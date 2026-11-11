#!/usr/bin/env python3
"""Run ChildPose on a video (or a folder of frames) for one or more tracked children.

Boxes JSON: one entry per child track, each mapping frame index -> [x1, y1, x2, y2] in pixels
(frames without an entry are skipped)::

    {"child_0": {"0": [100, 50, 300, 450], "1": [102, 51, 301, 452]},
     "child_1": {"0": [500, 80, 640, 400]}}

Example::

    python scripts/run_video.py --checkpoint vitpose_base \\
        --video clip.mp4 --boxes boxes.json --out poses.json --vis poses.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from childpose import COCO_KEYPOINTS, COCO_SKELETON, ChildPosePredictor  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def open_frames(video: Optional[Path], frames_dir: Optional[Path]) -> Tuple[Iterator, Optional[float]]:
    """Iterator of RGB uint8 frames, and the frame rate when known."""
    if frames_dir is not None:
        from PIL import Image

        files = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        return (Image.open(p).convert("RGB") for p in files), None
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or None

    def gen():
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    return
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()

    return gen(), fps


def draw(video: Optional[Path], frames_dir: Optional[Path], tracks: Dict[str, Dict[int, dict]], fps: float,
         out_path: Path, min_score: float, keep_every: int = 1) -> None:
    import cv2
    import numpy as np

    frames, _ = open_frames(video, frames_dir)
    writer = None
    colors = [(0, 200, 255), (255, 120, 0), (0, 220, 0), (220, 0, 220), (255, 255, 0)]
    for i, frame in enumerate(frames):
        if i % keep_every:
            continue
        img = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / keep_every, (w, h))
        for t, (_tid, poses) in enumerate(tracks.items()):
            pose = poses.get(i)
            if pose is None:
                continue
            color = colors[t % len(colors)]
            kp, sc = pose["keypoints"], pose["scores"]
            for a, b in COCO_SKELETON:
                if sc[a] >= min_score and sc[b] >= min_score:
                    cv2.line(img, tuple(int(v) for v in kp[a]), tuple(int(v) for v in kp[b]), color, 2, cv2.LINE_AA)
            for j, (x, y) in enumerate(kp):
                if sc[j] >= min_score:
                    cv2.circle(img, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)
        writer.write(img)
    if writer is not None:
        writer.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="checkpoint file, or a variant name (e.g. vitpose_base) to download from the Hugging Face Hub")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path)
    src.add_argument("--frames-dir", type=Path, help="folder of frames, read in sorted file-name order")
    ap.add_argument("--boxes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output JSON")
    ap.add_argument("--fps", type=float, default=None,
                    help="frame rate of the input (default: read from the video; for --frames-dir, 5)")
    ap.add_argument("--mode", choices=["interleave", "downsample", "sequential"], default="interleave",
                    help="how a video faster than 5 fps is fed to the memory: interleave (every frame, memory of "
                         "frames ~0.2 s apart; default), downsample (predict only every round(fps/5)-th frame), "
                         "sequential (every frame, memory of the immediately preceding frames)")
    ap.add_argument("--memory-fps", type=float, default=5.0, help="frame rate the models were trained at (default 5)")
    ap.add_argument("--no-memory", action="store_true", help="predict every frame independently")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-amp", action="store_true", help="run in float32 on CUDA")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--vis", type=Path, default=None, help="also write a video with the predicted skeletons")
    ap.add_argument("--vis-min-score", type=float, default=0.2)
    args = ap.parse_args()

    raw = json.loads(args.boxes.read_text())
    track_boxes = {tid: {int(k): v for k, v in boxes.items()} for tid, boxes in raw.items()}
    predictor = ChildPosePredictor(args.checkpoint, device=args.device, amp=not args.no_amp, batch_size=args.batch_size)

    fps = args.fps
    tracks: Dict[str, Dict[int, dict]] = {}
    for tid, boxes in track_boxes.items():
        frames, video_fps = open_frames(args.video, args.frames_dir)
        fps = fps or video_fps
        results = predictor.predict_video(
            frames, boxes, fps=fps, mode=args.mode, memory_fps=args.memory_fps, use_memory=not args.no_memory
        )
        tracks[tid] = {i: r for i, r in enumerate(results) if r is not None}
        print(f"{tid}: {len(tracks[tid])} frames", file=sys.stderr)

    out = {
        "variant": predictor.model.cfg.variant,
        "fps": fps,
        "mode": args.mode,
        "keypoint_names": list(COCO_KEYPOINTS),
        "tracks": {
            tid: {
                str(i): {"keypoints": r["keypoints"].round(2).tolist(), "scores": r["scores"].round(4).tolist(),
                         "score": round(r["score"], 4)}
                for i, r in poses.items()
            }
            for tid, poses in tracks.items()
        },
    }
    args.out.write_text(json.dumps(out))
    if args.vis is not None:
        keep_every = 1
        if args.mode == "downsample" and fps:
            keep_every = max(1, int(round(fps / args.memory_fps)))
        draw(args.video, args.frames_dir, tracks, fps or args.memory_fps, args.vis, args.vis_min_score, keep_every)


if __name__ == "__main__":
    main()
