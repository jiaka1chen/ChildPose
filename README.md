# ChildPose: Foundation for Children Pose Modeling

**[NeurIPS 2026]**

<p align="center">
  <a href="https://pediamed.ai/ChildPose/">🌐 Project Page</a> •
  <a href="https://huggingface.co/Asp1rinn/ChildPose">🤗 Model</a> •
  <a href="#usage">💻 Inference Code</a>
</p>

### Authors
Jiakai Chen<sup>1,2*</sup>,
[Yifan Shen](https://shenyifans.github.io/)<sup>1,2*</sup>,
[Boyi Li](https://resurgamm.github.io/)<sup>1,2*</sup>,
Chuanmiao Dong<sup>2</sup>,
[Xu Cao](https://www.irohxucao.com/)<sup>1,2†</sup>

<sup>1</sup>PediaMed AI, <sup>2</sup>University of Illinois Urbana-Champaign

<sup>*</sup>Equal Contribution, <sup>†</sup>Corresponding Author

### Abstract
Human pose estimation has advanced rapidly driven by large-scale adult-centric models. However, children
remain significantly underrepresented due to data scarcity, distinct body morphology, and unique motion
dynamics. We introduce **ChildPose**, a modular video foundation model designed for child-centric pose
estimation. ChildPose transforms pretrained pose encoders into a streaming pediatric model via
**memory-augmented temporal attention** and **age-aware co-training**. By maintaining a memory bank of
adjacent frame representations, the model achieves robust localization under motion blur, occlusion, and
atypical poses, while an auxiliary age-prediction objective enforces representations that capture
developmental morphology. We curate a diverse, video-based pediatric dataset spanning various age groups
and capture conditions, evaluating out-of-distribution generalizability on entirely unseen subjects.
Across multiple 2D baselines, ChildPose consistently outperforms both off-the-shelf models and child-data
fine-tuned variants, while ensuring stable sequence-level predictions. Our results underscore that
effective pediatric pose estimation necessitates child-specific modeling beyond standard adult-centric
estimators.

---

## About this repository

This repository contains the **inference code** of ChildPose; the model weights are hosted on the
Hugging Face Hub at [Asp1rinn/ChildPose](https://huggingface.co/Asp1rinn/ChildPose). Training code and
data are not part of this release.

ChildPose estimates COCO-17 2D keypoints of a child in video. It is a top-down model: each frame is cropped
around the child's bounding box, encoded by an image backbone, conditioned on a memory of the child's
previous frames, and decoded into keypoint heatmaps. The memory only looks back a fixed number of frames,
so videos of any length are processed frame by frame. The age-prediction branch is used only during
training and is not included in the released models.

## Models

| Variant | Backbone | Input (H x W) | Parameters | Checkpoint size |
|---|---|---|---|---|
| `hrnet_w48` | HRNet-W48 | 288 x 384 | 102.5M | 0.41 GB |
| `hrformer_base` | HRFormer-B | 288 x 384 | 82.1M | 0.33 GB |
| `vitpose_base` | ViTPose-B | 256 x 192 | 108.2M | 0.43 GB |
| `sapiens_0.3b` | Sapiens-0.3B | 1024 x 1024 | 346.1M | 1.38 GB |
| `sapiens_0.6b` | Sapiens-0.6B | 1024 x 1024 | 696.3M | 2.79 GB |
| `sapiens_1b` | Sapiens-1B | 1024 x 1024 | 1227.3M | 4.91 GB |

Pass a variant name wherever a checkpoint is expected and the file is downloaded from the Hub
(and cached); a local `.safetensors` path works as well.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.9+ and PyTorch 2.0+. `huggingface_hub` is used to download checkpoints, OpenCV only by
`scripts/run_video.py`.

## Usage

### Command line

```bash
python scripts/run_video.py --checkpoint vitpose_base \
    --video clip.mp4 --boxes boxes.json --out poses.json --vis poses.mp4
```

`boxes.json` gives the child's box for each frame, one entry per tracked child (frames without an entry
are skipped):

```json
{"child_0": {"0": [100, 50, 300, 450], "1": [102, 51, 301, 452]},
 "child_1": {"0": [500, 80, 640, 400]}}
```

Boxes are `[x1, y1, x2, y2]` in pixels and should be tight boxes around the child; the model enlarges them
by 10% before cropping. ChildPose does not detect or track children; use any person detector / tracker.

The output JSON holds, for every track and frame, the 17 keypoints in image pixels (COCO order, see
`keypoint_names`), a per-keypoint score (heatmap peak value) and their mean.

### Python

```python
from childpose import ChildPosePredictor

predictor = ChildPosePredictor("sapiens_1b")  # or a path to a .safetensors file; CUDA if available

# frames: any iterable of PIL images or RGB uint8 arrays (a generator is fine for long videos)
# boxes:  list or {frame_index: [x1, y1, x2, y2]}; None / missing = child not visible
results = predictor.predict_video(frames, boxes, fps=30)
for i, r in enumerate(results):
    if r is not None:
        print(i, r["keypoints"].shape, r["score"])   # (17, 2) image pixels

# single images, no temporal memory
results = predictor.predict_images(images, boxes)
```

### Temporal memory and frame rate

Each frame attends to the encoded tokens of up to 6 earlier frames, with a learned embedding of how many
steps back each one is. The models were trained on frames sampled at about **5 fps**, i.e. memory frames
about 0.2 s apart. The video is processed in a single pass in frame order, every frame is encoded once, and
the memory runs over the whole video (it is not reset between fixed-length clips). For a faster video,
`predict_video(..., fps=F, mode=...)` (`--mode` on the command line) chooses how frames reach the memory,
with `step = round(F / 5)` (6 for 30 fps):

| `mode` | Frames predicted | Memory of frame `i` |
|---|---|---|
| `"interleave"` (default) | all | frames `i - step`, `i - 2*step`, ... (about 0.2 s, 0.4 s, ... earlier, as in training) |
| `"downsample"` | `0, step, 2*step, ...` (the video at about 5 fps, as in training) | the previous predicted frames |
| `"sequential"` | all | frames `i - 1`, `i - 2`, ... (closer in time than in training) |

With `"interleave"`, frame `i` uses one of `step` memories (`i % step`): consecutive frames do not share a
memory, so frame `i - 1` is not in the memory of frame `i`. Leave `fps=None` if the frames are already at
about 5 fps; the three modes are then the same.

- `clip_len=N` empties each memory every `N` frames it has seen; by default it is never reset.
- `use_memory=False` (`--no-memory`) predicts every frame on its own.
- Frames without a box are skipped and do not enter the memory.

The backbone runs on batches of frames (`batch_size`); the memory then advances one frame at a time.
On CUDA, inference uses float16 autocast by default (`amp=False` / `--no-amp` for float32).

### Lower-level API

```python
from childpose import load_model

model = load_model("hrnet_w48", "cuda")
tokens = model.encode(crops)                 # [N, 3, H, W] normalized crops -> [N, tokens, dim]
fifo = []                                    # memory of one child, empty at the start
z, fifo = model.step(tokens[i:i + 1], fifo)  # one frame
heatmaps = model.heatmaps(z)                 # [1, 17, Hh, Wh], 4x the token grid
clip_heatmaps = model(clips)                 # [B, T, 3, H, W] -> [B, T, 17, Hh, Wh], memory from frame 0
```

Cropping and normalization are in `childpose/preprocess.py`.

## Repository layout

```
childpose/
  config.py        variant configurations, COCO keypoint names / skeleton
  model.py         ChildPoseModel, load_model
  memory.py        temporal memory (FIFO of past frames, lag embeddings, attention blocks)
  head.py          heatmap pose head and arg-max decoding
  preprocess.py    box enlargement, crop / resize / normalize, crop -> image mapping
  predictor.py     ChildPosePredictor: videos of any length and single images
  hub.py           checkpoint download from the Hugging Face Hub
  backbones/       Sapiens, ViTPose-B, HRNet-W48 and HRFormer-B encoders
scripts/run_video.py
RESPONSIBLE_USE.md   responsible use policy (intended use, prohibited uses, limitations, disclaimer)
```

## License

The code and the model weights are released under the
[Creative Commons Attribution-NonCommercial 4.0 International](LICENSE) license (CC BY-NC 4.0), the most
restrictive license among the components they build on (Meta's Sapiens is CC BY-NC 4.0). Parts of the
backbone code are adapted from ViTPose and MMPretrain (Apache-2.0) and from HRNet, HRFormer and
openseg.pytorch (MIT); their copyright and license notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

### Responsible use

Use of the code and weights is subject to the [ChildPose Responsible Use Policy](RESPONSIBLE_USE.md).
In short:

- **Intended use**: non-commercial research on children's pose and movement, on lawfully recorded video
  with the consent of the children's parents or legal guardians, under human oversight.
- **Prohibited**: re-identification or identity linkage (including from body shape or gait); covert
  surveillance or monitoring of children; use on its own for clinical diagnosis, screening, treatment
  or any other high-stakes decision about a child; commercial use; any use that violates the law.
- **Not a medical device**: not clinically validated or approved by any regulatory authority.
  Predictions can be wrong, and performance across skin tone, body habitus, clinical condition and
  capture setting has not been validated. Deployment in medical, educational or monitoring settings
  needs its own validation, fairness evaluation, privacy safeguards and human oversight.
- **No warranty, no liability**: the models are provided "as is"; the authors and their institutions
  are not liable for any use of them or of their outputs (see `LICENSE` and the policy).

## Citation

```bibtex
@inproceedings{chen2026childpose,
  title     = {ChildPose: Foundation for Children Pose Modeling},
  author    = {Chen, Jiakai and Shen, Yifan and Li, Boyi and Dong, Chuanmiao and Cao, Xu},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

---

<p align="center">
  <b>PediaMed AI</b> • University of Illinois Urbana-Champaign
</p>
