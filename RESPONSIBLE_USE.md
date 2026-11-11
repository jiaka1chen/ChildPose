# ChildPose Responsible Use Policy

ChildPose models estimate the body pose of children from video. Children are a vulnerable population and
video of children is sensitive personal data, so the ChildPose code and model weights (together, "the
Models") are released only for the uses described below. This policy applies to every copy of the
Models, to models derived from them, and to their outputs. It applies in addition to the license
(CC BY-NC 4.0, see `LICENSE`).

## 1. Intended use

The Models are research artifacts. They are intended for non-commercial research on 2D pose estimation
and movement analysis of children, for example research on pediatric movement analysis, child-centered
human-computer interaction and assistive video understanding, carried out:

- on video that was recorded and is processed lawfully, with the informed consent of the children's
  parents or legal guardians (and the children's assent where appropriate) and, for human-subjects
  research, under the approval of the responsible ethics board (IRB or equivalent);
- under qualified human oversight, with the outputs treated as estimates that may be wrong.

## 2. Prohibited uses

The Models and their outputs must not be used:

1. **For re-identification or identity linkage**: to identify a child, to recognize or verify who a child
   is (including from body shape, gait or movement patterns, which are biometric signals), or to link a
   child's data to other records or datasets.
2. **For covert surveillance or monitoring of children**: to observe, track or monitor children without
   the knowledge and consent of their parents or legal guardians, or in any setting where the children or
   their guardians would not expect to be monitored.
3. **On their own for clinical diagnosis, treatment or other high-stakes decisions**: as a standalone
   tool for diagnosis, screening, triage, treatment, or developmental or clinical assessment, or as the
   basis of any other decision with significant consequences for a child or family (for example
   educational placement, child-welfare or custody decisions, insurance or eligibility decisions).
   The Models are not a medical device and have not been cleared or approved by any regulatory authority.
4. **For any commercial purpose**, as stated in the license.
5. **In violation of applicable law**, including privacy, data-protection, child-protection, consumer-
   protection and medical-privacy laws, or to harm, exploit, harass or discriminate against children.

Anyone who redistributes the Models or a derivative of them must pass on this policy with them.

## 3. Known limitations and risks

- **Errors.** Pose predictions can be wrong, especially under occlusion, unusual postures, crowded scenes,
  poor lighting, low resolution or motion blur. Erroneous predictions can lead to misinterpretation of a
  child's movements, in particular in clinical settings, if relied upon without human oversight.
- **Unvalidated populations.** Beyond age, dimensions such as skin tone, body habitus, clinical
  condition and capture setting were not annotated in the training data, so performance across them has
  not been validated and may differ between groups of children.
- **Deployment needs more work.** Any use in medical, educational or monitoring settings requires
  additional validation on the target population and setting, a deployment-specific fairness
  evaluation, privacy safeguards and human oversight.
- **Scope of the model.** The Models produce 2D keypoints (COCO-17) for a child whose bounding box is
  given; they do not detect or track children, do not estimate 3D pose, and their quality depends on the
  quality of the boxes. They were trained on frames sampled at about 5 fps.
- **Temporal behavior.** Predictions that are smooth over time are not necessarily faithful to the true
  motion, especially during rapid or complex movements. Keypoint scores are heatmap peak values, not
  calibrated probabilities.
- **Dual use.** Pose models for children can be misused, for example for covert monitoring. Such uses are
  prohibited by Section 2.

## 4. Privacy

Video of children is personal data. Users are solely responsible for having a lawful basis and the
required consent to process any video they use with the Models, for complying with the privacy and
data-protection laws that apply to them, and for storing videos and outputs securely. Keypoint outputs
are less identifying than video but are not anonymous: body shape, gait and movement are weak biometric
signals.

## 5. Training data

The Models were trained on de-identified pediatric video. Data from private sources was collected under
protocols approved by the institutional review boards of the originating institutions, with documented
informed consent from parents or legal guardians; faces were blurred and identifying information was
removed before the data was accessed. This release contains no training data: no videos, images,
annotations or personal information. Raw clinical videos are not released under any terms.

## 6. Disclaimer

The Models are provided "AS IS" and "AS AVAILABLE", without representations or warranties of any kind,
including warranties of accuracy, fitness for a particular purpose or non-infringement (see Section 5 of
`LICENSE`). To the extent permitted by law, the authors and their institutions are not liable for any use
of the Models, for any decision or action taken on the basis of their outputs, or for any damages
arising from them. Users are solely responsible for how they use the Models and for complying with this
policy and with the law. The authors do not endorse, and are not responsible for, any use that violates
this policy.

ChildPose builds on third-party models (Sapiens, ViTPose, HRNet, HRFormer). Their authors are not
affiliated with this release and do not endorse it.

To report a misuse of the Models or a problem with this policy, please open an issue in the code
repository or contact the authors.
