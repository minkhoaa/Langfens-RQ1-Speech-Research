# SM13 native-large local — metric/calibration registration

Decision: `SM13-NATIVE-LOCAL-20261114-v1`; local only, no Kaggle calls or submission.

## Locked evaluator v1

`research/sm13_backbone_side/detection.py`, spec `SM13-COMMON-EVENT-DETECTION-v1`.
Gold is joined after inference by `L2_ARCTIC|speaker|utterance|target/gap|position`.
Exactly N targets and N+1 exact gaps; duplicate/missing/extra IDs fail, no dropped cases.
Positive target=S or O (count once even wrong S/O type), positive gap=exact insertion.
Scores: target=p(S)+p(O), gap=p(I); decision score >= threshold.
Micro P/R/F1 from pooled TP/FP/FN, separate target/gap and per-speaker counts,
FP/utterance, binary AP. Undefined ratios/AP return null, never an invented zero.
This is NOT paper hierarchical alignment F1. Typed S/O/I AP/F1 remains separately
available from the unchanged head metric helper; do not replace it with binary AP.

Fixtures: all-correct, S/O wrong typing counted once, exact-gap mislocalization,
N=0/N=1, repeated phones, boundary gaps, missing support, shuffled keys,
duplicate/missing/extra IDs, invalid scores/probabilities, threshold ties.
CROTTC event adapter and final common-model validation remain PENDING.

## First S1 fitting stage: TRAIN-derived speaker-disjoint calibration

- Native pretrained WavLM Large revision `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c`,
  frozen/eval FP32, indices0,2,...24; 315453120 backbone parameters; fresh13x1024 heads.
- Full TRAIN1800 cache is independently persisted/hash/geometry/finite verified.
- Fit1350 utterances/9 TRAIN speakers; hold450 ABA/ASI/BWC, same historical outer0
  used by the verified S0 epoch50 trial. This is **reused exploratory TRAIN development**.
- First target50 + gap50, seed101/gap1101, unchanged losses/LR/weight decay/batches
  and head topology except registered geometry. Fixed terminal50 checkpoint per head.
- This stage supplies speaker-disjoint TRAIN calibration before later fullTRAIN1800
  fitting; it is NOT the final fullTRAIN1800->DEV_SEARCH599 result or CROTTC ranking.
- Primary thresholds for the later fullTRAIN fit: choose from21x21 pairs `[0,.05,...1]`
  using epoch50 held TRAIN predictions, maximizing pooled micro detection F1;
  tie-break higher precision, then higher target threshold, then higher gap threshold.
  Reuse the same search rule/budget for S0; never search on DEV_SEARCH/test labels.
- Fixed0.5 per-epoch detection is diagnostic. No best epoch selection; no primary
  quality claim from calibration F1. Retain per-head typed metrics/AP and loss curves.
- S0 historical normalization is HF only; S1 uses HF then declared SB layernorm.
  S0/S1 head capacity also differs. Do not call a difference a pure backbone effect.
- Conditional50->100 remains possible only after curve review and a separate
  bounded receipt under GOAL_SM13§5.4; this run does not auto-extend.

## Execution

Shared local GPU lock; two CPU threads; max4h window; initial RAM16GiB,
runtime RAM4GiB, GPU512MiB and disk14GiB safety floors. Main CROTTC export has priority.
Atomic full state each epoch and safe optimizer-boundary pause, immediate reload/SHA;
stop nonfinite/mismatch, no batch skipping. No remote use/fallback, no new backbone
fine-tuning, no sealed DEV_CAL/test. Checkpoints/features remain outside Git.
