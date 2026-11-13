# RC2 H-001 — Omission-aware alignment

**Status: EXECUTED / SUPPORTED EXPLORATORY for ranking; not verified as an independent improvement.** A matched CPU-only frozen-frame comparison ran A (current forced Viterbi evidence) against B (one-phone optional/null path max-sum evidence) on the complete authorized TRAIN, DEV_CAL and DEV_SEARCH populations. EVAL-V3.1 recomputation matched the published run metrics exactly. DEV_SEARCH is development-exposed, has only four speakers, and is not independent confirmation. H001-C was not run: the cached CharSiU outputs are frame-classifier emissions, not verified CTC-trained phoneme posteriors.

The initial Pi/Codex `create_thread` route rejected the full alias `openai-codex/gpt-6-luna`; the native collaboration subagent mechanism then ran this H-001 task with `gpt-6-luna`.

## Question, protocol and exact contrast

Question: does a zero-duration/null alternative for one intended phone at a time improve omission ranking when all frame inputs, phone references, event population and all other class heads remain fixed?

- **A — forced Viterbi control:** align with the existing source topology; use negative mean target-phone log posterior on the frames assigned to that phone as the omission evidence (higher means more omission-like).
- **B — optional phone:** for each target slot independently, remove only that phone's emitting state from the same monotonic max-sum graph, retain all other phone and SIL states, and score `(best deletion-path score − best full-path score) / frame_count`. The skip penalty was predeclared as zero; no penalty was fit or tuned. The normalization uses the same utterance frame count for all its targets.
- All A/B computations use identical stored frame logits and intended references. Exact same `STU-01-GOLD-ONLY_seed101` substitution probabilities and exact-gap insertion scores are held fixed. For decisions, a 1-D logistic map per arm is fit on TRAIN correct/omission target events only, with fixed C=1; it estimates P(O | C or O), then composes with the unchanged P(S) as P(O)=(1−P(S))P(O|C/O), preserving a valid C/S/O simplex. The evaluator's primary AP uses the uncalibrated evidence in `score_omission`, not the logistic output. The decision rule is argmax; no DEV_SEARCH threshold or model selection was used.

This contrast is exploratory: the method changes the omission evidence/readout while retaining the existing S/I student heads. The TRAIN logistic calibrator is fit in-sample and is not an independent calibration estimate. No neural model was loaded, run or trained in this Cycle 2 scoring job.

## Implementation and path contract

`tools/extract_acoustic_features.py::viterbi_align` creates `[phone_0, SIL, phone_1, SIL, …]`, with stay, +1 and a +2 transition when the skipped state is SIL. There is no arbitrary deletion transition for phones 1…N−1. Important source nuance: initialization permits both state 0 and state 1 on frame 0, so phone 0 can already receive zero frames. There is no distinct leading-SIL state; the first separator is after phone 0 and is assigned to internal gap 1. The routine does not handle N=0. These topology quirks are reproduced in A and recorded as limitations; the evaluator event population is never changed to match the aligner.

B removes the selected phone state and leaves both adjacent SIL states. This is one explicit optional-path construction, not a uniquely determined linguistic alignment. Adjacent duplicate SIL states may make the deletion alternative too permissive; no SIL-collapse ablation was run. That caveat is material to interpreting the improvement.

The frame cache is `experiments/evaluation_v3/closure/waveform_full/utterances/{uid}.npz`; its `ctc_logits` key is the stored frame-classifier output. The cache's 5,398 indexed NPZ/JSON files were all verified against `LOCAL_CACHE_INDEX_SHA256.json`; the receipt reports 2,699 WAV utterances and exact full-cohort feature replay. The model config identifies `Wav2Vec2ForFrameClassification`, and the CharSiU source describes W2V2-FC-10ms as a frame classifier trained on forced-aligned labels. So CTC forward-sum values from these logits would not be valid CTC sequence likelihoods. [CharSiU model description](https://congzhang-linguist.github.io/project/charsiu/)

## EVAL-V3 results

All rows below were computed by the single authoritative `UnifiedEvaluatorV3` (EVAL-V3.1). A separate replay from the canonical prediction arrays reproduced AP/F1 exactly (maximum numeric delta 0.0). The population remained unchanged: TRAIN 1,800 utterances / 59,562 targets / 61,362 gaps; DEV_CAL 300 / 10,043 / 10,343; DEV_SEARCH 599 / 19,755 / 20,354. DEV_SEARCH has S/O/I support 1,768/209/125.

### Matched overall results

| Split | Arm | AP O | AP S | Exact-gap AP I | Macro-3 error AP | O precision | O recall | O F1 | Macro-3 error F1 | Macro-4 F1 | Joint FWR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | A forced | 0.113722 | 0.428201 | 0.355257 | 0.299060 | 0.412698 | 0.014247 | 0.027542 | 0.295722 | 0.447936 | 2.642222 |
| TRAIN | B optional | 0.207296 | 0.428201 | 0.355257 | 0.330251 | 0.083333 | 0.000548 | 0.001089 | 0.287481 | 0.441850 | 2.607778 |
| DEV_CAL | A forced | 0.195631 | 0.528023 | 0.176870 | 0.300174 | 0.357143 | 0.010823 | 0.021008 | 0.270099 | 0.422432 | 2.213333 |
| DEV_CAL | B optional | 0.323131 | 0.528023 | 0.176870 | 0.342675 | 0.333333 | 0.002165 | 0.004301 | 0.263984 | 0.417861 | 2.183333 |
| DEV_SEARCH* | A forced | 0.036767 | 0.364602 | 0.186715 | 0.196028 | 0 | 0 | 0 | 0.215857 | 0.393203 | 2.896494 |
| DEV_SEARCH* | B optional | 0.087932 | 0.364602 | 0.186715 | 0.213083 | 0 | 0 | 0 | 0.215646 | 0.393133 | 2.871452 |

`*` DEV_SEARCH is development-exposed and exploratory. The raw ranking moved in the expected direction: AP O increased by +0.093574 TRAIN, +0.127500 DEV_CAL, and +0.051165 DEV_SEARCH. The corresponding Macro-3 AP deltas were +0.031191, +0.042501 and +0.017055. S and exact-gap I APs are exactly unchanged because those score heads were held fixed. At the fixed argmax policy, O F1 did not improve: it is zero on DEV_SEARCH for both arms, while Macro-3 F1 is slightly lower for B. Thus the evidence supports an exploratory ranking change, not a demonstrated operating-point improvement.

Joint FWR is shown as descriptive behavior for this single argmax policy; it is not a pass/fail criterion. This H-001 run did not produce a threshold sweep or make FWR≤0.20 a gate.

### Speaker and correct-phone diagnostics

DEV_SEARCH omission AP rose for each speaker; with four clusters these differences remain descriptive:

| Speaker | O positives | A AP O | B AP O | Delta |
|---|---:|---:|---:|---:|
| NCC | 55 | 0.053347 | 0.158506 | +0.105159 |
| RRBI | 65 | 0.029054 | 0.073643 | +0.044589 |
| YBAA | 31 | 0.031118 | 0.087663 | +0.056546 |
| YDCK | 58 | 0.050957 | 0.092323 | +0.041366 |

At the same argmax policy, false O decisions among gold CORRECT phones were 5→1 of 7,800 on DEV_CAL and 8→4 of 17,778 on DEV_SEARCH. Using gold duration only for post-inference diagnostics, false O decisions among short (≤30 ms) correct phones were 1→0 of 148 on CAL and 1→0 of 298 on SEARCH. This limited stratum does not establish short-phone robustness; false omissions on correct phones are evaluated with one fixed decision rule, and gold timing never entered alignment or scoring.

## H001-C and remaining science

The intended CTC comparison was not fabricated. The cached head is a frame classifier; its config/lineage does not establish a CTC training objective or calibrated CTC blank semantics. Therefore no CTC marginal sequence likelihood is reported. To run C, obtain or locally train on CUDA a verified phoneme-CTC model with a documented blank/token contract, cache its frame log posteriors, and compare full versus one-phone-deleted CTC forward probabilities on matched events. Adjacent identical phones need the standard intervening-blank rule; phone/length normalization must be fixed using TRAIN speaker groups. A CTC blank probability alone is not omission truth.

Primary literature makes optional paths plausible but does not confirm this result: Yuan and Liberman's skip-state HMM permits zero-duration phones and improves boundary agreement on Mandarin broadcast speech; it reports no L2 omission AP. Lin and Wang jointly learn MDD alignment but do not establish an explicit null-phone path. Cao et al. propose segmentation-free CTC GOP including deletion/insertion on different datasets. These studies motivate methods, not LANGFENS performance. [Yuan & Liberman, Interspeech 2015](https://www.isca-archive.org/interspeech_2015/yuan15_interspeech.pdf), [Lin & Wang, ICASSP 2022](https://ieeexplore.ieee.org/document/9746727/), [Cao et al., Interspeech 2024](https://www.isca-archive.org/interspeech_2024/cao24b_interspeech.html).

## Provenance, artifacts and reproduction

- Experiment ID: `RC2-H001-A-B-OPTIONAL-PHONE-CPU-20261107`.
- Source revision: `aa4e6ea2147180a4a820d77a71911f333fcb6721`.
- Device: CPU conventional dynamic programming/statistical calibration only; no neural forward/backward or CUDA job.
- Frame model weight SHA-256: `6dc8a18422db7c22e951d5f72dc2afc267b942eb0b8459ac6dcc0cf412536de1`; config SHA-256: `7406aa4f917267640865688aa62f2337664a3abb9a49a2f204d932b53aeb6cb7`.
- Frozen shared S/I control checkpoint SHA-256: `8ce4ebe5f5e4ebb6fcbd02a9599328059e51d5ecc3bd9f52bdb66c822f994090`.
- Cache index SHA-256: `460b99072fbab7d5d91c4dc63fc1ddd4da682a9018e6f31fbcbfca5267bcbafb`.
- EVAL-V3 evaluator SHA-256: `76e41f906e5d953ee55cee8156379644bdd96bf96efab69e991b87d7f857d4a2`.
- Script SHA-256: `c0e9602c130e24b18681f0ae792fd3574e1d5a2102e45f02a536edd5cd54050a`.
- Result folder: `research/long_running_method_discovery/cycle_02/h001_alignment/run_20261107_viterbi_optional_cpu_v3/`.
- EVAL-V3 receipt SHA-256: `a33af22380141b174c86d790a0d2c4959e1a31ffcca9ebb7b6c197c91ab1ced8`.
- The folder contains `event_scores.csv`, raw intermediate and canonical full-event prediction NPZs, and `EVALUATION_RECEIPT.json`. Canonical output hashes are listed in that receipt; every prediction has `dataset:speaker:utterance:domain:slot` identity.

Canonical prediction SHA-256 by split/arm:

| File | SHA-256 |
|---|---|
| `train_A_viterbi_forced_canonical_predictions.npz` | `ce66c4fbd369292591663e32c5d4058362a350cd9c19d42dd15e601923de0e1c` |
| `train_B_optional_phone_canonical_predictions.npz` | `8a1f00356514a27a7d486a447e4be48d488c0c312b4f040340cfbb828846e254` |
| `dev_cal_A_viterbi_forced_canonical_predictions.npz` | `199e901e5a1752aaaf7cd517b3b81301c516c70670cee59a3647b7de995334a5` |
| `dev_cal_B_optional_phone_canonical_predictions.npz` | `8c4d7b908d37b4c4e6983a2a158457d59185c67cc386d61ebcdde54cae2dfc83` |
| `dev_search_A_viterbi_forced_canonical_predictions.npz` | `24d9a7d8f1a7f7a0e0e15a8352df468231a07c19b71e9f5d625592e53d7930fa` |
| `dev_search_B_optional_phone_canonical_predictions.npz` | `866a2a77723b44ca7d4553f934f79068d4c3b8bc511d79a21f12bb55a1e730cc` |

Reproduce the score run from the repository root with a fresh output folder by overriding `RUN` before calling the recorded script's `main()`:

```bash
.venv/bin/python -c 'import runpy; from pathlib import Path; m=runpy.run_path("research/long_running_method_discovery/cycle_02/h001_alignment/run_h001_viterbi_optional_cpu.py",run_name="h001"); m["RUN"]=Path("research/long_running_method_discovery/cycle_02/h001_alignment/run_20261107_viterbi_optional_cpu_reproduction"); m["main"]()'
```

Canonicalize the new folder's prediction arrays by running `canonicalize_h001_predictions.py` after changing its `RUN` constant to that fresh folder. No SEARCH-based tuning is performed by either command.

## Decision

**Supported exploratory for O ranking under this one frozen frame-classifier path contrast; not verified as a generalizable method.** B improved raw omission AP over A on TRAIN, CAL, and all four SEARCH speakers, while S/I scores were unchanged. O F1 did not improve under argmax, and the search cohort is already exposed. Preserve this result as a valid negative for operating F1 / positive for ranking; do not call it a validated winner. Next, test whether the gain survives (1) speaker-grouped TRAIN out-of-fold scorer calibration, (2) collapsing or otherwise controlling the duplicated adjacent SIL states, and (3) an independently trained/verified phoneme-CTC model for H001-C.

## CUDA screening addendum — optional deletion evidence plus acoustic student

### Registered comparison

`RC2-H001-OPTIONAL-ACOUSTIC-CUDA130-S101` adds one TRAIN-standardized scalar, the prior B optional-phone score, to the Phase 3 Gold-Only 51D per-target feature stream. All existing acoustic/reference inputs remain; the substitution/omission target softmax and independent gap head remain, with exactly N+1 gaps. The control is the freshly retrained CUDA 13.0 Gold-Only model, seed 101, same 12 epochs and 684 updates. The added feature increases parameter count from 78,661 to 78,725 (+64). Gold timing, realized phones, and error labels are not inputs to the score. The CharSiU source remains a frame classifier; its stored `ctc_logits` are not verified CTC sequence posteriors, so this is an optional-phone path score over frame emissions, not CTC deletion likelihood.

All event joins were exact: 59,562 TRAIN targets, 10,043 DEV_CAL targets, and 19,755 DEV_SEARCH targets. The official Phase 3 EVAL-V3 evaluator preserved 20,354 exact gaps (leading, internal, and trailing) and 599 DEV_SEARCH utterances. Threshold policies were selected on DEV_CAL; DEV_SEARCH remains exposed and exploratory.

### Results versus matched CUDA Gold-Only seed 101

| Metric | Gold-Only | Optional + acoustic | Delta |
|---|---:|---:|---:|
| AP S | 0.364602 | 0.356376 | -0.008226 |
| AP O | 0.159433 | 0.161252 | +0.001819 |
| AP exact-gap I | 0.186715 | 0.169413 | -0.017302 |
| Macro-3 Error AP | 0.236917 | 0.229014 | -0.007903 |
| Macro-3 Error F1, DEV_CAL budget 0.05 | 0.025063 | 0.028008 | +0.002945 |
| Realized false warnings at budget 0.05 | 5 | 21 | +16 |
| Macro-3 Error F1, DEV_CAL budget 0.20 | 0.104897 | 0.103815 | -0.001082 |
| Realized false warnings at budget 0.20 | 63 | 75 | +12 |
| Macro-4 F1, budget 0.05 | 0.255615 | 0.257775 | +0.002160 |

The class F1 values at the 0.05 operating point were S/O/I `0.003388 / 0.034483 / 0.046154` for the candidate versus `0 / 0 / 0.075188` for control. At budget 0.20 they were `0.006757 / 0.049587 / 0.255102` versus `0.001131 / 0.026432 / 0.287129`. These threshold results trade errors and warnings; no FWR cutoff is used as a gate.

**Decision: NOT_SUPPORTED_EXPLORATORY for discrimination.** O AP rose only 0.0018 while S and exact-gap I AP fell, leaving Macro-3 Error AP down 0.0079. Macro-3 F1 rose by 0.0029 at the 0.05 policy but with 16 more false warnings; it was slightly lower at 0.20. This single seed on exposed SEARCH does not justify integration. H-001's earlier frozen-score ranking gain is not transferred into the trained student by this simple concatenated scalar.

### CUDA and reproducibility record

- Wrapper job `RC2-H001-OPTIONAL-ACOUSTIC-CUDA130-S101-R2`, lease returned code 0; duration 53.61 sec. An initial attempt failed immediately because direct script invocation lacked repository import path; it was preserved in the ledger and corrected before the R2 run.
- `.venv-cuda`: PyTorch 2.14.1+cu130; GeForce GTX 1650; peak allocated VRAM 44.10 MiB; 684 updates; 12 epochs.
- Checkpoint SHA-256: `751e3e4b601dd6c54e443074c88ec516b73d95f4036c2b9b7dd153cbc5c0d454`.
- Script SHA-256: `cfd83bc1ee307fb4261b361bc0c94eee32aa7bf8ef1c56ee22e39afc1a73bfc8`.
- Optional event score source SHA-256: `233205c746ae9356f3810d745c7c6d26a4962a8b07bf4b989b2168b1a2f133a3`.
- DEV_CAL prediction SHA-256: `73b57f0e5d2b3e2ee4ca9307484158c5c21b45d42c53ffadf31ba5cbade8dc61`.
- DEV_SEARCH prediction SHA-256: `e60615fa06c148d9a8355f636f52fbda8717004435e4cb0b37fee9a2e2607f5a`.
- EVAL-V3 output SHA-256: `6321c4d22f499f3723d1556fa4e13c3219526d5630b73ad6ba62bfef5bda9947`.
- Full receipt: `research/long_running_method_discovery/cycle_02/h001_alignment/combined_acoustic/RC2-H001-OPTIONAL-ACOUSTIC-CUDA130_seed101_receipt.json`.

Reproduction from repository root:

```bash
.venv/bin/python coordination/phase3_gpu_job.py \
  --job RC2-H001-OPTIONAL-ACOUSTIC-CUDA130-S101-R2 \
  -- /home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python -u \
  research/long_running_method_discovery/cycle_02/h001_alignment/combined_acoustic/train_optional_acoustic.py

.venv/bin/python experiments/student_evaluation/evaluate_student_predictions.py \
  --dev-cal research/long_running_method_discovery/cycle_02/h001_alignment/combined_acoustic/RC2-H001-OPTIONAL-ACOUSTIC-CUDA130_seed101_dev_cal.npz \
  --dev-search research/long_running_method_discovery/cycle_02/h001_alignment/combined_acoustic/RC2-H001-OPTIONAL-ACOUSTIC-CUDA130_seed101_dev_search.npz \
  --dev-cal-manifest data/derived/l2arctic/manifests/dev_cal.jsonl \
  --dev-search-manifest data/derived/l2arctic/manifests/dev_search.jsonl \
  --output research/long_running_method_discovery/cycle_02/h001_alignment/combined_acoustic/RC2-H001-OPTIONAL-ACOUSTIC-CUDA130-S101-eval.json
```
