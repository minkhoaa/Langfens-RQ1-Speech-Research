# H-004 multi-teacher / disagreement learning — provenance audit

Status: **source-audited; Cycle 2 experiment NOT RUN**. This review is based on existing Phase 3 artifacts and reports. It adds no model, training, inference, or evaluation result.

## Source classes: independent neural teachers versus score compositions

| Source | What it is | Independent neural teacher? | H-004 use |
|---|---|---:|---|
| `TEACHER_S_CHAMPION` | Binary substitution specialist checkpoint | Yes | S teacher score / possible feature source |
| `TEACHER_O_CHAMPION` | Binary omission specialist checkpoint | Yes | O teacher score / possible feature source |
| `TEACHER_I_CHAMPION` | Insertion precision-oriented specialist checkpoint | Yes | I teacher score / possible feature source |
| `TEACHER_I_EXP01` | Insertion F1-oriented specialist checkpoint | Yes | Alternative I source; distinct task/training recipe |
| EXP-24 | Standalone acoustic/target-gap neural model | Yes, independently trained model | Potential teacher source, but no OOF TRAIN supervision source established in this audit |
| EXP-34 | Standalone phonotactic neural probe | Yes, independently trained model | Potential teacher source; lexical provenance/OOF TRAIN source must be audited before KD |
| EXP-28 | Standalone prosodic/energy neural model | Yes, independently trained model | Potential teacher source; OOF TRAIN supervision source not established here |
| EXP-36 | Multi-model score composition / fused score recipe | **No** | Treat only as a score composition, never as an independent teacher or calibrated posterior |

The five specialist checkpoint sources above are actual neural checkpoints. The Phase 3 KD scores are not a simplex: S and O are independent binary probabilities, and insertion has independent gap semantics. No renormalization of S/O/I into a fabricated mutually exclusive distribution is justified.

## Existing speaker-held-out TRAIN OOF procedure

Reviewed `research/student/kd/run_teacher_oof.py` and `experiments/student_training/kd/provenance_v2_correct_i_sources.json`.

- Three `GroupKFold` folds group by TRAIN speaker; each fold holds out four speakers and fits on the other eight of the 12 TRAIN speakers.
- For each model/fold, normalizer statistics are estimated only from fold-training speakers. S fitting excludes gold omission examples by its declared specialist contract, but S scores are written for every held-out target. O scores all held-out targets. I score models fit their registered eligible gap rows and output held-out gap scores.
- The code constructs IDs from dataset/speaker/utterance/event-kind/index and checks duplicate IDs. It preallocates all TRAIN score vectors, writes held-out-fold predictions, and rejects any output vector still containing NaNs. This establishes an implementation-level coverage check; the existing run receipt supplies the published event counts and artifact hashes.
- All four sources (S, O, I-F1, I-precision) have three fold checkpoints in `folds_v2_correct_i_sources/`. These are economical replicas (eight epochs/fold), not replicas of each champion's full training budget.
- Frozen full-TRAIN specialist checkpoints are used to score DEV_CAL and DEV_SEARCH. Those development scores are not OOF. DEV_SEARCH is exposed and cannot select the H-004 method.
- The fold code requires CUDA and records GTX 1650 / CUDA 12.4, seed 101, 21.85 seconds, and 22.23 MiB peak allocated VRAM in the existing provenance JSON. This is a historical CUDA receipt, not current GPU availability.

## Material provenance caveat: calibrated OOF pseudo-targets

The raw `teacher_scores_v2_train.npz` scores are speaker-held-out model predictions. However, the v2 script fits isotonic mappings on DEV_CAL labels and applies those mappings to the OOF TRAIN teacher scores to create `probs_*_calibrated`. Those calibrated TRAIN values are therefore **not label-independent TRAIN-only teacher targets**: the mappings were fit using DEV_CAL supervision, then reused to construct the student training targets. The provenance accurately describes this as an extrapolated mapping; it must not be conflated with raw OOF validity.

For any new H-004 primary KD comparison, use raw TRAIN OOF teacher scores, or fit any score transform solely inside each outer TRAIN speaker fold using only that fold's in-fold training speakers. Reserve DEV_CAL for checkpoint/calibrator/policy selection according to a frozen protocol. Any reuse of DEV_CAL-fitted teacher mappings as TRAIN targets belongs in a separately labeled sensitivity arm, not the clean primary comparison.

Other inherited limits: no frame-level acoustic representation in these teacher caches; OOF replicas have a shortened training budget; isotonic calibration can improve in-sample DEV_CAL reliability while changing AP and degrading ranking on DEV_SEARCH; and the four-speaker SEARCH cohort remains exploratory.

## Evidence hashes checked

`sha256sum` was run on the current local files:

| Artifact | SHA-256 | Match to v2 provenance |
|---|---|---|
| `data/derived/l2arctic/manifests/train.jsonl` | `1f4da8518d9b184fb737b93c7887ecd7c3e73d6e83260e54651074831f61bcff` | Match |
| `data/derived/l2arctic/features/train_features.npz` | `e85cc2844c5b5b785e9f3b8b9b424ec60c5979829a1511a6fd804d1761ec9fd1` | Match |
| `research/student/kd/run_teacher_oof.py` | `215f2fc0d852ddcbebe74fc5c86984f3b7694045cb815376b8432db13a41655c` | Match |
| `experiments/student_training/kd/teacher_scores_v2_train.npz` | `d8c4376fdf7e413fbebf668cd8ef51670088bcedecbaad93ee156378b5fe14e2` | Match |
| `experiments/student_training/kd/provenance_v2_correct_i_sources.json` | `9de40375a5d7fe6d1985519671402e35e81f2979fe41d3fd8deb4c33ae181a88` | File hash checked; not embedded in itself |

Checkpoint hashes listed in the provenance were not recomputed in this pass. No new checkpoint or prediction artifact was created.

## Smallest defensible Cycle 2 comparisons

All neural student arms require CUDA and should use matched architecture, TRAIN population, seed, update budget and DEV_CAL checkpoint rule. Run at least seeds 101/202/303 for any serious repeatability claim; preserve every seed.

| Arm | Intervention | Isolation / interpretation |
|---|---|---|
| H004-A | Gold-Only student | Existing matched control; preserve historical checkpoints |
| H004-B | Original scalar KD using raw speaker-grouped OOF probabilities | Rebuild only if necessary; binary S/O/I heads remain task-independent; no DEV_CAL-fitted map applied to TRAIN |
| H004-C | One teacher-to-student feature loss, one task at a time | Requires accessible, semantically aligned hidden features on the same events; compare against B with same student/budget |
| H004-D | Disagreement-aware sample weighting/loss for raw teacher scores | Predefine disagreement statistic and direction using TRAIN folds; do not treat correlated compositions as extra teachers |
| H004-E | Specialist-head multi-task student | Keep S/O target probabilities as independent binary teacher signals; gold target remains mutually exclusive CORRECT/S/O; gaps remain independent |
| H004-F | Acoustic-aware teacher gate | Gate must consume permitted audio-derived student features; compare against the same ungated teacher loss; no gold label, speaker ID, or SEARCH-tuned gate |

Primary first causal comparison should be Gold-Only vs scalar raw OOF KD, because it verifies the clean baseline contract before adding feature transfer or disagreement. Feature, disagreement, specialist-head, and gate objectives each require separate single-intervention arms; do not bundle them and attribute a joint result to any one mechanism. Evaluate with EVAL-V3 on full matched events; report per-class AP/PR, class precision/recall/F1, Macro-3 Error AP/F1, Macro-4 F1, raw calibration, speaker summaries, teacher coverage, score correlations, and compute cost. Select any hyperparameters only inside speaker-grouped TRAIN validation. Do not use SEARCH to choose a winner.

## Verdict

The historic Phase 3 raw OOF pipeline satisfies a credible **speaker-held-out score-generation contract** for four specialist replicas, with fold-wise normalization and event coverage checks. It does not prove OOF parity with full champion training, and its calibrated TRAIN variant incorporates DEV_CAL-label-fitted mappings. Phase 3's matched Gold-Only/KD-All results were broadly comparable across three seeds; no KD superiority was established. H-004 is therefore **UNRESOLVED for advanced transfer**, and Cycle 2 has not tested feature-level distillation, disagreement learning, or an audio-aware gate. No hypothesis support/rejection claim is made.

## Reproduction / safe review

Read-only provenance/hash review:

```bash
cd /home/khoa/Projects/New-Method-Research
sed -n '1,260p' research/student/kd/run_teacher_oof.py
cat experiments/student_training/kd/provenance_v2_correct_i_sources.json
sha256sum data/derived/l2arctic/manifests/train.jsonl data/derived/l2arctic/features/train_features.npz research/student/kd/run_teacher_oof.py experiments/student_training/kd/teacher_scores_v2_train.npz
```

Do not rerun `run_teacher_oof.py` unless the CUDA lease is available: the script asserts `torch.cuda.is_available()` and writes checkpoints/scores into existing Phase 3 paths. New Cycle 2 training must use unique output paths and a fresh, registered runner/receipt.
