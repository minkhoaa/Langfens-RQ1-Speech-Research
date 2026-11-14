# H-004 Cycle 2 screen: raw speaker-OOF specialist KD

**Experiment:** `RC2-H004-RAW-OOF-KD-CUDA130-S101-R4` — `EXECUTED`; numerical metrics are exploratory, not an H-004 victory.

## Question and intervention

A fresh Gold-Only CUDA130 seed-101 control was already trained in this cycle. This arm changed only the training objective: it added the Phase 3 raw, event-ID joined, three-speaker-fold OOF specialist probabilities for S, O, I-F1 and I-precision. It used the same 78,661-parameter student, data, seed, epochs, update count, class weights, and DEV_CAL gold-loss checkpoint rule. Four probabilities remain four separate binary tasks: S/O target marginals use the full 3-way student softmax; I teachers apply to independent exact-gap events. No simplex was fabricated. No DEV_CAL-fitted calibrated teacher targets were used.

This is a clean raw multi-teacher KD re-screen. It does **not** test feature-level transfer, disagreement weighting, or new teacher architectures. A separate OOF S/O hidden-state cache and feature-KD arm were subsequently executed with a matched zero-loss control; detailed results follow below. CPU hidden-state preflight files are retained but disqualified from model training.

## Provenance and execution

- Runtime: `/home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python`, PyTorch 2.14.1+cu130, CUDA 13.0, GTX 1650.
- Source: `code/train_raw_oof_kd.py`; copied from Phase 3 KD trainer with only unique RC2 output paths and import/environment fixes.
- Source SHA-256 at final source state: `192e60438abf41f24c2bc9f53a0ea881d5695659c51fa6a782a3bb2c1605f979`; the training run used the same logic with import fixes. Training-execution source revision was not separately hashed before the run.
- Gold-Only source script SHA at preflight: `80c506b8118d4e210ba75e66674d31fd041481c99de775482f479104983ce11b`.
- TRAIN feature SHA-256: `e85cc2844c5b5b785e9f3b8b9b424ec60c5979829a1511a6fd804d1761ec9fd1`.
- Raw grouped OOF teacher cache SHA-256: `d8c4376fdf7e413fbebf668cd8ef51670088bcedecbaad93ee156378b5fe14e2`.
- Teacher procedure: 3 speaker-grouped folds; raw held-out TRAIN scores only. The separate calibrated cache applies mappings fitted on DEV_CAL and was not used.
- Training: seed 101, 12 epochs, 684 updates, 85.09 s, 44.14 MiB peak allocated VRAM, 78,661 params. CUDA checkpoint SHA-256 `f1df7b2724090718d61ded519b5a6965dd400adee7775b52e8f1d93eea0bbc01`.
- Checkpoint: `checkpoints/RC2-H004-RAW-OOF-KD-CUDA130_seed101.pt`.
- Search predictions: `predictions/RC2-H004-RAW-OOF-KD-CUDA130_seed101_dev_search.npz`, SHA-256 `7fe351e400fc463898772c65f893e1bedc40bd1b75b1f872ec46328cff7bd4c1`.
- CAL predictions SHA-256 `cfd1ab9cb9027d34517ffbaf2a0e8d85066068fb675dd752a9446fc80c4b7ff3`.
- EVAL-V3 output: `RC2-H004-RAW-OOF-KD-CUDA130-S101-eval.json`; full population 19,755 targets and 20,354 exact gaps (599 utterances).
- GPU lease receipt: `coordination/PHASE3_GPU_LEDGER.jsonl`, final event for job `...-R4`, return code 0; device PID observed, peak sampled utilization 58%, lease released.
- Initial two job attempts failed before training because `.venv-cuda` lacks sklearn and the script import path was incomplete; both returned code 1 and released leases. The third attempt fixed both issues and succeeded. No failed output was reused.

## EVAL-V3 results

DEV_SEARCH is exposed development data; these are descriptive only. For comparison, the fresh same-cycle Gold-Only seed-101 `.venv-cuda` control had AP S/O/I `0.364602 / 0.159433 / 0.186715`, Macro-3 Error AP `0.236917`. This KD run produced:

| Metric | Gold-Only control | Raw OOF multi-teacher KD | Delta |
|---|---:|---:|---:|
| AP S | 0.364602 | 0.368655 | +0.004053 |
| AP O | 0.159433 | 0.158587 | -0.000846 |
| Exact-gap AP I | 0.186715 | 0.192241 | +0.005526 |
| Macro-3 Error AP | 0.236917 | 0.239828 | +0.002911 |

At the EVAL-V3 DEV_CAL-selected policy with the 0.20 operating budget, KD had Macro-3 Error F1 `0.104513` versus Gold-Only `0.104897` (slightly lower); at 0.05 it was `0.029851` versus `0.025063`. class P/R/F1: S `1.000/0.0017/0.0034`, O `0.118/0.0096/0.0177`, exact-gap I `0.356/0.248/0.292`. Single-seed small AP deltas do not establish reliable improvement. Report FWR as a secondary policy behavior, not a criterion to accept/reject a ranking method.

## Decision

**Raw multi-teacher KD: unresolved; small mixed AP deltas, with 0.20-budget Macro-3 F1 slightly below control. Feature-level S/O KD: not supported exploratory; all three AP values declined versus the matched control in one seed. Disagreement-aware learning remains untested. No H-004 component merits integration on this evidence.** Preserve the one-seed result and all event populations.

## Reproduction

```bash
.venv/bin/python coordination/phase3_gpu_job.py --job RC2-H004-RAW-OOF-KD-CUDA130-S101-R4 -- /home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python -u research/long_running_method_discovery/cycle_02/h004_multiteacher/code/train_raw_oof_kd.py --arm RC2-H004-RAW-OOF-KD-CUDA130 --epochs 12 --seed 101 --teachers S,O,IF1,IPREC
.venv/bin/python experiments/student_evaluation/evaluate_student_predictions.py --dev-cal research/long_running_method_discovery/cycle_02/h004_multiteacher/predictions/RC2-H004-RAW-OOF-KD-CUDA130_seed101_dev_cal.npz --dev-search research/long_running_method_discovery/cycle_02/h004_multiteacher/predictions/RC2-H004-RAW-OOF-KD-CUDA130_seed101_dev_search.npz --dev-cal-manifest data/derived/l2arctic/manifests/dev_cal.jsonl --dev-search-manifest data/derived/l2arctic/manifests/dev_search.jsonl --output research/long_running_method_discovery/cycle_02/h004_multiteacher/RC2-H004-RAW-OOF-KD-CUDA130-S101-eval.json
```

## H-004-C feature-level KD: controlled one-seed screen

After H-002 released the GPU lease, S/O teacher embeddings were extracted with `.venv-cuda` under wrapper job `RC2-H004-OOF-TEACHER-FEATURE-EXTRACT-CUDA130-R2` (return code 0; 6.45 s; 14.82 MiB peak allocated). TRAIN uses each target event's speaker-held-out OOF S/O fold checkpoint; DEV_CAL and DEV_SEARCH use frozen full-TRAIN champions. The S target is the L2-normalized 128D `TeacherS_ChampionNet.encoder` output. The O target is the L2-normalized 64D penultimate activation (`TeacherO_ChampionNet.encoder[:-1]`). These were not logits or class posteriors.

The learner uses normalized Gold-Only target hidden state (128D) and a cosine-distance loss against S; for O it uses the normalized first 64 target-hidden dimensions. This is a deliberately small exploratory alignment, with no scalar KD. Feature loss weight was 0.1. A same-seed, same-architecture, same-budget zero-feature-weight control was run. Both have 78,661 parameters, 12 epochs and 684 updates; selection uses the same minimum unweighted Gold DEV_CAL target+gap CE. Teacher feature sources are target-level S/O only; I's exact-gap feature path was not included in this intervention.

The first CUDA embedding-extraction attempt failed strict state loading before a forward because the hand-copied O architecture had an extra BatchNorm/GELU pair. Inspection of the saved `o_fold0.pt` state keys showed `encoder.8` Linear128→64, `.9` BatchNorm64, `.10` GELU and `.11` Linear64→2. The corrected inline class matched strictly and produced all three split caches. The first feature-KD training attempt then spent >2 minutes repeatedly indexing compressed `.npz` arrays inside per-event comprehensions, expanding RSS to ~21 GiB before epoch 1. I interrupted only that owned child, released the lease, retained the failure record, pre-cached exact joined arrays outside the lease and rewrote the runner to read each array once and slice by verified metadata order. The corrected new-ID run succeeded. No output from either failed attempt was used.

| DEV_SEARCH AP | Gold/zero-feature control | Feature KD weight 0.1 | Delta |
|---|---:|---:|---:|
| S | 0.364602 | 0.358274 | -0.006328 |
| O | 0.159433 | 0.156798 | -0.002635 |
| Exact-gap I | 0.186715 | 0.178040 | -0.008675 |
| Macro-3 Error AP | 0.236917 | 0.231038 | -0.005879 |

All three class APs declined, so this tested feature-level KD intervention is **NOT_SUPPORTED_EXPLORATORY**. At the EVAL-V3 DEV_CAL-selected policy budget 0.20, Macro-3 Error F1 was `0.093246` vs `0.104897` control; Macro-4 F1 was `0.306718` vs `0.315438`. Budget-level Macro-3 F1 (treatment vs control) was: 0.05 `0.069530 vs 0.025063`; 0.10 `0.079054 vs 0.034250`; 0.20 `0.093246 vs 0.104897`; 0.50 `0.165515 vs 0.163591`. This threshold behavior is mixed and doesn't overcome the AP decline. FWR remains an ancillary reported operating metric.

Treatment run `RC2-H004-FEATURE-KD-CUDA130-S101-R2`: 47.03 s, 48.78 MiB peak allocated, checkpoint SHA-256 `cb15339540dc86b5732ade995df3a5b43ce8ce518a2e53246d5a2f11580db192`. Control run `RC2-H004-FEATURE-KD-ARCH-CONTROL-CUDA130-S101`: 47.53 s, 48.78 MiB, checkpoint SHA-256 `8ccd02aaf0385ba66f0f0be46675385e81254f8769b29812057834929cec917d`. Both wrapper jobs returned 0 and released leases. TRAIN/CAL/SEARCH populations were 59,562 / 10,043 / 19,755 targets; EVAL-V3 included all 20,354 SEARCH exact-gap events and 599 utterances.

Feature data hashes: CUDA OOF TRAIN embedding cache `af955b5ab8fe7c88c06f8842d9b098b9c414f0326f4be8b0e5170e93a881e7d1`; prepared joined TRAIN arrays `582ffdc8b608e5927174940eaab63389f957c0c7cfdd3d03aa687237b09ba3cd`; EVAL-V3 treatment JSON `research/long_running_method_discovery/cycle_02/h004_multiteacher/RC2-H004-FEATURE-KD-CUDA130-V2-S101-eval.json`; SEARCH predictions SHA-256 `9d431ac7d646d4d40faf043fc221009fce6edd740fcaa89cc66ecfd9bff24a9c`. Teacher full checkpoint hashes: S `47e804011f6fa744bf09dbd13896cfcd9ef874534d9eaa9f403e37265b000510`; O `80618baaefe59eff0da3db0a9a64941fb2bdb1a10146c60ee93b286f9634db76`. Fold-0 OOF checkpoint hashes: S `c3b7459969e9a0430f17986c6f5871f43f48f56c83e6da2be4155b7b97012c87`; O `36a0c8952fa9b112856c34a904439cc3d8912b4a59fe1db93563d4ad67ded7f4`.

Reproduce CUDA feature extraction, one joined-data CPU prep, candidate, and matched control:

```bash
.venv/bin/python coordination/phase3_gpu_job.py --job RC2-H004-OOF-TEACHER-FEATURE-EXTRACT-CUDA130-R2 -- /home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python -u research/long_running_method_discovery/cycle_02/h004_multiteacher/code/extract_cuda_teacher_features.py
.venv/bin/python research/long_running_method_discovery/cycle_02/h004_multiteacher/code/prepare_feature_kd_arrays.py
.venv/bin/python coordination/phase3_gpu_job.py --job RC2-H004-FEATURE-KD-CUDA130-S101-R2 -- /home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python -u research/long_running_method_discovery/cycle_02/h004_multiteacher/code/train_feature_kd_student.py --arm RC2-H004-FEATURE-KD-CUDA130-V2 --epochs 12 --seed 101 --feature-weight 0.1
.venv/bin/python coordination/phase3_gpu_job.py --job RC2-H004-FEATURE-KD-ARCH-CONTROL-CUDA130-S101 -- /home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python -u research/long_running_method_discovery/cycle_02/h004_multiteacher/code/train_feature_kd_student.py --arm RC2-H004-FEATURE-KD-ARCH-CONTROL-CUDA130 --epochs 12 --seed 101 --feature-weight 0
```
