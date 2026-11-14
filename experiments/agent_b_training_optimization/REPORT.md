# Agent B — focal objective optimization

## Pipeline review and hypothesis

The current supervised reference is `research/student/train_light_student.py`: a 78,661-parameter bidirectional GRU-64 with separate target (3-way) and exact-gap (2-way) heads, AdamW (`lr=1e-3`, `weight_decay=1e-4`), 12 epochs/684 updates, TRAIN-only acoustic normalization, and square-root inverse class weights derived from TRAIN. Its selection checkpoint minimizes unweighted gold DEV_CAL target+gap cross-entropy. EVAL-V3 policy thresholds are fitted on DEV_CAL and the candidate is then evaluated on DEV_SEARCH. This is a robust, small, matched baseline; the obvious objective-level opportunity is to reduce domination by easy majority-class examples while retaining TRAIN-derived reweighting.

Tested one controlled objective change: modulate the same class-weighted cross-entropy by focal factor `(1-p_t)^gamma`, gamma=1.5, for both target and gap tasks. Architecture, seed, optimizer, updates, TRAIN-only normalization and class weights, and checkpoint selection criterion remain fixed. Compared with the existing seed-101 gold-only arm. No TRAIN labels outside TRAIN were used for weights; DEV_CAL was used only for unweighted-loss checkpoint selection and locked EVAL-V3 threshold fitting; DEV_SEARCH labels were not used for fitting or selection.

## EVAL-V3 DEV_SEARCH results

AP is ranking-based; F1 uses each arm's independently DEV_CAL-fitted `CAL_MAX_ERROR_MACRO_F1` policy. Full per-class counts/metrics and policy records are in `results.json`.

| Domain/class | Baseline AP | Baseline F1 | Focal AP | Focal F1 |
|---|---:|---:|---:|---:|
| Correct (target; secondary) | 0.975879 | 0.882695 | 0.975237 | 0.863409 |
| Substitution | 0.364602 | 0.386590 | 0.358542 | 0.367920 |
| Omission | 0.159433 | 0.251012 | 0.160967 | 0.264591 |
| Insertion (exact gap) | 0.186715 | 0.262821 | 0.200208 | 0.271186 |
| **Macro3 error** (S/O/I) | **0.236917 AP** | **0.300141 F1** | **0.239906 AP** | **0.301233 F1** |

Additional macros: macro3 target F1 0.506766 baseline vs 0.498640 focal; macro4 mixed-domain F1 0.445779 vs 0.441777 (not a four-way softmax). Thus macro3 error AP increased only +0.002989 and macro3 error F1 +0.001092, while macro4 F1 fell 0.004003. Results are mixed, not a demonstrated high-impact win.

DEV_CAL-fitted unbounded macro3-error policies: baseline thresholds S/O/I = 0.329358/0.610187/0.563435; focal = 0.389546/0.536561/0.516100. The CAL-selected policies had very high DEV_CAL joint FWR (5.29 and 6.05 warnings/utterance respectively); held-out SEARCH FWR was 5.738 baseline and 6.765 focal. Both exceed the 0.20 reference by a wide margin. They are macro-F1 operating points, not safe/deployable settings. The per-class scores/AP/F1 were evaluated across complete event populations (19,755 targets, 20,354 gaps, 599 utterances), no missing events or simplex violations.

## Run configuration and artifacts

- Objective: class-weighted focal loss, gamma 1.5, alpha weights = square-root inverse TRAIN frequency normalized to mean 1; summed target+gap losses.
- Matched setup: GRU-64, seed 101, 12 epochs, batch 32, AdamW 0.001 / 1e-4, 684 updates. Best epoch 11 by minimum unweighted DEV_CAL target+gap cross-entropy.
- Runtime: 47.846 s (end-to-end script runtime); NVIDIA GeForce GTX 1650; peak allocated VRAM 44.094 MiB.
- Checkpoint: `focal_gamma1p5_seed101.pt`, 320,342 bytes, SHA-256 `eed8b6d0a60efce69d223c9e00c4f76287d900fafc4874f7dea8b8497579c687`.
- Command: `.venv/bin/python coordination/phase3_gpu_job.py --job agent-b-focal-objective-final -- .venv/bin/python experiments/agent_b_training_optimization/train_focal_student.py > experiments/agent_b_training_optimization/train.log 2>&1`
- Run-specific files: `train_focal_student.py`, `train.log`, `results.json`, `focal_gamma1p5_seed101.pt`, `focal_dev_cal.npz`, `focal_dev_search.npz`. No shared model/training/evaluator code was modified.

## Negative findings and limits

- Focal loss shifted AP/F1 unevenly: insertion improved, omission slightly improved, substitution regressed, and the majority Correct F1 and macro4 F1 decreased. Aggregate macro3 changes are very small and a single seed cannot establish a causal/robust improvement.
- Macro3-error AP is ranking-based and the held-out class AP changes are not threshold effects; policy-specific F1/FWR are a separate CAL-fit/SEARCH-assess result.
- DEV_CAL is only two speakers (ERMS, THV); calibration/selection variance is substantial. This run used one seed and DEV_SEARCH remains development-exposed, not a blind final test.
- Both unbounded CAL macro-F1 policies yield extremely high SEARCH false-warning rates. This experiment is not deployment or safety evidence.
- No EVAL-V3 metric is reported for protected/sealed test data; `test_sealed` was not opened.

## Next experiment

Run a preregistered, matched-seed gamma comparison (gamma 0 vs 1 vs 2) with exact same TRAIN-only class weights, optimizer/update budget, DEV_CAL unweighted-loss checkpoint selection, and CAL-fitted policy. Compare seeds 101/202/303; use DEV_SEARCH only once for final comparison. Track macro3 error AP/F1, per-class AP/F1, macro4 F1, and risk-budgeted FWR, and reject the objective if substitution/macro4 regressions persist. Do not tune from sealed data.
