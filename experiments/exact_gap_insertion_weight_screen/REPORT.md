# Exact-gap insertion positive-weight screening (seed 101)

## Design
A single controlled screening run used the existing lightweight gold-only, bidirectional GRU-64 with seed 101 and 12 epochs. TRAIN alone supplied normalization statistics, target class weights, and gap counts. The target-head loss remained the original sqrt-inverse TRAIN-frequency weighted cross-entropy. Only gap-head class weighting changed: `[negative=1, positive=min(TRAIN negatives / TRAIN positives, 10)]`; TRAIN gap counts were 60,802 negatives and 560 insertion positives, so the capped insertion weight was 10. AdamW used lr 0.001 and weight decay 1e-4, batch size 32. Checkpoint selection used unweighted target+gap CE on DEV_CAL (epoch 11 of 12). The predeclared EVAL-V3 policy was selected on DEV_CAL only; EVAL-V3 was then run once on DEV_SEARCH. No FWR<=0.20 filter was applied.

## DEV_SEARCH result (EVAL-V3)
- Exact-gap insertion: AP **0.1873**, F1 **0.2587** (41 TP / 151 FP / 84 FN).
- Substitution: AP **0.3648**, F1 **0.3873**.
- Omission: AP **0.1593**, F1 **0.2540**.
- Error macro (S/O/I) AP **0.2371**, F1 **0.3000**.
- Mixed-domain macro-4 F1 **0.4458** (includes CORRECT; not a four-way softmax).
- Joint FWR **5.7362**, above 0.20; reported as-is, not filtered.

The selected DEV_CAL policy was `CAL_MAX_ERROR_MACRO_F1`, with thresholds S=.33035, O=.60800, I=.54855; it yielded DEV_CAL error macro-F1 .45542. Class metrics and details are in `results.json`.

## Runtime and artifacts
GTX 1650 CUDA runtime was 182.08 seconds (including data loading, training, inference and evaluation); peak allocated VRAM 44.09 MiB. The selected checkpoint is `gap_positive_capped_ratio_seed101.pt` (320,571 bytes; SHA-256 `22cbc2f97f60b74f35e1962c010b01ac87c8e50631d2c33bd2dba91eb3a6b416`). Exact command: `source .venv/bin/activate && python experiments/exact_gap_insertion_weight_screen/run.py`.

## Limitations and next experiment
This is one seed and one weight, so it screens the intervention but cannot establish a weight-response trend or robust improvement. The capped ratio deliberately imposes a strong positive emphasis (10x); no claim of improvement over other matched arms is made here. DEV_SEARCH is a single held-out split. Next, preregister a compact weight sweep (for example positive weights 1, 3, 10, and capped TRAIN ratio), hold architecture/seed/training fixed, choose the candidate solely on DEV_CAL insertion AP or a declared macro objective, then evaluate that single selection on DEV_SEARCH once. Keep target loss unchanged and report FWR without filtering.
