# Pairwise-ranking auxiliary loss experiment

Seed-101, bidirectional GRU-64; unchanged data/features, TRAIN-only normalization and square-root inverse class weights; AdamW (0.001, 1e-4), batch 32, 12 epochs on NVIDIA GeForce GTX 1650. Added predeclared `lambda=0.1` times the mean pairwise logistic loss for target substitution and omission log-odds and exact-gap insertion log-odds. Each minibatch uses deterministic sorted-index pairs with at most 128 negatives per positive; classes without a positive contribute no loss. No FWR filter was applied. Checkpoint and candidate selection used only maximum DEV_CAL Macro3 error AP; the selected checkpoint was calibrated with `CAL_MAX_ERROR_MACRO_F1` on DEV_CAL and evaluated once on DEV_SEARCH.

## Selected DEV_SEARCH metrics

| Class | AP | F1 |
|---|---:|---:|
| SUBSTITUTION | 0.3651 | 0.3837 |
| OMISSION | 0.1564 | 0.2618 |
| INSERTION | 0.1773 | 0.2485 |
| Macro3 error | 0.2330 | 0.2980 |

Macro3 target F1 is 0.5087. Unfiltered joint FWR is **5.9299**. The DEV_CAL thresholds were substitution 0.3273845, omission 0.5468582, insertion 0.7315081.

## Matched seed-101 baseline

The matched CE-only baseline is factor 1 from `omission_weight_sweep` (same seed, GRU-64, optimizer, features and data partition). Its reported DEV_SEARCH class AP/F1 was substitution 0.3646/0.3866, omission 0.1594/0.2510, insertion 0.1867/0.2628, Macro3 error AP/F1 0.2369/0.3001, and unfiltered joint FWR 5.7379. Baseline checkpoint selection used DEV_CAL CE, whereas the present run predeclared DEV_CAL Macro3 AP selection, so this is a matched training setup but not identical checkpoint-selection rule. Pairwise result changes relative to reported baseline: Macro3 AP -0.0039, Macro3 F1 -0.0021, and FWR +0.1920; class-level differences are shown in the tables.

## Run receipt

- DEV_CAL Macro3 AP: **0.6515767**; selected epoch **9** (earliest epoch wins ties).
- DEV_SEARCH evaluation count: 1.
- Runtime: **177.61 seconds**.
- Checkpoint: `pairwise_lambda_0p1_seed101.pt` (313 KB), SHA-256 `2fe8ab66161cd56d1d600b449fe5ea6b4a34d0ab80ca028d3387c9727103b15b`.
- Full per-class and evaluator results, epoch history, and prediction files are retained in this directory.
