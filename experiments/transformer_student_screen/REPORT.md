# Transformer student architecture screen

## Protocol

Seed 101; 12 epochs; batch size 32; AdamW (lr 0.001, weight decay 1e-4); CUDA GTX 1650. Model: two 64D TransformerEncoder layers, four heads, 128D feed-forward, dropout 0.1. Inputs are existing intended-reference-only 51D target features; class logits are 3-way and gap logits preserve N+1 slots from adjacent encoded target states plus existing 6D gap features. Acoustic normalization and sqrt-inverse target/gap class weights are derived from TRAIN only. Fitting uses TRAIN only. Checkpoint selection is minimum unweighted target+gap CE on DEV_CAL; EVAL-V3 policy is fit on DEV_CAL and DEV_SEARCH is evaluated once.

## DEV_SEARCH result

| Metric | Transformer | Matched seed-101 GRU-64 |
|---|---:|---:|
| Substitution AP | 0.33161 | 0.36460 |
| Omission AP | 0.11096 | 0.15943 |
| Exact-gap insertion AP | 0.13465 | 0.18672 |
| Macro3 AP (S/O/I) | 0.19241 | 0.23692 |
| Correct F1 | 0.87685 | 0.88269 |
| Substitution F1 | 0.37391 | 0.38659 |
| Omission F1 | 0.18896 | 0.25101 |
| Insertion F1 | 0.18227 | 0.26282 |
| Macro3 F1 | 0.24838 | 0.30014 |
| Macro4 mixed-domain F1 | 0.40549 | 0.44578 |
| Raw joint FWR | 6.24541 | 5.73790 |

The Transformer does not improve any requested aggregate error metric over the matched GRU reference, and its Macro3 AP is lower by 0.04451. EVAL-V3 coverage is complete (19,755 targets; 20,354 gaps; no missing predictions). These are development-screen results, not protected evaluation.

## Reproducibility and next step

Selected epoch 9; runtime 45.04 seconds; checkpoint SHA-256 `a2470e316fd8785e925b132a23621b529eede4c96373cdb21b27f5f4215ad3c9`. Full results, policy, and class metrics are in `results.json`; saved predictions and checkpoint are alongside this report.

Next step: retain the seed-101 GRU as the architecture baseline and do not promote this Transformer. If pursuing self-attention further, first diagnose the same-run DEV_CAL target/gap validation trajectory and encoder masking/position handling before committing to another screen; no additional hyperparameter search is justified by this negative single-seed result.

Limitations: one seed and DEV_CAL reuse for checkpoint/policy selection; DEV_SEARCH is development-exposed. Raw FWR is reported without FWR filtering.
