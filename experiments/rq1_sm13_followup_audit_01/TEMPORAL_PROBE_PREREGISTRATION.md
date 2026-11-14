# TRAIN-only spectrotemporal evidence probe: fixed pilot plan

**Frozen before scoring:** 2026-11-13  
**Question:** Does explicit local spectral trajectory information improve vowel and stop substitution ranking at low false-warning risk beyond the existing static CharSiu features, and does it approach the SM13 operating target?

## Data and candidate

- Population: exact 1,800 TRAIN utterances, 12 speakers, 59,562 target slots, original four speaker-held-out outer folds. No DEV or TEST files may be loaded.
- Audio anchor: the **predicted CharSiu center** already cached for SM13. Gold manual boundaries and raw `*` labels are excluded from model inputs.
- Temporal evidence: seven 32 ms Hann windows centered at offsets `[-120, -80, -40, 0, +40, +80, +120]` ms. For each window, compute 16 fixed triangular mel-band log powers, log RMS, and zero-crossing rate. Add adjacent-window mel differences. This yields 222 temporal dimensions.
- Static control: the existing 51D `target_X` (39D target-phone one-hot + 12 CharSiu acoustic summaries). Temporal candidate: same 51D plus the 222D trajectory. Both use identical per-manner binary `SUBSTITUTION` versus other target labels, the same `HistGradientBoostingClassifier` configuration (150 iterations, learning rate 0.05, 15 leaves, 50 minimum leaf rows, L2 1.0, no early stopping), and train-fold-derived square-root inverse-prevalence positive weights. No target-phone or error labels from held speakers enter fitting.
- SM13 reference: archived seed-101 outer-OOF Neural-S scores. Fixed-O, gap-I, and other-manner S alerts are held at archived scores for operating-point comparisons.

## Measurements and decision

Primary: pooled and each outer-fold AP for vowel and stop, plus exact retrospective single-manner threshold frontier subject to joint FWR ≤0.20 and manner precision ≥50%. Report TP, FP, recall, precision, and FWR, including the no-warning case. Same-data threshold search is explicitly an optimistic screening bound, not deployment calibration.

Stage 1 is a **single fixed pilot**. If the temporal candidate does not improve both manner AP and low-risk attainable recall relative to the static control, stop this feature recipe. If it improves and reaches the illustrative 15% recall / 50% precision gate for both manners, run a separate fully nested threshold-selection study, at least three fixed seeds where stochasticity applies, and prospective confirmation on new speakers. A positive pooled AP alone is insufficient.

The historical [EXP-68 temporal-pyramid run](../EXP-68-MULTI-RESOLUTION-TEMPORAL-PYRAMID-NET/RUN_SPEC.yaml) reported FWR 1.75 on DEV_REPLICATION under a different protocol, far above the present 0.20 budget. Its reported metrics are not a matched comparator for this pilot.
