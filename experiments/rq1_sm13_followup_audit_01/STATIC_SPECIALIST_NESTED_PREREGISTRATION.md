# Development transfer check for the static 51D specialist

**Frozen before this run:** 2026-11-13. The candidate was selected after viewing the exploratory [temporal probe](TEMPORAL_PROBE_PREREGISTRATION.md), so this is a **development-exposed transfer check**, not independent confirmation.

## Fixed candidate and threshold procedure

- Candidate: the same 51D `target_X` per-manner binary HistGradientBoostingClassifier configuration used in the temporal probe. No hyperparameters or input features are changed.
- Four original outer speaker groups; each outer model is already fit on nine speakers and scored on three held speakers in `TEMPORAL_PROBE_OOF.npz`.
- For each outer fold, train three additional inner models per manner on six of the nine outer-training speakers, score the remaining three, and aggregate nine-speaker inner OOF scores. No outer-held speaker labels or feature rows enter these fits or threshold selection.
- Select one vowel and one stop threshold **jointly** on inner OOF. Each manner must have inner precision ≥50% if it emits alerts. The two manners together receive at most **0.08 extra false warnings per inner-training utterance**, reserving 0.12 for frozen other-manner S, O, and I streams. Choose the pair maximizing the smaller of the two manner recall fractions; break ties by summed TP and then fewer extra FP. If no positive-precision option exists, the no-warning threshold is allowed.
- Apply both thresholds unchanged to the three outer-held speakers using their outer-OOF scores. Score-margin S/O arbitration and all frozen other alert streams follow the existing evaluator. Report per-fold and pooled TP, FP, precision, recall, and **realized** joint FWR. No threshold is picked on outer-held results.

The illustrative success gate is ≥15% recall and ≥50% precision **for both vowel and stop**, with realized joint FWR ≤0.20. A one-seed TRAIN result passing this gate would still require multi-seed replication and genuinely new speakers; failure stops promotion of this candidate. Historical TEST and DEV_REPLICATION are not used.
