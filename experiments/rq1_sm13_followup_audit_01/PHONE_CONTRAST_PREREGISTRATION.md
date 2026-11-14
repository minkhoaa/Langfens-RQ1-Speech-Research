# TRAIN-only phone-contrast specialist pilot

**Frozen before fitting:** 2026-11-13. This is a development experiment; even a pass requires fresh-speaker confirmation.

## Question and scope

Can phone-conditioned binary specialists improve useful low-warning recall for the high-support confusions **IH→IY, AH→AO, OW→AO, D→T, and T→D** while keeping pooled vowel/stop recall, precision and joint false-warning rate within the agreed gate? These five directed pairs reflect the dominant misses in the frozen-score phone audit. They are only a subset; results do not establish performance on every vowel or stop.

## Fixed candidate

- Use TRAIN event labels and the existing 51D static acoustic/phone feature matrix. No manual boundaries, raw `*` annotations, or held-speaker labels enter fitting or calibration.
- For each listed target phone, fit a binary target-phone SUBSTITUTION vs all other target-phone event classifier. The pair's realized-phone identity is **not** an input or fitting stratum; otherwise the model would use the answer being predicted.
- HGB configuration and square-root inverse-prevalence weighting match the prior static specialist. One fixed seed (101), no tuning.
- Produce one out-of-fold score per event from the same four speaker groups. Each candidate pair uses an inner-speaker threshold selected at precision ≥50%; the five thresholds are then jointly capped at 0.08 incremental FP per utterance on inner OOF. The thresholds transfer unchanged to outer held speakers. Pair-specific alerts pass through existing S/O margin arbitration, and all frozen S/O/I streams count toward realized FWR.
- Pair scores can only alert events whose target phone is that pair's first phone. When two targets ever overlap (they do not for this list), use the largest calibrated margin. Other phones retain frozen SM13 scores. No test/dev data.

## Gate and reporting

For the combined phone-specific policy, require pooled precision ≥50% and recall ≥15% **separately** for Vowel and Stop, realized pooled FWR ≤0.20, and no outer fold above 0.20. Report every fold and each of the five pair's TP, FP, support, recall and precision. Compare to the unchanged frozen policy and static 51D manner specialist under identical outer groups. If any gate fails, do not promote; use the confusion table to identify whether the limitation is weak pair ranking, score transfer, or remaining warning budget. No threshold may be selected from outer held results.
