# METHOD-BENCH-07 — Verdicts on hypotheses

## Supported within the executed nested TRAIN experiments

- **RQ-A, E4 complementarity:** supported across two strict speaker-grouped nested seeds (101/202). Each seed used four outer × three inner folds, 12 CTC heads fit on six inner-training speakers, E4 competitor statistics fit only on those six speakers, exact-fold MB05 outer CTC/E4 scores, and inner-only transforms/meta fitting. All 59,562 outer target event IDs join uniquely; outer labels were evaluation-only.
- Learned neural+phone-prior+E4 fusion exceeded the nested direct H-002 neural base in **4/4 folds for both S and O at each seed**. Pooled seed101 AP-S/O `.42328/.32509` vs direct `.38408/.26565` (Δ +.03920/+.05943); seed202 `.42300/.32862` vs `.38017/.25608` (Δ +.04283/+.07254). It also exceeds fixed neural+prior in every fold for both classes.
- E4-only is weaker than neural-only (seed101 `.27222/.22901` vs `.38408/.26565`; seed202 `.28815/.23684` vs `.38017/.25608`). The evidence supports conditional E4 use, not standalone E4 superiority.
- A prespecified fixed `.5` neural+E4 simplex mean also improves pooled AP over neural: seed101 `.40274/.33078`; seed202 `.40207/.33400`. It wins S in 3/4 and O in 4/4 folds for seed101; seed202 fixed blend gains are broadly similar. Learned fusion’s pooled O AP is slightly below fixed blend in both seeds, despite higher mean fold AP; do not claim learned is uniformly best.
- The nested neural+prior integration without E4 is negative: learned seed101 `.36748/.21949` and fixed mean `.38251/.24001` are below direct neural. E4 is needed for the observed learned-fusion lift over this control.
- **Decision-policy evidence is more mixed than AP:** target-only warning-budget thresholds selected inner-OOF and evaluated outer at 0.2/0.5/1/2 warnings per utterance. Fixed neural+E4 has the highest target Macro-2 F1 at all four budgets in both seeds. Learned fusion beats neural-only at budgets 1 and 2, but not consistently at 0.2/0.5. This is not a complete S/O/I policy because no insertion warnings are included.
- The temporal Conv1d on rank-warped Wav2Vec2 frames has AP-I `.01283`, only slightly above prevalence `.00913`, below static-local `.01147` and far below neural insertion `.162125`. This rejects the rank-warp + 9-frame CNN recipe, not alignment modeling generally.
- The phonological-attribute mismatch adaptation reached AP-S/O `.23462/.12643`, above prevalence but below neural/E4 historical controls. It is not a faithful reproduction.

## Refuted at the tested scope

- A rank-warped temporal CNN improves exact-gap insertion: unsupported; top-560 precision `.01964` is lower than static `.02143` and neural `.24286`.
- The tested phonological attribute adaptation competes with direct neural target detection: unsupported.
- Phone priors alone or neural+phone-prior composition explain the nested E4 ranking gains: contradicted by the fold-matched controls, though attribution is limited to this meta-model and does not establish universal acoustic independence.

## Still inconclusive / not tested

- E4 normalization-choice sensitivity, external speakers and independent blind validation. The two seeds repeat the same four TRAIN speaker groups, not an external cohort.
- Probability calibration/reliability. Inner-only thresholding is a warning-budget policy, not a calibrated posterior claim.
- Full S/O/I warning policy, insertion-inclusive FWR/Macro-3 F1. The target-only operating analysis does not invent gap scores.
- Whether alignment uncertainty/explicit insertion-state localization improves AP-I. Current rank-warp model did not estimate alignment.
- Phonological attributes under faithful sequence alignment.
- A unified deployed architecture.

## Decision

The nested ranking gain is replicated across two seeds and all four folds, but the target-only operating comparison favors the fixed E4 blend rather than learned fusion at every tested budget. Do **not** train a unified model yet. Next, study one reference-conditioned monotonic alignment with an explicit optional insertion state; then construct a complete warning evaluation with the same validated insertion scorer and inner-only policy selection. No DEV_SEARCH tuning or held-out gold timing.
