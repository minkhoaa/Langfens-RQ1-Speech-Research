# Corrected RQ1-TRN-04R factorial comparison

**Real-data comparison status: PENDING.** This report contains no historical or guessed results. The coordinator runs the independent accounting CLI once after freezing integrated source and the immutable prediction exchange. Until then no numerical arm ranking, gate disposition or scientific conclusion is asserted.

## Analysis contract

Each cell is scored separately from the same frozen VAL rows and CAL threshold rows. C0/T2 use `operation_ce`; T1/T3 use `class_focal`. All four three-way objectives must supply `s_omit`/`s_sub` separately from `s_error`; missing class-focal subtype channels fail validation. Binary and legacy omission/substitution AP use `s_error`; dedicated omission/substitution AP use `s_omit`/`s_sub`, respectively. Dedicated subtype metrics are null/unavailable only for an explicitly `binary_collapsed` B-Control. AP uses exact grouped ties. CAL correct-only thresholds at FPR ceilings 5%, 10%, and 15% are reported independently from retrospective VAL curves. Operational omission alarms use the full 9,956-position gold census; prompt-only extras are a distinct, out-of-census alarm group.

The primary matched cell contrasts are **T1−C0, T2−C0, T3−C0**. The factorial interaction is **T3−T2−T1+C0**. Every contrast reports endpoint, population, score semantics, coverage and denominator. Missing or unsupported values make that contrast unavailable, not zero.

## Gates (reported separately from contrasts)

Frozen Protocol Z absolute gates are evaluated for each cell after row accounting passes:

- **Level A:** dedicated `s_omit` precision >= .25; all-gold recall >= .30; false alarms/1,000 full gold <= 30; binary AP >= .313619; legacy substitution AP >= .303974; PER <= .246980.
- **Level B:** dedicated `s_omit` precision >= .30; all-gold recall >= .40; false alarms/1,000 full gold <= 15; the same binary AP, legacy substitution AP and PER noninferiority thresholds; matched-minus-mismatched binary AP >= .010.

These absolute external-reference gates are separate from T1-C0/T2-C0/T3-C0 contrasts. Missing required metrics make a gate unavailable, not passed. Coverage/accounting checks (actual row counts, label census, strict CAL/VAL speaker IDs, runtime/offline join identity, abstentions, extras, finite scores, and `FN=145−TP`, `TP<=137`) precede gate interpretation.

No CAL/VAL role changes or threshold tuning are permitted. Saved CTC recognition edits/PER remain separate from the human operation labels. Speaker/phone results are disaggregated evidence and do not convert a two-speaker validation partition into population-level generalization.

## Required coordinator resumption

Verify the frozen exchange manifest and hashes; preserve originals and private-only record storage; run `audit_scripts/recompute_trn04r_metrics.py --input-root <immutable-private-root> --endpoint fixed_budget_3600|best --output <new-private-output>`; its frozen cell map requires C0/T2 `operation_ce` and T1/T3 `class_focal`, so `class_focal` cannot be treated as `binary_collapsed`; review output invariants and compare independently before preparing a final synthesized result. Until this is done results are **pending**, not executed.
