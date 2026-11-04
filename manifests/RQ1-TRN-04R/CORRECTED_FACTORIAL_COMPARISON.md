# Corrected RQ1-TRN-04R factorial comparison

**Real-data comparison status: PENDING.** This report contains no historical or guessed results. The coordinator runs the independent accounting CLI once after freezing integrated source and the immutable prediction exchange. Until then no numerical arm ranking, gate disposition or scientific conclusion is asserted.

## Analysis contract

Each cell is scored separately from the same frozen VAL rows and CAL threshold rows. C0/T2 use `operation_ce`; T1/T3 use `class_focal`. All four three-way objectives must supply `s_omit`/`s_sub` separately from `s_error`; missing class-focal subtype channels fail validation. Binary and legacy omission/substitution AP use `s_error`; dedicated omission/substitution AP use `s_omit`/`s_sub`, respectively. Dedicated subtype metrics are null/unavailable only for an explicitly `binary_collapsed` B-Control. AP uses exact grouped ties. CAL correct-only thresholds at FPR ceilings 5%, 10%, and 15% are reported independently from retrospective VAL curves. Operational omission alarms use the full 9,956-position gold census; prompt-only extras are a distinct, out-of-census alarm group.

The primary matched cell contrasts are **T1−C0, T2−C0, T3−C0**. The factorial interaction is **T3−T2−T1+C0**. Every contrast reports endpoint, population, score semantics, coverage and denominator. Missing or unsupported values make that contrast unavailable, not zero.

## Gates (reported separately from contrasts)

1. **Registration gate:** pending the frozen numeric criterion and valid comparator. It will not be inferred from a positive contrast.
2. **Noninferiority vs external reference:** pending the named external reference and frozen margin.
3. **Noninferiority vs matched C0:** pending the named endpoint and frozen margin; a matched difference is not itself a noninferiority test.
4. **Coverage/accounting gate:** actual row counts, label census, runtime/offline alignment, unmapped abstentions, extras, finite score validity, and `FN=145−TP`, `TP<=137` invariants must pass before interpreting endpoint values.

No CAL/VAL role changes or threshold tuning are permitted. Saved CTC recognition edits/PER remain separate from the human operation labels. Speaker/phone results are disaggregated evidence and do not convert a two-speaker validation partition into population-level generalization.

## Required coordinator resumption

Verify the frozen exchange manifest and hashes; preserve originals and private-only record storage; run `audit_scripts/recompute_trn04r_metrics.py --input-root <immutable-private-root> --output <new-private-output>`; its frozen cell map requires C0/T2 `operation_ce` and T1/T3 `class_focal`, so `class_focal` cannot be treated as `binary_collapsed`; review output invariants and compare independently before preparing a final synthesized result. Until this is done results are **pending**, not executed.
