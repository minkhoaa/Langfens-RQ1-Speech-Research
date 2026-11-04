# TRN-04R — correction specification (pre-benchmark)

Original evidence is retained under `manifests/RQ1-TRN-04/` at original commit `d332997e732315292d673a3bd42a03698182f5f6`. This is an evaluation correction, not a training campaign.

## Binding compatibility reference

Use recovered P-SUP benchmark `final-v2`, its original `audit_scripts/run_psup_recovered_benchmark_v1.py`, frozen Evaluation Contract V1 (SHA-256 `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`), original prompt JSON and CMUdict, strict OOV policy/first pronunciation, original 39-phone normalizer and `alignment.py` `sub_first` policy. No newly selected pronunciation, alternative alignment, or VAL-derived threshold is primary.

## Corrections required before execution

1. **Oracle input:** old TRN-04 evaluator constructs `prompt_phone_ids` through the annotation-backed P-SUP dataset. New runtime must read waveform and written prompt only; freeze raw CAL/VAL matched/mismatched predictions and hashes before opening annotations. Offline joining cannot change those predictions. Keep O and Z distinct.
2. **Population:** no additive `fn + 8` repair. Count actual all-gold rows. Expected VAL: 9,956 gold = 9,919 aligned + 37 unmapped; 37 extra prompt positions. Full correct/substitution/omission 8,799/1,012/145; aligned 8,774/1,008/137. `FN_all = 145 - TP_aligned`; `TP_aligned <= 137`. Gold abstentions never receive fabricated probabilities. AP is explicitly aligned-only, not full-population AP.
3. **Unsupported adjudication:** no recycled disagreement-rate multiplication or unsupported Horvitz–Thompson performance estimate. Strict original-gold measures are primary. Broad-event performance remains unavailable absent valid case-level reviews and inclusion probabilities for the relevant predictions.
4. **Artifact identity:** hash actual original checkpoint bytes, reconcile pinned remote bytes and persisted receipts, retain conflicting historical claims additively. Keep fixed-budget and SELECT-best strata separate.
5. **Additional source finding:** shared `p_sup/runtime.py` accepts `class_focal` but emits subtype scores only for `operation_ce`, incorrectly treating focal as unidentified binary collapse. The old evaluator then falls back from null `s_omit` to `s_error` at deployment while applying a threshold calibrated on `p_deletion`, and labels legacy AP as dedicated AP when the dedicated-score list is empty. Thus historical T1/T3 alarm and dedicated-AP comparisons contain a score-family mismatch, in addition to the oracle/Z defect. New TRN-04R serialization must expose the trained three-way classes for CE and focal; genuine B-Control retains unsupported/null subtype scores. Preserve the original serializer unchanged. This source finding does not establish the corrected empirical outcome.

## Registered settings retained before corrected measurements

Primary: fixed-budget 3,600 successful updates per arm, CPU FP32 batch-one, seed 1337, frozen original backbone/head, CAL NCC/THV only, VAL NJS/RRBI. Nominal 15% CAL correct-token FPR is the registered primary operating point; 5% and 10% are secondary, not replacements selected after VAL inspection. Alarm means `score >= threshold`, threshold determined using CAL correct scores and complete tied groups. Dedicated omission uses deletion versus correct for AP; deployed omission precision also counts substitution false alarms. FA per 1,000 divides gold-linked false alarms by all 9,956 gold canonical positions; extra-prompt alarms are disclosed separately because their strict learner-event labels are unidentified.

Level A: precision >= .25, all-gold recall >= .30, FA/1,000 <= 30, binary AP >= .313619, legacy substitution AP >= .303974, PER <= .246980. Level B: precision >= .30, recall >= .40, FA/1,000 <= 15, same noninferiority conditions, matched-minus-mismatched binary AP >= .010. These absolute registered external-reference gates are not a substitute for internally matched C0 contrasts. Report T1-C0, T2-C0, T3-C0 and interaction T3-T2-T1+C0 separately, with directionality qualified per metric.

One seed and two repeatedly used VAL speakers permit descriptive exploratory comparisons, not population superiority, general focal-loss/acoustic-feature invalidation, deployment approval, or automatic new-training authorization. No sealed TEST access is permitted.

## Execution state

Implementation/audit lanes in progress; no corrected inference, metrics or verdict is asserted by this pre-benchmark specification. See `PROCESS_CHECKPOINT.md` for current campaign state.
