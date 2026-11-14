# Static 51D specialist: inner-threshold to outer-speaker transfer

**Run:** 2026-11-13. See the [predeclared protocol](STATIC_SPECIALIST_NESTED_PREREGISTRATION.md), [runner](validate_static_specialist_nested.py), [results](NESTED_STATIC_RESULTS.json), and [log](NESTED_STATIC.log). The 51D candidate was chosen after the exploratory temporal pilot, so this is a **development-exposed** check. It is not a prospective confirmation.

For each of four outer folds, two per-manner 51D models were fit on nine TRAIN speakers and scored on three held speakers. A separate set of six-speaker inner fits produced nine-speaker out-of-fold scores to choose the vowel and stop thresholds **jointly**. The inner choice allowed at most 0.08 additional false warnings per utterance for these two manners, reserving the historical 0.12 for the other frozen alerts. No outer-held speaker label was used to choose a threshold.

| Outer fold | Held speakers | Vowel TP / FP | Vowel recall / precision | Stop TP / FP | Stop recall / precision | Joint FWR |
|---|---|---:|---:|---:|---:|---:|
| 0 | ABA, ASI, BWC | 23 / 14 | 4.76% / 62.16% | 11 / 4 | 3.97% / 73.33% | 0.176 |
| 1 | EBVS, HJK, HKK | 31 / 29 | 5.50% / 51.67% | 6 / 13 | 5.22% / 31.58% | **0.227** |
| 2 | HQTV, LXC, NJS | 38 / 22 | 5.37% / 63.33% | 24 / 11 | 14.63% / 68.57% | 0.156 |
| 3 | PNV, SKA, SVBI | 34 / 25 | 6.46% / 57.63% | 25 / 9 | 8.17% / 73.53% | **0.204** |
| **Pooled** | 12 speakers | **126 / 90** | **5.53% / 58.33%** | **66 / 37** | **7.66% / 64.08%** | **0.191** |

The frozen SM13 policy gives zero vowel and stop true positives on these same 1,800 utterances, with 216 total false warnings (FWR 0.120). The specialist raises recall above zero, but fails the declared **15% recall for each manner** gate; two folds also exceed FWR 0.20, and fold 1 stop precision falls below 50%. Pooled precision and FWR conceal these speaker-level failures. **Do not promote this candidate.**

This result explains part of the gap between retrospective frontier and usable policy: thresholds chosen on development speakers do not reliably transfer to held speakers, and the two manners share one warning budget. The inner models were trained on six speakers while outer scorers used nine, so score-scale differences may contribute; the experiment does not isolate calibration drift from generalization error. Another threshold sweep on the same OOF labels would not be a validation.

The fixed O/I and other-manner streams already consumed **0.136, 0.133, 0.082, and 0.129 FP/utterance** in folds 0–3, rather than exactly 0.120 in each fold. Fold 1 also gained 42 additional FP from the transferred vowel/stop thresholds, versus the 36 FP implied by its 0.08 allocation. Fold 3 gained 34, but its base 58 FP alone used 0.129 FP/utterance. The fold failures therefore reflect both speaker variation in the fixed streams and threshold transfer; reducing the allocation based on each held fold's labels would leak evaluation information.

## Next decision

1. Collect two independent blinded judgments for the [136-clip packet](AUDIO_REVIEW_PROTOCOL.md) before altering the target or declaring raw `*` labels incorrect. Keep disagreement and uncertain votes visible.
2. Use the [phone-level error report](ERROR_REGION_REPORT.md) to preregister one targeted acoustic/context hypothesis. Vowel `IH`, `AH`, `OW` and stop `D`, `T` carry most of the missed positives. A new candidate must improve actual joint low-warning recall, not AP alone.
3. Require inner-fit threshold choice, outer-speaker transfer, per-fold precision and FWR, then confirmation on genuinely new speakers before any operational claim. Existing historical DEV/TEST sets have been exposed to prior research and are not fresh holdouts.
