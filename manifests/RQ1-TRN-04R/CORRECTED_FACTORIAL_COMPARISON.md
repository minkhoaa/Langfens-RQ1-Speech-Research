# Corrected RQ1-TRN-04R factorial comparison — independently recomputed

**State: BOTH ENDPOINTS RECOMPUTED AND INDEPENDENTLY AUDITED; FIXED-BUDGET GATES FAIL; PRIVATE-STORE READBACK PENDING. No overall PASS.** This is a model-free result update. It does not train or infer, and does not edit old files. Full endpoint JSON stays private at `/home/khoa/.cache/langfens/trn04r/genuine-z-v1-20261104T083210Z/metrics/`.

## Provenance and independent verification

The frozen run records identify source commit `c60a84fb044f049dc30f61046f819ff0ac36dc72`, four completed arm processes (C0/T1/T2/T3, return code 0, two threads per arm), CPU FP32, and no optimizer/backward operations. Workstream B independently ran `recompute_trn04r_metrics.py` for `fixed_budget_3600` and `best` against the immutable rows using its permitted `arms/<CELL>/<CELL>_<endpoint>` layout. Both B recomputations were byte-for-byte equal to the existing private metrics JSON. The parent’s read-only standard-library auditor independently returned `INDEPENDENT_RECOMPUTATION_AGREES` for both endpoints, verifying global/speaker/phone AP, CAL thresholds, deployed counts, omission FN, PER, row coverage and runtime freeze hashes. Every cell had 9,956 gold rows = 9,919 mapped + 37 unmapped and 37 separate prompt extras; reference CTC count was 9,881. It checked numerical agreement at 1e-12. This validates metric reproduction, not durable-store readback; the coordinator still has that readback pending.

## Historical alarm-score correction

The former focal-arm alarm result (985.436 false alarms/1,000, 9,811 false alarms, every canonical VAL token alarmed) is not a valid `s_omit` operating result. In the old evaluator, CAL omission thresholds came from deletion probability `b_probs[idx][2]` (`src/langfens_prep/trn04/evaluator.py`, CAL block); VAL then used `s_omit` **or fell back to `s_error`** when `s_omit` was null (same file, operating-point loop). The old `p_sup/runtime.py` serializer populated `s_omit`/`s_sub` only for `operation_ce`; `class_focal` left those scores null while retaining `s_error`. Thus the focal thresholds were deletion-score thresholds applied to the binary error score. The corrected c60 runtime preserves all three probability channels for both three-way objectives. The old dedicated-omission AP claims from fallback rows are likewise not dedicated-score AP.

The corrected fixed-budget T1/T3 values are substantially different (CAL-15 FA 60.667 and 51.728 per 1,000, not 985.436). Those corrected measurements still fail registered precision/FA gates and have lower dedicated omission AP than C0; this corrects the historical score-family artifact, it does not establish a winner. The single seed, two validation speakers, and unknown historical training-source provenance do not support claims that all focal-loss or acoustic methods are invalid.

## Fixed-budget 3,600-update endpoint: global metrics

AP values are recomputed on answered, aligned populations with exact grouped ties. PER is saved greedy-CTC recognition, not human omission labels. Acoustic gap is matched minus within-speaker cyclic-mismatch **binary AP**.

| Arm | Binary AP | Omission legacy AP (`s_error`) | Omission dedicated AP (`s_omit`) | Substitution legacy AP (`s_error`) | Substitution dedicated AP (`s_sub`) | Greedy PER | Binary acoustic gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.316555 | 0.033766 | 0.104690 | 0.306519 | 0.333787 | 0.238943 | +0.016809 |
| T1 | 0.311007 | 0.024290 | 0.029928 | 0.308645 | 0.341036 | 0.239045 | +0.011738 |
| T2 | 0.318160 | 0.031205 | 0.086464 | 0.310865 | 0.335312 | 0.238943 | +0.020904 |
| T3 | 0.313886 | 0.025012 | 0.031706 | 0.311598 | 0.337822 | 0.239045 | +0.018128 |

### Registered primary omission operating point: dedicated `s_omit`, CAL 15%

`FP` comprises `FP_correct + FP_substitution`; abstained true omissions remain FN. FA/1,000 uses all 9,956 gold positions; extras are separate and excluded.

| Arm | CAL threshold | TP | FP correct | FP substitution | FN (=145−TP) | Precision | All-gold recall | FA / 1,000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.058011 | 80 | 999 | 59 | 65 | 7.030% | 55.172% | 106.268 |
| T1 | 0.105885 | 17 | 591 | 13 | 128 | 2.738% | 11.724% | 60.667 |
| T2 | 0.059710 | 67 | 922 | 60 | 78 | 6.387% | 46.207% | 98.634 |
| T3 | 0.104366 | 22 | 493 | 22 | 123 | 4.097% | 15.172% | 51.728 |

### Secondary CAL 5% and 10% omission points (`s_omit`)

Cells below are `TP / FP / precision / all-gold recall / FA per 1,000`.

| Arm | CAL 5% | CAL 10% |
|---|---|---|
| C0 | 27 / 176 / 13.300% / 18.621% / 17.678 | 50 / 520 / 8.772% / 34.483% / 52.230 |
| T1 | 5 / 81 / 5.814% / 3.448% / 8.136 | 11 / 294 / 3.607% / 7.586% / 29.530 |
| T2 | 18 / 144 / 11.111% / 12.414% / 14.464 | 43 / 481 / 8.206% / 29.655% / 48.313 |
| T3 | 2 / 113 / 1.739% / 1.379% / 11.350 | 8 / 266 / 2.920% / 5.517% / 26.718 |

### Fixed-budget absolute registration gates (separate from factorial contrasts)

All four cells fail both levels. C0/T2 pass recall, binary AP, substitution AP and PER; both fail precision and FA. T1 fails precision, recall, FA and binary AP; it passes legacy substitution AP and PER. T3 fails precision, recall and FA; it passes the binary AP threshold narrowly, legacy substitution AP and PER. The Level B acoustic-gap criterion passes in each cell, but cannot offset failures in other required criteria.

| Level | Frozen requirements | C0 | T1 | T2 | T3 |
|---|---|---|---|---|---|
| A | Precision >=.25; recall >=.30; FA/1k <=30; binary AP >=.313619; legacy sub AP >=.303974; PER <=.246980 | FAIL | FAIL | FAIL | FAIL |
| B | Precision >=.30; recall >=.40; FA/1k <=15; same AP/sub/PER criteria; acoustic gap >=.010 | FAIL | FAIL | FAIL | FAIL |

## SELECT-best endpoint

This is a separately reported checkpoint stratum; fixed-budget registered gates are **not applicable**. AP and CAL-15 `s_omit` results:

| Arm | Binary AP | Omission legacy AP | Omission dedicated AP | Substitution legacy AP | Substitution dedicated AP | PER | CAL-15 P / all-gold R / FA per 1,000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.309462 | 0.035882 | 0.110943 | 0.297626 | 0.323354 | 0.249367 | 7.126% / 41.379% / 78.546 |
| T1 | 0.308140 | 0.023896 | 0.025631 | 0.306309 | 0.339991 | 0.282360 | 3.125% / 8.276% / 37.364 |
| T2 | 0.318160 | 0.031205 | 0.086464 | 0.310865 | 0.335312 | 0.238943 | 6.387% / 46.207% / 98.634 |
| T3 | 0.281907 | 0.026972 | 0.024485 | 0.273944 | 0.325520 | 0.249469 | 2.362% / 10.345% / 62.274 |

For completeness, CAL 5% and 10% endpoint secondary P/recall/FA are in private `metrics/best.json` and the lightweight `INDEPENDENT_METRIC_RECOMPUTATION.json`; no absolute gate or selection conclusion is attached to these best-checkpoint values.

## Matched factorial contrasts and interaction

Each row reports `T1−C0 / T2−C0 / T3−C0 / (T3−T2−T1+C0)`. Higher is better for AP, precision, recall and acoustic gap; lower is better for FA and PER. These are descriptive within the same two-speaker VAL set.

| Endpoint | Fixed-budget contrasts / interaction | Best-checkpoint contrasts / interaction |
|---|---|---|
| Binary AP | −0.005548 / +0.001605 / −0.002669 / +0.001273 | −0.001322 / +0.008699 / −0.027555 / −0.034932 |
| Dedicated omission AP | −0.074762 / −0.018226 / −0.072984 / +0.020004 | −0.085311 / −0.024479 / −0.086458 / +0.023332 |
| CAL-15 omission precision | −0.042924 / −0.006428 / −0.029330 / +0.020022 | −0.040009 / −0.007389 / −0.047637 / −0.000239 |
| CAL-15 all-gold omission recall | −0.434483 / −0.089655 / −0.400000 / +0.124138 | −0.331034 / +0.048276 / −0.310345 / −0.027586 |
| CAL-15 FA / 1,000 | −45.601 / −7.634 / −54.540 / −1.306 | −41.181 / +20.088 / −16.272 / +4.821 |
| Legacy substitution AP | +0.002126 / +0.004347 / +0.005080 / −0.001393 | +0.008682 / +0.013239 / −0.023682 / −0.045603 |
| Dedicated substitution AP | +0.007249 / +0.001525 / +0.004034 / −0.004739 | +0.016637 / +0.011958 / +0.002165 / −0.026429 |
| Greedy PER | +0.000101 / 0.000000 / +0.000101 / 0.000000 | +0.032993 / −0.010424 / +0.000101 / −0.022467 |
| Binary AP acoustic gap | −0.005071 / +0.004096 / +0.001320 / +0.002295 | −0.000141 / −0.000733 / −0.000196 / +0.000679 |

## Interpretation boundary

All fixed-budget arms fail the registered joint gates, despite several individual binary/substitution/PER criteria passing. The score-family correction invalidates the earlier 985.4/1k focal-arm alarm claim as an `s_omit` deployment estimate; the corrected focal-arm outputs still show low omission precision/recall and low dedicated omission AP under this particular registered run. This single seed and two repeatedly used VAL speakers support descriptive results only—not population-level superiority, deployment approval, or a conclusion that focal loss/acoustic approaches generally fail. The full private rows/results were not copied into Git. Parent storage readback remains pending; therefore no overall campaign PASS is asserted.
