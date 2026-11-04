# LANGFENS RQ1 — TRN-04R executive rebenchmark

**Executed corrected benchmark; practical acceptance failed; scientific conclusions partially supported, not blanket PASS.**

Frozen inference revision `c60a84fb044f049dc30f61046f819ff0ac36dc72`; final shared-cache CPU FP32, batch size 1 execution completed in 155.637409449 seconds, all 8 endpoints. Earlier 4 parallel runs are preserved but not final factorial evidence because one process had nonidentical encoder feature hashes. Final version shares one immutable 600-utterance cache across all heads, with unchanged model/source fingerprints. Independent production and standard-library saved-row recomputation agree, including global/speaker/phone AP, CAL thresholds, confusion counts and greedy PER.

Primary operating point: dedicated omission score, CAL-correct 15% FPR ceiling on NCC/THV; VAL NJS/RRBI never tunes thresholds. AP is finite aligned-only, not full-gold AP. Rates are fractions unless marked %. FA/1,000 uses 9,956 full gold positions. Extra-prompt alarms are separate and unlabelled.

| Arm | Dedicated omit AP | Legacy omit AP | Precision | Recall all | Recall aligned | FA/1,000 | Binary AP | Legacy sub AP | Dedicated sub AP | PER |
|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 0.104690254 | 0.033765022 | 0.070298770 | 0.551724138 | 0.583941606 | 106.267577340 | 0.316560146 | 0.306524807 | 0.333795147 | 0.238943427 |
| T1 | 0.029927641 | 0.024291675 | 0.027375201 | 0.117241379 | 0.124087591 | 60.666934512 | 0.311004684 | 0.308639164 | 0.341037241 | 0.239044631 |
| T2 | 0.086463470 | 0.031205329 | 0.063870353 | 0.462068966 | 0.489051095 | 98.633989554 | 0.318158944 | 0.310863253 | 0.335316426 | 0.238943427 |
| T3 | 0.031705546 | 0.025011971 | 0.040968343 | 0.151724138 | 0.160583942 | 51.727601446 | 0.313896816 | 0.311610067 | 0.337825388 | 0.239044631 |

No arm passes Level A/B. Historical focal alarm explosion is not reproduced; score-family fallback was invalid. T2 is not globally Pareto-dominated; T3 has favorable descriptive omission interaction without practical benefit over C0. C0 remains an omission reference, not a universal architecture champion. Unsupported broad-event HT claims are withdrawn from decision evidence.

One seed and two repeatedly used VAL speakers support exploratory descriptive contrasts only. No population-superiority, causal acoustic-localization, deployment, universal focal-loss/acoustic-feature invalidation, or new-training claim follows. Historical training-source commit, empirical FIT acoustic-normalization receipt, and frozen FIT-only prior control remain unverified/unrecovered.

See NEXT_RESEARCH_DECISION.md for all 12 explicit decisions; original-vs-corrected table, fixed/best, gates, speaker/phone analyses, provenance ledger, independent audit and private persistence receipts provide detailed evidence. Final process/durability state is in PROCESS_CHECKPOINT.md.
