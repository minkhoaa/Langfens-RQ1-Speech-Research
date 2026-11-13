# METHOD-BENCH-01 — comparative analysis (Agent C)

## Status

Benchmark protocol and leaderboard schema prepared from the official EVAL-V3.1 specification and scientific protocol. No new method predictions have been supplied yet; therefore this file contains no new comparative result or claim of improvement. GPU is not used by this workstream.

## Metric and event compatibility

Comparable score-based results require event-level scores joined by exact event key. Substitution and omission are one-vs-rest target rankings across the full target population. Insertion is one-vs-rest ranking across the full N+1 exact-gap population. An utterance-level insertion flag or token-aligned insertion label does not identify the correct gap and cannot be scored as exact-gap AP without a valid mapping. Likewise, native binary MDD outputs cannot support diagnosis AP or insertion AP.

Macro-3 Error AP is the unweighted mean of AP S, AP O, and AP I exact-gap, despite their separate target/gap domains. F1 values are threshold-dependent and must be reported separately from AP. Macro-4 averages correct, substitution, omission, and insertion F1 across mixed target/gap domains; it is not a four-way classifier metric.

## Development split and leakage guard

DEV_SEARCH is development-exposed, with 599 utterances from four speakers. It is exploratory rather than independent final validation. No learned fit, threshold, calibrator, hyperparameter, or candidate selection may use DEV_SEARCH gold labels. The canonical protocol fits learned models on TRAIN and any policy/calibration on DEV_CAL, then freezes it before DEV_SEARCH assessment. Report that DEV_SEARCH checkpoints were previously selected in the broader project where relevant.

## Baseline context (not matched reruns)

The EVAL-V3 leaderboard gives seed-101 student-only DEV_SEARCH Macro-3 Error AP of 0.236917 (GOLD-ONLY), 0.240230 (KD-ALL), and 0.221976 (LIGHT-GOLD). GOLD-ONLY seed results are 0.236917, 0.234517, and 0.231667 for seeds 101/202/303; KD-ALL results are 0.240230, 0.236279, and 0.233021. These scores are evidence of development-set ranking performance, not blind-test generalization. Compare new methods on matched manifests, cohorts, score semantics, and seeds wherever possible.

Earlier exploratory system values listed in the campaign brief are heterogeneous historical context and are not promoted onto the EVAL-V3 leaderboard absent matched event-level artifact and protocol audit.

## Pending

- Candidate prediction bundles and provenance from Agent B.
- Matched EVAL-V3.1 evaluation and class PR curves.
- Confusion matrices and K-budget retrieval summaries.
- Seed replication and paired deltas, if resources permit.
- Analysis of acoustic evidence versus prior-driven rank changes.
