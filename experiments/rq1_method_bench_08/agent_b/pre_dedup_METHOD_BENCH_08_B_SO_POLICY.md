# MB08 Agent B — S/O policy analysis

## Status and scope

**MEASURED** on TRAIN, seeds 101/202 and the four frozen speaker folds. **VERIFIED** event IDs, labels, speaker assignments, checkpoint hashes, MB07 prediction/receipt hashes, nested E4 joins and this lane's exports. **EXPLORATORY** matched-realized-FWR interpolation. **NOT_TESTED** insertion and any full-system S/O/I decision metric: no insertion policy was available in this lane. No TEST_SEALED or DEV_SEARCH inputs were read; no base-model reruns were made. This is target-only S/O evaluation, not unified training.

Population was 59,562 target events / 1,800 utterances / 12 speakers (S=7,381; O=1,825), with folds F0 ABA/ASI/BWC, F1 EBVS/HJK/HKK, F2 HQTV/LXC/NJS, F3 PNV/SKA/SVBI. `analysis_config.json` froze the corrected false-warning-prefix threshold procedure and score-family definitions before the corrected policy run. The initial total-warning-budget analysis is retained separately as `legacy_total_warning_budget_*`; it is not used for the primary findings.

## Provenance and correction notes

- MB07 Agent A seed-101/202 direct and nested fusion score NPZ hashes match their receipts; fusion seed 101's current prediction hash is `c5a6bcee061a8d108ea38b9d078f702f2384c30ff635b0432ae736f484307568`. The supervisor's independent input audit independently reports this file hash and verifies the E4 checkpoint hashes, event joins and label/speaker mapping. This current receipt-backed hash supersedes the stale `ac228...` hash appearing in earlier prose; no MB07 artifact was modified.
- The second-level meta crossfit omits the held meta-fold labels from the meta fit, but base inner models for the other OOF groups can have trained on speakers held by that meta fold. The learned-policy thresholds/meta predictions are therefore **outer-safe**, but their calibration is **not strictly inner-generalization-valid**. This applies to corrected and retained MB07 meta score families and is recorded in `provenance.json`.
- **MB07 FWR reporting bug:** its source sets `realized_target_FWR = warning.sum()/n_utts`; the separately counted `false_warnings` are warnings whose predicted S/O class differs from the gold label. Therefore MB07 columns labeled FWR are total warning rates, not false-FWR. Its threshold selection also targeted total warning counts. This lane's corrected procedure chooses, from sorted inner confidence/tie prefixes, the largest prefix whose cumulative false warnings (including C→S/O confusions) are at most `floor(target × inner utterances)`. Outer **actual false-FWR** is `false_warnings / unique utterances`; total warning rate is a distinct column and must not be substituted for it.
- Phone-prior audit: `train_reference.npy` is 39D, while MB07's `argmax(X[:,-42:])` includes the three preceding acoustic summary values. The index disagrees with `argmax(train_reference)` on 59,559/59,562 events. The original MB07 fixed/learned prior-containing results are preserved under explicit `mb07_original_*_contaminated42` names; they are not silently presented as valid phone-only controls. The v2 controls recompute a 39-category smoothed prior using only the fitting speakers. Direct neural and E4 predictions are unchanged.
- Historical MB03 PLAIN scores were keyed by the audited fold-specific producer order and manifest slot order. With no eligible inner OOF predictions for that historical model, only ranking AP is reported; no operating threshold is invented.

## Ranking and calibration results

Pooled AP-S/AP-O, ECE-15; all scores are outer held-speaker predictions, pooled over the four folds:

| Method | Seed 101 AP-S / AP-O | Seed 202 AP-S / AP-O | Seed 101 / 202 ECE-15 |
|---|---:|---:|---:|
| Direct H-002 neural | .3841 / .2657 | .3802 / .2561 | .1059 / .1098 |
| E4-only | .2722 / .2290 | .2882 / .2368 | .1689 / .1585 |
| Fixed neural + E4, 50/50 | .4027 / .3308 | .4021 / .3340 | .0782 / .0693 |
| Learned neural + E4, no prior (v2) | .4155 / .3100 | .4086 / .3107 | .0106 / .0143 |
| Learned neural + corrected 39-phone prior + E4 (v2) | .4233 / .3254 | .4230 / .3290 | .0118 / .0144 |
| Historical MB03 PLAIN (ranking only) | .4232 / .2767 | same historical scores | not applicable |

The corrected 39-phone fixed and learned-prior controls, as well as preserved contaminated42 legacy controls, are in `pooled_metrics.csv`. Fold AP and calibration are in `fold_metrics.csv`; per-speaker AP and calibration are in `speaker_metrics.csv`. Brier and NLL are also provided for every pooled/fold/speaker score set. The learned no-prior ablation is measured for both seeds; it gives repeatable ranking signal, but the simpler fixed neural+E4 blend has higher O AP and simpler inference.

Pooled AP need not equal mean fold AP: AP is non-decomposable and sensitive to positive prevalence and cross-fold score ordering. For example, direct H-002 seed-101 AP-O is .2657 pooled versus .2705 mean of the four fold APs; its AP-S is .3841 pooled versus .3864 fold mean. Neither aggregation is erroneous; they answer different aggregation questions. Per-fold variation and class supports are available in the compact tables.

MB03 PLAIN is not an architectural/input-matched ablation of H-002. MB03's grouped plain head consumes the mean of cached layer-6 crop frames plus 39D intended-phone one-hot and trains for five epochs. H-002 direct uses mean, standard deviation and end-minus-start over an 11-frame H-002 crop plus the 39D reference, with a 256/128 MLP and 12 epochs. The PLAIN ranking comparison is descriptive, not evidence that one architecture dominates under matched input/training.

## Frozen operating policies

`policy_metrics.csv` contains per-fold decision counts and class precision/recall/F1, Macro2, total warning rate and actual false-FWR for four nominal targets (.2/.5/1/2) and the predeclared dense grid. `policy_pooled.csv` sums disjoint outer-fold counts per seed. The target is an inner false-warning constraint, **not** a promise that outer achieved FWR equals the nominal number. Keyed decisions and frozen thresholds are in `keyed_policy_predictions.jsonl.gz`; keyed scores are in the NPZ export.

At nominal inner false-FWR target .2, pooled outer results for fixed neural+E4 versus direct neural were:

| Seed / scorer | Actual outer false-FWR | Total warning rate | O recall | Macro2 F1 |
|---|---:|---:|---:|---:|
| 101 fixed neural+E4 | .163 | .473 | .143 | .153 |
| 101 direct neural | .097 | .293 | .040 | .074 |
| 202 fixed neural+E4 | .195 | .507 | .136 | .149 |
| 202 direct neural | .141 | .392 | .039 | .084 |

At target .5, fixed neural+E4 achieved FWR .447/.459 (seeds 101/202), O recall .208/.212 and Macro2 .236/.235; direct neural achieved FWR .293/.356, O recall .083/.083 and Macro2 .150/.160. These nominal-policy comparisons have unmatched outer FWR and are not evidence of superiority at exactly equal false-warning rates.

`matched_fwr_exploratory.csv` linearly interpolates between achieved outer false-FWR points for already-frozen policies only. Common support across the clean comparison methods spans approximately .068–5.22 FWR/utterance for seed 101 and .077–5.10 for seed 202. At interpolated .2 FWR, fixed neural+E4 Macro2/O recall is .166/.153 (101) and .150/.138 (202), versus direct neural .114/.061 and .108/.051. This supports a practical low-FWR operating advantage in this **EXPLORATORY** interpolation, not an exact deployable match or a post-hoc threshold. No threshold or scorer was tuned from outer labels. Deployment still requires policy calibration on an eligible development set.

## Recommendation

Primary S/O scorer: **fixed 50/50 H-002 neural + E4**. It is a simple fixed rule, has repeatable pooled AP-O improvement over direct H-002, and the low-FWR interpolated omission-recall/Macro2 behavior is favorable across both seeds. Use the inner-fitted frozen policy thresholds at deployment only after the planned independent calibration; do not promise the nominal outer FWR. **Fallback:** direct H-002 neural if E4 evidence is unavailable or adds unacceptable compute/latency. The fallback is simpler, but its low-FWR omission recall is weaker here. Learned neural+E4 without prior is a useful ablation and ranking alternative, not the recommended policy. No sweeping matched-rate superiority claim is made.

Full-system F1 remains **null / NOT_TESTED** because insertion predictions and a validated insertion policy are absent in this lane. EVAL-V3.1 full-system reporting should be handled by the integrating supervisor after joining the separately keyed insertion output; no legacy evaluator was used here.

## Artifacts

- `analysis_config.json`, `run_analysis.py`, `summarize_policy.py`
- `fold_metrics.csv`, `speaker_metrics.csv`, `pooled_metrics.csv`, `inner_calibration.csv`
- `policy_metrics.csv`, `policy_pooled.csv`, `matched_fwr_exploratory.csv`
- `keyed_outer_scores.npz` (seed/fold/event_id/utterance_id/speaker_id/slot_idx/labels and method score matrices)
- `keyed_policy_predictions.jsonl.gz` (event identifiers, label, prediction, warning, method, nominal inner false-FWR target and frozen threshold/tie provenance)
- `historical_mb03_plain_ranking.csv`, `mb03_plain_keyed.npz`, `provenance.json`
- `legacy_total_warning_budget_metrics.csv`, `legacy_total_warning_budget_predictions.jsonl.gz` retained for audit only; their threshold target is total warning volume, not false-FWR.
- `receipt.json` records hashes, commands, environment and validation outcomes.
