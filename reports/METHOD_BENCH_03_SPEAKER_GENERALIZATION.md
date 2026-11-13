# Speaker generalization — RQ1-METHOD-BENCH-03

## Matched split diagnostic

A four-fold comparison used the same small plain-head recipe, feature construction, optimizer, class weighting, five epochs, CPU and seed, varying only fold assignment: 450 utterances per random fold versus three entire speakers held out per grouped fold. All TRAIN target and exact-gap events are predicted once.

| TRAIN OOF AP | Utterance-random | Speaker-grouped | Difference |
|---|---:|---:|---:|
| S | .478970 | .423209 | +.055761 |
| O | .359747 | .276706 | +.083040 |
| Exact-gap I | .220765 | .162125 | +.058640 |
| Macro-3 | .353161 | .287347 | +.065814 |

Random folds include all 12 speakers on both fit and validation sides. The difference shows that random-utterance validation overstates this model's observed cross-speaker performance. It is not a causal estimate of speaker identity, because fold utterances and label composition also differ. It is also not directly matched to the much larger U2 training architecture.

## Speaker adversarial method

Experiment `RQ1-MB03-C-SPEAKER-GRL-GROUPCV-S101`: 4 folds, each fits nine speakers and validates on three disjoint speakers. The plain and GRL arms share audio/intended-phone features, task heads, optimizer, seed and five epochs. GRL adds speaker-classification nuisance supervision (coefficient .10); speaker ID is TRAIN-only and not an inference feature. No realized phone or gold error labels are inference inputs.

| Event-OOF metric | Plain | GRL | Delta |
|---|---:|---:|---:|
| AP-S | .423209 | .423371 | +.000163 |
| AP-O | .276706 | .276634 | −.000072 |
| AP-I exact gap | .162125 | .165887 | +.003762 |
| Macro-3 AP | .287347 | .288631 | +.001284 |
| Binary MDD AP | .459202 | .458696 | −.000506 |

Fold-macro I AP instead changes .163123→.162838 (−.000284); three of four folds decline under GRL. Hence no repeatable transfer gain. No threshold F1 is claimed for these OOF scores because no fold-local calibration policy was predeclared.

## Other observed shifts and limits

U2 exact-gap insertion AP is .976979 in TRAIN vs .138111 DEV_CAL and .117992 DEV_SEARCH. The gap is strong evidence of in-sample degradation but not its cause. U2 per-speaker DEV_SEARCH AP-I spans .0450–.2506 (22–44 insertion positives per speaker). TRAIN S/O/I prevalence ranges by speaker .0646–.1749 / .0008–.0947 / .0014–.0206; DEV_SEARCH ranges .0671–.1083 / .0063–.0131 / .0043–.0086. Prevalence/label shift plausibly contributes, especially for omission. Intended-phone unigram JS distance is small: .000280 TRAIN/CAL, .000131 TRAIN/SEARCH; this only argues against a large marginal phone-frequency shift. No matched channel/noise analysis or annotation/alignment artifact decomposition was performed.

The GRL result is a negative finding for one small intervention, not a rejection of speaker-invariant representation learning. One seed, one corpus, four folds; no Spin-style representation training. Grouped OOF metrics are not directly comparable with DEV_SEARCH full-model scores. CTC and edit event scores are not both available on the same grouped OOF IDs; complementarity is therefore unestablished.

Receipts and predictions: `experiments/rq1_method_bench_03/agent_c/`. Full setup and source discovery context: `reports/METHOD_BENCH_03_C_GENERALIZATION_AND_RESEARCH.md`.