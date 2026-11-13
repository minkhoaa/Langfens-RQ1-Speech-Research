# METHOD-BENCH-03 Agent C — speaker transfer and independent method discovery

**Date:** 2026-11-08  
**Scope:** speaker-generalized acoustic error ranking on the authorized L2-ARCTIC TRAIN/DEV splits; focused primary-source review of methods outside the CTC work owned by Agents A and B.  
**Status:** one new CPU experiment completed; DEV_SEARCH metrics below are descriptive because the split has been exposed by earlier campaigns.

## Findings

1. **Utterance-random validation is substantially more optimistic than speaker-grouped validation, while neither recreates U2 in-sample fitting.** The same lightweight plain head obtains exact-gap event-OOF AP .220765 under four-fold utterance-random splits versus .162125 under four-fold speaker-grouped splits (delta +.058640); S/O AP likewise rise .478970/.359747 vs .423209/.276706. U2's distinct train-fit AP-I .976979 falls to .138111 on CAL and .117992 on SEARCH. Random folds include all 12 speakers on both sides; grouped folds hold out three entire speakers. These matched split results demonstrate validation-protocol sensitivity on TRAIN, not speaker identity causality, and neither is a matched architecture/data replication of U2.

2. **Speaker-wise error prevalence shifts substantially.** Across 12 TRAIN speakers, S prevalence ranges .0646–.1749 and O prevalence .0008–.0947; I prevalence ranges .0014–.0206. On DEV_SEARCH, speaker S prevalence ranges .0671–.1083, O .0063–.0131, and I .0043–.0086. DEV_CAL O prevalence is also highly uneven across its two speakers (.0112 and .0807). The train-wide rates are S .1239, O .0306, I .0091; DEV_CAL rates are .1773, .0460, .0099; DEV_SEARCH rates are .0895, .0106, .0061. Label-prevalence shift is therefore a plausible contributor, especially for O, but does not by itself explain the near-perfect in-sample insertion ranking. Existing U2 development AP varies sharply across the four SEARCH speakers, with exact-gap I AP .0450–.2506 on only 22–44 positives per speaker.

3. **A large intended-phone distribution shift is not supported by the aggregate marginals.** Squared Jensen–Shannon distance (base 2) for intended-phone unigram distributions is .000280 TRAIN-vs-CAL and .000131 TRAIN-vs-SEARCH. This rules against a large *marginal phone-frequency* shift as the main explanation; it does not test phone-conditioned acoustic shift, error-label shift within phone, prompt overlap, or realized-phone distribution.

4. **The tested speaker-adversarial projection did not reliably improve transfer.** In four folds, each with nine TRAIN speakers for fitting and three disjoint speakers for validation, gradient reversal changed the primary event-OOF AP from .423209 to .423371 for S, .276706 to .276634 for O, and .162125 to .165887 for exact-gap I. The insertion increase is small and not stable over folds: macro-fold I AP changes from .163123 to .162838, with three of four folds lower under GRL. This is an exploratory negative result for this specific small head, objective, and coefficient, not a rejection of speaker-invariant learning broadly.

## Speaker/generalization experiments

### Utterance-random versus speaker-grouped TRAIN OOF

To quantify the validation-design difference directly, `train_utterance_random_oof.py` reuses the exact `fit_predict` implementation, input construction, plain architecture, optimizer, class weights, five epochs, and fold seeds from the grouped experiment. NumPy RandomState(101) permuted 1,800 TRAIN utterances, assigning 450 per fold. Each validation fold shares all 12 speakers with its fitting set; this is not an independent-speaker estimate. Every target and exact gap is predicted once. No development data were used. Both experiments use CPU and the same 101 seed; their fold compositions differ by design.

| Event-OOF AP | Utterance-random | Speaker-grouped | Difference (random − grouped) | Positive support |
|---|---:|---:|---:|---:|
| S | .478970 | .423209 | +.055761 | 7,381 |
| O | .359747 | .276706 | +.083040 | 1,825 |
| Exact-gap I | .220765 | .162125 | +.058640 | 560 |
| Macro-3 error AP | .353161 | .287347 | +.065814 | — |

This is a direct comparison of the same training recipe with a changed split unit, but it does **not** isolate speaker identity as a causal factor: utterance-random folds also change which utterances and event-label composition enter each fit, and all speakers are represented in both arms. It does show that random-utterance validation overstates the observed cross-speaker transfer for this model/data setup. The high-dimensional U2 in-sample estimate (.976979) is still a different, more optimistic estimand; the small-head random OOF result does not reproduce it.

### Design and data

Experiment ID: `RQ1-MB03-C-SPEAKER-GRL-GROUPCV-S101`.

The 12 TRAIN speakers were divided into four fixed groups of three, with 9 speakers used to fit each fold and the other 3 used only for evaluation. Every TRAIN event receives one out-of-speaker prediction. A plain head and a gradient-reversal speaker-adversarial head have the same feature path, task head, optimizer, class weighting, five epochs, and initialization seed per fold. Only the adversarial arm uses an auxiliary speaker classification loss; GRL coefficient is .10. Speaker labels are used only as TRAIN nuisance supervision. Speaker identity is not an inference feature, and the held-out speaker identities are absent from each fold's adversarial classifier.

Target input is the mean of cached frozen Wav2Vec2 layer-6 crop frames at offsets 3:14 concatenated with the 39D intended-phone one-hot. Exact-gap input is the existing U2-style 1,614D vector: left/right edge means from neighboring predicted-center crops plus adjacent intended-phone one-hots. No gold error label, realized phone, or speaker identity is an inference input. The cached features are derived from audio and intended-phone alignment/crop preparation.

### Results

| Metric | Event-OOF AP, plain | Event-OOF AP, GRL | Δ | Support |
|---|---:|---:|---:|---:|
| S | .423209 | .423371 | +.000163 | 7,381 |
| O | .276706 | .276634 | −.000072 | 1,825 |
| exact-gap I | .162125 | .165887 | +.003762 | 560 |
| Macro-3 Error AP (mean of S/O/I AP) | .287347 | .288631 | +.001284 | — |
| Binary MDD AP | .459202 | .458696 | −.000506 | 9,206 target events |

Arithmetic mean of the four fold APs is S .420373→.420996, O .277785→.278365, and I .163123→.162838; it differs from the event-OOF estimand and is retained separately in the receipt.

Event-OOF AP was replayed through EVAL-V3.1 after joining by exact `(utterance_id, slot_idx)` and `(utterance_id, gap_idx)` identities. All 59,562 targets and 61,362 exact gaps from 1,800 utterances were covered once with zero missing or duplicate IDs; each utterance passed N-target/N+1-gap validation. Fold AP is shown separately because it is a different averaging estimand. The event-OOF I delta should not be called repeatable: fold I AP plain→GRL is .188792→.186671, .102708→.097053, .109449→.106200, and .251541→.261430. The apparent aggregate gain is driven by the fourth group and reverses when equally averaging folds. Threshold F1 is unsupported: no threshold/policy was predeclared or selected, and evaluator decisions are placeholders solely to obtain AP from EVAL-V3.1.

| Held-out TRAIN speakers | Target S AP plain → GRL | Target O AP plain → GRL | Gap I AP plain → GRL | I positive support |
|---|---:|---:|---:|---:|
| ABA, ASI, BWC | .4440 → .4438 | .2656 → .2649 | .1888 → .1867 | 134 |
| EBVS, HJK, HKK | .3610 → .3634 | .1903 → .1894 | .1027 → .0971 | 127 |
| HQTV, LXC, NJS | .4595 → .4582 | .3555 → .3560 | .1094 → .1062 | 150 |
| PNV, SKA, SVBI | .4169 → .4186 | .2997 → .3031 | .2515 → .2614 | 149 |

The score represents ranking only. No threshold F1 is reported for this OOF experiment: a defensible threshold policy would need fold-local calibration that does not use the held-out speakers. DEV_SEARCH U2 threshold results remain available in the existing EVAL-V3.1 artifact and are not mixed with these TRAIN OOF metrics.

### Independent benchmark and complementarity boundary

The primary fair comparison here is the matched plain-head split experiment above; the GRL comparison is also matched within the speaker-grouped folds (same target/gap inputs, task head, optimizer, seed, class weighting and epochs, differing only by auxiliary GRL loss). On event-OOF AP, GRL deltas versus grouped plain are S +.000163, O −.000072, I +.003762; fold-macro I changes −.000284, and three of four I folds decline. This is no robust complementarity signal. The existing Agent A continuous-log-mel CTC artifact has its own two-speaker holdout (HJK/HKK), while Agent B's probabilistic edit outputs are DEV_CAL/DEV_SEARCH, not identical TRAIN OOF event-score vectors. Consequently there is no same-population, same-event CTC/edit prediction pair to compute controlled correlation, conditional AP, or fusion lift against Agent C; cross-split score fusion would be unfair and is not claimed. CTC's standalone phone error metrics cannot be compared numerically to target/gap event AP. The reasonable conclusion from available evidence is complementarity **unestablished**, not absent. A matched event-level grouped OOF score export from A/B would be needed for a valid follow-up.

## U2 speaker transfer and alternative explanations

U2's DEV_SEARCH per-speaker AP-S is .2714–.5344, AP-O .1096–.3117, and AP-I .0450–.2506. Positive support is 331–536 for S, 31–65 for O, and 22–44 for I per speaker. The O/I AP range is large relative to the number of events, so the four-speaker estimates are descriptive and imprecise. This variation demonstrates heterogeneity among unseen speakers; it does not show speaker identity is the only cause of the U2 train-to-dev gap.

For the existing U2 EVAL-V3.1 policy selected on DEV_CAL at the 0.20 warning budget, the exposed DEV_SEARCH class F1s are S .0243, O .2092, exact-gap I .1429; Macro-3 Error F1 is .1254, Macro-4 mixed-domain F1 .3307, binary MDD F1 .0575, and joint FWR .1786. These operating-point values are reported separately from raw AP-S/O/I = .371360/.185318/.117992 and Macro-3 AP .224890. They are existing U2 results, not metrics for the new grouped OOF arms; the latter did not use a threshold policy.

| Candidate explanation | Evidence and limit |
|---|---|
| In-sample memorization / fitting | U2 train AP-I .976979 vs CAL .138111 / SEARCH .117992 is direct evidence of severe in-sample-to-held-out degradation. The grouped small-head OOF AP-I .162125 supports the need for speaker-held-out selection, but is not a matched causal comparison against U2. |
| Speaker-specific acoustic patterns | U2's per-speaker SEARCH AP ranges are wide; GRL did not produce a repeatable improvement. A direct decomposition into identity vs other acoustic factors was not performed. |
| Error prevalence | Speaker and split S/O/I prevalences are heterogeneous, with the largest split movement for O. AP is prevalence-sensitive; prevalence shift can contribute. It cannot explain the in-sample near-perfect score on its own. |
| Intended-phone distribution shift | Aggregate intended-phone unigram JS divergence is very small. Phone-conditioned error distributions and acoustic realizations remain untested. |
| Acoustic conditions | No matched analysis of microphone, channel, loudness, or noise metadata was available in this experiment. |
| Label and alignment artifacts | TRAIN annotations have highly heterogeneous speaker-wise rates. The U2 gap features use predicted-center crop edges and intended-phone context; we did not stratify crop/alignment error or annotator agreement by speaker. |
| Training procedure | The U2 gap branch is strongly overfit in-sample. Grouped fitting changes which speakers contribute to training and improves the evaluation design, but the standalone GRL intervention did not resolve transfer. |

## Focused external research

| Primary source | Mechanism and applicability | Implementation / feasibility verdict |
|---|---|---|
| Zhou et al., Interspeech 2025, [“Towards Accurate Phonetic Error Detection Through Phoneme Similarity Modeling”](https://www.isca-archive.org/interspeech_2025/zhou25h_interspeech.html), DOI 10.21437/Interspeech.2025-2417 | Verbatim phone recognition aims to predict what was produced rather than the intended sequence; paper describes multitask phoneme-similarity modeling and a simulated VCTK-accent benchmark. This is relevant to substitutions and sequence realization; deletion/insertion localization still needs an event mapping. | Authors' [public repository](https://github.com/Berkeley-Speech-Group/Phonetic-Error-Detection) identifies itself as the project website and contains simulation/static/evaluation assets, not a readily reproducible full recognizer training pipeline or weights. The paper mechanism is complementary to Agent A's CTC baseline; no result here is claimed for the published system. |
| Phan et al., Interspeech 2025, [Finland Swedish low-resource MDD](https://www.isca-archive.org/interspeech_2025/phan25_interspeech.pdf), DOI 10.21437/Interspeech.2025-1375 | Trains multilingual wav2vec 2.0 from L1 spontaneous speech with entropy regularization, then temperature scaling and top-k normalization for MDD. The published test reports a precision/recall trade-off on a different language and corpus; it motivates native-only uncertainty calibration, but does not establish L2-ARCTIC gains. | [Aalto-Speech/FinSwedish](https://github.com/aalto-speech/FinSwedish/) is linked from the original paper. Adapting its language-specific L1 pipeline is not a direct fit for the current labels; no copy or benchmark was run. |
| Lavechin & Hueber, EMNLP 2025, [acoustic invariance for articulatory learning](https://aclanthology.org/2025.emnlp-main.1217/) | Their paper reports intermediate wav2vec 2.0 representations that balance phonetic discrimination and speaker invariance in a vocal-imitation/articulatory task. This supports investigating speaker-robust representations, but its objective and endpoint differ from MDD. | A recent, adjacent representation result; used as rationale only. |
| Dao et al., arXiv 2026, [speaker-invariant spoofing detection with GRL and variational bottleneck](https://arxiv.org/abs/2606.08678) | Uses a teacher-student speaker-invariance objective and a variational information bottleneck for spoofing detection. Abstract reports relative EER reduction on nine spoofing datasets. That is not MDD evidence. | No verified implementation repository was located in this focused scan. We tested a lower-cost supervised GRL analogue using existing caches; results above are our own and not a reproduction of the spoofing paper. |
| Chang et al., Interspeech 2023, [speaker-invariant clustering (Spin)](https://github.com/vectominist/spin) | Swapped prediction between original and speaker-perturbed speech to preserve content while removing speaker information. This is a plausible self-supervised alternative to an adversarial speaker classifier. | Official code is available, but it requires waveform augmentation, s3prl dependencies and GPU fine-tuning; no adaptation was run in this campaign. If GRL remains weak, this is a stronger non-label-adversarial follow-up candidate, subject to a controlled TRAIN-only setup. |

### New method actually tested

The tested non-CTC method is the small speaker-adversarial projection described above. It is inspired by domain-invariant learning but is not a paper reproduction. Its target classes are correct/substitution/omission; its gap head predicts positive/negative insertion at every N+1 gap. The experiment runs entirely from the local frozen feature cache and CPU. It produces valid grouped OOF scores but no development checkpoint, threshold policy, or claim of better DEV performance. The insertion event-OOF AP rises by .0038 with event-level concatenation, while equally weighting folds slightly lowers it and the per-fold effects are inconsistent. Verdict: **no evidence-backed generalization improvement; do not integrate**.

## Reproduction and artifact map

- Source: `experiments/rq1_method_bench_03/agent_c/train_speaker_adversary.py`
- Training/OOF receipt: `experiments/rq1_method_bench_03/agent_c/speaker_grouped_oof_receipt.json`
- Event-level predictions: `experiments/rq1_method_bench_03/agent_c/speaker_grouped_oof_predictions.npz`
- Fold-level metrics: `experiments/rq1_method_bench_03/agent_c/speaker_grouped_oof_metrics.json`
- EVAL-V3.1 exact event join, raw AP, and coverage receipt: `experiments/rq1_method_bench_03/agent_c/speaker_grouped_oof_EVAL-V3.1.json`
- Speaker AP and support: `experiments/rq1_method_bench_03/agent_c/speaker_support_and_ap.csv`
- Split-by-speaker event supports/prevalences: `experiments/rq1_method_bench_03/agent_c/speaker_split_support.csv`
- Existing U2 per-speaker source: `experiments/rq1_unified_01/agent_c/RQ1-U2-FLAT-TWOHEAD-S101_EVAL-V3.1-AGENT-C.json`
- Speaker-grouped command: `.venv/bin/python -u experiments/rq1_method_bench_03/agent_c/train_speaker_adversary.py`
- Utterance-random command: `OMP_NUM_THREADS=4 .venv/bin/python -u experiments/rq1_method_bench_03/agent_c/train_utterance_random_oof.py`
- Random OOF receipt/predictions/metrics: `utterance_random_oof_receipt.json`, `utterance_random_oof_predictions.npz`, `utterance_random_oof_metrics.json` in the same `agent_c/` directory.
- Device/runtime/versions, fold membership, class supports, input hashes, code hash, prediction hash, and event-OOF vs fold AP definitions are recorded in the receipt.

## Limitations and recommended next step

Four folds are only four fold compositions and the data are still from one corpus. The random-vs-grouped comparison uses a single seed and does not disentangle speaker, utterance, or label-composition effects. The 12 speaker groups are balanced in utterance count but not in error prevalence. The DANN uses explicit speaker labels as auxiliary TRAIN supervision, with only nine known speaker classes per fold. The acoustic feature path remains crop-based and inherits the underlying alignment/center preparation. No acoustic-condition stratification, repeated training seeds, separate per-phoneme shift test, or downstream fusion is claimed.

The most useful next step is to join Agent A's genuine phone-CTC posteriors with Agent B's probabilistic event scores, evaluate them using speaker-grouped TRAIN OOF and fixed DEV_CAL policy, and retain the DANN features only if they add cross-speaker ranking evidence. A future self-supervised Spin-style intervention would be a distinct method family; it should be attempted only if the new sequence evidence and grouped diagnostics justify the extra encoder work.
