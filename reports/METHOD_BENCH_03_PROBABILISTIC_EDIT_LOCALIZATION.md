# Probabilistic edit-path and exact-gap localization — RQ1-METHOD-BENCH-03

## Final three-seed W2V2 result

The initial log-mel beam-3 run below is retained as an early, weaker recognizer diagnostic. Final primary comparisons use the frozen Wav2Vec2 layer-6 CTC head, beam-10 candidates and three recognizer seeds. On DEV_SEARCH, E3 path marginalization reached mean AP-S/O/I `.237754/.040945/.034013` and Macro-3 AP `.104238` across seeds 101/202/303. E3 beat matched E1 in S/O/I for every seed; E3 mean exact-gap AP is slightly above historical H05 `.02825`, but far below Gold-Only `.186715` and EXP43 `.285056`. Full per-seed E1/E2/E3 values, speaker supports and CAL-frozen policy results are in [Agent B report](METHOD_BENCH_03_B_PROBABILISTIC_EDIT.md).

Beam-10 candidate coverage remains inadequate for a full posterior interpretation: mean top-candidate CTC mass is only `1.3e-5` to `3.8e-5` on SEARCH, with median below `2.5e-12`; 567–576 of 599 utterances have ambiguous candidate edit paths. E3 exactly marginalizes minimum-cost edit paths within each candidate but only approximately covers sequence alternatives. Its path prior is uniform over equal-cost paths. Thus the empirical matched gain is evidence for reducing deterministic tie-breaking sensitivity, not evidence of calibrated joint edit posterior inference.

## Scoring definition

`experiments/rq1_method_bench_03/agent_b/probabilistic_edit.py` implements exact CTC sequence log probability by forward DP, truncated prefix beam, and dynamic programming over all minimum-cost Levenshtein paths. It emits N target scores for substitutions/deletions and N+1 exact-gap insertion probabilities (event means at least one insertion at that gap). Repeated phones require blank separation in CTC; utterance boundaries retain leading/trailing gaps. All equally optimal edit paths are marginalized, so ambiguity is explicit and no gold error label selects an alignment.

For E2/E3, candidates are normalized using exact CTC sequence scores over the unique supplied beam candidates plus canonical reference. E3 uses a uniform weight over minimum unit-cost edit paths within each candidate. This is a **finite-candidate conditional approximation**, not an exact posterior over all CTC sequences or a learned joint edit posterior. Candidate mass is reported and extremely low.

## Matched empirical metrics

DEV_CAL has 300 utterances / 10,043 targets / 10,343 gaps; DEV_SEARCH 599 / 19,755 / 20,354. N/N+1 coverage is complete. Same saved Agent A posterior outputs and event populations are used. SEARCH is exposed exploratory development data.

| Split / arm | AP-S | AP-O | AP-I exact gap | Macro-3 AP | Macro-3 F1 at saved hard decisions |
|---|---:|---:|---:|---:|---:|
| CAL E1 greedy hard edit | .18912 | .04858 | .00986 | .08252 | .09467 |
| CAL E2 beam-3 weighted, hard path | .20505 | .05615 | .00986 | .09035 | .12010 |
| CAL E3 beam-3 marginal paths | .26092 | .07058 | .01007 | .11386 | .11244 |
| SEARCH E1 greedy hard edit | .10043 | .01248 | .00614 | .03968 | .06393 |
| SEARCH E2 beam-3 weighted, hard path | .11065 | .01531 | .00614 | .04403 | .07386 |
| SEARCH E3 beam-3 marginal paths | .13106 | .01816 | .01218 | .05380 | .06828 |

E3 improves AP over matched E2 for all three classes on both splits, but hard-decision Macro-3 F1 is not uniformly better. On SEARCH, insertion AP improves +.00604 over E2, but remains far below Gold-Only .186715, EXP43 .285056, and historical frame-argmax hard mapping .02825. E3 omission AP .01816 remains below Gold-Only .159433 and EXP24 .225092. CTC sequence AP-S .13106 is far below Gold-Only .364602/H-002 .377810.

At fixed thresholds selected only on DEV_CAL (`CAL_MAX_ERROR_MACRO_F1`), DEV_SEARCH macro3 F1 / joint FWR: E1 .05883 / 63.7997; E2 .07928 / 54.8965; E3 .08687 / 52.9249. These operating points are warning-heavy and not operationally good; AP and policy F1 are distinct. Saved E1/E2/E3 default hard decisions had zero I true positives. P@K was not available in the existing receipts and is not invented.

## Ambiguity and truncation

| Beam | Mean candidate CTC mass | Median | Mean fraction of candidates with ambiguous path | Candidate posterior mass on ambiguous paths | Utterances ambiguous |
|---|---:|---:|---:|---:|---:|
| CAL B1 | 3.57e-7 | 3.99e-15 | .47467 | .98667 | 296/300 |
| CAL B3 | 9.54e-7 | 1.13e-14 | .70915 | .98769 | 299/300 |

Multiple minimum edit paths are common. Marginalization removes arbitrary tie-breaking, but the uniform measure over optimal paths is a modeling convention, not an acoustically calibrated posterior. Extremely low candidate mass precludes interpreting normalized beam weights as full sequence posterior. The truncation sensitivity is exposed via beam-1 vs beam-3 results; no reliable convergence is demonstrated.

## Validation and reproduction

Synthetic exhaustive fixtures passed: seven CTC sequence probabilities checked against path enumeration, eight edit alignment cases checked against exhaustive optimal-path enumeration including repeated phones and boundary gaps. They validate implementation only, not MDD improvement.

```sh
python experiments/rq1_method_bench_03/agent_b/synthetic_fixtures.py
# Existing saved-run reproduction commands (will regenerate/overwrite those ledgers):
python experiments/rq1_method_bench_03/agent_b/evaluate_ctc_edit.py --splits dev_cal dev_search --k 3
python experiments/rq1_method_bench_03/agent_b/apply_cal_policy.py --run-dir experiments/rq1_method_bench_03/agent_b/ctc_edit_run/beam_3 --beam-k 3
```

The latter commands were not rerun in this report; empirical outputs were read from saved receipts/ledgers. E0 historical frame-argmax and E4 alignment-free constrained scoring were not implemented as matched runs here. Avoid claims of superiority across unmatched generations. Agent B full receipt trail is under `experiments/rq1_method_bench_03/agent_b/ctc_edit_run/`.
