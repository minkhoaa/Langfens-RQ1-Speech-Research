

## Follow-up: G3-S101 exact-gap inference perturbations (CPU-only)

Ran `g3_gap_inference_perturbations.py` on the frozen G3-S101 gap-head checkpoint in CPU inference mode. This is an inference-time perturbation study, not a retrained modality ablation. The original 1614D U2 gap context is `[left acoustic 768, right acoustic 768, previous-phone one-hot 39, next-phone one-hot 39]`; G3 appends the unchanged 6D `gap_X`. For every arm, canonical IDs, gold labels, and the matched U2-S101 target probabilities were held fixed. Only gap-head input features were changed. Model predictions in the baseline reproduced stored G3 scores to max absolute error `7.75e-7` on DEV_CAL and `5.96e-7` on DEV_SEARCH.

| DEV_SEARCH arm | Exact-gap AP-I | Macro-3 error AP | CAL20 Macro-3 F1 | CAL20 FWR | CAL20 insertion F1 | Δ AP-I vs base | Δ Macro-3 AP vs base |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | .134162 | .230280 | .124811 | .18197 | .141026 | — | — |
| Zero 1536D W2V acoustic blocks | .046932 | .201203 | .088138 | .15192 | .031008 | −.087230 | −.029077 |
| Zero 6D `gap_X` only | .083611 | .213430 | .110436 | .16694 | .097902 | −.050551 | −.016850 |
| Shuffle paired W2V blocks across gap events | .011221 | .189299 | .077802 | .14858 | .000000 | −.122941 | −.040980 |
| Reference-only (zero acoustic + 6D; keep phone one-hots) | .019925 | .192201 | .077802 | .14858 | .000000 | −.114237 | −.038080 |

All metrics are EVAL-V3.1. For each arm, DEV_CAL independently selected target/gap thresholds under CAL20 and those thresholds were applied to DEV_SEARCH; AP and Macro-3 AP are threshold-independent. CAL20 insertion confusion counts TP/FP/FN/TN were baseline `11/20/114/20209`; zero W2V `2/2/123/20227`; zero gap_X `7/11/118/20218`; shuffled acoustic `0/0/125/20229`; reference-only `0/0/125/20229`. Since target scores are fixed but the target threshold is jointly reselected under CAL20, the small Macro-3 F1 deltas are policy outcomes, whereas AP-I directly measures changed gap ranking. The JSON contains full per-arm policies, class metrics, and confusion matrices, including unrestricted-threshold diagnostics (which have high false-warning rates and are not the selected low-warning operating point).

The sensitivity ordering suggests both groups of input matter to G3: removing `gap_X` costs .05055 AP-I, while removing the W2V acoustic blocks costs .08723; removing/shuffling acoustics nearly collapses the rare-positive ranking (AP-I .01992/.01122 vs prevalence .00614). This is not a feature-attribution or causal decomposition: the interventions can move inputs far outside the TRAIN distribution, acoustic blocks are paired from arbitrary donor events in the shuffle, and their interactions are not additive. Reference-only is an OOD control and retaining phone one-hots does not establish a valid standalone reference model. The result supports reliance of this frozen G3 scorer on acoustic input and gap_X; it does not estimate gains from retraining without either modality.

Population counts were 10,043 targets/10,343 gaps (102 insertion positives, 300 utterances) on DEV_CAL and 19,755 targets/20,354 gaps (125 insertion positives, 599 utterances) on DEV_SEARCH. SEARCH remains development-exposed and exploratory. No CUDA was used.

Reproduction: `.venv/bin/python experiments/rq1_unified_02/agent_c/g3_gap_inference_perturbations.py`. The script and `g3_gap_inference_perturbation_results.json` record the G3 checkpoint, U2 and G3 prediction, feature/context, evaluator adapter, and script SHA-256 values. No files outside this Agent C directory were edited for this follow-up.
