# P-SUP benchmark comparison V1

## Initial actual inference results — acceptance pending remediation

These numbers are independently recomputed from the initial **actual fresh CPU FP32** runtime predictions and gold joins, not copied from the preparation report. Evidence: `INITIAL_FRESH_RUN_INDEPENDENT_RECOMPUTATION.json`; raw bytes preserved under private HF revision `0d50af15823b67b27766551545f284c6aa33d00a`, prefix `benchmarks/psup-v1/initial-fresh-inference-20261104`. The initial runner's summaries/provenance/coverage failed review; corrected versioned runs are required before a completion claim. These tables preserve the initial evidence and are not an assertion that the whole benchmark passed.

### Protocol O — full canonical positions (N=9956)

| Arm / endpoint | Actual updates | Binary AP | Omission AP `s_error` | Dedicated omission AP `s_omit` | Substitution AP `s_error` | Dedicated substitution AP `s_sub` |
|---|---:|---:|---:|---:|---:|---:|
| T-OP fixed |3600|0.324081|0.037610|0.107724|0.312767|0.338962|
| B-Control fixed |3600|0.322633|0.038033|unsupported|0.311232|unsupported|
| T-OP SELECT-best |2700|0.321707|0.035379|0.099018|0.311845|0.331358|
| B-Control SELECT-best |2250|0.327272|0.034692|unsupported|0.319735|unsupported|

### Genuine Protocol Z — prompt-only aligned gold positions (N=9919)

| Arm / endpoint | Actual updates | Binary AP | Omission AP `s_error` | Dedicated omission AP `s_omit` | Substitution AP `s_error` | Dedicated substitution AP `s_sub` |
|---|---:|---:|---:|---:|---:|---:|
| T-OP fixed |3600|0.318616|0.033942|0.098566|0.308970|0.334103|
| B-Control fixed |3600|0.316196|0.034288|unsupported|0.306581|unsupported|
| T-OP SELECT-best |2700|0.315957|0.031754|0.086877|0.307971|0.325620|
| B-Control SELECT-best |2250|0.321359|0.031754|unsupported|0.314797|unsupported|

The four Z comparisons use identical eligible aligned keys and frontend policy; inference uses batch1, whereas initial O uses batch2. Do not interpret O-minus-Z differences as canonical-query effects alone, and never directly rank their different populations. Best endpoints have unequal training exposure, so retain fixed3600 as the exposure-matched contrast.

### Retrospective VAL operating-point caution

Initial T-OP fixed dedicated omission at correct-token FPR<=15%:
- O: TP106/145, recall73.1034%; subset precision7.4386%, full deployed omission-event precision7.1046%.
- Z aligned: TP96/137, recall70.0730%; subset precision6.7989%, full deployed omission-event precision6.4909%.

These are **VAL-derived diagnostic thresholds**, not CAL-calibrated deployment guarantees. High recall does not imply acceptable omission-feedback precision. The initial source used a negative-only threshold candidate set and incorrectly reported O recall105/145=72.4138%; the independent all-score grouped-tie calculation yields106/145 at the same achieved FPR. Full TP/FP/TN/FN, thresholds, achieved FPR, precision/recall/F1 at5/10/15% are in the machine receipt.

## Archival baselines — separate evidence stratum

Independently recomputed from recovered fullprecision archives on the identical9919-key Z-aligned cohort (`BASELINE_ARCHIVAL_RECOMPUTATION.json`):

| Baseline | Binary AP | Omission AP `s_error` | Substitution AP `s_error` | Inference evidence |
|---|---:|---:|---:|---|
| E0 |0.313530|0.032564|0.304811|archival-only|
| P-SCORE SF-S |0.259094|0.029411|0.246800|archival-only|
| P-SCORE SF-SD |0.250049|0.051026|0.222544|archival-only|
| P-SCORE GOP-SA |0.254370|0.049846|0.228104|archival-only|

Do not mix these with full9956 O rankings. E0 used original TRAIN2399, while P-SUP used reduced FIT1799; E0 is not the matched experimental control. P-SCORE score families are raw graph-derived error rankings, not subtype probabilities. Greedy-DP answers only4212tokens and is excluded from this matched-population table. Baseline arithmetic is reproduced, but no new E0/P-SCORE inference was executed in this process.

## Explicit reconciliation with prior reports

1. Original P-SUP serialized `Z-PROMPT-V1` rows were actually annotation-canonical O; their AP arithmetic reproduces, but the original prompt-only interpretation does not.
2. T-OP and B-Control runs reached3600successful/0skipped updates, but SELECT-best checkpoints are2700and2250respectively. Fixed endpoints are epoch7 zero-based (8epochs at450updates/epoch), not the historical baseline's epoch6 convention.
3. Executed training source was `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f`, not the later evaluator commit9ad1e5e or earlier staging pin df55f12.
4. The first implementation's pre-inference 'reproduced' comparison table, O-aligned-as-Z rows, negative-only FPR threshold result and conclusive 'tradeoff resolved/proved' language were rejected. Preserve that report only as historical agent evidence.
5. The initial word-final G2P-index proxy was invalid for comparison. Correct frozen gold midpoint-to-word-tier keys independently reconcile to1871full word-final consonants/95deletions and1866aligned word-final consonants/93deletions, with exact equality to archived P-SCORE aligned keys. Do not use earlier87-deletion proxy numbers.
6. Initial per-run durations cover O forward inference only; summing them does not measure total O+Z+mismatch wall time.
7. B-Control subtype logits remain unidentified; unsupported dedicated metrics are null, never zero. Different valid scoring policies may be discussed only with named score semantics, not as invented like-for-like subtype outputs.
8. Two speakers/one seed and repeatedly reused VAL do not establish population significance or acoustic diagnosis readiness. No causal or broad pedagogical deployment claim is accepted.

## Pending acceptance boundaries

Corrected raw outputs, full coverage, mismatch prediction retention/equal batching, independently decodable PER records and source/output manifests are undergoing remediation. A frozen permitted FIT-only prior artifact has not been recovered; no new prior model is fitted locally. CAL-derived thresholds must not be claimed unless actual CAL inference/selection is executed and recorded. These are explicit unresolved measurements, not fabricated access failures.
