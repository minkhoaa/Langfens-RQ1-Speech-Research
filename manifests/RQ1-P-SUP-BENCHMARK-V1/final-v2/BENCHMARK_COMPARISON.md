# Recovered P-SUP — executed benchmark V2

## Disposition

**Four checkpoint configurations reproduced under both protocols; required frozen FIT-only prior control unavailable. Full-contract completion is NOT claimed.** These are actual frozen-weight CPU FP32 inference results, not a preparation plan. No training, backward pass, optimizer update or sealed TEST access occurred.

All newly measured numbers below have evidence status **reproduced**. E0/P-SCORE inference is **archival-only** (metric arithmetic independently recomputed). Unsupported subtype metrics and the missing prior are **unavailable**. Historical rounded prior AP is **reported-only**.

## Protocol O: oracle-canonical, 9,956 positions

| Arm / endpoint | Updates | Binary AP | Omit AP: s_error | Omit AP: s_omit | Sub AP: s_error | Sub AP: s_sub | Greedy PER |
|---|---:|---:|---:|---:|---:|---:|---:|
| T-OP fixed |3600|0.324085|0.037610|0.107724|0.312772|0.338959|0.241980|
| B-Control fixed |3600|0.322600|0.038033|unavailable|0.311195|unavailable|0.241980|
| T-OP SELECT-best |2700|0.321704|0.035379|0.099026|0.311843|0.331351|0.244611|
| B-Control SELECT-best |2250|0.327270|0.034692|unavailable|0.319732|unavailable|0.252910|

O labels: 8,799 correct / 1,012 substitutions / 145 omissions. O omission population: 8,944; substitution population: 9,811. Queries use annotation-derived identities, never a prompt-only claim.

## Protocol Z: genuine prompt-only, 9,919 aligned positions

| Arm / endpoint | Updates | Binary AP | Omit AP: s_error | Omit AP: s_omit | Sub AP: s_error | Sub AP: s_sub | Greedy PER |
|---|---:|---:|---:|---:|---:|---:|---:|
| T-OP fixed |3600|0.318619|0.033942|0.098560|0.308974|0.334102|0.241980|
| B-Control fixed |3600|0.316194|0.034288|unavailable|0.306580|unavailable|0.241980|
| T-OP SELECT-best |2700|0.315955|0.031754|0.086886|0.307969|0.325617|0.244611|
| B-Control SELECT-best |2250|0.321359|0.031755|unavailable|0.314796|unavailable|0.252910|

Z aligned labels: 8,774 correct / 1,008 substitutions / 137 omissions. Omission population: 8,911; substitution: 9,782. Every cell also retains 37 unmapped gold abstentions (25 correct / 4 sub / 8 omit) and 37 extra prompt positions: all 9,956 gold positions are accounted for. AP is conditional on aligned coverage, with identical eligible keys across arms and endpoints. No full-gold AP imputation. CAL operating-point recall below instead uses all gold positives. Do not rank the O and Z tables together.

B-Control's within-error logits are not identified subtype probabilities. Dedicated scores are null, not zero. Comparing T-OP dedicated omission with B-Control legacy omission compares explicitly different scoring policies; it is not a like-for-like subtype-probability comparison. On the shared legacy omission score, fixed T-OP does not improve AP over fixed B-Control.

## CAL-selected omission operating point: nominal 15% CAL correct-token FPR

Actual CAL inference: NCC/THV, 300 utterances; 8,054 O or 8,036 aligned Z correct scores. Thresholds use only these correct scores, grouped ties and score >= tau. They were not tuned on VAL. FPR ceilings are empirical CAL targets, not guaranteed VAL bounds. All-gold Z recall counts unaligned omissions as missed.

| Protocol | Arm / endpoint | Score | Threshold | TP/FP/TN/FN | Recall | Subset precision | Achieved VAL FPR |
|---|---|---|---:|---|---:|---:|---:|
|O|T-OP fixed|s_omit|0.066317663|91/1083/7716/54|0.627586|0.077513|0.123082|
|O|B-Control fixed|s_error|0.223545223|52/1069/7730/93|0.358621|0.046387|0.121491|
|O|T-OP best|s_omit|0.054014653|89/1038/7761/56|0.613793|0.078971|0.117968|
|O|B-Control best|s_error|0.213849880|32/931/7868/113|0.220690|0.033229|0.105807|
|Z|T-OP fixed|s_omit|0.066045389|82/1068/7731/63|0.565517|0.071304|0.121377|
|Z|B-Control fixed|s_error|0.223344855|41/1056/7743/104|0.282759|0.037375|0.120014|
|Z|T-OP best|s_omit|0.053000309|81/1027/7772/64|0.558621|0.073105|0.116718|
|Z|B-Control best|s_error|0.217066891|24/897/7902/121|0.165517|0.026059|0.101943|

Subset omission precision excludes substitutions. Fixed T-OP full deployed omission-event precision is only 0.073864 O / 0.067825 Z at these thresholds. This is not feedback-readiness evidence. Z unresolved-correct FPR upper bounds add 25/8,799 to the displayed lower-bound/no-alarm FPR. Each `metrics.json` supplies full-precision thresholds, precision/recall/F1, confusion counts and FPR bounds at **5%, 10%, 15%**, for all supported score families. `retrospective_VAL_fpr_points` is separately labeled diagnostic, not calibrated.

For comparison with historical recall claims, retrospective T-OP fixed dedicated omission at VAL FPR <=15% gives O 106/145 = 0.731034, and Z aligned 96/137 = 0.700730 (all-gold 96/145 = 0.662069). These are NOT the CAL-selected recalls above.

## Acoustic controls and recognition

All cells use the same batch-one FP32 encoder features, waveform normalization, prompts and no-fixed-point same-speaker cyclic shift-1 mapping. Raw mismatched predictions and offline joins are retained, not just a historical gap.

| Arm / endpoint | O matched / mismatched binary AP | Z matched / mismatched binary AP |
|---|---|---|
|T-OP fixed|0.324085 / 0.304549|0.318619 / 0.301171|
|B-Control fixed|0.322600 / 0.300207|0.316194 / 0.294788|
|T-OP best|0.321704 / 0.301001|0.315955 / 0.297848|
|B-Control best|0.327270 / 0.304529|0.321359 / 0.301486|

Canonical-only matched control: **unavailable**. No frozen FIT1799 phone/position prior lookup was recovered from the inspected local artifacts/private HF inventory. This is an artifact gap, not missing dataset access; CAL and VAL are accessible. A new fit is prohibited. The historical TRAIN2399 prior AP 0.303063 is reported-only ([baseline summary](../../RQ1-TRN-02A/BASELINE_COMPARISON.json)), not a matching FIT-only control.

Greedy realized-phone recognition, 9,881 reference phones, identical O/Z hypothesis per endpoint:

| Endpoint | S / D / I | Edits | PER |
|---|---|---:|---:|
|T-OP fixed|1710 / 517 / 164|2391|0.241980|
|B-Control fixed|1711 / 513 / 167|2391|0.241980|
|T-OP best|1727 / 470 / 220|2417|0.244611|
|B-Control best|1768 / 518 / 213|2499|0.252910|

Recognition S/D/I is edit alignment against realized phones, not human annotated substitution/omission counts. Human omissions have no realized reference phone. Hypotheses, references, integer edit counts and per-speaker PER are saved for recomputation.

## Speaker and word-final breakdown

| Endpoint | O binary AP NJS / RRBI | Z binary AP NJS / RRBI | WFC dedicated omission AP O / Z |
|---|---|---|---|
|T-OP fixed|0.395531 / 0.266961|0.389152 / 0.261812|0.183310 / 0.184137|
|B-Control fixed|0.397691 / 0.265452|0.390718 / 0.258457|unavailable|
|T-OP best|0.404781 / 0.254860|0.397485 / 0.250007|0.180851 / 0.171154|
|B-Control best|0.413613 / 0.259275|0.404777 / 0.254511|unavailable|

Word-final consonants use the frozen offline midpoint-to-word tier rule, not a G2P-index proxy: 1,871 gold / 95 omissions; 1,866 aligned / 93 omissions. Full family counts, AP and operating points for each speaker/stratum are in cell metrics. Only two speakers and one seed; sparse omission strata and reused VAL preclude population-significance claims.

## Archival-only baselines: identical Z-aligned population

| Baseline | Binary AP | Legacy omission AP | Legacy substitution AP | Inference evidence |
|---|---:|---:|---:|---|
|E0|0.313530|0.032564|0.304811|archival-only|
|P-SCORE SF-S|0.259094|0.029411|0.246800|archival-only|
|P-SCORE SF-SD|0.250049|0.051026|0.222544|archival-only|
|P-SCORE GOP-SA|0.254370|0.049846|0.228104|archival-only|

Gold keys and labels were independently cross-checked; see `../BASELINE_ARCHIVAL_RECOMPUTATION.json`. These are not new contemporaneous baseline inference. E0 used TRAIN2399, whereas P-SUP used FIT1799; E0 is not the matched control. P-SCORE uses raw graph error scores, not identified subtype probabilities. Greedy-DP only answers 4,212 tokens and is excluded from this ranking. No O9956 versus Z9919 AP ranking.

## Reconciliation and interpretation

1. The old P-SUP rows labeled Z were actually O. Their arithmetic reproduces on the original bytes; prompt-only interpretation is retracted. This run freezes genuine Z runtime before first annotation access.
2. Both runs reached 3,600 successful updates / zero skips, but selected endpoints are 2,700 T-OP and 2,250 B-Control. Fixed endpoints are epoch 7 zero-based, not the original baseline's epoch 6 convention.
3. Executed training SHA is `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f`, not `9ad1e5e` or staging `df55f12`. Unchanged head/encoder/trainer/runtime/label implementation was checked against that commit.
4. Fixed endpoint embedded history ends at 3,150 because it is saved before final epoch-history append. All optimizer/scheduler/payload counters and external histories support 3,600; this discrepancy is preserved.
5. Final O AP differs slightly from initial/archival arithmetic because this is a separately named batch-one frozen-feature FP32 stratum, rather than the initial O batch-two run. For example B-Control fixed binary AP is now 0.322600, not 0.322633. Do not silently overwrite or promise bit parity across batching. Z also has small backend numerical differences; exact source/dependencies/hashes accompany this stratum.
6. Initial negative-only FPR candidate search missed a valid positive-score operating point. The all-score grouped-tie recomputation recovers O T-OP 106/145 at diagnostic FPR15. An initial wrong WFC proxy and missing unmapped rows were rejected, not retrospectively accepted.
7. At fixed exposure T-OP binary AP is slightly above B-Control, legacy omission AP is slightly below, and PER is equal. Dedicated treatment omission ranking is better than the named legacy control policy; this does not identify B-Control's subtype logits. At selected endpoints B-Control binary AP is higher, with unequal exposure. No blanket 'tradeoff solved', causal diagnosis or deployment conclusion follows.
8. The unexecuted prior is an explicit remaining control gap. All four inference cells, saved-token recomputation, calibration and private preservation are executed evidence, but the full requested control set is not complete.

Evidence, immutable storage, commands and safe restart: `../REPRODUCTION_REPORT.md`, `../FINAL_PRIVATE_STORAGE_RECEIPT.json`, `RUN_MANIFEST.json`, and `../PROCESS_CHECKPOINT.md`.
