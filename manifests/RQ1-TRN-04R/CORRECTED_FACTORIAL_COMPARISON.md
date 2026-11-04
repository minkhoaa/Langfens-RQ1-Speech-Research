# Final TRN-04R factorial metric report — v2 shared cache

**Final metric version: v2 shared encoder cache.** Workstream B recomputed both endpoints from v2 at `/home/khoa/.cache/langfens/trn04r/genuine-z-v2-shared-20261104T084300Z/predictions/`. Both full-precision B reports are byte-identical to private `metrics/*.json`; the independent standard-library auditor agrees with both reports and stored `audits/*.json`. Fixed-budget Level A/B calculations fail in all four arms. Private-store readback/review remains pending, so no overall PASS is claimed.

## Provenance, source integrity and v2 input hashes

Coordinator's final v2 inference run finished successfully (exit 0, 155.63740944862366 seconds). `predictions/RUN_MANIFEST.json` identifies source commit `c60a84fb044f049dc30f61046f819ff0ac36dc72`, clean source state, verified unchanged source hashes, zero backward passes/optimizer updates, and one process producing all eight arm/endpoints from one shared immutable cache. It records 600 shared-cache utterances and cache identity SHA-256 `a9744f39fc4ae36ab3c688cf6eaa1c1ab37b2dde730a1e6c5207fa718084bc56`.

The recompute CLI and metrics implementation match the frozen c60 snapshot by SHA-256 (`audit_scripts/recompute_trn04r_metrics.py`: `507f872e922877a82b685455481ef224a2f02efc6352d31b2318af0bcbb731a7`; `src/langfens_prep/trn04r/metrics.py`: `717a45d072f477b0dd1811cc0b6ea9f846dea4b9991c8a7cfcd432e7c78994bb`). The auditor's flat-layout support is a later layout-only parent patch (SHA-256 `817b7a073a97dcebbc1e1f8ed6b303b7a6b2d844156e56c24fbcb4c233a2e894`); it does not alter frozen runtime/model code.

The 64 v2 JSONL metric inputs (32 files per endpoint) are hashed individually in `INDEPENDENT_METRIC_RECOMPUTATION.json`. Their canonical aggregate input-manifest SHA-256 is `6f6903b05e8ce667045e4734000ab67980d4acdc0bd3125e457fee630f038567`. Other provenance SHA-256 values: source snapshot `7fdf1233f284a6a10ec5891d30ae48393a49367b1ccf1c780063b41d331619c8`; run manifest `a1872589948a4d587414168400f066da03152167e7e04bb884b1846b6894ff50`; feature-identity file `a9744f39fc4ae36ab3c688cf6eaa1c1ab37b2dde730a1e6c5207fa718084bc56`.

B output/audit agreement:

| Endpoint | Private metrics SHA-256 = B recomputation SHA-256 | Private audit SHA-256 = B audit SHA-256 | Independent audit |
|---|---|---|---|
| fixed_budget_3600 | `ac9c1d476f5cce124d630570fd569effef161623255b4aeb7f05e45576c4c2e9` | `04b237db0f8112007dad33efe0497cf75e8032862bc9f0bf3ac7d779782f2d75` | `INDEPENDENT_RECOMPUTATION_AGREES`, all 4 arms verified |
| best | `f41e763a6c5c02ef8dbb42bbc560d02bcb6d6c7ab5cb11c4bfc686058c778c34` | `0fb23935cae75733be1ea9ae08c7e710bd503338376bd62b99d8ddd7e803a8f5` | `INDEPENDENT_RECOMPUTATION_AGREES`, all 4 arms verified |

Each arm: 9,956 gold = 9,919 mapped + 37 unmapped; 37 separate prompt extras; 9,881 reference CTC phones. Auditor covers global/speaker/phone AP, CAL thresholds and deployed TP/FP, omission FN, global/speaker PER, runtime freeze and gold-field audit.

## v1 to v2 comparison

The coordinator-reported v1 cache review found `fp32_feature_sha` differing between T2 and C0 on 464/600 rows, while T1/T3 matched C0 and waveform/4D-cache hashes matched. Cause remains unresolved; no cause is inferred. V1 raw outputs remain preserved. V2 uses the one shared cache across the eight heads.

Recomputed comparison: all CAL-15 omission operating counts and rates are identical between v1 and v2 (the CAL threshold values themselves can shift slightly); registered fixed-budget Level A/B pass/fail statuses are unchanged. Across all global AP families, both endpoints and all arms, the maximum absolute v1→v2 AP delta is `0.000011630032492992992`, at fixed-budget T3 legacy substitution AP (v1 `0.31159843681069727`; v2 `0.31161006684319026`). Use v2 values below and in the JSON as final metrics; do not copy v1 exact AP values into final results. The version-difference ledger SHA-256 is `ea7a774001442899a861ed2ff9d8d77964211e387e1a5181093d419ec2d54b19`.

## Fixed-budget 3,600-update endpoint — full-precision metrics

AP uses answered/aligned rows with exact grouped ties. `s_error` legacy and dedicated subtype channels remain separate. Acoustic gap is matched minus within-speaker cyclic-mismatch binary AP.

| Arm | Binary AP | Omission legacy AP (`s_error`) | Omission dedicated AP (`s_omit`) | Substitution legacy AP (`s_error`) | Substitution dedicated AP (`s_sub`) | Greedy PER | Binary acoustic gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.31656014550795514 | 0.033765022018883512 | 0.10469025371998517 | 0.30652480684992928 | 0.33379514686611167 | 0.23894342677866612 | 0.016813967215313641 |
| T1 | 0.31100468425723188 | 0.024291675478182017 | 0.029927641127363395 | 0.30863916406672176 | 0.34103724127670482 | 0.23904463111021151 | 0.011733800163423347 |
| T2 | 0.3181589435136053 | 0.031205329125611664 | 0.08646346950159671 | 0.31086325334417908 | 0.33531642642509474 | 0.23894342677866612 | 0.020901151639767301 |
| T3 | 0.3138968158730544 | 0.025011971166871809 | 0.031705545989403325 | 0.31161006684319026 | 0.33782538792514943 | 0.23904463111021151 | 0.018138827539707636 |

### Primary omission operating point: dedicated `s_omit`, CAL 15%

`FP=FP_correct+FP_sub`; abstained true omissions remain FN. FA/1,000 uses all 9,956 gold positions.

| Arm | Threshold | TP | FP correct | FP sub | FN | Precision | All-gold recall | FA / 1,000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.058004841208457954 | 80 | 999 | 59 | 65 | 0.070298769771529004 | 0.55172413793103448 | 106.26757734029731 |
| T1 | 0.10588592290878297 | 17 | 591 | 13 | 128 | 0.027375201288244767 | 0.11724137931034483 | 60.666934511852148 |
| T2 | 0.059713609516620643 | 67 | 922 | 60 | 78 | 0.063870352716873219 | 0.46206896551724136 | 98.633989554037768 |
| T3 | 0.10436487942934038 | 22 | 493 | 22 | 123 | 0.040968342644320296 | 0.15172413793103448 | 51.727601446363998 |

### Secondary CAL 5% and 10% omission operating points

Each cell is `threshold; TP/FP; precision; all-gold recall; FA/1,000`.

| Arm | CAL 5% | CAL 10% |
|---|---|---|
| C0 | 0.13683906197547915; 27/176; 0.13300492610837439; 0.18620689655172415; 17.677782241864204 | 0.09212201833724977; 50/520; 0.08771929824561403; 0.34482758620689657; 52.229811169144234 |
| T1 | 0.13905045390129092; 5/81; 0.058139534883720929; 0.034482758620689655; 8.1357975090397758 | 0.12292923033237459; 11/294; 0.036065573770491806; 0.07586206896551724; 29.5299316994777 |
| T2 | 0.13237573206424716; 18/144; 0.1111111111111111; 0.12413793103448276; 14.463640016070711 | 0.092629268765449538; 43/481; 0.082061068702290074; 0.29655172413793102; 48.312575331458419 |
| T3 | 0.14024150371551516; 2/113; 0.017391304347826087; 0.013793103448275862; 11.349939734833267 | 0.12201289832592012; 8/266; 0.029197080291970802; 0.055172413793103448; 26.717557251908396 |

### Fixed-budget registered absolute gates

All four v2 cells fail both joint levels. C0/T2 pass recall, binary AP, legacy substitution AP and PER but fail precision/FA. T1 fails precision, recall, FA and binary AP; T3 fails precision, recall and FA. T3 passes the binary AP threshold narrowly; the matched acoustic-gap requirement passes all arms but does not offset other failures.

| Level | Requirements | C0 | T1 | T2 | T3 |
|---|---|---|---|---|---|
| A | Precision ≥.25; all-gold recall ≥.30; FA/1k ≤30; binary AP ≥.313619; legacy sub AP ≥.303974; PER ≤.246980 | FAIL | FAIL | FAIL | FAIL |
| B | Precision ≥.30; recall ≥.40; FA/1k ≤15; same AP/sub/PER criteria; acoustic gap ≥.010 | FAIL | FAIL | FAIL | FAIL |

## SELECT-best endpoint — full-precision metrics

Fixed-budget registered gates are not applicable to this checkpoint stratum.

| Arm | Binary AP | Omission legacy AP | Omission dedicated AP | Substitution legacy AP | Substitution dedicated AP | Greedy PER | Binary acoustic gap | CAL-15 precision / recall / FA per 1,000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.30946171260621264 | 0.035882129896804386 | 0.11094248775038876 | 0.29762524338818663 | 0.32335432799185815 | 0.24936747292784131 | 0.021636882356507903 | 0.071258907363420429 / 0.41379310344827586 / 78.545600642828447 |
| T1 | 0.30814067105141246 | 0.023895988797492256 | 0.025631732543235752 | 0.30630943553820622 | 0.33999208887453342 | 0.28236008501163851 | 0.021497837980288037 | 0.03125 / 0.082758620689655171 / 37.364403374849338 |
| T2 | 0.3181589435136053 | 0.031205329125611664 | 0.08646346950159671 | 0.31086325334417908 | 0.33531642642509474 | 0.23894342677866612 | 0.020901151639767301 | 0.063870352716873219 / 0.46206896551724136 / 98.633989554037768 |
| T3 | 0.28191227805391811 | 0.026971582616048796 | 0.024483826029202127 | 0.27395037926893101 | 0.32551797030706686 | 0.24946867725938671 | 0.021448437593927672 | 0.023622047244094488 / 0.10344827586206896 / 62.274005624748895 |

Best endpoint secondary points (same field order as fixed-budget table):

| Arm | CAL 5% | CAL 10% |
|---|---|---|
| C0 | 0.11644901335239412; 29/159; 0.15425531914893617; 0.2; 15.970269184411411 | 0.075900241732597365; 47/461; 0.092519685039370081; 0.32413793103448274; 46.303736440337488 |
| T1 | 0.19713181257247928; 0/62; 0; 0; 6.2274005624748892 | 0.16556027531623843; 7/179; 0.037634408602150539; 0.048275862068965517; 17.979108075532341 |
| T2 | 0.13237573206424716; 18/144; 0.1111111111111111; 0.12413793103448276; 14.463640016070711 | 0.092629268765449538; 43/481; 0.082061068702290074; 0.29655172413793102; 48.312575331458419 |
| T3 | 0.17374667525291446; 2/121; 0.016260162601626018; 0.013793103448275862; 12.153475291281639 | 0.15397538244724276; 7/302; 0.022653721682847898; 0.048275862068965517; 30.333467255926074 |

## Matched contrasts and factorial interactions — v2 full precision

Each row is `T1−C0 / T2−C0 / T3−C0 / (T3−T2−T1+C0)`. Higher is better for AP, precision, recall and acoustic gap; lower for FA and PER. Descriptive two-speaker VAL contrasts only.

| Measure | Fixed-budget contrasts / interaction | Best contrasts / interaction |
|---|---|---|
| Binary AP | −0.0055554612507232592 / 0.0015987980056501594 / −0.0026633296349007352 / 0.0012933336101723647 | −0.001321041554800173 / 0.0086972309073926612 / −0.027549434552294527 / −0.034925623904887015 |
| Dedicated omission AP | −0.074762612592621763 / −0.018226784218388456 / −0.072984707730581841 / 0.020004689080428378 | −0.085310755207153005 / −0.024479018248792048 / −0.08645866172118663 / 0.023331111734758422 |
| CAL-15 omission precision | −0.042923568483284241 / −0.0064284170546557856 / −0.029330427127208708 / 0.020021558410731319 | −0.040008907363420429 / −0.0073885546465472102 / −0.047636860119325941 / −0.00023939810935830197 |
| CAL-15 all-gold recall | −0.43448275862068964 / −0.089655172413793116 / −0.40000000000000002 / 0.12413793103448278 | −0.33103448275862069 / 0.048275862068965503 / −0.31034482758620691 / −0.027586206896551724 |
| CAL-15 FA / 1,000 | −45.60064282844516 / −7.6335877862595396 / −54.53997589393331 / −1.3057452792286171 | −41.181197267979108 / 20.088388911209321 / −16.271595018079552 / 4.8212133386902281 |
| Legacy substitution AP | 0.002114357216792484 / 0.0043384464942498036 / 0.0050852599932609821 / −0.0013675437177813055 | 0.0086841921500195873 / 0.013238009955992447 / −0.023674864119255623 / −0.045597066225267657 |
| Dedicated substitution AP | 0.0072420944105931473 / 0.0015212795589830685 / 0.0040302410590377646 / −0.0047331329105384512 | 0.016637760882675268 / 0.01196209843323659 / 0.0021636423152087136 / −0.026436217000703144 |
| Greedy PER | 0.00010120433154539588 / 0 / 0.00010120433154539588 / 0 | 0.032992612083797196 / −0.010424046149175192 / 0.00010120433154539588 / −0.022467361603076635 |
| Binary AP acoustic gap | −0.0050801670518902942 / 0.0040871844244536604 / 0.0013248603243939949 / 0.0023178429518306287 | −0.00013904437621986565 / −0.00073573071674060131 / −0.00018844476258023102 / 0.00068633033038023594 |

## Historical score-family correction and claim boundary

The historical T1/T3 alarm explosion (985.436 false alarms/1,000; 9,811 false alarms) is not a valid `s_omit` operating result. The old evaluator calibrated from deletion probability but fell back to `s_error` on VAL where the class-focal serializer left `s_omit` null. V2's corrected T1/T3 CAL-15 FA rates are 60.666934511852148 and 51.727601446363998 per 1,000. The score-family explanation is independent of the v1 cache mismatch; the latter's cause remains unknown.

The final v2 calculations show all arms fail the joint fixed-budget gates, although individual binary/substitution/PER criteria pass. Results are one seed and two VAL speakers; contrasts are descriptive, not inferential. They do not establish population superiority, deployment approval, or that focal-loss/acoustic methods generally fail. V2 raw outputs and v1 originals remain preserved in private storage. Parent private-store readback/review remains pending; therefore no overall campaign PASS is asserted.
