# RQ1-UNIFIED-01 — Đường đánh giá và phân tích so sánh

## Trạng thái

Đây là đánh giá CPU-only sau các lượt huấn luyện U2 (flat) và U3 (hierarchical) do Agent A hoàn tất. Các file dự đoán được đối chiếu cùng EVAL-V3.1 và chung population ID. Tất cả DEV_SEARCH đã exposed; số liệu là phát triển exploratory, không phải kiểm thử mù.

## Đường đánh giá bắt buộc

Entrypoint chính là `research.evaluation_v3.evaluator.UnifiedEvaluatorV3.evaluate(...)`; protocol và API được ghi tại `research/evaluation_v3/specification.md`. Không dùng `evaluator/unified_evaluator.py` legacy để tạo leaderboard mới: evaluator cũ điền xác suất giả cho score thiếu, mặc định nhãn CORRECT khi thiếu prediction, cho phép overwrite ID trùng và gọi Macro-3 target là `macro_3_f1`. Những hành vi này làm sai so sánh cho mục tiêu hiện tại.

Mỗi lần đánh giá cần có gold target/gap ledger của đúng split, target predictions theo khóa event và continuous class scores, gap predictions theo exact gap ID và continuous insertion score, cùng expected utterances. Hợp đồng dữ liệu giữ N target slots và N+1 gap slots trên mỗi utterance, gồm gap đầu và cuối. Duplicate ID, slot sai topology và ID thừa bị từ chối; prediction thiếu được giữ trong mẫu số như abstention/failure. Scores thô `score_*` phải tách khỏi xác suất `prob_*`; AP dùng continuous score đúng hướng và không đổi theo threshold. `binary_md_ap` chỉ tính khi có `binary_md_score` hoặc target probability simplex hợp lệ.

Báo cáo mỗi arm riêng: AP-S/O/I, `macro3_error_ap`, class-wise P/R/F1, `macro3_error_f1`, `macro4_mixed_domain_f1`, binary MD, exact-gap counts và FWR. Ghi rõ chọn threshold trên DEV_CAL và áp nguyên ngưỡng sang DEV_SEARCH; không dùng DEV_SEARCH để fit calibration, chọn checkpoint, hoặc chọn chính sách. Nếu chưa có calibration provenance, trạng thái confidence là chưa hiệu chuẩn hậu nghiệm. FWR là mô tả trade-off, không phải cổng loại.

## Mốc số học DEV_SEARCH

| Arm | Loại | AP-S | AP-O | AP-I exact-gap | Macro-3 Error AP | Macro-3 Error F1 | Macro-4 F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| U2 flat two-head seed 101 | Unified neural | 0.371360 | 0.185318 | 0.117992 | 0.224890 | 0.125421† | 0.330707† |
| U2 gap positive loss weight 3 | Unified neural ablation | 0.371360 | 0.185318 | 0.127349 | 0.228009 | 0.118343† | 0.325398† |
| U3 hierarchical two-head seed 101 | Unified neural, omission + conditional S/C | 0.362413 | 0.176117 | 0.123822 | 0.220784 | 0.072340‡ | 0.282708‡ |
| Gold-Only seed 101 | Neural | 0.364602 | 0.159433 | 0.186715 | 0.236917 | 0.300141* | 0.445779* |
| H-002 normalized attention | Neural target + Gold-Only gap scores | 0.377810 | 0.167294 | 0.186715 | 0.243940 | 0.3095† | 0.4592† |
| EXP24 | Score composition | 0.357737 | **0.225092** | 0.247628 | 0.276819 | 0.305855 | 0.450294 |
| EXP29 | Score composition | 0.378062 | 0.196539 | 0.272068 | 0.282223 | **0.342914** | **0.482871** |
| EXP36 | Score composition | **0.387088** | 0.204687 | 0.275924 | **0.289233** | 0.331090 | 0.470515 |
| EXP43 | Score composition | 0.363453 | 0.205771 | **0.285056** | 0.284760 | 0.332231 | 0.472479 |
| Phase 4 omission-specialist fusion | Score composition; five-seed mean | — | — | — | **0.3018** | 0.3178 | — |
| Phase 4 fixed headwise blend | Score composition; five-seed mean | — | — | — | 0.2955 | 0.3296 | — |

`*` Gold-Only values are the matching CAL_MAX_ERROR_MACRO_F1 entry from the closure ledger. `†` U2 selected DEV_CAL warning-budget 0.20; its F1 is shown for that explicitly named policy. H-002 unrestricted F1 (M3=.3095, Macro4=.4592) is a different policy and remains a historical diagnostic only. These operating-policy F1 values must not be ranked as if they used a common threshold. The source-specific five-seed fusion aggregate does not expose its per-class AP in this table; its macro value is retained separately rather than inventing components. The complete captured baseline rows and provenance are in `experiments/rq1_unified_01/agent_c/HISTORICAL_BASELINES.csv`.
`‡` U3 selected DEV_CAL warning-budget 0.20 under the same evaluator adapter. The U2 and U3 policy selection was on DEV_CAL; their SEARCH F1 figures are development estimates.

## U2 actual CUDA result — seed 101

Agent A trained `RQ1-U2-FLAT-TWOHEAD-S101` for 8 epochs on the frozen normalized Wav2Vec2 `hidden_states[6]` H-002 representation. The checkpoint has 653,893 learned parameters, trained on GTX 1650 in 24.88 s, peak allocated VRAM 41.33 MiB. Checkpoint SHA-256 is `1978633b08034904d566273cb29f3f8305c604ed86e3be1145e1bc70d6375c92`. The model predicts flat CORRECT/SUBSTITUTION/OMISSION probabilities and a newly trained binary insertion score for each exact gap. EVAL-V3.1 reports 19,755/19,755 targets and 20,354/20,354 gaps, zero missing predictions and zero simplex violations.

Against the matched Gold-Only seed 101 score cache, AP changes are S `+0.006758`, O `+0.025885`, I `−0.068723`, yielding Macro-3 Error AP `−0.012027` (0.224890 vs 0.236917). Thus the unified target head is learning useful target evidence, especially omission, but the newly learned gap head causes the aggregate ranking score to fall. Against H-002's target AP specifically, U2 S is `−0.006450` while O is `+0.018024`; H-002's insertion score is exactly the Gold-Only legacy score and is not an independent neural-gap comparator.

With the U2 DEV_CAL-selected 0.20 warning-budget policy, SEARCH class P/R/F1 is S `.4783/.0124/.0243`, O `.3299/.1531/.2092`, and exact-gap I `.3793/.0880/.1429`; Macro-3 Error F1 is 0.125421, Macro-4 is 0.330707, and observed joint FWR is 0.178631. At unrestricted thresholds, F1 rises to 0.280346 while FWR rises to 4.654 warnings/utterance. This illustrates why thresholds and ranking must remain separate. The remaining 1,746/1,768 S positives are still classified CORRECT at the 0.20 budget; substitution recall is the main target operating-point weakness.

The main failure is insertion ranking: AP-I 0.117992 is below Gold-Only/H-002 0.186715, EXP43 0.285056, and even the rare-class prevalence 0.006141 only by a modest absolute signal given the huge event imbalance. There are 560 insertion positives among 61,362 TRAIN gaps. U2's gap context concatenates averaged acoustic windows adjacent to the two intended-phone centers and the two intended-phone embeddings. The training gap loss fell to 0.012738 by epoch 8, while SEARCH AP-I remained low; this is consistent with overfit or a context-feature mismatch, but does not identify which cause. The smallest justified next insertion experiment is a controlled gap-context/window variant that represents both sides of the intended boundary more directly, retaining the same TRAIN labels, frozen encoder, seed and evaluation population. Avoid changing gap loss, architecture and target representation together.

Agent A also ran a genuine WAV inference smoke using the single U2 checkpoint and an intended-phone reference: 39 target outputs and 40 exact-gap outputs, simplex maximum absolute error `1.19e−7`, GTX 1650 inference 1.61 s and 438.76 MiB peak allocated VRAM for that utterance. This confirms the code path can produce N/N+1 outputs from a WAV, but it is one sampled smoke and does not establish full-population end-to-end throughput or independent intended-reference provenance.

### U3 hierarchy ablation

The matched hierarchical arm uses omission BCE over every target and conditional S/C binary loss only on non-omission targets, with valid simplex construction `pO`, `(1−pO)×pS|nonO`, `(1−pO)×pC|nonO`. Against U2, AP-S declines by 0.008947, AP-O by 0.009202, AP-I improves only 0.005830, and Macro-3 AP declines 0.004106. Macro-3 F1 at DEV_CAL warning budget 0.20 falls from 0.125421 to 0.072340. In this matched seed screen, explicit hierarchical omission specialization did not help either omission or substitution ranking; prefer the flat target head while addressing the exact-gap representation separately.

### U2 gap-loss-weight ablation

Changing only the insertion positive-loss weight to 3 leaves target scores and AP unchanged. On seed 101, AP-I rises by 0.009356 (0.117992→0.127349) and Macro-3 AP by 0.003119 (0.224890→0.228009), but CAL20 Macro-3 F1 falls by 0.007079 (0.125421→0.118343). The paired seed-202 gain and seed-303 regression, together with the supplied three-seed means, make the ranking result unstable and show no reproducible mean gain. This supports studying the gap-boundary representation instead of continuing the weight sweep.

The paired seed-202/303 comparisons reverse direction: GAPW3−U2 AP-I/Macro-3 AP deltas are +0.030188/+0.010063 for seed 202 and −0.039485/−0.013162 for seed 303. Across these two replication seeds, mean U2 AP-S/O/I/Macro-3 AP is .36681/.17188/.14950/.22940; GAPW3 is .36681/.17188/.14485/.22785. Their CAL20 Macro-3 F1 means are .10826 for U2 and .12095 for GAPW3, a mixed, policy-sensitive result. Including seed 101, the full three-seed arithmetic means from all six official JSONs are U2 AP-S/O/I/Macro-3 AP = .368327/.176356/.138999/.227894 and GAPW3 = .368327/.176356/.139019/.227901. The paired full-three-seed mean deltas are 0/0/+.000020/+.000007. Per-seed AP-I/Macro-3 AP deltas are seed 101 +.009356/+.003119, seed 202 +.030188/+.010063, and seed 303 −.039485/−.013162. These small mean differences and direction reversal show no reproducible gain from the weight increase. Source rows are preserved per seed in `HISTORICAL_BASELINES.csv`.

## Failure analysis and implications for new runs

1. **Omission ranking is the largest target gap.** H-002 improves on the same-seed Gold-Only from AP-O 0.159433 to 0.167294 (+0.007861), but remains 0.057798 below EXP24's 0.225092. A unified omission specialist should therefore be assessed first on AP-O and then on AP-S/AP-I deltas to expose negative transfer.
2. **H-002 cannot establish insertion learning.** Its AP-I is exactly unchanged at 0.186715 because the gap scores were inherited from Gold-Only. Against EXP43, this is a 0.098341 AP-I shortfall. Any claim for a new insertion head requires its own scores over all 20,354 exact gaps and explicit leading/trailing slot coverage.
3. **Target error F1 remains threshold sensitive.** H-002's AP gains are small, and its unrestricted metrics differ from CAL-selected metrics. Its reported CAL 10% point has very low recalls (S .0096, O .0383, I .0400), while unrestricted recall rises with a large warning-rate cost. Compare ranking capability first, then compare the same named CAL policy separately.
4. **Specialist recipes are complementary evidence, not a deployable neural checkpoint.** EXP24 leads omission AP; EXP36 leads substitution AP and macro AP; EXP43 leads exact-gap insertion AP; EXP29 leads the cited CAL-selected Macro-3 F1. This motivates head-aligned ablations, but does not imply their score advantages can be transferred by training automatically.
5. **Population uncertainty is limited.** SEARCH comprises 599 utterances and four speaker clusters. Per-speaker metrics are useful diagnostics but four clusters do not support precise population claims. Report seed variation and the observed speaker variation descriptively.

For every new model, first report one matched row with common event IDs, score direction, exact counts, checkpoint and prediction hashes, parameter count, training duration, peak CUDA memory and inference input type. Then inspect errors: whether omission positives rank low, whether S is routed to CORRECT or OMISSION, and whether insertion positives at exact gaps rank low. Do not use gold realized-phone labels or gold boundaries as features in these diagnoses.

## New-model evaluation handoff

As soon as Agent A/B publishes a prediction ledger, compare against the nearest matched Gold-Only and H-002 seed on exactly the same population. Report:

- Per-class AP and delta vs matched baseline, plus Macro-3 Error AP.
- CAL-selected threshold policy, class P/R/F1, Macro-3 Error F1, Macro-4 F1 and SEARCH FWR as a secondary trade-off.
- Counts TP/FP/FN/TN for S/O and exact-gap insertion; preserve warning-type conflicts and event denominators.
- N/N+1 coverage and any missing/duplicate event IDs.
- Per-speaker AP and omission subtype/phone-family errors only where the canonical metadata is available; label four-speaker slices descriptive.
- Comparisons against EXP24/29/36/43 and Phase 4 fusions explicitly labeled score-composition comparisons, not unified-neural checkpoint comparisons.

The evaluator's strict topology, ID and score validation is itself a precondition to publishing a row. If model artifacts use a different input schema, adapt only the serialization layer; do not recompute metrics outside EVAL-V3.1.

## U2/U3 EVAL-V3.1 results

The official CPU replay adapter is `experiments/rq1_unified_01/agent_c/evaluate_twohead_v3.py`. It reads Agent A's `p_target`, `p_gap`, `target_meta`, and `gap_meta` NPZ interface, exports those values as both `prob_*` and `score_*` under their proper meanings, and invokes `UnifiedEvaluatorV3`. It also scores CAL-selected threshold policies at 10% and 20% warning budgets and records per-speaker metrics. Full outputs and input/evaluator hashes are in the three `*_EVAL-V3.1-AGENT-C.json` receipts next to the script.

| Arm | Split | AP-S | AP-O | AP-I exact-gap | Macro-3 error AP | CAL 10% Macro-3 F1 | CAL 20% Macro-3 F1 | Unrestricted Macro-3 F1 | Unrestricted FWR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| U2 flat two-head, seed 101 | DEV_SEARCH | 0.371360 | 0.185318 | 0.117992 | 0.224890 | 0.059048 | 0.125421 | 0.280346 | 4.6544 |
| U2 flat two-head, seed 202 | DEV_SEARCH | 0.362558 | 0.170159 | 0.137553 | 0.223423 | 0.092882 | 0.119090 | 0.272415 | — |
| U2 flat two-head, seed 303 | DEV_SEARCH | 0.371064 | 0.173591 | 0.161451 | 0.235369 | 0.103852 | 0.097422 | 0.280706 | 4.6694 |
| U2 flat, gap loss weight 3 (GAPW3), seed 101 | DEV_SEARCH | 0.371360 | 0.185318 | 0.127349 | 0.228009 | 0.054113 | 0.118343 | 0.295560 | 4.4457 |
| GAPW3, seed 202 | DEV_SEARCH | 0.362558 | 0.170159 | 0.167741 | 0.233486 | 0.118273 | 0.144482 | 0.295156 | — |
| GAPW3, seed 303 | DEV_SEARCH | 0.371064 | 0.173591 | 0.121966 | 0.222207 | 0.079876 | 0.097422 | 0.292252 | 4.4274 |
| U3 hierarchical two-head, seed 101 | DEV_SEARCH | 0.362413 | 0.176117 | 0.123822 | 0.220784 | 0.061169 | 0.072345 | 0.265451 | 4.2955 |

Across the three seeds, U2 Macro-3 AP mean±sample SD is 0.227894±0.006515; GAPW3 is 0.227901±0.005640. Their mean insertion AP is 0.138999 vs 0.139019, essentially identical. The GAPW3 insertion AP delta reverses at seed 303 (−0.039485 vs U2); therefore positive-loss weighting did not provide a stable ranking improvement. Mean CAL20 Macro-3 F1 is 0.113978 vs 0.120082, but seed-wise changes vary (−0.007079, +0.025392, 0). Treat this as noisy operating-point evidence, not a validated gain. U3 seed 101 gains 0.005830 exact-gap AP over U2 but loses −0.008947 substitution AP, −0.009202 omission AP, and −0.004106 Macro-3 AP. Neither hierarchy nor loss reweighting closes the gap to historical systems. All runs have 19,755 targets, 20,354 gaps, and 599 utterances; target/gap coverage is 100%, with zero missing predictions and zero target-simplex violations.

Class F1 keeps operating policy separate from AP:

| Arm and policy | S F1 | O F1 | I F1 | Macro-3 error F1 | Joint FWR |
|---|---:|---:|---:|---:|---:|
| U2, CAL 10% | 0.0023 | 0.1143 | 0.0606 | 0.0590 | 0.0518 |
| GAPW3, CAL 10% | 0.0023 | 0.1143 | 0.0458 | 0.0541 | 0.0518 |
| U3, CAL 10% | 0.0000 | 0.0870 | 0.0966 | 0.0612 | 0.0401 |
| U2, CAL 20% | 0.0243 | 0.2092 | 0.1429 | 0.1254 | 0.1786 |
| GAPW3, CAL 20% | 0.0243 | 0.2092 | 0.1216 | 0.1183 | 0.1720 |
| U3, CAL 20% | 0.0000 | 0.1205 | 0.0966 | 0.0723 | 0.0634 |
| U2, unrestricted | 0.4451 | 0.2163 | 0.1797 | 0.2803 | 4.6544 |
| GAPW3, unrestricted | 0.4451 | 0.2163 | 0.2254 | 0.2956 | 4.4457 |
| U3, unrestricted | 0.4363 | 0.1734 | 0.1867 | 0.2655 | 4.2955 |

At low warning budgets, all variants miss most S/O positives. GAPW3 changes CAL20 Macro-3 F1 inconsistently across seeds and has no mean AP-I gain. Its unrestricted insertion F1 is higher than U2 for all three seeds: 0.1797/0.1554/0.1809 vs 0.2254/0.2236/0.2155 (seeds 101/202/303); unrestricted FWR is still above 4.4 warnings/utterance for both variants. U3's 20% CAL policy issues 38 false warnings over 599 utterances; U2 seed 101 issues 107. FWR is the observed false-warning count divided by utterances; unrestricted points are diagnostic and generate many false warnings.

### Class-wise PR and error concentration

AP is the area under each class's stepwise precision-recall ranking, so the S/O/I AP columns above are the class-wise PR summary. The official receipts also include class confusion counts and operating-point precision/recall. For U2, unrestricted O AP is uneven across speakers: RRBI 0.3117 (65 omissions), YDCK 0.2236 (58), YBAA 0.1108 (31), and NCC 0.1096 (55). U3 remains heterogeneous: RRBI 0.3126 (65), YDCK 0.1987 (58), NCC 0.1149 (55), and YBAA 0.0939 (31). These are four exposed speaker slices and descriptive only.

Under U2's unrestricted policy, omission F1 by speaker is NCC 0.1629 (25 TP, 227 FP, 30 FN), RRBI 0.2699 (39, 185, 26), YBAA 0.1547 (14, 136, 17), and YDCK 0.2612 (35, 175, 23). U3 has NCC 0.1311 (28, 344, 27), RRBI 0.2347 (44, 266, 21), YBAA 0.1107 (14, 208, 17), and YDCK 0.2033 (37, 269, 21). U3 finds a few additional omissions in most speakers but at substantially more false positives. Across the aggregate U2 target confusion matrix, 96 of 209 gold omissions are missed at unrestricted threshold: 30 at NCC, 26 RRBI, 17 YBAA, and 23 YDCK.

Phone-conditioned omission ranking also varies. With at least two positive omissions, U2's weakest observed phone APs are DH 0.0083 (2 positives), ER 0.0106 (2), HH 0.0189 (3), TH 0.0227 (3), and IH 0.0243 (8). Stronger slices include W 0.5696 (12) and R 0.4477 (52). Very small supports make the rare-phone values unstable. The largest raw count of missed omissions is R (11), then D (9), AH (8), IH (7), and Z (5) at the unrestricted policy; those counts reflect support as well as ranking.

### Comparison with historical systems

Historical values remain in the inspected and preserved `experiments/rq1_unified_01/agent_c/HISTORICAL_BASELINES.csv`. EXP24/29/36/43 values match the canonical DEV_SEARCH composition comparison in `reports/PHASE4_MODEL_COMPARISON.csv`. Their metrics are score-composition replays (EXP24 is a standalone neural checkpoint replay); they are not matched reruns of U2/U3. The relevant differences are:

| Comparator | AP-S | AP-O | AP-I | Macro-3 AP | Macro-3 F1 (recorded CAL_MAX_ERROR_MACRO_F1) | Macro-4 F1 |
|---|---:|---:|---:|---:|---:|---:|
| Gold-Only seed 101 | 0.364602 | 0.159433 | 0.186715 | 0.236917 | 0.300141 | 0.445779 |
| H-002 normalized attention + Gold-Only gaps | 0.377810 | 0.167294 | 0.186715 | 0.243940 | 0.3095* | 0.4592* |
| EXP24 | 0.357737 | 0.225092 | 0.247628 | 0.276819 | 0.305855 | 0.450294 |
| EXP29 | 0.378062 | 0.196539 | 0.272068 | 0.282223 | 0.342914 | 0.482871 |
| EXP36 | 0.387088 | 0.204687 | 0.275924 | 0.289233 | 0.331090 | 0.470515 |
| EXP43 | 0.363453 | 0.205771 | 0.285056 | 0.284760 | 0.332231 | 0.472479 |
| Phase 4 fixed headwise blend, 5-seed mean | — | — | — | 0.2955 | 0.3296 | — |
| Phase 4 omission-specialist fusion, 5-seed mean | — | — | — | 0.3018 | 0.3178 | — |
| U2 flat, seed 101 | 0.371360 | 0.185318 | 0.117992 | 0.224890 | 0.280346† | 0.436738† |
| U2 flat GAPW3, seed 101 | 0.371360 | 0.185318 | 0.127349 | 0.228009 | 0.295560† | 0.448149† |
| U3 hierarchical, seed 101 | 0.362413 | 0.176117 | 0.123822 | 0.220784 | 0.265451† | 0.426971† |

`*` H-002 F1 values are the cited unrestricted diagnostic in its historical report and do not share U2/U3's policy. `†` The new-system values are unrestricted diagnostic F1, included to show the high-warning trade-off and not as the selected operating policy. The single-seed U2/GAPW3 AP comparison against Gold-Only shows that target AP improved modestly under U2 while insertion AP fell by 0.0687; GAPW3 recovers 0.0094 versus U2 but remains 0.0594 below the Gold-Only insertion ranking. U3 insertion remains 0.0629 below Gold-Only. The four Phase 2 specialist/composition systems have higher Macro-3 AP, with EXP24 strongest on omission AP, EXP36 on substitution and Macro-3 AP, and EXP43 on exact-gap AP. Five-seed blend aggregates are reported separately because their per-class APs are not present in the source aggregate.

The central architectural result is therefore limited: a shared frozen acoustic representation with new target and exact-gap heads gives complete coverage and learnable rankings, but the new exact-gap head is still materially weaker than Gold-Only and all cited score compositions. The omission specialist's stronger AP and F1 cannot be attributed to a standalone learned omission architecture from this comparison; it is evidence about score composition. The CAL policy controls how the available ranking is exposed to users, and should not be conflated with either architecture or composition.

**Next research direction:** investigate a gap-boundary representation built from frozen utterance-wide acoustic states around the boundary between adjacent predicted phone centers, then compare it with the current left/right phone-crop edge pooling while keeping the U2 target head and training schedule fixed. Do not use gold boundaries, realized phones, or gold gap labels as input features. Choose the policy on DEV_CAL, report AP-I and CAL-policy insertion F1/FWR on DEV_SEARCH descriptively, and replicate before claiming a gain. This is a recommendation for future work; the representation experiment has not been run.
