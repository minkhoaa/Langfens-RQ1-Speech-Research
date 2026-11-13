# LANGFENS RQ1-METHOD-BENCH-05 — Báo cáo khoa học cuối

**Ngày:** 08-11-2026 · **Đánh giá:** EVAL-V3.1 · **Dữ liệu chính:** TRAIN, speaker-grouped OOF · **Thiết bị:** CPU do CUDA driver không khả dụng.

## Kết luận điều hành

E4 cải thiện xếp hạng lỗi substitution (S) và omission (O) so với E3 trên cả bốn fold speaker-held-out, lặp lại ở seed 101 và 202. Trên OOF gộp, E4 canonical `mean` tăng AP-S/AP-O từ `.23888/.09205` lên `.31972/.25936` ở seed 101, và từ `.24859/.10568` lên `.32829/.27157` ở seed 202. Tăng AP-I không ổn định: E4 `.03159` so với E3 `.03462` ở seed 101; `.03743` so với `.03717` ở seed 202. Vì vậy E4 là cải thiện chuyển giao cho S/O trong phạm vi 12 speaker TRAIN, không phải scorer cân bằng S/O/I.

Collision trong không gian chuỗi CTC là hiện tượng thật nhưng bằng chứng không ủng hộ nó là nguyên nhân chính của insertions yếu. Với top-8 candidate, 6,795/10,139 gaps (67.0%) có *ít nhất một* giả thuyết candidate trùng collapsed sequence với gap khác. Tuy nhiên chỉ 2/46 gold-positive gaps có chính giả thuyết mang gold inserted phone bị trùng đa-gap. Top-8 chỉ bao phủ 23/46 positive gaps; thêm toàn bộ 42 ký hiệu tokenizer tăng coverage lên 46/46 nhưng AP-I giảm `.036681 → .034919` trong phép thử max-vs-max. Tần suất collision rộng không đồng nghĩa số event gold không định danh.

Temporal score T2, dùng Viterbi reference alignment và bằng chứng posterior cục bộ quanh predicted gap, đạt AP-I `.004263`, thấp hơn E3 `.009966` và E4 canonical mean `.038718` trên cùng 10,139 gaps. Thử forward-backward T3 không tạo được score hợp lệ. Do đó không có bằng chứng rằng can thiệp temporal đã thử cải thiện exact-gap localization; kết quả âm chỉ bác bỏ cơ chế local-max đã thử.

Một MLP trực tiếp C/S/O độc lập CTC đạt AP-S/AP-O `.38302/.27410` trên cùng bốn nhóm speaker và 59,562 target events; cao hơn E4 mean `.31972/.25936`, nhưng thấp hơn neural direct-event baseline MB03 `.423209/.276706`. Acoustic-only và reference-only controls lần lượt đạt `.29915/.18003` và `.30966/.08945`: âm thanh đóng góp rõ hơn cho O, trong khi phone prior đóng góp đáng kể cho S. MLP này không dự đoán I nên không có Macro-3 so sánh.

## Thực thi và kiểm soát dữ liệu

- Đã phục hồi MB03 Wav2Vec2-CTC posteriors, tokenizer, blank ID 42, 20 ms clock và scripts E1/E3/E4; không retrain các recognizer MB03 hay ghi đè artifact lịch sử.
- Bốn fold chính thức giữ lần lượt ABA/ASI/BWC; EBVS/HJK/HKK; HQTV/LXC/NJS; PNV/SKA/SVBI. Mỗi recognizer mới fit chín speaker và chỉ chấm ba speaker bị giữ lại. Seed 101 và 202 đều dùng realized-phone CTC targets, Linear(768,43), AdamW, 8 epochs; không dùng gold timing hay realized phones làm feature.
- E1, E3 và E4 trong mỗi fold dùng cùng posterior. Thống kê competitor E4 chỉ fit trên chín speaker huấn luyện của fold. Cả target N và gap N+1 được giữ đủ.
- Agent A dùng posterior cố định trên TRAIN HJK/HKK holdout và DEV_CAL; Agent C dùng bốn TRAIN folds. DEV_SEARCH không dùng để chọn kiến trúc/candidate/policy, protected holdout không truy cập.
- E4 `mean` là cấu hình canonical MB04 và là kết quả chính; `max` được giữ riêng làm sensitivity analysis. Raw-score AP được báo riêng với F1/policy. E3 raw AP hơi khác EVAL summary vì EVAL chuẩn hóa S/O marginals thành xác suất ba lớp; bảng OOF chính giữ raw score semantics đồng nhất giữa E1/E3/E4.

## Kết quả E4: grouped transfer

Pooled OOF gồm 59,562 target events, 61,362 exact gaps; supports S/O/I là 7,381/1,825/560, prevalence/prior-only AP là `.123921/.030640/.009126`.

| Seed | Method | AP-S | AP-O | AP-I exact-gap | Macro-3 AP |
|---:|---|---:|---:|---:|---:|
| 101 | E1 | .16686 | .05084 | .01271 | .07680 |
| 101 | E3 | .23888 | .09205 | .03462 | .12185 |
| 101 | **E4 mean** | **.31972** | **.25936** | .03159 | **.20356** |
| 202 | E1 | .17347 | .05217 | .01576 | .08047 |
| 202 | E3 | .24859 | .10568 | .03717 | .13048 |
| 202 | **E4 mean** | **.32829** | **.27157** | .03743 | **.21243** |

E4 S/O gains held in all 8 fold-seed comparisons. Exact-gap AP-I was higher for E4 in only 3/8 fold-seed comparisons. Per-speaker seed101 AP-S increased in all 12 speakers; AP-O increased in all 12 (HJK has only 4 omission positives, so its estimate is unstable). AP-I increased in 6 and decreased in 6 speakers. Speaker breakdown is in the Agent B report and CSV artifact.

Beam-10 realized-phone PER across folds was `.40579–.44347` for seed101 and `.38016–.41539` for seed202. Across eight fold-seed points, beam PER had Pearson r `.31/.19/.29` with E4 raw AP-S/O/I; this descriptive sample does not establish a monotonic PER–MDD relationship. Better recognition is not a sufficient explanation for event-ranking differences.

### Threshold and warning policy

Raw AP and decision policy diverge. Averaging the four EVAL-V3.1 fold summaries, E4 default Macro-3 F1 is `.1561` (seed101) and `.1654` (seed202), Macro-4 mixed-domain F1 `.3139/.3246`, and joint false-warning rates `13.61/12.35` per utterance. Mean binary MD AP is `.4216/.4288`. E3's corresponding four-fold mean Macro-3 F1 is `.1998/.2081` and joint FWR `11.19/10.53`. E4's S/O ranking gain does not improve default point F1; no low-warning operating claim is made and FWR≤.20 is not a success gate. Pooled classwise P/R/F1 and binary metrics are tabulated below; fold metrics, warnings, and coverage are in the grouped operating-metrics CSV/JSON and EVAL receipts.

Pooled classwise precision/recall/F1 below use EVAL-V3.1 default argmax decisions on concatenated OOF event rows. These values differ from fold-averaged Macro-F1 above; raw AP remains the ranking result.

| Seed / method | S P/R/F1 | O P/R/F1 | I P/R/F1 | Correct F1 | Binary MD AP/F1 |
|---|---|---|---|---:|---:|
| 101 E1 | .238/.376/.292 | .064/.607/.116 | .048/.091/.063 | .712 | .236/.389 |
| 101 E3 | .281/.464/.350 | .091/.540/.155 | .065/.191/.097 | .790 | .314/.434 |
| 101 E4 mean | .308/.241/.270 | .084/.794/.152 | .028/.241/.050 | .787 | .428/.427 |
| 202 E1 | .250/.392/.306 | .067/.591/.120 | .061/.127/.083 | .731 | .245/.400 |
| 202 E3 | .295/.471/.363 | .099/.544/.167 | .067/.200/.100 | .804 | .323/.444 |
| 202 E4 mean | .341/.233/.277 | .089/.810/.160 | .036/.248/.063 | .802 | .436/.437 |

Machine-readable pooled/fold operating metrics, supports, and class AP are saved in `experiments/rq1_method_bench_05/agent_b/grouped_oof_operating_metrics.csv`.

## Identifiability, candidate coverage, and timing

On the fixed HJK/HKK population, top-eight broad collision flags mark 67.0% of all gaps and 31/46 positive gaps. Yet only 2/46 gold-positive gaps have the actual gold inserted-phone sequence mapping to multiple exact gaps. The E4 collision-participating subset has AP-I `.02148` (31/6,795) versus `.09181` (15/3,344) for identifiable gaps, but prevalence is nearly equal (`.456%` vs `.449%`) and support is low. This descriptive split cannot establish that collision caused the AP difference.

Candidate coverage and ambiguity are distinct. Top-eight scores cover 23/46 positive gaps; the full nonblank tokenizer covers 46/46. Under max aggregation, AP-I falls by `.001762` on this small population as candidates broaden. The full tokenizer also makes every gap collision-participating under the broad “any candidate collides” rule. These results argue against treating the broad flag as a causal measure or assuming more alternatives improve ranking.

T0/T1/T2 matched exact-gap results were `.038718/.009966/.004263` (E4 mean/E3/T2), with 46 positives among 10,139 gaps. T2's ±2-frame local max over the top-eight candidate-phone posterior odds did not help. The score used predicted reference-conditioned CTC Viterbi alignment, not gold annotation timing. Uncertainty-marginalized T3 failed before valid evaluation; no T3 result is claimed.

## Independent non-CTC evidence

The direct event model `RQ1-MB05-C-DIRECT-MLP-OOF-S101` scores S/O at targets only. Pooled grouped OOF AP is `.38302/.27410`, above TRAIN prevalence and above matched-population E4 mean by `.06330/.01474`. This is a matched population, not a one-factor architecture comparison: representations, labels/objective, and scoring mechanisms differ. It is below MB03's earlier grouped neural target head `.423209/.276706`. Acoustic-only is `.29915/.18003`; reference-only is `.30966/.08945`. These controls support an acoustic contribution—especially for O—but show that phone-conditioned priors explain part of S. The model has no gap output, so insertion AP and Macro-3 are unsupported.

External discovery covered direct SSL MDD, weakly supervised word error detection, learned acoustic/phoneme alignment, MixGoP multi-cluster phone modeling, and phonological-attribute MDD. The implemented MLP is an adaptation of the direct supervised route, not a reproduction of those papers. Agent C's report distinguishes published claims from LANGFENS measurements and provides source links; for the ICASSP 2022 alignment paper, only the publisher abstract was accessible.

## So sánh với lịch sử và giới hạn quần thể

Historical DEV_SEARCH (already exposed) values include Gold-Only Macro-3 `.236917`, H-002 `.243940`, EXP24 `.276819`, EXP43 `.284760`, EXP36 `.289233`, and Phase 4 omission-specialist composition about `.3018`. They are not matched speaker OOF populations and are not evidence that the new TRAIN OOF model beats those systems. E4's principal value here is the paired E1/E3 comparison under valid grouped folds. Historical Phase 4 value is a score composition, not a new unified checkpoint.

The four folds exhaust these 12 TRAIN speakers; repeating seeds does not create independent cohorts. Phone prevalence and phonetic mix vary by fold. The frozen Wav2Vec2 encoder is not retrained, and no external-speaker test was run. DEV_CAL remains available for future declared calibration; DEV_SEARCH should not guide further choices.

## Główne odpowiedzi badawcze

1. **Collision phổ biến tới đâu?** 67% gaps có ít nhất một top-eight collision; toàn bộ gaps với all-42 inventory. Đây là định nghĩa rộng.
2. **Bao nhiêu gold events bị ảnh hưởng?** 31/46 positive gaps có broad collision flag, nhưng chỉ 2/46 gold insertion hypotheses bị trùng đa-gap; không có bằng chứng phần lớn positive events bị mất định danh.
3. **Temporal evidence có giúp không?** T2 không; T3 không có kết quả hợp lệ.
4. **Ambiguity-aware localization vượt E3/E4?** Chưa. T2 thấp hơn E3 và E4; T3 chưa đo.
5. **E4 S/O có transfer?** Có trên 4 folds, 2 seeds; tăng cả 8 fold-seed comparisons và pooled OOF.
6. **Insertion có transfer?** Không ổn định; AP-I thấp và chỉ 3/8 fold-seed comparisons tăng.
7. **PER liên quan AP?** Có tương quan mô tả yếu, không đơn điệu trên 8 điểm.
8. **Acoustic hay phone prior?** Direct MLP controls cho thấy cả hai góp phần; acoustic tăng O mạnh hơn S so với reference-only.
9. **Non-CTC có thắng matched CTC?** Có với S/O so với E4 mean trên cùng targets; không thắng historical grouped neural direct head; không có I.
10. **Phát hiện ổn định?** E4 S/O pooled gain gần như giống nhau ở hai seed; E4 I, default F1 và temporal localization không có lợi ích ổn định.

### Tái lập nhanh

Xem [reproduction report](METHOD_BENCH_05_REPRODUCTION.md), [hypothesis verdicts](METHOD_BENCH_05_HYPOTHESIS_VERDICTS.md), [leaderboard](METHOD_BENCH_05_EXPERIMENT_LEADERBOARD.csv), [ablations](METHOD_BENCH_05_ABLATION_RESULTS.csv), và [next research](METHOD_BENCH_05_NEXT_RESEARCH.md). Các fold-specific checkpoints, posterior hashes, score ledgers, PR curves, OOF arrays, receipts và source nằm trong `experiments/rq1_method_bench_05/`.
