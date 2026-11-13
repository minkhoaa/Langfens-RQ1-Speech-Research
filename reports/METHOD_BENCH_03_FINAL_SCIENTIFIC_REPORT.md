# LANGFENS RQ1-METHOD-BENCH-03 — Báo cáo khoa học cuối

**Ngày:** 2026-11-08 · **Evaluator:** EVAL-V3.1 · **Trạng thái:** campaign đã chạy; có bằng chứng lặp lại rằng CTC Wav2Vec2 và marginalization edit path cải thiện ranking so với matched CTC hard-path control, nhưng chưa cạnh tranh với neural/specialist frontier. DEV_SEARCH đã exposed trong chiến dịch trước, chỉ dùng thăm dò; không truy cập holdout được bảo vệ.

## Kết luận

Đã huấn luyện recognizer phoneme CTC thật với hai encoder, realized-phone TRAIN targets, blank tường minh, CTC loss, beam decoding và posterior export. Frozen `facebook/wav2vec2-base-960h` hidden layer 6 cùng trainable linear CTC head đạt beam-10 realized PER trung bình **0.3303** trên TRAIN-speaker holdout HJK/HKK, **0.3458** DEV_CAL và **0.3216** DEV_SEARCH qua seed 101/202/303. Kết quả này tốt hơn nhiều so với log-mel BiGRU beam PER 0.7207/0.6664/0.6802 tương ứng, nhưng vẫn kém canonical-sequence control PER 0.2320/0.1013 trên CAL/SEARCH. PER không đồng nghĩa MDD.

Với cùng Wav2Vec2 posteriors và cùng event population, E3 marginal hóa tất cả edit path tối thiểu trong từng candidate. Trên ba seed DEV_SEARCH, E3 cải thiện AP so với greedy E1 trong cả S/O/I ở từng seed; mean AP-S/O/I là **0.2378/0.0409/0.0340**, Macro-3 AP **0.1042**. Đây là bằng chứng lặp lại về giá trị tương đối của sequence evidence và path marginalization. Tuy nhiên Macro-3 AP còn thấp hơn Gold-Only 0.2369; AP-O thấp hơn 0.1594; AP-I chỉ nhỉnh hơn H05 frame-argmax 0.02825 và rất thấp hơn Gold-Only 0.1867/EXP43 0.2851. Candidate beam giữ rất ít tổng CTC mass, nên đây là phép xấp xỉ conditional chứ không phải posterior đầy đủ.

Speaker-grouped TRAIN OOF cho thấy split sensitivity lớn: plain-head Macro-3 AP là 0.2873 so với 0.3532 khi utterance-random split. GRL speaker-adversarial analogue không có cải thiện ổn định. Những kết quả này cho thấy validation theo speaker quan trọng, nhưng không chứng minh speaker identity là nguyên nhân duy nhất của U2 insertion gap. Không tích hợp CTC E3/GRL hiện tại vào score composition.

## Bảng so sánh

| Hệ thống / split | AP-S | AP-O | AP-I exact-gap | Macro-3 AP | Loại |
|---|---:|---:|---:|---:|---|
| Gold-Only / DEV_SEARCH | .364602 | .159433 | .186715 | .236917 | Neural reference |
| H-002 / DEV_SEARCH | .377810 | .167294 | .186715 | .243940 | Neural target + Gold-Only gaps |
| U2-S101 / DEV_SEARCH | .371360 | .185318 | .117992 | .224890 | Unified neural checkpoint |
| EXP24 / DEV_SEARCH | .357737 | .225092 | .247628 | .276819 | Learned checkpoint |
| EXP36 / DEV_SEARCH | .387088 | .204687 | .275924 | .289233 | Score composition |
| EXP43 / DEV_SEARCH | .363453 | .205771 | .285056 | .284760 | Score composition |
| Phase 4 omission specialist / DEV_SEARCH | — | — | — | ≈.3018 | Score composition; reported aggregate |
| Log-mel CTC E3 beam-10 / DEV_SEARCH | .135783 | .017343 | .013578 | .055568 | Trained CTC + conditional path marginal |
| W2V2 CTC E3 S101 / DEV_SEARCH | .235737 | .041842 | .029214 | .102264 | Trained CTC + conditional path marginal |
| W2V2 CTC E3 S202 / DEV_SEARCH | .241050 | .042150 | .042219 | .108473 | Trained CTC + conditional path marginal |
| W2V2 CTC E3 S303 / DEV_SEARCH | .236475 | .038843 | .030607 | .101975 | Trained CTC + conditional path marginal |
| W2V2 CTC E3 mean / DEV_SEARCH | .237754 | .040945 | .034013 | .104238 | Three-seed mean; separate recognizers |
| Plain neural head / grouped TRAIN OOF | .423209 | .276706 | .162125 | .287347 | Four held-speaker folds |
| GRL neural head / grouped TRAIN OOF | .423371 | .276634 | .165887 | .288631 | Four held-speaker folds |
| Plain neural head / random TRAIN OOF | .478970 | .359747 | .220765 | .353161 | Utterance-random diagnostic |

Historical expert compositions remain a separate category from trained models. TRAIN OOF is a different population from DEV_SEARCH and must not be read as a directly matched leaderboard comparison.

## Trả lời 17 câu hỏi khoa học

1. **Có recognizer phoneme CTC thật không?** Có. Log-mel BiGRU-S101 được train bằng `torch.nn.CTCLoss`; sau đó ba recognizer frozen-Wav2Vec2 + linear CTC head với seed 101/202/303 được train và đánh giá. Đây là sequence recognition, không phải CharSiU frame classifier.
2. **Targets và tokenizer?** 42 lớp phoneme LANGFENS cộng blank ID 42. TRAIN targets là chuỗi realized-phone lấy từ `realized_phone` ở target không bị omission và các insertion được gán gap, sắp theo annotation time. Chuỗi canonical được đánh giá riêng. Gold labels/timing chỉ dùng làm supervision/evaluation, không làm inference input. Checkpoint eSpeak có 392 IPA/eSpeak tokens, nhưng chưa có converter IPA→LANGFENS đã kiểm chứng.
3. **PER?** W2V2 CTC beam-10 mean realized PER là .3215 train-fit, .3303 held-out HJK/HKK, .3458 CAL, .3216 SEARCH. Greedy mean là .3961/.4220/.4177/.3988 tương ứng. Seed SD và từng seed nằm trong CTC report/CSV. Log-mel BiGRU beam-10 đạt .6240/.7207/.6664/.6802. Canonical sequence control PER là .1529 TRAIN, .2320 CAL, .1013 SEARCH; control này dùng annotation và không phải inference baseline.
4. **Generalize sang speaker mới?** W2V2 held-out TRAIN speakers có mean beam PER cao hơn nhẹ so với train-fit (.3303 so với .3215); biến thiên theo seed/speaker có trong Agent A receipts. Chỉ có hai TRAIN-speaker holdout và bốn SEARCH speaker, nên chưa đủ để kết luận generalization rộng.
5. **CTC có tăng AP-S?** Có trên matched CTC E1 control qua từng seed: E3 S AP .2357–.2411 so với E1 mean thấp hơn; W2V2 E3 mean .2378, nhưng vẫn dưới Gold-Only .3646 và H-002 .3778. Không tuyên bố vượt baseline tổng quát.
6. **Deletion evidence probabilistic có tăng AP-O?** Có mức tăng tương đối lặp lại so với E1 ở cả ba seed (E3 mean .0409), nhưng tuyệt đối yếu hơn nhiều Gold-Only .1594, EXP24 .2251. Cần ghi nhận only 209 omission events ở DEV_SEARCH.
7. **Posterior-aware mapping có tăng exact-gap AP-I?** Có ở cả ba seed so với matched E1; mean .0340, range .0292–.0422. Mức này chỉ cao hơn chút so với H05 .02825, và thấp xa Gold-Only .1867, EXP43 .2851. Không có căn cứ để gọi insertion localization đã giải quyết.
8. **Ambiguity thường xuyên thế nào?** Với W2V2 top-10, 567–576/599 SEARCH utterances có candidate chứa nhiều edit path tối ưu; ambiguous candidates nhận .813–.855 sequence weight. E3 biểu diễn bất định bằng cách marginal hóa đều các path unit-cost tối ưu, nhưng phép đều đó là quy ước, không phải acoustic posterior. Candidate top-10 + canonical có mean CTC mass khoảng 1.3e-5–3.8e-5, median 4.5e-13–2.4e-12; phần lớn xác suất nằm ngoài candidate set.
9. **Acoustic recognition có hơn canonical/prior-only control không?** Không theo PER: W2V2 beam PER .32–.35 kém canonical sequence control .10–.23 trên dev. CTC dùng acoustic evidence, nhưng MDD AP còn yếu. Control canonical dựa trên gold realization và chỉ là trần tham chiếu chẩn đoán, không phải inference competitor. Không có bằng chứng CTC vượt prior-only control trên matched MDD metric.
10. **Speaker dependence giải thích U2 gap?** Có split sensitivity đo được nhưng chưa phân rã nhân quả. Plain-head random OOF Macro-3 AP .3532 so với grouped .2873; insertion .2208 so với .1621. TRAIN speaker prevalence và U2 per-speaker AP không đồng nhất. Phone unigram JSD nhỏ (.000280 TRAIN/CAL, .000131 TRAIN/SEARCH), nhưng điều đó không loại trừ phone-conditioned, acoustic-channel, label hoặc alignment shift. U2 insertion train .976979 so DEV_CAL .138111/SEARCH .117992 cho thấy severe in-sample-to-dev gap.
11. **Speaker-invariant method nào được train?** Một supervised GRL speaker-adversarial small-head analogue, TRAIN-only speaker nuisance labels và bốn grouped OOF folds. Macro-3 AP aggregate chỉ tăng .0013; insertion aggregate +.0038 nhưng fold mean giảm và 3/4 folds giảm. Không có gain ổn định; Spin-style augmentation chưa train.
12. **External methods nào được tìm và thử?** Nguồn phát hiện gồm Cao et al. (Interspeech 2024, CTC alignment-free GOP), Parikh et al. (Interspeech 2025, substitution-aware GOP), Zhou et al. (Interspeech 2025, phoneme similarity/verbatim recognition), Phan et al. (Interspeech 2025, multilingual low-resource MDD), Lavechin & Hueber (EMNLP 2025, acoustic invariance), Dao et al. (2026, GRL/variational bottleneck cho spoofing), và Spin speaker-invariant clustering. Đã train GRL analogue nội bộ; các paper method khác chưa được tái lập trực tiếp. HF eSpeak CTC weights được tải và probe, nhưng phone map chưa hợp lệ nên không benchmark LANGFENS.
13. **Có method mới nào hơn Gold-Only/H-002 matched metric?** Không có candidate CTC hoàn chỉnh trên DEV_SEARCH. TRAIN OOF grouped AP-O/S cao hơn một số historical DEV_SEARCH values nhưng khác split/population và không phải comparison matched; grouped AP-I .1621 vẫn dưới .1867. Không tuyên bố thắng.
14. **Có gần EXP24/36/43 hoặc Phase 4 fusion?** Không. W2V2 E3 Macro-3 .1042 so với .2768/.2892/.2848 và ≈.3018.
15. **Cải thiện nào lặp qua seed/speaker?** W2V2 E3 > matched E1 ở S/O/I trên cả 3 seeds. Macro-3 AP range hẹp .1020–.1085. Per-speaker event AP có biến thiên, nhất là do omission/insertion supports nhỏ; không đủ để khẳng định đồng đều theo speaker. GRL không repeat qua seed.
16. **Giả thuyết nào thất bại?** Compact log-mel CTC không nhận dạng đủ tốt; W2V2 PER tốt hơn nhưng vẫn không đạt canonical-control PER và sequence event scores yếu; uniform path marginalization tăng AP matched nhưng không đưa O/I đến frontier; GRL chưa cho gain speaker-stable. Các thất bại này không bác bỏ full CTC, lattice hay mọi invariance method.
17. **Hướng tiếp theo mạnh nhất?** Nâng chất lượng sequence model trong speaker-grouped TRAIN validation, sau đó tăng lattice coverage/calibration và đánh giá target/gap event marginal trên cùng held-speaker folds. Trước đó không blend với U2. Ưu tiên benchmark acoustic evidence trước/sau recognizer và kiểm soát canonical/prior-only trên matched event identities.

## Cơ sở và phân loại bằng chứng

- **Đã xác minh implementation/execution:** hai họ genuine CTC; ba W2V2 seeds; CTC blank/lengths/posteriors/clocks; checkpoint và environment receipts; beam decoding; exact CTC forward scoring; edit-path DP; exhaustive synthetic fixtures; EVAL-V3.1 event coverage; speaker-grouped OOF và GRL.
- **Đo thực nghiệm:** PER/AP/F1/support và candidate mass tại các báo cáo agent, receipts và CSVs liên kết bên dưới. DEV_SEARCH là exposed, exploratory.
- **Được báo cáo trong literature:** novelty/method claims ở external sources; các kết quả trên dataset LANGFENS không được paper thay thế.
- **Diễn giải thăm dò:** speaker-grouped/random chênh lệch và U2 train-dev gap ủng hộ kiểm soát speaker split; chưa xác định nguyên nhân.
- **Chưa hỗ trợ:** CTC sequence evidence vượt model frontier; speaker identity là nguyên nhân duy nhất; GRL cải thiện tổng quát; beam weights là full posterior; IPA tokenizer tương thích.
- **Chưa hoàn tất:** E4 alignment-free likelihood ratio; full lattice posterior; matched grouped-OOF sequence event streams; validated eSpeak conversion; independent blind test; inference P@K report.

## Tài liệu và reproduction

[CTC recognition](METHOD_BENCH_03_CTC_RECOGNITION.md) · [probabilistic edit localization](METHOD_BENCH_03_PROBABILISTIC_EDIT_LOCALIZATION.md) · [speaker generalization](METHOD_BENCH_03_SPEAKER_GENERALIZATION.md) · [external discovery](METHOD_BENCH_03_EXTERNAL_DISCOVERY.md) · [leaderboard](METHOD_BENCH_03_EXPERIMENT_LEADERBOARD.csv) · [ablations](METHOD_BENCH_03_ABLATION_RESULTS.csv) · [reproduction](METHOD_BENCH_03_REPRODUCTION.md) · [next research](METHOD_BENCH_03_NEXT_RESEARCH.md).
