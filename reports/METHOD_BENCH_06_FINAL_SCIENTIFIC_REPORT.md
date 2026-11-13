# LANGFENS METHOD-BENCH-06 — Báo cáo khoa học cuối (tạm kết chiến dịch)

**Ngày kiểm kê:** 08-11-2026 · **Quần thể chính:** TRAIN, 4-fold speaker-grouped OOF · **EVAL:** V3.1. **Phạm vi:** ba lane đã khởi chạy song song theo yêu cầu, model `gpt-6-luna`. Kết quả dưới đây phân biệt phép đo đã chạy với giả thuyết chưa được thử; không coi báo cáo chẩn đoán là bằng chứng của mô hình tích hợp đã huấn luyện.

## Tóm tắt

- **E4 bổ sung xếp hạng khác neural, nhưng chưa chứng minh lợi ích độc lập hữu dụng.** Trên matched pooled OOF 59,562 target events, neural MB03 vượt E4 mean cho cả S (`.42321` vs `.31972`) và O (`.27671` vs `.25936`). E4 có các true positive riêng và có vẻ giúp O trong nhóm neural confidence thấp, nhưng đây là phân tích hậu nghiệm; phone identity là control mạnh với S. Rank-mean pooled cho AP `.44547/.35708` chỉ là exploratory vì chuẩn hoá/ghép dùng toàn bộ OOF, không nested, không phải phép thử confirmatory. **Nested meta-learning chưa chạy.**
- **Lane B đã train và đánh giá một probe exact-gap temporal mới**, `RQ1-MB06-B-GROUPED-TEMPORAL-LOGREG-S101`: pooled AP-I `.02220`, thấp hơn static six-feature control `.03557` và neural baseline `.162125`. Thắng static chỉ ở 1/4 folds. Đây là negative result cho local pooling đã thử, không phải bác bỏ mọi temporal localization.
- **MB03 grouped baseline** AP-I `.162125` (560/61,362); per-fold `.18879/.10271/.10945/.25154`. Probe mới dùng frame CTC logits trong replay cache và predicted alignment; không dùng gold insertion timing ở inference. Historical T2 `.004263` trên 10,139 gaps/46 positives là một test khác.
- **Không có calibration/F1/warning-policy mới** và không có repeated seed. Probe đo timing evidence chứ không chạy waveform encoder inference; raw WAVs hiện diện nhưng không cần tái trích xuất.
- **MixGoP-inspired GMM adaptation đã chạy**, là non-CTC nhưng không faithful reproduction: S/O `.19307/.12649`, dưới MB03 neural `.42321/.27671`. Không có insertion output.
- **Không có hiệu chỉnh calibration, F1/warning budget, hoặc kiểm tra cảnh báo thực tế mới.** Không học sigmoid hậu nghiệm để gọi là posterior.

## Thí nghiệm đã xác minh

### A — matched E4 / neural

ID baseline neural: `RQ1-MB03-C-SPEAKER-GRL-GROUPCV-S101` (receipt MB03; PLAIN head). E4: MB05 seed101 canonical mean scores. Ghép chính xác `(utterance_id, slot_idx)`, kiểm tra mỗi fold và số target; 59,562 rows, 7,381 S và 1,825 O; cùng bốn speaker groups chính thức.

| Lớp | Neural AP | E4 AP | Spearman | AP trong neural max-probability quartile thấp (neural / E4) |
|---|---:|---:|---:|---:|
| S | .42321 | .31972 | .41049 | .33016 / .33927 |
| O | .27671 | .25936 | .22936 | .19626 / .28315 |

Fold S neural/E4 AP: F0 `.4440/.2858`; F1 `.3610/.3041`; F2 `.4595/.3598`; F3 `.4169/.3279`. Fold O: `.2656/.2337`; `.1903/.1755`; `.3555/.3688`; `.2997/.2148`. E4-only top-5% unique TP: S 194 pooled, O 56; khác biệt support không tự nó là cải thiện quyết định. Pooled phone-prevalence-only AP S `.33608`, O `.10240`; within-phone E4 percentile AP S `.22168`, O `.20351`. Các control này là mô tả hậu nghiệm, chưa phải cross-fitted model conditional.

Composition rank-mean pooled toàn OOF: AP-S `.44547`, AP-O `.35708`, so với neural `.42321/.27671`. Đây là **exploratory**, không dùng để tuyên bố cross-fitted improvement: phép percentile normalization nhìn pooled OOF, không có outer-test isolation. Không có fixed per-fold composition benchmark đủ hợp lệ hay learned fusion. Theo fold, E4 O AP tốt hơn neural chỉ ở F2; E4 không vượt neural S ở fold nào.

### B — insertion

MB03 grouped receipt: AP-I `.1621249`, fold AP `.18879/.10271/.10945/.25154`, support `134/127/150/149` trong tổng `560`, 61,362 gaps. Các predictions có thể kiểm chứng trong `experiments/rq1_method_bench_03/agent_c/speaker_grouped_oof_predictions.npz`, keys `gap_PLAIN_{0..3}`; kích thước lần lượt `15266×2`, `15298×2`, `15439×2`, `15359×2`. Đây là phát hiện của supervisor audit và sửa nhận định “không tìm thấy score array” trong lane report; cần ghép labels/event IDs để tái tạo chi tiết trước khi dùng làm control mới.

Probe MB06 dùng sáu thống kê từ cửa sổ ±5 frame quanh predicted canonical gap center của frozen CTC logits, fit balanced logistic regression trong từng outer fold. Pooled AP-I `.02220`, pooled static same-classifier control `.03557`, neural `.162125`; held-fold temporal AP `.02004/.03068/.01982/.02621`, vượt static chỉ Fold 3. P@5% pooled `.0313`; train-fit pooled AP `.02363`. Runtime CPU 15.13 s; peak memory chưa đo. Không lặp seed. Kết quả bác bỏ lợi ích của phép local temporal pooling cụ thể này, không chứng minh mọi timing evidence vô ích.

### C — external method

`RQ1-MB06-C-MIXGOP-ADAPT-GROUPED-S101`, CPU 4.34 s, peak memory chưa đo, không checkpoint. GMM theo phone được fit trên correct TRAIN targets và PCA fit riêng trong mỗi outer fold; K=1/K=4. K4 pooled AP `.19307 S/.12649 O`; K1 `.19170/.11509`. K4 per-fold S/O: F0 `.1761/.0870`; F1 `.1741/.1042`; F2 `.2165/.2181`; F3 `.2112/.0936`. Chỉ có target S/O. Không lặp seed vì negative screen; không reproduction fidelity claim. Nguồn MixGoP (Choi et al., NAACL 2025), Vidal et al. SLaTE 2023, Korzekwa et al. Interspeech 2021 được trích ở báo cáo Agent C.

## Đáp án cho câu hỏi chiến dịch

1. E4 và neural có rankings khác nhau, nhưng **complementarity theo nghĩa conditional useful information chưa established**.
2. E4 có TP ngoài neural top 5%: 194 S và 56 O; không chứng minh có lợi ở policy chung hoặc chi phí FP chấp nhận được.
3. E4 thêm thông tin vượt intended-phone priors? **Inconclusive**: S phone-only AP `.33608`, within-phone AP `.22168`; O within-phone `.20351` và phone-only `.10240`, gợi ý O hơn S nhưng retrospective, chưa cross-fit.
4. Fixed score integration? Rank-mean pooled có AP tăng exploratory, không leakage-safe confirmatory; **không có fixed validated gain**.
5. Learned cross-fitted integration? **Không chạy**.
6. Nested speaker-grouped? **Không hoàn tất; không có leakage-free fusion claim**.
7. Calibration/F1/warning trade-offs? **Không chạy** trong MB06.
8. Insertion method mới? **Có**: six-feature local temporal pooling probe trên frozen CTC frame logits với predicted alignment, balanced logistic regression.
9. Beat neural AP-I? **Không**: `.02220` so với `.162125`; cũng dưới matched static control `.03557`.
10. Consistency insertion? held AP `.02004/.03068/.01982/.02621`; chỉ hơn static ở fold 3. Không có seed replication.
11. Timing evidence materially help? **Không được hỗ trợ cho local pooling này**; timing methods rộng hơn vẫn inconclusive.
12. External methods: MixGoP GMM, direct SSL MDD, weakly-supervised word-level pronunciation detection; joint alignment/phonological edit methods ở Agent B.
13. Implement? MixGoP-inspired GMM adaptation chạy (không faithful reproduction); Agent B chạy temporal pooling probe.
14. Có phương pháp nào vượt neural relevant reference? **Không**; GMM dưới neural S/O.
15. Repeated positives across seeds? **Không**; không có MB06 positive claim cần lặp.
16. Bác bỏ: chưa được phép bác bỏ timing nói chung; T2 local-max cụ thể đã không giúp trong MB05; GMM adaptation kém direct neural; fusion positive chưa được chứng minh.
17. Unified trained model? **Chưa có cơ sở** cho E4 fusion; insertion cần thí nghiệm riêng trước.
18. Bằng chứng tốt nhất hiện tại là neural direct head cho S/O/I; E4 omission evidence là follow-up hypothesis, chưa xác nhận; insertion-localization là gap ưu tiên nhưng cần real frame-level input.

## Tái lập

Lệnh và artefact đầy đủ ở `METHOD_BENCH_06_REPRODUCTION.md`; chi tiết tại Agent A/B/C reports. Môi trường lúc kiểm kê: `.venv`, Python Torch 2.6.0+cu124, `torch.cuda.is_available()=True`; GTX 1650 4GB idle trong `nvidia-smi`. MB06 GMM/analysis chạy CPU; không có GPU checkpoint/job, không đo peak memory. Không thay đổi MB05 lịch sử, không dùng DEV_SEARCH để fit/chọn, không truy cập sealed data.

**Kết luận:** không đủ bằng chứng để build unified fusion. E4 complementarity vẫn inconclusive; MixGoP adaptation và temporal pooling probe đều negative so với controls. Insertion có một MB06 experiment thực thi, nhưng thất bại trước static lẫn neural, nên không có cải thiện exact-gap.
