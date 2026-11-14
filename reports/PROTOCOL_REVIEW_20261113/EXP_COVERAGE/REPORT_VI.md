# Mức độ nghiên cứu trong các thư mục EXP — 2026-11-13

**Kết luận:** Có nhiều thử nghiệm thực chất, nhưng chưa đủ bằng chứng để kết luận đã nghiên cứu hết các phương pháp, hoặc CROTTC-IF/PER-MDD/X-MDD đã được áp dụng đầy đủ rồi thất bại. Giữ quyết định chọn phương pháp/protocol ở trạng thái chưa chốt.

## Phạm vi và nhận dạng

Theo chỉ dẫn của người dùng, phạm vi chính là các thư mục con trực tiếp `experiments/EXP-<số>-...` của repo hiện tại. Snapshot kiểm tra có đủ EXP-01…110 và ba thư mục EXP-A101/B101/C101. Không tìm thấy thư mục đánh số trực tiếp EXP-111…130 trong phạm vi đó. `REQUESTED_001_130_COVERAGE.csv` ghi riêng cả 130 số, không lấp khoảng thiếu bằng experiment ở namespace khác.

Có thêm campaign `Pronunciation-Research-EXP101` nằm trong repo, và một bản ở thư mục Projects kế bên. Campaign này có registry EXP-101…120, nhưng trùng số EXP-101…110 với campaign chính và mang phương pháp khác. `NESTED_CAMPAIGN_IDENTITY.csv` ghi 20 dòng registry của bản nested làm phụ lục nhận dạng, không cộng vào 110 EXP chính hoặc gọi là một chuỗi 130 thí nghiệm liên tục. EXP-116/120 trong campaign nested là synthesis/packaging, không mặc định checkpoint mới.

Việc không tìm thấy một thư mục trong phạm vi nêu trên không chứng minh chưa từng có experiment đó trên máy hoặc trong lịch sử. Ba nhánh A/B/C cũng được ghi riêng, không biến thành EXP-111/112/113.

## Kiểm tra từng hồ sơ

Đã đọc RUN_SPEC, RUN_STATE, REVIEW, TRAIN.log và bảng hash hiện có ở cả 110 thư mục. Phân tích AST 109 source `methods/M*/run.py`; EXP-20 là synthesis, không thấy source tương ứng theo quy tắc liên kết đã kiểm. Cả 110 spec có hypothesis ID; 104 hồ sơ có thư mục hypothesis khớp tên chính xác, đã đọc Markdown trong các thư mục này. Sáu hồ sơ không có thư mục hypothesis khớp tên không được diễn giải là chưa có nghiên cứu ở bất kỳ nơi nào khác.

`EXPERIMENT_EVIDENCE_MATRIX.csv` có một dòng cho từng EXP chính, gồm hypothesis, source/classes, loại bằng chứng, trạng thái closure, receipt, phạm vi đối chiếu và giới hạn. EXP-12 có method ID trỏ M11 và source M12 đóng vai evaluator; không tự coi đây là sai liên kết. Các chỉ dấu AST chỉ mô tả source, không chứng minh một lệnh đã được thực thi hoặc một cơ chế có hiệu quả.

| Loại bằng chứng ở 110 EXP chính | Số hồ sơ | Cách hiểu |
|---|---:|---|
| Neural checkpoint có receipt replay | 50 | Có model và evidence suy luận đã lưu; không đồng nghĩa 50 cơ chế độc lập hoặc 50 paper reproduced. |
| Composition có receipt replay | 24 | Có kết hợp score/model; không mặc định thêm backbone hoặc training mới. |
| Policy có receipt replay | 5 | Có hiệu chỉnh/quy tắc quyết định; tách khỏi cải thiện representation. |
| Deterministic control | 3 | Có đối chứng; không phải model neural mới. |
| Audit/report/packaging | 12 | Công việc nghiên cứu/hạ tầng có giá trị, không đếm như 12 thử nghiệm kiến trúc. |
| Source hoặc bằng chứng giới hạn | 16 | Có protocol ngoài phạm vi, thiếu artifact, thiếu seed weights, runtime bị chặn, hoặc chỉ aggregate. Không đồng loạt gọi là “chưa chạy”. |

Phân loại bám receipt và disposition đã công bố, không chỉ bám tên folder. 82 EXP chính có receipt trực tiếp; 28 còn lại có disposition riêng. Toàn bộ closure còn có teacher và variant ngoài 110 EXP này, nên tổng closure 92 không mâu thuẫn với 82.

## Kiểm chứng evidence đã tồn tại

Lượt này không forward lại model hoặc training. Đã kiểm lại trực tiếp 92 receipt closure, SHA-256 của 184 score files CAL/SEARCH và 7 output của publication: tất cả khớp. Phạm vi gồm 50 neural, 24 composition, 6 policy/variant, 3 control và 9 teacher.

Publication có 1.340 dòng: 500 baseline lịch sử giữ lại và 840 operating evidence mới. Từ JSON counts của 840 dòng mới, đã kiểm riêng công thức FWR và 7.560 phép tính precision/recall/F1 S/O/I; khớp số học. Không có dòng nào trong 840 dòng này đồng thời có precision/recall mỗi S/O/I ≥80% và FWR ≤0,20. Đây là kết quả trên các operating point đã lưu, không chứng minh mọi threshold khả dĩ hoặc mọi phương pháp đều không đạt.

Đây là **hash/count verification của release và score artifacts**, không phải tái lập training, đánh giá lại toàn bộ score ledger bằng evaluator, hay xác nhận all-history scientific validity. Replay mạnh nhất trước đó đã làm gì và giới hạn ở đâu được ghi trong `reports/EVAL_V3_CLOSURE_REPORT.md`; không nhận công việc trước đó là vừa chạy lại trong lượt này.

CSV COMPARISONS cũ có 38 dòng mà các cột macro AP/F1 và F1 S/O/I được kiểm đều bằng 0. Không dùng CSV đó một mình để kết luận 38 phương pháp thất bại: nó không phản ánh đầy đủ evidence mới ở closure.

## Đối chiếu tên phương pháp với code

| EXP | Cơ chế đọc được | Điều chưa được chứng minh |
|---|---|---|
| 17 | Focal/CE/BCE trên event heads; không có CTCLoss/ctc_loss call trong source này | Không dùng tên “CTC” để nhận là đã thử full realized-phone CTC training. |
| 24 | Attention trên slot features và event losses | Chưa phải OTTC/CR/IF hoặc paper CROTTC-IF. |
| 22/26/49 | Learnable prototypes cho lớp C/S/O, contrastive regularization | Chưa phải truy xuất phone embeddings kiểu PER-MDD. |
| 59 | Có diffusion classes và neural model thực | Không phải bằng chứng đã thử một trong ba paper được hỏi. |
| 101 | LPC area features tính trên các đoạn theo manifest start/end | Chưa chứng minh end-to-end learned physical vocal-tract inversion. |
| 102 | Low-rank adapter trên đầu vào đặc trưng 72 chiều | Chưa phải fine-tuning WavLM transformer backbone. |
| 104 | `y_soft = 0.90 × one_hot + 0.10 × prior` | Chưa có nhiều người gán nhãn độc lập hoặc adjudication. |
| 89/107 | Routing/cascade có các giới hạn được closure ghi riêng | Score-space routing không tự chứng minh end-to-end speedup. |

Những khác biệt này không làm mất giá trị của thí nghiệm đã thực hiện. Chúng giới hạn việc dùng kết quả để bác bỏ một cơ chế rộng hơn tên trong báo cáo.

## Ba phương pháp đề xuất có được áp dụng không?

**Trong 110 thư mục EXP chính đã kiểm: chưa xác nhận EXP nào áp dụng đầy đủ ba phương pháp.** Không có trace paper → cơ chế đầy đủ trong source → recipe/run → kết quả matched đủ để gọi là full application/reproduction.

- **CROTTC-IF:** Có những EXP về attention/CTC, nhưng chưa thấy tổ hợp OTTC + consistency regularization + Indirect Fusion teacher. Ngoài các EXP đánh số, `MB02-B-OTTC-ORDERED-STATES-S101` có một static transport adaptation trên window; receipt tự khai `INSPIRED_IMPLEMENTATION`, thiếu learned marginals/full-sequence decoder. M1 PhnMonoSSL dùng baseline từ repo IF-MDD; `REFERENCE_FIDELITY.json` ghi rõ không phải CROTTC-IF và `paper_result_replication=false`.
- **PER-MDD:** EXP-22/26/49 dùng prototypes C/S/O được học với classifier; không đồng nhất với pool embeddings theo phone và truy xuất frame. Campaign nested có PER-MDD trong literature và H2 kNN trong candidate hypotheses; proposal không phải run receipt. Chưa xác nhận pipeline HuBERT/pool-phone/retrieval tương ứng trong các EXP chính.
- **X-MDD:** Chưa thấy chain Qwen3-ASR + canonical speech/phones + SFT + GRPO với reward detection/PER trong các EXP chính. Teacher/distillation hoặc attention ở các EXP khác không đủ để coi là X-MDD.

Kết luận chỉ nói về bằng chứng tìm được trong phạm vi rà soát. Không dùng keyword vắng mặt làm bằng chứng duy nhất; đã đối chiếu source/classes và các dấu vết gần cơ chế. Những adaptation gần nhau không được gọi là đã tái lập paper; việc chưa tái lập cũng không bảo đảm phương pháp mới sẽ tốt hơn.

## Có “đủ nghiên cứu” để chốt chưa?

Đủ để nói đã có một tập nghiên cứu rộng và nhiều model thật, đồng thời các operating points đã kiểm chưa đạt yêu cầu joint SOI/FWR. Chưa đủ để nói toàn bộ không gian phương pháp đã được nghiên cứu, ba paper đều đã thử đầy đủ, hay chất lượng thấp chỉ do supervision.

Một paper reproduction không phải yêu cầu cho mọi ý tưởng mới. Nhưng muốn dùng kết quả của một EXP để khẳng định một paper/hướng đã thất bại, cần chỉ rõ cơ chế nào được giữ, phần nào bị thay, data/supervision nào đã dùng, controls/ablations nào cô lập tác dụng và metric nào tương đương bài toán. Đây là khoảng thiếu trong traceability, không được đồng nhất với “không có nghiên cứu”.

Vì vậy, chưa chốt architecture hoặc thay protocol thành quyết định đã thông qua. Giữ hai vấn đề audit hiện tại là bằng chứng có phạm vi; dùng bảng từng EXP để chọn đúng phép thử còn thiếu. Không khởi động thêm training trong lượt này.

## Tái kiểm

```sh
.venv/bin/python reports/PROTOCOL_REVIEW_20261113/audit_experiment_coverage.py
```

Outputs: `SUMMARY.json`, `INPUT_HASHES.json`, `SOURCE_DETAILS.json`, `EXPERIMENT_EVIDENCE_MATRIX.csv`, `REQUESTED_001_130_COVERAGE.csv`, `NESTED_CAMPAIGN_IDENTITY.csv`. Không đọc audio, dataset archives, checkpoint weights hoặc official test. Các source/proposal chỉ được đọc như evidence, không import hay chạy.
