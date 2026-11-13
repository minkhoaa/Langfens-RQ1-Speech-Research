# Kiểm toán độc lập EVAL-V3 (lane C)

**Phạm vi:** tính lại số liệu từ raw prediction NPZ seed 101 bằng arithmetic riêng; xác minh manifest/speaker; kiểm tra đường suy luận WAV và nguồn tham chiếu. Script không import evaluator EVAL-V3. Không chạy neural forward, không đọc TEST_SEALED/GPG. Nguồn: `03e4968ffd8908782d0bf463b6d101e6b6872372`.

## Đối chiếu manifest

| Split | Utterance | Speaker | Target | Gap | Gold C/S/O/I |
|---|---:|---:|---:|---:|---|
| TRAIN | 1.800 | 12 | 59.562 | 61.362 | 50.356 / 7.381 / 1.825 / 560 |
| DEV_CAL | 300 | 2 | 10.043 | 10.343 | 7.800 / 1.781 / 462 / 102 |
| DEV_SEARCH | 599 | 4 | 19.755 | 20.354 | 17.778 / 1.768 / 209 / 125 |

Mỗi utterance thỏa N target/N+1 gap; ID utterance duy nhất trong từng split. Ba giao speaker đều rỗng. Hash manifest và ledger nằm trong `experiments/evaluation_v3/INDEPENDENT_AUDIT.json`. Số 23.044/23.643 là sai khác narrative lịch sử, không phải kích thước manifest hiện tại.

## Tính lại từ raw score ledger

AP được tính độc lập bằng cách sắp điểm giảm dần, gộp tie groups nguyên vẹn, rồi cộng `positive_in_group × precision_at_group_end` / tổng positives (stepwise AP). Không lấy AP từ evaluator hoặc từ confusion matrix.

| Seed101 / split | AP S | AP O | AP I exact-gap | Macro3 error AP | Hỗ trợ S/O/I |
|---|---:|---:|---:|---:|---|
| DEV_CAL | 0,528023 | 0,478873 | 0,176870 | 0,394589 | 1.781 / 462 / 102 |
| DEV_SEARCH | 0,364602 | 0,159433 | 0,186715 | 0,236917 | 1.768 / 209 / 125 |

DEV_SEARCH argmax confusion target (hàng gold C/S/O, cột predicted C/S/O): `[[15467,1546,765],[770,863,135],[54,11,144]]`. One-vs-rest confusion `TP/FP/FN/TN`: S `863/1557/905/16430`; O `144/900/65/18646`; I `43/200/82/20029` trên gap. Precision/Recall/F1 tương ứng S `0,356612/0,488122/0,412130`; O `0,137931/0,688995/0,229848`; I `0,176955/0,344000/0,233696`. Macro3-error F1 = `0,291891`. Joint false warnings = 2.657, FWR = 4,435726/utterance. DEV_CAL tương ứng 1.271 argmax warnings trên 300 utterances, FWR = 4,236667 (mẫu số lấy trực tiếp từ metadata ledger; không hard-code 599).

Đây là kiểm tra argmax thô, **không phải** lịch sử threshold operating point hay một chính sách khả thi theo rủi ro. Độc lập áp dụng các threshold đã lưu (không chạy threshold search) trong `experiments/student_evaluation/so_diagnosis/policy_counterfactuals.csv` lên seed101 NPZ cho kết quả shared Gold S/O/I TP `1/3/29`, FP `0/15/48`, 63 cảnh báo, FWR `0,105175`; separated S/O/I TP `27/3/29`, FP `12/15/48`, 75 cảnh báo, FWR `0,125209`. Cả hai khớp stored CSV. Code arithmetic, thresholds và cờ khớp được lưu trong JSON. Đây là replay frozen thresholds, không phải tái hiện việc lựa chọn threshold trên DEV_CAL.

## Tham chiếu và WAV inference: điểm cần phân biệt

`research/student/infer_student.py` có genuine audio+intended-ARPAbet đường đi, dùng local-only CTC (`charsiu/en_w2v2_fc_10ms`), alignment trên audio/reference rồi student checkpoint, và xuất N target/N+1 gap. Receipt hiện có (`experiments/student_training/student_audio_inference_receipt.json`) chứng minh đã chạy CUDA trên WAV `ABA_arctic_a0003.wav`, shape 39/40; checkpoint thật ở `models/student/STU-01-GOLD-ONLY_seed101.pt` (SHA kỳ vọng `8ce4ebe5…f994090`). Không lặp neural forward trong audit.

Tuy nhiên `tools/parse_l2arctic.py` tạo `target_phone` và thứ tự từ phone intervals của TextGrid chú giải. Receipt ghi sequence intended lấy trực tiếp từ `target_phone`. Việc inference không nạp gold timing/realized label **không tự chứng minh nguồn reference độc lập với annotation**. Các phone identity có thể biểu diễn canonical target, nhưng provenance độc lập prompt-derived không được chứng minh bởi đường hiện tại. Cũng ghi nhận kiểm tra parent: `ERMS_arctic_a0003` trong DEV_CAL bắt đầu F,R,ER,... trong khi inference_sample trước đó F,AO,R,...; mismatch này nhấn mạnh không thể giả định sequence annotations là transcript intended chuẩn.

Parent đã chạy smoke sampled trên first DEV_CAL prompt bằng local CMUDict: `experiments/evaluation_v3/waveform/WAVEFORM_SMOKE_RECEIPT.json` (SHA-256 trong audit JSON). Receipt ghi WAV+orthographic prompt, 39 target/40 gap, leading/trailing gap 0/39, GPU GTX 1650, 413,535 MiB peak, 0,8902 s, và sentinel fields `target_phones`, `realized_phone`, `xmin`, `xmax`, `gold_label`, `TextGrid` bị từ chối. Đây là bằng chứng **PASS_SAMPLED_WAVEFORM_ONLY**, không chạy bởi lane C. Receipt chủ động không khẳng định match slots với manifest; prompt CMUDict reference F,AO,R,... trong khi TextGrid-derived manifest ERMS record có F,R,ER,... . Do đó genuine prompt-derived waveform path đã có sampled evidence; alignment giữa reference intended và nhãn canonical manifest, và mọi cached full-population reference provenance, vẫn **UNVERIFIED**.

## Báo cáo nghiệm thu lịch sử và semantic caveat

- Đọc được Phase 2 definitive report và Phase 3 final/diagnosis/independent review. Không tìm thấy Phase 1 acceptance report dưới tên/đường đã dò trong `reports/`; tình trạng Phase 1 acceptance: **UNVERIFIED / không thấy artifact**, không suy diễn. Repository source snapshot audit là `03e4968ffd8908782d0bf463b6d101e6b6872372`; historical diagnostic reference là commit `45cd3d2` (`Add Phase 3 CUDA student distillation experiments and corrected evaluation`). Báo cáo chẩn đoán hiện tại được đọc từ tree 03e; script `research/student/diagnose_so_failure.py` không tồn tại trong tree 45cd3d2. Không nhập nhằng mã nguồn snapshot với provenance báo cáo.
- Known issue 23.044/23.643: báo cáo cũ lưu narrative sai; manifest thực là 19.755/20.354.
- Đã kiểm tra trực tiếp 19.755 target event trong `experiments/complementarity/CPL01_JOINT_PREDICTIONS.jsonl` (unique event IDs): gold O support 209; flag S trên O=48, flag O trên O=68, giao cùng flag=31, hợp raw events=48+68−31=**85**. Con số report 94 hàm ý overlap 22 và không khớp flags thô. Raw event ledger vì vậy giải quyết inconsistency: union đúng theo flags là 85/209; giữ 94 như lỗi lịch sử, không sửa old report.
- Teacher S/O là binary scores riêng, không phải simplex 3 lớp. Phase 3 argmax score distributions là ba lớp từ student; decision FWR counts wrong-type predicted warning một lần theo lớp dự đoán. Không nhập FP_CORRECT vào false warnings.
- Phase 3 calibration/isotonic và threshold selection cùng dùng DEV_CAL theo report: in-sample calibration/selection reuse; DEV_SEARCH đã development-exposed, không phải test độc lập.
- Kiểm toán này chỉ chọn raw seed101 student ledger; không audit độc lập mọi checkpoint/experiment/teacher. Không tuyên bố toàn bộ inventory/replay đã được kiểm chứng.

## Tái lập

```bash
python research/evaluation_v3/independent_audit.py \
  --root /home/khoa/Projects/New-Method-Research \
  --output experiments/evaluation_v3/INDEPENDENT_AUDIT.json
```

Không có neural forward, network, training hoặc access tới protected holdout trong lệnh này.
