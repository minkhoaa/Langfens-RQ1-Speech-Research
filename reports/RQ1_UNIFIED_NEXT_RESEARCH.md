# RQ1-UNIFIED-01 — Hướng nghiên cứu tiếp theo

## Quyết định sau chiến dịch hiện tại

Không tiếp tục tinh chỉnh trọng số insertion. Paired seed 101/202/303 cho GAPW3−U2 ở AP-I là `+.00936`, `+.03019`, `−.03948`; Macro-3 AP là `+.00312`, `+.01006`, `−.01316`. Trung bình 3 seed lần lượt chỉ `+.00002` và `+.00001`. Chênh lệch đảo dấu ở seed 303 nên hiệu ứng GAPW3 không ổn định. Hierarchical U3 giảm AP S/O/Macro-3 so với U2, dù AP-I tăng `+.00583` ở seed 101.

K1–K4 distillation chưa chạy: không có TRAIN speaker-held-out OOF predictions cho EXP24/36/43. EXP36/43 canonical artifacts chỉ là score composition trên DEV_CAL/DEV_SEARCH; chạy checkpoint fitted trên TRAIN để tạo score TRAIN sẽ là in-sample. Không dùng score development làm target TRAIN. Bổ sung OOF expert models cần thiết kế và ngân sách riêng.

## Thí nghiệm kế tiếp được đề xuất

Thử một biến đổi gap-boundary cụ thể, giữ nguyên U2 flat target head, seed, optimizer, epoch budget và gold loss:

1. Trích frozen Wav2Vec2 layer-6 frame states trên WAV từng utterance theo normalized H-002 contract.
2. Dùng center do frame classifier/Viterbi dự đoán từ intended reference để tạo mỗi internal gap window 17 frame quanh midpoint hai predicted phone centers; leading/trailing gap dùng cửa sổ sát acoustic edge. N=0 dùng pooled utterance representation, báo là OOD nếu vẫn không có ví dụ TRAIN tương ứng.
3. Condition gap head bằng hai intended-phone vectors; train đúng N+1 gap labels. Không dùng gold timing/realized phones.
4. So sánh matched gap head hiện tại với boundary-window MLP trên cùng seed trước; chỉ nhân rộng seeds nếu AP-I và Macro-3 tăng cùng hướng, và CAL policy không làm giảm F1 nghiêm trọng. Chọn ngưỡng trên DEV_CAL; DEV_SEARCH chỉ là mô tả do đã lộ.

Đây là một thay đổi biểu diễn gắn với lỗi đã quan sát (AP insertion 0.139 trung bình ba seed so với Gold-Only/H-002 0.1867), không phải phép tìm trọng số. H-003 local linear pooling đã âm tính; đề xuất này khác ở việc dùng hidden-state window và neural nonlinear classifier quanh predicted boundary. Cần ghi rõ nếu phép so khớp không thành công.

## Điều kiện trước khi distillation

Tạo hoặc tìm nguồn TRAIN_OOF speaker-held-out hợp lệ cho từng teacher. Recipe EXP24 cho omission, EXP36 C/S và EXP43 gaps phải khớp kiến trúc/score source đúng; receipts cần model hash, fold speaker, training speaker list, prediction hash và exact TRAIN event order. Bắt đầu K1 riêng sau K0; omission soft loss phụ trợ, giữ gold loss. K2/K3 chỉ sau khi từng can thiệp độc lập có tác dụng. Không gộp trước kết quả.

## Điểm phải giữ trong báo cáo kế tiếp

- DEV_SEARCH exposed, bốn speaker; không dùng làm ước lượng kiểm thử độc lập.
- Mô hình hiện tại ASR-finetuned Wav2Vec2, không pure SSL; intended-reference provenance độc lập chưa được xác minh đầy đủ.
- Checkpoint U2 seed101 là một model neural duy nhất cho S/O/C target và insertion gaps, nhưng frontend inference cần local frozen CharSiU alignment model và Wav2Vec2 weights.
- U2 chưa vượt Gold-Only hoặc H-002 Macro-3 AP; nguyên nhân chính là ranking insertion yếu. Omission AP tốt hơn Gold-Only nhưng còn cách EXP24.
