# RQ1-UNIFIED-01 — Báo cáo Agent B: chuyển giao tri thức chuyên gia

## Kết luận hiện tại

Đã khôi phục được giao diện score và công thức của các kết quả Phase 4. Agent A đã huấn luyện các baseline neural gold-only U2, U3 và GAPW3 trên CUDA. Tuy nhiên chưa có teacher target TRAIN hợp lệ cho EXP-24, EXP-36 hay EXP-43. Vì vậy K1–K4 chưa chạy do provenance teacher không hợp lệ: score từ checkpoint đã fit trên cùng TRAIN là in-sample, còn vector DEV_CAL/DEV_SEARCH không thể dùng làm target TRAIN.

Đây là giới hạn về provenance của mục tiêu distillation, không phải kết quả phủ định distillation. K0 gold-only đã có để làm đối chứng matched. Phân tích hiện có và thiết kế ablation nằm trong [SOURCE_RECOVERY.md](../experiments/rq1_unified_01/agent_b/SOURCE_RECOVERY.md).

## Đối chứng gold-only đã đo

Các kết quả DEV_SEARCH dưới đây là exploratory, một seed (101), 19,755 target và 20,354 gap; các AP không phụ thuộc ngưỡng. U2 là K0 gold-only gần nhất: target head flat C/S/O CE cùng neural exact-gap head. GAPW3 chỉ đổi tỷ lệ positive/negative loss cho gap head thành 3; target branch giữ nguyên. U3 thay target loss bằng omission BCE cộng conditional C/S BCE.

| Arm | AP S | AP O | AP I | Macro-3 AP | Thời gian train | VRAM đỉnh |
|---|---:|---:|---:|---:|---:|---:|
| K0 / U2 flat, gold-only | 0.37136 | 0.18532 | 0.11799 | 0.22489 | 24.88 s | 41.33 MiB |
| U3 hierarchical, gold-only | 0.36241 | 0.17612 | 0.12382 | 0.22078 | 24.62 s | 41.33 MiB |
| GAPW3, gold-only | 0.37136 | 0.18532 | 0.12735 | 0.22801 | 23.56 s | 41.33 MiB |

GAPW3 tăng AP I 0.00936 và Macro-3 AP 0.00312 so với U2, không đổi AP S/O. AP I vẫn thấp hơn Gold-Only legacy 0.18672 và EXP-43 0.28506; do đó GAPW3 là đối chứng gần nhất để đo distillation, không phải bằng chứng chuyển giao đã thành công. Ba arm dùng GTX 1650 4 GB. Chi tiết recipe/checkpoint có trong [báo cáo Agent A](RQ1_UNIFIED_A_ACOUSTIC_TRAINING.md).

## Công thức score đã xác nhận

| Recipe | Omission | Correct/Substitution | Insertion |
|---|---|---|---|
| Omission-specialist | EXP-24 omission | Phân bổ khối lượng không-omission theo tỷ lệ C/S gốc của 0.8 EXP-36 + 0.2 Gold-Only khớp seed | 0.8 EXP-43 + 0.2 Gold-Only khớp seed |
| Fixed head-wise blend | EXP-36 nằm trong head target: 0.8 EXP-36 + 0.2 Gold-Only | Cùng vector target đó | 0.8 EXP-43 + 0.2 Gold-Only |

Các recipe là score composition cố định; không phải checkpoint neural hợp nhất. Chúng báo cáo 19,755 target và 20,354 gap trên DEV_SEARCH, với event order được đối chiếu. DEV_SEARCH là dữ liệu phát triển đã lộ.

Năm seed của omission-specialist đạt Macro-3 AP 0.3018 và Macro-3 F1 0.3178; fixed blend đạt 0.2955 và 0.3296. Đây là tham chiếu để đánh giá chuyển giao, không chứng minh rằng ba chuyên gia có thể distill vào một mô hình.

## Provenance mục tiêu

- EXP-24 có checkpoint và đầu ra DEV_SEARCH/DEV_REPLICATION; spec xác nhận fit TRAIN, nhưng chưa tìm thấy đầu ra TRAIN speaker-held-out.
- EXP-36 và EXP-43 có checkpoint-inference replay, recipe, hashes và score vectors DEV_CAL/DEV_SEARCH. Receipt ghi rõ replay chỉ làm phép toán score CPU, không huấn luyện hay neural forward mới.
- Báo cáo Phase 3 có teacher OOF ba fold theo speaker, nhưng đó là bản sao kiến trúc champion/EXP01, không phải EXP-24/36/43. Calibration DEV_CAL áp vào TRAIN được mô tả là ngoại suy có shift; nếu dùng proxy này sau, cần raw OOF scores và một thí nghiệm riêng.

## Kế hoạch K ablation

K0 hiện có. K1–K4 vẫn chưa chạy vì chưa có OOF teacher theo speaker. Khi có OOF hợp lệ, giữ nguyên gold loss, dữ liệu, seed, budget và acoustic backbone; thêm soft loss cho từng head riêng:

1. K1 chỉ distill omission từ EXP-24 OOF.
2. K2 chỉ distill xác suất substitution có điều kiện `pS/(pC+pS)` từ EXP-36 OOF 3-way; omission examples không có conditional C/S loss nhưng vẫn nhận omission gold loss.
3. K3 chỉ distill binary insertion ở đúng N+1 gap từ EXP-43 OOF.
4. K4 gộp những can thiệp đơn lẻ đã tăng AP lớp mục tiêu mà không làm hại rõ rệt lớp còn lại.

Không giả định các teacher độc lập hoặc đã hiệu chuẩn hoàn hảo. Mỗi so sánh cần K0 matched và lựa chọn checkpoint trên DEV_CAL; DEV_SEARCH chỉ để đọc kết quả khám phá.

## Hành động tiếp theo

Tạo speaker-fold OOF output tương thích với từng expert là bước cần trước K1–K3. Không chạy distillation từ score in-sample hay score DEV. Sau khi có OOF, chạy K1 riêng trước; chỉ tiếp tục K2/K3 nếu K1 có tín hiệu, rồi K4 chỉ gộp các can thiệp head-wise hữu ích.

## Giao diện phục hồi và chuẩn bị OOF

NPZ EXP-24 có `p_target (N,3)`, `p_gap (M,)`, logits tương ứng và metadata target/gap kích thước `(rows,2)`; thứ tự lớp target là C/S/O. NPZ replay EXP-36 và EXP-43 có `p_target (N,3)`, `p_gap (M,)`, cùng metadata. Hai EXP sau là composition của 7 và 9 checkpoint score, không phải checkpoint neural riêng; receipt nói rõ chỉ có score composition CPU, không train và không forward neural. EXP-24 là checkpoint model đã train trên TRAIN nên điểm tự suy diễn trên TRAIN là in-sample.

Đã thêm [distillation.py](../experiments/rq1_unified_01/agent_b/distillation.py): script CPU chỉ chấp nhận NPZ TRAIN_OOF kèm receipt nguồn/hash/fold, kiểm tra speaker-held-out và ID/order chính xác, sau đó xuất riêng ba vector K1/K2/K3. Mô-đun cung cấp thêm soft loss cho omission, C/S có điều kiện (mask hàng gold omission) và insertion exact-gap; mọi loss đều là phụ trợ và phải giữ gold loss hiện có. Hiện không chạy script tạo output vì chưa có đủ nguồn OOF đạt kiểm tra. Đã kiểm tra syntax và giao diện `--help`; không chạy neural training.

Kế hoạch ứng viên cụ thể: trước tiên K0 gold-only so với K1, chỉ thêm omission BCE từ EXP-24 OOF với hệ số 0.25. Giữ cùng seed, khởi tạo, dữ liệu, budget và gold CE của K0. Nếu K1 có tín hiệu, chạy K2 conditional C/S từ EXP-36 OOF và K3 exact-gap insertion từ EXP-43 OOF riêng lẻ, không gộp trước khi thấy hiệu quả từng head. Đây là head-score supervision và không lặp lại H-004 feature-cosine.
