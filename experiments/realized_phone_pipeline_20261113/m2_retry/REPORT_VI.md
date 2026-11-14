# M2 retry — nhận diện âm lỗi và sửa kết luận cổng M2

## Kết luận và lý do thử lại

Kết luận cũ ‘chưa đạt M2’ đã trộn ba vấn đề: training đã hoàn tất; recognizer có bằng chứng âm học cho lỗi thật hay không; chẩn đoán S/O/I đạt operating point hay chưa. `m2_report.py` còn gán cứng `M2_quality_gate_complete=False`. Ngưỡng tăng Macro-3 AP +0,02 thuộc nhánh tiến bộ MDD, không phải tiêu chí M2 ở GOAL §8. Giữ nguyên receipt cũ và sửa cách diễn giải bằng audit mới, không sửa lịch sử để tạo PASS.
M2 yêu cầu model/checkpoint, realized predictions, PER/phân tích speaker/phone và bằng chứng hữu ích cho lỗi thật. Không có ngưỡng PER tuyệt đối hoặc điều kiện nghiệm thu sản phẩm đã được user chốt. Audit lần này kiểm bằng đối chứng audio cùng prompt; không xem model là đã sẵn sàng sản phẩm.
Hạn chế cũ có thật: fit CTC gần 0 nhưng cal PER quanh 26–27%, gợi ý overfit; chỉ giữ đúng khoảng 47–48% realized S và 33–36% S nguyên âm khi căn chỉnh với realized gold. Inference vẫn có lỗi; yếu về MDD không chỉ là vấn đề cách ghi trạng thái.

## Lượt training mới

Đã chạy local hai arm × 4 fold × seeds 101/202 = 16 run, mỗi run 32 epoch. Pointwise kernel1 và temporal kernel5 dùng cùng residual head/dropout0,2, weight decay0,001, frozen H6, CTC supervision, AdamW/lr cosine/batch16/clip5 và chọn checkpoint bằng cal PER mỗi 4 epoch. Shared initial weights và identity context được kiểm; thứ tự train/update giống nhau, chạy xen kẽ arm mỗi fold/seed. Không tune bằng held.
Trainable parameters: pointwise 460,457, temporal 461,993; chênh 1.536. Kernel5 thêm ±2 frame (20ms/frame), H6 vốn đã contextual. Old M2 có kiến trúc/regularization khác, nên chỉ new matched pair hỗ trợ quy thuộc tác động temporal. Không tuyên bố đã tách riêng mọi hiệu ứng capacity.
Dùng lại 1.243/1.800 exact realized sequences TRAIN; 6 fit/3cal/3held speaker. MDD đánh giá đủ 1.800 câu/59.562 target/61.362 gap mỗi seed; PER trên 39.477 phones/1.243 câu. Không official test, nhãn phone giả, gold boundary input hoặc thêm speaker. Prior temporal CNN dùng SM13 target/gap; lần này là continuous realized-sequence CTC.

## Kết quả nhận diện và chẩn đoán

FWR là held thực tế tại policy cal budget0,20. PER và MDD được báo riêng; hard edit là adapter chẩn đoán, chưa phải event posterior.

| Model | Seed | PER | Macro-3 AP | Macro-3 F1 | FWR |
|---|---:|---:|---:|---:|---:|
| Old M2 | 101 | 26.36% | 0.12829 | 0.01892 | 0.19000 |
| Old M2 | 202 | 26.83% | 0.12524 | 0.01762 | 0.20111 |
| pointwise | 101 | 26.07% | 0.13406 | 0.01856 | 0.18778 |
| pointwise | 202 | 26.52% | 0.12809 | 0.02385 | 0.20167 |
| temporal | 101 | 26.03% | 0.13399 | 0.01929 | 0.19000 |
| temporal | 202 | 26.30% | 0.12899 | 0.02232 | 0.19611 |

PER temporal − pointwise theo speaker, không coi seed/fold là mẫu độc lập:

Pooled PER delta: seed101 -0.038, seed202 -0.215 điểm %. Temporal có PER thấp hơn ở cả hai seed; chưa có xác nhận độc lập hoặc claim significance.

- Seed 101: median -0.082 điểm %, IQR [-0.165; +0.169], temporal tốt hơn ở 8/12 speaker.
- Seed 202: median -0.265 điểm %, IQR [-0.306; +0.065], temporal tốt hơn ở 7/12 speaker.

## Đối chứng âm học cùng prompt

Giữ reference/prompt nguyên, xoay hypothesis audio giữa held speaker có cùng prompt suffix và reference phones hoàn toàn giống nhau. Chỉ so trên cùng error events có cặp; đây là diagnostic, không đưa nhãn held vào train. Căn chỉnh về source slot để đo recognized realized S; I tính phone trùng trong đúng source gap (giữ multiplicity). Chặt hơn retention căn chỉnh với realized sequence; không xem số này là policy recall.
Reference-only không nhận audio. Cặp hoán đổi có thể vẫn mang cùng cách phát âm/lỗi tương tự; uplift chỉ đo trong nhóm kiểm tra, không chứng minh mọi lỗi/biến thể. Coverage không đủ toàn corpus: báo bên dưới.

| Model | Seed | Câu có cặp/1.243 | S đúng audio / swapped / ref (support) | I đúng audio / swapped / ref (support) | Mean speaker uplift S+I; interval mô tả |
|---|---:|---:|---|---|---|
| old_M2_retrospective | 101 | 543/1243 | 613 / 301 / 0 (1684) | 15 / 1 / 0 (130) | 17.01%; [12.42%; 21.00%] |
| old_M2_retrospective | 202 | 543/1243 | 655 / 306 / 0 (1684) | 18 / 1 / 0 (130) | 19.56%; [13.79%; 24.90%] |
| pointwise | 101 | 543/1243 | 656 / 339 / 0 (1684) | 18 / 1 / 0 (130) | 18.27%; [11.81%; 22.90%] |
| pointwise | 202 | 543/1243 | 630 / 302 / 0 (1684) | 20 / 1 / 0 (130) | 19.11%; [13.46%; 24.26%] |
| temporal | 101 | 543/1243 | 653 / 338 / 0 (1684) | 17 / 2 / 0 (130) | 16.89%; [10.94%; 22.05%] |
| temporal | 202 | 543/1243 | 633 / 301 / 0 (1684) | 17 / 1 / 0 (130) | 19.02%; [13.77%; 23.86%] |

Khoảng mô tả dùng bootstrap theo 12 speaker với seed cố định, không p-value hoặc xác nhận trên speaker mới. Các prompt, fold train set và seed có phụ thuộc; không suy diễn bootstrap thành bằng chứng độc lập. Audit old M2 là retrospective, không được gọi là preregistered.

Bằng chứng âm học có ích theo tiêu chí nghiên cứu M2: {'pointwise': True, 'temporal': True, 'old_M2_retrospective': True}. Cờ này tính từ kết quả đối chứng, không gán cứng. Nếu evidence=True, đủ cơ sở nghiên cứu alignment/uncertainty M3 với các giới hạn đã ghi; không đồng nghĩa nhận diện hoàn hảo hoặc product acceptance.

## Metric S/O/I tại policy B=0,20

| Arm | Seed | Lớp | Support | TP | FP | FN | Precision | Recall | AP |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| pointwise | 101 | SUBSTITUTION | 7381 | 93 | 190 | 7288 | 32.86% | 1.26% | 0.24273 |
| pointwise | 101 | OMISSION | 1825 | 18 | 82 | 1807 | 18.00% | 0.99% | 0.12352 |
| pointwise | 101 | INSERTION | 560 | 4 | 66 | 556 | 5.71% | 0.71% | 0.03593 |
| pointwise | 202 | SUBSTITUTION | 7381 | 89 | 185 | 7292 | 32.48% | 1.21% | 0.23056 |
| pointwise | 202 | OMISSION | 1825 | 23 | 97 | 1802 | 19.17% | 1.26% | 0.11957 |
| pointwise | 202 | INSERTION | 560 | 8 | 81 | 552 | 8.99% | 1.43% | 0.03415 |
| temporal | 101 | SUBSTITUTION | 7381 | 97 | 195 | 7284 | 33.22% | 1.31% | 0.23938 |
| temporal | 101 | OMISSION | 1825 | 19 | 95 | 1806 | 16.67% | 1.04% | 0.12581 |
| temporal | 101 | INSERTION | 560 | 4 | 52 | 556 | 7.14% | 0.71% | 0.03679 |
| temporal | 202 | SUBSTITUTION | 7381 | 86 | 182 | 7295 | 32.09% | 1.17% | 0.23237 |
| temporal | 202 | OMISSION | 1825 | 22 | 97 | 1803 | 18.49% | 1.21% | 0.12016 |
| temporal | 202 | INSERTION | 560 | 7 | 74 | 553 | 8.64% | 1.25% | 0.03445 |

Nhánh AP advancement so old M2: {'pointwise': False, 'temporal': False}. Nhánh precision/recall chưa đánh giá; không claim pass/fail mọi nhánh.

Không dùng PER giảm hoặc cờ M2 để tuyên bố MDD tốt hơn. Hard edit xuất score0/1, nhiều event hòa điểm, policy có hash tie break; đây có thể hạn chế xếp hạng/chọn cảnh báo. Chưa có ablation đủ để quy toàn bộ recall thấp cho adapter, recognizer hoặc dữ liệu. Phần định vị/uncertainty thuộc M3. 837 source S cùng base/deviation trong M0 không trở thành realized gold mới.

## Kiểm chứng và artifacts

- 16 run ×32 epoch, hai smoke20updates, checkpoint reload, full topology/held coverage; 192 policy replay checks; independent edit-distance PER và evaluator counts/classes. 2061 historical hash khớp, giữ code/manifest/checkpoint cũ bất biến.
- 5 tests pass: shared initialization, padding invariant sau context khác identity, CTC finite gradient, optimizer/scheduler/RNG restart, same-prompt grouping/excluded support. Tests.log là kết quả mới, không trộn với 20 historical tests.
- Waveform smoke 38 target/39gap, CPU decode bằng cache ở câu đã kiểm; không claim bitwise toàn corpus. Checkpoint metadata ghi ContextHead/kernel và weight decay thật, scaler disabled/fp32, optimizer/scheduler/RNG atomic mỗi epoch. Resume CPU optimizer/RNG được test; chưa kill/restart campaign ở mọi crash window.
- Wall 16 run gồm cal/export: 496.93s; peak PyTorch allocated 114.62MiB, khác tổng driver VRAM. Training budget3600s, cache extraction lịch sử tính riêng từ M1. Không claim speed superiority, không sweep tiếp hoặc promote.
- Environment.json ghi machine/package/params/branch; PREREGISTRATION.md ghi trước training. DIAGNOSTICS.json có PER từng speaker/phone, confusion/deletion/spurious insertion, nhóm nguyên âm/âm tắc và gap; PAIRED_SPEAKERS.json giữ đủ 12 speaker, không bỏ nhóm recall0.
- ACOUSTIC_CONTROLS.json giữ support/coverage/speaker/interval; completion.json tách M2 evidence khỏi product/M3–M6. selected và epoch8 có đủ realized predictions, policies, metrics, posteriors. Model weights/cache/dataset không vào Git.

## Tái lập

```sh
.venv/bin/python -m research.realized_phone_pipeline.m2_retry --mode smoke
.venv/bin/python -m research.realized_phone_pipeline.m2_retry --mode train
# Khi đã có run, chỉ --resume để kiểm/tiếp tục, không overwrite:
.venv/bin/python -m research.realized_phone_pipeline.m2_retry --mode train --resume
.venv/bin/python -m research.realized_phone_pipeline.m2_retry_report
.venv/bin/python -m research.realized_phone_pipeline.m2_retry_writeup
.venv/bin/python -m research.realized_phone_pipeline.m2_retry_infer --help
```

M2 là mốc nghiên cứu recognizer, không phải nghiệm thu sản phẩm; M3–M6 và goal tổng thể chưa hoàn thành. Không bắt đầu training/M3 mới chỉ vì sửa cờ milestone.
