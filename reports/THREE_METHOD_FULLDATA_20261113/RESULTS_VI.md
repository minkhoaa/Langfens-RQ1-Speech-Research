# Kết quả full-FIT và đường cong so với CTC / SM13

Đã chạy thật trên toàn bộ **628 câu FIT đủ nhãn chuỗi**, CAL 450, HELD 450, fold 0; CTC/CROTTC/SM13 có hai seed 101/202. CROTTC compact tăng **0.61 điểm phần trăm F1** so với CTC M1 ở epoch cuối, nhưng SM13 vẫn đạt F1 cao hơn và ít cảnh báo sai hơn. PER-MDD không cải thiện macro F1 khi tăng kho. X-MDD chưa có đường cong full-data.

## Kết quả cùng 450 câu HELD

Các điểm cố định ở epoch 8 CTC, epoch 16 CROTTC, epoch 12 target/5 gap SM13, kho 628 PER; không chọn epoch theo HELD. CTC/CROTTC dùng greedy phone sequence và hard edit alignment; SM13 dùng argmax target, insertion ≥0,5. Bảng này chưa áp chính sách giới hạn cảnh báo.

| Cấu hình cùng FIT 628 | S/O/I macro F1 (%) | Cảnh báo sai/câu | Khoảng F1 hai seed (%) |
|---|---:|---:|---:|
| CTC baseline | 13.22 | 17.13 | 13.17–13.26 |
| CROTTC compact | 13.83 | 16.99 | 13.72–13.94 |
| PER-MDD | 6.46 | 17.58 | — |
| SM13 | 26.33 | 3.99 | 25.54–27.12 |

PER chỉ có một kho cố định tại mỗi kích thước; không có khoảng biến thiên giữa seed. Hai seed và một fold không đủ để suy rộng thống kê.

| Cấu hình | Recall S | Recall O | Recall I |
|---|---:|---:|---:|
| CTC baseline | 27.62% | 73.95% | 8.21% |
| CROTTC compact | 59.04% | 31.51% | 13.81% |
| PER-MDD | 10.97% | 0.56% | 76.87% |
| SM13 | 38.81% | 50.14% | 7.84% |

Không cấu hình mặc định nào đồng thời đạt recall từng S/O/I ≥80% và FWR ≤0,20/câu. F1 macro không phải mục tiêu recall 80%.

## Đường cong đo được

![Đường cong checkpoint](epoch_curves.png)

Đường liền là trung bình hai seed, vùng mờ là min–max hai seed, không phải khoảng tin cậy. Trục epoch tương ứng các mục tiêu và số cập nhật khác nhau, không biểu thị compute ngang nhau. HELD đã được xem như tập chẩn đoán; đây không phải xác nhận trên test niêm phong.

![Đường cong kích thước kho](pool_scaling.png)

PER giữ k=10 và threshold=0,7 ở cả bốn điểm, cùng 450 query HELD. F1 từ 9,45% ở kho 24 xuống 6,46% ở kho 628; FWR giảm từ 23,22 xuống 17,58/câu. Realized PER cũng giảm nhưng không kéo macro S/O/I F1 lên. Xem chi tiết tại [PER_CURVE.md](PER_CURVE.md).

Checkpoint CROTTC pilot 24 cũ, giữ nguyên lựa chọn bằng CAL 6 cũ và chỉ chạy inference lại trên cùng 450 HELD, đạt **9.13% F1, 26.25 cảnh báo sai/câu**. Lượt full-FIT cải thiện so với checkpoint đó; lượng dữ liệu, số cập nhật, lịch learning rate và tập CAL đều thay đổi, nên không thể quy riêng mức tăng cho dữ liệu. Không so điểm pilot 6 HELD cũ với điểm450 HELD mới.

## SM13 dùng 900 FIT và chính sách FWR

Replay lịch sử SM13 dùng đủ 900 FIT event-labeled có **34.09% F1, 3.99 cảnh báo sai/câu** trên cùng 450 HELD, trung bình hai seed. Đã xác minh hash các source artifacts rồi chấm lại raw probabilities bằng argmax/0,5. Đây chỉ là điểm cuối lịch sử; không có dự đoán từng epoch để vẽ thành đường cong. Khác lượng FIT và lịch tối ưu với SM13 instrumented 628, nên không dùng chênh lệch này làm ước lượng nhân quả của việc thêm 272 câu.

Với SM13 instrumented 628, áp quy tắc cũ chia ngân sách cảnh báo 80% target/20% gap, fit threshold **chỉ trên CAL 450** tại epoch 12 cố định: F1 HELD trung bình **9.33%**, FWR hai seed **0.173/0.207**. Fit FWR trên CAL không đảm bảo HELD đạt cùng giới hạn. Xem `sm13_cal_budget02.json`; không suy từ các đường cong argmax rằng SM13 không thể giảm cảnh báo.

## Phạm vi “toàn bộ dữ liệu” và giới hạn

FIT gốc có 900 câu, trong đó272 thiếu chuỗi realized-phone đầy đủ nên CTC/CROTTC hiện vẫn chỉ dùng628. CAL/HELD không đưa vào train; toàn bộ official test vẫn niêm phong. Chấm MDD trên đủ 450 HELD kể cả 221 câu không có exact sequence; realized PER chỉ dùng 229 câu HELD đủ nhãn (CAL 386). Vì vậy lượt này **chưa sửa lỗ hổng supervision** đã audit.

CROTTC ở đây là bản compact dùng frozen W2V2-H6 và decoder nhỏ, có OT/consistency/teacher distillation nhưng khác đáng kể recipe paper. SM13 instrumented dùng feature 13 layer với reference/alignment lịch sử từ annotation; lịch target/gap xen kẽ và clip norm 5 khác runner SM13 cũ. Không diễn giải kết quả thành thứ hạng của các paper hoặc sẵn sàng chẩn đoán acoustic độc lập.

X-MDD pilot vẫn sinh chữ thường thay vì CMU phones và có 0 effective GRPO updates. Ước lượng một lượt SFT 628 khoảng 71,7 phút chỉ là ngoại suy từ một khoảng cập nhật đã đo; chưa gồm đánh giá 900 câu hay GRPO. Không có số F1/đường cong full-data để kết luận về X-MDD. Xem [X_SCALING.md](X_SCALING.md).

## Kiểm chứng và tái lập

- 152 điểm role/seed/step được kiểm tra không trùng/thiếu; mỗi điểm đủ 450 câu, zero missing target/gap predictions.
- 92 prediction files có SHA-256 được kiểm tra lại trước khi dựng biểu đồ; toàn bộ metric qua EVAL-V3.1 strict typed / exact gap.
- Baseline CTC epoch cuối khớp toàn bộ 1.800 chuỗi dự đoán CAL/HELD từ hai seed của lượt lịch sử.
- CROTTC hai seed: 542.83s; controls hai seed: 183.79s; đều trong cap 1.800s riêng. Đây là wall time có checkpoint/inference, không phải benchmark tốc độ ngang compute. PER lượt uncached ban đầu 200,56s được ghi trong tool transcript; receipt gốc đã bị lượt cache-hit 21,29s ghi đè. `per_mdd/TIMING_PROVENANCE.json` ghi rõ hạn chế lưu giữ này; đây là thời gian cả pipeline, không riêng extraction.
- Checkpoint mỗi epoch có optimizer/scheduler/scaler/RNG, ghi atomic và reload kiểm tra. Lượt controls đầu lỗi import trước optimizer được giữ riêng; retry hoàn tất.
- 16 tests cơ chế OT/distillation, retrieval, topology, cảnh báo sai loại và phạm vi PER đã pass; kết quả test được ghi tại `TEST_RESULT.json`.

Dữ liệu biểu đồ: [curve_points.csv](curve_points.csv), [curve_points.json](curve_points.json). Biểu đồ vector: [epoch_curves.svg](epoch_curves.svg), [pool_scaling.svg](pool_scaling.svg). Môi trường và budget: `experiments/three_method_fulldata_20261113/ENVIRONMENT.json`, `scaling.json`, `STAGING_RECEIPT.json`.

```sh
.venv/bin/python -m research.three_method_scaling.stage
# Runner commands and individual receipts are in CROTTC_CURVE.md / PER_CURVE.md / X_SCALING.md.
.venv/bin/python -m research.three_method_scaling.calibrate_final
.venv/bin/python -m research.three_method_scaling.build_report
.venv/bin/python -m pytest -q research/three_method_scaling/test_crottc_curve.py research/three_method_scaling/test_evaluate.py
```
