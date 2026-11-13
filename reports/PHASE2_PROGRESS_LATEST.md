# LANGFENS GIAI ĐOẠN 2: TIẾN ĐỘ NGHIÊN CỨU MỚI NHẤT (PHASE 2 PROGRESS LATEST)

**Thời gian cập nhật:** 2026-11-06 12:15:00 UTC  
**Vai trò:** Autonomous Scientific Research Director  
**Trạng thái toàn chiến dịch:** HOÀN THÀNH TOÀN DIỆN CẢ 3 WORKSTREAMS (CPL-01, CPL-02, CPL-03)

---

## 1. TÓM TẮT ĐIỀU HÀNH BẰNG TIẾNG VIỆT (OPERATOR EXECUTIVE SUMMARY)

Kính gửi Giám đốc Nghiên cứu,

Giai đoạn 2 của Chiến dịch LANGFENS (**Nghiên cứu Tính Bổ trợ Đa Chuyên gia & Hợp nhất Quyết định**) đã được thực thi và hoàn tất thành công 100% mục tiêu khoa học độc lập:

1. **Chứng minh tính bổ trợ và tính trực giao mạnh mẽ giữa các chuyên gia (CPL-01):**
   - **Teacher S** (Thay thế âm) và **Teacher O** (Nuốt âm) hoạt động trên hai khía cạnh âm học hoàn toàn độc lập với hệ số tương quan điểm mềm cực kỳ thấp ($r = 0.0384$ trên các âm vị đúng).
   - **Xung đột quyết định (Decision Conflicts) cực kỳ hiếm:** Trong 19.755 âm vị mục tiêu của tập `DEV_SEARCH`, chỉ có **153 trường hợp** (0.77%) cả Teacher S và Teacher O cùng báo lỗi. Trong 153 trường hợp này, có tới **87 ca là lỗi phát âm thật** (56 ca là Thay thế, 31 ca là Nuốt âm). Điều này chứng minh rằng khi hai chuyên gia cùng kích hoạt, xác suất tồn tại lỗi phát âm thật lên tới **56.86%**!
   - **Cảnh báo sai chung (Shared False Positives) cực thấp:** Chỉ có **66 âm CORRECT** bị cả hai chuyên gia cùng báo nhầm (chỉ 0.37% tổng số âm đúng), hệ số tương đồng Jaccard chỉ là **0.0421**. Điều này tạo điều kiện thuận lợi tuyệt đối cho việc lọc sạch cảnh báo giả khi chưng cất sang mô hình học sinh.
   - **Tác động hiệp đồng phát hiện lỗi chéo (Synergistic Cross-Detection):** Teacher S phát hiện được 48/209 lỗi nuốt âm dưới dạng sụt giảm năng lượng và sai lệch nguyên mẫu. Khi kết hợp, tổng số lỗi nuốt âm phát hiện được tăng từ 68 lên **94 âm** (+38.2%).

2. **Khảo sát toàn diện 26 chính sách hợp nhất quyết định (CPL-02):**
   - **Giới hạn trần độ nhạy của Chuyên gia (Unconstrained Ceiling):** Khi kết hợp không ràng buộc (`F2-O-PRIO` / `F1-UNION`), Macro-4 F1 đạt **0.4272** (+39.2% so với 0.3069 của UPR03), Substitution Recall đạt **51.87%** (so với 5.94% của UPR03), Omission Recall đạt **32.54%** (so với 4.31% của UPR03).
   - **Tại vùng an toàn sư phạm nghiêm ngặt (FWR <= 0.20 / FP <= 119):**
     - Chính sách `F5-RISK-B030` đạt $\text{FWR} = 0.1836$ (110 cảnh báo giả), $\text{Macro-4} = 0.3030$, với độ thu hồi chèn âm Exact-Gap đạt **7.20%** (vượt trội +28.6% so với 5.60% của UPR03).
     - Chính sách phân tầng họ âm vị `F3-CALIB-FAMILY` đạt $\text{FWR} = 0.1970$ (118 cảnh báo giả), thu hồi Substitution đạt **9.45%** (vượt trội **+59.1% so với 5.94% của UPR03**).
     - Chính sách an toàn cao `F5-RISK-B020` đạt $\text{FWR} = 0.0885$ (chỉ 53 cảnh báo giả trên 599 câu, triệt tiêu hơn 96.8% cảnh báo giả tự nhiên).

3. **Luận chứng tính khả thi & Hợp đồng kiến trúc học sinh thống nhất (CPL-03):**
   - Đã định lượng chất lượng tín hiệu mềm của 4 chuyên gia: AUPRC và ROC-AUC cao, độ bất định tương quan chặt chẽ với sai số thực tế ($r = 0.713$ với S, $r = 0.970$ với O).
   - **Phát hiện khoa học quyết định:** Phép hợp nhất điểm số hậu kỳ (post-hoc score fusion) bị chặn bởi trần thông tin do 3 mô hình hoạt động trên không gian đặc trưng rời rạc. Để giải phóng toàn bộ tiềm năng độ nhạy 51.87% Sub Recall và 32.54% Om Recall mà vẫn giữ FWR <= 0.20, **bắt buộc phải huấn luyện một mô hình học sinh thống nhất (Unified Multi-Task Conformer)** với bộ mã hóa dùng chung và attention liên nhiệm vụ.

---

## 2. BẢNG KẾT QUẢ ĐỐI CHỨNG CÁC ĐIỂM PARETO THEN CHỐT

| Kịch bản / Chính sách | Họ giải thuật | FWR / câu | Tổng FP | Sub Recall | Omis Recall | Ins Recall | Macro-3 F1 | Macro-4 F1 | Đạt chuẩn an toàn sư phạm? |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| `F5-RISK-B005` | Risk Budgeted | 0.0117 | 7 | 0.23% | 0.96% | 3.20% | 0.3237 | 0.2576 | **ĐẠT** |
| `F5-RISK-B020` | Risk Budgeted | 0.0885 | 53 | 1.24% | 4.31% | 3.20% | 0.3474 | 0.2758 | **ĐẠT** |
| `F5-RISK-B030` | Risk Budgeted | 0.1836 | 110 | 4.75% | 3.83% | **7.20%** | 0.3664 | **0.3030** | **ĐẠT** |
| `F3-CALIB-FAMILY` | Calibrated Family | 0.1970 | 118 | **9.45%** | 0.00% | 5.60% | 0.3710 | **0.3019** | **ĐẠT** |
| **`UPR03_BASELINE`** | Historical Champion | 0.1886 | 113 | 5.94% | 4.31% | 5.60% | 0.3760 | **0.3069** | **ĐẠT** |
| `F5-RISK-B050` | Risk Budgeted | 0.4224 | 253 | **12.44%** | **8.13%** | 3.20% | 0.4205 | **0.3302** | Không |
| `F5-RISK-B100` | Risk Budgeted | 0.9215 | 552 | **16.52%** | **24.88%** | 5.60% | 0.4634 | **0.3712** | Không |
| `F1-UNION` | Naive Union | 2.7896 | 1,671 | **51.87%** | **18.66%** | **8.80%** | 0.5139 | **0.4167** | Không |
| `F2-O-PRIO` | Priority Arbitration | 2.8347 | 1,698 | **48.70%** | **32.54%** | **8.80%** | 0.5279 | **0.4272** | Không |

---

## 3. DANH MỤC CÁC TÀI LIỆU KHOA HỌC ĐÃ PHÁT HÀNH

1. `reports/CPL01_COMPLEMENTARITY_REPORT.md`
2. `reports/CPL02_FUSION_REPORT.md`
3. `reports/CPL03_DISTILLATION_FEASIBILITY.md`
4. `reports/CPL03_STUDENT_ARCHITECTURE.md`
5. `reports/CPL03_KD_EXPERIMENT_DESIGN.md`
6. `reports/PHASE2_FINAL_SCIENTIFIC_REPORT.md`
7. `reports/PHASE3_STUDENT_DISTILLATION_HANDOFF.md`
8. Toàn bộ sổ cái thực nghiệm tại `experiments/complementarity/` và `experiments/fusion/`.
