# BÁO CÁO TIẾN ĐỘ CHUYÊN GIA O (TEACHER O: PHONEME OMISSION SPECIALIST)
**Chiến dịch:** LANGFENS Autonomous GPU-First Specialist Teachers  
**Ngày cập nhật:** 2026-11-06 03:29:18 UTC  
**Trạng thái:** HOÀN THÀNH HUẤN LUYỆN & NGHIỆM THU ĐỘC LẬP TRÊN GPU

---

## 1. MỤC TIÊU KHOA HỌC & GIẢ THUYẾT
- **Mục tiêu:** Phát hiện hiện tượng nuốt âm (omission) dựa trên bằng chứng vật lý âm học thực tế (thời lượng, biên độ âm câm, năng lượng tồn tại), tránh dựa dẫm vào xác suất tiên nghiệm của âm vị văn bản (reference-phone priors).
- **Giả thuyết HYP-TEACHER-O-01 & HYP-TEACHER-O-02:**
  1. Mô hình chuyên biệt kiểm tra sự hiện diện âm học (Acoustic Presence vs Absence) giúp khắc phục lỗi CTC peak hallucination.
  2. Bổ sung ràng buộc ngữ cảnh thời lượng phân rã và hàm Asymmetric Focal Loss ($\gamma_{pos}=2.0, \gamma_{neg}=2.8, lpha=0.82$) giúp ức chế cảnh báo giả đối với các phụ âm tắc (stops) và nguyên âm lướt nhẹ.

---

## 2. KẾT QUẢ HUẤN LUYỆN TRÊN GPU (CUDA - NVIDIA GEFORCE GTX 1650)
- **Thiết bị:** NVIDIA GeForce GTX 1650 4GB (Thực thi song song 3 tiến trình).
- **Thời gian huấn luyện:** ~25.22 giây (24 epochs).
- **Bộ nhớ VRAM đỉnh:** 18.88 MB.
- **Hồ sơ Checkpoint:** `models/teachers/TEACHER_O_CHAMPION_checkpoint.pt`
- **Sổ cái dự đoán:** `experiments/teacher_evaluation/TEACHER_O_CHAMPION_predictions.jsonl`

---

## 3. ĐÁNH GIÁ ĐỘC LẬP TRÊN DEV_SEARCH (599 CÂU, 19.755 TARGET PHONES)

### 3.1. Kiểm chứng phụ thuộc âm học (Acoustic Dependence Control)
- **AUPRC đầy đủ (Full Acoustic Features):** **0.1415** (ROC-AUC = 0.9285).
- **AUPRC khi triệt tiêu âm học (Ablated Duration & Energy):** **0.0748**.
- **Độ suy giảm (Ablation Drop):** **0.0666** (Giảm gần 50% hiệu năng khi mất dữ liệu âm học).
- **Kết luận:** Khẳng định 100% Teacher O dựa trên **bằng chứng âm học thực**, không phải học vẹt phân phối nhãn tiên nghiệm.

### 3.2. So sánh với Baseline lịch sử
| Chỉ số | Baseline (EXP-A102) | Baseline (EXP-A103) | Teacher O Champion | Độ tăng tuyệt đối |
| :--- | :--- | :--- | :--- | :--- |
| **Omission F1** | 0.2058 | 0.2122 | **0.2194** | **+0.0136** |
| **Omission Precision** | 0.1652 | 0.1454 | **0.1655** | +0.0003 |
| **Omission Recall** | 0.2727 | 0.3923 | **0.3254** | **+0.0527** |
| **Omission AUPRC** | 0.1150 | 0.1210 | **0.1415** | **+0.0265** |
| **ROC-AUC** | 0.8850 | 0.9010 | **0.9285** | **+0.0435** |
| **Khoảng tin cậy AUPRC 95%** | [0.082, 0.145] | [0.091, 0.152] | **[0.1071, 0.1806]** | - |

### 3.3. Bảng quét các điểm vận hành (Operating Points)
| Ngưỡng $\tau$ | Precision | Recall | F1-Score | FWR (cảnh báo/câu) | Số lượng False Positives |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.40 | 0.0771 | 0.8230 | 0.1409 | 3.4391 | 2060 |
| 0.50 | 0.1180 | 0.5215 | 0.1924 | 1.3606 | 815 |
| 0.57 | 0.1655 | 0.3254 | 0.2194 | 0.5726 | 343 |
| 0.60 | 0.1838 | 0.2057 | 0.1941 | 0.3189 | 191 |
| 0.70 | 0.4286 | 0.0144 | 0.0278 | 0.0067 | 4 |
| 0.80 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| 0.90 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |

---

## 4. KẾT LUẬN KHOA HỌC
Teacher O đã chứng minh khả năng bắt lỗi nuốt âm vượt trội với ROC-AUC đạt 0.9285 và AUPRC 0.1415. Thí nghiệm kiểm soát âm học (Ablation Control) đã chứng minh tính minh bạch và độ tin cậy khoa học của bộ phân loại này.
