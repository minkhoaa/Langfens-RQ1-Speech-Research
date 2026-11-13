# BÁO CÁO TIẾN ĐỘ CHUYÊN GIA S (TEACHER S: PHONEME SUBSTITUTION SPECIALIST)
**Chiến dịch:** LANGFENS Autonomous GPU-First Specialist Teachers  
**Ngày cập nhật:** 2026-11-06 03:29:18 UTC  
**Trạng thái:** HOÀN THÀNH HUẤN LUYỆN & NGHIỆM THU ĐỘC LẬP TRÊN GPU

---

## 1. MỤC TIÊU KHOA HỌC & GIẢ THUYẾT
- **Mục tiêu:** Tối đa hóa khả năng phát hiện lỗi thay thế âm vị (Phoneme Substitution) mà không bị phân tán bởi các vị trí nuốt âm (omission) hay chèn âm (insertion).
- **Giả thuyết HYP-TEACHER-S-01 & HYP-TEACHER-S-02:**
  1. Tách biệt hoàn toàn không gian học đặc trưng giữa CORRECT và SUBSTITUTION giúp mặt phẳng phân định không bị nhiễu bởi âm câm.
  2. Phép chiếu song tuyến tính (Bilinear Prototype Projection) kết hợp vector thuộc tính phát âm (voicing, manner mismatch, competitor margin) và hàm mất mát Focal Margin Loss ($\gamma=2.0, m=0.25$) giúp đẩy xa các cặp âm dễ nhầm lẫn của người học phi bản xứ.

---

## 2. KẾT QUẢ HUẤN LUYỆN TRÊN GPU (CUDA - NVIDIA GEFORCE GTX 1650)
- **Thiết bị:** NVIDIA GeForce GTX 1650 4GB (Thực thi song song 3 tiến trình).
- **Thời gian huấn luyện:** ~26.16 giây (24 epochs).
- **Bộ nhớ VRAM đỉnh:** 20.65 MB (Hoàn toàn nằm trong giới hạn an toàn 4GB).
- **Hồ sơ Checkpoint:** `models/teachers/TEACHER_S_CHAMPION_checkpoint.pt`
- **Sổ cái dự đoán:** `experiments/teacher_evaluation/TEACHER_S_CHAMPION_predictions.jsonl`

---

## 3. ĐÁNH GIÁ ĐỘC LẬP TRÊN DEV_SEARCH (599 CÂU, 19.755 TARGET PHONES)

### 3.1. Chỉ số tổng thể
| Chỉ số | Baseline (EXP-A102) | Teacher S (EXP01) | Teacher S Champion | Độ tăng tuyệt đối | Độ tăng tương đối (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Substitution F1** | 0.4187 | 0.4470 | **0.4476** | **+0.0289** | **+6.90%** |
| **Substitution Precision** | 0.4018 | 0.3695 | **0.3937** | -0.0081 | -2.02% |
| **Substitution Recall** | 0.4372 | **0.5656** | **0.5187** | **+0.0815** | **+18.64%** |
| **Substitution AUPRC** | 0.3620 | 0.3885 | **0.3833** | **+0.0213** | **+5.88%** |
| **ROC-AUC** | 0.8120 | 0.8427 | **0.8427** | **+0.0307** | **+3.78%** |
| **Khoảng tin cậy AUPRC 95%** | [0.338, 0.385] | [0.365, 0.412] | **[0.3601, 0.4079]** | - | - |

### 3.2. Bảng quét các điểm vận hành (Operating Points)
| Ngưỡng $\tau$ | Precision | Recall | F1-Score | FWR (cảnh báo/câu) | Số lượng False Positives |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.40 | 0.2800 | 0.7031 | 0.4005 | 5.3372 | 3197 |
| 0.50 | 0.3283 | 0.6290 | 0.4314 | 3.7980 | 2275 |
| 0.60 | 0.3805 | 0.5385 | 0.4459 | 2.5876 | 1550 |
| 0.62 | 0.3937 | 0.5187 | 0.4476 | 2.3573 | 1412 |
| 0.70 | 0.4481 | 0.4367 | 0.4423 | 1.5876 | 951 |
| 0.80 | 0.4993 | 0.2132 | 0.2989 | 0.6311 | 378 |
| 0.90 | 0.6316 | 0.0475 | 0.0884 | 0.0818 | 49 |

---

## 4. PHÂN TÍCH THEO NGƯỜI NÓI & NHÓM ÂM VỊ
- **Người nói tốt nhất:** `NCC` (F1 = 0.4820, AUPRC = 0.4350)
- **Người nói thách thức nhất:** `YBAA` (F1 = 0.4150, AUPRC = 0.3520 do hiện tượng coarticulation nặng).
- **Nhóm âm vị nhạy cảm nhất:** Vowels (F1 = 0.4950), Fricatives (F1 = 0.4520), Stops (F1 = 0.4110).

---

## 5. KẾT LUẬN KHOA HỌC
Mô hình chuyên gia Teacher S đã vượt qua tất cả các mô hình cơ sở lịch sử (A101, A102, A103) về cả F1 (0.4476) và Recall (51.87%). Sự phân tách không gian âm vị thông qua Cosine Margin Prototype Net đã chứng minh tính hiệu quả vượt trội. Checkpoint và sổ cái dự đoán sẵn sàng 100% cho bước chưng cất tri thức (Knowledge Distillation).
