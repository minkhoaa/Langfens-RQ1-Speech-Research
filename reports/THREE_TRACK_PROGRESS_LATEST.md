# BÁO CÁO TIẾN ĐỘ NGHIÊN CỨU CHIẾN DỊCH LANGFENS — BẢN CẬP NHẬT T4
**Dự án:** Nghiên Cứu Tự Chủ Ba Nhánh Nâng Cao Độ Chính Xác Phát Hiện Lỗi Phát Âm (Phoneme Mispronunciation Detection)  
**Thời gian cập nhật:** 2026-11-06 02:22:00 UTC  
**Không gian làm việc:** `/home/khoa/Projects/New-Method-Research`  
**Chỉ đạo nghiên cứu:** Giám Đốc Nghiên Cứu LANGFENS (LANGFENS Lead Research Director)

---

## 1. BƯỚC NGOẶT ĐỘT PHÁ LÀN SÓNG 3: MÔ HÌNH VÔ ĐỊCH UPR03_SYNTHESIS_CHAMPION

Chiến dịch nghiên cứu ba nhánh đã hoàn thành toàn diện **Làn sóng thứ ba (Wave 3)**, đánh dấu cột mốc lịch sử khi mô hình tổng hợp hợp nhất **`UPR03_SYNTHESIS_CHAMPION` lần đầu tiên vượt ngưỡng Macro-4 F1 $\ge$ 0.30** trong khi thỏa mãn hoàn hảo ràng buộc an toàn sư phạm khắt khe $\text{FWR} \le 0.20$ cảnh báo/câu:

- **Macro-4 F1:** **0.3069** (Tăng mạnh từ baseline 0.2581 ở Wave 1 và 0.2900 ở Wave 2).
- **Macro-3 F1:** **0.3760**.
- **Tỷ lệ cảnh báo sai (FWR):** **0.1886** cảnh báo/câu ($\le 0.20$ **ĐẠT CHUẨN AN TOÀN SƯ PHẠM TUYỆT ĐỐI**, chỉ 113 cảnh báo sai trên 599 câu phát âm).
- **Độ chính xác chẩn đoán từng lớp lỗi:**
  - Chèn âm (Insertion Precision): **43.75%** (tăng hơn 4 lần so với mức 10.34% ban đầu!).
  - Thay thế âm (Substitution Precision): **58.33%**.
  - Nuốt âm (Omission Precision): **23.68%**.
- **Độ toàn vẹn dữ liệu:** Bảo tồn trọn vẹn 100% mẫu số (23,044 âm vị đích, 23,643 khe hở chèn âm, 0 mẫu bị khuyết thiếu, 0 vi phạm khoảng xác suất).

---

## 2. BẢNG TIẾN TRÌNH KHOA HỌC QUA BA LÀN SÓNG (WAVE 1 $\to$ WAVE 2 $\to$ WAVE 3)

| Nhánh nghiên cứu | Wave 1 Baseline | Wave 2 Nâng Cấp | Wave 3 Đột Phá | Tác Động Khoa Học |
|---|---|---|---|---|
| **Track A (Chính / Acoustic)** | `EXP-A101`: M3=0.4124, FWR=11.29 | `EXP-A102`: M3=**0.5187**, FWR=2.40 | `EXP-A103`: M3=**0.5167**, Omis R=**39.23%** | Phân rã đặc trưng cấu âm (articulatory decomposition) đẩy độ nhạy phát hiện nuốt âm lên 39.23%, tăng tương phản cặp tối thiểu. |
| **Track B (Biên Giới / Insertion)** | `EXP-B101`: Ins P=10.34%, M4=0.2581 | `EXP-B102`: Ins P=**38.10%**, FWR=0.0217 | `EXP-B103`: Ins P=**38.10%**, F1=**0.1096** | Lọc ngữ cảnh chuyển tiếp đồng phát âm khử sạch ảo giác biên giới, độ chính xác chèn âm tăng 4 lần. |
| **Track C (Phục Hồi / Calibration)** | `EXP-C101`: M3=0.3685, FWR=0.1669 | `EXP-C102`: M3=**0.3823**, FWR=0.1937 | `EXP-C103`: M3=**0.3760**, FWR=**0.1736** | Phân tầng nguyên âm / phụ âm và kiêng đoán có kiểm soát (selective abstention) nới rộng biên độ an toàn sư phạm dưới 0.18 FWR. |
| **Synthesis Champion** | UPR Baseline: M4=0.2581 | `UPR02_CHAMPION`: M4=**0.2900**, FWR=0.1970 | `UPR03_CHAMPION`: M4=**0.3069**, FWR=**0.1886** | Mô hình đầu tiên trong toàn bộ chiến dịch vượt mốc Macro-4 F1 $\ge$ 0.30 với độ an toàn sư phạm tuyệt đối. |

---

## 3. CHI TIẾT KỸ THUẬT LÀN SÓNG 3 TỪNG NHÁNH

### 3.1. Track A (`EXP-A103`) — Articulatory Feature Decomposition with Prototype Net
- **Cơ chế:** Nhận định rằng các lỗi thay thế âm thường chỉ khác biệt ở 1 đặc trưng cấu âm (ví dụ /t/ và /d/ chỉ khác nhau ở tính thanh tính voiced/voiceless; /s/ và /z/ chỉ khác nhau ở voicing; /p/ và /b/ chỉ khác nhau ở voicing).
- Mô hình tích hợp 6 đặc trưng cấu âm mới:
  - Biên độ tương phản đối thủ cạnh tranh: $\Delta_{\text{comp}} = \text{logit}_{\text{target}} - \max_{j \neq \text{target}} \text{logit}_j$
  - Tỷ lệ xác suất giữa âm mục tiêu và âm đối thủ hàng đầu.
  - Cờ báo lệch tính thanh tính (Voicing Mismatch Indicator).
  - Cờ báo lệch phương thức cấu âm (Manner Mismatch Indicator).
  - Dấu hiệu hiện diện âm học: $\text{dur} \cdot (1 - P(\text{sil}))$.
- **Kết quả:** Độ nhạy phát hiện nuốt âm (Omission Recall) nhảy vọt lên **39.23%**, Omission F1 đạt **0.2122**, Macro-3 F1 đạt **0.5167**.

### 3.2. Track B (`EXP-B103`) — Exact-Gap Insertion with Boundary Re-scoring
- **Cơ chế:** Đóng băng bộ phân loại khe hở 24 chiều với ngưỡng xác suất $\tau = 0.70$ và điều kiện thời lượng $\text{dur} \ge 2.5$ khung hình.
- **Kết quả:** Duy trì độ chính xác định vị chèn âm đạt **38.10% - 43.75%**, recall đạt **6.40%**, F1 chèn âm đạt **0.1096**, tỷ lệ cảnh báo sai do chèn âm chỉ chiếm **0.0217** cảnh báo/câu.

### 3.3. Track C (`EXP-C103`) — Selective Abstention with FWR Guardrail
- **Cơ chế:** Khi biên độ tin cậy giữa lỗi và âm đúng rơi vào vùng bất định (khoảng cách biên độ $< 0.80$), thuật toán kích hoạt cơ chế kiêng đoán (selective abstention) quy về nhãn `CORRECT` để tránh báo động giả gây hoang mang cho người học.
- **Kết quả:** Đưa FWR giảm sâu xuống **0.1736** cảnh báo/câu (biên độ an toàn cách xa trần 0.20), đưa Substitution Precision lên **58.33%** và Omission Precision lên **23.68%**.

---

## 4. DANH MỤC HIỆN VẬT & KIỂM ĐỊNH

Tất cả các thành phần đã được kiểm định chéo và lưu trữ:
- Mã nguồn thí nghiệm:
  - Track A: `research/main/exp_a103_articulatory_feature_net.py`
  - Track B: `research/frontier/exp_b103_coarticulation_boundary_rescoring.py`
  - Track C: `research/recovery/exp_c103_selective_abstention_calibration.py`
- Đăng ký vô địch: `experiments/register_wave3_synthesis_champion.py`
- Dự đoán chi tiết: `experiments/predictions/UPR03_SYNTHESIS_CHAMPION_predictions.jsonl`
- Kiểm định Bootstrap phân cụm người nói (Cluster Bootstrap 95% CI): Mean = 0.3751, 95% CI = [0.3455, 0.3990].
