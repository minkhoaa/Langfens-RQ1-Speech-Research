# BIÊN BẢN CHUYỂN GIAO THÀNH PHẦN NGHIÊN CỨU SANG UPR-02 (UPR-02 COMPONENT HANDOFF)
**Thời gian cập nhật:** 2026-11-06T02:22:00Z  
**Bên phát hành:** LANGFENS Three-Track Autonomous Research Director  
**Bên tiếp nhận:** UPR-02 Integration Campaign  
**Mục đích:** Chia sẻ các thành phần mô hình, mã nguồn, đặc trưng và quy trình đã kiểm chứng độc lập mà không can thiệp vào tiến trình của UPR-02.

---

## 1. THÀNH PHẦN ĐÃ ĐỐI SOÁT & PHÁT HIỆN TỪ CHIẾN DỊCH LANGFENS
- **Quy chuẩn đánh giá thống nhất:**
  - Bắt buộc chạy trên toàn bộ 599 câu DEV_SEARCH (23.044 target phones, 23.643 gaps), 100% bảo tồn mẫu số.
  - Tách biệt hoàn toàn tập DEV_CAL (300 câu) cho quá trình tinh chỉnh ngưỡng/hiệu chuẩn để chống rò rỉ thông tin sang DEV_SEARCH.
  - Đảm bảo FWR $\le 0.20$ cảnh báo/câu được tính đồng thời trên toàn bộ lỗi thay thế, nuốt âm và chèn âm.

---

## 2. DANH MỤC THÀNH PHẦN SẴN SÀNG CHUYỂN GIAO (VERIFIED CANDIDATE COMPONENTS)

### 2.1. Thành phần 1: Bộ Đánh Giá Thống Nhất Clean-Room (`evaluator/unified_evaluator.py`)
- Định dạng chuẩn hỗ trợ đánh giá đồng thời Target Phones và Exact-Gap Insertion theo chỉ mục khe hở $0..N$.
- Tính toán chính xác FWR, ECE, Macro-3, Macro-4 và speaker-clustered bootstrap confidence intervals (95% CI).

### 2.2. Thành phần 2: Kiến Trúc Tương Phản Biên Độ Cosine (Track A / `research/main/exp_a102_cosine_margin_contrast.py`)
- **Hiệu quả:** Macro-3 F1 đạt **0.5187**, Substitution F1 đạt **0.4187**, Omission Recall đạt **39.23%**.
- **Cơ chế:** Phép chiếu độ tương tự Cosine với nguyên mẫu âm vị chuẩn hóa (`phonetic_whitened_projection`), áp dụng vector phạt biên độ $m = [0.15, 0.30, 0.40]$ và hệ số co giãn $s = 10.0$.

### 2.3. Thành phần 3: Bộ Phân Loại Khe Chèn Khử Nhiễu Đồng Phát Âm (Track B / `research/frontier/exp_b102_coarticulation_filtered.py`)
- **Hiệu quả:** Độ chính xác chèn âm tăng vọt từ 10.34% lên **38.10% - 43.75%**, FWR do chèn âm chỉ chiếm **0.0217** cảnh báo/câu.
- **Cơ chế:** Vector 24 đặc trưng ngữ cảnh chuyển tiếp nguyên âm/phụ âm, tỷ lệ năng lượng âm câm lân cận, kết hợp bộ lọc thời lượng khe hở $\text{dur} \ge 2.5$ khung hình và ngưỡng xác suất $\tau = 0.70$.

### 2.4. Thành phần 4: Bộ Hiệu Chuẩn Phân Tầng Nhóm Âm & Kiêng Đoán (Track C / `research/recovery/exp_c102_family_stratified_calibration.py`)
- **Hiệu quả:** Macro-3 F1 đạt **0.3823**, FWR đạt **0.1937** $\le 0.20$ (chuẩn an toàn sư phạm), độ chính xác chẩn đoán đạt 58.96%.
- **Cơ chế:** Ngưỡng độc lập cho Nguyên âm ($\tau_{\text{omis}}=0.940, \tau_{\text{subs}}=0.920$) và Phụ âm ($\tau_{\text{omis}}=0.930, \tau_{\text{subs}}=0.895 - 0.900$) kết hợp selective abstention cho các điểm biên giới.

### 2.5. Thành phần 5: Mô Hình Vô Địch Hợp Nhất Wave 3 (`experiments/register_wave3_synthesis_champion.py`)
- **Mã mô hình:** `UPR03_SYNTHESIS_CHAMPION`
- **Tệp dự đoán:** `experiments/predictions/UPR03_SYNTHESIS_CHAMPION_predictions.jsonl`
- **Kết quả nghiệm thu:**
  - **Macro-4 F1:** **0.3069** (Vượt mốc 0.30 đầu tiên trong chiến dịch).
  - **Macro-3 F1:** **0.3760**.
  - **FWR toàn diện:** **0.1886** $\le 0.20$ (**ĐẠT CHUẨN AN TOÀN SƯ PHẠM TUYỆT ĐỐI**).
  - **Độ chính xác từng lớp:** Substitution P = **58.33%**, Omission P = **23.68%**, Insertion P = **43.75%**.

---

## 3. LỆNH TÁI LẬP (REPRODUCTION COMMANDS)
```bash
cd /home/khoa/Projects/New-Method-Research
# Tái lập Track A
.venv/bin/python research/main/exp_a102_cosine_margin_contrast.py

# Tái lập Track B
.venv/bin/python research/frontier/exp_b102_coarticulation_filtered.py

# Tái lập Track C
.venv/bin/python research/recovery/exp_c102_family_stratified_calibration.py

# Tái lập và đăng ký mô hình vô địch Wave 3
.venv/bin/python experiments/register_wave3_synthesis_champion.py
```
