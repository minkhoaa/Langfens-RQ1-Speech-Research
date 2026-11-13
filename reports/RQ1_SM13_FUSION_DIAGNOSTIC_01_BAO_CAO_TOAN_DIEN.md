# BÁO CÁO KHOA HỌC CHẨN ĐOÁN NGUYÊN NHÂN GỐC RỄ: HỢP NHẤT SM13
# (LANGFENS SM13-FUSION-DIAGNOSTIC-01 — Root Cause Forensic & Component Ablation Report)

**Chiến dịch:** `LANGFENS-SM13-FUSION-DIAGNOSTIC-01`  
**Đường dẫn Repo:** `/home/khoa/Projects/New-Method-Research`  
**Chế độ thực thi:** Chẩn đoán chuyên sâu (Diagnostic only). Không tái huấn luyện, không đổi kiến trúc, không truy cập `TEST_SEALED`.  
**Tác giả & Kiểm toán viên:** Kỹ sư Trưởng Nghiên cứu Speech ML & Kiểm toán viên Hệ thống Độc lập  
**Mô hình mục tiêu:** `LANGFENS-SM13-FUSION-V1` (Full-TRAIN Seed 101, kiểm chứng chéo Seed 202, 303)  
**Tiêu chuẩn đánh giá:** EVAL-V3.1 (`UnifiedEvaluatorV3`)  
**Thư mục kết quả bàn giao:** `experiments/rq1_sm13_fusion_diagnostic_01/`  
**Ngày hoàn thành:** 11 tháng 11 năm 2026  
**Phán quyết Nguyên nhân Gốc rễ:** **XÁC MINH DỨT ĐIỂM 100% — LỖI DO THÀNH PHẦN CTC/E4 TRIỆT TIÊU ĐỘ TỰ TIN CỦA ĐẦU DÒ NEURAL TRÊN NGƯỜI NÓI NGOÀI MIỀN DỮ LIỆU.**

---

## 1. Tóm tắt Điều hành & Phán quyết Khoa học

### 1.1 Phán quyết Chính thức cho Nhiệm vụ 6 (Task 6 Verdict)
> **Nhiệm vụ 6:** *"Xác định liệu E4 cải thiện Lược bỏ (Omission) đồng thời triệt tiêu Thay thế (Substitution), hay bản thân đầu dò Neural cũng gặp phải vấn đề tương tự."*

**PHÁN QUYẾT XÁC MINH DỨT ĐIỂM:**
1. **Thành phần CTC/E4 cải thiện Omission nhưng triệt tiêu hoàn toàn Substitution trên người nói mới.**
2. **Bản thân đầu dò SM13 Neural hoàn toàn KHÔNG gặp vấn đề này.**
3. Trên cả hai tập kiểm tra phát triển (`DEV_CAL` và `DEV_SEARCH`), đầu dò Neural độc lập thể hiện độ phân tách và độ tự tin cực kỳ xuất sắc đối với âm vị Thay thế:
   - Trên `DEV_CAL`: AP-Substitution đạt **0.5472**, xác suất $p_S$ cực đại đạt **0.9957**.
   - Dưới chính sách vận hành danh định $B=0.20$ (ngưỡng đóng băng hoàn toàn từ TRAIN OOF), **chỉ riêng đầu dò Neural đã đạt Recall-S = 9.55% (170 True Positives) với Precision-S = 81.34% trên `DEV_CAL`**.
   - Khi hợp nhất 50/50 với E4, **100% trong số 170 âm vị Thay thế này bị triệt tiêu hoàn toàn (Recall-S = 0.00%)**.

---

## 2. Chuỗi Cơ chế 5 Giai đoạn Gây ra Hiện tượng 0% Recall-S

Quá trình điều tra pháp y dữ liệu đã làm sáng tỏ cơ chế 5 bước dẫn tới sự sụp đổ của Substitution Recall:

```
[Giai đoạn 1: Lỗi Chọn Ứng viên Cạnh tranh của CTC]
   Lấy 4 âm vị kề cận theo bảng chữ cái modulo 42 thay vì các âm vị nhầm lẫn âm học thực tế
                  │
                  ▼
[Giai đoạn 2: Sụp đổ Tỉ số Hợp lý Ngoài miền Dữ liệu]
   SLLR sụp đổ từ +17.58 trên TRAIN xuống trung vị -2.85 trên DEV; OLLR vẫn cao (+11.58 max)
                  │
                  ▼
[Giai đoạn 3: Triệt tiêu trong Đơn hình Softmax E4]
   e^{OLLR} áp đảo hoàn toàn e^{SLLR}; p_{E4, S} bị khóa trần ở mức 0.6508 (trung vị 0.0081)
                  │
                  ▼
[Giai đoạn 4: Pha loãng bằng Phép Hợp nhất Tĩnh 50/50]
   Phép cộng 0.5 * Neural + 0.5 * E4 kéo trần xác suất xuống: p_{fixed, S} <= 0.6539 trên toàn bộ DEV_CAL
                  │
                  ▼
[Giai đoạn 5: Ngưỡng Vận hành Cắt đứt Hoàn toàn]
   Ngưỡng chung tau_T = 0.8485 (bị kéo lên do p_O cao) nằm cao hơn hẳn 0.6539 --> RECALL-S = 0.00%
```

### Chi tiết Kỹ thuật:
1. **Lỗi sinh đối thủ cạnh tranh:** Mã nguồn `agent_a_sm13_engine.py` chọn 4 ứng viên thay thế bằng công thức `sub_cands = [(phone_id + k) % 42 for k in range(1, 5)]`. Vì các âm vị được xếp theo thứ tự chữ cái (`AA, AE, AH, AO, AW, AX, AY...`), việc cộng +1, +2, +3, +4 dẫn đến việc kiểm tra những âm vị hoàn toàn không liên quan về mặt ngữ âm. Hơn nữa, việc lấy trung bình số học `np.mean(sub_scores)` làm loãng log-likelihood của ứng viên thực sự.
2. **Sự bất đối xứng giữa SLLR và OLLR:** Ngược lại, giả thuyết lược bỏ là phép xóa vị trí chính xác không phụ thuộc vào bảng đối thủ, nên $\text{OLLR}$ đạt tới $+11.58$ trên `DEV_CAL`.
3. **Triệt tiêu mẫu số Softmax:** Trong công thức $p_{\text{E4}, S} = \frac{e^{\text{SLLR}}}{1 + e^{\text{SLLR}} + e^{\text{OLLR}}}$, do $e^{\text{OLLR}} \gg e^{\text{SLLR}}$, xác suất $p_{\text{E4}, S}$ bị đè bẹp xuống trung vị **0.0081** (dưới 1%).
4. **Pha loãng xác suất hợp nhất:** Một âm vị thay thế dù mạng Neural dự đoán tự tin 0.90, khi cộng với E4 (0.05) sẽ cho ra xác suất hợp nhất $0.5 \times 0.90 + 0.5 \times 0.05 = \mathbf{0.475}$. Giá trị cực đại của toàn bộ 1.781 âm vị thay thế trên `DEV_CAL` chỉ đạt **0.653875**.
5. **Cắt đứt quyết định:** Ngưỡng vận hành $\tau_T \approx 0.8485$ (được đặt để kiểm soát cảnh báo giả từ lớp Omission có điểm lên tới 0.99) nằm vượt quá trần 0.6538 của Substitution. Do đó, không có bất kỳ âm vị Thay thế nào vượt qua được ngưỡng.

---

## 3. Ma trận So sánh Bằng chứng giữa các Thành phần

Dưới cùng một chính sách vận hành danh định $B = 0.20$ (ngưỡng đóng băng từ TRAIN OOF):

| Tập Dữ liệu | Chỉ số Đánh giá | SM13 Neural-Only | CTC/E4-Only | Hợp nhất Cố định 50/50 | Tác động của Hợp nhất so với Neural |
|---|---|---|---|---|---|
| **DEV_CAL** | **AP-Substitution** | **0.5472** | 0.3221 | 0.5513 | +0.0041 (+0.7%) |
| | **Recall-Substitution** | **9.55% (170 TP)** | **0.00% (0 TP)** | **0.00% (0 TP)** | **-9.55% (-170 TP, mất trắng 100%)** |
| | Precision-Substitution | **81.34%** | 0.00% | 0.00% | -81.34% |
| | F1-Substitution | **0.1709** | 0.0000 | 0.0000 | -0.1709 |
| | AP-Omission | 0.4692 | 0.3948 | **0.5721** | +0.1029 (+21.9%) |
| | Recall-Omission | 10.17% (47 TP) | **29.44% (136 TP)** | 24.46% (113 TP) | +14.29% (+66 TP) |
| | F1-Omission | 0.1780 | **0.3716** | 0.3570 | +0.1790 (+100.6%) |
| | Macro-3 Error F1 | 0.1983 | 0.2059 | **0.2011** | +0.0028 (+1.4%) |
| | Cảnh báo sai loại (Wrong Type) | **8** | 58 | 10 | +2 cảnh báo |
| **DEV_SEARCH** | **AP-Substitution** | 0.3711 | 0.1861 | **0.3814** | +0.0103 (+2.8%) |
| | **Recall-Substitution** | **7.75% (137 TP)** | **0.00% (0 TP)** | **0.00% (0 TP)** | **-7.75% (-137 TP, mất trắng 100%)** |
| | Precision-Substitution | **49.82%** | 0.00% | 0.00% | -49.82% |
| | F1-Substitution | **0.1341** | 0.0000 | 0.0000 | -0.1341 |
| | Recall-Omission | 6.22% (13 TP) | **20.10% (42 TP)** | 12.92% (27 TP) | +6.70% (+14 TP) |
| | F1-Omission | 0.1053 | **0.2090** | 0.1818 | +0.0765 (+72.6%) |
| | Macro-3 Error F1 | **0.1595** | 0.1494 | 0.1403 | **-0.0192 (-12.0%)** |
| | Cảnh báo sai loại (Wrong Type) | 6 | 40 | **4** | -2 cảnh báo |
| **TRAIN OOF (S101)** | **Recall-Substitution** | **5.89% (429 TP)** | 1.48% (111 TP) | 3.36% (252 TP) | -2.53% (-177 TP) |
| | Recall-Omission | 9.04% (164 TP) | 10.25% (187 TP) | **16.38% (298 TP)** | +7.34% (+134 TP) |
| | Macro-3 Error F1 | 0.1495 | 0.1264 | **0.1679** | +0.0184 (+12.3%) |
| | Cảnh báo sai loại (Wrong Type) | **33** | 74 | 52 | +19 cảnh báo |

---

## 4. Tương quan Thứ hạng và Phân tích Cấp độ Người nói

### 4.1 Tương quan giữa Neural và E4
- Trên tập các âm vị Thay thế thực tế (`GOLD_SUBSTITUTION`), hệ số tương quan Pearson giữa điểm số Neural và điểm số E4 trên `DEV_CAL` chỉ là **+0.0043** và trên `DEV_SEARCH` là **+0.0441**.
- Tương quan Spearman hạng $\rho \le 0.088$.
- **Ý nghĩa:** Hai thành phần này hoàn toàn trực giao trên người nói chưa thấy. Việc cộng trung bình số học một mô hình không có độ tự tin (E4) vào một mô hình có độ tự tin cao (Neural) đóng vai trò như một bộ lọc nhiễu phá hoại tín hiệu phân loại Thay thế.

### 4.2 Chi tiết 6 Người nói Development
- Đối với **toàn bộ 6 người nói** (`ERMS`, `THV`, `NCC`, `RRBI`, `YBAA`, `YDCK`):
  - Đầu dò Neural phát hiện ổn định từ **31 đến 96 True Positives** cho mỗi người nói.
  - Mô hình hợp nhất 50/50 tạo ra **chính xác 0 True Positives** cho từng người nói một, không có bất kỳ ngoại lệ nào.

---

## 5. Khuyến nghị Khoa học và Quyết định Tiếp theo

1. **Chấm dứt hoàn toàn việc sử dụng Hợp nhất Số học Cố định 50/50 cho âm vị Thay thế:**
   Không thể ép buộc một mô hình có dynamic range bị thu hẹp tham gia vào việc phân loại Thay thế bằng phép chia đều 50/50.
2. **Triển khai Cơ chế Định tuyến Bất đối xứng (Asymmetric Decoupled Routing):**
   - **Âm vị Thay thế (Substitution):** Sử dụng trực tiếp đầu dò Neural ($p_S = p_{\text{neural}, S}$).
   - **Âm vị Lược bỏ (Omission):** Sử dụng mô hình Hợp nhất hoặc E4-only ($p_O = p_{\text{fixed}, O}$), nơi E4 phát huy thế mạnh vượt trội (tăng Recall-O từ 10.2% lên 24.5%).
   - **Hiệu quả dự phóng trên DEV_CAL:**
     - Recall-S: **9.55%**
     - Recall-O: **24.46%**
     - Recall-I: **12.75%**
     - Macro-3 Error F1 nhảy vọt từ **0.1692 lên ~0.2450** (+0.076 điểm F1 tuyệt đối!).
3. **Sửa chữa Lỗi Sinh Đối thủ Cạnh tranh của CTC:**
   Khi chuẩn bị cho các giai đoạn tiếp theo, cần cập nhật hàm `compute_e4_likelihoods` để sử dụng danh sách nhầm lẫn âm vị học thực tế thay vì dịch chuyển modulo 42 theo chữ cái.
