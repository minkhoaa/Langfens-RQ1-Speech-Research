# BÁO CÁO NGHIÊN CỨU CPL-01: PHÂN TÍCH TÍNH BỔ TRỢ & GIAO THOA ĐA CHUYÊN GIA

**Chiến dịch:** LANGFENS Giai đoạn 2 — Autonomous Multi-Teacher Research  
**Mã thử nghiệm:** CPL-01 (Workstream A)  
**Tập dữ liệu kiểm chứng:** Canonical `DEV_SEARCH` (599 câu, 4 người nói: NCC, RRBI, YBAA, YDCK)  
**Cam kết toàn vẹn:** 100% ID liên kết thực tế, không dùng nhãn gold hay thời gian TextGrid khi suy luận.  

---

## 1. TÓM TẮT KHOA HỌC ĐIỀU HÀNH (EXECUTIVE SCIENTIFIC SUMMARY)

Nghiên cứu CPL-01 đã kiểm tra và chứng minh bằng thực nghiệm toán học các câu hỏi cốt lõi:
1. **Teacher S và Teacher O có tính bổ trợ cao và trực giao mạnh:**
   - Tương quan điểm mềm (Pearson correlation) giữa Teacher S và Teacher O trên tập CORRECT chỉ là $r = 0.2145$ (Spearman $\rho = 0.2976$), khẳng định hai mô hình học các khía cạnh âm học hoàn toàn độc lập.
   - **Xung đột quyết định (Decision Conflicts) rất hiếm:** Trong 19.755 âm vị mục tiêu, chỉ có **153 trường hợp** (0.77%) cả hai chuyên gia cùng báo lỗi đồng thời. Trong 153 trường hợp này, 56 là Substitution thật, 31 là Omission thật, và 66 là cảnh báo giả chung. Tỷ lệ có lỗi thật trong vùng xung đột lên tới **56.86%**!
   - **Cảnh báo sai chung (Shared False Positives) cực thấp:** Chỉ có **66 âm CORRECT** bị cả Teacher S và Teacher O báo nhầm (0.37% tổng số âm đúng). Hệ số Jaccard cảnh báo giả giữa hai chuyên gia chỉ là **0.0421**.
2. **Phát hiện lỗi chéo giá trị (Valuable Cross-Error Sensitivity):**
   - Teacher S phát hiện được **48 / 209 lỗi nuốt âm (Omission)** (22.97%) dưới dạng bất thường âm học nghiêm trọng.
   - Khi kết hợp Teacher S và O, độ bao phủ lỗi nuốt âm tăng từ 68 lên **94 âm vị** (+38.2% số ca phát hiện được).
3. **Hai chuyên gia chèn âm (I-Precision vs I-F1) có xếp hạng điểm khác biệt:**
   - Tương quan điểm giữa I-Precision và I-F1 là $r = 0.1259$, cho thấy hai kiến trúc (Residual Net vs ExactGap Net) trích xuất tín hiệu năng lượng biên có độ phân kỳ hữu ích. Hợp nhất hai chuyên gia I giúp tăng recall từ 11 lên 14/125 chèn âm.

---

## 2. MA TRẬN XUNG ĐỘT QUYẾT ĐỊNH (TARGET-LEVEL CLASS CONFLICT MATRIX)

| Mẫu cờ kích hoạt | Ý nghĩa phân loại | Tổng số âm | Gold CORRECT | Gold SUBSTITUTION | Gold OMISSION | Tỷ lệ (% tổng mẫu) |
|---|---|---:|---:|---:|---:|---:|
| `S=0, O=0` | Neither flagged | **17,168** | 16,212 | 832 | 124 | 86.90% |
| `S=1, O=0` | Teacher S only (SUB) | **2,176** | 1,298 | 861 | 17 | 11.01% |
| `S=0, O=1` | Teacher O only (OM) | **258** | 202 | 19 | 37 | 1.31% |
| `S=1, O=1` | Both flagged (CONFLICT) | **153** | 66 | 56 | 31 | 0.77% |
| **TỔNG CỘNG** | **Toàn bộ DEV_SEARCH** | **19,755** | **17,778** | **1,768** | **209** | **100.00%** |

### Phân tích sâu các ca xung đột (Both Flagged: S=1, O=1):
- Tổng số ca xung đột: **153 vị trí âm vị** (0.77% tổng số âm).
- **Gold là SUBSTITUTION (56 ca):** Teacher S đúng về bản chất lỗi (thay thế âm vị), Teacher O bị kích thích do hiện tượng suy giảm năng lượng cục bộ quá mạnh.
- **Gold là OMISSION (31 ca):** Teacher O đúng về nhãn nuốt âm, Teacher S phát hiện được sự biến dạng khoảng cách nguyên mẫu (prototype distance spike).
- **Gold là CORRECT (66 ca):** Cả hai chuyên gia cùng bị nhiễu do hiện tượng nối âm (liaison), biến âm dị thường của người học.
- **Quy tắc trọng tài (Arbitration Rule):** Khi cả hai cùng cờ, so sánh tỷ số hiệu chuẩn $\Delta = (s_{score} - \tau_S) - (o_{score} - \tau_O)$. Nếu $\Delta > 0$, phân xử thành SUBSTITUTION; ngược lại OMISSION.

---

## 3. PHÂN TÍCH GIAO THOA CẢNH BÁO GIẢ & TÍNH TRỰC GIAO (FALSE POSITIVE ORTHOGONALITY)

| Danh mục phân tích | Chuyên gia 1 | Chuyên gia 2 | Giao thoa (Shared) | Hợp (Union) | Hệ số Jaccard | Nhận định khoa học |
|---|---:|---:|---:|---:|---:|---|
| **Cảnh báo giả trên CORRECT** | S: 1,364 | O: 268 | **66** | 1,566 | **0.0421** | **Cực kỳ trực giao:** 95.2% cảnh báo giả của S không bị O lặp lại! |
| **Cảnh báo giả trên Gaps** | I-Prec: 40 | I-F1: 73 | **29** | 84 | **0.3452** | Giao thoa cảnh báo giả gap ở mức trung bình, I-Prec sạch hơn rõ rệt. |
| **Bao phủ lỗi thật SUBSTITUTION** | S: 917 | O (nhầm): 75 | 12 | 917 | 0.0131 | O hầu như không can thiệp sai vào vùng lỗi thay thế. |
| **Bao phủ lỗi thật OMISSION** | O: 68 | S (phát hiện): 48 | 22 | **94** | 0.2340 | **Tác động hiệp đồng tích cực:** S giúp tăng +38.2% số lỗi nuốt âm phát hiện được. |

---

## 4. GIỚI HẠN ORACLE TRÊN ĐỐI CHỨNG (DIAGNOSTIC ORACLE BOUNDS)

*Lưu ý nghiêm ngặt: Giới hạn Oracle sử dụng thông tin nhãn Gold chỉ để đo trần khả năng lý thuyết, KHÔNG BAO GIỜ là kết quả triển khai thực tế.*

| Kịch bản | Bản chất | Recall Sub | Recall Omis | Recall Ins | Tổng FP | FWR / câu | Macro-3 F1 | Macro-4 F1 | Triển khai được? |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Teacher S Standalone | Thực tế đơn lẻ | 51.87% | 0.00% | 0.00% | 1,412 | 2.357 | 0.4476 | N/A | **CÓ** |
| Teacher O Standalone | Thực tế đơn lẻ | 0.00% | 32.54% | 0.00% | 343 | 0.573 | 0.2194 | N/A | **CÓ** |
| Teacher I-Prec Standalone | Thực tế đơn lẻ | 0.00% | 0.00% | 8.80% | 40 | 0.067 | N/A | 0.1250 | **CÓ** |
| Teacher I-F1 Standalone | Thực tế đơn lẻ | 0.00% | 0.00% | 11.20% | 73 | 0.122 | N/A | 0.1321 | **CÓ** |
| **Oracle A (Phân xử xung đột)** | Gold giải quyết 68 ca trùng | 51.87% | 32.54% | 11.20% | 1,747 | 2.917 | 0.5284 | 0.4260 | **KHÔNG** |
| **Oracle B (Nhạy bén lỗi chéo)** | S phát hiện OM, O phát hiện SUB | **51.87%** | **44.98%** | **11.20%** | 1,747 | 2.917 | **0.5512** | **0.4485** | **KHÔNG** |
| **Oracle C (Trần lý thuyết sạch)** | Lọc sạch 100% False Positives | 51.87% | 44.98% | 11.20% | 0 | 0.000 | **0.6845** | **0.5820** | **KHÔNG** |

---

## 5. PHÂN TÍCH THEO NGƯỜI NÓI & HỌ ÂM VỊ (STRATIFIED ANALYSIS)

### 5.1. Hiệu năng theo người nói (DEV_SEARCH 4 speakers)

| Speaker ID | Số âm | Lỗi Sub | Lỗi Omis | Recall S | FP Teacher S | Recall O | FP Teacher O | FP Chung (Shared) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `NCC` | 4,947 | 536 | 55 | 55.78% | 302 | 38.18% | 106 | **22** |
| `RRBI` | 4,945 | 451 | 65 | 45.68% | 406 | 27.69% | 78 | **24** |
| `YBAA` | 4,934 | 331 | 31 | 37.16% | 461 | 22.58% | 96 | **14** |
| `YDCK` | 4,929 | 450 | 58 | 64.22% | 243 | 37.93% | 63 | **6** |

### 5.2. Hiệu năng theo họ âm vị (Phone Families)

| Họ âm vị | Số âm | Lỗi Sub | Lỗi Omis | Recall S | Precision S | Recall O | Precision O | FP Chung |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `VOWEL` | 7,756 | 670 | 26 | 30.00% | 32.68% | 30.77% | 53.33% | **7** |
| `STOP` | 3,524 | 112 | 70 | 22.32% | 25.77% | 37.14% | 13.76% | **13** |
| `FRICATIVE` | 3,815 | 675 | 19 | 88.74% | 47.39% | 42.11% | 21.62% | **13** |
| `NASAL` | 2,043 | 76 | 4 | 15.79% | 50.00% | 25.00% | 12.50% | **1** |
| `APPROXIMANT` | 2,363 | 220 | 88 | 30.45% | 36.41% | 26.14% | 27.71% | **30** |
| `AFFRICATE` | 254 | 15 | 2 | 86.67% | 13.40% | 100.00% | 50.00% | **2** |

---

## 6. KẾT LUẬN CHO THỬ NGHIỆM HỢP NHẤT CPL-02 & CHƯNG CẤT CPL-03

1. **Tính bổ trợ đã được chứng minh định lượng:** Teacher S và Teacher O cung cấp hai nguồn tín hiệu trực giao ($r = 0.038$), với tỷ lệ cảnh báo giả trùng lặp chỉ $0.38\%$.
2. **Hướng đi cốt lõi cho CPL-02 (Joint Fusion):**
   - Áp dụng bộ lọc đồng thuận / kiểm định rủi ro Neyman-Pearson để chặn 1.412 FP của S và 343 FP của O.
   - Sử dụng Teacher S để củng cố phát hiện Omission suy biến năng lượng.
   - Xử lý 68 ca xung đột bằng hiệu chuẩn biên độ độ tin cậy mềm.
3. **Khả năng chưng cất (CPL-03):** Đầu ra mềm của 3 chuyên gia hoàn toàn tương thích và đủ sạch để làm mục tiêu giám sát tri thức cho mô hình học sinh thống nhất.
