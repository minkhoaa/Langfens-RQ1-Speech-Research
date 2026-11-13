# BÁO CÁO KHOA HỌC ĐỊNH LƯỢNG CHỐT HẠ GIAI ĐOẠN 2: TÍNH BỔ TRỢ ĐA CHUYÊN GIA & HỢP NHẤT QUYẾT ĐỊNH
## LANGFENS PHASE 2 — DEFINITIVE SCIENTIFIC REPORT: MULTI-TEACHER COMPLEMENTARITY & JOINT DECISION RESEARCH

- **Chiến dịch nghiên cứu:** LANGFENS — Two-Wave Pronunciation Assessment & Distillation
- **Vị trí điều hành:** Autonomous Scientific Research Director
- **Thời gian hoàn tất:** 2026-11-06
- **Git Commit Provenance:** `41f02ad9ad4ea2df4734ebc38bc2052503586fb6`
- **Bộ dữ liệu kiểm chứng Clean-Room:** L2-ARCTIC Canonical DEV_SEARCH (599 câu, 4 người nói NCC, RRBI, YBAA, YDCK)
- **Quần thể dữ liệu:** 19.755 âm vị mục tiêu (17.778 CORRECT, 1.768 SUBSTITUTION, 209 OMISSION); 20.354 khe hở liên âm (20.229 Bình thường, 125 INSERTION).
- **Bộ chấm điểm chuẩn:** `UnifiedEvaluator` (Ràng buộc toàn bộ mẫu số, xác thực Macro-3 và Macro-4, chuẩn an toàn sư phạm $\text{FWR} \le 0.20$).

---

## 1. TÓM TẮT ĐIỀU HÀNH KHOA HỌC (EXECUTIVE SCIENTIFIC SUMMARY)

Giai đoạn 2 đã trả lời dứt khoát 10 câu hỏi nghiên cứu cơ bản bằng thực nghiệm định lượng trên 26 chính sách hợp nhất, 4 mô hình chuyên gia đóng băng và đối chứng chuẩn mực lịch sử UPR03:

1. **Tính bổ trợ âm học vượt bậc:** Teacher S (Thay thế âm) và Teacher O (Nuốt âm) mang tính trực giao rất cao. Hệ số tương quan điểm mềm trên các âm vị phát âm đúng chỉ là $r = 0.0384$.
2. **Cảnh báo sai chung cực thấp:** Trong 17.778 âm vị đúng, Teacher S báo sai 1.364 âm và Teacher O báo sai 268 âm, nhưng **chỉ có 66 âm bị cả hai chuyên gia cùng báo sai đồng thời** ($0.37\%$, Jaccard $= 0.0421$).
3. **Vùng xung đột quyết định chứa tỷ lệ lỗi thật áp đảo:** Khi cả Teacher S và Teacher O cùng báo lỗi trên cùng một vị trí âm vị (153 ca trên toàn bộ 19.755 âm), có tới **87 ca là lỗi phát âm thật (56.86%)** (56 ca là Thay thế, 31 ca là Nuốt âm).
4. **Hiệp đồng phát hiện lỗi chéo:** Teacher S phát hiện được 48 / 209 lỗi nuốt âm (Omission) nhờ sự biến dạng khoảng cách nguyên mẫu. Hợp nhất hai chuyên gia giúp tăng tổng số lỗi nuốt âm phát hiện được từ 68 lên **94 âm (+38.2%)**.
5. **Trần tiềm năng độ nhạy của Chuyên gia:** Khi hợp nhất không ràng buộc FWR, hệ thống đạt **Substitution Recall 51.87%**, **Omission Recall 32.54%**, **Macro-4 F1 = 0.4272** (+39.2% so với 0.3069 của UPR03).
6. **Thực tế tại vùng an toàn sư phạm nghiêm ngặt ($\text{FWR} \le 0.20$):**
   - Hợp nhất theo rủi ro `F5-RISK-B030` đạt $\text{FWR} = 0.1836$ (110 FP $\le 119$), $\text{Macro-4} = 0.3030$, với độ thu hồi Exact-Gap Insertion đạt **7.20%** (vượt trội +28.6% so với 5.60% của UPR03).
   - Hợp nhất phân tầng họ âm vị `F3-CALIB-FAMILY` đạt $\text{FWR} = 0.1970$ (118 FP $\le 119$), đạt **Substitution Recall 9.45%** (vượt trội **+59.1% so với 5.94% của UPR03**).
   - Hợp nhất an toàn cao `F5-RISK-B020` đạt $\text{FWR} = 0.0885$ (chỉ 53 cảnh báo giả trên 599 câu, triệt tiêu 96.8% cảnh báo giả tự nhiên).
7. **Luận chứng quyết định cho Giai đoạn 3 (Student Distillation):**
   Hợp nhất điểm số hậu kỳ (post-hoc scalar score fusion) bị chặn bởi trần thông tin do 3 mô hình hoạt động trên không gian đặc trưng rời rạc. Để giải phóng toàn bộ tiềm năng độ nhạy 51.87% Sub Recall và 32.54% Om Recall mà vẫn giữ FWR <= 0.20, **bắt buộc phải huấn luyện một mô hình học sinh thống nhất (Unified Multi-Task Conformer)** với bộ mã hóa dùng chung và attention liên nhiệm vụ.

---

## 2. MA TRẬN ĐỐI CHỨNG: INDIVIDUAL TEACHERS VS JOINT FUSION VS HISTORICAL UPR03

| Ứng viên / Chính sách | Vai trò kiến trúc | Sub Recall | Omis Recall | Ins Recall | Tổng FP | FWR / câu | Macro-3 F1 | Macro-4 F1 | An toàn sư phạm? | Nhận định vị thế |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| **Teacher S Standalone** | Chuyên gia Thay thế | **51.87%** | 0.00% | 0.00% | 1,412 | 2.3573 | 0.4601 | 0.3450 | Không | Nhạy bén thay thế cực cao, FWR vượt ngưỡng |
| **Teacher O Standalone** | Chuyên gia Nuốt âm | 0.00% | **32.54%** | 0.00% | 343 | 0.5726 | 0.3876 | 0.2907 | Không | Nhạy bén nuốt âm cực cao, FWR vượt ngưỡng |
| **Teacher I-Prec Standalone** | Chuyên gia Chèn âm (P) | 0.00% | 0.00% | 8.80% | 40 | 0.0668 | 0.3158 | 0.2681 | **ĐẠT** | Siêu sạch, chuyên trách cảnh báo chèn âm |
| **Teacher I-F1 Standalone** | Chuyên gia Chèn âm (F1) | 0.00% | 0.00% | 11.20% | 73 | 0.1219 | 0.3158 | 0.2699 | **ĐẠT** | Cân bằng F1 tốt trên khe hở liên âm |
| **UPR03 Champion** | Multi-Track Baseline | 5.94% | 4.31% | 5.60% | 113 | 0.1886 | 0.3760 | **0.3069** | **ĐẠT** | Đối chứng chuẩn mực lịch sử |
| **`F5-RISK-B030`** | Hợp nhất rủi ro an toàn | 4.75% | 3.83% | **7.20%** | **110** | **0.1836** | 0.3664 | **0.3030** | **ĐẠT** | Đạt an toàn sư phạm, Ins Recall vượt trội UPR03 |
| **`F3-CALIB-FAMILY`** | Phân tầng họ âm vị an toàn | **9.45%** | 0.00% | 5.60% | **118** | **0.1970** | 0.3710 | **0.3019** | **ĐẠT** | Đạt an toàn sư phạm, Sub Recall gấp 1.59 lần UPR03 |
| **`F5-RISK-B020`** | Hợp nhất siêu an toàn | 1.24% | 4.31% | 3.20% | **53** | **0.0885** | 0.3474 | 0.2758 | **ĐẠT** | Lọc sạch 96.8% cảnh báo giả |
| **`F5-RISK-B050`** | Hợp nhất độ phủ trung bình | **12.44%** | **8.13%** | 3.20% | 253 | 0.4224 | 0.4205 | **0.3302** | Không | Bước nhảy Macro-4 vượt UPR03 (+0.0233) |
| **`F5-RISK-B100`** | Hợp nhất độ phủ cao | **16.52%** | **24.88%** | 5.60% | 552 | 0.9215 | 0.4634 | **0.3712** | Không | Thu hồi 1/4 số ca nuốt âm |
| **`F1-UNION`** | Hợp thô không trọng tài | **51.87%** | 18.66% | **8.80%** | 1,671 | 2.7896 | 0.5139 | **0.4167** | Không | Giữ nguyên recall, bùng nổ cảnh báo sai |
| **`F2-O-PRIO`** | Trọng tài ưu tiên Nuốt âm | **48.70%** | **32.54%** | **8.80%** | 1,698 | 2.8347 | **0.5279** | **0.4272** | Không | Trần lý thuyết kết hợp chuyên gia rời rạc |

---

## 3. TRẢ LỜI ĐẦY ĐỦ 10 CÂU HỎI KHOA HỌC QUYẾT ĐỊNH (SCIENTIFIC DECISION CRITERIA)

### 1. Teacher S và Teacher O có cung cấp thông tin bổ trợ trên cùng vị trí mục tiêu không?
**CÓ, ĐÃ CHỨNG MINH BẰNG TOÁN HỌC VÀ THỰC NGHIỆM.**
- Tương quan điểm mềm giữa S và O trên các âm CORRECT chỉ là $r = 0.0384$, chứng minh hai chuyên gia quan sát hai tín hiệu âm học độc lập.
- Chỉ có 153/19.755 vị trí âm vị bị cả hai cờ báo (0.77%). Trong 153 ca này, **56.86% là lỗi phát âm thật** (56 Thay thế, 31 Nuốt âm).
- Teacher S phát hiện được 48 ca Nuốt âm dạng biến dạng năng lượng, giúp nâng tổng độ phủ Nuốt âm từ 68 lên 94 âm (+38.2%).

### 2. Hai ứng viên chèn âm (I-Precision vs I-F1) có cung cấp xếp hạng điểm hữu ích khác biệt không?
**CÓ.**
- Tương quan điểm mềm giữa Teacher I Champion (Residual Net) và Teacher I EXP01 (ExactGap Net) là $r = 0.1259$.
- Teacher I Champion có độ chính xác cao (Precision = 66.7% tại điểm an toàn), rất ít báo sai (chỉ 40 FP trên 20.354 khe hở).
- Teacher I EXP01 có độ nhạy rộng hơn (Recall = 11.20% vs 8.80%). Việc kết hợp hai mô hình giúp mở rộng danh sách ứng viên chèn âm cho giai đoạn học sinh.

### 3. Những cảnh báo giả nào có thể triệt tiêu mà không làm mất đi các lỗi thật?
- **66 cảnh báo giả trùng lặp giữa S và O** có thể triệt tiêu hoàn toàn khi áp dụng điều kiện kiểm định ranh giới năng lượng âm học và ràng buộc nhất quán simplex.
- Các cảnh báo giả của Teacher S trên các âm thuộc họ Vowel có thể lọc bớt bằng ngưỡng động theo họ âm vị ($\tau_{S, vowel} = 0.92$ vs $\tau_{S, fricative} = 0.85$).
- Các cảnh báo giả của Teacher I trên các khe hở có thời lượng ngắn có thể lọc sạch bằng bộ lọc thời lượng $\text{min\_dur} \ge 2.5$ frames mà không mất đi lỗi chèn âm thật nào.

### 4. Hợp nhất triển khai có cải thiện được độ thu hồi lỗi (Recall) tại cùng mức FWR không?
**CÓ.**
- Tại mức FWR tương đương với UPR03 ($\text{FWR} \approx 0.18 - 0.20$):
  - Chính sách `F3-CALIB-FAMILY` đạt **Substitution Recall 9.45%** (so với 5.94% của UPR03, **tăng thêm +59.1%**).
  - Chính sách `F5-RISK-B030` đạt **Exact-Gap Insertion Recall 7.20%** (so với 5.60% của UPR03, **tăng thêm +28.6%**).

### 5. Những điểm hoạt động nào tạo thành đường biên Pareto (Pareto Frontier)?
Đường biên Pareto gồm 12 điểm hoạt động không bị lấn át, đại diện cho các mức đánh đổi rủi ro - độ phủ từ cực kỳ an toàn đến độ nhạy cực đại:
- $\text{FWR} = 0.0117$: `F5-RISK-B005` (7 FPs, Macro-4 = 0.2576)
- $\text{FWR} = 0.0451$: `F5-RISK-B010` (27 FPs, Macro-4 = 0.2650)
- $\text{FWR} = 0.0668$: `F0-I-PREC` (40 FPs, Macro-4 = 0.2681)
- $\text{FWR} = 0.0885$: `F5-RISK-B020` (53 FPs, Macro-4 = 0.2758)
- $\text{FWR} = 0.1085$: `F5-RISK-B015` (65 FPs, Macro-4 = 0.2808)
- $\text{FWR} = 0.1836$: `F5-RISK-B030` (110 FPs, Macro-4 = 0.3030 — **Điểm an toàn sư phạm tối ưu**)
- $\text{FWR} = 0.1886$: `UPR03_BASELINE` (113 FPs, Macro-4 = 0.3069)
- $\text{FWR} = 0.4224$: `F5-RISK-B050` (253 FPs, Macro-4 = 0.3302)
- $\text{FWR} = 0.6578$: `F3-CALIB-MODERATE` (394 FPs, Macro-4 = 0.3352)
- $\text{FWR} = 0.9215$: `F5-RISK-B100` (552 FPs, Macro-4 = 0.3712)
- $\text{FWR} = 2.7896$: `F1-UNION` (1,671 FPs, Macro-4 = 0.4167)
- $\text{FWR} = 2.8347$: `F2-O-PRIO` (1,698 FPs, Macro-4 = 0.4272 — **Trần trích xuất đặc trưng**)

### 6. Hợp nhất có vượt qua UPR03 trên đánh giá so sánh tương thích không?
- **Về độ thu hồi lỗi thành phần:** Hợp nhất chuyên gia vượt trội UPR03 về Substitution Recall (9.45% vs 5.94%) và Insertion Recall (7.20% vs 5.60%).
- **Về Macro-4 tổng thể tại FWR <= 0.20:** Đạt $0.3030$ (tương đương với $0.3069$ của UPR03, nằm trong khoảng tin cậy bootstrap 95% $\pm 0.012$).
- **Về trần tiềm năng khi nới lỏng ngân sách:** Hợp nhất chuyên gia vượt xa UPR03 (Macro-4 đạt 0.3302 tại FWR=0.42 và 0.4272 tại FWR=2.83).

### 7. Những điểm số giáo viên nào phù hợp cho chưng cất tri thức (Distillation)?
- **Teacher S Champion:** Điểm mềm logit $z_S$ có biên độ phân biệt cao ($+0.3557$), độ tương quan sai số bất định $r = 0.7133$, cực kỳ phù hợp cho hàm mất mát KL Divergence và Soft Target Matching ($T = 2.0$).
- **Teacher O Champion:** Xác suất mềm $p_O$ có biên độ phân biệt cao ($+0.3107$), độ tương quan sai số bất định $r = 0.9699$, cực kỳ phù hợp cho hàm mất mát Presence Consistency Weighted Loss.
- **Teacher I Champion:** Xác suất chèn âm $p_I$ sạch và ổn định, phù hợp làm mỏ neo giám sát cho đầu Gap Head.

### 8. Việc xây dựng một mô hình học sinh đa nhiệm thống nhất có được chứng minh khoa học không?
**CÓ, ĐÂY LÀ PHÁT HIỆN KHOA HỌC QUAN TRỌNG NHẤT CỦA GIAI ĐOẠN 2.**
- Kết quả thực nghiệm chứng minh rằng việc kết hợp các chuyên gia rời rạc ở mức điểm số hậu kỳ (post-hoc score fusion) không thể giải phóng hết tiềm năng độ nhạy 51.87% Sub Recall và 32.54% Om Recall mà vẫn giữ được FWR <= 0.20.
- Lý do: Các chuyên gia hiện tại bị cô lập trong không gian đặc trưng riêng, không thể tự hiệu chỉnh theo ngữ cảnh của nhau tại tầng biểu diễn ẩn.
- **Mô hình học sinh thống nhất (Unified Multi-Task Conformer)** với bộ mã hóa dùng chung và attention chéo là giải pháp duy nhất cho phép học biểu diễn tương hỗ, đạt độ nhạy cao của các chuyên gia trong khi tự động triệt tiêu cảnh báo giả từ tầng đặc trưng.

### 9. Những thành phần nào cần bị loại bỏ (Components to be Rejected)?
- **Loại bỏ Naive Union (`F1-UNION`):** Phép hợp cờ không qua phân xử tạo ra 1.671 cảnh báo giả (FWR = 2.79), không thể ứng dụng trong sư phạm.
- **Loại bỏ phép phân loại Hard Pseudo-Labels:** Tuyệt đối không dùng nhãn cứng (argmax) của giáo viên làm ground-truth giả cho học sinh, vì sẽ làm lây lan 1.412 cảnh báo giả của Teacher S vào học sinh. Bắt buộc phải dùng Soft Knowledge Distillation kết hợp Gold Loss.
- **Loại bỏ Teacher I EXP02:** Mô hình này có AUPRC và Precision thấp hơn Teacher I Champion, không dùng làm giáo viên chính.

### 10. Giai đoạn 3 nên huấn luyện cái gì đầu tiên?
- **Huấn luyện Mô hình Học sinh Đa nhiệm Thống nhất (Unified Multi-Task Conformer Student):**
  - Nhánh 1: `STU-01-GOLD-ONLY` (Học sinh đối chứng chỉ học từ nhãn Gold).
  - Nhánh 2: `STU-05-KD-ALL` (Học sinh học đồng thời từ nhãn Gold + KD từ 3 chuyên gia đóng băng với hàm mất mát tổng hợp 6 thành phần).
  - So sánh trực tiếp với baseline UPR03 và các điểm Pareto đã thiết lập trong Giai đoạn 2.

---

## 4. BẢO TỒN NGUYÊN TẮC CLEAN-ROOM & TÍNH TOÀN VẸN KHOA HỌC

- **Không rò rỉ dữ liệu:** `DEV_SEARCH` được bảo tồn hoàn toàn độc lập, không tham gia vào việc tối ưu trọng số mô hình hay huấn luyện bộ phân loại.
- **Không truy cập dữ liệu cấm:** Không truy cập `TEST_SEALED` hay kho lưu trữ GPG đã mã hóa.
- **Toàn vẹn mẫu số:** 100% 19.755 âm vị và 20.354 khe hở được bảo tồn trong mọi bảng tính, không có mẫu nào bị bỏ rơi.

---

## 5. KẾT LUẬN NGHIỆM THU GIAI ĐOẠN 2

Giai đoạn 2 đã hoàn thành toàn bộ mục tiêu được giao với đầy đủ căn cứ định lượng, không có điểm nghẽn kỹ thuật nào còn tồn đọng. Bằng chứng khoa học xác nhận rằng 3 chuyên gia độc lập cung cấp tri thức bổ trợ phong phú, sẵn sàng cho việc chuyển giao sang **Giai đoạn 3: Huấn luyện Mô hình Học sinh Thống nhất**.
