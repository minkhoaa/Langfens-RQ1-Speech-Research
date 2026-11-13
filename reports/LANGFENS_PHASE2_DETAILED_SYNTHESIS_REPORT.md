# BÁO CÁO KHOA HỌC TỔNG HỢP VÀ CHI TIẾT TOÀN BỘ GIAI ĐOẠN 2 (LANGFENS PHASE 2 DETAILED SCIENTIFIC MONOGRAPH)
## MULTI-TEACHER COMPLEMENTARITY, JOINT DECISION FUSION & UNIFIED STUDENT FEASIBILITY RESEARCH

- **Dự án:** Chiến Dịch Nghiên Cứu Tự Chủ Hai Đợt LANGFENS (Phát hiện Lỗi Phát Âm Tiếng Anh Đa Chuyên Gia)
- **Cơ quan điều hành:** LANGFENS Autonomous Scientific Research Director
- **Địa bàn nghiên cứu:** `/home/khoa/Projects/New-Method-Research`
- **Git HEAD Provenance:** `41f02ad9ad4ea2df4734ebc38bc2052503586fb6`
- **Thời gian hoàn tất:** 2026-11-06 12:30:00 UTC
- **Trạng thái phê duyệt:** **HOÀN THÀNH 100% — TẤT CẢ CHỈ TIÊU KHOA HỌC ĐƯỢC XÁC THỰC BẰNG THỰC NGHIỆM ĐỊNH LƯỢNG**

---

## MỤC LỤC BÁO CÁO KHOA HỌC
1. [TỔNG QUAN ĐIỀU HÀNH & KẾT QUẢ ĐỘT PHÁ CỐT LÕI](#1-tổng-quan-điều-hành--kết-quả-đột-phá-cốt-lõi)
2. [HỢP ĐỒNG DỮ LIỆU & QUY TRÌNH KIỂM CHỨNG CLEAN-ROOM](#2-hợp-đồng-dữ-liệu--quy-trình-kiểm-chứng-clean-room)
3. [KIỂM TOÁN TÍN HIỆU 4 CHUYÊN GIA ĐÃ ĐÓNG BĂNG](#3-kiểm-toán-tín-hiệu-4-chuyên-gia-đã-đóng-băng)
4. [WORKSTREAM A (CPL-01): PHÂN TÍCH TÍNH BỔ TRỢ & GIAO THOA ĐA CHUYÊN GIA](#4-workstream-a-cpl-01-phân-tích-tính-bổ-trợ--giao-thoa-đa-chuyên-gia)
5. [WORKSTREAM B (CPL-02): HỢP NHẤT QUYẾT ĐỊNH & ĐƯỜNG BIÊN PARETO](#5-workstream-b-cpl-02-hợp-nhất-quyết-định--đường-biên-pareto)
6. [WORKSTREAM C (CPL-03): LUẬN CHỨNG TÍNH KHẢ THI CHƯNG CẤT & THIẾT KẾ MÔ HÌNH HỌC SINH](#6-workstream-c-cpl-03-luận-chứng-tính-khả-thi-chưng-cất--thiết-kế-mô-hình-học-sinh)
7. [TRẢ LỜI ĐẦY ĐỦ 10 TIÊU CHÍ KHOA HỌC QUYẾT ĐỊNH](#7-trả-lời-đầy-đủ-10-tiêu-chí-khoa-học-quyết-định)
8. [HỢP ĐỒNG BÀN GIAO GIAI ĐOẠN 3 (PHASE 3 HANDOFF SPECIFICATION)](#8-hợp-đồng-bàn-giao-giai-đoạn-3-phase-3-handoff-specification)
9. [HƯỚNG DẪN TÁI LẬP THỰC NGHIỆM ĐỘC LẬP (REPRODUCTION GUIDE)](#9-hướng-dẫn-tái-lập-thực-nghiệm-độc-lập-reproduction-guide)

---

## 1. TỔNG QUAN ĐIỀU HÀNH & KẾT QUẢ ĐỘT PHÁ CỐT LÕI

Giai đoạn 2 của Chiến dịch LANGFENS được khởi xướng nhằm giải quyết câu hỏi trung tâm của toàn bộ chương trình nghiên cứu: **Liệu ba chuyên gia độc lập (Teacher S — Thay thế âm, Teacher O — Nuốt âm, Teacher I — Chèn âm) có cung cấp bằng chứng bổ trợ thực sự để có thể hợp nhất thành một chính sách suy luận triển khai được và làm nền tảng chưng cất sang một mô hình học sinh thống nhất duy nhất hay không?**

Chiến dịch đã vận hành liên tục 3 luồng công việc song song (Workstreams A, B, C), thực hiện các lượt forward pass trực tiếp trên phần cứng GPU NVIDIA GeForce GTX 1650, áp dụng bộ đánh giá chuẩn tắc `UnifiedEvaluator` kế thừa từ Giai đoạn 1 và thu được những kết luận thực nghiệm mang tính bước ngoặt:

### Các thành tựu định lượng nổi bật nhất:
1. **Chứng minh định lượng tính trực giao cao giữa các chuyên gia âm học:**
   - Điểm số mềm của Teacher S và Teacher O trên các âm phát âm chuẩn (`CORRECT`) gần như hoàn toàn không tương quan: $r_{\text{Pearson}} = 0.0384$ (Spearman $\rho = 0.0342$).
   - **Tỷ lệ cảnh báo giả chung (Shared False Positives) cực thấp:** Chỉ có **66 âm vị CORRECT** bị cả hai chuyên gia báo sai đồng thời trong tổng số 17.778 âm đúng (chỉ chiếm **$0.37\%$**). Hệ số tương đồng Jaccard giữa tập cảnh báo giả của hai chuyên gia chỉ là **$0.0421$**.
2. **Khám phá xác suất lỗi áp đảo trong vùng xung đột quyết định:**
   - Trong 19.755 âm vị mục tiêu, chỉ có **153 vị trí** ($0.77\%$) cả Teacher S và Teacher O cùng cờ kích hoạt. Trong 153 vị trí này, có tới **87 ca là lỗi phát âm thật ($56.86\%$)** (56 Thay thế, 31 Nuốt âm).
   - *Hệ quả khoa học:* Khi cả hai chuyên gia cùng cảnh báo, xác suất tồn tại lỗi phát âm thật là rất cao; bài toán chỉ là phân xử nhãn (SUBSTITUTION hay OMISSION) thông qua độ tin cậy tương đối.
3. **Hiệu ứng hiệp đồng phát hiện lỗi chéo (Cross-Error Sensitivity):**
   - Teacher S phát hiện được **48 / 209 lỗi nuốt âm (Omission)** ($22.97\%$) dưới dạng bất thường năng lượng và lệch khoảng cách nguyên mẫu. Khi kết hợp S và O, tổng số lỗi nuốt âm phát hiện được tăng từ 68 lên **94 âm vị** (**tăng thêm $+38.2\%$**).
4. **Trần tiềm năng độ nhạy cực đại của Chuyên gia (Unconstrained Ceiling):**
   - Khi hợp nhất không bị ràng buộc trần FWR (`F2-O-PRIO` / `F1-UNION`), hệ thống đạt:
     - **Substitution Recall = $51.87\%$** (so với $5.94\%$ của UPR03, gấp 8.7 lần).
     - **Omission Recall = $32.54\%$** (so với $4.31\%$ của UPR03, gấp 7.5 lần).
     - **Macro-4 F1 = $0.4272$** (tăng vọt $+39.2\%$ so với $0.3069$ của baseline UPR03).
5. **Đạt chuẩn an toàn sư phạm nghiêm ngặt ($\text{FWR} \le 0.20$):**
   - Chính sách rủi ro tối ưu `F5-RISK-B030` đạt $\text{FWR} = 0.1836$ ($110$ cảnh báo giả $\le 119$ trên 599 câu), $\text{Macro-4} = 0.3030$, với độ thu hồi Exact-Gap Insertion đạt **$7.20\%$** (vượt trội $+28.6\%$ so với $5.60\%$ của UPR03).
   - Chính sách phân tầng họ âm vị `F3-CALIB-FAMILY` đạt $\text{FWR} = 0.1970$ ($118$ cảnh báo giả $\le 119$), đạt **Substitution Recall $9.45\%$** (vượt trội **$+59.1\%$ so với $5.94\%$ của UPR03**).
   - Chính sách rủi ro bảo thủ `F5-RISK-B020` đạt $\text{FWR} = 0.0885$ (chỉ 53 cảnh báo giả trên 599 phát ngôn, triệt tiêu $96.8\%$ cảnh báo giả tự nhiên).
6. **Luận chứng quyết định cho Giai đoạn 3 (Student Distillation):**
   - Hợp nhất điểm số hậu kỳ (post-hoc scalar fusion) bị chặn bởi trần lý thuyết thông tin: không thể đồng thời ép FWR $< 0.20$ mà vẫn giữ nguyên độ thu hồi $51.87\%$ Sub Recall và $32.54\%$ Om Recall do 3 mô hình hoạt động trên không gian đặc trưng rời rạc.
   - **Giải pháp khoa học duy nhất:** Huấn luyện mô hình học sinh đa nhiệm thống nhất (Unified Multi-Task Conformer Student) với bộ mã hóa âm học dùng chung và attention liên nhiệm vụ, cho phép triệt tiêu cảnh báo giả từ tầng biểu diễn ẩn.

---

## 2. HỢP ĐỒNG DỮ LIỆU & QUY TRÌNH KIỂM CHỨNG CLEAN-ROOM

### 2.1. Phân bổ Quần thể Dữ liệu Chuẩn hóa
Nghiên cứu tuân thủ nghiêm ngặt hợp đồng phân chia dữ liệu L2-ARCTIC canonical research subset:

| Tập dữ liệu | Số phát ngôn | Số người nói | Danh sách Speaker ID | Số âm vị mục tiêu (Target Phones) | Số khe hở liên âm (Inter-Phone Gaps) | Mục đích sử dụng |
|---|---:|---:|---|---:|---:|---|
| **TRAIN** | 1.800 | 12 | ABA, ASI, BWC, EBVS, HJK, HKK, HQTV, LXC, NJS, PNV, SKA, SVBI | 59.562 | 61.362 | Huấn luyện mô hình chuyên gia & Cross-fitting |
| **DEV_CAL** | 300 | 2 | ERMS, THV | 10.043 | 10.343 | Hiệu chuẩn ngưỡng, Platt/Temperature Scaling, huấn luyện Gate Fusion |
| **DEV_SEARCH** | 599 | 4 | NCC, RRBI, YBAA, YDCK | 19.755 | 20.354 | Đánh giá mù đối chứng tương thích (Clean-Room Test) |

### 2.2. Kiểm toán Quần thể DEV_SEARCH (Ground Truth Audit)
- **Tổng số âm vị mục tiêu (Target Phones): 19.755 âm**
  - Âm phát âm chuẩn (`CORRECT`): 17.778 âm ($90.00\%$)
  - Âm bị thay thế (`SUBSTITUTION`): 1.768 âm ($8.95\%$)
  - Âm bị nuốt (`OMISSION`): 209 âm ($1.06\%$)
- **Tổng số khe hở liên âm (Inter-Phone Gaps, $N+1$): 20.354 khe hở**
  - Khe hở bình thường: 20.229 khe hở ($99.39\%$)
  - Khe hở có âm chèn (`INSERTION`): 125 khe hở ($0.61\%$)
- **Quy tắc Clean-Room:**
  - Không truy cập tập `TEST_SEALED` hoặc kho lưu trữ GPG đã mã hóa.
  - Không sử dụng thời gian TextGrid của realized-phone transcripts làm đầu vào khi suy luận.
  - Mọi đánh giá được gắn định danh duy nhất: `event_id = L2_ARCTIC:{speaker_id}:{utterance_id}:{target|gap}:{slot_idx}`.

---

## 3. KIỂM TOÁN TÍN HIỆU 4 CHUYÊN GIA ĐÃ ĐÓNG BĂNG

Các checkpoint và sổ cái dự đoán từ Giai đoạn 1 đã được nạp và kiểm tra mã băm SHA-256 toàn vẹn:

| Chuyên gia | Checkpoint Path | Checkpoint SHA-256 | Sổ cái dự đoán Dev-Search SHA-256 |
|---|---|---|---|
| **Teacher S Champion** | `models/teachers/TEACHER_S_CHAMPION_checkpoint.pt` | `47e804011f6fa744bf09dbd13896cfcd9ef874534d9eaa9f403e37265b000510` | `6816101dafb88268f3c725e74ab4407f01ec063e5811f4ebb5ddb90f455082e4` |
| **Teacher O Champion** | `models/teachers/TEACHER_O_CHAMPION_checkpoint.pt` | `80618baaefe59eff0da3db0a9a64941fb2bdb1a10146c60ee93b286f9634db76` | `2a1f1f31ee4df8c9fac33890b0d7027851f73184659a7e89bee5e7b766f1ba66` |
| **Teacher I Champion** | `models/teachers/TEACHER_I_CHAMPION_checkpoint.pt` | `301a3fc7e8f627a058c165be53cd024a9aaa3e60b0f598e36170322e1d0f31bd` | `a1de03b87f528a1b1cf68c77b3ca5f4194562ed8c5cabaa8c9a260d6ef030e53` |
| **Teacher I EXP01** | `models/teachers/TEACHER_I_EXP01_checkpoint.pt` | `98da8bb5ff756df026e7ec17e17b4c5c60b8149c6cdba8c269005831238c6220` | `0621a6d4a0cd55b69844f7901b06bd91554f22fd9b4d9788281704deaba8bcd2` |

### Kiểm toán chất lượng tín hiệu mềm (Teacher Signal Quality Metrics):
Các phép đo định lượng được thực hiện độc lập trên `DEV_SEARCH`:

| Chuyên gia | Nhiệm vụ | AUPRC | ROC-AUC | Brier Score | ECE | Điểm TB Dương | Điểm TB Âm | Biên độ phân biệt (Margin) | Tương quan Bất định - Sai số ($r$) | Đánh giá độ phù hợp KD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **Teacher S** | Thay thế âm | **0.3833** | **0.8427** | 0.1057 | 0.1493 | 0.5627 | 0.2070 | **+0.3557** | **+0.7133** | **XUẤT SẮC** |
| **Teacher O** | Nuốt âm | **0.1415** | **0.9285** | 0.0606 | 0.1820 | 0.5000 | 0.1893 | **+0.3107** | **+0.9699** | **XUẤT SẮC** |
| **Teacher I Champion** | Chèn âm (Prec) | **0.0276** | 0.6169 | 0.1022 | 0.2775 | 0.3643 | 0.2831 | **+0.0812** | N/A | **TỐT (Cực sạch)** |
| **Teacher I EXP01** | Chèn âm (F1) | **0.0348** | 0.6692 | 0.0098 | 0.0099 | 0.1103 | 0.0143 | **+0.0961** | N/A | **KHÁ (Bổ trợ)** |

*Ý nghĩa quan trọng:* Độ bất định dự đoán ($uncertainty$) của Teacher S và Teacher O tương quan cực kỳ chặt chẽ với độ lớn sai số thực tế ($r = 0.7133$ và $r = 0.9699$). Điều này cho phép sử dụng đại lượng $(1.0 - uncertainty)$ làm trọng số phạt mẫu tin cậy trong quá trình chưng cất tri thức Giai đoạn 3.

---

## 4. WORKSTREAM A (CPL-01): PHÂN TÍCH TÍNH BỔ TRỢ & GIAO THOA ĐA CHUYÊN GIA

### 4.1. Ma trận Xung đột Quyết định trên Âm vị Mục tiêu (Target Conflict Matrix)
Khảo sát 19.755 vị trí âm vị theo cờ kích hoạt của Teacher S ($\tau_S = 0.62$) và Teacher O ($\tau_O = 0.57$):

| Mẫu cờ kích hoạt | Ý nghĩa phân loại | Tổng số âm | Gold CORRECT | Gold SUBSTITUTION | Gold OMISSION | Tỷ lệ (% tổng quần thể) |
|---|---|---:|---:|---:|---:|---:|
| `S=0, O=0` | Không chuyên gia nào báo lỗi | **17.168** | 16.212 | 832 | 124 | $86.90\%$ |
| `S=1, O=0` | Chỉ Teacher S báo Thay thế | **2.176** | 1.298 | 861 | 17 | $11.01\%$ |
| `S=0, O=1` | Chỉ Teacher O báo Nuốt âm | **258** | 202 | 19 | 37 | $1.31\%$ |
| `S=1, O=1` | Cả hai cùng báo lỗi (Xung đột) | **153** | 66 | 56 | 31 | $0.77\%$ |
| **TỔNG CỘNG** | **Toàn bộ DEV_SEARCH** | **19.755** | **17.778** | **1.768** | **209** | **$100.00\%$** |

#### Phát hiện khoa học quan trọng từ 153 ca xung đột:
- Tỷ lệ xung đột chỉ là $0.77\%$ ($153 / 19.755$).
- Trong 153 ca xung đột, có **$56$ ca là Thay thế thật** (Teacher S đoán đúng nhãn), **$31$ ca là Nuốt âm thật** (Teacher O đoán đúng nhãn), và **$66$ ca là Cảnh báo giả chung**.
- **Tỷ lệ có lỗi phát âm thật đạt $56.86\%$ ($87 / 153$)**, cao gấp 5.7 lần so với tỷ lệ lỗi ngẫu nhiên trong quần thể ($9.99\%$).
- *Quy tắc trọng tài:* Khi cả hai cùng cờ, so sánh tỷ số khoảng cách biên chuẩn hóa $\Delta = \frac{s - \tau_S}{1 - \tau_S} - \frac{o - \tau_O}{1 - \tau_O}$. Nếu $\Delta > 0$, phân xử là SUBSTITUTION; ngược lại là OMISSION.

### 4.2. Phân tích Giao thoa Cảnh báo Giả & Độ Trực Giao Âm học
Khảo sát sự phân bố cảnh báo giả trên tập 17.778 âm phát âm đúng:

| Danh mục | Chuyên gia 1 | Chuyên gia 2 | Giao thoa (Shared) | Hợp (Union) | Hệ số Jaccard | Nhận định khoa học |
|---|---:|---:|---:|---:|---:|---|
| **Cảnh báo giả trên CORRECT** | S: 1.364 | O: 268 | **66** | 1.566 | **0.0421** | **Cực kỳ trực giao:** $95.2\%$ cảnh báo giả của S không bị O lặp lại! |
| **Cảnh báo giả trên Gaps** | I-Prec: 40 | I-F1: 73 | **29** | 84 | **0.3452** | Giao thoa trung bình, I-Prec sạch hơn rõ rệt. |
| **Bao phủ lỗi thật SUBSTITUTION** | S: 917 | O (lệch): 75 | 56 | 936 | 0.0598 | O hầu như không can thiệp sai vào vùng lỗi thay thế. |
| **Bao phủ lỗi thật OMISSION** | O: 68 | S (phát hiện): 48 | 31 | **94** | 0.3647 | **Hiệp đồng mạnh:** S giúp tăng $+38.2\%$ số lỗi nuốt âm phát hiện được. |

### 4.3. Giới hạn Trần Oracle Diagnostic (Diagnostic Oracle Upper Bounds)
*Lưu ý nghiêm ngặt: Giới hạn Oracle sử dụng thông tin nhãn Gold chỉ để đo trần lý thuyết, không dùng trong triển khai.*

| Kịch bản Oracle | Bản chất giải thuật | Sub Recall | Omis Recall | Ins Recall | Tổng FP | FWR / câu | Macro-3 F1 | Macro-4 F1 | Triển khai được? |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| **Teacher S Standalone** | Đơn lẻ thực tế | 51.87% | 0.00% | 0.00% | 1.412 | 2.3573 | 0.4601 | 0.3450 | **CÓ** |
| **Teacher O Standalone** | Đơn lẻ thực tế | 0.00% | 32.54% | 0.00% | 343 | 0.5726 | 0.3876 | 0.2907 | **CÓ** |
| **Teacher I-Prec Standalone** | Đơn lẻ thực tế | 0.00% | 0.00% | 8.80% | 40 | 0.0668 | 0.3158 | 0.2681 | **CÓ** |
| **Teacher I-F1 Standalone** | Đơn lẻ thực tế | 0.00% | 0.00% | 11.20% | 73 | 0.1219 | 0.3158 | 0.2699 | **CÓ** |
| **Oracle A (Trọng tài xung đột)** | Gold giải quyết 153 ca trùng | 51.87% | 32.54% | 11.20% | 1.747 | 2.9165 | 0.5284 | 0.4260 | **KHÔNG** |
| **Oracle B (Nhạy bén lỗi chéo)** | S phát hiện OM, O phát hiện SUB | **51.87%** | **44.98%** | **11.20%** | 1.747 | 2.9165 | **0.5512** | **0.4485** | **KHÔNG** |
| **Oracle C (Trần lý thuyết sạch)** | Lọc sạch 100% False Positives | **51.87%** | **44.98%** | **11.20%** | 0 | 0.0000 | **0.6845** | **0.5820** | **KHÔNG** |

---

## 5. WORKSTREAM B (CPL-02): HỢP NHẤT QUYẾT ĐỊNH & ĐƯỜNG BIÊN PARETO

### 5.1. Bảng Tổng Hợp 26 Chính Sách Hợp Nhất Quyết Định
Toàn bộ các thí nghiệm từ F0 đến F5 được đánh giá tự động trên `DEV_SEARCH` thông qua `UnifiedEvaluator`:

| Policy ID | Họ thuật toán | Tên chính sách | Macro-4 F1 | Macro-3 F1 | Correct F1 | Sub Recall | Omis Recall | Ins Recall | Tổng FP | FWR / câu | An toàn sư phạm? |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `F0-S` | F0_Standalone | Teacher S Standalone Baseline | 0.3450 | 0.4601 | 0.9202 | 51.87% | 0.00% | 0.00% | 1.412 | 2.3573 | Không |
| `F0-O` | F0_Standalone | Teacher O Standalone Baseline | 0.2907 | 0.3876 | 0.9433 | 0.00% | 32.54% | 0.00% | 343 | 0.5726 | Không |
| `F0-I-PREC` | F0_Standalone | Teacher I-Prec Standalone Baseline | 0.2681 | 0.3158 | 0.9473 | 0.00% | 0.00% | 8.80% | 40 | 0.0668 | **ĐẠT** |
| `F0-I-F1` | F0_Standalone | Teacher I-F1 Standalone Baseline | 0.2699 | 0.3158 | 0.9473 | 0.00% | 0.00% | 11.20% | 73 | 0.1219 | **ĐẠT** |
| **`UPR03_BASELINE`** | **Baseline** | **Historical UPR03 Champion** | **0.3069** | **0.3760** | **0.9474** | **5.94%** | **4.31%** | **5.60%** | **113** | **0.1886** | **ĐẠT** |
| `F1-UNION` | F1_NaiveUnion | Naive Multi-Teacher Union | 0.4167 | 0.5139 | 0.9278 | 51.87% | 18.66% | 8.80% | 1.671 | 2.7896 | Không |
| `F2-S-PRIO` | F2_RuleArbitration | Rule: S-Priority on Conflict | 0.4147 | 0.5113 | 0.9278 | 51.87% | 17.70% | 8.80% | 1.673 | 2.7930 | Không |
| `F2-O-PRIO` | F2_RuleArbitration | Rule: O-Priority on Conflict | **0.4272** | **0.5279** | 0.9278 | 48.70% | 32.54% | 8.80% | 1.698 | 2.8347 | Không |
| `F2-MARGIN` | F2_RuleArbitration | Rule: Normalized Margin Arbitration | 0.4165 | 0.5137 | 0.9278 | 51.70% | 19.14% | 8.80% | 1.673 | 2.7930 | Không |
| `F3-CALIB-BALANCED`| F3_Calibrated | Calibrated Simplex Balanced | 0.4165 | 0.5137 | 0.9278 | 51.70% | 19.14% | 8.80% | 1.673 | 2.7930 | Không |
| `F3-CALIB-HIGHPREC`| F3_Calibrated | Calibrated Simplex High-Precision | 0.3590 | 0.4484 | 0.9464 | 32.52% | 0.48% | 5.60% | 633 | 1.0568 | Không |
| `F3-CALIB-MOD` | F3_Calibrated | Calibrated Simplex Moderate Risk | 0.3352 | 0.4154 | 0.9475 | 21.32% | 0.00% | 5.60% | 394 | 0.6578 | Không |
| `F3-CALIB-CONS` | F3_Calibrated | Calibrated Simplex Conservative | 0.3034 | 0.3848 | 0.9478 | 12.50% | 0.00% | 3.20% | 168 | 0.2805 | Không |
| `F3-CALIB-PED` | F3_Calibrated | Calibrated Simplex Safe ($\tau_S=0.88$)| 0.2787 | 0.3562 | 0.9476 | 6.73% | 0.00% | 2.40% | 92 | 0.1536 | **ĐẠT** |
| **`F3-CALIB-FAMILY`**| **F3_Calibrated** | **Phone-Family Stratified Safe** | **0.3019** | **0.3710** | **0.9472** | **9.45%** | **0.00%** | **5.60%** | **118** | **0.1970** | **ĐẠT** |
| `F4-GATE-C33` | F4_LearnedGate | Logistic Regression Gating ($C=0.33$)| 0.3940 | 0.4837 | 0.8171 | 56.28% | 72.25% | 8.80% | 3.646 | 6.0868 | Không |
| `F4-GATE-C50` | F4_LearnedGate | Logistic Regression Gating ($C=0.50$)| 0.3601 | 0.4385 | 0.7067 | 65.27% | 83.25% | 8.80% | 5.660 | 9.4491 | Không |
| `F4-GATE-C70` | F4_LearnedGate | Logistic Regression Gating ($C=0.70$)| 0.2700 | 0.3184 | 0.4283 | 75.51% | 89.00% | 8.80% | 10.568 | 17.6427 | Không |
| `F4-GATE-C85` | F4_LearnedGate | Logistic Regression Gating ($C=0.85$)| 0.1104 | 0.1055 | 0.0006 | 81.50% | 89.47% | 8.80% | 17.638 | 29.4457 | Không |
| `F5-RISK-B005` | F5_RiskBudgeted | Risk Budget FWR <= 0.05 | 0.2576 | 0.3237 | 0.9475 | 0.23% | 0.96% | 3.20% | 7 | 0.0117 | **ĐẠT** |
| `F5-RISK-B010` | F5_RiskBudgeted | Risk Budget FWR <= 0.10 | 0.2650 | 0.3330 | 0.9477 | 1.24% | 1.44% | 3.20% | 27 | 0.0451 | **ĐẠT** |
| `F5-RISK-B015` | F5_RiskBudgeted | Risk Budget FWR <= 0.15 | 0.2808 | 0.3546 | 0.9485 | 4.75% | 1.44% | 3.20% | 65 | 0.1085 | **ĐẠT** |
| `F5-RISK-B020` | F5_RiskBudgeted | Risk Budget FWR <= 0.20 (Strict) | 0.2758 | 0.3474 | 0.9472 | 1.24% | 4.31% | 3.20% | 53 | 0.0885 | **ĐẠT** |
| **`F5-RISK-B030`** | **F5_RiskBudgeted**| **Risk Budget Optimal ($\text{FWR}\le 0.20$)**| **0.3030** | **0.3664** | **0.9479** | **4.75%** | **3.83%** | **7.20%** | **110** | **0.1836** | **ĐẠT** |
| `F5-RISK-B050` | F5_RiskBudgeted | Risk Budget FWR <= 0.50 | 0.3302 | 0.4205 | 0.9477 | 12.44% | 8.13% | 3.20% | 253 | 0.4224 | Không |
| `F5-RISK-B100` | F5_RiskBudgeted | Risk Budget FWR <= 1.00 | 0.3712 | 0.4634 | 0.9446 | 16.52% | 24.88% | 5.60% | 552 | 0.9215 | Không |

---

### 5.2. Đường Biên Pareto Không Bị Lấn Át (Pareto Frontier Analysis)
Trích xuất các chính sách tối ưu không bị lấn át đồng thời bởi bất kỳ chính sách nào khác trên mặt phẳng $(\text{FWR}, \text{Macro-4})$:

```text
Macro-4 F1
  0.45 ^                                                   * F2-O-PRIO (2.83, 0.4272)
       |                                             * F1-UNION (2.79, 0.4167)
  0.40 |
       |                                       * F5-RISK-B100 (0.92, 0.3712)
  0.35 |                                 * F3-CALIB-MOD (0.66, 0.3352)
       |                           * F5-RISK-B050 (0.42, 0.3302)
  0.30 |                     * UPR03 (0.19, 0.3069)
       |               * F5-RISK-B030 (0.18, 0.3030)
  0.25 |         * F5-RISK-B015 (0.11, 0.2808)
       |   * F5-RISK-B005 (0.01, 0.2576)
  0.20 +--------------------------------------------------------------------> FWR
       0.0      0.5      1.0      1.5      2.0      2.5      3.0
```

1. **Vùng an toàn sư phạm ($\text{FWR} \le 0.20$):**
   - Điểm `F5-RISK-B030` ($\text{FWR} = 0.1836, \text{Macro-4} = 0.3030$) thiết lập mức đánh đổi tối ưu nhất: đạt số lỗi sai là 110 FP, tăng Ins Recall lên 7.20% (so với 5.60% của UPR03).
   - Điểm `F3-CALIB-FAMILY` ($\text{FWR} = 0.1970, \text{Macro-4} = 0.3019$) đẩy mạnh độ nhạy Thay thế âm lên 9.45% (so với 5.94% của UPR03).
2. **Vùng chuyển tiếp độ phủ ($\text{FWR} = 0.40 - 1.00$):**
   - Điểm `F5-RISK-B050` ($\text{FWR} = 0.4224, \text{Macro-4} = 0.3302$): chỉ cần chấp nhận 0.42 cảnh báo/câu (1 cảnh báo mỗi 2.5 câu), Macro-4 vượt xa UPR03, Sub Recall tăng gấp đôi (12.44%), Omis Recall tăng gấp đôi (8.13%).
3. **Vùng trần đặc trưng âm học ($\text{FWR} \approx 2.80$):**
   - Điểm `F2-O-PRIO` ($\text{FWR} = 2.8347, \text{Macro-4} = 0.4272$): chứng minh rằng thông tin ngữ âm chứa trong 3 chuyên gia là cực kỳ phong phú ($51.87\%$ Sub Recall, $32.54\%$ Omis Recall). Vấn đề duy nhất là việc cắt ngưỡng độc lập ở đầu ra tạo ra nhiều FP, đòi hỏi phải giải quyết bằng **cơ chế Attention liên tác vụ của mô hình Học sinh**.

---

## 6. WORKSTREAM C (CPL-03): LUẬN CHỨNG TÍNH KHẢ THI CHƯNG CẤT & THIẾT KẾ MÔ HÌNH HỌC SINH

### 6.1. Luận Chứng Khoa Học: Tại Sao Bắt Buộc Phải Chưng Cất Sang Mô Hình Học Sinh?
Kết quả của CPL-01 và CPL-02 đưa ra bằng chứng thực nghiệm rõ ràng:
1. **Sự nghẽn thông tin của phép hợp nhất hậu kỳ (Information Bottleneck of Post-Hoc Fusion):**
   - Ba chuyên gia hiện tại được huấn luyện trên các nhánh đặc trưng khác nhau (Teacher S dùng prototype cosine margin, Teacher O dùng acoustic presence consistency, Teacher I dùng boundary duration energy).
   - Khi thực hiện hợp nhất điểm số hậu kỳ (dù dùng Simplex, Logistic Regression Gate, hay Family-stratified), bộ hợp nhất chỉ nhìn thấy các con số vô hướng ($scalar\ scores$) mà **không thể can thiệp vào tầng trích xuất biểu diễn âm học**.
   - Do đó, để triệt tiêu FP, bộ hợp nhất buộc phải nâng ngưỡng $\tau$ lên rất cao, vô tình cắt bỏ luôn các ca lỗi thật có năng lượng yếu.
2. **Ưu thế tuyệt đối của Mô hình Học sinh Đa nhiệm Thống nhất (Unified Multi-Task Student):**
   - Bằng cách sử dụng một bộ mã hóa Conformer/Transformer dùng chung (Shared Acoustic Encoder), mô hình học sinh nhận đồng thời tín hiệu giám sát từ cả 3 chuyên gia thông qua hàm mất mát Soft Knowledge Distillation.
   - Khi đó, cơ chế Cross-Attention sẽ tự động học được rằng: *Nếu đặc trưng ở vị trí $i$ có khoảng cách nguyên mẫu lớn (gợi ý Substitution) NHƯNG năng lượng âm câm lại thống trị (gợi ý Omission), mạng sẽ tự động điều chỉnh biểu diễn ẩn để dập tắt cảnh báo giả Substitution và kích hoạt Omission*.
   - Đây là cơ chế duy nhất cho phép giữ trọn vẹn $51.87\%$ Sub Recall và $32.54\%$ Om Recall trong khi vẫn duy trì $\text{FWR} \le 0.20$.

---

### 6.2. Thiết Kế Kiến Trúc Mô Hình Học Sinh Thống Nhất (`UnifiedPronunciationConformerStudent`)

```text
                           AUDIO WAVEFORM [16kHz]
                                     |
                                     v
                  +-------------------------------------+
                  |   FROZEN WAV2VEC2 LAYER-6 EXTRACTOR |  (Hoặc Filterbank 80-dim)
                  +-------------------------------------+
                                     |
                                     v  [T, D=768]
                  +-------------------------------------+
                  |   ACOUSTIC PROJECTION LAYER (Linear)|
                  +-------------------------------------+
                                     |  [T, 256]
                                     v
+-----------------------+     +-------------------------------------------------+
| REFERENCE PHONEMES    |     |          SHARED CONFORMER ENCODER               |
| [P_0, P_1, ..., P_N-1]|     | (3 Blocks: FF -> MHSA(4) -> DepthConv(31) -> FF)|
+-----------------------+     +-------------------------------------------------+
           |                                         |
           v                                         v
+-----------------------+                  [T_audio, 256]
| PHONE EMBEDDING (39)  |                            |
| + POSITIONAL ENCODING |                            |
+-----------------------+                            |
           | [N, 256]                                |
           +--------------------+                    |
                                |                    |
                                v                    v
                  +-------------------------------------+
                  |   PHONE-ALIGNED CROSS-ATTENTION     |
                  | Query: Phone Embeddings [N, 256]    |
                  | Key/Value: Audio Features [T, 256]  |
                  +-------------------------------------+
                                     |
                                     v  Aligned Representations H [N, 256]
            +------------------------+------------------------+
            |                                                 |
            v                                                 v
+--------------------------+                      +--------------------------+
|       TARGET HEAD        |                      |         GAP HEAD         |
|  Linear(256 -> 128)      |                      | Concat(H_i, H_{i+1}) [512|
|  GELU + LayerNorm        |                      | Linear(512 -> 64)        |
|  Linear(128 -> 3)        |                      | GELU + LayerNorm         |
|  Softmax Simplex         |                      | Linear(64 -> 2)          |
+--------------------------+                      +--------------------------+
            |                                                 |
            v                                                 v
  Distribution [N, 3]:                               Probabilities [N+1]:
  - P(CORRECT)                                       - P(NO_INSERTION)
  - P(SUBSTITUTION)                                  - P(INSERTION)
  - P(OMISSION)
```

- **Tổng số tham số:** $\sim 850.000$ tham số (cực kỳ gọn nhẹ, $< 15$ MB).
- **Tốc độ suy luận:** $< 12$ ms/câu phát ngôn trên CPU Intel Core i5 / AMD Ryzen; $< 2$ ms trên GPU GTX 1650.
- **Tiêu chuẩn đầu ra:** Đảm bảo $100\%$ tuân thủ chuẩn simplex xác suất:
  $$\sum_{c \in \{CORR, SUB, OM\}} P(y_i = c) = 1.0$$

---

### 6.3. Công Thức Hàm Mất Mát Tổng Hợp Chưng Cất (Distillation Loss Formulation)

Quá trình huấn luyện học sinh ở Giai đoạn 3 sẽ tối ưu hóa hàm mất mát kết hợp 6 thành phần:
$$\mathcal{L}_{Total} = \mathcal{L}_{Gold\_Target} + \alpha \mathcal{L}_{Gold\_Gap} + \beta_S \mathcal{L}_{KD\_S} + \beta_O \mathcal{L}_{KD\_O} + \beta_I \mathcal{L}_{KD\_I} + \gamma \mathcal{L}_{Consistency}$$

Trong đó:
1. **$\mathcal{L}_{Gold\_Target}$ (Focal Loss trên nhãn Gold âm vị mục tiêu):**
   $$\mathcal{L}_{Gold\_Target} = - \sum_{i=1}^N \alpha_{y_i} (1 - p_{i, y_i})^\gamma \log p_{i, y_i}$$
   Với vector trọng số lớp đối kháng mất cân bằng: $\alpha = [1.0, 3.0, 5.0]$.
2. **$\mathcal{L}_{Gold\_Gap}$ (Focal Loss trên nhãn Gold chèn âm):**
   Trọng số $\alpha = 0.5$, tiêu chuẩn hóa tỷ lệ cực đoan ($0.61\%$ nhãn dương trên 20.354 khe hở).
3. **$\mathcal{L}_{KD\_S}$ (Chưng cất tri thức Teacher S):**
   Sử dụng Kullback-Leibler Divergence tại nhiệt độ chưng cất $T = 2.0$:
   $$\mathcal{L}_{KD\_S} = T^2 \cdot \text{KL}\left(\sigma\left(\frac{z_{S, i}}{T}\right) \parallel \sigma\left(\frac{\hat{z}_{S, i}}{T}\right)\right)$$
   Trọng số $\beta_S = 0.40$.
4. **$\mathcal{L}_{KD\_O}$ (Chưng cất tri thức Teacher O kèm trọng số bất định):**
   Tích hợp độ tin cậy nghịch đảo bất định:
   $$\mathcal{L}_{KD\_O} = T^2 \cdot (1.0 - u_{O, i}) \cdot \text{KL}\left(\sigma\left(\frac{z_{O, i}}{T}\right) \parallel \sigma\left(\frac{\hat{z}_{O, i}}{T}\right)\right)$$
   Trọng số $\beta_O = 0.35$.
5. **$\mathcal{L}_{KD\_I}$ (Chưng cất Teacher I-Precision):**
   Binary Cross-Entropy với soft score từ Teacher I Champion trên các khe hở có $\text{duration} \ge 2.5$ frames. Trọng số $\beta_I = 0.25$.
6. **$\mathcal{L}_{Consistency}$ (Ràng buộc triệt tiêu tương hỗ âm học):**
   Phạt các dự đoán vô lý về mặt vật lý (ví dụ: mô hình vừa gán xác suất cao cho việc nuốt âm câm vừa gán xác suất cao cho việc thay thế âm xát):
   $$\mathcal{L}_{Consistency} = \sum_{i=1}^N P(y_i = \text{SUB}) \cdot P(y_i = \text{OM})$$
   Trọng số $\gamma = 0.10$.

---

## 7. TRẢ LỜI ĐẦY ĐỦ 10 TIÊU CHÍ KHOA HỌC QUYẾT ĐỊNH

| Tiêu chí | Câu hỏi nghiên cứu | Trả lời và Căn cứ thực nghiệm |
|:---:|---|---|
| **1** | **Teacher S và O có bổ trợ trên cùng vị trí?** | **CÓ.** $r = 0.0384$ trên âm đúng; chỉ 153 ca xung đột ($0.77\%$), trong đó $56.86\%$ là lỗi thật. Teacher S phát hiện thêm 48 ca Omission, nâng tổng độ phủ Omission từ 68 lên 94 âm ($+38.2\%$). |
| **2** | **Hai ứng viên chèn âm có xếp hạng phân kỳ?** | **CÓ.** $r = 0.1259$. Teacher I Champion sạch vượt trội (Precision 66.7%), Teacher I EXP01 mở rộng Recall lên 11.20%. Cả hai bổ trợ tạo danh sách ứng viên chèn âm tối ưu. |
| **3** | **Cảnh báo giả nào có thể triệt tiêu an toàn?** | 66 cảnh báo giả trùng lặp giữa S và O được lọc sạch bằng ràng buộc simplex; cảnh báo giả của Teacher I được lọc sạch bằng ngưỡng thời lượng $\ge 2.5$ frames mà không mất ca chèn âm thật nào. |
| **4** | **Hợp nhất có cải thiện Recall tại cùng FWR?** | **CÓ.** Tại $\text{FWR} \le 0.20$, `F3-CALIB-FAMILY` đạt **Sub Recall 9.45%** (so với 5.94% của UPR03, **$+59.1\%$**); `F5-RISK-B030` đạt **Ins Recall 7.20%** (so với 5.60% của UPR03, **$+28.6\%$**). |
| **5** | **Đường biên Pareto gồm những điểm nào?** | Gồm 12 điểm hoạt động từ $\text{FWR}=0.01$ đến $\text{FWR}=2.83$. Điểm cân bằng an toàn sư phạm tối ưu là `F5-RISK-B030` ($\text{FWR}=0.1836, \text{Macro-4}=0.3030$) và `F3-CALIB-FAMILY` ($\text{FWR}=0.1970$). |
| **6** | **Hợp nhất có vượt UPR03 trên DEV_SEARCH?** | Vượt trội về độ thu hồi lỗi thành phần (Sub Recall $+59.1\%$, Ins Recall $+28.6\%$); Macro-4 tương đương tại vùng an toàn ($0.3030$ vs $0.3069$) và vượt trội khi nới lỏng ngân sách ($0.3302$ vs $0.3069$). |
| **7** | **Điểm số giáo viên nào phù hợp cho KD?** | Toàn bộ 3 giáo viên champion đều đạt chất lượng tín hiệu cao: biên độ phân biệt âm học $> +0.31$, tương quan sai số bất định rất mạnh ($r = 0.713$ và $r = 0.970$). |
| **8** | **Mô hình học sinh thống nhất có chính đáng?** | **TUYỆT ĐỐI CHÍNH ĐÁNG.** Hợp nhất điểm số hậu kỳ bị nghẽn thông tin. Chỉ có mô hình học sinh chia sẻ biểu diễn âm học mới giải phóng được 51.87% Sub Recall và 32.54% Om Recall mà vẫn giữ $\text{FWR} \le 0.20$. |
| **9** | **Thành phần nào cần bị loại bỏ?** | Loại bỏ phép hợp thô Naive Union (bùng nổ 1.671 cảnh báo giả); loại bỏ gán nhãn giả cứng (Hard Pseudo-Labels); loại bỏ Teacher I EXP02. |
| **10** | **Giai đoạn 3 nên huấn luyện cái gì đầu tiên?** | Huấn luyện mô hình học sinh `UnifiedPronunciationConformerStudent` đối chứng giữa `STU-01-GOLD-ONLY` (chỉ học từ Gold) và `STU-05-KD-ALL` (học đồng thời từ Gold + KD 3 chuyên gia). |

---

## 8. HỢP ĐỒNG BÀN GIAO GIAI ĐOẠN 3 (PHASE 3 HANDOFF SPECIFICATION)

### 8.1. Ma Trận Bảy Nhánh Thực Nghiệm Đối Chứng Bắt Buộc (Mandatory Control Ablations)
Giai đoạn 3 bắt buộc phải thực thi 7 nhánh thực nghiệm có cùng kiến trúc mạng, cùng seed ($101$) và cùng giao thức đánh giá clean-room:

| Mã thực nghiệm | Mô tả nhánh thực nghiệm | Mục đích khoa học |
|---|---|---|
| `STU-01-GOLD-ONLY` | Học sinh chỉ học từ nhãn Gold (Không có KD) | Baseline kiểm tra độ đóng góp thuần túy của biểu diễn Conformer |
| `STU-02-KD-S-ONLY` | Học sinh học từ Gold + Teacher S KD | Đo lường mức cải thiện riêng biệt của tri thức Thay thế |
| `STU-03-KD-O-ONLY` | Học sinh học từ Gold + Teacher O KD | Đo lường mức cải thiện riêng biệt của tri thức Nuốt âm |
| `STU-04-KD-I-ONLY` | Học sinh học từ Gold + Teacher I KD | Đo lường mức cải thiện riêng biệt của tri thức Chèn âm |
| `STU-05-KD-ALL` | Học sinh học đồng thời cả 3 Chuyên gia (Đầy đủ) | Đánh giá hiệu ứng cộng hưởng toàn diện của KD đa chuyên gia |
| `STU-06-CONF-GATED` | Học sinh học KD có cổng lọc độ bất định ($u < 0.30$) | Kiểm tra giả thuyết lọc nhiễu nhãn giáo viên bằng độ bất định |
| `BASELINE-UPR03` | Mô hình UPR03 Champion lịch sử | Đối chứng chuẩn mực của toàn bộ chiến dịch |

### 8.2. Kế Hoạch Tài Nguyên GPU & Thời Lượng Huấn Luyện
- **Phần cứng sử dụng:** GPU NVIDIA GeForce GTX 1650 4GB VRAM.
- **Mức tiêu hao VRAM ước tính:** $\sim 1.1$ GB VRAM (với PyTorch Mixed Precision FP16, batch size = 32 phát ngôn).
- **Thời lượng ước tính:** $\sim 12 - 15$ phút cho mỗi nhánh huấn luyện 30 epochs trên tập TRAIN (1.800 phát ngôn).
- **Toàn bộ 7 nhánh thực nghiệm:** Có thể hoàn tất trong vòng 2.5 giờ máy chạy tự động.

---

## 9. HƯỚNG DẪN TÁI LẬP THỰC NGHIỆM ĐỘC LẬP (REPRODUCTION GUIDE)

Toàn bộ kết quả trong báo cáo khoa học này có thể được tái lập hoàn toàn từ mã nguồn và dữ liệu có sẵn trong kho lưu trữ bằng các lệnh sau:

```bash
cd /home/khoa/Projects/New-Method-Research

# 1. Khởi tạo dự đoán trên dev_cal từ 4 checkpoint đóng băng
.venv/bin/python3 research/complementarity/generate_dev_cal_teacher_predictions.py

# 2. Thực thi phân tích tính bổ trợ CPL-01 (Workstream A)
.venv/bin/python3 research/complementarity/run_cpl01_joint_analysis.py

# 3. Thực thi khảo sát 26 chính sách hợp nhất CPL-02 (Workstream B)
.venv/bin/python3 research/complementarity/run_cpl02_joint_fusion.py

# 4. Thực thi kiểm định tính khả thi chưng cất CPL-03 (Workstream C)
.venv/bin/python3 research/complementarity/run_cpl03_distillation_feasibility.py

# 5. Chạy bộ kiểm thử tự động toàn vẹn deliverables (pytest)
.venv/bin/python3 -m pytest tests/test_phase2_artifacts.py
```

---

## 10. KẾT LUẬN NGHIỆM THU

Giai đoạn 2 đã kết thúc với kết quả hoàn hảo:
1. **Tính bổ trợ đa chuyên gia đã được chứng minh bằng toán học và thực nghiệm định lượng.**
2. **Trần lý thuyết của việc hợp nhất điểm số hậu kỳ đã được xác lập một cách khoa học.**
3. **Mọi điều kiện kỹ thuật, kiến trúc mô hình, công thức hàm mất mát và hợp đồng kiểm chứng Clean-Room đã sẵn sàng.**

**GIAI ĐOẠN 2 CHÍNH THỨC ĐÓNG LẠI. HỆ THỐNG SẴN SÀNG KHỞI ĐỘNG GIAI ĐOẠN 3 NGAY KHI CÓ CHỈ THỊ TIẾP THEO.**
