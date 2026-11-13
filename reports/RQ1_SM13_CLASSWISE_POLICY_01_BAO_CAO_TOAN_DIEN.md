# BÁO CÁO KHOA HỌC TOÀN DIỆN VÀ KIỂM TOÁN HỆ THỐNG
# CHIẾN DỊCH NGHIÊN CỨU CHÍNH SÁCH PHÂN LỚP ĐỘC LẬP SM13
## (LANGFENS SM13-CLASSWISE-POLICY-01 — Final Scientific Evaluation & Verification Report)

**Chiến dịch:** `LANGFENS-SM13-CLASSWISE-POLICY-01`  
**Đường dẫn Repo:** `/home/khoa/Projects/New-Method-Research`  
**Chế độ thực thi:** Thực thi tức thì, đánh giá thực nghiệm trực tiếp, không tái huấn luyện, không đổi kiến trúc, không truy cập `TEST_SEALED`.  
**Tác giả & Kiểm toán viên:** Kỹ sư Trưởng Nghiên cứu Speech ML & Kiểm toán viên Hệ thống Độc lập  
**Mô hình mục tiêu:** `LANGFENS-SM13-FUSION-V1` (Seed 101 Full-TRAIN Refit, đối chiếu chéo Seed 202, 303)  
**Tiêu chuẩn đánh giá:** EVAL-V3.1 (`UnifiedEvaluatorV3`, $N$ sự kiện âm vị mục tiêu và $N+1$ khoảng trống chèn)  
**Thư mục kết quả bàn giao:** `experiments/rq1_sm13_classwise_policy_01/`  
**Ngày hoàn thành:** 11 tháng 11 năm 2026  
**Trạng thái kiểm thử:** **100% ĐẠT (25/25 ca kiểm thử hồi quy toàn repo)**  
**Trạng thái Ranh giới Bảo mật:** **`TEST_SEALED` TIẾP TỤC ĐƯỢC KHÓA CHẶT 100% (MÃ BĂM SHA-256 BẤT BIẾN)**  
**Quyết định chính thức:** **`NO_GAIN` (KHÔNG ĐẠT LỢI THẾ CẢI TIẾN THỰC CHẤT)**

---

## 1. Tóm tắt Điều hành & Quyết định Chính thức

### 1.1 Quyết định: **`NO_GAIN`**

Theo đúng quy chuẩn phân loại của đề bài:
- **`PROMISING`:** Chính sách phân lớp cải thiện đánh đổi S/O/I dưới mức FWR tương đương.
- **`NO_GAIN`:** Việc tăng Recall-S gây suy giảm không thể chấp nhận đối với Omission/Insertion hoặc làm tăng FWR.
- **`INCONCLUSIVE`:** Bằng chứng không đầy đủ hoặc xuất xứ chính sách không hợp lệ.

**Cơ sở khoa học và kỹ thuật xác định phán quyết `NO_GAIN`:**
1. **Sự đánh đổi khốc liệt trên tập TRAIN OOF:** Trong quy trình kiểm định chéo ngoài mẫu (4-fold speaker-grouped OOF) trên tập huấn luyện TRAIN (12 người nói, 1.800 câu, 59.562 sự kiện), việc trích ngân sách cảnh báo giả để tăng tỷ lệ phát hiện Thay thế (Substitution Recall) từ ~3.3% lên ~5.7% (Seed 101) đã buộc ngưỡng của class Lược bỏ (Omission) phải siết chặt lại. Hậu quả là Omission Recall **suy giảm nghiêm trọng từ 16.38% xuống 9.81%** (mất mát tương đối tới 40%), kéo điểm tổng hợp Macro-3 F1 **sụt giảm từ 0.1679 xuống 0.1526** dưới cùng ràng buộc danh định $B \le 0.20$.
2. **Không có lợi thế tại mức FWR thực tế tương đương (Matched Realized FWR):** Khi kiểm soát chặt chẽ để FWR thực tế đạt đúng 0.2000, chính sách phân lớp đạt Macro-3 F1 = 0.1716 so với 0.1693 của chính sách ngưỡng chung (+0.0023, không có ý nghĩa thống kê), nhưng Substitution Recall thực tế lại *giảm nhẹ* (3.25% so với 3.41%) vì cả hai lớp đều bị đẩy vào vùng ngưỡng quá cao. Trung bình trên cả 3 seed (101, 202, 303), chênh lệch Macro-3 F1 chỉ đạt **+0.0003** (0.1726 so với 0.1723), hoàn toàn nằm trong phương sai ngẫu nhiên.
3. **Thất bại hoàn toàn khi chuyển giao chẩn đoán sang `DEV_CAL`:** Khi chuyển giao candidate chính sách phân lớp đóng băng từ TRAIN OOF ($\tau_S \approx 0.7570 - 0.8147$) sang tập chẩn đoán `DEV_CAL` (2 người nói `ERMS`, `THV`), **Substitution Recall vẫn bằng 0.00% tuyệt đối**.
4. **Phát hiện nguyên nhân gốc rễ (Root Cause Forensic Discovery):** Việc Substitution Recall bị triệt tiêu trên `DEV_CAL` không phải do ngưỡng chung quá khắt khe, mà do **thành phần tỉ số hợp lý chuỗi CTC (E4) bị nén biên độ nghiêm trọng trên người nói chưa từng thấy (unseen speakers)**, khiến $p_{\text{E4}, S} \le 0.6508$. Khi hợp nhất 50/50 với mạng neural, xác suất $p_S$ trên toàn bộ `DEV_CAL` bị chặn trên ở mức $0.6699$ (trên các sự kiện gán nhãn thực tế là Substitution, giá trị cực đại chỉ đạt $0.6538$). Do đó, bất kỳ ngưỡng $\tau_S$ hợp lệ nào được chọn từ TRAIN OOF ($\tau_S \ge 0.75$) đều cao hơn giá trị cực đại này, dẫn đến việc không có bất kỳ âm vị Thay thế nào được cảnh báo trên `DEV_CAL`.
5. **Tuân thủ tuyệt đối quy tắc Firewall:** Nếu cố tình hạ ngưỡng $\tau_S \le 0.50$ trực tiếp trên `DEV_CAL` để ép Recall-S > 0, khi đưa ngưỡng này áp dụng vào TRAIN OOF sẽ tạo ra một thảm họa cảnh báo giả với 1.467 cảnh báo sai (FWR = 0.815, vượt trần 0.20 gấp 4 lần). Hơn nữa, việc tinh chỉnh ngưỡng dựa trên kết quả `DEV_CAL` bị cấm tuyệt đối theo quy tắc phòng vệ chống rò rỉ dữ liệu.

---

## 2. Bối cảnh Nghiên cứu và Thiết lập Thử nghiệm

### 2.1 Chính sách Đóng băng Hiện tại trên DEV_CAL
Trước chiến dịch này, chính sách vận hành của `LANGFENS-SM13-FUSION-V1` (Seed 101 Full-TRAIN) được đóng băng tại `experiments/rq1_sm13_dev_calibration_01/FROZEN_CALIBRATED_POLICY.json`:
- Ngưỡng âm vị mục tiêu (Target): $\tau_T = 0.848521$
- Ngưỡng khoảng trống (Gap Insertion): $\tau_G = 0.873637$
- Hiệu năng tổng thể: Macro-3 Error F1 = **0.169221**; Joint FWR = **0.176667** cảnh báo giả/câu (53 cảnh báo giả / 300 câu)
- Hiệu năng phân lớp:
  - **Recall-S = 0.00%** (Precision = 0.00%, F1 = 0.0000, 0 True Positives trên 1.781 sự kiện Thay thế)
  - **Recall-O = 19.26%** (Precision = 67.42%, F1 = 0.299663)
  - **Recall-I = 12.75%** (Precision = 56.52%, F1 = 0.208000)

Sự bất đối xứng hoàn toàn giữa Recall-O (19.26%) và Recall-S (0.00%) đặt ra câu hỏi khoa học: *Liệu việc thiết lập ngưỡng độc lập $\tau_S$ và $\tau_O$ kèm cơ chế phân xử xung đột có thể khôi phục Recall-S mà không làm tổn hại Recall-O, Recall-I và giữ FWR $\le 0.20$?*

### 2.2 Quần thể Dữ liệu và Kiểm toán Ranh giới
- **Tập huấn luyện TRAIN (12 người nói L2-ARCTIC, 1.800 câu):**
  - Nhóm 0: `ABA`, `ASI`, `BWC` (450 câu)
  - Nhóm 1: `EBVS`, `HJK`, `HKK` (450 câu)
  - Nhóm 2: `HQTV`, `LXC`, `NJS` (450 câu)
  - Nhóm 3: `PNV`, `SKA`, `SVBI` (450 câu)
  - Sự kiện Target: 59.562 (Chính xác: 50.356; Thay thế: 7.381; Lược bỏ: 1.825)
  - Sự kiện Gap: 61.362 (Không chèn: 60.802; Chèn: 560)
- **Tập hiệu chuẩn chẩn đoán DEV_CAL (2 người nói, 300 câu):**
  - Người nói: `ERMS` (Tây Ban Nha, Nam, 150 câu), `THV` (Việt Nam, Nữ, 150 câu)
  - Sự kiện Target: 10.043 (Chính xác: 7.800; Thay thế: 1.781; Lược bỏ: 462)
  - Sự kiện Gap: 10.343 (Không chèn: 10.241; Chèn: 102)
- **Tập kiểm tra đóng kín TEST_SEALED:**
  - File manifest: `data/derived/l2arctic/manifests/test_sealed.jsonl`
  - Mã băm SHA-256 xác thực: `a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6`
  - **Trạng thái:** Tuyệt đối không đọc nhãn, không mở file, bảo vệ toàn vẹn 100%.

---

## 3. Bản chất Kỹ thuật: Phân tích Nén Điểm (Score Compression) trên DEV_CAL

Để hiểu tại sao Substitution Recall bằng 0% trên `DEV_CAL`, chúng tôi tiến hành phân tích phân rã các thành phần xác suất cấu thành:

Mô hình `SM13-FUSION-V1` sử dụng cơ chế hợp nhất cố định 50/50 giữa đầu dò Neural (13 tầng wav2vec2) và mô hình chuỗi E4 (CTC likelihood ratio):
$$p_{\text{fixed}} = 0.5 \cdot p_{\text{neural}} + 0.5 \cdot p_{\text{E4}}$$

Khi kiểm tra giá trị phân phối điểm trên tập `DEV_CAL`:
- **Đầu dò Neural trên DEV_CAL:**
  - $p_{\text{neural}, S} \in [0.0000, 0.9957]$
  - Điểm của mạng neural hoàn toàn bình thường, có độ tự tin cao trên cả người nói chưa thấy.
- **Thành phần CTC E4 trên DEV_CAL:**
  - $p_{\text{E4}, S} \in [0.0000, \mathbf{0.6508}]$
  - Điểm tỉ số hợp lý chuỗi CTC cho nhãn Substitution bị nén chặt dưới $0.6508$. Trên `DEV_CAL`, mô hình CTC không bao giờ tạo ra log-likelihood ratio đủ lớn để đơn hình E4 tiến gần đến 1.0 đối với lớp Thay thế.
- **Hệ quả trên xác suất hợp nhất $p_{\text{fixed}, S}$:**
  - Giá trị cực đại của $p_{\text{fixed}, S}$ trên toàn bộ 10.043 sự kiện của `DEV_CAL` là **0.6699**.
  - Riêng trên tập các sự kiện thực tế mang nhãn Substitution ($y_t = 1$), giá trị cực đại là **0.6538**. Phân vị 90 là 0.4675, phân vị 95 là 0.4903, phân vị 99 là 0.5453.
- **Ngược lại, trên tập TRAIN OOF:**
  - Mô hình CTC được đánh giá trên các ngữ cảnh âm học của tập huấn luyện, do đó $p_{\text{E4}, S}$ đạt tới $0.9999$.
  - Xác suất hợp nhất $p_{\text{fixed}, S}$ trên TRAIN OOF đạt tới **0.9920**.

**Kết luận khoa học quan trọng:**
Ngưỡng đóng băng danh định $\tau_T = 0.848521$ được hiệu chuẩn từ quy tắc tối ưu hóa F1/FWR chung. Do $p_{\text{fixed}, O}$ của lớp Lược bỏ trên `DEV_CAL` vẫn đạt tới $0.9923$, toàn bộ hạn mức cảnh báo sai của âm vị mục tiêu đã được "hấp thụ" bởi các sự kiện có $p_O$ cao, đẩy ngưỡng chung lên mức $0.8485$. Ở ngưỡng này, do $p_{\text{fixed}, S} \le 0.6538 < 0.8485$, việc Substitution Recall bằng 0% là một tất yếu toán học của hiện tượng nén điểm out-of-domain.

---

## 4. Thiết kế Chính sách Phân lớp Độc lập & Giải quyết Xung đột

### 4.1 Quy tắc Phân ngưỡng Độc lập
Với mỗi lớp cảnh báo lỗi $c \in \{\text{SUBSTITUTION}, \text{OMISSION}, \text{INSERTION}\}$, ta thiết lập ngân sách cảnh báo giả danh định $B_c$ sao cho:
$$B_S + B_O + B_I = B \le 0.20 \quad (\text{hoặc } B_{eff} \le 0.18)$$

Trên tập huấn luyện OOF nội bộ (inner OOF với 9 người nói, 1.350 câu):
- Ngưỡng $\tau_c$ và giá trị cắt hòa $\text{cutoff}_c$ được chọn là tiền tố lớn nhất thỏa mãn:
  $$\text{Số cảnh báo giả của lớp } c \le \lfloor B_c \times N_{\text{inner\_utts}} \rfloor$$
- Trong đó, định nghĩa cảnh báo giả (False Warning) tuân thủ chặt chẽ EVAL-V3.1:
  - Cảnh báo sai Substitution: $\mathbf{1}(\hat{y}_t = 1 \land y_t \ne 1)$ (gồm cả âm vị đúng bị báo nhầm $C \to S$ và âm vị lược bỏ bị báo nhầm $O \to S$).
  - Cảnh báo sai Omission: $\mathbf{1}(\hat{y}_t = 2 \land y_t \ne 2)$ (gồm cả $C \to O$ và $S \to O$).
  - Cảnh báo sai Insertion: $\mathbf{1}(\hat{y}_g = \text{True} \land y_g = 0)$ (khoảng trống bình thường bị báo nhầm là chèn).

### 4.2 Cơ chế Phân xử Xung đột (S/O Conflict Arbitration)
Khi một sự kiện thỏa mãn đồng thời cả hai điều kiện: $p_S \ge \tau_S$ và $p_O \ge \tau_O$, hệ thống sẽ kích hoạt bộ phân xử xác định. Năm quy tắc phân xử đã được lập trình và đối chiếu:
1. **Biên vượt ngưỡng (`score_margin`):** So sánh $m_S = p_S - \tau_S$ và $m_O = p_O - \tau_O$. Chọn nhãn có biên vượt ngưỡng lớn hơn.
2. **Biên chuẩn hóa (`norm_margin`):** So sánh $\frac{p_S - \tau_S}{1 - \tau_S}$ và $\frac{p_O - \tau_O}{1 - \tau_O}$.
3. **Xác suất thô (`prob_comparison`):** So sánh trực tiếp $p_S$ và $p_O$.
4. **Ưu tiên Thay thế (`priority_substitution`):** Luôn chọn Substitution để tối đa hóa khả năng cứu vãn Recall-S.
5. **Ưu tiên Lược bỏ (`priority_omission`):** Luôn chọn Omission.

**Định lý loại trừ xung đột ở vùng vận hành:**
Vì $p$ nằm trên đơn hình ($p_C + p_S + p_O = 1.0$), ta có bất đẳng thức:
$$p_S + p_O \le 1.0$$
Nếu $\tau_S + \tau_O > 1.0$, thì không thể tồn tại bất kỳ sự kiện nào có cả $p_S \ge \tau_S$ và $p_O \ge \tau_O$.

*Kiểm chứng thực nghiệm:*
- Ở ngân sách $B = 0.20$, các ngưỡng $\tau_S$ được chọn trên TRAIN OOF đều $\ge 0.7397$ và $\tau_O \ge 0.7916$.
- Tổng ngưỡng: $\tau_S + \tau_O \ge 1.5313 \gg 1.0000$.
- **Số lượng xung đột thực tế trên toàn bộ 59.562 sự kiện TRAIN: 0 sự kiện.**
- **Số lượng xung đột thực tế trên toàn bộ 10.043 sự kiện DEV_CAL: 0 sự kiện.**
- Cả 5 quy tắc phân xử đều cho kết quả đồng nhất 100% trong miền vận hành.

---

## 5. Kết quả Thực nghiệm Toàn diện

### 5.1 Khảo sát Phân bổ Ngân sách trên TRAIN OOF (Seed 101, Gộp 4 Folds)

Tổng số câu $N = 1.800$. Ngân sách chèn cố định $B_I = 0.040$. Ngân sách âm vị mục tiêu $B_T = 0.160$ được quét chia giữa $B_S$ và $B_O$:

| Chính sách | $B_S$ | $B_O$ | $\tau_S$ | $\tau_O$ | $\tau_I$ | Joint FWR | Tổng FW | Báo sai Loại | Macro-3 F1 | Recall-S | F1-S | Recall-O | F1-O | Recall-I | F1-I |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Ngưỡng chung (Shared)** | — | — | 0.8178 | 0.8178 | 0.7380 | **0.2017** | 363 | 52 | **0.1679** | 3.36% | 0.0643 | 16.38% | 0.2569 | 11.43% | 0.1826 |
| Phân lớp (0.02 S / 0.14 O) | 0.02 | 0.14 | 0.8669 | 0.7916 | 0.7380 | 0.2028 | 365 | 58 | 0.1661 | 1.83% | 0.0358 | 18.52% | 0.2799 | 11.43% | 0.1826 |
| Phân lớp (0.04 S / 0.12 O) | 0.04 | 0.12 | 0.8229 | 0.8099 | 0.7380 | 0.2022 | 364 | 51 | **0.1694** | 3.27% | 0.0626 | 16.93% | 0.2631 | 11.43% | 0.1826 |
| **Phân lớp (0.05 S / 0.11 O)** | 0.05 | 0.11 | 0.8147 | 0.8217 | 0.7380 | 0.2028 | 365 | 51 | 0.1679 | 3.59% | 0.0685 | 16.00% | 0.2526 | 11.43% | 0.1826 |
| Phân lớp (0.06 S / 0.10 O) | 0.06 | 0.10 | 0.8022 | 0.8333 | 0.7380 | 0.2028 | 365 | 49 | 0.1645 | 4.04% | 0.0765 | 14.58% | 0.2344 | 11.43% | 0.1826 |
| Phân lớp (0.08 S / 0.08 O) | 0.08 | 0.08 | 0.7769 | 0.8575 | 0.7380 | 0.2028 | 365 | 40 | 0.1587 | 4.81% | 0.0901 | 12.22% | 0.2035 | 11.43% | 0.1826 |
| Phân lớp (0.10 S / 0.06 O) | 0.10 | 0.06 | 0.7570 | 0.8826 | 0.7380 | 0.2106 | 379 | 32 | 0.1526 | 5.74% | 0.1061 | 9.81% | 0.1689 | 11.43% | 0.1826 |
| Phân lớp (0.12 S / 0.04 O) | 0.12 | 0.04 | 0.7397 | 0.9100 | 0.7380 | 0.2028 | 365 | 23 | 0.1431 | 6.58% | 0.1202 | 7.01% | 0.1265 | 11.43% | 0.1826 |

**Quy luật thực nghiệm nổi bật:**
- Số lượng cảnh báo sai loại (`wrong_type_warnings`: các trường hợp Substitution thực tế nhưng bị báo nhầm là Omission) giảm rõ rệt từ 58 xuống 23 khi tăng $B_S$.
- Tuy nhiên, độ dốc đánh đổi rất bất lợi: để tăng Recall-S thêm +3.22% (từ 3.36% lên 6.58%), hệ thống phải hy sinh tới **-9.37% Recall-O** (sụt từ 16.38% xuống 7.01%).
- Vì số lượng mẫu Omission ít hơn và có F1 cao hơn, việc sụt giảm Omission làm Macro-3 F1 lao dốc không phanh từ 0.1679 xuống 0.1431.

---

### 5.2 Đối chiếu Đa Seed và Đa Fold (Seeds 101, 202, 303)

So sánh chính sách ngưỡng chung danh định $B=0.20$ với chính sách phân lớp ứng viên tốt nhất ($B_S = 0.05, B_O = 0.11, B_I = 0.04$):

| Seed | Tập đánh giá | Chính sách | Joint FWR | FW Count | Wrong Type | Macro-3 F1 | Recall-S | Recall-O | Recall-I |
|---|---|---|---|---|---|---|---|---|---|
| **101** | Pooled (4 Folds) | Shared Baseline | 0.2017 | 363 | 52 | **0.1679** | 3.36% | 16.38% | 11.43% |
| 101 | Pooled (4 Folds) | Classwise Candidate | 0.2028 | 365 | 51 | **0.1679** | 3.59% | 16.00% | 11.43% |
| 101 | Fold 0 (`ABA, ASI, BWC`) | Shared Baseline | 0.2867 | 129 | 24 | 0.1834 | 2.89% | 19.33% | 16.42% |
| 101 | Fold 0 (`ABA, ASI, BWC`) | Classwise Candidate | 0.2911 | 131 | 24 | 0.1841 | 3.27% | 18.66% | 16.42% |
| 101 | Fold 1 (`EBVS, HJK, HKK`) | Shared Baseline | 0.2111 | 95 | 9 | 0.1368 | 2.30% | 9.30% | 15.75% |
| 101 | Fold 1 (`EBVS, HJK, HKK`) | Classwise Candidate | 0.2111 | 95 | 8 | 0.1378 | 2.65% | 8.86% | 15.75% |
| 101 | Fold 2 (`HQTV, LXC, NJS`) | Shared Baseline | 0.1933 | 87 | 11 | 0.1744 | 4.04% | 17.85% | 9.33% |
| 101 | Fold 2 (`HQTV, LXC, NJS`) | Classwise Candidate | 0.1978 | 89 | 11 | 0.1724 | 4.43% | 17.44% | 9.33% |
| 101 | Fold 3 (`PNV, SKA, SVBI`) | Shared Baseline | 0.1156 | 52 | 8 | 0.1772 | 4.22% | 20.30% | 4.03% |
| 101 | Fold 3 (`PNV, SKA, SVBI`) | Classwise Candidate | 0.1111 | 50 | 8 | 0.1772 | 4.02% | 20.30% | 4.03% |
| **202** | Pooled (4 Folds) | Shared Baseline | 0.2050 | 369 | 54 | 0.1663 | 2.09% | 16.55% | 12.86% |
| 202 | Pooled (4 Folds) | Classwise Candidate | 0.2100 | 378 | 49 | **0.1673** | 3.36% | 14.85% | 12.86% |
| **303** | Pooled (4 Folds) | Shared Baseline | 0.2033 | 366 | 47 | **0.1828** | 3.82% | 16.66% | 13.75% |
| 303 | Pooled (4 Folds) | Classwise Candidate | 0.2072 | 373 | 48 | 0.1825 | 3.90% | 16.49% | 13.75% |
| **Mean** | **Trung bình 3 Seeds** | **Shared Baseline** | **0.2033** | **366.0** | **51.0** | **0.1723** | **3.09%** | **16.53%** | **12.68%** |
| **Mean** | **Trung bình 3 Seeds** | **Classwise Candidate** | **0.2067** | **372.0** | **49.3** | **0.1726** | **3.62%** | **15.78%** | **12.68%** |

---

### 5.3 So sánh tại Mức FWR Thực tế Tương đương (Matched Realized FWR)

Khảo sát trên Seed 101 TRAIN OOF với các mốc cảnh báo sai được kiểm soát chính xác:

| Mốc Cảnh báo Sai | FWR Thực tế | Macro-3 F1 (Shared) | Macro-3 F1 (Classwise) | Recall-S (Shared) | Recall-S (Classwise) | Recall-O (Shared) | Recall-O (Classwise) |
|---|---|---|---|---|---|---|---|
| **180 FW** | **0.1000** | 0.1154 | 0.1155 | 1.53% | 1.83% | 10.25% | 9.86% |
| **270 FW** | **0.1500** | 0.1411 | 0.1414 | 2.47% | 2.55% | 13.42% | 13.37% |
| **360 FW** | **0.2000** | 0.1693 | 0.1716 | 3.41% | 3.25% | 16.33% | 17.15% |
| **424 FW** | **0.2356** | 0.1811 | 0.1820 | 3.96% | 3.55% | 17.75% | 18.74% |
| **540 FW** | **0.3000** | 0.2011 | 0.1988 | 4.95% | 3.89% | 20.16% | 21.48% |

**Nhận định kiểm toán:** Tại chính xác mốc ngân sách sư phạm $FWR = 0.2000$, chính sách phân lớp không mang lại lợi thế phát hiện lỗi Thay thế so với chính sách ngưỡng chung (Recall-S đạt 3.25% so với 3.41%).

---

### 5.4 Đánh giá Chuyển giao Chẩn đoán trên DEV_CAL (Fixed Policy Transfer)

Áp dụng các candidate chính sách đóng băng từ TRAIN OOF sang tập `DEV_CAL` mà **hoàn toàn không tái điều chỉnh**:

| Tên Chính sách | Nguồn gốc Ngưỡng | $\tau_S$ | $\tau_O$ | $\tau_I$ | Joint FWR | FW Count | Wrong Type | Macro-3 F1 | Recall-S | Recall-O | Recall-I |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Chính sách DEV_CAL Đóng băng** | DevCal Frozen | 0.8485 | 0.8485 | 0.8736 | **0.1767** | 53 | 7 | **0.1692** | **0.00%** | **19.26%** | 12.75% |
| Cand 1: Split 0.04 S / 0.12 O | Frozen TRAIN OOF | 0.8229 | 0.8099 | 0.8424 | 0.2400 | 72 | 10 | 0.2050 | **0.00%** | 25.54% | 15.69% |
| **Cand 2: Split 0.05 S / 0.11 O** | Frozen TRAIN OOF | 0.8147 | 0.8217 | 0.8424 | 0.2267 | 68 | 10 | 0.1997 | **0.00%** | 24.03% | 15.69% |
| Cand 3: Split 0.06 S / 0.10 O | Frozen TRAIN OOF | 0.8022 | 0.8333 | 0.8424 | 0.2033 | 61 | 9 | 0.1902 | **0.00%** | 21.43% | 15.69% |
| Cand 4: Split 0.08 S / 0.08 O | Frozen TRAIN OOF | 0.7769 | 0.8575 | 0.8424 | 0.1633 | 49 | 6 | 0.1810 | **0.00%** | 18.83% | 15.69% |
| Cand 5: B=0.18 Margined | Frozen TRAIN OOF | 0.8208 | 0.8267 | 0.8540 | 0.2033 | 61 | 9 | 0.1777 | **0.00%** | 21.65% | 12.75% |

**Phân tích chuyển giao chẩn đoán:**
- Bất kể phân bổ ngân sách nào được tối ưu trên TRAIN OOF, $\tau_S$ đều nằm trong khoảng $0.7769 - 0.8229$. Vì điểm $p_{\text{fixed}, S}$ cao nhất trên `DEV_CAL` chỉ là 0.6538, **Substitution Recall vẫn là 0.00% trên toàn bộ các candidate**.
- Hơn nữa, vì ngưỡng $\tau_O$ và $\tau_I$ lấy từ TRAIN OOF thấp hơn ngưỡng bảo thủ của DEV_CAL ($\tau_T = 0.8485, \tau_G = 0.8736$), số lượng cảnh báo giả bị đẩy vọt lên 68 - 72 cảnh báo, khiến FWR thực tế tăng lên $0.2267 - 0.2400$, vi phạm trần an toàn $0.2000$.
- Nếu chọn Cand 4 để giữ FWR $\le 0.20$ (FWR = 0.1633), Omission Recall lại sụt giảm xuống 18.83% (thấp hơn mức 19.26% của chính sách đóng băng ban đầu).

---

## 6. Phân tích Chi tiết Cấp độ Người nói (Speaker-Level Breakdown)

Dữ liệu chi tiết của 12 người nói TRAIN và 2 người nói DEV_CAL (Seed 101, mỗi người 150 câu):

| Người nói | Giọng mẹ đẻ | Số Target | Số Gap | AP-S | AP-O | AP-I | FWR (Shared) | Sub R (Shared) | Omi R (Shared) | FWR (Classwise) | Sub R (Classwise) | Omi R (Classwise) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ABA` | Ả Rập | 4.960 | 5.110 | 0.4491 | 0.3694 | 0.1802 | 0.2667 | 3.32% | 19.82% | 0.2800 | 3.99% | 19.37% |
| `ASI` | Ả Rập | 4.834 | 4.984 | 0.3721 | 0.4278 | 0.1610 | 0.2667 | 2.50% | 18.66% | 0.2600 | 2.80% | 17.54% |
| `BWC` | Trung Quốc | 5.054 | 5.204 | 0.4507 | 0.3470 | 0.1820 | 0.4733 | 2.81% | 19.54% | 0.4800 | 3.01% | 19.16% |
| `EBVS` | Tây Ban Nha | 4.942 | 5.092 | 0.3951 | 0.2600 | 0.1666 | 0.2267 | 2.62% | 11.26% | 0.2200 | 3.12% | 10.38% |
| `HJK` | Hàn Quốc | 4.930 | 5.080 | 0.3667 | 0.2311 | 0.2223 | 0.1600 | 1.83% | 7.97% | 0.1600 | 2.44% | 7.97% |
| `HKK` | Hàn Quốc | 4.976 | 5.126 | 0.3601 | 0.1883 | 0.1685 | 0.2467 | 2.41% | 8.21% | 0.2533 | 2.41% | 8.21% |
| `HQTV` | Việt Nam | 5.049 | 5.199 | 0.4632 | 0.4912 | 0.1558 | 0.1867 | 3.73% | 17.70% | 0.1867 | 3.86% | 17.70% |
| `LXC` | Trung Quốc | 5.124 | 5.274 | 0.4907 | 0.5186 | 0.1477 | 0.1933 | 3.59% | 18.52% | 0.1867 | 3.99% | 17.90% |
| `NJS` | Tây Ban Nha | 4.904 | 5.054 | 0.4842 | 0.4658 | 0.1488 | 0.2000 | 5.08% | 17.37% | 0.2067 | 5.46% | 17.01% |
| `PNV` | Việt Nam | 4.874 | 5.024 | 0.4354 | 0.3703 | 0.2505 | 0.2067 | 3.32% | 15.52% | 0.2067 | 3.90% | 14.54% |
| `SKA` | Ả Rập | 5.038 | 5.188 | 0.3846 | 0.3957 | 0.2081 | 0.2000 | 3.65% | 14.93% | 0.2067 | 3.94% | 14.93% |
| `SVBI` | Hindi | 4.877 | 5.027 | 0.4243 | 0.3475 | 0.1931 | 0.1867 | 3.03% | 14.49% | 0.1933 | 3.33% | 14.49% |
| **`ERMS` (DEV)** | Tây Ban Nha | 5.129 | 5.279 | 0.3697 | 0.4285 | 0.1706 | 0.2267 | 0.00% | 20.30% | 0.2667 | 0.00% | 23.31% |
| **`THV` (DEV)** | Việt Nam | 4.914 | 5.064 | 0.3712 | 0.4278 | 0.1751 | 0.1267 | 0.00% | 18.23% | 0.1867 | 0.00% | 24.74% |

---

## 7. Danh mục Bàn giao & Kiểm toán Toàn vẹn Hệ thống

Tất cả các tệp bàn giao theo yêu cầu đã được lưu trữ hoàn chỉnh tại thư mục `experiments/rq1_sm13_classwise_policy_01/`:

1. **`FINAL_REPORT.md`:** Báo cáo khoa học bằng tiếng Anh, phân tích cấu trúc, định lý toán học và kết luận kiểm toán.
2. **`POLICY_COMPARISON.csv`:** Bảng dữ liệu 336 dòng ghi nhận chi tiết toàn bộ các lần chạy, chính sách, độ phân bổ ngân sách, ngưỡng phân lớp, chỉ số FWR và P/R/F1 trên từng fold và toàn bộ tập TRAIN/DEV_CAL.
3. **`CLASSWISE_METRICS.csv`:** Bảng dữ liệu 76 dòng bóc tách chi tiết từng người nói cho 12 người nói TRAIN và 2 người nói DEV_CAL.
4. **`CANDIDATE_CLASSWISE_POLICY.json`:** Tệp cấu hình chính sách ứng viên định dạng JSON độc lập, lưu trữ đầy đủ tham số, ngưỡng, mã băm, và kết quả chẩn đoán.
5. **`REGRESSION_RESULTS.md`:** Biên bản kiểm toán kỹ thuật xác nhận 100% ca kiểm thử tự động đạt yêu cầu.
6. **`run_experiments.py`:** Mã nguồn Python thực thi toàn bộ pipeline thử nghiệm với tính năng tái lập bit-for-bit.
7. **`tests/test_sm13_classwise_policy_regression.py`:** Bộ kiểm thử hồi quy Pytest chuyên dụng (6 ca kiểm thử độc lập).

### Bảng Mã băm Kiểm toán Mật mã học (Cryptographic Verification)
| Tệp Artifact | Đường dẫn | Mã băm SHA-256 | Trạng thái |
|---|---|---|---|
| Chính sách DEV_CAL gốc | `experiments/rq1_sm13_dev_calibration_01/FROZEN_CALIBRATED_POLICY.json` | `dc7da3bcbc28918fcf39bdb1beba4ab990219d4467e56cdabbaebb0ac35e88e1` | **BẢO TỒN NGUYÊN VẸN** |
| Manifest TEST_SEALED | `data/derived/l2arctic/manifests/test_sealed.jsonl` | `a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6` | **KHÓA CHẶT 100%** |
| Dự đoán DEV_CAL NPZ | `experiments/rq1_sm13_dev_inference_01/predictions_sm13_seed101_dev_cal.npz` | `73f7cc8335a761bedb9a8f1346dd7416de648a98eebdd9a1a802c2d628776841` | **KHỚP CHÍNH XÁC** |
| Dự đoán TRAIN S101 NPZ | `experiments/rq1_method_select_02r/agent_a/predictions/predictions_sm13_seed101.npz` | `2c27addc691502b7758cb265b3e8f03648986d3e378d89a03ae87a5eb5cca094` | **KHỚP CHÍNH XÁC** |
| Dự đoán TRAIN S202 NPZ | `experiments/rq1_method_select_02r/agent_a/predictions/predictions_sm13_seed202.npz` | `bb5e4a9895a97247cb399c1e2a13046f7101408bc4d9b785425ad36e4d4b4ae0` | **KHỚP CHÍNH XÁC** |
| Dự đoán TRAIN S303 NPZ | `experiments/rq1_method_select_02r/agent_a/predictions/predictions_sm13_seed303.npz` | `85392bfaf88139376074641150b3be6974f11c7aac4246365c8581e262c5aec6` | **KHỚP CHÍNH XÁC** |

---

## 8. Khuyến nghị Khoa học và Bước đi Tiếp theo

Việc điều chỉnh chính sách ngưỡng vận hành (Operating Policy Calibration) về bản chất chỉ là **cơ chế phân xử ranh giới quyết định trên không gian xác suất hậu nghiệm đã tồn tại**, không thể bù đắp hay "tự sinh" ra độ tự tin phân loại nếu biểu diễn đặc trưng âm học cơ sở không cung cấp đủ tín hiệu phân tách.

Hiện tượng Recall-S bằng 0% trên `DEV_CAL` bắt nguồn từ sự sụp đổ biên độ điểm của tỉ số hợp lý chuỗi CTC (E4) khi gặp người nói chưa từng thấy. Do đó:
1. **Không tiếp tục lãng phí thời gian vào việc điều chỉnh ngưỡng hậu xử lý (Post-hoc Threshold Tuning):** Việc chia nhỏ ngân sách hay đổi quy tắc phân xử không thể giải quyết bài toán này.
2. **Tập trung nghiên cứu vào Khâu Trước Hợp nhất (Pre-Fusion Normalization & Calibration):**
   - Chuẩn hóa phân phối điểm tỉ số hợp lý $SLLR$ và $OLLR$ thích ứng theo từng người nói (Speaker-adaptive temperature scaling hoặc Platt scaling).
   - Huấn luyện lại trọng số hợp lý chuỗi CTC hoặc áp dụng cơ chế Attention-based Fusion động thay vì trọng số tĩnh cố định 50/50.
3. **Bảo toàn nguyên vẹn Chính sách Đóng băng `FROZEN_CALIBRATED_POLICY.json`:** Giữ nguyên chính sách hiện tại cho các đánh giá prospective tiếp theo, tuyệt đối không triển khai candidate phân lớp do không đem lại lợi thế thực chất (`NO_GAIN`).
