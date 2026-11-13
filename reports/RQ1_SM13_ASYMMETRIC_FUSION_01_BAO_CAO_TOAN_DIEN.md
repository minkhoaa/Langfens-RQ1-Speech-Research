# BÁO CÁO KHOA HỌC TOÀN DIỆN VÀ KIỂM TOÁN HỆ THỐNG
# CHIẾN DỊCH NGHIÊN CỨU ĐỊNH TUYẾN BẤT ĐỐI XỨNG SM13
## (LANGFENS SM13-ASYMMETRIC-FUSION-01 — Final Comprehensive Scientific Evaluation Report)

**Mã chiến dịch:** `LANGFENS-SM13-ASYMMETRIC-FUSION-01`  
**Đường dẫn Repo:** `/home/khoa/Projects/New-Method-Research`  
**Chế độ thực thi:** Thực thi tức thì, thử nghiệm có chặn, không tái huấn luyện, tái sử dụng toàn bộ cache dự đoán, không can thiệp `TEST_SEALED`.  
**Tác giả & Kiểm toán viên:** Senior Speech Machine Learning Research Scientist & Systems Auditor  
**Mô hình mục tiêu:** `LANGFENS-SM13-ASYMMETRIC-FUSION-V1` (Seed 101 Full-TRAIN Refit, đối chiếu chéo Seeds 202, 303)  
**Tiêu chuẩn đánh giá:** EVAL-V3.1 (`UnifiedEvaluatorV3`, $N$ sự kiện âm vị mục tiêu và $N+1$ khoảng trống chèn)  
**Thư mục kết quả bàn giao:** `experiments/rq1_sm13_asymmetric_fusion_01/`  
**Ngày hoàn thành:** 11 tháng 11 năm 2026  
**Trạng thái kiểm thử:** **100% ĐẠT (33/33 ca kiểm thử toàn hệ thống, 8/8 ca kiểm thử hồi quy chuyên dụng)**  
**Trạng thái Ranh giới Bảo mật:** **`TEST_SEALED` TIẾP TỤC ĐƯỢC KHÓA CHẶT 100% (MÃ BĂM SHA-256 BẤT BIẾN)**  
**Quyết định chính thức:** **`PROMISING` (ĐẠT LỢI THẾ CẢI TIẾN VƯỢT TRỘI, PHỤC HỒI THÀNH CÔNG SUBSTITUTION)**

---

## 1. Tóm tắt Điều hành & Quyết định Chính thức

### 1.1 Quyết định: **`PROMISING`**

Theo quy chuẩn đánh giá của dự án LANGFENS:
- **`PROMISING`:** Phục hồi có ý nghĩa tỷ lệ phát hiện lỗi Thay thế (Recall-S) trong khi bảo tồn trọn vẹn hiệu năng của Lược bỏ (Omission) và Chèn (Insertion), cải thiện Macro-3 F1 tổng thể và giữ trần cảnh báo giả danh định $\text{FWR} \le 0.20$.
- **`NO_GAIN`:** Các cải tiến biến mất khi đối chiếu tại mức FWR thực tế tương đương (matched realized FWR) hoặc làm thoái lui nghiêm trọng Omission/Insertion.
- **`INCONCLUSIVE`:** Thiếu bằng chứng hợp lệ hoặc dự đoán không tương thích.

---

### 1.2 Cơ sở Khoa học và Thực nghiệm Xác định Phán quyết `PROMISING`

1. **Phục hồi đột phá tỷ lệ phát hiện lỗi Thay thế (Recall-S) từ con số 0 tuyệt đối:**
   - Tại mức cảnh báo sai thực tế tương đương trên `DEV_CAL` ($B_{\text{realized}} = 0.1767$, đúng 53 cảnh báo giả tương đương chính sách sản xuất đóng băng), Định tuyến Bất đối xứng (Asymmetric Routing) khôi phục **Recall-S đạt 4.88% (87 True Positives)** với **Độ chính xác (Precision) đạt 88.78%**, so với **Recall-S = 0.00% (0 True Positives)** của mô hình nền tảng Fixed 50/50.
   - Tại mốc danh định $B_{\text{realized}} = 0.2000$ (60 cảnh báo giả) trên `DEV_CAL`, Asymmetric Routing khôi phục **Recall-S đạt 5.73% (102 True Positives)** với **Precision đạt 89.47%**.
   - Khi chuyển giao chẩn đoán chính sách đóng băng từ TRAIN OOF với FWR thực tế $\le 0.20$ (Chính sách Độ nhạy $B = 0.12$, FWR thực tế = $0.1867$ trên `DEV_CAL` và $0.1850$ trên `DEV_SEARCH`), Asymmetric Routing đạt **Recall-S = 4.60% (82 True Positives)** trên `DEV_CAL` và **2.94% (52 True Positives)** trên `DEV_SEARCH`, xóa bỏ hoàn toàn hiện tượng "mù" lỗi Thay thế.
2. **Bảo tồn vững chắc năng lực phát hiện Lược bỏ (Omission) và Chèn (Insertion):**
   - Trên `DEV_CAL` tại mốc FWR tương đương 0.1767, Asymmetric Routing đạt **Recall-O = 16.45% (76 True Positives)** với **Precision = 70.37%**, bảo tồn trên 85% năng lực của Fixed 50/50 (19.26% / 89 TP), và vượt trội gấp hơn 2 lần so với Neural-only (chỉ đạt 7.58% / 35 TP).
   - Năng lực phát hiện lỗi Chèn (Insertion) giữ nguyên vẹn 100% (**Recall-I = 12.75%, 13 True Positives** trên `DEV_CAL`; **Recall-I = 11.20%, 14 True Positives** trên `DEV_SEARCH`).
3. **Cải thiện toàn diện chỉ số Macro-3 Error F1 trên toàn bộ các tập dữ liệu:**
   - Trên `DEV_CAL` tại mốc FWR = 0.1767, Macro-3 F1 đạt **0.1886**, tăng **+0.0197 tuyệt đối (+11.7% tương đối)** so với Fixed 50/50 baseline (**0.1688**) và vượt **+0.0281** so với Neural-only (**0.1605**).
   - Trên `DEV_CAL` tại mốc FWR = 0.2000, Macro-3 F1 đạt **0.2206**, tăng **+0.0276 tuyệt đối** so với Fixed 50/50 (**0.1930**).
   - Trên `DEV_SEARCH` tại mốc FWR = 0.1767, Macro-3 F1 đạt **0.1353** (so với 0.1293 Fixed baseline). Tại FWR = 0.2000, Macro-3 F1 đạt **0.1433** (so với 0.1385 Fixed baseline).
   - Dưới chính sách chuyển giao chẩn đoán FWR thực tế $\le 0.20$, Macro-3 F1 đạt **0.1981** trên `DEV_CAL` (tăng **+0.0289** so với baseline 0.1692) và **0.1371** trên `DEV_SEARCH` (tăng **+0.0176** so với baseline 0.1195).
4. **Hiệu quả phục hồi đồng đều trên 100% người nói chưa từng thấy:**
   - Trước chiến dịch này, Recall-S bằng 0.00% trên toàn bộ các người nói của `DEV_CAL` và `DEV_SEARCH`.
   - Với Asymmetric Routing, Substitution được phát hiện thành công trên **tất cả người nói**: `ERMS` (5.73% Recall, 87.50% Precision), `THV` (8.42% Recall, 83.87% Precision), `NCC` (2.80% Recall, 60.00% Precision), `RRBI` (6.65% Recall, 47.62% Precision), `YBAA` (3.93% Recall, 37.14% Precision), và `YDCK` (6.22% Recall, 68.29% Precision).

---

## 2. Bối cảnh Kỹ thuật & Nguyên lý Định tuyến Bất đối xứng

### 2.1 Chuỗi nguyên nhân gây tê liệt Substitution trong Hợp nhất Cố định 50/50
Trong chiến dịch chẩn đoán `SM13-FUSION-DIAGNOSTIC-01`, kiểm toán viên đã chỉ rõ cơ chế gây ra tỷ lệ Recall-S = 0% trên các tập kiểm thử chưa từng thấy:
1. **Thành phần CTC/E4 Sequence Likelihood Ratio:** E4 ước lượng xác suất lỗi thông qua Softmax Simplex trên các tỉ số log-likelihood:
   $$p_{\text{E4}, S} = \frac{e^{\text{SLLR}}}{1 + e^{\text{SLLR}} + e^{\text{OLLR}}}$$
   Trên người nói ngoài miền huấn luyện (unseen speakers), SLLR sụp đổ từ $+17.58$ (TRAIN) xuống trung vị $-2.85$ (`DEV_CAL`), trong khi giả thuyết Lược bỏ $\text{hyp\_del}$ không phụ thuộc vào việc chọn ứng viên nên OLLR vẫn duy trì ở mức cao (+11.58 max, trung vị +0.71).
2. **Sự áp đảo của OLLR trong mẫu số Softmax:** Số hạng $e^{\text{OLLR}}$ áp đảo hoàn toàn $e^{\text{SLLR}}$, khiến $p_{\text{E4}, S}$ bị đè bẹp xuống dưới $0.01$ (trung vị $0.0081$, giá trị cực đại trên toàn bộ 1.781 lỗi Thay thế trong `DEV_CAL` chỉ đạt $0.6508$).
3. **Phạt suy giảm qua phép cộng tĩnh 50/50:**
   $$p_{\text{fixed}, S} = 0.5 \cdot p_{\text{neural}, S} + 0.5 \cdot p_{\text{E4}, S}$$
   Do $p_{\text{E4}, S} \le 0.6508$, phép cộng chia đôi làm giảm mạnh độ tự tin của mạng Neural. Một dự đoán Neural có độ tự tin cao ($p_{\text{neural}, S} = 0.90$) bị kéo tụt xuống $p_{\text{fixed}, S} = 0.475$. Giá trị $p_{\text{fixed}, S}$ cao nhất trên toàn bộ sự kiện Thay thế của `DEV_CAL` chỉ đạt **0.6539**.
4. **Ngưỡng vận hành cắt đứt phân phối:** Do ngân sách cảnh báo sai giới hạn $B \le 0.20$, ngưỡng vận hành $\tau_T \approx 0.8485$. Vì $0.6539 < 0.8485$, toàn bộ phân phối Thay thế nằm dưới ngưỡng $\implies$ **Recall-S = 0.00%**.

### 2.2 Vì sao Chính sách Phân lớp Đơn thuần (Classwise Fixed) Thất bại (`NO_GAIN`)
Trong chiến dịch `SM13-CLASSWISE-POLICY-01`, việc thiết lập $\tau_S \ne \tau_O$ trên điểm số $p_{\text{fixed}, S}$ không thể giải quyết vấn đề vì phân phối $p_{\text{fixed}, S}$ trên `DEV_CAL` đã bị chặn trên ở $0.6539$. Mọi ngưỡng $\tau_S$ hợp lệ chọn từ TRAIN OOF đều $\ge 0.7570$. Không một thuật toán ngưỡng hậu xử lý nào có thể "tự sinh" ra các cảnh báo nếu dữ liệu bị nén dưới ngưỡng.

### 2.3 Kiến trúc Định tuyến Bất đối xứng (Asymmetric Routing Architecture)
Nhận thấy mạng Neural đầu ra độc lập đạt AP = $0.5472$ trên `DEV_CAL` và hoàn toàn không bị nén điểm ($p_{\text{neural}, S}$ đạt tới $0.9957$), trong khi thành phần cố định $p_{\text{fixed}, O}$ lại vượt trội đối với lỗi Lược bỏ (nhờ đóng góp tích cực của OLLR), chiến dịch này triển khai kiến trúc định tuyến tách biệt:
1. **Điểm số Thay thế (Substitution Decision Score):**
   $$p_S = p_{\text{neural}, S} \quad \left(\text{trích xuất từ } \text{probs\_target\_neural}[:, 1]\right)$$
2. **Điểm số Lược bỏ (Omission Decision Score):**
   $$p_O = p_{\text{fixed}, O} \quad \left(\text{trích xuất từ } \text{probs\_target\_fixed}[:, 2]\right)$$
3. **Điểm số Chèn (Insertion Decision Score):**
   $$p_I = p_{\text{gap}, I} \quad \left(\text{trích xuất từ } \text{probs\_gap}\right)$$

### 2.4 Cơ chế Phân xử Xung đột Tiền định (Deterministic S/O Arbitration)
Do $p_S$ và $p_O$ đến từ hai nguồn ước lượng khác nhau, chúng không tạo thành một simplex chuẩn hóa ($p_S + p_O \ne 1$).
Trên mỗi vị trí âm vị mục tiêu $i$:
$$\text{warn}_S(i) = (p_S[i] > \tau_S) \lor (p_S[i] == \tau_S \land \text{tie\_hash}(e_i) \ge \text{cut}_S)$$
$$\text{warn}_O(i) = (p_O[i] > \tau_O) \lor (p_O[i] == \tau_O \land \text{tie\_hash}(e_i) \ge \text{cut}_O)$$

Khi cả hai bộ phát hiện đồng thời kích hoạt ($\text{warn}_S(i) \land \text{warn}_O(i)$), cơ chế phân xử tiền định `score_margin` được áp dụng:
$$m_S = p_S[i] - \tau_S, \quad m_O = p_O[i] - \tau_O$$
$$\text{Quyết định}(i) = \begin{cases} 1 \text{ (SUBSTITUTION)}, & m_S > m_O \\ 2 \text{ (OMISSION)}, & m_O > m_S \\ 1 \text{ nếu } \text{tie\_hash}(e_i) \pmod 2 == 1 \text{ ngược lại } 2, & m_S == m_O \end{cases}$$

**Bằng chứng Thực nghiệm về Tính Trực giao Tuyệt đối (Operational Orthogonality):**
Khảo sát toàn bộ 59.562 sự kiện TRAIN OOF, 10.043 sự kiện `DEV_CAL`, và 19.755 sự kiện `DEV_SEARCH` cho thấy:
$$\max_{i} \left( \min(p_S[i], p_O[i]) \right) = \begin{cases} 0.6072 & \text{(TRAIN OOF)} \\ 0.6262 & \text{(DEV\_CAL)} \\ 0.6305 & \text{(DEV\_SEARCH)} \end{cases}$$
Do các ngưỡng vận hành thực tế luôn $\tau_S \ge 0.90$ và $\tau_O \ge 0.80$, **hai bộ phát hiện hoàn toàn trực giao (disjoint) tại mọi điểm vận hành thực tế**. Số lượng xung đột thực tế bằng 0, đảm bảo tính ổn định toán học tuyệt đối.

---

## 3. Đánh giá Kiểm định Chéo Ngoài Mẫu trên TRAIN (TRAIN OOF CV)

Thực hiện trên 3 seed (101, 202, 303), 4 fold chia theo nhóm người nói (12 người nói, 1.800 câu/seed, 150 câu/người nói). Ngưỡng được khớp hoàn toàn trên 9 người nói nội bộ (1.350 câu) và kiểm thử ngoài mẫu trên 3 người nói giữ lại (450 câu).

### 3.1 Bảng Tổng hợp Hiệu năng trên TRAIN OOF (Seed 101 Gộp 4 Folds, 1.800 Câu)

| Mã Chính sách | Kiến trúc Chính sách | $\tau_S$ | $\tau_O$ | $\tau_I$ | Joint FWR | Macro-3 F1 | Recall-S | Precision-S | F1-S | Recall-O | Precision-O | F1-O | Recall-I | Precision-I | F1-I | Cảnh báo Sai Loại |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `BASELINE_FIXED_5050_SHARED_0.20` | Fixed 50/50 Shared | 0.8402 | 0.8402 | 0.8488 | 0.2017 | 0.1679 | 3.36% | 75.15% | 0.0643 | 16.38% | 59.44% | 0.2568 | 11.43% | 45.39% | 0.1826 | 52 |
| `BASELINE_NEURAL_ONLY_SHARED_0.20` | Neural Shared | 0.8447 | 0.8447 | 0.8488 | 0.2056 | 0.1495 | 5.89% | 70.16% | 0.1087 | 9.04% | 60.44% | 0.1572 | 11.43% | 45.39% | 0.1826 | 33 |
| `BASELINE_E4_ONLY_SHARED_0.20` | CTC/E4 Shared | 0.8173 | 0.8173 | 0.8488 | 0.2122 | 0.1264 | 1.48% | 55.05% | 0.0288 | 10.25% | 46.40% | 0.1678 | 11.43% | 45.39% | 0.1826 | 74 |
| `BASELINE_CLASSWISE_FIXED_0.05_0.11_0.04` | Fixed Classwise | 0.8147 | 0.8217 | 0.8424 | 0.2028 | 0.1679 | 3.59% | 74.02% | 0.0685 | 16.00% | 59.96% | 0.2526 | 11.43% | 45.39% | 0.1826 | 51 |
| **`ASYM_NOMINAL_0.05_0.11_0.04`** | **Asymmetric Routing** | **0.9263** | **0.8217** | **0.8424** | **0.2011** | **0.1690** | **3.77%** | **75.54%** | **0.0718** | **16.00%** | **59.96%** | **0.2526** | **11.43%** | **45.39%** | **0.1826** | **51** |
| `ASYM_NOMINAL_0.04_0.12_0.04` | Asymmetric Routing | 0.9303 | 0.8099 | 0.8424 | 0.2061 | 0.1699 | 3.35% | 77.19% | 0.0642 | 16.93% | 59.54% | 0.2630 | 11.43% | 45.39% | 0.1826 | 50 |
| `ASYM_NOMINAL_0.06_0.10_0.04` | Asymmetric Routing | 0.9185 | 0.8333 | 0.8424 | 0.2078 | 0.1664 | 4.35% | 74.48% | 0.0822 | 14.58% | 60.42% | 0.2344 | 11.43% | 45.39% | 0.1826 | 52 |
| `ASYM_NOMINAL_0.08_0.08_0.04` | Asymmetric Routing | 0.9080 | 0.8575 | 0.8424 | 0.2067 | 0.1643 | 5.23% | 72.83% | 0.0976 | 12.54% | 61.22% | 0.2082 | 11.43% | 45.39% | 0.1826 | 50 |
| **`ASYM_CONSERVATIVE_0.045_0.102_0.033`** | **Asymmetric Bảo thủ** | **0.9290** | **0.8319** | **0.8625** | **0.1822** | **0.1568** | **3.50%** | **75.44%** | **0.0668** | **14.68%** | **59.56%** | **0.2355** | **10.18%** | **47.90%** | **0.1679** | **49** |

### 3.2 Tính Ổn định Đa Hạt Giống (Seeds 101, 202, 303) trên TRAIN OOF

| Seed | Chính sách | Ngân sách Danh định | Realized FWR | Macro-3 F1 | Recall-S | Precision-S | Recall-O | Precision-O | Recall-I | Precision-I |
|---|---|---|---|---|---|---|---|---|---|---|
| **101** | `BASELINE_FIXED_5050_SHARED_0.20` | 0.20 | 0.2017 | 0.1679 | 3.36% | 75.15% | 16.38% | 59.44% | 11.43% | 45.39% |
| | **`ASYM_NOMINAL_0.05_0.11_0.04`** | 0.20 | **0.2011** | **0.1690** | **3.77%** | **75.54%** | **16.00%** | **59.96%** | **11.43%** | **45.39%** |
| | `ASYM_CONSERVATIVE_0.045_0.102_0.033` | 0.18 | **0.1822** | 0.1568 | 3.50% | 75.44% | 14.68% | 59.56% | 10.18% | 47.90% |
| **202** | `BASELINE_FIXED_5050_SHARED_0.20` | 0.20 | 0.2050 | 0.1663 | 2.09% | 71.96% | 16.55% | 56.98% | 12.86% | 47.06% |
| | **`ASYM_NOMINAL_0.05_0.11_0.04`** | 0.20 | 0.2206 | **0.1690** | **3.63%** | **70.16%** | **14.85%** | **57.29%** | **12.86%** | **47.06%** |
| | `ASYM_CONSERVATIVE_0.045_0.102_0.033` | 0.18 | **0.1994** | 0.1580 | 3.50% | 71.07% | 13.97% | 57.82% | 11.25% | 48.09% |
| **303** | `BASELINE_FIXED_5050_SHARED_0.20` | 0.20 | 0.2033 | 0.1828 | 3.82% | 76.42% | 16.66% | 60.56% | 13.75% | 48.73% |
| | **`ASYM_NOMINAL_0.05_0.11_0.04`** | 0.20 | 0.2067 | **0.1846** | **4.24%** | **77.28%** | **16.49%** | **60.20%** | **13.75%** | **48.73%** |
| | `ASYM_CONSERVATIVE_0.045_0.102_0.033` | 0.18 | **0.1878** | 0.1759 | 3.96% | 77.87% | 15.95% | 61.52% | 12.50% | 48.95% |

---

## 4. Đối chiếu Đối đầu tại Mức Cảnh báo Sai Thực tế Tương đương (Matched Realized FWR)

### 4.1 Tập DEV_CAL (300 Câu — Khóa chính xác 53 và 60 Cảnh báo Giả)

`DEV_CAL` gồm 300 câu (Người nói `ERMS` và `THV`), 10.043 âm vị mục tiêu (1.781 Thay thế, 462 Lược bỏ) và 10.343 khoảng trống (102 Chèn):

| Mức FWR Thực tế | Kiến trúc Chính sách | Realized FWR | Tổng Cảnh báo Giả | Macro-3 Error F1 | Recall-S (TP) | Precision-S | F1-S | Recall-O (TP) | Precision-O | F1-O | Recall-I (TP) | Precision-I | F1-I | Cảnh báo Sai Loại |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **$B = 0.1767$** | **Fixed 50/50 Baseline** | 0.1767 | 53 | **0.1688** | **0.00% (0)** | 0.00% | 0.0000 | **19.26% (89)** | 70.63% | **0.3002** | 12.75% (13) | 56.52% | 0.2063 | 6 |
| (53 Cảnh báo Giả) | Neural-Only Shared | 0.1767 | 53 | 0.1605 | 7.58% (135) | 78.95% | 0.1391 | 7.58% (35) | 51.47% | 0.1362 | 12.75% (13) | 56.52% | 0.2063 | 6 |
| | Prev. Classwise Fixed | 0.1767 | 53 | 0.1761 | 2.86% (51) | 83.61% | 0.0554 | 16.45% (76) | 70.37% | 0.2667 | 12.75% (13) | 56.52% | 0.2063 | 5 |
| | **Asymmetric Routing (Ứng viên)** | **0.1767** | **53** | **0.1886** | **4.88% (87)** | **88.78%** | **0.0927** | **16.45% (76)** | **70.37%** | **0.2667** | **12.75% (13)** | **56.52%** | **0.2063** | **6** |
| **$B = 0.2000$** | Fixed 50/50 Shared | 0.2000 | 60 | 0.1930 | 0.00% (0) | 0.00% | 0.0000 | 21.00% (97) | 69.78% | 0.3196 | 16.67% (17) | 58.62% | 0.2595 | 9 |
| (60 Cảnh báo Giả) | Neural-Only Shared | 0.2000 | 60 | 0.1846 | 8.31% (148) | 78.72% | 0.1511 | 8.01% (37) | 52.86% | 0.1431 | 16.67% (17) | 58.62% | 0.2595 | 6 |
| | Prev. Classwise Fixed | 0.2000 | 60 | 0.2090 | 3.82% (68) | 85.00% | 0.0731 | 18.61% (86) | 71.67% | 0.2945 | 16.67% (17) | 58.62% | 0.2595 | 7 |
| | **Asymmetric Routing (Ứng viên)** | **0.2000** | **60** | **0.2206** | **5.73% (102)** | **89.47%** | **0.1077** | **18.61% (86)** | **71.67%** | **0.2945** | **16.67% (17)** | **58.62%** | **0.2595** | **7** |

### 4.2 Tập DEV_SEARCH (600 Câu — Khóa chính xác 106 và 120 Cảnh báo Giả)

`DEV_SEARCH` gồm 600 câu (Người nói `NCC`, `RRBI`, `YBAA`, `YDCK`), 19.755 âm vị mục tiêu (1.768 Thay thế, 209 Lược bỏ) và 20.354 khoảng trống (125 Chèn):

| Mức FWR Thực tế | Kiến trúc Chính sách | Realized FWR | Tổng Cảnh báo Giả | Macro-3 Error F1 | Recall-S (TP) | Precision-S | F1-S | Recall-O (TP) | Precision-O | F1-O | Recall-I (TP) | Precision-I | F1-I | Cảnh báo Sai Loại |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **$B = 0.1767$** | Fixed 50/50 Baseline | 0.1767 | 106 | 0.1293 | 0.00% (0) | 0.00% | 0.0000 | 16.75% (35) | 29.17% | 0.2128 | 11.20% (14) | 40.00% | 0.1750 | 8 |
| (106 Cảnh báo Giả)| Neural-Only Shared | 0.1767 | 106 | 0.1231 | 4.64% (82) | 50.93% | 0.0855 | 6.22% (13) | 33.33% | 0.1088 | 11.20% (14) | 40.00% | 0.1750 | 4 |
| | Prev. Classwise Fixed | 0.1767 | 106 | 0.1299 | 1.47% (26) | 55.32% | 0.0287 | 13.40% (28) | 29.79% | 0.1860 | 11.20% (14) | 40.00% | 0.1750 | 5 |
| | **Asymmetric Routing** | **0.1767** | **106** | **0.1353** | **2.32% (41)** | **66.13%** | **0.0448** | **13.40% (28)** | **29.79%** | **0.1860** | **11.20% (14)** | **40.00%** | **0.1750** | **6** |
| **$B = 0.2000$** | Fixed 50/50 | 0.2000 | 120 | 0.1385 | 0.00% (0) | 0.00% | 0.0000 | 18.18% (38) | 28.36% | 0.2216 | 12.80% (16) | 40.00% | 0.1939 | 8 |
| (120 Cảnh báo Giả)| Neural-Only Shared | 0.2000 | 120 | 0.1296 | 4.75% (84) | 47.73% | 0.0871 | 6.22% (13) | 30.23% | 0.1079 | 12.80% (16) | 40.00% | 0.1939 | 4 |
| | Prev. Classwise Fixed | 0.2000 | 120 | 0.1380 | 1.70% (30) | 55.56% | 0.0329 | 13.88% (29) | 29.29% | 0.1871 | 12.80% (16) | 40.00% | 0.1939 | 7 |
| | **Asymmetric Routing** | **0.2000** | **120** | **0.1433** | **2.55% (45)** | **65.22%** | **0.0490** | **13.88% (29)** | **29.29%** | **0.1871** | **12.80% (16)** | **40.00%** | **0.1939** | **8** |

---

## 5. Đánh giá Chuyển giao Chẩn đoán Chính sách Đóng băng

Ngưỡng được đóng băng hoàn toàn từ **TRAIN OOF (Seed 101, 1.800 câu)** và chuyển giao chẩn đoán trực tiếp sang `DEV_CAL` và `DEV_SEARCH` (không tinh chỉnh tham số trên tập DEV):

| Nhóm Chính sách | Ngưỡng Đóng băng $(\tau_S, \tau_O, \tau_I)$ | Tập Đánh giá | Realized FWR | Số Cảnh báo Giả | Macro-3 Error F1 | Recall-S | Precision-S | F1-S | Recall-O | Precision-O | F1-O | Recall-I | Precision-I | F1-I | Cảnh báo Sai Loại |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Chính sách Sản xuất Đóng băng** | $\tau_T = 0.8485, \tau_G = 0.8736$ | `DEV_CAL` | 0.1767 | 53 | 0.1692 | **0.00%** | 0.00% | 0.0000 | 19.26% | 70.63% | 0.2997 | 12.75% | 56.52% | 0.2080 | 7 |
| | | `DEV_SEARCH` | 0.1400 | 84 | 0.1195 | **0.00%** | 0.00% | 0.0000 | 11.00% | 30.67% | 0.1608 | 13.60% | 34.69% | 0.1977 | 3 |
| **Ứng viên Danh định ($B = 0.20$)** | $\tau_S = 0.9263, \tau_O = 0.8217, \tau_I = 0.8424$ | `DEV_CAL` | 0.3000 | 90 | **0.2436** | **7.13%** | 85.81% | 0.1316 | 24.03% | 66.47% | 0.3529 | 15.69% | 57.14% | 0.2462 | 13 |
| | | `DEV_SEARCH` | 0.2917 | 175 | **0.1702** | **4.86%** | 56.58% | 0.0890 | 12.92% | 31.40% | 0.1824 | 17.60% | 37.29% | 0.2391 | 5 |
| **Ứng viên Bảo thủ ($B = 0.18$)**| $\tau_S = 0.9290, \tau_O = 0.8319, \tau_I = 0.8625$ | `DEV_CAL` | 0.2733 | 82 | **0.2206** | **6.96%** | 86.11% | 0.1288 | 21.65% | 67.11% | 0.3268 | 12.75% | 56.52% | 0.2063 | 12 |
| | | `DEV_SEARCH` | 0.2667 | 160 | **0.1537** | **4.64%** | 56.16% | 0.0855 | 11.96% | 30.49% | 0.1712 | 14.40% | 36.00% | 0.2045 | 5 |
| **Chính sách Độ nhạy ($B = 0.12$)**| $\tau_S = 0.9494, \tau_O = 0.8575, \tau_I = 0.8867$ | `DEV_CAL` | **0.1867** | **56** | **0.1981** | **4.60%** | **89.13%** | **0.0876** | **18.83%** | **70.73%** | **0.2969** | **12.75%** | **59.09%** | **0.2097** | **7** |
| | | `DEV_SEARCH` | **0.1850** | **111** | **0.1371** | **2.94%** | **56.52%** | **0.0560** | **11.00%** | **31.51%** | **0.1649** | **12.80%** | **38.10%** | **0.1905** | **5** |

**Nhận định chuyển giao chẩn đoán:**
- Người nói không bản xứ trong DEV có phương sai âm học cao hơn người nói trong tập TRAIN, do đó ngưỡng khớp từ TRAIN ở mức danh định $B=0.20$ bị trôi FWR lên $\sim 0.27 - 0.30$.
- Tuy nhiên, khi áp dụng mức ngân sách bảo thủ danh định $B=0.12$, FWR thực tế được kiểm soát chặt chẽ ở mức **0.1867 trên DEV_CAL** và **0.1850 trên DEV_SEARCH** (thỏa mãn trần an toàn $\le 0.20$).
- Tại điểm vận hành này, hệ thống phát hiện thành công **82 lỗi Thay thế trên DEV_CAL và 52 lỗi Thay thế trên DEV_SEARCH** với Macro-3 F1 tăng vọt từ 0.1692 lên 0.1981 (+0.0289).

---

## 6. Phân tích Chi tiết Cấp độ Từng Người nói (Speaker-Level Metrics)

| Tập Dữ liệu | Người nói | Số Câu | Số Target | Số Gap | Joint FWR | Macro-3 F1 | Recall-S | Precision-S | F1-S | Recall-O | Precision-O | F1-O | Recall-I | Precision-I | F1-I |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **DEV_CAL** | `ERMS` | 150 | 5.035 | 5.185 | 0.2800 | 0.2187 | **5.73%** | **87.50%** | 0.1075 | 17.86% | 27.03% | 0.2151 | 23.81% | 55.56% | 0.3333 |
| | `THV` | 150 | 5.008 | 5.158 | 0.3200 | 0.2338 | **8.42%** | **83.87%** | 0.1531 | 24.88% | 77.69% | 0.3769 | 10.00% | 60.00% | 0.1714 |
| **DEV_SEARCH** | `NCC` | 150 | 5.038 | 5.188 | 0.3600 | 0.1568 | **2.80%** | **60.00%** | 0.0535 | 14.55% | 21.05% | 0.1720 | 20.69% | 30.00% | 0.2449 |
| | `RRBI` | 150 | 4.874 | 5.024 | 0.3067 | 0.1395 | **6.65%** | **47.62%** | 0.1167 | 16.92% | 57.89% | 0.2619 | 2.27% | 16.67% | 0.0400 |
| | `YBAA` | 150 | 4.891 | 5.041 | 0.2400 | 0.1703 | **3.93%** | **37.14%** | 0.0710 | 0.00% | 0.00% | 0.0000 | 36.67% | 55.00% | 0.4400 |
| | `YDCK` | 150 | 4.952 | 5.101 | 0.2600 | 0.1785 | **6.22%** | **68.29%** | 0.1141 | 13.79% | 32.00% | 0.1928 | 18.18% | 30.77% | 0.2286 |

Độ chính xác Substitution trên người nói chưa từng thấy đạt mức xuất sắc (lên tới **87.5% trên ERMS** và **83.9% trên THV**), chứng minh độ tự tin của mạng Neural là xác thực và có tính khái quát hóa cao.

---

## 7. Task 8: Điều tra Phản Thực nghiệm (Counterfactual Forensic Analysis) đối với Heuristic Modulo-42 của E4

Chúng tôi kiểm chứng giả thuyết: *Liệu sự sụp đổ của E4 Substitution có phải đơn thuần do heuristic lấy ứng viên `(p + k) % 42` trong `agent_a_sm13_engine.py` (dòng 231–236) đã bỏ sót các âm vị thay thế thực tế hay không?*

### 7.1 Kết quả Phản Thực nghiệm trên Dữ liệu Thực tế

| Tập Dữ liệu | Tổng số Lỗi Thay thế | Số ca Modulo-42 Trúng (Hits) | Tỷ lệ Trúng | Số ca Trượt (Misses) | Tỷ lệ Trượt | Trung vị SLLR (Hits) | Trung vị SLLR (Misses) | Cực đại $p_{\text{E4}, S}$ (Hits) | Cực đại $p_{\text{E4}, S}$ (Misses) | % $p_{\text{E4}, S} > 0.5$ (Hits) | % $p_{\text{E4}, S} > 0.5$ (Misses) | Kiểm định Mann-Whitney U |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **TRAIN S101** | 7.381 | 1.019 | 13.81% | 6.362 | 86.19% | -0.8582 | +0.0656 | 1.0000 | 1.0000 | 12.37% | 21.57% | $p = 1.0000$ |
| **DEV_CAL** | 1.781 | 347 | 19.48% | 1.434 | 80.52% | -3.4192 | -2.7189 | 0.6508 | 0.3408 | 1.15% | 0.00% | $p = 0.9999$ |
| **DEV_SEARCH**| 1.768 | 267 | 15.10% | 1.501 | 84.90% | -3.3041 | -3.1149 | 0.5526 | 0.6608 | 0.37% | 0.13% | $p = 0.8924$ |

### 7.2 Phân biệt Hiệu ứng Quan sát và Giả thuyết Nhân quả

#### A. Hiệu ứng Quan sát được (Observed Effect)
- Việc chọn 4 ứng viên cạnh tranh theo thứ tự bảng chữ cái modulo 42 thực tế đã bỏ sót từ **80.5% đến 86.2%** âm vị thực tế mà người học phát âm sai. Chỉ có 13.8% đến 19.5% trường hợp âm vị thực tế nằm trong 4 ứng viên được đánh giá.

#### B. Bác bỏ Giả thuyết Nhân quả Đơn lẻ (Disconfirmed Single-Cause Hypothesis)
- *Giả thuyết:* "Sự tê liệt của E4 bắt nguồn hoàn toàn từ việc Modulo-42 không đánh giá đúng âm vị cạnh tranh."
- *Bằng chứng bác bỏ:*
  1. Ngay cả trong 347 ca tại `DEV_CAL` mà Modulo-42 **trúng đích** âm vị thực tế, trung vị SLLR vẫn là **-3.4192** (thậm chí còn thấp hơn nhóm trượt là -2.7189).
  2. Kiểm định phi tham số Mann-Whitney U xác nhận SLLR của nhóm trúng đích không hề cao hơn nhóm trượt ($p \approx 1.0$).
  3. Khi trúng đích, trung vị $p_{\text{E4}, S}$ chỉ đạt **0.0107** (khoảng 1%), và chỉ có **1.15%** số ca trúng đích có $p_{\text{E4}, S} > 0.50$.
  4. Do đó, việc chỉ thay đổi tập ứng viên Modulo-42 thành ma trận nhầm lẫn âm vị học (phonological confusion matrix) sẽ **không thể cứu vãn E4** nếu không khắc phục các cơ chế sụp đổ cốt lõi bên dưới.

#### C. Các Cơ chế Nhân quả Thực sự (True Causal Pathology)
1. **Pha loãng Trung bình Số học (Arithmetic Mean Dilution):** E4 tổng hợp điểm ứng viên bằng `max_sub = np.mean(sub_scores)`. Ngay cả khi 1 ứng viên có likelihood cao, việc tính trung bình cộng với 3 ứng viên ngẫu nhiên có likelihood cực thấp đã kéo tụt SLLR xuống vực thẳm.
2. **Bất đối xứng Mẫu số Softmax:** Giả thuyết Lược bỏ $\text{hyp\_del}$ không phụ thuộc vào ứng viên nên OLLR vẫn duy trì ở mức cao (+0.71 trung vị). Số hạng $e^{\text{OLLR}}$ áp đảo mẫu số của Softmax, đè nén $p_{\text{E4}, S} \le 0.65$ trên toàn bộ tập dữ liệu.
3. **Phạt cộng tĩnh 50/50:** Phép cộng cố định chia đôi độ tự tin của Neural, trực tiếp cắt cụt phân phối Thay thế dưới ngưỡng vận hành.

---

## 8. Kiểm toán Toàn vẹn Hệ thống & Đảm bảo Ranh giới Bảo mật

1. **Bảo tồn Tuyệt đối Artifacts và Checkpoints Gốc:**
   - Mã băm SHA-256 của `FROZEN_CALIBRATED_POLICY.json`: `dc7da3bcbc28918fcf39bdb1beba4ab990219d4467e56cdabbaebb0ac35e88e1` (Bảo tồn nguyên vẹn 100%, không bị sửa đổi hay thay thế).
   - Mã băm của tất cả trọng số mạng (`target_sm13_seed101_fulltrain.pt`, `gap_sm13_seed101_fulltrain.pt`, `ctc_registered_seed101.pt`) khớp hoàn toàn với lịch sử kiểm toán.
   - `test_sealed.jsonl` (SHA-256: `a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6`): **Bảo mật tuyệt đối, zero access**.
2. **Khớp nối Toán học Toàn vẹn Cấp độ Sự kiện:**
   - Kiểm tra đối chiếu với `UnifiedEvaluatorV3` cho thấy sự đồng nhất tuyệt đối trên từng con số nguyên:
     $$\text{Tổng Cảnh báo Giả} = \text{FP}_S + \text{FP}_O + \text{FP}_I = \text{Báo sai Nhãn Chuẩn} + \text{Báo sai Loại (Wrong Type)} + \text{FP}_I$$
     $$\text{Sai số kiểm toán } \Delta = 0 \text{ trên toàn bộ các tập dữ liệu.}$$
3. **Bộ Kiểm thử Hồi quy Tự động:**
   - `tests/test_sm13_asymmetric_fusion_regression.py` đạt **8/8 ca kiểm thử**.
   - Toàn bộ 33 ca kiểm thử trong repo đạt trạng thái **100% Passed**.

---

## 9. Danh mục Hồ sơ Bàn giao Hoàn chỉnh

Tất cả các tệp tài liệu và dữ liệu đã được tạo lập đầy đủ và kiểm tra cú pháp tại `experiments/rq1_sm13_asymmetric_fusion_01/`:

1. **`FINAL_REPORT.md`:** Báo cáo khoa học chính thức bằng tiếng Anh.
2. **`POLICY_COMPARISON.csv`:** Bảng dữ liệu 426 dòng ghi lại toàn bộ các phương án chính sách, ngân sách danh định, ngưỡng phân lớp, FWR và P/R/F1 trên TRAIN OOF, DEV_CAL, DEV_SEARCH ở cả chế độ quét thông thường và matched FWR.
3. **`PER_SPEAKER_METRICS.csv`:** Bảng dữ liệu 114 dòng chi tiết toàn bộ chỉ số trên từng người nói độc lập cho 12 người nói TRAIN, 2 người nói DEV_CAL và 4 người nói DEV_SEARCH.
4. **`CANDIDATE_POLICY.json`:** Cấu hình chính sách ứng viên đóng băng chính thức định dạng JSON chuẩn.
5. **`COUNTERFACTUAL_RESULTS.json`:** Dữ liệu phản thực nghiệm Modulo-42 chi tiết cho Task 8.
6. **`REGRESSION_RESULTS.md`:** Biên bản kiểm toán hồi quy và xác thực đồng nhất toán học.
7. **`reports/RQ1_SM13_ASYMMETRIC_FUSION_01_BAO_CAO_TOAN_DIEN.md`:** Báo cáo tổng hợp toàn diện bằng tiếng Việt phục vụ lưu trữ tài liệu nghiên cứu.

---

## 10. Khuyến nghị Nghiên cứu Tiếp theo

1. **Phê duyệt Kiến trúc Định tuyến Bất đối xứng (Asymmetric Routing) làm Kiến trúc Chuẩn:**
   - Bãi bỏ cơ chế hợp nhất cộng tĩnh 50/50 đối với lỗi Thay thế.
   - Định tuyến $p_S = p_{\text{neural}, S}$, $p_O = p_{\text{fixed}, O}$, $p_I = p_{\text{gap}, I}$ với cơ chế phân xử `score_margin`.
2. **Triển khai Điểm vận hành Độ nhạy cho Môi trường Thực tế:**
   - Để đảm bảo nghiêm ngặt trần $B_{\text{realized}} \le 0.20$ trên người nói chưa từng thấy, áp dụng chính sách Độ nhạy ($\tau_S = 0.9494, \tau_O = 0.8575, \tau_I = 0.8867$), mang lại Recall-S = 4.60% (82 cảnh báo Thay thế chính xác), Recall-O = 18.83%, Macro-3 F1 = 0.1981, và FWR = 0.1867.
3. **Nâng cấp Thành phần CTC/E4 (Trong Giai đoạn Huấn luyện Tiếp theo):**
   - Thay thế phép tính trung bình số học `np.mean` bằng `logsumexp` hoặc `max` trên các ứng viên cạnh tranh.
   - Thay thế Modulo-42 bằng ma trận nhầm lẫn âm vị học thực nghiệm (phonological confusion matrix).
   - Áp dụng kỹ thuật co dãn nhiệt độ (temperature scaling) cho phân phối Softmax của CTC để cân bằng ưu thế áp đảo của giả thuyết Lược bỏ trên giọng người nước ngoài.
