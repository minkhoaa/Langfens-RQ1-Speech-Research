# BÁO CÁO KHOA HỌC CUỐI CÙNG — ĐIỀU TRA NĂNG LỰC CÁC MÔ HÌNH HÀNG ĐẦU SAU ĐỢT PHÁT HÀNH EVAL-V3
## LANGFENS Post EVAL-V3 Scientific Research Campaign: Final Monograph

**Cơ quan chủ quản:** LANGFENS Autonomous Speech ML Research Directorate  
**Tác giả:** Autonomous ML Research Director & Agents A, B, C  
**Không gian làm việc & Nguồn chân lý duy nhất:** `/home/khoa/Projects/New-Method-Research` — `experiments/evaluation_v3/closure/release_final/`  
**Các tệp dữ liệu kiểm chứng đi kèm:**  
- `reports/TMCI_MODEL_CAPABILITY_COMPARISON.csv`
- `reports/TMCI_COMPONENT_CONTRIBUTION.csv`
- `reports/TMCI_A_COMPONENT_ATTRIBUTION.md`
- `reports/TMCI_B_DISCRIMINATIVE_CAPABILITY.md`
- `reports/TMCI_C_VALIDITY_AND_ROBUSTNESS.md`
- `reports/TMCI_REPRODUCTION.md`
- `reports/TMCI_NEXT_RESEARCH_HANDOFF.md`  
**Ngày công bố chính thức:** 07/11/2026  

---

## 1. TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)

Chiến dịch nghiên cứu khoa học chuyên sâu **TMCI (Top-Model Capability Investigation)** được tiến hành ngay sau khi đợt đánh giá chuẩn hóa **EVAL-V3** hoàn tất và đóng task chính thức (`aa4e6ea`). Mục tiêu tối thượng của chiến dịch là giải mã căn nguyên toán học, kiến trúc mạng nơ-ron, bản chất âm học và chiến lược kết hợp điểm số giúp 4 hệ thống:
- **EXP-36-FINAL-HEXAD-PNC-SUPER-FUSION** (Macro-3 Error AP = **0.289233**)
- **EXP-43-CONTINUOUS-AUTONOMOUS-FRONTIER-EXPANSION** (Macro-3 Error AP = **0.284760**)
- **EXP-29-PROSODIC-PENTAD-FUSION** (Macro-3 Error AP = **0.282223**)
- **EXP-24-FINAL-ROADMAP-EXPANSION** (Macro-3 Error AP = **0.276819**)

đạt được năng lực phân biệt cân bằng 3 lớp lỗi phát âm (Thay thế - Substitution, Bỏ sót - Omission, Chèn - Insertion) vượt trội hoàn toàn so với các thế hệ trước trong lịch sử dự án LANGFENS (cụm giáo viên chuyên gia Phase 2: Macro-3 AP 0.1865; mô hình học sinh chưng cất tri thức GRU Phase 3 STU-05: Macro-3 AP 0.2365).

### Các kết luận khoa học đột phá:
1. **EXP-24 là "viên ngọc kiến trúc" nơ-ron đơn lẻ:** EXP-24 không phải là một tổ hợp hòa trộn điểm mà là một mô hình mạng nơ-ron độc lập (`JointSoftAttentionCTCNetwork`, 533k tham số). Nhờ cơ chế **Monotonic Local Gaussian Attention ($\sigma=3.0$)**, EXP-24 loại bỏ triệt để hiện tượng rung giật biên giới (boundary jitter) của bộ canh hàng cưỡng bức, thiết lập kỷ lục chiến dịch về **AP Omission đơn lẻ đạt 0.225092** (cao hơn cả tổ hợp EXP-36).
2. **Nguyên lý Tam giác Trụ cột tạo nên đỉnh cao EXP-36:** Thành công của EXP-36 không đến từ việc hòa trộn ngẫu nhiên 7 mô hình mà đến từ sự hội tụ của 3 nguồn chứng cứ âm học mang tính nhân quả:
   - *Trụ cột chú ý âm học:* EXP-24 (JSAC-Net) chống rung biên, neo giữ Omission.
   - *Trụ cột tiên nghiệm âm vị học:* EXP-34 (PNC-Net) khai thác xác suất chuyển tiếp n-gram từ điển và cờ hợp thức cụm phụ âm, neo giữ Substitution (AP S đạt 0.3871).
   - *Trụ cột động lực năng lượng:* EXP-28 (APED-Net) đưa năng lượng dạng sóng RMS và độ dốc phổ vào khe hở, neo giữ Insertion (AP I đạt 0.2759).
3. **Phát hiện cấu hình tối giản vượt trội [EXP-24 + EXP-34]:** Phép đo tích lũy từng bước chứng minh chỉ cần kết hợp 2 mô hình EXP-24 và EXP-34, Macro-3 Error AP đạt tới **0.292012** (vượt qua cả EXP-36 với 0.289233), trong khi AP Omission đạt đỉnh kỷ lục **0.242612**.
4. **Giải mã thất bại của EXP-43 so với EXP-36:** EXP-43 (9 mô hình) bổ sung thêm EXP-38 và EXP-42. Trong lịch sử, EXP-38 đạt kết quả ảo do rò rỉ nhãn gold qua biến định tuyến `target_m`. Khi kiểm định non-oracle nghiêm ngặt trong EVAL-V3, EXP-38 bị sụp đổ hiệu năng và trở thành lực cản kéo lùi AP Substitution của EXP-43 xuống 0.3635.
5. **Tính không đồng nhất giữa các người nói (Speaker Non-Uniformity):** EXP-36 không thắng tuyệt đối trên mọi người nói. Trên người nói tiếng Trung (`NCC`) và tiếng Ả Rập (`YBAA`), EXP-36 áp đảo. Tuy nhiên, trên người nói tiếng Hindi (`RRBI`) và tiếng Hàn (`YDCK`), mô hình đơn lẻ **EXP-24 mới là kẻ chiến thắng** nhờ khả năng bắt lỗi nuốt âm (Omission) vượt trội.
6. **Hợp đồng suy luận và tính độc lập tham chiếu:** Toàn bộ chuỗi suy luận từ file âm thanh thực WAV 16kHz đã được xác minh khớp 100% không sai lệch đặc trưng (`delta = 0`). Tuy nhiên, việc sinh chuỗi tham chiếu từ văn bản thuần túy qua CMUdict gặp **39 trường hợp OOV (6.5%)** cần cơ chế G2P dự phòng để tránh việc phải từ chối chấm điểm (abstain).

---

## 2. TRẢ LỜI CHI TIẾT 12 CÂU HỎI QUYẾT ĐỊNH KHOA HỌC (THE 12 SCIENTIFIC DECISION CRITERIA)

### Câu hỏi 1: Tại sao EXP-36 đạt Macro-3 Error AP cao nhất trong báo cáo phát hành?
**Trả lời:**  
EXP-36 đạt Macro-3 Error AP cao nhất (**0.289233**) vì nó kết hợp thành công **Tam giác Trụ cột (The Core Triad)** của 3 nguồn chứng cứ âm học trực giao nhau mà không bị pha loãng bởi các thành phần rò rỉ:
1. **Biểu diễn âm học liên tục không bị rung biên** từ mô hình mạng nơ-ron EXP-24 (chiếm trọng số target 0.20).
2. **Mô hình tiên nghiệm xác suất âm vị học n-gram từ vựng** từ EXP-34 (chiếm trọng số target 0.20 và gap 0.20).
3. **Đặc trưng động lực năng lượng phổ dạng sóng** từ EXP-28 (chiếm trọng số target 0.15 và gap 0.18).
Các thành phần còn lại (EXP-17 chuẩn hóa tốc độ nói, EXP-21 adapter âm học, EXP-22 khoảng cách tương phản, EXP-10 conformer) đóng vai trò điều hòa phương sai (regularizers), giúp đẩy đều cả 3 chỉ số thành phần lên mức cân bằng tối ưu trên DEV_SEARCH: AP S = **0.387088**, AP O = **0.204687**, AP I = **0.275924**.

---

### Câu hỏi 2: Những thành phần nào chịu trách nhiệm cho hiệu năng xếp hạng S, O và I của EXP-36?
**Trả lời:**  
Phép kiểm định cắt bỏ thành phần (Leave-One-Out Ablations) và đánh giá đơn lẻ xác định chính xác:
- **Xếp hạng lỗi Thay thế (AP S = 0.3871):**
  Thành phần chủ đạo là **EXP-34 (PNC-Net)** (đơn lẻ đạt AP S = 0.3746) kết hợp với **EXP-22 (PSC-Net)** và **EXP-24**. EXP-34 sử dụng xác suất chuỗi $P(p_t \mid p_{t-1})$ từ CMUdict để loại bỏ các âm allophone có xác suất từ vựng cực thấp, giúp phân định rõ ràng giữa âm chuẩn và âm thay thế.
- **Xếp hạng lỗi Bỏ sót (AP O = 0.2047):**
  Thành phần chịu trách nhiệm tuyệt đối là **EXP-24 (JSAC-Net)** (đơn lẻ đạt kỷ lục AP O = 0.2251). Khi cắt bỏ EXP-24 và refit trên DEV_CAL, AP Omission của tổ hợp rơi tự do xuống 0.1856.
- **Xếp hạng lỗi Chèn (AP I = 0.2759):**
  Thành phần quyết định là **EXP-28 (APED-Net)** (đơn lẻ đạt AP I = 0.2563) phối hợp cùng cờ hợp thức cụm phụ âm của **EXP-34** và chuẩn hóa thời lượng khe hở của **EXP-17**. Năng lượng RMS và độ dốc phổ tại khe hở giúp nhận diện năng lượng phát âm dư thừa tại vị trí chèn.

---

### Câu hỏi 3: EXP-36, EXP-43, EXP-29 và EXP-24 có thực sự khác biệt về mặt bằng chứng dự đoán không?
**Trả lời:**  
- **EXP-36 và EXP-29:** Rất ít khác biệt về bản chất. Hệ số tương quan Pearson giữa điểm số dự đoán của 2 mô hình là **0.9950** (Substitution), **0.9902** (Omission), **0.9872** (Insertion). Tỷ lệ đồng thuận quyết định đạt **98.60%**, Cohen's Kappa $\kappa = 0.9392$. EXP-36 chỉ là phiên bản tinh chỉnh nhẹ của EXP-29 bằng cách cộng thêm EXP-34 (+0.007 Macro-3 AP).
- **EXP-36 và EXP-43:** Tương quan rất cao (r ~ 0.98 - 0.99), nhưng EXP-43 bị thoái hóa nhẹ do bổ sung EXP-38 và EXP-42 (hai mô hình có AP Substitution đơn lẻ chỉ 0.200 và 0.244).
- **EXP-36 và EXP-24:** **Thực sự khác biệt sâu sắc!**
  Hệ số tương quan điểm số khe hở giữa EXP-24 và EXP-36 chỉ là **0.870**, độ tương đồng quyết định chèn âm có $\kappa = 0.635$. EXP-24 sở hữu một biểu diễn âm học độc lập bắt nguồn từ mạng nơ-ron đơn lẻ, trong khi EXP-36 là không gian hòa trộn đa miền (acoustic + prosodic + n-gram).

---

### Câu hỏi 4: Có mô hình nào thống trị tuyệt đối (dominate) mô hình khác trên toàn bộ đường cong PR không?
**Trả lời:**  
**KHÔNG CÓ SỰ THỐNG TRỊ TUYỆT ĐỐI.**
- Trên vùng **Recall thấp (High-Precision, Recall < 15%) của lỗi Bỏ sót (Omission)**:
  **EXP-24 thống trị hoàn toàn EXP-36 và EXP-43.** Precision@100 của EXP-24 đạt **37.0%** (so với EXP-36 là 28.0% và STU-05 là 24.0%).
- Trên vùng **Recall trung bình và cao của lỗi Thay thế (Substitution, Recall 20% - 60%)**:
  **EXP-36 chiếm ưu thế rõ rệt** nhờ bổ trợ của n-gram priors từ EXP-34 (duy trì Precision > 50% ở quy mô 1.000 cảnh báo).
- Trên vùng **Recall cao của lỗi Chèn (Insertion, Recall > 30%)**:
  **EXP-43 nhỉnh hơn EXP-36 khoảng 1-2% Precision** nhờ bổ sung các đặc trưng thời gian đa phân giải.
Sự vượt trội của EXP-36 chỉ là ưu thế tổng hợp trên giá trị tích phân diện tích dưới đường cong (Macro AP), không phải sự vượt trội đồng nhất ở mọi ngưỡng quyết định.

---

### Câu hỏi 5: Sự khác biệt về AP giữa các mô hình có ổn định trên các người nói khác nhau không?
**Trả lời:**  
**HOÀN TOÀN KHÔNG ỔN ĐỊNH. Có sự phân hóa sâu sắc theo ngôn ngữ mẹ đẻ (L1):**
- Trên người nói tiếng Trung (`NCC`) và tiếng Ả Rập (`YBAA`):
  EXP-36 vượt trội áp đảo EXP-24 (NCC: 0.3441 vs 0.3149; YBAA: 0.2380 vs 0.1890).
- Trên người nói tiếng Hindi (`RRBI`) và tiếng Hàn (`YDCK`):
  **EXP-24 lật ngược thế cờ và đánh bại EXP-36!**
  - RRBI: EXP-24 đạt Macro-3 AP = **0.2456** (EXP-36 chỉ đạt 0.2375).
  - YDCK: EXP-24 đạt Macro-3 AP = **0.4267** (EXP-36 chỉ đạt 0.4100).
Nguyên nhân: Người học tiếng Hàn và tiếng Hindi có tỷ lệ nuốt âm phụ âm cuối cực cao. Khả năng bảo toàn biểu diễn chuyển tiếp formant của EXP-24 giúp AP Omission trên YDCK đạt đỉnh **0.3132** (so với 0.2436 của EXP-36).

---

### Câu hỏi 6: Mô hình nào đạt hiệu năng phát hiện phát âm sai nhị phân (Binary Mispronunciation Detection) tốt nhất?
**Trả lời:**  
- Về mặt xếp hạng không ngưỡng (Binary MD AP trên target simplex $p_S + p_O$):
  **EXP-43** (0.387222) và **EXP-36** (0.386926) đồng hạng dẫn đầu, vượt trội so với EXP-24 (0.359881), học sinh STU-05 (0.344302) và giáo viên chuyên gia (0.349102).
- Về mặt điểm quyết định (F1 nhị phân tại ngưỡng tối ưu DEV_CAL):
  **EXP-43** đạt F1 nhị phân cao nhất: **0.4375** (Precision = 31.9%, Recall = 69.8%), theo sát bởi **EXP-29** (0.4361) và **EXP-36** (0.4347).

---

### Câu hỏi 7: Mô hình nào chẩn đoán chính xác loại lỗi tốt nhất (Exact Error Diagnosis)?
**Trả lời:**  
- **Về tổng thể đa lớp (Macro-3 Error F1 và Macro-4 F1):**
  **EXP-29** (Macro-3 F1 = **0.3356**, Macro-4 F1 = **0.4721**) và **EXP-36** (Macro-3 F1 = **0.3308**, Macro-4 F1 = **0.4688**) đạt kết quả chẩn đoán chính xác cao nhất.
- **Về chẩn đoán phân loại phụ âm lỏng bị bỏ sót (Liquid Omission Diagnosis):**
  Lớp âm lỏng /r, l/ chiếm 35.4% tổng số lỗi bỏ sót (74 trường hợp). Trên phân lớp này, **EXP-24 đạt độ nhạy chẩn đoán cao nhất** với Recall đạt **52.70%** (so với EXP-36 là 44.60%).

---

### Câu hỏi 8: Chính sách vận hành nào tạo ra dự đoán xếp hạng theo độ tin cậy hữu ích nhất?
**Trả lời:**  
1. **Chính sách chọn lọc theo ngân sách cố định (Precision@K / Fixed-Budget Policy):**
   Đây là chính sách hữu ích nhất cho ứng dụng sư phạm:
   - Nếu ngân sách cảnh báo hẹp ($K \le 100$ cảnh báo/tập dữ liệu): Sử dụng điểm số của **EXP-24** mang lại độ chính xác cao nhất trên Omission (Precision@100 = 37.0%) và Insertion (Precision@100 = 36.0%).
   - Nếu ngân sách cảnh báo rộng ($K \ge 1000$ cảnh báo): Sử dụng điểm số của **EXP-36** mang lại độ bao phủ tốt nhất (bắt được 502 lỗi Thay thế với Precision 50.2%, và tóm được 71.3% tổng số lỗi Bỏ sót).
2. **Chính sách ngưỡng hiệu chuẩn DEV_CAL (`CAL_MAX_ERROR_MACRO_F1`):**
   Mang lại F1 cân bằng tối đa (~0.331), nhưng chấp nhận mức FWR thực tế là **4.5 - 5.1 cảnh báo sai/câu**.
3. **Cảnh báo về chính sách FWR $\le 0.20$ (`PREDECLARED_RAW_0.9`):**
   Mặc dù triệt tiêu được cảnh báo sai xuống 0.25 lỗi/câu, chính sách này gây sụp đổ Recall của các lỗi thiểu số xuống 2.4%, làm mất đi giá trị chẩn đoán sư phạm.

---

### Câu hỏi 9: Độ tin cậy của các xác suất dự đoán (Predicted Probabilities) ra sao?
**Trả lời:**  
1. **Xác suất thô (Raw probabilities) của tất cả các mạng nơ-ron đều bị sai lệch tự tin (over-confident):**
   ECE thô của các mô hình trên lớp Đúng và Thay thế dao động từ **0.15 đến 0.18**.
2. **Hiệu chuẩn đẳng trương độc lập (Isotonic Calibration) giải quyết triệt để vấn đề:**
   - Khi áp dụng bản đồ Isotonic được khớp trên DEV_CAL sang DEV_SEARCH:
     - Brier score của EXP-36 được cải thiện từ **17% đến 24%** trên toàn bộ các lớp.
     - ECE lớp Chèn (Insertion) giảm từ 0.0065 xuống mức lý tưởng **0.0026** (EXP-36) và **0.0021** (EXP-24).
     - Brier score của EXP-36 đạt mức thấp nhất chiến dịch: **0.07467** (Substitution), **0.01168** (Omission), **0.00558** (Insertion).
3. **Mô hình học sinh STU-05 có độ tin cậy thô rất kém ở lớp thiểu số:**
   ECE thô của STU-05 trên Omission là **0.0482** (cao gấp hơn 2 lần EXP-36) và trên Insertion là **0.0355** (cao gấp 5.5 lần EXP-36).

---

### Câu hỏi 10: Các tổ hợp hàng đầu có cải thiện thực chất so với học sinh Phase 3 và giáo viên chuyên gia không?
**Trả lời:**  
**CÓ, CẢI THIỆN ĐỘT PHÁ VÀ CÓ Ý NGHĨA THỐNG KÊ RÕ RỆT.**
- Macro-3 Error AP tăng từ **0.1865** (Cụm giáo viên chuyên gia) và **0.2365** (Học sinh STU-05 KD-All trung bình) lên **0.2768** (EXP-24 đơn lẻ) và **0.2892** (EXP-36).
- Kiểm định cặp Student's t-test và Wilcoxon signed-rank test trên 599 câu phát ngôn xác nhận sự vượt trội của EXP-36 so với EXP-24 có ý nghĩa thống kê cực kỳ mạnh mẽ ($t = 5.655$, $p = 2.41 \times 10^{-8}$; Wilcoxon $p = 2.34 \times 10^{-11}$).
- **Bản chất của sự nhảy vọt:**
  Giáo viên chuyên gia và học sinh Phase 3 bị "mù" gần như hoàn toàn ở lớp Chèn âm (AP Insertion của giáo viên chỉ đạt 0.027 - 0.035). EXP-36 và EXP-24 đã giải quyết triệt để điểm mù này, đưa AP Insertion lên **0.2476** (EXP-24) và **0.2759** (EXP-36) nhờ khai thác động lực năng lượng phổ và cơ chế tự chú ý cục bộ.

---

### Câu hỏi 11: Mỗi hệ thống có thể chạy được từ âm thanh thực và chuỗi tham chiếu dự định độc lập không?
**Trả lời:**  
1. **Âm thanh thực tế (WAV Audio Input): HOÀN TOÀN KHẢ THI VÀ ĐÃ KIỂM CHỨNG.**
   Toàn bộ 2.699 câu âm thanh WAV 16kHz đã được chạy lại trên GPU và cho kết quả đặc trưng khớp 0 tuyệt đối so với cache. Pipeline suy luận âm học từ WAV là có thật và hợp lệ.
2. **Chuỗi tham chiếu dự định (Intended Reference): KHẢ THI VỚI ĐIỀU KIỆN CÓ G2P.**
   - Tra cứu từ điển CMUdict qua `prompt_phones` thành công trên **93.5% (560/599 câu)**.
   - Tuy nhiên, có **6.5% (39 câu)** chứa từ ngoài từ điển (OOV, như tên riêng địa danh). Nếu triển khai độc lập hoàn toàn mà không có mô hình nơ-ron Grapheme-to-Phoneme (G2P) dự phòng, hệ thống sẽ buộc phải từ chối xử lý 39 câu này.
   - Về mặt lý thuyết, tính độc lập của chuỗi tham chiếu trong tập L2-ARCTIC vẫn ở trạng thái **chưa được chứng nhận độc lập hoàn toàn (unverified independence)** do bắt nguồn từ chú thích TextGrid chuẩn hóa của tập dữ liệu.

---

### Câu hỏi 12: Những biểu diễn nền tảng nào đáng được chuyển giao sang kiến trúc mô hình thống nhất tương lai?
**Trả lời:**  
Chiến dịch TMCI xác định **3 thành phần kiến trúc cốt lõi duy nhất** đáng giá để kế thừa vào mô hình tương lai:
1. **Monotonic Local Gaussian Attention ($\sigma=3.0$) từ EXP-24:**
   Đây là đột phá kiến trúc quan trọng nhất toàn bộ chiến dịch. Nó giải quyết tận gốc hiện tượng rung giật biên giới của bộ canh hàng forced alignment, cho phép mô hình nơ-ron gom tụ thông tin âm học trên toàn bộ cửa sổ đồng cấu âm tự nhiên mà không bị cắt cụt nhân tạo. Nó là xương sống tạo nên AP Omission kỷ lục 0.2251.
2. **Phonotactic N-Gram Transition Prior Conditioning từ EXP-34:**
   Việc đưa xác suất chuyển tiếp âm vị $P(p_t \mid p_{t-1})$ và cờ hợp thức cụm phụ âm tiếng Anh vào đầu vào giúp mô hình sở hữu tri thức ngôn ngữ học từ điển, đưa AP Substitution lên 0.3871 và hỗ trợ phát hiện chèn âm cụm phụ âm (epenthesis).
3. **Waveform Prosodic Energy Dynamics từ EXP-28:**
   Tích hợp trực tiếp các tham số năng lượng RMS tương đối, tỷ lệ zero-crossing rate và độ nghiêng phổ vào biểu diễn khe hở (gap), cung cấp bằng chứng vật lý trực tiếp để phân biệt khe hở yên lặng với âm phát âm chèn thêm.

**CÁC THÀNH PHẦN PHẢI LOẠI BỎ:**
- Cấu trúc phân nhánh hai giai đoạn phân cấp (HTC-Net của EXP-38 và VTLN của EXP-42): Đã chứng minh gây sụt giảm hiệu năng trong chế độ non-oracle.
- Cơ chế hòa trộn điểm số hậu kỳ (post-hoc score ensembling) của 7 mô hình cồng kềnh: Phải được thay thế bằng một mạng nơ-ron thống nhất duy nhất có thể tối ưu hóa end-to-end.

---

## 3. BẢNG TỔNG HỢP SO SÁNH NĂNG LỰC TOÀN DIỆN (SYNTHESIS LEDGER)

Trích xuất từ `reports/TMCI_MODEL_CAPABILITY_COMPARISON.csv` và `reports/TMCI_COMPONENT_CONTRIBUTION.csv`:

| Mô hình | Bản chất kiến trúc | Số tham số | AP S | AP O | AP I | **Macro-3 AP** | Cal M3 F1 | Cal FWR | Ưu thế nổi bật | Điểm hạn chế chính |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| **EXP-36** | Score Ensemble (7 models) | ~4.4M | **0.3871** | 0.2047 | 0.2759 | **0.2892** | 0.3308 | 5.14 | Macro AP cao nhất toàn cục; AP S cao nhất; Brier/ECE tối ưu. | Cồng kềnh (cần 7 lượt forward); AP O thấp hơn EXP-24; thua EXP-24 trên Hindi/Korean. |
| **EXP-43** | Score Ensemble (9 models) | ~5.5M | 0.3635 | 0.2058 | **0.2851** | **0.2848** | 0.3305 | 4.56 | AP Insertion cao nhất (0.2851); Binary MD AP cao nhất. | EXP-38 và 42 gây nhiễu phân tán; AP S bị kéo lùi từ 0.387 xuống 0.363. |
| **EXP-29** | Score Ensemble (6 models) | ~3.9M | 0.3781 | 0.1965 | 0.2721 | **0.2822** | **0.3356** | 5.10 | Cân bằng tốt; Macro-3 F1 đạt 0.3356. | Kém hơn EXP-36 do thiếu thông tin n-gram prior của EXP-34. |
| **EXP-24** | **Standalone NN (JSAC-Net)** | **533k** | 0.3577 | **0.2251** | 0.2476 | **0.2768** | 0.3068 | 6.22 | **Mô hình đơn lẻ mạnh nhất; AP O cao nhất chiến dịch (0.2251); Precision@100 Omission = 37%; thắng trên Hindi & Korean.** | AP Thay thế (0.3577) thấp hơn các tổ hợp do thiếu priors từ vựng. |
| **STU-05** | Student GRU (KD-All, 3-seed mean) | 1.15M | 0.3669 | 0.1540 | 0.1887 | **0.2365** | 0.3094 | 4.28 | Kiến trúc nhỏ gọn, nhanh; đã vượt qua cụm giáo viên Phase 2. | Bị tụt hậu xa so với EXP-24 (-0.040 AP); ECE lớp thiểu số rất kém. |
| **TEACHER_BANK**| Specialist Teachers (3 Heads) | ~1.5M | 0.3833 | 0.1415 | 0.0348 | **0.1865** | 0.2482 | 2.87 | Từng là chuẩn so sánh của Phase 2. | Thất bại nặng nề ở lớp chèn âm (AP I = 0.0348); kiến trúc phân mảnh. |

---

## 4. KẾT LUẬN KHOA HỌC VÀ KHUYẾN CÁO CHIẾN LƯỢC

1. **Khẳng định tính ưu việt của hướng tiếp cận mới:**
   Các phương pháp dựa trên biểu diễn thời gian cục bộ mềm (Monotonic Local Attention) và tiêm đặc trưng âm vị học/năng lượng trực tiếp đã vượt trội hoàn toàn so với mô hình học sinh GRU Phase 3 và cụm giáo viên chuyên gia Phase 2. Mức tăng trưởng từ Macro-3 AP 0.1865 lên 0.2892 là bước tiến lớn nhất trong lịch sử dự án.
2. **Bác bỏ việc tiếp tục ensembling điểm số thô:**
   Việc tăng số lượng mô hình hòa trộn từ 6 (EXP-29) lên 7 (EXP-36) mang lại mức tăng biên nhỏ (+0.007 AP), và tăng lên 9 (EXP-43) đã bắt đầu phản tác dụng do nhiễu (-0.005 AP). Hướng đi gia tăng số lượng mô hình hòa trộn điểm đã chạm trần bão hòa và không có tính khả thi trong môi trường sản xuất có ràng buộc về độ trễ.
3. **Chỉ đạo chiến lược cho giai đoạn tiếp theo:**
   Toàn bộ nỗ lực kỹ thuật phải tập trung vào việc **thống nhất hóa (Unification)**: Xây dựng một mạng nơ-ron duy nhất tích hợp cơ chế Monotonic Local Attention của EXP-24, tầng mã hóa priors n-gram của EXP-34, và nhánh năng lượng khe hở của EXP-28, huấn luyện đồng thời với hàm mất mát tương phản có giám sát và phụ trợ đa nhiệm.
