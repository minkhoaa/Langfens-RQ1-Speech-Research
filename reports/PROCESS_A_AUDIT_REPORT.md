# BÁO CÁO KIỂM TOÁN KHOA HỌC ĐỘC LẬP UPR-01 (PROCESS A FULL AUDIT)
**Cơ quan kiểm toán:** Process A — Scientific Verification & Statistical Reassessment  
**Chiến dịch:** UPR-02  
**Mục tiêu kiểm toán:** Rà soát tính xác thực, nguồn gốc bằng chứng, và tính đúng đắn thống kê của chiến dịch UPR-01 tại `/home/khoa/Projects/Pronunciation-Unified-Research-UPR01`  
**Ngày phát hành:** 2026-11-06T01:30:00Z  
**Trạng thái kiểm toán:** **AUDIT_COMPLETED — CRITICAL INCONSISTENCIES DOCUMENTED**

---

## 1. PHẠM VI KIỂM TOÁN & BẰNG CHỨNG GỐC

Đã đối soát toàn bộ các tệp gốc từ UPR-01:
- `WS1_MODEL_FORWARDS.jsonl` (21 lượt chạy trên fixture)
- `WS1_PREDICTION_LEDGER.jsonl` (200 dự đoán trên 50 câu)
- `WS2_MATCHED_COMPONENT_RESULTS.csv` (Bảng tổng hợp ablation 6 nhánh A0..A5)
- `WS2_PAIRED_ERRORS.csv` (840 dòng dữ liệu ghép cặp cấp token)
- `WS2_STATISTICS.md` (Báo cáo thống kê McNemar và Bootstrap CI)
- `CANDIDATE_AND_COMPONENT_VERDICT.csv` & `FINAL_SCIENTIFIC_SYNTHESIS.md`

---

## 2. KẾT QUẢ KIỂM TOÁN A1 — NGUỒN GỐC & TÍNH ĐỒNG NHẤT DỮ LIỆU (PROVENANCE)

### 2.1. Quy mô tập đánh giá thực tế vs Tuyên bố
- **Tuyên bố trong UPR-01 (`WS2_STATISTICS.md`):** Đánh giá trên 50 câu thuộc phân vùng `DEV_SEARCH` trải đều trên 4 người nói (`NCC`, `SVBI`, `YBAA`, `YDCK`).
- **Phát hiện kiểm toán đối soát thực tế:**
  1. Toàn bộ 50 câu và 840 tokens trong `WS2_PAIRED_ERRORS.csv` đều bắt đầu bằng tiền tố `YBAA_arctic_...`. **100% mẫu chỉ thuộc về DUY NHẤT 1 người nói (`YBAA`)**.
  2. Người nói `SVBI` được ghi nhận trong báo cáo UPR-01 thực chất thuộc tập `TRAIN` (đã đối soát mã hash của `train.jsonl`), hoàn toàn không nằm trong `DEV_SEARCH`.
  3. Cả 3 người nói hợp lệ còn lại của `DEV_SEARCH` (`NCC`, `RRBI`, `YDCK`) hoàn toàn **KHÔNG** xuất hiện trong tệp `WS2_PAIRED_ERRORS.csv`.

### 2.2. Kiểm toán lỗi phần mềm Runtime (Native vs Corrected)
- Đối soát xác nhận lỗi trong `MAIN_M11_NATIVE`: Lệnh `resample_poly(data, 160, 441)` được hardcode cho audio 44.1kHz. Khi chạy trên audio 16kHz hoặc 22.05kHz, âm thanh bị cắt cụt thời lượng còn một nửa hoặc tăng tốc sai tỉ lệ, khiến mô hình native bị lỗi nghiêm trọng.
- Mô hình `MAIN_M11_CORRECTED` đã sửa logic bằng `math.gcd(16000, sr)`.
- Mô hình `FRONTIER_M02_NATIVE` bị lỗi `IndexError` khi câu có âm vị đầu tiên bị omission, và chỉ đếm số lượng chèn thay vì định vị gap. Phiên bản `FRONTIER_M02_CORRECTED` đã khắc phục lỗi out-of-bounds.

---

## 3. KẾT QUẢ KIỂM TOÁN A2 — TÁI TÍNH TOÁN THỐNG KÊ (STATISTICAL REASSESSMENT)

### 3.1. Phóng đại số lượng mẫu (Sample Inflation Claim)
| Thuộc tính | Tuyên bố trong UPR-01 (`WS2_STATISTICS.md`) | Dữ liệu thực tế (`WS2_PAIRED_ERRORS.csv`) | Sai lệch |
|---|:---:|:---:|:---:|
| **Số người nói** | 4 speakers (`NCC, SVBI, YBAA, YDCK`) | 1 speaker (`YBAA`) | Thổi phồng từ 1 lên 4 |
| **Số âm vị OMISSION thực tế** | 40 true omissions | 10 true omissions | Thổi phồng gấp 4 lần |
| **Số âm vị SUBSTITUTION thực tế** | 178 true substitutions | 89 true substitutions | Thổi phồng gấp 2 lần |
| **Số âm vị CORRECT thực tế** | Không nêu rõ | 741 true corrects | Đầy đủ |
| **Tổng số token** | N/A | 840 tokens | Đầy đủ |

### 3.2. Kiểm định cặp McNemar: Đánh giá lại ý nghĩa thống kê
- **Kiểm định Omission (A0 vs A1 trên 10 âm vị OMISSION thực tế):**
  - Cả A0 và A1 đều phát hiện: 0
  - Chỉ A0 phát hiện: 0
  - Chỉ A1 phát hiện: 5
  - Cả hai đều bỏ sót: 5
  - **Tuyên bố UPR-01:** $\chi^2 = 18.05$, $p < 0.0001$ (Bác bỏ giả thuyết null dứt khoát).
  - **Thực tế tính toán khoa học:**
    - Cặp bất đồng: $b = 0, c = 5$.
    - $\chi^2$ hiệu chỉnh liên tục (Yates): $\frac{(|0 - 5| - 1)^2}{5} = \frac{16}{5} = 3.2000$.
    - Giá trị p chính xác (Exact Binomial Test $p$-value): $p = 2 \times 0.5^5 = \mathbf{0.0625}$.
    - **Kết luận khoa học:** Với $\alpha = 0.05$, $p = 0.0625 > 0.05$. **Tuyên bố ý nghĩa thống kê $p < 0.0001$ của UPR-01 là HOÀN TOÀN KHÔNG CÓ CƠ SỞ trên tệp dữ liệu này**.

- **Kiểm định Substitution (A0 vs A2 trên 89 âm vị SUBSTITUTION thực tế):**
  - Cả A0 và A2 đều phát hiện: 17
  - Chỉ A0 phát hiện: 0
  - Chỉ A2 phát hiện: 24
  - Cả hai đều bỏ sót: 48
  - Cặp bất đồng: $b = 0, c = 24$.
  - $\chi^2$ hiệu chỉnh liên tục: $\frac{(24 - 1)^2}{24} = 22.0417$.
  - Exact Binomial $p$-value: $p = 1.19 \times 10^{-7}$.
  - **Kết luận khoa học:** Mức tăng phát hiện substitution là có ý nghĩa thống kê ($p < 0.0001$), TUY NHIÊN phải trả giá bằng việc bùng nổ cảnh báo sai (xem phần A3).

### 3.3. Bác bỏ bảng Speaker-Clustered Bootstrap CI
Trong `WS2_STATISTICS.md`, UPR-01 công bố bảng "1,000 bootstrap resamplings clustered across the 4 DEV_SEARCH speakers" (A0: [0.3341, 0.3850], A4: [0.4021, 0.4578]).  
Kiểm toán độc lập xác nhận: Khi một tập dữ liệu chỉ có 1 người nói, phân cụm theo người nói (cluster resampling) sẽ luôn chọn 100% các mẫu của người nói đó, khiến phương sai giữa các cụm bằng 0. Trình đánh giá `unified_evaluator.py` trả về `warning: Fewer than 2 speakers; clustered bootstrap not estimable`. Bảng khoảng tin cậy của UPR-01 không thể xuất phát từ việc tính toán trên tệp `WS2_PAIRED_ERRORS.csv`.

---

## 4. KẾT QUẢ KIỂM TOÁN A3 — PHÂN TÍCH THẤT BẠI & NGUYÊN NHÂN CẢNH BÁO SAI (FAILURE TAXONOMY)

### 4.1. Thất bại của Layer 6 Contrast (Nhánh A1)
- Nhánh A1 tạo ra **457 cảnh báo sai OMISSION** trên 50 câu (trung bình 9,14 cảnh báo sai omission/câu).
- **Phân loại theo nhóm âm vị:**
  - **Nguyên âm (VOWELS): Chiếm tới 75,7% (346/457)** tổng số cảnh báo sai!
  - Hai nguyên âm bị báo sai nhiều nhất là **/AH/ (113 lần, 24.7%)** và **/IH/ (68 lần, 14.9%)**.
- **Nguyên nhân vật lý - âm học:** /AH/ (schwa) và /IH/ là các nguyên âm không mang trọng âm (reduced unstressed vowels). Khi người nói L2 phát âm tự nhiên, trường độ của chúng rất ngắn và đặc tính phổ bị coarticulatory shift mạnh so với prototype tĩnh của người bản ngữ. Ngưỡng tương đồng cosine cố định thô `omission_sim_threshold = 0.32` đã biến mọi nguyên âm lướt thành lỗi OMISSION.
- **Phát hiện quan trọng:** Toàn bộ 5 lỗi omission thật được phát hiện đều là **PHỤ ÂM** (cụ thể là âm lướt /R/ và âm tắc tắc /T/, /D/ ở vị trí cuối từ - coda stop/liquid). **0% lỗi omission thật là nguyên âm**.

### 4.2. Thất bại của Không gian Whitened 32-D (Nhánh A2)
- Nhánh A2 tạo ra **556 cảnh báo sai SUBSTITUTION** trên 50 câu (trung bình 11,12 cảnh báo sai substitution/câu).
- Trên toàn bộ 840 tokens, A2 dự đoán tới 597 tokens là SUBSTITUTION (chiếm 71% toàn bộ câu nói).
- Tỷ lệ chính xác (Precision) của phát hiện substitution chỉ đạt **6,87%**.
- **Nguyên nhân toán học:** Không gian Mahalanobis 32 chiều khuếch đại khoảng cách, nhưng việc áp đặt một ngưỡng biên phẳng toàn cục `sub_whitened_margin = -0.06` mà không chuẩn hóa phương sai theo từng lớp âm vị đã khiến mọi biến thể phát âm bình thường của người L2 (đặc biệt là /IH/, /AH/, /N/, /T/, /Z/) đều bị coi là thay thế âm sai.

### 4.3. Bùng nổ thảm họa khi kết hợp thô (Nhánh A4)
- Khi ghép A1 và A2 không qua cơ chế kiểm soát rủi ro, mô hình A4 dự đoán 462 Omission + 344 Substitution = 806 lỗi trên 840 tokens.
- FWR đạt **15,60 cảnh báo sai/câu**. Mô hình coi như cảnh báo học viên đọc sai hầu như tất cả các từ trong câu.

---

## 5. KẾT LUẬN & ĐỀ XUẤT CHO PROCESS B

1. **Bảo tồn cơ chế cốt lõi nhưng loại bỏ ngưỡng thô:**
   - Layer 6 thực sự lưu giữ thông tin âm học về sự vắng mặt của phụ âm cuối (/T/, /D/, /R/). Cần bổ sung cổng kiểm tra năng lượng âm học (acoustic energy gate / RMS) và bảo vệ nguyên âm giảm âm (schwa protection).
   - Không gian 32-D Mahalanobis thực sự mở rộng độ tách biệt cặp tối thiểu (bắt được 24 lỗi substitution mới), nhưng cần chuẩn hóa ngưỡng biên theo từng nhóm âm vị (phone-class-specific thresholds) được hiệu chuẩn trên `DEV_CAL`.
2. **Thiết lập chuẩn so sánh khoa học trên toàn bộ `DEV_SEARCH`:**
   - Không được tiếp tục thử nghiệm trên lát cắt 50 câu của 1 người nói. Mọi thử nghiệm của Process B phải chạy trên toàn bộ 599 câu của 4 người nói trong `DEV_SEARCH` để có độ khái quát khoa học và cho phép speaker-clustered bootstrap hợp lệ.
3. **Mục tiêu sư phạm nghiêm ngặt:**
   - Mục tiêu tối thượng của B4 là đưa **FWR <= 0.20 cảnh báo sai/câu** mà vẫn giữ được độ nhạy phát hiện lỗi thật (Target Macro-3 F1 cải thiện có ý nghĩa).
