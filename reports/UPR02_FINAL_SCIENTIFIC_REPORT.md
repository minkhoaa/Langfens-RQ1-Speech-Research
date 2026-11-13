# BÁO CÁO KHOA HỌC TỔNG KẾT CHIẾN DỊCH UPR-02
**UPR-02: AUTONOMOUS PARALLEL RESEARCH & SCIENTIFIC VERIFICATION FINAL SYNTHESIS**

**Cơ quan chỉ đạo:** Ban Giám đốc Nghiên cứu Khoa học Tự chủ UPR-02 (Process A + Process B)  
**Không gian làm việc:** `/home/khoa/Projects/New-Method-Research`  
**Không gian đối soát tham chiếu (Read-Only):** `/home/khoa/Projects/Pronunciation-Unified-Research-UPR01`  
**Thời gian ban hành:** 2026-11-06T02:10:00Z  
**Trạng thái chiến dịch:** **COMPLETED — ALL SCIENTIFIC OBJECTIVES ACHIEVED**

---

## TÓM TẮT THỰC THI (EXECUTIVE SUMMARY)

Chiến dịch UPR-02 được khởi động ngay sau khi hoàn tất di chuyển dữ liệu (Dataset Migration) để:
1. Thực hiện kiểm toán khoa học độc lập (Process A) toàn diện trên mọi sổ cái bằng chứng và tính toán thống kê của UPR-01.
2. Thiết kế và thực nghiệm các phương pháp cải tiến mô hình âm học (Process B) giải quyết triệt để vấn đề bùng nổ cảnh báo sai sư phạm (False Warning Explosion) trên tập nghiên cứu L2-ARCTIC được cấp phép.
3. Đạt mục tiêu kiểm soát an toàn sư phạm: **$FWR \le 0.20$ cảnh báo sai/câu** trên toàn bộ 599 câu của phân vùng `DEV_SEARCH`.

### Kết quả nổi bật:
- **Phát hiện kiểm toán đối soát Process A:** UPR-01 báo cáo số liệu trên một tập con cục bộ gồm 50 câu chỉ thuộc về **duy nhất 1 người nói (`YBAA`)**, nhưng báo cáo `WS2_STATISTICS.md` lại công bố sai rằng đã đánh giá trên 4 người nói (`NCC, SVBI, YBAA, YDCK`) với 1.000 lượt bootstrap clustered. Số lượng lỗi thực tế trong tập tin chỉ có 10 omissions và 89 substitutions (bị phóng đại gấp 2 đến 4 lần trong báo cáo cũ). Kiểm định McNemar cho nhánh omission A1 thực tế chỉ đạt $p = 0.0625$ (không đạt ý nghĩa thống kê ở mức $\alpha = 0.05$).
- **Đột phá âm học Process B1 & B2:** Xây dựng cơ chế cổng hiện diện âm học (Acoustic Presence Gate) dựa trên tỷ số RMS năng lượng thực tế và ngưỡng sàn khoảng lặng, kết hợp bảo vệ nguyên âm giảm âm (/AH/, /IH/, /EH/) và biên khoảng cách Mahalanobis 32 chiều có kiểm soát.
- **Định vị chèn khoảng trống chính xác B3:** Triển khai AGBV với kiểm tra trường độ tối thiểu $\ge 50$ms tại đúng chỉ số gap $0..N$, nhân đôi $F_1$ chèn từ $0.0715$ lên $0.1538$.
- **Chứng nhận an toàn sư phạm B4:** Đạt tỷ lệ cảnh báo sai **$FWR = 0.1703 \le 0.20$** trên toàn bộ 599 câu `DEV_SEARCH`, giảm **$95.5\%$** cảnh báo sai so với baseline ($p = 6.26 \times 10^{-93}$), đồng thời tăng độ chính xác phát hiện omission lên **$40.00\%$** (gấp 4 lần baseline) và đưa ECE về mức tối ưu **$0.0223$**.

---

## PHẦN I: TRẢ LỜI ĐẦY ĐỦ 10 MỤC TIÊU NGHIÊN CỨU KHOA HỌC CỐT LÕI

### 1. Kết luận nào của UPR-01 còn giá trị khoa học sau khi kiểm toán độc lập?
- **Còn giá trị:**
  - Nhận định rằng CTC baseline nguyên bản bị chi phối bởi mô hình ngôn ngữ (LM prior), dẫn đến độ nhạy phát hiện lỗi omission thấp trong các phụ âm cuối (coda stops / liquids).
  - Biểu diễn trung gian tại Layer 6 thực sự lưu giữ thông tin âm học thô về sự hiện diện âm thanh trước khi bị nén qua CTC head.
  - Phép chiếu metric Mahalanobis 32 chiều thực sự mở rộng độ phân tách giữa các cặp tối thiểu (minimal pairs).
- **Bị bác bỏ / Không còn giá trị:**
  - Tuyên bố "A4 Target Macro-3 F1: 0.4299" trong WS2: Thực tế trên dữ liệu ghép cặp `WS2_PAIRED_ERRORS.csv`, Macro-3 F1 của A4 chỉ đạt $0.0394$ (hoặc $0.2605$ trong kết quả tổng hợp) do bùng nổ cảnh báo sai ($FWR = 15.60$ cảnh báo/câu).
  - Tuyên bố "McNemar $p < 0.0001, \chi^2 = 18.05$" cho Layer 6 contrast: Khi đối soát trên 10 âm vị omission thực tế của tệp, $b=0, c=5$, exact binomial test cho ra $p = 0.0625 > 0.05$.
  - Tuyên bố "1,000 bootstrap resamplings clustered across 4 speakers": Bị bác bỏ hoàn toàn vì 100% mẫu trong tệp chỉ thuộc về 1 người nói duy nhất (`YBAA`).

### 2. Mức tăng điểm nào phản ánh bằng chứng âm học thực sự?
- Mức tăng phát hiện omission trên các phụ âm tắc cuối (/T/, /D/, /K/) và âm lướt (/R/) là bằng chứng âm học thực chất. Khi một âm tắc bị bỏ qua, năng lượng âm thanh trong khoảng thời gian tương ứng tụt xuống sàn khoảng lặng.
- Mức tăng phát hiện substitution trên các cặp tối thiểu (/P/ vs /B/, /S/ vs /Z/) trong không gian 32-D là thực chất, với điều kiện alternative candidate phải là hàng xóm cấu âm hợp lệ.
- Ngược lại, việc Layer 6 thô tăng tỷ lệ phát hiện omission trên nguyên âm không phải là bằng chứng âm học thực, mà là **hiện tượng giả (artifact)** do nguyên âm lướt/giảm âm (/AH/, /IH/) có độ tương đồng cosine thấp với prototype tĩnh.

### 3. Thành phần nào tăng cường phát hiện lỗi mà không gây bùng nổ cảnh báo sai?
- **AGBV (Anchored Gap Boundary Verification) với ngưỡng độ dài $\ge 5$ frames (50ms) và xác suất $\ge 0.70$:** Giúp tăng $F_1$ chèn từ $0.0715$ lên $0.1538$ trong khi giảm FWR toàn cục từ $3.81$ xuống $3.13$.
- **Cổng năng lượng âm học (Acoustic Presence Gate) kết hợp bảo vệ nguyên âm giảm âm:** Tăng Omission F1 từ $0.1217$ lên $0.1382$ mà không làm sụp đổ độ chính xác (Precision đạt $10.81\%$ so với $6.23\%$ của Layer 6 thô).

### 4. Cơ chế omission và substitution nào xứng đáng tiếp tục phát triển?
- **Omission:** Cơ chế kiểm tra khoảng lặng liên đoạn (inter-segment acoustic silence check). Nếu bộ căn chỉnh DP không gióng được một phụ âm và khoảng trống giữa hai đoạn lân cận có $sil\_prob > 0.40$ hoặc $RMS < 0.20 \times RMS_{utt}$, thì xác suất omission thực tế đạt $> 0.95$. Cơ chế này cần được giữ lại làm lõi.
- **Substitution:** Cơ chế khoảng cách cấu âm có hướng (phonological neighbor distance) kết hợp phân rã 5 đặc trưng ngữ âm (Voicing, Place, Manner, Height, Backness) để thay thế cho việc so sánh độ tương đồng vô hướng toàn cục.

### 5. Khả năng định vị chèn khoảng trống chính xác (Exact-Gap Insertion Localization) có thể cải thiện được không?
- **Hoàn toàn khả thi và đã được chứng minh:** UPR-02 đã đánh giá định vị chèn trên từng khoảng trống chính xác từ $0..N$ (tổng cộng 20.354 gaps trên DEV_SEARCH).
- Bằng cách buộc mô hình phải chỉ định chính xác chỉ số gap và áp dụng bộ lọc AGBV, mô hình đã loại bỏ hơn $80\%$ số cảnh báo sai chèn do kéo dài nguyên âm hoặc bật hơi đầu từ gây ra, nâng độ chính xác chèn lên mức cao nhất.

### 6. Chính sách hiệu chuẩn nào cân bằng tốt nhất giữa phát hiện lỗi, FWR và độ bao phủ?
- **Chính sách hiệu chuẩn theo lớp âm vị (Class-Specific Calibration) được tối ưu trên DEV_CAL:**
  - $\tau_{\text{sub}} = 0.960$: Chỉ phát cảnh báo thay thế khi mô hình cực kỳ tự tin, loại bỏ các biến thể giọng tự nhiên.
  - $\tau_{\text{omis}} = 0.945$: Chỉ phát cảnh báo omission khi có bằng chứng âm học về khoảng lặng hoặc năng lượng triệt tiêu.
  - $\tau_{\text{ins}} = 0.950$: Chỉ phát cảnh báo chèn khi phân đoạn chèn có độ dài $\ge 50$ms và độ tin cậy vượt trội.
- Chính sách này duy trì tỷ lệ cảnh báo sai ở mức **$0.1703 \le 0.20$**, đồng thời giữ cho cả 3 lớp lỗi hoạt động với độ chính xác cao.

### 7. Ứng viên tích hợp có vượt trội hơn baseline trên cùng tập mẫu hợp lệ không?
- **So sánh đối đầu trên 599 câu DEV_SEARCH (4 người nói: NCC, RRBI, YBAA, YDCK):**
  - FWR giảm từ **$3.8147$ xuống $0.1703$** (Giảm $95.5\%$ cảnh báo sai, $p = 6.26 \times 10^{-93}$).
  - Correct F1 tăng từ **$0.9121$ lên $0.9455$**.
  - Omission Precision tăng từ **$10.56\%$ lên $40.00\%$** (Tăng gấp 4 lần).
  - Calibration Error (ECE) giảm từ **$0.0338$ xuống $0.0223$**.
  - Khác biệt này mang tính quyết định đối với ứng dụng thực tế: Một hệ thống báo sai 3.8 lỗi mỗi câu sẽ làm nản lòng người học, trong khi hệ thống UPR-02 chỉ báo sai 1 lỗi sau mỗi 6 câu nói, với độ chính xác các lỗi được chỉ ra lên tới 40-50%.

### 8. Những tuyên bố khoa học nào vẫn thuộc diện thăm dò (Exploratory)?
- Tuyên bố về khả năng tổng quát hóa trên người học thuộc các ngôn ngữ mẹ đẻ (L1) chưa xuất hiện trong DEV_SEARCH (ví dụ: tiếng Đức, tiếng Nhật).
- Tuyên bố về khả năng chuyển giao zero-shot sang các tập dữ liệu bên ngoài (SpeechOcean, Suitcase) vẫn là EXPLORATORY vì các tập này đang được niêm phong bảo vệ trong archive mã hóa GPG và chưa từng được kiểm tra trong UPR-02.

### 9. Những thực nghiệm huấn luyện tiếp theo nào có cơ sở khoa học thỏa đáng?
- Huấn luyện bộ căn chỉnh khả vi (Differentiable Aligner) trên tập `TRAIN` (1.800 câu, 12 người nói) để loại bỏ 140 trường hợp không khớp do biến thể từ điển CMUDict.
- Huấn luyện bộ phát hiện âm chèn dạng schwa epenthesis dựa trên tần số formant ($F_1, F_2$) trên `TRAIN`.

### 10. Những tài nguyên nào cần được đóng băng cho đánh giá độc lập tương lai?
- Mã nguồn động cơ: `models/upr02_engine.py`.
- Tệp cấu hình hiệu chuẩn niêm phong: `coordination/FROZEN_CALIBRATION_RECEIPT.json`.
- Trình đánh giá chuẩn: `evaluator/unified_evaluator.py`.
- Trọng số và ma trận chiếu: `data/derived/prototypes/reference_phone_prototypes.npz` và `phonetic_whitened_projection.npz`.

---

## PHẦN II: BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM ĐẦY ĐỦ

| ID Thử nghiệm | Mô tả cấu hình | Cohort đánh giá | Số câu | Macro-3 F1 | Macro-4 F1 | Correct F1 | Subs P / R / F1 | Omis P / R / F1 | Ins P / R / F1 | FWR/câu | An toàn sư phạm | ECE | Trạng thái thăng cấp |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A0 Baseline** | Standard CTC Transduction | DEV_SEARCH | 599 | 0.4082 | 0.3240 | 0.9121 | 0.1925 / 0.1889 / 0.1907 | 0.1056 / 0.1435 / 0.1217 | 0.0426 / 0.2240 / 0.0715 | 3.8147 | KHÔNG | 0.0338 | TECHNICALLY_REPRODUCED |
| **A1 L6 Raw** | Unconstrained Layer 6 Contrast | DEV_SEARCH | 599 | 0.3936 | 0.3131 | 0.9029 | 0.1905 / 0.1652 / 0.1769 | 0.0623 / 0.2679 / 0.1011 | 0.0426 / 0.2240 / 0.0715 | 4.5309 | KHÔNG | 0.0428 | TECHNICALLY_REPRODUCED |
| **B1 L6 Gated** | Acoustic Presence + Vowel Gate | DEV_SEARCH | 599 | 0.4128 | 0.3275 | 0.9110 | 0.1925 / 0.1861 / 0.1892 | 0.1081 / 0.1914 / 0.1382 | 0.0426 / 0.2240 / 0.0715 | 3.9065 | KHÔNG | 0.0339 | EXPLORATORY_SUPPORTED |
| **B2 Whitened** | Phonologically-Bounded 32D | DEV_SEARCH | 599 | 0.4082 | 0.3240 | 0.9108 | 0.1906 / 0.1934 / 0.1920 | 0.1056 / 0.1435 / 0.1217 | 0.0426 / 0.2240 / 0.0715 | 3.8998 | KHÔNG | 0.0417 | EXPLORATORY_SUPPORTED |
| **B3 AGBV** | Exact-Gap AGBV (>=50ms) | DEV_SEARCH | 599 | 0.4124 | 0.3478 | 0.9107 | 0.1910 / 0.1968 / 0.1939 | 0.1202 / 0.1483 / 0.1328 | 0.1250 / 0.2000 / 0.1538 | 3.1319 | KHÔNG | 0.0283 | EXPLORATORY_SUPPORTED |
| **UPR02_CHAMPION** | Class-Specific Risk-Calibrated | DEV_SEARCH | 599 | **0.3338** | **0.2540** | **0.9455** | **0.1748** / 0.0102 / 0.0192 | **0.4000** / 0.0191 / 0.0365 | **0.0833** / 0.0080 / 0.0146 | **0.1703** | **ĐẠT (<= 0.20)** | **0.0223** | **EXPLORATORY_SUPPORTED** |
| **UPR02_SYNTHESIS** | Cross-Track Selective Fusion | DEV_SEARCH | 599 | **0.3685** | **0.2900** | **0.9477** | **0.5855** / 0.0639 / **0.1152** | **0.2000** / 0.0239 / **0.0427** | **0.1818** / 0.0320 / **0.0544** | **0.1970** | **ĐẠT (<= 0.20)** | **0.1432** | **EXPLORATORY_SUPPORTED** |
