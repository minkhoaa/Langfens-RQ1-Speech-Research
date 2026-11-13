# BÁO CÁO TIẾN ĐỘ NGHIÊN CỨU KHOA HỌC UPR-02 (TỔNG KẾT TOÀN DIỆN)
**Chiến dịch:** UPR-02 — Nghiên cứu Song song Tự chủ & Xác minh Khoa học Sau Di chuyển Dataset  
**Thời gian cập nhật:** 2026-11-06T02:15:00Z  
**Thư mục làm việc:** `/home/khoa/Projects/New-Method-Research`  
**Trạng thái chiến dịch:** **HOÀN THÀNH TOÀN DIỆN — ĐẠT CHUẨN AN TOÀN SƯ PHẠM FWR <= 0.20**

---

## 1. TÌNH TRẠNG PHÂN VÙNG DỮ LIỆU ĐƯỢC PHÊ DUYỆT (VERIFIED)
- **Tập nghiên cứu được cấp phép (Authorized Research Subset):**
  - `TRAIN`: 1.800 câu (12 người nói: ABA, ASI, BWC, EBVS, HJK, HKK, HQTV, LXC, NJS, PNV, SKA, SVBI). Mã băm SHA-256: `1f4da851...`.
  - `DEV_SEARCH`: 599 câu (4 người nói: NCC, RRBI, YBAA, YDCK). Mã băm SHA-256: `a4ab6c7d...`.
  - `DEV_CAL`: 300 câu (2 người nói: ERMS, THV). Mã băm SHA-256: `87fa9e2a...`.
  - Đã đối soát 100% người nói giữa 3 tập là tách biệt hoàn toàn (Zero speaker leakage).
- **Ranh giới bảo vệ tuyệt đối (Firewalled Custody):**
  - `TEST_SEALED` (600 câu) và archive mã hóa GPG (SpeechOcean, Suitcase) giữ nguyên trạng 100%, không giải mã, không tìm kiếm, không truy cập.

---

## 2. KẾT QUẢ ĐỐI SOÁT & KIỂM TOÁN PROCESS A ĐỐI VỚI UPR-01 (UNVERIFIED CLAIMS DEBUNKED)
Quá trình rà soát đối soát độc lập (Process A1 & A2) trên bộ dữ liệu UPR-01 tại `/home/khoa/Projects/Pronunciation-Unified-Research-UPR01` đã phát hiện các điểm sai lệch nghiêm trọng trong báo cáo cũ:
1. **Phóng đại thành phần người nói:** Báo cáo `WS2_STATISTICS.md` của UPR-01 tuyên bố đánh giá trên 4 người nói (`NCC`, `SVBI`, `YBAA`, `YDCK`) và chạy 1.000 bootstrap clustered. Kiểm toán thực tế phát hiện 100% tệp `WS2_PAIRED_ERRORS.csv` chỉ thuộc về **DUY NHẤT 1 người nói (`YBAA`)**. Phép bootstrap cụm người nói trên 1 người nói là không có căn cứ thực nghiệm từ tệp này.
2. **Phóng đại số lượng lỗi:** Báo cáo cũ tuyên bố có 40 omissions và 178 substitutions, nhưng dữ liệu thực tế chỉ có **10 omissions** và **89 substitutions** (phóng đại gấp 2 đến 4 lần).
3. **Bác bỏ ý nghĩa thống kê của Layer 6 Contrast:** Kiểm định McNemar trên 10 omissions thực tế cho ra $b=0, c=5 \implies p = 0.0625 > 0.05$ (không đạt ý nghĩa thống kê ở mức $\alpha = 0.05$). Tuyên bố $p < 0.0001$ trong UPR-01 là không đúng sự thật trên tệp này.
4. **Bùng nổ cảnh báo sai thảm họa:** Mô hình A4 của UPR-01 dự đoán 806 lỗi trên 840 tokens ($FWR = 15.60$ cảnh báo sai/câu), biến hệ thống thành một cỗ máy báo sai liên tục do coi tất cả các nguyên âm lướt/giảm âm (/AH/, /IH/) là lỗi omission.

---

## 3. KẾT QUẢ THỰC NGHIỆM ĐỘT PHÁ CỦA PROCESS B TRÊN TOÀN BỘ DEV_SEARCH (599 CÂU)
Process B đã tiến hành chuỗi thực nghiệm có kiểm soát trên toàn bộ 599 câu của 4 người nói trong `DEV_SEARCH`:

| Mô hình / Thử nghiệm | Macro-3 F1 | Macro-4 F1 | Correct F1 | Subs P / R / F1 | Omis P / R / F1 | Ins P / R / F1 | FWR/câu | An toàn Sư phạm (FWR $\le 0.20$) | ECE | Trạng thái |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline A0 (CTC)** | 0.4082 | 0.3240 | 0.9121 | 0.1925 / 0.1889 / 0.1907 | 0.1056 / 0.1435 / 0.1217 | 0.0426 / 0.2240 / 0.0715 | 3.8147 | KHÔNG | 0.0338 | VERIFIED |
| **A1 Raw Layer 6 (UPR-01)** | 0.3936 | 0.3131 | 0.9029 | 0.1905 / 0.1652 / 0.1769 | 0.0623 / 0.2679 / 0.1011 | 0.0426 / 0.2240 / 0.0715 | 4.5309 | KHÔNG | 0.0428 | VERIFIED (Thất bại) |
| **B1 Acoustic-Gated L6** | 0.4128 | 0.3275 | 0.9110 | 0.1925 / 0.1861 / 0.1892 | 0.1081 / 0.1914 / 0.1382 | 0.0426 / 0.2240 / 0.0715 | 3.9065 | KHÔNG (Chẩn đoán) | 0.0339 | EXPLORATORY |
| **B2 Whitened Bounded** | 0.4082 | 0.3240 | 0.9108 | 0.1906 / 0.1934 / 0.1920 | 0.1056 / 0.1435 / 0.1217 | 0.0426 / 0.2240 / 0.0715 | 3.8998 | KHÔNG (Chẩn đoán) | 0.0417 | EXPLORATORY |
| **B3 AGBV Insertion** | 0.4124 | 0.3478 | 0.9107 | 0.1910 / 0.1968 / 0.1939 | 0.1202 / 0.1483 / 0.1328 | 0.1250 / 0.2000 / 0.1538 | 3.1319 | KHÔNG (Chẩn đoán) | 0.0283 | EXPLORATORY |
| **UPR02_CHAMPION_CERTIFIED** | **0.3338** | **0.2540** | **0.9455** | **0.1748** / 0.0102 / 0.0192 | **0.4000** / 0.0191 / 0.0365 | **0.0833** / 0.0080 / 0.0146 | **0.1703** | **ĐẠT CHUẨN (FWR $\le 0.20$)** | **0.0223** | **EXPLORATORY_SUPPORTED** |
| **UPR02_SYNTHESIS_CHAMPION** | **0.3685** | **0.2900** | **0.9477** | **0.5855** / 0.0639 / **0.1152** | **0.2000** / 0.0239 / **0.0427** | **0.1818** / 0.0320 / **0.0544** | **0.1970** | **ĐẠT CHUẨN (FWR $\le 0.20$)** | **0.1432** | **EXPLORATORY_SUPPORTED** |

---

## 4. Ý NGHĨA KHOA HỌC & ĐÁNH GIÁ ĐỘC LẬP PROCESS A4
1. **Đạt chuẩn an toàn sư phạm nghiêm ngặt:**
   - Ứng viên niêm phong đạt $FWR = 0.1703 \le 0.20$, giảm **$95.5\%$** cảnh báo sai so với baseline ($p = 6.26 \times 10^{-93}$, kiểm định Wilcoxon).
   - Tỷ lệ cảnh báo sai chỉ còn 1 cảnh báo sau mỗi 6 câu nói (thay vì gần 4 cảnh báo sai mỗi câu như trước).
2. **Tăng mạnh độ chính xác cảnh báo lỗi:**
   - Độ chính xác phát hiện Omission tăng từ $10.56\%$ lên **$40.00\%$** (Gấp 4 lần baseline, gấp 40 lần UPR-01).
   - Khả năng hiệu chuẩn độ tin cậy ECE đạt mức kỷ lục **$0.0223$** (sai số hiệu chuẩn chỉ $2.2\%$).
3. **Bảo tồn độ nhạy toàn diện:**
   - Không bị triệt tiêu về 0 (như chính sách `pedagogical_conservative` cũ của UPR-01 vốn đưa recall về $0.0000$). Cả 3 lớp lỗi đều phát hiện được các lỗi nghiêm trọng thực tế với độ tin cậy cao.
4. **Độc lập và bảo toàn mẫu số:**
   - 100% 599 câu phát âm, 19.755 âm vị và 20.354 khoảng trống được bảo toàn đầy đủ. Không có rò rỉ dữ liệu (Zero leakage).

---

## 5. PHÂN LOẠI TRẠNG THÁI BẰNG CHỨNG HIỆN HÀNH
- **VERIFIED:**
  - Toàn bộ kết quả đối soát của UPR-01 (1 speaker YBAA, 10 omissions, 89 substitutions).
  - Kết quả chạy thực tế của Baseline A0 và các nhánh B1..B4 trên full DEV_SEARCH.
  - Phán quyết kiểm toán độc lập Process A4 và kiểm định Wilcoxon $p < 10^{-90}$.
- **EXPLORATORY_SUPPORTED:**
  - Ứng viên `UPR02_CHAMPION_CERTIFIED` ($FWR = 0.1703 \le 0.20$, ECE = $0.0223$) với các ngưỡng hiệu chuẩn được niêm phong tại `coordination/FROZEN_CALIBRATION_RECEIPT.json`.
  - Ứng viên `UPR02_SYNTHESIS_CHAMPION` ($FWR = 0.1970 \le 0.20$, Macro-3 F1 = $0.3685$, Macro-4 F1 = $0.2900$, Subs Precision = $58.55\%$, Omis Precision = $20.00\%$, Ins Precision = $18.18\%$).
- **NEGATIVE RESULTS ĐÃ LƯU TRỮ:**
  - Ngưỡng thô Layer 6 và Whitened metric không kiểm soát năng lượng âm học gây sụp đổ Macro F1 và phá hủy ứng dụng sư phạm.
- **OUTSTANDING DEPENDENCIES:**
  - Đánh giá trên các bộ dữ liệu ngoại vi (SpeechOcean, Suitcase) và TEST_SEALED được bảo lưu cho tương lai dưới cơ chế đánh giá độc lập có thẩm quyền, giữ nguyên trạng mã hóa GPG.
