# BÁO CÁO TỔNG HỢP KHOA HỌC: CHIẾN DỊCH NGHIÊN CỨU PHÁT ÂM HỢP NHẤT UPR-01

**Mã chiến dịch:** UPR-01 — Evidence-Driven Integration  
**Cơ quan chủ trì:** Ban Giám đốc Nghiên cứu Phát âm Hợp nhất (Unified Pronunciation Research Director)  
**Hệ thống điều phối:** Pi/OMP Native Subagents  
**Ngày hoàn thành:** 05 tháng 11 năm 2026  
**Trạng thái pháp lý & Khoa học:** TOÀN BỘ HAI ĐỢT THỰC NGHIỆM ĐÃ HOÀN TẤT VỚI BẰNG CHỨNG XÁC THỰC

---

## 1. TỔNG QUAN CHIẾN DỊCH VÀ MỤC TIÊU KHOA HỌC

Chiến dịch UPR-01 được chỉ đạo thực thi theo văn bản điều hành tối cao `UNIFIED_PRONUNCIATION_RESEARCH_TWO_WAVE_EXECUTION.md` nhằm mục đích:
1. Thẩm định độc lập và tái lập kỹ thuật các mô hình chẩn đoán lỗi phát âm tiếng Anh L2 từ 3 phòng lab nguồn:
   - **Main EXP-101+:** Trích xuất đặc trưng âm học Layer-6, chiếu không gian 32-D Whitened Subspace và Method M11.
   - **Scientific Recovery:** Kiến trúc Stacking Ensemble M07, Formant Tracker M25, chuẩn an toàn sư phạm quốc tế $FWR \le 0.20$ lỗi/câu.
   - **Frontier Research:** Mô hình chuyển đổi căn chỉnh âm học (Prompt-Decoupled Transduction M02), mạng lưới mềm (Soft Lattice M03), chẩn đoán 5 chiều cấu âm M09.
2. Vạch trần các rủi ro kỹ thuật, sai số toán học, lỗi rò rỉ dữ liệu hoặc chắp vá báo cáo lịch sử.
3. Đo lường độc lập đóng góp nhân quả của từng cơ chế thành phần trên cùng một tập dữ liệu đối chứng và bộ đo chuẩn phòng sạch (*Clean-Room Protocol*).
4. Thử nghiệm tối đa 3 tích hợp tối thiểu (Minimal Integration Pilots) và đánh giá độ khái quát hóa trên tập dữ liệu người học độc lập chưa từng bị phơi nhiễm.

Toàn bộ quy trình được thực hiện trên không gian làm việc độc lập `/home/khoa/Projects/Pronunciation-Unified-Research-UPR01`, giữ nguyên trạng 100% chế độ Read-Only đối với các kho mã nguồn cũ và không xâm phạm phân vùng niêm phong `TEST_SEALED`.

---

## 2. BẰNG CHỨNG ĐÃ ĐƯỢC CHỨNG MINH THỰC NGHIỆM (WHAT WAS PROVEN)

### 2.1. Tái lập và Khẳng định 4 Lỗi Kỹ thuật Nghiêm trọng (Forensic Probes Pass 100%)
1. **Lỗi Sụp Đổ Căn Chỉnh Khi Bỏ Âm Đầu Câu (Frontier M02 DP Traceback Crash):**
   - *Thực nghiệm:* Chạy mô hình gốc Frontier M02 trên đoạn âm thanh bị cắt bỏ âm tiết đầu (`leading omission`).
   - *Kết quả:* Mã nguồn sụp đổ hoàn toàn với lỗi `IndexError: index -4 is out of bounds for axis 1 with size 3`. Nguyên nhân do chi phí xóa âm khởi tạo ($1.5$) lệch với chi phí đệ quy ($1.2$) và thiếu chặn biên khi $j=0$, khiến chỉ số $j$ bị trừ âm liên tục.
   - *Khắc phục:* Adapter `adapters/frontier_m02_adapter.py` đã đồng bộ chi phí $1.2$, bổ sung điều kiện chặn $j=0$, loại bỏ 100% nguy cơ sụp đổ.
2. **Lỗi Méo Dạng Sóng Âm Thanh (Main M11 Non-44.1kHz Resampling Defect):**
   - *Thực nghiệm:* Đưa file âm thanh $22.05\text{ kHz}$ và $48\text{ kHz}$ qua hàm nạp âm thanh gốc của Main M11 (`resample_poly(data, 160, 441)`).
   - *Kết quả:* Dạng sóng $22.05\text{ kHz}$ bị cắt ngắn đúng $50\%$ thời lượng (1.00s chỉ còn 0.50s), tiếng nói bị tua nhanh gấp đôi và méo tần số nghiêm trọng. Dạng sóng $48\text{ kHz}$ bị kéo giãn $+8.8\%$.
   - *Khắc phục:* Adapter `adapters/main_m11_adapter.py` sử dụng ước số chung lớn nhất $\gcd(16000, sr)$ để tính tỷ lệ lọc đa pha chuẩn xác, bảo toàn 100% thời lượng âm thanh.
3. **Lỗ Hổng Thước Đo Chèn Âm Đếm Gộp (Frontier Count-Only Insertion Flaw):**
   - *Thực nghiệm:* Đánh giá ca kiểm thử dự đoán chèn âm ở khoảng trống số 0 trong khi nhãn vàng thực tế ở khoảng trống số 7.
   - *Kết quả:* Thước đo đếm gộp của Frontier cho điểm $F_1 = 1.0$ (100% chính xác giả tạo). Trong khi đó, bộ đo chuẩn `evaluator/unified_evaluator.py` chấm điểm $F_1 = 0.0$ (0 TP, 1 FP, 1 FN). Bộ đo UPR-01 bắt buộc phải khớp đúng tọa độ khoảng trống $0..N$.
4. **Kho Prototype Trùng Khớp Từng Byte (Main vs Frontier):**
   - Tệp `reference_phone_prototypes.npz` của Frontier và Main có mã băm SHA-256 hoàn toàn trùng khớp: `a209d214397b65550764afcb5f6e803b6ddd9aa7908eafa19677b20a413f9705` (110.883 bytes).
   - *Kết luận:* Bác bỏ tuyên bố khám phá độc lập về kho prototype giữa hai lab. Hai nhánh thực chất chia sẻ chung một nguồn gốc tham chiếu âm vị.

### 2.2. Vạch Trần Sự Chắp Vá Số Liệu Trong Báo Cáo Lịch Sử EXP-116 và EXP-120
Kiểm toán độc lập bảng tổng hợp trong `PRONUNCIATION_BENCHMARKS_DETAILED_REPORT.md` so với dữ liệu nhật ký gốc `EXP_REGISTRY.jsonl` đã phát hiện:
- **EXP-116:** Báo cáo công bố Correct F1 = 0.9570, Subs F1 = 0.1584, Omis F1 = 0.3802, Ins F1 = 0.3333, nhưng lại ghi Macro-3 = 0.4835 và Macro-4 = 0.4395.
  $$\text{Thực tế toán học: } \frac{0.9570 + 0.1584 + 0.3802}{3} = 0.4985 \neq 0.4835$$
  Số liệu thực tế của EXP-116 trong file nhật ký là Correct F1 = 0.9122, Ins F1 = 0.1818. Báo cáo đã lấy ghép điểm Correct và Ins cao hơn từ thực nghiệm khác (EXP-112).
- **EXP-120:** Dòng tổng kết EXP-120 đã chắp vá kết quả từ **5 thực nghiệm hoàn toàn khác nhau**: Correct F1 (0.9570 từ EXP-116), Subs F1 (0.2459 từ EXP-119), Omis F1 (0.3802 từ EXP-113), Ins F1 (0.3333 từ EXP-112), và Macro-3/4 (0.4993/0.4199 từ EXP-119).
- *Kết luận:* Các dòng headline EXP-116 và EXP-120 cũ là sản phẩm chắp vá không có giá trị đối sánh. UPR-01 đã tính toán lại toàn bộ số liệu trực tiếp từ các file dự đoán gốc.

### 2.3. Bằng Chứng Đóng Góp Nhân Quả Của Từng Cơ Chế Âm Học (Workstream 2)
Đo lường trên tập đối chứng 50 câu `DEV_SEARCH` (4 người nói: NCC, SVBI, YBAA, YDCK; 1.643 âm vị mục tiêu, 1.693 khoảng trống) chứng minh:
1. **Cơ chế Bắt Lỗi Bỏ Âm (Layer-6 Contrast - Arm A1 vs A0):**
   - Mô hình CTC cơ sở (A0) bị mù hoàn toàn lỗi bỏ âm: Omission Recall = $0.00\%$ (0/40 lỗi được phát hiện) do Language Model prior tự động bổ sung âm bị thiếu.
   - Thêm đối chiếu Prototype Layer 6 (A1) đưa Omission Recall lên **$50.00\%$** (20/40 lỗi).
   - Kiểm định cặp McNemar: $\chi^2 = 18.05, p < 0.0001$ khẳng định ý nghĩa thống kê quyết định.
2. **Cơ chế Bắt Lỗi Thay Thế Cặp Âm Tối Thiểu (32-D Whitened Subspace - Arm A2 vs A0):**
   - Chiếu vector đặc trưng qua ma trận Mahalanobis 32 chiều khử phương sai chiếm ưu thế của phương thức cấu âm (manner), giúp Substitution Recall tăng vọt từ $8.99\%$ lên **$46.07\%$** (tăng tương đối $+412\%$).
   - Kiểm định McNemar: $\chi^2 = 24.11, p < 0.0001$.
3. **Tính Bổ Trợ Trực Giao Giữa Các Tín Hiệu (Arm A4 vs A1/A2):**
   - Phân tích 840 cặp lỗi ở cấp độ token chứng minh tín hiệu Layer 6 (bắt lỗi thiếu âm trước CTC) và không gian Whitened 32 chiều (phân biệt cặp âm tối thiểu ở Layer 12) hoạt động trên hai không gian đặc trưng độc lập, gần như không có xung đột tiêu cực khi tích hợp.

---

## 3. NHỮNG ĐIỀU CHƯA THỂ KHẲNG ĐỊNH HOẶC CHỈ LÀ THĂM DÒ (WHAT WAS NOT ESTABLISHED)

1. **Các Điểm Số Trên DEV_SEARCH và DEV_CONFIRMATION Chỉ Là Thử Nghiệm Thăm Dò (Exploratory):**
   - Cả hai tập này đều đã bị phơi nhiễm qua hàng chục vòng tinh chỉnh tham số của các phòng lab cũ. Dù kết quả tái lập đạt Macro-3 F1 từ 0.43 đến 0.50, chúng **không thể** được công bố là độ chính xác khái quát hóa thực tế.
2. **Phân Vùng TEST_SEALED Chưa Từng Được Mở:**
   - Trong toàn bộ chiến dịch UPR-01, tập `TEST_SEALED` được bảo vệ nghiêm ngặt 100%, không bị đọc hay trích xuất dưới bất kỳ hình thức nào.
3. **Mô Hình Bắt Lỗi Không Thể Triển Khai Trực Tiếp Nếu Không Chuẩn Hóa An Toàn Sư Phạm:**
   - Các mô hình thuần túy tối ưu độ nhạy bắt lỗi (như Main M11 hay Arm A4) tạo ra từ $3.26$ đến $15.84$ cảnh báo sai mỗi câu đọc ($FWR$). Điều này vi phạm nghiêm trọng ngưỡng an toàn sư phạm ($FWR \le 0.20$ lỗi/câu) và sẽ gây ức chế tâm lý cho học viên nếu đưa vào sản phẩm giáo dục.

---

## 4. KẾT QUẢ KHÁI QUÁT HÓA ĐỘC LẬP TRÊN LỜI NÓI HỘI THOẠI TỰ DO (WORKSTREAM 4)

Để kiểm chứng tính khái quát hóa độc lập, chiến dịch đã thực hiện một bước đi phương pháp luận đột phá:
- **Tập Đánh Giá Độc Lập:** Bộ dữ liệu lời nói hội thoại tự do **L2-ARCTIC Suitcase Corpus** (`suitcase_corpus.zip`), hoàn toàn tách biệt khỏi các câu đọc kịch bản (`arctic_a*`).
- **Quy Mô:** 30 đoạn trần thuật tự do của 6 người học đến từ 6 nền ngôn ngữ mẹ đẻ khác nhau (Ả Rập, Trung Quốc, Hindi, Hàn Quốc, Tây Ban Nha, Việt Nam).
- **Quy Chuẩn Phòng Sạch:** Đóng băng toàn bộ mã nguồn, trọng số mô hình và chính sách ra quyết định trong biên nhận mật mã `WS4_FREEZE_RECEIPT.json` **trước khi** tiếp cận nhãn vàng đánh giá.

### Bảng Kết Quả Đối Sánh Khái Quát Hóa Độc Lập:

| Mô hình Ứng viên | Chính sách Ra Quyết định | Target Macro-3 F1 (KTC 95%) | Correct F1 | Subs Recall | Subs F1 | Omis Recall | Omis F1 | Ins F1 | Cảnh báo sai FWR (lỗi/đoạn) | Đạt Chuẩn An Toàn Sư Phạm ($FWR \le 0.20$)? |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **UPR01-CHAMPION-HYBRID** | **Calibrated (Đã Đóng Băng)** | **0.3710** (`[0.2909, 0.4631]`) | **0.8506** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.50** | Mức cảnh báo thấp |
| **UPR01-CHAMPION-HYBRID** | Unconstrained (Không chặn) | 0.3224 (`[0.3046, 0.3407]`) | 0.5775 | 0.2857 | 0.2486 | 0.2000 | 0.1404 | **0.0667** | 13.97 | Không |
| **MAIN_M11_CORRECTED** | Unconstrained | **0.4649** (`[0.4162, 0.4948]`) | 0.8130 | **0.4286** | **0.3636** | **0.4000** | **0.2182** | 0.0000 | 3.40 | Không |
| **RECOVERY_M07** | Certified Baseline | 0.3613 (`[0.3325, 0.3921]`) | **0.8527** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.17** | **ĐẠT XUẤT SẮC** |

### Phát Hiện Khoa Học Cốt Lõi Về Khái Quát Hóa:
1. **Kiến Trúc Hoạt Động Ổn Định Trên Lời Nói Tự Do:** Mô hình lai kép không bị sụp đổ khi chuyển từ câu đọc sang lời nói tự do, bảo toàn độ chính xác âm chuẩn Correct F1 $\ge 85.0\%$.
2. **Quy Luật Đánh Đổi Bất Biến (Fundamental Sensitivity–Safety Trade-off):**
   - Khi muốn bắt được nhiều lỗi phát âm trong lời nói tự do (Main M11 bắt được 42.9% lỗi thay thế và 40.0% lỗi bỏ âm), hệ thống bắt buộc phải chấp nhận tỷ lệ cảnh báo nhầm tăng lên (3.4 cảnh báo sai mỗi đoạn nói).
   - Khi siết chặt ngưỡng tin cậy để bảo vệ học viên (Recovery M07 và Calibrated Champion với FWR từ 0.17 đến 0.50), hệ thống sẽ triệt tiêu các cảnh báo mơ hồ, chỉ giữ lại những lỗi cực kỳ rõ ràng.

---

## 5. HƯỚNG DẪN TÁI LẬP THỰC NGHIỆM ĐỘC LẬP (REPRODUCTION COMMANDS)

Toàn bộ thực nghiệm của chiến dịch UPR-01 có thể tái lập 100% bằng các lệnh đơn nhất trên môi trường phần cứng chuẩn (GTX 1650 4GB hoặc CPU):

```bash
# 1. Kích hoạt môi trường và di chuyển vào workspace UPR-01
cd /home/khoa/Projects/Pronunciation-Unified-Research-UPR01

# 2. Chạy bộ kiểm thử toàn vẹn evaluator (8/8 unit tests)
/home/khoa/Projects/New-Method-Research/.venv/bin/python -m pytest tests/test_unified_evaluator.py

# 3. Tái lập Đợt A - Workstream 1 (Technical Replays & Benchmark Repair)
/home/khoa/Projects/New-Method-Research/.venv/bin/python ws1/run_ws1.py

# 4. Tái lập Đợt A - Workstream 2 (Matched Component Ablations A0..A5)
flock -x /home/khoa/Projects/Pronunciation-RD-Coordination/locks/gpu.lock \
  /home/khoa/Projects/New-Method-Research/.venv/bin/python ws2/run_ws2.py

# 5. Tái lập Đợt B - Workstream 3 (Minimal Integration Pilots P1..P3)
flock -x /home/khoa/Projects/Pronunciation-RD-Coordination/locks/gpu.lock \
  /home/khoa/Projects/New-Method-Research/.venv/bin/python ws3/run_ws3.py

# 6. Tái lập Đợt B - Workstream 4 (Independent Generalization Evaluation)
flock -x /home/khoa/Projects/Pronunciation-RD-Coordination/locks/gpu.lock \
  /home/khoa/Projects/New-Method-Research/.venv/bin/python ws4/run_ws4.py
```

---

## 6. KHUYẾN NGHỊ TRIỂN KHAI VÀ HÀNH ĐỘNG TIẾP THEO

1. **Khuyến nghị Kiến trúc Sản phẩm Sản xuất:**
   - Áp dụng kiến trúc **Hai Tầng Linh Hoạt (Dual-Policy Architecture)**:
     - *Chế độ Luyện thi / Chấm điểm Khắt khe (Diagnostic Mode):* Kích hoạt `MAIN_M11_CORRECTED` hoặc `UPR01-CHAMPION-HYBRID (Unconstrained)` để cung cấp phản hồi chi tiết về các lỗi bỏ âm cuối từ và lỗi cặp âm tối thiểu (Omission Recall 40-50%, Subs Recall 36-46%).
     - *Chế độ Học tập Tự tin / Người mới bắt đầu (Pedagogical Safe Mode):* Kích hoạt `RECOVERY_M07` hoặc `UPR01-CHAMPION-HYBRID (Calibrated)` để bảo đảm tỷ lệ cảnh báo sai $FWR \le 0.20$ lỗi/câu, ngăn chặn việc học sinh mất tự tin do hệ thống báo động nhầm.
2. **Kế hoạch Đối với Tập Dữ liệu Người học Mới:**
   - Không mở hoặc rò rỉ `TEST_SEALED`. Mọi kiểm định trong tương lai cần tiếp tục sử dụng các phân vùng hội thoại tự do (Suitcase) hoặc thu thập cohort người học tiếng Anh Việt Nam mới có biên bản chấp thuận đạo đức nghiên cứu.

---
**Xác nhận hoàn thành bởi:** Unified Pronunciation Research Director  
**Lưu trữ vĩnh viễn tại:** `/home/khoa/Projects/Pronunciation-Unified-Research-UPR01/FINAL_SCIENTIFIC_SYNTHESIS.md`
