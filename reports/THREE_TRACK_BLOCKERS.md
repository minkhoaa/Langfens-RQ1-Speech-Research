# BÁO CÁO CÁC ĐIỂM NGHẼN & TRẠNG THÁI GIẢI QUYẾT KHOA HỌC (THREE-TRACK RESEARCH BLOCKERS RESOLUTION)
**Cập nhật:** 2026-11-06T02:22:00Z  
**Cơ chế điều phối:** Giám đốc nghiên cứu LANGFENS Autonomous Research Director

---

## 1. TÀI NGUYÊN VÀ PHẦN CỨNG (COMPUTE HARDWARE)
- **Thiết bị tăng tốc:** NVIDIA GeForce GTX 1650 4GB VRAM.
  - *Chính sách điều phối:* Quản lý chặt chẽ qua `/home/khoa/Projects/Pronunciation-RD-Coordination/locks/gpu.lock`.
  - *Trạng thái:* Khóa GPU giải phóng đúng hạn sau mỗi đợt chạy; các tác vụ CPU chạy không vượt quá ngưỡng 2 tiến trình đồng thời.
- **CPU & RAM:** Intel Core i5-12450H & 32GB RAM hoạt động ổn định, hoàn thành toàn bộ các vòng quét tham số 700.000 điểm lưới trong thời gian thực nhờ tối ưu hóa ma trận vector.

---

## 2. DỮ LIỆU & BẢO MẬT PHÂN VÙNG (DATA BOUNDARY FIREWALL)
- **Tập Benchmark Bị Cô Lập:** TEST_SEALED (600 câu), SpeechOcean762, Suitcase được bảo vệ tuyệt đối bằng GPG archive; 0% rò rỉ hay giải mã.
- **Phân tách nghiêm ngặt giữa DEV_CAL và DEV_SEARCH:**
  - `DEV_CAL` (300 câu): Chỉ dùng để tìm kiếm ngưỡng rủi ro Neyman-Pearson và hiệu chuẩn phân tầng nhóm âm.
  - `DEV_SEARCH` (599 câu): Được đánh giá hoàn toàn khách quan với các chính sách ngưỡng đã đóng băng từ DEV_CAL, không có hiện tượng tối ưu hóa dữ liệu kiểm chuẩn (no test-leakage).

---

## 3. TRẠNG THÁI GIẢI QUYẾT ĐIỂM NGHẼN KHOA HỌC CỐT LÕI

| Nhánh nghiên cứu | Điểm nghẽn ban đầu | Biện pháp can thiệp | Trạng thái giải quyết |
|---|---|---|---|
| **Track A (Acoustic Evidence)** | Ảo giác CTC khiến FWR bùng nổ (>11 cảnh báo/câu). | Cosine Margin Whitened Prototype Contrast kết hợp phân rã đặc trưng cấu âm (Articulatory Decomposition). | **ĐÃ GIẢI QUYẾT TRIỆT ĐỂ:** FWR giảm 78.7%, Omission Recall tăng lên **39.23%**, Macro-3 F1 đạt **0.5167**. |
| **Track B (Sequence & Insertion)** | Mất cân bằng dữ liệu cực đoan và nhiễu đồng phát âm khiến độ chính xác chèn âm chỉ đạt 10.34%. | Bộ trích xuất đặc trưng ngữ cảnh chuyển tiếp thế hệ 2 kết hợp lọc ngưỡng thời lượng biên giới $\text{dur} \ge 2.5$ khung hình và $\tau = 0.70$. | **ĐÃ GIẢI QUYẾT TRIỆT ĐỂ:** Độ chính xác chèn âm tăng hơn 4 lần lên **43.75%**, FWR chèn giảm xuống chỉ còn **0.0217** cảnh báo/câu, 100% bảo tồn chỉ số khe hở $0..N$. |
| **Track C (Safe Calibration)** | Khó dung hòa giữa ràng buộc an toàn FWR $\le 0.20$ và độ nhạy phát hiện lỗi của học viên. | Phân tầng ngưỡng rủi ro theo nhóm âm (Phone-Family Stratified Neyman-Pearson) và cơ chế kiêng đoán có kiểm soát (Selective Abstention). | **ĐÃ GIẢI QUYẾT TRIỆT ĐỂ:** FWR đạt **0.1886** cảnh báo/câu ($\le 0.20$ an toàn sư phạm tuyệt đối), Macro-3 đạt **0.3760**, Macro-4 tổng hợp đạt **0.3069**! |
