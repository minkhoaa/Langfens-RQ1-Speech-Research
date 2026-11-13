# BÁO CÁO TIẾN ĐỘ CHUYÊN GIA I (TEACHER I: EXACT-GAP INSERTION SPECIALIST)
**Chiến dịch:** LANGFENS Autonomous GPU-First Specialist Teachers  
**Ngày cập nhật:** 2026-11-06T10:45:00Z  
**Trạng thái:** HOÀN THÀNH HUẤN LUYỆN & NGHIỆM THU HAI ỨNG VIÊN PARETO TRÊN GPU

---

## 1. MỤC TIÊU KHOA HỌC & BẢN CHẤT BÀI TOÁN
- **Mục tiêu:** Định vị chính xác khe hở phát sinh âm chèn (Exact-Gap Insertion Detection) trên toàn bộ $N+1$ khe hở của câu nói ($20.354$ khe hở trên 599 câu `DEV_SEARCH`), tách bạch hiện tượng chèn âm thừa (epenthesis) khỏi hiện tượng kéo dài do đồng phát âm (coarticulation).
- **Đặc thù phân bố cực đoan:** Trong $20.354$ khe hở, chỉ có đúng **$125$ khe hở có âm chèn thật ($0.61\%$)**, $99.39\%$ là âm tính. Do đó, việc tối ưu hóa bài toán này đòi hỏi sự cân nhắc cẩn trọng giữa độ thu hồi ($F_1$) và độ chọn lọc cảnh báo sai (Precision / False Warning Rate).

---

## 2. DUY TRÌ SONG SONG HAI ỨNG VIÊN PARETO CHO TEACHER I
Thay vì gán nhãn một mô hình duy nhất là "tốt nhất trên mọi phương diện", chiến dịch chính thức lưu trữ và chuyển giao **hai ứng viên chuyên biệt với trade-off rõ ràng**:

| Ứng viên chuyên gia | Mã hiệu thử nghiệm | Vai trò thiết kế | Precision | Recall | $F_1$-Score | Số ca cảnh báo sai (FP) | FWR (cảnh báo/câu) | Đánh giá & Khuyến nghị sử dụng |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **I-F1** | `TEACHER_I_EXP01` | **Tối đa hóa $F_1$ & Độ bao phủ** | $16.09\%$ | **$11.20\%$** | **$0.1321$** | $73$ | $0.1219$ | Phù hợp khi mục tiêu là thu thập tối đa bằng chứng lỗi âm chèn cho mô hình Student distillation. |
| **I-Precision** | `TEACHER_I_CHAMPION` | **Chọn lọc cao & Giảm cảnh báo sai** | **$21.57\%$** | $8.80\%$ | $0.1250$ | **$40$** | **$0.0668$** | Phù hợp khi làm chốt chặn an toàn sư phạm, giảm $45\%$ số ca phạt nhầm học viên. |

---

## 3. LƯU Ý KHOA HỌC VỀ ĐIỂM NGƯỠNG $\tau = 0.95$
Tại ngưỡng cắt cực trị $\tau = 0.95$, mô hình `TEACHER_I_CHAMPION` có độ chính xác hiển thị là $100.0\%$. Tuy nhiên:
- Số ca phát hiện: $\text{TP} = 1, \text{FP} = 0$.
- Độ thu hồi (Recall) tụt xuống chỉ còn: $1 / 125 = \mathbf{0.8\%}$.
- **Kết luận thẩm định:** Điểm ngưỡng $\tau = 0.95$ là một **hiện tượng biên thống kê (boundary artifact)** khi mô hình chỉ dám kích hoạt đúng duy nhất 1 dự đoán. Điểm vận hành này **gần như không có giá trị chẩn đoán thực tế** nếu sử dụng độc lập, và không được dùng làm căn cứ tuyên bố an toàn sư phạm đơn lẻ.

---

## 4. KẾT QUẢ TÁI KIỂM ĐỊNH TRỰC TIẾP TRÊN GPU & CHECKPOINT
- **Checkpoint I-F1:** `models/teachers/TEACHER_I_EXP01_checkpoint.pt`
- **Checkpoint I-Precision:** `models/teachers/TEACHER_I_CHAMPION_checkpoint.pt`
- **Sổ cái dự đoán:** 
  - `experiments/teacher_evaluation/TEACHER_I_EXP01_predictions.jsonl`
  - `experiments/teacher_evaluation/TEACHER_I_CHAMPION_predictions.jsonl`
- Cả hai checkpoint đều đã được xác thực 100% không rò rỉ dữ liệu (Zero Data Leakage) và tái lập chính xác các chỉ số $F_1$ trên phần cứng NVIDIA GeForce GTX 1650.
