# LANGFENS — Phát triển phương pháp chẩn đoán và hướng dẫn sửa lỗi phát âm

Decision checkpoint: `THESIS-METHOD-DECISION-20261113-v1`.

Ngày chốt: 2026-11-13. Trạng thái: **DECISION_FROZEN — RESEARCH_NOT_COMPLETE**.

Đây là GOAL hiện hành duy nhất của repository. Theo yêu cầu người dùng, kế hoạch/milestone M0–M6 cũ được thay thế về vai trò điều hành; không tiếp tục chúng như danh sách công việc bắt buộc. Giữ kết quả, checkpoint và báo cáo cũ làm bằng chứng nghiên cứu, không xóa hoặc đổi kết luận.

## 1. Mục tiêu khóa luận

Phát triển và kiểm chứng **một phương pháp cải tiến** giúp hệ thống xác định một số nhóm lỗi phát âm có căn cứ và đưa ra hướng dẫn sửa lỗi **chính xác, rõ ràng, phù hợp**.

Đóng góp phải nằm ở phương pháp và bằng chứng thực nghiệm về cải tiến. Chỉ tích hợp model có sẵn với giao diện hoặc sinh lời khuyên không đủ để coi mục tiêu đã hoàn thành. Tái lập công trình trước là bước xây nền và đối chứng, không phải đóng góp cuối cùng.

Không đồng nhất thành công với một điểm Macro-F1, recall hoặc đứng đầu baseline. Không tự đặt precision/recall 80% cho mọi S/O/I hay FWR ≤0,20/câu thành điều kiện tốt nghiệp khi chưa được thống nhất. Vẫn báo đầy đủ chất lượng, độ phủ và cảnh báo sai; không bỏ ca khó để làm đẹp kết quả.

## 2. Nền nghiên cứu đã chọn

**CROTTC-IF là nền chính cho nhận diện/chẩn đoán.** Ưu tiên implementation, recipe, checkpoint và evaluator chính thức để xác lập đối chứng trước khi sửa phương pháp.

- [Paper — Geng và cộng sự, Interspeech 2026](https://www.isca-archive.org/interspeech_2026/geng26_interspeech.html).
- [Code — Secondtonumb/IF-MDD](https://github.com/Secondtonumb/IF-MDD).
- [Model bundle — Haopeng/CROTTC-IF-l2-arctic](https://huggingface.co/Haopeng/CROTTC-IF-l2-arctic).
- F1 71,77% là detection theo protocol paper, không phải Macro-F1 S/O/I của EVAL-V3.1 hoặc kết quả local đã tái lập.

**L2-Arctic-plus / ALMs4Learning là nguồn thiết kế benchmark và đánh giá phản hồi.** Tận dụng dữ liệu, cấu trúc giải thích/gợi ý sửa và công cụ đánh giá phù hợp; không mặc định ghép nguyên hai kiến trúc hoặc train lại toàn bộ ALM.

- [Paper — Liu và cộng sự, Findings of EACL 2026](https://aclanthology.org/2026.findings-eacl.190/).
- [Code — zoeyada/ALMs4Learning](https://github.com/zoeyada/ALMs4Learning).
- Phân biệt nhãn do người xác minh, dữ liệu sinh tự động và kết quả người đánh giá. Tương đồng văn bản không tự chứng minh lời hướng dẫn đúng ngữ âm.

ATP-LLM, PP-ATP, PER-MDD, X-MDD và các EXP cũ là tài liệu/đối chứng phụ khi có câu hỏi cụ thể. Chúng không phải các nhánh bắt buộc chạy song song. Không chọn công trình chỉ vì điểm F1 cao.

## 3. Phạm vi đóng góp và giới hạn độ phức tạp

- Chọn **một vấn đề trọng tâm, một phương pháp cải tiến chính, một baseline gốc và ablation tối thiểu cần thiết**.
- Ghi trước phạm vi lỗi, giả thuyết/cơ chế, protocol, compute budget và quy tắc chọn model, trước khi đánh giá phương pháp mới. Quyết định hiện tại chưa chốt các chi tiết này.
- Đối chiếu implementation, experiment và literature trước khi gọi cơ chế là mới. Không đổi tên OT, alignment, distillation, attribute hoặc calibration đã thử thành đóng góp mới.
- Chỉ mở nhánh bổ sung để giải quyết thiếu hụt cụ thể của giả thuyết chính; không sweep kiến trúc hoặc mở rộng đủ mọi S/O/I như nghĩa vụ mặc định.
- Không cam kết tái lập điểm paper hoặc gain trước thực nghiệm. Có code/checkpoint không đồng nghĩa recipe training vừa GPU 4 GB.

## 4. Bằng chứng thành công

| Thành phần | Bằng chứng cần tạo |
|---|---|
| Căn cứ âm học | Cảnh báo đối chiếu được với audio và nhãn độc lập; inference không nhận gold transcript, nhãn lỗi hoặc gold timestamp của câu đánh giá. |
| Chẩn đoán | Đúng vị trí và loại/đặc điểm lỗi trong phạm vi; đánh giá âm thực tế khi có gold phù hợp; ghi bất định thay vì đoán phần thiếu. |
| Hướng dẫn sửa | Tương ứng lỗi thực, đúng ngữ âm, cụ thể, dễ hiểu và phù hợp người học; có người đánh giá theo rubric định nghĩa trước. |
| Cải tiến phương pháp | So baseline gốc với phương pháp mới trên dữ liệu/protocol phù hợp và ablation; công bố khác biệt compute, dữ liệu, seed và giới hạn kết luận. |
| Toàn tuyến | Đánh giá từ audio đến phản hồi cuối cùng, tính cả cảnh báo sai và bỏ sót; không chỉ chấm lời khuyên cho lỗi đã phát hiện đúng. |

Báo precision/recall/F1 theo phạm vi, cảnh báo sai/câu và độ phủ. Bổ sung hai phép đo toàn tuyến với đơn vị/rubric đăng ký rõ:

1. **Độ đúng của phản hồi:** số cảnh báo đồng thời đúng vị trí, đúng chẩn đoán và đúng hướng dẫn / tổng cảnh báo đã đưa ra.
2. **Độ phủ hữu ích:** số lỗi thật trong phạm vi nhận được phản hồi đúng trọn vẹn / tổng lỗi thật trong phạm vi.

Báo số đếm và trường hợp không ước lượng được. Ngưỡng định lượng, cỡ mẫu và deadline chưa chốt; phải đăng ký, không tự gán trạng thái đạt. Người đánh giá cần bằng chứng phù hợp, được che danh tính phương pháp khi phù hợp, và ghi bất đồng. Phản hồi tốt không tự chứng minh người học tiến bộ sau luyện tập.

## 5. Dữ liệu và tài nguyên

- Giữ ranh giới dữ liệu và official test niêm phong theo AGENTS.md. Benchmark chứa test không đồng nghĩa được phép mở test.
- Kiểm tra training-speaker provenance của checkpoint L2 trước phép so sánh speaker-disjoint. Public checkpoint không mặc nhiên sạch với các fold local.
- CROTTC compact trong `research/three_method_pilot/` và `research/three_method_scaling/` không phải bản gốc đạt 71,77%; điểm của nó không thay kết quả tái lập chính thức.
- Giữ EVAL-V3.1 strict S/O/I và exact-gap làm mốc kỹ thuật khi phù hợp. Metric paper/word-level feedback phải ghi đơn vị, mẫu số và split; không xếp hạng trực tiếp khác protocol.
- Chọn nhóm lỗi trước đánh giá; không chỉ chọn ví dụ đẹp. Tập đã dùng nhiều lần vẫn là development, không đổi tên thành xác nhận mới.
- Thực hiện local theo quyền/giới hạn đã có trong AGENTS.md: profile tài nguyên, GPU lock, budget hữu hạn, checkpoint và provenance. Không provisioning Colab/Kaggle cho GOAL này.

## 6. Công việc tiếp theo

- [ ] **Nền thực nghiệm:** xác minh artifact chính thức, provenance, khả năng chạy và baseline ở phạm vi ghi rõ. Tận dụng audit/smoke có sẵn, không làm lại phần đã xác minh.
- [ ] **Đăng ký cải tiến:** chọn một thiếu hụt có bằng chứng, phạm vi lỗi, giả thuyết/cơ chế, đối chứng, rubric phản hồi và budget theo thời gian còn lại; kiểm tra trùng nghiên cứu cũ.
- [ ] **Thực nghiệm:** triển khai phương pháp, so sánh có kiểm soát với baseline và ablation cần thiết; giữ kết quả âm và ca thất bại.
- [ ] **Đánh giá toàn tuyến:** tạo phản hồi từ dự đoán thật, thẩm định chẩn đoán/hướng dẫn, báo độ đúng/độ phủ và giới hạn; hoàn thiện báo cáo khóa luận.

Các ô trên chưa được hoàn thành bởi milestone M0–M6 cũ. Cập nhật này chỉ đóng băng quyết định, không chứng nhận kết quả phương pháp mới.

## 7. Quyết định thay thế và lưu vết

- [GOAL trước quyết định, nguyên văn](reports/GOAL_DECISION_20261113/GOAL_SUPERSEDED.md).
- [Bản ghi quyết định](reports/GOAL_DECISION_20261113/DECISION.json).
- [Biên nhận hash snapshot](reports/GOAL_DECISION_20261113/ARCHIVE_RECEIPT.json).
- Mọi `GOAL_before_*`, `GOAL_after_*`, báo cáo/marker hoàn tất campaign cũ chỉ là lịch sử, không phải GOAL hoạt động song song hoặc bằng chứng hoàn thành khóa luận.
