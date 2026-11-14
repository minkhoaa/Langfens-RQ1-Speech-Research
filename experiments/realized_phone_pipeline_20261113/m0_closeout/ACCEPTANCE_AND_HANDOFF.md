# Tiêu chí nghiên cứu, đề xuất sản phẩm và handoff

## M0 đã chốt điều gì?

Theo GOAL §8, đầu ra M0 là schema, inventory, reference matrix, protocol/split; cổng là hiểu nhãn có thật, phần thiếu và phép so sánh hợp lệ. Theo §9, M0 phải **đề xuất** mục tiêu và phân biệt nghiên cứu với nghiệm thu đã thống nhất. Việc chốt product acceptance, tạo human gold mới hoặc reproduce model là công việc khác; không tự coi chúng đã đạt để đóng M0.

`locked_policy.json` là chuẩn nghiên cứu hiện hành. Protocol v1 và preregistration M1 giữ nguyên cho kết quả lịch sử; v2 chỉ áp dụng các run tương lai. Tất cả mốc dưới đây phải được ghi trong config trước run.

## 1. Cổng tiến bộ nghiên cứu

- Primary: Macro-3 error AP tăng tuyệt đối ≥0,02 so matched control; hoặc recall descriptive tại precision ≥0,75 tăng ≥0,05. Nhánh nào/lớp nào được tuyên bố cải thiện phải ghi trước; không dùng Macro-3 để che một lớp chưa đo.
- Hướng cải thiện lặp ở cả seeds 101/202; không lớp S/O/I nào giảm AP quá 0,01. Xuất paired per-speaker delta/support, kể cả zero recall. Các mức này là lựa chọn sàng lọc thực dụng của nghiên cứu, không phải ngưỡng thống kê đã được corpus chứng minh.
- Oracle PR chỉ mô tả khả năng ranking, giữ cả tie; không dùng để tạo policy deploy. Muốn tuyên bố lợi ích vận hành phải dùng policy cal-frozen và báo precision/recall cùng FWR held thực tại B=0,10/0,20/0,50, không đổi threshold bằng held gold.
- Matched control phải cùng fit utterances/speakers, inventory, feature dtype/normalization, optimizer update và mức tiếp xúc nhãn thật; nếu cố ý thay một yếu tố phải có ablation giải thích. Lịch sử SM13 chỉ là context khi subset/budget khác.
- Promote cần xác nhận ít nhất seeds 101/202/303, freeze phương pháp/policy và speaker mới. Không gán p-value độc lập cho fold/seed dùng lại speaker. Đạt một research gate không đồng nghĩa giải quyết sản phẩm hay cả C/S/O/I.

## 2. Đề xuất nghiệm thu sản phẩm — CHƯA ĐƯỢC NGƯỜI DÙNG CHỐT

Đề xuất để thảo luận: **precision ≥95% và recall ≥50% riêng S/O/I; joint FWR ≤0,20/câu; đủ 100% event output**, đo cùng lúc trên fresh speakers sau freeze. Biến thể hợp lệ và cảnh báo trên C cần tập human audit phù hợp; support phải đủ để nêu uncertainty. Chưa chốt population người học/điều kiện thu nên đây chỉ là proposal, không phải cam kết hay điều kiện đã được user đồng ý.

Với error prevalence TRAIN `e=5,42556 event/câu`, tại **đúng** recall 0,50 và precision 0,95, FWR tính theo sizing khoảng 0,14278/câu; điểm này khả thi về đại số. Tại recall 0,80 với cùng precision, FWR khoảng 0,22844/câu, vượt 0,20. Vì vậy lower bounds precision/recall không tự bảo đảm FWR; phải đo cả ba ràng buộc. Tại recall 1,0 cần precision khoảng 96,44% để FWR=0,20 theo prevalence TRAIN.

`acceptance_feasibility.csv` ghi sizing cho TRAIN/DEV, không phải capability model hoặc bảo đảm trên fresh speakers. Không suy ra CI/ý nghĩa thống kê từ sizing. Proposal không ngăn hoàn tất M0 định nghĩa, nhưng phải có thỏa thuận trước quyết định nghiệm thu sản phẩm ở M6.

## 3. Phân công phần còn lại theo milestone

| Công việc | Mốc/điều kiện đúng | Trạng thái |
|---|---|---|
| Policy uncertainty/reference và missing-label handling | M0 định nghĩa | Đã chốt conservative v2 và kiểm tra thực thi |
| Acceptance sizing, research gate, phân biệt proposal/user-agreed | M0 | Đã xuất; product agreement không bị giả mạo |
| Pin/reproduce IF-MDD hoặc sequence reference tương thích | M1 | Chưa hoàn thành; xem REFERENCE_MATRIX.json |
| Đối chứng SM13/legacy CTC cùng subset/update; preprocessing và reference inference parity | M1 | Còn mở; diagnostic mismatch đã đo, chưa giải quyết |
| Recognizer có bằng chứng hữu ích cho lỗi thật | M2 | Baseline hiện tại chưa đạt; không promote |
| Uncertainty decoding, exact-gap và candidate coverage | M3 | Chưa hoàn thành |
| Attribute mapping/ablation thực sự từ realized supervision | M4 | Chưa hoàn thành |
| Human adjudication, adapter SpeechOcean, thêm speaker | M5/data work song song | Chưa có nhãn con người mới |
| Chốt sản phẩm, valid variants đã human-audit, fresh-speaker confirmation | M6 | Chưa hoàn thành |

## 4. Handoff thực thi

M1 phải bắt đầu bằng kiểm tra source/recipe chính thức và lập fidelity matrix; pin commit/checkpoint/training speakers trước khi chạy. Không dùng public L2-trained checkpoint trên speaker đã từng train model đó rồi gọi là speaker-disjoint. Profile nhỏ trước khi cam kết large backbone; CUDA phiên trước không khả dụng. Full reference reproduction chưa được thực hiện trong closeout này.

Nếu tiếp tục recognizer, thiết kế matched comparisons tách supervision, capacity và optimizer budget trước run. Baseline 8 epoch đã chạy chỉ là kết quả âm của một recipe; học curve còn giảm không tự chứng minh cứ train lâu sẽ tốt hơn. Chỉ chuyển M3/M4 sau khi có bằng chứng recognizer hoặc data phù hợp. Không mở sweep hoặc train mới trong lượt đóng M0 này.
