# Schema và guideline annotation — REALIZED-MDD-1

Đây là đặc tả triển khai M0, không phải nhãn mới do con người xác nhận. Mã thực thi: `research/realized_phone_pipeline/contract.py`.

## Hai miền event

- Target có `slot_idx=0..N-1`, reference phone, nhãn C/S/O. Xác suất xuất theo tên trường `prob_correct`, `prob_substitution`, `prob_omission`, tổng bằng 1. Quyết định thuộc đúng ba lớp.
- Gap có `gap_idx=0..N`, kể cả đầu/cuối; xác suất `prob_insertion`, quyết định boolean. Danh sách âm chèn giữ thứ tự và số lượng, không rút nhiều âm thành một âm giả.
- ID ổn định: `L2_ARCTIC|speaker|utterance|target hoặc gap|index`. Adapter evaluator dùng riêng `position` hoặc `gap_position`; không truyền chuỗi ID làm nhãn. EVAL-V3.1 biểu diễn cùng bộ khóa bằng dấu `:` trong `EventKey` nội bộ.
- Độ tin cậy là xác suất lớp được quyết định hoặc xác suất nhị phân tương ứng, ghi rõ trạng thái chưa hiệu chuẩn. Không suy diễn score likelihood hoặc xác suất chuẩn hóa trong n-best là posterior event đã hiệu chuẩn.
- Candidate âm thực tế, span và uncertainty là thông tin bổ sung; metric span/phone riêng. Không lấy boundary hoặc nhãn gold làm input inference.

## Nhãn nhận diện âm thực tế

Trình tự dựng bằng cách đọc gap 0, target 0 nếu không O, gap 1, target 1,…, gap N. Không sắp xếp lại bằng timestamp vì omission, âm lặp và biên có thể gây đổi danh tính. Token provenance giữ event và chỉ số âm trong insertion.

`realized_phone=null` của O nghĩa là không phát ra token. Null ở target khác nghĩa là thiếu supervision; không điền bằng reference và không biến thành O. `ERR`, `SPN`, `D_`, `Z_`, `Y_`, `ER)` chưa được ánh xạ âm chính xác trong schema mới. Không dùng chúng như một âm cụ thể hoặc tự sửa chính tả âm vị.

`exact_phone_sequence=null` khi raw annotation chưa đối chiếu được, khi có âm thiếu/chưa xác định, hoặc có dấu deviation cần policy riêng. `observed_phone_tokens`, `raw_phone_labels`, lý do và provenance vẫn giữ nguyên. Policy strict hiện tại chỉ là cách tạo một tập supervision chắc hơn; nó làm thay đổi thành phần dữ liệu, chưa chứng minh nhãn còn lại đều đúng và chưa được chọn làm recipe tối ưu.

Nguồn định nghĩa raw label: [L2-ARCTIC documentation](https://psi.engr.tamu.edu/l2-arctic-corpus-docs/), phần Manual annotation: `CPL,PPL,s`, `sil,PPL,a`, `CPL,sil,d`; `err` là âm khó xác định, `*` là deviation. Parser mới giữ nguyên raw label và từ chối tag lạ; parser lịch sử đã bỏ `*`. Đây là lý do cần audit uncertainty trước khi train.

## Reference và inference

`train_inference_requests.jsonl` chỉ chứa đường dẫn audio, prompt, chuỗi reference cùng identity/provenance; không chứa realized phones, error labels, gaps hoặc boundary. Đây là allowlist dữ liệu đầu vào, chưa phải inference service đã hoàn thiện.

Reference hiện lấy từ manifest lịch sử, vốn dựng bằng annotation. Nó là population cố định để so với baseline cũ; không chứng minh khả năng tự tạo reference chuẩn từ văn bản. M1 cần so sánh riêng prompt → lexicon/G2P với reference lịch sử, thống kê mismatch và định nghĩa chính sách biến thể trước khi đưa ra kết luận về inference từ text. Không remap gold event một cách hậu nghiệm để làm metric đẹp hơn.

## Quy trình audit bởi con người cần thực hiện

1. Mỗi đoạn được hai annotator nghe độc lập, không nhìn prediction model; sau đó adjudicator xem bất đồng. Lưu reviewer ID, thời gian, version guideline và quyết định cuối, không ghi đè nhãn nguồn.
2. Xác nhận intended reference và biến thể hợp lệ trước khi gán lỗi. Xử lý riêng lược âm/nối âm tự nhiên; chưa có danh sách biến thể được thống nhất thì đánh dấu cần adjudication.
3. Phân biệt silence, hesitation, tiếng không rõ và một âm chèn thực. Không dùng duration, blank hoặc confidence thấp để kết luận O.
4. Phát âm trung gian có thể giữ nhiều candidate/nhãn mềm khi annotator có cơ sở; không ép về âm chuẩn. Insertion nhiều âm phải ghi thứ tự và gap; âm lặp hoặc S so với O+I cần lưu cách giải thích cạnh tranh.
5. Báo cáo agreement trước adjudication, tỷ lệ nhãn thiếu và kết quả theo nhóm mẫu. Lấy mẫu ngẫu nhiên song song với mẫu có vấn đề; đánh giá agreement có trọng số thiết kế nếu suy rộng. Không dùng chỉ mẫu khó để ước tính lỗi cả corpus.

M0 xuất queue có provenance với các cột reviewer/adjudication để trống. Không có annotator mới nào đã làm việc; không có “human gold” mới được tạo. Recording condition và adjudication history không có trong manifest hiện tại; giữ trạng thái thiếu, không suy đoán từ speaker/L1.
