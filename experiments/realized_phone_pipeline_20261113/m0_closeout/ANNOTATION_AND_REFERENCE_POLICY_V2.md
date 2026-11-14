# Policy nhãn và reference đã chốt — M0 v2

Policy máy đọc: `locked_policy.json`; code: `research/realized_phone_pipeline/m0_policy.py`. Hiệu lực cho nghiên cứu tiếp theo, không sửa policy hoặc nhãn của tám lượt M1 đã chạy.

## 1. Miền nhãn và các trường hợp chưa chắc chắn

| Trường hợp nguồn | Quyết định nghiên cứu v2 | Việc không được suy ra |
|---|---|---|
| C từ corpus | Giữ C và identity target | Không tự gọi tất cả prompt-only audio là C |
| `CPL,PPL,s` | Giữ S, kể cả CPL/PPL cùng ký hiệu cơ sở | Không chuyển S thành C chỉ vì strip stress/deviation rồi hai token giống nhau |
| `CPL,sil,d` | Giữ O, không phát realized token | Không lấy CTC blank, silence hoặc duration ngắn làm gold O |
| `sil,PPL,a` | Giữ âm chèn theo gap và chỉ số trong chuỗi | Không làm mất âm thứ hai/thứ ba hoặc bỏ gap đầu/cuối |
| `*` | Giữ raw marker, base candidate và S/I nguồn; exclude câu khỏi primary exact-sequence CTC | Không gán base candidate như một nhãn chắc chắn |
| ERR/SPN/ký hiệu lạ/missing | Giữ event MDD, đánh dấu uncertainty; không tạo complete realized sequence | Không điền bằng reference, đoán phone hay xóa token chưa biết để làm CTC thường |
| S có cùng base symbol | Giữ S và ghi hạn chế nhận diện bằng token | Không kết luận gold sai hoặc phone-only edit đủ để chẩn đoán deviation |

Các định nghĩa raw tag/deviation đối chiếu với [publisher documentation](https://psi.engr.tamu.edu/l2-arctic-corpus-docs/). Quyết định exclude là lựa chọn thận trọng của nghiên cứu này, không phải quy tắc do publisher yêu cầu.

Xuất `train_event_policy.jsonl` cho 1.800 câu: từng event có raw label, class nguồn, candidate, lý do uncertainty, token mask và provenance. O là no-emission đã gán nhãn; missing là chưa biết. Mask chỉ phục vụ audit. CTC sequence chỉ hợp lệ khi toàn câu đủ thông tin; không giả định zeroing token loss hoặc xóa token chưa biết là partial-CTC đúng.

Strict policy vẫn giữ đúng 1.243 câu như M0 trước. Selection bias theo speaker vẫn còn và phải báo cáo. Không bỏ 557 câu khó khỏi mẫu số MDD; không dùng cùng tên “matched control” nếu hai arm khác subset hoặc update.

## 2. Biến thể hợp lệ và reference

**Profile corpus:** valid realization/error class do annotation nguồn xác định, giữ nguyên population lịch sử. Không sửa nhãn bởi dictionary, model hoặc kết quả metric. Bốn reference symbol ngoài inventory được giữ/flag như giới hạn benchmark; không tự map chúng về âm thường. Kết quả trên population này chưa chứng minh khả năng từ text bất kỳ.

**Profile supplied-reference:** reference độc lập learner realization; phone phải thuộc inventory. Quan hệ token tương đương mặc định là identity. Không tự gộp AX/AH, AO/AA, flapping, linking, assimilation hoặc coi mọi deletion tự nhiên là đúng. Profile này có chủ đích hẹp; muốn chấp nhận các biến thể ngữ cảnh phải có danh sách được xác định trước, provenance và tập audit riêng. Không gọi phạm vi hẹp là đã giữ được mọi accent/biến thể hợp lệ trong sản phẩm.

**Profile text-only nghiên cứu:** dictionary local được pin SHA-256, giữ apostrophe, bỏ stress trong phép so phone, chọn pronunciation đầu tiên một cách xác định. OOV và numeral cần explicit reference/text expansion; hàm từ chối thay vì bỏ qua. Các pronunciation còn lại được giữ như candidate. Chúng chưa được xác nhận hợp ngữ cảnh nên không tự coi tất cả là correct. “Read” là ví dụ cần ngữ cảnh; việc có hai entry trong lexicon không chọn được tense.

`train_reference_parity.jsonl` đo cả primary và best lexical-variant match với annotation. Best-variant dùng annotated reference để chọn nên là **oracle diagnostic**, không phải text-only inference. Không dùng kết quả đó để remap gold, đổi event ID hoặc tuyên bố đã giải quyết reference parity.

## 3. Silence, hesitation, nối/lược âm và ranh giới

Silence/pause không tự là insertion. Hesitation hoặc tiếng không rõ chỉ được gán I nếu có annotation đủ cơ sở; otherwise giữ ambiguous/unjudged trong audit mới. Với dữ liệu nguồn hiện tại, giữ nhãn corpus để so sánh. Không tự hợp thức hóa hoặc đổi lỗi do agent nghe/suy đoán.

Biến thể reduction/linking của người bản ngữ và accent-acceptable realizations phải được phân biệt với lỗi mục tiêu trong guideline của tập thu mới. Nếu chưa adjudicate, không ép nhãn C hay S. S so với O+I, phone lặp, insertion nhiều âm và gap đầu/cuối phải lưu candidate/giải thích cạnh tranh trước khi quyết định cuối.

## 4. Audit bởi con người và metadata nhiều annotator

L2 queue 641 câu vẫn `PENDING_HUMAN_REVIEW`: 557 câu có issue + 120 câu random phân tầng, giao 36. Đây là queue cho M5/data work, không có 641 câu đã được gán nhãn lại. M0 hoàn tất việc xác định phạm vi, phát hiện thiếu và chốt policy không cần đoán các nhãn đó.

Khi review: hai annotator nghe độc lập và ẩn model predictions; lưu reviewer ID, waveform checksum, guideline version, class/candidate/uncertainty và thời gian; adjudicator xử lý bất đồng. Chỉ tính agreement trên phần thật sự đã review; phân biệt random/targeted strata. Không dùng agreement sau adjudication như agreement ban đầu. Reviewer chưa nghe phải giữ unjudged.

SpeechOcean có sẵn năm phone-score notations mỗi từ. `speechocean_multiannotator_audit.jsonl` giữ các ca có literal disagreement hoặc bracket insertion cùng annotation nguồn. Khác chuỗi ký hiệu không đồng nghĩa bất đồng C/S/O/I; `(phone)` là score 0 có thể incorrect hoặc missed. `[phone]` có thông tin âm chèn, nhưng chưa tự là exact-gap event toàn câu theo EVAL-V3.1. Cần adapter và expert audit trước khi tạo complete realized-phone supervision. Không tự gọi phần metadata này là hoàn toàn chỉ có score, cũng không gọi nó là gold S/O/I đã tương thích.

## 5. Điều kiện đổi policy

Mọi thay đổi phải có policy version mới, giải thích bằng chứng, train-fit speaker exclusion và đối chứng giữ nguyên các thành phần còn lại. Không thay policy sau khi nhìn held score để cải thiện số liệu cùng run. Các lớp ambiguity được giữ lại cho nghiên cứu mở rộng; quyết định conservative hiện tại không khẳng định mọi phát âm khác canonical là sai.
