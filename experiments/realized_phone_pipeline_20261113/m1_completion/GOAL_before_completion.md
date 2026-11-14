# LANGFENS — Goal mới: học bằng chứng phát âm thực tế và chẩn đoán C/S/O/I

Ngày lập: 2026-11-13.

Trạng thái: M0 đã hoàn tất theo phạm vi định nghĩa/audit ở §8–9, policy v2 ngày 2026-11-13. M1 đã pin IF-MDD, smoke public CTC và chạy matched linear/MLP (mỗi arm 4 fold × 2 seed); MLP cải thiện AP/PER nhưng recall vận hành còn rất thấp, không promote. Full reference reproduction, matched controls còn lại, M2–M6 và goal tổng thể chưa hoàn thành; product acceptance chỉ là đề xuất chưa được người dùng chốt.

## 1. Mục tiêu

Xây dựng và kiểm chứng một pipeline nhận audio cùng nội dung người học được yêu cầu đọc, sau đó xác định chính xác:

- **CORRECT (C):** âm mục tiêu được phát âm phù hợp với các biến thể hợp lệ đã định nghĩa.
- **SUBSTITUTION (S):** âm mục tiêu được thay bằng âm khác hoặc phát âm lệch theo quy tắc gán nhãn của bộ dữ liệu.
- **OMISSION (O):** âm mục tiêu bị bỏ.
- **INSERTION (I):** có âm được chèn tại một khe cụ thể trong chuỗi mục tiêu.

Mục tiêu là hiệu quả trên speaker chưa xuất hiện trong train, với precision, recall và số cảnh báo sai đủ để hỗ trợ người học. Khả năng tái lập hoặc đứng đầu một tập baseline yếu chưa đủ để kết luận đã giải quyết bài toán.

Hướng ưu tiên: **nhận diện âm thực tế + dữ liệu có giám sát phù hợp**, sau đó kiểm tra lợi ích bổ sung của **thuộc tính cấu âm**.

“Train lại từ đầu” trong goal này là xây lại pipeline supervision, mô hình và quy trình đánh giá từ checkpoint speech pretrained. Khởi tạo ngẫu nhiên toàn bộ encoder chỉ là phương án riêng nếu có dữ liệu và bằng chứng phù hợp; không phải cấu hình mặc định với quy mô dữ liệu hiện có.

## 2. Bối cảnh và bằng chứng hiện tại

SM13 chưa chứng minh được năng lực MDD đáng tin cậy. TEST lịch sử của bản asymmetric ghi nhận recall S/O/I khoảng 4,34% / 14,84% / 12,30%. Ở ngưỡng được kiểm tra, không có substitution nguyên âm hoặc âm tắc nào được phát hiện. Những con số này mô tả kết quả tại điểm vận hành đó, không chứng minh nguyên nhân hay giới hạn tuyệt đối của encoder.

Chiến dịch `sm13_three_directions_20261113` đã thử ba biến thể với bốn split speaker và hai seed:

| Cấu hình | Macro-3 error AP | Macro-3 error F1 | FWR thực tế, cảnh báo sai/câu |
|---|---:|---:|---:|
| SM13 neural đủ 13 tầng, train lại theo split chung | 0,2488 | 0,1274 | 0,2269 |
| Thêm 600 câu bằng pseudo-label | 0,2472 | 0,1323 | 0,2342 |
| Temporal CNN target + gap SM13 | 0,1952 | 0,1004 | 0,1858 |
| Frozen control năm tầng giữa cho LoRA | 0,2338 | 0,1274 | 0,2147 |
| LoRA năm tầng giữa + gap SM13 | 0,2339 | 0,1264 | 0,2131 |

Đây là trung bình hai kết quả gộp theo seed trên TRAIN CV, không so trực tiếp với TEST lịch sử và không đại diện toàn bộ deployed asymmetric/E4. Pseudo-label chỉ bổ sung câu từ cùng 12 speaker; chưa bổ sung speaker mới hoặc nhãn lỗi do con người xác nhận. LoRA chỉ được thử với lịch adaptation ngắn. Temporal CNN chỉ là một cấu hình cụ thể.

Các thử nghiệm CTC/E4, soft alignment, focal loss và auxiliary head trong lịch sử repo cũng phải được đọc trước khi thiết kế lại. Chỉ thử lại khi nêu được thay đổi cụ thể về supervision, dữ liệu, mô hình hoặc đối chứng; không đổi tên một hướng cũ rồi xem là giả thuyết mới.

Bằng chứng và giới hạn: [tổng hợp SM13](experiments/sm13_three_directions_20261113/SYNTHESIS_VI.md), [kiểm toán các nhánh](experiments/sm13_three_directions_20261113/temporal/AUDIT.md), [định hướng sau CTC cũ](reports/METHOD_BENCH_03_NEXT_RESEARCH.md).

## 3. Hợp đồng đầu ra

Với N âm vị chuẩn, pipeline phải xuất đúng:

- N target event, mỗi event có xác suất hợp lệ cho C/S/O, quyết định, độ tin cậy và event ID ổn định.
- N+1 gap event, gồm cả đầu/cuối câu, mỗi event có xác suất chèn và quyết định.
- Khi khả thi, kèm âm thực tế ứng viên và khoảng thời gian hỗ trợ chẩn đoán; thông tin này có uncertainty và được đánh giá riêng.

Chuỗi âm thực tế có thể dài hơn hoặc ngắn hơn N. Một gap có thể chứa nhiều âm chèn; phải giữ chuỗi ứng viên nếu có, đồng thời ánh xạ về sự kiện có/không chèn để đánh giá EVAL-V3.1.

C/S/O thuộc miền target; I thuộc miền gap. Không giả định đây là một softmax bốn lớp trên cùng một miền mẫu. Target bị báo sai loại lỗi phải tính vào cảnh báo sai theo evaluator.

Inference nhận waveform và reference được cung cấp; không nhận nhãn lỗi, realized-phone transcript hoặc gold boundary của câu đang đánh giá.

## 4. Hướng A — Học âm thực sự được phát ra

Giả thuyết: supervision trực tiếp cho chuỗi âm thực tế có thể tạo bằng chứng giúp nhận diện và định vị lỗi tốt hơn các score CTC/E4 hiện có. Đây là giả thuyết cần kiểm chứng.

Pipeline dự kiến:

```text
Audio → encoder speech pretrained → phân phối về âm thực tế theo thời gian
                                             ↓
Reference chuẩn → đối chiếu chuỗi có xét bất định → target C/S/O + gap I
```

Yêu cầu supervision:

- Nếu người học nói /t/ thay /θ/, nhãn nhận diện thực tế là /t/.
- Nếu bỏ âm, chuỗi thực tế không chứa âm đó.
- Nếu chèn âm, chuỗi thực tế có thêm âm tại vị trí được gán nhãn.
- Prompt chỉ cung cấp nội dung được yêu cầu đọc. Không lấy prompt làm realized transcript hoặc mặc định mọi âm là CORRECT.
- Chỉ dựng realized sequence khi annotation đủ thông tin. Dòng thiếu âm thực tế phải được đánh dấu thiếu; không tự chế âm thay thế để hoàn thiện nhãn.

Thử nghiệm bắt đầu bằng recognizer được train từ realized transcript thật, với CTC hoặc mục tiêu sequence phù hợp và một baseline đơn giản có thể tái lập. Việc chọn loss/decoder cần dựa vào audit dữ liệu và reproduction reference, chưa khóa một kiến trúc duy nhất.

Đo riêng chất lượng nhận diện âm, khả năng giữ lại âm phát âm sai và lỗi theo speaker/nhóm âm. PER tốt hơn chưa đủ để promote nếu AP hoặc định vị S/O/I không tốt hơn.

## 5. Hướng B — Đối chiếu âm thực tế với reference và định vị lỗi

Giả thuyết: giữ các cách giải thích âm học/căn chỉnh hợp lý có thể giảm chẩn đoán sai khi một đoạn audio phù hợp nhiều đường edit.

Các bước cần so sánh:

1. Baseline: nhận diện chuỗi âm rồi căn chỉnh edit xác định với reference.
2. Ứng viên: đối chiếu có xét phân phối âm và nhiều đường alignment, gồm match/substitute/delete/insert.
3. Quy về đúng N target và N+1 gap, giữ bất định của vị trí và loại lỗi.

Yêu cầu kỹ thuật và đối chứng:

- Đánh giá độ phủ candidate/lattice trước khi diễn giải score là xác suất sự kiện.
- Tính toán trên nhiều đường hợp lý khi khả thi; ghi rõ phần xấp xỉ và độ hội tụ theo độ rộng tìm kiếm.
- Không đồng nhất CTC blank với omission.
- Không dùng duration ngắn, silence hoặc alignment confidence làm nhãn bỏ âm chắc chắn.
- Đối chiếu âm học-only với bản có reference; kiểm tra ảnh hưởng khi thay reference trên cùng audio để phát hiện phụ thuộc quá mức vào prompt.
- Giữ CharSiu làm baseline/neo đối chứng; học alignment mới phải chứng minh lợi ích. Các soft alignment trước đã có kết quả âm.
- Báo cáo nhầm S với O+I, gap lệch một vị trí, âm lặp, nhiều âm chèn, đầu/cuối câu và đoạn không rõ tiếng.

Sự khác biệt cần tạo ra so với CTC/E4 cũ phải nằm ở chất lượng recognizer, supervision hoặc phép suy luận được kiểm chứng. Đổi decoder trên score cũ chưa đủ để xem là giải pháp mới.

## 6. Hướng C — Supervision bằng thuộc tính cấu âm

Giả thuyết: thêm mục tiêu học thuộc tính ngữ âm giúp model giữ lại các khác biệt quan trọng cho substitution, đặc biệt nguyên âm và âm tắc.

Các nhóm thuộc tính xem xét: hữu thanh/vô thanh; phương thức và vị trí cấu âm; nguyên âm cao/thấp, trước/sau, tròn môi và các đặc trưng phù hợp inventory.

- Xây bảng ánh xạ âm → thuộc tính có nguồn và version rõ ràng.
- Tạo nhãn thuộc tính từ âm thực tế đã xác nhận, không mặc định lấy thuộc tính của reference cho audio sai.
- Với phát âm trung gian hoặc annotator bất đồng, lưu ambiguity/nhãn mềm nếu có cơ sở, không ép một nhãn chắc chắn.
- So sánh cùng encoder/dữ liệu/lịch train: phone-only, attribute-only khi có ý nghĩa, và phone+attribute.
- Kiểm tra âm học-only trước khi thêm thông tin reference để phát hiện việc head chỉ đoán thuộc tính từ văn bản.
- Thuộc tính cấu âm bổ trợ nhận diện và chẩn đoán; cần cơ chế sequence/gap để xử lý omission và insertion.

Không promote chỉ vì attribute accuracy tăng. Cần cải thiện metric lỗi thực tế và không làm O/I suy giảm đáng kể.

## 7. Dữ liệu — ưu tiên nhãn thật và đa dạng speaker

Lập inventory vật lý về audio, transcript, nhãn lỗi, realized phones, boundary, speaker, accent/L1, điều kiện thu, licence, nguồn và checksum. Phân biệt dữ liệu đang có trên máy với quy mô được công bố của corpus.

Các lớp dữ liệu phải được ghi riêng:

| Dữ liệu | Cách dùng được phép về mặt khoa học |
|---|---|
| Audio có realized-phone/error label đã xác nhận | Supervision nhận diện và chẩn đoán theo schema tương thích |
| Audio chỉ có prompt | Học không giám sát hoặc pseudo-label có ghi provenance/uncertainty |
| Score phát âm/nhãn chất lượng không phân biệt S/O | Auxiliary task tương thích; không tự chuyển thành nhãn S hoặc O |
| Dữ liệu chỉnh sửa/tổng hợp | Thử nghiệm augmentation riêng; không thay thế bằng chứng trên lỗi thật |

Ưu tiên mở rộng speaker và tình huống phát âm; bổ sung lỗi nguyên âm/âm tắc, omission, insertion và các phát âm đúng dễ bị báo sai. Chọn mẫu gán nhãn bằng độ bất định/bất đồng model kèm lấy mẫu ngẫu nhiên để kiểm soát selection bias.

Cần guideline xác định biến thể hợp lệ, nối âm/lược âm tự nhiên, silence, hesitation và ranh giới lỗi. Có tập audit đa annotator, lưu bất đồng và cách adjudication. Agent không được tự nhận mình đã tạo nhãn con người hoặc biết chắc realized phone của audio chưa được kiểm chứng.

Pseudo-label phải dùng teacher loại trừ speaker đánh giá, báo cáo coverage, C/S/O distribution và chất lượng trên phần được kiểm chứng. Nhãn teacher thiên về CORRECT không được dùng làm bằng chứng đã mở rộng đủ lỗi khó.

## 8. Lộ trình và cổng quyết định

| Mốc | Công việc và artifact bắt buộc | Điều kiện chuyển bước |
|---|---|---|
| M0 — Định nghĩa và audit | Schema C/S/O/I; inventory dữ liệu; reference matrix; protocol và manifest split | Hiểu được nhãn nào có thật, thiếu gì, và phép so sánh nào hợp lệ |
| M1 — Reproduce reference và baseline | Ít nhất một reference sequence MDD phù hợp; baseline SM13 theo cùng protocol; preprocessing parity | Chạy lại được, đầu ra đúng topology; ghi rõ khác biệt so với paper |
| M2 — Recognizer âm thực tế | Model/checkpoint; realized-phone predictions; PER và phân tích lỗi theo speaker/phone | Có bằng chứng âm học hữu ích cho lỗi thật; nếu không có thì sửa dữ liệu/recognizer trước |
| M3 — Chẩn đoán và exact-gap | Baseline edit xác định, ứng viên có uncertainty, audit độ phủ và vị trí | S/O/I được đo đầy đủ; có lợi ích so với đối chứng phù hợp |
| M4 — Thuộc tính cấu âm | Attribute mapping, supervision, ablation phone/attribute | Lợi ích chuyển được sang MDD, không chỉ accuracy auxiliary |
| M5 — Dữ liệu mở rộng | Manifest nguồn mới, provenance nhãn; ablation dữ liệu và compute | Gain không chỉ đến từ thêm bước train hoặc dùng speaker bị giữ lại |
| M6 — Xác nhận và handoff | Candidate/policy freeze, đánh giá speaker mới, báo cáo và inference package | Đạt tiêu chí đã đăng ký; nếu không đạt phải ghi rõ goal chưa hoàn thành |

M0–M1 xác định phương án thực thi trước khi chạy sweep lớn. Dữ liệu có thể được cải thiện song song; M2–M3 là đường nghiên cứu chính, M4 là kiểm tra bổ sung. M5 không bị trì hoãn nếu audit M0 cho thấy dữ liệu hiện tại thiếu supervision cần thiết.

## 9. Đánh giá và tiêu chí thành công

**Metric chính:** precision/recall/F1/AP riêng S/O/I; Macro-3 error AP/F1; FWR tổng và số cảnh báo sai nguyên; sai loại lỗi; exact-gap insertion. Với C, đo false warning trên phát âm đúng và khả năng giữ các biến thể hợp lệ. Accuracy tổng không thay thế các metric này.

Báo cáo thêm theo speaker, accent/L1, nhóm âm, độ dài câu, nhóm lỗi và mức bất định. Cần nêu support và số đếm; không bỏ nhóm có recall bằng 0 ra khỏi kết luận.

**Hai loại phép đo phải tách rõ:**

- Vận hành: chọn threshold/calibration bằng dữ liệu hiệu chuẩn, đóng băng rồi áp dụng sang speaker đánh giá.
- Chẩn đoán: PR curve, recall ở precision cố định hoặc FWR khớp dùng nhãn đánh giá. Ghi rõ oracle/descriptive, không trình bày như policy triển khai.

Đăng ký các operating point trước khi xem kết quả. Giữ mức tham chiếu FWR 0,20 cảnh báo sai/câu để nối với nghiên cứu cũ, đồng thời báo cáo nhiều operating point hợp lý. Calibration đạt 0,20 không chứng minh speaker mới cũng đạt 0,20.

Trước khi khóa mục tiêu sản phẩm, tính tính khả thi giữa error prevalence, recall, precision và FWR. Không tự đặt các ngưỡng mâu thuẫn nhau. Mức precision/recall tối thiểu cho sản phẩm chưa được người dùng xác định trong phiên này; M0 phải đề xuất và ghi rõ mức nào là mục tiêu nghiên cứu, mức nào là điều kiện nghiệm thu đã thống nhất.

**Đề xuất cổng tiến bộ nghiên cứu:**

- Recall tăng có ý nghĩa thực dụng tại precision tương đương, hoặc precision tăng tại recall tương đương, với đối chứng cùng dữ liệu/split và phạm vi lớp.
- Nếu tuyên bố tăng đồng thời precision và recall, phải thể hiện đúng điều đó ở operating point tương ứng; tăng F1 đơn lẻ chưa đủ.
- Có cải thiện xếp hạng hoặc lợi ích vận hành lặp lại qua speaker và seed; không chỉ tăng recall bằng tăng cảnh báo sai.
- Tuyên bố giải quyết đủ C/S/O/I cần bằng chứng cho tất cả các miền sự kiện; head dùng chung score baseline không chứng minh lớp đó đã cải thiện.
- Có xác nhận trên speaker đánh giá mới sau khi đã freeze phương pháp. Giữ kết luận exploratory nếu chưa có tập xác nhận phù hợp.

Các cổng này là kế hoạch kiểm chứng, không phải kết quả đã đạt. Không đánh dấu mục tiêu tổng thể hoàn thành chỉ vì một mô hình tốt hơn SM13 một lượng nhỏ.

## 10. Chống leakage và so sánh công bằng

- Chia theo speaker trước mọi học tham số, feature normalization có fit, pseudo-label, chọn siêu tham số hoặc threshold. Dữ liệu bổ sung của speaker giữ lại cũng bị loại khỏi adaptation.
- Dùng TRAIN grouped CV để phát triển. Các holdout đã được xem trong lịch sử không được đổi tên thành test mới chưa bị sử dụng.
- Tái sử dụng protocol 6 fit / 3 cal / 3 held và seeds 101/202 khi cần so trực tiếp chiến dịch cũ; nếu dữ liệu/split thay đổi phải train lại các baseline tương ứng.
- Xác nhận ứng viên triển vọng bằng ít nhất ba seed được đăng ký trước; không chọn seed tốt nhất để đại diện mô hình.
- Ghi rõ budget dữ liệu, optimizer updates, mức tiếp xúc nhãn thật, trainable params và công sức tuning. Có đối chứng đủ để tách hiệu ứng kiến trúc, dữ liệu và thời gian train.
- Chạy smoke đầy đủ train → predict → calibration → evaluate → save trước run lớn. Kiểm tra normalization, sample rate, PCM/dtype, inventory, mask và cache parity.
- Sai khác fp16/fp32 giữa hai nhánh phải được kiểm soát hoặc đo rõ; không gán chênh lệch rất nhỏ cho học model khi input chưa khớp.
- Thời lượng train cần được giải thích bằng learning curve và budget được đăng ký. Kết quả âm của lịch train ngắn chỉ áp dụng cho lịch đó; không suy ra mọi fine-tune đều vô ích.
- Seed lặp và fold có train set giao nhau không phải mẫu độc lập. Báo cáo uncertainty theo đơn vị speaker phù hợp, không dùng bootstrap để tạo p-value vượt quá thông tin dữ liệu.
- Giữ checkpoint/policy lịch sử làm mốc; lưu mọi run mới, thất bại và kết quả bị loại với lý do rõ ràng.

## 11. Reference và khoảng trống cần chứng minh

Không dùng giả định “Langfens gần như không có reference” làm cơ sở cho thiết kế hoặc tuyên bố novelty. MDD có nghiên cứu về substitution, deletion và insertion từ trước. Sự tương đương về metric, exact-gap, speaker split và khả năng triển khai cần được kiểm tra từng reference.

Reference khởi đầu, đã được tra cứu trong phiên thảo luận:

1. [Qian, Meng & Soong, 2016 — A Two-Pass Framework of Mispronunciation Detection and Diagnosis](https://www1.se.cuhk.edu.hk/~hccl/publications/pub/2016_07401018.pdf): phát hiện ba loại lỗi và nhận diện âm thực tế ở đoạn lỗi.
2. [Shahin & Ahmed, Interspeech 2024 — Phonological-Level Mispronunciation Detection and Diagnosis](https://www.isca-archive.org/interspeech_2024/shahin24_interspeech.pdf): supervision các thuộc tính ngữ âm, wav2vec2 và CTC đa nhãn.
3. [CROTTC-IF, preprint 2026 — Beyond Acoustic Sparsity and Linguistic Bias](https://arxiv.org/abs/2604.22133): tách bằng chứng âm học khỏi canonical prior; cần đọc và tái lập trước khi chấp nhận các kết luận thực nghiệm.
4. [Microsoft Pronunciation Assessment](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-pronunciation-assessment): reference sản phẩm về điểm âm vị và lỗi ở cấp từ, không mặc định tương đương contract âm vị/exact-gap của Langfens.

Lập reference matrix: task, nhãn, corpus, speaker split, inference input, boundary oracle, loại lỗi hỗ trợ, metric, code/checkpoint, licence, compute và khả năng ánh xạ sang EVAL-V3.1. Điểm F1 công bố không được so trực tiếp nếu chưa thống nhất protocol.

## 12. Artifact và điều kiện dừng

Mỗi chiến dịch cần có protocol/config versioned, manifest/hash, provenance dữ liệu/nhãn, checkpoint, learning curve, command/log, environment, keyed predictions và cal policy. Báo cáo chứa classwise/per-speaker results, ablation, lỗi còn lại, runtime/VRAM và hướng dẫn tái lập. Có kiểm tra độc lập bằng evaluator cho ứng viên được đề nghị promote.

Phần cứng hiện có là GTX 1650 4 GB; profile trước khi cam kết kiến trúc hoặc thời gian. Tận dụng cache phần encoder frozen khi hợp lệ, dùng khóa GPU khi chạy nhiều job và ghi chi phí trích đặc trưng riêng với chi phí train.

Dừng hoặc đổi một nhánh khi bằng chứng lặp lại cho thấy không có lợi ích thực dụng, khi candidate coverage không đủ để diễn giải score, hoặc khi gain phụ thuộc leakage/nhãn không phù hợp/cảnh báo sai tăng. Đăng ký giới hạn budget cho từng thí nghiệm; không sweep vô hạn. Giữ kết quả âm và nêu câu hỏi còn chưa trả lời.

**Phạm vi khi lập tài liệu (lịch sử):** yêu cầu ban đầu là ghi goal và kế hoạch chi tiết. Người dùng đã cho phép triển khai M0 ở lượt tiếp theo. Việc hoàn thành audit/tài liệu không có nghĩa các thí nghiệm M1–M6 đã chạy hay goal nghiên cứu đã đạt.

## 13. Checklist tiến độ ban đầu

- [x] Ghi mục tiêu, bối cảnh, ba hướng nghiên cứu và contract C/S/O/I.
- [x] Ghi reference, tiêu chí đánh giá, ranh giới kết luận và bài học từ các thử nghiệm cũ.
- [x] M0: hoàn tất audit dữ liệu, định nghĩa nhãn, reference matrix, protocol/split và acceptance targets theo yêu cầu đề xuất ở §9. Đã chốt policy conservative deviation/unknown/reference, research gate và product proposal có feasibility; không coi proposal là user agreement. Xem [báo cáo tổng hợp M0](experiments/realized_phone_pipeline_20261113/m0_closeout/REPORT_VI.md).
- [ ] M1: reproduce reference và lập baseline theo protocol thống nhất. Đã train baseline realized CTC, pin IF-MDD/public CTC và smoke strict-load; matched capacity control linear/MLP hoàn tất trên 6/3/3, seeds 101/202. Full original-reference recipe, reference-from-text parity và matched controls đầy đủ còn mở; xem §17.
- [ ] M2: train recognizer bằng realized-phone supervision và phân tích bằng chứng âm học.
- [ ] M3: train/đánh giá chẩn đoán C/S/O/I với exact-gap và uncertainty.
- [ ] M4: kiểm tra lợi ích thêm thuộc tính cấu âm.
- [ ] M5: đo tác động của dữ liệu được bổ sung/gán nhãn phù hợp.
- [ ] M6: freeze, xác nhận trên speaker mới và quyết định handoff dựa trên tiêu chí đã đăng ký.

## 14. Triển khai M0 ban đầu — snapshot trước train, 2026-11-13

- Công cụ: `research/realized_phone_pipeline/audit.py`, contract và protocol JSON cùng thư mục. Tái chạy: `.venv/bin/python -m research.realized_phone_pipeline.audit`.
- Artifact và kết luận: [báo cáo M0](experiments/realized_phone_pipeline_20261113/m0/REPORT_VI.md), [schema/guideline](experiments/realized_phone_pipeline_20261113/m0/SCHEMA_AND_ANNOTATION.md), [reference matrix](experiments/realized_phone_pipeline_20261113/m0/REFERENCE_MATRIX.md), [protocol và M1](experiments/realized_phone_pipeline_20261113/m0/PROTOCOL_AND_M1.md).
- Đối chiếu raw annotation TRAIN: 1.800/1.800 khớp event manifest; strict exact-phone supervision có 1.243 câu. Giữ nhãn nguồn và uncertainty, không tự tạo nhãn con người.
- Đã phát hiện registry/audio metadata cũ không phản ánh đủ hiện trạng; xem inventory mới trước khi dùng. Không sửa/ghi đè dữ liệu và checkpoint lịch sử.
- M1 chưa chạy. Không promote từ audit; cổng M0 đầy đủ vẫn chưa thông qua khi chưa giải quyết các mục còn mở nêu trong báo cáo.

## 15. Train baseline M1 — 2026-11-13, sau audit M0

- Người dùng đã yêu cầu train. Đã chạy preflight waveform/cache, smoke toàn tuyến và 8 lượt train CPU (4 fold × 2 seed), 8 epoch/lượt, trên supervision strict thuộc speaker fit.
- Báo cáo train: [M1 training report](experiments/realized_phone_pipeline_20261113/m1/REPORT_VI.md). Mục 14 và báo cáo M0 ban đầu mô tả trạng thái tại thời điểm audit trước train; trạng thái M0 hiện hành ở mục 16.
- Realized PER trên 1.243 câu held đủ nhãn: 61,86% / 61,11% cho seeds 101/202. Macro-3 error AP trên toàn bộ held events: 0,06728 / 0,06748; FWR tại policy B=0,20: 0,18833 / 0,19833. Recall S/O/I đều rất thấp; chưa đạt yêu cầu năng lực.
- Đã lưu checkpoint/policy/keyed predictions/posteriors, replay EVAL-V3.1, kiểm tra PER bằng implementation riêng và thử inference WAV + supplied phone reference. Không dùng test/DEV_REPLICATION; không thay checkpoint lịch sử.
- Không promote. M1 reference reproduction, M2 recognizer đủ tốt và các milestone sau vẫn chưa hoàn thành. Kết quả âm chỉ áp dụng cho recipe frozen linear CTC, strict subset và budget đã chạy.

## 16. Hoàn tất M0 — policy v2, 2026-11-13

- Báo cáo hiện hành: [M0 closeout và tổng hợp](experiments/realized_phone_pipeline_20261113/m0_closeout/REPORT_VI.md). Policy máy đọc: `research/realized_phone_pipeline/m0_policy_v2.json`; registry overlay, reference matrix và bằng chứng bổ sung nằm trong thư mục closeout.
- Đã chốt cách giữ source C/S/O/I, deviation/unknown, complete-sequence mask, silence/hesitation và các profile biến thể/reference. Không đoán phone hoặc đổi source S thành C sau khi strip marker. Primary exact-sequence subset vẫn là 1.243 câu; MDD denominator giữ full held events.
- Đã kiểm tra lại 7.944 WAV + 12 archive, waveform QA và snapshot lịch sử; đo reference parity trên đủ 1.800 câu/300 prompt. Chỉ 573 câu khớp pronunciation đầu tiên; best lexical-variant oracle khớp 1.534 câu, không phải text-only inference đã đạt.
- Bổ sung audit: 837 source S có cùng base phone (836 có deviation), không nằm trong strict eligible subset; SpeechOcean có 5 notation/từ và bracket insertion candidates, nhưng chưa là complete realized/exact-gap gold. Human review L2 vẫn chưa được thực hiện và không bị giả mạo.
- Đã đăng ký research gate và đề xuất sản phẩm P≥95%, R≥50% riêng S/O/I, FWR≤0,20/câu cùng coverage; có sizing theo prevalence. Product proposal chưa được user đồng ý; thỏa thuận nghiệm thu và xác nhận thực tế là gate M6, không phải điều kiện phải tự tạo ra để đóng M0 định nghĩa.
- M0 hoàn tất không có nghĩa đã giải quyết reference inference, reproduce paper, cải thiện model hoặc tạo human gold. Các việc đó vẫn thuộc M1/M2 và M5/M6 như handoff; không train mới hoặc promote trong closeout này.
- M0/M1 snapshot cũ giữ nguyên; GOAL trước closeout được lưu với hash lịch sử và mapping tường minh. Không cập nhật policy v1 hoặc ghi lại kết quả cũ như thể đã dùng v2.

## 17. Tiếp tục M1 — reference và matched capacity, 2026-11-13

- Báo cáo: [M1 reference continuation](experiments/realized_phone_pipeline_20261113/m1_reference/REPORT_VI.md). Đăng ký trước ở cùng thư mục; code mới `m1_capacity.py`, `m1_public_ctc.py`, `verify_m1_reference.py` trong `research/realized_phone_pipeline/`.
- Pin IF-MDD commit `cca150ec4ff00e01d61a613092d1d178e661dfb2`, public CTC revision `de46aaf5778c5759c003ffa17657176fee772e77`. Public bundle thiếu training-speaker manifest đã xác minh nên chỉ smoke, không dùng làm clean grouped-CV evidence. Tokenizer bundle khác token IDs source hiện hành; giữ đúng mapping đi cùng weights.
- Public CTC strict-load encoder/head và smoke ba câu TRAIN qua waveform → phone → edit → calibration → EVAL-V3.1 → save/reload. Normalization logits và greedy collapse khớp implementation đối chiếu. Đây là interface smoke, không phải reproduce metric paper hoặc xác nhận speaker mới.
- Chạy 16 matched run: linear và IF-style MLP head, mỗi arm 4 fold × seeds 101/202, tám epoch, cùng H6 features/strict supervision/update/decoder/calibration. Cả tám linear checkpoint khớp tensor baseline cũ. MLP chỉ là partial head-topology adaptation, không IF teacher transfer hoặc CROTTC-IF reproduction.
- PER linear 61,86% / 61,11% → MLP 27,03% / 27,14%; Macro-3 AP 0,06728 / 0,06748 → 0,12460 / 0,12448. AP cả S/O/I tăng ở cả hai seed, đạt nhánh AP research screening v2. Full held population giữ nguyên; PER có mask coverage riêng.
- Tại policy cal B=0,20, FWR MLP held 0,21333 / 0,20222, recall S/O/I chỉ khoảng 0,93–1,32%. Không tuyên bố lợi ích tại FWR held bằng nhau; không promote hoặc coi đã đạt nghiệm thu sản phẩm.
- 45 test pass; replay policies/evaluator/PER và snapshot hash. Tất cả compute lượt này chạy CPU; CUDA hiện khả dụng nhưng standard fp32 Adam WavLM Large có lower bound 4,638 GiB trước activation, vượt GPU 4 GiB. Không chạy OOM hoặc mở sweep.
- M1 vẫn mở cho full reference recipe trên split sạch, matched legacy-label/SM13 controls khi cần và reference-from-text parity. Dừng sau hai arm đã đăng ký. GOAL sau M0 được giữ trong `m1_reference/input_snapshots/GOAL_after_M0_closeout.md`; verifier mới resolve snapshot cũ tường minh, không sửa manifest M0/M1 lịch sử.
