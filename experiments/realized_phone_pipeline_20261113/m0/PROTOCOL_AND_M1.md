# Protocol và kế hoạch thực thi M1

Protocol máy đọc: `research/realized_phone_pipeline/protocol.json`; split cụ thể và hash TRAIN: `folds.json`. Trạng thái: đã đăng ký cho bước triển khai, chưa có checkpoint/policy mới.

## Population và phép so sánh

Dùng TRAIN 1.800 câu/12 speaker. Giữ bốn nhóm cũ; mỗi fold dùng 6 fit / 3 cal / 3 held. Mọi bước có fit, gồm scaler, vocabulary học từ dữ liệu, teacher, candidate prior và chọn hyperparameter, phải loại speaker cal/held. Với inventory có nguồn định nghĩa sẵn, version cố định độc lập held labels. Seeds so sánh 101/202; candidate xác nhận dùng thêm 303 theo đăng ký trước.

DEV_SEARCH và DEV_CAL đã có lịch sử sử dụng, chỉ dùng mô tả/audit hiện trạng trong M0. TEST lịch sử cũng đã được xem; tên `test_sealed` không làm nó trở lại thành test mới. Không mở manifest/audio TEST hoặc DEV_REPLICATION. M6 cần dữ liệu speaker mới sau freeze; M0 chưa có tập xác nhận như vậy.

Strict subset chỉ dùng để phân tích và thử supervision. Khi train trên subset, mọi đối chứng của ablation đó phải dùng cùng subset và budget. Đánh giá MDD vẫn trên toàn bộ held events, gồm câu chứa nhãn nhận diện bất định; PER chính xác có thể phải dùng subset, phải công bố coverage, support và thay đổi thành phần speaker. Không dùng strict filter để bỏ các lỗi khó khỏi mẫu số MDD.

## Operating points và tính khả thi

Đăng ký FWR 0,10 / 0,20 / 0,50 cảnh báo sai/câu; 0,20 là mốc chính. Dùng chung policy cũ để so công bằng: target/gap chia budget 80/20, score target `(max(pS,pO)-pC+1)/2`, type `argmax(pS,pO)`, threshold và tie rule chỉ fit trên calibration. Lưu toàn bộ policy. Cal đạt budget không bảo đảm held đạt budget.

Xuất precision/recall/F1/AP và TP/FP/FN/support riêng S/O/I, Macro-3 error AP/F1, raw FP tổng, FWR, cảnh báo trên C và sai loại lỗi. Gộp held predictions theo seed rồi evaluate lại, không lấy trung bình precision/recall các fold. Báo cáo theo speaker, nhóm phone, gap đầu/cuối, insertion nhiều âm và lệch gap. Giữ nhóm có recall 0; precision không xác định khi không có warning phải ghi null/status thay vì biến thành thành công.

PR và recall tại precision 0,75/0,80 dùng held labels chỉ là oracle/descriptive, giữ toàn bộ tie. Không dùng các threshold này làm policy triển khai.

Với tổng mật độ lỗi `e`, recall gộp `r`, precision gộp `p`, `FWR=e*r*(1-p)/p`. TRAIN có `e=(7381+1825+560)/1800=5,42556`. Nếu recall 0,50 và precision 0,80 thì FWR khoảng 0,678; muốn FWR ≤0,20 tại recall 0,50 cần precision ít nhất 0,9313. Đây là tính toán cho tổng event lỗi, không thay được constraint từng lớp hoặc đảm bảo model đạt được. `operating_point_feasibility.csv` ghi nhiều điểm và prevalence từng split.

**Mục tiêu nghiên cứu đề xuất:** tăng Macro-3 AP ít nhất 0,02; hoặc tăng recall ít nhất 0,05 tại precision descriptive 0,75; hướng cải thiện lặp qua cả hai seed, báo cáo từng speaker và không lớp AP nào giảm quá 0,01. Với lợi ích vận hành, so policy cal-frozen tại các budget đã đăng ký và báo FWR held thực. Nếu trade-off khác phải ghi là exploratory. Đây là cổng sàng lọc do agent đề xuất, chưa là điều kiện nghiệm thu sản phẩm được người dùng chốt. Không tuyên bố ý nghĩa thống kê từ seed/fold giao train; uncertainty dùng đơn vị speaker, kèm giới hạn mẫu nhỏ.

## Thứ tự M1 và các đối chứng cần có

1. Pin code IF-MDD và kiểm tra training-speaker manifest/licence của checkpoint. Bundle L2-trained chỉ dùng smoke giao diện nếu bị overlap; không lấy kết quả của bundle làm grouped-CV sạch. Chọn một reference sequence phù hợp có thể reproduce; ghi rõ mức fidelity. Phone-only wav2vec2 CTC theo reference Shahin là phương án nhẹ để bắt đầu, base/frozen thay large-robust phải gọi là partial adaptation.
2. Reproduce baseline MB03 continuous frozen W2V2-H6 CTC trước khi thêm capacity, giữ cả recipe nhãn cũ như control nếu thích hợp. Đối chiếu recipe strict, categorical-deviation hoặc unknown-mask bằng ablation riêng sau khi policy annotation được định nghĩa. Không đồng thời thay supervision, head và decoder rồi gán gain cho một thành phần.
3. Giữ SM13 all-13 trên cùng protocol, epoch/budget của campaign cũ. Saved predictions là neo tham khảo; nếu thay population hoặc preprocessing thì retrain baseline phù hợp. Ghi preprocessing và dtype đầy đủ; không ghép crop cache thành continuous audio sequence.
4. Export audio-only realized predictions, PER S/D/I và phone/speaker errors. So E1 deterministic edit, E4 finite likelihood control và candidate uncertainty trên cùng recognizer. Không gọi CTC blank là O. Candidate uncertainty phải đo mass/coverage và hội tụ theo beam/search width; chưa đạt thì chỉ là score xếp hạng.
5. Kiểm tra thay reference trên cùng audio: recognizer audio-only phải giữ output; event diagnosis có thể thay theo reference. So prompt-derived reference với population annotation lịch sử riêng, không tráo ontology của evaluator.
6. Smoke đủ train → predict → calibration → EVAL-V3.1 → save → reload, có ID và topology contract, trước run lớn. Tối đa 24 câu/20 update cho smoke; đây chỉ là kiểm tra plumbing, không làm số liệu hiệu quả. Tổng cap M1 2 GPU-hour hoặc 2 CPU-hour; hết budget ghi trạng thái, không tự mở sweep.

## Hardware và điểm chưa thông qua

`environment.json` ghi truy vấn NVIDIA thất bại trong runtime này. M0 không train GPU. M1 cần kiểm tra PyTorch CUDA thực tế và profile batch/duration/peak VRAM; dùng GPU lease khi chạy CUDA. Chưa thể cam kết WavLM Large/large-robust vừa GTX 1650 4 GB hoặc thời gian train. Frozen cache chỉ dùng khi provenance, normalization và waveform parity được xác minh.

Pipeline M1 vẫn chưa chạy. Cổng reproduction reference, annotation policy cho deviation/unknown, reference-from-text parity và acceptance sản phẩm còn mở. Không đánh dấu M1–M6 hoàn thành hoặc promote model từ kết quả M0.
