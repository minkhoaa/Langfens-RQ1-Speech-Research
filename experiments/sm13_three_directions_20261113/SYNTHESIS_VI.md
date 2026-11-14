# SM13 — tổng hợp thử nghiệm ba hướng, 2026-11-13

**Kết luận:** Ba subagent `gpt-6-luna` đã thực thi và tự benchmark ba hướng. Các biến thể đã chạy chưa chứng minh được khả năng tăng mạnh đồng thời precision và recall. Chưa có cơ sở thay checkpoint SM13 hiện tại bằng một trong các ứng viên này.

**Giao thức và cách đọc số liệu.** Dùng riêng 1.800 câu TRAIN của L2-ARCTIC, 12 speaker chia bốn nhóm. Với mỗi lượt: sáu speaker train, ba speaker hiệu chuẩn và ba speaker đánh giá, không trùng nhau. Chạy seeds 101 và 202. Mỗi seed gộp dự đoán của bốn nhóm đánh giá thành đúng 1.800 câu, 59.562 target và 61.362 gap, rồi tính lại bằng `UnifiedEvaluatorV3` EVAL-V3.1. Các con số dưới đây là trung bình của hai kết quả gộp theo seed; không lấy trung bình precision/recall của các fold.

Ngưỡng được cố định trên nhóm hiệu chuẩn với ngân sách danh định 0,20 cảnh báo sai/câu. FWR thực tế trên nhóm đánh giá có thể vượt 0,20; không sửa ngưỡng bằng nhãn đánh giá. Các seed dùng lại cùng speaker nên đây là mô tả biến thiên, không phải các mẫu độc lập hay bằng chứng thống kê xác nhận.

Baseline là **thành phần neural SM13 scalar mixing đủ 13 tầng được train lại theo split mới**, với target 12 epoch và gap 5 epoch theo cấu hình gốc. Benchmark này không chạy lại toàn bộ hệ thống asymmetric có E4 và không thể so trực tiếp những số dưới đây với TEST lịch sử. Không mở thêm tập test niêm phong để chọn ứng viên.

| Cấu hình | Macro-3 AP | Macro-3 F1 | FWR sai/câu | Precision-S | Recall-S |
|---|---:|---:|---:|---:|---:|
| SM13 neural 13 tầng, baseline | 0,2488 | 0,1274 | 0,2269 | 67,11% | 5,51% |
| Thêm dữ liệu bằng pseudo-label | 0,2472 | 0,1323 | 0,2342 | 67,24% | 6,06% |
| Temporal CNN target + gap SM13 | 0,1952 | 0,1004 | 0,1858 | 64,62% | 4,47% |
| Đối chứng frozen 5 tầng giữa cho LoRA | 0,2338 | 0,1274 | 0,2147 | 66,18% | 3,94% |
| LoRA 5 tầng giữa + gap SM13 | 0,2339 | 0,1264 | 0,2131 | 66,25% | 3,92% |

S/O/I lần lượt là thay thế/bỏ/chèn âm. Macro-3 chỉ gồm ba loại lỗi đó. Bảng đầy đủ từng lớp và từng seed nằm trong [CSV tổng hợp](shared/baseline_pooled_oof.csv).

**1. Thêm dữ liệu bằng pseudo-label: cải thiện nhỏ ở điểm vận hành, chưa cải thiện xếp hạng lỗi.** Đã lấy 600 câu mới, 50 câu cho mỗi speaker TRAIN, không trùng các câu có nhãn trong manifest. Mỗi fold chỉ dùng câu mới thuộc sáu speaker train. Teacher chỉ học từ nhóm train đó, giữ nhãn mềm có độ tin cậy tối đa ít nhất 0,90. Nhãn âm vị chuẩn trong prompt chỉ dùng làm tham chiếu/căn chỉnh, không bị xem là bằng chứng phát âm đúng.

Lịch train và số bước tối ưu khớp baseline; mỗi batch thay 25% mẫu có nhãn bằng pseudo-label. Vì vậy số lần tiếp xúc với nhãn thật giảm, đây cũng là giới hạn của phép so sánh. Recall-S tăng khoảng 0,55 điểm phần trăm; precision-S gần như giữ nguyên. AP giảm 0,0016, trong khi F1 tăng 0,0049 và FWR tăng 0,0072. Seed 101 giảm F1, seed 202 tăng F1; cả hai đều vượt FWR 0,20. Chưa thể xem đây là một cải thiện ổn định về năng lực phân biệt.

96,25% trong 39.767 dòng pseudo được giữ qua tám lượt được teacher dự đoán là CORRECT. Đây là dự đoán chưa được con người xác nhận; các dòng lặp qua fold/seed không phải 39.767 quan sát mới độc lập. Thử nghiệm chỉ kiểm tra self-training trên cùng 12 speaker, chưa kiểm tra bổ sung speaker/accent mới hoặc nhãn lỗi mới từ con người. [Báo cáo nhánh dữ liệu](data_expansion/BENCHMARK_REPORT.md).

**2. Temporal CNN: kết quả âm rõ với biến thể đã thử.** Dùng chuỗi 17 frame tầng 6 quanh neo CharSiu, hai lớp Conv1d và attention pooling. Ứng viên target-only dùng chung gap head SM13 để tách tác động lên target. Macro-3 AP giảm ở cả tám lượt, so với cả baseline 13 tầng và đối chứng MLP pooling tầng 6. Thử thay cả gap head cũng kém hơn: AP trung bình 0,1843 và F1 0,0841.

Kết quả phản đối cấu hình CNN/cửa sổ/lịch train cụ thể này. Nó chưa bác bỏ mọi temporal model. Cache frame dùng fp16; sai số so với đặc trưng gốc đã được đo và ghi trong kiểm toán. [Báo cáo temporal](temporal/TEMPORAL_RESULTS.md).

**3. LoRA: hiệu ứng rất nhỏ và không nhất quán.** Adapter được gắn vào q/v của các block transformer 4–8, dùng mixture các hidden state 5–9. Có hai epoch warmstart head và hai epoch cho cặp frozen/LoRA theo cùng lịch cập nhật. Đối chứng đã được chạy lại từ cùng prefix fp16, cùng warmstart và cùng gap SM13, để tránh trộn hiệu ứng học với sai số cache.

So với đối chứng khớp, AP tăng 0,000739 ở seed 101 nhưng giảm 0,000535 ở seed 202; trung bình chỉ tăng khoảng 0,000102. F1 giảm ở cả hai seed. Precision-S trung bình tăng khoảng 0,07 điểm phần trăm, recall-S giảm khoảng 0,02 điểm phần trăm. Chưa có lợi ích thực dụng để chọn LoRA này. Lịch train ngắn và mixture chỉ có năm tầng giữa giới hạn kết luận; đây không phải phủ định mọi phương án fine-tune encoder. [Báo cáo LoRA](encoder_adapter/REPORT.md).

**Kiểm tra và tính toàn vẹn.** Hai nhóm kết quả không hợp lệ được tách khỏi phép tổng hợp: pseudo-label ban đầu dùng tiền xử lý waveform khác baseline, và đối chứng LoRA ban đầu dùng cache full-fp32 khác prefix fp16 của ứng viên. Sau sửa, waveform probe 44,1→16 kHz khớp từng mẫu với audio trong repo, tham chiếu 39 âm vị khớp toàn bộ 59.562 target, và đặc trưng probe khớp trong sai số số học. Các kết quả bị loại được giữ riêng/đánh dấu superseded, không dùng để chọn cấu hình.

Agent temporal kiểm tra độc lập các file dự đoán dữ liệu và LoRA bằng evaluator, đối chiếu AP/F1/FWR, số cảnh báo, cardinality và speaker. Số liệu khớp với báo cáo. Ba checkpoint gốc target/gap/CTC vẫn khớp SHA-256 đã ghi trong hồ sơ kiểm toán trước. [Kiểm toán độc lập](temporal/AUDIT.md).

[Recall tại precision 75% và 80%](temporal/pooled_held_oracle_precision_recall.csv) là chẩn đoán dùng nhãn đánh giá để chọn ngưỡng, có gộp tie đầy đủ. Những điểm này không phải kết quả vận hành có thể triển khai.

**Quyết định hiện tại:** Giữ mô hình đang đóng băng. Không promote ba cấu hình vừa chạy. Nếu tiếp tục hướng dữ liệu, câu hỏi còn mở có giá trị là hiệu quả của dữ liệu lỗi được gán nhãn thật và đa dạng speaker hơn; đợt này chưa đo được điều đó. Mọi cải thiện tương lai cần được kiểm chứng trên tập đánh giá mới chưa dùng để phát triển.

