# Đề xuất xem lại protocol LANGFENS — 2026-11-13

Trạng thái: **DRAFT_NOT_ADOPTED**. Đây là đề xuất nghiên cứu, không phải protocol đã được áp dụng hay chứng nhận đạt chất lượng. Không thay đổi protocol, benchmark, GOAL, nhãn hoặc kết quả cũ; không chạy training mới.

## Kết luận

Nên dừng vòng lặp đề xuất thêm kiến trúc và thiết kế lại protocol dữ liệu–supervision–đánh giá trước. Chưa có bằng chứng đủ để kết luận SOI không thể học, nhãn sai toàn bộ, hay một phương pháp mới chắc chắn đạt 80%. Việc giới thiệu lại CROTTC-IF, PER-MDD, X-MDD mà chưa đối chiếu đầy đủ kho thử nghiệm là thiếu sót; không coi ba hướng đó là ý tưởng chưa thử.

## Bằng chứng và giới hạn

- Có 113 thư mục EXP-*, 110 dòng COMPARISONS, 14 dòng EXPERIMENT_REGISTRY và 12 dòng ALL_METRICS_MASTER. Đây không phải bốn tập độc lập; không cộng chúng thành số thử nghiệm. Số thư mục không chứng minh số lần tái lập paper đầy đủ. Review này chưa replay toàn bộ lịch sử và chưa xác định đầy đủ mọi lần chạy của ba phương pháp người dùng nhắc.
- Protocol realized-phone hiện tại chỉ dùng 1.243/1.800 câu cho supervision chuỗi đầy đủ; 557 câu không đủ điều kiện, nhưng vẫn được đánh giá MDD. Có 462 câu mang issue deviation marker; các issue có thể trùng nhau. Đây là lệch miền supervision cần kiểm tra, chưa chứng minh nguyên nhân nhân quả của điểm thấp. Không suy rộng con số này cho mọi experiment cũ.
- Trên 1.800 câu, support S/O/I lần lượt 7.381/1.825/560. Recall mỗi lớp ≥80% cần tối thiểu 7.813 TP. FWR ≤0,20 chỉ cho phép 360 FP; precision gộp các cảnh báo đúng loại khi đó phải ≥95,60%. Đây là điều kiện cần, không phải mâu thuẫn toán học, cũng không đòi precision riêng mọi lớp đều 95,60%.
- M6 tại policy hiệu chỉnh cho budget 0,20 có macro F1 khoảng 8,84–10,44%; FWR held thực tế 0,203–0,223. Budget trên calibration không bảo đảm held cũng đạt budget. Không thể gọi đây là kết quả đáp ứng sản phẩm.
- CTC chuỗi phoneme không trực tiếp tối ưu đầy đủ nhãn sự kiện S/O/I. Nhãn S có cả trường hợp deviation; không được mặc định mọi S là thay thế bằng một phoneme rời rạc khác. Chưa có audit người độc lập để định lượng mức bất đồng nhãn.

Nguồn và SHA-256: `evidence.json`; các số trên có phạm vi là artifacts được liệt kê trong đó.

## Protocol kế tiếp được đề xuất

1. **Chốt bài toán sản phẩm trước khi chọn loss.** Nếu vẫn cần chẩn đoán đủ S/O/I: giữ mục tiêu precision và recall ≥80% cho từng lớp; báo TP/FP/FN, support và FWR. Định nghĩa “accuracy” rõ mẫu số; accuracy chứa đa số CORRECT không thay thế recall lỗi. AP là chỉ số xếp hạng phụ, không phải nghiệm thu 80%. Đăng ký rõ FWR 0,20 có là điều kiện sản phẩm hay chỉ một điểm trên đường cong.
2. **Audit gold trước.** Dùng batch 100 câu đã chuẩn bị tại `experiments/realized_phone_pipeline_20261113/m3_m6/m5/human_review/ANNOTATION_TEMPLATE.jsonl`: hai người gán nhãn độc lập, nghe audio, không xem dự đoán model; phân xử bất đồng. Báo agreement riêng detection, loại S/O/I và vị trí gap. Tách báo cáo 50 câu chọn theo bất đồng/độ tin cậy và 50 control ngẫu nhiên; không ước lượng tỷ lệ lỗi toàn bộ quần thể từ cả batch chọn lọc. Nếu độ thống nhất thấp, sửa hướng dẫn và adjudicate trước khi train.
3. **Tách các đầu ra và nguồn supervision.** Đánh giá riêng phát hiện lỗi, phân loại S/O/I và định vị chính xác. Dùng nhãn event đã biết cho detection/type kể cả khi realized phone không biết; chỉ mask phần phone identity chưa xác định. Không đoán phone để lấp nhãn, không đưa gold TextGrid timestamp vào inference. Đo riêng deviation và substitution có phone identity rõ. Nếu đổi ontology phải version nhãn mới và công bố bảng ánh xạ.
4. **Giữ strict benchmark làm mốc hồi quy.** Exact-gap insertion và mọi FP/FN vẫn được báo cáo. Có thể thêm track xử lý vùng mơ hồ nếu người gán nhãn độc lập xác định ambiguity trước khi chấm; không chọn alignment có lợi theo dự đoán, không bỏ ca khó sau khi xem kết quả. Binary detection chỉ là track phụ; chuyển sang sản phẩm binary là đổi phạm vi, không phải đã giải SOI.
5. **Sửa quy trình chọn model/policy.** Tách FIT, CAL, development validation theo speaker; chọn threshold và phần ngân sách target/gap chỉ trong CAL, không mặc định tỷ lệ 80/20 là tối ưu. Đăng ký tổng budget, cấu hình và quy tắc dừng trước chạy. Những speaker đã dùng lặp lại là development; cần dữ liệu speaker mới sau freeze để xác nhận, không đổi tên một fold cũ thành test mới. Giữ official test đang niêm phong nguyên trạng.
6. **Chạy một pilot có khả năng bác bỏ giả thuyết.** Sau audit nhãn, kiểm tra khả năng fit một tập TRAIN nhỏ đã adjudicate và chẩn đoán riêng lỗi reference frontend/alignment/search bằng oracle chỉ dùng phân tích. Nếu không fit nổi TRAIN thì sửa implementation/objective trước; nếu fit tốt nhưng speaker-disjoint kém thì ưu tiên dữ liệu và generalization. Chỉ sau đó đăng ký một so sánh supervision cũ với event supervision mới, cùng split, backbone, compute và evaluator; không mở thêm sweep hàng chục kiến trúc. Không có cam kết trước rằng pilot sẽ đạt 80%.

## Quyết định và độ chắc chắn

Khuyến nghị: thay protocol theo các bước trên, bắt đầu từ audit nhãn và đăng ký acceptance; chưa thay architecture. Độ chắc chắn cao về việc các kết quả hiện tại chưa đáp ứng yêu cầu; vừa về việc event supervision tận dụng dữ liệu tốt hơn; chưa biết nó cải thiện bao nhiêu. Nếu audit cho thấy agreement tốt, supervision đầy đủ và ceiling/alignment tốt nhưng generalization vẫn thấp, ưu tiên thêm dữ liệu speaker/domain phù hợp thay vì tiếp tục sửa metric. Chỉ dữ liệu mới sau freeze mới xác nhận được lợi ích cuối cùng.
