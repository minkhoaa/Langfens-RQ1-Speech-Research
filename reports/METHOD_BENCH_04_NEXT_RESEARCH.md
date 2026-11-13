# METHOD-BENCH-04 — Hướng nghiên cứu tiếp theo

## Ưu tiên 1: cải thiện chấm exact-gap insertion

E4 event-conditioned likelihood là kết quả mới có bằng chứng lặp lại rõ nhất: tăng AP-S/O/I so với E3 trên cả ba TRAIN speaker-holdout seeds. Nhưng E4 insertion AP thấp hơn E3 trên CAL và SEARCH, P@10 bằng 0 trên CAL. Hãy giữ nguyên acoustic recognizer và thử local gap evidence trong TRAIN speaker-grouped OOF trước khi đánh giá CAL. Candidate phải được định danh riêng cho mọi `N+1` gap; kiểm tra leading/trailing gap, repeated phones và vị trí hai gap cạnh nhau. So sánh raw LR, local frame posterior mass và constrained local alignment, không suy ra insertion từ target scores.

## Ưu tiên 2: mở rộng candidate likelihood một cách có kiểm soát

Hiện E4 dùng top-4 confusion competitors và top-8 insertion phones xây từ TRAIN-fit. Thử tăng candidate coverage hoặc phonological competitor neighborhoods theo các bậc đã khai báo trước; ghi candidate count, runtime, mass/coverage và AP trên grouped TRAIN/CAL. Hãy đánh giá max, logmeanexp và logsumexp có chuẩn hóa theo số đối thủ. Không diễn giải các likelihood ratios như xác suất event đã hiệu chuẩn.

## Ưu tiên 3: đo speaker transfer ngoài hai holdout speakers

E4 lặp qua seeds nhưng TRAIN speaker holdout hiện chỉ có HJK/HKK. Tiếp theo nên tính event predictions theo bốn grouped TRAIN folds nếu có acoustic posterior hợp lệ cho từng fold; mỗi model chỉ được fit trên 9 speakers và chấm 3 speaker còn lại. Điều này trả lời transfer tốt hơn so với một cặp held-out speakers. Không lặp GRL cũ. Jitter cần lặp seed 202/303 chỉ khi O gain giữ được dấu và fold consistency; residualization hiện đã bị bác bỏ.

## Ưu tiên 4: thay vì fusion, kiểm tra complementary information

SylPh-inspired cosine margin thêm một dạng non-CTC evidence nhưng không vượt neural baselines và không hỗ trợ insertion. Nếu xem tiếp, so sánh độc lập trên grouped TRAIN các lỗi mà E4 bỏ sót, đặc biệt omission; chỉ đánh giá fusion sau khi có split-based cross-fitting weights. Không tìm trọng số trên DEV_SEARCH.

## Không nên làm ở vòng kế tiếp

- Không retrain lại ba recognizer Wav2Vec2 hiện có chỉ để lấy thêm E4 score.
- Không tăng beam width đơn thuần rồi gọi đó là full posterior; MB03 cho thấy coverage vẫn rất thấp.
- Không tích hợp E4 vào model unified trước khi insertion evidence và grouped-fold stability được cải thiện.
- Không đánh giá các C score target-only bằng Macro-3 hoặc gán I giả.

## Điều kiện để tiếp tục tích hợp

Giữ E4 như component nghiên cứu cho S/O. Chỉ cân nhắc unified integration khi E4 tái lập grouped-fold gains trên nhiều speakers, gap/I không suy giảm so với E3 hoặc được bổ sung bởi một phương pháp I độc lập, và score calibration/policy được chọn trên CAL. DEV_SEARCH vẫn chỉ là exploratory; chưa có protected holdout access.
