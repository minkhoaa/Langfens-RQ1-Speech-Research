# METHOD-BENCH-05 — Tiếp theo nên nghiên cứu gì

## Ưu tiên 1: learned exact-gap localization trên grouped TRAIN

E4 mean đã chứng minh tín hiệu S/O chuyển speaker; insertion AP-I vẫn xấp xỉ E3 và temporal T2 âm tính. Thí nghiệm tiếp theo nên giữ recognizer Wav2Vec2-CTC/posteriors cố định lúc đầu, dùng predicted reference-conditioned boundary distributions và một gap-local acoustic head có thể chấm span candidate phone hoặc blank-to-phone transition. Không dùng gold timing, realized phone hay error label tại inference.

Thiết kế:

1. Dùng đúng bốn speaker-held-out fold; fit mọi normalization, competitor prior và head chỉ trên chín speaker train.
2. So sánh E3, E4 mean, T2 local baseline và một learned temporal head trên cùng N+1 gap IDs.
3. Báo AP-I, prevalence, PR curves, P@K/R@K, F1/FWR policy được fit ngoài fold, support identifiable vs collision subsets, candidate coverage và PER.
4. Chạy ít nhất seed101/202 cho đủ bốn folds; dùng DEV_CAL chỉ cho quyết định đã khai báo; giữ DEV_SEARCH ngoài selection.
5. Kiểm tra acoustic dependence bằng reference-only, audio-only, boundary-shuffle/zero controls và prior-only score.

Trước khi mở model, nên bắt đầu với một forward-backward implementation đã kiểm chứng trên fixture nhỏ, có phép thử tổng posterior alignment và event ID invariants. T3 trong MB05 không cho kết quả khoa học vì lỗi số học/thời gian thực thi; không tái dùng score giả định.

## Ưu tiên 2: đánh giá non-CTC temporal direct head

Direct MLP có target S/O signal và acoustic/reference controls, nhưng thấp hơn neural direct-event grouped baseline và không có I. So sánh mean-only, mean+std+delta và frame-level attention/GRU trên cùng H-002 cache, cùng folds/seeds. Trước tiên chỉ mở rộng S/O; insertion cần một biểu diễn gap inference-time và control riêng. Thử MixGoP-style phone-conditioned subclusters chỉ sau khi so với supervised head trên cùng folds.

## Ưu tiên 3: củng cố speaker inference

Seed101/202 bao phủ đủ 12 speakers nhưng không tạo cohort độc lập. Nếu dữ liệu bổ sung có thể được phê duyệt, giữ nguyên EVAL-V3.1 event contract và khóa pipeline trước external-speaker evaluation. Nếu không, bootstrap theo speaker/group và per-speaker uncertainty nên được báo như phân tích sensitivity, không diễn giải fold lặp là independent samples.

## Không tích hợp vội

Giữ E4 S/O, direct MLP, và các score lịch sử tách rời. Chỉ thử composition hay unified student sau khi insertion specialist mới thể hiện gain ổn định trên grouped OOF và DEV_CAL; không tune composition trên exposed DEV_SEARCH. `EXP43`, Phase 4 fusion, và các historical score mixes vẫn phải gắn nhãn composition thay vì neural checkpoint.
