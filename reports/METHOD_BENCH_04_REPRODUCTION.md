# METHOD-BENCH-04 — Hướng dẫn tái lập

## Trạng thái dữ liệu và compute

Campaign dùng các split L2-ARCTIC đã được duyệt và EVAL-V3.1. Posterior Wav2Vec2 CTC, phone inventory, blank ID 42, frame lengths/offsets và clock 20 ms được tái sử dụng từ METHOD-BENCH-03; không recognizer nào được huấn luyện lại. E4 likelihood scoring chạy CPU từ posterior lưu sẵn. GTX 1650 khả dụng nhưng không dùng cho E4. Agent B jitter chạy CUDA; B residualization và Agent C anomaly chạy CPU.

DEV_SEARCH đã bị exposed trong chiến dịch trước. Chỉ dùng các số SEARCH như kết quả thăm dò sau khi giữ nguyên cấu hình từ CAL; không dùng để chọn aggregator, K hay ngưỡng. Không truy cập protected holdout.

## A. E4 và matched CTC controls

Thư mục mã: `experiments/rq1_method_bench_04/agent_a/`.

- `ctc_forward.py`: forward CTC Python log-space, exhaustive sequence fixture.
- `ctc_batch.cpp`: C++ batch kernel cho nhiều sequence likelihood.
- `validate_ctc_kernel.py`: kiểm tra kernel độc lập.
- `e4_event_ctc.py`: E4 score, PR/AP/P@K, prior và length controls.
- `matched_baselines.py`: E1/E3 cùng recognizer và event population.
- Output JSON/JSONL: mỗi `s{seed}_{split}_e4.json` cùng `.scores.jsonl`; matched E1/E3 có `*_matched_baselines.json` cùng score ledgers.

Các lệnh đã dùng (thư mục gốc repository):

```bash
.venv/bin/python experiments/rq1_method_bench_04/agent_a/validate_ctc_kernel.py
.venv/bin/python experiments/rq1_method_bench_04/agent_a/e4_event_ctc.py --prefix w2v2_ --split train_speaker_holdout --output experiments/rq1_method_bench_04/agent_a/s101_train_holdout_e4.json
.venv/bin/python experiments/rq1_method_bench_04/agent_a/e4_event_ctc.py --prefix w2v2_ --split dev_cal --output experiments/rq1_method_bench_04/agent_a/s101_dev_cal_e4.json
.venv/bin/python experiments/rq1_method_bench_04/agent_a/e4_event_ctc.py --prefix w2v2_ --split dev_search --output experiments/rq1_method_bench_04/agent_a/s101_dev_search_e4.json
.venv/bin/python experiments/rq1_method_bench_04/agent_a/e4_event_ctc.py --prefix w2v2_s202_ --split train_speaker_holdout --output experiments/rq1_method_bench_04/agent_a/s202_train_holdout_e4.json
.venv/bin/python experiments/rq1_method_bench_04/agent_a/e4_event_ctc.py --prefix w2v2_s202_ --split dev_cal --output experiments/rq1_method_bench_04/agent_a/s202_dev_cal_e4.json
.venv/bin/python experiments/rq1_method_bench_04/agent_a/e4_event_ctc.py --prefix w2v2_s303_ --split train_speaker_holdout --output experiments/rq1_method_bench_04/agent_a/s303_train_holdout_e4.json
.venv/bin/python experiments/rq1_method_bench_04/agent_a/e4_event_ctc.py --prefix w2v2_s303_ --split dev_cal --output experiments/rq1_method_bench_04/agent_a/s303_dev_cal_e4.json
```

Max-alternative ablation thêm `--agg max`. E1/E3 matched commands dùng `.venv/bin/python experiments/rq1_method_bench_04/agent_a/matched_baselines.py --prefix PREFIX --split SPLIT`; đã chạy với seed 101/202/303 trên TRAIN speaker holdout, seed 101 trên CAL và SEARCH. Script này ghi output mặc định vào namespace Agent A; nếu cần rerun, đổi thư mục output trong bản sao script để giữ các receipt hiện tại.

Nguồn input: `experiments/rq1_method_bench_03/agent_a/w2v2_{,s202_,s303_}{train_speaker_holdout,dev_cal,dev_search}_posteriors.npz` và manifest `data/derived/l2arctic/manifests/*.jsonl`. Checkpoint/hash, số frame, clock và tokenizer nằm trong receipt METHOD-BENCH-03; hash posterior được ghi lại trong từng JSON E4.

## B. Speaker-generalizable event learning

- Feature jitter: `experiments/rq1_method_bench_04/agent_b/train_aug.py`.
- Utterance-centered residualization: `experiments/rq1_method_bench_04/agent_b/train_utterance_centered_oof.py`.
- Các receipt: `speaker_grouped_oof_aug_receipt.json` và `utterance_centered_s101_receipt.json`.

```bash
.venv/bin/python experiments/rq1_method_bench_04/agent_b/train_utterance_centered_oof.py
```

Feature jitter đã có paired PLAIN/AUG outputs; receipt ghi CUDA, 20.44 giây, seed/folds/epochs, prediction hash và input hashes. Residualization chạy CPU trong 20.61 giây. Cả hai dùng TRAIN-only fit/supervision, 4 grouped speaker folds và 5 epochs. Chỉ jitter có số liệu S/O/I; residualization script/changelog ở receipt. Không có PER vì hai phương pháp là direct event-head variants, không recognizer.

## C. Non-CTC prototype anomaly

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u experiments/rq1_method_bench_04/agent_c/normal_phone_anomaly.py
```

Script fit normal phone prototypes trên TRAIN-correct events theo fold; held-out speaker labels chỉ dùng để đánh giá. Runtime CPU 3.31 giây. Kết quả, fold supports, hashes và limitations nằm trong `experiments/rq1_method_bench_04/agent_c/normal_phone_anomaly_receipt.json`. Mô hình hỗ trợ target S/O, không phát sinh I.

## Báo cáo và giới hạn tái lập

Chi tiết kết quả xem `METHOD_BENCH_04_A_EVENT_CTC.md`, `METHOD_BENCH_04_B_SPEAKER_LEARNING.md`, `METHOD_BENCH_04_C_EXTERNAL_ALTERNATIVES.md`, leaderboard và ablation matrix. Peak memory không được instrument trong B/C và một số receipt A; không suy diễn số đo bị thiếu. Các lệnh trên mô tả lần chạy đã thực hiện; đừng ghi đè các JSON/score ledgers hiện có khi tái lập.
