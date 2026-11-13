# RQ1-UNIFIED-01 — Hướng dẫn tái lập

## Môi trường và nguồn

- Repo: `/home/khoa/Projects/New-Method-Research`; mã RQ1 ở `experiments/rq1_unified_01/`.
- CUDA: GTX 1650 4 GB, PyTorch `.venv-cuda` tại `/home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python`; tất cả job chạy tuần tự qua `coordination/phase3_gpu_job.py`.
- H-002 cache đầu vào tái sử dụng: `research/long_running_method_discovery/cycle_02/h002_ssl/normalized_v1/`; encoder `facebook/wav2vec2-base-960h`, `AutoFeatureExtractor.do_normalize=true`, hidden state 6, crop 17 frame. Đây là encoder ASR-finetuned, không phải SSL pretraining-only. CharSiU frame classifier tạo center alignment từ WAV + intended phones.
- TRAIN: 59,562 targets, 61,362 exact gaps; 50,356 C / 7,381 S / 1,825 O targets; 60,802 non-insertion / 560 insertion gaps. DEV_CAL: 10,043 targets / 10,343 gaps. DEV_SEARCH: 19,755 / 20,354, 599 utterances, 4 speakers; exposed exploratory.

## Huấn luyện đã chạy

Các lệnh U2, U3 và GAPW3 được bọc bằng GPU lease. Ví dụ U2 seed 101:

```bash
/home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python coordination/phase3_gpu_job.py \
  --job RQ1-U2-FLAT-TWOHEAD-S101-R2 -- \
  /home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python -u \
  experiments/rq1_unified_01/agent_a/train_twohead.py --seed 101 --epochs 8 --target-head flat
```

U3 seed 101 dùng `--job RQ1-U3-HIERARCHICAL-TWOHEAD-S101` và thêm `--target-head hierarchical`. GAPW3 dùng flat head và `--gap-positive-weight-ratio 3`. Các so sánh lặp lại dùng seed 202 và 303, mỗi job có ID riêng; xem `experiments/rq1_unified_01/EXPERIMENT_REGISTRY.csv` cùng receipt để biết ID/seed/checkpoint chính xác. Không chạy lại đè các checkpoint hiện có.

Trong CUDA environment không cài scikit-learn; phần huấn luyện hoàn tất và ghi receipt/checkpoint nhưng adapter tích hợp cuối lệnh trả lỗi import. EVAL-V3.1 sau đó chạy CPU trong `.venv` của repo, ví dụ:

```bash
.venv/bin/python experiments/rq1_unified_01/agent_c/evaluate_twohead_v3.py \
  --prefix RQ1-U2-FLAT-TWOHEAD --seed 101
```

Các receipt EVAL-V3.1 đã lưu bên cạnh mỗi arm trong `agent_a/` và replay độc lập seed 101 ở `agent_c/`. Thresholds fit riêng trên DEV_CAL; không dùng DEV_SEARCH để chọn mô hình hoặc ngưỡng.

## Suy luận WAV + tham chiếu

Tạo `intended_phones.json` là JSON array ARPAbet, rồi chạy:

```bash
/home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python coordination/phase3_gpu_job.py \
  --job RQ1-U2-WAV-INFERENCE-SMOKE -- \
  /home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python \
  experiments/rq1_unified_01/agent_a/infer_twohead.py \
  --audio path/to/audio.wav --phones-json path/to/intended_phones.json \
  --checkpoint RQ1-U2-FLAT-TWOHEAD-S101.pt --output predictions.json
```

Đã chạy trên `YBAA_arctic_a0003.wav`: 39 target, 40 gap; 1.612 s; peak VRAM 438.76 MiB; target/gap simplex sai số cực đại `1.2e-7`. WAV SHA-256 `45a0246c2aeee9f9348f5b8690f17dfc7f627ce860a772b74331ef23781f9813`, checkpoint SHA-256 `1978633b08034904d566273cb29f3f8305c604ed86e3be1145e1bc70d6375c92`, output SHA-256 `550ff1752907b7a8ec56b959d86397581515247bfeb6c6b4b6984ed8a43dc80a`. Inference nạp một checkpoint neural RQ1 và hai snapshot frontend local (CharSiU frame classifier + frozen Wav2Vec2); không nạp EXP24/36/43 hay Gold-Only expert checkpoints. Nhánh N=0 có trong code, nhưng chưa runtime-verified; gap duy nhất dùng pooled whole-utterance frame state và là trường hợp ngoài phân bố huấn luyện.

## Checkpoints và receipts chính

- U2 seed 101: `experiments/rq1_unified_01/agent_a/RQ1-U2-FLAT-TWOHEAD-S101.pt`; SHA-256 `1978633b08034904d566273cb29f3f8305c604ed86e3be1145e1bc70d6375c92`; 653,893 parameters; 24.88 s; 41.33 MiB peak.
- U3 seed 101: `.../RQ1-U3-HIERARCHICAL-TWOHEAD-S101.pt`; SHA-256 `c3bd772121c854cfb8c0118422a5b2a658ab5e6931a9df18f7b4b955ecd15538`; 653,764 parameters; 24.62 s; 41.33 MiB.
- GAPW3 seed 101: receipt `.../RQ1-U2-FLAT-TWOHEAD-GAPW3-S101_receipt.json`; SHA-256 `d408cae05f85c43c3b21bffe0082d405e5219fa234bd2210f84e12ba9c963206`; 653,893 parameters; 23.56 s; 41.33 MiB.
- Mỗi arm receipt bao gồm checkpoint/prediction hashes, input hashes, event counts, model params, thời gian, VRAM, encoder và hạn chế. Hai CSV leaderboards là bảng tóm tắt; per-class confusion/AP và policy đầy đủ nằm trong EVAL-V3.1 JSON.

## Class-wise precision-recall curves

Exact threshold/precision/recall points for S, O and exact-gap I on DEV_SEARCH for the seed101 U2, U3 and GAPW3 arms are exported by `experiments/rq1_unified_01/agent_c/export_pr_curves.py` to the compressed artifact `reports/RQ1_UNIFIED_CLASSWISE_PR_CURVES.csv.gz`. The AP field is computed from the same continuous scores and is a consistency reference; all split results remain exploratory.
