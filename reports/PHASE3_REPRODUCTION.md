# Phase 3 — Tái lập local CUDA

Root duy nhất: `/home/khoa/Projects/New-Method-Research`. Cần `.venv`, approved TRAIN/DEV_CAL/DEV_SEARCH cache, frozen teacher checkpoints và hai HuggingFace snapshots local ghi trong inference receipt. Không tải mạng hoặc truy cập sealed/replication holdout. Các dependency/data có sẵn không được tự thêm vào commit Phase3.

## 1. Kiểm tra artifact hiện tại — không train lại

```bash
cd /home/khoa/Projects/New-Method-Research
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi
PYTHONPATH=research/student .venv/bin/python coordination/phase3_gpu_job.py \
  --job phase3-tests -- .venv/bin/python -m pytest \
  research/student/test_student.py \
  experiments/student_evaluation/test_evaluate_student_predictions.py -q

# Chỉ kiểm tra hash/metadata; torch.load CPU ở đây không phải neural training.
PYTHONPATH=experiments/student_evaluation .venv/bin/python - <<'PY'
from run_corrected_v2_evaluation import preflight
failures, records = preflight()
assert not failures, failures
print('Verified artifact receipts:', len(records))
PY
```

Supervisor đã chạy 26 tests thành công. Canonical raw predictions/checkpoints là local artifacts; `.pt/.npz` bị Git ignore, nhưng có SHA-256 trong receipts và `coordination/PHASE3_ARTIFACT_MANIFEST.json`.

## 2. Đánh giá lại mà không đè output đã công bố

```bash
OUT="experiments/student_evaluation/reproduction-$(date +%Y%m%dT%H%M%S)"
mkdir -p "$OUT"
.venv/bin/python experiments/student_evaluation/evaluate_student_predictions.py \
  --dev-cal experiments/student_training/ARM_seed101_dev_cal.npz \
  --dev-search experiments/student_training/ARM_seed101_dev_search.npz \
  --dev-cal-manifest data/derived/l2arctic/manifests/dev_cal.jsonl \
  --dev-search-manifest data/derived/l2arctic/manifests/dev_search.jsonl \
  --output "$OUT/GOLD_seed101.json"
.venv/bin/python experiments/student_evaluation/evaluate_student_predictions.py \
  --dev-cal experiments/student_training/STU-05-KD-ALL_seed101_dev_cal.npz \
  --dev-search experiments/student_training/STU-05-KD-ALL_seed101_dev_search.npz \
  --dev-cal-manifest data/derived/l2arctic/manifests/dev_cal.jsonl \
  --dev-search-manifest data/derived/l2arctic/manifests/dev_search.jsonl \
  --output "$OUT/KD_seed101.json"
```

Evaluator chỉ làm metrics/statistics trên CPU. Neural forward/backward đã hoàn thành trên CUDA. Protocol cuối là `student-eval-v2-upper-tail`, không dùng số liệu từ các thư mục superseded. Với Gold seed202/303, tên NPZ kế thừa là `ARM_seed202_seed202_*` / `ARM_seed303_seed303_*`.

Full 14-run commands, candidate counts, source hashes: `experiments/student_evaluation/PHASE3_EVALUATOR_PROVENANCE.json`. Runner `run_corrected_v2_evaluation.py` cố ý từ chối đè `individual/*.json` đang có. Không xóa evidence để chạy lại; dùng output mới như ví dụ trên. Các script summary chỉ dùng current `individual/`, không glob archive.

## 3. Train lại OOF + matched arms (ghi đè canonical local artifacts)

**Chỉ chạy phần này khi thực sự muốn retrain.** Trước đó snapshot các thư mục local artifact; các receipts/hash công bố cũ không còn mô tả checkpoint mới sau overwrite. Source checkpoints frozen không bị sửa. Không cam kết bitwise-identical SHA giữa hai lần save/NPZ hoặc giữa các mức concurrency.

```bash
cd /home/khoa/Projects/New-Method-Research
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
P="$PWD/.venv/bin/python"
LEASE="$PWD/coordination/phase3_gpu_job.py"
# Score TRAIN = speaker-held-out OOF; DEV_CAL/DEV_SEARCH = frozen full-TRAIN teachers.
"$P" "$LEASE" --job reproduce-oof -- "$P" research/student/kd/run_teacher_oof.py

for seed in 101 202 303; do
  "$P" "$LEASE" --job "reproduce-gold-$seed" -- "$P" \
    research/student/train_light_student.py --arm STU-01-GOLD-ONLY --epochs 12 --seed "$seed"
  "$P" "$LEASE" --job "reproduce-kd-$seed" -- "$P" \
    research/student/train_kd_student.py --arm STU-05-KD-ALL --epochs 12 --seed "$seed" --teachers S,O,IF1
done
```

Trainer Gold phát checkpoint/predictions và stdout JSON; receipt lịch sử có thêm metadata từ reconciliation. Sau retrain, dùng checkpoint/predictions mới để đánh giá trực tiếp, không sửa tay receipt cũ để giả khớp. Teacher/feature inputs, TRAIN weights và seed phải giữ giống nhau giữa Gold/KD.

## 4. Các ablation seed101 đã thực thi

Dùng cùng wrapper và 12 epochs. Ví dụ commands tương ứng dưới đây; mỗi command neural phải được bọc bởi `"$P" "$LEASE" --job NAME -- "$P"`:

```bash
"$P" "$LEASE" --job light -- "$P" research/student/train_light_student.py --arm STU-00-LIGHT-GOLD --hidden 32 --epochs 12 --seed 101
"$P" "$LEASE" --job kd-s -- "$P" research/student/train_kd_student.py --arm STU-02-KD-S-ONLY --teachers S --epochs 12 --seed 101
"$P" "$LEASE" --job kd-o -- "$P" research/student/train_kd_student.py --arm STU-03-KD-O-ONLY --teachers O --epochs 12 --seed 101
"$P" "$LEASE" --job kd-if1 -- "$P" research/student/train_kd_student.py --arm STU-04-KD-I-F1 --teachers IF1 --epochs 12 --seed 101
"$P" "$LEASE" --job kd-gate -- "$P" research/student/train_kd_student.py --arm STU-06-CONF-GATED --teachers S,O,IF1 --gated --epochs 12 --seed 101
"$P" "$LEASE" --job kd-iprec -- "$P" research/student/train_kd_student.py --arm STU-07-KD-I-ALTERNATE --teachers IPREC --epochs 12 --seed 101
"$P" "$LEASE" --job kd-idual -- "$P" research/student/train_kd_student.py --arm STU-08-KD-I-DUAL --teachers IF1,IPREC --epochs 12 --seed 101
"$P" "$LEASE" --job kd-cal -- "$P" research/student/train_kd_student.py --arm STU-09-KD-ALL-CALIBRATED --teachers S,O,IF1 --calibrated --epochs 12 --seed 101
```

Lệnh trên chạy tuần tự an toàn. Đợt thực tế đã thử ba wrapper jobs chồng nhau; xem ledger. Không bỏ qua global lease để ép concurrency, không claim ba-way kernel overlap khi thiếu trace. CUDA unavailable hoặc lease/VRAM không an toàn phải dừng, không CPU fallback.

## 5. Inference audio + intended phones — chỉ local CUDA

Chuẩn bị `$AUDIO` và `$PHONES` từ caller. `$PHONES` là JSON array ARPAbet không stress, không phải realized/gold label. Ví dụ interface:

```bash
P="$PWD/.venv/bin/python"; LEASE="$PWD/coordination/phase3_gpu_job.py"
AUDIO=/absolute/path/to/approved/local/learner.wav
PHONES='["DH","IH","S"]'
"$P" "$LEASE" --job audio-inference -- "$P" research/student/infer_student.py \
  --audio "$AUDIO" --phones-json "$PHONES" \
  --checkpoint models/student/STU-01-GOLD-ONLY_seed101.pt \
  --output-json experiments/student_training/my_inference.json
```

Không dùng placeholder path như lệnh chạy thật: thay bằng audio approved local. Sample đã chạy nằm trong `experiments/student_training/inference_sample.json` và `inference_empty_reference.json`, logs trong `logs_corrected_kd/`. Caller có thể truyền `--phones-json '[]'`: output giữ 0 target và 1 insertion gap. Chưa tích hợp orthographic text→phone. API `predict(...)` trả cả distributions, CLI `--output-json` ghi indexed probabilities; stdout có shapes, latency/device.

## 6. Công bố và đọc kết quả

`coordination/publish_phase3.py` chỉ tổng hợp output hiện tại, không train. Không chạy lại sau retrain khi receipts cũ chưa được reconciliation hợp lệ.

- `reports/PHASE3_FINAL_SCIENTIFIC_REPORT.md`: kết luận và giới hạn tiếng Việt.
- `reports/PHASE3_EXPERIMENT_RESULTS.csv`: 98 điểm với provenance/commands.
- `reports/PHASE3_GOLD_VS_KD_PAIRED.md`: matched three-seed table.
- `reports/PHASE3_FWR_RECALL_PARETO.csv`: student-only realized-risk frontier.
- `reports/PHASE3_TEACHER_OOF_PROVENANCE.md`: nguồn teacher đúng.
- `experiments/student_evaluation/PHASE3_FROZEN_CONTROL_METRICS.csv`: controls threshold frozen, không giả retune.

Bản hiện tại không chứng minh KD superiority. Insertion recall tăng trong một số điểm đi kèm omission regression. DEV_SEARCH exploratory, chỉ bốn speaker, không được gọi là blind test.
