# EVAL-V3 — Tái lập không huấn luyện

Chạy tại repository root, dùng dependency local `.venv`. Không chạy legacy training/reproduction commands. Các output mới phải ghi thư mục mới nếu muốn giữ bản phát hành; initial source freeze và WAV smoke cố ý từ chối ghi đè receipt.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHONPATH=. .venv/bin/python -m pytest research/evaluation_v3 tools/test_eval_v3_safety.py tests/test_phase2_artifacts.py tests/test_upr02_engine_and_evaluator.py research/student/test_so_diagnosis.py evaluation/tests/test_evaluator.py -q
PYTHONPATH=. .venv/bin/python research/evaluation_v3/migration.py --root "$PWD" --output experiments/evaluation_v3/reproduction
PYTHONPATH=. .venv/bin/python research/evaluation_v3/independent_audit.py --root "$PWD" --output experiments/evaluation_v3/reproduction/INDEPENDENT_AUDIT.json
PYTHONPATH=. .venv/bin/python tools/eval_v3_robustness.py --root "$PWD" --output experiments/evaluation_v3/reproduction
PYTHONPATH=. .venv/bin/python tools/eval_v3_policy_statistics.py --output experiments/evaluation_v3/reproduction
# Tùy chọn một acoustic forward CUDA thực; không cần cho metric replay:
PYTHONPATH=. .venv/bin/python tools/eval_v3_waveform_smoke.py --output experiments/evaluation_v3/waveform-reproduction
```

`tools/freeze_eval_v3.py` chỉ chạy lần đầu, không dùng để thay freeze sau khi có thay đổi. Initial1499 input hashes, source revision/session receipt và read-only archive nằm trong `experiments/evaluation_v3`. Waveform receipt bổ sung hash WAV/CMUDict/local acoustic weights/tokenizer. `OUTPUT_MANIFEST_SHA256.json` hash code/report/output mới; tránh circular self-hash. Lượt tái lập riêng trong `/tmp/langfens-eval-v3-reproduction-03e4968` đã PASS: primary metrics khớp và PR/risk/cohort/arithmetic/delta CSV byte-identical; source reports và initial freeze không đổi. Receipt: `REPRODUCTION_VERIFICATION.json`. Hai CSV lớn được version lossless dưới `.csv.gz`; giải nén bằng `gzip -dk FILE.csv.gz` nếu bản CSV chưa tồn tại. Không copy protected data hoặc khẳng định cached latency là waveform latency.
