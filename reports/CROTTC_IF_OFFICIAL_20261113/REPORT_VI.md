# Khả năng tái hiện đầy đủ CROTTC-IF — 2026-11-13

**Đã chạy thành công inference của toàn bộ model chính thức được phát hành. Chưa tái lập training đầy đủ hoặc benchmark F1 71,77%.**

## Kết quả đã kiểm chứng

| Hạng mục | Bằng chứng |
|---|---|
| Nguồn | `Haopeng/CROTTC-IF-l2-arctic`, revision `9149e5107b3332b9aa3f5573f6328a4e312006ad` |
| Model | WavLM Large và IF model nguyên bản phát hành, 327.457.277 tham số trong các module được nạp; không thay bằng frozen-H6/GRU compact |
| Integrity | 27 file khớp Git blob hoặc LFS SHA-256 upstream; hai checkpoint 73.894.282 và 1.262.009.130 byte |
| Loading | Hai recoverable `model` và `perceived_ssl` được official wrapper load với `strict=True`; label encoder 44 token được kiểm hash |
| Inference | Ba WAV thuộc FIT của historical TRAIN đã được stage/hash; chỉ truyền đường WAV vào wrapper, không truyền reference/gold labels/timestamps |
| Decoding | Beam 10, CTC decoding weight 0,99; cả ba câu có chuỗi phone đầu ra |
| Tài nguyên | GTX 1650 4 GB; hai CPU thread; 12,35 giây gồm load/verification/inference; peak CUDA allocated 2,47 GiB, reserved 2,58 GiB; peak process RSS 2,19 GiB |
| Thay đổi weights | 0 optimizer update, không train/fine-tune; bản phát hành nguyên vẹn sau lượt chạy |

Ba câu được chọn xác định trước theo request manifest, không dùng gold để chọn: `HQTV_arctic_a0005`, `HQTV_arctic_a0007`, `HQTV_arctic_a0011`. Đây chỉ là kiểm tra khả năng thực thi, không phải mẫu đánh giá chất lượng hoặc xác nhận speaker-disjoint. Raw hypotheses có trong `SMOKE_RECEIPT.json`; không suy ra chất lượng diagnosis hoặc feedback từ việc sinh được chuỗi phone.

Runtime cô lập: Torch/torchaudio 2.6.0+cu124, Transformers 4.57.0, SpeechBrain 1.0.3, huggingface-hub 0.36.0; wandb 0.21.4 được cài vì runtime import module này, logging bị tắt. Official requirements ghi Torch/torchaudio 2.7.1 và huggingface-hub 0.33.2; đây là khác biệt runtime đã ghi nhận. Không sửa code/YAML/checkpoint của bundle để chạy. Lượt cài wandb không pin ban đầu được dừng trong setup; lượt pin hoàn tất, không có training trong các lượt setup.

## Ba mức tái hiện cần phân biệt

1. **Tái hiện inference của release:** đã xác minh ở mức smoke trên ba câu với toàn bộ weights và runtime chính thức. Có thể dùng làm điểm xuất phát kỹ thuật để nghiên cứu tiếp.
2. **Tái hiện training của paper:** chưa thực hiện. Paper mô tả AM 300 epoch, LM/teacher 200 epoch, batch 32, chọn model bằng validation F1, trên một NVIDIA GH200. Có thêm giai đoạn huấn luyện CTC/Transformer ban đầu trước khi thay bằng AM CROTTC. Không được gọi một lượt compact ngắn hoặc chỉ train head là tái lập toàn bộ recipe.
3. **Tái lập benchmark 71,77%:** chưa thực hiện. Cần khớp preprocessing/nhãn, manifest train/validation/test, các giai đoạn training hoặc lineage checkpoint và evaluator/operating point. Paper dùng 2.429 train, 268 validation, 900 test; các tập FIT628/CAL450/HELD450 trước đây không phải split paper.

## Vì sao chưa chạy nguyên training trên GPU hiện tại

Static accounting trên chính checkpoint: chỉ các tensor weight/bias trong WavLM `model.encoder.*` đã có 302.330.048 phần tử. Với FP32 Adam thông thường, riêng weights + gradients + hai moments cần khoảng **4,51 GiB**, chưa gồm phần còn lại của model và activation; vượt tổng VRAM 4 GiB. Đây là ước lượng bảo thủ theo tensor, không phải một lần backward/OOM thực nghiệm.

Offload, mixed precision, accumulation hoặc CPU training cần profile và budget riêng; không kết luận mọi cách training đều bất khả thi, nhưng cũng không cam kết chúng tái lập nguyên recipe batch32 trong thời gian khóa luận. Inference vừa VRAM không chứng minh full training cũng vừa.

## Ranh giới benchmark và việc tiếp theo

- Official test/dev_rep tiếp tục niêm phong theo AGENTS.md; không stage/extract hoặc evaluate chúng trong lượt này.
- Training-speaker manifest của checkpoint public chưa được xác minh đầy đủ. Vì vậy model smoke này chưa được chứng nhận là baseline sạch cho grouped CV hiện có.
- Cần xác minh lineage, dữ liệu/preprocessing và protocol trước phép so sánh hiệu quả. Không dùng điểm công bố như điểm của checkpoint trên tập local.
- Benchmark detection F1 của paper khác Macro-F1 S/O/I exact-gap; chưa có đánh giá hướng dẫn sửa lỗi trong lượt này.
- Checklist nền thực nghiệm trong GOAL vẫn mở: smoke thành công không đồng nghĩa toàn bộ baseline/benchmark đã được tái lập.

## Artifact và lệnh chạy

Nguồn và bằng chứng local ở `experiments/crottc_if_official_20261113/`: `MODEL_API.json`, `COMPUTE_BUDGET.json`, `DOWNLOAD_RECEIPT.json`, `BUNDLE_INTEGRITY.json`, `SMOKE_RECEIPT.json`, `SMOKE.log`, `TRAINING_RESOURCE_ESTIMATE.json`, `RUNTIME_FREEZE.txt`. Weights `.ckpt` bị gitignore.

Entrypoint local: `research/crottc_if_official/smoke.py`, dùng trực tiếp `bundle/custom_interface.py`. Runner từ chối overwrite receipt; cần giữ receipt/failed provenance trước lượt tiếp theo.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled WANDB_DISABLED=true \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
timeout --signal=TERM --kill-after=10s 300s \
flock -n /home/khoa/Projects/Pronunciation-RD-Coordination/locks/gpu.lock \
.venv-crottc-official/bin/python -m research.crottc_if_official.smoke
```

Nguồn gốc: [Paper §6.1–6.3](https://arxiv.org/html/2604.22133v1), [official code](https://github.com/Secondtonumb/IF-MDD), [pinned release](https://huggingface.co/Haopeng/CROTTC-IF-l2-arctic/tree/9149e5107b3332b9aa3f5573f6328a4e312006ad).
