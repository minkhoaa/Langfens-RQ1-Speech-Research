# M2 — Training local realized-phone recognizer, 2026-11-13

Đã thực hiện 8 run (4 fold × seeds 101/202), 32 epoch/run. Training/evaluation hoàn tất; cổng chất lượng M2 chưa được chứng minh. Không promote hoặc mở sweep/M3 mới.

## Cấu hình và dữ liệu

Người dùng yêu cầu training local từ nay; đã cập nhật AGENTS.md, không dùng Colab/Kaggle. Nhánh `feat/m2-local-realized-recognizer`. Legacy Fold-1 contract tách khỏi protocol milestone.
Frozen W2V2-base-960h đúng revision M1, cache H6 đã hash/audit; chỉ học IF-style MLP CTC. Đây là thử lịch học dài hơn, không phải kiến trúc mới hoặc encoder fine-tuning. Supervision gồm 1.243 complete realized sequences thật; giữ speaker split 6 fit / 3 cal / 3 held trên TRAIN. Không mở official test/dev_replication, không dùng gold boundary làm inference input.
AdamW lr 0,001, weight decay 0,0001, batch 16, clip norm 5, fp32, cosine tới 0,0001 trong 32 epoch. Chọn checkpoint bằng PER cal ở epoch 4,8,…,32, tie chọn sớm nhất; không dùng held. Epoch-8 control thuộc cùng lịch cosine; M1 MLP dùng CPU constant-LR, nên không gán chênh lệch nhỏ cho riêng số epoch.

## Kết quả

Mỗi seed: 1.800 held câu, 59.562 target/61.362 gap. PER trên 1.243 strict câu/39.477 phones. FWR là held thực tế tại policy cal budget 0,20.

| Arm | Seed | PER | Macro-3 AP | Macro-3 F1 | FWR |
|---|---:|---:|---:|---:|---:|
| M2 epoch8 | 101 | 26.94% | 0.12461 | 0.01795 | 0.19833 |
| M2 epoch8 | 202 | 27.02% | 0.12541 | 0.01584 | 0.20222 |
| M2 selected | 101 | 26.36% | 0.12829 | 0.01892 | 0.19000 |
| M2 selected | 202 | 26.83% | 0.12524 | 0.01762 | 0.20111 |
| M1 frozen MLP | 101 | 27.03% | 0.12460 | 0.02311 | 0.21333 |
| M1 frozen MLP | 202 | 27.14% | 0.12448 | 0.01885 | 0.20222 |
| M1 end-to-end | 101 | 38.90% | 0.09009 | 0.01061 | 0.18556 |
| M1 end-to-end | 202 | 46.01% | 0.07258 | 0.00792 | 0.20778 |
| M1 SM13 full-fit | 101 | — | 0.24687 | 0.12495 | 0.21889 |
| M1 SM13 full-fit | 202 | — | 0.25071 | 0.12987 | 0.23500 |

AP tăng +0,00369 / +0,00076 so M1 MLP, dưới ngưỡng +0,02 ở cả hai seed. Seed 202 giảm AP O/I; selected AP thấp hơn epoch-8 control dù PER tốt hơn. Nhánh AP gate không đạt. Nhánh precision/recall chưa được đánh giá; không claim mọi nhánh đã thất bại hoặc đã vượt gate.

## S/O/I ở policy cal B=0,20

| Seed | Lớp | Support | TP | FP | FN | Precision | Recall | AP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 101 | SUBSTITUTION | 7381 | 78 | 179 | 7303 | 30.35% | 1.06% | 0.23649 |
| 101 | OMISSION | 1825 | 23 | 96 | 1802 | 19.33% | 1.26% | 0.11661 |
| 101 | INSERTION | 560 | 4 | 67 | 556 | 5.63% | 0.71% | 0.03178 |
| 202 | SUBSTITUTION | 7381 | 90 | 197 | 7291 | 31.36% | 1.22% | 0.23566 |
| 202 | OMISSION | 1825 | 13 | 106 | 1812 | 10.92% | 0.71% | 0.10541 |
| 202 | INSERTION | 560 | 5 | 59 | 555 | 7.81% | 0.89% | 0.03465 |

## Bằng chứng âm học

Greedy recognizer chỉ nhận feature từ audio. Retention dưới đây căn chỉnh edit xác định với complete realized gold; âm lặp có thể làm đổi quy thuộc token. Không đồng nhất retention với exact-gap diagnosis.

| Seed | S phone nhận đúng | I phone nhận đúng | S nguyên âm đúng | S âm tắc đúng |
|---|---:|---:|---:|---:|
| 101 | 2092/4480 | 164/405 | 645/1961 | 484/861 |
| 202 | 2170/4480 | 167/405 | 702/1961 | 497/861 |

Có bằng chứng nhận ra một phần lỗi thật, nhưng retention nguyên âm chỉ 33–36%, insertion 40–41%; PER giảm ít và recall diagnosis chỉ khoảng 0,7–1,3%. Chưa đủ bằng chứng recognizer vượt cổng chất lượng M2. Strict subset loại missing/unknown/deviation; 837 source S cùng base phone trong M0 không tự biến thành realized gold chắc chắn. Không ngoại suy retention sang toàn bộ 7.381 substitution.
DIAGNOSTICS.json chứa per-speaker PER S/D/I/support, từng phone (support/correct/substituted/deleted/recall), confusion, spurious insertions, phone-family S/O, edge/multiple-gap và lệch vị trí. Không bỏ nhóm recall thấp. Không có human gold/speaker mới, three-seed confirmation hoặc claim significance; fold/seed không độc lập.

## Kiểm chứng và chi phí

- 20 tests pass: optimizer/scheduler/RNG restart, CTC padding/finite backward và contract/M1 completion regressions; hai Swig deprecation warnings.
- 96 policy checks replay khớp (16 export run × 3 budget × 2 miền); EVAL-V3.1 counts/classes khớp; PER đối chiếu edit-distance độc lập, full held exactly-once coverage và speaker exclusion.
- 1724 historical hash khớp. Live GOAL đã cập nhật; verifier m2_report resolve seal M1 qua GOAL_before_M2.md, không sửa manifest lịch sử. close_m1 trực tiếp vẫn kiểm hash GOAL cũ nên sẽ báo khác khi dùng live GOAL mới.
- Smoke 20 updates chạy đủ train/predict/calibration/evaluate/save/reload. WAV HKK_arctic_a0003: 38 target/39 gap, decode bằng cache. CPU/GPU backend drift có thể có ở câu khác; không claim corpus-wide bitwise parity.
- last.pt lưu atomic mỗi epoch: model/optimizer/scheduler/scaler-disabled/RNG/metadata; best.pt theo cal, epoch8.pt giữ riêng. Checkpoint reload logits bằng nhau, artifacts được hash. Chưa thử kill/restart toàn campaign cho mọi crash window. Không đưa weights/dataset vào Git.
- Tổng 8 run gồm validation/export: 217.15 giây; peak PyTorch allocated 101.18 MiB (khác tổng driver VRAM). Budget mới 3.600 giây; không sweep. Cache được tái sử dụng, chi phí extraction lịch sử đã báo ở M1. environment.json/train_timing.json ghi cấu hình và wall time.
- GPU lease đã nhả; không còn training job. Chưa commit/push/merge.

## Chạy lại

```sh
.venv/bin/python -m research.realized_phone_pipeline.m2_local --mode smoke
.venv/bin/python -m research.realized_phone_pipeline.m2_local --mode train
# Đã có artifacts thì phải dùng --resume, không overwrite:
.venv/bin/python -m research.realized_phone_pipeline.m2_local --mode train --resume
.venv/bin/python -m research.realized_phone_pipeline.m2_report
.venv/bin/python -m research.realized_phone_pipeline.m2_infer --help
```

Artifact: PREREGISTRATION.md, completion.json, DIAGNOSTICS.json, selected/SUMMARY.json, epoch8/SUMMARY.json, seed*/{last,best,epoch8}.pt, waveform_inference_smoke.json, environment.json, verification.log.

Đã thực hiện training M2 local theo yêu cầu. Giữ cổng chất lượng M2 mở; sửa dữ liệu/recognizer trước bước M3 lớn, không tự mở sweep mới.
