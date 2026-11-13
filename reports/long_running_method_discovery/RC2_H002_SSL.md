# RC2 H-002 — Biểu diễn khung Wav2Vec2 và reference-conditioned attention

**Kết luận: `SUPPORTED_EXPLORATORY` cho một arm nhạy cảm dùng encoder ASR-finetuned; giả thuyết về pretraining-only SSL vẫn `UNRESOLVED`.** Một seed sàng lọc cho thấy AP SUB/O tăng nhẹ so với Gold-Only, nhưng tăng F1 phụ thuộc operating point, I được giữ nguyên từ Gold-Only, và SEARCH đã exposed. Không dùng các số này làm xác nhận độc lập hoặc tuyên bố thắng EXP24/29/36/43.

## Phạm vi và model provenance

Primary H-002 yêu cầu frozen intermediate-layer SSL features. Cache cục bộ có `facebook/wav2vec2-base-960h`, nhưng checkpoint đã ASR-finetune; không tìm thấy checkpoint pretraining-only trong vùng cache kiểm tra. Vì vậy thử nghiệm dưới đây được ghi chính xác là **ASR-finetuned Wav2Vec2 sensitivity arm**, không phải phép thử thuần SSL. Model repository: `facebook/wav2vec2-base-960h`, pinned local snapshot `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`; weights SHA-256 `8aa76ab2243c81747a1f832954586bc566090c83a0ac167df6f31f0fa917d74a`. Encoder frozen; chỉ attention target head được huấn luyện.

Mỗi target dùng 17 hidden frames (±8 quanh tâm đường Viterbi suy ra từ logits của frame classifier CharSiU và chuỗi intended phones) và 39D reference one-hot. Tâm alignment được suy từ emission âm học và reference, không dùng `xmin/xmax`, realized phone hoặc nhãn lỗi. Tín hiệu được chuẩn hóa bằng `AutoFeatureExtractor` của checkpoint, `do_normalize=true`, theo `preprocessor_config.json`. Unknown spelling của reference dùng SIL cho align, và giữ zero one-hot giống feature pipeline hiện có.

Encoder có 12 transformer blocks, hidden size 768; dùng `hidden_states[6]`. Phần 17-frame attention head có 223,875 tham số, 8 epoch, seed 101. Gap head giữ nguyên điểm dự đoán của Gold-Only, bảo toàn đúng 20,354 gap N+1 trên SEARCH.

## CUDA execution và reproducibility

GPU GTX 1650 4GB, PyTorch `2.14.1+cu130`, CUDA `13.0`; mỗi neural job chạy qua `coordination/phase3_gpu_job.py` với `.venv-cuda`. Các lease đã trả mã 0:

| Job | Kết quả | Peak allocated VRAM | Thời gian |
|---|---:|---:|---:|
| `RC2-H002-ASR-FINETUNED-NORM-EXTRACT-S101` | 2,699 utt; 89,360 target frames pooled | 495.68 MiB | 187.96 s |
| `RC2-H002-ASR-FINETUNED-NORM-ATTN-TRAIN-S101` | 8 epoch, 59,562 targets | 32.45 MiB | 18.10 s training |
| `RC2-H002-ASR-FINETUNED-NORM-ACOUSTIC-CONTROLS-S101` | reference-only + slot shuffle | 18.42 MiB | 7.52 s |
| `RC2-H002-ASR-FINETUNED-NORM-ZEROREF-S101` | zero-reference perturbation | 18.42 MiB | 4.90 s |

Lệnh tái chạy:

```bash
.venv/bin/python coordination/phase3_gpu_job.py --job RC2-H002-ASR-FINETUNED-NORM-EXTRACT-S101 -- /home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python -u research/long_running_method_discovery/cycle_02/h002_ssl/run_h002_asr_sensitivity.py extract
.venv/bin/python coordination/phase3_gpu_job.py --job RC2-H002-ASR-FINETUNED-NORM-ATTN-TRAIN-S101 -- /home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python -u research/long_running_method_discovery/cycle_02/h002_ssl/run_h002_asr_sensitivity.py train
.venv/bin/python coordination/phase3_gpu_job.py --job RC2-H002-ASR-FINETUNED-NORM-ACOUSTIC-CONTROLS-S101 -- /home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python -u research/long_running_method_discovery/cycle_02/h002_ssl/normalized_v1/run_acoustic_controls.py
.venv/bin/python coordination/phase3_gpu_job.py --job RC2-H002-ASR-FINETUNED-NORM-ZEROREF-S101 -- /home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python -u research/long_running_method_discovery/cycle_02/h002_ssl/normalized_v1/run_acoustic_controls.py
.venv/bin/python research/long_running_method_discovery/cycle_02/h002_ssl/normalized_v1/evaluate_h002_evaluator_v3.py
```

## EVAL-V3.1 discrimination

DEV_SEARCH có 19,755 targets (17,778 CORRECT, 1,768 SUB, 209 OMIT) và 20,354 exact gaps (125 insertions). Toàn bộ event IDs, class denominators và target/gap topology khớp manifest. AP không đổi theo threshold; operating point được chọn chỉ trên DEV_CAL. Các F1 dưới đây dùng policy tối đa Macro-3 Error F1 dưới warning budget đã nêu trên DEV_CAL; FWR SEARCH là kết quả quan sát, không phải ràng buộc bắt buộc.

| Mô hình / inference control | AP S | AP O | AP I exact-gap | Macro-3 Error AP | Macro-3 Error F1 @ CAL 10% | Macro-4 F1 @ CAL 10% |
|---|---:|---:|---:|---:|---:|---:|
| Gold-Only CUDA130 seed 101 | 0.364602 | 0.159433 | 0.186715 | 0.236917 | 0.0343 | 0.2625 |
| ASR-finetuned W2V2 + reference attention | 0.377810 | 0.167294 | 0.186715 | 0.243940 | 0.0546 | 0.2778 |
| Reference-only perturbation (hidden states=0) | 0.293833 | 0.024503 | 0.186715 | 0.168350 | 0.0251 | 0.2556 |
| Slot-shuffled acoustic context | 0.239753 | 0.023376 | 0.186715 | 0.149948 | 0.0251 | 0.2556 |
| Zero-reference perturbation | 0.095798 | 0.011834 | 0.186715 | 0.098116 | 0.0251 | 0.2556 |

So với cùng-seed CUDA Gold-Only, full arm có ΔAP S `+0.0132083`, O `+0.0078609`, I `0`, Macro-3 Error AP `+0.0070231`. Gold-Only checkpoint SHA-256 là `52c6c26bb52c119915942da8a3aaf9fdbc109911b77a55df8cf813ca46f7dc4f`; baseline dùng seed 101 và 12 epoch Phase 3.

Ở policy DEV_CAL 10%, SEARCH class metrics của full model là:

| Lớp | Precision | Recall | F1 |
|---|---:|---:|---:|
| SUBSTITUTION | 0.4857 | 0.0096 | 0.0189 |
| OMISSION | 0.4000 | 0.0383 | 0.0699 |
| exact-gap INSERTION | 0.6250 | 0.0400 | 0.0752 |

Gold-Only tại cùng operating point đạt S `1.0000/0.0006/0.0011`, O `0.1667/0.0144/0.0264`, I `0.6250/0.0400/0.0752`. Macro-3 F1 tăng `+0.0203`, Macro-4 tăng `+0.0153`; SEARCH FWR lần lượt `0.0551` và `0.0301` false warnings/utterance theo EVAL-V3.1. Tại CAL budget 20%, Macro-3 F1 là `0.1253` so với `0.1049`, nhưng đây vẫn là một seed và exposed SEARCH. Tại CAL budget 40%, model đạt Macro-3 F1 `0.1745` so với `0.1519`, còn SEARCH FWR là `0.2938` so với `0.1953`.

Các class metrics của H2 tại mọi policy (P/R/F1 theo thứ tự):

| Budget chọn trên CAL | S P/R/F1 | O P/R/F1 | I P/R/F1 | Macro-3 F1 | Macro-4 F1 | SEARCH FWR |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.4857 / 0.0096 / 0.0189 | 0.4000 / 0.0383 / 0.0699 | N/E / 0 / 0 | 0.0296 | 0.2590 | 0.0501 |
| 0.10 | 0.4857 / 0.0096 / 0.0189 | 0.4000 / 0.0383 / 0.0699 | 0.6250 / 0.0400 / 0.0752 | 0.0546 | 0.2778 | 0.0551 |
| 0.20 | 0.4857 / 0.0096 / 0.0189 | 0.4000 / 0.0383 / 0.0699 | 0.3766 / 0.2320 / 0.2871 | 0.1253 | 0.3308 | 0.1302 |
| 0.40 | 0.4863 / 0.0503 / 0.0912 | 0.3585 / 0.0909 / 0.1450 | 0.3766 / 0.2320 / 0.2871 | 0.1745 | 0.3675 | 0.2938 |
| unrestricted | 0.3542 / 0.5990 / 0.4451 | 0.1457 / 0.4545 / 0.2207 | 0.2193 / 0.3280 / 0.2628 | 0.3095 | 0.4592 | 4.3973 |

`N/E` means no predicted positive event, so precision is not estimable; the report preserves this instead of converting it to zero.

Ở operating point unrestricted, AP vẫn như trên, nhưng Macro-3 F1 là `0.3095` so với Gold-Only `0.3102`, Macro-4 là `0.4592` so với `0.4619`; SEARCH FWR tăng `4.3973` so với `3.6912`. Vì vậy lựa chọn threshold ảnh hưởng quyết định phân loại; không có cải thiện toàn diện tại mọi điểm.

Confidence tại CAL-selected 10% policy: H2 SEARCH Brier/ECE lần lượt CORRECT `0.11610/0.15184`, SUB `0.08956/0.11250`, O `0.02267/0.03934`, I `0.01238/0.03146`; Gold-Only tương ứng `0.11947/0.16645`, `0.08935/0.11816`, `0.02611/0.04831`, `0.01238/0.03146`. Đây là độ tin cậy thô trên SEARCH, không được hiệu chuẩn hậu nghiệm.

## Acoustic-dependence diagnostics và speaker variation

Các perturbation dùng nguyên checkpoint, không refit threshold cho AP, và không phải ablation được huấn luyện lại. So với full model, AP của reference-only (hidden states=0) là S/O/I `0.293833/0.024503/0.186715`, giảm ΔS `−0.083977`, ΔO `−0.142791`; slot-shuffled acoustic là `0.239753/0.023376/0.186715`, giảm ΔS `−0.138057`, ΔO `−0.143918`; zero-reference là `0.095798/0.011834/0.186715`, giảm ΔS `−0.282012`, ΔO `−0.155460`. AP insertion không đổi vì mọi arm giữ nguyên Gold-Only gap scores. Đây là dấu hiệu rằng dự đoán phụ thuộc cả hidden acoustic context lẫn reference identity, nhưng các phép can thiệp tại inference nằm ngoài phân phối huấn luyện, không chứng minh quan hệ nhân quả hoặc ưu thế của pure SSL.

AP theo speaker trên SEARCH (descriptive, chỉ 4 speaker):

| Speaker | Gold-Only S/O/I AP | H2 S/O/I AP |
|---|---|---|
| NCC | 0.4787 / 0.1650 / 0.1859 | 0.4571 / 0.1018 / 0.1859 |
| RRBI | 0.3272 / 0.1913 / 0.0822 | 0.2866 / 0.2834 / 0.0822 |
| YBAA | 0.1499 / 0.0942 / 0.3636 | 0.2658 / 0.1004 / 0.3636 |
| YDCK | 0.5347 / 0.2343 / 0.2608 | 0.5439 / 0.2259 / 0.2608 |

Kết quả dị biệt theo speaker: S cải thiện ở YBAA/YDCK nhưng giảm ở NCC/RRBI; O tăng rõ ở RRBI, giảm ở NCC/YDCK. Bốn speaker không đủ để suy luận generalization thống kê.

## Lỗi đã gặp và cách xử lý

- R2 torchaudio loader cần TorchCodec không cài trong `.venv-cuda`; dùng SoundFile + SciPy resampling thay thế.
- R3 manifest có spelling `ER)`, `D_`, `Z_`, `Y_`; aligner dùng SIL fallback đúng như pipeline đã tồn tại, reference one-hot đọc trực tiếp cache và giữ all-zero với OOV.
- R4 raw waveform đầu tiên bỏ qua `do_normalize=true`; head đã chạy 8 epoch nhưng run bị loại hoàn toàn. Cache và checkpoint được giữ trong `research/.../h002_ssl/failures/attempt4_raw_input_contract/`; không dùng metrics. Lượt chuẩn hóa sau đó dùng `AutoFeatureExtractor` đúng config.
- Training normalized hoàn tất thành công. Không có artifact lịch sử nào bị ghi đè; baseline Gold-Only seed 101 có tên riêng RC2.

## Hashes và artifact index

Mọi hash bên dưới là SHA-256. Chi tiết đầy đủ, input per-split hashes, population counts, output metrics, speaker summaries và calibration nằm trong [EVAL-V3 receipt](../../research/long_running_method_discovery/cycle_02/h002_ssl/normalized_v1/EVAL_V3_RECEIPT.json), [extraction receipt](../../research/long_running_method_discovery/cycle_02/h002_ssl/normalized_v1/EXTRACTION_RECEIPT.json), [training receipt](../../research/long_running_method_discovery/cycle_02/h002_ssl/normalized_v1/TRAINING_RECEIPT.json), và [control receipt](../../research/long_running_method_discovery/cycle_02/h002_ssl/normalized_v1/ACOUSTIC_CONTROL_RECEIPT.json).

- Model weights `facebook/wav2vec2-base-960h`: `8aa76ab2243c81747a1f832954586bc566090c83a0ac167df6f31f0fa917d74a` (local ASR-finetuned checkpoint).
- Normalized TRAIN frame-context array: `eb8b23a22c2b497e5ac41a07c2ab20ab4f4baac1228605f9481a5e949489392d`.
- Normalized DEV_CAL frame-context array: `82fa09c5b7811b546608fab8d3938d54833c9fe018c5c16cd5e18bddbf5a7bd9`.
- Normalized DEV_SEARCH frame-context array: `bb39b12c6959c1ccd99a8f8a7e142533443d089bc26a72bf2c4b7a94705f684e`.
- Attention checkpoint `RC2-H002-ASR-FINETUNED-ATTN-NORM_seed101.pt`: `6739c8c9bdf00df5632693f2feebb5ef5907a499338af803ed6095a0aa120f50`.
- Gold-Only baseline checkpoint seed 101: `52c6c26bb52c119915942da8a3aaf9fdbc109911b77a55df8cf813ca46f7dc4f`.
- DEV_CAL prediction NPZ: `ced1fa895bbdde4e67fe88dd68728315afe3c139cabb08c16c0ec10c7435fbf5`.
- DEV_SEARCH prediction NPZ: `79dc2e48cde89c583355ca747f199aa5a59741938360aa36a80f3ff81393fe77`.
- EVAL-V3 receipt SHA-256: `785a0bbbe418f301acfc954a59841f8cc2c229603c1969b458df31ed236a3ceb`. Extraction receipt SHA-256: `56973127011b5689d0a0603b10ba8fcec7d5cb47f66fba715d861a7e81cd0508`; training receipt SHA-256: `bb5de96821c81eafb5e7a4af58ba16f8b5262fa836e404ca307a02420d271940`; acoustic-control receipt SHA-256: `3200a3ebb4e87326dc7abd991996ece7b208e506f2daf96379920cb7a88befd4`. `EVAL-V3.1 evaluator.py` SHA-256: `76e41f906e5d953ee55cee8156379644bdd96bf96efab69e991b87d7f857d4a2`.

## Quyết định tiếp theo

Giữ arm này như kết quả exploratory, không tích hợp vào Phase 4. Để xác nhận H-002 cần checkpoint pretraining-only thực sự, training lại seed lặp lại, reference-only network được huấn luyện matched, modality masking trong lúc training hoặc perturbation in-distribution, TRAIN speaker-grouped selection, và comparison độc lập không chọn winner trên SEARCH. Intended-reference independence vẫn chưa được kiểm chứng.
