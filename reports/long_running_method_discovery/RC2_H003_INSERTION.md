# LANGFENS RC2 H-003 — mô hình hóa insertion tại exact gap

**Trạng thái:** đã chạy một đối chứng thống kê CPU có kiểm soát; `SUPPORTED` chưa đạt. Phép thử pooling tuyến tính cục bộ **không được dữ liệu DEV_SEARCH ủng hộ**. Đây là kết quả thăm dò trên DEV_SEARCH đã development-exposed, không phải xác nhận độc lập và không bác bỏ attention/neural pooling.

**Experiment ID:** `RC2-H003-CPU-LOGREG-001`  
**Agent:** R3 / `h003_insertion`  
**Model agent:** `gpt-6-luna`  
**Source revision:** `aa4e6ea2147180a4a820d77a71911f333fcb6721`

## Câu hỏi và thiết kế

Kiểm tra liệu một biểu diễn sáu chiều gom đặc trưng CTC theo thời gian quanh gap có tăng xếp hạng insertion so với sáu thống kê gap hiện hữu không. Hai arm dùng cùng LogisticRegression (`C=1`, `class_weight=balanced`, `lbfgs`, tối đa 500 vòng, `random_state=101`), cùng dữ liệu TRAIN, cùng số chiều đầu vào và cùng cách chuẩn hóa chỉ fit trên TRAIN. **Chỉ biểu diễn của insertion gap thay đổi.** Điểm target S/O/C giữ nguyên từ frozen `STU-01-GOLD-ONLY_seed101` trong cả hai arm.

- **Control — `GAP6_STATIC_CONTROL`:** sáu đặc trưng gap hiện có.
- **Intervention — `GAP6_LOCAL_TEMPORAL_INTERVENTION`:** sáu thống kê trên cửa sổ 11 frame quanh gap: trung bình/cực đại xác suất phone không-silence lớn nhất mỗi frame, chênh lệch giữa nửa sau và nửa đầu, xác suất silence trung bình, entropy trung bình, và tỷ lệ frame có xác suất không-silence vượt silence.

Tâm gap nội bộ dùng midpoint của blank span CTC nếu có; nếu không, dùng midpoint của các span phone dự đoán lân cận. Leading gap dùng onset phone dự đoán đầu tiên; trailing gap dùng blank span cuối hoặc offset phone dự đoán cuối. Reference rỗng có một gap và pooling trên toàn bộ utterance. Không dùng gold boundary, realized phone hoặc error label để tạo đặc trưng. Unknown reference phones dùng đúng fallback legacy SIL của waveform replay.

Ngưỡng từng arm được chọn riêng trên DEV_CAL từ lưới cố định 201 điểm `0.001..0.999`, tối đa F1 insertion, tie-break ngưỡng cao hơn; khóa lại trước DEV_SEARCH. Đây chỉ là threshold policy, không ảnh hưởng AP liên tục. Mọi metric so sánh dùng `UnifiedEvaluatorV3` / EVAL-V3.1.

## Kết quả

| Split / arm | Ngưỡng CAL | Exact-gap AP I | ROC-AUC I | Precision I | Recall I | F1 I | Macro-3 Error AP | Macro-3 Error F1 | Joint FWR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DEV_CAL — static control | 0.89421 | 0.08970 | 0.64455 | 0.15033 | 0.22549 | 0.18039 | 0.36553 | 0.39051 | 4.3500 |
| DEV_CAL — local temporal | 0.78942 | 0.02231 | 0.71282 | 0.03036 | 0.16667 | 0.05136 | 0.34307 | 0.34749 | 5.7267 |
| DEV_SEARCH — static control | 0.89421 | **0.04002** | 0.58593 | 0.11290 | 0.11200 | 0.11245 | **0.18802** | **0.25148** | 4.2855 |
| DEV_SEARCH — local temporal | 0.78942 | **0.01132** | 0.67798 | 0.01304 | 0.05600 | 0.02115 | **0.17845** | **0.22104** | 4.9866 |

DEV_SEARCH intervention minus control: **AP I −0.02870**, **F1 I −0.09130**, **Macro-3 Error AP −0.00957**, **Macro-3 Error F1 −0.03043**. At frozen CAL-selected thresholds, control has 14 TP / 110 FP / 111 FN / 20,119 TN over 20,354 gaps; intervention has 7 / 530 / 118 / 19,699. Both arms retain all 599 utterances, 19,755 target slots and **20,354 exact insertion gaps** (125 positives).

Independent numerical check: recomputing AP from saved score arrays with a separate grouped-tie precision/recall summation gave 0.040022890777949904 (static) and 0.011320187270796615 (local), matching EVAL-V3 to floating-point roundoff.

Per-speaker DEV_SEARCH AP I was lower for the local intervention in all four speakers:

| Speaker | Static control AP I | Local temporal AP I | Positive gaps |
|---|---:|---:|---:|
| NCC | 0.07574 | 0.01519 | 29 |
| RRBI | 0.03308 | 0.00998 | 44 |
| YBAA | 0.05244 | 0.01404 | 30 |
| YDCK | 0.05098 | 0.01684 | 22 |

Speaker rows are descriptive; four speakers do not support strong population-level inference. Joint FWR is reported as operating behavior, not used as a pass/fail criterion. The large FWR here also reflects the unchanged Gold-Only target head under this independently fitted gap probe; it does not change the paired insertion AP comparison.

## Interpretation

The six local trajectory statistics had higher ROC-AUC but substantially worse AP, precision and recall/F1 than the static gap control. With insertion prevalence only 125/20,354 on SEARCH, the AUC gain did not translate into better ordering of the rare positive tail. The CAL-selected operating point likewise produced many more false insertion alerts. This does **not** support this specific low-capacity temporal-pooling representation.

The result does not test reference-conditioned attention, learned transition features, energy/voicing trajectories, joint phone/gap decoding, or a matched CUDA neural head. Those remain unresolved. Because the experiment used one deterministic fit and only four exposed SEARCH speakers, classify it `NOT_SUPPORTED_EXPLORATORY`, not a final hypothesis rejection.

## Data and artifact provenance

The existing full authorized waveform replay contains **2,699** utterance archives with `ctc_logits` shaped frames × 42 vocabulary entries. Its receipt reports exact feature parity for 1,800 TRAIN, 300 DEV_CAL and 599 DEV_SEARCH utterances. The frozen cache avoids another CTC forward pass. Cache inputs preserve annotation-derived canonical intended phones; their independent-reference lineage remains unverified.

Selected SHA-256 values:

- Full waveform replay receipt: `6cb0de822e6298651af6e0df572f492c8eb9e3ead79876cd4806898a6b976f3a`
- Sorted manifest hash of 2,699 per-utterance waveform archives: `4b8a6788bdb11734a5bbc471f46400a5df0621ccb758abe2659c06bb2c8f4478`
- H-003 runner: `066ae9bcee39fc82e491a6b42aa58e5a540e286234dd862a9ba50f3c5a29dd40`
- Results JSON: `a7f201851570fb79569fcd72f2a84a0c66c46cff8503c2d39ca655d01187ded4`
- Static control score artifact: `7468d0ca25a34b7d05b0989b5f88c5aa849fac28b926c2304c4e7e36f4bd2eda`
- Local temporal score artifact: `1f21197eab4e4c80a20c62c6ae7490fa4d4c99590f90570cda1a46b3a9644e35`
- Frozen target checkpoint: `8ce4ebe5f5e4ebb6fcbd02a9599328059e51d5ecc3bd9f52bdb66c822f994090`

Input hashes for feature caches, extractor, tokenizer and replay implementation are in [`RESULTS.json`](../../research/long_running_method_discovery/cycle_02/h003_insertion/linear_probe/RESULTS.json). Full exact event score arrays are saved beside it.

## Runtime and reproduction

`nvidia-smi` could not communicate with the NVIDIA driver, so no live GPU lease or VRAM telemetry was available. No neural model was trained or inferred in this experiment. Only conventional CPU LogisticRegression was fitted to the frozen features; thus there is no CUDA receipt or neural checkpoint. The failed system `python` import was resolved by using the repository `.venv`, which has scikit-learn 1.9.1 and SciPy 1.17.1.

From the repository root:

```bash
.venv/bin/python research/long_running_method_discovery/cycle_02/h003_insertion/run_h003_linear_probe.py
```

The runner validates all N/N+1 populations before evaluation and writes versioned outputs under `research/long_running_method_discovery/cycle_02/h003_insertion/linear_probe/`. Re-running into the same output folder overwrites this arm's own generated score files and results JSON.
