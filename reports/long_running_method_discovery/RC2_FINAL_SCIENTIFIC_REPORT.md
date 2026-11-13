# LANGFENS Research Cycle 2 — Báo cáo khoa học tổng hợp

Ngày: 2026-11-07. Model của sáu workstream: `gpt-6-luna` qua native collaboration subagent. Source revision khi khởi động: `aa4e6ea2147180a4a820d77a71911f333fcb6721`.

## Kết luận điều hành

Sáu workstream đã được dispatch bằng model nhỏ `gpt-6-luna`. Công cụ subagent của phiên cho tối đa ba agent ngoài supervisor chạy đồng thời, vì vậy workstreams được khởi chạy thành các đợt song song. Hai so sánh mới chạy được trên CPU mà không chạy neural model: (1) forced Viterbi so với optional-phone max-sum trên frame outputs đã đóng băng (H-001); (2) baseline 6D gap so với local temporal pooling 6D bằng logistic regression (H-003). H-001 cải thiện omission ranking trên DEV_SEARCH đã exposure, nhưng omission argmax F1 vẫn bằng 0. H-003 local pooling giảm exact-gap insertion AP. Cả hai là kết quả khám phá trên development set, không phải xác nhận độc lập.

Không có huấn luyện/inference neural Cycle 2 và không có CUDA receipt: `nvidia-smi` không kết nối được driver. H-002/H-004/H-006 là source/provenance preflight; H-005 audit lại isotonic artifact đã có, không fit calibrator mới. Vì vậy không tuyên bố Cycle 2 đã hoàn tất nghiệm thu kỹ thuật, không chọn mô hình thắng, và không đề nghị tích hợp vào Phase 4.

## 1. Trạng thái sáu giả thuyết

| ID | Phương pháp thực sự kiểm tra | Kết quả Cycle 2 | Phán quyết |
|---|---|---|---|
| H-001 | CPU DP trên frozen frame-classifier outputs, forced Viterbi A so với optional-phone max-sum B; cùng event population | AP O trên SEARCH 0,036767→0,087932; Macro-3 Error AP 0,196028→0,213083; O F1=0 ở cả hai arm; tất cả 4 speaker có AP O tăng | `SUPPORTED_EXPLORATORY` cho ranking của score path này; chưa có lợi ích operating F1 và chưa kiểm tra CTC sequence marginalization |
| H-002 | Kiểm kê cache và model cục bộ, thiết kế matched SSL probe | Baseline GRU cache 51D target/6D gap, không phải frame states. Checkpoint Wav2Vec2 tại máy đã fine-tune ASR; không tìm thấy checkpoint pretraining-only trong vị trí đã kiểm tra | `UNRESOLVED`; chưa có inference, training, AP mới |
| H-003 | CPU LogisticRegression cùng 6D input/control, static gap so với temporal pooling; full N+1 gaps | AP I 0,040023→0,011320; I F1 0,112450→0,021148; Macro-3 AP 0,188019→0,178452. ROC-AUC I tăng 0,58593→0,67798 nhưng rare-positive ranking ở AP giảm | `NOT_SUPPORTED_EXPLORATORY` cho local linear pooling này; không bác bỏ attention/neural gap model |
| H-004 | Kiểm toán lineage teacher/OOF | Raw Phase 3 speaker-grouped OOF specialist replicas có provenance; calibrated TRAIN targets dùng isotonic fit trên DEV_CAL. EXP-36 là composition, không phải teacher độc lập. Không train transfer mới | `UNRESOLVED`; Phase 3 trước đó không chứng minh KD-All hơn Gold-Only |
| H-005 | Đọc lại frozen EVAL-V3 isotonic results | EXP-24 isotonic trên SEARCH giảm Brier/ECE so với raw ở S/O/I; omission Brier vẫn cao hơn baseline prevalence-only | `PARTIAL_EXISTING_EVIDENCE`; không phải fit mới hay xác nhận robust calibration |
| H-006 | Kiểm tra bảng phonological và baseline | Bảng 16D hiện có gộp nhiều contrast; intended-phone one-hot đã vào model, nên auxiliary task có shortcut sao chép reference | `UNRESOLVED`; chưa train place/manner/voicing head |

## 2. H-001 — omission-aware alignment

Experiment `RC2-H001-A-B-OPTIONAL-PHONE-CPU-20261107` dùng cùng frame emissions CharSiU, intended phone sequences, model scores còn lại, speaker membership và event keys. Forced alignment replay khớp feature cache tối đa `1,15e-6` chênh mean log posterior và bằng 0 chênh duration. EVAL-V3.1 xác nhận 89.360 target, 92.059 gap trên TRAIN/CAL/SEARCH; DEV_SEARCH có 19.755 target, 20.354 gap và 209 omission.

| DEV_SEARCH arm | AP O | AP S | Exact-gap AP I | Macro-3 Error AP | O P/R/F1 tại argmax | Macro-3 Error F1 | Macro-4 F1 | Joint FWR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A forced Viterbi | 0,036767 | 0,364602 | 0,186715 | 0,196028 | 0/0/0 | 0,215857 | 0,393203 | 2,8965 |
| B optional-phone | 0,087932 | 0,364602 | 0,186715 | 0,213083 | 0/0/0 | 0,215646 | 0,393133 | 2,8715 |

ΔAP O là `+0,051165`; ΔMacro-3 Error AP là `+0,017055`. S/I scores được giữ nguyên nên AP của chúng không đổi. AP O tăng ở cả NCC, RRBI, YBAA, YDCK; bốn cụm vẫn quá ít cho suy luận speaker population. Tại chính sách argmax, false omission decisions trên CORRECT giảm 8→4/17.778 ở SEARCH và trên gold-duration ≤30 ms giảm 1→0/298; đây là strata mô tả, gold timing chỉ dùng sau inference. O precision/recall/F1 vẫn bằng 0, vì vậy chưa có quyết định omission hữu ích.

**Giới hạn quan trọng:** field `ctc_logits` trong cache là output của `Wav2Vec2ForFrameClassification` CharSiU, có lineage frame-classification trên forced-aligned labels. Không gọi chúng là CTC sequence posterior. H001-C forward-sum likelihood không chạy. H001-B bỏ phone state nhưng giữ hai SIL state kề nhau; chưa có SIL-collapse ablation. Kết quả ủng hộ score ranking exploratory, không xác nhận deployment benefit hay generalization.

Receipt: `research/long_running_method_discovery/cycle_02/h001_alignment/run_20261107_viterbi_optional_cpu_v3/EVALUATION_RECEIPT.json`, SHA-256 `a33af22380141b174c86d790a0d2c4959e1a31ffcca9ebb7b6c197c91ab1ced8`. Script SHA-256 `c0e9602c130e24b18681f0ae792fd3574e1d5a2102e45f02a536edd5cd54050a`. Frozen STU-01 checkpoint SHA-256 `8ce4ebe5f5e4ebb6fcbd02a9599328059e51d5ecc3bd9f52bdb66c822f994090`.

## 3. H-003 — exact-gap insertion

Experiment `RC2-H003-CPU-LOGREG-001` so sánh hai LogisticRegression `C=1`, balanced, 500 iterations, cùng chuẩn hóa TRAIN-only, cùng frozen STU-01 target scores. Chỉ gap representation đổi: static 6D so với sáu thống kê từ cửa sổ 11 frame quanh gap. Ngưỡng được chọn trên lưới cố định 201 điểm ở DEV_CAL; không dùng SEARCH để chọn arm. Cả hai giữ 599 utterance, 19.755 targets, đủ **20.354/20.354 exact gaps**, gồm leading/trailing.

| DEV_SEARCH arm | AP I exact-gap | ROC-AUC I | I precision | I recall | I F1 | Macro-3 Error AP | Macro-3 Error F1 | Macro-4 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Static 6D | 0,040023 | 0,58593 | 0,11290 | 0,11200 | 0,11245 | 0,188019 | 0,251476 | 0,415602 |
| Local temporal 6D | 0,011320 | 0,67798 | 0,01304 | 0,05600 | 0,02115 | 0,178452 | 0,221042 | 0,392777 |

AP I giảm `0,028703`, F1 giảm `0,091302`; speaker AP I giảm ở cả bốn speaker. ROC-AUC tăng nhưng không cải thiện thứ tự ở đuôi hiếm theo AP. Một phép tính grouped-tie độc lập từ saved scores và gold exact-gap IDs xác nhận AP lần lượt `0,040022890777950` và `0,011320187270797`. Đây là một lần fit thống kê CPU; không phải neural gap modeling. Giữ kết luận `NOT_SUPPORTED_EXPLORATORY` đúng phạm vi local linear pooling.

Runner SHA-256 `066ae9bcee39fc82e491a6b42aa58e5a540e286234dd862a9ba50f3c5a29dd40`; results SHA-256 `a7f201851570fb79569fcd72f2a84a0c66c46cff8503c2d39ca655d01187ded4`. Frozen target checkpoint SHA-256 như trên.

## 4. H-005 — calibration evidence hiện có

Đây là trích xuất từ EVAL-V3 closure đã có, không phải Cycle 2 calibration fit. Isotonic EXP-24 fit trên DEV_CAL (10.043 targets/10.343 gaps, 2 speakers) rồi đánh giá trên DEV_SEARCH (19.755/20.354, 4 speakers; set đã exposure):

| Head | Brier raw | Brier isotonic | Baseline prevalence | ECE raw | ECE isotonic |
|---|---:|---:|---:|---:|---:|
| S | 0,101525 | 0,077436 | 0,081487 | 0,165060 | 0,076090 |
| O | 0,016898 | 0,011893 | 0,010468 | 0,020753 | 0,015668 |
| I | 0,009312 | 0,005327 | 0,006104 | 0,010242 | 0,002080 |

Reliability so với raw tốt hơn trong artifact này; O Brier vẫn không bằng baseline prevalence. Không có đối chiếu hợp lệ temperature/beta/isotonic bằng speaker-grouped TRAIN OOF. Hai speaker CAL không đủ để tuyên bố hiệu chuẩn robust trên speaker mới; raw AP là discrimination chính và isotonic ties có thể làm thay đổi AP.

## 5. H-002, H-004, H-006 và baseline context

- **H-002:** cached GRU baseline là 51D/6D summaries, không chứa frame hidden states. Máy có Wav2Vec2-Base ASR fine-tuned và CharSiU phone-frame classifier; chưa nhận diện checkpoint pretraining-only trong cache đã kiểm tra. Không có CUDA nên không chạy extraction/head training. Không kết luận SSL giúp hay không.
- **H-004:** Phase 3 raw TRAIN specialist scores là speaker-held-out OOF replicas nhưng ngân sách thấp hơn champion. Các calibrated TRAIN scores áp dụng isotonic fit trên DEV_CAL, do đó không phải TRAIN-only supervision. EXP-36 là composition. Không chạy feature/disagreement distillation mới. Historical Phase 3 Gold-Only/KD-All ba seed chỉ cho thấy kết quả gần nhau, không khẳng định KD thắng.
- **H-006:** bảng phonological 16D có place/manner contrasts thô, nhiều vowel pairs chung vector. Vì input đã có intended-phone identity, auxiliary labels derived từ intended phones có thể làm mô hình copy reference; cần reference-only và acoustic-only controls. Chưa train.
- Gold-Only seed 101 historical EVAL-V3 baseline trên SEARCH: AP S/O/I `0,3646021 / 0,1594330 / 0,1867152`, Macro-3 Error AP `0,2369168`. SEARCH development-exposed; không phải xác nhận độc lập.

## 6. Tổng hợp 15 câu hỏi nghiên cứu

1. **Thực sự thử:** H-001 frozen-emission alignment A/B; H-003 static-vs-local 6D CPU logistic. H-005 chỉ kiểm tra lại artifact closure. H-002/H-004/H-006 mới audit/preflight.
2. **Cải thiện đo được:** H-001 tăng AP omission exploratory, nhưng không tăng omission F1. Không có cải thiện mới được xác nhận độc lập.
3. **Không được ủng hộ:** H-003 temporal pooling tuyến tính làm AP/F1 insertion giảm; chưa kết luận về attention hay neural trajectories.
4. **Deletion-aware omission:** optional-phone path tăng AP O `0,036767→0,087932` trên SEARCH; O F1 vẫn 0. Vì SEARCH đã exposure, đây là tín hiệu cần tái kiểm tra.
5. **Frame-level SSL:** chưa thử; checkpoint pretraining-only và CUDA hiện chưa sẵn sàng.
6. **Gap-centered insertion:** local 11-frame pooling không cải thiện exact-gap AP; ROC-AUC không phản ánh đuôi positive hiếm trong phép thử này.
7. **Advanced multi-teacher:** chưa thử; không có cơ sở nói vượt Gold-Only.
8. **Held-out speaker calibration:** artifact isotonic cải thiện reliability trên 4 speaker SEARCH nhưng không phải test mù và không đủ cho khẳng định robust unseen-speaker.
9. **Articulatory auxiliary:** chưa thử; prior-copying là rủi ro thiết kế chính.
10. **Lặp seed:** H-001 là deterministic scoring, H-003 một fit; Cycle 2 chưa có lặp seed cho các intervention mới. Không có cải thiện mới nào xác nhận lặp lại qua seed.
11. **Lexical/reference so với audio:** Cycle 2 chưa chạy đủ reference-only/audio-only controls. H-001 phụ thuộc intended reference để định nghĩa path; H-003 dùng frame emissions quanh vị trí suy ra từ alignment. Chưa tách causal contribution.
12. **GTX 1650 4GB:** CPU path scoring và logistic probe chạy được mà không GPU. Khả năng chạy neural H-002/H-004/H-006 trên GPU chưa đo trong phiên vì driver không hoạt động; historical Phase 3 resource receipts không thay thế phép đo Cycle 2.
13. **Reference independence:** chưa giải quyết. Canonical intended phones vẫn annotation-derived; không tuyên bố độc lập với annotation.
14. **Tích hợp Phase 4:** chưa đề nghị tích hợp. H-001 đáng cho một controlled follow-up; H-003 linear intervention không đáng tích hợp. Calibration/auxiliary/distillation cần OOF và CUDA evidence trước.
15. **Bước tiếp theo:** kiểm soát topology SIL và OOF scorer của H-001 trước khi dùng model CTC hợp lệ; sau đó khôi phục CUDA để chạy matched neural feature/auxiliary arms. DEV_SEARCH không được dùng làm bộ chọn người thắng; cần speaker cohort độc lập để xác nhận.

## Giới hạn và phán quyết

Hai result dùng DEV_SEARCH mô tả trên bốn speaker đã exposure; chúng không phải test độc lập. Chỉ H-001 và H-003 có controlled Cycle 2 comparison; H-001 chỉ dùng max-sum discriminative path score từ frame-classifier emissions, không phải CTC likelihood. Không có neural training/inference mới, CUDA lease, CUDA telemetry, hoặc kết luận generalization tới TEST_SEALED. Không áp dụng FWR≤0,20 làm điều kiện thành công.

Các báo cáo độc lập: [H-001](/home/khoa/Projects/New-Method-Research/reports/long_running_method_discovery/RC2_H001_ALIGNMENT.md), [H-002](/home/khoa/Projects/New-Method-Research/reports/long_running_method_discovery/RC2_H002_SSL.md), [H-003](/home/khoa/Projects/New-Method-Research/reports/long_running_method_discovery/RC2_H003_INSERTION.md), [H-004](/home/khoa/Projects/New-Method-Research/reports/long_running_method_discovery/RC2_H004_MULTITEACHER.md), [H-005](/home/khoa/Projects/New-Method-Research/reports/long_running_method_discovery/RC2_H005_CALIBRATION.md), [H-006](/home/khoa/Projects/New-Method-Research/reports/long_running_method_discovery/RC2_H006_ARTICULATORY.md).

## Cập nhật CUDA Cycle 2 — khởi động lại trong `.venv-cuda` (2026-11-07)

Driver NVIDIA đã hoạt động trong phiên Linux hiện tại: GTX 1650, driver `615.78.08`; `nvidia-smi` hiện bình thường và báo không có tiến trình GPU sau baseline. Smoke test CUDA 13.0 đã qua trước đó. Baseline Gold-Only Phase 3 được huấn luyện bằng `/home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python`, seed 101, 12 epoch, 684 updates, 78,661 tham số; thời gian 44.39 giây, peak allocated VRAM 44.09 MiB. Job `RC2-GOLD-ONLY-CUDA130-S101` trả mã 0 và lease đã được giải phóng.

Đánh giá EVAL-V3 trên DEV_SEARCH (chỉ thăm dò, không dùng chọn mô hình): 19,755 target, 20,354 exact gaps, 599 utterance. AP S/O/I lần lượt `0.364602 / 0.159433 / 0.186715`; Macro-3 Error AP `0.236917`. Theo ngưỡng chọn trên DEV_CAL trong budget 0.05, F1 S/O/I là `0 / 0 / 0.075188`, Macro-3 Error F1 `0.025063`, Macro-4 F1 `0.255615`; số false warning quan sát trên SEARCH là 5. Đây là baseline sàng lọc một seed và DEV_SEARCH đã phơi nhiễm; không phải kết luận tổng quát hóa.

Receipt và hash đầy đủ: `research/long_running_method_discovery/cycle_02/baseline_control/RC2-GOLD-ONLY-CUDA130-S101-receipt.json`. Checkpoint SHA-256 `52c6c26bb52c119915942da8a3aaf9fdbc109911b77a55df8cf813ca46f7dc4f`. So sánh chính thức với EXP24/29/36/43 và các thí nghiệm mới vẫn cần cùng evaluator/population; không chọn kiến trúc từ riêng SEARCH.

## Cập nhật bổ sung: các sàng lọc CUDA mới (2026-11-07)

### H-001: optional-phone + acoustic student

Đã so với Gold-Only seed 101, 12 epoch/684 updates, cùng event universe: 59.562 TRAIN, 10.043 CAL, 19.755 SEARCH targets và 20.354 exact gaps. AP S/O/I là `0.356376 / 0.161252 / 0.169413` so với `0.364602 / 0.159433 / 0.186715`; Macro-3 Error AP `0.229014` so với `0.236917`. Kết quả một seed này **không ủng hộ** thêm scalar optional-phone score: O tăng nhẹ nhưng S và I giảm. Checkpoint SHA-256 `751e3e4b601dd6c54e443074c88ec516b73d95f4036c2b9b7dd153cbc5c0d454`; EVAL-V3 JSON SHA-256 `6321c4d22f499f3723d1556fa4e13c3219526d5630b73ad6ba62bfef5bda9947`. Xem [báo cáo H-001](/home/khoa/Projects/New-Method-Research/reports/long_running_method_discovery/RC2_H001_ALIGNMENT.md) và receipt trong `cycle_02/h001_alignment/combined_acoustic/`.

### H-004: raw speaker-grouped OOF scalar KD

Job `RC2-H004-RAW-OOF-KD-CUDA130-S101-R4` dùng raw OOF teacher probabilities (không dùng DEV_CAL-calibrated scores), Gold-Only student cùng kiến trúc/seed/budget, 12 epoch/684 updates, GTX 1650 CUDA 13.0, peak VRAM `44.14 MiB`, runtime `85.09 s`; checkpoint SHA-256 `f1df7b2724090718d61ded519b5a6965dd400adee7775b52e8f1d93eea0bbc01`. SEARCH AP S/O/I `0.368655 / 0.158587 / 0.192241`, Macro-3 Error AP `0.239828`, so với Gold-Only `0.364602 / 0.159433 / 0.186715`, `0.236917`: delta nhỏ và hỗn hợp. Ở budget 0.20, Macro-3 Error F1 `0.104513` so với `0.104897`; ở 0.05 là `0.029851` so với `0.025063`. Một seed không xác nhận KD thắng. EVAL-V3 JSON SHA-256 `cd77f4cf97562acb41104a392806408334c9e99dee7347e8be06211a705e29f1`. Feature-level KD đang được chuẩn bị như can thiệp riêng; không trộn với scalar KD.

### H-006: auxiliary articulatory objective

Một seed 101 so với architecture-matched control có cùng auxiliary head nhưng weight 0. Control xác nhận thêm nhánh không đổi AP ở seed này. Treatment đạt AP S/O/I `0.364057 / 0.160935 / 0.192187`, Macro-3 Error AP `0.239060`, delta Macro-3 AP `+0.002143`; F1 Macro-3 tại budget 0.20 là `0.113712` so với `0.104897`, nhưng tại 0.05 hai arm đều `0.025063`. S AP giảm nhẹ và O/I tăng nhẹ. Đây là **SUPPORTED_EXPLORATORY** cho mixed shift, không đủ chọn mô hình hay tích hợp: một seed, bốn speaker SEARCH, table 16D coarse, chưa có riêng place/manner/voicing ablations. Treatment checkpoint SHA-256 `a30a734ccaba7fb91c8df855a35927efa96936f032dbd52e2ac007e3b91c8e79`; control `b3432da6381451e10a64588d43f239d38ab41dd857142f0c14b8a8b95056aa52`. Chi tiết/commands trong [báo cáo H-006](/home/khoa/Projects/New-Method-Research/reports/long_running_method_discovery/RC2_H006_ARTICULATORY.md).

### H-002 input-contract correction; H-004 feature KD in preparation

Cache Wav2Vec2 lượt đầu dùng waveform thô dù checkpoint `facebook/wav2vec2-base-960h` khai báo `do_normalize=true`; cache/checkpoint được giữ trong `h002_ssl/failures/attempt4_raw_input_contract/` và bị loại khỏi metric. Lượt attention đó kết thúc mã 1 trước khi tạo prediction. Run chuẩn hóa mới `RC2-H002-ASR-FINETUNED-NORM-S101` đang chạy bằng `AutoFeatureExtractor` trên CUDA, cache riêng `normalized_v1/`; model hiện có là ASR-finetuned, không phải pretraining-only. Chưa có kết quả H-002 hợp lệ. H-004 đang dựng embedding OOF độc lập từ teacher fold checkpoints dưới GPU wrapper để thử feature-level KD; chưa có kết quả để so sánh.

Đây là các phép sàng lọc một seed trên DEV_SEARCH development-exposed, không phải kiểm định độc lập và không chứng minh tổng quát hóa tới TEST_SEALED. So sánh EXP24/29/36/43 trong `unified_candidate/UNIFIED_SCREEN_SPEC.md` chỉ là tham chiếu từ release, vì một số là composition và provenance/reference independence khác nhau; chưa chọn một winner hoặc hợp nhất Phase 4.

## Bổ sung cuối: H-002 chuẩn hóa và H-004 feature KD đã có kết quả

### H-002: Wav2Vec2 ASR-finetuned + local reference-conditioned attention

Checkpoint `facebook/wav2vec2-base-960h` cục bộ là ASR-finetuned, chưa tìm thấy pretraining-only; vì vậy đây là **sensitivity arm**, không phải thuần SSL. Frame cache dùng `AutoFeatureExtractor` với `do_normalize=true`. Encoder đông băng ở `hidden_states[6]`, 17 frame quanh tâm alignment suy từ acoustic emissions + intended phones; không dùng gold boundary hoặc realized phone. Head có 223,875 tham số, train CUDA seed 101 8 epoch; peak allocated 32.45 MiB, runtime head 18.10 giây. Checkpoint SHA-256 `6739c8c9bdf00df5632693f2feebb5ef5907a499338af803ed6095a0aa120f50`; EVAL-V3.1 receipt SHA-256 `785a0bbbe418f301acfc954a59841f8cc2c229603c1969b458df31ed236a3ceb`.

Giữ nguyên 19,755 target và 20,354 gap exact; gap scores lấy nguyên từ Gold-Only, nên I AP/F1 không đổi. SEARCH AP S/O/I `0.377810 / 0.167294 / 0.186715`, Macro-3 Error AP `0.243940`, so với Gold-Only `0.364602 / 0.159433 / 0.186715`, `0.236917`: S/O ranking tăng, I không đổi. Ở policy được chọn trên CAL budget 0.10, class P/R/F1 treatment là S `0.4857/0.0096/0.0189`, O `0.4000/0.0383/0.0699`, I `0.6250/0.0400/0.0752`; Macro-3 Error F1 `0.054638`, Macro-4 `0.277771`, so với baseline `0.034250 / 0.262453`. SEARCH false-warning count/FWR là `33 / 0.0551`, so với `18 / 0.0301`. Ở CAL budget 0.20 Macro-3 F1 `0.125285` so với `0.104897`; unrestricted Macro-3 F1 gần như ngang `0.309546` so với `0.310212` và FWR cao hơn. Vì vậy F1 thay đổi theo operating point; không lấy một policy để tuyên bố thắng toàn diện.

Inference perturbations (không retrain, không refit AP; phân phối đầu vào bị thay đổi) làm AP S/O giảm: reference-only `0.293833/0.024503`, slot-shuffled acoustic `0.239753/0.023376`, zero-reference/acoustic-only `0.095798/0.011834`; full model là `0.377810/0.167294`. Đây là bằng chứng nhạy cảm với acoustic context và reference, nhưng không phải modality retraining controls hay kiểm định nhân quả. SEARCH đã exposed; speaker results dị biệt và chỉ có bốn speaker. Không tuyên bố generalization độc lập.

### H-004-C: feature-level teacher transfer

CUDA speaker-OOF embeddings S/O được trích từ fold checkpoints; feature KD dùng cosine loss weight 0.1, cùng seed 101, 12 epoch/684 updates, cùng kiến trúc 78,661 tham số so với weight-0 control. Candidate AP S/O/I `0.358274/0.156798/0.178040`, Macro-3 AP `0.231038`, thấp hơn control `0.364602/0.159433/0.186715`, `0.236917` ở cả ba lớp. Budget 0.20 Macro-3 F1 `0.093246` so với `0.104897`. **NOT_SUPPORTED_EXPLORATORY** ở lượt này; không tích hợp.

H-004-A raw scalar OOF KD vẫn có mixed delta nhỏ (Macro3 AP `0.239828`, budget-0.20 F1 `0.104513` so với `0.104897`), cũng chưa hỗ trợ superiority. H-006 vẫn exploratory; H-001 combined optional feature và H-004 feature KD không ủng hộ tích hợp. Do đó không có căn cứ để chọn một kiến trúc thắng mọi lớp hoặc đưa thành phần mới vào Phase 4. H-002 là candidate đáng chạy grouped TRAIN validation/seed repeats; hiện chỉ là arm sàng lọc. Không ghép H-002 với H-006/H-004 theo giả định additive.

Các thông tin chi tiết mới nhất, speaker tables và lệnh tái lập nằm trong [H-002](/home/khoa/Projects/New-Method-Research/reports/long_running_method_discovery/RC2_H002_SSL.md), [H-004](/home/khoa/Projects/New-Method-Research/reports/long_running_method_discovery/RC2_H004_MULTITEACHER.md), [H-006](/home/khoa/Projects/New-Method-Research/reports/long_running_method_discovery/RC2_H006_ARTICULATORY.md) và `research/long_running_method_discovery/cycle_02/unified_candidate/`.

### Score-level unified candidate assembly receipt

Đã tạo assembler theo manifest, join target `(utterance_id, slot_idx)` từ H-002 và exact-gap `(utterance_id, gap_idx)` từ CUDA Gold-Only; kiểm tra mỗi key đúng một lần, so khớp đầy đủ với N/N+1 manifest và xác nhận labels/probabilities tương ứng trước khi replay. DEV_CAL có 10,043 target/10,343 gap; DEV_SEARCH 19,755/20,354. EVAL-V3.1 replay khớp chính xác với H-002 receipt ở CAL policy budgets 0.10 và 0.20 cho AP S/O/I, Macro-3 AP/F1, Macro-4 F1 và joint FWR. Reproduction: [`assemble_screened_bundle.py`](/home/khoa/Projects/New-Method-Research/research/long_running_method_discovery/cycle_02/unified_candidate/assemble_screened_bundle.py), [`evaluate_screened_bundle.py`](/home/khoa/Projects/New-Method-Research/research/long_running_method_discovery/cycle_02/unified_candidate/evaluate_screened_bundle.py). Assembly receipt SHA-256 `1628b25225341919c4711d8c2181922d4c864265d45cdcd9e4320bcfc1e63481`; EVAL-V3.1 replay receipt SHA-256 `f8d3959c987990c2e03512f53327b2b8c43eb22b0be76db8fad7a3e837f1747c`; manifest [`RC2_SCREENED_UNIFIED_BUNDLE.json`](/home/khoa/Projects/New-Method-Research/research/long_running_method_discovery/cycle_02/unified_candidate/RC2_SCREENED_UNIFIED_BUNDLE.json) SHA-256 `9414e923f791b1c98fb21e7476e02c3878014dac8f275286d6616b2da7b9de5a`. Đây là score-level assembled result, không phải fused neural checkpoint. H-002 gap scores vẫn là Gold-Only; vì vậy candidate không thể được diễn giải là mô hình insertion mới.
