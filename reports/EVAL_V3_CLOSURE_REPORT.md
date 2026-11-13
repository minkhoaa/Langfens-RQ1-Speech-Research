# EVAL-V3 — Báo cáo đóng task

## 1. Kết luận bàn giao

**Đóng task trong phạm vi đã được phép.** Pipeline, frozen replay, calibration/evaluation, đối soát, kiểm toán và publication cuối đã được thực hiện. Không tuyên bố mọi thí nghiệm lịch sử đều tái hiện được: các trường hợp thiếu fitted artifacts, native runtime không có CUDA, protected cohort và nguồn tham chiếu chưa độc lập vẫn giữ trạng thái giới hạn có bằng chứng.

Bản phát hành chính thức: `experiments/evaluation_v3/closure/release_final/`. Các thư mục `standardized_final*`, inventory v2–v4 và progress trước đây là snapshot trung gian, không phải kết luận mới nhất. Commit nguồn baseline là `cc4e331`; commit closure được ghi trong Git khi bàn giao. Chỉ commit local, không push/deploy.

### Quy mô thực tế

| Hạng mục | Kết quả |
|---|---:|
| Frozen neural experiments legacy đã suy luận CUDA | 50 |
| Teacher checkpoints đã suy luận CUDA native FP16 autocast | 9 |
| Deterministic controls | 3 |
| Canonical compositions/router | 24 |
| Rule/policy variants | 6 |
| Tổng model/variant closure có receipt CAL + SEARCH | **92** |
| Student baseline giữ nguyên | 14 arm/seed |
| Original inventory | 290 dòng |
| Inventory phát hành, thêm riêng M57-on-EXP52 | **291 dòng experiment/seed/variant** |
| Results phát hành | **1.340 dòng = 500 baseline + 840 operating evidence mới** |
| Subset primary matched/eligible | 865 dòng; không đồng nghĩa 865 mô hình độc lập |

CPL01 có thêm bốn arbitration operating points từ ledger registry đã tồn tại, không phải bốn checkpoint mới. Hai lựa chọn I-bank là hai score-vector khác nhau, không bị gọi nhầm là thay đổi decision policy trên cùng scores.

## 2. Cohort, waveform và định nghĩa đánh giá

- Full authorized WAV replay: **2.699 utterances**, sai khác feature với cache **0**. Receipt và trace tại `closure/waveform_full/`.
- **28/28** student split-score comparisons, 14 students × CAL/SEARCH: sai khác **0**.
- **9/9** teacher SEARCH score comparisons với ledger native AMP: sai khác **0**. Kết quả FP32 khác native được giữ riêng làm diagnostic, không dùng làm benchmark.
- Orthographic-reference mode: **2.660 PASS, 39 OOV abstentions**. Không ép phone khác nhau thành gold canonical để tạo parity giả.
- CAL: 300 utterances, 10.043 targets, 10.343 exact gaps. SEARCH: **599 utterances, 19.755 targets, 20.354 exact gaps**.
- SEARCH supports: C **17.778**, S **1.768**, O **209**, I **125**; bốn speakers.

Một `UnifiedEvaluatorV3` duy nhất quyết định metric chính. Target C/S/O là mutually exclusive; insertion dùng đúng N+1 gap slots, có leading/trailing gap. Không trộn thành four-way softmax. Macro3 error là S/O/I, không phải C/S/O. AP dùng native probability-valued ranking scores, không thay bằng class logits hoặc calibrated ranks khi so sánh khả năng gốc.

`FWR = (FP_S + FP_O + FP_I) / số utterances`; wrong-type error được tính đúng, không chỉ false alarms trên gold CORRECT. CORRECT là metric thứ cấp vì prevalence. Thiếu head/score hoặc zero-support được ghi NOT_ESTIMABLE, không điền số 0 để tạo cảm giác đã đo.

## 3. Calibration, operating policies, confidence và robustness

- Fit threshold, isotonic maps và finite source mixture grids trên **DEV_CAL**, rồi mới assessment SEARCH. SEARCH là **development-exposed, không blind test**.
- Fit receipt ràng buộc scores, labels, unique event IDs và speaker-event mapping bằng hash; không chỉ nhận một danh sách speaker chung.
- Raw AP được giữ bất biến khi chỉ đổi decision policy. Publication verification đã xác minh **94 score-vector identity groups**: 92 closure models và hai CPL01 bank selections.
- Isotonic target maps là các binary maps độc lập, **không renormalize thành simplex giả**.
- Raw threshold 0,9 cố định được đổi nhãn `PREDECLARED_RAW_0.9_DIAGNOSTIC`; không gọi là CAL-fit hoặc bảo đảm precision 90%.
- PR đầy đủ được stream và đóng gói lossless chunks có hash. Recall/FP-count views dùng cùng dữ liệu, không phải thêm một scorer. Với class CORRECT, FP density **không phải false-warning rate**; per-head S/O/I sweeps cũng không tự trở thành joint mutually-exclusive policy FWR.
- `classification_selective_v2/`: **92 models, 175 domain tasks, 15.447 classification risk/coverage points**, giữ full-population FN khi abstain. Đây là classification error among covered cases, khác alert precision/risk.
- Cùng thư mục có **682 Top-K rows**: raw và CAL-isotonic cho 341 supported heads. Ties dùng canonical dataset/speaker/utterance/domain/**numeric slot**, không dùng gold, invariant khi shuffle rows.
- `stratified_full/`: **92 models, 175 domain tasks, 9.664 strata rows**, bao gồm reference-phone families, exact-gap context, speaker, CTC spans và CTC-vs-gold-duration.
- Có **3.354 derived error-subtype recall/FN rows**. Đây là gold-evaluation-only diagnostics; không gọi subtype derived là dataset-native subtype, không tính AP trên các nhóm gold-positive-only rồi diễn giải như end-to-end capability.
- Gold timing/realized phones chỉ dùng phân tích strata, không cấp vào inference hay fit policies. CTC frames đổi sang seconds với hệ số **0,01**, có test tránh lỗi lệch đơn vị 100 lần.
- `reliability_figures_final/`: **92 SVGs, 341 supported-head panels**, raw và calibrated reliability; teacher head thiếu được ghi NOT_ESTIMABLE. Rendering tái dùng canonical diagnostics, teacher raw reliability dùng chính helper canonical, không thêm metric engine.

## 4. Kết quả khoa học đáng chú ý

### Capability ranking mô tả, không chọn deployment winner

| Canonical variant | AP S | AP O | AP I exact-gap | Macro3 error AP |
|---|---:|---:|---:|---:|
| EXP36 | 0,387088 | 0,204687 | 0,275924 | **0,289233** |
| EXP43 | 0,363453 | 0,205771 | 0,285056 | 0,284760 |
| EXP29 | 0,378062 | 0,196539 | 0,272068 | 0,282223 |
| EXP24 | 0,357737 | 0,225092 | 0,247628 | 0,276819 |

EXP36 đứng đầu raw Macro3 AP của bảng phát hành; đây là canonical CAL-selected composition mới, **không phải historical parity** hay chứng minh superiority trên blind test. Teacher S_EXP01 có AP S **0,388469522**, chỉ trong supported S-head scope, không tự có O/I capability.

### Các phát hiện ngăn diễn giải sai

1. Source audit xác nhận **38 historical methods có gold-conditioned inference routing** qua `target_m` từ realized/error-conditioned auxiliary labels. Các replay mới không cấp gold auxiliary labels; dùng native `target_m=None` head average và gắn `NONORACLE_NATIVE_HEAD_AVERAGE`. Không quy toàn bộ chênh lệch lịch sử cho việc đổi evaluator.
2. Intended-reference lineage vẫn annotation-derived/unverified independence. WAV parity không chứng minh reference độc lập hoặc zero leakage.
3. M93/M99/M106 có native CAL proxy khác SEARCH real components; canonical variants sửa sự không nhất quán đó, không nhận là đã tái hiện đúng kết quả native cũ.
4. M98 abstention được giữ thật, FN không biến mất. M89 chỉ replay score-space routing: **không có bằng chứng measured skipped compute hoặc latency speedup**.
5. M57 native grid objective là hằng số phân tích, không phải CAL optimization thực tế; variant mới fit CAL riêng. M72 dùng dictionary constants, không giả là empirically learned. M88 dùng fused entropy proxy, không giả là constituent disagreement.
6. M14 nạp M13 rồi phân tích sáu L1 groups; không refit CV folds. M74 có unsupervised transductive preprocessing trên authorized audio, không được gọi TRAIN-only.
7. Native quirks như M68 duration-index behavior và unused centroids của M61 được giữ và công bố, không âm thầm thay model để làm kết quả đẹp hơn.

## 5. Kiểm toán và kiểm thử

- **159 tests passed**, sáu NumPy warnings từ native empty-slice feature behavior; không có test failure. Log: `coordination/EVAL_V3_CLOSURE_TESTS_FINAL.log`.
- All-family receipt audit **PASS**: 50 neural + 9 teachers + 3 deterministic + 24 composition + 6 policy; **184 CAL/SEARCH output hashes** đối soát theo inventory, exact event populations, finite probability bounds, simplex ở nơi protocol yêu cầu và execution-source snapshots.
- Original source freeze: **1.490 hashes verified, 9 protected/replication paths không mở lại**; không nói đã đọc lại đủ 1.499 files.
- Independent adapter review: **342/342 helper hashes**, 50 receipts/100 outputs, metadata order/bounds/simplex PASS; maximum simplex error khoảng 1,79e-7.
- Independent numerical review: năm NN/compositions (EXP13/24/38/11/86), mỗi model sáu selected policy rows, AP/F1 khớp ≤1e-10 và TP/FP/FN/TN khớp tuyệt đối; teacher S_EXP01 được kiểm tra riêng. Hai composition arrays khớp independent reconstruction **delta 0**.
- Independent PR stream validation: **119.696 rows**, counts và precision/recall khớp. Code độc lập không import evaluator/policies chính và không trở thành scorer production thứ hai.
- `release_final/PUBLICATION_VERIFICATION.json`: full denominator, FWR arithmetic, policy-invariant raw AP và unique **experiment_id + arm_seed** keys PASS. 291 inventory rows không có nghĩa 291 experiment IDs khác nhau.

Không che sự cố: child từng chạy 18 synthetic learned CPU forwards trái contract; evidence đó bị loại, có global guard và official CUDA replay thay thế. Child khác từng ghi sáu draft fusion files ở root; đã quarantine nguyên trạng, parent tự tái tạo official results. Bản strata child chỉ 83 models và chưa đủ semantics không được nhận làm final; parent hoàn thiện đủ 92. Render thử với schema thiếu gap/teacher bị loại; bản final có đủ 341 panels. Chi tiết tại `coordination/EVAL_V3_EXECUTION_INCIDENTS.md`.

## 6. Runtime và tài nguyên: giới hạn so sánh

Full waveform invocation ghi **265,887 giây**, peak **486,606 MiB** trên GTX1650; có reuse pilot cache, **không phải cold end-to-end latency benchmark cho toàn bộ 2.699 utterances**. NN/teacher receipts có parameter counts, CUDA walltime và peak VRAM riêng. Cached-head timings và score-composition CPU timings không được quảng bá như walltime end-to-end hoặc so sánh kiến trúc đã được chuẩn hóa. Không suy ra cascade speedup từ tỷ lệ routing.

## 7. Các blocker/giới hạn được đóng bằng disposition, không làm giả recovery

| Trường hợp | Disposition cuối |
|---|---|
| F4 C33/C50/C70/C85 | Thiếu fitted historical LR coefficients; không refit để tạo lịch sử giả. |
| 5 dynamic-quantized checkpoint files: EXP33/47/66×2/96 | Native dynamic operators có CPU dispatch nhưng không CUDA dispatch. Không chạy learned CPU inference, không dequantize/substitute float. Không tuyên bố mọi loại quantization đều thiếu CUDA. |
| M107 | Phụ thuộc M96 INT8 bị chặn; native source còn chọn exit threshold trên SEARCH. Không thay bằng M94 float hay bịa IDs. |
| M32/M46, original M14 mixed cohort | Phụ thuộc protected DEV_REPLICATION. Không chạy noisy SEARCH rồi gắn ID lịch sử. |
| EXP18 và A/B/C102/103 runs | Thiếu persisted per-seed weights/continuous score artifacts trong authorized sources; không retrain. Không alias số 102 sang legacy EXP102. |
| External protocols 09/35/41/51/78 | Source-only/outside authorization; không probe/download corpus. |
| UPR controls khác canonical topology | Giữ distinct-key/cohort mismatch; không ép 49 extra controls hoặc 140 missing canonical slots thành matched evidence. |
| Intended reference independence | Chưa xác minh; không blind-validation/zero-leakage/deployment certification. |
| Dataset-native fine-grained subtype | Không có; chỉ công bố derived evaluation-only types, không đổi tên cho giống ground truth native. |

Source candidates còn pending trong inventory final: **không có**. Điều đó không biến các blocker thật thành PASS. Hành chính/nonmodels và protocol-only analyses đã phân loại từ registry/source AST, không dựa vào tên như ROADMAP.

## 8. Artifact bàn giao và tái lập

- Bảng đầy đủ, primary subset, specialist subset, operating points, inventory và leaderboard: `experiments/evaluation_v3/closure/release_final/`.
- CAL-fit/confidence/curves: bốn `closure/confidence_*_v1/` bundles; Top-K canonical và classification-selective final ở `closure/classification_selective_v2/`.
- Full robustness: `closure/stratified_full/`; plots: `closure/reliability_figures_final/`.
- Inventory/dispositions: `closure/inventory_final/`; source audits: `closure/source_validity/`, `closure/checkpoint_source_chain/`.
- Review độc lập: `closure/independent_adapter_review/`, `closure/independent_numerical_final/`.
- Requirement-by-requirement acceptance: `reports/EVAL_V3_REQUIREMENTS_ACCEPTANCE.md`.

Lệnh kiểm thử:

```sh
PYTHONPATH=. .venv/bin/python -m pytest -q research/evaluation_v3 tools/test_eval_v3_safety.py
```

Large PR gzip được lưu lossless trong `*.parts/INDEX.json` và các chunks; dùng `research.evaluation_v3.chunked_artifact restore` để phục hồi trước khi đọc symlink views trên clone mới. Waveform per-utterance caches chỉ giữ local, có hash index; không bulk-commit dữ liệu người dùng hoặc checkpoint/corpus ngoài scope. Replay CUDA cần authorized inputs và GPU lock theo contract. Không training/backward/optimizer, restart Phase3B, protected/sealed access, download, push hay deployment được dùng để lấp blocker.

**Điểm dừng:** bàn giao đầy đủ evidence có thể tạo trong scope, giữ nguyên lịch sử, ghi rõ những gì không thể chứng nhận; không mở thêm vòng research hay lựa chọn winner bằng SEARCH.
