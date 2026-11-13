# M1 — đã train baseline realized-phone CTC

Ngày: 2026-11-13. **Hoàn tất 8 lượt train: 4 fold × seeds 101/202, mỗi lượt 8 epoch. Baseline còn yếu; không promote.** Đây là bước triển khai đầu tiên sau M0, chưa phải hoàn thành M1 reference reproduction hoặc goal nghiên cứu.

## Model và supervision đã train

Encoder `facebook/wav2vec2-base-960h` được giữ frozen; train **31.529 tham số** của head Linear(768,41) bằng CTC trên chuỗi âm thực tế. Inventory gồm 39 âm ARPAbet + AX và blank riêng. Không khởi tạo lại encoder và không dùng checkpoint đã fine-tune trên L2 speaker giữ lại.

Supervision lấy từ 1.243 câu strict đủ điều kiện trong audit M0, rồi loại speaker calibration/held ở từng fold. Không tự biến ERR hoặc deviation thành âm chuẩn. Các fold lần lượt có **628 / 473 / 615 / 770 câu fit**, tương ứng **320 / 240 / 312 / 392 optimizer update** mỗi seed. Tổng 8 lượt có **2.528 update**. Không chọn checkpoint bằng held/cal; tất cả lấy epoch cuối đã đăng ký.

MDD được đánh giá trên đầy đủ **1.800 câu, 59.562 target và 61.362 gap mỗi seed**. PER dùng riêng 1.243 câu có realized reference rõ, tổng 39.477 âm. Do subset, speaker fit, inventory và update budget khác lịch sử, không dùng bảng này để khẳng định strict supervision tốt/xấu hơn recipe cũ hoặc SM13.

## Kết quả gộp held theo seed

Operating point chính: policy được fit riêng trên 3 calibration speaker/fold với budget 0,20 cảnh báo sai/câu, sau đó đóng băng. Không dùng held labels sửa threshold.

| Seed | Realized PER, strict subset | Macro-3 error AP | Macro-3 error F1 | FWR held | Số cảnh báo sai |
|---|---:|---:|---:|---:|---:|
| 101 | 61,86% | 0,06728 | 0,00537 | 0,18833/câu | 339 |
| 202 | 61,11% | 0,06748 | 0,00699 | 0,19833/câu | 357 |

| Lớp | Precision 101 / 202 | Recall 101 / 202 | AP 101 / 202 | TP 101 / 202 | Support |
|---|---:|---:|---:|---:|---:|
| S | 15,15% / 16,98% | 0,203% / 0,244% | 0,15023 / 0,15024 | 15 / 18 | 7.381 |
| O | 4,55% / 4,76% | 0,493% / 0,548% | 0,04063 / 0,04179 | 9 / 10 | 1.825 |
| I, exact gap | 1,49% / 2,82% | 0,179% / 0,357% | 0,01097 / 0,01041 | 1 / 2 | 560 |

Các điểm FWR 0,10 và 0,50 cũng được xuất đúng đăng ký; xem `CLASSWISE.csv` và `pooled_seed*_B*.json`. Ở budget 0,50, FWR held của seed 202 là 0,50056; calibration không bảo đảm budget trên speaker mới. FWR chung thấp tại 0,20 đi kèm recall gần 0, không phải bằng chứng đủ chất lượng sử dụng.

PER breakdown S/D/I: seed 101 = **6.593 / 17.397 / 429**, seed 202 = **6.524 / 17.163 / 438** trên 39.477 reference phone. Recognizer xóa nhiều âm; learning curve fit vẫn giảm khi hết 8 epoch. Chưa tách được tác động của ít dữ liệu, selection bias, số update, capacity head hoặc frozen representation. Kết quả chỉ áp dụng cho recipe đã chạy, không bác bỏ mọi fine-tune hay realized-phone learning.

Decoder hiện là greedy CTC rồi edit xác định, tie diagonal → O → I. Score event chỉ là 0/1 của đường edit nên xếp hạng thô; policy phải xử lý nhiều tie bằng event hash. Đây không phải posterior event được hiệu chuẩn. Không có beam/lattice uncertainty mới để tuyên bố cải thiện định vị. Chưa có dữ liệu speaker mới xác nhận sau freeze.

## Kiểm tra trước/sau train

- Cache full-utterance H6 khớp hash/order manifest TRAIN, hash WAV audit, dtype fp16, shape, frame clock và finite values cho đủ 1.800 câu.
- Trích lại từ hai waveform float32: max absolute difference khoảng 0,000966, RMSE khoảng 0,0000543 so cache; đạt tolerance đăng ký trước. Không dùng gold timing/realized label làm feature.
- Smoke đủ 20 update/24 câu qua train → predict → calibration → evaluate → save → reload. Smoke có lớp không có support và metric null; chỉ kiểm tra plumbing, không dùng làm bằng chứng hiệu quả.
- Replay toàn bộ 8 receipt/checkpoint/artifact hash; loại trừ speaker; dựng lại toàn bộ policy từ calibration decodes; kiểm tra budget và policy khớp chính xác.
- Gộp held đúng một lần/câu/seed, EVAL-V3.1 replay khớp; coverage target/gap 100%, không thiếu event hoặc simplex violation. PER được tính lại bằng implementation Levenshtein lịch sử và khớp S/D/I/N.
- **29 test pass** cho contract, CTC/edit, evaluator và perturb reference. Thay reference trên cùng feature giữ nguyên nhận diện âm học; topology chẩn đoán thay theo reference.
- CLI inference thật từ WAV + reference phone đã chạy trên `ABA_arctic_a0003`: output 39 target + 40 gap; recognized sequence khớp đường cache ở probe này. Đây là utterance thuộc held fold 0, đã exposed development; không gọi nó là kiểm chứng speaker mới.

`verification.log`, `TESTS.log`, `preflight.json` và `waveform_inference_smoke.json` lưu bằng chứng. Không có independent agent review trong lượt này; replay dùng evaluator hiện có, còn PER dùng một implementation riêng.

## Chi phí và artifact

CUDA không khả dụng nên tất cả chạy CPU, 4 torch thread. Tổng thời gian 8 lượt train + predict/calibrate/evaluate/save là **56,38 giây**; preflight **5,29 giây** và smoke **0,24 giây**. Các bước tổng hợp/replay và kiểm thử nằm ngoài con số 56,38 giây. Không đo peak RAM; không có số VRAM train GPU để báo.

Đây là thời gian dùng cache encoder sẵn có. Full extraction lịch sử được receipt ghi khoảng **653,06 giây CPU**; không nhập nó thành extraction mới hoặc bỏ qua khi ước lượng pipeline từ waveform. Chỉ hai probe waveform được trích lại trong preflight.

Mỗi `seed{101,202}_fold{0..3}/` có:

- `checkpoint.pt`, `learning_curve.json`, `receipt.json`;
- `policy_B*.json`, keyed target/gap predictions và classwise/per-speaker metrics;
- `realized_predictions.jsonl` và `posteriors.npz`, giữ emission spans và frame clock. Emission span không phải phone boundary chắc chắn.

`SUMMARY.json`, `RUNS.csv`, `CLASSWISE.csv`, `PHONE_GROUPS_B0.2.csv` và `pooled_seed*_per_speaker.json` tổng hợp toàn chiến dịch. `artifact_hashes.json` niêm snapshot kết quả và source mới. Không chọn seed/fold tốt nhất làm đại diện; checkpoint dùng trong ví dụ là seed101/fold0 theo thứ tự cố định.

## Tái chạy và inference

Đăng ký trước: [PREREGISTRATION.md](PREREGISTRATION.md). Các lệnh đã chạy:

```bash
.venv/bin/python -m research.realized_phone_pipeline.train --mode preflight
.venv/bin/python -m research.realized_phone_pipeline.train --mode smoke
.venv/bin/python -m research.realized_phone_pipeline.train --mode train
.venv/bin/python -m research.realized_phone_pipeline.summarize_training
.venv/bin/python -m pytest -q research/realized_phone_pipeline/test_contract.py research/realized_phone_pipeline/test_train.py research/evaluation_v3/test_evaluator_v3.py
```

Runner từ chối ghi đè run đã hoàn tất; để tái chạy train cần đổi namespace có version thay vì xóa kết quả. Preflight và summary có thể chạy lại, nhưng sẽ đổi snapshot hash và cần niêm lại. Chưa có chức năng resume giữa epoch; receipt chỉ ghi COMPLETE sau khi save/evaluate xong.

Inference nhận WAV mono 16 kHz và **chuỗi phone reference được cung cấp**, không nhận gold label/boundary:

```bash
.venv/bin/python -m research.realized_phone_pipeline.infer \
  --run-dir experiments/realized_phone_pipeline_20261113/m1/seed101_fold0 \
  --audio data/derived/l2arctic/wavs_16k/ABA_arctic_a0003.wav \
  --reference 'F AO R DH AH T W EH N T IY IH TH T AY M DH AE T IY V N IH NG DH AH T UW M EH N SH UH K HH AE N D Z' \
  --speaker-id ABA --utterance-id ABA_arctic_a0003 \
  --output /tmp/langfens_m1_prediction.json
```

G2P từ text và reference-variant parity chưa hoàn thiện; không trình bày CLI này thành sản phẩm nhận text bất kỳ đã được kiểm chứng.

## Quyết định

Giữ nguyên model/policy lịch sử. Không promote baseline mới. Đã có checkpoint và inference để tái lập, nhưng chưa có bằng chứng âm học đủ tốt cho M2 hoặc lợi ích chẩn đoán cho M3. Bước tiếp theo cần cải thiện/đối chứng recognizer và budget học trước khi mở rộng decoder; giữ ablation dữ liệu, update, capacity và supervision để biết thành phần nào có tác dụng. Không tự chạy tiếp sweep sau kết quả này.
