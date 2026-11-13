# LANGFENS — triển khai và audit M0, 2026-11-13

**Đã chạy audit thực tế, chưa train model mới.** Có công cụ tái chạy, inventory/hash, đối chiếu raw annotation, export supervision có uncertainty, contract inference/prediction, split manifest, reference matrix và protocol M1. Cổng M0 đầy đủ vẫn mở vì policy nhãn bất định, biến thể reference và acceptance sản phẩm chưa được giải quyết. M1–M6 chưa hoàn thành; chưa có model được promote.

## 1. Dữ liệu thực có và provenance

Số liệu dưới đây được tính lại từ file, không sao chép quy mô công bố của corpus. `audio_inventory.jsonl` lưu SHA-256, kích thước, subtype, sample rate, channel và frame count cho từng WAV thuộc ba partition đang dùng. Không mở manifest/audio TEST hay DEV_REPLICATION.

| Partition | Speaker | Câu | Target C/S/O | Gap / gap có I | Thời lượng audio |
|---|---:|---:|---|---:|---:|
| TRAIN | 12 | 1.800 | 50.356 / 7.381 / 1.825 | 61.362 / 560 | 6.392,314 s |
| DEV_SEARCH, đã exposed | 4 | 599 | 17.778 / 1.768 / 209 | 20.354 / 125 | 2.239,232 s |
| DEV_CAL, đã dùng lịch sử | 2 | 300 | 7.800 / 1.781 / 462 | 10.343 / 102 | 1.193,071 s |

TRAIN có 620 âm chèn trong 560 gap: 507 gap một âm, 46 gap hai âm, 7 gap ba âm; 3 gap đầu câu và 23 gap cuối câu có I. Không thiếu realized_phone ở target khác O về mặt trường dữ liệu, nhưng trường có giá trị không đồng nghĩa đã biết chắc âm thực tế.

Không trùng speaker giữa ba partition; không có WAV trùng SHA-256 trong 2.699 câu. Đây là kiểm tra trùng nội dung byte, không chứng minh không có hai bản encode khác nhau của cùng một recording. `speaker_support.csv` có L1 và support; recording condition không có trong manifest nên ghi thiếu.

12 speaker archive của campaign trước đang có vật lý trong `experiments/sm13_three_directions_20261113/data_expansion/raw_archives/`. Audit kiểm tra lại SHA-256 khớp receipt, kiểm kê 13.291 WAV member và 1.800 manual annotation member; 11.491 câu không có manual annotation tương ứng. Chúng không tạo speaker mới và không được mặc định là phát âm đúng. `source_archives.json` giữ số lượng từng speaker và checksum. Không giải nén thêm hoặc dùng các câu đó để train trong M0.

SpeechOcean đang có **5.245 WAV / 250 thư mục speaker**, còn scores.json có **5.000 câu** (2.500 train / 2.500 test), tất cả có audio; **245 WAV không có score**. Speaker train/test theo metadata không giao nhau. Inventory giữ riêng audio vật lý và metadata score. Nguồn này chỉ thích hợp cho auxiliary quality scoring theo schema hiện có; chưa có nhãn xác nhận S/O hay realized sequence trong phần audit. Không dùng dữ liệu này để train ở M0. CMUDict có checksum khớp registry, chỉ là reference lexicon.

`data/raw_registry.jsonl` và tài liệu migration cũ chưa phản ánh SpeechOcean và các archive mới xuất hiện. WAV L2 hiện là **16 kHz, mono, IEEE float32** cho cả 2.699 câu, khác mô tả PCM16 cũ. Đây là khác biệt cần kiểm soát khi reproduce cache; audit không chuyển đổi/ghi đè waveform. M1 phải đọc float32 đúng scale, kiểm tra normalization và parity thay vì chia lại theo thang int16 hoặc tự roundtrip không đăng ký.

Licence registry L2 ghi CC-BY-NC-SA-4.0, trong khi [trang publisher](https://psi.engr.tamu.edu/l2-arctic-corpus/) và licence receipt archive ghi CC-BY-NC-4.0. Giữ nguyên registry lịch sử và ghi bất đồng; dùng nguồn publisher làm tài liệu đối chiếu, không suy ra quyền thương mại. Licence SpeechOcean và checkpoint cần pin cùng nguồn trong M1; sự hiện diện trên đĩa không tự xác nhận mọi quyền sử dụng.

## 2. Nhãn raw và tập supervision dùng được

Đã đọc annotation trong 12 archive TRAIN và parse bằng code nghiêm ngặt: **1.800/1.800 câu khớp toàn bộ target/gap/time với manifest**, không có mismatch. Kết quả này xác nhận phép biến đổi event, không thay thế đánh giá chất lượng annotation bởi con người.

Parser lịch sử đã bỏ dấu `*` đánh dấu deviation. Trong TRAIN có 462 câu mang dấu này ở raw label. Các realized target còn có `ERR` 140 lần, và mỗi ký hiệu `ER)`, `D_`, `SPN`, `Z_` một lần. CTC lịch sử nhận ERR như một lớp phone và ánh xạ một số ký hiệu khác về phone chuẩn; M0 không tự khẳng định đó là nhãn phone chính xác.

Policy strict hiện tại giữ **1.243/1.800 câu (69,06%)** đủ điều kiện exact sequence sau raw parity và kiểm tra uncertainty. **557 câu** còn lại vẫn giữ toàn bộ observed tokens, raw labels, lý do và source member/hash; `exact_phone_sequence=null`. Các nhóm issue có giao nhau; không cộng chúng để suy ra số câu độc lập. `ERR` không được suy đoán là phone nào; deviation chưa bị ép về nhãn chắc chắn.

| Subset supervision | Câu | C | S | O | Gap I |
|---|---:|---:|---:|---:|---:|
| TRAIN đầy đủ | 1.800 | 50.356 | 7.381 | 1.825 | 560 |
| Strict exact sequence | 1.243 | 34.592 | 4.480 | 1.352 | 361 |

**Strict filtering có selection bias rõ:** chỉ còn 48 câu ABA và 49 câu ASI nhưng 148 câu HJK. Không được train trên subset rồi chỉ đánh giá các câu dễ còn lại, hoặc gán gain cho kiến trúc khi dữ liệu/label exposure đã đổi. `strict_subset_support.csv` cho phép kiểm tra support theo speaker trước khi chọn recipe.

`annotation_review_queue.csv` có **641 câu chờ người đánh giá**, gồm toàn bộ 557 câu có issue và 120 câu lấy mẫu ngẫu nhiên phân tầng speaker (10/speaker, seed 101), có 36 câu giao nhau. Giữ cả chỉ báo random và lý do targeted để tránh diễn giải sai prevalence. Các cột reviewer/adjudication để trống. **Không tạo nhãn con người mới.**

## 3. Contract, reference và protocol đã triển khai

- `train_supervision.jsonl`: sequence theo thứ tự gap/target, token provenance và raw annotation; omission không phát token, missing/unknown khác omission.
- `train_inference_requests.jsonl`: allowlist audio + prompt + supplied reference cùng identity/provenance; không có nhãn lỗi, realized phone hoặc boundary.
- `contract.py`: kiểm tra N target/N+1 gap, ordered slots, stable ID, C/S/O simplex và gap probability/decision; từ chối probability không hữu hạn và ID sai.
- `folds.json`: 4 fold đúng nhóm cũ, 6 fit / 3 cal / 3 held, danh sách utterance và hash TRAIN. Không load feature cache hoặc nạp test để dựng split.
- [Reference matrix](REFERENCE_MATRIX.md): đọc nguồn gốc Qian 2016, Shahin 2024, CROTTC-IF 2026 và Microsoft; chỉ rõ chỗ khác metric, data, model, compute và exact-gap. Chưa reproduce paper.
- [Protocol/M1](PROTOCOL_AND_M1.md): operating points đăng ký trước, research gate đề xuất, budget smoke/profile và các đối chứng cần có. Không dùng threshold chọn trên held labels làm policy triển khai.

Reference lịch sử lấy từ annotation và chưa được chứng minh độc lập prompt. Allowlist inference không giải quyết được vấn đề này: M1 phải kiểm tra prompt→reference parity riêng trước khi tuyên bố pipeline đầy đủ từ text.

Với prevalence TRAIN, FWR 0,20/câu và recall gộp 50% yêu cầu precision gộp khoảng **93,1%**; precision 80% tại recall đó cho FWR khoảng **0,678/câu**. Bảng feasibility đi kèm tính cho nhiều operating point. Mục tiêu nghiên cứu đề xuất và acceptance sản phẩm được tách rõ; chưa có acceptance sản phẩm do người dùng chốt.

## 4. Kiểm tra và giới hạn

Kiểm thử contract cùng evaluator EVAL-V3.1: **24 test pass**. Các fixture kiểm tra omission so với missing, ký hiệu chưa xác định, deviation, insertion đầu/cuối và nhiều âm, phone lặp, topology, ID, probability không hữu hạn và tag raw không hợp lệ. `VALIDATION.log` lưu kết quả. Audit vật lý chạy thành công; script verification kiểm tra split coverage, source parity, output allowlist, strict mask và checksums.

Truy vấn NVIDIA thất bại; PyTorch `2.6.0+cu124` trả `cuda_available=false`, device_count=0 trong runtime hiện tại. Không có training/profile GPU hoặc chi phí VRAM mới để báo. Chi phí M0 là đọc metadata/hash, không phải benchmark inference. `environment.json` ghi môi trường và `RUN_LOG.md` ghi lệnh, kể cả lần audit đầu phát hiện WAV FLOAT.

Artifacts được tái tạo bởi audit và kiểm tra checksum, nhưng không có independent agent review trong lượt này. Kiểm tra evaluator sử dụng implementation hiện có và fixture; điều đó không đồng nghĩa ứng viên model đã được kiểm chứng độc lập.

## 5. Cổng M0 và việc cần làm tiếp

| Artifact/cổng | Trạng thái |
|---|---|
| Physical inventory, manifest/audio/source hash, support và provenance | Đã xuất trong phạm vi nguồn nêu trên |
| Schema C/S/O/I, raw-label preservation, missing/uncertainty và queue | Đã triển khai; chưa có adjudication mới |
| Reference matrix và audit các hướng cũ | Đã ghi; chưa pin/reproduce reference |
| TRAIN split manifest, operating points, nghiên cứu và budget M1 | Đã đăng ký |
| Policy cho deviation/unknown và biến thể phát âm hợp lệ | Chưa chốt bằng dữ liệu/annotation guideline đã kiểm chứng |
| Acceptance sản phẩm | Chưa được người dùng xác định; chỉ có đề xuất nghiên cứu |
| Hardware profile, full training smoke và original-reference reproduction | Chưa chạy; thuộc M1 |
| Dữ liệu speaker mới cho xác nhận sau freeze | Chưa có trong inventory này |

Bước thực thi tiếp theo là M1 theo protocol đã ghi: pin reference/code/checkpoint provenance, xác định policy supervision có đo coverage/bias, rồi smoke toàn tuyến và reproduce các baseline trên cùng split. Không chạy sweep lớn hoặc coi audit M0 là bằng chứng đã giải quyết C/S/O/I.

## Tái chạy

```bash
.venv/bin/python -m research.realized_phone_pipeline.audit
.venv/bin/python -m pytest -q research/realized_phone_pipeline/test_contract.py research/evaluation_v3/test_evaluator_v3.py
.venv/bin/python -m research.realized_phone_pipeline.verify
```

`artifact_hashes.json` kiểm tra snapshot artifact/source; audit tái chạy sẽ cập nhật hash, bao gồm tài liệu sẵn có trong thư mục. Không đổi dữ liệu gốc, evaluator, checkpoint/policy lịch sử hay các kết quả cũ.
