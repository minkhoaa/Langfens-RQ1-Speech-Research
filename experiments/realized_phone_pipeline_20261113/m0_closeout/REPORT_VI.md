# LANGFENS — báo cáo tổng hợp và hoàn tất M0

**Ngày:** 2026-11-13. **Trạng thái: M0 hoàn tất cho định nghĩa và audit nghiên cứu theo GOAL §8–9.** Đã chốt schema, inventory/provenance, policy nhãn bất định và reference, reference matrix, split/protocol, cổng nghiên cứu và đề xuất nghiệm thu có kiểm tra tính khả thi. **Chưa nghiệm thu sản phẩm, chưa có human gold mới, chưa promote model hoặc hoàn thành M1.**

Quyết định: dùng policy conservative có version cho các nghiên cứu tiếp theo; giữ full event population để đánh giá; hoàn thiện reproduction và đối chứng M1 trước khi train thêm cấu hình. Kết quả baseline M1 đã chạy được giữ như một kết quả âm của recipe cụ thể. M0 không thất bại vì model yếu; trước closeout này M0 còn thiếu một số quyết định và phép audit, nay đã được thực hiện và kiểm tra.

## 1. Bảng hoàn tất M0 theo đúng GOAL

| Yêu cầu | Artifact có thể kiểm tra | Kết quả |
|---|---|---|
| Schema C/S/O/I, N target/N+1 gap, ID và uncertainty | Contract v1 + `locked_policy.json`, `train_event_policy.jsonl` | Hoàn tất; không gộp bốn lớp vào một softmax |
| Inventory vật lý, nhãn, speaker/L1, nguồn/licence/hash và phần thiếu | M0 inventories + `data_registry_overlay.json`, `waveform_quality.jsonl` | Hoàn tất trong phạm vi nguồn đã nêu; missing metadata ghi rõ |
| Chính sách deviation/unknown, silence, hesitation và biến thể hợp lệ | `ANNOTATION_AND_REFERENCE_POLICY_V2.md`, code `m0_policy.py` | Đã chốt phạm vi nghiên cứu, không đoán nhãn |
| Reference matrix, metric/domain equivalence và phương án reproduce | `REFERENCE_MATRIX.json` + matrix M0 ban đầu | Đủ trường; chỗ chưa biết được ghi là unverified và đặt gate M1 |
| Protocol và manifest split | `locked_policy.json`, `../m0/folds.json` | Giữ TRAIN 6 fit/3 cal/3 held, seeds 101/202; không đổi holdout đã exposed thành test mới |
| Research targets và acceptance targets được phân biệt, có feasibility | `ACCEPTANCE_AND_HANDOFF.md`, `acceptance_feasibility.csv` | Research gate đã đăng ký; product proposal chưa được user đồng ý |
| Tái lập và kiểm tra | `completion.json`, `TESTS.log`, `artifact_hashes.json`, verifier | 42 test pass; input/output và snapshot kiểm tra được |

Cổng M0 là **hiểu nhãn nào có thật, thiếu gì và phép so sánh nào hợp lệ**. Nó không yêu cầu mọi nhãn còn thiếu đã được con người tạo mới hoặc model đã tốt. Yêu cầu acceptance trong GOAL là đề xuất và tách rõ trạng thái; không tự đổi proposal thành agreement. Việc tôi trước đây coi agreement sản phẩm là lý do phải để M0 mở là chưa đúng phạm vi này.

## 2. Bằng chứng vật lý và nguồn dữ liệu

Đã kiểm tra lại hash **7.944 WAV**: 2.699 L2 thuộc TRAIN/DEV_SEARCH/DEV_CAL và 5.245 SpeechOcean. Đã đọc waveform để QA: **không có NaN/Inf, không có waveform toàn zero**, không có nhóm trùng byte SHA-256 trong phạm vi này. Đây không phải kiểm tra người nghe hoặc bằng chứng không có hai bản encode khác nhau của cùng nội dung. Peak và tỷ lệ sample biên độ lớn được giữ như QA, không được biến thành nhãn clipping hay O.

**12 archive L2** có hash khớp receipt. Chúng có 13.291 WAV member; 1.800 annotation tương ứng phần TRAIN hiện có và 11.491 recording không có manual annotation tương ứng. Đây là thêm recording của cùng 12 speaker, không phải thêm speaker mới hay thêm gold lỗi.

| Population | Speaker | Câu | C / S / O | Gap / gap có I |
|---|---:|---:|---|---:|
| TRAIN | 12 | 1.800 | 50.356 / 7.381 / 1.825 | 61.362 / 560 |
| DEV_SEARCH, exposed | 4 | 599 | 17.778 / 1.768 / 209 | 20.354 / 125 |
| DEV_CAL, dùng lịch sử | 2 | 300 | 7.800 / 1.781 / 462 | 10.343 / 102 |

TRAIN có **620 âm chèn** trong 560 gap: giữ riêng count phone và count event. M0 đã đối chiếu raw annotation với manifest cho đủ 1.800 câu; closeout kiểm tra lại nguồn/hash và bảo toàn các event này. Không mở manifest hoặc audio TEST/DEV_REPLICATION.

L2 WAV thực tế là **float32, 16 kHz, mono**, khác mô tả PCM16 cũ. Recording condition và lịch adjudication chi tiết không có trong manifest, nên không điền suy đoán. L1/support theo speaker nằm ở `../m0/speaker_support.csv`.

Overlay mới ghi L2 licence theo [publisher](https://psi.engr.tamu.edu/l2-arctic-corpus/) là CC-BY-NC-4.0, tách khỏi assertion CC-BY-NC-SA trong registry cũ. [Publisher card SpeechOcean](https://huggingface.co/datasets/mispeech/speechocean762) ghi Apache-2.0; việc ghép licence text với archive local chưa được xác nhận, không tự biến publisher claim thành chứng chỉ archive. CMUDict pin nội dung bằng checksum; nhãn version “0.7b” từ registry cũ chưa được xác thực. Registry lịch sử không bị viết đè.

## 3. Nhãn bất định và biến thể: quyết định đã chốt

Policy `REALIZED-MDD-M0-CLOSEOUT-v2` giữ nguyên class nguồn C/S/O và các âm I có thứ tự. O không phát token; missing/unknown không phải O. ERR/SPN và các symbol malformed không được gán thành âm thường. Giữ raw `*`, candidate và lý do uncertainty; không tự xem base candidate là realized phone chắc chắn.

Đã xuất ledger cho đủ 1.800 câu với từng raw event, phone status và mask. Có **1.112 realized token không exact** theo policy. Primary complete-sequence CTC vẫn chỉ có **1.243 câu**, khớp subset M0/M1 cũ. Mask token là dữ liệu audit; không xóa token chưa biết để tạo nhãn CTC thường vì như vậy sẽ ngầm dạy model bỏ âm.

**Phát hiện quan trọng:** 837/7.381 S (**11,34%**) có CPL/PPL trùng base symbol; 836 trường hợp có deviation marker. Không trường hợp nào trong 837 này nằm trong strict eligible subset. Điều này cho thấy một phần lỗi nguồn không biểu diễn bằng thay đổi phone cơ sở và bị loại khỏi primary exact-sequence training. Nó không chứng minh sai annotation, không chứng minh nguyên nhân duy nhất của baseline yếu, và không cho phép đổi S thành C.

Strict filtering cũng làm lệch thành phần speaker: ví dụ ABA còn 48 câu, ASI 49, HJK 148. Các arm so sánh phải giữ cùng fit subset/budget khi cần tách tác động kiến trúc. **Đánh giá MDD vẫn đủ 1.800 câu**, kể cả 557 câu không có complete exact sequence; PER phải báo mask coverage và support riêng.

Biến thể hợp lệ được định nghĩa theo profile, không dựa vào model prediction:

- Corpus profile giữ nhãn nguồn và event topology lịch sử, không sửa gold bằng dictionary.
- Supplied-reference profile dùng token identity trong inventory, không tự alias AX/AH hoặc AO/AA và không tự hợp thức hóa lược/nối âm.
- Text-only research profile dùng CMUDict pinned, pronunciation đầu tiên xác định; giữ apostrophe, từ chối OOV/numeral thay vì bỏ token. Các variant còn lại là candidate chưa được xác nhận ngữ cảnh.
- Natural/accent variants ngoài phạm vi này cần danh sách xác định trước và human audit trong các mốc dữ liệu/xác nhận. Phạm vi hẹp hiện tại không được quảng bá như đã giữ mọi phát âm hợp lệ cho sản phẩm.

Đây là policy thực thi được cho nghiên cứu hiện tại, không phải khẳng định mọi khác biệt với canonical đều sai. Chi tiết và điều kiện đổi version nằm trong [policy guide](ANNOTATION_AND_REFERENCE_POLICY_V2.md).

## 4. Reference từ text: đã đo khoảng trống thay vì để chung chung

Audit đủ 1.800 utterance instance TRAIN, tương ứng **300 prompt ID**, với dictionary checksum cố định:

| Phép kiểm tra | Số câu / 1.800 |
|---|---:|
| Tất cả token có trong dictionary | 1.774 |
| Pronunciation đầu tiên khớp reference annotation | 573 (31,83%) |
| Một cách chọn các lexical variant có thể khớp annotation | 1.534 (85,22%) |
| Công cụ cũ có token bị bỏ âm thầm | 34 |
| Prompt có apostrophe | 116 |

26 câu có OOV trong policy mới: `PROMOTER'S` 12, `NIGHTGLOW` 12, `TOMFOOLERY` 2 occurrence. Các occurrence theo speaker không phải các từ mới độc lập. Hàm mới trả trạng thái explicit-reference-required và không trả chuỗi một phần làm reference hoàn chỉnh.

Best-variant match dùng annotated reference để chọn, nên là **oracle diagnostic**. 1.534 match không phải độ chính xác text-only inference. Không chọn variant bằng gold đang đánh giá, không remap event hoặc đổi denominator theo cách chọn đó. M0 đã xác định được vấn đề và chốt policy; giải quyết reference inference parity là M1, không được tuyên bố đã đạt.

## 5. SpeechOcean: sửa kết luận cũ và audit nhiều annotator

Vật lý có 5.245 WAV/250 speaker directory, nhưng chỉ **5.000 câu có score**; 245 audio không có score giữ riêng. Metadata chi tiết có **31.816 word record × 5 notation = 159.080 annotator-word record**.

- 12.658 word record khác nhau ở ít nhất một chuỗi notation, khoảng **39,79%**. Đây là literal string disagreement, không phải tỉ lệ bất đồng gold C/S/O/I hoặc Cohen's kappa.
- Có **2.653 dấu bracket insertion** qua các bản annotation, xuất hiện ở 1.777 word record. Số dấu qua annotator không phải số insertion event độc lập.
- 13 word record có năm danh sách insertion candidate giống nhau và không rỗng; chưa xác nhận vị trí gap hoặc adjudicate âm học từ đó.
- `(phone)` biểu diễn score 0, vẫn gộp incorrect và missed; không tự tách thành S/O. `[phone]` có thông tin âm chèn, nên cách mô tả cũ “chỉ có score” là quá rộng.

Kết luận sử dụng: có thể dùng quality scoring đúng schema và nghiên cứu uncertainty/candidate I; **chưa dùng như complete realized-phone/exact-gap gold** nếu chưa có adapter được kiểm tra và expert audit. Source notations và các case có disagreement/bracket được xuất ở `speechocean_multiannotator_audit.jsonl`. Không tạo nhãn người mới hoặc train SpeechOcean trong closeout này.

L2 queue 641 câu vẫn chờ human review. Điều đã hoàn tất ở M0 là audit, quyết định thiếu nhãn nào và cách không sử dụng sai nhãn; việc nghe/gán nhãn mới thuộc M5/data work và phải được báo riêng.

## 6. Reference và những gì chưa được reproduce

`REFERENCE_MATRIX.json` có đủ task, labels, corpus, speaker split, inference input, boundary oracle, S/O/I, metrics, code/checkpoint, licence, compute và EVAL-V3 mapping cho Qian 2016, Shahin 2024, CROTTC-IF 2026 và Microsoft product context. Matrix ban đầu và các báo cáo cũ vẫn được giữ làm nguồn.

M1 priority là kiểm tra recipe sequence-MDD công khai phù hợp, pin source/training provenance rồi smoke/profile. Code IF-MDD có source/recipe công khai; commit và checkpoint-speaker exclusion chưa pin local nên chưa được dùng như một reproduction sạch. Metrics paper khác EVAL-V3.1 không được xếp hạng trực tiếp. Unknown về artifact/compute được ghi rõ, không đổi thành “không tồn tại”.

Đã đọc lại các negative result về CTC/E4, learned alignment, focal, auxiliary/crop head và attribute adaptation trong matrix M0. Không thiết kế lại chỉ bằng cách đổi tên một nhánh cũ. Baseline frozen H6 vừa train **không phải** original-reference reproduction.

## 7. Tiêu chí đã chốt và tính khả thi

Research gate: tăng Macro-3 AP tuyệt đối ≥0,02 trên matched control, hoặc recall descriptive tại precision ≥0,75 tăng ≥0,05; hướng lặp ở seeds 101/202, không class AP giảm quá 0,01, báo support/delta đủ speaker. Đây là ngưỡng nghiên cứu do agent đăng ký cho tương lai, không là product acceptance hoặc bằng chứng thống kê. Operating points B=0,10/0,20/0,50 được giữ; fit threshold trên cal và áp dụng held nguyên trạng.

**Product proposal chưa được user đồng ý:** precision ≥95%, recall ≥50% riêng S/O/I, joint FWR ≤0,20/câu và đủ event coverage; đo đồng thời trên fresh speakers sau freeze. Với prevalence TRAIN, tại đúng P=0,95/R=0,50 sizing FWR≈0,14278/câu. Tại R=0,80 với P=0,95, FWR≈0,22844 nên vẫn phải kiểm tra ràng buộc FWR riêng. Đại số khả thi không chứng minh model đạt được hoặc bảo đảm prevalence speaker mới.

Không cần giả mạo user agreement để hoàn tất việc đề xuất ở M0. Thỏa thuận nghiệm thu phải có trước M6. Chi tiết cổng, ablation và handoff ở [ACCEPTANCE_AND_HANDOFF.md](ACCEPTANCE_AND_HANDOFF.md).

## 8. Tổng hợp với kết quả train và quyết định tiếp theo

Baseline trước closeout đã train 4 fold × 2 seed × 8 epoch; frozen W2V2-H6 và head CTC 31.529 tham số. Pooled held PER 61,86% / 61,11%; Macro-3 AP 0,06728 / 0,06748; recall S/O/I dưới 0,6%. Giữ kết luận **không promote**, không đổi checkpoint lịch sử. Không so trực tiếp với SM13 để suy ra tác động nhân quả của strict policy vì dữ liệu và budget khác.

| Hướng tiếp theo | Bằng chứng ủng hộ | Hạn chế/chi phí | Quyết định |
|---|---|---|---|
| Hoàn thiện reference reproduction và matched controls M1 | Có code/recipe, cache và audit; hiện thiếu phép so sánh tách yếu tố | Cần pin provenance/fidelity và budget trước run | Ưu tiên trước train thêm |
| Adjudication và mở rộng speaker song song theo M5 | Có ambiguity ledger, queue và metadata đa annotator | Cần người đánh giá; chưa có gold/speaker xác nhận mới | Chuẩn bị dữ liệu, không giả lập human label |
| Đầu tư ngay large backbone/decoder/attribute sweep | Có giả thuyết nghiên cứu trong GOAL | Recognizer hiện yếu; GPU không khả dụng ở phiên train; chưa có causal diagnosis | Chưa mở sweep; profile/đối chứng trước |

**Độ chắc chắn:** cao cho count/hash/parity có thể tái chạy; trung bình cho việc chọn policy conservative là cách khởi đầu thích hợp vì nó mất coverage và có selection bias; chưa biết hiệu quả model tương lai. Không có bằng chứng một nguyên nhân duy nhất giải thích train yếu.

Điều làm thay đổi quyết định: human adjudication chứng minh cách giữ thêm dữ liệu bất định an toàn; reference selector được kiểm chứng ngữ cảnh độc lập annotation; hoặc reproduction/đối chứng mới cho thấy lợi ích recognizer chuyển sang đủ S/O/I. Khi đó phải version policy và đăng ký ablation mới, không viết lại kết quả M1 cũ.

## 9. Nguồn và tính độc lập

| ID | Nguồn | Phạm vi và chất lượng |
|---|---|---|
| S1 | GOAL §8–9, snapshot trước closeout | Hợp đồng công việc và ranh giới nghiệm thu |
| S2 | M0 inventories, source archive receipts; waveform QA closeout | Count/hash của 7.944 WAV + 12 archive; không là human validation |
| S3 | 1.800 raw label lists/parity + event ledger v2 | Định nghĩa/uncertainty theo nguồn; chưa có re-annotation |
| S4 | CMUDict pinned + `train_reference_parity.jsonl` | Deterministic/oracle diagnostics trên 300 prompt/1.800 instance |
| S5 | SpeechOcean local README và scores-detail hash | Năm source notation/từ; chưa ánh xạ đầy đủ C/S/O/I |
| S6 | M1 SUMMARY, keyed predictions, policies/receipts | 8 run; cùng speaker qua seed, không là 8 mẫu độc lập |
| S7 | Primary sources trong REFERENCE_MATRIX và publisher links ở trên | Method/metadata nguồn, không phải local reproduction |

Các insight về coverage, reference mismatch, annotation semantics và baseline yếu được nối với S2–S6; diễn giải về bước tiếp theo là đề xuất, không phải kết quả thử nghiệm đã chạy. Không có independent agent review trong lượt này. Số liệu mới được tính bằng code audit; verifier tách khỏi bước tạo artifact kiểm tra tính nhất quán và hash.

## 10. Snapshot và tái lập

M0/M1 cũ được giữ nguyên. Vì GOAL là tài liệu sống, bản trước closeout được lưu ở `input_snapshots/GOAL_before_closeout.md`, hash khớp manifest M1. `historical_snapshot_resolution.json` ghi mapping cụ thể này; mọi đường dẫn khác vẫn phải khớp hash cũ. Không re-seal artifact cũ để che thay đổi. **218 entry snapshot lịch sử** được kiểm tra.

```bash
# Build lại audit v2; không train và không sửa M0/M1 cũ
.venv/bin/python -m research.realized_phone_pipeline.close_m0

# Kiểm thử
.venv/bin/python -m pytest -q research/realized_phone_pipeline/test_m0_policy.py research/realized_phone_pipeline/test_contract.py research/realized_phone_pipeline/test_train.py research/evaluation_v3/test_evaluator_v3.py

# Kiểm tra snapshot đã niêm; rebuild sau thay đổi cần niêm lại hash v2
.venv/bin/python -m research.realized_phone_pipeline.close_m0 --verify
```

Code, policy, dữ liệu audit, tài liệu và GOAL hiện hành được pin ở `artifact_hashes.json` của closeout. Xem `completion.json` để đọc trạng thái máy. M0 được đánh dấu hoàn thành; M1 reference reproduction, M2–M6 và goal tổng thể vẫn chưa hoàn thành.
