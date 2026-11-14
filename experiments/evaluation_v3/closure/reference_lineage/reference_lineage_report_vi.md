# Kiểm toán lineage tham chiếu/toàn bộ đặc trưng

## Phạm vi và phương pháp
Đã duyệt độc lập 2699 bản ghi TRAIN/DEV_CAL/DEV_SEARCH từ `prompt_text`, dùng `tools/lexicon.py` và `data/original_sources/cmudict/cmudict.dict` local, không tải mạng, không inference. Kết quả từng utterance ở `reference_lineage.csv`; bảng split-speaker ở `speaker_split_mismatches.csv`; số liệu máy đọc ở `reference_lineage.json`.

Lexicon chuẩn hoá theo triển khai thực tế: regex `[A-Za-z]+(?:'[A-Za-z]+)?`, tra cứu uppercase và bỏ ký tự ngoài ASCII A-Z; biến thể CMUDict `(n)` được bỏ, lấy cách phát âm đầu tiên; stress digit bị xoá. Hàm sentence_to_phones bỏ qua token OOV âm thầm, nên báo cáo này ghi lại OOV thay vì xem chuỗi thiếu là so sánh hợp lệ. Không tự sửa dấu nháy/chính tả hoặc thêm G2P.

## Kết quả
Toàn bộ: **2699** utterances; **830** chuỗi phone quan sát được khớp (trong đó **830** không có OOV; khớp khi có OOV chỉ là so sánh sau khi lexicon bỏ âm thầm OOV, không phải bằng chứng sequence đầy đủ); **2232** cùng độ dài (**467** khác độ dài); **51** utterances có OOV, tổng **51** token OOV (8 dạng). Manifest có 89360 phone tokens, CMUDict prompt có 89118 phone tokens. Inventory chỉ có trong manifest: `D_, ER), SPN, Y_, Z_`; chỉ có trong prompt: `không có`.

| Split | Utterances | Exact sequence | Same length | Length mismatch | Có OOV / token OOV |
|---|---:|---:|---:|---:|---:|
| TRAIN | 1800 | 555 | 1504 | 296 | 34 / 34 |
| DEV_CAL | 300 | 97 | 254 | 46 | 7 / 7 |
| DEV_SEARCH | 599 | 178 | 474 | 125 | 10 / 10 |

Số liệu từng speaker/split có trong CSV; không gộp các mismatch thành lỗi âm vị học, vì CMUDict first-pronunciation là một tham chiếu chính tả độc lập, còn manifest theo annotation và có thể chứa biến thể dự kiến. Exact mismatch không tự nó chứng minh leakage.

## X và Y / kết luận validity
Trong `tools/extract_acoustic_features.py`, chuỗi `target_phone` của manifest được dùng làm điều kiện Viterbi alignment và one-hot X. Waveform/model logits/probabilities tạo acoustic X. `label`/`has_insertion` chỉ tạo `target_Y`/`gap_Y`; `xmin`, `xmax`, `realized_phone` không dùng trong X/alignment. Vì vậy X không dùng trực tiếp timing gold, realized phones hoặc error label theo luồng đã đọc. Nhưng intended phone sequence là annotation-derived canonical reference. Đánh giá non-oracle chỉ hợp lệ khi caller-supplied intended reference được cho phép; không thể khẳng định sequence ấy độc lập annotation.

`research/student/infer_student.py` không đọc gold fields, nhưng target empty-frame competitor margin (dimension46) là **-10** thay vì **-15** trong cache cũ; silence margin (dimension47) đã là **-15** và khớp; gap empty mặc định khớp. Bounded fix parent: thống nhất đúng công thức/giá trị empty-frame trước replay và regression test so sánh X. Không sửa canonical labels/IDs.

## Kế hoạch waveform an toàn
1. Replay cached canonical mode với đúng chuỗi manifest do caller cung cấp; kiểm tra X tái lập so với cache cùng WAV/model/tokenizer/preprocess/normalization sau khi xử lý lệch margin. Đây chỉ xác nhận feature reproducibility, không xác nhận reference độc lập.
2. Chạy riêng orthographic mode `prompt_text → local CMUDict`, ghi OOV và N/N+1 riêng; không gán lại phone/gap đầu ra lên canonical IDs khi số event khác. Tuyệt đối không truyền `xmin/xmax`, `realized_phone`, label hay TextGrid vào inference. Không thay canonical gold/IDs bằng kết quả mode này.

**Phân loại:** dữ liệu tham chiếu full-population: `ANNOTATION_DERIVED_CANONICAL_REFERENCE`; X-vs-gold timing/realization/label: `NO_DIRECT_FEATURE_LEAKAGE_IN_INSPECTED_EXTRACTOR`; độc lập lineage intended sequence: `UNVERIFIED / contract-dependent`; waveform replay full population chưa được thực hiện tại lane audit này.
