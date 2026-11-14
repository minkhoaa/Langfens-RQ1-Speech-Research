# Kết quả triển khai ba nhánh bằng Luna — 2026-11-13

**Đã triển khai và chạy pilot cả ba nhánh bằng ba agent `gpt-6-luna`.** Hai nhánh xuất được chuỗi phone để chấm. X-MDD đã chạy Qwen3-ASR thật và SFT thật, nhưng chưa tạo được đầu ra phoneme hợp lệ; không coi đó là MDD thành công.

Nhánh Git: `feat/three-method-local-pilot-20261113`. Không push, không merge, không đổi GOAL/protocol nghiệm thu hay dữ liệu cũ.

## Phạm vi cố định

Pilot chính dùng 24 FIT, 6 CAL và 6 HELD, tách speaker theo M0 fold 0 trong population TRAIN lịch sử. Audio/feature đã đối soát SHA-256 trước chạy. HELD là development, không phải test mới. Các query tối đa 5 giây. X-MDD dùng subset smoke riêng 2 FIT, 1 CAL, 1 HELD để giới hạn CPU/RAM.

Các agent triển khai song song; công việc GPU dùng lock chung. Không chạy ba job GPU cùng lúc trên GPU 4 GB. CROTTC và X chạy CPU, PER dùng GPU để encode. Budget được lưu trước thực thi; không truy cập official test hoặc stream data từ Drive.

## Kết quả thực thi

| Nhánh | Đã chạy thật | Kết quả và giới hạn chính |
|---|---|---|
| CROTTC-IF compact | Full-utterance monotone OT với learned frame mass; CR; decoder phone tự hồi quy; teacher canonical/error chỉ FIT truyền phân phối phone vào student; 2 epoch CPU | 24 FIT, CAL chọn epoch 1 theo **realized PER 1,0592**. Xuất 6 CAL/6 HELD; checkpoint đầy đủ trạng thái và reload. Là adaptation W2V2-H6/GRU/teacher nhỏ, không phải WavLM-Large + kiến trúc/lịch train nguyên paper. |
| PER-MDD HuBERT-large | Đúng backbone `facebook/hubert-large-ls960-ft`; pool midpoint frame theo phone thực tế FIT, cosine top-k 10, threshold 0,7, plurality, collapse/blank | 723 vector từ 24 FIT, 6 CAL/6 HELD. 0 optimizer updates, khoảng 7,11 giây. Pool nhỏ hơn 500 câu của recipe paper. Bản HuBERT-base trước đó cũng được giữ riêng. |
| X-MDD CPU LoRA | Qwen3-ASR-1.7B thật; canonical audio/phones và learner audio; 2 SFT updates; lấy mẫu group 2 và tính GRPO reward/reference/current | Hoàn tất đường chạy khoảng **85,64 giây**. SFT loss 3,2765 rồi 1,5777; gradient hữu hạn. Cả hai mẫu GRPO và 4 đầu ra greedy vẫn là **văn bản, không phải phone**. Reward `[0,0]`, advantage/gradient GRPO bằng 0: **không có optimizer update RL có tác dụng**. Không chấm SOI trên các đầu ra sai định dạng. |

X-MDD dùng decoder LoRA và audio tower đóng băng thay vì SFT gần toàn bộ weights; canonical speech dùng eSpeak phoneme input thay CosyVoice voice cloning. Đây là các khác biệt lớn với paper, không phải reproduction. Trọng số Qwen revision `7278e1e70fe206f11671096ffdd38061171dd6e5`; HuBERT-large revision `ece5fabbf034c1073acae96d5401b25be96709d8`. Trọng số được tải về cache local và kiểm hash, không đưa vào Git.

X-MDD model load ghi RSS khoảng 9,41 GB; sau SFT khoảng 10,91–11,44 GB; cuối chạy khoảng 12,00 GB. Đây là các mốc RSS, không dùng chúng để khẳng định một peak toàn tiến trình đã được đo độc lập. Budget RSS 16 GiB và 600 giây CPU được giữ trong runner. Không tạo thêm một bản sao đầy đủ của Qwen làm reference; dùng snapshot adapter SFT.

## Chấm điểm mô tả hai nhánh có đầu ra phone

Root dùng deterministic edit alignment và EVAL-V3.1 hiện có; không dùng gold timestamps khi query inference. Toàn bộ 6 HELD vẫn nằm trong mẫu số, không bỏ câu ineligible. Hard sequence không có continuous event confidence nên **AP không estimable**, không bịa xác suất.

| Nhánh | HELD Macro F1 S/O/I | HELD FWR/câu |
|---|---:|---:|
| CROTTC compact | 7,56% | 22,67 |
| PER HuBERT-base adaptation | 14,62% | 16,67 |
| PER HuBERT-large paper backbone | 8,74% | 21,00 |
| X-MDD | Không chấm: đầu ra sai inventory | Không chấm |

Đây không phải head-to-head công bằng về compute/data/recipe và không đủ để xếp hạng phương pháp. Sáu HELD chỉ có S/O/I support **22/2/1**; một sự kiện làm thay đổi recall rất lớn. PER realized chỉ estimable ở 3/6 HELD đủ exact sequence (80 realized phones); không thay mẫu số SOI bằng subset PER. Không so các F1 này với headline detection F1 của paper.

Các điểm pilot còn thấp và FWR cao; chưa có nhánh nào là nghiệm thu sản phẩm. Riêng X-MDD xác định được một blocker cụ thể: SFT smoke chưa chuyển được codec đầu ra từ transcript sang phone, khiến GRPO không có tín hiệu reward phân biệt. Không biến transcript thành phone bằng hậu xử lý rồi nhận đó là acoustic phone prediction.

## Kiểm tra và sửa trước bàn giao

19 tests đạt trong isolated Qwen runtime, gồm OT marginals/monotonicity/gradient, CR, teacher không lọt vào inference, length predictor có gradient, retrieval voting/blank/nonfinite/speaker isolation, full evaluation denominator, và objective/mask/detach của GRPO. Có test dùng **cùng class Qwen3-ASR chính thức với config nhỏ, random weights** để kiểm split hai audio và PEFT API; đó là test tương thích, không phải bằng chứng hiệu quả pretrained model.

Review đã phát hiện và sửa các lỗi có ảnh hưởng: CROTTC ban đầu thiếu decoder teacher transfer đúng vào phone logits, length predictor chưa được học, và chọn checkpoint theo canonical thay realized CAL; X có các lỗi package/device/validation/autograd snapshot. Những run CROTTC bị supersede được giữ nguyên ở thư mục riêng, không đưa vào bảng kết quả cuối. Không nhận việc test pass là chứng minh tái lập paper.

## Artifacts và lệnh

- Source: `research/three_method_pilot/{crottc_if,per_mdd,x_mdd}/`.
- Manifest, budgets, staging/integrity, checkpoints và raw outputs: `experiments/three_method_pilot_20261113/`.
- Tổng hợp có hash: `experiments/three_method_pilot_20261113/SUMMARY.json`, `VALIDATION.json`.
- Báo cáo từng nhánh: `CROTTC_IF.md`, `PER_MDD.md`, `X_MDD.md` trong thư mục này.

```sh
# CROTTC: CAL labels chỉ dùng chọn checkpoint, không vào query inference.
.venv/bin/python -m research.three_method_pilot.crottc_if.run \
  --pilot experiments/three_method_pilot_20261113/pilot.json \
  --cal-supervision experiments/three_method_pilot_20261113/cal_supervision.jsonl \
  --output experiments/three_method_pilot_20261113/crottc_if_rerun \
  --epochs 2 --max-fit 24 --threads 2

# X-MDD: isolated runtime giữ nguyên venv dự án; source Qwen chính thức ở /tmp.
PYTHONPATH=/tmp/langfens_qwen_runtime:/tmp/qwen3-asr-src \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m research.three_method_pilot.x_mdd.real_pilot

# Tests.
PYTHONPATH=/tmp/langfens_qwen_runtime:/tmp/qwen3-asr-src \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m pytest research/three_method_pilot -q
```

Lệnh X trên dùng output_dir trong config; khi chạy lại phải dùng config/output mới để giữ run cũ. Runtime/source ở `/tmp` cần được dựng lại nếu hệ thống xóa chúng; xem dependency/version và nguồn hash trong integrity receipts. Chưa có tuyên bố reproduction toàn recipe hoặc đạt 80%.
