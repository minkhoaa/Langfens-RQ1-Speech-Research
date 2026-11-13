# M1 — hoàn tất reproduction triển khai theo protocol local

Ngày: 2026-11-13. **M1 hoàn tất theo cổng GOAL §8:** có sequence-MDD reference
chạy được trên split sạch, baseline SM13 cùng protocol, preprocessing parity,
đầu ra đúng topology và sai khác với paper được ghi rõ. **Không reproduce metric
paper, không hoàn tất recipe WavLM Large 100 epoch, không promote hoặc nghiệm thu
sản phẩm. M2–M6 và goal tổng thể vẫn chưa hoàn thành.**

Kết luận thực nghiệm: reference end-to-end chạy được nhưng lịch hai epoch còn
cho kết quả yếu và khác biệt đáng kể giữa seed. E4 cải thiện ranking so với E1
trên cùng posterior, nhất là omission, nhưng chưa vượt SM13 full-fit về Macro-3
AP. Chưa có giải pháp C/S/O/I đủ chất lượng sản phẩm.

## 1. Cổng M1 và bằng chứng

| Yêu cầu GOAL | Artifact | Kết quả |
|---|---|---|
| Ít nhất một sequence-MDD reference phù hợp | `sequence_reference/`, `REFERENCE_FIDELITY.json` | Train encoder + CTC head thực sự, 4 fold × 2 seed; supported base-backend adaptation của public PhnMonoSSL baseline |
| SM13 theo cùng protocol | `sm13_full_replay/`, `sm13_strict/` | Replay đủ full-fit lịch sử và train lại strict-fit, cùng 6/3/3 và held population |
| Legacy CTC control | `legacy42/`, `legacy_strict_sequence_parity.json` | 1.243/1.243 strict sequence giống recipe legacy; kiểm tra khác biệt inventory, không bịa supervision ablation |
| Preprocessing parity | `preprocessing_parity_cuda.json`, CPU profile riêng | GPU reconstruction khớp bitwise 52 kiểm tra trên hai WAV, đủ 13 tầng; hash/shape/dtype/finite cho 26 cache |
| N target/N+1 gap, inference không nhận gold | `inference_verification.json`, waveform outputs | CLI WAV + supplied phones/text/OOV đã chạy, thay reference giữ nguyên acoustic recognition |
| Calibration và evaluator có thể replay | `CALIBRATION_REPLAY.json`, per-run metrics | 144 policy khớp; EVAL-V3.1 full held và PER được tính lại |
| Kiểm thử và snapshot | `TESTS.log`, `verification.log`, `artifact_hashes.json` | 52 test; giữ nguyên snapshot lịch sử với mapping GOAL tường minh |

Phân biệt rõ hai nghĩa: **implementation/protocol reproduction** được hoàn tất;
**original-paper result replication** chưa thực hiện. GOAL §8 cho phép ghi rõ
khác biệt với paper. Lượt trước mới có frozen-head adaptation và public smoke;
lượt này bổ sung end-to-end reference, controls và kiểm tra toàn tuyến để đủ cổng
triển khai M1. Không gọi baseline CTC này là phương pháp CROTTC-IF hoặc IF teacher.

## 2. Reference và mức fidelity

Reference là public **PhnMonoSSL phone-CTC baseline** trong
[official IF-MDD source](https://github.com/Secondtonumb/IF-MDD/blob/cca150ec4ff00e01d61a613092d1d178e661dfb2/hparams/phnmonossl.yaml),
commit `cca150ec4ff00e01d61a613092d1d178e661dfb2`. Source hỗ trợ backend Wav2Vec2;
local checkpoint override dùng `facebook/wav2vec2-base-960h`, revision
`22aad52d435eb6dbaf354bdad9b0da84ce7d6156`. Pretrained weights chưa được adaptation
bằng L2 trong pipeline này; public L2-trained bundles chỉ giữ vai trò smoke cũ.

Giữ cơ chế audio-only realized-phone CTC, train speech encoder với CNN frozen,
head hai lớp 384 + LeakyReLU/LayerNorm, Adam lr encoder/head 1e-5/3e-4, speed
95/100/105, greedy decode rồi deterministic edit. **90.630.953 tham số trainable**;
kiểm tra encoder thực sự thay đổi và CNN không đổi. Không dùng gold boundary
hoặc realized transcript làm inference input.

Sai khác đã đăng ký: base-960h thay WavLM Large mặc định; inventory 40 phone +
blank thay 44 token; local strict supervision và split 6/3/3; **hai epoch cố định**
thay 100; effective batch 16 qua microbatch 1 thay batch 16 × accumulation 2;
fp32/gradient checkpointing, gradient clip 5; không chọn checkpoint bằng cal/held.
Functional resampling khác cached transform ở mức số học float32, có regression
fixture. Hai parameter group của Adam cập nhật encoder/head riêng như hai optimizer
trên các parameter rời nhau. Upstream autocast branch zero gradient mỗi microbatch;
local runner dùng accumulation thực sự, không sao chép lỗi đó.

Đây là sai khác thực chất, không chỉ đổi tên checkpoint. Kết quả yếu của hai epoch
không bác bỏ recipe dài hơn hoặc backbone khác. Không trích metric paper để xếp
hạng trực tiếp với EVAL-V3.1. Chi tiết đầy đủ ở `REFERENCE_FIDELITY.json` và các
repair/runtime khác biệt ở `IMPLEMENTATION_REPAIRS.md`.

## 3. Population và phép so sánh

TRAIN 1.800 câu/12 speaker; bốn fold 6 fit / 3 cal / 3 held; seeds 101/202.
Recognizer và strict SM13 chỉ fit các câu complete-sequence eligible thuộc sáu
fit speaker. Primary subset toàn TRAIN là 1.243 câu. **MDD giữ đủ 1.800 câu,
59.562 target và 61.362 gap mỗi seed**; PER dùng 39.477 realized phone có exact gold.
Không mở TEST/DEV_REPLICATION, không tạo nhãn người mới hoặc speaker xác nhận mới.

Có 24 lượt train mới: tám reference, tám legacy CTC và tám SM13 strict (mỗi SM13
run có target/gap head). Thêm tám full-SM13 historical replay và 16 E1/E4 decoder
replay tạo **48 run artifact được kiểm tra**, không phải 48 lượt train mới.

Chỉ các phép so đúng yếu tố mới có diễn giải tương ứng: legacy42/linear41 kiểm
inventory trên cùng strict labels; SM13 full/strict đo ảnh hưởng fit population;
E1-export/E4 giữ cùng frame posterior. Reference/SM13 có cùng split/population
đánh giá nhưng khác supervision task, feature và optimizer update, nên không gán
mọi chênh lệch cho encoder hoặc kiến trúc.

## 4. Kết quả pooled held, policy cal B=0,20

Policy được fit trên calibration speakers và giữ nguyên sang held; không sửa
threshold để ép held FWR về 0,20. Gộp held predictions rồi evaluate một lần/seed.

| Cấu hình | Seed | Macro-3 AP | Macro-3 F1 | Held FWR | Raw FP | PER strict |
|---|---:|---:|---:|---:|---:|---:|
| Reference CTC end-to-end, 2 epoch | 101 | 0.09009 | 0.01061 | 0.18556 | 334 | 38.90% |
| Reference CTC end-to-end, 2 epoch | 202 | 0.07258 | 0.00792 | 0.20778 | 374 | 46.01% |
| Legacy CTC 42 phone, strict | 101 | 0.06582 | 0.00542 | 0.18667 | 336 | 62.89% |
| Legacy CTC 42 phone, strict | 202 | 0.06767 | 0.00707 | 0.18611 | 335 | 60.74% |
| SM13 all-13, strict-fit | 101 | 0.20083 | 0.08950 | 0.22444 | 404 | — |
| SM13 all-13, strict-fit | 202 | 0.20945 | 0.08348 | 0.24500 | 441 | — |
| SM13 all-13, full-fit lịch sử | 101 | 0.24687 | 0.12495 | 0.21889 | 394 | — |
| SM13 all-13, full-fit lịch sử | 202 | 0.25071 | 0.12987 | 0.23500 | 423 | — |
| Frozen MLP + E4, export-parity | 101 | 0.21375 | 0.10432 | 0.19778 | 356 | 27.04% |
| Frozen MLP + E4, export-parity | 202 | 0.21443 | 0.10421 | 0.21278 | 383 | 27.14% |
| Frozen MLP + E1, export-parity | 101 | 0.12460 | 0.02311 | 0.21333 | 384 | 27.04% |
| Frozen MLP + E1, export-parity | 202 | 0.12445 | 0.01885 | 0.20222 | 364 | 27.14% |

S/O/I có support 7.381 / 1.825 / 560. B=0,10 và 0,50, classwise counts, warnings
trên C/sai loại lỗi, per-speaker metrics và phân tích nhóm âm/gap đều được lưu.
Không tính trung bình precision/recall các fold để thay pooled result.

Reference CTC có AP S/O/I seed101 = 0,18345 / 0,07066 / 0,01616; seed202 =
0,16464 / 0,04177 / 0,01134. Recall S/O/I chỉ 0,88% / 0,27% / 0,54% và
0,60% / 0,16% / 0,54%. Đây là baseline chạy được, chưa có chất lượng để promote.

E4 so E1-export tăng Macro-3 AP khoảng 0,08915 / 0,08998; AP cả ba class tăng ở
cả hai seed. E4 đạt nhánh AP của research screen với matched inference control,
không phải đóng góp mới so với E4 đã có trong lịch sử. Ở B=0,20, E4 có precision
S/O/I khoảng 46–51% / 56–58% / 16–20%, recall khoảng 1,06–1,15% / 15,84–16,00% /
2,32–2,50%. Chưa đạt product proposal và held FWR seed202 vẫn vượt 0,20.

SM13 full-fit vẫn có Macro-3 AP cao hơn. SM13 strict thấp hơn full-fit ở cả hai
seed cho thấy cần đo coverage/selection bias, không chỉ thay head hoặc decoder.
Không suy ra ý nghĩa thống kê từ các fold/seed dùng lại speaker.

## 5. Chất lượng bằng chứng âm học, gap và uncertainty

`DETAILED_DIAGNOSTICS.json` giữ AP/TP/FP/FN/support cho nguyên âm, âm tắc và nhóm
khác; gap đầu/cuối, internal, nhiều âm chèn; gold S bị báo O kèm I lân cận; warning
lệch một gap. Nhóm nhiều âm chèn là positive-only: recall/support có ý nghĩa,
AP trong nhóm là trivial và không được dùng làm thành tích ranking.

Trên strict subset có 4.480 realized token thuộc source S. Recognizer end-to-end
giữ đúng realized phone theo deterministic alignment: nguyên âm 27,74% / 12,65%
(1.961 token), âm tắc 45,76% / 40,65% (861), nhóm khác 55,19% / 55,37% (1.658).
Phép đo phụ thuộc alignment và không bao phủ toàn bộ 7.381 S, đặc biệt deviation
cùng base phone bị loại khỏi strict supervision. Không đồng nhất token retention
với khả năng định vị/chẩn đoán S ở operating point.

Candidate diagnostic có 48 câu chọn trước, widths 1/4/8, top12 token expansion
mỗi frame, exact CTC rescore unique sequences. Median retained mass lần lượt
0,125% / 0,447% / **0,737%**; conditional event score có thể đổi tới 0,5 khi tăng
beam 4→8. Không đạt bằng chứng hội tụ/coverage để gọi finite-candidate normalized
score là full event posterior. E4 dùng likelihood-ratio trên đối thủ hữu hạn,
được ghi **uncalibrated scores**, không mang diễn giải posterior.

Bốn decoded sequence cal/held thay đổi sau fp16 export trong tổng 7.200 lượt decode.
E1-export được thêm theo addendum để dùng đúng normalized posterior của E4;
PER và hash posterior E1/E4 khớp hoàn toàn. Không dùng E1 fp32 trực tiếp để gán
chênh lệch nhỏ do export thành lợi ích decoder. Reference symbol ngoài inventory
được giữ trong denominator; E4 lưu unsupported mask và neutral scores cho câu
không thể chấm likelihood, không tự alias hoặc bỏ event.

## 6. Reference, preprocessing và inference

Supplied-reference topology parity đạt 1.800/1.800. Text-first CMUDict chỉ khớp
corpus reference ở 573/1.800 câu; 26 câu cần explicit reference vì OOV. Fit-only
unanimous prompt catalog có 470 câu held dùng được và chỉ 274 câu khớp corpus.
Không lấy held gold chọn lexical variant, không remap gold hoặc đổi denominator.
**Arbitrary-text parity với corpus chưa được giải quyết.** M1 khóa contract:
supplied phones cho benchmark; text có profile/ID riêng và OOV refusal rõ ràng.
Đây là giới hạn đầu vào được đo và xử lý tường minh, không tuyên bố text-only
sản phẩm đã chính xác. Phát âm tự nhiên/acceptable variants cần human audit sau.

GPU fresh extraction khớp bitwise cached SM13 cho đủ 13 tầng trên hai WAV mẫu;
26 cache đã kiểm shape/dtype/finite/hash. CPU/reference runtime có last-layer
drift max 2,62e-4, RMSE 1,75e-5 và được giữ trạng thái không đạt strict guard.
SM13 controls luôn dùng cùng original float32 cache. H6 có tolerance fp16 riêng.
Các input normalization, waveform float32 mono/16k và clock 20 ms đều ghi rõ.

CLI thật đã chạy trên ABA_arctic_a0003, checkpoint seed101/fold0 chọn cố định:
WAV + supplied reference → 39 target/40 gap, save/reload; thay phone reference
hoặc text giữ nguyên acoustic recognized sequence; NIGHTGLOW trả explicit-reference
required và không xuất chuỗi một phần. Đây là held development đã exposed,
không phải fresh-speaker confirmation. Pipeline không nhận gold error/boundary.

## 7. Kiểm tra, budget và snapshot

**52 test pass**, gồm các regression cho masked SpecAugment initialization,
speed-rate formula, E4 CTC kernel độc lập, unknown/OOV, topology, policy và head.
**144 policy replay** khớp toàn bộ counts/tie cutoffs; E4 dùng PyTorch float64 CTC
độc lập với C++ (max log-likelihood difference 2,84e-14; threshold tolerance 1e-9).
Pooled evaluator và independent Levenshtein PER chạy đủ held, không thiếu event.

715 entry lịch sử được xác minh bằng hash, gồm GOAL được resolve qua snapshot.
Run lỗi/interrupted và source trước repair được giữ riêng, không vào pooled result.
Không re-seal manifest cũ để che thay đổi. GOAL trước lượt này ở
`GOAL_before_completion.md`; dùng verifier mới cho chuỗi snapshot hiện tại.
Không có independent agent review trong lượt này.

Reference train/predict/calibrate/save của tám run dùng 1691.64 giây GPU-wall,
peak allocated VRAM 1.806 MiB trên GTX 1650 4 GiB. Cumulative M1 GPU accounting,
gồm failed attempts và historical SM13 extraction, là 2257.56 giây;
CPU measured stages gồm historical extraction là 3789.11 giây,
cộng allowance 600 giây cho các check/replay nhỏ chưa đo riêng vẫn dưới cap
7.200 giây mỗi device account. Đây là stage wall times có thể chạy song song,
không phải tổng elapsed tuần tự hoặc CPU core-hours. Không tự mở thêm sweep.

## 8. Reproduce và handoff

Runtime reference: `/tmp/langfens-ifmdd-env`, Transformers 4.57.0/tokenizers 0.22.2,
SpeechBrain 1.0.3 và torch 2.6.0+cu124; public recipe torch 2.7.1 là một khác biệt.
`../m1_reference` giữ source pin và script dựng isolated runtime. Các thư mục
run hoàn tất từ chối overwrite; chạy mới cần namespace version mới.

```bash
# Reference, dưới cooperative GPU lease
.venv/bin/python coordination/phase3_gpu_job.py --job M1-REFERENCE-REPRO \
  -- /tmp/langfens-ifmdd-env/bin/python -m research.realized_phone_pipeline.m1_sequence_reference --mode smoke
# Đổi --mode train sau khi smoke đạt, trong namespace mới đã đăng ký.

# Verify snapshot và bằng chứng đóng M1 hiện tại
.venv/bin/python -m research.realized_phone_pipeline.close_m1

# Inference nhận WAV + reference độc lập, không gold boundary/error label
/tmp/langfens-ifmdd-env/bin/python -m research.realized_phone_pipeline.m1_inference \
  --run-dir experiments/realized_phone_pipeline_20261113/m1_completion/sequence_reference/seed101_fold0 \
  --audio data/derived/l2arctic/wavs_16k/ABA_arctic_a0003.wav \
  --reference 'F AO R DH AH T W EH N T IY IH TH T AY M DH AE T IY V N IH NG DH AH T UW M EH N SH UH K HH AE N D Z' \
  --output /tmp/m1_inference.json
```

M1 bàn giao runner/checkpoint, baseline/control, policies và bộ đánh giá cho M2.
M2 cần cải thiện bằng chứng âm thực tế trên lỗi thật với supervision/budget đã
đăng ký, không chỉ PER hoặc một operating point. M3 còn phải giải quyết posterior
coverage/localization; M4–M6, human adjudication, fresh-speaker confirmation và
thỏa thuận sản phẩm vẫn mở. Không tiếp tục train ngoài M1 trong lượt này.
