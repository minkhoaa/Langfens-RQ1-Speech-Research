# M1 tiếp tục — reference pin, interface smoke và matched capacity control

Ngày: 2026-11-13. **Có tiến bộ nghiên cứu, không promote; M1 chưa hoàn tất toàn bộ.**

Đã pin source IF-MDD, chạy interface thật của public CTC checkpoint và hoàn tất
16 run đối chứng (hai head × bốn fold × hai seed). Head MLP cải thiện nhận diện
và ranking S/O/I so với linear trên cùng dữ liệu/update. Recall vận hành vẫn rất
thấp; reproduction đầy đủ paper và các matched control khác còn mở.

## 1. Kết quả matched control

Giữ frozen W2V2-H6, strict subset 1.243 câu, cùng fit/cal/held 6/3/3, seeds
101/202, tám epoch, batch 16, AdamW/lr/clip, inventory, decoder và calibration.
MDD giữ đủ 1.800 câu/59.562 target/61.362 gap mỗi seed. PER dùng 39.477 âm có
exact realized gold. MLP chỉ thay topology head: hai lớp 384, LeakyReLU,
LayerNorm và projection. Có **459.689** tham số so với linear **31.529**.
MLP không có thêm context thời gian, không fine-tune encoder, không đổi nhãn.

| Head / seed | PER | Macro-3 AP | Macro-3 F1 | Held FWR tại cal B=0,20 | Raw FP |
|---|---:|---:|---:|---:|---:|
| Linear / 101 | 61,856% | 0,06728 | 0,00537 | 0,18833 | 339 |
| Linear / 202 | 61,112% | 0,06748 | 0,00699 | 0,19833 | 357 |
| IF-style MLP / 101 | 27,033% | 0,12460 | 0,02311 | 0,21333 | 384 |
| IF-style MLP / 202 | 27,137% | 0,12448 | 0,01885 | 0,20222 | 364 |

| Lớp | AP linear 101 / 202 | AP MLP 101 / 202 | Precision MLP 101 / 202 | Recall MLP 101 / 202 | TP MLP 101 / 202 | Support |
|---|---:|---:|---:|---:|---:|---:|
| S | 0,15023 / 0,15024 | 0,23061 / 0,22682 | 30,58% / 28,84% | 1,206% / 1,043% | 89 / 77 | 7.381 |
| O | 0,04063 / 0,04179 | 0,11227 / 0,11110 | 19,83% / 13,71% | 1,315% / 0,932% | 24 / 17 | 1.825 |
| I exact gap | 0,01097 / 0,01041 | 0,03093 / 0,03551 | 7,61% / 8,22% | 1,250% / 1,071% | 7 / 6 | 560 |

Macro-3 AP tăng **0,05733 / 0,05700**; cả ba class AP đều tăng ở cả hai seed.
Theo speaker: S và O tăng ở 12/12 speaker mỗi seed; I tăng ở 12/12 và 11/12,
speaker còn lại hòa. Đạt nhánh AP của research screening gate v2 (≥0,02 và
không class giảm quá 0,01). Không suy ra ý nghĩa thống kê từ các fold/seed
dùng lại speaker. `paired_speakers_seed*.json` giữ support và metric từng speaker.

Precision và recall đều tăng tại các policy cal-frozen tương ứng, nhưng FWR
cũng tăng và vượt 0,20 ở cả hai seed MLP. Vì vậy **không tuyên bố đã cải thiện
vận hành tại FWR held bằng nhau**. Không đổi threshold bằng held gold. B=0,10
và 0,50 được lưu đầy đủ trong từng arm. Score vẫn là hard edit 0/1, không phải
posterior event được hiệu chuẩn. Đây là cải thiện một baseline yếu, chưa đủ
thay SM13 hoặc đáp ứng sản phẩm; không có fresh-speaker confirmation.

## 2. Reference đã pin và giới hạn fidelity

- [Official source](https://github.com/Secondtonumb/IF-MDD/tree/cca150ec4ff00e01d61a613092d1d178e661dfb2):
  commit `cca150ec4ff00e01d61a613092d1d178e661dfb2`; bản source ở `upstream/`.
- [Public CTC bundle](https://huggingface.co/Haopeng/CTC_for_IF-MDD/tree/de46aaf5778c5759c003ffa17657176fee772e77):
  revision `de46aaf5778c5759c003ffa17657176fee772e77`.
- [CROTTC-IF bundle metadata](https://huggingface.co/Haopeng/CROTTC-IF-l2-arctic/tree/9149e5107b3332b9aa3f5573f6328a4e312006ad):
  revision `9149e5107b3332b9aa3f5573f6328a4e312006ad`; không download/train
  weights CROTTC-IF trong lượt này. Paper target là
  [arXiv v1](https://arxiv.org/html/2604.22133v1), không gọi CTC baseline là CROTTC-IF.

Public checkpoints không kèm training-speaker manifest đủ để xác minh loại trừ
speaker của grouped CV này. Không dùng chúng cho efficacy hoặc clean baseline.
Public inventory có sil/ERR/BOS/EOS, không có AX. Đặc biệt **token ID của bundle
CTC khác source hiện hành** ở HH/F/TH/IH/M/DH/S. Dùng tokenizer.ckpt đi cùng
weights, không dùng `upstream/utils/label_encoder.txt` thay thế. Metadata và
URL/hash của từng nguồn nằm ở `metadata/sources.json`.

Public recipe có WavLM Large trainable, head hai lớp 384, speed perturbation,
hai optimizer, 100 epoch ở config CTC; runtime main dùng perceived_train_target.
Comment `training_target` trong YAML không tự chứng minh nhãn canonical được
dùng để train. Không có source train JSON được phát hành trong checkout để audit
toàn bộ preprocessing nhãn. Paper/CTC metrics khác EVAL-V3.1, không xếp hạng trực tiếp.

Matched MLP chỉ tái lập **head topology với input/output dimensions thay đổi**.
Encoder/H6, nhãn strict, optimizer, budget, augmentation và decoder khác public
recipe. Không thực hiện IF teacher transfer hoặc CROTTC loss. Vì vậy đây là
partial architecture adaptation và capacity ablation, không full paper reproduction.

Root LICENSE/README ghi Apache-2.0; header train.py còn ghi non-commercial.
Ghi lại sự không nhất quán này; chưa coi source licence là bảo đảm cho mọi
artifact/checkpoint/dataset. Không publish hoặc deploy bundle trong lượt này.

## 3. Interface smoke public CTC

Strict-load toàn bộ encoder/head bằng `weights_only=True`, không missing/unexpected
keys. Dùng ba câu TRAIN chọn cố định đầu tiên đủ strict ở từng role fold 0:
SKA/HKK/ABA `arctic_a0003`. Checkpoint có thể đã gặp các speaker này; **chỉ smoke**.

Đọc WAV float32 mono 16 kHz, không crop hoặc dùng gold boundary. Dùng SpeechBrain
WavLM wrapper với config-only construction rồi restore toàn bộ released weights;
không execute Python/HyperPyYAML download. Logits khớp chính xác đường WavLM
với input layer_norm viết riêng; greedy tokens khớp implementation collapse riêng.
Giữ raw tokens và trace loại structural sil/BOS/EOS; ERR nếu xuất hiện là
uncertain candidate, không tự chế phone hoặc alias AX/AH. Adapter edit giữ N
target/N+1 gap và nhiều insertion candidate. Calibrate một câu riêng, evaluate
một câu held, save/reload keyed outputs và kiểm tra reference perturbation.
Metrics của smoke không có diễn giải efficacy.

SpeechBrain 1.0.3 không tương thích Transformers mới của project. Runtime riêng
`/tmp/langfens-ifmdd-env` dùng Transformers 4.57.0 (bản upstream pin, package bị
yank vì lỗi setup), tokenizers 0.22.2 và huggingface-hub 0.36.0; chia sẻ torch
2.6.0 và các dependency hiện có. Đây là sai khác với torch 2.7.1 của upstream.
Không downgrade Transformers project. Các package SpeechBrain/HyperPyYAML,
ruamel.yaml và sentencepiece được thêm vào project venv để chạy head/parity tests.

## 4. Kiểm tra, chi phí và snapshot

- **45 test pass**: topology, repeated CTC/edit, policy v2/evaluator, head topology
  bằng SpeechBrain, runtime isolation và structural/unknown-token adapter.
- Replay cả 16 run: artifact/checkpoint hash, speaker exclusion, mọi policy từ
  calibration, EVAL-V3.1, exactly-once held coverage, PER bằng implementation
  Levenshtein độc lập. Cả tám linear checkpoint có tensor và fit-ID order khớp
  chính xác snapshot baseline cũ.
- Hai smoke capacity: 4,07 giây kể cả load/check input. Train/predict/evaluate/save
  16 run: **220,11 giây** kể cả load. Public smoke: **7,56 giây**. Replay/tổng hợp,
  tests và source/download là chi phí riêng; không gộp download thành training.
  Tất cả compute chạy CPU/four threads. Không chạy CUDA training.
- Tính từ config WavLM: standard fp32 Adam cần ít nhất **4,638 GiB** cho trainable
  encoder parameters/gradients/two moments khi freeze CNN, chưa có activation/head.
  GTX 1650 có 4 GiB. Đây là lower bound, không phải đo OOM hoặc chứng minh offload,
  optimizer khác hay kiến trúc khác không chạy được. Không mở sweep để vượt budget.
- Các receipt compute cộng historical extraction/baseline khoảng 16 phút; replay
  và kiểm thử vẫn để tổng lượt này thấp hơn cap hai giờ. Không đo peak RAM.
- Source runner có một sửa accounting trong lúc chạy: fallback 654 giây được
  thay bằng trường receipt extraction_seconds=653,0627. Chỉ đổi budget dư, không
  đổi model/data/optimizer. Source trước sửa được snapshot để hash từng receipt
  vẫn kiểm tra được; không re-seal các receipt để che thay đổi.

M0/M1 cũ giữ nguyên. GOAL trước lượt này được giữ ở
`input_snapshots/GOAL_after_M0_closeout.md`; verifier mới resolve hash GOAL cũ
bằng snapshot tường minh. Manifest M0 closeout không bị cập nhật. Verifier M0
cũ kiểm GOAL live sẽ cần dùng snapshot này khi GOAL đã tiến sang M1.

## 5. Trạng thái và bước kế tiếp

M1 đã có source/checkpoint pin, public interface smoke và matched capacity evidence.
**Còn mở:** full sequence-reference recipe trên split sạch; matched legacy-label
control; SM13 cùng population/update cho causal comparison nếu cần; giải quyết
reference từ text so với corpus ontology. SM13 lịch sử chỉ là context trong
capacity ablation này. Không đánh dấu M1 hoàn thành, không promote, không coi
M2–M6 hoặc product acceptance đã đạt.

Kết quả cho thấy capacity head là một yếu tố có tác động trong recipe này;
không chứng minh đó là nguyên nhân duy nhất. Sau M1 cần cải thiện recognizer/
supervision và đánh giá phone-error retention; hard-edit scores vẫn giới hạn
ranking/vận hành. Dừng sau hai arm đã đăng ký, không tự train tiếp decoder,
large backbone, thêm epoch hoặc seed chọn theo kết quả.

## 6. Tái chạy

`research/realized_phone_pipeline/m1_reference_runtime.sh` dựng runtime isolated.
Checkpoint phải được download pinned trước smoke; runtime compute offline.
Namespace hiện có từ chối overwrite run đã hoàn tất; reproduction mới cần
namespace version mới, giữ nguyên artifacts đã niêm.

```bash
.venv/bin/python -m research.realized_phone_pipeline.m1_capacity --mode smoke
.venv/bin/python -m research.realized_phone_pipeline.m1_capacity --mode train
.venv/bin/python -m research.realized_phone_pipeline.m1_capacity --mode summarize
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /tmp/langfens-ifmdd-env/bin/python \
  -m research.realized_phone_pipeline.m1_public_ctc
.venv/bin/python -m research.realized_phone_pipeline.verify_m1_reference
```
