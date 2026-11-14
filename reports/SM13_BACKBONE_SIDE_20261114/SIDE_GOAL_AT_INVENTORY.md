# SIDE_GOAL — SM13: backbone thay thế và mức độ thích nghi

Decision: **SM13-BACKBONE-SIDE-20261114-v1**. Ngày2026-11-14.
User: “ghi toàn bộ phần này vào SIDE_GOAL, thực hiện song song với training hiện tại”.
Trạng thái: **AUTHORIZED — IMPLEMENTATION_AND_AUDIT_IN_PROGRESS**, chưa có gain được chứng minh.

## 1. Vai trò và ranh giới

`GOAL.md` vẫn là mục tiêu khóa luận chính; full-paper CROTTC-IF training trên Kaggle tiếp tục, không reset/ngắt/restart để nhường tài nguyên cho nhánh này. Đây là **side experiment được user cho phép mới**, không khôi phục hàng loạt EXP/M0–M6 lịch sử và không thay tuyên bố full-paper completion. Cùng branch feature hiện hành; không commit/push/merge nếu chưa được yêu cầu.

Song song có nghĩa là phát triển code, unit tests, audit/protocol/package trên CPU local trong khi Kaggle train chính. Feature extraction và head fitting/native backward dùng cửa sổ Kaggle riêng có budget và quota admission, không cài vào runtime train chính, không tranh haiGPU của nó. Không tự train production local/Colab, không tạo cron/timer/subagent. Không tự mở test hoặc DEV_CAL.

## 2. Những điều đã giải thích và bằng chứng hiện có

- **Dataset** là tập câu/audio/nhãn. **Epoch** là một lượt học qua tập đó;300epoch không tạo300datasets. **Update** là một bước optimizer, **checkpoint** là trạng thái đã học để resume/inference.
- SM13 formal dùng TRAIN1800câu/12speakers,59562targets và61362exactgaps. CROTTC hiện hành cũng dùng1800TRAIN, không phải có thêm dữ liệu nhờ300epoch. Nhãn/preprocessing/task khác nhau phải được ghi rõ.
- SM13 đã áp dụng học nhiều epoch: targethead12epoch/batch512, gaphead5epoch/batch1024, seeds101/202/303, chọn terminal12/5 cố định. NhánhCTC/E4 dùng checkpoint đăng ký từ trước. Chữ13 chỉ13hidden representations, không phải13epoch.
- Backbone SM13 là `facebook/wav2vec2-base-960h`, **frozen/eval khi train SM13**; nó đã qua pretrained/ASR fine-tuning trước đó nhưng không được fine-tune trên dữ liệu phát âm trong lượt SM13. Scalar-mixing logits/scale và target/gap MLP được học.
- CROTTC hiện hành fine-tune WavLM Large Transformer, CNN frontend frozen, full architecture giữ nguyên. WavLM24Transformer layers/1024D, tức25hidden representations nếu tính đầu vào; Wav2Vec2Base12layers/768D,13representations. Checkpoint không thể nạp nguyên vào headSM13cũ; cần features/mixing/head/cache mới.
- Tăng epoch SM13 có thể giảm trainloss nhưng không bảo đảm generalization; dữ liệu mới/đa dạng khác với lặp lại cùng dữ liệu. SM13seed101targettrainloss0.7620→0.3833qua12epoch;gap0.3293→0.1620qua5epoch; ba seed cùng xu hướng. Formalrun không có validationcurve để chọn epoch vì fixedterminal/protectedsplit policy, không thể kết luận underfit/overfit chỉ từtrainloss.
- CROTTC DEV_SEARCH ở8completedepochs: PER65.96→30.97%;detectionF140.03→51.14%,bestepoch4≈51.19%; PER giảm không đảm bảo MDD F1 tăng. Đây là developmental metrics, không phải paper71.77% hay SM13strictS/O/I.
- Đổi backbone có thể cải thiện bằng chứng âm học; frozen replacement nhẹ hơn fullfine-tune. Fine-tune cho phát âm có cơ sở để thử nhưng dễ overfit với12speakers. Model lớn hơn/mới hơn không tự động tốt hơn; lỗi alignment, crop/context, supervision, class imbalance và operatingpolicy không biến mất khi đổi backbone.
- Lịch sử `RQ1-ENCODER-BENCH-01` đã thử HuBERTBase/WavLMBase dưới đối chứng single-layer và không xác nhận gain lớn; đây không phải bằng chứng WavLMLarge-adaptedSM13 đã được thử. `experiments/sm13_three_directions_20261113/encoder_adapter/` cũng có hướng adapter/matched-control cũ. Phải đọc và phân biệt trước khi gọi nhánh mới là novelty.

Bằng chứng gốc: `experiments/rq1_sm13_formal_train_01/PREREGISTRATION.json`, `FORMAL_TRAINING_RECEIPT.json`, `agent_a_sm13_engine.py`; `reports/ENCODER_BENCH_01_BACKBONE_COMPARISON.md`; full-model config/source receipts của CROTTC. Giữ nguyên artifacts/lịch sử.

## 3. Một câu hỏi nghiên cứu trọng tâm

**Giữ nguyên supervision/event population/canonical reference/alignment/head recipe, thay frozen Wav2Vec2 features bằng WavLM Large (pretrained hoặc CROTTC-adapted) có cải thiện SM13-style target/gap ranking và operating quality trên permitted development speakers không?**

Không gọi đây là phương pháp mới hay gain trước thực nghiệm. Không gộp một bảng điểm khác protocol thành so sánh trực tiếp.

### Đối chứng khóa trước khi fit

| ID | Frozen backbone | Vai trò |
|---|---|---|
| S0 | Wav2Vec2Base960h | SM13 gốc/replay reference; feature contract13×768 |
| S1 | Native pretrained WavLMLarge, pinned source revision | Matched backbone-family/size control |
| S2 | WavLMLarge từ CROTTC checkpoint đã xác minh | Kiểm tra lợi ích adaptation cho phát âm |

- S1 vàS2 dùng **cùng layer indices `[0,2,4,...,24]`**,13relative-depth representations;1024D/crop-target3072D/gap2048D. Không sweep chọn layers bằng development.
- S1/S2 cùng scalar-mixing/head widths/loss/LR/batch/epochs, nên soS2−S1 cách ly adaptation tốt hơn soS2−S0. S0 có inputdim/parametercount nhỏ hơn; báo chênh capacity, không gọi là matched capacity.
- Stage đầu giữ backbonefrozen, trainheadsseed101,12/5epoch, không đổi sang300chỉ vì paper300. Seeds202/303 là replication đã đăng ký, chạy sau correctness/resource gate; không chọn seed theo kết quả đẹp.
- Giữ nguyên registeredCTC/E4 branch và fixed50/50fusion cho cảS0/S1/S2; báo cả neural-only và fused. Không lén thay E4/scorer/alignment cùng lúc với backbone. Nâng E4/backbone end-to-end là biến thể khác, không được gộp gain.
- Full fine-tune SM13 hoặc epoch-extension là **giai đoạn sau có amendment riêng**, không mở sweep lớn song song. Giả thuyết được giữ, chưa triển khai mặc định.

## 4. Data, leakage và evaluation gates

- Chỉ frozenallowlistTRAIN1800/DEV_SEARCH599. TestSEALED, không đọc rawarchives/test/DEV_CAL. Inference dùng audio + suppliedcanonicalreference/đã dự đoán alignment, không dùng gold timestamps/nhãn lỗi làm feature. Gold chỉ ởfitTRAIN hoặcmetric sau inference.
- Featureextractor phải tách inference request khỏi gold labels; giữ Ntargets/N+1gaps, kể cả N=0/N=1, roworder/speaker IDs. Cache hash phụ thuộc modelcheckpoint/config/layers/normalization/alignment/manifest/order/sourcecode, không reuse cache chỉ vì filename giống nhau.
- **Current CROTTC backbone đã fit toàn bộ12TRAINspeakers. Không dùng nó để tạo “held-out TRAIN OOF” rồi coi là independent.** S2 OOF chỉ hợp lệ nếu backbone mỗifold không nhìn thấy fold đó (foldwise training mới, không nằm trong budget đầu).
- Stage đầu fullTRAINfit→DEV_SEARCH chỉ là **exploratory development comparison**, vì DEV_SEARCH đã được dùng nhiều lần và CROTTC selection cũng dùng nó. Không trình bày là xác nhận mới/unseen benchmark. Không tự mở test để giải quyết giới hạn này.
- Chờ trainedCROTTC checkpoint cuối/validation-selected đãverify để có S2 chính thức. Nếu dùng checkpoint sớm để smoke, pinSHA/epoch và ghiPROVISIONAL, không gọi fulltrained/final. Historical checkpoints không bị ghi đè.
- Metricprimary: classwise AP vàmacro3AP(S/O/I) theo EVAL-V3.1; báo F1/P/R, falsewarnings/câu, coverage, counts/byspeaker và resourcecost. Dùng fixed historicalwarningpolicy chỉ để so diagnostic, không tuyên bố giữ FWRbudget trên phân phối mới. Không retune threshold theo mỗi contender rồi gọi công bằng; mọi threshold-selection phải preregister riêng và giữ cùng dữ liệu/quy tắc.
- Báo mean/spread ba seeds, pairedbyspeaker/bootstrapuncertainty khi đủ support; improvement claim cần baseline matched, không chỉ trainloss giảm. Không ép threshold80%/budget mới làm điều kiện hoàn tất.

## 5. Thực thi hữu hạn và storage

- BướcA ngay: CPUread-only inventory audit + genericSM13-style feature/head adapter + unitparity tests với13×768gốc, dimension/cache/leakage failclosed. Không fitmodel hoặc backprop native trên laptop.
- BướcB: private Kaggle feature/nativeprofile, tối đa1800sdriver/3600splatform, TRAINonly representative allowlist, không input officialtest. Chỉ launch khi maintraining không bị thay đổi và platform admission/resourcebudget đã ghi.
- BướcC: preregister frozenS1/S2 productioncache/headtraining, 4hwindow maximum mỗi lượt; checkpoint atomic đủoptimizer/RNG/cursor. Stopnonfinite/mismatch/OOM/diskfloor. Không skipbatch, không rút dataset để làm đẹp score.
- Head recipe12/5 cố định cho lượt đầu, tối đa3seeds, không automaticgridsearch. Mọi productiontraining trênKaggle. Chưa có nativeprofile không được gọi modelready.
- Cache fullS1/S2ước tính mỗi biến thể~16.0GiBfloat32 hoặc~8.0GiBfloat16. Vì disk hiện hạn chế, **không materialize cả hai cachelocal**, không đổi precision âm thầm: chuẩn bịper-utterance/shardedcache, báoprecisionparity và profiletrước. Maintrainingdownload có ưu tiên14GiBfloor; sidejob yêu cầu thêmheadroom, không xóa artifacts để lấy chỗ.
- Sideoutputs ở `reports/SM13_BACKBONE_SIDE_20261114/` và runtime `/var/tmp/.../sm13_backbone_side/` riêng; không overwriteformalSM13/features/checkpoints/source/historicalreceipts.

## 6. Acceptance và tiến độ

- [x] Userauthorization/scope ghi nhận; thảo luận, hiểu đúngepoch/dataset/backbone được lưu.
- [ ] Realmanifest/alignment/cache/modelprovenance inventory audited, receipts hashed.
- [ ] Genericextractor/headadapter unitparity + failclosed leakage/cachetests.
- [ ] Kaggleprofile và privatepackageverified; native throughput/head numerical checks.
- [ ] S0replay/S1matchedcontrol/S2adapted frozenfeatures +12/5headtraining, sameeventdenominators.
- [ ] Learningcurves + permitteddevelopment scores + uncertainty/limits; không chứng nhận testgeneralization.
- [ ] Quyết định tiếp tục/dừng dựa trên evidence; nếu gain không rõ ghi no-improvement, không đổi thước đo/loại ca khó.

**Không đồng nghĩa:** hoàn thành SIDE_GOAL≠tái lập fullpaper≠hoàn thành đóng góp khóa luận. Currentstatus và evidence sẽ được cập nhật sau khi thực sự chạy/audit/test.
