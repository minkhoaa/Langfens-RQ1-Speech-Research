# CROTTC-IF fidelity ledger — 2026-11-14, audit từ 17:05 +07

**Verdict: FULL_PAPER_NOT_REPRODUCED; F1–F7 vẫn OPEN.** Đây là audit tiếp nối,
không sửa recipe/driver hiện hành, không cấp quyền campaign mới, không mở test/DEV_CAL.

## Audit tiếp nối19:44 — bổ sung lỗi eval dropout, sửa giả thuyết BN

[Chẩn đoán mới](../DIAGNOSIS_1945/REPORT_VI.md) xác nhận SpeechBrain1.0.3RoPEMHA
vẫn attentiondropout.1 trongeval;F7cóknownimplementationdefect chưarepairproduction,
chưađoactualnativeF1delta. ActualAMheads **0BN/14LayerNorm**,khôngphảiBNbatchnhỏ.
Các mô tả BN/groupparity lịch sử không thay actualmoduleinspection. Ledger bên dưới
làaudit17:05;khôngcoi90testspasslàevaldeterminismđãpass. Newtests58passed/1strictxfail.

## Primary sources và phạm vi chứng cứ

- **P1:** [paper arXiv v1](https://arxiv.org/html/2604.22133v1), §§3.3, 4.2–4.3,
  6.1–6.3, Tables1–4. Đọc lại trực tiếp bằng `fetch_content` ngày 2026-11-14;
  `read_url_content` cùng URL lỗi, không dùng lỗi crawler để kết luận paper thiếu thông tin.
  Retrieval ID `mwg8qw2sz0ljzp`; không dùng snippet thứ cấp thay nội dung paper.
- **S1:** [official pinned source](https://github.com/Secondtonumb/IF-MDD/tree/cca150ec4ff00e01d61a613092d1d178e661dfb2).
  Đọc lại README public tại đúng commit. Replay kiểm SHA **65/65** files của
  `experiments/crottc_if_full_local_20261113/SOURCE_RECEIPT.json`.
  `PINNED_PUBLIC_TREE.json`: GitHub recursive tree tại đúng commit trả **79 entries,
  truncated=false**; không có data JSON/frozen split manifests. `utils/data_prep.py`
  và `utils/DataPrepIO.py` là loaders cho JSON đầu vào, không cung cấp raw TextGrid→split
  generator. Không kiểm/chuyển sang HEAD mới; không kết luận mọi artifact khác trên web đều vắng mặt.
- **D1:** frozen authorized `train.jsonl`, `dev_search.jsonl` và `DATA_RECEIPT.json`
  trong `experiments/crottc_if_full_local_20261113/`; cả hai manifest SHA khớp receipt.
  E1 chỉ đọc metadata; raw TRAIN audit A2 riêng được mô tả bên dưới. Không đọc WAV.
- **L1:** `LIVE_STREAM_1707.log` là stream của exact private window001/v1,
  chứa update2163–3150 và validation38–55. Ghép pinned curve1–51 để đủ epoch1–55.
- **T1:** `tests/test_crottc_fidelity_audit.py` và năm test modules liên quan;
  Lượt đầu **87 passed / 10 warnings / 4.68s**, CPU-only, `TESTS.log`.
  Bổ sung3 malformed-phone fixtures sau audit raw TRAIN; lượt cuối **90passed/10warnings/7.43s**
  ở `TESTS_FINAL.log`.
- **E1:** `FIDELITY_EVIDENCE.json`, replay bằng `tools/audit_crottc_fidelity.py`.
  Source/data/implementation hashes nằm trong E1. E1 không xác minh checkpoint.
- **R1:** `RECONCILIATION.json`: full downloaded EPOCHS1–55 khớp merged curve,
  overlap pinned1–51 không đổi;8 implementation hashes local khớp remote contract.
  Metadata đã tải không đồng nghĩa full weights/optimizer export đã verified.
- **A2:** `F2_UNKNOWN_C_AUDIT.json`: chỉ đọc3 TextGrid members thuộc allowlisted TRAIN,
  sau khi verify3 archive hashes và annotation hashes; không extract, không đọc WAV/test.
  Ba unknown-C positions đến từ raw untagged `Z_`, `ER)`, `D_`, không phải khôi phục phone truth.

## Ledger và acceptance

| Gate | Evidence đã kiểm lại | Kết luận / điều kiện đóng |
|---|---|---|
| **F1 split/population/lineage** | P1 §6.1/Table1:2429/268/900, test NJS/TLV/TNI/TXHC/YKWK/ZHAA. D1:1800 TRAIN (12×150),599 DEV (4 speakers), TRAIN/DEV IDs không overlap; **NJS150 đã ở TRAIN**. | **BLOCKED cho exact paper result từ lineage này.** Cần author utterance manifests/version/hashes, speaker+utterance+audio overlap audit, quyền data protocol cụ thể. Paper counts không đủ tái dựng split. Campaign faithful phải từ clean pretrained native; không reuse current AM để certify paper test NJS. Không ngắt/xóa campaign hiện tại. |
| **F2 supervision/preprocessing** | `prepare_full.raw_sequences`: star→err, AX→AH, malformed→err, silence giữ ở TRAIN, deletion bỏ khỏi target. E1:557 TRAIN câu/1110 err target tokens; teacher labels C/S/D/I đầy đủ. **3 aligned err positions mang label C**,1091 S,16 I. Đây là unknown realization, không gold realized phone được phục hồi. | **OPEN.** Cần author raw-to-reference script/policies và fixtures silence/star/deletion/insertion/normalization. Nguồn3 unknown-C positions đã truy về raw untagged malformed `Z_`, `ER)`, `D_` bằng A2 và3 fixtures; giữ nguyên provenance, không tự đổi chúng thành lỗi hay sửa labels. Không âm thầm drop557 câu hoặc đoán realized phones. |
| **F3 validation labels** | E1:599/599 DEV có provenance `authorized_derived_manifest_without_silence_or_raw_star_markers`;0 parallel-silence intervals;12 err target tokens/12 câu. TRAIN có7577 parallel-silence intervals. | **OPEN.** Cần frozen authorized raw→reference DEV pipeline và parity về inventory, markers, counts/denominators. Tái đọc metadata không khôi phục thông tin đã mất. Không mở DEV_CAL/test để giải quyết. |
| **F4 AM recipe/augmentation** | P1 §6.2:time warp80, tối đa3 time/freq masks, ratio(.1,.3]. S1 `_confEnc` YAML:DropFreq/DropChunk, speed95/100/105, Warping window5 được khai báo nhưng không trong Augmenter. `NativeBatch.prepare_views` gọi speed và waveform augmentation; native SSL SpecAugment bật. Full WavLM315453120 + heads3822765, kernel3, objective paper Eq5/6 đã hiện diện. | **OPEN, xác nhận non-equivalence.** Cần author confirmation/effective recipe và deterministic augmentation fixtures. **Thêm câu hỏi stride:** P1 nói kernel AND stride3; CPU source-sized Conformer kernel3 có toàn Conv1d stride1 và15→15 frames. Không thấy stride3 subsampler trong current AM path. Cần xác nhận stride ám chỉ module nào hay lỗi mô tả; chưa tự thêm subsampling giữa campaign. |
| **F5 IF architecture/recipe** | P1 §§4.2/4.3/6.2:2-layer384 LM/FuN,2 fusion branches,downsample4,C/S/D/I, Eq18 omega1=.3 nhưng §6.2=.5;200 LM/teacher epochs chưa tách stage. S1 YAML alias fuse_net_dec, regular/relative attention và CNN khác mô tả. Local `if_components.py`:independent branches, full Conformer memory,4classes, .5/1/10, noConPCO. T1 kiểm teacher→LM/memory gradients, ragged masks, causal prefix, no teacher/gold in beam search. | **OPEN.** Implementation safeguards có scoped tests nhưng không chứng minh author-equivalence/native training. Khóa paper→source→effective config (RoPE vs regular/RelPos,128shared+64position branch vs current shared128→64), xác nhận stage allocation/freeze/loss, profile native IF và full-state resume/export trên Kaggle trước production. Historical local IF commands không phải quyền chạy local. |
| **F6 execution numerics** | P1 singleGH200; current2T4, FP16autocast/FP32master-loss-Adam, DP và per-utterance2-view unpadded heads. L1 contract minh bạch numerical parity=false, LayerDrop per-parameter ledger; T1 kiểm optimizer/grad/RNG/cursor algebra. Native resource profile trước đó có actual measurements (không chạy lại). | **OPEN.** Actual export3150 full-state verify đã đạt lúc17:19 (receipt bên dưới); vẫn cần seed/RNG/LayerDrop/length/mask/gradient comparisons cho numerical paper-parity claim phù hợp. Source-sized/synthetic tests không thay native IF. True batch32 BN statistics không được suy từ physical SSL batch32. Profile speedup không đồng nghĩa learning trajectory parity. Profile cũ chỉ kiểm loss eval_without_augmentation:7.7527542 vs7.7524042 (abs3.50e-4); không phải gradient/train-mode/multi-step parity. |
| **F7 metric/selection/decoding** | P1 §6.3/Table2:TR là positive, F1=2TR/(2TR+FR+FA), COR=1−(S+D)/N. New Table2 fixture cho TA1/FR1/FA1/TR2/CD1/ED1 và F1=2/3. E1 replay validation38–55 F1/counts khớp; epoch55 TR1143/FR1079/FA857→54.144955%. Cùng log oldmetric39.284954%, **không phải cùng metric**. Existing fixtures chứng minh newmetric có thể bỏ spurious hypothesis insertion và undefined denominators phải explicit. | **OPEN.** **Sai khác mới được test:** trainer `kaggle_optimized_train.evaluate` tính PER với hypothesis giữ sil; offline `validation_report.score_records` bỏ sil cả hyp/reference. Fixture cùng hyp `[sil,ae,t]`,ref `[ae,t]`:online50%,offline0%. Chưa đo ảnh hưởng lên actual epoch55 vì không có per-utterance hypotheses export; không tính lại điểm bằng policy khác. Cần khóa/đối chiếu author sil/blank/repeat/error policy, old/new metric, denominator, AM greedy/IF beam10/temp1.1/AMweight và validation selection. Không sửa lịch sử/recipe để làm điểm đẹp. |

## Cập nhật C0 lúc17:20 +07

`WINDOW001_VERIFICATION.json` và `OPERATIONAL_FINAL.json` xác nhận3150 đã tải và
full-state verify (Adam/steps, scheduler/scaler, RNG2GPU, pendinggrad/cursor, TRAINorder),
SHA`47fbf1d91ab0838970b05d9d0af67d224c9c8294e3f3950c5c9069414ab5ceaf`.
Prerequisite preserve/export **cho window001** đã đạt trước khi controller có sẵn nối
window002. Đây là scoped operational PASS, **không đóng F6 numerical paper parity**.
Window002 API RUNNING, initiallog bootstrap, chưa có optimizer update mới được audit xác nhận.

## Những gì đã kiểm được, nhưng không đóng fidelity

1. Source hashes65files + manifest hashes + authorized population invariant: PASS ở mức integrity.
2. Paper Table2 expected-output fixture, no-gold teacher inference, teacher gradient paths,
   optimizer migration/resume algebra: PASS trong phạm vi CPU tests.
3. Data split/unknown labels/DEV provenance/augmentation/stride/IF schedule và metric-policy
   parity: còn OPEN hoặc BLOCKED; không có gate nào được promote thành CLOSED.
4. Không rerun performance benchmark trên local GPU. Median profile cũ5.0276s/update
   và7.742× chỉ là registered scratch comparison; captured production-tail median3.5179s
   không phải phép A/B mới, không gồm validation/checkpoint/IO overhead đầy đủ.

## Các bước tiếp theo có thứ tự

1. Bảo toàn/export/hash/load/Adam/RNG/cursor/order/pending-gradient verify window001 **đã đạt3150**.
   Giữ latestparent3150, không rollback482; áp dụng gate tương tự cho window002 và
   không bypass quota nếu admission sau không được cấp tài nguyên.
2. Lấy/xác nhận author manifest và preprocessing **ở mức metadata trước**, không xin
   quyền mở toàn bộ sealed data. Chưa liên hệ tác giả/thay mặt user gửi thư.
3. Chuẩn bị câu hỏi cụ thể: exact utterance lists; star/error/sil normalization;
   timewarp80 vs waveform perturbation; Conformer stride3; IF attention/CNN/branch aliases;
   omega1 và200epoch allocation; metric silence policy/old-new metric/selection decoder.
4. Đóng expected-output policy fixtures/effective config trước optimizer campaign faithful mới.
   Không stage production IF hay clean paper-split training khi prerequisites chưa đóng.

## Replay

```sh
.venv-crottc-official/bin/python tools/audit_crottc_fidelity.py \
  --live reports/CROTTC_IF_KAGGLE_20261114/AUDIT_1705/LIVE_STREAM_1707.log \
  --output /var/tmp/crottc_fidelity_replay_unique.json
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv-crottc-official/bin/python -m pytest -q \
  tests/test_crottc_fidelity_audit.py tests/test_crottc_validation_report.py \
  tests/test_crottc_if_components.py tests/test_crottc_if_train.py \
  tests/test_crottc_optimized_train.py tests/test_crottc_optimized_campaign.py
```

Output replay phải dùng path mới (exclusive creation); không overwrite historical evidence.
