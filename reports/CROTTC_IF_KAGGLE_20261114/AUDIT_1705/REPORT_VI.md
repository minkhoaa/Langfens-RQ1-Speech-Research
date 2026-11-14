# Kaggle AM CROTTC — benchmark phát triển và fidelity audit

Snapshot bắt đầu **2026-11-14 17:05 +07**; trạng thái cuối xem mục vận hành bên dưới.
**Không phải full-paper reproduction; test và DEV_CAL vẫn sealed.**

## 1. Campaign và trạng thái vận hành

- Kernel thật: [optimized-window001/v1](https://www.kaggle.com/code/minkhoaa/langfens-crottc-if-optimized-window001).
- API ban đầu RUNNING, sau đó **COMPLETE**; driver kết thúc **WINDOW_EXHAUSTED_RESUMABLE**,
  không phải hoàn tất target300, không phải bị audit dừng.
- Kết thúc **update3150, internal epoch55/cursor480**, tức đã hoàn thành55epochs và
  đang dở humanepoch56; pending accumulation0. Runtime driver14441.15s (bao gồm
  in-flight/finalization grace sau budget14400s). Metadata: `DOWNLOADED_METADATA/STATUS.json`.
- Local controller **PID1291861** có thật; đã tự chuyển sang download/verify export.
  Audit không launch/restart/stop/resubmit training, không tạo controller/retrieval trùng.
- **Tại17:14: download full weights/Adam còn đang chạy; latest independent verified
  checkpoint vẫn482.** Không được dùng STATUS/EPOCHS/receipt metadata để chứng nhận3150.
  `OPERATIONAL_FINAL.json` ghi snapshot cuối riêng (xem cập nhật17:20), không thay lịch sử đoạn này.
- Destination `/var/tmp/langfens-crottc-full-20261113-t_car8m4/kaggle_optimized_campaign_02/window001/`;
  disk40GiB trống lúc bắt đầu, đủ floor14GiB. Không xóa checkpoints/artifacts.
- Quota read-only: `QUOTA_1708.json` giữ raw `timeUsed=33047.36.0s`, allowed21600s,
  reserved0, reset2026-11-21UTC. Leading numeric seconds≈9.18h đã dùng vs6h allowed.
  Raw duration malformed được giữ nguyên. Next admission có thể BLOCKED/QUEUED;
  không hứa đủ quota300epochs và không tự fallback local/Colab.

### Cập nhật cuối17:20 +07

- **Full checkpoint3150 đãdownload vàfull-state verified**, không còn chỉ metadata.
  `WINDOW001_VERIFICATION.json`:3798392142bytes,SHA
  `47fbf1d91ab0838970b05d9d0af67d224c9c8294e3f3950c5c9069414ab5ceaf`.
  Verifier kiểm inventory/hash/size,last+best, strict contract, Adam moments/counters,
  scheduler/scaler,2CUDA RNG,pending-unscaled-gradients,cursors vàđúngTRAINorder.
  Receipt field`verified_microbatches=84248` ởcode là sốTRAIN**IDs** đã đối chiếu,
  không phải84248physicalGPUbatches.
- Controller có sẵn nối [optimized-window002/v1](https://www.kaggle.com/code/minkhoaa/langfens-crottc-if-optimized-window002)
  sau gate. API17:19:50 **RUNNING**, log17:20 ởbootstrap/runtime setup;
  **chưa xác nhận optimizer update>3150**, không gọi initial platform admission là tiến độ model mới.
- `OPERATIONAL_FINAL.json` lưu actualstatus/controller/verifiedparent; disk/var/tmp~20GiB.
  Quotaused>allowed không ngăn admission này, nhưng không bảo đảm các window sau.
  Không dừng kernel mới hay đổi nền tảng để xử lý quota.

## 2. Kết quả đo trên DEV_SEARCH599 — không phải paper test

Backbone WavLM Large315453120 params + heads3822765;2-layer Conformer384/kernel3;
objective CROTTC paper interpretation, batch32,2T4, FP16train/FP32validation,
seed3047. TRAIN1800/12speakers, DEV599/4speakers;greedy audio-only AM inference.

| Mốc | Completed epoch | Detection F1 (%) | PER (%) | P (%) | R (%) |
|---|---:|---:|---:|---:|---:|
| Latest validation |55|54.1450|25.9173|51.4401|57.1500|
| Best validation F1 |46|54.3590|25.9274|50.9170|58.3000|
| Best validation PER |28|54.1639|25.2566|48.9310|60.6500|
| Mean latest5epochs |51–55|52.7162|26.6358|—|—|

Epoch55 counts: TA16808,TR1143,FA857,FR1079,CD899,ED244.
F1 count replay=2×1143/(2×1143+1079+857)=54.144955%; CD+ED=TR.
Cùng source có **old-metric F1=39.284954%**; không đổi qua lại giữa hai metric.
PER ở đây theo current online policy, không âm thầm thay bằng offline scorer (xem F7).

### Đường cong đầy đủ

![CROTTC development curve](CURVE/CURVE.png)

- `CURVE/VALIDATION_CURVE.csv`: liên tục1–55, không chọn tail để báo best.
- `CURVE/SUMMARY.json`: nguồn/hashes, best/latest và movingmean.
- `RECONCILIATION.json`: downloaded EPOCHS1–55 khớp merged curve; overlap pinned1–51
  không đổi;8 current implementation hashes khớp remote contract.
- Từepoch1→10 PER giảm nhanh; từ khoảng20–30, PER/F1 cải thiện chậm và dao động.
  Mean F1 theo blocks1–10/11–20/21–30/31–40/41–50/51–55:
  **49.0421/50.0888/51.9124/51.8369/52.1313/52.7162%**.
  Latest chưa vượt best46; không có bằng chứng từ các điểm này rằng cần dừng train
  hay chắc chắn300epochs sẽ vượt benchmark. Đây là nhận xét mô tả, không kiểm định hội tụ.
- DEV được tái sử dụng mỗi epoch, một seed; chưa có independent test/uncertainty
  cho generalization. Không bootstrap các epoch như samples độc lập.

### Benchmark paper và compute: không so ngang trái protocol

Paper v1 Table3/4a báo **CROTTC AM F1=62.39%**; full **CROTTC-IF=71.77%**.
Current stage chỉ AM, chưa IF, split/labels/augmentation/metric parity chưa đóng;
không dùng hai số này làm local result hoặc kết luận chênh lệch apples-to-apples.

Registered resource profile trước đó (không chạy lại): median5.0276s/update,
95% bootstrap interval median[3.1710,6.9158]s,n10 sau2warmups,p95=8.4463s;
reported speedup7.742× so legacy scratch protocol. Nguồn:
`../OPTIMIZED_CAMPAIGN_02/PROFILE_VERIFICATION.json` và pinned profile JSON.
Current captured-tail988updates có median3.5179s,p10=2.4485,p90=5.0886s.
Những step timings này đổi theo utterance duration, không gồm đầy đủ validation/
checkpoint overhead, không phải independent A/B comparison và không chứng minh
numerical/training-trajectory equivalence. Một resource logGPUutil0% tại boundary
không chứng minh GPU idle suốt training; không dùng nó để đổi placement.

## 3. Fidelity audit đã tiếp tục

Chi tiết gate/primary-source evidence/acceptance:
**[FIDELITY_LEDGER.md](FIDELITY_LEDGER.md)**.

- Integrity:65/65 pinnedsourcefiles +2frozenmanifests khớp hashes;TRAIN/DEV ID overlap0.
  GitHub pinned tree79entries,truncatedfalse, không có frozen split/dataJSON manifests.
- Lineage: **NJS150** đã fit trong TRAIN; NJS thuộc paper test. Faithful paper split
  cần clean native pretrained initialization/campaign riêng; không tái chứng nhận current parent.
- TRAIN giữ557câu/1110unknownerr targets;DEV12câu/12err và599derived-only annotations.
  Truy3unknown-C positions về đúng3allowlisted TRAIN TextGrid sau hashverify:
  rawuntagged`Z_`,`ER)`,`D_`→err/err/C. Không sửa nhãn hay đoán phones.
- **F4 mới:** paper nói kernel/stride3;source-sized currentConformer kernel3 có
  Conv1dstride1,15→15frames. Cần author clarification, không tự thêm subsampling.
- **F7 mới:** onlinePER giữ hypothesis`sil`,offlinescorer bỏ;syntheticfixture cùng
  hyp/ref cho50%vs0%. Chưa có actualepoch55hypotheses để lượng hóa ảnh hưởng;
  không thay metric hoặc sửa điểm lịch sử.
- Table2paperfixture khớpTA/FA/FR/TR/CD/ED;IFgradient/masks/no-gold-inference,
  optimizer/resume guards được test. **90passed,10warnings,7.43s** tại
  `TESTS_FINAL.log`;CPUonly,syntheticmemorykhôngphải nativeIFtraining.
- **Không gate F1–F7 nào được đóng hoàn toàn.** NativeIFproduction/AMCTC/fullpapercoverage
  vẫn chưa hoàn tất; không dùng docs/tests thay thực nghiệm.

## 4. Artifact / replay

`tools/audit_crottc_fidelity.py` đọc metadata/source/logs, không model/audio/archive;
`FIDELITY_EVIDENCE.json` lưu số liệu và hashes. RawTRAIN3case audit riêng tại
`F2_UNKNOWN_C_AUDIT.json`; không extract/stage audio.

Lệnh replay và scopedtests ở cuối ledger. Không commit/push/merge hay đổi recipe.
