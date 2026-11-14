# GOAL 1 — Tái lập đầy đủ paper CROTTC-IF

**Decision:** `RESEARCH-GOALS-SPLIT-20261114-v1` — user yêu cầu tách hai GOAL rõ ràng.
**Vai trò:** tái lập baseline/paper; độc lập với GOAL phương pháp SM13.
**Trạng thái khoa học:** `FULL_PAPER_NOT_REPRODUCED — FIDELITY_GATES_OPEN`.
**Trạng thái vận hành tại snapshot 2026-11-14 16:46 +07:** AM CROTTC trên protocol local đang chạy Kaggle; không dừng/restart do việc tách tài liệu.
**Điều phối:** [GOAL.md](GOAL.md). Nhánh còn lại: [GOAL_SM13.md](GOAL_SM13.md).

## 1. Mục tiêu và cách gọi kết quả

Tái lập kiến trúc, dữ liệu/preprocessing, training recipe, các stage, decoding/evaluator và các thí nghiệm trong paper CROTTC-IF. Phải phân biệt:

1. **Release inference:** chạy đúng weights chính thức; smoke không chứng minh benchmark hay training reproduction.
2. **Implementation/training trên protocol local:** thực nghiệm phát triển; có thể hữu ích nhưng chưa phải tái lập đúng benchmark paper.
3. **Faithful main-result reproduction:** dữ liệu, lineage, recipe và đánh giá đã qua gate, hoàn thành model chính.
4. **Full-paper reproduction:** thêm baseline, ablation, phân tích và các corpus/thí nghiệm bổ sung đã đăng ký.

**Lượt đang chạy chỉ thuộc mức 2.** Không gọi nó là “đã tái lập full paper”, kể cả nếu hoàn thành 300 epoch. Việc ghi nhận sai khác không tự biến sai khác thành tương đương. Không lấy F1 công bố 71,77% làm điểm local.

Reproduction không đòi ra đúng từng chữ số 71,77%. Nếu protocol thực sự faithful mà kết quả khác, phải báo kết quả, uncertainty và điều tra; không đổi metric/split hoặc chọn run đẹp để đạt điểm tác giả. Thiếu fidelity thiết yếu thì kết luận phải là chưa xác lập exact reproduction, không phải PASS.

## 2. Nguồn chính và provenance

| Nguồn | Vai trò |
|---|---|
| [Paper ISCA](https://www.isca-archive.org/interspeech_2026/geng26_interspeech.html), [bản arXiv đã được audit trước](https://arxiv.org/html/2604.22133v1) | Phương trình, kiến trúc, recipe, bảng/figure; cần khóa phiên bản dùng khi đối chiếu |
| [Secondtonumb/IF-MDD](https://github.com/Secondtonumb/IF-MDD) | Source chính thức; pin commit/receipt, không mặc định mọi YAML khớp paper |
| [Released model](https://huggingface.co/Haopeng/CROTTC-IF-l2-arctic/tree/9149e5107b3332b9aa3f5573f6328a4e312006ad) | Revision đã kiểm integrity; không thay thế bằng chứng train từ khởi tạo native |
| [Báo cáo release](reports/CROTTC_IF_OFFICIAL_20261113/REPORT_VI.md) | Strict loading, 3-câu audio-only smoke, runtime và giới hạn |
| [Preregistration](reports/CROTTC_IF_FULL_LOCAL_20261113/PREREGISTRATION.md) | Quyết định recipe, population và fidelity gaps gốc |
| [Quyền Kaggle](reports/CROTTC_IF_KAGGLE_20261114/DECISION.json) | Nền tảng full CROTTC: Kaggle, không tự resume local |
| [Full-paper mandate](reports/CROTTC_IF_KAGGLE_20261114/OPTIMIZED_FULL_TRAINING_MANDATE.md), [coverage](reports/CROTTC_IF_KAGGLE_20261114/PAPER_COVERAGE_CURRENT.csv) | Phạm vi không chỉ AM training; CSV là snapshot lịch sử, không phải live progress |

Source/native/data receipts ở `experiments/crottc_if_full_local_20261113/`; bundle inference ở `experiments/crottc_if_official_20261113/`. Không sửa checkpoint, source upstream hay receipt lịch sử để hợp thức hóa một thay đổi mới.

## 3. Trạng thái hiện tại — snapshot có bằng chứng

### Diagnostic repair22:58 — v1FAILED preserved, explicitly authorized v2 CPU-only launched

- v1 đãERROR ởgradientforward:SpeechBrainRoPEcache được tạo tronginference_mode ởcleanprofile,
  sau đóautogradkhôngsaveđượccachedindex. Đây làlỗicôngcụdiagnostic, **không phải trainingbug**;
  không cócomplete2399cleanmetrics/6gradientprobes ởv1. Failedlogs/source/receiptsgiữnguyên.
- Userexplicitlyyêucầuthựchiệnsửatest/diagnostictrướccanthiệp. Repair
  `CROTTC-PLATEAU-DIAGNOSTIC-NOGRAD-REPAIR-20261114-v2`: cleanprofiletorch.no_grad
  thay inference_mode đểmemoizedconstantslànormaltensors; output numericalunchanged,
  gradienttoytestfinite/nonzero/weightsunchanged. **26CPUtests passed/1strictxfail**.
- Private [plateau-diagnostic-cpu-v2](https://www.kaggle.com/code/minkhoaa/langfens-crottc-plateau-diagnostic-cpu-v2)
  v1pushedonce/serverCPUprivacy/scriptreadbackverified,API22:57RUNNING/bootstrap;
  nativeprofile/probes/cleanpopulations vẫnPENDING đếnkhicóbằngchứngthật.
  Same5672/1800+599/6fixedprobes/budgets,0optimizer;khôngproductionrecipe/dependencychanges.
- [Repairprotocol/provenance](reports/CROTTC_IF_KAGGLE_20261114/PLATEAU_DIAGNOSTICS_REPAIR_20261114/PROTOCOL.md).
  FiniteretrievalPID1975664 at`PLATEAU_DIAGNOSTICS_REPAIR_20261114/retrieval_v2/`;
  oldPID1952258terminalfailednormally,noautothirdattempt. Mainremainsdiskfloorstopped,
  verified5672/pendinggradients intact;khôngrestart/hạfloor/xóacheckpoints.

### Plateau diagnostics22:28 — user authorized error/cleanloss/componentgrad pipeline (historical v1)

- **Mainwindow002đãCOMPLETE/export5672FULLSTATEVERIFIED**,internalepoch99/cursor928,
  pending32/32TRAINexamples chưaoptimizerstep đãgiữđủ. SHA
  `8d095b0f43cd1b1514efd548c4ec8416bf98938017791d662547c8d8a572695b`.
  Controller1291861đãterminal **STOPPED_DISK_FLOOR_PARENT_PRESERVED** trướcwindow003,
  khôngphảidiagnosticstopnó. Khônggọicampaignvẫnrunning;khôngresume/xóa/hạ14GiBfloor.
- Decision `CROTTC-PLATEAU-DIAGNOSTIC-20261114-v1`, [protocol](reports/CROTTC_IF_KAGGLE_20261114/PLATEAU_DIAGNOSTICS_20261114/PROTOCOL.md).
  [Text-onlyerroranalysis3150đãxong](reports/CROTTC_IF_KAGGLE_20261114/PLATEAU_DIAGNOSTICS_20261114/ERROR_ANALYSIS_VI.md),
  countsoverall/per-speakerexactmatchverifiedreplay; khôngstrictS/O/Iclaim.
- Mộtprivate **CPU-only** Kagglekernel
  [plateau-diagnostic-cpu-v1](https://www.kaggle.com/code/minkhoaa/langfens-crottc-plateau-diagnostic-cpu-v1)
  v1pushedonce/serverreadbackprivacy/sourceverified; API22:41RUNNING/bootstrap,
  **chưacóclean2399metrics/6nativegradientprobes**. Profile/admissiongates trướcfullrun.
  Budget12000driver/14400platformsec,2threads,fullnative/nooptimizer/noGPU.
- Diagnosticfixed5672:cleanTRAIN1800/DEV599sameevalpolicy+OTsingleview;6augmentedTRAIN
  OT/CRautogradprobes(3durations×2seeds),gradientnorm/dot/cosine/clipping/transportalpha.
  Khôngđổirecipe/optimizeweights/localnativebackward. GradientCPUFP32singlepairs≠productionbatch32.
- **24CPUtests passed/1strictxfail**, inclui synthetic2399textverification; khôngclaimnativePASS.
  FiniteretrievalPID1952258, `PLATEAU_DIAGNOSTICS_20261114/retrieval_v1/`;
  chỉVERIFIEDkhiactualfullresultsdownload vàtextreplaypass. Trainingablationvẫnpendingevidence/registration.

### Evaluator replay COMPLETE/VERIFIED; preservation trước rerun (21:17)

- [Kết quả đầy đủ](reports/CROTTC_IF_KAGGLE_20261114/EVAL_REPLAY_20261114/RESULTS_VI.md):
  fixed3150/599DEV, **2995predictions/5trials independentlyverified**.
  LegacymeanF1=54.859260%,repair=54.581238%,delta=−0.278022pp;
  repairpredictions exactequal across2seeds. ActualremoteattentionSHAkhớpdiagnosis.
- Lỗievaldropoutđãxácnhận nhưngpairedreplaynàykhôngủnghộnólànguyênnhânchínhcủaplateau60%.
  KhôngrewritehistoricalGPUmetrics;CPUbackend/onecheckpointlimitationsgiữnguyên.
- User yêu cầu lưu mọi bằngchứng vàchạylại;preservationkhôngtựstop/resetcurrentwindow.
  Scope rerun (eval-only/freshtraining/effective recipe) cần xácnhận trướcnewproductionlaunch.

### Eval replay được user đồng ý20:13 — CPU-only Kaggle diagnostic đã launch (historical)

- Decision **`CROTTC-EVAL-REPLAY-20261114-v1`**, [protocol](reports/CROTTC_IF_KAGGLE_20261114/EVAL_REPLAY_20261114/PROTOCOL.md).
  Mộtfixedverifiedcheckpoint3150,599frozenDEV_SEARCH,3legacyseeds/2repairseeds;
  audio-onlysharednativeWavLMmemory,postdecodepinnedscorer/per-speaker/per-utteranceoutputs.
- Private [CPU replay v1](https://www.kaggle.com/code/minkhoaa/langfens-crottc-eval-dropout-replay-cpu-v1)
  đượcpush1lần,serverprivacy/noGPU/sourcehashreadbackverified. API20:28RUNNING;
  nativeCPUprofilemedian/p95/longest **PASSED**,1.43/2.31/3.00s,0visibleGPU,RSS~2.3GB,
  estimatedfeaturetime880s<6000sbudget. Đã25/599features lúc20:28,
  **chưa có actual599DEVresult**.7200driver/10800platformseconds,2threads,zerooptimizer.
- Không tranhGPU/dừngmainwindow002;không sửa8productiondrivers/dependency/recipe.
  CPUunpadding/cachekháchistorical2T4eval;chỉso pairedarms,khôngrewritehistoricalF1.
- CPUtests **53passed/1strictxfail** (originalevaldeterminismknownbroken). FiniteCLIretrieval
  PID1786236,noGPU/no trainlaunch,at `reports/CROTTC_IF_KAGGLE_20261114/EVAL_REPLAY_20261114/retrieval_v1/`.
  Chỉ gọi VERIFIED khi `VERIFICATION.json` tồn tại và status thành công;không lấyAPIstatus làm kết quả.

### Chẩn đoán plateau19:44–20:00 — eval dropout defect, chưa sửa production

- [Báo cáo chẩn đoán](reports/CROTTC_IF_KAGGLE_20261114/DIAGNOSIS_1945/REPORT_VI.md):
  **SpeechBrain1.0.3 RoPEMHA vẫn apply attentiondropout.1 trong eval** vì functionalSDPA
  nhận dropout_p=self.dropout không kiểm trainingflag. Sameinput/weightsrepeats khôngdeterministic;
  disposableCPUcontrolattentiondropout0 hếtnoise. Chưa đo actualDEVF1delta, khônghứavượt60%.
- **Correction cho giải thích BN cũ:** actualAMheads có **0BatchNorm/14LayerNorm**,
  strictloaded3150; không quy lỗi vềper-utteranceBN.21nativeCNNtensorsgiữnguyên,
  q_proj24SSLTransformer layersđềuđổi. Fullmodel/optimizer cóhọc.
- Logs483–4655:4173updatesfinite/nonzeroprobe;TRAINlossmean41–55=2.1454→76–81=1.7065,
  DEVF1khôngtăngtươngứng. OTcost/gradientfixturesindependentpass; CRrelativebalancekhácpublicsource
  **110×trongca1unpaddedpair**, chưađoactualcomponentdominance. Không đủkếtluậnuniqueoverfit/capacitycause.
- Tests **58passed/1strictxfail** (evaldeterminismKNOWNBROKEN), khônggọiđãfix.
  Userchỉyêucầuchẩnđoán; khôngsửadriver/dependency/recipe/restarttrain. Nextpriority là
  registeredboundednativeeval-onlyreplay trênKaggle,khôngGPUcompetition/currentruntimehotpatch.
  CurrentAMlocalprotocol/F1–F7vẫnOPEN, test/DEV_CALsealed.

### 3.0 Audit mới 2026-11-14 17:05–17:20 +07 — export3150 verified, window002 được admit

- Private `optimized-window001/v1` đã **COMPLETE** bình thường; driver
  `WINDOW_EXHAUSTED_RESUMABLE` tại **update3150/internal epoch55/cursor480**
  (55 completed epochs, dở human epoch56/300), không phải target300 đã hoàn tất.
- Latest val **epoch55: F1=54.1450%, PER=25.9173%**. BestF1 vẫn epoch46=54.3590%;
  bestPER epoch28=25.2566%; mean51–55 F1=52.7162%/PER=26.6358%.
  [Curve1–55](reports/CROTTC_IF_KAGGLE_20261114/AUDIT_1705/CURVE/CURVE.png),
  [report/limits](reports/CROTTC_IF_KAGGLE_20261114/AUDIT_1705/REPORT_VI.md).
- ControllerPID1291861 tự download/verify; tại17:16 checkpoint còn download,
  đến17:19 **latest independent verified đã3150**, SHA
  `47fbf1d91ab0838970b05d9d0af67d224c9c8294e3f3950c5c9069414ab5ceaf`.
  [Full-state receipt](reports/CROTTC_IF_KAGGLE_20261114/AUDIT_1705/WINDOW001_VERIFICATION.json).
  Controller có sẵn nối `optimized-window002/v1`; API17:19:50 báo **RUNNING**,
  log17:20 ở bootstrap/runtime setup, **chưa xác nhận optimizer update mới >3150**.
  [Snapshot vận hành](reports/CROTTC_IF_KAGGLE_20261114/AUDIT_1705/OPERATIONAL_FINAL.json).
  Quota có thể chặn các admission sau; current window002 đã được admit, không stop vì quota.
- [Fidelity ledger mới](reports/CROTTC_IF_KAGGLE_20261114/AUDIT_1705/FIDELITY_LEDGER.md):
  source65files/2manifests hashverified, paperTable2fixture và **90CPUtests passed**;
  F1–F7 vẫnOPEN. Phát hiện PER online/offline khác silencepolicy; kernel3 hiệnstride1,
  chưa xác lập paperstride3. Truy3unknown-C tokens về rawTRAIN malformed labels,
  không sửa data/recipe. Không stop/restart/resubmit, không đọc test/DEV_CAL.

### 3.1 Campaign phát triển đang được bảo toàn — snapshot lịch sử16:46

| Thuộc tính | Giá trị tại 16:46 +07 |
|---|---|
| Kernel | [optimized-window001](https://www.kaggle.com/code/minkhoaa/langfens-crottc-if-optimized-window001), private version 1, `RUNNING` |
| Controller local | PID **1291861**, `tools/kaggle_crottc_optimized_campaign.py`; chỉ điều khiển CLI, không train CROTTC local |
| Stage/target | `am_crottc`, **300 total epochs**, không phải 300 epoch bổ sung từ checkpoint |
| Live progress | **update 2950, internal epoch 51/cursor 1376**, tức đang học human epoch **52** |
| Completed validation mới nhất | Epoch **51**: detection F1 **51,4233%**, PER **27,4774%** |
| Best F1 đã ghi trong 1–51 epoch | Epoch **46**: F1 **54,3590%**, PER **25,9274%** |
| Best PER | Epoch **28**: PER **25,2566%**; không phải checkpoint best F1 |
| Population | TRAIN **1800/12 speakers**, DEV_SEARCH **599**; test/DEV_CAL chưa mở |
| Compute | 2×T4; WavLM Large DataParallel; FP16 autocast, FP32 loss/master/Adam; physical/effective batch 32 |
| Heads/BN | Xử lý acoustic heads trên cặp two-view không padding của từng utterance; không tuyên bố true paper-batch BN equivalence |
| Checkpoint mới nhất độc lập download/verify | **update 482, internal epoch 8/cursor 832**, không phải update 2950 |
| Full pipeline | AM CTC/IF production và các thí nghiệm paper **chưa hoàn tất** |

Bằng chứng snapshot:
- [Remote/quota snapshot](reports/GOAL_SPLIT_20261114/REMOTE_SNAPSHOT.json).
- [Đường cong đầy đủ](reports/GOAL_SPLIT_20261114/CROTTC_CURVE/CURVE.png), [CSV](reports/GOAL_SPLIT_20261114/CROTTC_CURVE/VALIDATION_CURVE.csv), [summary/source hashes](reports/GOAL_SPLIT_20261114/CROTTC_CURVE/SUMMARY.json).
- [Controller snapshot](reports/GOAL_SPLIT_20261114/CROTTC_CONTROLLER_SNAPSHOT.json); live file hiện hành: `reports/CROTTC_IF_KAGGLE_20261114/OPTIMIZED_CAMPAIGN_02/CONTROLLER_STATUS.json`.
- [Verified parent 482](reports/CROTTC_IF_KAGGLE_20261114/OPTIMIZED_CAMPAIGN_02/resume_smoke/VERIFICATION.json), SHA `7db07c849fd1ca0c0a2f14720212c9f7954e0b197d662d8c7265b05a3e753582`.
- [Long-window registration](reports/CROTTC_IF_KAGGLE_20261114/OPTIMIZED_CAMPAIGN_02/window001/WINDOW_REGISTRATION.json), [launch](reports/CROTTC_IF_KAGGLE_20261114/OPTIMIZED_CAMPAIGN_02/LAUNCH_RECEIPT.json).

**Cách đọc:** internal epoch bắt đầu từ 0; log validation `epoch` là số epoch đã hoàn tất. Live log/remote checkpoint không đồng nghĩa đã tải checkpoint về và kiểm chứng đủ state. Kaggle stream có thể chỉ trả tail: phải merge snapshot đã pin và kiểm liên tục epoch trước khi báo “best”/moving mean. Controller timestamp cũ không thay cho kiểm process, platform và log update thật.

### 3.2 Quota và giới hạn tiếp tục

Read-only SDK `get_accelerator_quota_statistics` tại snapshot báo `timeUsed=31899.255732.0s`, `totalTimeAllowed=21600s`, `timeReserved=0s`, reset **2026-11-21 00:00 UTC**. Raw duration có suffix thập phân bất thường: giữ nguyên payload; chỉ dẫn giải leading numeric seconds, không suy quota mặc định 30 giờ.

Current admitted kernel vẫn RUNNING; **không vì quota này mà dừng nó**. Cửa sổ sau có thể bị chặn. Controller hiện hành không phải cam kết có đủ quota cho 300 epoch, càng không phải cả paper. Khi window kết thúc phải ưu tiên download/verify full export; nếu admission/quota lỗi, giữ parent mới nhất và báo BLOCKED, không retry training vô hạn, không rollback và không tự chuyển CROTTC sang local/Colab.

## 4. Fidelity gates — phải giải quyết trước một campaign benchmark đúng paper

| Gate | Hiện trạng có bằng chứng | Điều kiện đóng gate / deliverable |
|---|---|---|
| **F1 — split/population** | Paper 2429 train / 268 val / 900 test; local 1800/599. Paper test: NJS/TLV/TNI/TXHC/YKWK/ZHAA; local TRAIN chứa NJS; local seal không cùng population. | Manifest chính xác theo utterance và speaker, source/version/hash, chống overlap, quyền truy cập riêng. Chỉ khớp counts là chưa đủ. Thiếu manifest tác giả: tìm artifact/liên hệ tác giả hoặc ghi không thể xác lập exact split, không đoán. |
| **F2 — supervision/preprocessing** | 557 câu từng thiếu complete realized sequence được giữ bằng local `err`; không khôi phục realized phones. Có 1110 unknown `err` tokens trong audit IF. Star→err, AX→AH và malformed phones là policy local. | Script/policy tác giả hoặc bằng chứng tái tạo tương đương từ annotation; audit phone inventory, silence, star, deletion và C/S/D/I. Không tự điền gold không tồn tại. Không “fix” bằng âm thầm bỏ câu khó. |
| **F3 — validation labels** | DEV_SEARCH derived thiếu raw silence/star markers. Dùng upstream detection logic không chứng minh annotation-policy parity. | Frozen raw-to-reference pipeline, kiểm mẫu số và các ca lỗi. Đánh giá audio-only; labels chỉ đi vào metric sau decode. Không dùng test để chỉnh mapping. |
| **F4 — augmentation/AM recipe** | Paper time-warp80/time-frequency masking khác public waveform DropFreq/DropChunk/speed95/100/105; current theo source ở phần này. | Xác nhận recipe tác giả hoặc đăng ký rõ interpretation/variant và parity tests; không đổi augmentation giữa campaign rồi đổi tên lịch sử. |
| **F5 — IF architecture/recipe** | Public source/paper có khác attention, CNN trunk, fusion alias, loss omega1=.3/.5 và lịch 200 epoch chưa rõ phân bổ. Component/trainer code có nhưng native IF chưa chạy. | Ma trận paper→source→effective config đã khóa; giải quyết/ghi bất định, kiểm module thật/masks/gradients/decoder, xác nhận epoch allocation trước production IF. |
| **F6 — execution numerics** | Runtime khác official requirements; FP16/DP/physical batching và head grouping đã khai báo. | Profile native, so loss/gradient/valid lengths, lượng hóa sai khác số học, seed/RNG/LayerDrop/optimizer counters; không gọi speedup/parity smoke là quỹ đạo học tương đương. |
| **F7 — metric/selection/decoding** | Local AM audio-only greedy chọn F1 nhưng SpeechBrain1.0.3 RoPEMHA attentiondropout vẫn bật trong eval (diagnosis19:44); chưa sửa production/đo nativeF1delta. Source có default chọn PER và IF teacher-forced validation. | Khóa selection metric, decoder, beam/weights, normalization, denominator và kiểm evaluator bằng fixtures/reference outputs. Phân biệt detection F1 paper với strict S/O/I/exact-gap và metric GOAL2. |

Đã xử lý trong current AM: full WavLM Large, Conformer 2 lớp/kernel3 và chọn objective CR theo phương trình paper. Không liệt kê chúng như thể vẫn chưa được triển khai. Nhưng các sửa này **không đóng F1–F7**.

### Hệ quả lineage bắt buộc

Current checkpoint đã học NJS. Không được tiếp tục checkpoint đó để chứng nhận speaker-disjoint trên paper test có NJS. Nếu F1 cho phép dựng đúng paper split, campaign benchmark phải **khởi tạo lại từ pretrained native sạch**, không “quên” leakage bằng xóa speaker khỏi các epoch sau. Đây là campaign mới với lineage/budget riêng, không phải lệnh xóa hoặc ngắt current campaign.

## 5. Kế hoạch thực thi theo thứ tự

### C0 — Tiếp nhận và bảo toàn phiên hiện hành
- [x] Phân loại đúng: local-protocol AM development, không full-paper reproduction.
- [x] Xác minh live kernel/controller và checkpoint parent độc lập tại snapshot.
- [ ] Để window đang được user cho phép tiếp tục; không stop/restart/reset vì SM13 hay sửa docs.
- [x] Download/hash/load/optimizer/RNG/cursor/order/pending-gradient verify export window001
  tại3150 trước continuation window002. Các window sau vẫn phải qua cùng gate.
- [ ] Kiểm quota/disk thực tế; giữ failed/interrupted provenance. Không khẳng định “đang chạy” nếu chỉ QUEUED.

### C1 — Đóng fidelity ở mức metadata/code/evaluator, không cần ngắt current runtime
- [x] Dựng fidelity ledger F1–F7 với primary-source evidence, unresolved questions và acceptance cụ thể:
  [audit17:05](reports/CROTTC_IF_KAGGLE_20261114/AUDIT_1705/FIDELITY_LEDGER.md).
  Ledger hoàn tất không đồng nghĩa các fidelity gates đã đóng.
- [ ] Reconstruct/obtain đúng manifest và preprocessing; nếu cần truy cập vùng đang niêm phong, xin quyền protocol **cụ thể**, chưa extract/stage/evaluate trước.
- [ ] Kiểm evaluator, model modules, losses, teacher labels, attention/masking và augmentation bằng tests có expected outputs.
- [ ] Đăng ký effective config/schedule và ngân sách sau khi profile, trước optimizer update campaign mới.

### C2 — Faithful main training, chỉ sau prerequisites

| Stage | Quan hệ cha/con | Lịch phải xác nhận |
|---|---|---|
| AM CTC | native sạch, đúng paper population | 300 epoch theo paper |
| AM CROTTC | native sạch, đúng paper population | 300 epoch theo paper |
| IF CTC initialization | completed validation-selected AM CTC + LM/teacher | Paper nói 200 LM/teacher epochs; phân bổ stage phải giải quyết |
| IF CROTTC transfer | IF initialization + **toàn bộ** selected AM CROTTC, freeze policy đúng | 200/stage trong local contract chỉ provisional, chưa author-verified |

Không rút dataset, layers hay lịch để gọi compact/pilot là full. Native IF resource/backward/resume/export gates phải chạy thật; synthetic memory unit tests không thay thế. Mỗi parent có hash/selection history, child strict-load và transfer/freeze receipt.

### C3 — Main benchmark và full coverage
- [ ] Chốt model/decoder/operating parameters bằng đúng permitted validation, không bằng test.
- [ ] Chỉ chấm test sau khi quyền đánh giá riêng và data/fidelity gates được đóng; report counts, F1/P/R/FRR/FAR/DER/PER/COR theo định nghĩa gốc, kèm uncertainty phù hợp.
- [ ] Table4a: CTC, CRCTC, OTTC, CROTTC.
- [ ] Table4b/main: CTC-IF, bỏ encoder fusion, bỏ decoder fusion, bỏ error head, CTC-LM, CROTTC-IF.
- [ ] Table3/4c: LLaMA/Qwen, basic/canonical/potential-pronunciation; oracle error-position phải tách nhãn, không gọi audio-only deployment.
- [ ] Figures4–6: posterior, teacher attention ablations, joint-decoding AM-weight analysis; example/grid khóa trước, không cherry-pick test.
- [ ] Table5: SO762/ERJ fine-tuning và zero-shot theo điều kiện paper; cần corpus/labels/split/permission trước.
- [ ] Table7: Iqra'Eval2 và decoding-weight conditions; không tự provision corpus/TTS mới.
- [ ] Bảng coverage mới trỏ vào receipts/checkpoints/metrics cho từng hàng. Có hàng thiết yếu NOT_REPRODUCED thì không tuyên bố full-paper complete.

## 6. Ranh giới và operational contract

- CROTTC full training **Kaggle**; local chỉ code/tests/audit/package/CLI/artifact verification cho nhánh này. Không tự resume local checkpoint124; SM13 local là quyền riêng của GOAL2.
- Preserve weights, Adam moments/steps, scheduler/scaler, Python/NumPy/Torch/CUDA RNG, cursors, pending gradients và TRAIN order; checkpoint atomic mỗi epoch/window và cadence đã đăng ký. Không drop gradient/batch để vượt resume gate.
- Current bounded window: 14400 driver seconds /16800 platform seconds; max64 windows chỉ là trần controller, không quota bảo đảm. Disk floor local download14GiB; không xóa checkpoint/artifacts để lấy chỗ.
- Dữ liệu hiện được phép: manifest-allowlisted TRAIN/DEV_SEARCH. **Official test và DEV_CAL sealed**. Không stage/extract audio/labels test, toàn archive, credentials hay toàn data tree. Giữ archive Drive nguyên vẹn; đọc training từ runtime disk.
- Không tự mở sweep, thay platform, thêm local CROTTC, cron/timer/subagent hay commit/push/merge. Không động runtime chính để nhường GPU cho SM13.
- Thay đổi đường dẫn/runtime/placement/recipe luôn là provenance mới; không sửa receipt lịch sử.

## 7. File map để tiếp nhận

| File/thư mục | Dùng để làm gì |
|---|---|
| `tools/kaggle_crottc_optimized_campaign.py` | Controller current AM, bounded windows, immutable parent/download/verify/attach |
| `research/crottc_if_official/kaggle_optimized_train.py` | Native optimized AM driver |
| `research/crottc_if_official/kaggle_native_batch.py` | Batch/normalization/OT grouping/gradient export |
| `tools/kaggle_crottc_run/bootstrap.py` | Template; **current remote script là immutable version đã push**, không tự nhận edit local |
| `tools/verify_crottc_kaggle_export.py` | Full-state/order verification |
| `research/crottc_if_official/prepare_full.py` | Current local annotation policy — phải audit, không mặc định author-equivalent |
| `research/crottc_if_official/if_components.py`, `research/crottc_if_official/if_train.py` | IF component/trainer implementation, chưa production evidence |
| [IF component contract](reports/CROTTC_IF_FULL_LOCAL_20261113/IF_COMPONENT_REGISTRATION.md), [IF trainer](reports/CROTTC_IF_FULL_LOCAL_20261113/IF_TRAINER_PROTOCOL.md) | Source/paper gaps, stage parents, loss/freeze/validation contracts; một số backend mô tả lịch sử FP32/local, chưa có optimized IF adapter/profile |
| [Optimization protocol](reports/CROTTC_IF_KAGGLE_20261114/RESOURCE_OPTIMIZATION_PROTOCOL.json), [handover](reports/CROTTC_IF_KAGGLE_20261114/RESOURCE_HANDOVER_STATUS.md) | Benchmark/numerical changes và migration evidence, không chứng nhận recipe parity |
| `reports/CROTTC_IF_KAGGLE_20261114/OPTIMIZED_CAMPAIGN_02/` | Live controller status, attachment/launch/registrations và export receipts |
| `/var/tmp/langfens-crottc-full-20261113-t_car8m4/kaggle_optimized_campaign_02/` | Local downloads/state; không đưa weights vào Git |
| [HANDOFF.md](HANDOFF.md) | Entry point vận hành; đọc update mới nhất, không dùng snapshot cũ làm live status |

## 8. Điều kiện hoàn tất và quan hệ khóa luận

Hoàn tất GOAL1 khi có đúng protocol/lineage, tất cả coverage thiết yếu có bằng chứng thực nghiệm tái chạy được, fidelity/limitations được báo trung thực, và package kết quả đủ để kiểm chứng độc lập. Thiếu artifacts tác giả hoặc corpus/quyền truy cập có thể làm full exact reproduction **BLOCKED**, không được tự nâng best-effort thành full.

Đây là nền đối chứng, không tự là đóng góp phương pháp của khóa luận. Cải tiến SM13 thuộc GOAL2; hoàn thành một GOAL không chứng nhận GOAL kia hay toàn thesis. Bối cảnh thesis và đánh giá feedback vẫn ở [GOAL.md](GOAL.md), không tự mở campaign ALM khác.
