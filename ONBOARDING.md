# LANGFENS — Onboarding cho nhóm nghiên cứu

> **Đọc đầu tiên.** Đây là bản đồ lịch sử và trạng thái dự án, không phải bằng chứng mọi experiment đã PASS.
> Snapshot tài liệu: **2026-11-14, Asia/Ho_Chi_Minh**. Không tự khởi chạy training khi clone repo.

## 1. Dự án làm gì?

LANGFENS nghiên cứu **Mispronunciation Detection and Diagnosis — MDD**: từ audio người học và câu/phone reference cần đọc, phát hiện và phân loại **substitution (S), omission (O), insertion (I)**, hướng tới phản hồi sửa phát âm có căn cứ âm học.

Dự án đã đi qua nhiều vòng baseline, teacher/student, fusion, evaluation audit và SM13. **Hiện chỉ có hai GOAL hoạt động**:

| GOAL | Câu hỏi | Điều hành |
|---|---|---|
| **GOAL1 — CROTTC-IF** | Tái lập paper và baseline theo đúng data/recipe/metric/stages | [GOAL_CROTTC_IF.md](GOAL_CROTTC_IF.md) |
| **GOAL2 — SM13** | Backbone lớn hơn có cải thiện phát hiện lỗi trên cùng protocol không? | [GOAL_SM13.md](GOAL_SM13.md) |

[GOAL.md](GOAL.md) là index và bối cảnh thesis; [SIDE_GOAL.md](SIDE_GOAL.md) chỉ là compatibility pointer. Các kế hoạch EXP/Phase/M0–M6 cũ là **lịch sử**, không phải hàng đợi được phép tự chạy lại.

**Thesis chưa hoàn tất.** Có metric tốt hơn hoặc reproduce được một model không đồng nghĩa phản hồi phát âm đúng, có ích hoặc người học tiến bộ.

## 2. Đọc theo thứ tự nào?

### Trong 15 phút

1. Tài liệu này.
2. [AGENTS.md](AGENTS.md): ranh giới dữ liệu, platform, tài nguyên, provenance.
3. [GOAL.md](GOAL.md), sau đó GOAL tương ứng.
4. Phần mới nhất của [HANDOFF.md](HANDOFF.md): đọc từ trên xuống; **snapshot mới thắng running header cũ**.
5. [docs/EXPERIMENT_CATALOG.md](docs/EXPERIMENT_CATALOG.md): tìm đúng họ experiment và report.

### Khi cần kiểm một con số

`report → protocol/split/metric → VERIFICATION receipt → prediction/checkpoint hash → artifact được cấp quyền`.

Report hoặc file mang tên `FINAL`, `PASS`, `LOCAL_REPRODUCED` **không tự chứng minh** kết luận. Registry lịch sử có placeholder/zero và nhiều protocol; đọc các disposition/correction ở EVAL-V3 và report mới hơn. Phiên onboarding này không verify lại toàn bộ lịch sử.

## 3. Trạng thái thực tế ở thời điểm bàn giao

### GOAL1: chưa full-paper reproduction

- Campaign AM trên protocol local đã export **update5672**, internal epoch99/cursor928; **32/32 accumulated examples và pending gradients được giữ**, chưa optimizer step cho group đó.
- Receipt: `reports/CROTTC_IF_KAGGLE_20261114/OPTIMIZED_CAMPAIGN_02/window002/VERIFICATION.json` (local evidence; not included in docs-only push), status `DOWNLOADED_FULL_STATE_VERIFIED`, checkpoint SHA `8d095b0f43cd1b1514efd548c4ec8416bf98938017791d662547c8d8a572695b`.
- Main controller đã dừng ở disk-floor gate, **không tự restart**, không gọi vẫn đang train. Native CPU plateau diagnostic v2 có launch snapshot riêng; trạng thái cuối phải kiểm ở HANDOFF và retrieval receipt, không suy từ PID cũ.
- Current AM là **local-protocol development**, chưa đóng data/augmentation/metric/all-stage fidelity gates; không gọi faithful paper benchmark hay IF đã hoàn tất. Không lấy F1 paper71.77% làm kết quả local.
- Paired evaluator replay tại checkpoint3150,599DEV: legacy mean F1 **54.8593%**, repaired **54.5812%**; sửa eval-dropout không giải thích plateau trong phép kiểm đó. Không hotpatch production hoặc đổi số lịch sử. [Replay report](reports/CROTTC_IF_KAGGLE_20261114/EVAL_REPLAY_20261114/RESULTS_VI.md).

### GOAL2: normalized300 đã xong, dropout control chưa train

| Experiment | Population/metric | Kết quả và giới hạn |
|---|---|---|
| S0 original W2V2,50/head |1350fit/450held TRAIN; neural-only |102states verified; macro3AP33.4066%; không full-fusion result |
| S1 native WavLM Large raw50 |Cùng split; fixed.5 event detection |F1 **38.6507%**, macro3AP25.7442% |
| S1 raw350 |Continuation cùng recipe |404states verified; F1 **37.3146%**; không epoch-driven gain |
| Fresh raw50 vs fit-RMS normalized50 |Paired init/seed/data/recipe |F1 **38.6507→40.8143%**;6gates passed;204states verified |
| Normalized300 |Resume exact normalized50; fixed endpoint |602states verified; F1 **40.8843%**, macro3AP33.8588%; không F1 gain bền vững từ thêm epoch |
| Normalized baseline vs dropout×2 |Fresh paired50/head/arm được chuẩn bị |Driver/verifier đã có; **chưa có candidate training/result** |

Nguồn: `reports/SM13_EPOCH50_20261114/LOCAL_VERIFICATION.json` (local evidence; not included in docs-only push), [paired/300 results](reports/SM13_RAW_NORMALIZED_20261114/RESULTS_VI.md), `reports/SM13_RAW_NORMALIZED_20261114/FINAL_VERIFICATION.json` (local evidence; not included in docs-only push).

CPU audit đo lại pooled fitF1 **96.6907→99.6100%**, heldF1 **40.8143→40.8843%** từ normalized50→300. Target held CE1.6889→3.0304; model tự tin sai hơn. [Diagnosis](reports/SM13_NORMALIZED_DIAGNOSIS_20261114/REPORT_VI.md).

**Dropout runtime hoạt động đúng.** Chưa chứng minh p hiện tại gây overfit; tăng dropout/label smoothing/R-Drop đều chưa có quality outcome trên run này. [Causality audit](reports/SM13_DROPOUT_CAUSALITY_20261114/REPORT_VI.md). User yêu cầu **tái hiện và đối chứng trước khi quyết định**, không thay bằng một danh sách recommendation.

- Experiment hiện tại: [SM13_DROPOUT_PAIRED_20261114](reports/SM13_DROPOUT_PAIRED_20261114/STATUS_VI.md).
- Lần thực thi22:38 bị disk gate chặn trước GPU/training. Recheck23:21 chỉ qua **CPU storage/RAM admission**, GPU profile/lock và paired fit vẫn pending. `reports/SM13_DROPOUT_PAIRED_20261114/LATEST_ADMISSION_STATUS.json` (local evidence; not included in docs-only push).
- Việc dung lượng đủ trở lại **không tự resume/launch** trong task documentation này.

## 4. Kiến trúc: những tên dễ nhầm

**SM13 dependency declaration đầy đủ:** [docs/SM13_DEPENDENCIES.md](docs/SM13_DEPENDENCIES.md) — từ planning, data/alignment, runtime/backbone/cache đến training, benchmark và verification; gồm SHA và ranh giới Git/private storage.

### SM13

**13 =13 hidden representations, không phải13epochs.**

```text
Audio + supplied intended phones
  → predicted alignment
  → frozen SSL backbone
  → per-layer pooled target/gap features
  → scalar mixing + small target/gap heads
  → p(C/S/O) ở N targets; p(I) ở N+1 exact gaps
```

- **S0:** `facebook/wav2vec2-base-960h`,13×768; ASR-fine-tuned upstream, frozen trong SM13.
- **S1:** native pretrained WavLM **Large**,24Transformer layers/1024D; chọn `[0,2,…,24]` thành13representations;315,453,120backbone params, frozen.
- **S1 normalized:**13layer RMS scales fit **chỉ trên fit speakers**, frozen buffers trong fresh heads; không đổi historical cache/head.
- **S2 conditional:** CROTTC-adapted WavLM; chưa có quyền bỏ qua parent/lineage/leakage gates. Update482 trước đây chỉ smoke parent, không final selected model.
- Các run50/350/normalized300 hiện tại là **neural-only**. Full historical SM13 có thêm CTC/E4 và fusion; không gắn nhãn full fusion cho neural-only results.
- Frozen backbone không nhận optimizer updates. Train head300epochs từ fixed cache **không phải** fine-tune WavLM300epochs.

### CROTTC-IF

Native WavLM Large + acoustic/sequence components với CROTTC/IF stages. Source chính thức và model release có lineage riêng; xem [GOAL1 sources](GOAL_CROTTC_IF.md). Release-inference smoke, local training và faithful full-paper reproduction là ba mức evidence khác nhau.

## 5. Dữ liệu, splits và metric — không được trộn

### Các population thường gặp

| Protocol | Fit / calibration / evaluation | Dùng cho |
|---|---|---|
| Legacy Fold1 RQ1-TRN-02A |2399train,300val(NJS/RRBI),900test sealed |Thử nghiệm W2V2 CTC+Binary đời đầu; không phải split GOAL2 |
| Local common-development |TRAIN1800/12speakers; DEV_CAL300; DEV_SEARCH599; test900 |Nền tảng nhiều benchmark sau; active tasks chỉ dùng split được phép |
| TRAIN speaker-grouped OOF |TRAIN1800 chia4nhóm,3held speakers/fold |Method/encoder benchmarks lịch sử |
| Current SM13 calibration |1350fit/9speakers;450held ABA/ASI/BWC |Reused TRAIN development, không independent confirmation |

TRAIN1800 có59,562targets và61,362gaps. Các protocol/paper có thể khác người nói dù số câu trùng. **NJS đã được current CROTTC học**, nên không thể dùng checkpoint đó để chứng nhận speaker-disjoint paper test cóNJS.

**Official test/DEV_CAL hiện sealed cho công việc mới.** Một số report lịch sử có evaluation được cấp quyền riêng; đó không phải quyền mở lại dữ liệu. Không chạy script cũ chỉ vì tên có `test` hoặc `final`.

### Metric không tương đương nhau

- **AP-S/O/I, macro3AP:** ranking; phải giữ đúng scores và population.
- **Typed macroF1:** phân loại đúng từng S/O/I; khác pooled binary detection.
- **SM13 event detectionF1@.5:** target positive nếuS hoặcO; score `p(S)+p(O)`, gapscore `p(I)`; pooledTP/FP/FN trên cùng keyedN+N+1population.
- **CROTTC hierarchical detectionF1:** phone alignment/decoding evaluator; chưa được dùng thay tên cho SM13 eventF1.
- **PER:** lỗi nhận dạng chuỗi, không detectionF1.
- **EVAL-V3 FWR:** wrong-type-inclusive false warnings/câu; khác total warnings/câu hoặc chỉFP trênCORRECT.
- MeanfoldAP khác pooledAP; nominalFWR budget khác achievedFWR. Missing/unsupported metric là undefined/NOT_ESTIMABLE, không thay bằng0.

**Không có một leaderboard hợp lệ gộp tất cả số trong repo.** Phải match metric, cohort, reference information, checkpoint/threshold selection và training exposure trước khi so.

## 6. Lịch sử nghiên cứu và những gì đã học

Các hàng dưới là bản đồ đọc **report lịch sử**, không chứng nhận mới trong phiên onboarding.

| Giai đoạn / họ experiment | Đã khảo sát | Kết luận cần giữ / đường dẫn |
|---|---|---|
| RQ1-AUD/STD/SPLIT/PREP/TRN |Corpus integrity, phone inventory, speaker splits, loaders, W2V2 CTC+binary |Contract nền tảng; `manifests/` và `src/langfens_prep/`; không dùng denominator cũ như metric active mặc định |
| EXP001–100 và các đợt tiếp theo |Priors, mismatch/GOP, fusion/router, temporal/articulatory/confidence variants |Xem [first consolidated report](reports/LANGFENS_EXPERIMENT_FIRST_CONSOLIDATED_REPORT.md), [detailed synthesis](reports/LANGFENS_DETAILED_SYNTHESIS_REPORT.md), [scientific review](reports/SCIENTIFIC_REVIEW_EXP100.md). IDs/registry gates không tự là audited gains |
| EVAL-V3 closure |Chuẩn hóa replay, scores, calibration, event IDs và audit historical routing |Report ghi92variants cóCAL+SEARCHreceipts;38methods lịch sử cógold-conditioned routing được gắn nonoracle replacement, không giảhistorical parity. [Closure](reports/EVAL_V3_CLOSURE_REPORT.md), [comparability](reports/EVAL_V3_COMPARABILITY.md) |
| Teacher/student Phase2–4 |S/O/I specialists, complementarity, KD, operating points, composition |Đọc [Phase2](reports/PHASE2_FINAL_SCIENTIFIC_REPORT.md), [Phase3](reports/PHASE3_FINAL_SCIENTIFIC_REPORT.md), [Phase4](reports/PHASE4_FINAL_SCIENTIFIC_REPORT.md); negative results và OOF lineage quan trọng hơn chọn bảng đẹp |
| UPR02/CPL/TMCI |Unified routing, complementarity, top-model capability attribution |[UPR02](reports/UPR02_FINAL_SCIENTIFIC_REPORT.md), [CPL01](reports/CPL01_COMPLEMENTARITY_REPORT.md), [TMCI](reports/TMCI_FINAL_SCIENTIFIC_REPORT.md); coi đúng scope và exposure |
| METHOD-BENCH01–08 |Sequence methods, probabilistic edits, direct exact-gap insertion, speaker validation, evidence fusion |[MB01](reports/METHOD_BENCH_01_FINAL_SCIENTIFIC_REPORT.md), [MB02](reports/METHOD_BENCH_02_FINAL_SCIENTIFIC_REPORT.md), [MB03](reports/METHOD_BENCH_03_FINAL_SCIENTIFIC_REPORT.md), [MB04](reports/METHOD_BENCH_04_FINAL_SCIENTIFIC_REPORT.md), [MB05](reports/METHOD_BENCH_05_FINAL_SCIENTIFIC_REPORT.md), [MB06](reports/METHOD_BENCH_06_FINAL_SCIENTIFIC_REPORT.md), [MB07](reports/METHOD_BENCH_07_FINAL_SCIENTIFIC_REPORT.md), [MB08](reports/METHOD_BENCH_08_FINAL.md). MB08 giữfixedneural+E4 và directgap; newgapaligner không đạt adoption gate |
| TRAIN-PHASE01/02 |Modular neural+CTC/E4+gap, class-sensitive warning policies |[Phase01](reports/TRAIN_PHASE_01_FORMAL_FINAL.md), [Phase02](reports/TRAIN_PHASE_02_FINAL_SCIENTIFIC_REPORT.md). Policy improvement không đồng nghĩa representation gain |
| ENCODER-BENCH01 |W2V2 layers, SSL-only vs ASR-pretrained, HuBERT/WavLM Base, scalar mixing/adapters |[Final report](reports/ENCODER_BENCH_01_FINAL.md), [detailed report](reports/ENCODER_BENCH_01_BAO_CAO_CHI_TIET_TOAN_BO_KET_QUA.md). Multi-layer mixing dẫn tớiSM13; WavLM **Base** cũ không phải Large hiện tại; overparameterized adapter là negative result cụ thể |
| RQ1-UNIFIED01/02 |Acoustic training, specialist transfer, unified outputs/ablations |[Unified01](reports/RQ1_UNIFIED_FINAL_RESEARCH_REPORT.md), [Unified02](reports/RQ1_UNIFIED_02_FINAL_SCIENTIFIC_REPORT.md); không tự phục hồi làm active queue |
| Formal SM13, calibration/fusion/asymmetric |13layer heads,3seeds, fixed12target/5gap, inference/calibration/policies |[Formal report](reports/RQ1_SM13_FORMAL_TRAIN_01_BAO_CAO_TONG_HOP_TOAN_DIEN.md), [fusion diagnostic](reports/RQ1_SM13_FUSION_DIAGNOSTIC_01_BAO_CAO_TOAN_DIEN.md), [asymmetric](reports/RQ1_SM13_ASYMMETRIC_FUSION_01_BAO_CAO_TOAN_DIEN.md). FullTRAINtrainloss không là held curve |
| SM13 three-directions,2026-11-13 |Same-speaker pseudo-label data, temporalCNN, short matchedLoRA |[Synthesis](experiments/sm13_three_directions_20261113/SYNTHESIS_VI.md). Không có stable practical gain; rejected preprocessing/precision controls được giữ; không phủ định mọiLoRA/augmentation |
| Realized-phone M0–M6 / three-method pilot/full |Reference/annotation contracts, realized-phone recognition, alternative methods |`reports/GOAL_DECISION_20261113/DECISION.json` (local evidence; not included in docs-only push); các snapshots/report vẫn giữ nhưng không active mandates |
| Thesis decision,2026-11-13 |CROTTC baseline + method evidence + grounded feedback evaluation |`reports/GOAL_DECISION_20261113/DECISION.json` (local evidence; not included in docs-only push). Không tự áp80%P/R hoặcFWR≤.20 làm điều kiện thesis |
| CROTTC release/local/Kaggle,2026-11-13–14 |Strict release loading, local native training, migration/pending-gradient preservation, runtime optimization |[Release report](reports/CROTTC_IF_OFFICIAL_20261113/REPORT_VI.md), [local preregistration](reports/CROTTC_IF_FULL_LOCAL_20261113/PREREGISTRATION.md), [GOAL1](GOAL_CROTTC_IF.md). Local124 và482migration là lịch sử; latestverified5672 |
| Goals split,2026-11-14 |Hai GOAL độc lập; platform và validity boundaries |`reports/GOAL_SPLIT_20261114/DECISION.json` (local evidence; not included in docs-only push). CROTTC Kaggle; SM13localonly |
| S0/S1 extensions / RMS / overfit / dropout-control |12/5→50; raw50→150→350; freshnormalized50→300; CPUaudit; pairedpilot implementation |[GOAL2](GOAL_SM13.md), metrics ở§3 của tài liệu này; dropout pilot chưa có outcomes |

Tìm report và code theo từng thư mục ở [catalog](docs/EXPERIMENT_CATALOG.md). Không lặp lại thử nghiệm đã làm rồi gọi novelty.

## 7. Zero-oracle và reference provenance

Inference không được nhận gold timestamps, realized phones, error labels hoặc gold-conditioned routing. Gold vẫn hợp lệ cho fitTRAIN/metric sau inference theo protocol.

**Supplied intended-phone reference** và **text→G2P reference** là hai information budgets khác nhau. Manifest intended reference hiện có lineage annotation-derived; predicted alignment không tự chứng minh text-only independence. Nếu G2P tạo chuỗi khác, sốNtargets/N+1gaps có thể đổi; không ép output vào gold canonical IDs để tạo parity giả.

Zero-oracle là contract validity, **không phải regularizer** hoặc cách tự động tăngF1. Xem [EVAL-V3 closure](reports/EVAL_V3_CLOSURE_REPORT.md), [GOAL2 data boundary](GOAL_SM13.md).

## 8. Bản đồ code và artifact

```text
src/langfens_prep/              # data preparation / legacy model contract
research/evaluation_v3/         # canonical evaluation/replay/audits
research/sm13_backbone_side/    # active frozen backbone/head/cache/conditioning/pair driver
research/crottc_if_official/    # active CROTTC implementation/runtime/migration/diagnostics
research/realized_phone_pipeline/ # historical realized-phone plan
experiments/                   # experiment-specific source + lightweight reports; weights local only
reports/                       # scientific reports, registrations, verification receipts
manifests/                     # versioned legacy split/audit metadata; not permission to open test
configs/                       # runtime/model configuration
literature/                    # paper/dataset/method research indices
coordination/                  # historical ledgers and shared locks/contracts; not an automatic queue
ONBOARDING.md                   # this human-facing map
GOAL*.md / HANDOFF.md           # goals and timestamped operational handoff
```

### Những gì không nằm trên GitHub

Audio,TextGrid,rawarchives,modelweights,optimizerstates,acoustic/framecaches,per-examplepredictions,venvs,credentials và runtime scratch không được đưa lênGit. Không `git add .` từ laptop nghiên cứu.

Machine-local artifacts thường ở:

- `/var/tmp/langfens-crottc-full-20261113-t_car8m4/`: CROTTC exports/nativeSM13cache/history.
- `/home/khoa/.local/share/langfens/`: normalized SM13weights.
- `.cache/huggingface/`: model assets, có thể cần offline; không coi là disposable chỉ vì tên cache.
- `/home/khoa/.local/share/langfens_maintenance/`: environment freeze/cleanup/restore receipts.

Các đường dẫn trong historical registrations là **identity/provenance**, không portable configuration. Teammate không tự sửa receipt cũ; cần runtime registration mới ràng buộc path/hash mới. Xin quyền truy cập artifacts/dataset riêng và xác minh checksum trước inference/training.

## 9. Môi trường và bước đầu an toàn

### Đừng dùng README/lockfile lịch sử như active runtime

`requirements-lock.txt` ởroot là snapshot lịch sử, **không mô tả đúng current CROTTC/SM13 runtime**. Active owner-machine CPU test environment:

- Python3.11.17; PyTorch2.6.0+cu124; torchaudio2.6.0+cu124;
- transformers4.57.0; speechbrain1.0.3;
- danh sách version hiệu dụng: [environments/research-runtime-20261114.txt](environments/research-runtime-20261114.txt).

File này là **environment inventory**, không phải lock cówheelhash/mộtcleaninstall đã được chứng minh trên máy teammate. Availability trên publicpackageindex/OS/CUDA có thể khác. Không cài đè một env đang chạy để “giống onboarding”.

**Cạm bẫy workstation:** `.venv-crottc-official` củaowner có `.pth` trỏ vào `.venv` chính. Xóa `.venv` làm `torch` biến mất trong officialenv. Trong phiêncleanup, dependencyenv đã cần khôi phục từcachedpayloads; versions/imports và85targetedtests được kiểm, không claim bitwiseparity mọi file củaenv cũ. Teammate nên tạo env độc lập, không copy `.pth` chứa `/home/khoa`.

### Trên clone mới

1. Đọcgoals/safety rồi chọn task. Không launch `--prepare`/training/watchers chỉ để kiểm setup.
2. Tạo Python3.11 environment độc lập; coreGPUdependencies dùngindexPyTorchcu124, cácpackage còn lại theo inventory. Trao đổi nếu đúngversion khôngresolve; khôngâmthầmupgrade.
3. ExportCPU-only khi chạyfixturetests: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`.
4. Scopedunit command dưới đây không cần model/audio, không mở test/DEV_CAL, khôngfitrealdata:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m pytest -q \
  tests/test_sm13_dropout_paired.py tests/test_sm13_dropout_causality.py \
  tests/test_sm13_debug.py tests/test_sm13_conditioning.py \
  tests/test_sm13_detection.py tests/test_sm13_raw_normalized.py
```

Trongworkstationtại23:16:85tests passed sauenvironmentrestoration. Clean-checkoutsafety sẽ được kiểm ở `reports/TEAM_ONBOARDING_20261114/VALIDATION.md` (local evidence; not included in docs-only push); đọc kết quả thực tế ở đó, không suy mọitests PASS. Cótests khác cầnprivateassets/runtime/lock và knownstrictxfail ởCROTTCevaldeterminism. Không hạtolerance/nhảproductionlockđểtestxanh.

## 10. Làm việc cùng teammates

| Workstream | Có thể làm độc lập | Không tự ý làm |
|---|---|---|
| Evaluation/data contracts |Syntheticfixtures,keyuniqueness,metric/provenanceaudit |Mởsealedsets,đổiannotation/denominator/thresholdselection |
| SM13 |Headtests,pairedverifier review,read-onlycurves |SwitchLS/LoRA/fullFT,sweep,retunehistoricalthresholds,extendepochs |
| CROTTC |Paperfidelitymatrix,CPUunit/audit,downloadedreceiptanalysis |Restart/reset/patchcurrentcampaign,migrateplatform,discardpendinggradients |
| Documentation/reproducibility |Updateonboarding/catalog/environment/artifactmap |Commitweights/data/secrets,claimfullreproductionfromsmoke |

Gitworkflow: featurebranch → explicitreviewablefiles → scopedtests → stagedsize/secretcheck → commit → pushbranch → review/PR. Khôngforcepush hoặcmerge`main`mặcđịnh. Userđãủyquyềnpushgóionboardingnày; điềuđókhôngtựủyquyềntraining/dataupload.

SharedGPUlock là `/home/khoa/Projects/Pronunciation-RD-Coordination/locks/gpu.lock` trênworkstationowner. Gitrepo và filesystemsharedlocks làhai thứkhácnhau. Ghi taskowner, fileownership, decisionID, inputs/hash, boundedbudget trướcproduction. Khônghaiagentsđồngthờixóa/rebuildvenv hoặccùnglaunchmộtjob.

## 11. Definition of done cho một experiment

- Protocol/decision, population, informationbudget, metric và endpoint/thresholdselection đã khóa.
- Resource/profile/platform/admission và sharedGPUlease đạt; localSM13≠localCROTTC.
- Checkpoint fullstate atomic/resumable gồm optimizer/RNG/cursors/pendinggradients khi có.
- Predictions keyedđủpopulation; khôngskipfailedcases/dropclasses khó.
- Verifier độc lập/checksums và scope rõ; reportedprogress khác downloadedverifiedstate.
- Report fullcurves/fixedendpoints/per-speaker/S/O/I/falsewarnings, negativeoutcomes/failurerecovery.
- Kếtluận `improved / no clear improvement / regression / not tested` đúng evidence; không cherrypickepoch/seed/metric.
- Source/config/tests/reportnhẹ được review; heavy/privateartifacts ởstorageđượcphép.

**Research success không bắt buộc candidate phải thắng.** Negative result có đối chứng hợp lệ là evidence; recommendation chưa thử không phải kết luận.
