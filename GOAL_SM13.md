# GOAL 2 — Phương pháp SM13 với backbone lớn và đánh giá detection F1

**Decision điều phối:** `RESEARCH-GOALS-SPLIT-20261114-v1`.
**Kế thừa quyền:** `SM13-BACKBONE-SIDE-20261114-v1`, `SM13-EPOCH50-20261114-v1`; **SM13 chạy local, không break tiến trình CROTTC hiện tại**. Quyền gốc **sau50epoch, nếu đường cong vẫn tăng thì có thể thử thêm50epoch** ở§5.4 được giữ làm provenance. Quyền **thêm100, từ50→150epoch/head** ở§5.5 đã thực hiện và verified. Quyền thêm200 từ150→350epoch/head (§5.6) đã thực hiện/verified. **Latest user: đối chứng raw vs normalized trước; nếu ổn thì train lại300epoch** — pairedpilot50vs50, conditionalfreshnormalizedtotal300 theo§5.7.
**Quyền thực thi mới nhất:** `SM13-NATIVE-LOCAL-20261114-v1` — user yêu cầu tiếp tục backbone mới; **mọi tác vụ SM13 mới chạy LOCAL ONLY, không dùng Kaggle**. Giữ nguyên CROTTC hiện tại. Ưu tiên S1 native pretrained WavLM Large frozen + heads mới; không tiếp tục quanh S0 cũ thay cho S1, không nhầm WavLM Large với WavLM Base.
**Trạng thái:** `S0_EPOCH50_VERIFIED — S1_RAW350_VERIFIED — PAIRED_RAW_NORMALIZED50_AND_NORMALIZED300_VERIFIED`; pilot/normalized300 đãterminal, không còn training trong run này (§5.7). Full-TRAIN head fit và common DEV_SEARCH comparison vẫn pending gates.
**Điều phối:** [GOAL.md](GOAL.md). Nhánh paper: [GOAL_CROTTC_IF.md](GOAL_CROTTC_IF.md).

## 1. Mục tiêu, giả thuyết và phạm vi

Phát triển và kiểm chứng một SM13-style model dùng backbone lớn hơn, trước hết **frozen WavLM Large**, so với SM13 gốc bằng **detection precision/recall/F1 chung**, đồng thời giữ AP/F1 theo S/O/I, exact-gap insertion, false warnings và chi phí tài nguyên để phát hiện đánh đổi.

Câu hỏi chính: **cùng TRAIN, intended-phone reference, predicted alignment, head recipe và evaluation policy, SM13 với biểu diễn WavLM Large có cải thiện chất lượng bắt lỗi so với SM13 gốc không?**

Đổi metric không tự cải thiện model. Model lớn hơn không tự tốt hơn; tăng epoch có thể tăng một lớp và giảm lớp khác. Đây là giả thuyết/thử nghiệm phương pháp, **chưa có bằng chứng backbone gain hay novelty**. Không gọi lại các thử nghiệm HuBERT/WavLM Base/LoRA lịch sử thành phương pháp mới.

GOAL2 được phát triển song song nhưng không dừng/reset/resubmit hoặc chiếm runtime hai T4 của GOAL1. Không mặc định mở full backbone fine-tuning, broad sweep, nhiều layer-selection trials, thêm ALM hay corpus mới. Các biến thể tốn compute hơn phải có amendment/budget riêng.

## 2. SM13 hiện tại là gì?

- Backbone gốc: `facebook/wav2vec2-base-960h`, revision **22aad52d435eb6dbaf354bdad9b0da84ce7d6156**, frozen/eval; pretrained/ASR fine-tuning trước đó không đồng nghĩa đã được fine-tune trong SM13.
- **13 là 13 hidden representations**, không phải 13 epoch. Wav2Vec2 Base: 12 Transformer layers/768D + input representation.
- Acoustic features được cache; train scalar mixing/scale và small target/gap heads. Target phân loại C/S/O, gap phân loại insertion/no insertion. Giữ N target events và N+1 exact gaps.
- Formal lịch sử: target12/gap5, seeds101/202/303; CTC/E4 branch được giữ từ registered source, fixed50/50 fusion. Formal full-TRAIN trainloss giảm, nhưng không có formal held-validation curve để kết luận generalization.
- Epoch50 experiment mới là **neural-only**, không replay E4/fusion; không gắn kết quả đó cho full SM13 fusion hoặc backbone mới.
- Dataset là tập câu/nhãn; epoch chỉ là một lượt lặp lại. 300 epoch không tạo 300 datasets và không tự bù thiếu diversity.

## 3. Trạng thái có bằng chứng

### 3.1 Inventory, adapter và native smoke đã xong

| Hạng mục | Bằng chứng/kết luận |
|---|---|
| TRAIN inventory | 1800 utterances/12 speakers; toàn bộ WAV SHA verified; 59562 target events, 61362 exact gaps |
| Original cache | 26 acoustic files/13layers FP32 hash/geometry checked; lịch sử không bị overwrite |
| Generic adapter | W2V2 pooling và cả hai head có unit parity với historical SM13; cache/leakage/geometry guards |
| Alignment finding | **18 utterances có predicted centers không monotonic**; giữ original event order, không sort/drop âm thầm |
| Head capacity | S0 tổng **848481** params; WavLM heads **1110625**; không gọi S0/S1 matched capacity |
| Native large-backbone smoke | Kaggle CPU: 8 TRAIN IDs × native/adapted WavLM Large315453120 params, fully frozen; feature geometry/finite checks đã download/verify |
| Adapted smoke parent | Update **482** là provisional early CROTTC; không phải final/selected trained backbone |
| Cache cost | Full TRAIN WavLM arm khoảng **14,95GiB FP32 /7,47GiB FP16**, chưa gồm DEV_SEARCH; precision change cần parity, không tự chọn FP16 vì thiếu disk |

References: [inventory audit](reports/SM13_BACKBONE_SIDE_20261114/inventory_v1/AUDIT.json), [native smoke verification](reports/SM13_BACKBONE_SIDE_20261114/native_smoke_v1/NATIVE_CPU_FEATURE_VERIFICATION.json), [side-study report](reports/SM13_BACKBONE_SIDE_20261114/STATUS.md).

### 3.2 S0 epoch-extension đã hoàn tất, không còn đang train

- Local GTX1650, shared GPU lock; **target50 + gap50**, seed101 (gap RNG1101).
- Fit **1350 TRAIN utterances/9 speakers**, hold **450 TRAIN utterances/ABA+ASI+BWC**, historical outer0.
- Các speakers/fold đã được nghiên cứu trước: **exploratory reused TRAIN development**, không gọi mới/unseen confirmation.
- Original FP32 cache được read-only reuse/hash verify, không chạy backbone backward và không tạo thêm 11GiB cache.
- Đã verify **102 head checkpoint states**, optimizer counters/RNG/cursors, predictions mỗi epoch, AP/F1, fixed endpoints và retained original cache hashes.
- Runtime **551,57s (~9,19 phút)**. Training PID1448557 và verifier PID1451451 đã terminal; **không resume tự động chỉ vì còn số epoch để thử**.

| Held metric | Target12/gap5 | Target50/gap50 | Chênh điểm phần trăm |
|---|---:|---:|---:|
| Substitution AP | 43,5389% | 44,2078% | +0,6689 |
| Omission AP | 32,4963% | 30,4143% | **−2,0820** |
| Exact-gap insertion AP | 16,6571% | 25,5978% | +8,9408 |
| **Macro-3 AP** | **30,8974%** | **33,4066%** | **+2,5092** |

Paired empirical speaker bootstrap: 27 ordered resamples từ chỉ3speakers; interval mô tả95% của delta khoảng **+0,2169 đến +3,9759 điểm phần trăm**. Không coi đây là xác nhận thống kê mạnh: chỉ1seed/1reusedfold. ABA/ASI tăng, BWC giảm; omission cũng giảm. Không chọn AP đẹp rồi bỏ các regression.

Bằng chứng chính:
- [Full verification](reports/SM13_EPOCH50_20261114/LOCAL_VERIFICATION.json), [terminal verifier status](reports/SM13_EPOCH50_20261114/LOCAL_VERIFIER_STATUS.json).
- [Learning curves](reports/SM13_EPOCH50_20261114/LEARNING_CURVES.png), [log](reports/SM13_EPOCH50_20261114/LOCAL_TRAIN.log), [launch receipt](reports/SM13_EPOCH50_20261114/LOCAL_LAUNCH_RECEIPT.json).
- Runtime giữ nguyên: `/var/tmp/langfens-crottc-full-20261113-t_car8m4/sm13_epoch50_local_v1/`, gồm `STATUS.json`, `COMPARISON.json`, histories, checkpoint/receipt mỗi epoch và keyed held predictions.

**Chưa có:** S1 DEV_SEARCH cache/final head result, S2 full cache/head fit, matched large-backbone result, CROTTC common detection adapter/end-to-end validation, full-TRAIN S0/S1 paired DEV_SEARCH evaluation, final fusion comparison hay test result. S1 TRAIN cache và evaluator v1 đã tiến triển ở§3.4; không dùng snapshot S0 cũ phủ nhận chúng.

### 3.3 Remote job tồn dư và tài nguyên

Kaggle GPU S0 job [epoch50-s0](https://www.kaggle.com/code/minkhoaa/langfens-sm13-epoch50-s0) v1 vẫn **QUEUED**, không phải đang train và **chưa được hủy**. Read-only quota API chứng minh dùng GPU vượt21600s budget, reserved0, reset21NovUTC. Không suy quota chỉ từ QUEUED. [Latest snapshot](reports/GOAL_SPLIT_20261114/REMOTE_SNAPSHOT.json), [local admission](reports/SM13_EPOCH50_20261114/LOCAL_ADMISSION.json).

Public cancel SDK cần session ID chưa resolve được từ owner/slug; **không đoán ID, không delete kernel/artifacts**. Historical local window có guard60s và kết thúc trước quota reset; guard đã hết khi process local kết thúc, không giả vờ vẫn chạy. Old finite retrieval PID1418016 còn chờ; nó không phải training launcher/cancellation controller. Cần xử lý đúng queued session trước một duplicate future run; nếu không resolve cancellation được phải báo operator, không gọi đã cancel. Không động GOAL1 để giải quyết job này.

### 3.4 S1 native WavLM Large đã tiến sang local, không phải S0 chạy tiếp

- Local offline profile đã qua median/p95/longest TRAIN trên GTX1650: full315M frozen FP32, peak allocated câu dài nhất khoảng1,35GiB, hai head mới qua disposable backward batches512/1024. Không quality claim từ smoke.
- **TRAIN1800 S1 cache hoàn tất**, 59562target/61362gap; 14,95GiB FP32 mới, không overwrite S0. Full-file và1800row SHA/geometry/finite được verifier đọc lại độc lập. [Profile](reports/SM13_NATIVE_LOCAL_20261114/LOCAL_PROFILE_VERIFIED.json), [cache verification](reports/SM13_NATIVE_LOCAL_20261114/LOCAL_CACHE_VERIFICATION.json).
- Đã launch local first S1 head stage **TRAIN-derived speaker-disjoint calibration1350fit/450held ABA+ASI+BWC**, seed101, target50/gap50, cùng recipe S0 trừ geometry; fresh heads1024D. PID1550406, [registration](reports/SM13_NATIVE_LOCAL_20261114/FIT_CALIBRATION50_REGISTRATION_v1.json), [log](reports/SM13_NATIVE_LOCAL_20261114/FIT_CALIBRATION50_v1.log), work `/var/tmp/langfens-crottc-full-20261113-t_car8m4/sm13_s1_calibration50_v1/`; PID là launch snapshot, phải kiểm STATUS/process trước gọi đang chạy. Đây là reused TRAIN development/calibration trước fullTRAINfit, **không phải** full1800→DEV_SEARCH599 result hoặc quyền xếp hạng CROTTC trên450TRAIN.
- [Metric/calibration registration](reports/SM13_NATIVE_LOCAL_20261114/METRIC_AND_CALIBRATION_REGISTRATION.md): keyed event-level detection evaluator v1, targetscore p(S)+p(O), gapscore p(I), fixed0.5 diagnostic curve; threshold21×21 trên terminal50 TRAINheld, tie-break microF1→precision→higher target/gap threshold, cùng budget/policy cho S0. CROTTC adapter và common DEV_SEARCH validation còn pending.
- **S1 target50/gap50 đã COMPLETE/VERIFIED**, runtime440,05s; trainingPID1550406 và verifierPID1553467 đã terminal. [Independent102-state verification](reports/SM13_NATIVE_LOCAL_20261114/FIT_VERIFICATION.json), [verifier status](reports/SM13_NATIVE_LOCAL_20261114/FIT_VERIFIER_STATUS.json). Macro3AP25,7442%, diagnostic detectionF1@0.5=38,6507%, thấp hơn S0 cùng450held (33,4066%/46,3634%). Cuốicurve không tăng rõ: meanF1epochs41–45=39,9751%,46–50=39,5886%; không tự gán backbone gain.
- Drivers `local_s1.py` (profile/cache), `local_s1_fit.py` (calibration heads); GPU lock/atomic full state/window≤4h/floors, **không có Kaggle API/launcher**. Historical104CPUtests passed; extension regression124passed (CUDA disabled), không claim full repo PASS. Cache ở `/var/tmp/langfens-crottc-full-20261113-t_car8m4/sm13_s1_train_cache_v1/`. User nay cho phép extension bất kể đường cong plateau theo§5.5, không rewrite điều kiện/quyết định cũ.

### 3.5 Debug theo yêu cầu user — CPU audit xong, chưa refit phương pháp sửa

- [Debug report](reports/SM13_S1_DEBUG_20261114/REPORT_VI.md), [completed CPU audit](reports/SM13_S1_DEBUG_20261114/cpu_audit_v2/AUDIT.json): labels/referenceadjacency/eventorder/split/migrations không thấy mismatch. FreshCPUheadinference toànfit/held ở50/150/350 tái tạo savedGPUprobs (maxdelta~5,94e-6,0argmaxdisagreement); sklearn independently replay P/R/F1/AP fixedendpoints. Không tìm thấy labelshuffle/resumereset/evaluator-countbug giải thíchregression.
- Overfitconfirmed: target-onlyfitF1@.5 **90,61→98,79%** nhưng heldF1 **39,43→37,74%** từ50→350; pooledweightedCEfit **0,1024→0,0209**, held **1,4602→3,9133**; wrongheldconfidence≥99% **626→1330**. Recompute cảfit/held ởevalmode/logits loại trừ dropout/batchlosslogging là lời giải thích duy nhất.
- Conditioningweakness: rawlayerRMS S1 max/min **89,48×target/100,19×gap** trên toànfit-only; same256fit-sample S0~3,10×/6,51×. WavLMstableLayerNorm lưu intermediatepre-finalnorm nhưng finalstatepostnorm; trộn rawlayers làm scalarweights không phản ánh contribution ngang nhau. Đây là hypothesis về nguyên nhân qualityloss, chưa causalproof hoặc backbonegainfix.
- NativeCPUreplay3TRAIN: relativeL2~1,5e-6…3,7e-6, downstreamheadprobdelta≤3,32e-6/noargmaxdifference. **Sparse pooledfeatures không qua strictCPU/GPUallclose gate ở2/3samples**; auditv1failed được giữ, auditv2measure/report thay vì hạ tolerance/overwritecache. Doublewaveformnormalizationcounterfactual1sample có pooleddelta nhỏ, không đủ evidence đổ lỗi nó cho gap chất lượng.
- Newopt-in `research/sm13_backbone_side/conditioning.py`: explicitfit-speaker-only RMSnormalizer + NEWheadwrapper/checkpointbuffer; chốngheldleak, khôngloadlegacyheadnhưđãcorrected. [Stats](reports/SM13_S1_DEBUG_20261114/CONDITIONING_CANDIDATE_STATS.json) computed44746target/46096gapfit-events; normalizedfixedsampleRMS~1. **Chưa real-datarefit/gainclaim**, không sửa cache/350checkpoints hoặcproductionrecipe.157SM13CPUtests passed, khôngfullrepoPASS.
- Nextcandidate là registeredmatchedraw-vs-conditionedfreshheadrefit (sameprotocol/seed/recipe), không tự combineLR/layer/architecturechanges hoặc tựresume350. Earlystopping/selectionrule cho tươnglai phải khóa trước, không cherry-pickfixedhistory. Localonly/CROTTCpreservation/testDEV_CALseal vẫn giữ nguyên; debug không tự cấp quyềnproductionrun mới.

### 3.6 Investigation sau normalized300 — CPU endpoint audit mới, không training

- [Report tiếng Việt](reports/SM13_NORMALIZED_DIAGNOSIS_20261114/REPORT_VI.md), [fresh CPU audit](reports/SM13_NORMALIZED_DIAGNOSIS_20261114/AUDIT.json): rehash cache/audio/inputs, strict-load4normalizedstates target/gap50/300 đối chiếu prior602-state receipt; replay toànfit/held ởevalmode/logits, fixed.5, khôngthresholdsearch/backboneforward/GPU/optimizer. Runtime49,03s; protectedartifacts khôngđổi;59targetedCPUtests passed, khôngfullrepoPASS.
- **Pooled fitF1 96,6907→99,6100% nhưng held40,8143→40,8843%**; cùngpooledweightedCEestimator targetfit0,04650→0,00516/held1,68894→3,03045. Wrongheldtargetconfidence≥99%867→1290. Overfit/overconfidence vẫn rõ sauconditioning, khôngchỉlosslogging/dropoutartifact.
- MacroAPgain chủyếu insertion: S/O/IAPdelta50→300 **+1,3427/−0,1220/+8,0268pp**; S/Ochiếm94,30%heldpositives. TargetbinaryAPchỉ+1,0095pp. OmissionTP141→103/357; substitution643→647/1859; totalTP822→786/FP856→709: giảmfalsewarningsđổibằngrecall, khônggainF1bềnvững. Target1240positivesFNởcả50và300.
- Kếtluận: recipehiệntạioverfit/recallbottleneck, **khôngphảichứngminhmodelđãchạmgiớihạnkhảnăngtuyệtđối**. Frozenbackbone/cache khônghọcthêmtheoepoch; task/speakerdiversity/pooling/alignment/regularization làcácgiảthuyếtcầnđốichứng, khôngrootcausecausalđãchốt. Khôngautoextend/retune/chọnbestepoch; matchedfullTRAIN→DEV_SEARCHgates vẫn pending. Investigationkhôngcấpquyềnrunmới/finetune/sealeddata/CROTTCchanges.

### 3.7 User yêu cầu verify hướng cải thiện — anti-overfit pilot còn BLOCKED tài nguyên

- User làmrõ cầnverifyhướngcókhảnăngtăngđiểm, khôngchỉloadbackboneS2. Chọnđốichứnghẹp **gấp2dropout** trênfreshnormalizedheads: target.25/.20→.50/.40,gap.15→.30; proposedpaired50/head/arm với baselinefreshnormalized, sameinit/seed/split/recipe/fixed.5. [Protocol](reports/SM13_DROPOUT_VERIFY_20261114/PROTOCOL.json), [preflight](reports/SM13_DROPOUT_VERIFY_20261114/PREFLIGHT.json), [status](reports/SM13_DROPOUT_VERIFY_20261114/RESULTS_VI.md).
- CPU-onlyconstructionchecks đạt: exactinitialstate/paramcount/evaloutputs, finite/stochastictrainmode; **khôngoptimizerupdate/qualityevaluation/backbone/GPU**. KhôngcoismokelàF1gain hoặcđãtrainpilot.
- **PREFLIGHT_ONLY_RESOURCE_BLOCKED**: /var/tmp12,53GiB<14GiBCROTTCreserve; home4GiBfloor+1,5GiBbudget/RAMđạt. ChưaGPUprofile/fit/productiondriver/independentqualityverification; khôngautolaunch/waiter/delete/moveartifact/hạfloor, khôngđụngCROTTCretrieval. Chỉtiếptụcsauvalidstorage/resourceadmission; khôngS2/fullfinetune/sweep/extension/testDEV_CAL.

### 3.8 Debug dropout causality — không có bằng chứng dropout là root cause

- User hỏi liệu overfit/plateau thực sự dodropout. [Report](reports/SM13_DROPOUT_CAUSALITY_20261114/REPORT_VI.md), [CPUaudit](reports/SM13_DROPOUT_CAUSALITY_20261114/AUDIT.json):22,74s/2threads,4trainednormalizedheads50/300,8fit/heldsampleprobes vàfullheldlogitCEreplay. Sourcehashkhớpregistration/cache/audio/SHAđãkiểm; khôngoptimizer/backbone/GPU/thresholdsearch/remote/sealeddata.
- Runtimecontrols đạt: measuredtrainmasktarget~25%/20%,gap~15%; evalrepeats exact; trainp0=eval exact; gấp2p saufitvẫneval thìoutputkhôngđổi. Modes/p/RNG/tensors/artifactsđượcgiữnguyên. **Chưathấybugdropouttrain/eval**, nhưngkhôngchứngminhpđủregularization. Prior2xdropoutproposal chỉlàhypothesis/chưarefit/chưaqualitygain.
- HeldmeanCE50→300 theoS/O **4,5968→7,9076 /5,6086→12,0728**, chiếm~98,3%phầntargetweightedCEtăng; fitF1gần100%nhưngheld~41%vẫnconsistentoverfit. MeanheldF1window11–20/41–50/291–300 **44,4078/42,1783/39,6455%**, descriptivekhôngchọnbestepochhậunghiệm.
- Verdict: generalization/overconfidencegapđãxácnhận; dropoutstrength/LR/capacity/diversity/representation/alignmentchưatáchcausal.62targetedCPUtestspassed,nofullrepoPASS. Pairedrefit§3.7vẫnblocked14GiBmainreserve; khôngautomatictraining/finetune/extensionhoặchạgate.

### 3.9 Research xử lý overfit — recommendation, chưa đổi quyền chạy

- User yêu cầu research/continue. [Tổng hợp tiếng Việt](reports/SM13_OVERFIT_RESEARCH_20261114/REPORT_VI.md), [11primaryworks/evidence](reports/SM13_OVERFIT_RESEARCH_20261114/EVIDENCE.md), [searchlog/limits](reports/SM13_OVERFIT_RESEARCH_20261114/SEARCH_NOTES.md). Focusedreview, khôngsystematicreview/SOTAclaim; khôngtraining/evalnewpopulation/remotejobs.
- Ưutiênnghiêncứu: futurepreregisteredearlystop/checkpointpolicy; mộtfreshpaired **label smoothing nhẹ** vsnormalizedbaseline đểkiểmoverconfidence; smallerhead hoặcRDrop làcácalternativeđợtriêng. ConfidencepenaltyalternativeLS; khônggộpLoss/LR/dropout/capacitycùnglúc. ExternalASR/TIMIT/RDropNLP-CVevidencekhôngphảibằngchứngSM13F1gain. WeightedCE+smoothingsemantics/denominatorcầntoytests; khôngâmthầmđổiweighting.
- MDDaugmentationpapersFu2021/SpeechBlender2023 cóevidencephùhợpnhưngkhácmodel/metric; referencecorruptioncóthểđổiS/O/I/eventmapping, pooledSSLmaskkhôngđồngnghĩaSpecAugmentfilterbank. Longtermerrorlabels/speakerdiversitykhác600same-speakerpseudo-labeltrialcũ; LoRA/CNNhistoricalnegativefindingsđượcgiữ, khôngclaimnovelty/tựrerun.
- Đây **khôngtựthayprotocol/quyềnpilotdropout§3.7 bằngLS**, khôngmởgrid/finetune/data/sealedsets. Cầnquyếtđịnhcanthiệpcụthể+registration/localresource/sharedlock/admission. Noautoextension/thresholdretuning/besthistoryselection;14GiBCROTTCreservevẫngiữ.

### 3.10 Latest user — tái hiện và đối chứng trước, không quyết định từ recommendation chưa verify

- User yêu cầu **khôngđềxuấtquyếtđịnhkhichưacóđốichứng; tái hiện rồi đối chứng trước**. Scope thực thi giữboundedfreshnormalizedbaseline vs2xdropout50/head/arm§3.7; khôngtựđổisangLS/grid/fullFT vìresearch§3.9. Qualitydecisionchỉsauverifiedrealoutcomes.
- [Freshreplay/admission](reports/SM13_REPRODUCE_CONTROL_20261114/REPLAY_AND_ADMISSION.json), [status](reports/SM13_REPRODUCE_CONTROL_20261114/RESULTS_VI.md): CPU37,54s/2threads,cache/audio/inputSHA/fullfit-heldfreshnormalized50headinference vàsklearn100savedpredictionfiles/50epochsP/R/F1/S/O/IAP táihiệnđúng. FitF196,6907%,held40,8143%. Đây **inference/metricreplay**, khôngfresh50epochtraining/pairedqualityoutcome.
- **BASELINE_REPLAY_VERIFIED_PAIRED_TRAINING_RESOURCE_BLOCKED**: rootfree14.597.033.984bytes<14GiB(15.032.385.536),thiếu435.351.552bytes(~415,18MiB);homebudget/RAMđạt. NoGPUprofile/train/backbone/remote/sealeddata, noautolaunch/schedule/delete/move/hạfloor. Pairfit/vérifierproductionchưatriểnkhai; khôngclaimdropoutgiúp/LSgain. GiữCROTTC/artifactscũ.

### 3.11 User “tiếp tục đi” — paired driver/verifier đã implement, actual admission vẫn BLOCKED

- Đãtriểnkhai `research/sm13_backbone_side/dropout_paired.py`, `tools/verify_sm13_dropout_paired.py`, `tests/test_sm13_dropout_paired.py`; **85targetedCPUtests passed**, khôngfullrepo/GPUproductionPASS. [Registeredsource](reports/SM13_DROPOUT_PAIRED_20261114/REGISTRATION.json), [implementation/status](reports/SM13_DROPOUT_PAIRED_20261114/STATUS_VI.md).
- Boundedlocalpair50/head/arm: firstfreshnormalizedbaseline→blocking102-state/historicalnormalized50reproductionverifier→candidate2xdropout→blocking204-state/keyedmetrics/Adam/RNG/fit-onlyRMS/fullcurve/freshCPUfitheldverification. Chỉdropoutđổi, noLS/FT/newcache/thresholdtuning/bestepoch/extension. ActualGPUprofile/sharedlock/resourcegatecótrongdriver,chưathựcthibởiadmissionblocked.
- Lệnhdriverthậtđãgọi: [ATTEMPT.log](reports/SM13_DROPOUT_PAIRED_20261114/ATTEMPT.log), [admission](reports/SM13_DROPOUT_PAIRED_20261114/LATEST_ADMISSION_STATUS.json) **RESOURCE_BLOCKED_NO_TRAINING**, rootthiếu435.351.552bytes(~415,18MiB)so14GiB; home/RAMđạt. Workchưatạo/noGPUfit/noautowaiter/retry; exit0làadmissionrefusal, khôngtrainingPASS. Outputestimate1,298GiB+15%margintrongregistered1,5GiB. Khôngdelete/moveprotectedartifacts/hạfloor/độngCROTTC. Codeđãcó, cầnvalidadmissionđểchạyrồiđốichứng; chưacóqualityoutcome/gainclaim.

## 4. Các model/đối chứng đã chốt phạm vi

| Arm | Backbone/geometry | Vai trò và gate |
|---|---|---|
| **S0** | Frozen Wav2Vec2 Base960h; indices0…12; 13×768 | Reference SM13; replay/fit cùng protocol với S1, không lấy in-sample formal score làm baseline held-out |
| **S1** | Frozen **native WavLM Large**; indices `[0,2,…,24]`; 13×1024 | Phương pháp lớn hơn đầu tiên cần triển khai, không đổi layers bằng DEV search |
| **S2 — conditional** | Frozen CROTTC-adapted WavLM Large, same S1geometry | Kiểm tra adaptation sau khi có final/selected parent đã verify; checkpoint482 chỉ smoke. Không tự khởi chạy campaign backbone-foldwise |
| **CROTTC comparator** | Existing full native AM trên protocol local; IF chỉ khi thật sự hoàn tất | Model so sánh, không giả vờ là faithful paper benchmark. Không nhận học từ held evaluation speakers |

WavLM có 24Transformer layers/1024D + input =25representations; cần trích features mới và **train heads mới**, không load nguyên heads768D. Target input13×3072, gap13×2048. Generic pooling radius8/target crop3:14/gap11:16 và1:6 giữ nguyên.

S1/S2 phải cùng normalization và head recipe; hiện FrozenWavLM smoke khai báo HF normalization rồi SpeechBrain layernorm nếu do_normalize, no output norm, eval/no SpecAugment. S0 original cache dùng HF feature extractor only. **Normalization/capacity khác nhau phải được audit và công bố trước khi quy gain cho riêng backbone.** Không reuse cache chỉ vì filename giống nhau; identity phải bind model/config/layers/precision/normalization/alignment/manifest/order/source hashes.

## 5. Protocol đánh giá mới — detection F1 chính, AP không bị bỏ

### 5.1 Không đánh tráo thước đo

- GOAL2 chuyển trọng tâm sang **detection P/R/F1**. Đầu tiên phải khóa common evaluator/fixtures/denominators; chưa có số F1 chung được chứng nhận.
- Bản contract dự kiến **event-level detection**: positive target = S hoặc O; positive gap = exact-gap insertion. Gold population N target +N+1gap; một positive target chỉ được đếm một lần dù dự đoán sai giữa S/O. Đếm TP/FP/FN trên cùng keyed population; micro F1=`2TP/(2TP+FP+FN)`, kèm precision/recall và từng target/gap subgroup.
- Đây **không tự đồng nhất với hierarchical phone-alignment detection F1 của paper**. Paper-style F1 nếu báo phải có adapter và đối chiếu độc lập; báo tên/protocol riêng, không chuyển tên để so trực tiếp với CROTTC51–54% hiện có.
- S/O classification, exact gap localization và false warnings phải còn báo. **Secondary:** S/O/I AP, macro3AP, typedF1, FWR/câu, coverage, per-speaker support, latency/RAM/VRAM/cache cost. Metric undefined khi thiếu support thì báo undefined, không thay bằng0 hay loại subgroup khó.
- SM13 binary target score có thể dùng `p(S)+p(O)`, gap dùng`p(I)`; điều này khác historical warning policy dùng max error probability. **Phải khóa score/threshold policy trước fit/evaluation**, không âm thầm áp policy mới rồi gọi là replay original SM13.
- Fixed0.5 có thể báo như diagnostic đã định trước; primary threshold/checkpoint selection phải dùng cùng một quy tắc TRAIN-derived speaker-disjoint calibration, finite search/budget/tie-break đăng ký trước. Không chọn threshold riêng bằng chính final evaluation labels rồi gọi là khách quan. Không tự áp FWR≤0,20 hoặc precision/recall80% làm điều kiện thesis.

### 5.2 Common data và leakage boundary

1. **Epoch50 trial hiện tại** chỉ trả lời12/5vs50/50 của S0 trên reusedouter0. Không dùng450câu đó để xếp hạng CROTTC: CROTTC đã train toàn bộ1800, gồm450câu này.
2. Phép so S0/S1/CROTTC chính trên protocol local: cùng **TRAIN1800 fit → DEV_SEARCH599 evaluation**, cùng supplied intended reference, annotation mapping và event IDs. Cần fit/replay S0/S1 đầy đủ TRAIN; checkpoint S0 mới fit1350 không được gọi là full1800matched baseline.
3. DEV_SEARCH đã dùng lặp lại, và CROTTC chọn checkpoint bằng nó: tất cả kết quả này vẫn **exploratory development**, không gọi unseen confirmation hay test paper.
4. SM13 inference chỉ nhận audio/canonical reference do bài tập cung cấp và predicted alignment; không gold timestamps/error labels/realized phones. CROTTC event adapter dùng decoded output + cùng intended reference sau acoustic decoding, không đưa teacher/gold vào inference. Nhãn lỗi chỉ vào fit TRAIN hoặc metric sau inference.
5. S2 toàn-TRAIN-adapted **không hợp lệ cho independent TRAIN OOF**. Nếu cần OOF/calibration backbone-disjoint thì checkpoint mỗifold phải không học fold đó; đây không phải quyền tự mở4campaign300epoch. Nếu không có lineage thích hợp, không đưa S2 vào claimed-independent experiment; đăng ký alternative development-only protocol trước.
6. Test/DEV_CAL sealed; câu hỏi “so ai tốt hơn” không tự cấp quyền mở chúng.

### 5.3 Tách yếu tố để biết gain đến từ đâu

- C0 metric replay: tính detection cho S0 hiện có để kiểm evaluator, không đổi backbone và gọi tăng điểm do model.
- C1 epoch control: giữ kết quả12/5vs50/50 đã đo; **không tuyên bố50tốt cho mọi lớp hoặc mọi backbone**.
- C2 backbone comparison: S0/S1 cùng seed101, TRAIN/DEV population, loss/LR/batches/epoch target/checkpoint rule và score policy. Mốc50/50 là bounded first recipe; cần registration trước run. Extension50→100 có điều kiện theo§5.4; không chọn riêng epoch đẹp của từng arm để tuyên bố backbone gain. Có thể giữ12/5snapshots làm đối chiếu epoch trong cùng run.
- C3 replication: seeds202/303 kế thừa kế hoạch trước, chỉ sau correctness/resource gate và receipt budget riêng; không chọn seed tốt nhất, không tự mở full grid.
- C4 optional S2 và fusion: sau parent/leakage gates. Historical E4/CTC branch và fixed50/50fusion giữ nguyên cho paired fusion experiments; khi E4/DEV replay chưa verify chỉ báo neural-only, không gắn nhãn fullfusion.

### 5.4 Quyền thử thêm50epoch nếu đường cong còn tăng

**Amendment `SM13-EPOCH100-CONDITIONAL-20261114-v1`:** user cho phép sau khi hoàn thành50epoch, **nếu đường cong đánh giá vẫn cải thiện gần mốc50**, có thể thử thêm **50epoch**, tức tối đa **100epoch tổng cộng** cho mỗi head trong arm đủ điều kiện. Không phải mặc định chạy100 hoặc tự nối tiếp150/200epoch.

- Xét **held-out development metric**, không chỉ train loss giảm hay điểm50 cao hơn12/5. Với S0 trial hiện có dùng macro3AP đã đăng ký; với protocol detection mới dùng detectionF1 theo policy đã khóa. Xem cả S/O/I và per-speaker regressions, không chỉ lớp tăng đẹp nhất.
- Trước extension, lưu curve review và receipt nêu metric, cửa sổ cuối dùng xét xu hướng, tiêu chí cải thiện/plateau, parent checkpoint/hash và budget. Không đổi metric hoặc chọn cửa sổ hậu nghiệm để hợp thức hóa việc chạy tiếp. Nếu plateau/giảm hoặc chưa đủ bằng chứng xu hướng: giữ mốc50, chưa kích hoạt extension. Trial S0 hiện có vẫn **COMPLETE/VERIFIED**, chưa được kết luận đủ điều kiện chỉ từ chênh12/5→50/50.
- Resume từ **full state epoch50 đã verify** (model/optimizer/scheduler nếu có/RNG/cursor), không restart từ đầu rồi gọi continuation. Giữ data/split/seed/backbone/head recipe và score/threshold policy; giữ nguyên artifacts50, ghi outputs51–100 và provenance riêng, atomic checkpoint mỗi epoch.
- Với matched S0/S1, giữ kết quả50vs50 làm đối chứng chung. Chỉ gọi100vs100 là matched epoch comparison khi cả hai được chạy cùng extension protocol/budget đã đăng ký; extension của một arm chỉ là exploratory within-arm, không phải bằng chứng gain do backbone.
- Vẫn phải qua local resource/profile/shared-lock/queued-duplicate gates, bounded window≤4h và checkpoint verification; **không ảnh hưởng CROTTC**, không mở test/DEV_CAL hay full-backbone fine-tuning. Tác vụ cập nhật tài liệu này không khởi chạy training.

### 5.5 Latest user — S1 thêm100epoch, từ50→150 để quan sát đường cong

**Amendment `SM13-S1-PLUS100-20261114-v1`:** user yêu cầu **“train tiếp100epoch xem đường cong thế nào”** sau khi được báo S1epoch50 kém hơn và cuốicurve không tăng rõ. Diễn giải: **100epoch bổ sung cho cảtarget/gap**, dừng ở **150epoch tổng/head**. Đây là quyền trực tiếp mới, thay gate còn tăng/max100 của§5.4 **chỉ cho continuation S1 này**; không phải mọi arm tự được150 hoặc tự nối tiếp sau150.

- Resume **full verified epoch50 state**: giữ nguyên model/Adam/RNG/history/cursors/scheduler/scaler; migration sang outputlineage mới chỉ đổi decision/identity tags và có exact-state comparison/receipt. Không restart/reseed model để gọi là continuation; không sửa checkpoint/outputs50 gốc.
- Giữ nguyên nativeLargefrozen/cacheFP32/split1350fit450held/seed101/gap1101/headrecipe/LR/loss/weightdecay/batches. Không backbonebackward, S0extension, fullTRAINfit hay DEV_SEARCH/test access mới trongrun này.
- Báo complete1–150curves, fixed endpoints50/100/150, fit-vs-heldloss, S/O/IAP/macroAP, per-speakerP/R/F1 và falsewarnings. Detection dùng fixed0.5 và **thresholdđãcalibrateởepoch50 được giữ nguyên** làm diagnostic; không retune theo từngepoch hoặc lấybestepoch thay fixedendpoint.
- [Registration](reports/SM13_S1_PLUS100_20261114/REGISTRATION.json), [train log](reports/SM13_S1_PLUS100_20261114/TRAIN.log); driver `research/sm13_backbone_side/local_s1_extend.py`; launchPID1579721 (checkprocess/STATUS để biết liveprogress). Work `/var/tmp/langfens-crottc-full-20261113-t_car8m4/sm13_s1_plus100_v1/`; curve report/PNG khi terminal nằm trongreportdir.
- **Đã hoàn tất target150/gap150**, trainingPID1579721 và observerPID1583121 đã terminal; window855,09s. Training đủ150 nhưng originaldriver/observer báoFAILED vì postprocess gặp7 derived FP32 scores=1+1ulp. CPU-only repair không train lại/đổi checkpoints/predictions/threshold decisions; giữ originalfailedstatus/source/log và ghi [recovery receipt](reports/SM13_S1_PLUS100_20261114/POSTPROCESS_RECOVERY.json). Subsequent verifier đã kiểm **204states** (200new+2migrated50+2last), Adam/RNG/metadata-only migration, complete1–150curves/keyed metrics/frozen50thresholds và retained cache hashes. [Verified receipt](reports/SM13_S1_PLUS100_20261114/VERIFICATION.json), [recovery verifier status](reports/SM13_S1_PLUS100_20261114/RECOVERY_VERIFIER_STATUS.json), [full report](reports/SM13_S1_PLUS100_20261114/STATUS.md), [verified curve](reports/SM13_S1_PLUS100_20261114/VERIFIED_LEARNING_CURVES.png). Không rewrite originalFAILED thành success.
- Fixed endpoints50/100/150: diagnostic detectionF1@0.5 **38,6507 /41,7724 /40,6829%**; frozen50thresholdF1 **42,6627 /44,0037 /43,7263%**; macro3AP **25,7442 /27,3496 /28,9543%**. MeanF1last10 quanh50/100/150 khoảng39,78/39,95/39,80%: AP tăng nhưng detectionF1 dao động/gần plateau. Targetfitloss0,2011→0,0958 trong khi heldloss1,4602→2,3781, phù hợp overfitting. Không tự train sau150; chưa backbonegain/finalDEV_SEARCHclaim. CPU126SM13tests passed sau numericalguardfix, không fullrepoPASS.
- **LOCAL ONLY**, sharedGPUlock,2CPUthreads, bounded4h; admission thêm2GiBoutputbudget ngoài14GiBCROTTCexportreserve, không tạo cachemới. Full atomic states mỗi epoch/safeoptimizerboundary, resource/nonfinite/mismatch stop, không skipbatch. Không động CROTTC, Kaggle/Colab, sealedtest/DEV_CAL. Đây vẫn exploratoryreusedTRAIN/neural-only, không matchedbackbonegain hay finalcommonDEV_SEARCHresult.

### 5.6 Latest user — S1 thêm200epoch, từ150→350

**Amendment `SM13-S1-PLUS200-20261114-v1`:** user yêu cầu **“train thử thêm200epoch xem”** sau báo verified150/overfitting. Quyền trực tiếp **200epoch bổ sung/head, tổng350**, riêng S1 local; thay điểm dừng150 cho continuation mới, không tự cấp thêm sau350.

- Resume verified150parent, không rollback/restart/reseed. Outputlineage mới, migration chỉ đổi decision/identity, fullmodel/Adam/RNG/history/cursor/scheduler/scaler exact equality/receipts. Original50/150artifacts và failure/recoveryprovenance giữ nguyên. Numericalderived-scoreFP32guard đã có trong source registration mới, không âm thầm thay source cũ.
- Same nativeLargefrozen/cacheFP32/1350fit450held/seed101gap1101/headrecipe/LR/loss/weightdecay/batches; no backboneforward/backward/newcache/test/DEV_CAL/DEV_SEARCH. Fixed0.5 và frozenepoch50thresholddiagnostics; không thresholdretuning/bestepochselection. Report complete1–350curve, fixed50/100/150/200/250/300/350, losses/S/O/IAP/macroAP/per-speakerP/R/F1/FWR.
- Driver `research/sm13_backbone_side/local_s1_plus200.py`; [registration](reports/SM13_S1_PLUS200_20261114/REGISTRATION.json), [log](reports/SM13_S1_PLUS200_20261114/TRAIN.log); launchPID1675056. Work `/var/tmp/langfens-crottc-full-20261113-t_car8m4/sm13_s1_plus200_v1/`. Launch/live snapshot không thay verifiedterminalreceipt.
- LOCAL ONLY/sharedGPUlock/2CPUthreads/max4h, unchangedRAM16initial/4runtimeGiB/VRAM512MiB/disk14GiBmainreserve. Admission **3GiBnewoutputs ngoài14GiB**, không deleteweights/cache đểláchgate. Atomicfullstateeach-epoch/safeoptimizerpause, stopnonfinite/mismatch/resource, khôngskipbatch; khôngđộngCROTTC/Kaggle/Colab. CPU143SM13tests passed (CUDA disabled), khôngfullrepoPASS.
- **Target350/gap350 đã COMPLETE và404states VERIFIED**, target30800/gap16100updates, window1674,63s (~27,91phút); trainingPID1675056/verifierPID1676328 đã terminal. [Verification](reports/SM13_S1_PLUS200_20261114/VERIFICATION.json), [verifier status](reports/SM13_S1_PLUS200_20261114/VERIFIER_STATUS.json), [verified curve](reports/SM13_S1_PLUS200_20261114/VERIFIED_LEARNING_CURVES.png). Parentexactstate/Adam/RNG/completecurve/keyedmetrics/frozen50thresholds/cachehashes đã kiểm, không cần reportrepair mới.
- Fixed150→350: F1@0.5 **40,6829→37,3146%**, frozen50thresholdF1 **43,7263→41,2178%**, macro3AP **28,9543→29,0901%**. MeanF1epochs141–150/341–350 **39,7981/38,3997%**. TargetfitCE0,0958→0,0558 nhưng heldCE2,3781→3,9133: detectionregression/overfitting, không tăng chất lượng rõ từ thêm200. Dừng350, khôngautoextend. Đây vẫn exploratoryTRAINcalibration/neural-only, không finalDEV_SEARCH/commonCROTTC hoặc backbonegainclaim.

### 5.7 Latest user — đối chứng raw/normalized trước, nếu ổn thì300total

**Amendment `SM13-RAW-NORMALIZED-CONDITIONAL300-20261114-v1`:** userủyquyền pairedlocalraw-vs-fit-onlyRMS trước; chỉ khi correctness/qualitygateđạt mới freshnormalized300total/head. [Protocol](reports/SM13_RAW_NORMALIZED_20261114/PROTOCOL.md), driver `research/sm13_backbone_side/raw_normalized.py`, verifier `tools/verify_sm13_raw_normalized.py`.

- Pilotđãlaunch localPID1763886; [registration](reports/SM13_RAW_NORMALIZED_20261114/REGISTRATION.json), [log](reports/SM13_RAW_NORMALIZED_20261114/TRAIN.log); checkworkSTATUS/process thay vìcoi launchsnapshot làliveprogress. Freshpaired50/head/arm, seed101/gap1101, identicaltrainableinitialweights/RNG/batchorder, same1350fit/450held/layers/cache/recipe; chỉ acousticper-layerRMS khác nhau. Recompute13RMS trên fit44746target/46096gap, frozenbuffers persisted; noheldstats/labeluse. Interleave rawtarget/normtarget/rawgap/normgap; onepairedseed/reuseddevelopment, no stablegainclaim.
- **Gate đăng ký trước outcomes:** fixed50diagnosticmicroF1@.5delta≥+1pp; meanpairedlast10F1delta≥0; macro3APregression≤0.5pp; mỗiS/O/IAPregression≤2pp; ít nhất2/3speakersF1khônggiảm; FWRincrease≤0.5/câu. SeparateCPUverifier checks all204pilotstates/predictions/freshheldinference/rawhistoricalparity/fit-onlystats/gate trước300admission. Khôngretune/chọnbestepoch/đổigate đểqua; descriptive27speakerresamples khôngstrongconfirmation.
- **Đã COMPLETE/VERIFIED**: cả6pilotqualitygatesPASS,204pilotstatesverified trướcconditionaladmission; normalized300total/head đã xong,602normalizedstatesverified, PID1763886terminal. [Results](reports/SM13_RAW_NORMALIZED_20261114/RESULTS_VI.md), [pilot verification](reports/SM13_RAW_NORMALIZED_20261114/PILOT_VERIFICATION.json), [final verification](reports/SM13_RAW_NORMALIZED_20261114/FINAL_VERIFICATION.json), [curve](reports/SM13_RAW_NORMALIZED_20261114/NORMALIZED300_CURVES.png). Raw50→norm50F1@.5 **38,6507→40,8143%**, macroAP **25,7442→30,7764%**; tất cả3heldspeakers tăng, FWRgiảm0,5289/câu. Normalized300F1 **40,8843%**, macroAP **33,8588%**: thêmepochchủyếutăngAP, khôngF1gainrõ; cuối10epochmeanF1~39,6455% so42,1783%ở41–50, targetheldCEtăng1,6889→3,0304. Dừng300, noautoextension. Onepairedseed/reusedTRAINcalibration, khôngfinaltest/commonDEV_SEARCHclaim.
- Quyền đãthựchiện: khiPASS normalized mới0→50pilot rồi **resume exactfullstate50→300**, total300 (không300bổsung, khôngoldraw350warmstart). Pilotartifacts/gateverification giữ nguyên; sameLR/loss/batches/scale/thresholdfixed.5, fixedendpoints50/100/150/200/250/300/fullcurves, finalCPU602stateverification. Normalized300vsraw50khôngmatched-epochgainclaim; pairedpilot50vs50 là normalizationcomparison. NếuFAIL: stopafterpilot, khôngauto300/retry/sweep.
- **LOCAL ONLY**, GPUsharedlock/2CPUthreads/≤4h, RAM16initial4runtimeGiB/VRAM512MiB. Existingcache /var/tmpreadonly; vìrootdiskhẹp, weights mới ở `/home/khoa/.local/share/langfens/sm13_raw_normalized_20261114_v1/` ngoàiGit: homefloor4GiB+5.5GiBnewoutputadmission, độc lập **giữnguyên14GiB/var/tmpCROTTCexportfloor**. Khôngdeletecache/checkpoints/độngCROTTC hayremote/test/DEV_CAL; stopnonfinite/resource/mismatch/safeoptimizercheckpoint, no skipbatch.179SM13CPUtests passed trướcadmission; nofullrepoPASS.

## 6. Kế hoạch thực thi và acceptance gates

### S0 — Tiếp nhận kết quả hiện có
- [x] Audio/feature/alignment audit; generic pooling/head parity và leakage guards.
- [x] Full frozen native/adapted feature smoke8×2; không quality claim từ16forwards.
- [x] Local original-backbone target50/gap50, checkpoint/predictions/terminal comparison verified.
- [x] Báo +2,5092pp macro3AP và omission/BWC regressions, không cherry-pick chỉ gain.
- [ ] Cập nhật report/head loss curves về train-vs-held và detection replay sau khi evaluator đã khóa; không reinterpret argmax classF1 thành paper detectionF1.

### S1 — Common detection/evaluator/data contract
- [ ] Freeze metric spec, score/threshold/calibration/checkpoint rule, event mapping, canonical information budget và denominator.
- [ ] Fixtures: all-correct, S/O/I, N=0/N=1, repeated phones, omission/insertion ambiguity, boundary gaps, missing class, duplicate/missing event IDs; so expected counts.
- [ ] CROTTC adapter/audio-only provenance; common predictions export đầy đủ, không drop failed cases.
- [ ] Freeze TRAIN1800/DEV_SEARCH599 manifests/hashes; chuẩn bị **predicted** DEV_SEARCH alignment/features, không dùng gold TextGrid làm inference input. Hiện chưa có DEV_SEARCH feature cache được verify.

### S2 — WavLM Large local profile/cache/head fitting
- [ ] Kiểm source/native normalization, full315M frozen architecture, dimension/head capacity và cache identity.
- [x] Profile local4GiBGPU thật trên median/p95/longest authorized TRAIN và disposable head backward; bằng chứng§3.4, không dùng KaggleCPU smoke thay localGPUproof.
- [ ] Đăng ký bounded production window≤4h, explicit disk/RAM/VRAM floors và storage plan trước tạo cache; **không materialize S1+S2fullcaches cùng lúc**. Main CROTTC export có ưu tiên14GiBdownload floor. Không delete artifacts để lấy chỗ.
- [ ] Extract only allowlisted TRAIN/DEV_SEARCH, hash per-shard/row order/geometry/finite; stage inference requests tách khỏi gold.
- [ ] Fit/replay matched full-TRAIN S0/S1; same head recipe, seed101 và full state atomic/resumable; report neuron-only trước nếu E4 gates còn mở.

### S3 — Paired comparison và kết luận
- [ ] Cùng evaluation599event population, same metric/selection policy, đầy đủ per-speaker P/R/F1/AP/FWR và support.
- [ ] Không dùng current450TRAINheld để so CROTTC. Không lấy paper71,77% làm điểm comparator.
- [ ] Báo paired deltas, uncertainty phù hợp, omission/insertion/cost regressions; DEV_SEARCH repeated-development caveat rõ.
- [ ] Replication đã gate qua nếu cần claim ổn định; nếu không, ghi single-seed exploratory.
- [ ] Kết luận improved / no clear improvement / regression dựa trên metric đã đăng ký. Không đổi thước đo/loại ca khó để “hoàn tất có gain”.

**Hoàn tất GOAL2:** có implementation/cache/lineage tái chạy được, matched comparison trên protocol rõ ràng, kết quả đúng metric và giới hạn được báo trung thực. Chưa cần thắng để hoàn tất thí nghiệm; một negative result hợp lệ hơn một positive result do leakage. “Phương pháp mới” chưa đồng nghĩa novelty/contribution thesis cho đến khi chứng minh cơ chế và evidence.

## 7. Thực thi an toàn và quyền hiện hành

- **SM13 LOCAL ONLY**: không submit, dùng runtime hay chuyển fallback Kaggle/Colab cho các tác vụ SM13 mới. User đã yêu cầu tiếp tục thực thi S1, không chỉ sửa docs; native extraction/head fitting bắt đầu sau profile và bounded registration tương ứng, không cần hỏi lại lựa chọn local. CROTTC trên Kaggle thuộc GOAL1 và không bị tác động.
- Local-only drivers mới: `research/sm13_backbone_side/local_s1.py` (offline native FP32 profile median/p95/longest TRAIN, disposable head backward smoke, resumable TRAIN cache), `local_s1_fit.py` (fresh WavLM head calibration50) và `detection.py` (keyed evaluator v1; CROTTC adapter/common DEV_SEARCH validation vẫn pending). Receipts nguồn, native weights và budget được khóa trước run. Remote S0 QUEUED là tồn dư lịch sử, không được gọi đã hủy; không launch lại S0 hoặc tự dùng Kaggle để xử lý nó trong tác vụ local-only này.
- Local shared GPU lock: `/home/khoa/Projects/Pronunciation-RD-Coordination/locks/gpu.lock`. Kiểm existing jobs/VRAM/RAM/disk trước; 2CPUthreads cho run ban đầu. Resource floors có trong per-run receipt, không hạ để lách gate.
- Atomic checkpoint mỗi epoch và safe optimizer boundary khi signal/budget: model/optimizer/RNG/shuffle/cursor/partial losses; immediate reload/SHA verification. Stop nonfinite/mismatch/OOM, không skip batches.
- Không đồng thời launch remote/local duplicate. Remote S0 QUEUED không phải đã cancel; xử lý riêng, không tác động remote CROTTC.
- Không full-fine-tune backbone, mở methodsweep, thêm corpus, cron/timer/subagent hay commit/push/merge mặc định.
- Không sửa historical formalSM13, originalcache, checkpoint124/482 hay CROTTCcurrentrecipe. Output mới tách thư mục; weights/cache/audio/credentials không vào Git.
- CPU regression history: prelaunch222passed; localadmission tests passed; concurrent broad regression227passed/3fixtureerrors vì productionGPUlock đang được giữ. Không claim toàn suite PASS, không nhả productionlock để test xanh.

## 8. File map và provenance references

| File/thư mục | Vai trò |
|---|---|
| `research/sm13_backbone_side/features.py`, `heads.py` | Generic geometry/pooling, scalar mixing/heads, cache identity/leakage guards |
| `research/sm13_backbone_side/audit.py` | Read-only authorized inventory và inference requests không gold |
| `research/sm13_backbone_side/backbone.py` | Frozen WavLM native/adapted loader và normalization declaration |
| `research/sm13_backbone_side/kaggle_profile.py` | Historical CPU-only8×2feature smoke, không head fit |
| `research/sm13_backbone_side/epoch50.py` | Original recipe/head training/checkpoint/evaluation helpers và historical Kaggle driver |
| `research/sm13_backbone_side/local_epoch50.py` | Completed local cached-S0 driver, quota/admission/shared lock/resume guard |
| `tools/verify_sm13_epoch50.py`, `tools/watch_local_sm13_epoch50.py` | 102-state/prediction/metric verification, bootstrap/curve generation; finite observer, không launcher |
| `tests/test_sm13_backbone_side.py`, `tests/test_sm13_epoch50.py`, `tests/test_sm13_local_epoch50.py` | Feature/head/cache, resumability, metric và local admission checks |
| [Historical formal preregistration](experiments/rq1_sm13_formal_train_01/PREREGISTRATION.json), [receipt](experiments/rq1_sm13_formal_train_01/FORMAL_TRAINING_RECEIPT.json) | Old3seed/full1800/fusion provenance; không overwrite hoặc gọi in-sample là held-out |
| `experiments/rq1_sm13_formal_train_01/agent_a_sm13_engine.py`, `train_sm13_formal.py` | Historical exact head/training source; có hardcoded paths và fixed12/5, không gọi trực tiếp để chạy50mới |
| `experiments/rq1_encoder_bench_01/common/dataset_util.py`, `experiments/rq1_encoder_bench_01/features/alignment_centers.npz` | Historical order/reference/fold mapping và frozen predicted alignment |
| [Backbone comparison history](reports/ENCODER_BENCH_01_BACKBONE_COMPARISON.md), [adapter history](experiments/sm13_three_directions_20261113/encoder_adapter/PREREGISTRATION.md) | Không lặp lại công việc rồi gọi novelty; không phải bằng chứng WavLMLarge gain |
| [Epoch50 status](reports/SM13_EPOCH50_20261114/STATUS.md), [verification](reports/SM13_EPOCH50_20261114/LOCAL_VERIFICATION.json) | Latest terminal result thắng các running snapshots cũ trong status/handoff |
| [Pre-split full discussion](reports/GOAL_SPLIT_20261114/PRE_SPLIT_SIDE_GOAL.md), [archive hash receipt](reports/GOAL_SPLIT_20261114/ARCHIVE_RECEIPT.json) | Toàn bộ thảo luận và amendments trước tách GOAL, giữ nguyên làm lịch sử |

## 9. Quan hệ với GOAL1 và thesis

GOAL2 cần baseline CROTTC để so sánh phương pháp local, nhưng **không phụ thuộc việc gọi sai current CROTTC là full reproduction**. S1 native không cần chờ CROTTC300; S2 adapted mới cần parent đủ điều kiện. Metric common giữa models là nhiệm vụ riêng, không tự làm đúng benchmark paper.

Hoàn tất SM13 experiment không chứng minh full paper đã tái lập, không chứng minh phản hồi sửa phát âm đúng hoặc người học tiến bộ. Bối cảnh L2-Arctic-plus/ALMs4Learning, rubric feedback và thesis-level evidence giữ ở [GOAL.md](GOAL.md); không mở thêm campaign ngoài hai GOAL này.
