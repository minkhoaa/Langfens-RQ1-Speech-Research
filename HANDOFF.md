# HANDOFF — Hai GOAL: CROTTC-IF full-paper reproduction / SM13 phương pháp mới

Cập nhật: **2026-11-14, Asia/Ho_Chi_Minh**. Workspace:
`/home/khoa/Projects/New-Method-Research`.
Nhánh: `feat/crottc-if-official-20261113`, Git HEAD tại bàn giao:
`879edc2ea5f85ce4ba8f56a97ceb49b4f83cb8a0`.

## Tiếp nhận session mới — đọc routing mới trước snapshots lịch sử

### Storage23:13 — user approved Pacmancachekeep1; cleanup BLOCKED by sudo authentication

- User`ok` sauproposal chỉprunePacmancachegiữ1version. Dryrun192candidates≈1.93GiB.
  Actual`sudo -n paccache --remove --keep 1 --nocolor` returned1:`sudo: a password is required`.
  **Chưa xóafile/khôngfreebytes**,khôngsudointeractive/passwordcollection/bypass.
  Receipt`reports/STORAGE_AUDIT_20261114/PACMAN_CLEAN_20261114_231325/RECEIPT.json`.
  User cần tự chạyterminal`sudo paccache -r -k 1`;khôngtựresumeproduction saucleanup.
- User23:33chophépthaotáctrongtmux`code`. Openedseparate`code:1`window`cache-cleanup`
  (original`code:0`ncduuntouched),script`reports/STORAGE_AUDIT_20261114/PACMAN_TMUX_20261114_2333/cleanup.sh`.
  At23:34`STATE.txt=WAITING_SUDO_OR_CLEANUP`,sudoPID2018447waitingonpts/5,noRESULT.jsonyet.
  Userneedsattach`tmux a -t code`andenterpassworddirectlythere;neverrequestpasswordinchat.
  Scriptwillrecord`RESULT.json`afterexit,includingdfbefore/afterandinstalledpackagelistcomparison.
  **Do notduplicatecleanupwhilewaiting**. Rootfree~15.42GiBwasalreadyhigherbeforethisattempt
  duetoothermaintenance;donotattributethattoPacmanuntilreceiptshowsactualcompletion/delta.
- Root/var/tmpfree14,596,730,880B vẫn<14GiBfloor15,032,385,536B;homefree~13.36GiB
  đãthayđổidoothermaintenance,khôngđượcgọilàPacmancleanupsuccess.
- Read-onlyenvironmentprobe23:14:`.venv-crottc-official/bin/python -c 'import torch,speechbrain'`
  fail`ModuleNotFoundError:No module named torch`. `.venv`directoryexistsbutRecoveryvenvabsent;
  potentialconcurrentenvironmentmaintenance,**recheckbeforelocalverifierPASSclaim**.
  This task didnotdelete/recreate/installvenv,onlyPacmanattempt. Cloudkernelunmodified;
  existingnativeCPUdiagnosticlocaltextverificationmayneedruntime restoration after remotecomplete.


### CROTTC diagnostic repair22:58 — v1FAILED preserved; manualauthorized v2RUNNING/bootstrap

- v1terminalERROR tại`RoPEMHA/_rope_rotate/torch.index_select`:inference tensors cannotbesavedforbackward.
  Cleanprofileinference_mode tạo cachedsin/cos/indexinference tensors rồi nativegradient dùnglại.
  **Không phải maintrainingbug/khôngcócompletegradientor2399cleanmetrics**. Originalfailedlogs/profile/env
  ở`PLATEAU_DIAGNOSTICS_20261114/retrieval_v1/failed_output/`;oldretrieval1952258terminal,
  khônggọicònwait. Nativev1forwardprofileđãpassestimate3926snhưnggradientchưathànhcông.
- Userexplicitlyủyquyềnsửatest/diagnosticthứtựtrướccanthiệp. Repairdecision
  `CROTTC-PLATEAU-DIAGNOSTIC-NOGRAD-REPAIR-20261114-v2`, reportdir
  `reports/CROTTC_IF_KAGGLE_20261114/PLATEAU_DIAGNOSTICS_REPAIR_20261114/`.
  Original5sourcespreservedhashesinPROVENANCE. Diagnosticcleancontextno_gradthayinference_mode;
  progress/profile/repairprovenanceonly,loss/data/model/seed/budgetunchanged.
- RegressiontinyRoPEreproducesfailure,thenrepaircleanoutputexactequal/finitegradient/weightsunchanged;
  **26CPUtests passed/1strictxfail** (originalevaldropoutknownbroken,notthiscontextrepair).
  Fullsynthetic2399textverifiertestscoverv1andv2registrations;no localnativebackward.
- PrivateCPU https://www.kaggle.com/code/minkhoaa/langfens-crottc-plateau-diagnostic-cpu-v2 /1
  pushedonce,serverprivacy/noGPU/scriptsourceexactreadbackverified; API22:57RUNNING/bootstrap.
  **Chưanativev2profile/probes/cleanmetricsclaim** nếu chỉplatformstatus. Same5672parent,
  1800TRAIN/599DEV,6fixedTRAINauggradprobes,12000driver/14400platformsec/2threads,
  16initial12prebackward4runtimeGiBRAM;0optimizer/checkpointweightupdates.
- FiniteCLIretrieval **PID1975664**,`tools/watch_crottc_plateau_diagnostic.py --attempt2`,
  max16200swait+downloadverifygrace;`PLATEAU_DIAGNOSTICS_REPAIR_20261114/retrieval_v2/RETRIEVAL_STATUS.json`.
  VerifierchecksrepairID+2399textpredictions/lossmeans/6gradientcoverage/scalaralgebra;
  **khôngindependentlyreplaynativegradients**. Noautomaticthirdattempt/cancel/relaunch.
- MainremainsSTOPPED_DISK_FLOOR_PARENT_PRESERVED afterverified5672/pending32/32TRAINgradients.
  No production8driver/dependency/recipe/parentmutation/restart/localCROTTC/sweep/sealeddata/commit/push.
  Erroranalysis3150done;chooseinterventiononlyafteractualdiagnosticresultsreview.


### CROTTC latest22:41 — mainterminaldiskfloor5672preserved; nativeCPUdiagnostic RUNNING/bootstrap

- User approved pipelineerroranalysis→matchedcleanTRAINDEV→OTCRgradientdiagnostic.
  **Maincontroller1291861đãterminalSTOPPED_DISK_FLOOR_PARENT_PRESERVED** sauwindow002
  COMPLETE/exportverified5672/internalepoch99cursor928, pending32/32TRAINexamplesgradients
  preserved(nooptimizerstepforthatpendinggroup). Latestverifiedkhôngcòn3150.
  `OPTIMIZED_CAMPAIGN_02/window002/VERIFICATION.json`,SHA
  `8d095b0f43cd1b1514efd548c4ec8416bf98938017791d662547c8d8a572695b`.
  Khôngdiagnosticstop/reset/delete/recipepatch;**khôngtựrestart**hoặchạ14GiBfloor.
- Reportdir `reports/CROTTC_IF_KAGGLE_20261114/PLATEAU_DIAGNOSTICS_20261114/`:
  DECISION/PROTOCOL/LAUNCH_RECEIPT,SERVERSCRIPTreadbackverified. PrivateCPUonly
  https://www.kaggle.com/code/minkhoaa/langfens-crottc-plateau-diagnostic-cpu-v1 /1
  API22:41RUNNING/bootstrap;**chưanativeclean2399result/6gradientprobes/profilePASS**.
- Fixedverified5672fullnative315M+heads;cleanTRAIN1800/DEV599sameeval/normalization/pinnedmetrics,
  OTsingleview/PERonlinevsno-sil;6TRAINsinglepairprobes at899/1709/1799×seeds3047/3048,
  trainaug+SpecAug+fullSSLautograd,gradientOTCRnorm/dot/cosine/clip/alphatransport.
  **0optimizer**,noGPU/localnativebackward,weights/source/parentpendingunchanged.
  CPU_Fp32singlepairnotproduction2T4FP16batch32causalproof;noautomaticablation.
- Profilegates12kdriver/14.4kplatformseconds,2CPUthreads,RAM16initial12prebackward4runtimeGiB;
  nativeforwardestimate<10200s trước6probes/fullpopulation. Probecompletion>300s=>stop,
  globalwatchdogboundsinflightcomputation. Failpreservesoutputs/noauto-retry/noquietskip.
- PhaseA**text-onlyanalysisCOMPLETE** atverified3150repaired599DEV,overall/per-speakerexactmatch:
  F1NCC64.16/YDCK63.90/RRBI46.87/YBAA37.21%;base-ratesdifferso notspeakerbiasproof.
  342/1325FRtopz→s137/r→err130/dh→d75. Hyp361errvs12golderr;TRAIN355r→errtargetpositions,
  supervisionpolicyhypothesisnotcausalproof. `ERROR_ANALYSIS_VI.md`, `ERROR_ANALYSIS_V2/ANALYSIS.json`.
  Per-typepositive-onlyprecision/F1notdefined;onlyrecall;notstrictS/O/I/exactgap.
- **24CPUtests passed/1strictxfail** (originalevaldeterminismknownbroken),includes2399synthetic
  textverifier;notnativePASS. Source/newtests+diagnostictools only,production8driverhashespreserved.
- FiniteCLIretrieval**PID1952258**,`tools/watch_crottc_plateau_diagnostic.py`,max16200swait
  +downloadverifygrace,nottraininglauncher. `retrieval_v1/RETRIEVAL_STATUS.json`;
  actual`VERIFICATION.json`onlyfull2399/6probesmetadata+independenttextscoreonpass.
  Gradientverificationisfinite/coverage/scalaralgebraonly,notindependentnativeautogradreplay.
  NoTRAIN/DEV_CALtestunseal/newcorpus/localCROTTC/SM13scopechange/commit/push.


### SM13 latest22:39 — paired driver/verifier IMPLEMENTED; actual launch admission BLOCKED

- User “rồi sao dừng brainstorm, tiếp tục đi”: implemented `research/sm13_backbone_side/dropout_paired.py`,
  `tools/verify_sm13_dropout_paired.py`, `tests/test_sm13_dropout_paired.py`.85targetedCPUtestspassed,nofullrepo/GPUproductionPASS.
- `reports/SM13_DROPOUT_PAIRED_20261114/REGISTRATION.json`+PROVENANCE pins sources/protocol/historicalreceipt/stats.
  Driverfitsfreshnormalizedbaseline50/head→blocking102-state/historicalnormalized50reproduction→candidate2xdropout50/head
  →blocking204-state/keyedpredictions/sklearnmetrics/fullcurves/Adam/RNG/fit-onlyRMS/freshCPUfitheldchecks; noautoextension.
- Actualdriverinvocation22:38:55 returned RESOURCE_BLOCKED_NO_TRAINING (`ATTEMPT.log`,`LATEST_ADMISSION_STATUS.json`).
  Root14,597,033,984bytes<14GiB15,032,385,536;short435,351,552bytes(~415.18MiB);home/RAMpass.Worknotcreated,
  noGPUprofile/train/verifierlive/noautolaunchwaiter. Exit0meanscleanadmissionrefusal,nottrainingPASS.Productionruntimeunverified.
- `STATUS_VI.md` includesrealentrypoint/limits/storageestimate1.298GiB+15%within1.5GiBbudget. Needvalidstorageadmission,
  notmorebrainstorming/switchtoLS/RDrop/FT. Do notdelete/moveprotectedcache/checkpoints/lower14GiBfloor/haltCROTTC.
  GOAL_SM13§3.11winsold “drivernotimplemented” snapshots. Noqualityoutcomesyet.

### SM13 latest — user yêu cầu reproduce/control trước recommendation; pairfit RESOURCE BLOCKED

- Latestuser: khôngđềxuấtquyếtđịnhchưachắc/cần táihiệnrồiđốichứng. Acknowledgedresearchprioritiesarehypotheses,
  notSM13qualityevidence. Keepexistingboundednormalizedbaseline-vs2xdropout50/head/arm,nounauthorizedswitchtoLS/grid/FT.
- NewCPU37.54s/2threads`tools/replay_sm13_baseline_and_admission.py`:cache/audio/inputSHA,stricttrainednormalized50heads,
  freshallfit/heldreplay,independentsklearn50epochs/100savedpredictionfilesmetrics/order/APverified. FitF196.6907%,held40.8143%.
  **Notfreshtrainingreproductionorcandidatequalitytrial**. `reports/SM13_REPRODUCE_CONTROL_20261114/{REPLAY_AND_ADMISSION.json,RESULTS_VI.md}`.
- StatusBASELINE_REPLAY_VERIFIED_PAIRED_TRAINING_RESOURCE_BLOCKED:rootfree14,597,033,984bytes<15,032,385,536floor,
  short435,351,552bytes(~415.18MiB). Home4GiB+1.5GiBbudget/RAMpass;noGPUprofile/lockadmission/training/backbone/remote/sealeddata.
  Noautowaiter/launch/delete/move/haltCROTTC/lowerfloor/CPUtrainingbypass. Pairedproductiondriver/verifiernotimplementedyet.
  Nextqualityclaimrequiresfreshbaselineparity+pairedfit+independentstate/predictionverification,notpreflightsmoke.

### SM13 overfit research —11primarysources, recommendation ONLY/no new training

- Useryêucầuresearchcácphươngánxửlýoverfit,rồicontinue. Report`reports/SM13_OVERFIT_RESEARCH_20261114/REPORT_VI.md`,
  `EVIDENCE.md`11deduplicatedprimaryworks,`SEARCH_NOTES.md`queries/rejectedleads/limits. Focusedreview,notmeta-analysis/SOTA.
- Mechanismpriority: futurelockedcheckpoint/earlystoppolicy; cheapfreshpairedlightLS(ε.05proposednotoptimalproven)
  vsnormalizedbaseline; smallerhead/RDropalternativesseparately. ConfidencepenaltyalternativeLS;outputregularizationmatches
  localoverconfidence,butnoopenedpaperprovescurrentSM13F1gain. WeightedCE+smoothingrequiresdefinedlossdenominator/toytests.
- MDDaugmentationFu2021/SpeechBlender2023relevantbutdifferentmetrics/architecture;referencechangesmustremaplabels/events;
  SpecAugmentfilterbank≠randommaskpooledSSLcache. Moreerrorlabels/newspeakerdiversity≠priormostlyCORRECTsame-speakerpseudo.
  HistoricalCNN/shortLoRAnegativefindingsretained;noautomaticrerun/newnovelty/fullFTclaim.
- ResearchrecommendationdoesNOTreplacependingdropoutpilotwithLSauthorization. Usermustsettleinterventionbeforefresh
  registration/admission/profile/sharedlock. NoGPU/train/newmetricdata/remote/CROTTC/testDEV_CALcommands;14GiBreserveunchanged.

### SM13 dropout-causality debug — runtime đúng, strength chưa có causal evidence

- Userhỏioverfit/plateaucóthựcsựdodropoutkhông. `reports/SM13_DROPOUT_CAUSALITY_20261114/REPORT_VI.md`
  và`AUDIT.json`:CPU-only22.74s/2threads,4trainednormalizedheads50/300;8fit/held256-eventprobes/fullheldCEreplay.
- Actualtrainmasks~25/20%target,15%gap;evaldeterministic;trainp0=eval;doublepafterfitwithevaloutputsunchanged.
  Source/cache/audio/checkpointSHAverified; modes/p/RNG/weights/artifactsunchanged. Nooptimizer/backward/GPU/backbone/remote/test.
- **Dropoutruntimebugnotfound;insufficientdropoutNOTproven.** Proposed2xdropoutisoneuntestedintervention,notdiagnosedfix.
  HeldS/OmeanCE4.60→7.91/5.61→12.07 from50→300;S/O~98.3%targetweightedCEincrease. Generalizationgapconfirmed,
  rootcauseamongdiversity/capacity/representation/alignment/regularizationstillunresolved.62CPUtestspassed,nofullrepoPASS.
- Priorpairedpilotstillpreflightonly/resourceblocked(root<14GiB);noautolaunch/CPUtrainingbypass/delete/CROTTCchanges.
  GOAL_SM13§3.8;newtool`tools/debug_sm13_dropout_causality.py`,tests`tests/test_sm13_dropout_causality.py`.

### SM13 next verify — anti-overfit dropout preflight ONLY, RESOURCE BLOCKED

- User yêu cầu verifyhướngcảithiệnđiểm, khôngphảichỉloadadaptedS2backbone. Selectednarrowhypothesis:
  freshnormalizedbaseline vs2xdropoutcandidate,target.25/.20→.50/.40,gap.15→.30; proposedpaired50/head/arm,
  sameinit/seed101gap1101/split/recipe/fixed.5. `reports/SM13_DROPOUT_VERIFY_20261114/PROTOCOL.json`.
- `tools/verify_sm13_dropout_preflight.py` ran CPU-onlyconstructionchecks: initialstate/params/evaloutputs exact,
  trainmodefinite/stochastic. **Nooptimizer/backbone/GPU/F1evaluation**. `PREFLIGHT.json`statusPREFLIGHT_ONLY_RESOURCE_BLOCKED.
- /var/tmpfree12.53GiB<14GiBmainexportfloor; home4GiB+1.5GiBoutputbudget/RAMpass. CROTTCretrievalactive,
  untouched. Do notdelete/moveweights/cache/haltretrieval/lowerfloor orqueueautomatictraining. Noactualpilotdriver,
  training/profile/independentqualityverificationyet; do notclaimqualitygain. Needvalidstorageadmissionfirst.
  `RESULTS_VI.md`; GOAL_SM13§3.7. NotS2/fullbackbonefinetune, nosweep/extension/sealeddata.

### SM13 diagnosis21:49 — normalized50/300 fresh CPU audit, no new training

- User hỏi vì sao model không tăng khả năng. `reports/SM13_NORMALIZED_DIAGNOSIS_20261114/REPORT_VI.md`
  và`AUDIT.json`: freshCPU49.03s/2threads, cache/audio/inputs rehashed,4endpointstates strictload/SHA với prior602receipt,
  toànfit/held evalmode/logitCE replay vàfixed.5counts matched; khôngGPU/backboneforward/optimizer/remote/sealeddata.
- PooledfitF1 **96.6907→99.6100%**, held **40.8143→40.8843%**; targetfitpooledweightedCE.04650→.00516,
  held1.68894→3.03045; wrongheldconfidence≥99%867→1290. Overfit persists afterconditioning, notjustlosslogging.
- MacroAPgainmainlyinsertion(+8.0268pp), S+1.3427/O−.1220pp; targetS/O94.3%positives. OmissionTP141→103/357,
  substitution643→647/1859, totalTP822→786/FP856→709. Target1240FNbothendpoints. APgain≠sustainedmicroF1gain.
- Noevidencemodelabsoluteceiling; speaker/task/representation/pooling/alignment/regularization remainhypotheses.
  Noautoepoch/retune/bestepoch/finetune/newrun; fullTRAIN→DEV_SEARCH/commonadapterpending.59CPUtargetedtestspassed,
  nofullrepoPASS. Newread-onlytool`tools/diagnose_sm13_normalized_plateau.py`, refusesexistingoutputdirectory.
  Historicalreceipts unchanged; latestnormalizedrunremainsCOMPLETE, CROTTCuntouched.

### SM13 latest21:26 — pairedpilotPASS và normalized300 COMPLETE/VERIFIED, không còn train

- TrainingPID1763886terminal; workSTATUS=COMPLETE_NORMALIZED300_VERIFIED, target26400/gap13800updates,
  totalwindow3744.75s(~62.41min). `reports/SM13_RAW_NORMALIZED_20261114/FINAL_VERIFICATION.json`602states
  VERIFIED; `PILOT_VERIFICATION.json`204states/gateverified trướcconditional300. No postprocesserror/repair mới.
- Matched50vs50 raw→normalized: F1@.5 38.6507→40.8143%, macroAP25.7442→30.7764%; all6gatesPASS,
  cả3heldspeakersF1tăng/FWRgiảm. Normalized300: F1@.5 **40.8843%**, macroAP **33.8588%**,
  precision52.5753%/recall33.4468%;APtăngnhưngF1gầnngang50. Last10F1meanat30039.6455%,at5042.1783%;
  targetfitCE.1096→.0321 nhưngheld1.6889→3.0304: overfitvẫn còn. Không chọnbestepoch200F143.2289% thayfixed300.
- `RESULTS_VI.md`, `NORMALIZED300_CURVES.png`, `NORMALIZED300_REPORT.json` trongreportdir.
  Stop300; noautomaticextension/thresholdretuning. Historical350 giữnguyên, khôngrunactive.
  Same1350fit/450reusedTRAINheld, onepairedseed/neuralonly; normalizationevidence=pilot50vs50,
  khôngnorm300vsraw50matchedgain; fullTRAIN→DEV_SEARCH/commonCROTTCadapter/test vẫn pendinggates.


### CROTTC latest21:17 — evaluator replay COMPLETE/VERIFIED; user yêu cầu preserve rồi rerun

- CPUreplayđãCOMPLETE/download/independentlyverified:599DEV×5=2995predictions,
  `reports/CROTTC_IF_KAGGLE_20261114/EVAL_REPLAY_20261114/retrieval_v1/VERIFICATION.json`.
  Legacy3F1=54.917319/54.939759/54.720701%,mean54.859260%;repaired54.581238%,
  delta−0.278022pp,repair2seedsidentical. ActualremoteattentionSHAkhớppublic/localdiagnosis.
  `EVAL_REPLAY_20261114/RESULTS_VI.md` chứacounts/PER/limits;scopefixed3150/CPUpaired,
  **khôngủnghộevaldropoutlànguyênnhânchínhplateau60%trongphépkiểmnày**.
- RetrievalPID1786236đãterminalnormally;khônggọicònwaiting. Maincontroller1291861/window002
  API21:17RUNNING;latestindependentlyverified3150untilwindow002exportgate.
- User “lưu lại toàn bộ rồi chạy lại là biết”:preservecode/config/predictions/metrics/logs/receipts,
  checkpointrefs/hashkhôngduplicateweights trênlowdisk. Rerunmeaningchưarõeval-onlyvsfreshtrain/
  recipe nào;đãhỏixácnhận,**chưa stop/reset/launchnewtraining**. GiữKagglemainđangchạy,
  khônglocalCROTTC/Colab/sweep/nhảGPU/alterLR/data/testDEV_CAL. Snapshotdựkiến
  `reports/CROTTC_IF_KAGGLE_20261114/PRESERVED_20261114_2117/`;chỉcertify khiPRESERVATION_RECEIPT.json
  vàhash/tarextractionverificationđãPASS,khôngcoifoldertồntạilàđãarchiveđủ.


### CROTTC eval replay20:28 — CPU native profile PASSED, 25/599features, fullmetrics pending

- Decision `CROTTC-EVAL-REPLAY-20261114-v1`; reports
  `reports/CROTTC_IF_KAGGLE_20261114/EVAL_REPLAY_20261114/{DECISION.json,PROTOCOL.md,LAUNCH_RECEIPT.json}`.
  Private https://www.kaggle.com/code/minkhoaa/langfens-crottc-eval-dropout-replay-cpu-v1 v1,
  **enable_gpu=false**,serverprivacy/scriptreadbackverified. API20:28RUNNING;
  nativeCPUprofilemedian/p95/longest **PASSED**1.43/2.31/3.00s,RSS~2.3GB,0visibleGPU,
  conservativefeatureestimate880s<6000s. Live25/599features;**chưa actual599DEVscore**.
  Evidence `EVAL_REPLAY_20261114/STREAM_2028.log`;progresskhôngđồngnghĩaindependentdownloadverification.
- Nativefull315MWavLM+heads CPUFP32/frozen/eval;fixedparent3150SHA47fbf1d9...ab5ceaf,
  exact599DEV_SEARCH/audiohashverified,read-onlyRAMfeaturecache. Trials interleaved
  legacy3047/repair3047/legacy3048/repair3048/legacy3049. Repaironlyevalattentiondropout0
  disposableRAMcontext,weights/RNGpreserved;nogolddecoderinputs/postdecodeoriginalmetrics.
  CPUunpadding/cache khácmain2T4padding nên chỉpairedcontrast,khôngrewritehistoricalmetrics.
- ActualCPUmedian/p95/longestfeatureprofile trướcfullpopulation;7200driver/10800platformsec,
  2threads/RAM12initial4runtimeGiB,zerooptimizer/noGPU. Failbudget/mismatch/nonfinite→preserve,
  khôngauto-retry/relaunch/changepopulation. Test/DEV_CALsealed;assetsv1+window001/1 reuse,
  khônguploadWAV/weights/newdataset/credentials.
- `research/crottc_if_official/eval_replay.py`;preparetool/kernel under`tools/*crottc_eval_replay*`.
  **53CPUtests passed/1strictxfail** (originalRoPEMHAevaldeterminismknownbroken,notfixedproduction).
- FiniteCLIretrieval **PID1786236** (`tools/watch_crottc_eval_replay.py`,11700smaxwait+downloadverifygrace),
  report`EVAL_REPLAY_20261114/retrieval_v1/RETRIEVAL_STATUS.json`. NoGPU/traininglauncher/cancel/retry.
  Oncecomplete downloadonlysmallJSON/JSONL/logs andindependentlyscore2995predictions/599IDs/all5trials,
  matchper-speaker/counts/PER/silpolicies/repairrepeat;`VERIFICATION.json`onlyonpass.
- MaincontrollerPID1291861/window002stillRUNNING,8productiondrivers/dependencyrecipeunmodified.
  Khôngdừng/restart/upgrade để nhườngGPU; khôngcommit/pushGit/testopening. Checklatestlogs/status
  beforeclaimresult. CPUdiagnosticplatformselected forCROTTC, khôngđộngSM13localonlyscope.


### SM13 latest — fresh raw-vs-normalized pilot50 RUNNING; conditionaltotal300 đãủyquyền nếuPASS

- Latestuser **“đối chứng raw vs normalized trước, nếu ổn thì train lại với300epoch xem”**.
  Decision `SM13-RAW-NORMALIZED-CONDITIONAL300-20261114-v1`, GOAL_SM13§5.7. Historical350terminal giữnguyên;
  đây là NEWfreshheads, khôngold350extension. Localonly/noKaggle/Colab/CROTTCcommands/sealeddata.
- **TrainingPID1763886**, module `research/sm13_backbone_side/raw_normalized.py`, statusPILOT_TRAINING/rawtarget
  vừaobserved; disposablelocalheadprofile4batchespassed. SharedGPUlock/2threads/max4h.
  `reports/SM13_RAW_NORMALIZED_20261114/{REGISTRATION.json,PROTOCOL.md,TRAIN.log}`;
  work `/home/khoa/.local/share/langfens/sm13_raw_normalized_20261114_v1/` ngoàiGit vàtrênseparate/homeFS.
- /var/tmpfree~15.41GiB khôngđủ checkpointcampaign+14GiBmainreserve, nên newstatesONLY/home:
  **4GiBhomefloor+5.5GiBoutputadmission**, root/var/tmp14GiBmainexportfloor khônghạ. Existing14.95GiBcache
  read-onlyreuse/rehashed; no delete/compressoldartifacts/newcache. RAM16initial4runtimeGiB/VRAM512MiB.
- Pair50/head/arm, same1350fit450held/seed101gap1101/trainableinit/RNG/batchorder/recipe/LR/loss/batches;
  only13fit-onlylayerRMSscaling differs. Orderrawtarget,normtarget,rawgap,normgap.179SM13CPUtests passed,
  includesfactoryresumebuffer/Adam/init/gatefixtures (no fullrepoPASS). `epoch50.fit_head` thêmoptionalmodel_factory,
  defaultbehaviorunchanged; `conditioning` addsread-onlyscale/layer_logitsproperties; originals inPROVENANCE/.
- GatePREOUTCOMES: fixed50microF1@.5delta≥+1pp, meanlast10delta≥0, macroAPregression≤.5pp,
  eachS/O/IAPregression≤2pp, ≥2/3heldspeakersnotworse, FWRincrease≤.5/câu. No thresholdtuning/bestepoch/gateloosen.
  Onepairedseed/reuseddevelopment;27orderedspeakerbootstrapdescriptiveNOTconfirmation.
- **BlockingindependentCPUverifier subprocess** `tools/verify_sm13_raw_normalized.py --stage pilot`, noGPU/training/remote,
  checks204pilotstates/predictions/freshCPUheldinference/recomputedfit-onlystats/rawhistoricalparity/gate.
  Driver invokes it; no separatewaitingprocess/cron/subagent. `PILOT_VERIFICATION.json` only onsuccess.
- Ifquality+correctnessPASS: save `CONDITIONAL300_ADMISSION.json` and exactfreshnormalized50fullstate,
  continue50→300 =**300total/head, not300additional**, no oldraw350warmstart. Same frozenstats/recipe/fixed.5;
  verify602normalizedstates/curves via `--stage final`, `FINAL_VERIFICATION.json` onlyonsuccess.
  IfFAIL: stopafterpilot, noauto300/retry/sweep. Fixed300vsraw50 is NOT matchednormalizationgain evidence;
  onlypilot50vs50 is paired. Allartifacts preserved; fullTRAIN→DEV_SEARCH/commonCROTTCpending.


### CROTTC diagnosis20:00 — CONFIRMED evaldropout defect; chưa patch production

- User hỏi vìsaoF1chưavượt60,thuậttoán/model/cáchtrain cóvấnđềkhông;usercontinuechẩnđoán.
  Report `reports/CROTTC_IF_KAGGLE_20261114/DIAGNOSIS_1945/REPORT_VI.md` vàdiagnosticcurves.
  **SpeechBrain1.0.3 RoPEMHA passes dropout_p=self.dropout(.1) trongeval**;PyTorch2.6SDPA
  luônapplydropoutnếu>0,bấtkểmodule.eval(). Publicv1.0.3sourcebyteSHAkhớplocal;
  remotedependencyversionkhớp, chưa trực tiếpfetchremoteattentionfilebytes.
- Actualverified3150headsstrictload+syntheticWavLMshapedmemory8repeats chooutputs khác;
  disposableinRAMattentiondropout0controlchoidenticaloutputs,weightsunchanged.
  **KhôngcóactualnativeDEVF1delta**,khônghứafixvượt60/đổmọiplateaucholỗi.
  HistoricalbestF1stochasticobservationsgiữnguyên;productiondriver/dependency/recipeUNMODIFIED.
- **CorrectionBNexplanation:** actualAM có0BatchNorm/14LayerNorm,norunningstats.
 21CNNstate tensorsgiữnguyênnative;24Transformerqprojectionsđãđổiở3150. Khôngđổper-utteranceBN.
-4173updates483–4655finite/nonzeroprobe;gradientclip100%preclipnorm>5khôngtựlàbug;
 TRAINlossfullblock41–55mean2.1454→76–81mean1.7065,DEVF1plateau/regress.
 OTmathloss/logitgrad/alphagradkhớpindependentCDFtransport ở8CPUcases;CRrelativepublicsource
 factor110×trong1unpaddedpair,khôngđượcgọinativebatch32componentdominanceđãđo.
- Tests `DIAGNOSIS_TESTS.log`:58passed/1strictxfail (evaldeterminismknownbroken),khôngallpass/fixedclaim.
 Initial10pass/1failedprobeđượcpreserve. Newtoolsdiagnosticloganalysis/evaldropoutprobe+testonly;
 khôngnativeWavLMforward/localtrain/newKagglelaunch/testDEV_CAL/commit/push.
- Nextpriority:registeredboundednativeeval-onlysamecheckpointrepeats/repaircontrol trênKaggle,
 saveper-utterancepredictions/matchedcleanTRAINDEV/componentloss;khônghotpatchcurrentruntime,
 khôngstop/restart/upgradepackagegiữacampaign/khôngđổibackbonehayLRbừakhichưatáchcauses.
 Controller1291861alive/window002RUNNINGat19:45;latestindependentverified3150. Checklivebeforeclaim.


### SM13 debug latest — CPU-onlyaudit complete, opt-inconditioningcandidate CHƯA train

- User yêu cầudebug/continue sau350. Không newproductiontraining/GPU/remote/CROTTCchanges.
  `reports/SM13_S1_DEBUG_20261114/REPORT_VI.md`, `cpu_audit_v2/AUDIT.json` statusCPU_FORENSIC_AUDIT_COMPLETE_NO_TRAINING.
  Fullfit/heldCPUheadinference50/150/350 reproduce savedGPUprobs maxdelta5.94e-6/0argmaxdisagreement;
  labels/referenceadjacency/eventorder/split/migration khôngmismatch; sklearnfixedendpointmetricsreplay.
- Overfitconfidenceconfirmed: targetfitF1 .9061→.9879 nhưngheldF1 .3943→.3774, evalmodefitweightedCE
  .1024→.0209 nhưngheld1.4602→3.9133, wrongconfidence≥.99 626→1330. Noresumereset/labelshufflefound.
- Layerconditioningweakness: allfitS1RMSmax/min89.48target/100.19gap vs same256sampleS0~3.10/6.51;
  HF WavLMstableencoder intermediatepre-finalnorm/finalpostnorm. Rawmixing recipe không cân layeramplitudes;
  contributiontoquality vẫnhypothesis, chưacausalproof. Scale .4339→.0437 compensatesgrowingheadweights.
- FreshCPU315Mfeature replay3TRAINrelativeL2~1.5e-6..3.7e-6, headprobdelta≤3.32e-6/0argmaxchange,
  nhưng strictpooledCPU/GPUallclosefailedsparsecoordinatesở2samples. Preservefailedauditv1, v2measurefailures,
  KHÔNG nới tolerance/overwritecache/gọi fullnativeparityPASS. Doublewaveformnorm1counterfactualsmall,
  khôngevidence đủ đổ lỗi nó. Fourinheritedzero canonicalrefs ER)/D_/SPN/Z_ sharedS0S1, khôngrelabel.
- New `conditioning.py`: opt-infit-only13layerRMS/frozenbuffers/newheadnamespace, rejectsleaks/nonfinite/zero,
  cannotloadoldheadasconverted. `CONDITIONING_CANDIDATE_STATS.json` hasfit-only44746target/46096gapstats,
  oldcacherehashed, noheadfit.157SM13CPUtests passed(nofullrepoPASS). Nochangeproductionheads/epoch50/backbone.
- Auditv1probabilityfloorCE underflowartifact fixedinauditv2 via freshlogitCE; productionevalalreadyusedlogits.
  `tests/test_sm13_debug.py` underflowfixture; nohistoricalmetricsrewrite.
- Nextboundedexperiment needsfreshregistration/admission: matchedraw-vs-RMSconditionedfreshS1heads,
  same1350/450split/seed/recipe/evaluator; noLR/layersweep/conflatedchanges or old350automaticresume.
  Earlystoprule futureonly/preregistered, current350fixedresult retained. FullTRAIN→DEV_SEARCHpending,
  localonly/CROTTCreserve/testDEV_CALsealed; userdebugrequest khôngnewproductionauthorization.


### SM13 latest19:41 — target350/gap350 COMPLETE và404states VERIFIED, không còn train

- `reports/SM13_S1_PLUS200_20261114/VERIFICATION.json`: S1_PLUS200_STATES_AND_FULL_CURVES_VERIFIED,
  404states, full1..350curves/keyedmetrics/frozen50thresholds/retainedcache/parentAdamRNG verified.
  TrainingPID1675056/verifierPID1676328 terminal; target30800/gap16100updates, runtime1674.63s(~27.91min).
  Không lỗi/reportrepair mới, không newtraining/remote/CROTTCcommands trong statuscheck này.
- Fixed150→350: diagnosticF1@.5 **40.6829→37.3146%**, frozen50thresholdF1 **43.7263→41.2178%**,
  macro3AP **28.9543→29.0901%**. MeanF1last10at150/35039.7981/38.3997%; targetfitCE.0958→.0558,
  heldCE2.3781→3.9133: detectionregression/overfitting, không gọi thêm200 cải thiện rõ.
- Report `CURVE_REPORT.json`, `VERIFIED_LEARNING_CURVES.png`, `VERIFIER_STATUS.json` trongreportdir.
  Stop350; khôngautomaticextension. Same450reusedTRAINheld, exploratoryneural-only, khôngfinalDEV_SEARCH/test.
  FullTRAIN1800→DEV_SEARCH599/commonCROTTCadapter vẫn pendinggates, SM13LOCALONLY/testDEV_CALsealed.


### SM13 latest — user thêm200epoch, LOCAL150→350 active

- Latestuser **“train thử thêm200epoch xem”** sau verified150/plateau/overfitting. Decision
  `SM13-S1-PLUS200-20261114-v1`, GOAL_SM13§5.6: **200bổsung/head,total350**, khôngautoextend sau350.
  Localonly, noKaggle/Colab/CROTTCcommands. Prior150terminal/repair/204stateverification vẫn giữ nguyên.
- **TrainingPID1675056**, driver `research/sm13_backbone_side/local_s1_plus200.py`; target158/update13904
  vừaobserved, gap sẽresume150 sau target350. Đây là livesnapshot, kiểmSTATUS/process trước báođangtrain.
  Registration/log: `reports/SM13_S1_PLUS200_20261114/`; work
  `/var/tmp/langfens-crottc-full-20261113-t_car8m4/sm13_s1_plus200_v1/`.
- Newlineage migration chỉ decision/identitytags, exactfull150model/Adam/RNG/history/cursor/scaler/scheduler;
  frozen native315M/cacheFP32/same1350fit450held/seed101gap1101/recipe/LR/loss/weightdecay/batches.
  No newcache/backbonefitting/dataaccess. Frozen50threshold+fixed.5diagnostics, no retune/bestepochselection.
  Complete1..350curve(prefix1..150fromverifiedparent), fixed50/100/150/200/250/300/350endpoints.
- SharedGPUlock/2threads/max4h, RAM16initial4runtimeGiB/VRAM512MiB/disk14GiBmainreserve,
  **3GiBnewoutputadmission** (free~17.53GiBprelaunch), no deleteartifacts/reducefloors. Atomicfullepochstates,
  safeoptimizerboundaryresourcepause, stopnonfinite/mismatch/no skipbatch.143SM13CPUtests passed(noCUDA),
  includes150migration/350curve/frozen-thresholdchecks; no fullrepoPASSclaim.
- FiniteCPUverifier **PID1676328**, `tools/verify_sm13_local_s1_plus200.py`,max4h,noGPU/remote/launcher;
  checks404states(400new+2migrated150+2last), parentstate/Adam/RNG/curves/metrics/retainedcache.
  `VERIFIER_STATUS.json`; `VERIFICATION.json` onlywhensuccess. No terminal350claimyet.
  FullTRAIN1800→DEV_SEARCH599/commonCROTTCadapter stillpending; sealedtest/DEV_CALunchanged.


### SM13 terminal150 — VERIFIED sau CPU-only report recovery, không còn train

- S1 đãCOMPLETE target150/gap150 (target13200/gap6900updates), window855.09s; PID1579721/1583121terminal.
  **Latest verification** `reports/SM13_S1_PLUS100_20261114/VERIFICATION.json` xác nhận204states/fullcurves/cache;
  `RECOVERY_VERIFIER_STATUS.json` thắng oldwaiting/failureobserver khi xác định kết quả đãverify.
- Training đủ150 rồi originaldriverfailed `invalid detection score` lúc postprocess:7 FP32derivedpS+pO=1+1ulp.
  OriginalfailedSTATUS/log/observer/source được giữ nguyên (runtime và `POSTPROCESS_INCIDENT/`), khôngrewrite.
  `detection.scores` chỉ bound derivedroundoff≤2dtypeeps; savedprobabilities khôngđổi, invalidmaterial/directexports
  vẫn fail; allregisteredthresholddecisions unchanged. `POSTPROCESS_RECOVERY.json` pinssourcechange vàprotectedSHA.
  CPU-only reportrepair/verifier KHÔNG train/GPU/remote/reseed; registration/training sourcehashes lịch sử giữ nguyên.
- Fixed50/100/150: diagnosticF1@.5=38.6507/41.7724/40.6829%; frozen50thresholdF1=42.6627/44.0037/43.7263%;
  macro3AP=25.7442/27.3496/28.9543%. Meanlast10F1~39.78/39.95/39.80: plateau/noisy dùAPtăng.
  Targetfitloss .2011→.0958 nhưng heldloss1.4602→2.3781, consistentoverfitting; khôngbackbonegainclaim.
- Report `STATUS.md`, `CURVE_REPORT.json`, `VERIFIED_LEARNING_CURVES.png` trongreportdir.126SM13CPUtests passed,
  khôngfullrepoPASS. Original50parents và14.95GiBS1cache retained/rehashed. Noautomaticextension sau150.
  FullTRAIN1800→DEV_SEARCH599 matchedS0/S1/commonCROTTCadapter vẫn pendinggates; test/DEV_CALsealed/localonly.


### SM13 latest — user yêu cầu THÊM100epoch, S1 local50→150 đang chạy

- Latestuser **“train tiếp100epoch xem đường cong thế nào”** sau báo plateau/regression. Decision
  `SM13-S1-PLUS100-20261114-v1`, GOAL_SM13§5.5: **100bổsung/head, total150**, trực tiếp override
  priorconditionalmax100 **chỉ choS1run này**. Localonly, khôngKaggle/Colab/canthiệpCROTTC; khôngautoextend sau150.
- Parent S1target50/gap50 đãCOMPLETE/VERIFIED102states, runtime440.05s; oldPID1550406/verifier1553467terminal.
  `reports/SM13_NATIVE_LOCAL_20261114/FIT_VERIFICATION.json` thắng oldrunningheader17:23.
  Same450heldmacro3AP25.7442%/diagnosticF1@.5=38.6507%, thấp hơnS0(33.4066%/46.3634%);
  cuốicurvekhôngtăngrõ. Khôngbópmetric/cherry-pick để gọi backbonegain.
- Newdriver `research/sm13_backbone_side/local_s1_extend.py`, **trainingPID1579721**; livecheck vừa có
  targetepoch78/update6864 (launch/progresssnapshot, khôngterminalclaim). Registration/log ở
  `reports/SM13_S1_PLUS100_20261114/`; work `/var/tmp/langfens-crottc-full-20261113-t_car8m4/sm13_s1_plus100_v1/`.
- Exactfullstate50model/Adam/RNG/history/cursor/scaler/scheduler được giữ nguyên khi migrate outputlineage;
  chỉ decision/identitytags đổi, có migrationreceipt/exactequality. Originalepoch50/checkpoints/features khôngoverwrite.
  Same1350fit/450held, seed101/gap1101, geometry/headrecipe/LR/loss/batches/normalization/cacheFP32.
  No backboneforward/backward, no newcache, no newdata/test/DEV_CAL. Checkpoints51..150atomic/fullstate mỗi epoch.
- Fixed0.5 và **frozenepoch50calibrationthresholds** làm diagnostic; KHÔNG retune thresholds theoepoch.
  Report complete1..150curve, fixed50/100/150endpoints, loss/S/O/IAP/F1/per-speaker/falsewarnings;
  `CURVE_REPORT.json`/`LEARNING_CURVES.png` khi terminal. KhôngsoS1@150 vớiS0@50 rồi gọi purebackbonegain.
- SharedGPUlock/2threads/bounded4h; RAMinitial16/runtime4GiB,VRAM512MiB,disk14GiBmainreserve +2GiBoutputadmission.
  Resourcepause savesfullstate atsafeoptimizerboundary; noautomaticretry. CPU124SM13tests passed(CUDA disabled),
  gồm metadata-onlymigration/full150curve/frozen-thresholdfixtures; khôngclaimfullrepoPASS.
- FiniteCPUverifier **PID1583121**, `tools/verify_sm13_local_s1_extend.py`, max4h, noGPU/launcher/remote;
  `VERIFIER_STATUS.json` và `VERIFICATION.json` chỉ cókhi204states/curve/metrics/cacheverified.
  Chưaverifiedterminal150lúc snapshotnày. CommonfullTRAIN1800→DEV_SEARCH599 vẫn pendinggates.


### SM13 cập nhật17:23 — NEW NATIVE LARGE LOCAL ONLY, S1 calibration heads đang train

- Latestuser yêu cầu **tiếp tục backbone/modelbase mới, mọi tác vụ SM13 mới LOCAL ONLY, không dùng Kaggle**.
  Decision `SM13-NATIVE-LOCAL-20261114-v1`; GOAL_SM13/AGENTS đã cập nhật. Không gọi lại S0epoch50 là đangtrain;
  không dùng S0 chạy thêm thay cho S1. CROTTC không bị stop/reset/resubmit/đổirecipe bởi tác vụ này.
- S1 = **native pretrained WavLM Large315453120 frozen/eval FP32**, revisionc1423ed94bb01d80a3f5ce5bc39f6026a0f4828c,
  layers0,2,...24, fresh1024Dheads. LocalGTX1650median/p95/longest+disposableheadbackward profile PASSED;
  fullTRAIN1800cache14.95GiB đãCOMPLETE/independentlyverified1800row/fullfileSHA/geometry/finite.
  `reports/SM13_NATIVE_LOCAL_20261114/{LOCAL_PROFILE_VERIFIED,LOCAL_CACHE_VERIFICATION}.json`.
- **Localheadtraining PID1550406**, target50/gap50, seed101/gap1101, calibration1350fit/450heldABA+ASI+BWC.
  Đây là speaker-disjointTRAINcalibration/reuseddevelopment trước fullTRAINfit, **không phải** full1800→DEV_SEARCH599
  comparison/independentconfirmation/fullfusion hay quyền soCROTTC trên450TRAIN. Check STATUS/process trước liveclaim.
  Work `/var/tmp/langfens-crottc-full-20261113-t_car8m4/sm13_s1_calibration50_v1/`; log+registration trongreportdir.
  Cache `/var/tmp/langfens-crottc-full-20261113-t_car8m4/sm13_s1_train_cache_v1/`; khôngoverwriteS0cache.
- Newmodules `local_s1.py`, `local_s1_fit.py`, `detection.py`; bounded4h/sharedGPUlock/2threads,
  initialRAM16GiB/runtimeRAM4GiB/VRAM512MiB/disk14GiBfloor, ưu tiênmainCROTTCexport. NoKagglecalls/fallback.
  `epoch50.fit_head` thêm optionalgeometry/decision/resourcecallback, defaultS0 giữ nguyên;
  prechange source preserved `reports/SM13_NATIVE_LOCAL_20261114/epoch50_before_geometry_extension.py`.
  CPU104tests passed với CUDA disabled; không claim fullsuitePASS.
- Metriccontract v1 + fixtures đã khóa: event-leveltargetS/Opositiveonce + exactgapI; scorepS+pO/pI,
  threshold441pairs TRAINheldterminal50, maxmicroF1→precision→higherthresholds, samebudgetS0;
  fixed0.5curve diagnostic. S/O/ItypedAP/F1 retained; no paperhierarchicalF1claim. CROTTCadapter cònpending.
  `METRIC_AND_CALIBRATION_REGISTRATION.md` + `FIT_CALIBRATION50_REGISTRATION_v1.json` trongreportdir.
- FiniteCPUverifier **PID1553467**, CUDA disabled, max4h, khônglauncher/retry/remote:
  `tools/verify_sm13_local_s1_fit.py`; `FIT_VERIFIER_STATUS.json`/`FIT_VERIFICATION.json` (nếuterminalverified).
  Kiểm102states/Adam/RNG/keys/predictions/metrics/retainedcache; chưa cóverifiedterminalresult lúc bàn giao.
- Quyền conditional50→100 theoGOAL_SM13§5.4, khôngauto-extend. Saucalibration: verifycurves/states,
  matchedfullTRAIN1800S0/S1fit và DEV_SEARCH599predictedalignment/cache/evaluation vẫnpendinggates;
  giữnormalization/capacitycaveats, test/DEV_CALsealed. RemoteS0QUEUED làhistorynotcancelled; khônglaunch/cancel/CLI nó ở tasklocalonly.


### Cập nhật17:20 — export3150 verified; window002 admitted; fidelity audit vẫn OPEN

- **Full window001 export3150 đãdownload/SHA/load/Adam/RNG/scaler/cursor/order/pendinggradient
  verified** bởicontrollercósẵn, trướccontinuation. Latestindependentlyverifiedkhôngcòn482:
  update3150/internalepoch55cursor480,emptypending,SHA
  `47fbf1d91ab0838970b05d9d0af67d224c9c8294e3f3950c5c9069414ab5ceaf`.
  Receipt `reports/CROTTC_IF_KAGGLE_20261114/AUDIT_1705/WINDOW001_VERIFICATION.json`.
- ControllerPID1291861 tựnối **optimized-window002/v1**, API17:19:50**RUNNING**/failureMessagenull.
  Initialstream17:20đangbootstrap/runtimeverify,**chưacóupdate>3150đượcxácnhận**.
  KhônggọiplatformRUNNINGlàđãcóproductionupdates. Khôngauditlauncher/duplicate/stop/restart.
  Link https://www.kaggle.com/code/minkhoaa/langfens-crottc-if-optimized-window002 .
- `AUDIT_1705/OPERATIONAL_FINAL.json` pinsactualplatformstatus vàlatestverifiedparent;
  disk/var/tmpcòn~20GiB. Quotađãvượtallowednhưngwindow002vẫnđượcadmit;khôngdừngnó,
  khônghứacácwindowsau300/fullpaper. Benchmark/curve55epochs vàledger90tests ởentrybêndưới.

### Cập nhật17:16 — user yêu cầu check Kaggle/curve rồi audit fidelity (download snapshot)

- Không stop/restart/resubmit/thay recipe. Mainoptimizedwindow001v1 đãCOMPLETE,
  driverWINDOW_EXHAUSTED_RESUMABLE tạiupdate3150/internalepoch55cursor480,
  55completedepochs vàdởhuman56/300. ControllerPID1291861 cònhoạtđộng vàđangtải
  fullstateexport vào `/var/tmp/langfens-crottc-full-20261113-t_car8m4/kaggle_optimized_campaign_02/window001/`.
  Tại17:16latestindependentlyverifiedvẫn482;metadata3150khôngthayfullcheckpointproof.
  Finalsnapshotnếucó: `reports/CROTTC_IF_KAGGLE_20261114/AUDIT_1705/OPERATIONAL_FINAL.json`.
- Đủcontiguouscurve1–55, reconcilevớidownloadedEPOCHS. Latest55F154.1450%/PER25.9173%;
  bestF1vẫn46=54.3590%,bestPER28=25.2566%. Mean51–55F152.7162%/PER26.6358%.
  `reports/CROTTC_IF_KAGGLE_20261114/AUDIT_1705/REPORT_VI.md`, `CURVE/CURVE.png`.
  Localdevelopmentonly;khôngpaperbenchmark;khôngsoAMvớiIF71.77%nhưsameprotocol.
- FidelityledgerF1–F7đãdựngở`AUDIT_1705/FIDELITY_LEDGER.md`;gatesvẫnOPEN.
  NewCPUfixturesTable2paper;onlinePERkeepssil/offlinestrips→50%vs0%synthetic;
  effectiveConformerkernel3stride1(length15→15),paperwordingstride3unresolved.
  Raw3allowlistedTRAINTextGrid/hashverifiedtruyunknown-CtừZ_/ER)/D_;khôngđổilabel.
  Source65files+2manifestshashverified,GitHubpinnedtree79entries/notruncated/nodataJSON;
  **90CPUtests passed**,khôngnativeIFtraining. Tool `tools/audit_crottc_fidelity.py`
  metadata-only vàtest `tests/test_crottc_fidelity_audit.py`;production8driverhashesunchanged.
- Freshquotarawused33047.36.0s/allowed21600s/reset21NovUTC;keeprawpayload.
  Nextadmissionmayblocked/queued,preserve3150exportfirst;khôngfallbacklocal/Colab.
  Test/DEV_CALsealed;khôngGPUtraining/sweep/cron/timer/subagent/commit/push.

### Cập nhật16:46 — user tách thành HAI GOAL, docs-only, không đổi training

- **`GOAL.md` là index**, không còn soleactiveworkqueue. Hai tài liệu điều hành:
  1. **`GOAL_CROTTC_IF.md`** — full-paper target, fidelity/data/recipe/metric/stages/coverage gates,
     preservation của currentAMdevelopmentcampaign và Kagglequota/export contract.
  2. **`GOAL_SM13.md`** — SM13 lớn hơn/frozenWavLMLarge, common detectionF1 và retainedS/O/IAP,
     matchedcontrols/leakage/calibration/resource gates, kết quả S0epoch50 đầy đủ.
- `SIDE_GOAL.md` chỉ compatibilitypointer, khôngGOALthứ3. Decision `RESEARCH-GOALS-SPLIT-20261114-v1`.
  OriginalGOAL/SIDE_GOAL/AGENTS/HANDOFF được copynguyênvẹn/hash ở `reports/GOAL_SPLIT_20261114/`.
  Các header cũ bên dưới là lịch sử, không thắng snapshot/goals mới.
- **CROTTC main vẫnRUNNING**, controllerPID1291861, privateoptimizedwindow001v1.
  Snapshotliveupdate2950/internalepoch51cursor1376 (humanepoch52/300AM), completedval51
  F151.4233%/PER27.4774%; bestF1epoch46=54.3590%, bestPER28=25.2566%.
  Fullcontiguous1–51curve/sources `reports/GOAL_SPLIT_20261114/CROTTC_CURVE/`.
  Latestindependentlydownloadverifiedcheckpointvẫn482; phải download/verifylongwindowexport.
- **Không gọi currentrun là fullreproduction**: local1800/599 khácpaper2429/268/900,
  NJSđãfit nhưngpapertestcóNJS, annotation/augmentation/IFrecipe cònopen. Exactpaperbenchmark
  cầnfreshcleanpretrainedlineage, không promote currentcheckpoint bằng đổi tên/loạiNJSởepochsau.
  User yêu cầu giữ currentrun khôngbreak; lần táchdocs này khôngstop/restart/resubmit/sửa driver/recipe.
- **SM13local S0 đãCOMPLETE/VERIFIED**, khôngcònđangtrain: target50+gap50,102states/predictions/
  metricsverified, runtime551.57s, PID1448557/verifier1451451đãterminal. Same450TRAINheldseed101:
  macro3AP30.8974→33.4066%(+2.5092pp), omission32.4963→30.4143%; one-reused-fold/neural-only.
  `reports/SM13_EPOCH50_20261114/LOCAL_VERIFICATION.json`, `LEARNING_CURVES.png`.
  FullS1/S2fit và common detectionF1 chưađượcchứngnhận. Khôngso33.4AP với54.4CROTTCF1.
- SDKquotafreshsnapshotused31899.255732.0s >21600sallowed/reserved0, reset21NovUTC;
  rawpayload `reports/GOAL_SPLIT_20261114/REMOTE_SNAPSHOT.json`. Currentadmittedmainvẫntrain;
  nextwindowmaybequota-blocked. Khônghứauninterrupted300/fullpaper, khôngautoCROTTCfallbacklocal.
- RemoteSM13S0v1 vẫnQUEUED/notcancelled; oldretrievalPID1418016cònwait, khônglauncher.
  Localguardđãhếtcùngcompletedlocalrun; đừngclaimguardvẫnalive hoặcguesscancelSessionID.
  Trước futureSM13run, xử lý queuedduplicate/profiling/resourcegates theo GOAL2.
- Test/DEV_CALsealed; mới chỉ táchGOAL, không mở dataaccess/newGPUrun/cron/subagent/commit/push.

## Historical handoff snapshots — không dùng thay live verification

### Cập nhật16:00 — BOTH PIPELINES ACTIVE, SM13 LOCAL / CROTTC KAGGLE

- Latestuser: **“sm13 chạy local, không break tiến trình training full paper hiện tại”**.
- LocalSM13 **PID1448557** đangtrainthật trênGTX1650/sharedGPUlock, targetepoch16/update1408
  lúc15:58. Cảhaiheads50epochs, seed101, fixed12/5vs50/50 trênreusedTRAINouter0; khôngE4/fullfusionclaim.
  `research/sm13_backbone_side/local_epoch50.py` reuse26originalFP32cachefiles SHAverified,
  khôngre-extract/copy11GiB hoặcload/trainbackbone. Fullatomicstateeach-epoch+safe-batchboundary.
 4hwindow/2threads/VRAM512MiBfloor/RAMadmission16GiB/disk14GiBmainreserve.
- Runtime `/var/tmp/langfens-crottc-full-20261113-t_car8m4/sm13_epoch50_local_v1/`,
  `STATUS.json`/`target_history.json`/`gap_history.json`, logs/receipts in
  `reports/SM13_EPOCH50_20261114/LOCAL_TRAIN.log`, `LOCAL_LAUNCH_RECEIPT.json`.
- OnefiniteCPUverifier **PID1451451** waitsfor50/50thenchecks102headstates/Adam/RNG/cursors,
  allheldpredictions/AP/F1, fixedendpoints/bootstrap/curves; `LOCAL_VERIFIER_STATUS.json`.
  Itdoesnottrain/resubmit. Command: `tools/watch_local_sm13_epoch50.py`; nativecheckpointresume
  via localdriver`--resume` needsfreshadmissionproof andsamepreservedwork, noautomaticretry.
- **QuotaAPI discovered** via installedkagglesdk`get_accelerator_quota_statistics`:
  GPUused28855.827985s >allowed21600s/reserved0, reset2026-11-21UTC. Rawdurationhasmalformed
  seconddecimal; keeporiginalpayload, parseleadingnumericvalue. `LOCAL_ADMISSION.json`/`QUOTA_1551.json`.
  MainalreadyadmittedwindowstillRUNNING; nextwindowsmaybequota-blocked, donotpromiseuninterrupted300epochs.
- RemoteSM13v1 remainsquota-blockedQUEUED, **notcancelled**: cancellationSDKexistsbutrequiresunknownsessionID;
  publicowner/slugstatusdoesnotreturnID, doNOTguessIDs/deletekernels. Local4hdeadlinebefore6-dayreset;
  in-processguardchecksSM13remotestatus60s; unexpectedRUNNING/APIuncertainty=>pauselocalwithfullstate.
  ItsfiniteoldretrievalPID1418016stillwaits, notatraininglauncher. No mainruntimecommandschanged.
- **MainCROTTC PID1291861 active on Kaggle**, liveupdate2400/internalepoch42cursor192 (human43)
  at15:58; proof `MAIN_UNINTERRUPTED_1600.log`. **No mainstop/restart/resubmit/STOPfile orlocalCROTTC**.
  Latestindependentlydownloadverifiedparent482untilcurrent4hwindowexport. Preserve/download/verifycurrent
  outputs evenifnextwindowquotaadmissionfails. Test/DEV_CALsealed; no originalartifacts overwritten.

### Cập nhật15:43 — userủyquyền SM13 50epoch, GPUadmissionpending (historical)

- Amendment **SM13-EPOCH50-20261114-v1** ở SIDE_GOAL.md/AGENTS.md: useryêucầu
  thử50epochSM13, Kagglefirst, fallbacklocalSM13 **chỉ khi Kagglethiếutài nguyên**.
  MainCROTTCkhôngđượcrestartlocal/đổirecipe. Test/DEV_CALsealed.
- Registeredfirstarm **S0frozenWav2Vec2Base960h**, seed101, refit9TRAINspeakers1350câu,
  heldABA/ASI/BWC450câu (reusedhistoricalouter0development). So target12/gap5 với
  target50/gap50, neural-onlymacro3AP; khôngfullfusion/E4/bestepochcherrypick/backbonegainclaim.
- Code `research/sm13_backbone_side/epoch50.py`, package/verify/watch tools
  `tools/*sm13_epoch50*`. FP32regeneration+16realparitychecks beforefitting,
  atomicper-epoch+optimizer-boundaryfullstate (Adam/RNG/order/cursor), safe2threads/4h.
- Private409MiBinputpackagev1 gồmTRAINrequests/labels/refs/predictedalignment/nativepublicweights/
  smallparitysamples; reusesauthorizedassets1, khôngreuploadWAV/test/rawarchives/credentials.
  Source/privacy/scriptreadbackverified. **221CPUregressiontests passed** trướccurveloggateaddition.
- GPUjob **QUEUED15:43**, chưacóproductionbatch:
  https://www.kaggle.com/code/minkhoaa/langfens-sm13-epoch50-s0
  v1pushed15:29; APIstatusQUEUED/failureMessagenull khôngchứngminhquotaexhaustion.
  **Khôngchạy localduplicate** khiremotevẫncóthểadmit. Nếuquota/resourceERRORthật,
  preserveevidence/confirmnolivejob vàresourceGPUlock/budget trướcfallback; khônggọi codeerrorlàthiếutài nguyên.
- OnefiniteCLIretrieval **PID1418016**, `reports/SM13_EPOCH50_20261114/retrieval_v1/RETRIEVAL_STATUS.json`;
  khôngcron/timer/resubmit/localfit. Checkpoint/prediction outputs downloadverified;11GiBfeaturearrays
  đểprivatekerneloutput,khôngtựclaimđãdownloadcaches. Xem side `STATUS.md`.
- Main **PID1291861/remotekernelRUNNING** khônggiánđoạn. Livelogupdate2218/internalepoch38cursor1664,
  tứcđangepoch39; completedepoch38 F150.3142%/PER27.1725%; bestboth28 F154.1639%/PER25.2566%.
  Full38epochcurve `reports/CROTTC_IF_KAGGLE_20261114/CURVES/SNAPSHOT_1543/`.
  **Kagglelivelogcóthểchỉtrảtail**: mergepinnedpriorcurve, requirecontiguousepochs trướcbest/mean.
  `INVALID_TAIL_ONLY_1542/` làreportkhônghợplệđãpreserve, khôngdùngtríchđiểm;1543đãsửa/merge đủ38.
  Fullcheckpointdownloadverifiedvẫn482untillongwindowexport. Khôngclaim300/fullpaper/gain.

### Cập nhật14:23 — side native feature smoke VERIFIED

- `SIDE_GOAL.md` lưu toàn bộ thảo luận và bounded sideexperiment userauthorized.
- SideCPUfeature kernel đãCOMPLETE/download/verified8TRAIN×2full315Mfrozen,
  driver43.91s,đúng13layerfeature/N+1gaps. Proof:
  `reports/SM13_BACKBONE_SIDE_20261114/native_smoke_v1/NATIVE_CPU_FEATURE_VERIFICATION.json`.
  RetrievalPID1310766đãterminalnormally. KhôngGPU/backward/fit/gainclaim;
  source482làprovisional, finalparent/fullcache/headfit/developmentmetrics vẫnpending.
- CPUregression209passed. Maintraining khônggiánđoạn; controllerPID1291861ACTIVE
  reattachedexistingwindow001, khôngpushlại. Xem side `STATUS.md`/SIDE_GOAL gates.

### Cập nhật14:18 — SIDE_GOAL được user cho phép và controller phục hồi (launch snapshot)

- User yêu cầu ghi toàn bộ thảo luậnSM13/epoch/backbone vào **`SIDE_GOAL.md`** và
  thực hiện bounded song song. Decision `SM13-BACKBONE-SIDE-20261114-v1`;
  GOAL chính/fullpapertraining không bị thay. AGENTS/GOAL có ghi ngoại lệ mới này.
- Code mới `research/sm13_backbone_side/`: generic13layerpooling/heads, cache
  provenance, leakageguards, read-onlyinventory, frozenWavLMloader/nativeCPUprofile.
  Real audit1800WAVhashes/26legacyfeaturefiles/59562targets/61362gaps; phát hiện18
  predictedcenterarraysnonmonotonic, giữ nguyên/khôngâmthầmsort.209CPUtests đạt.
- **SideprivateCPUkernel RUNNING**, khôngGPU/backward/headfit:
  https://www.kaggle.com/code/minkhoaa/langfens-sm13-backbone-cpu-feature-smoke
 8TRAINIDs×native/adapted315Mfrozen, provisionalparent482,1800sdriver/3600splatform.
  Khôngfinal/gainclaim. Package nhỏ khôngWAV/weights/test/credentials, reuseassets1.
- SideCLIretrieval **PID1310766**, reports `reports/SM13_BACKBONE_SIDE_20261114/`,
  `native_smoke_v1/RETRIEVAL_STATUS.json`; xem `STATUS.md` cho đầy đủ evidence/gates.
  FullS1/S2cache/headfit/developmentmetrics vẫnpending, khônglấy smoke làm kết quả.
- **Mainremotetrain luônRUNNING**, nhưng localPID1141950 đã dừng13:22do APIstatus403.
  Đã preserveerror và reattach exactexistingwindow001 **khôngresubmit/restart**.
  Maincontroller mới **PID1291861**, `OPTIMIZED_CAMPAIGN_02/CONTROLLER_STATUS.json`,
  code `tools/kaggle_crottc_optimized_campaign.py --attach-window 1`; nativev8/2T4/
  target300epochs không đổi. `window001/ATTACH_RECEIPT.json` vàpreviousstatus preserved.
  Read-onlytransportstatus retries tối đa3(lưu lỗi), khôngretryremote/trainingERROR.
- Không mainGPUcompetition/localproductiontrain/cron/timer/subagent/commit/push.

### Cập nhật13:05 — optimized long training đã có updates thật (snapshot cũ)

- Livelogwindow001 đã cập nhật483–485/internalepoch8cursor928 (epoch9/300),
  loss3.6988,step4.18s,SSLweightdelta9.82e-6. Evidence:
  `OPTIMIZED_CAMPAIGN_02/window001/LIVE_START.log`. Verifiedexportparentvẫn482.
- Không chỉ platformRUNNING/initializing: traininglong optimized đã thực sự bắtđầu.

### Cập nhật13:02 — optimized long training RUNNING (launch snapshot)

- **Link monitor actual longtraining:**
  https://www.kaggle.com/code/minkhoaa/langfens-crottc-if-optimized-window001
  Privatev1platformRUNNING/initializing; serverprivacy/script/registrationverified.
- Optimizedsmoke480→482 đã download/SHA/load/fullstate/ordered64TRAIN/Adamledger
  verified, source482SHA `7db07c849fd1ca0c0a2f14720212c9f7954e0b197d662d8c7265b05a3e753582`.
  `OPTIMIZED_CAMPAIGN_02/resume_smoke/VERIFICATION.json` và `EVIDENCE/`.
- Longkernel uses2T4FP16/FP32loss-master-Adam,physical32/effective32,
  expandableallocator;14400driverseconds/16800platformseconds,**no updatecap**,
  target300totalAMepochs, not300additional. Startinternalepoch8cursor832.
- ControllerPID1141950ACTIVE, state `OPTIMIZED_CAMPAIGN_02/CONTROLLER_STATUS.json`;
  immutableparent/hash gates and fullcheckpoint export/download/verifyeachwindow.
  Launchreceipt `OPTIMIZED_CAMPAIGN_02/LAUNCH_RECEIPT.json`. Latestverified482,
  actualupdatesinsideactivewindow must distinguish livelog from exportedproof.
- Usermandatefullpaperoptimized notbenchmark-only: `OPTIMIZED_FULL_TRAINING_MANDATE.md`.
  AMCTC/IF/ablations/generalization/challenge stillpending nativegates; fullpaper
  notreproducedyet. TestSEALED, localCLIcontrolonly, nocron/timer.

### Cập nhật12:50 (snapshot trước longtraininglaunch)

- User tái xác nhận phải trainfullpaper bằng cấu hìnhoptimized, không dừng ở
  benchmark/smoke; gửi linklongtraining đangchạy để monitor.
  Đọc `reports/CROTTC_IF_KAGGLE_20261114/OPTIMIZED_FULL_TRAINING_MANDATE.md`.
- Latestlegacy480/epoch8/cursor768 đã download/SHA/load/fullstate/orderverified,
  SHA `382bc2e0ee69caaf54ac0ca8d52ed5a1713166615840e9bbd0c17dc0a23f6179`.
- ConfirmationCOMPLETE/downloadverified chọn2T4FP16mb32/effective32,
  expandableallocator, median5.03s/update(~7.74xbenchmark),1.5GiBreservekhôngđổi.
- Optimizedsmoke `minkhoaa/langfens-crottc-if-optimized-resume-smoke/1` đã
  realupdates481–482, platformCOMPLETE; đangdownloadverifyexport.
- ControllerPID1141950 `OPTIMIZED_CAMPAIGN_02/` sẽ tự nối longkernel
  `minkhoaa/langfens-crottc-if-optimized-window001` **sau exportgate**.
  Tại12:50longkernelchưalaunch: đừng gọi URLdựkiến làđangtrain.
- Longwindows4hdriver/16800splatform,target300totalAM,nocapupdates;
  fullstatecheckpoint50updates hoặc300s + eachepoch/window. Mỗi export tải/
  verifytrước cửa sổ sau; khôngrollback/khôngxóaartifacts. Các stagepaperkhác
  vẫn pending gates/adapters; không gọi AM300 là fullpapercomplete. TestSEALED.

### Cập nhật12:24 (snapshot lịch sử)

- Window002 **COMPLETE**, remote receipt480/internalepoch8/cursor768/no pending,
  tức8completed epochs và đangepoch9; fulloutput đangdownload, chưacertify480local.
- Allocatorconfirmation **RUNNING**. OptimizedcontrollerPID1141950 vẫn waitprofile
  + latestparent, **chưa launch optimizedproduction**. Native unit/regression163passed.
- Xem `RESOURCE_HANDOVER_STATUS.md` và `OPTIMIZED_CAMPAIGN_02/CONTROLLER_STATUS.json`
  dưới reportKaggle. Không restart legacy/khôngrollback125/không duplicatecontroller.

### Cập nhật12:14 — tối ưu tận dụng2T4 (snapshot trước12:24)

- **Đọc `reports/CROTTC_IF_KAGGLE_20261114/RESOURCE_HANDOVER_STATUS.md` trước**.
  User đã yêu cầu tận dụng tối đa GPU/CPU/RAM; không fake-fill hay intentionalOOM.
- Window002/campaign02 vẫn RUNNING, snapshot11:56update450/internalepoch7cursor1632.
  `AM_CAMPAIGN_02/STOP` ngăn legacywindow tiếp theo, **không kill current**.
  PID863482 sẽ tải/verify latestexport; tuyệt đối không rollback125.
- Primary native scratch profileCOMPLETE/downloaded/verified: safeFP16microbatch4
  GPUAdam/twoGPU7.36s/update vs38.65slegacy, ~5.25x; largerbatch ran but rejected
  by1.5GiBreservedheadroom. Actualtensorallocation lower than allocatorcache.
- Allocator confirmation kernel `minkhoaa/langfens-crottc-if-resource-allocator-confirmation/1`
  code-v8 **QUEUED**, tests expandable_segments:True with unchanged safetymargin.
  Do not assume selectedbatch32 or production speedup before evidence.
- **Optimized waiting controllerPID1141950**, reports `OPTIMIZED_CAMPAIGN_02/`:
  waits confirmation→latestwindow002verified→2update native resume smoke→download/
  fullstate/orderaudit→4hoptimizedwindows toward300. Code-v8READY/private;
  full model/objective/target32 unchanged, numerical precision/batch/RNG changes explicit.
- Per-parameter Adam counters and pendingNone gradients from original LayerDrop are
  preserved with537orderednames/inactiveIDs ledger, not forced toglobalupdate.
  Atomiccheckpoint every50updates or300s plus each epoch/window, no silent skip.
- Superseded CPU-only waitingPID1094443 stopped before native launch, preserved at
  `OPTIMIZED_CAMPAIGN_01/STOP` + `PRE_POLICY_REVIEW_STATUS.json`. Do not restart it.
- All5historicaldrivers unchanged; no localproductiontraining/cron/testopening.

### Cập nhật thực thi08:26 (2026-11-14; snapshot lịch sử)

- Window001/campaign01 đã ERROR lúc02:51 trước batch mới do audit so `{}`
  với `{'ssl':{},'heads':{}}`; không có tiến độ qua125 trong lần thất bại đó.
- Đã sửa và đăng ký adapter-only migration125→code-v5, giữ nguyên năm historical
  drivers/recipe/state.115tests passed. Evidence: `WINDOW001_FAILURE_REPAIR_V5.md`
  và `ADAPTER_REPAIR_V5_REGISTRATION.json` trong report Kaggle.
- **Window002 RUNNING, đã train thật**: sau chờ300s theo yêu cầu,08:26 log live
  đến update141/internal epoch2/cursor864 (epoch3/300), loss4.5744, weight delta
  5.72e-6. Remote checkpoint140 được log báo đã lưu; chưa download/verify,
  nên durable local verified parent vẫn125. Không coi live update là exported proof.
- Controller **PID863482**, live state/log tại
  `reports/CROTTC_IF_KAGGLE_20261114/AM_CAMPAIGN_02/`; code-v5, first window2,
  max63windows,4hdriver/16800splatform, target300totalAM epochs. Output base:
  `/var/tmp/langfens-crottc-full-20261113-t_car8m4/kaggle_am_campaign_02/`.
- Đã bật tee driver output sang live Kaggle logs; controller tải diagnostics nếu
  remote lỗi. Không tự retry lỗi; từng export phải download/verify trước cửa sổ sau.
- **Không tạo cron/systemd timer**: user hủy yêu cầu đó, chỉ yêu cầu sleep300s
  rồi check một lần (đã thực hiện). CLI controller training vẫn hoạt động bình thường.
- Bằng chứng live: `AM_CAMPAIGN_02/window002/LIVE_AFTER_300S.log`.
  Local không train production, test SEALED; nhiệm vụ vẫn tái lập toàn bộ paper.

### Nhiệm vụ tái lập full paper được tái xác nhận (2026-11-14; snapshot02:45, đã được cập nhật ở trên)

- Người dùng yêu cầu tiếp tục full training300epoch và hoàn thành **toàn bộ paper
  sau training**, không dừng tại AM300. Phạm vi/gates/control:
  `reports/CROTTC_IF_KAGGLE_20261114/FULL_PAPER_EXECUTION.md`.
- Native profile **kernel v3 COMPLETE, DOWNLOADED/VERIFIED**:32TRAIN samples,
  1scratch update,49.41driver seconds, peak allocated2.51GiB. Receipt:
  `NATIVE_PROFILE_V3_VERIFICATION.json`.
- Resume smoke `minkhoaa/langfens-crottc-if-resume124-smoke` v1 **COMPLETE,
  DOWNLOADED_FULL_STATE_VERIFIED**:125/epoch2/cursor352, đúng25microbatches8–32,
  đủ537Adam states/scheduler/RNG. SHA checkpoint mới:
  `66fcf24144d74b575cf173def0e5f9f69a19d9117a70f4416fb9d90da30d7194`.
  Receipt: `reports/CROTTC_IF_KAGGLE_20261114/RESUME124_SMOKE_V1_VERIFICATION.json`.
- **Sustained kernel `minkhoaa/langfens-crottc-if-am-window001` v1 RUNNING**,
  source `minkhoaa/langfens-crottc-if-resume124-smoke/1`, code-v4. Target300total
  AM epochs, không cap updates;14400driver seconds/16800platform seconds.
- `tools/kaggle_crottc_campaign.py`: CPU-only CLI controller **PID511439 ACTIVE**,
  tối đa64bounded windows;98CPU regression tests passed. Live state/log:
  `reports/CROTTC_IF_KAGGLE_20261114/AM_CAMPAIGN_01/CONTROLLER_STATUS.json`
  và `CONTROLLER.log`; không launch duplicate. Mỗi output phải
  download/SHA/load/order/full-state audit trước cửa sổ sau. Không xóa artifact;
  dừng diskfloor14GiB/quota/error/STOP. Tạo file `AM_CAMPAIGN_01/STOP` để ngăn
  window tiếp theo (không kill window đang chạy). Giữ laptop awake/networked
  để controller retrieve/verify/resubmit; laptop không chạy model training.
- Verified parent125 nằm ở `/var/tmp/langfens-crottc-full-20261113-t_car8m4/kaggle_resume124_smoke_v1/`.
  Các output tương lai ở cùng base `kaggle_am_campaign_01/windowNNN/`.
  Tiến độ epoch/update trong active window chưa có export để độc lập chứng nhận.
- Giữ schedule300/300/200/200 và parent/native gates; AM300 không thay IF,
  baselines/ablations/generalization/challenge. Exact test vẫn SEALED, không
  gọi fullpaper complete khi coverage còn `NOT_REPRODUCED`.

### Khôi phục sau lỗi provider (2026-11-14; lịch sử trước mốc trên)

- Session trước dừng do provider trả `Bad Request`, không phải lỗi checkpoint.
  Transcript được đọc tham chiếu, không chỉnh sửa hoặc resume session cũ.
- Native profile v1 lỗi HyperPyYAML/ruamel.yaml0.19.1 (`Loader.max_depth`).
  Bootstrap đã pin YAML theo local và probe parser; regression31passed.
- Retry v2 qua parser nhưng chặn đúng tại hash code do chọn nhầm local package
  v1 khi mount code-v3; không optimizer update. Đã lưu cả hai thất bại, không nới gate.
- Native scratch profile **kernel v3 RUNNING** ở lần kiểm gần nhất, dùng đúng
  frozen `kaggle_code_v3` (hash inventory đã đối chiếu server), không mount state124.
- Private code dataset **v4** hiện khớp source workspace (adapter resume-parent đã
  sửa trong session trước nhưng chưa upload). Profile/configure/migration gate
  giống hệt v3; năm historical driver hashes không đổi.
- Smoke một update chuẩn bị tại `tools/kaggle_crottc_resume124_v2/`, code-v4,
  state-v1,1800s; **chưa push**, phải download/verify native profile trước.
  Không dùng `resume124_v1` (prepared only với package cũ).
- Chi tiết, evidence và đường dẫn: `reports/CROTTC_IF_KAGGLE_20261114/SESSION_RECOVERY.md`.
  Local vẫn dừng tại124/7pending; test SEALED, chưa chứng minh campaign update Kaggle.

- Đã đọc AGENTS/GOAL/HANDOFF và verify lại SHA/load của checkpoint update124,
  537 Adam states step124, scheduler124 và 7/32 pending; không resume local.
- Probe GPU kernel `minkhoaa/langfens-crottc-if-kaggle-resource-probe` v1 đã
  COMPLETE:2×T4, RAM~31GiB, `/kaggle/working`20GB. V2 CPU đo `/tmp` riêng
  (ephemeral overlay, không tính output20GB). GPU quota còn lại chưa được CLI
  cung cấp; allocation thành công không chứng minh đủ giờ cho full stages.
- `minkhoaa/langfens-crottc-if-environment-check` v1 CPU COMPLETE: uv dựng
  Python3.11.17/Torch2.6.0+cu124 cùng primary packages local; imports thành công.
  Không dùng mặc định Python3.13/Torch2.11/transformers5 của image mới.
- Package local private ở `/var/tmp/langfens-crottc-full-20261113-t_car8m4/kaggle_packages_v1/`.
  Chỉ2399WAV + frozen authorized supervision/source manifests, pinned upstream,
  native weights, full update124/best/receipts. Không raw archive/test/credentials.
- Upload datasets `minkhoaa/langfens-crottc-if-authorized-assets-v1`,
  `minkhoaa/langfens-crottc-if-update124-v1`, `minkhoaa/langfens-crottc-if-runtime-v1`.
  Phải verify server READY/isPrivate/version trước native launch. Runtime code v2
  bổ sung strict recipe/implementation gate và actual scheduler/scaler resume audit.
- Kaggle tự expand tar thành thư mục `assets/` và `code/`; bootstrap phải verify
  từng file theo SHA inventory sau mount (không giả định tar còn nguyên).
- Adapter mới `research/crottc_if_official/kaggle_runtime.py` bind path/input hooks
  cho driver cũ **không sửa năm file hash historical**, giữ FP32/CPU offload và
  microbatch1×32. Chỉ GPU0 visible để giữ inventory CUDA RNG; không tự DDP haiGPU.
- Các bounded windows/gates tại `reports/CROTTC_IF_KAGGLE_20261114/WINDOW_PLAN.json`.
  Native profile và resume campaign chưa được chứng minh tại mốc tiếp nhận này.
  Cần download/verify profile rồi smoke update125 đủ state trước sustained window.
- Tests adapter+existing resume/offload/migration:30passed (`ADAPTER_TESTS_V2.log`).
  Test không thay native resource/continuation evidence. Original checkpoint124
  luôn bất biến, official test vẫn SEALED; không claim full reproduction.

Phần dưới giữ chi tiết bàn giao ban đầu; câu “chưa tạo/upload” ở các mục lịch sử
phản ánh trạng thái trước khi session mới tiếp nhận, không phải trạng thái live.

## 1. Quyết định mới nhất và thứ tự ưu tiên

**Quyết định có hiệu lực: `CROTTC-IF-FULL-KAGGLE-20261114-v1`.**
Người dùng yêu cầu: “thôi, tôi nghĩ nên chuyển lên kaggle để chạy full, tôi có
kaggle cli. Đây là quyết định mới, cần ghi lại lịch sử phiên train, quyết điện
thay đổi lên kaggle vào HANDOFF và GOAL để tôi switch qua session mới, yêu cầu
HANDOFF phải chi tiết nhé”.

- Chuyển full training/reproduction sang **Kaggle**. Local giữ vai trò chuẩn bị,
  phát triển, test, audit, phân tích, lưu checkpoint và log lịch sử.
- Quyết định này thay hướng “hoàn toàn local” ngày 2026-11-13 và “ưu tiên GPU,
  thiếu VRAM dùng RAM” ngày 2026-11-14 về nơi chạy tiếp. Những quyết định cũ vẫn
  là lịch sử hợp lệ của các run đã thực hiện, không phải lệnh tự resume local.
- Job train local đã dừng sạch và checkpoint được xác minh; **không tự khởi chạy
  lại local**. Không có Kaggle training/kernel/dataset mới được tạo trong bước
  bàn giao này. Session mới tiếp nhận việc chuẩn bị rồi thực thi trên Kaggle.
- Giữ nguyên phạm vi full model/full schedule/toàn bộ coverage paper. Không
  đổi thành compact pilot hoặc gọi một AM run là “full paper reproduced”.
- Official test vẫn SEALED. Chuyển nền tảng không cấp quyền mở test, đổi split
  hoặc upload toàn bộ archive có test. Không provisioning Colab như fallback.
- `GOAL.md` vẫn là mục tiêu nghiên cứu duy nhất: xây baseline CROTTC-IF rồi phát
  triển một phương pháp cải tiến cho chẩn đoán và phản hồi. Chưa hoàn tất nghiên cứu.

Quyết định, snapshot GOAL/HANDOFF/AGENTS trước thay đổi và bằng chứng bàn giao ở
`reports/CROTTC_IF_KAGGLE_20261114/`. Đọc `DECISION.json`,
`LOCAL_STOP_VERIFICATION.json`, `LOCAL_SESSION_HISTORY.json`,
`KAGGLE_CLI_READINESS.json`. HANDOFF trước chuyển Kaggle được giữ nguyên tại
`reports/CROTTC_IF_KAGGLE_20261114/HANDOFF_BEFORE_KAGGLE.md`; các dòng “ACTIVE local”
hoặc “IF chưa triển khai” trong snapshot đó chỉ phản ánh các mốc lịch sử.

## 2. Trạng thái thật tại điểm dừng local

Đã gửi **SIGTERM**, trainer nhận signal, dừng ở ranh giới microbatch và ghi
atomic checkpoint đủ partial gradient. Không SIGKILL, không bỏ batch lỗi.

- PID train cuối 324148, exec session 69438: đã kết thúc, kiểm `/proc` không còn.
- Shared GPU lock đã kiểm có thể acquire độc quyền:
  `/home/khoa/Projects/Pronunciation-RD-Coordination/locks/gpu.lock`.
- Status cuối: `WINDOW_EXHAUSTED_RESUMABLE`, không phải training failure hoặc
  hoàn tất lịch300 epoch. Lý do dừng thực tế: yêu cầu chuyển Kaggle của người dùng;
  chuỗi status này cũng được trainer dùng cho signal stop.
- `RUNNING_WINDOW.json` và `EXECUTION_CONTROL.json` dưới experiment root đã đổi
  thành `LOCAL_STOPPED_FOR_KAGGLE_SESSION_HANDOFF`, `local_auto_resume_authorized=false`.
  PID/session trong manifest được giữ làm lịch sử, không còn là job live.
- Full native WavLM Large315,453,120 tham số; SSL trainable311,246,528;
  acoustic heads3,822,765. Native CNN frozen; không phải toàn SSL frozen.
- Đã xong **2/300 epoch**. Internal `epoch=2` nghĩa là đang epoch 3.
  `updates=124`, `cursor=320`, pending7/32 microbatch: đã xử lý327 câu của epoch 3,
  nhưng mới320 câu thuộc các optimizer update hoàn tất trong epoch đó.

### Checkpoint chính phải giữ và chuyển

`experiments/crottc_if_full_local_20261113/am_crottc/last.pt`

- SHA-256: `d060833722f1898800d00778c05e39df636c3aecf3584723a3ae0dfe0a71ee5c`.
- Bytes: **5,058,796,894** (~4.71GiB). Lớn hơn checkpoint3.8GB thông thường vì
  chứa thêm khoảng1.26GB gradient RAM của batch chưa xong.
- `epoch=2`, `cursor=320`, `updates=124`.
- Accumulation: `completed=7`, `group_size=32`, `loss_sum=32.19321870803833`.
  Nếu tiếp tục cùng recipe, bắt đầu **microbatch 8**, không lặp lại 7 câu đầu hoặc
  xóa gradient. Không lấy `best.weights.pt` thay full resumable state.
- 537 Adam parameter states đều `step=124`; scheduler `last_epoch=124`.
- Có SSL/head weights, optimizer, scheduler, disabled scaler FP32, RNG
  Torch/CUDA/Python/NumPy, named gradients, cursor, objective, provenance,
  execution contract. Gradient inventory: SSL 467 tensors/1,244,986,112bytes;
  heads 70 tensors/15,291,060bytes.
- Đã hash khớp `CHECKPOINT_RECEIPT.json`, load CPU bằng
  `torch.load(..., weights_only=True, mmap=True)`; metadata/status/receipt khớp.
  Năm hash implementation đang train giữ nguyên.
- Data/source/native và toàn bộ2399 authorized WAV đã verify lại bằng
  `full_train.verify_inputs()` sau dừng, khớp provenance trong checkpoint.

Selected weights chỉ phục vụ model selection:
`experiments/crottc_if_full_local_20261113/am_crottc/best.weights.pt`.
Được chọn ở human epoch 2 (`epoch=1`, `cursor=1800`, `updates=114`),
source F1 `0.46682054241200227`. SHA-256: `ddddc065bdc13c022b5f86f21c7ced2d6afcb32d721b034114ac4c8c266caff0`;
kích thước 1,277,328,598 bytes. Metadata đầy đủ ở `LOCAL_STOP_VERIFICATION.json`.
File này **không có Adam/RNG/partial gradients để resume đầy đủ**.

Checkpoint profile ban đầu vẫn nguyên tại `am_crottc_profile/last.pt`, SHA
`f4c97a353ce392e166c60965b68aa49868cab598d169ff218bd5626ddde87bd7`,
update 1/cursor 32/internal epoch 0. Đã verify lại SHA lúc bàn giao. Không resume
nhầm profile này thay checkpoint production update 124.

## 3. Lịch sử train của phiên này

Các mốc theo Asia/Ho_Chi_Minh; JSON `LOCAL_SESSION_HISTORY.json` chứa giờ bắt đầu,
giờ kết thúc, PID, cursor, pending gradients và SHA của các log.

1. Ban đầu có native profile thật trên 32 câu ngắn nhất, một Adam update,
   WavLM/Adam trên CPU, head/loss CUDA. Profile này chứng minh backward/update,
   chưa đại diện câu dài; không dùng nó làm ETA toàn campaign.
2. Sửa strict resume: `--resume`/`--resume-from` không âm thầm reset khi thiếu
   checkpoint; chỉ nhận legacy migration cho đúng SHA profile đã đăng ký.
   Bổ sung RNG/partial-gradient checkpoint, checkpoint mỗi5update và clean stop.
3. Numerical continuation tests4passed trên model nhỏ CPU/CUDA. Profile đại
   diện 32 câu:16 median neighbors+8 p95 neighbors+8 longest, max9.054s, một scratch
   update; input checkpoint không thay đổi. Peak RSS~9.89GiB ở CPU backend.
4. CPU production tiếp tục qua nhiều cửa sổ clean stop: update 1→17→19→20→22,
   giữ pending gradients khi ngắt để profile GPU. Các log WINDOW001–005 giữ nguyên.
5. Bốn profile full-GPU trước chưa offload gradient sau từng microbatch đều CUDA
   OOM, **zero scratch optimizer updates**: CPU Adam đơn thuần, thêm activation
   offload, sửa frozen-CNN waveform autograd, rồi AMP. Đây là thất bại của các
   cấu hình đó, không phải bằng chứng full model không thể chạy trên local.
6. Profile thứ5 thành công: toàn SSL/head FP32 trên CUDA, Adam/master weights,
   accumulated gradients và saved activations trong CPU RAM. Sau từng microbatch
   copy/add gradient vào RAM và set GPU `p.grad=None`. Peak CUDA allocated
   2,696,370,688bytes (~2.51GiB), RSS8,588,914,688bytes (~8GiB), đủ 32 câu/một update.
7. CPU→GPU migration có receipt riêng: update 22/cursor 704/pending 3 →
   update 23/cursor 736. Sau đó native GPU clean-stop giữ 12/32 gradient, resume từ
   microbatch 13; đã chứng minh optimizer/scheduler/RNG tiếp tục, không train lại.
8. GPU/RAM sustained window chạy khoảng3902 s (~65 phút), từ update 23 đến 124,
   xong epoch 1 và 2, vào epoch 3. Dừng sạch theo quyết định chuyển Kaggle.

### Các attempt production, không tính scratch profile

| PID | Bắt đầu (local) | Updates đầu→cuối | Cursor cuối | Pending cuối | Kết thúc |
|---|---|---:|---:|---:|---|
|246890|2026-11-13T23:12:45.951669+07:00|1→1|32|23/32|WINDOW_EXHAUSTED_RESUMABLE|
|248779|2026-11-13T23:14:21.836931+07:00|1→17|544|28/32|WINDOW_EXHAUSTED_RESUMABLE|
|289349|2026-11-13T23:47:26.786790+07:00|17→19|608|19/32|WINDOW_EXHAUSTED_RESUMABLE|
|295868|2026-11-13T23:53:01.732195+07:00|19→20|640|3/32|WINDOW_EXHAUSTED_RESUMABLE|
|300886|2026-11-13T23:57:18.838845+07:00|20→20|640|17/32|WINDOW_EXHAUSTED_RESUMABLE|
|306911|2026-11-14T00:02:36.639216+07:00|20→22|704|3/32|WINDOW_EXHAUSTED_RESUMABLE|
|320568|2026-11-14T00:14:45.398879+07:00|22→23|736|0/0|UPDATE_WINDOW_COMPLETE_RESUMABLE|
|322697|2026-11-14T00:16:25.338587+07:00|23→23|736|12/32|WINDOW_EXHAUSTED_RESUMABLE|
|324148|2026-11-14T00:17:40.938555+07:00|23→124|320|7/32|WINDOW_EXHAUSTED_RESUMABLE|

### Validation đã đo

Mỗi epoch decode audio-only đủ599 DEV_SEARCH; gold chỉ được dùng tính metric
sau decode. Log AM cũ chỉ lưu aggregate, không lưu hypothesis từng câu để rescore.

| Epoch hoàn tất | Updates | Source new hierarchical F1 | Source old F1 | PER (%) |
|---|---:|---:|---:|---:|
|1|57|40.0283%|23.0536%|65.9569|
|2|114|46.6821%|26.1382%|52.0276|

Đây là **local development** với annotation policy/split khác paper; không phải
strict typed S/O/I, không chứng minh acoustic diagnosis/feedback readiness.
Old/new source metric có định nghĩa khác nhau, giữ cả hai; không chọn định nghĩa
đẹp hơn để claim gain. Paper F1=71.77% vẫn là kết quả tác giả, chưa được tái lập.
Bằng chứng: `AM_EPOCH_001_VERIFICATION.json`, `AM_EPOCH_002_VERIFICATION.json`
trong `reports/CROTTC_IF_FULL_LOCAL_20261113/`.

### Vì sao đổi nơi chạy

GTX 1650 4GB + RAM31 GiB/swap35 GiB thực sự chạy full model được, nhưng phải offload
nhiều. Snapshot khoảng 1 giờ cho~38.7 s/update kể cả validation/checkpoint;
AM CROTTC 300 epoch riêng ước~7.7 ngày nếu giữ tốc độ. Đây là ước lượng từ cửa sổ
ngắn, không bảo đảm hiệu năng nhiều ngày. Chưa đo tốc độ native AM CTC/IF nên
không gán cùng ETA; toàn campaign có thể nhiều tuần. Chi tiết
`reports/CROTTC_IF_FULL_LOCAL_20261113/GPU_RAM_COMPUTE_PROJECTION.json`.
Người dùng quyết định chuyển Kaggle, không phải trainer chứng minh “local bất khả thi”.

## 4. Dữ liệu, source và môi trường có thể tái sử dụng

Experiment root cũ: `experiments/crottc_if_full_local_20261113/`.
Không đổi tên/xóa để giả vờ toàn bộ run đã chạy trên Kaggle.

- Source `Secondtonumb/IF-MDD`, commit
  `cca150ec4ff00e01d61a613092d1d178e661dfb2`, bản pin ở `upstream/`.
  Không có `.git` trong bản sao; `git -C upstream rev-parse HEAD` có thể trả
  HEAD repo cha. Dùng `SOURCE_RECEIPT.json` và hash từng file.
- Native `microsoft/wavlm-large`, revision
  `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c`, đã tải tại `native_wavlm_large/`.
  `pytorch_model.bin` SHA
  `fdee460e529396ddb2f8c8e8ce0ad74cfb747b726bc6f612e666c7c1e1963c9d`.
  Không dùng released L2-finetuned checkpoint làm native initialization.
- TRAIN 1800/12 speakers: ABA, ASI, BWC, EBVS, HJK, HKK, HQTV, LXC, NJS, PNV,
  SKA, SVBI. DEV_SEARCH 599/4: NCC, RRBI, YBAA, YDCK. Không fitting DEV_CAL.
- Prepared `train.jsonl` SHA
  `8db52b5dcc86ca0c0f6854a2ffc920ec44e5bf506f8414fd8006dcb9fa4cd3cc`;
  `dev_search.jsonl` SHA
  `41c26dc0afa58a515d543a1043bc67e5525fe576c9a9d52e041c71c6a25773bc`.
- Source manifests và archive hashes/counts/label policy đầy đủ trong
  `DATA_RECEIPT.json`; source manifest paths
  `data/derived/l2arctic/manifests/train.jsonl` và `dev_search.jsonl`.
- Authorized WAV ở `data/derived/l2arctic/wavs_16k/`; package Kaggle phải lấy
  **allowlist IDs từ đúng hai prepared manifests**, không copy cả thư mục vì có
  thể chứa population khác. Mỗi row có `audio_sha256`; train annotation provenance
  có `source_member`, `archive_sha256`, `annotation_sha256`.
- Raw archive local phục vụ lịch sử chuẩn bị:
  `experiments/sm13_three_directions_20261113/data_expansion/raw_archives/`.
  Không upload full archive hoặc original Drive zip. Chỉ stage train/validation
  được phép; không mở/extract/stage official test audio hoặc labels.
- Local sealed speakers: ZHAA, TNI, MBMPS, TLV. Paper test còn có NJS/TXHC/YKWK;
  NJS đang ở local TRAIN, nên không thể gọi current checkpoint là paper-split clean.
- `.venv-crottc-official/bin/python`: Python3.11, torch2.6.0+cu124,
  torchaudio2.6, SpeechBrain1.0.3, transformers4.57.0, numpy2.4.6, numba0.68.1;
  tokenizers0.22.2, huggingface-hub0.36.0, wandb0.21.4.
  Base `.venv` dùng transformers5 không phù hợp source này. Freeze tham khảo
  `experiments/crottc_if_official_20261113/RUNTIME_FREEZE.txt`; effective versions
  trong checkpoint execution contract mới là bằng chứng của run local.
  Không copy nguyên venv local sang Kaggle; dựng môi trường và ghi versions thực tế.

## 5. Code đã có và phần chưa được chứng minh

| Thành phần | Trạng thái tại bàn giao |
|---|---|
|`full_train.py`, `checkpointing.py`|CPU backend, strict resume, full state/partial gradient, atomic save; đã thực thi native training.|
|`cpu_adam_offload.py`, `gpu_train.py`|GPU/RAM native AM backend; đã chạy production đến update 124.|
|`placement_migration.py`|Migration CPU→GPU đã dùng đúng một lần; receipt cũ không dùng cho Kaggle.|
|`if_components.py`|Full source IF decoder/teacher15,965,361 params, 384 dim/2 layers, C/S/D/I và audio-only beam decode ; 7 component tests.|
|`stage_contract.py`|Strict completed-parent gates,300/300/200/200, hash/validation/selected weights checks ; 5 tests.|
|`if_train.py`|Đã triển khai IF CTC initialization và CROTTC transfer, GPU/RAM, partial resume, atomic checkpoint, audio-only validation ; 13 trainer+gate tests. **Chưa native IF resource profile/training.**|
|`validation_report.py`|Offline DEV scoring/report helper và tests; không được import vào AM driver đang chạy để thay evaluator giữa campaign.|
|`local_storage.py`|11 storage tests, root private đã provision và verify; **chưa production relocation**.|
|Kaggle adapter/package/notebook|**Chưa tạo, chưa upload, chưa launch**; session mới phải triển khai và kiểm chứng.|
|Automatic next-local-window controller|Mới có ý định; chưa tạo, không có background auto-resume controller.|

Năm code hash đã pin trong local checkpoint:

| File | SHA-256 |
|---|---|
|`full_train.py`|`dacbd50baeec9910700653a1fdc726cd66b1af5338da89a85abf859760f88450`|
|`checkpointing.py`|`fa886cb24816fb03520788249f20accae967874b832363e50ec55447ca7076e8`|
|`gpu_train.py`|`bf18c682e3811e1c5c4a63799dcc3d448cbc3fe5a8d97af9b393c3252f52bd50`|
|`cpu_adam_offload.py`|`4f43514c8ab026f27514c9bf912db19a53fe818281367be2f7ab6876f130ead1`|
|`placement_migration.py`|`176c8f339af7507457ba5cf76c31ef0699a0facdb95169b88fe29dff53ea6d57`|

Không sửa hash trong checkpoint hoặc receipt cũ cho khớp code mới. Kaggle adapter,
đổi package/device placement/path resolution cần migration receipt mới, hash input
checkpoint/config/runtime cũ và mới, cùng kiểm tra giữ nguyên weights/Adam/cursor/RNG/
partial gradients. GPU khác hoặc runtime khác không được claim bitwise-equivalent
chỉ vì load thành công. Không âm thầm bỏ 7 pending gradients để chạy batch 32 mới.

## 6. Lịch full stages và fidelity còn thiếu

| Stage | Target đăng ký | Dependency | Đã đạt |
|---|---:|---|---|
|AM CROTTC|300 epoch|Native WavLM Large|2epoch +320 committed/7pending TRAIN rows của epoch 3;124updates|
|AM CTC|300 epoch|Native WavLM Large|Chưa train|
|IF CTC initialization|200 epoch provisional|AM CTC 300 hoàn tất|Trainer đã triển khai; chưa profile/train|
|IF CROTTC transfer|200 epoch provisional|AM CROTTC 300 + IF CTC 200 hoàn tất|Trainer đã triển khai; chưa profile/train|

- `stage_contract.py` không nhận live/profile checkpoint làm completed parent.
  IF CTC: train SSL+AM heads+IF (native CNN frozen); IF CROTTC: thay toàn AM bằng
  selected CROTTC AM, chỉ chuyển decoder/teacher từ IF CTC, freeze/eval toàn AM,
  fresh IF optimizer. Không bypass gates bằng AM2epoch.
- TRAIN giữ raw silence; starred→`err`; AX→AH; malformed er)/d_/z_/AH)→`err`;
  bỏ deletion khỏi target.557 TRAIN rows có1110 `err` tokens. Không suy diễn lại
  exact realized phone từ canonical. DEV derived thiếu raw silence/star,12 `err`.
- Exact paper split2429/268/900 khác local1800/599. Author preprocessing script/
  frozen manifests chưa có trong pin; preprocessing parity UNVERIFIED.
- AM dùng full WavLM + Conformer 384/2 layer, kernel override31→3 theo paper;
  paper Eq5/6 detached symmetric KL. Waveform DropFreq/DropChunk+speed và native
  SSL SpecAugment chưa khớp paper timewarp 80/ratio masks. `objective=paper` không
  đồng nghĩa toàn recipe fully faithful. Không đổi augmentation/objective giữa
  resumed campaign mà không tách/đăng ký thay đổi.
- Microbatch1 accumulation32 không có cùng BatchNorm statistics như real batch 32.
- Source IF có alias fusion branches, thiếu acoustic Conformer registration,
  binary teacher labels dù head4classes, ConPCO ngoài paper. Local components
  đã tách branches, nối full AM memory, C/S/D/I, bỏ ConPCO theo registered choice.
- Source LM decoder thực tế regular MHA + positional embedding, teacher RelPosMHA;
  không gọi fully RoPE chỉ vì tên YAML. Paper omega1 .3 ở Eq18 vs .5 implementation;
  local registered .5, omega2=1, omega3=10.200 epochs mỗi IF stage là provisional vì
  paper không chia rõ allocation; chưa gọi là đã giải quyết ý định tác giả.
- `PAPER_COVERAGE.csv` trong report local theo dõi main/baselines/ablations,
  LLM variants, SO762, ERJ, Iqra, figures. Các phần chưa có bằng chứng vẫn
  `NOT_REPRODUCED`; chuyển Kaggle không tự hoàn thành chúng.

## 7. Kaggle CLI đã kiểm tra và việc session mới cần làm

CLI có sẵn: `/home/khoa/.local/bin/kaggle`, **2.2.5**, user **minkhoaa**, OAuth.
`kaggle kernels list --mine --page-size 1 --csv` chạy thành công (exit0), đã xác
thực read-only, không chỉ kiểm binary tồn tại. Không in/copy/upload credential.
Chưa xác minh accelerator được cấp, quota còn lại, notebook limits, internet và
storage thực tế của tài khoản. Không hứa GPU cụ thể hoặc300 epoch trong một session.

**Thứ tự triển khai tiếp theo:**

1. Đọc AGENTS/GOAL/HANDOFF và decision mới, xác minh local checkpoint SHA124,
   đủ 7 gradients, local job đã terminal. Không tự resume local.
2. Khảo sát Kaggle runtime/accelerator/quota thực tế bằng CLI/UI phù hợp; chọn
   budget từng window trước optimizer update, chừa thời gian checkpoint và
   xuất output trước giới hạn nền tảng. Giữ target300/300/200/200.
3. Tạo package private tối thiểu từ allowlist 2399 WAV + authorized supervision,
   manifest/hash/speaker allowlist/denylist. Stage riêng source pin, code/config,
   native weights và full checkpoint/selected weights/receipts/logs cần thiết.
   Không upload nguyên repo, full data tree, raw zip hoặc credentials. Kiểm SHA
   cả trước upload và sau mount. Tạo một lần/version, tái sử dụng qua các window.
4. Triển khai adapter cho `/kaggle/input` và writable work/output. Current driver
   **chưa portable nguyên trạng**: prepared rows có WAV path tuyệt đối `/home/khoa/...`,
   source receipt chứa path tương đối repo, native/source roots được suy ra từ
   `ROOT/OUT`, GPU flock dùng absolute laptop path, threads/preflight/placement
   theo local. Nếu rewrite manifest, ghi mapping old/new hashes và chứng minh
   IDs/audio/labels/order không đổi; không sửa receipt cũ rồi gọi same provenance.
5. Chọn resume-compatible path/placement trước. Nếu tận dụng VRAM lớn hơn để đổi
   Adam placement, precision, microbatch/BatchNorm hoặc augmentation, đăng ký rõ
   migration/recipe mới và giữ historical local run riêng; không tự reset native,
   giảm epoch hoặc che mất124updates. Preserve partial group trước mọi thay đổi
   không tương thích. Hạn chế claim numerical equivalence giữa hardware/runtime.
6. Profile scratch native full model đủ 32 median/p95/max, real forward/backward/
   Adam, đo VRAM/RAM/disk/time. Profile không được tự promote thành production.
   Kiểm load/resume ở Kaggle với full optimizer/scheduler/RNG/partial gradients,
   tiếp tục microbatch 8, save/download rồi verify hash/load local. Không gọi chỉ
   load weight hay inference smoke là full training validation.
7. Khởi chạy AM CROTTC continuation bằng Kaggle CLI khi adapter/profile/provenance
   đã xác minh; lưu kernel slug/version, dataset versions, accelerator/runtime,
   log, checkpoints và status. CLI auth đã có; không hỏi lại user quyền chuyển
   Kaggle vốn đã được quyết định rõ ở phiên này.
8. Mỗi window phải có checkpoint atomic, nonfinite/data/resource stop, log thất
   bại, full resume state; lấy output về và verify trước restart. Không giả định
   filesystem của một runtime còn nguyên khi runtime kết thúc. Không có output
   verified thì không claim checkpoint đã persist.
9. Tiếp tục AM CTC/IF theo dependency gates, validate DEV-only/audio-only, rồi
   thực hiện coverage paper còn lại trong quyền dữ liệu. Tách paper MDD metric,
   strict S/O/I và feedback thesis. Chỉ chuyển cải tiến chính sau baseline đủ chứng cứ.

Tài liệu CLI chính thức đã đối chiếu ngày 2026-11-14:
[Kernel commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md),
[kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md),
[dataset commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md).
CLI `push` sẽ chạy kernel trừ khi dùng `--no-run`; dataset create mặc định private.
Luôn xác minh metadata private và nguồn input trước khi push. Chưa có slug để đưa
lệnh launch thật ở đây; không copy slug của experiment Fold1 cũ vào population mới.

Các lệnh read-only có thể dùng đầu session mới:

```sh
kaggle --version
kaggle kernels list --mine --page-size 1 --csv
kaggle kernels push --help
kaggle kernels output --help
```

## 8. Storage local, log, tests và các lưu ý bàn giao

- Checkpoint production vẫn ở HOME, không symlink/copy sang vùng mới.
  Root riêng `/var/tmp/langfens-crottc-full-20261113-t_car8m4` còn khoảng62 GiB
  lúc provision; registry `LOCAL_STORAGE_REGISTRY.json`. Chỉ có owner marker,
  retention lock và16 MiB I/O probe; **không chứa production checkpoint**.
- Guard PID389676/session34894 đã dừng, status `GUARD_STOPPED_FILES_PINNED`;
  sticky regular files được giữ. Không cần resume guard để chuyển Kaggle. Code
  relocation/local-next-window trong report cũ là hướng cũ, không phải queue mới.
- Không xóa checkpoint/profile cũ để lấy chỗ. Home chỉ~10 GiB free sau partial
  checkpoint; đo lại trước tạo package lớn. Drive archives giữ nguyên.
- Tests đã có log: resume4passed; IF components7passed; IF trainer+stage13passed;
  optimizer/migration/stage16passed ở nhóm kiểm trước; storage11passed.
  Các nhóm có overlap, **không cộng thành một tổng độc lập**. Scope IF tests là
  components/synthetic AM memory/transfer/freeze/gates, không native IF training.
- Tests/reports thường ở `reports/CROTTC_IF_FULL_LOCAL_20261113/` và
  `experiments/crottc_if_full_local_20261113/*TESTS.log`. Không rerun toàn bộ
  training chỉ để xác minh docs. Native Kaggle profile/continuation vẫn chưa có.
- Official released bundle smoke cũ:3 FIT audio-only,
  `reports/CROTTC_IF_OFFICIAL_20261113/REPORT_VI.md`, release revision
  `9149e5107b3332b9aa3f5573f6328a4e312006ad`; không thay native training evidence.
- Repo có nhiều untracked artifacts từ trước. Không `git add -A`, không commit
  dataset/weights/checkpoints/credentials, không force-push/merge. Chưa commit/push
  bàn giao này. Git HEAD không chứa đủ untracked implementation; package source
  cho Kaggle phải dựa trên file/hash inventory, không chỉ git clone HEAD.
- Không spawn agents nếu user/AGENTS/skill không yêu cầu. Không resume M0–M6/
  compact pilot/parallel method sweeps vì GOAL thesis hiện hành đã supersede chúng.

## 9. Prompt mở session mới

> Đọc AGENTS.md, GOAL.md và HANDOFF.md. Resume mục tiêu full CROTTC-IF theo quyết
> định CROTTC-IF-FULL-KAGGLE-20261114-v1: chuyển training sang Kaggle bằng CLI đã
> xác thực. Local đã dừng sạch ở update 124, epoch 2/cursor320 với 7/32 pending
> gradients; giữ checkpoint và lịch sử, không khởi tạo lại hoặc tự train local.
> Chuẩn bị private package chỉ TRAIN 1800/DEV_SEARCH 599, adapter/migration/profile
> Kaggle, xác minh resume rồi chạy các stage đầy đủ theo dependency. Official test
> vẫn SEALED; fidelity và coverage paper còn thiếu phải theo dõi, không claim full
> reproduction chỉ vì chạy được một AM stage.
