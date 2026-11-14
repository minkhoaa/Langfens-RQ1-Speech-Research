# SM13 — Dependency declaration từ kế hoạch đến benchmark hiện tại

**Snapshot:**2026-11-14. Đọc cùng [ONBOARDING](../ONBOARDING.md), [GOAL_SM13](../GOAL_SM13.md), [HANDOFF](../HANDOFF.md).
**Không có training trong task publication này.** Dependency có trongGit không đồng nghĩa privateartifacts đã được phát hành hoặc mộtrunđãđượcủyquyền.

## 1. Dependency graph

```text
GOAL_SM13 + user decision + registered protocol
  ├─ runtime/code source hashes + resource floors + GPU lock
  ├─ frozen TRAIN1800 manifest / authorized WAV hashes
  │    ├─ canonical references + TRAIN supervision → TRAIN_HEAD_INPUTS.npz
  │    └─ predicted alignment centers / frame clock → TRAIN_ALIGNMENT.npz
  ├─ pinned native backbone/config/preprocessor
  │    └─ geometry + pooling + normalization policy
  │         └─ target/gap acoustic cache + FEATURE_RECEIPT + independent verification
  └─ explicit fit/held speakers + seed/recipe
       ├─ optional fit-only per-layer RMS → frozen normalization buffers
       └─ fresh target/gap heads + scalar mixing
            └─ atomic model/Adam/RNG/cursor states + per-epoch keyed probabilities
                 ├─ fixed.5 event-detection P/R/F1, S/O/I AP, per-speaker/FWR
                 ├─ full curves + fixed endpoints + preregistered quality safeguards
                 └─ independent state/hash/metric/CPU-inference verifier
```

**S0, S1raw, S1normalized, dropoutcontrol là các lineage khác nhau.** Khôngloadhead768D lên1024D hoặcápnormalizationmới vàocheckpointrawcũ.

## 2. Python/system dependencies

### Current S1 train/benchmark/audit runtime

[Exact effective package inventory](../environments/research-runtime-20261114.txt) chứa105uniqueversions ởworkstationowner. Critical imports:

| Dependency | Dùng ở đâu | Constraint/ghi chú |
|---|---|---|
| Python3.11.17 |Drivers,verifiers,tests |Linux runtime; không dùngrootlockPython3.14 cũ làm activeenvironment |
| torch2.6.0+cu124 |Head/backbone forward,AdamW,atomicstate/RNG |CurrentCUDArecipe; CPU fixtures cũngimportđược; preserveoptimizer/APIsemantics |
| torchaudio2.6.0+cu124 |HistoricalSSL/SpeechBrain audio stack |KhớpPyTorchbuild; khôngmixCUDA13variant |
| transformers4.57.0 |NativeWavLM/W2V2/config/featureextractor |Localfiles/offlineproductionidentity; noimplicitmodelrevisionupgrade |
| speechbrain1.0.3 |CROTTC lineage/historicalnormalizationcontext |SM13nativeadapter gọiHFWavLM; HF→SB-stylelayernorm được code explicit, khôngtựđổi policy |
| numpy2.4.6 |FP32memmaps,referencearrays,shuffleRNG,keyorder |KhôngcastFP16đểláchdisk; precisionchange cầnparity |
| scipy,soundfile |AudioI/O/resampling và analysis utilities |Mono16k/audiohashcontract; versions ởinventory |
| scikit-learn |IndependentAP/P/R/F1replay |Verifierkhôngchỉđọclại số report |
| matplotlib |Learningcurveplots |Khôngphảimodelmetric hoặctrainingdependencychính |
| psutil |RAM/resourceadmission |Initial/runtimefloors khôngđượchạ |
| pytest |Syntheticcontracts/resume/corruptiontests |Testscopeghi rõ; khôngcoimọi suite/data-dependenttests làPASS |
| fcntl/os/signal/subprocess |SharedGPUlease,safeinterrupt,blockingCPUverifier |Linux/stdlibrary; khôngportableWindowsdriver chưađượcchứngminh |
| NVIDIA driver + CUDA-capabletorch |LocalGTX1650profile/training |ActualVRAM/profilegate; khôngsuyadmissiontừCUDAversionstring |
| Git/SHA256 |Sourceidentity/publication/provenance |Sourcechangesphảiregisterlineagemới, khôngrewritehistoricalreceipts |

KaggleCLI/API chỉ phục vụ remoteCROTTC/historicalSM13jobs. **Không phải dependency cho localSM13qualitytrial hiện tại.** Khôngđưakaggle.json/tokens lênGit.

### Environment reconstruction caveat

Owner `.venv-crottc-official` có `.pth` trỏ `.venv` chính; khôngxóa `.venv` khi cònphụthuộc. Cleanupincidentđãcầnofflinerepair; versioninventory/imports/85scopedCPUtestsđượckiểm, khôngclaimoldenvbitwiseparity. Xem [environment notes](../environments/README.md). Trênmáyteammate nêncài một envđộc lập; khôngcopyowner-specificpth.

Inventorykhôngchứaallwheelhashesvàkhôngbảohànhpublicindexavailability. Original `requirements-lock.txt` làhistoricalsnapshot, khôngcurrentS1lock.

## 3. Planning/protocol dependencies

| Giai đoạn | Dependency trong Git | Ý nghĩa |
|---|---|---|
| Historical formalSM13 |`experiments/rq1_sm13_formal_train_01/PREREGISTRATION.json` (local evidence; not included in docs-only push),[formalreport](../reports/RQ1_SM13_FORMAL_TRAIN_01_BAO_CAO_TONG_HOP_TOAN_DIEN.md) |3seeds101/202/303,target12/gap5,13representationheads,neural+E4fusionlineage; không gọifullTRAINfitloss làheldcurve |
| Originalbackboneepoch50 |`reports/SM13_EPOCH50_20261114/PROTOCOL.json` (local evidence; not included in docs-only push),`reports/SM13_EPOCH50_20261114/EPOCH50_INPUT_INVENTORY.json` (local evidence; not included in docs-only push) |Frozenoriginalcache,fit1350/held450,reusedouter0development |
| S1nativecache |`reports/SM13_NATIVE_LOCAL_20261114/CACHE_REGISTRATION_v1.json` (local evidence; not included in docs-only push),`reports/SM13_NATIVE_LOCAL_20261114/LOCAL_CACHE_VERIFICATION.json` (local evidence; not included in docs-only push) |PinnednativeLarge/model/config/audio/alignment/hashgeometry |
| S1firstheadfit |`reports/SM13_NATIVE_LOCAL_20261114/FIT_CALIBRATION50_REGISTRATION_v1.json` (local evidence; not included in docs-only push),[metric registration](../reports/SM13_NATIVE_LOCAL_20261114/METRIC_AND_CALIBRATION_REGISTRATION.md) |Fresh1024Dheads,50/head,fixedterminal50; finiteTRAINcalibrationgrid riêng vớifixed.5diagnostic |
| Raw50→150→350 |`reports/SM13_S1_PLUS100_20261114/REGISTRATION.json` (local evidence; not included in docs-only push),`reports/SM13_S1_PLUS200_20261114/REGISTRATION.json` (local evidence; not included in docs-only push) |Exactfullstateresume/migration,RNG/Adam/cursor; frozen50thresholds,noautoretune |
| Raw-vs-RMS/normalized300 |[protocol](../reports/SM13_RAW_NORMALIZED_20261114/PROTOCOL.md),`reports/SM13_RAW_NORMALIZED_20261114/REGISTRATION.json` (local evidence; not included in docs-only push) |Freshpaired50,onlyfitRMSdiffers; afterverifiedpilotgate normalized50→300total,noextension |
| Dropoutcontrol |`reports/SM13_DROPOUT_VERIFY_20261114/PROTOCOL.json` (local evidence; not included in docs-only push),`reports/SM13_DROPOUT_PAIRED_20261114/REGISTRATION.json` (local evidence; not included in docs-only push) |Freshbaseline50→blockingreproductionverification→dropout2x50→pairedverifier; noLS/RDrop/FTswitch |

Currentcontrolsareonepairedseed/reusedTRAINdevelopment. Aprotocol/registerfileisNOTprooftrainingran.

## 4. Data/reference/alignment dependencies — PRIVATE / outside Git

| Artifact | Identity / owner-machine location | Required for |
|---|---|---|
| Frozen TRAIN manifest |`data/derived/l2arctic/manifests/train.jsonl`;SHA`1f4da8518d9b184fb737b93c7887ecd7c3e73d6e83260e54651074831f61bcff` |TRAIN1800speaker/eventorder/labels; noofficialtestextraction |
| AllowlistedWAV1800 |PathsandSHAinprivateTRAINinference requests |Featureextraction/profile/audioreplay; noDriveaudio streaming |
| Intendedphones/refvectors/labels |`sm13_epoch50_package_v1_staging/TRAIN_HEAD_INPUTS.npz`;SHA`94a180c4ddd199292e8174919655f8944e1cc48c96c621648128df89f876c7e3` |59562targetrefs39D/y;61362gaprefs78D/y; speakers/UIDs. Goldonlysupervisesfit/scoring, notinferenceinputs |
| Frozen predictedalignment |`experiments/rq1_encoder_bench_01/features/alignment_centers.npz`; also`TRAIN_ALIGNMENT.npz`;SHA`275d707949c4fef785611ef37e2f18906b912974eb7dbd841c06baed85c069c7` |Framecenters/clock,order/pooling.18nonmonotonicutterancesinherited; donotsort/drop |
| Minimalinference requests |`TRAIN_REQUESTS.jsonl` / `TRAIN_INFERENCE_REQUESTS.jsonl`;separatehashesinreceipts |Requestfieldallowlist:audio/intendedphones, notgoldtimestamp/realized/errorlabels |
| Inputinventory |`reports/SM13_EPOCH50_20261114/EPOCH50_INPUT_INVENTORY.json` (local evidence; not included in docs-only push) |Verifiesfiles/sizes/order beforeheads; inventorymetadata isGit, NPZ/audio are not |

Fit1350speakers=`EBVS,HJK,HKK,HQTV,LXC,NJS,PNV,SKA,SVBI`; held450=`ABA,ASI,BWC`. Eventsfit44746target/46096gap. SameTRAINhasbeenreuseddevelopment; notunseenconfirmation.

Referenceiscurrently**suppliedcanonical/annotation-derived**, notproventext-G2Pindependence. Zero-oracletextmodecannotblindlyreusecanonicaleventIDsifthephonesequencechanges.

## 5. Backbone/feature-cache dependencies

### S0

- Model`facebook/wav2vec2-base-960h`,revision`22aad52d435eb6dbaf354bdad9b0da84ce7d6156`.
-13states×768; originalFP32layercaches at `experiments/rq1_encoder_bench_01/features/target_acoustic_w2v2_960h_l*.npy` andgapcounterparts.
-Do notoverwrite/reextract/normalizehistoricalcachesundertheirsameidentity.

### S1nativeLarge

- PinnednativeWavLMrevision`c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c`.
-24Transformer layers/1024D;315453120backboneparams; selectedindices`[0,2,…,24]`.
- RegisterednativeweightsSHA`fdee460e529396ddb2f8c8e8ce0ad74cfb747b726bc6f612e666c7c1e1963c9d`.
- Canonicalnativebundle metadata: `reports/SM13_NATIVE_LOCAL_20261114/CACHE_REGISTRATION_v1.json` (local evidence; not included in docs-only push); realweights remainoutsideGit.
- Policy:`HF_feature_extractor_then_SB_layer_norm_if_do_normalize;batch1;no_output_norm;eval_specaugment_disabled`. This differsfromS0historicalHF-onlypolicy; disclosebeforepurebackbonegainclaims.
- Cache:`/var/tmp/langfens-crottc-full-20261113-t_car8m4/sm13_s1_train_cache_v1/target.npy`,`gap.npy`;≈14.95GiBFP32.
- Geometry:target`(59562,13,3072)`;gap`(61362,13,2048)`. Byte/rowhashes andgeometrycheckedin `reports/SM13_NATIVE_LOCAL_20261114/LOCAL_CACHE_VERIFICATION.json` (local evidence; not included in docs-only push).
- Targetmean/std/enddelta; gapleft/rightmeansaroundpredictedalignment. This isnotrawtime-frequencySpecAugmentinput.

Cacheidentitymustbindmodel/revision/layers/precision/normalization/alignment/manifest/order/source. CPU/GPUstrictfeatureparityhadsparsefailuresinaudit; donotclaimuniversalallcloseorsilentlyloosentolerance.

### S1fit-onlyRMS

`conditioning.py` computes13RMSscalesonlyfromfitfeatures, persistedas`normalizer.rms`buffersinsideNEWheads. Statsfiles underprivate normalizedwork;SHA`ce4350d9dc700737fd2867985eaefb697810d806d0a89215b7cd987796cbf972`. Do notfitonheld/dev/testorreusetheoldrawheadasconverted.

### S2conditional

Needsselected/finaladaptedparentwithverifiedlineage. AbackbonetrainedonallTRAIN1800alreadyknows450TRAINheldspeakers: cannotclaimindependentTRAINOOF. NoautomaticS2cache/fine-tuningcampaignauthorizedbythisdocumentation.

## 6. Code dependency map

Allpathsbelowareversion-controlledsource;heavyinputsareseparate.

| Layer | Files / imports | Responsibilities |
|---|---|---|
| Historicalreferenceimplementation |`experiments/rq1_sm13_formal_train_01/agent_a_sm13_engine.py`,`train_sm13_formal.py`;`experiments/rq1_encoder_bench_01/common/dataset_util.py` |Originalgeometry/order/recipeprovenance; hardcodedhistoricalpathsarenotnewrunentrypoints |
| Genericfeature/head |`research/sm13_backbone_side/features.py`,`heads.py`,`backbone.py`,`audit.py` |Geometry/pooling/leakageguards,scalarheads,frozenofflinebackbone,hashinventory |
| Commonheadengine |`epoch50.py` |Seed101/gap1101,AdamW,fit/heldspeakerpolicy,atomicfullstates,RNG,cursors,predictions |
| Nativeprofile/cache |`local_s1.py`;`tools/prepare_sm13_local_s1.py`,`verify_sm13_local_s1_cache.py` |Resourceprofile/cacheadmission/read-onlyidentitychecks |
| Firstfit/metric |`local_s1_fit.py`,`detection.py` |KeyedN/N+1population,scorepolicy,thresholdcalibrationvsfixed.5diagnostic |
| Continuations |`local_s1_extend.py`,`local_s1_plus200.py`;corresponding`tools/verify_sm13_*` |Exactstateparent/migrations/curves,notrestartfromscratch |
| Numericalreportrepair |`tools/recover_sm13_s1_plus100_report.py` |Preserveoriginalfailure;derivedFP32scoreguard/CPUrepair,noretraining |
| RMScomparison |`conditioning.py`,`raw_normalized.py`,`tools/verify_sm13_raw_normalized.py` |Fit-onlyscales,freshpairedinit/gates/exact50→300resume,independentproof |
| CPUdiagnosis |`tools/debug_sm13_s1.py`,`diagnose_sm13_normalized_plateau.py`,`debug_sm13_dropout_causality.py`,`replay_sm13_baseline_and_admission.py` |Inference-onlycontrols/evaluatorreplay,notcandidatequalitytraining |
| Currentpair |`dropout_paired.py`,`tools/verify_sm13_dropout_paired.py` |Importsengine/priorRMSfactory/detection/localresourcehelpers;verifieralsouses`tools.debug_sm13_s1.forward_all`andprobabilitydiagnostics |
| Validation |`tests/test_sm13_*.py` |Syntheticgeometry/leakage/resume/RNG/Adam/metric/resource/recipefixtures;somebroaderhistoricaltestsneedprivateassets |

Driver sourcehashchecksareintentional. AnewGitcheckoutcanreadhistory/testsyntheticfixturesbutcannotdirectlyrunanowner-pathregistrationwhosehashes/pathsdo notmatch. Createanexplicitnewregistrationfornewruntimeafterapproval; nevereditoldreceiptsuntiltheyPASS.

## 7. Head training recipe dependencies

Bothcurrentnormalizedarms:13×1024features,targetrefs39/gaprefs78,samefrozenRMS/nativecache.

| Setting | Target | Gap |
|---|---:|---:|
| Labels |C/S/O |noI/I |
| Hiddenwidths |256/128 |128/64 |
| Activations |ReLU |GELU |
| AdamWLR |.001 |.0008 |
| Weightdecay |.01 |.0001 |
| Batchsize |512 |1024 |
| Lossweights |[1,2.5,4] |Fit-onlynormalizedsqrtinversefrequency |
| Seed |101 |1101 |
| BaselineDropout |.25/.20 |.15 |
| CandidateDropout |.50/.40 |.30 |

Freshpairedweights/RNG/order/stepsmustmatch; **onlydropoutchanges**. AllnewcheckpointweightsoutsideGit. No automaticLS/architecture/LRchangeandnoconditional300extensionforthedropoutpilot.

Eachcheckpointcontainsmodel/Adam/RNG/epoch/cursor/shuffleorder/history/partiallosses/scheduler/scaler/pendinggradientmetadata. Atomicwrite+reload+SHA; nofailedbatchskip. Checkpointsatsafeoptimizerboundaries; resumesmustnotresetAdam/reseed/dropcursor.

## 8. Benchmark/evaluator/verification dependencies

- Spec:`SM13-COMMON-EVENT-DETECTION-v1`;scores=`pS+pO`target,`pI`gap.
- Keyedfullpopulation; TP/FP/FN/TN countatfixed.5;S/OtypedAPandinsertionexactgapAPretained; per-speaker/FWR/supportrequired.
- FirstS1calibration50usedfinite21×21gridonTRAINheldterminal50. Rawextensionskeepthose50thresholdsunchanged; currentRMS/dropoutcomparisonsusefixed.5diagnostic,notper-epochretuning.
- HistoricalEVAL-V3typedF1/FWRandpaperhierarchicalCROTTCF1arenotthismetric. NevercompareS0fullTRAINin-samplescorewithS1heldscore.
- Pairedpilotgates:fixed50F1delta≥1pp,last10meanF1delta≥0,macroAPregression≤.5pp,eachS/O/IAPregression≤2pp,≥2/3speakersnotworse,FWRincrease≤.5/câu.
- Freshbaselineverificationmustreproducehistoricalnormalized50state/predictionsbeforecandidatefit. Pairedverifierchecksall204states,Adam/RNG/buffers,NumPyshufflecontinuity,allper-epochpredictions/AP/counts,freshCPUfit/heldoutputreplay. **Plannedchecksnotverifiedoutcomesyet.**
- Endpointselection/runtimes/missingclasses/failedrunsmustbereported; onepairedseed/reusedTRAINlimitsremain.

### Evidence state of completed lineages

| Lineage | Proof | What it certifies / does not |
|---|---|---|
| S0epoch50 |`reports/SM13_EPOCH50_20261114/LOCAL_VERIFICATION.json` (local evidence; not included in docs-only push) |102states;notLargebackbonegain |
| S1first50 |`reports/SM13_NATIVE_LOCAL_20261114/FIT_VERIFICATION.json` (local evidence; not included in docs-only push) |102states;notfull1800→599comparison |
| S1raw150 |`reports/SM13_S1_PLUS100_20261114/VERIFICATION.json` (local evidence; not included in docs-only push),`reports/SM13_S1_PLUS100_20261114/POSTPROCESS_RECOVERY.json` (local evidence; not included in docs-only push) |204statesafterCPUreportrepair;originalFAILEDreceiptsretained |
| S1raw350 |`reports/SM13_S1_PLUS200_20261114/VERIFICATION.json` (local evidence; not included in docs-only push) |404states;overfit/regression,notagainclaim |
| Raw/RMS50 |`reports/SM13_RAW_NORMALIZED_20261114/PILOT_VERIFICATION.json` (local evidence; not included in docs-only push) |204states/gatebeforeconditionaladmission |
| Normalized300 |`reports/SM13_RAW_NORMALIZED_20261114/FINAL_VERIFICATION.json` (local evidence; not included in docs-only push) |602states;APgainbutnoendpoint-drivenF1gainclear |
| Dropoutpair |`reports/SM13_DROPOUT_PAIRED_20261114/LATEST_ADMISSION_STATUS.json` (local evidence; not included in docs-only push) |CPUadmissiononly,nottrain/qualityverification |

## 9. Resource/operational dependencies

SM13 **localonly**, CROTTCfulltrainingKaggle; donotinterchangescopes.

- SharedGPUlock:`/home/khoa/Projects/Pronunciation-RD-Coordination/locks/gpu.lock`.
- Root`/var/tmp`independentCROTTCexportfloor **14GiB**.
- Newdropoutweightsin`/home/khoa/.local/share/langfens/sm13_dropout_verify_20261114_v1/`,outsideGit;homefloor4GiB+initial1.5GiBoutputadmission.
- Currentpair≤3600s/2CPUthreads/RAM16GiBinitial,4GiBruntime/VRAM512MiBfloor;profileactualbatchbeforefit.
- Driverrechecksresources,stops/pausessafely;noautowaiter/retry/fallback/extension. Donotdeletecaches/checkpointstobypassgate.
- Documentation/pushdoesnotauthorizeproductionrelaunch;latesttaskispublication,nottraining.

## 10. Git vs private asset handoff

**Git:**source,configs,syntheticfixtures,planning/docs,smallregistrations/hashreceipts/aggregates,versioninventory.
**Privateauthorizedstorage:**audio/annotations/manifests,alignment/headinputNPZ,modelweights,allcaches/states/optimizer/RNGpayloads,per-examplepredictionsandlogs.

Machine-readabledeclarationsandpinnedidentities: `reports/TEAM_ONBOARDING_20261114/SM13_DEPENDENCY_AUDIT.json` (local evidence; not included in docs-only push).

Teammatechecklist:
1. Selecttaskandreadscope;dontlauncholdentrypoint.
2. Buildindependentenv;runscopedCPUfixturesoncleancheckout.
3. Obtainauthorizedprivateassetsseparatelyandvalidateallhashes/IDs/sourceidentity.
4. Preservehistoricallineage;registernewpaths/runtimechangeswithoutrewritingoldproofs.
5. Gateactualprofile/resources/sharedlockbeforefit.
6. Reportmatchedpopulation/epoch/metricwithindependentreceipts,includingnegativeoutcomes.

**Pending:**fullTRAIN1800headfit→commonDEV_SEARCH599;verifiedDEV_SEARCHalignment/features;commonCROTTCadapter;finalfusioncomparison;independentconfirmation. Theseareexplicitgates,notcompleteddependenciesjustbecausefilesexist.
