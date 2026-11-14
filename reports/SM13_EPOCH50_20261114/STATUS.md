# SM13 epoch50 — bounded user amendment, 2026-11-14

## LIVE16:03 — target50complete / gap7+ running, mainuntouched

LocalPID1448557 hascompleted **target50epochs/4400updates**, latestcheckpointSHAverified `455a65f330e36ed1…`; gapheadhascompleted7epochs/322updatesandcontinues toward50. MainKagglewindow001stillRUNNING/controller1291861alive; no maininterruptions. Full50/50comparisonnotclaimeduntilfinalverifier.

Localadapter/admissiontests passed beforelaunch. ConcurrentfullCPUregression:227passed,3fixtureerrors because `tests/test_crottc_full_resume.py` correctlycouldnotacquiretheglobalGPUlockheldbyrealSM13training. Preserved `LOCAL_FULL_REGRESSION.log`; **notclaimedallPASS**, doNOTrelease/bypasstheproductionlockorrunCUDAunitjobsduringtraining justtogreentests. Prelaunch222CPUregressionandlocaladmissionunitlogremainvalid.

## LIVE16:00 — SM13LOCAL RUNNING; MAINUNINTERRUPTED

Latestuser explicitlyrequests SM13local withoutbreakingfullpapertraining. **LocaltrainingPID1448557** onGTX1650/sharedGPUlock hasrealoptimizerupdates/epochs, targetepoch16/update1408seen15:58 (notjuststartup). Usesverifiedoriginal26FP32featurefiles directly, notbackboneextraction/fine-tuning or11GiBnewcache. FullTRAINcache59562targets/61362gaps; fit9/held3speakers unchanged. Target50+gap50/seed101/4hbudget/2threads/atomicfullstates. Runtime `/var/tmp/langfens-crottc-full-20261113-t_car8m4/sm13_epoch50_local_v1/` (`STATUS.json`, histories, epochcheckpoints/predictions); logs `LOCAL_TRAIN.log`, proof `LOCAL_LAUNCH_RECEIPT.json`.

Read-onlySDKquotaendpoint discovery: used~28856GPU-seconds >21600secondsallowed, reserved0, reset21NovUTC. RawAPI hasmalformedfractionalsecondssuffix, preserved/parsedleadingnumericseconds. **This confirmsquotaexhaustion**, not merelyQUEUED inference. `LOCAL_ADMISSION.json`, `QUOTA_1551.json`. Theknownalreadyadmittedmainkernelstillhasupdates; don'tstopit orassumequotaaffectsitscurrentwindowimmediately. Futureadmissionmayfail.

RemoteSM13v1stillQUEUED/**notclaimedcancelled**: publiccancelSDKrequiresknownsessionID, owner/sluglookupnotexposed. NeverguesssessionIDs/deleteartifacts. Local4hwindowendsbefore6dayquotareset; in-process60sread-onlyguardpauseslocalatoptimizerboundaryifremoteunexpectedlyRUNNINGorAPIuncertain. No cron/timer. FiniteCPUverifierPID1451451willcheckall102headstates/Adam/RNG/history/predictionmetrics/fixed12/5vs50/50/bootstrap andrendercurvesoncecomplete; state `LOCAL_VERIFIER_STATUS.json`. Noimprovementclaimbeforeheldevaluation.

**MainCROTTCcontrollerPID1291861/remotewindow001unchanged**, liveupdate2400/internalepoch42cursor192(human43)at15:58 in `MAIN_UNINTERRUPTED_1600.log`. No stop/restart/rollback/resubmit/STOPfile orlocalCROTTC. Existingmainexport/retrievalremainspriority. Test/DEV_CALremainsealed. Earlierqueue-onlysnapshots below arehistorical.

**Live15:43+07:** privateGPUkernel stillQUEUED aftersuccessful15:29push; **zero SM13productionupdates observed**. StructuredKaggleAPI reports `status=QUEUED,failureMessage=null`, not quotaexhaustion. No localduplicate started while remote mayadmit. Maintraining remainsRUNNING. CPUregression latest **222passed**, `CPU_REGRESSION_V3.log`, including partial-live-log curve regression.

Decision `SM13-EPOCH50-20261114-v1`; authorization/protocol in `SIDE_GOAL.md`. User explicitly permits Kaggle-first training, local SM13 fallback only if Kaggle resources unavailable. Local CROTTC remains prohibited; main2T4training stays untouched.

## Registered experiment

- **S0 original frozen Wav2Vec2 Base960h**, revision22aad52d435eb6dbaf354bdad9b0da84ce7d6156. Full13layers/FP32features; target/gapheads exactly the previously unit-verified SM13 architecture. No backbone gradients, E4/CTC changes, fullfine-tune or sweep.
- **50 target epochs AND50gap epochs**, seed101; same LR/loss/batch/weightdecay as originalformalrecipe. Refit from scratch on1350TRAINutterances/9speakers, retain450TRAINutterances/ABA+ASI+BWC as historicalouter0development. Gold inference-label separation preserved. This is reusedresearchdevelopment, not newly independent confirmation/test.
- Compare **target12/gap5 versus target50/gap50** on identical held events. Everyepoch logs heldloss/AP/F1 plus trainloss; primaryneural-onlymacro3AP. ArgmaxF1 isdiagnostic, not registeredfullfusionwarningpolicy. Do not compare itsnumbers directly with CROTTCpaperdetectionF1.
- NativeFP32feature generation uses originalHFnormalization and frozenpredictedalignment;16realparitychecks (8TRAINIDs×2heads) against originalcache must pass before fitting, with declaredFP32tolerance1e-3relative/3e-4absolute. No goldtiming. Full1800features/59562targets/61362gaps, not a compactdata replacement.
-4hdriver/16800platformseconds,2CPUthreads/onevisibleGPU. Resourceguards, sharedlocalGPUlock if fallback, atomiccheckpoint per epoch and on optimizer-boundary pause. Fullweights/Adam/RNG/cursor/shuffleorder/partiallosses, no partialgradients at save; immediate fullrecursiveCPUreload equality plusSHA receipt. CPU tests prove interrupted resume is tensor/optimizer/metric identical.

## Execution evidence / live state

- **219CPUregression tests passed**, `CPU_REGRESSION.log`.
- Privateinputs dataset `minkhoaa/langfens-sm13-epoch50-inputs-v1/1`: about409MiB, onlypinnedpublicnativeweights + TRAINrequests/labels/referencevectors/predictedcenters/eightrealfeatureparitysamples/source. **No raw audio reupload, test/DEV_CAL/credentials or full11GiBfeature reupload.** Reuses immutableauthorizedassets1. Private/versionreadback in `PRIVATE_DATASETS_PRERUN.json`.
- Initial datasetstatus403during creation preserved in `DATASET_ADMISSION_STATUS.log`; followingstatusREADY andprivate/versionverification succeeded. This is **not** GPUresource failure and doesnottriggerlocalfallback.
- GPUkernel submitted successfully15:29, **QUEUED at15:32**, not yet claimed actualtraining:
  https://www.kaggle.com/code/minkhoaa/langfens-sm13-epoch50-s0
  Pinnedversion1, private/GPUflag/server-scriptSHAverified: `LAUNCH_RECEIPT.json`, `tools/kaggle_sm13_epoch50_v1/WINDOW_REGISTRATION.json`, `PUSH_GPU.log`.
- Onefinite CLIretrieval **PID1418016**, state `retrieval_v1/RETRIEVAL_STATUS.json`; log `RETRIEVAL.log`. It only waits/downloads/verifies thisonejob, neverresubmits/fitslocally. No cron/timer. Outputbase `/var/tmp/langfens-crottc-full-20261113-t_car8m4/sm13_epoch50_s0_v1/`.
- Retrieval preserves/checks all100epoch+2terminalheadstates/optimizer/RNG and allheldpredictions; **11GiBfeature arrays remain in privateimmutablekerneloutput** to prioritize maincheckpointexport disk. Do not claim thosearrays independentlydownload-verified. Numericalparity/model/inputprovenance receipts are downloaded. Completion verifier recomputesAP/F1 fromeachheldprediction and fixed12/5vs50/50comparison, plus exact27resample paired3-speakerbootstrap (exploratorylimitedsupport).
- No Kaggleadmissionerror exists yet; no localproductiontraining was started. Do not launch duplicatefallback while remotequeued/running. If remotecode/data/parityfails, stop/preserve; do not relabelcodeerror as lackofresources. S1/S2replacement experiments remainpending separategates.

## Main CROTTC development curve

At15:42source snapshot,38completedepochs; update2218/internalepoch38/cursor1664 (humanepoch39). DEV_SEARCH599latestepoch38 F1=50.3142%, PER=27.1725%; bestbothsofar epoch28 F1=54.1639%, PER=25.2566%. Recent5means F1=51.0753%, PER=26.9001%.

Fullcurve: `reports/CROTTC_IF_KAGGLE_20261114/CURVES/SNAPSHOT_1543/CURVE.png` (sourcecaptured15:42), `VALIDATION_CURVE.csv`, `SUMMARY.json` andpinnedsourceSHA. KaggleliveAPIcanreturnonlytail: `INVALID_TAIL_ONLY_1542` ispreservedinvalidreport, donotuseitsbest/recentmeans. Fixedreport1543mergespinnedpriorCSVandrequirescontiguousepochs1–38; sourceSHAincluded. EarlyPERdecreasesfast, thenimprovesmore slowly/oscillates; F1isflat-isharound50–54% andnotmonotonic. Latestepochisnotbest. Theseobservations donotproveoverfitting or justifychangingregistered300epochschedule. Trainlossalonecannotanswerquality. No testopened/authorbenchmarkparityclaimed. Latestindependentlyverifiedfullcheckpointstill482untilcurrentwindowexport.
