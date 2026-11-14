# Paired reproduction/control — implementation complete, actual admission BLOCKED

## Đã làm trong lượt “tiếp tục đi”

Không dừng ởbrainstorm: đãviếtdriver `research/sm13_backbone_side/dropout_paired.py`, independentCPUverifier `tools/verify_sm13_dropout_paired.py`, testfixtures `tests/test_sm13_dropout_paired.py`, rồi chạytests và gọi productiondriver thật qua admission. Sourceđượcpin trong `REGISTRATION.json` và snapshot `PROVENANCE/`; khôngsửadriver/head/recipeSM13lịchsử.

**85targetedCPUtests passed** (`CPU_TESTS.log`), khôngfullrepoPASS hoặcGPUproductionPASS. Các tests optimizer là disposabletoyfixtures, khôngrealdatafit/candidatequalitytrial.

## Driver làm gì khi admission đạt?

1. CPUstorage/RAMcheck trướctạoworkdir/CUDA/GPUlock; rootreserve14GiB + homefloor4GiB/outputadmission1.5GiB.
2. SharedGPUlock/existingjob/actualVRAMcheck; localonly/noCPUorremotefallback.
3. Rehash registeredcache/audio/inputs, recomputeexactfit-onlyRMS, comparehistoricalstats; samefreshheadinitialstate/RNG, sourcepin.
4. Actualdisposabletarget/gap×2armsbackwardprofile(batch512/1024), finitegradient/resourcechecks.
5. Trainfreshnormalizedbaseline target50/gap50 atomicfullstates; bounded1h, safeoptimizerboundaryonbudget/signal/resourcepause.
6. **Blockingbaselineverifier** kiểm102states+freshfit/heldinference, all50predictionepochs/modelstateparityvớihistoricalnormalized50. Khôngtraincandidate nếureproductionkhôngđạt.
7. Candidate target50/gap50, chỉexistingdropoutp×2; sameweightsinit/data/batchorder/seed/LR/loss/weightdecay/fit-onlyRMS/fixed.5.
8. **Blockingpairedverifier** kiểm204states/Adamcounters/RNG/buffers/keyedpredictions/independentcounts+sklearnAP/fullcurves/fit-vs-heldlogitCE/frozenqualitygate; matchedNumPyshuffleRNG/updateseach-epoch, baselineartifactsnotmodified, plots.
9. Stop50/head/arm dùqualityPASS/FAIL; không300extension/bestepochselection/thresholdretune/grid. Onepairedseed/reusedTRAINdevelopment, khôngcommonDEV_SEARCH/testconfirmation.

Cácguardfixturesđãkiểm: wrongscope/source/epochs/platform; exactinitstate/RNG/paramcount/chỉpđổi; exactresumeweights/Adam/RNG; fakeAdamreset dùreceiptSHAđãupdate; RMSbuffer/predictionUIDcorruption; wrongmetriccounts; rootshortfall dùhomedư; resume khôngđượcbypassfreshoutputadmission; admissionblocked khôngtạoworkdir/đụngCUDA.

## Ước tính storage từartifact thật

Samecapacityhistoricaltargetfullstate10,004,653bytes;gap3,408,685bytes, pluspredictions.204states+200predictionfiles≈1,393,941,376bytes(~1.298GiB), +15%margin trong1.5GiBoutputadmission. Đâyestimate, khôngthayactualresourceprofile/runtimeguard.

## Kết quả lệnh launch thực tế

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv-crottc-official/bin/python -u -m research.sm13_backbone_side.dropout_paired \
  --registration reports/SM13_DROPOUT_PAIRED_20261114/REGISTRATION.json
```

`ATTEMPT.log` và `LATEST_ADMISSION_STATUS.json`: **RESOURCE_BLOCKED_NO_TRAINING**.
`ADMISSION_1794670735238805419.json`: rootfree14,597,033,984bytes <15,032,385,536bytes(14GiB),thiếu435,351,552bytes(~415.18MiB). Home/RAMđạt.

Driverexit0là cleanadmissionrefusal, **khôngphảitrainingthànhcông**. Work `/home/khoa/.local/share/langfens/sm13_dropout_verify_20261114_v1` **chưađượctạo**. Khôngtrainer/verifierđangchạy; GPUprofilechưachạy. Khôngtạowaiter/scheduler/autoretry hoặcxóa/moveprotectedartifacts/hạfloor/độngCROTTCđểlấychỗ.

**Chưa có freshbaselinefit, candidateF1 hoặc qualitygain.** ProductionreadinessmớiđượcCPUfixturekiểm; actualGPUprofile/reproduction/qualityverificationvẫnpendingvalidadmission.

## Tiếp nhận

Driverđãcóentrypointthật ởtrên, khôngcầnbrainstormhoặctựđổiLS/RDrop. Cầnphươngánstorageđượccấpphépđểđảm bảoindependent14GiBrootreserve; recheckmanualadmissiontrướcrun. Khôngxóacheckpoints/cacheshoặcreducemarginđểláchgate. Khiadmissionđạt,chạyboundedpairvàchỉkếtluậntừverifiedreceipts.
