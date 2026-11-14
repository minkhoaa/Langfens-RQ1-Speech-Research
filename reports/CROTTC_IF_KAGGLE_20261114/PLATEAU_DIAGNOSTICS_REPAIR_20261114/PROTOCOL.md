# Authorized diagnostic repair v2 — 2026-11-14

Decision: **CROTTC-PLATEAU-DIAGNOSTIC-NOGRAD-REPAIR-20261114-v2**.
Authority: current user "oke, thực hiện theo thứ tự bạn đề xuất đi" after explicit
report of failed diagnostic and proposal to repair/test diagnostic before choosing
training intervention. This is a manual authorized bounded attempt, **not automatic retry**.

## Failure preserved and reproduced

v1 kernel `minkhoaa/langfens-crottc-plateau-diagnostic-cpu-v1/1` is ERROR/terminal.
Downloaded original failed logs/registration/CPUprofile/environment are retained at
`../PLATEAU_DIAGNOSTICS_20261114/retrieval_v1/failed_output/` and original diagnostic
sources are copied/hash-bound in `PROVENANCE/ORIGINAL_SOURCE_HASHES.json`.
No existing file/checkpoint/receipt is deleted/rewritten to pretend v1 succeeded.

Root mechanism reproduced with installedSpeechBrain1.0.3 tinyRoPEMHA:
cleanwarmup under `torch.inference_mode()` populates memoizedsin/cos/index tensors
as inference tensors. A later grad-enabledforward tries saving cachedindex for
backward and raises `Inference tensors cannot be saved for backward`.
This was a **diagnostic execution bug**, not evidence of failed modeloptimizer or
causalexplanation oflowF1. v1 has no completegradient/clean2399results.

## Only v2 changes

1. Diagnosticcleanforwardcontext uses **torch.no_grad()** instead of inference_mode.
   It still disablesgrad, but creates normalcachetensors valid for later autograd.
2. Print completednativeforwardprofile for visible resourceadmissionevidence.
3. Separate unique v2kernel/package/output/repairprovenance, verifier/retrieval binds
   explicitv2registration and repairID. No change to underlyingloss/model/data.

SyntheticCPUregression test shows exactsamecleanevaloutput between contexts,
replicatesoldexception, then repair permitsfinite/nonzerogradients withnooptimizer/
weightchange. Cachedtestdata is isolated/restored so other tests do not hide bug.
**25tests passed/1strictxfail**;xfail is separateknownoriginaleval-dropoutdefect,
notthisfixedcontextbug. No nativeWavLMbackward is run locally.

## Frozen experimental design / limits

Inherit base `CROTTC-PLATEAU-DIAGNOSTIC-20261114-v1` protocol unchanged:
- Fixedfullstateverifiedcheckpoint5672/internal99cursor928,
  SHA`8d095b0f43cd1b1514efd548c4ec8416bf98938017791d662547c8d8a572695b`.
  Parentpending32/32TRAINexamplesgradients remaininplaceunconsumed.
- ExactlyTRAIN1800/DEV599, samecleanrepairedeval/normalization/decoder/scorer/OTsingleview;
  fullnative315M+heads,CPUFP32;nosameviewCRqualityclaim.
- Sixgradientprobes:fixedTRAINindices899/1709/1799 ×seeds3047/3048,
  unchangedtrainmodeaugmentation/nativeSSLSpecAugment andseparateOT/CRautograd.
  Lossmathunchanged,onlydevice-literalCPUadapterinpinnedOToperator.
- **Zero optimizer/weight updates**,noLR/scheduler/campaign/label/productiondependencychange.
- PrivateCPU-onlyKaggle,2threads;16GiBinitialRAM/12GiBprebackward/4GiBruntimefloor,
 12000sdriver/14400splatform;forwardprofileestimate<10200s beforefulladmission,
 probecompletion>300sstopnextprobe;globalwatchdogboundsinflightcomputation.
- Partialfailure outputs preserved;no skips/no automatic thirdattempt/newtraining/
 ablation/sweep/localCROTTC/Colab/testDEV_CALaccess.
- CPUoneutterancepairs do not proveproduction2T4FP16batch32gradbalance; cleanTRAIN/DEV
 annotationpolicies differ,so gapalonecannotisolatesingleoverfitmechanism.

Mainproductioncontroller alreadyterminaldiskfloor afterverified5672export,not stopped
bydiagnostic;donotresume/delete/moveprotectedweights/cache/lower14GiBfloor.
Independent local textverifier checks2399predictions/lossmeans/per-speaker metrics
and6gradientprobeartifactcoverage/finite/scalaralgebra;notindependentnativegradreplay.
