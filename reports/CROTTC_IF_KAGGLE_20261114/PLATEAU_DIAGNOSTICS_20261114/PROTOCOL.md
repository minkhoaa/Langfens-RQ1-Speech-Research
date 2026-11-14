# CROTTC plateau diagnostics — CROTTC-PLATEAU-DIAGNOSTIC-20261114-v1

User authorized the proposed sequence: saved-prediction error analysis, matched clean
TRAIN/DEV diagnosis, OT/CR component-gradient probes. **No counterfactual training
or recipe/production restart is authorized by this diagnostic.**

## Phase A — text-only error analysis (completed locally)

Fixed already-verified repaired3150predictions,599DEVutterances. Replay pinned
new-hierarchical counts from saved three-way alignments and require exact overall
and per-speaker agreement. Report C/S/D/I source-annotation subgroup support,
error-detection recall, correct-phone false rejection, phone-confusions and first3
lexicographic examples per outcome. Do not manufacture per-typeprecision/F1 by
ignoring correct-phonefalsewarnings. Strict S/O/I/exact-gap remains different/notcomputed.
Source files/hashes pinned; no audio, native forward, labels changed or threshold retuned.

## Phase B — bounded clean full-native diagnosis on Kaggle CPU ONLY

- Parent **5672/internalepoch99/cursor928**, independently downloaded/full-state verified;
  SHA`8d095b0f43cd1b1514efd548c4ec8416bf98938017791d662547c8d8a572695b`.
  Original32pendinggradients/group remain in immutable parent; diagnostic does not consume,
  discard, overwrite or promote them. Full-state resume is a separate existing campaign gate.
- Full native WavLM315453120 + heads3822765. Frozen clean evaluation FP32,
  nativeSpecAugmentoff, all modules.eval(), scoped2RoPEMHAattentiondropout0.
- Exactly **TRAIN1800 + DEV_SEARCH599**, existing frozen allowed manifests/WAVSHA,
  same clean decoder/normalization/metric/objective policy for both populations.
  RawTRAINsilence/unknownpolicy still differs from derivedDEV labels; gap alone
  cannot establish a unique causal-overfit mechanism. TRAIN is in-sample.
- Compute one-view OT objective AFTER audio-only native forward; no meaningful
  same-viewCRclaim. Greedyphonehypotheses, standard hierarchical counts,
  productionPER+no-silPER reported separately; output keyed hypotheses/losses.
- Do not compare nativeCPU/unpadded output as bitwise/equivalent to historical2T4FP16train.
  No featurearrays/modelweights saved or downloaded; only small text metrics/predictions.

## Phase C — component autograd, ZERO optimizer updates

- **3 fixed TRAIN IDs at frozen duration indices899/1709/1799** (median/p95/longest),
  **2 fixed seeds3047/3048 =6probes**, no outcome-based selection.
- Recreate actual source speed95/100/105, DropFreq/DropChunk and nativeSSLSpecAugment;
  fullSSLtrainmodewithCNNfrozen,heads/trainattentiondropout. Two views per utterance,
  independent native forward; gradientcheckpointing enabled as in production.
- Same OTTC operator as pinned source, **only device literal CUDA→CPU changed in
  disposable AST adapter**. No source math/detach/reduction edits. Toy tests compare
  CPU adapter, component derivative algebra and finitegradient geometry.
- `autograd.grad` separately OT and CR; report loss components,norms/dot/cosine
  for all/SSL/heads, combinednorm and hypotheticalclip5scale; transportalphaentropy/max.
  No optimizer instantiated/step, no LR/scheduler updates, no saved weight changes.
- **Interpretation limit:**6CPUFP32singlepairprobes are not productionbatch32/2T4gradient
  distributions. A large or negative ratio is evidence to investigate, not proof
  of causaltraininggain. No automatic CRcoefficient change, ablation or sweep.

## Resource and preservation gates

- Private separate **CPU-only** Kaggle kernel,visibleCUDA0,count0,no mainGPUcompetition.
  CPU2threads;RAM≥16GiBinitial,≥12GiBbeforebackward,≥4GiBbetweenitems.
- Platform14400s,driver12000s,bootstrap/finalizationreserve. Actualmedian/p95/longest
  native clean forward profile before6gradientprobes/full2399:
  `max(seconds/audioseconds)*total2399duration <10200s`. Fail→preserve/no retry.
- Each completed gradientprobe measured; if>300s stop before admitting another probe.
  This is a completion-time bound check, not mid-autograd hard preemption; driver
  process has the separate overall platform/driver watchdog. No skipped failed batches.
- Hash all weights before/after. Results COMPLETE only when all6gradientprobes and
  full1800/599cleanpopulations complete. Preserve partial outputs if resourcefailure.
- Existing maincontroller already stopped **naturally** after window002 export because
  storage<14GiBfloor; no stop/restart by diagnostic. Checkpoint5672/pendinggradients preserved.
  Do not delete/move checkpoints, lowerfloor or automatically resume to fund diagnostics.
- No local native backward/productiontraining/test/DEV_CAL/newcorpus access.
  Local receives lightweight text outputs; no new gigabytes of weight/cache download.

## Results/verification

Report means for cleanTRAIN/DEVOT and counts/F1/PER, source-policy caveats; subgroup
analysis and gradient probes all reported. Independent local text scorer recomputes
all2399hypotheses and keyedlossmeans, validatescountinvariants/gradnormcosinealgebra,
checkpoint/source/model-integrity receipts. Nativeprofile pass is not finalmetricsPASS.
No numerical acceptance is adjusted after seeing outcomes.
