# TRAIN-PHASE-01 frozen formal-training protocol

**Status: frozen design, NOT_RUN; launch remains CONDITIONAL_GO pending the exact orchestration conditions in the GO gate.** This is not permission to start training during this task. The authoritative scientific design is unchanged from `METHOD_BENCH_08_TRAIN_PHASE_01_SPEC.md`, frozen before new preflight outcomes. No hyperparameter, architecture, seed or threshold-budget choice was changed after seeing preflight metrics.

Machine snapshot: `experiments/rq1_train_phase_01/preflight/supervisor/FORMAL_PROTOCOL_FROZEN.json`, byte-identical to the original MB08 config; SHA256 **6206c19db6cad9d99727a16cd9c220d077e09a9662a0229ee0971945bf1cbcb1**. Phase0 source/criteria freeze: `supervisor/{ACCEPTANCE_FREEZE.md,phase0_receipt.json}`. HEAD `aa4e6ea2147180a4a820d77a71911f333fcb6721` plus dirty-source hashes—not HEAD alone—defines provenance.

## 1. Inputs, architecture and objectives

Frozen acoustic source: ASR-finetuned `facebook/wav2vec2-base-960h`, snapshot `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`, hidden_states[6],768D/20ms; no encoder backprop. Existing audited caches are reused. Inference is audio plus **supplied intended phones**, or their certified cached features—not certified prompt-text-only inference. No gold timing, realized phones, error labels, speaker features or held-speaker fitted priors. Speaker IDs are export/split metadata only.

| Branch | Inputs / network | Objective / optimizer | Fixed budget |
|---|---|---|---|
| H-002 target | Predicted crop[3:14]: mean768 + population std768 + last−first768 + intended39 =2343; Linear256/ReLU/Dropout.25→Linear128/ReLU/Dropout.20→Linear3, order C/S/O | Weighted-mean CE[1,2.5,4]; AdamW lr.001/wd.01; batch512 | 12 epochs |
| Independent CTC/E4 | Full continuous frozen H6; Linear768→43,42 phones+blank42; exact canonical-mean sequence likelihood; ≤4 substitution/8 insertion competitors fit only on current fit speakers | CTCLoss blank42/mean/zero_infinity; AdamW.001/.0001;16 utterances; clipgrad5 | 8 epochs |
| Independent MB03 gap | At gapj: left cropj−1[11:16]mean768, right cropj[1:6]mean768, left/right intended39; missing edge inputs zero;1614→128/GELU/Dropout.15→64/GELU→2 | Fit-count weights sqrt(n/(2n_class)), normalized mean1; weighted CE; AdamW.0008/.0001; batch1024 | 5 epochs |

No shared trainable trunk, joint target/gap loss or teacher-to-gap dependency. Primary target scores `.5*neural + .5*softmax([0,SLLR,OLLR])`; preserve raw LLRs. Direct H-002 is fallback. No learned mixture, rank/percentile normalization, phone prior or meta-calibration in primary. Historical MB03 PLAIN target is a separately labeled ranking baseline, not a matched one-factor H-002 ablation.

Target scaler: fit targets only, mean/population std+1e-5. Gap scaler: fit gaps only, std floor1e-4. CTC no extra fitted scaler. Missing class in a fit is a failed split contract, not invented prevalence or an undefined head.

## 2. Data, folds, seeds and selection

TRAIN only:1,800 utterances/12 speakers/59,562 targets (C50,356/S7,381/O1,825),61,362 gaps (I560). No exclusions or gold rewrites.

| Outer | Held speakers | Targets | S / O | Gaps / I |
|---|---|---:|---:|---:|
|0|ABA/ASI/BWC|14,816|1,859/357|15,266/134|
|1|EBVS/HJK/HKK|14,848|1,524/301|15,298/127|
|2|HQTV/LXC/NJS|14,989|2,079/773|15,439/150|
|3|PNV/SKA/SVBI|14,909|1,919/394|15,359/149|

Outer fit nine speakers/1,350 utterances; three remaining official groups form local inner0/1/2, six fit speakers/900 utterances and three held/450. All fit-dependent transforms/competitors/model fits exclude validation and outer speakers. Never use outer-OOF scores as inner features.

Formal seeds101 primary/202 replication/303 sensitivity, no primary seed ensemble. Reset Python/NumPy/Torch **before each module construction**; inner seed=base+10×outer+local_inner; outer seed=base+100+outer. Save all epoch checkpoints; evaluate only fixed terminal12/5/8. No outer-driven checkpoint selection, early stopping, variable fold budgets or difficult-speaker omissions. Checkpoint must bind model+optimizer+RNG+sampler order/cursor+epoch+exact fit/held IDs+scalers+config/source hashes.

## 3. Exact identity and unknown rules

Emit N target and N+1 gap events per utterance, including both edges. Full key=(dataset,speaker,utterance,domain,index). Reject duplicates, missing/extra IDs, nonfinite outputs and broken topology. Accepted strict cached export facade: `preflight/supervisor/validated_inference.py`; A's permissive low-level multi-utterance helper alone is not the formal boundary.

Preserve zero39 UNKNOWN references and event slots at `ABA_arctic_a0025::21` ER), `BWC_arctic_a0434::28` D_, `SVBI_arctic_a0006::7` SPN, `HJK_arctic_a0104::4` Z_. CTC42 explicitly normalizes ER)→ER,D_→D,Z_→Z,Y_→Y,AH)→AH; SPN is class32. Preserve raw token/normalized provenance; unregistered tokens fail. H002's OOV→SIL alignment-only behavior is documented separately; no silent AA/phone0 reference replacement. Prompt phones originate from TextGrid canonical annotations, not verified text dictionary lineage; do not claim prompt-only input.

## 4. Joint policy and evaluator

B∈{.2,.5,1,2} false warnings/utterance; primary .2, allocation80% target/20% gap. Both streams must have exact eligible inner coverage and per-event fit exclusion/checkpoint/normalizer/source provenance. Target confidence=max(pS,pO), type argmaxS/O (tie S); descending confidence then descending first8 SHA256 bytes of `LANGFENS-MB07-warning-tie-v1:`+compact event ID. Choose largest inner prefix whose cumulative **wrong-type-inclusive FP**≤floor(allocatedB×inner utterances). Gap analogous. Freeze all thresholds/fit IDs before outer application; never use outer labels to fit cutoffs, calibrators or forced warning counts.

Use `research.evaluation_v3.evaluator.UnifiedEvaluatorV3`, EVAL-V3.1. Report S/O/I P/R/F1/AP, Macro3 Error F1/AP, binary target MD, correct secondary metrics, warnings, FP classes, wrong types, jointFWR=(FP_S+FP_O+FP_I)/1,800, utterance FP histogram/fraction,10-bin ECE/Brier/NLL, pooled/fold/speaker/seed supports and coverage. Ranking diagnostics P@560/P@100/P@500 are not policies. Missing valid provenance means null/NOT_ESTIMABLE. Nominal budget is not achieved-rate assurance. Optional paired whole-speaker bootstrap1,000 draws/seed8808 is conditional TRAIN uncertainty, not external confirmation.

## 5. Controls, budgets and stopping

Mandatory comparisons: direct H-002, E4-only, fixed fusion, MB03 PLAIN historical ranking reference, independent gap baseline. Mandatory diagnostic controls: zero/shuffled audio with topology/intended reference fixed, reference-only, explicit UNKNOWN handling. Their **formal Phase01 execution is NOT_TESTED**; historical MB08 controls stay historical, not silently relabeled a new ablation. Shared-trunk and staged-target-to-gap architecture ablations remain future NOT_TESTED alternatives; exact implementations/count matching need a separate prelaunch freeze. No encoder fine-tuning or new method search in this budget.

CPU default, CUDA_VISIBLE_DEVICES='', OMP/MKL/OPENBLAS/Torch2 threads;45-minute wall cap per formal seed. On cap/error save complete resumable state and stop, preserving unfinished folds. No outcome-directed budget reduction. Any uniform timing-based amendment must freeze before new outer outcomes. CUDA is outside current preflight; future use needs a newly valid cooperative exclusive lease, availability/VRAM checks, job/PID/config start/end records, telemetry and finally release. No historical missing lease is precedent.

**Feasibility evidence:** bounded24-utterance smoke61.955s/865,252KiB and gap-policy16fit wall33.57s; historical CTC12inner wall614.95s (different run/thread context). These suggest compact-head feasibility but do **not** measure a complete3-branch formal seed. Enforced coordinated hard-stop/orchestration remains a launch condition; no full run is performed merely to claim readiness.

## 6. Unfinished engineering gate and stop

Before any later formally authorized launch, implement/freeze/test the complete seed/fold driver, branch seed resets, all-epoch archive/fixed-terminal selection, strict per-utterance export/score provenance and global45-minute stop/resume as one workflow. Verify scheduling/stopping with mocks or tiny synthetic data; do not launch a full budget to test the gate. Current fit-loop/resume/identity/policy receipts are reusable and must not be restarted or tuned.

No DEV_SEARCH tuning, protected/TEST_SEALED access, destructive changes or remote push. This task stops at the documented gate. All raw scores, configs, source hashes, tests and receipts remain under the preflight root. No formal-run command was invoked.
