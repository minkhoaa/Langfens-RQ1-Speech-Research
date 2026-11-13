# RQ1-TRAIN-PHASE-01 — controlled training specification

**Decision:** CONDITIONAL_GO; implement the bounded preflight and joint-policy work below before the formal run. **Status: specification only; formal training NOT RUN.** MB08 stopped at method verification. Authoritative selection is this supervisor specification; the independent alternative/design discussion remains in `experiments/rq1_method_bench_08/agent_c/TRAINING_SPEC.md`.

Machine-readable planned configuration: `experiments/rq1_method_bench_08/supervisor/TRAIN_PHASE_01_CONFIG.json` (specification only, not an executed run).

Source revision `aa4e6ea2147180a4a820d77a71911f333fcb6721`, dirty user work preserved. Untracked source/config files are identified by `experiments/rq1_method_bench_08/EVIDENCE_MANIFEST_SHA256.json`, not HEAD alone. Environment: Python3.11.17, Torch2.6.0+cu124, NumPy2.4.6, sklearn1.9.1; CPU i5-12450H/32GB RAM. No protected data or DEV_SEARCH tuning.

## 1. Methodology and architecture selected

**Primary: independently trained modules over a common frozen acoustic feature extractor.** This is genuinely modular: no shared trainable trunk, no joint target/gap loss and no sequential teacher-to-gap dependency.

| Component | Frozen design | Reason |
|---|---|---|
| Acoustic source | `facebook/wav2vec2-base-960h`, snapshot `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`, hidden_states[6],768D,20ms | Existing verified continuous features; ASR-finetuned, not untouched SSL |
| S/O primary | H-002 direct C/S/O MLP + independent CTC/E4; fixed 50/50 probability average | Replicated omission ranking and favorable low-FWR operating evidence; no prior/meta-fitting dependency |
| S/O fallback | Direct H-002 C/S/O MLP, same fitted checkpoint | Removes E4 compute/dependency; weaker low-FWR omission recall |
| Insertion | MB03 PLAIN exact-gap adjacent-edge MLP | Valid keyed baseline AP-I .162124897/P@560 .242857; both MB08 alignment seeds lose |
| Historical reference | MB03 PLAIN target + gap scores | Stronger historical S ranking; different architecture/input/budget, not the H-002 direct baseline |

A shared **trainable** trunk with target/gap heads is a unified multi-head model, not the modular primary. No controlled transfer benefit supports making it primary. Staged S/O-then-gap training adds cross-fit obligations and error propagation without demonstrated insertion gain. Both are future architectural ablations, not approved new search families or prerequisites for the first controlled modular run. The MB08 learned-emission monotonic aligner remains a recorded negative comparison; do not promote it or tune it from its outer results.

### Exact model definitions

1. **Target neural head:** predicted-alignment target crop offsets3:14 from the cached17-frame H6 crop. Concatenate mean768, population std768, last-minus-first768 and intended reference39: input2343. `Linear(2343,256),ReLU,Dropout(.25),Linear(256,128),ReLU,Dropout(.20),Linear(128,3)`. Softmax class order `[CORRECT,SUBSTITUTION,OMISSION]`.
2. **CTC/E4 branch:** independent `Linear(768,43)` over continuous frozen H6 frames; fixed42-phone inventory and blank42 from the audited MB03 CTC producer. Train with realized-phone sequences **only as eligible training supervision**. Construct exactly the existing canonical-mean E4 recipe: at most4 substitution competitors per intended phone and8 insertion competitors, fitted from the current fit speakers only; exact CTC sequence log likelihood; S=logmeanexp(substitution alternatives)-canonical loglikelihood; O=deleted-candidate loglikelihood-canonical. Preserve raw LLRs separately. Target fusion uses `e4=softmax([0,SLLR,OLLR])`; `p_target=.5*p_neural+.5*e4`. No pooled percentile/rank normalization, learned mixture coefficient or phone prior.
3. **Gap head:** for gapj, left target crop offsets11:16 mean768 and right offsets1:6 mean768, plus left/right39 intended-phone vectors: input1614. Missing left/right utterance-edge inputs are zero. `Linear(1614,128),GELU,Dropout(.15),Linear(128,64),GELU,Linear(64,2)`. Softmax class1 is insertion. Target and gap parameters, scalers and optimizers are independent.

## 2. Input/output and data contract

Inference accepts audio16kHz and **supplied intended phonemes**, or prompt text only after the text-to-phone lineage gate passes. Intended-phone conditioning is operator-authorized. It must never consume gold timing, realized phones, error labels, speaker identity or held-speaker fitted priors. Acoustic/predicted reference alignment uses frozen audio predictions plus intended phones only. Gold fields may supervise eligible fit partitions and evaluate predictions, never form held-out inputs.

Emit exactly N target rows and N+1 gap rows including both edges. Full evaluator key is `(dataset_id,speaker_id,utterance_id,domain,index)`; compact `uid::index` is permitted only with explicit target/gap arrays and proven TRAIN UID uniqueness. Target output: logits and C/S/O simplex. Gap output: insertion logit/probability. Persist seed,fold,checkpoint hash,normalizer hash and feature-producer lineage with each stream. Scores must be finite; duplicate/extra/missing IDs fail the preflight, not silently reorder or drop.

Population is **TRAIN1800 utterances/12speakers/59562targets** (C50356,S7381,O1825), **61362exact gaps** (I560/negative60802). No data exclusions or gold-label rewrites. Four official outer groups:

| Fold | Held speakers | Target count | S/O support | Gap count/I support |
|---|---|---:|---:|---:|
|0|ABA/ASI/BWC|14816|1859/357|15266/134|
|1|EBVS/HJK/HKK|14848|1524/301|15298/127|
|2|HQTV/LXC/NJS|14989|2079/773|15439/150|
|3|PNV/SKA/SVBI|14909|1919/394|15359/149|

The current39D reference cache has four explicit unknowns: `ABA_arctic_a0025::21 ER)`, `BWC_arctic_a0434::28 D_`, `SVBI_arctic_a0006::7 SPN`, `HJK_arctic_a0104::4 Z_`. Retain their zero reference vectors and event slots under a documented UNKNOWN rule; never silently argmax them to phone0 or change gold. Do not adopt the MB08 aligner's AA fallback. A future phone-prior control must use the reference39 slice with an explicit unknown category, never `X[:,-42:]`.

## 3. Splits, seeds and checkpoints

For each outer fold, fit on the other **nine** speakers/1350utterances. Three inner groups are the remaining official three-speaker groups in sorted speaker order; each inner base fit uses6speakers/900utterances and scores3unseen speakers/450utterances. All fit-dependent operations—normalization, class weights, emissions, competitors, projections, calibrators and any meta learner—must exclude the speakers whose predictions are evaluated. Never use outer-OOF predictions as inner training features.

Formal seeds: **101 primary,202 replication,303 sensitivity**, identical folds and budgets, no seed ensemble as primary. Before constructing any model seed Python/NumPy/Torch. For each module use inner seed `base+10*outer+inner`; outer refit `base+100+outer`. This new recorded gap schedule differs from the historical MB03 baseline's `101+fold`; keep historical artifacts unchanged. Save RNG states and exact fit/held IDs.

Checkpoint selection is **fixed terminal epoch**, not best outer or best pooled AP: target12epochs,gap5epochs,CTC8epochs. Save all checkpoints but evaluate the predeclared terminal only. No checkpoint/architecture selection from outer labels. No early stopping or per-fold adaptive epoch budget. If an input-only timing pilot requires a uniform budget amendment, register it before any new outer outcomes; otherwise abort/resume without selecting favorable folds.

### Optimizers and imbalance

| Module | Loss / weights | Optimizer / batch |
|---|---|---|
|Target|weighted C/S/O CE, fixed historical weights `[1,2.5,4]`; weighted-mean denominator sum of batch weights|AdamW lr.001,wd.01,batch512,12epochs|
|Gap|weighted binary CE; `w_c=sqrt(n_fit/(2*max(n_c,1)))`, normalize weights to mean1, fit-partition counts only|AdamW lr8e-4,wd1e-4,batch1024,5epochs|
|CTC|CTCLoss blank42,reduction mean,zero_infinity=True; no inference labels|AdamW lr.001,wd1e-4,batch16 utterances,8epochs,clipgrad5|

Target feature scaler: mean/std from fit targets only, `sd=std(ddof0)+1e-5`, matching MB07. Gap scaler: fit gaps only, `sd=max(std(ddof0),1e-4)`, matching MB03. CTC consumes raw frozen H6 frames with no extra fitted scaler, matching its producer. Preserve native deterministic waveform normalization. No global TRAIN scaler or validation prevalence fit. Do not oversample evaluated populations. Report zero-positive fit partitions as a failed split contract rather than fitting an undefined head.

## 4. Inner-OOF joint warning policy

Generate **both target and gap** inner-OOF predictions from their own six-speaker fits, covering every outer-training event exactly once. This missing gap-policy integration is bounded required work; MB03's archived outer scores cannot substitute as inner predictions.

For S/O confidence `max(p_S,p_O)`, warning type is argmaxS/O (tieS). Sort inner events by descending confidence, then descending first8bytes SHA256 of `'LANGFENS-MB07-warning-tie-v1:'+event_id`. For each prefix count false warnings as **predicted error class != gold**, including S↔O wrong-type errors. Choose the largest prefix satisfying cumulativeFP≤floor(B_target*N_inner_utterances). Freeze score/tie cutoff and fit IDs before outer application. Gap selection is analogous using pI and false=gold-negative gap. Do not fit thresholds to outer labels or rank the outer population to force a warning count.

Predeclare total false-warning budgets B∈{.2,.5,1,2}/utterance. Primary joint policy allocates **80% target/20% gap**; fit each threshold on innerOOF at .8B/.2B. At the operational primary B=.2, budgets are .16target/.04gap. Report all four policies, not whichever looks best. Target-only comparisons at budgets.2/.5/1/2 are separate and align with MB08.

Outer achievedFWR need not equal the nominal inner budget; this is not a guarantee of safety. Report achieved values, warnings and FP separately. Common-support interpolation of already-frozen outer curves is **EXPLORATORY**, never a deployable threshold or evidence of an exact matched operating point. If no common support, report no comparison.

No fitted calibration by default; supplied simplex values may be uncalibrated. If calibration is later preregistered, use a disjoint inner calibration group and refit the **entire** base pipeline excluding it. MB07's ordinary second-stage meta crossfit is insufficient because base fits for meta-training rows can have seen meta-held speakers. Learned meta-policy results remain ineligible until that full exclusion is implemented. No sigmoid/isotonic/temperature/scaler may be fitted on outer labels.

## 5. Baselines, controls and analysis limits

Mandatory score/policy comparisons: directH002,E4-only,fixed.5neural+E4,MB03PLAIN ranking reference,independent gap baseline. Preserve original MB07 prior-containing AP under historical labels; v2correct39 prior and learned no-prior ablations are separately versioned, not interchangeable controls. Learned meta-policy is not primary.

Mandatory controls: exact ID/topology/speaker checks; all-zero or deterministically shuffled audio at inference with reference/topology fixed; reference-only diagnostic; explicit unknown-token handling. The MB08 insertion ablation removes gap states while reusing identical learned emissions; it is a structural inference ablation, not independently optimized alignment training. If the new aligner is replayed, retain its fixed recipe and acoustic shuffle; do not tune it from outer results.

Architecture ablations are **NOT_TESTED** in MB08 and not required to begin the modular protocol: (a) shared trainable128Dprojection with separate C/S/O and gapheads, jointly `Ltarget+Lgap`, versus independent projections with same total epoch/data budgets; (b) stage target training then frozen target representations to gap, using strictly cross-fitted representations. Their exact implementation/count-matching must be frozen before launch; no vague architecture claim is an executed result. Encoder fine-tuning is outside the first controlled budget.

No hypothesis of universal alignment failure follows from MB08. Repeated seeds share the same12speakers; TRAIN has been repeatedly explored and is not blind external validation. DEV_SEARCH remains exposed and is not used to tune; protected/TEST_SEALED access is prohibited. External/new-speaker confirmation needs a separately authorized later protocol, not an unannounced holdout read.

## 6. Complete EVAL-V3.1 outputs

Use **`research.evaluation_v3.evaluator.UnifiedEvaluatorV3`**, not older generic target-macro evaluators.

- AP-S/O over all59562targets; exact-gapAP-I over61362gaps; **Macro3 Error AP=(AP-S+AP-O+AP-I)/3**.
- Classwise TP/FP/FN,precision,recall,F1 for S/O/I; **Macro3 Error F1=(F1-S+F1-O+F1-I)/3**. Correct metrics and C/S/O target macro are secondary and explicitly named.
- Binary targetMDAP/P/R/F1; exact-gapP@560,P@100/500 and corresponding recall. TopK diagnostics are not frozen deployment policies.
- Total warnings/utterance, **jointFWR=(FP-S+FP-O+FP-I)/1800**, wrong-type count,false-alarm-on-C count,per-utteranceFP histogram/fraction withFP,coverage/topology. Wrong S/O type counts once as a false warning.
- Calibration: classwise equal-width10binECE/Brier/reliability plus multiclassNLL; label ECE15 from MB08B separately, not as identical to official10binECE. No posterior-calibration claim from simplex alone.
- Pooled,perfold,perspeaker and perseedeffects with supports and actual execution status. Optional paired speaker bootstrap1000draws,seed8808,report conditional CIs and no independent-cohort claim.
- Missing valid threshold/prediction provenance means **null/NOT_ESTIMABLE**, never zero, invented threshold or fabricated aggregate. Current MB08 complete-systemF1/FWR are null; this does not prevent valid score-only AP assembly.

## 7. Compute, lease and stopping

CPU default: `CUDA_VISIBLE_DEVICES=''`,OMP/MKL/OPENBLASthreads2,Torchthreads2. Frozen features and compact heads avoid encoder backprop; existing timings support feasibility, but full3seed protocol wall/RSS are not yet measured. Pilot on an eligible training partition before outer outcomes; record maxRSS and step counts. Per-seed wall cap45minutes; on cap/error save resumable state, report unfinished folds and stop. Do not reduce epochs after inspecting outcomes, omit difficult speakers or launch an unbounded search.

GTX16504GB is optional only with a new valid cooperative lease: exclusive shared coordination lock, availability/VRAM check,LEASE_START with job/PID/config,telemetry,finally LEASE_END/release including failure. Do not rely on sampled idle GPU or the legacy3-slot700MB rule as proof of4GB feasibility. CPU fallback is valid. MB07's missing lease receipt is retained as a historical deviation; no MB08 CUDA training occurred.

## 8. Reproduction commands and required receipts

Run from repository root. These are **implemented audit/replay commands**, not training launches:

```bash
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python experiments/rq1_method_bench_08/supervisor/audit_inputs.py
.venv/bin/python experiments/rq1_method_bench_08/supervisor/audit_new_results.py
.venv/bin/python experiments/rq1_method_bench_08/supervisor/verify_policy_results.py
.venv/bin/python -m pytest -q research/evaluation_v3/test_evaluator_v3.py research/evaluation_v3/test_numerical_contract.py research/evaluation_v3/test_final_diagnostics.py experiments/rq1_method_bench_08/supervisor/test_assembly.py experiments/rq1_method_bench_08/supervisor/test_policy_contract.py
.venv/bin/python experiments/rq1_method_bench_08/supervisor/evaluate_assembly.py \
 --target experiments/rq1_method_bench_07/agent_a/nested_e4_fusion_seed101_predictions.npz \
 --target-key fixed50_neural_e4 --gap experiments/rq1_method_bench_08/agent_c/mb03_plain_keyed.npz \
 --seed 101 --method fixed_neural_E4_plus_MB03_I \
 --out experiments/rq1_method_bench_08/supervisor/assembly_fixed_seed101
```

New alignment reproduction, **already executed for101/202; do not rerun without need**:
`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python experiments/rq1_method_bench_08/agent_a/align.py 101` (change101→202 for replication). This experimental producer overwrites its own seed outputs; preserve the verified bundle before any independently authorized rerun.

Formal training CLI is **PLANNED, NOT IMPLEMENTED, NOT RUN**: create `experiments/rq1_train_phase_01/train.py` with `--spec experiments/rq1_method_bench_08/supervisor/TRAIN_PHASE_01_CONFIG.json --seed 101 --outer all --device cpu --threads 2 --output <fresh-run-dir>`, then repeat202/303 after integrity checks. Do not invoke this nonexistent interface as evidence or use legacy producer mains that overwrite historical artifacts. Implementation must follow sections1–7, unit-test exact contracts, then freeze its source/config hashes and pass the readiness preflight. The specification is implementation-ready; it is not a claim the new runner exists.

Required run bundle: training-spec/config and source hashes; fit/held speaker/eventIDs; tokenizer/alignment/model provenance; optimizer/checkpoints/scalers/RNGstates; innerOOF and outerkeyed scores; frozen threshold fitIDs/FPcounts/tie rule; EVALJSON/compactCSV; timing/maxRSS;GPUlease if any; exact commands/exitcodes and explicit MEASURED/VERIFIED/EXPLORATORY/INCONCLUSIVE/NOT_TESTED claims. No remote push,destructive operations or formal training was performed by MB08.
