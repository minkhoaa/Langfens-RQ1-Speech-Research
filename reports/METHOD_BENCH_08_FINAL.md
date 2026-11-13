# LANGFENS RQ1-METHOD-BENCH-08 — final verification

**Verdict: CONDITIONAL_GO.** Train a **modular** system: independent H-002 target neural + CTC/E4 branches with fixed50/50 S/O fusion, and the MB03 direct exact-gap insertion head. Direct H-002 is the S/O fallback. Do not adopt the new insertion aligner. Formal training was **NOT RUN**; complete the bounded runner/joint-policy preflight in the [training specification](METHOD_BENCH_08_TRAIN_PHASE_01_SPEC.md).

## 1. Scope, freeze and evidence quality

All new measurements use **TRAIN1800 utterances,12speakers,59562targets(C50356/S7381/O1825),61362exact gaps(I560)**, four official speaker-held-out folds. F0 ABA/ASI/BWC;F1 EBVS/HJK/HKK;F2 HQTV/LXC/NJS;F3 PNV/SKA/SVBI. Each base outer fit excludes3speakers and uses9; each eligible inner fit uses6 and excludes its3validation speakers and the3outer speakers. No protected/TEST_SEALED access,DEV_SEARCH tuning,remote push,destructive operations or formal unified training.

Source revision `aa4e6ea2147180a4a820d77a71911f333fcb6721`; dirty user work preserved. MB08 initially contained only `FROZEN_SPEC.md`. Supervisor adopted it, recorded an execution addendum and **122 source/report/receipt hashes at2026-11-08T11:01:34Z**, before new outer outcomes. The original freeze's earlier date is a historical assertion, not independently established. Exact implementation corrections and their pre-valid-outcome timestamps are retained. Three native `openai-codex/gpt-6-luna` agents ran concurrently in isolated lane directories; only unfinished requirements were resumed. Supervisor owns integration. See `supervisor/orchestration_receipt.json`.

Claim labels throughout: **MEASURED** actually executed; **VERIFIED** independently checked; **EXPLORATORY** descriptive/posthoc; **INCONCLUSIVE** unresolved; **NOT_TESTED** not executed. Seed replication uses the same repeatedly explored12speakers, **not an independent cohort**. Hash-indexed evidence is under `experiments/rq1_method_bench_08/`; paths below are relative to that root unless stated otherwise.

## 2. Audit findings and corrections

**VERIFIED:** authoritative manifest joins and supports, unique target/gap IDs, official speaker groups, MB03 producer ordering, MB07 inner index coverage and exclusions, and all24 reused inner CTC checkpoint hashes. Historical MB03 arrays lack embedded IDs; AgentC independently reconstructed exact producer order—not just matching fold counts—and persisted `agent_c/mb03_plain_keyed.npz` (SHA256 `8a5692e43f64b08d60252b9f6a8c63d8be7f8c5e6d6dcfa4d0e26ef7e65bb42b`). AP-I `.162124897`,P@560 `.242857` reproduced. Historical direct MLP checkpoints were not saved; source/seed/input/prediction reproduction is available, **not checkpoint-level replay**.

Material qualifications, preserved rather than silently rewriting history:

1. **MB07 FWR bug:** `realized_target_FWR` divides total warnings by utterances. Correct EVALFWR uses wrong-type-inclusive `FP_S+FP_O` (andFP_I for a complete system), not total warnings or narrower gold-C-only binary false alarms. MB08 refits thresholds on eligible inner **false-warning** prefixes and reports achieved outer rates separately.
2. **Prior slice bug:** reference is39D; MB07argmax(last42features) includes3acoustic values. Most apparent index disagreement is harmless +3renumbering, but **14/59562events** select an acoustic category. Original prior-containing results remain explicitly historical; corrected39 controls and learned no-prior ablation are separately named. Four all-zero reference rows still map to phone0 in the exploratory v2prior; do not adopt that fallback in formal training. Primary fixedneural+E4 does not use this prior.
3. **Meta-calibration limitation:** second-stage meta crossfit omits held labels at the meta layer, but some base fits generating meta-training features include meta-held speakers. Outer predictions remain outer-speaker-held-out; strict inner calibration validity is **not established**. Learned meta-policy is not primary.
4. **Stale prose hash:** current seed101fusion prediction SHA256 `c5a6bcee061a8d108ea38b9d078f702f2384c30ff635b0432ae736f484307568` matches its receipt; older prose `ac228...` is stale. No MB07 artifact changed.
5. **Historical GPU governance:** MB07 temporal insertion used CUDA without a verifiable cooperative lease receipt. Preserve its ranking result and limitation; do not pretend it was a compliant MB08 run. All new MB08 work usedCPU.
6. **Reference lineage:** reviewed features use predicted alignment and intended phones; no prohibited realized-phone/gold-timing inference dataflow found. Operator-authorized intended-phone-conditioned evaluation is supported. Raw-audio+text-only phone provenance remains **INCONCLUSIVE** upstream. Four unsupported intended tokens/zero vectors are explicitly inventoried; stronger prompt-only claims require a bounded lineage check.

## 3. A — actual new insertion alignment experiment

**MEASURED:** a trained linear acoustic emission head over continuous768D W2V2H6 frames, fixed39phone+insertion inventory,8fixedepochs/fold,AdamWlr.003/wd.0001,batch4096,insertionweight5. Fit-only frame normalization and eligible training frame labels; gold timing/realized phones supervise training only. Inference conditions on intended phones using optional monotonic states `G0,P0,G1,...,P(N-1),GN`. Fixed transition **weights**, learned emissions; forward/backward gap-visit probabilities, not rank-warped windows and not end-to-end learned transitions.40classes/30760parameters. ExactlyN+1scores and stableIDs.

First AgentA attempt was only a sketch. Its resumed implementation initially had inconsistent DPterminal normalization and compulsory edge gaps; supervisor interrupted the exact process **before any completed fold/checkpoint/outcome**. Corrected code/config froze at11:18:27Z. Agent tiny enumeration passed; supervisor independently checked **13 tiny enumeration fixtures**, posterior normalization and visit probabilities. Invalid attempt is not scientific evidence. Preserved config says“eight”fit speakers; executable and independent audit show **nine/1350utterances**, a documentation typo explicitly corrected in report.

| Seed | Scorer | AP-I pooled | P@560 | TP@560/560 | Status |
|---:|---|---:|---:|---:|---|
|101|MB03 direct baseline|.162125|.242857|136|VERIFIED reused|
|101|MB07 rank-window CNN|.012828|.019643|11|historical context|
|101|MB08 explicit-gap alignment|.094538|.189286|106|MEASURED|
|101|Same emissions,no-gap-state control|.040030|.091071|51|MEASURED structural ablation|
|101|Held-audio-frame shuffle|.009147|.014286|8|MEASURED diagnostic|
|202|MB08 explicit-gap alignment|.101253|.191071|107|MEASURED replication|
|202|Same emissions,no-gap-state control|.042691|.087500|49|MEASURED structural ablation|
|202|Held-audio-frame shuffle|.009089|.012500|7|MEASURED diagnostic|

P@560 is a ranking diagnostic, **not a calibrated warning policy**. Gap prevalence `.009126`. Explicit states beat their structurally different no-state control, and shuffling removes useful localization. Nevertheless both seeds fail the frozen MB03 adoption rule on AP-I **and** P@560. Preserve MB03. This is a negative result for this fixed-weight learned-emission model, **not all alignment methods**.

| Fold / gap support | MB03 AP-I | New101 AP-I | New202 AP-I |
|---|---:|---:|---:|
|F0 /15266,I134|.188792|.089117|.083992|
|F1 /15298,I127|.102708|.088784|.100447|
|F2 /15439,I150|.109449|.126868|.140680|
|F3 /15359,I149|.251541|.119994|.127713|

OnlyF2beatsMB03 in either seed. SpeakerAPmean/SD(sample):101 `.10695/.05820`,range `.03684–.21585`;202 `.11835/.06110`,range `.03139–.22248`. HJK/ASI/BWC remain difficult. Mean frame-state entropy across folds is approximately `.212–.231`; uncertainty is descriptive, not confidence calibration. No-gap ablation reuses identical trained emissions, not a separately optimized alignment loss. Four unsupported reference tokens fall back toAA in this experiment; disclosed limitation, not adopted for formal training.

Seed101explicit P/R@100 `.160/.02857`,P/R@500 `.198/.17679`;seed202 `.260/.04643`,`.200/.17857`. Failure examples,allfold/speaker metrics and controls are in `supervisor/insertion_failure_cases.csv`, `supervisor/insertion_recomputed.csv` and `agent_a/results_seed*.json`. Two seeds each preserve4checkpoints and61362rawpredictions; all8checkpoint hashes checked. Internal train/inference time47.46s/47.96s excludes input/supervision preparation; end-to-end wall/maxRSS not measured. No new GPU use.

## 4. B — S/O ranking and practical warning policy

**MEASURED/VERIFIED:** reused MB07 base neural/E4 predictions, fitted only new bounded meta/prior analyses and inner false-budget policies. No base-model retraining. All scores use C/S/O simplex; **E4softmax([0,SLLR,OLLR]) is not calibrated posterior**, and differs in ranking from historical rawE4LLR.

| Method |101 AP-S/O|202 AP-S/O|101/202 ECE15 |
|---|---:|---:|---:|
|DirectH002|.384080/.265655|.380168/.256080|.1059/.1098|
|HistoricalMB03PLAIN|.423209/.276706|same101reference|not policy-comparable|
|E4simplex-only|.272225/.229009|.288154/.236843|.1689/.1585|
|**Fixedneural+E4**|**.402742/.330783**|**.402074/.333998**|.0782/.0693|
|Learnedneural+E4,noprior|.415479/.310009|.408627/.310705|.0106/.0143|
|Learnedneural+correct39prior+E4,v2|.423262/.325390|.422974/.328997|.0118/.0144|
|Historicallearnedneural+legacyprior+E4|.423282/.325088|.422996/.328622|see rawtables|

Fixed/learned neural+prior controls, original versuscorrect39 versions,calibrationBrier/NLL,allfold/speakerAP are retained in `agent_b/{pooled_metrics,fold_metrics,speaker_metrics}.csv`. No prior control is silently repaired. MB03PLAIN mean-crop+39input,128/64MLP,5epochs differs fromH002 mean/std/end-start+39,256/128MLP,12epochs; the historical comparison is not a one-factor ablation.

**Pooled-versus-fold reversal:** seed101learnedv2meanfoldAP-O `.34266` exceeds fixed `.32997`, but pooled `.32539` is below fixed `.33078`. AP is non-decomposable; cross-fold score calibration/prevalence/ranking changes pooled ordering. Pooled AP and meanfold AP answer different questions; report both, not whichever favors a model.

### Frozen inner false-budget policies — target only

Confidence=max(pS,pO);type=argmaxS/O; deterministic event-ID ties. Select largest eligible inner prefix with cumulative incorrectly typed/false warnings≤floor(B×1350utterances), freeze then apply450outerutterances. Outer labels never fit thresholds. Supervisor independently reproduced **192 eligible inner thresholds** for fixed/direct, verified **960 distinct fold policies/240 pooled policies**, and recomputed all outercounts. An initial duplicate-budget reporting issue doubled pooled denominators; deduplication preserved identical policies,ratios and pre-correction artifacts. **Every final pooled row uses1800utterances,S7381/O1825.**

| Seed | Innerfalsebudget | Direct achievedFWR /Macro2F1 | Fixed achievedFWR /Macro2F1 | Fixedtotalwarnings |
|---:|---:|---:|---:|---:|
|101|.2|.0967/.0736|.1633/.1529|852|
|101|.5|.2928/.1505|.4472/.2358|1951|
|101|1|.6844/.2406|.8861/.3164|3488|
|101|2|1.7250/.3376|1.7806/.3940|6122|
|202|.2|.1406/.0841|.1950/.1486|912|
|202|.5|.3556/.1597|.4589/.2352|1956|
|202|1|.8117/.2430|.9356/.3226|3626|
|202|2|1.8522/.3365|1.9122/.3987|6461|

AtB.2 fixed101S P/R/F1 `.7388/.04024/.07632`,O `.5800/.14301/.22945`,FP294/1800. Fixed202S `.6980/.04227/.07971`,O `.5355/.13644/.21747`,FP351/1800. DirectOrecall `.0400/.03890`. All budgets/classes are available in `agent_b/policy_pooled.csv`; frozen keyed decisions,thresholds and fit provenance are preserved. These nominal comparisons have **different achieved FWR**, not an exact matched-rate superiority claim.

**EXPLORATORY common-achieved-FWR comparison:** predeclared dense inner-budget policies produce overlap approximately `.0683–5.2183`(101),`.0767–5.0967`(202) across clean methods. Interpolate already-frozen outer curves only. At achievedFWR.2, fixed versusdirect Macro2F1 is `.1656vs.1138`(101),`.1504vs.1083`(202);Orecall `.1534vs.0609`,`.1380vs.0514`. No interpolated point is a deployment threshold; no outer labels select policies. This supports, but does not establish exact deployable low-FWR dominance.

**Robustness risk:** atB.2 fixed per-speaker achievedFWR ranges `.080–.2867`(101),`.0933–.3600`(202), despite pooled≤.2. Three/four speakers respectively exceed.2. Low omission recall and speaker variability preclude safety/generalization claims. Supervisor operating breakdown: `supervisor/speaker_policy_metrics.csv`.

**Selection:** fixed50/50H002+E4 primary, directH002fallback. Learnedmeta has better S/ranking/calibration diagnostics but no strictly inner-valid policy advantage; corrected39controls and noprior ablation do not justify additional meta complexity. E4 adds credible useful omission evidence beyond directH002; S does not clearly exceed stronger historicalMB03PLAIN.

## 5. Supervisor — evaluation-only complete score assembly

Selected fixedS/O plus independently keyed **MB03 insertionseed101**, no unified training. EVALV3.1 exactIDs/topology/support verified. Targetseed202 paired with insertionseed101 is a **hybrid target-seed replication**, not all-headseed202.

| Assembly | Targetseed/Iseed | AP-S | AP-O | AP-I | Macro3 Error AP |
|---|---:|---:|---:|---:|---:|
|MB03PLAINbaseline|101/101|.423209|.276706|.162125|.287347|
|**Selectedfixedneural+E4 +MB03I**|101/101|.402742|.330783|.162125|**.298550**|
|**Selectedfixedneural+E4 +MB03I**|202/101|.402074|.333998|.162125|**.299399**|
|Learnedcorrect39v2rankingalternative +MB03I|101/101|.423262|.325390|.162125|.303592|
|Learnedcorrect39v2rankingalternative +MB03I|202/101|.422974|.328997|.162125|.304699|

The rankingalternative has higherMacro3AP; it is not selected as primary operating policy. Original historicalmeta assemblies are separately retained (`assembly_rank_seed*`). Primary assembly foldMacro3AP:101 `.300331/.221633/.335890/.337843`;202 `.310202/.221682/.334937/.324547`. Raw keyed streams,per-speaker/foldmetrics,input hashes and receipts are in `supervisor/assembly_selected_seed*` (and explicitly versioned alternatives).

**Missing, not fabricated:** complete-system classIprecision/recall/F1,Macro3ErrorF1,totalwarnings and jointFWR are **null/NOT_ESTIMABLE_NO_HELD_OUT_INSERTION_POLICY**. No eligible archived inner gap-policy scores exist forMB03; the new insertion models also lack a predeclared calibrated deployment policy. S/O classP/R/F1 above remain valid target-only. The AP-only evaluator's required gapdecision placeholder is discarded; its decision output is never presented as a measured operating result. TopK insertionprecision is not a replacement threshold.

## 6. Architecture/readiness conclusion

| Option | Evidence for | Evidence against / decision |
|---|---|---|
|Independent modular target/E4/gap|Valid baselines,independent receipts,separabletopology,CPU-feasible smallheads|ExtraE4compute; jointpolicy integration remains. **SELECTED** |
|Shared trainable multi-head|Potential representation reuse|No demonstrated insertiontransfer or calibrated full-system gain. **NOT_TESTED ablation** |
|Staged target→gap|Could reuse targetrepresentation|Crossfit/leakage complexity and errorpropagation without measured benefit. **NOT_TESTED,notprimary** |

Confidence is **moderate** for the selected TRAIN methodology, low for unseen-population safety. Valid central event mapping and held-speaker AP are reproducible; no material outer leakage was found in primaryfixedscorers. Negative insertion alone does not justifyNO_GO. FullGO would overstate current jointpolicy/runner readiness.

**CONDITIONAL_GO means:** the scientific design/baselines are sufficient to proceed to bounded implementation and preflight, then formal modular controlled training after those checks pass. It is **not unconditional permission to launch training now**, and not a deployment/safety gate. Required conditions: implement/test the fresh phase01runner; preserve exact input/unknown rules; generate eligible inner target+gap scores and jointthresholds; record CPUtime/RSS or validGPUlease; freeze sources/config before outer evaluation. Rawtext-only claims additionally require lineage certification. No additional open-ended method search is needed.

What would change the recommendation: broken key/split/prohibited-inference dataflow→NO_GO; a controlled matched-budget shared model with valid gapoperating gain could justify unifiedarchitecture later; fully reproduced runner/jointpolicy with valid provenance and feasible resources→GO. Do not retune MB08 outcomes or access protected data to settle these questions.

## 7. Final evidence and reproduction index

- [Readiness gate](METHOD_BENCH_08_READINESS_GATE.md); [self-contained training specification](METHOD_BENCH_08_TRAIN_PHASE_01_SPEC.md).
- Agent reports copied to `reports/METHOD_BENCH_08_A_INSERTION.md`, `_B_SO_POLICY.md`, `_C_READINESS.md`; fullraw/checkpoint/configevidence remains under respectiveMB08agentdirectories.
- Freeze/incident/independentverification receipts: `supervisor/{phase0_receipt,independent_input_audit,new_insertion_audit,policy_verification_receipt,orchestration_receipt}.json` and insertionrecovery/DPincidentMarkdown.
- **Tests:**33EVALV3.1syntheticcontracttests +8supervisorID/policytests passed;13independentDPenumerationfixtures passed. No formaltrainingtests are implied.
- Reproduction/auditcommands are in the training specification and scripts under `supervisor/`. FinalSHA256index: `EVIDENCE_MANIFEST_SHA256.json`. No historical frozen source/report/receipt changed.

**Stop:** MB08 verification complete; formal unified/modular training not launched. Carry only the bounded conditions and selected frozen protocol into RQ1-TRAIN-PHASE-01.
