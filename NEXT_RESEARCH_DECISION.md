# LANGFENS RQ1 — Next Research Decision & Retraining Experiment Specification (Coordinator)

**Document ID:** `LANGFENS-RQ1-NEXT-RESEARCH-DECISION-02`  
**Author:** Senior ML Research Engineer, Phonetics Researcher & Reproducibility Auditor (Coordinator)  
**Experiment ID:** `LANGFENS-RQ1-PRECISION-RECOVERY-01`  
**Date:** 2026-11-04  
**Status:** **AUTHORIZATION PACKET PREPARED — TRAINING ON HOLD PENDING OPERATOR SIGN-OFF**  
**Policy Authority:** Repository `AGENTS.md` (Strict Remote GPU Execution Boundaries; Zero Local Training Invariant)  

---

## 1. Definitive Four-Outcome Scientific Decision Matrix

Following comprehensive empirical evaluation across all four independent specialist workstreams (Agents A, B, C, D) and Pareto frontier analysis, the Coordinator formally resolves the four prospective research outcomes:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               FOUR-OUTCOME SCIENTIFIC DECISION MATRIX                          │
├───────────────────────────────┬────────────┬───────────────────────────────────────────────────┤
│ Outcome Candidate             │ Verdict    │ Definitive Empirical Evidence                     │
├───────────────────────────────┼────────────┼───────────────────────────────────────────────────┤
│ Outcome A: Decision Policy    │ REJECTED   │ Precision >= 30% attainable only at 2.07% recall; │
│ Sufficient                    │ (NO-GO)    │ Selective muting {D,T,R} destroys 97.6% of TPs.   │
├───────────────────────────────┼────────────┼───────────────────────────────────────────────────┤
│ Outcome B: Acoustic Verifier  │ REJECTED   │ Verifier is strictly Pareto-dominated by CAL 10%; │
│ Beneficial Standalone         │ (NO-GO)    │ Precision drops (6.78% -> 6.31%); destroys 25 TPs.│
├───────────────────────────────┼────────────┼───────────────────────────────────────────────────┤
│ Outcome C: Annotation         │ ADOPTED    │ 23.61% of alarms are unreleased stops marked      │
│ Reliability is Major Blocker  │ (GO - REQ) │ "correct"; broad event precision is 31.62% [95%CI]│
├───────────────────────────────┼────────────┼───────────────────────────────────────────────────┤
│ Outcome D: Representation     │ ADOPTED    │ 67.90% of alarms are genuine model over-pred;     │
│ Insufficiency (Retraining)    │ (GO - RET) │ Internal deletion prior must be retrained on GPU. │
└───────────────────────────────┴────────────┴───────────────────────────────────────────────────┘
```

### Synthesis of the Combined Decision (Outcome C + Outcome D)
1. **Outcome A is decisively rejected:** Post-hoc thresholding cannot shift a concave ROC curve outward. Raising thresholds merely trades massive amounts of true omission recall ($-6.5$ to $-7.8\text{ pp}$ recall per $+1\text{ pp}$ precision) for trivial precision gains.
2. **Outcome B is decisively rejected as a standalone solution:** Standalone heuristic waveform verification suffers from acoustic coarticulation bleeding on true omissions, destroying $30.5\%$ of genuine detections and dropping deployed precision to $6.31\%$.
3. **Outcome C is adopted as a mandatory evaluation requirement:** All future evaluations must report the dual evaluation horizon: Strict Original Gold Ground Truth alongside the Horvitz-Thompson Adjudicated Pronunciation Event Horizon. Models must not be penalized for detecting genuine unreleased reductions.
4. **Outcome D is adopted as the primary scientific path:** Because $67.9\%$ of false alarms ($820.9$ alarms) are confirmed model errors driven by an uncalibrated deletion prior on coronal phonemes, the model's internal representations must be remediated through **Targeted Supervised Retraining on authorized Remote GPU hardware**.

---

## 2. Specification of the Next Experiment: RQ1-TRN-04A

The Coordinator prospectively specifies the exact architecture, loss formulation, experimental controls, and acceptance gates for the next investigation: **RQ1-TRN-04A (Class-Conditioned Deletion-Calibrated MDD)**.

### 2.1 Observed Failure Mode Addressed
The Subtype Supervision (`T-OP`) model exhibits an uncalibrated, exaggerated deletion prior that triggers omission alarms on:
- **$89.55\%$ of all canonical `/r/` tokens** (generating 362 false alarms), mistaking non-native alveolar taps [ɾ] for deletions.
- **$70.25\%$ of all canonical `/d/` and `/t/` tokens** (generating 691 false alarms), mistaking unreleased closures for deletions.

This failure occurs because the current MDD head shares a single scalar deletion logit across all 39 phonemes without phoneme-identity conditioning or frame-level acoustic energy anchoring.

### 2.2 Proposed Change in Representation, Supervision & Objective
To resolve this without altering the frozen Wav2Vec2 backbone:
1. **Phoneme-Class Conditioned Loss Formulation (Class-Focal Loss):**
   Replace the uniform cross-entropy operation loss $\mathcal{L}_{\text{op}}$ with a phoneme-conditioned focal loss:
   $$\mathcal{L}_{\text{cond}} = -\sum_{i=1}^N \alpha_{p_i, y_i} (1 - P(y_i \mid \mathbf{x}_i))^{\gamma} \log P(y_i \mid \mathbf{x}_i)$$
   where the class weighting factor $\alpha_{p_i, \text{del}}$ dynamically down-weights deletion penalties on coronal stops and rhotics ($p_i \in \{/d/, /t/, /r/\}$) during training:
   $$\alpha_{p_i, \text{del}} = \begin{cases} 0.25 & \text{if } p_i \in \{/d/, /t/, /r/\} \\ 1.00 & \text{otherwise} \end{cases}$$
2. **Acoustic Energy Residual Projection:**
   Concatenate frame-level log-mel energy and spectral centroid directly into the pooling projection before the operation classification head, giving the neural classifier direct physical awareness of release bursts and voicing bars.

### 2.3 Matched Baseline & Unchanged Experimental Conditions
To guarantee strict causal attribution:
- **Pretrained Backbone:** Pinned `facebook/wav2vec2-base-960h` (revision `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`), strictly frozen during evaluation.
- **Split & Partitions:** Identical Fold-1 split manifest (`manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`, SHA-256 `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`).
- **Data Partitions:** Train on 16 authorized training speakers ($2,399$ utterances); calibrate on NCC/THV ($300$ utterances); validate on NJS/RRBI ($300$ utterances).
- **Optimization Horizon:** Fixed-budget $3,600$ updates, batch size 2, gradient accumulation 2 (effective batch size 4), AdamW ($\text{lr} = 3 \times 10^{-5}$), cosine scheduler with 100 warmup updates, random seed `1337`.
- **Numerical Precision:** Strictly CUDA FP16 AMP on remote hardware; CPU FP32 deterministic replay.

### 2.4 Training Data & Label-Quality Requirements
- Training audio staged from immutable archive: `datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip` (SHA-256 `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`).
- Gold label schema: strict three-way operations ($0=\text{correct}$, $1=\text{substitution}$, $2=\text{deletion}$) adhering to `EVALUATION_CONTRACT_V1.md`.
- **Zero test partition access:** Test speakers `ABA, ASI, EBVS, HJK, LXC, PNV` remain completely sealed.

### 2.5 Remote GPU Execution Platform & Compute Budget
In strict compliance with `AGENTS.md`:
- **Local Laptop GPU/CPU training is strictly PROHIBITED.**
- **Authorized Execution Platform:** Kaggle GPU (official CLI, Nvidia Tesla T4 × 2, private notebook) OR Google Colab GPU (Tesla T4).
- **Compute Budget:**
  - Training updates: $3,600$ maximum updates ($\approx 6$ epochs).
  - Estimated execution duration: $2.8\text{ hours}$ on Tesla T4.
  - Quota impact: $\approx 2.8\text{ GPU hours}$ out of 30-hour weekly quota ($<10\%$).
- **Atomic Checkpointing & Durability:** Checkpoints (`last.pt`, `best.pt`, `fixed_budget_3600.pt`) mirrored atomically to private persistent cloud storage with SHA-256 readback verification before tearing down the VM runtime.

---

## 3. Prospective Acceptance Gates & Explicit Rejection Criteria

To prevent publication bias and protect against metric regression, the retrained model will be evaluated against strict, pre-declared quantitative gates:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             PROSPECTIVE EVALUATION GATES (RQ1-TRN-04A)                         │
├───────────────────────────────────────┬─────────────┬─────────────┬────────────────────────────┤
│ Metric Channel                        │ Baseline    │ Gate Target │ Rationale / Constraint     │
├───────────────────────────────────────┼─────────────┼─────────────┼────────────────────────────┤
│ Strict Deployed Omission Precision    │ 6.78%       │ >= 25.00%   │ Primary recovery goal      │
│ All-Gold Omission Recall              │ 56.55%      │ >= 40.00%   │ Preserves detection power  │
│ False Alarms per 1,000 Phones         │ 113.20      │ <= 30.00    │ Student feedback hygiene   │
│ Binary MDD Average Precision (AP)     │ 0.318619    │ >= 0.308619 │ Non-inferiority margin .01 │
│ Legacy Substitution AP                │ 0.308974    │ >= 0.298974 │ Non-inferiority margin .01 │
│ Greedy Phoneme Error Rate (PER)       │ 0.241980    │ <= 0.251980 │ Upstream ASR non-inferior  │
│ Broad Pronunciation Event Precision   │ 31.62%      │ >= 50.00%   │ Adjudicated horizon target │
└───────────────────────────────────────┴─────────────┴─────────────┴────────────────────────────┘
```

### Explicit Rejection Criteria (Causal Model Termination)
The retraining hypothesis will be declared **REJECTED** and the experiment terminated if ANY of the following occur:
1. **Precision Failure:** Deployed omission precision fails to reach $20.0\%$ at recall $\ge 35.0\%$ on Fold-1 VAL under Protocol Z.
2. **Binary AP Regression:** Binary MDD AP drops below $0.308619$ (exceeding the declared $0.010$ non-inferiority margin).
3. **CTC Blank Collapse:** CTC greedy PER degrades beyond $0.251980$ or blank frame fraction exceeds $0.99$.
4. **False Alarm Concentration Invariant:** Coronal stops (`/d/`, `/t/`) and rhotics (`/r/`) continue to account for $>85\%$ of all false alarms, demonstrating that focal re-weighting failed to alter the deletion prior.

---

## 4. Operational Boundaries & Handoff Directive

- **Current Status:** Phase 1 Investigation and Empirical Recovery Experiment 01 is **COMPLETE**.
- **Execution Freeze:** No training scripts may be executed locally. No remote kernels may be pushed until the operator reviews this authorization packet and grants explicit permission.
- **Handoff:** Deliver all 10 verified deliverables, update `docs/process-checkpoints/INDEX.md`, and await operator authorization for Phase 2 remote deployment.
