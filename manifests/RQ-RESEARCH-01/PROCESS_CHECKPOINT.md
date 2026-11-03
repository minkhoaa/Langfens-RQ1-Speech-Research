# RQ-RESEARCH-01 Process Checkpoint: Parallel Research & Thesis-Aligned Method Selection

**Document ID:** `manifests/RQ-RESEARCH-01/PROCESS_CHECKPOINT.md`  
**Phase:** `RQ-RESEARCH-01`  
**Status:** `RESEARCH_COMPLETED; AWAITING_OPERATOR_DECISION`  
**Execution Platform:** Local CPU (Research, literature audit, and subagent coordination only; 0 GPU training updates, 0 backward passes, 0 Kaggle jobs)  
**Execution Source Commit:** `baa4989243d5ff46b8b3bf5a3c280cdf2857382b` / branch `research-phase`  
**Date:** 2026-11-03  
**Policy Authority:** `AGENTS.md` (Mandatory Process Checkpoint Policy & Research Boundaries)  
**Authoritative Scope Authority:** UIT Graduation Thesis Proposal `KLTN0127` (Nguyễn Bá Trọng Khôi & Từ Minh Khoa; Advisor: ThS. Trần Thị Hồng Yến)  

---

## 1. Executive Summary & Verification Disposition

This process checkpoint formally records the execution and completion of the **RQ-RESEARCH-01 Parallel Research & Thesis-Aligned Method Selection Campaign**.

Six independent Pi subagents were spawned concurrently in background execution under workflow ID `06c9aabf-1eb7-490d-91aa-6b18f579aad3` (Mission `8522611a-18d4-4d52-b4c9-7dc4698d38ab`), completing exhaustive empirical and theoretical audits across the six designated research families:
- **Lane A:** Goodness of Pronunciation (GOP), CTC Posterior Scoring, and Segmentation-Free Deletion Scoring (`manifests/RQ-RESEARCH-01/LANE_A_CTC_GOP.md`, 47,992 B).
- **Lane B:** Direct Phoneme Recognition (Free-Phone ASR) and Acoustic Realization Modeling (`manifests/RQ-RESEARCH-01/LANE_B_ACOUSTIC_REALIZATION.md`, 48,541 B).
- **Lane C:** Deletion-Specific and Supervised Learning Objectives (`manifests/RQ-RESEARCH-01/LANE_C_SUPERVISED_OBJECTIVES.md`, 34,278 B).
- **Lane D:** Monotonic, Probabilistic, and Edit-Aware Alignment Mechanisms (`manifests/RQ-RESEARCH-01/LANE_D_ALIGNMENT_MECHANISMS.md`, 35,577 B).
- **Lane E:** Joint Acoustic-Recognition and Edit-Aware Modeling (`manifests/RQ-RESEARCH-01/LANE_E_JOINT_MODELING.md`, 46,083 B).
- **Lane F:** Retrieval, Phonological, and Hybrid Alternatives (`manifests/RQ-RESEARCH-01/LANE_F_RETRIEVAL_PHONOLOGICAL.md`, 40,236 B).
- **Integrated Synthesis:** Master Decision Report (`manifests/RQ-RESEARCH-01/INTEGRATED_DECISION_REPORT.md`, 35,355 B).

**Total Generated Research Evidence:** 7 master reports totaling **288,062 bytes (~281 KB)** of peer-reviewed analysis, mathematical formulations, codebase seam mappings, and pre-registered falsifiable pilot specifications.

---

## 2. Invariant & Boundary Verification

| Policy Invariant | Constraint Specification | Verification Evidence | Status |
|---|---|---|:---:|
| **Zero GPU Training** | No backward passes, optimizer steps, or GPU execution | CPU research pass only; 0 updates | **PASS** |
| **Sealed Test Isolation** | Active test partitions ($S_{\text{test}}^{(k)}$) strictly unread | Checked manifests and subagent logs; 0 test access | **PASS** |
| **Baseline R Freeze** | Baseline R weights and validation metrics unmodified | Baseline R remains frozen at `best.pt` (SHA-256 `a2445c8d...`) | **PASS** |
| **E1 Scientific Gate** | E1 skip-token failure acknowledged; no Folds 2–4 rollout | Documented in `manifests/RQ-MODEL-01/PROCESS_CHECKPOINT.md` | **PASS** |
| **Thesis Scope Demarcation**| Strict boundary between RQ1 (Speaking/MDD) and RQ2 (Multi-Agent) | Confirmed; no LLM/multi-agent code in acoustic seam | **PASS** |
| **Zero ASR Scope Drift** | No open-domain foundation ASR training | Rejection of Whisper / unconstrained foundation models | **PASS** |

---

## 3. Method Ranking Summary & Gate Status

All candidates evaluated against the pre-registered 5-criterion rubric (Academic Evidence 25%, Omission Potential 25%, Langfens Compatibility 20%, Feasibility & Compute 15%, Corrective Feedback 15%) and the authoritative UIT Thesis Gate:

| Rank | Candidate Method | Weighted Score | Thesis Gate | Recommendation |
|:---:|---|:---:|:---:|:---:|
| **#1** | **E.1: Edit-Aware Sequence Tagging (GECToR-Speech)** | **93.20** | **PASS (GO)** | **Top Shortlist Candidate** |
| **#2** | **D.1: CTC-Viterbi Segmental Pooling ("CTC-Align")** | **91.55** | **PASS (GO)** | **Top Shortlist Candidate (CPU Pilot)** |
| **#3** | **A.2: Dual-Path Hybrid MDD (CTC-DP + Cross-Attn)** | **90.40** | **PASS (GO)** | **Top Shortlist Candidate (Ensemble)** |
| **#4** | **D.2: CTC-Supervised Attention Alignment Loss** | **87.20** | **PASS (GO)** | Reserve Candidate |
| **#5** | **C.1: 3-Class Supervised Objective (Focal/Weights)** | **86.90** | **PASS (GO)** | Reserve Candidate |
| **#6** | **F.2: Phonological Attribute Modeling (SCTC-SB)** | **83.60** | **PASS (GO)** | Tier-3 Feedback Complement |
| **#7** | **A.1: Pure CTC Free-Phone Recognition + DP** | **83.70** | **PASS (GO)** | Baseline Control |
| — | **F.1: Retrieval-Based MDD (Tu et al. ICASSP 2026)** | 48.75 | **NO-GO** | REJECTED (104% PER; blind to omissions) |
| — | **D.3: Monotonic Multihead Attention (MoChA)** | 59.75 | **NO-GO** | REJECTED (Recurrent latency; no skip state) |
| — | **E.2: Full Conformer-AED (RefTextLAS)** | 57.00 | **NO-GO** | REJECTED (>30k steps; ASR error-erasure) |
| — | **E.3: Neural Transducer (Conformer-RNN-T)** | 51.75 | **NO-GO** | REJECTED (VRAM OOM hazard; training diverges) |
| — | **A.3 / F.3: External Forced Alignment (Kaldi/MFA)**| 53.75 | **NO-GO** | REJECTED (Breaks on omissions; heavy runtime) |
| — | **B.1: Whisper-based ASR MDD** | 47.25 | **NO-GO** | REJECTED (Autoregressive error-erasure; >2.5s) |

---

## 4. Shortlisted Pilot Experiment Recommendations

The coordinator recommends a phased, compute-optimal 2-stage verification plan:

1. **Phase 1: Pilot D-1 (Zero-GPU-Compute Local CPU Audit):**
   - Execute standalone Python evaluation (`audit_scripts/diagnose_ctc_viterbi_omissions.py`) on Fold 1 validation set (300 utterances, 9,956 phonemes) using existing Baseline R weights.
   - Verify whether CTC Viterbi duration $\Delta t_i$ and mean blank posterior $P_{\text{blank}}$ achieve Omission AP $\ge 0.1200$.
   - **Cost: 0 GPU hours, ~3 minutes CPU execution.**

2. **Phase 2: Pilot E-1 (Remote Kaggle GPU Pilot — Pending Operator Authorization):**
   - Implement `LangfensEditTaggingMDD` (41-class edit prediction: $\$KEEP, \$DELETE, \$REPLACE_{1..39}$) in `src/langfens_prep/model.py`.
   - Stage kernel to Kaggle CLI with `--no-run`.
   - Execute exactly 3,600 updates (~2.5h on Tesla T4) under the pre-registered contract.
   - Acceptance target: Omission AP $\ge 0.1500$, Omission Recall @ 15% FPR $\ge 0.5500$, Binary AP $\ge 0.3170$.

---

## 5. Artifact Ledger

| Deliverable Artifact | File Location | File Size (Bytes) | Verification Status |
|---|---|---|:---:|
| Lane A Report | `manifests/RQ-RESEARCH-01/LANE_A_CTC_GOP.md` | 47,992 | Verified |
| Lane B Report | `manifests/RQ-RESEARCH-01/LANE_B_ACOUSTIC_REALIZATION.md` | 48,541 | Verified |
| Lane C Report | `manifests/RQ-RESEARCH-01/LANE_C_SUPERVISED_OBJECTIVES.md` | 34,278 | Verified |
| Lane D Report | `manifests/RQ-RESEARCH-01/LANE_D_ALIGNMENT_MECHANISMS.md` | 35,577 | Verified |
| Lane E Report | `manifests/RQ-RESEARCH-01/LANE_E_JOINT_MODELING.md` | 46,083 | Verified |
| Lane F Report | `manifests/RQ-RESEARCH-01/LANE_F_RETRIEVAL_PHONOLOGICAL.md` | 40,236 | Verified |
| Integrated Decision Report | `manifests/RQ-RESEARCH-01/INTEGRATED_DECISION_REPORT.md` | 35,355 | Verified |
| Process Checkpoint | `manifests/RQ-RESEARCH-01/PROCESS_CHECKPOINT.md` | ~8,000 | Verified |

---

## 6. Restart & Safe Handoff Instructions

1. **Next Authorized State:** Research campaign is COMPLETE. All subagent outputs are merged and verified.
2. **Current Stop Boundary:** Do NOT begin implementation, code edits, or remote Kaggle GPU launches until the operator reviews the shortlist and grants explicit authorization for Pilot D-1 or Pilot E-1.

---

## 7. Append-only coordinated closure correction — 2026-11-03

**Current documentary disposition: CLOSURE_COMPLETE_WITH_ACCOUNTED_SOURCE_LIMITATIONS; NO_ARCHITECTURE_SELECTED; EXPERIMENTS_NOT_AUTHORIZED.** This is continuation of **RQ-RESEARCH-01**, not a new phase. The historical narrative/ranking above and the separate revalidation session remain preserved, but their unsupported winner, causal, resource, scope and independence claims are qualified/superseded by the [coordinated closure report](closure-2026-11-03/INTEGRATED_DECISION_REPORT.md) and [correction ledger](closure-2026-11-03/CROSS_SESSION_CORRECTION_LEDGER.md).

Six fresh-context agents launched concurrently in native workflow `723092bc-f51f-4a2a-8cd9-bf63b7c93d0b` (launch span 0.193s; all six execution intervals overlapped). Six substantive independently authored reports were collected, read and preserved unchanged, including a newly reconstructed Lane E. C successfully wrote its complete report before a usage-limit terminal failure; the **one** native retained recovery `8372f239-79f3-42c6-9b4f-54867e8137e0` confirmed unchanged bytes and completed. Do not claim six clean original terminal successes. A discloses incidental ranking exposure through required checkpoint reads; reports are separate analyses, not blind replications or independent experiments. Historical B/E cross-reads also qualify the original blanket independence claim.

Authoritative original thesis, pinned source/label/checkpoint/evaluator contracts, archived E0/E1 and DIAG-02B, and primary literature were reconciled. E1 remains NO-GO for cross-fold rollout; DIAG-02B parity remains PARTIAL. GECToR-Speech is an unvalidated transfer hypothesis, canonical alignment/blank/DELETE supervision do not prove acoustic absence, and the old universal 1.5×blank gate is retired as infeasible on documented controls. Segmentation-free scoring with relevant controls and one compact natural-subtype/realization direction are independent pilot options, not a winning architecture or P0-before-supervised dependency.

**Accounted source gaps:** no distinct latest omission proposal found in enumerated project documentary scopes; Joint-Align full primary text blocked HTTP420; several code/model licenses/provenances unresolved. No conjectural substitution or claim to have reviewed missing contents. The original thesis was located and fully reviewed. Completion here means documentary reconciliation under these explicit limits, **not** empirical RQ1/feedback completion or scientific PASS.

[Closure checkpoint/restart](closure-2026-11-03/PROCESS_CHECKPOINT.md), [verification evidence](closure-2026-11-03/VERIFICATION_EVIDENCE.json), [parallel pilot designs](closure-2026-11-03/PARALLEL_PILOT_SPECIFICATIONS.md). Outer base remains `b25a7595360294798d8d06b37592dd65dd4c9c33` on `research-phase`; pre-existing nested mode/symlink dirt preserved. Current process: no inference/training/GPU/deployment/TEST access; **0 successful updates, 0 skipped updates, no new model artifact**. E0/E1/datasets/evaluators unchanged. STOP after documentation; obtain separate operator authorization for any implementation, fitting, inference, training, staging/deployment or TEST evaluation. No commit/push performed.
