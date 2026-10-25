# LANGFENS — RQ1-TRN-01B1-R2: Blockers, Resolutions & Next Actions Roadmap

**Gate Reference:** RQ1-TRN-01B1-R2 — Acoustic-Conditioned MDD & Final Source Provenance Gate  
**Deliverable Path:** `manifests/RQ1-TRN-01B1/reconciliation/final_acoustic_gate/BLOCKERS_AND_NEXT_ACTIONS.md`  
**Repository Branch:** `feat/rq1-trn-01b1-r2-acoustic-gate`  
**Base Commit:** `493b7ab875174c7a6dd9812e3526a50717ecb5dc`  
**Lead Speech ML Systems Engineer & Scientific Reproducibility Auditor:** Senior Speech ML Research Engineer  
**Gate Acceptance Decision:** `RQ1-TRN-01B1-R2: PASS` (All Audit Standards Met; Architectural Redesign Directive Issued)  

---

## 1. Executive Gate Decision & Acceptance Criteria Review

$$\mathbf{RQ1-TRN-01B1-R2: PASS}$$

The objective of Gate R2 was to verify source provenance and definitively establish whether the Binary MDD and Substitution Diagnosis heads depend on acoustic evidence before authorizing full Fold 1 multi-task training. All 7 predeclared acceptance standards have been met or exceeded:

### Acceptance Criteria Checklist:

| Acceptance Criterion | Verification Standard | Delivered Evidence Artifact | Gate Status |
| :--- | :--- | :--- | :---: |
| **1. Verifiable Source Provenance** | Commit `493b7ab` pushed to origin; verified bit-identically via GitHub REST API; metadata lineage reconciled. | `GITHUB_SOURCE_PROVENANCE.md` | **PASS** |
| **2. Explicit Canonical Baselines** | Empirical prior ($56.35\%$) and learned MLP ($56.35\%$) benchmarked against multi-task model ($50.79\%$). | `CANONICAL_ONLY_BASELINE_COMPARISON.md` | **PASS** |
| **3. Acoustic Sensitivity Disentanglement** | Token agreement ($93.25\%$), logit cosine sim ($0.9934$), and ASI ($+0.0035$) prove honest architectural limitation. | `DIAGNOSIS_ACOUSTIC_CONTRIBUTION.md` | **PASS** |
| **4. Bounded Strategy Ablations** | 4 bounded strategies evaluated on train/holdout probes under matched vs mismatched audio. | `AUDIO_CONDITIONED_DIAGNOSIS_ABLATIONS.md` | **PASS** |
| **5. Deployable Pipeline Separation** | Oracle accuracy ($50.79\%$) separated from cascaded E2E Recall ($16.27\%$ @ $\tau=0.50$, $44.05\%$ @ $\tau=0.20$). | `END_TO_END_MDD_EVALUATION.md` | **PASS** |
| **6. Preserved CTC Recovery & Freeze** | Greedy baseline preserved (PER $0.4844$, length ratio $0.8573$); $\beta=1.0$ exploratory erratum documented. | `CTC_PROTOCOL_AND_DECODER_FREEZE.md` | **PASS** |
| **7. Passing Regression Suite & Boundaries** | 87 passed, 0 failed; GPU determinism claims corrected; zero split leakage confirmed. | `REGRESSION_AND_REPRODUCIBILITY_REPORT.md` | **PASS** |

---

## 2. Comprehensive Blocker Accounting & Resolution Registry

| Blocker ID | Description | Root Cause / Defect | Current Status | Permanent Resolution Mechanism |
| :--- | :--- | :--- | :---: | :--- |
| **BLK-TRN-01** | Catastrophic CTC blank collapse | Backbone unfreezing with high LR under low token exposure | **RESOLVED** | Pinned frozen-backbone protocol throughout training. |
| **BLK-TRN-02** | Confounded 2x2 Factorial | Freezing policy confounded with learning rate in pilot | **RESOLVED** | $2 \times 2$ factorial grid isolating main effects and interaction. |
| **BLK-TRN-03** | Erroneous hardcoded majority baseline | Literal `0.1452` based on class `S` instead of empirical majority | **RESOLVED** | Dynamic calculation from active dataset; 29-class census. |
| **BLK-TRN-04** | CTC deletion under-recognition | Low token exposure ($L_{\text{hyp}}/L_{\text{ref}} = 0.22$) | **RESOLVED** | Scaled to $256$ utts; length ratio recovered to $0.86$; PER to $0.48$. |
| **BLK-TRN-05** | Missing GitHub commit provenance | Commit `493b7ab` not pushed to origin tracking | **RESOLVED** | Pushed to origin; verified via GitHub REST API `200 OK`. |
| **BLK-TRN-06** | Official split gating policy | Zero contamination requirement for validation and test splits | **ACTIVE GATE** | Strict zero-leakage maintained; validation and test splits untouched. |
| **BLK-TRN-07** | **Substitution Diagnosis Prompt-Prior Dominance** | Cross-attention diagnosis head learns $P(R \mid C)$ rather than acoustics | **ARCHITECTURAL BLOCKER** | **Do NOT deploy cross-attention diagnosis head.** Redesign required. |

---

## 3. Scientific Findings & The Architectural Directive (BLK-TRN-07)

### 3.1 The Diagnostic Truth
The comprehensive battery of Task B and Task C audits conclusively demonstrated:
1. **The Substitution Diagnosis Head is Blind to Acoustics:**
   - Swapping audio produces virtually zero accuracy change ($50.79\% \to 51.19\%$, $\Delta = -0.40\%$).
   - Token prediction agreement between matched and mismatched audio is **$93.25\%$**.
   - The Acoustic Sensitivity Index when the actual spoken phone changes is **$+0.003461$** ($< 0.35\%$ shift).
   - An acoustic-free learned MLP given only the canonical prompt token achieves **$56.35\%$**, beating the neural acoustic model by $+5.56\%$.
2. **The CTC Recognition Head is Authentically Acoustic:**
   - On matched audio, greedy PER is **$0.4844$** ($L_{\text{hyp}} = 1,784$).
   - On mismatched audio, PER collapses to **$0.9471$**.
   - On silence, PER collapses to **$1.0000$** ($0$ tokens emitted).

### 3.2 Binding Policy Directive for Subsequent Phases
In accordance with the prompt's instructions:
*"If diagnosis remains dominated by prompt priors, do not claim acoustic diagnosis readiness. Recommend whether to redesign that head or proceed with a separately scoped CTC/Binary training study. Stop after this gate. Full Fold 1 multi-task training remains separately authorized."*

**We issue the following formal engineering directives:**
1. **NO Claim of Acoustic Substitution Diagnosis Readiness:** Under no circumstances should the project claim that the current cross-attention architecture has achieved phoneme substitution diagnosis capability.
2. **De-coupling of Training Scope (Recommendation A):** Fold 1 multi-task training should be scoped strictly to:
   - The verified **CTC Phone Recognition Head** (acoustic realization decoder).
   - The **Binary MDD Detection Head** (with explicit prompt-prior risk baseline calibration).
3. **Architectural Redesign Roadmap (Recommendation B):**
   In a dedicated follow-up study, substitution diagnosis must be re-architected around **CTC-Guided Forced Alignment**:
   - Step 1: Use the verified CTC head to generate acoustic frame phone posteriors.
   - Step 2: Perform Viterbi alignment between the canonical prompt and the CTC frame posteriors to extract time-aligned acoustic phoneme segments.
   - Step 3: Classify the time-aligned acoustic segments into realized phonemes.
   This guarantees that the diagnosis representation is strictly acoustic and mathematically bound to the time intervals where the phone was spoken.

---

## 4. Next Actions Roadmap

1. **Local Commit & Feature Branch Creation:**  
   Stage all 8 final acoustic gate reports, JSON metrics, audit scripts, model modifications, and new regression test suite on feature branch `feat/rq1-trn-01b1-r2-acoustic-gate`.
2. **Push to Remote GitHub Origin:**  
   Push `feat/rq1-trn-01b1-r2-acoustic-gate` to `origin` and verify remote presence.
3. **Synchronize Deliverables to Google Drive (`kltn_gdrive`):**  
   Upload all 8 markdown deliverables and 3 JSON metric files to:
   `kltn_gdrive:KLTN-Dataset/manifests/RQ1-TRN-01B1/reconciliation/final_acoustic_gate/`
   and verify file presence using `rclone lsf`.
4. **Formal Gate Closure:**  
   **STOP all execution after this gate.** Full Fold 1 training, official validation calibration, and held-out test inference remain separately unauthorized pending external review of this dossier.
