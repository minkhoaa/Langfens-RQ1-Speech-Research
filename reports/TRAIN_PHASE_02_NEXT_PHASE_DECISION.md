# LANGFENS RQ1-TRAIN-PHASE-02 — Next Phase Decision & Scientific Disposition

**Campaign:** LANGFENS RQ1-TRAIN-PHASE-02  
**Standard:** EVAL-V3.1  
**Decision Gate:** Phase-02 Scientific Adoption & Downstream Handoff  
**Date:** November 2026  
**Final Scientific Verdict:** **`PARTIAL_ADOPTION`**  

---

## 1. Formal Decision & Disposition

Based on rigorous outer cross-validation across three registered seeds (101, 202, 303) and four official speaker-grouped folds under EVAL-V3.1, the scientific committee adopts a **`PARTIAL_ADOPTION`** verdict:

```
+-----------------------------------------------------------------------------+
|                            PHASE-02 ADOPTION MATRIX                         |
+------------------------------+--------------------+-------------------------+
| Component / Dimension        | Adoption Verdict   | Scientific Action       |
+------------------------------+--------------------+-------------------------+
| Target Acoustic Trunk        | RETAIN_PHASE_01    | Keep H-002 Neural Head  |
| Acoustic CTC Sequence Branch | RETAIN_PHASE_01    | Keep E4 Native Kernel   |
| Exact-Gap Insertion Head     | RETAIN_PHASE_01    | Keep MB03 1614D Head    |
| Evidence Integration         | RETAIN_PHASE_01    | Keep Fixed 50/50 Linear |
| Decision Policy Engine       | ADOPT_PHASE_02     | Replace Joint Argmax    |
|                              |                    | with Decoupled Policy   |
+------------------------------+--------------------+-------------------------+
```

### 1.1 What Is Formally Adopted
- **Decoupled Class-Sensitive Operating Policy (`arm2_decoupled_fixed_50`):**
  - Replaces the frozen Phase-01 joint argmax cutoff rule.
  - Partitions target false warning budget equally ($w_S = 0.50$) on inner OOF, establishing separate thresholds $\tau_S$ and $\tau_O$, with relative standardized margin arbitration $\arg\max \{ P(S)/\tau_S, P(O)/\tau_O \}$ for dual-positive events.
  - **Measured Benefits:**
    1. **Substitution Recall:** Increases from **0.0356 ± 0.0028** to **0.0440 ± 0.0039 (+23.6% relative gain)** at nominal $B=0.2$.
    2. **Substitution F1:** Increases from **0.0680** to **0.0805 (+18.4% relative gain)**.
    3. **Wrong-Type False Warnings:** Decreases from **38.7 ± 8.2** down to **34.3 ± 4.5 (-11.4% reduction)**.
    4. **Budget Adherence:** Empirical FWR remains strictly controlled at **0.2078 ± 0.0045** (1.04x nominal budget).
    5. **Matched-FWR Operating Superiority:** Decoupled policy Pareto-dominates the baseline across all common achieved FWR operating points ($\text{FWR} \in [0.16, 0.28]$).

### 1.2 What Is Formally Retained from Phase-01
- **Acoustic Trunk:** Standalone H-002 target head (`Linear(2343, 256) -> 128 -> 3`) and standalone CTC/E4 sequence branch (`Linear(768, 43)`).
- **Exact-Gap Insertion Head:** MB03 boundary crop MLP (`1614 -> 128 -> 64 -> 2`).
- **S/O Evidence Fusion:** Fixed 50/50 linear blend between neural output probabilities and softmaxed CTC sequence log-likelihood ratios.

### 1.3 What Is Formally Rejected
- **Sequence CTC Insertion Scoring (ILLR):** Rejected due to catastrophic temporal smearing (AP-I collapses from 0.1586 to 0.0285).
- **Expanded Boundary Flux Feature Vectors:** Rejected due to dimensional explosion (3150D vs 426 positives) causing severe overfitting (AP-I collapses to 0.1147).
- **Learned Dynamic Evidence Gating:** Rejected because instance-level nonlinear gating scrambles ranking monotonicity (Macro-3 AP collapses from 0.3050 to 0.2762).
- **Unsupervised Test-Time Utterance Centering:** Rejected because subtracting utterance mean confidence suppresses true errors in dense error utterances (Macro-3 F1 collapses from 0.1604 to 0.1366).

---

## 2. Quantitative Rationale for `PARTIAL_ADOPTION`

A verdict of full `ADOPT_PHASE_02` would imply that the four registered Phase-02 hypotheses were confirmed. In reality:
- H-P2-01 was partially successful (improved Substitution recall by +23.6%, reduced wrong-type warnings by 11.4%), but failed the preregistered acceptance bar ($\ge 30\%$ wrong-type reduction, $\ge +0.015$ Macro-3 F1 gain).
- H-P2-02, H-P2-03, and H-P2-04 were all definitively **FALSIFIED** by negative empirical results.
- Conversely, a verdict of `RETAIN_PHASE_01` would discard a verified, zero-cost policy enhancement that demonstrably increases useful error recall and lowers cross-type misclassifications at identical realized FWR.

`PARTIAL_ADOPTION` accurately reflects scientific integrity: adopting the verified policy advancement while preserving the robust Phase-01 acoustic champion and acknowledging the falsification of architectural hypotheses.

---

## 3. Forward Research Roadmap (Phase-03 Priorities)

Based on the forensic failure analysis conducted by Agents A, B, and C, three priorities are identified for subsequent research cycles:

### Priority 1: Supervised Speaker Calibration / Enrollment Audio
- **Diagnosis:** Unsupervised test-time score centering failed because it lacks a reference baseline. Inter-speaker FWR disparity (0.0844 to 0.3222) is driven by acoustic idiosyncrasies (e.g. Speaker ABA's epenthetic stop bursts, Speaker HJK's unreleased stop lenition) that cannot be deduced from single unlabeled utterances.
- **Proposal:** Test a supervised calibration protocol using 3–5 phonetically balanced enrollment utterances per speaker to establish speaker-specific score priors without oracle test labels.

### Priority 2: Student Distillation & Latency Compression
- **Diagnosis:** The dual-engine architecture (H-002 neural MLP + CTC native batch sequence kernel) requires dual feature pipelines and C++ sequence scoring.
- **Proposal:** Distill the Phase-02 champion ensemble (Fixed 50/50 fusion + MB03 gap head + Decoupled Policy) into a unified single-pass student network for on-device deployment under strict latency and memory budgets.

### Priority 3: Forced-Alignment Gap Epenthesis Localizer
- **Diagnosis:** Sequence CTC forward scoring lacks temporal resolution for insertions because it integrates over all possible alignments.
- **Proposal:** Explore gap-bounded CTC forward-backward scoring that restricts insertion lattice paths strictly to the frame interval between phoneme $j-1$ offset and phoneme $j$ onset.
