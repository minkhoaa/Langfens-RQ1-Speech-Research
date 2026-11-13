# LANGFENS RQ1-TRAIN-PHASE-01 — Next Research Agenda (Phase-02)

**Role:** Senior Speech ML Research Engineer & Scientific Reproducibility Auditor  
**Scope:** Forward research directions arising from the formal controlled Phase-01 training campaign  
**Status:** **PROPOSED HYPOTHESES & EXPERIMENTAL ROADMAP**  
**Contract:** EVAL-V3.1  

---

## 1. Scientific State at Phase-01 Closure

Formal controlled training of the frozen modular architecture has established the first leakage-safe, fully audited benchmark across three registered seeds (101, 202, 303) and four official speaker-grouped outer folds.

### Key Measured Achievements:
1. **Macro-3 Error AP:** Fixed 50/50 fusion achieves **0.3042 ± 0.0034**, outperforming both direct standalone neural H-002 (**0.2717 ± 0.0022**) and the historical MB08 assembly reference (**0.298550**).
2. **Complementarity Confirmed:** Neural H-002 excels at phonetic substitution ranking (AP-S **0.4098** vs **0.2839** for E4), while the CTC sequence likelihood ratio breaks neural ambiguity on deletions, driving AP-O from **0.2681** to **0.3441** (+28.3% relative).
3. **Leakage-Safe Joint Policy Derivation:** Deriving joint operating thresholds exclusively on inner out-of-fold partitions yielded an empirical joint false warning rate of **0.2043 ± 0.0082** at nominal budget $B=0.2$ false warnings per utterance (a 1.02x adherence ratio).

---

## 2. Core Scientific Bottlenecks Identified

Despite these gains, rigorous forensic inspection of the 2,238 formal artifacts revealed four primary failure modes that constrain pedagogical effectiveness:

### Bottleneck A: Wrong-Type S↔O False Warnings
- **Observation:** At operating budget $B=0.2$, **38.7 ± 8.2 warnings (10.5% of all false warnings)** in the Fixed fusion system are not false alarms on correct speech, but *type misclassifications* (a true substitution mislabeled as an omission, or vice versa). In the standalone E4 model, wrong-type errors account for **20.3% of false warnings** (99.0 out of 487.3).
- **Mechanism:** The current policy applies an argmax rule over non-correct probabilities (`argmax(prob_S, prob_O)`) after an utterance-level confidence cutoff is passed. Because omission and substitution have vastly different prior prevalences (7,381 substitutions vs 1,825 omissions in TRAIN), symmetric probability thresholds cause severe cross-talk.
- **Pedagogical Impact:** Under EVAL-V3.1, mislabeling a student's sound substitution as an omission is penalized identically to alerting on correct speech.

### Bottleneck B: High Inter-Speaker Disparity in False Warning Rate
- **Observation:** While the global average FWR matches the nominal 0.20 budget, the achieved per-speaker FWR ranges from **0.0844** (Speaker SVBI, Hindi L1) to **0.3222** (Speaker ABA, Arabic L1)—a 3.8x spread.
- **Mechanism:** Inner OOF threshold calibration sets a uniform global cutoff across all training speakers. It does not calibrate for baseline acoustic variance across native language backgrounds (L1 accents). Speakers whose accent naturally diverges more from General American (GA) acoustic references trigger disproportionately more false alerts.
- **Pedagogical Impact:** A system that over-warns certain learner demographics by 4x is pedagogically discriminatory and unreliable for automated scoring.

### Bottleneck C: Exact-Gap Insertion Plateaus in Acoustic Isolation
- **Observation:** Standalone exact-gap insertion AP plateaus at **0.1586 ± 0.0048** (P@560 = **0.2476 ± 0.0072**), showing minimal variance across seeds and no interaction with the main acoustic trunk.
- **Mechanism:** The MB03 direct exact-gap head relies solely on static boundary frame averages (left crop `[11:16]` and right crop `[1:6]`). It has zero access to the continuous CTC sequence alignment or acoustic duration lattices. An epenthetic vowel often lasts only 20–40 ms; static frame pooling smooths out these rapid formant transients.

### Bottleneck D: Rigid 50/50 Heuristic Fusion
- **Observation:** The equal weighting (`0.5 * neural + 0.5 * e4`) is a fixed global constant.
- **Mechanism:** When the local neural model is extremely confident (e.g. $p_S > 0.95$), incorporating an uncertain E4 likelihood can dilute the score and induce false alarms. Conversely, when local acoustic context is corrupted by coarticulation, global CTC sequence likelihood should dominate.

---

## 3. Prioritized Hypotheses for Phase-02 Research

We formulate four falsifiable hypotheses for Phase-02, strictly maintaining the zero-leakage, speaker-grouped cross-validation contract:

### Hypothesis H-P2-01: Decoupled Asymmetric Cost-Sensitive Thresholding
- **Formal Statement:** Replacing the joint argmax decision rule with independent, cost-sensitive inner OOF thresholding for $S$ and $O$ (calibrating separate thresholds $\tau_S$ and $\tau_O$ to satisfy their respective false-warning budgets) will reduce wrong-type false warnings by $\ge 30\%$ and improve Macro-3 Error F1 at $B=0.2$ by $\ge +0.015$.
- **Falsification Condition:** Wrong-type false warnings do not decrease by at least 15%, or Macro-3 Error F1 decreases on outer-fold evaluation.

### Hypothesis H-P2-02: L1-Stratified Adaptive Warning Calibration
- **Formal Statement:** Calibrating warning thresholds conditionally on speaker L1 group (using inner OOF L1 clusters) will reduce inter-speaker FWR variance by $\ge 40\%$ (maximum speaker FWR $\le 0.25$ at nominal $B=0.2$) without reducing pooled Macro-3 F1.
- **Falsification Condition:** Inter-speaker FWR range exceeds 0.28, or pooled Macro-3 F1 drops by $>0.005$.

### Hypothesis H-P2-03: CTC Alignment-Aware Epenthesis Detection
- **Formal Statement:** Providing the exact-gap insertion head with CTC blank-duration features and temporal transition flux across the boundary will increase exact-gap AP-I from $0.1586$ to $\ge 0.1850$, and improve P@560 from $0.2476$ to $\ge 0.2800$.
- **Falsification Condition:** AP-I on outer folds fails to exceed 0.1650.

### Hypothesis H-P2-04: Confidence-Gated Dynamic Evidence Fusion
- **Formal Statement:** A learned gating network (trained on inner OOF features) that blends neural probabilities and E4 sequence likelihoods as $\alpha(x) \cdot P_{\text{neural}} + (1 - \alpha(x)) \cdot P_{\text{E4}}$ will increase Macro-3 Error AP from $0.3042$ to $\ge 0.3150$.
- **Falsification Condition:** Macro-3 Error AP fails to improve by at least +0.005 over the fixed 50/50 baseline.

---

## 4. Execution Governance & Hard Guardrails

All subsequent research must adhere to the following governance rules:
1. **No Tuning on Exploration Splits:** Hypotheses must be pre-registered before fitting. No hyperparameters may be tuned on outer evaluation splits.
2. **Protected Split Isolation:** The `TEST_SEALED` and `DEV_SEARCH` partitions remain strictly locked. They may not be accessed until the Phase-02 model architecture and calibration rules are frozen in an immutable gate report.
3. **Budget and Resource Parity:** Training must remain strictly bounded under cumulative time allowances on standardized hardware with full checkpoint archival.
