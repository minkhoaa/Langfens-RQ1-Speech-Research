# Recomputed Protocol O vs. Protocol Z Comparative Evaluation — RQ1-TRN-02A-ZERO-ORACLE-VERIFY

**Author:** Senior Speech ML Evaluation & Reproducibility Auditor  
**Date:** 2026-10-26  
**Subject:** Formal comparative evaluation of Oracle-Canonical (Protocol O) vs Zero-Oracle (Protocol Z) across explicitly declared scoring populations  

---

## 1. Executive Summary

This report establishes the verified empirical performance of the multi-task Wav2Vec2 MDD model under **Protocol Z (Zero-Oracle Text+Audio Inference)** in direct comparison to **Protocol O (Oracle TextGrid Interval Baseline)**.

### Principal Findings:
1. **Faithful Reproduction of Historical Audit Claim:**
   The reported exploratory Protocol Z validation AP of **0.312467** reproduces within **0.000004** ($0.312471$) under the audited validation prevalence imputation policy.
2. **Defensible Train-Only Scoring Policies:**
   When unaligned gold tokens are scored using a principled, predeclared train-only conditional prior $P_{\text{train}}(\text{error} \mid g_j)$ (Policy B1), Protocol Z achieves an AP of **0.314265** across the full 9,956 gold tokens—dropping by only **0.28 percentage points** (-0.002756) relative to Protocol O (0.317021).
3. **Aligned-Only Population Performance:**
   Restricted to the 9,919 phone tokens where G2P and Gold annotations align ($99.63\%$ coverage), Protocol Z achieves **AP = 0.313530** vs. Protocol O **AP = 0.316434** ($\Delta = -0.002904$).
4. **Exact Sequence Utterances:**
   Across the 82 validation sentences where CMUdict G2P matches annotator TextGrids bit-for-bit (2,338 tokens), Protocol Z and Protocol O APs are **mathematically identical** ($0.305362$, $\Delta = 0.000000$).
5. **Deployment Boundary Caveat:**
   A small $\Delta \text{AP}$ ($< 0.30$ points) does **not** imply immediate production readiness. The model was trained exclusively on oracle TextGrid interval queries. Complete transition to a zero-oracle deployment pipeline requires retraining or fine-tuning with G2P canonical inputs.

---

## 2. Quantitative Comparative Evaluation Across Populations

| Population / Policy | Evaluated Tokens ($N$) | Ground Truth Positives | Protocol O (Oracle TextGrid) AP | Protocol Z (CMUdict G2P) AP | Absolute $\Delta \text{AP}$ (Z $-$ O) | Optimal F1 ($\tau_{\text{opt}}$) | F1 at $\tau=0.50$ |
|---|---:|---:|---:|---:|---:|---|---|
| **Population 1: Aligned-Only (Zero Imputation)** | 9,919 (99.63%) | 1,145 | **0.316434** | **0.313530** | **-0.002904** | 0.3712 ($\tau=0.20$) | 0.2656 |
| **Population 2: Full Gold Benchmark ($N=9,956$)** | | | | | | | |
| ├── **Policy B1: Train Prior Lookup** | 9,956 (100.0%) | 1,157 | **0.317021** | **0.314265** | **-0.002756** | 0.3737 ($\tau=0.20$) | 0.2636 |
| ├── **Policy B2: Train Error Prevalence** | 9,956 (100.0%) | 1,157 | **0.317021** | **0.313006** | **-0.004015** | 0.3698 ($\tau=0.20$) | 0.2636 |
| ├── **Policy B3: Abstention Penalty (1.0)** | 9,956 (100.0%) | 1,157 | **0.317021** | **0.307811** | **-0.009210** | 0.3729 ($\tau=0.20$) | 0.2724 |
| └── **Audit Reference: Val Prevalence (0.116211)**| 9,956 (100.0%) | 1,157 | **0.317021** | **0.312471** | **-0.004550** | 0.3698 ($\tau=0.20$) | 0.2636 |
| **Population 3: Exact Sequence Utterances** | 2,338 (23.48%) | 259 | **0.305362** | **0.305362** | **+0.000000** | 0.3639 ($\tau=0.20$) | 0.2636 |

---

## 3. Detailed Operating Point Breakdown (Threshold $\tau = 0.50$ and $\tau = 0.20$)

### Standard Operating Point ($\tau = 0.50$):
- **Protocol O Full Benchmark ($N = 9,956$):**
  $\text{TP} = 201, \; \text{FP} = 208, \; \text{FN} = 956, \; \text{TN} = 8,591$  
  $\text{Precision} = 0.4914, \; \text{Recall} = 0.1737, \; \text{F1} = 0.2567$
- **Protocol Z Policy B1 ($N = 9,956$):**
  $\text{TP} = 208, \; \text{FP} = 213, \; \text{FN} = 949, \; \text{TN} = 8,586$  
  $\text{Precision} = 0.4941, \; \text{Recall} = 0.1798, \; \text{F1} = 0.2636$
- **Protocol Z Aligned-Only ($N = 9,919$):**
  $\text{TP} = 208, \; \text{FP} = 213, \; \text{FN} = 937, \; \text{TN} = 8,561$  
  $\text{Precision} = 0.4941, \; \text{Recall} = 0.1817, \; \text{F1} = 0.2656$

### Optimal F1 Operating Point ($\tau = 0.20$):
- **Protocol O Full Benchmark ($N = 9,956$):**
  $\text{TP} = 615, \; \text{FP} = 1,507, \; \text{FN} = 542, \; \text{TN} = 7,292$  
  $\text{Precision} = 0.2898, \; \text{Recall} = 0.5315, \; \text{F1} = 0.3751$
- **Protocol Z Policy B1 ($N = 9,956$):**
  $\text{TP} = 614, \; \text{FP} = 1,515, \; \text{FN} = 543, \; \text{TN} = 7,284$  
  $\text{Precision} = 0.2884, \; \text{Recall} = 0.5307, \; \text{F1} = 0.3737$

---

## 4. Per-Speaker Disaggregated Comparison

| Speaker | Metric Dimension | Protocol O Benchmark | Protocol Z Aligned-Only | Protocol Z Policy B1 | Delta (B1 vs O) |
|---|---|---:|---:|---:|---:|
| **NJS** | Total Gold Tokens | 5,011 | 4,998 (99.74%) | 5,011 (100.0%) | 0 |
| (Spanish L1) | Binary Errors | 641 (12.79%) | 636 (12.72%) | 641 (12.79%) | 0 |
| | **Binary AP** | **0.394927** | **0.390735** | **0.389965** | **-0.004962** |
| | F1 at $\tau=0.50$ | 0.3204 | 0.3235 | 0.3209 | +0.0005 |
| **RRBI** | Total Gold Tokens | 4,945 | 4,921 (99.51%) | 4,945 (100.0%) | 0 |
| (Russian L1) | Binary Errors | 516 (10.43%) | 509 (10.34%) | 516 (10.43%) | 0 |
| | **Binary AP** | **0.251519** | **0.247107** | **0.249739** | **-0.001780** |
| | F1 at $\tau=0.50$ | 0.1746 | 0.1905 | 0.1887 | +0.0141 |

### Observations:
1. Both speakers show slight, consistent AP decreases under Protocol Z (-0.50 AP points for NJS, -0.18 AP points for RRBI).
2. The larger drop on NJS occurs because NJS Gold TextGrids contain more non-standard vocalic transcriptions (206 substitutions vs 137 on RRBI).
3. In all conditions, NJS remains significantly easier to detect ($AP \approx 0.39$) than RRBI ($AP \approx 0.25$), mirroring native language transfer patterns.

---

## 5. Methodological Evaluation of Scoring Policies

### Primary Replay Metric — Population 1 (Aligned-Only Protocol Z AP):
- Evaluates strictly on aligned positions where the G2P frontend hypothesized a phone and a gold reference counterpart exists ($N = 9,919$, $99.63\%$ gold coverage).
- Zero score imputation; reflects genuine model prediction capability on the queried phonemes.
- Yields **AP = 0.313530** (vs Protocol O aligned counterpart $0.316434$, $\Delta = -0.002904$).
- Must always be reported alongside reported gold coverage ($99.63\%$) and omission count ($37$ unaligned gold tokens) to ensure full population transparency.

### Benchmark Analysis — Policy B1 (Train Prior Lookup Full-Gold Imputation):
- Evaluates across all 9,956 gold tokens by imputing omitted gold positions with conditional train priors $P_{\text{train}}(\text{error} \mid g_j)$ derived from Fold 1 training items ($N=79,426$).
- **Crucial Interpretation:** Policy B1 is an **oracle-referenced full-gold imputation analysis** designed solely for offline comparative benchmarking against Protocol O. It is **NOT** a deployment-time operating policy, because determining which gold tokens were omitted and looking up their priors requires the gold reference TextGrid, which is unavailable at deployment.
- Yields **AP = 0.314265** ($\Delta = -0.002756$ relative to Protocol O $0.317021$).

### Sensitivity & Audit Analyses (Policies B2, B3, and Validation Prevalence):
- **Policy B2 (Train Overall Prevalence Imputation):** Imputes unaligned gold tokens with the static train error rate ($0.149082$), yielding **AP = 0.313006**.
- **Policy B3 (Abstention Penalty):** Stress-tests the pipeline by assigning maximum error confidence ($1.0$) to G2P omissions, yielding **AP = 0.307811**.
- **Audit Reference (Validation Prevalence Imputation):** Re-evaluates the historical exploratory report ($0.116211$), reproducing **AP = 0.312471** (reproducing $0.312467$ to within $4 \times 10^{-6}$).
- **Status:** Policies B2, B3, and historical validation prevalence are preserved strictly as sensitivity/audit reconciliation evidence, not production operating policies.
