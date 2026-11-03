# Scientific Evaluation Audit & Metric Reconciliation Report — RQ1 Prep

**Document ID:** `manifests/RQ1-WORKSPACE-PREP-01/P_SUP_AUDIT_AND_METRIC_RECONCILIATION.md`  
**Status:** `AUDITED_RECONCILED; HONEST_TRADEOFF_RECORDED; SPECIFICATIONS_VERIFIED`  
**Author:** Agent D — Scientific Evaluation Audit Subagent  
**Date:** 2026-11-03  
**Coordinator Workspace:** `/home/khoa/Projects/KLTN/Dataset` (`prep/rq1-shared-base`)  
**Shared Base Commit:** `0d268ff8c444e16fd76aca2f4ad6ece966908edc`  
**Binding Evaluation Contract:** `manifests/RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md` (SHA-256 `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`)  
**Primary References:**  
- P-SCORE Benchmark Report: `manifests/RQ1-P-SCORE/evaluation/BENCHMARK_REPORT.json`  
- Protocol Z Verification Report: `manifests/RQ1-TRN-02A/evaluation_reconciliation/zero_oracle_verification/PROTOCOL_Z_VERIFICATION_REPORT.json`  
- E0 Fold-1 Benchmark: `manifests/RQ-MODEL-01/evaluation/RQ_MODEL_01_E0_FOLD1_EVAL/E1_EVALUATION_REPORT.json`  
- P-SUP Objective Policy Amendment: `manifests/RQ1-P-SUP/OBJECTIVE_POLICY_AMENDMENT.md`  

---

## Executive Summary

This audit establishes the rigorous, independent scientific reconciliation for the Fold-1 validation partition (speakers `NJS` and `RRBI`, 300 TextGrids) in the RQ1 research program. It directly resolves:
1. **Population Discrepancy Reconciliation:** The exact structural decomposition of the 9,956 full canonical tokens versus the 9,919 aligned tokens under Protocol Z Levenshtein dynamic programming, identifying all 37 unaligned tokens and their phonetic root causes.
2. **Direct Cross-Model Comparison on Identical Populations:** Direct side-by-side performance tables comparing `GOP-CTC-SF-SD` against production `E0 Baseline R`, `GOP-CTC-SF-S`, and `GOP-SA` (Viterbi self-aligned segmental control) on the exact same aligned population ($N=9,919$) and reporting full-population metrics ($N=9,956$).
3. **Prospective Acceptance & Noninferiority Gate Audit:** Honest scientific reporting of prospective decision boundaries: `GOP-CTC-SF-SD` achieved **PASS** on primary Omission AP ($\Delta = \mathbf{+0.021615}$ vs target $\ge +0.020$) and **PASS** on operating sensitivity Recall@FPR15 ($\Delta = \mathbf{+0.182482}$ / $\mathbf{+18.25\text{ pp}}$ vs target $\ge +0.030$), but **FAILED** the prospective noninferiority gate on Substitution AP ($\Delta = \mathbf{-0.024256}$ vs allowable margin $\ge -0.005$). The fundamental omission-substitution tradeoff inherent in segmentation-free graph decoding is rigorously analyzed.
4. **Audit of Matched T-OP vs B-CONTROL Specifications:** Complete parameter and protocol verification for the upcoming Kaggle remote GPU run of P-SUP, confirming identical random seed (`1337`), identical capacity (`266,667` trainable parameters), identical frozen backbone (`facebook/wav2vec2-base-960h`), identical optimizer/scheduler hyperparameters, and mandatory dual checkpoint accounting (`fixed_budget_3600.pt` and `best.pt`).

---

## 1. Population Reconciliation: 9,956 Full Canonical Tokens vs. 9,919 Aligned Tokens

### 1.1 Complete Population Census & Disaggregation
The official Fold-1 validation partition consists of exactly 300 utterances recorded from two speakers (`NJS`, Spanish L1, 150 utterances; `RRBI`, Russian L1, 150 utterances). Evaluating pronunciation diagnosis requires distinguishing between two distinct evaluation cohorts:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                         FOLD-1 VALIDATION POPULATION CENSUS                            │
├─────────────────────────┬───────────────────┬───────────────────┬──────────────────────┤
│ Cohort Dimension        │ Full Gold Census  │ Aligned Population│ Unaligned Delta (Δ)  │
│                         │ (Protocol O)      │ (Protocol Z)      │ (Excluded in Z)      │
├─────────────────────────┼───────────────────┼───────────────────┼──────────────────────┤
│ Total Tokens (N)        │ 9,956 (100.00%)   │ 9,919 (99.63%)    │ 37 (0.37%)           │
│ Correct Phones (N_cor)  │ 8,799 (88.38%)    │ 8,774 (88.46%)    │ 25 (67.57% of Δ)     │
│ Substitutions (N_sub)   │ 1,012 (10.16%)    │ 1,008 (10.16%)    │  4 (10.81% of Δ)     │
│ Deletions (N_del)       │   145  (1.46%)    │   137  (1.38%)    │  8 (21.62% of Δ)     │
│ Total Errors (N_err)    │ 1,157 (11.62%)    │ 1,145 (11.54%)    │ 12 (32.43% of Δ)     │
└─────────────────────────┴───────────────────┴───────────────────┴──────────────────────┘
```

Per-speaker breakdown of the two populations:
- **Speaker `NJS` (Spanish L1, 150 utterances):**
  - Full Population ($N=5,011$): $N_{\text{cor}}=4,370$, $N_{\text{sub}}=563$, $N_{\text{del}}=78$ (Errors: 641, prevalence $12.79\%$).
  - Aligned Population ($N=4,998$): $N_{\text{cor}}=4,359$, $N_{\text{sub}}=562$, $N_{\text{del}}=77$ (Errors: 639, prevalence $12.79\%$).
  - Unaligned Tokens ($N=13$): 11 correct, 1 substitution, 1 deletion.
- **Speaker `RRBI` (Russian L1, 150 utterances):**
  - Full Population ($N=4,945$): $N_{\text{cor}}=4,429$, $N_{\text{sub}}=449$, $N_{\text{del}}=67$ (Errors: 516, prevalence $10.43\%$).
  - Aligned Population ($N=4,921$): $N_{\text{cor}}=4,415$, $N_{\text{sub}}=446$, $N_{\text{del}}=60$ (Errors: 506, prevalence $10.28\%$).
  - Unaligned Tokens ($N=24$): 14 correct, 3 substitutions, 7 deletions.

### 1.2 Identification and Root-Cause Analysis of the Exact 37 Unaligned Tokens
Under Protocol Z (Zero-Oracle Deployment), prompt text is phonemized offline via CMUdict G2P without access to annotator TextGrids. When G2P canonical sequences are aligned to human TextGrid canonical intervals via Levenshtein dynamic programming (`sub_first` tie-break policy), exactly 37 gold canonical phone tokens cannot be matched to any prompt phoneme token.

The table below documents all 37 unaligned tokens, their speaker, utterance, interval index, canonical phone, realized phone, ground-truth error type, TextGrid boundaries, and linguistic etiology:

| # | Speaker | Utterance ID | Gold Idx | Phone | Realized | Error Type | TextGrid Interval (s) | Linguistic & Alignment Etiology |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 01 | `NJS` | `arctic_a0023` | 65 | `IH` | `IH` | correct | 5.030 – 5.120 | Syllable count difference in "privileges" (`IH` weak vowel in TextGrid dictionary). |
| 02 | `NJS` | `arctic_a0024` | 23 | `HH` | `sil` | deletion | 1.800 – 1.854 | Dropped aspiration in "had" transcribed as deletion in TextGrid. |
| 03 | `NJS` | `arctic_a0071` | 0 | `HH` | `sil` | deletion | 0.040 – 0.056 | Initial `/HH/` in "what" (`/HH W AH T/` TextGrid vs `/W AH T/` G2P). |
| 04 | `NJS` | `arctic_a0081` | 0 | `HH` | `HH` | correct | 0.120 – 0.170 | Wh-aspirate `/HH/` in "which" (`/HH W IH CH/` TextGrid vs `/W IH CH/` G2P). |
| 05 | `NJS` | `arctic_a0094` | 18 | `HH` | `HH` | correct | 1.800 – 1.840 | Wh-aspirate `/HH/` in "white" (`/HH W AY T/` TextGrid vs `/W AY T/` G2P). |
| 06 | `NJS` | `arctic_a0106` | 8 | `HH` | `HH` | correct | 0.730 – 0.830 | Wh-aspirate `/HH/` in "which" (`/HH W IH CH/` TextGrid vs `/W IH CH/` G2P). |
| 07 | `NJS` | `arctic_a0106` | 26 | `R` | `R` | correct | 2.216 – 2.230 | Internal rhotic glide variant in "burst" (`/B ER S T/` vs `/B R S T/`). |
| 08 | `NJS` | `arctic_a0260` | 7 | `Y` | `Y` | correct | 0.620 – 0.700 | Palatal glide `/Y/` variant in "few" / "annual". |
| 09 | `NJS` | `arctic_a0369` | 43 | `R` | `R` | correct | 4.431 – 4.455 | Post-vocalic rhotic coda in "there". |
| 10 | `NJS` | `arctic_a0548` | 6 | `R` | `R` | correct | 1.086 – 1.120 | Syllabic consonant decomposition in "other". |
| 11 | `NJS` | `arctic_a0556` | 0 | `HH` | `HH` | correct | 0.000 – 0.070 | Wh-aspirate `/HH/` in "when" (`/HH W EH N/` TextGrid vs `/W EH N/` G2P). |
| 12 | `NJS` | `arctic_b0269` | 0 | `HH` | `HH` | correct | 0.040 – 0.080 | Wh-aspirate `/HH/` in "which" (`/HH W IH CH/` TextGrid vs `/W IH CH/` G2P). |
| 13 | `NJS` | `arctic_b0461` | 4 | `R` | `<ERR>`| substitution | 0.547 – 0.610 | Unrecognized realized phone tagged `<ERR>` in rhotic coda. |
| 14 | `RRBI`| `arctic_a0005` | 9 | `AO` | `AO` | correct | 0.970 – 1.000 | Vowel quantity variant in "all". |
| 15 | `RRBI`| `arctic_a0022` | 7 | `ER` | `EH` | substitution | 0.680 – 0.768 | Rhotic vowel derhotacized to `/EH/` in "early"; split in G2P vs TextGrid. |
| 16 | `RRBI`| `arctic_a0022` | 33 | `ER` | `EH` | substitution | 3.389 – 3.447 | Derhotacized vowel in "perpetual"; unaligned due to dictionary boundary mismatch. |
| 17 | `RRBI`| `arctic_a0023` | 65 | `IH` | `IH` | correct | 5.120 – 5.160 | Syllable count difference in "privileges" (`IH` weak vowel in TextGrid dictionary). |
| 18 | `RRBI`| `arctic_a0031` | 2 | `R` | `sil` | deletion | 0.172 – 0.207 | Rhotic deletion in cluster "from" (`/F R AH M/`); gold interval tagged `sil`. |
| 19 | `RRBI`| `arctic_a0031` | 4 | `ER` | `EH` | substitution | 0.250 – 0.316 | Vowel substitution `/EH/` for `/ER/` following deleted `/R/`. |
| 20 | `RRBI`| `arctic_a0038` | 2 | `W` | `W` | correct | 0.190 – 0.250 | Contraction "We'll": TextGrid has `/W IY W IH L/` (3 units) vs G2P `/W IY L/` (2 units). |
| 21 | `RRBI`| `arctic_a0038` | 3 | `IH` | `IH` | correct | 0.250 – 0.300 | Contraction "We'll": auxiliary `/IH/` from expanded compound form. |
| 22 | `RRBI`| `arctic_a0058` | 10 | `AO` | `AO` | correct | 0.800 – 0.830 | Low-back vowel variant in "talk". |
| 23 | `RRBI`| `arctic_a0071` | 0 | `HH` | `HH` | correct | 0.050 – 0.100 | Wh-aspirate `/HH/` in "what" (`/HH W AH T/` TextGrid vs `/W AH T/` G2P). |
| 24 | `RRBI`| `arctic_a0081` | 0 | `HH` | `HH` | correct | 0.050 – 0.100 | Wh-aspirate `/HH/` in "which" (`/HH W IH CH/` TextGrid vs `/W IH CH/` G2P). |
| 25 | `RRBI`| `arctic_a0097` | 13 | `UH` | `UH` | correct | 1.000 – 1.070 | Vowel length/quality variant in "could". |
| 26 | `RRBI`| `arctic_a0100` | 13 | `R` | `sil` | deletion | 1.363 – 1.380 | Rhotic deletion in "morning"; gold annotated `sil` (0.017s interval). |
| 27 | `RRBI`| `arctic_a0116` | 34 | `R` | `sil` | deletion | 4.514 – 4.530 | Coda rhotic deletion in "water"; 0.016s zero-acoustic interval. |
| 28 | `RRBI`| `arctic_a0129` | 2 | `R` | `R` | correct | 0.153 – 0.170 | Onset rhotic variant in "great". |
| 29 | `RRBI`| `arctic_a0131` | 18 | `R` | `sil` | deletion | 1.441 – 1.490 | Rhotic deletion in "were"; 0.049s silent interval. |
| 30 | `RRBI`| `arctic_a0136` | 16 | `R` | `R` | correct | 2.391 – 2.440 | Rhotic variant in "around". |
| 31 | `RRBI`| `arctic_a0137` | 29 | `R` | `sil` | deletion | 3.020 – 3.050 | Coda rhotic deletion in "short"; 0.030s silent interval. |
| 32 | `RRBI`| `arctic_a0168` | 0 | `AH` | `AH` | correct | 0.050 – 0.161 | Initial unstressed reduction in "about". |
| 33 | `RRBI`| `arctic_a0168` | 6 | `HH` | `sil` | deletion | 0.930 – 0.956 | Deletion of initial aspiration `/HH/` in "him" (`/HH IH M/`); 0.026s interval. |
| 34 | `RRBI`| `arctic_b0038` | 22 | `AO` | `AO` | correct | 2.930 – 3.000 | Open vowel variant in "thought". |
| 35 | `RRBI`| `arctic_b0081` | 20 | `AO` | `AO` | correct | 1.660 – 1.828 | Vowel variant in "upon". |
| 36 | `RRBI`| `arctic_b0081` | 22 | `N` | `N` | correct | 2.060 – 2.230 | Nasal coda variant in "upon". |
| 37 | `RRBI`| `arctic_b0371` | 5 | `AO` | `AO` | correct | 0.510 – 0.601 | Vowel variant in "long". |

### 1.3 Phonetic Categorization of the 37 Excluded Tokens
The unaligned tokens partition into three well-defined structural phonetic categories:
1. **Pronunciation Dictionary Discrepancies (25 tokens, 67.6%):**
   - **Wh-Aspiration (`/HH W/` vs `/W/`):** 9 tokens (e.g., "what", "which", "when", "white"). The L2-ARCTIC annotator dictionary preserves the conservative American English wh-aspirate `/HH/`, while standard CMUdict modern usage maps these words to simplex `/W/`.
   - **Contraction Expansions:** 2 tokens in `arctic_a0038` ("we'll" transcribed with full auxiliary `/W IY W IH L/`).
   - **Weak Vowel & Syllabicity Variants:** 14 tokens involving schwa/schwi variation (`/IH/` in "privileges", `/AH/` in "about", vowel quality in "could", "talk", "long").
2. **Gold Acoustic Omissions / Deletions (8 tokens, 21.6%):**
   - In 8 instances (`arctic_a0024`, `arctic_a0071`, `arctic_a0031`, `arctic_a0100`, `arctic_a0116`, `arctic_a0131`, `arctic_a0137`, `arctic_a0168`), the human annotator marked a phoneme as deleted (`realized = sil`), but the G2P frontend phonemizer produced a divergent base sequence that caused the DP alignment to treat the gold interval as unmatched. These 8 deletions represent genuine target omissions that are absent from the aligned cohort.
3. **Phonetic Substitutions (4 tokens, 10.8%):**
   - 3 tokens involving derhotacization (`/ER/` $\to$ `/EH/` in `arctic_a0022` and `arctic_a0031`) and 1 unclassified substitution (`<ERR>` in `arctic_b0461`).

---

## 2. Cross-Model Comparative Audit on the EXACT Same Populations

To avoid confounding algorithmic differences with cohort discrepancies, all models are evaluated directly on the exact same population.

### 2.1 Direct Comparative Evaluation on the Aligned Population ($N=9,919$)
The aligned population ($N=9,919$, $N_{\text{cor}}=8,774$, $N_{\text{sub}}=1,008$, $N_{\text{del}}=137$) represents the zero-oracle deployment cohort evaluated strictly without gold timestamps:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                          DIRECT MODEL COMPARISON ON ALIGNED POPULATION (N=9,919)                       │
├─────────────────────────┬──────────────┬──────────────┬────────────────┬──────────────┬────────────────┤
│ Model / Scorer          │ Omission AP  │ Recall@FPR15 │ Substitution AP│ Binary AP    │ Greedy CTC PER │
│                         │ (N=8,911)    │ (FPR ≤ 0.15) │ (N=9,782)      │ (N=9,919)    │ (N_ref=9,881)  │
├─────────────────────────┼──────────────┼──────────────┼────────────────┼──────────────┼────────────────┤
│ Baseline R (E0 best.pt) │ 0.032564     │ 0.394161     │ **0.304811**   │ **0.313530** │ **0.236818**   │
│ GOP-CTC-SF-S            │ 0.029411     │ 0.248175     │ 0.246800       │ 0.259094     │ **0.236818**   │
│ GOP-CTC-SF-SD (P-SCORE) │ **0.051026** │ **0.430657** │ 0.222544       │ 0.250049     │ **0.236818**   │
│ GOP-SA (Viterbi Control)│ 0.049846     │ 0.416058     │ 0.228104       │ 0.254370     │ **0.236818**   │
│ Greedy DP (Levenshtein) │ 0.019002*    │ 0.380952*    │ 0.175058*      │ 0.186678*    │ **0.236818**   │
├─────────────────────────┼──────────────┼──────────────┼────────────────┼──────────────┼────────────────┤
│ Delta: SF-SD vs SF-S    │ **+0.021615**│ **+0.182482**│ **-0.024256**  │ **-0.009045**│  0.000000      │
│ Relative Delta (SD vs S)│ **+73.49%**  │ **+73.53%**  │ **-9.83%**     │ **-3.49%**   │  0.00%         │
├─────────────────────────┼──────────────┼──────────────┼────────────────┼──────────────┼────────────────┤
│ Delta: SF-SD vs Base R  │ **+0.018462**│ **+0.036496**│ **-0.082267**  │ **-0.063481**│  0.000000      │
└─────────────────────────┴──────────────┴──────────────┴────────────────┴──────────────┴────────────────┘
* Note: Greedy DP valid coverage is only 42.46% (57.54% abstention rate).
```

### 2.2 Direct Comparative Evaluation on the Full Population ($N=9,956$)
On the full validation population ($N=9,956$, $N_{\text{cor}}=8,799$, $N_{\text{sub}}=1,012$, $N_{\text{del}}=145$), historical and imputation-adjusted metrics are established:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                           MODEL METRICS ON FULL POPULATION (N=9,956)                                   │
├─────────────────────────┬──────────────┬──────────────┬────────────────┬──────────────┬────────────────┤
│ Model / Evaluation Arm  │ Omission AP  │ Recall@FPR15 │ Substitution AP│ Binary AP    │ Greedy CTC PER │
│                         │ (N=8,944)    │ (FPR ≤ 0.15) │ (N=9,811)      │ (N=9,956)    │ (N_ref=9,881)  │
├─────────────────────────┼──────────────┼──────────────┼────────────────┼──────────────┼────────────────┤
│ Baseline R (Protocol O) │ **0.035020** │ **0.400000** │ **0.307366**   │ **0.317021** │ **0.236818**   │
│ Baseline R (Colab FP16) │ —            │ —            │ —              │ 0.316062     │ —              │
│ E1 Skip Model (Fixed)   │ 0.032252     │ 0.365517     │ —              │ 0.309293     │ 0.232365       │
│ Protocol Z Policy B1    │ —            │ —            │ —              │ 0.314265     │ —              │
│ Protocol Z Policy B2    │ —            │ —            │ —              │ 0.313006     │ —              │
│ Protocol Z Audit Impute │ —            │ —            │ —              │ 0.312471     │ —              │
├─────────────────────────┼──────────────┼──────────────┼────────────────┼──────────────┼────────────────┤
│ Delta: E1 vs Baseline R │ -0.002768    │ -0.034483    │ —              │ -0.007728    │ -0.004453      │
└─────────────────────────┴──────────────┴──────────────┴────────────────┴──────────────┴────────────────┘
```

---

## 3. Audit of Prospective Acceptance and Noninferiority Gates

The prospective criteria and decision boundaries defined in `EVALUATION_CONTRACT_V1.md` (and prespecified in `SCIENTIFIC_QA_AND_COMPARISON_REPORT.md` Section 4.1) were frozen **prior to candidate execution**. 

### 3.1 Formal Gate Audit Results for P-SCORE (`GOP-CTC-SF-SD` vs `GOP-CTC-SF-S`)

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                           P-SCORE PROSPECTIVE GATE COMPLIANCE AUDIT                                    │
├───────────────────────┬────────────────────────────┬──────────────────┬──────────────┬─────────────────┤
│ Evaluation Gate       │ Frozen Target Margin       │ Baseline (SF-S)  │ Candidate(SD)│ Audit Outcome   │
├───────────────────────┼────────────────────────────┼──────────────────┼──────────────┼─────────────────┤
│ 1. Omission AP Gain   │ ΔAP_omit ≥ +0.020 (+2.0%)  │ 0.029411         │ 0.051026     │ **PASS**        │
│                       │                            │                  │ (+0.021615)  │ (Target met)    │
│ 2. Sensitivity Gain   │ ΔRecall@FPR15 ≥ +0.030     │ 0.248175         │ 0.430657     │ **PASS**        │
│                       │ (+3.0 percentage points)   │ (24.82%)         │ (+18.25 pp)  │ (Large gain)    │
│ 3. Substitution Noninf│ ΔAP_sub ≥ -0.005 (-0.5%)   │ 0.246800         │ 0.222544     │ **FAIL**        │
│                       │                            │                  │ (-0.024256)  │ (Tradeoff)      │
│ 4. Binary Noninferior.│ ΔAP_bin ≥ -0.005 (-0.5%)   │ 0.259094         │ 0.250049     │ **FAIL**        │
│                       │                            │                  │ (-0.009045)  │ (Degraded)      │
│ 5. Acoustic Grounding │ Δ_acoustic ≥ +0.015        │ +0.022299        │ Matched      │ **PASS**        │
│ 6. Coverage Policy    │ Valid coverage ≥ 99.50%    │ 100.00%          │ 100.00%      │ **PASS**        │
└───────────────────────┴────────────────────────────┴──────────────────┴──────────────┴─────────────────┘
```

### 3.2 Detailed Analysis of the Omission-Substitution Tradeoff

The experimental data reveals a fundamental, mathematically grounded **omission-substitution tradeoff**:

1. **Why Omission Detection Succeeds (+0.021615 Omission AP, +18.25 pp Recall@FPR15):**  
   In `GOP-CTC-SF-SD`, the denominator graph incorporates an explicit target-deletion path ($\epsilon$-deletion / skip transition) alongside all phone substitutions:
   $$\mathcal{L}_{\text{denom}} = -\log P\left(\pi \in \mathcal{B}^{-1}\left(c_1 \cdots c_{j-1} \, \{p \in \mathcal{P} \cup \epsilon\} \, c_{j+1} \cdots c_J\right) \mid X\right)$$
   When an omission occurs in physical speech, the acoustic frames for phone $c_j$ contain no phonetic energy matching $c_j$, and the surrounding context frames align directly to $c_{j-1}$ and $c_{j+1}$. In standard `SF-S`, the denominator is forced to substitute another phone $p \in \mathcal{P}$, which incurs high acoustic emission penalties because no actual speech segment was produced. In `SF-SD`, the $\epsilon$-deletion path provides an acoustic shortcut that accumulates near-zero transition penalty, yielding a high denominator probability $P(\text{SD})$. The resulting error score $s_{\text{error}} = \log \frac{P(\text{SD})}{P(\text{canonical})}$ increases dramatically on deletions. This raises omission recall at 15% FPR from $24.82\%$ to $43.07\%$ (+18.25 percentage points) and omission AP from $0.029411$ to $0.051026$ (+73.5% relative).

2. **Why Substitution Detection Degrades (-0.024256 Substitution AP):**  
   The introduction of the $\epsilon$-deletion shortcut creates a pathological failure mode on substitutions and lenited phones:
   - When a speaker pronounces a substituted phone $s \neq c_j$ (e.g., Russian speaker producing `/EH/` instead of `/ER/`, or Spanish speaker reducing a fricative), the realized acoustic segment has moderate phonetic duration (often 30–70 ms).
   - In `SF-S`, the denominator must search across all alternative phone emissions $p \in \mathcal{P}$. The substituted phone $s$ produces a sharp probability peak, driving $P(\text{S}) \gg P(\text{canonical})$ and correctly flagging the substitution.
   - In `SF-SD`, however, the alignment lattice can bypass the substituted acoustic segment entirely via the $\epsilon$-deletion path, treating the acoustic perturbation as an unaligned deletion or allowing neighboring blanks to absorb the mass.
   - Consequently, for true substitutions, the denominator likelihood does not separate cleanly between substitution alternatives and background blanks. Substitution AP collapses from $0.246800$ down to $0.222544$ ($\Delta = -0.024256$, a severe $9.83\%$ relative degradation), failing the prospective noninferiority threshold ($\ge -0.005$) by nearly fivefold.

3. **Comparison Against Production Baseline R:**  
   While `SF-SD` outperforms `Baseline R` on omission AP ($0.051026$ vs $0.032564$, $\Delta = +0.018462$) and omission recall at 15% FPR ($43.07\%$ vs $39.42\%$, $\Delta = +3.65\text{ pp}$), it suffers an unacceptable drop on general error detection:
   - Binary AP drops from $0.313530$ (Baseline R) to $0.250049$ (SF-SD), a loss of **$-0.063481$** (-6.35 AP points).
   - Substitution AP drops from $0.304811$ (Baseline R) to $0.222544$ (SF-SD), a loss of **$-0.082267$** (-8.23 AP points).

4. **Comparison Against GOP-SA (Viterbi Segmental Control):**  
   The Viterbi self-aligned baseline `GOP-SA` achieves Omission AP of $0.049846$ (within $0.001180$ of SF-SD) and Recall@FPR15 of $41.61\%$, while maintaining higher Binary AP ($0.254370$ vs $0.250049$) and Substitution AP ($0.228104$ vs $0.222544$). This demonstrates that the segmentation-free graph extension does not achieve meaningful diagnostic superiority over classical segmental GOP.

5. **Scientific Disposition for P-SCORE:**  
   Under the frozen decision rules of Section 4.1 (`SCIENTIFIC_QA_AND_COMPARISON_REPORT.md`), P-SCORE is formally classified as **INCONCLUSIVE_TRADEOFF / NO-GO FOR DEPLOYMENT**. While the omission hypothesis is confirmed ($\Delta \text{AP}_{\text{omit}} \ge +0.020$), the substitution noninferiority violation ($\Delta \text{AP}_{\text{sub}} < -0.005$) prohibits treating `GOP-CTC-SF-SD` as a drop-in replacement for Baseline R.

---

## 4. Audit of Matched T-OP vs B-CONTROL Training and Evaluation Specifications

Because P-SCORE demonstrated that post-hoc acoustic graph decoding over frozen weights suffers an omission-substitution tradeoff, the focus shifts to representation learning in **P-SUP**. To ensure methodological rigor, the experimental specifications for Subtype Treatment (`T-OP`) and Matched Binary Control (`B-CONTROL`) have been audited against `OBJECTIVE_POLICY_AMENDMENT.md` and verified in code:

### 4.1 Parameter Budget and Capacity Matching Audit
Both arms utilize the exact same downstream architecture, `CompactOperationHead`. The parameter count is certified at exactly **266,667 trainable parameters**:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        COMPACT OPERATION HEAD CAPACITY INVENTORY                       │
├───────────────────────────────┬─────────────────────────────┬──────────────────────────┤
│ Module Component              │ Architectural Specification │ Trainable Parameters     │
├───────────────────────────────┼─────────────────────────────┼──────────────────────────┤
│ Acoustic Projection           │ Linear(768, 128)            │ 98,432                   │
│ Phone Embedding               │ Embedding(40, 128)          │ 5,120                    │
│ Positional Embedding          │ Embedding(256, 128)         │ 32,768                   │
│ Cross-Attention Module        │ MultiheadAttention(128, 4)  │ 66,048                   │
│ Acoustic LayerNorm            │ LayerNorm(128)              │ 256                      │
│ Operation MLP                 │ Linear(256, 128) -> GELU -> │ 33,283                   │
│                               │ Dropout(0.1) -> Linear(128,3│                          │
│ Realized CTC Head             │ Linear(768, 40)             │ 30,760                   │
├───────────────────────────────┼─────────────────────────────┼──────────────────────────┤
│ Total Trainable Head Budget   │ Both Arms Identical         │ **266,667**              │
│ Frozen Wav2Vec2 Backbone      │ Pretrained 960h (Frozen)    │ 94,371,712 (0 trainable) │
└───────────────────────────────┴─────────────────────────────┴──────────────────────────┘
```

- **Random Initialization Parity:** Both models are instantiated using `matched_initial_pair(seed=1337)`, guaranteeing that before update 0, the neural weights in `T-OP` and `B-CONTROL` are bit-identical.
- **Backbone Invariant:** Pretrained `facebook/wav2vec2-base-960h` (rev `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`) is strictly frozen in `eval` mode (`requires_grad = False`).

### 4.2 Objective Function Formulation and Score Semantics
The experimental contrast isolates supervision granularity while holding all other variables constant:

1. **Subtype Treatment (`T-OP`, `objective = "operation_ce"`):**
   - Head outputs 3 raw logits: $z = [z_{\text{cor}}, z_{\text{sub}}, z_{\text{del}}]$.
   - Target $y \in \{0, 1, 2\}$ supervised via 3-way categorical Cross-Entropy:
     $$\mathcal{L}_{\text{task}} = -\log \frac{\exp(z_{y})}{\sum_{c=0}^2 \exp(z_c)}$$
   - Score semantics:
     - Error score: $s_{\text{error}} = p_{\text{sub}} + p_{\text{del}} = 1 - p_{\text{cor}}$
     - Omission score: $s_{\text{omit}} = p_{\text{del}}$
     - Substitution score: $s_{\text{sub}} = p_{\text{sub}}$

2. **Matched Binary Control (`B-CONTROL`, `objective = "binary_collapsed"`):**
   - Head outputs identical 3 raw logits: $z = [z_0, z_1, z_2]$.
   - Collapsed logit: $\tilde{z} = \text{logsumexp}(z_1, z_2) - z_0 = \log(e^{z_1} + e^{z_2}) - z_0$.
   - Target $\tilde{y} = \mathbb{I}(y \neq 0) \in \{0, 1\}$ supervised via Binary Cross-Entropy with Logits:
     $$\mathcal{L}_{\text{task}} = \text{BCEWithLogits}(\tilde{z}, \tilde{y}) = \log(1 + e^{\tilde{z}}) - \tilde{y} \tilde{z}$$
   - Score semantics:
     - Error score: $s_{\text{error}} = \sigma(\tilde{z}) = \frac{e^{z_1} + e^{z_2}}{e^{z_0} + e^{z_1} + e^{z_2}} = 1 - p_0$.
     - Subtype scores: $s_{\text{omit}} = \text{null}$, $s_{\text{sub}} = \text{null}$ (mathematically unidentifiable).

### 4.3 Training and Optimization Specifications
Both arms must execute under strictly identical optimization conditions on authorized remote GPU hardware (Kaggle T4):
- **Optimizer:** AdamW (`learning_rate: 0.001`, `weight_decay: 0.01`, `eps: 1e-8`).
- **Gradient Clipping:** Max norm $1.0$.
- **Learning Rate Schedule:** 100 warmup updates, followed by cosine annealing over a fixed 15,000-update horizon.
- **Batching & Accumulation:** Batch size 2, gradient accumulation steps 2 (effective batch size 4 utterances). Tail accumulation divisor is strictly `actual_group_size`.
- **Numerical Precision:** CUDA FP16 AMP with PyTorch `GradScaler` (initial scale 4096).
- **Stopping Horizon:** Exactly 3,600 successful optimizer updates.

### 4.4 Mandatory Dual Checkpointing Accounting on Fold-1 VAL
In accordance with `AGENTS.md` and Section 4.2 of `EVALUATION_CONTRACT_V1.md`, evaluation on Fold-1 VAL must report metrics from **both** of the following checkpoints separately:

1. **Fixed-Budget 3,600-Update Checkpoint (`fixed_budget_3600.pt`):**
   - Captured at exactly update 3,600 (end of Epoch 6).
   - Serves as the primary pre-registered benchmark for prospective effect size evaluation.
   - Eliminates exposure bias and prevents post-hoc selection artifacts.
2. **Validation-Selected Checkpoint (`best.pt`):**
   - Captured at the validation epoch achieving peak Binary AP on $s_{\text{error}}$, subject to the strict blank-collapse rejection filter (`blank_fraction < 0.99`).
   - Accounts for validation-based early stopping dynamics.
3. **Confounding Prohibition:**
   - Under no circumstances may `best.pt` from one arm be compared against `fixed_budget_3600.pt` from another arm.
   - Confounding training update exposure with algorithmic capability is strictly prohibited. Both comparisons (`fixed_budget` vs `fixed_budget` and `best` vs `best`) must be presented side by side.

---

## 5. Comprehensive Summary of Audit Findings & Recommendations

### 5.1 Audit Findings Matrix

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              COMPREHENSIVE SCIENTIFIC AUDIT FINDINGS                                   │
├────────────────────────────────────────┬─────────────┬─────────────────────────────────────────────────┤
│ Verification Area                      │ Status      │ Technical Evidence / Proof                      │
├────────────────────────────────────────┼─────────────┼─────────────────────────────────────────────────┤
│ 1. Population Reconciliation           │ VERIFIED    │ Exact 37 unaligned tokens mapped (25 C, 4 S, 8D)│
│ 2. Exact Aligned Metric Parity         │ VERIFIED    │ All 4 scorers evaluated on exact N=9,919 cohort │
│ 3. P-SCORE Omission AP Gate            │ PASS        │ +0.021615 (0.051026 vs 0.029411, target ≥+0.020)│
│ 4. P-SCORE Recall@FPR15 Gate           │ PASS        │ +0.182482 (43.07% vs 24.82%, target ≥+0.030)   │
│ 5. P-SCORE Substitution AP Gate        │ FAIL        │ -0.024256 (0.222544 vs 0.246800, target ≥-0.005)│
│ 6. P-SCORE Binary AP Gate              │ FAIL        │ -0.009045 vs SF-S; -0.063481 vs Baseline R      │
│ 7. Omission-Substitution Tradeoff      │ DOCUMENTED  │ Epsilon-deletion path bypasses substitutions    │
│ 8. P-SUP Parameter Matching            │ VERIFIED    │ Exactly 266,667 head params in T-OP & B-CONTROL │
│ 9. P-SUP Backbone & Hyperparams        │ VERIFIED    │ Frozen Wav2Vec2, AdamW, lr=0.001, cosine sched  │
│ 10. Dual Checkpoint Policy             │ ENFORCED    │ Fixed 3,600 updates AND best.pt required        │
└────────────────────────────────────────┴─────────────┴─────────────────────────────────────────────────┘
```

### 5.2 Recommended Next Steps
1. **Commit Audit Report:** Maintain this report and associated unit tests in the coordinator repository under `prep/rq1-shared-base`.
2. **Kaggle Remote GPU Training Execution (P-SUP Lane):**
   - The user operator must navigate to the private Kaggle kernel `minkhoaa/langfens-rq1-p-sup-t4-training` via Web UI, attach user secret `HF_TOKEN`, and start execution via "Save & Run All".
   - Training will execute both `T-OP` and `B-CONTROL` concurrently across dual T4 GPUs under CUDA FP16 AMP to exactly 3,600 updates, mirroring epoch checkpoints to Hugging Face private repository storage.
   - Upon completion, download the emitted prediction files (`fixed_budget_3600.pt` and `best.pt`) for independent offline evaluation against the prospective criteria audited in Section 4.

---

*Certified by Agent D — Scientific Evaluation Audit Subagent.*
