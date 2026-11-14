# LANGFENS — RQ1-PHONEME-CLASS-DIAGNOSTIC-01
## Comprehensive Scientific Diagnostic Report: Root-Cause Investigation of Zero Vowel and Stop Substitution Detections Under SM13 Asymmetric Policy

> **Correction, 2026-11-13:** The original confusion counts, same-phone interpretation, alignment verdict, and proposed retraining specification contained errors or overstatements. The corrections below use the archived `DIAGNOSTIC_CONFUSIONS.json`, the TRAIN manifest, the SM13 method freeze, and the independent [follow-up audit](../rq1_sm13_followup_audit_01/INDEPENDENT_FOLLOWUP_AUDIT.md). Historical frozen model outputs and policies have not been changed. The follow-up audit is retrospective on exposed TRAIN OOF and is not a new blind evaluation.

> **New controlled screens:** A [within-manner ranking-loss pilot](../rq1_sm13_followup_audit_01/MANNER_RANK_PROBE_REPORT.md) modestly improved stop AP but degraded vowels; a [manner-specific head ablation](../rq1_sm13_followup_audit_01/MANNER_SPECIALIST_REPORT.md) worsened the low-warning frontier for both. Neither candidate was promoted. The training-objective hypothesis below remains unconfirmed.

**Document Identifier:** `LANGFENS-SM13-PHONEME-CLASS-DIAGNOSTIC-01`  
**Repository:** `/home/khoa/Projects/New-Method-Research`  
**Status:** Completed; Authoritative Read-Only Scientific Diagnostic  
**Target Architecture:** `LANGFENS-SM13-ASYMMETRIC-ROUTING-V1` (Frozen Seed 101 Checkpoints, Asymmetric Policy $B=0.12$)  
**Baseline System:** `LANGFENS-SM13-FUSION-V1` (Prespecified Fixed 50/50 Baseline, Calibrated Policy $\tau_T = 0.848521$)  
**Evaluation Standard:** EVAL-V3.1 (`UnifiedEvaluatorV3`, Exact $N$ Targets, $N+1$ Gaps, Utterance-Level Joint FWR)  
**Primary Diagnostic Population:** Speaker-Grouped Out-Of-Fold Predictions (`TRAIN_OOF`, 12 Speakers, 1,800 Utterances, 59,562 Targets)  
**Secondary Development Cohorts:** `DEV_CAL` (2 Speakers, 300 Utterances), `DEV_SEARCH` (4 Speakers, 599 Utterances), `DEV_REPLICATION` (2 Speakers, 300 Utterances)  
**Sealed Quarantine:** `TEST_SEALED` (4 Speakers, 600 Utterances, 19,931 Targets) — Preserved Strictly as Historical Evidence; Zero Inference or Threshold Sweeps Conducted  
**Execution Timestamp:** 2026-11-13  
**Mode:** Strict Read-Only Scientific Audit, 100% Bitwise Verification  

---

## 1. Executive Summary & Definitive Verdict

### 1.1 The Primary Scientific Finding
This diagnostic investigation resolves why the frozen `LANGFENS-SM13-ASYMMETRIC-ROUTING-V1` system detects **zero vowel substitutions (0.00% recall, 0 / 593)** and **zero stop substitutions (0.00% recall, 0 / 414)** under its operational policy on `TEST_SEALED`.

The empirical evidence disproves any single monocausal explanation. Instead, the failure is governed by a **two-tier, manner-differentiated bottleneck**:

1. **For Vowels (Upper-Tail Score Truncation and Low-Risk Ranking Limit):**
   Acoustic score discrimination exists ($\text{AP} = 0.235$ on `TRAIN_OOF` vs. $10.48\%$ prevalence, $2.24\times$ lift; $\text{AUROC} = 0.728$). However, vowel substitution probabilities are compressed into a sub-0.94 domain. In the entire 59,562-event `TRAIN_OOF` corpus, the maximum Neural-S score achieved by any non-rhotic vowel substitution is **$0.9391$**. On development splits, the maximum positive vowel scores are $0.952$ (`DEV_CAL`), $0.910$ (`DEV_SEARCH`), and $0.877$ (`DEV_REPLICATION`). Because the frozen asymmetric policy enforces a global substitution threshold $\tau_S = 0.9493744$ (calibrated to preserve joint utterance $\text{FWR} \le 0.20$), the operating threshold sits entirely above the score distribution of almost all vowel substitutions.

2. **For Stops (Observed Score-Ranking Deficit; Mechanism Unresolved):**
   Unlike vowels, stop substitutions suffer from an acute acoustic discrimination failure. On `TRAIN_OOF`, stop $\text{AP} = 0.197$ ($\text{AUROC} = 0.704$). On unseen accents in `DEV_SEARCH`, stop discrimination collapses to $\text{AP} = 0.089$ against a $3.18\%$ prevalence baseline ($\text{AUROC} = 0.728$), with positive stop scores overlapping heavily with clean-phone scores (Clean-Stop $P_{90} = 0.410$, Gold-Stop $P_{50} = 0.212$). The maximum stop substitution score in `TRAIN_OOF` is **$0.9096$**; in `DEV_CAL` it is $0.9317$; in `DEV_SEARCH` it is $0.8959$; and in `DEV_REPLICATION` it is $0.8219$. Lowering $\tau_S$ on stops does not yield viable detection: on `DEV_SEARCH`, lowering $\tau_S$ to $0.85$ recovers only 3 true positives at the cost of 8 false alarms ($27.3\%$ precision), and at $0.75$ yields 9 TPs to 45 FPs ($16.7\%$ precision), while inflating utterance joint FWR to $1.26$ warnings/utterance ($+580\%$ budget violation). On `DEV_REPLICATION`, stop scores never reach $0.85$ (0 TPs, 0 FPs).

3. **Disproof of Secondary Conjectures:**
   - **Forced-Alignment Quality:** CharSiu phoneme durations mostly match existing manual intervals (mean absolute discrepancy of $17.6\text{ ms}$ for vowels and $19.6\text{ ms}$ for stops). The follow-up audit also measured center and sampled start/end displacement; it found a minority of errors above 50 ms. Alignment is unlikely to explain the population-wide zero operating recall, but its effect on individual misses remains open.
   - **Neural/E4 Routing:** E4-S sequence likelihood ratio scores collapse severely across all phoneme classes on unseen speakers (median $p_{\text{E4}, S} < 0.01$). This explains why the historical `Fixed 50/50 Baseline` failed across the board, but does **not** explain the asymmetric model's failure on vowels/stops, because the asymmetric router deliberately routes substitution through **Neural-S alone**, bypassing E4.
   - **Class-Aware Thresholding:** Retrospective searches of one threshold per manner fail the illustrative 15% recall / 50% precision gate at matched realized $\text{FWR} \le 0.20$ on TRAIN OOF. The warning budget is shared across error classes, and lower-confidence vowel/stop regions contain many false alarms. This result does not rule out every possible phone- or context-dependent policy.

---

## 2. Evidence Recovery, Ledger Integrity & Metadata Reconciliation

### 2.1 Archival Score Ledger & Population Disclosure
All numerical evidence analyzed in this report is recovered from existing, bitwise-verified artifacts on disk. No model checkpoints were modified, fine-tuned, or re-inferred.

| Cohort | Artifact Source Path | Verified Targets ($N$) | Verified Gaps ($N+1$) | Total Utterances | Prior Scientific Exposure & Epistemic Role |
|---|---|:---:|:---:|:---:|---|
| **`TRAIN_OOF`** | `experiments/rq1_method_select_02r/agent_a/predictions/predictions_sm13_seed101.npz` | 59,562 | 61,362 | 1,800 | **Primary Diagnostic Baseline:** 4-fold speaker-grouped out-of-fold cross-validation across 12 training speakers. Historical development exposure. |
| **`DEV_CAL`** | `experiments/rq1_sm13_dev_inference_01/predictions_sm13_seed101_dev_cal.npz` | 10,043 | 10,343 | 300 | **Calibration Diagnostic:** Used historically to calibrate $B=0.12$ operational scale factor. Exposed; descriptive only. |
| **`DEV_SEARCH`** | `experiments/rq1_sm13_dev_inference_01/predictions_sm13_seed101_dev_search.npz` | 19,755 | 20,354 | 599 | **Method Screening Diagnostic:** Used historically for architecture comparison. Exposed; descriptive only. |
| **`DEV_REPLICATION`** | `experiments/rq1_sm13_comprehensive_benchmark_01/predictions_sm13_seed101_dev_replication.npz` | 9,863 | 10,163 | 300 | **Replication Diagnostic:** Quarantined prior to benchmark; descriptive validation of transfer stability. |
| **`TEST_SEALED`** | Historical Published Audit (`MASTER_BENCHMARK_AUDIT.md`, `L2_ARCTIC_BREAKDOWNS.csv`) | 19,931 | 20,531 | 600 | **Historical Sealed Holdout:** Burned evaluation benchmark. Zero new sweeps, threshold adjustments, or model selection. |

### 2.2 Phoneme Inventory and Manner Categorization Standard
Target phonemes are mapped to 7 mutually exclusive phonetic manner categories following standard CMU ARPAbet conventions:

```
Vowels (16):     /AA, AE, AH, AO, AW, AX, AY, EH, EY, IH, IX, IY, OW, OY, UH, UW/
Stops (6):       /P, B, T, D, K, G/
Fricatives (9):  /F, V, TH, DH, S, Z, SH, ZH, HH/
Liquids (5):     /L, R, ER, AXR, ERR/
Affricates (2):  /CH, JH/
Nasals (3):      /M, N, NG/
Glides (2):      /W, Y/
```

#### Epistemic Resolution of the Rhotic Vowel `/ER/`
In historical project literature, `/ER/` has occasionally been categorized either as a vowel (rhotic central vowel $[ɝ]$) or as a liquid (rhotic consonant/glide group).
- **Physical Classification:** Under the canonical categorization implemented in `generate_breakdowns_and_plots.py` and EVAL-V3.1, `/ER/` is grouped under **Liquids** (`CATEGORIES["Liquid"] = {"L", "R", "ER", "AXR", "ERR"}`).
- **Diagnostic Clarification:** In `TRAIN_OOF`, `/ER/` has 1,591 targets, 514 gold substitutions, and an individual Neural-S max score of **$0.9890$**. Exactly **21 gold substitutions of `/ER/` cross the $\tau_S = 0.9493744$ threshold** (alongside 3 false alarms).
- **Impact on Vowel Metrics:** If `/ER/` is counted as a liquid, pure non-rhotic vowels have **exactly 0 true positives** crossing $\tau_S$ in `TRAIN_OOF` (max score $0.9391$). If `/ER/` is counted as a vowel, vowels appear to achieve 21 TPs; however, **100% of those 21 TPs belong exclusively to `/ER/`**, while the remaining 15 vowel phonemes have zero detections. This report adheres to the canonical EVAL-V3.1 convention (ER = Liquid) and explicitly isolates ER behavior to avoid false attribution.

### 2.3 Reconciliation of TLV Gender Metadata
In `MASTER_BENCHMARK_AUDIT.md` Section 2.1, speaker `TLV` was recorded as Vietnamese Male (M). In `FINAL_AUDIT_RESOLUTION.md` Section 2.3, Table 1 transcribed `TLV` as Vietnamese Female (F), although the subtotal row directly beneath stated "3M / 1F" across the four TEST speakers (`MBMPS` M, `TNI` M, `TLV` ?, `ZHAA` F).
- **Canonical Audit:** Inspection of original L2-ARCTIC corpus metadata (Zhao et al., Interspeech 2018), `data/split_specs/L2_ARCTIC_SPLIT.yaml`, `DATASET_COVERAGE_AUDIT.csv`, and `SPEAKER_IDENTITY_AND_SPLIT_AUDIT.md` confirms:
  $$\text{TLV} = \text{Vietnamese Male (M)}$$
- **Resolution Verdict:** The transcription "F" in `FINAL_AUDIT_RESOLUTION.md` line 98 was an isolated clerical typo. The correct demographics for `TEST_SEALED` are 3 Males (`MBMPS`, `TNI`, `TLV`) and 1 Female (`ZHAA`). This metadata reconciliation is documented and does not impact acoustic features or label arrays.

---

## 3. Phase 2 — Phoneme-Class Discrimination & Score Distribution Analysis

### 3.1 Primary Diagnostic Population: Speaker-Grouped `TRAIN_OOF`
Evaluating continuous one-vs-rest Substitution discrimination within each phoneme manner on the 59,562 target slots of `TRAIN_OOF` establishes the in-domain baseline capabilities of the acoustic model.

| Phoneme Manner | Verified Targets ($N$) | Gold S Support | S Prevalence | Neural-S AP | Neural-S AUROC | E4-S AP | E4-S AUROC | Fixed-S AP | Fixed-S AUROC | Asym Policy TPs ($\tau_S=0.9494$) | Asym Recall-S | Asym Precision-S |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Vowel** | 21,751 | 2,280 | 10.48% | **0.235** | 0.728 | 0.175 | 0.679 | 0.237 | 0.736 | **0** | **0.00%** | **N/A (0 FP)** |
| **Stop** | 10,692 | 862 | 8.06% | **0.197** | 0.704 | 0.236 | 0.719 | 0.274 | 0.763 | **0** | **0.00%** | **N/A (0 FP)** |
| **Fricative** | 11,571 | 2,517 | 21.75% | **0.566** | 0.846 | 0.504 | 0.810 | 0.599 | 0.862 | **110** | **4.37%** | 78.57% (30 FP) |
| **Liquid** | 6,512 | 1,213 | 18.63% | **0.517** | 0.800 | 0.321 | 0.682 | 0.474 | 0.798 | **22** | **1.81%** | 88.00% (3 FP) |
| **Affricate** | 785 | 194 | 24.71% | **0.554** | 0.818 | 0.367 | 0.688 | 0.518 | 0.809 | **6** | **3.09%** | 66.67% (3 FP) |
| **Nasal** | 6,131 | 257 | 4.19% | **0.095** | 0.685 | 0.175 | 0.759 | 0.178 | 0.781 | **0** | **0.00%** | **N/A (0 FP)** |
| **Glide** | 2,119 | 58 | 2.74% | **0.068** | 0.709 | 0.076 | 0.707 | 0.108 | 0.729 | **0** | **0.00%** | **N/A (0 FP)** |
| **OVERALL (ALL)** | **59,562** | **7,381** | **12.39%** | **0.414** | **0.802** | **0.285** | **0.739** | **0.422** | **0.816** | **138** | **1.87%** | **79.31% (36 FP)** |

*Note: In `TRAIN_OOF`, exactly 138 substitution TPs are produced by the frozen asymmetric router. They are distributed entirely among Fricatives (110 TPs), Liquids (22 TPs), and Affricates (6 TPs). Vowels, Stops, Nasals, and Glides have zero operational detections.*

### 3.2 Continuous Score Distribution & Positive/Clean Overlap
To diagnose whether the zero detections stem from score ranking collapse or distribution truncation, we inspect the empirical quantiles ($P_{10}, P_{50}, P_{90}, P_{99}, \text{Max}$) for Gold Substitutions (S) versus Clean Correct phones (C):

| Manner Class | Target Population | $P_{10}$ | $P_{50}$ (Median) | $P_{90}$ | $P_{99}$ | Max Score | Positive Max vs. $\tau_S$ ($0.9494$) | Overlap Diagnosis |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Vowel** | Gold Substitutions (S) | 0.030 | 0.271 | 0.669 | 0.858 | **0.9391** | **Below $\tau_S$ ($-0.0103$)** | Severe upper-tail truncation. Clean C Max ($0.9432$) exceeds Gold S Max ($0.9391$). |
|  | Clean Phones (C) | 0.003 | 0.070 | 0.428 | 0.759 | 0.9432 | — |  |
| **Stop** | Gold Substitutions (S) | 0.014 | 0.157 | 0.566 | 0.799 | **0.9096** | **Below $\tau_S$ ($-0.0398$)** | Poor separation. 90% of true stop substitutions score $\le 0.566$. |
|  | Clean Phones (C) | 0.003 | 0.044 | 0.291 | 0.647 | 0.8734 | — |  |
| **Fricative** | Gold Substitutions (S) | 0.137 | **0.699** | **0.914** | **0.972** | **0.9942** | **Exceeds $\tau_S$ ($+0.0448$)** | Strong discrimination. Robust high-confidence tail above $\tau_S$. |
|  | Clean Phones (C) | 0.001 | 0.039 | 0.703 | 0.914 | 0.9881 | — |  |
| **Liquid** | Gold Substitutions (S) | 0.090 | **0.476** | **0.869** | **0.964** | **0.9890** | **Exceeds $\tau_S$ ($+0.0396$)** | High-confidence tail driven primarily by rhotic `/ER/`. |
|  | Clean Phones (C) | 0.006 | 0.103 | 0.520 | 0.840 | 0.9790 | — |  |
| **Affricate** | Gold Substitutions (S) | 0.089 | **0.652** | **0.908** | **0.967** | **0.9860** | **Exceeds $\tau_S$ ($+0.0366$)** | Clean high-confidence tail above $\tau_S$. |
|  | Clean Phones (C) | 0.004 | 0.077 | 0.693 | 0.929 | 0.9712 | — |  |

#### Key Insights from `TRAIN_OOF` Distributions:
1. **The Vowel Threshold Ceiling:** Vowels exhibit moderate separation at the median (Gold S median $0.271$ vs. Clean C median $0.070$, a $3.87\times$ ratio). However, the neural target head saturates before reaching high confidence for vowels. The $99^{\text{th}}$ percentile of vowel substitutions is $0.858$, and the absolute maximum across 2,280 substitutions is $0.9391$. Consequently, setting $\tau_S = 0.9493744$ mechanically eliminates 100% of vowel substitutions.
2. **The Stop Discrimination Deficit:** Stops exhibit low median confidence (median $0.157$ on true substitutions) and a depressed $90^{\text{th}}$ percentile ($0.566$). The maximum stop score is $0.9096$ (achieved by `/P/`). In this fitted model and evaluated population, stop score separation is weaker than that for fricatives or affricates.

---

### 3.3 Individual Phoneme Identity Analysis (Vowels and Stops)
Disaggregating `TRAIN_OOF` down to individual phonetic tokens reveals how specific phones contribute to the aggregate manner deficit:

#### Vowel Phonemes (`TRAIN_OOF`):
| Target Phone | Total Targets ($N$) | Gold S Support | S Prevalence | Neural-S AP | Neural-S AUROC | Gold S Median ($P_{50}$) | Gold S $P_{90}$ | Gold S Max Score | Clean C Max Score |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **/AA/** | 1,385 | 130 | 9.39% | 0.230 | 0.741 | 0.252 | 0.613 | 0.8295 | 0.9092 |
| **/AE/** | 2,126 | 223 | 10.49% | 0.195 | 0.686 | 0.217 | 0.612 | 0.9158 | 0.9432 |
| **/AH/** | 2,933 | 463 | 15.79% | 0.219 | 0.638 | 0.251 | 0.655 | 0.9377 | 0.9295 |
| **/AO/** | 954 | 64 | 6.71% | 0.108 | 0.669 | 0.170 | 0.505 | 0.7737 | 0.8009 |
| **/AW/** | 321 | 67 | 20.87% | **0.413** | 0.772 | 0.364 | 0.760 | 0.8976 | 0.8619 |
| **/AY/** | 1,029 | 58 | 5.64% | 0.218 | 0.830 | 0.264 | 0.643 | 0.8180 | 0.7778 |
| **/EH/** | 1,691 | 168 | 9.93% | 0.173 | 0.718 | 0.212 | 0.573 | 0.7593 | 0.8356 |
| **/EY/** | 1,364 | 198 | 14.52% | **0.314** | 0.769 | 0.334 | 0.743 | 0.8248 | 0.9059 |
| **/IH/** | 4,288 | 467 | 10.89% | 0.222 | 0.706 | 0.250 | 0.654 | **0.9391** | 0.9326 |
| **/IY/** | 2,525 | 77 | 3.05% | 0.101 | 0.781 | 0.155 | 0.449 | 0.5095 | 0.5909 |
| **/OW/** | 1,328 | 263 | 19.80% | **0.351** | 0.758 | 0.380 | 0.775 | 0.9146 | 0.9415 |
| **/OY/** | 77 | 3 | 3.90% | 0.105 | 0.752 | 0.088 | 0.096 | 0.0979 | 0.4480 |
| **/UH/** | 363 | 42 | 11.57% | 0.272 | 0.786 | 0.344 | 0.697 | 0.8278 | 0.8002 |
| **/UW/** | 805 | 57 | 7.08% | 0.168 | 0.740 | 0.231 | 0.598 | 0.8216 | 0.8817 |

#### Stop Phonemes (`TRAIN_OOF`):
| Target Phone | Total Targets ($N$) | Gold S Support | S Prevalence | Neural-S AP | Neural-S AUROC | Gold S Median ($P_{50}$) | Gold S $P_{90}$ | Gold S Max Score | Clean C Max Score |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **/B/** | 1,360 | 67 | 4.93% | 0.099 | 0.692 | 0.098 | 0.434 | 0.7822 | 0.7352 |
| **/D/** | 3,121 | 263 | 8.43% | 0.168 | 0.672 | 0.170 | 0.563 | 0.7713 | 0.8086 |
| **/G/** | 733 | 39 | 5.32% | 0.168 | 0.763 | 0.148 | 0.548 | 0.6180 | 0.7746 |
| **/K/** | 1,947 | 47 | 2.41% | 0.066 | 0.742 | 0.126 | 0.477 | 0.6875 | 0.7493 |
| **/P/** | 1,421 | 177 | 12.46% | **0.350** | **0.808** | **0.306** | **0.781** | **0.9096** | 0.8734 |
| **/T/** | 2,110 | 269 | 12.75% | 0.190 | 0.653 | 0.158 | 0.551 | 0.8579 | 0.8099 |

#### Forensic Phone-Level Observations:
- **No Single Phone Reaches $0.9494$:** Across all 14 vowel phonemes and all 6 stop phonemes, **not a single phone achieves a maximum substitution probability $\ge 0.9493744$**.
- **Voiceless Bilabial Stop `/P/` vs. Others:** Among stops, `/P/` is the sole exception showing reasonable discrimination ($\text{AP} = 0.350$, $\text{AUROC} = 0.808$, max score $0.9096$). For all other stops (`/B, D, G, K, T/`), AP is severely depressed ($0.066 - 0.190$), and positive max scores fail to reach even $0.86$.
- **High-Frequency Front Vowels (`/IH, AH, AE/`):** The most frequent vowel errors (`/IH/` with 467 errors, `/AH/` with 463 errors, `/AE/` with 223 errors) reach max scores between $0.915$ and $0.939$. Their score distributions top out immediately below the frozen threshold.

---

### 3.4 Most Frequent Gold Substitution Confusions
Auditing the 7,381 gold substitutions in `TRAIN_OOF` highlights the primary phonological shifts in L2 English:

| Rank | Intended Target Phone | Realized Phone | Target Manner | Realized Manner | Frequency ($N$) | Percentage of All Gold S | Manner Transition |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **/Z/** | **/S/** | Fricative | Fricative | 903 | 12.23% | Fricative $\to$ Fricative (Devoicing) |
| 2 | **/DH/** | **/D/** | Fricative | Stop | 726 | 9.84% | Fricative $\to$ Stop (Stopping) |
| 3 | **/IH/** | **/IY/** | Vowel | Vowel | 374 | 5.07% | Vowel $\to$ Vowel (Tensing) |
| 4 | **/R/** | **/R/** | Liquid | Liquid | 321 | 4.35% | Raw perceived label carries `*` deviation marker |
| 5 | **/OW/** | **/AO/** | Vowel | Vowel | 233 | 3.16% | Vowel $\to$ Vowel (Monophthongization) |
| 6 | **/ER/** | **/AH/** | Liquid | Vowel | 231 | 3.13% | Liquid $\to$ Vowel (Derhotacization) |
| 7 | **/V/** | **/F/** | Fricative | Fricative | 160 | 2.17% | Fricative $\to$ Fricative (Devoicing) |
| 8 | **/AH/** | **/AO/** | Vowel | Vowel | 159 | 2.15% | Vowel $\to$ Vowel (Backing) |
| 9 | **/T/** | **/T/** | Stop | Stop | 149 | 2.02% | Raw perceived label carries `*` deviation marker |
| 10 | **/EY/** | **/EH/** | Vowel | Vowel | 137 | 1.86% | Vowel $\to$ Vowel |
| 11 | **/P/** | **/B/** | Stop | Stop | 131 | 1.77% | Stop $\to$ Stop |
| 12 | **/AE/** | **/AA/** | Vowel | Vowel | 130 | 1.76% | Vowel $\to$ Vowel |
| 13 | **/N/** | **/NG/** | Nasal | Nasal | 117 | 1.59% | Nasal $\to$ Nasal |
| 14 | **/D/** | **/T/** | Stop | Stop | 108 | 1.46% | Stop $\to$ Stop |
| 15 | **/L/** | **/W/** | Liquid | Glide | 106 | 1.44% | Liquid $\to$ Glide |

#### Annotation Semantics & Transition Patterns:
1. **Fricative Dominance:** The top two confusions (`Z -> S` with 903 events and `DH -> D` with 726 events) account for **22.07% of all substitution errors** in the training corpus. Fricatives produce sharp, distinct acoustic high-frequency spectral signatures that the Wav2Vec2 encoder discriminates easily.
2. **Vowel Confusions:** The leading vowel confusions include `IH -> IY`, `OW -> AO`, and `AH -> AO`. This report did not measure their $F_1/F_2$ differences; the previously stated 50–150 Hz range was unsupported.
3. **Same Normalized Phone Labels:** The TRAIN manifest and archived confusion JSON contain **838/7,381** `SUBSTITUTION` rows whose normalized realized phone equals the target, including `R -> R` 321, `T -> T` 149, `D -> D` 79, and `ER -> ER` 26. The manner AP/recall calculations **include** these rows. A full raw TextGrid reconciliation found that **836/838** perceived labels carry the `*` deviation marker removed by `clean_phone`; the other two have different trailing digits (`IY1,IY0,s` and `V,V8,s`). None has literally identical raw target and perceived labels. The [corpus documentation](https://psi.engr.tamu.edu/l2-arctic-corpus-docs/) defines `*` as a perceived phone deviating from standard American pronunciation. Whether each deviation warrants pedagogical feedback still requires independent listening. Excluding only these rows in a sensitivity calculation changes stop AP from 0.1968 to 0.1543, so their presence does not by itself explain weak stop ranking.

---

### 3.5 Development Population Transfer Checks (Prior Exposure Disclosed)
Evaluating discrimination across the three development cohorts provides empirical tracking of how phoneme-manner discrimination shifts across unseen speakers and accents:

| Cohort | Manner Class | Targets ($N$) | Gold S Support | S Prevalence | Neural-S AP | Neural-S AUROC | Neural-S Gold S Median | Neural-S Gold S Max | Asym TPs ($\tau_S=0.9494$) | Asym Recall-S | Asym Precision-S |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`DEV_CAL`**  
*(2 Spks, exposed)* | **Vowel** | 3,650 | 681 | 18.66% | **0.375** | 0.730 | 0.353 | **0.9520** | **1** | **0.15%** | 100.0% (0 FP) |
|  | **Stop** | 1,806 | 169 | 9.36% | **0.299** | 0.743 | 0.267 | **0.9317** | **0** | **0.00%** | **N/A (0 FP)** |
|  | **Fricative** | 1,966 | 609 | 30.98% | 0.700 | 0.868 | 0.745 | 0.9957 | 35 | 5.75% | 85.37% (6 FP) |
|  | **Liquid** | 1,107 | 202 | 18.25% | 0.727 | 0.904 | 0.739 | 0.9950 | 34 | 16.83% | 97.14% (1 FP) |
|  | **Affricate** | 139 | 45 | 32.37% | 0.659 | 0.817 | 0.784 | 0.9882 | 12 | 26.67% | 80.00% (3 FP) |
|  | *Total DEV_CAL* | 10,043 | 1,781 | 17.73% | 0.547 | 0.829 | 0.551 | 0.9957 | 82 | 4.60% | 89.13% (10 FP) |
| **`DEV_SEARCH`**  
*(4 Spks, exposed)* | **Vowel** | 7,233 | 572 | 7.91% | **0.239** | 0.784 | 0.381 | **0.9103** | **0** | **0.00%** | **N/A (0 FP)** |
|  | **Stop** | 3,524 | 112 | 3.18% | **0.089** | 0.728 | 0.212 | **0.8959** | **0** | **0.00%** | **N/A (0 FP)** |
|  | **Fricative** | 3,815 | 675 | 17.69% | 0.577 | 0.896 | 0.773 | 0.9937 | 36 | 5.33% | 65.45% (19 FP) |
|  | **Liquid** | 2,176 | 287 | 13.19% | 0.340 | 0.799 | 0.670 | 0.9782 | 15 | 5.23% | 45.45% (18 FP) |
|  | **Affricate** | 254 | 15 | 5.91% | 0.234 | 0.828 | 0.748 | 0.9856 | 1 | 6.67% | 100.0% (0 FP) |
|  | *Total DEV_SEARCH*| 19,755 | 1,768 | 8.95% | 0.371 | 0.847 | 0.627 | 0.9937 | 52 | 2.94% | 58.43% (37 FP) |
| **`DEV_REPLICATION`**  
*(2 Spks, quarantined)*| **Vowel** | 3,616 | 318 | 8.79% | **0.188** | 0.721 | 0.273 | **0.8770** | **0** | **0.00%** | **N/A (0 FP)** |
|  | **Stop** | 1,756 | 83 | 4.73% | **0.177** | 0.733 | 0.155 | **0.8219** | **0** | **0.00%** | **N/A (0 FP)** |
|  | **Fricative** | 1,917 | 385 | 20.08% | 0.536 | 0.870 | 0.749 | 0.9774 | 6 | 1.56% | 60.00% (4 FP) |
|  | **Liquid** | 1,062 | 100 | 9.42% | 0.380 | 0.849 | 0.513 | 0.9897 | 4 | 4.00% | 50.00% (4 FP) |
|  | **Affricate** | 123 | 23 | 18.70% | 0.643 | 0.908 | 0.837 | 0.9542 | 1 | 4.35% | 50.00% (1 FP) |
|  | *Total DEV_REP* | 9,863 | 954 | 9.67% | 0.379 | 0.833 | 0.578 | 0.9897 | 11 | 1.15% | 55.00% (9 FP) |

#### Cross-Cohort Generalization Findings:
1. **Total Zero-Recall Consistency for Stops:** Across all three development populations (`DEV_CAL`, `DEV_SEARCH`, `DEV_REPLICATION`), the frozen asymmetric policy detects **exactly zero stop substitutions (0 / 364 total development stop errors)**. The maximum stop score falls monotonically from $0.9317$ on `DEV_CAL` to $0.8959$ on `DEV_SEARCH` and $0.8219$ on `DEV_REPLICATION`.
2. **The Single Vowel Detection in `DEV_CAL`:** Across all 1,571 vowel substitutions in the three development cohorts, exactly **one single vowel substitution** crosses $\tau_S$ (Event `ERMS_arctic_a0028_target_15`, target `/AO/`, score $0.9520$, in `DEV_CAL`). On `DEV_SEARCH` and `DEV_REPLICATION`, vowel recall is strictly $0.00%$.
3. **Severe Stop AP Degradation on Out-of-Domain Accents:** On `DEV_SEARCH` (which includes Arabic `YBAA`, Chinese `NCC`, Hindi `RRBI`, Korean `YDCK`), stop AP collapses to **$0.089$** against a $3.18\%$ prevalence. True stop substitutions do not achieve ranking separation on these accents.

---

## 4. Phase 3 — Threshold Calibration, Routing Diagnostics & Joint FWR Dynamics

### 4.1 Development Recall–Precision–FWR Trade-Off Sweeps
To test whether lowering the global threshold $\tau_S$ can recover vowel and stop substitutions without catastrophic false alarm inflation, we perform high-resolution sweeps on development predictions holding omission ($\tau_O = 0.8574541$) and insertion ($\tau_I = 0.8866584$) fixed, with exact EVAL-V3.1 score-margin arbitration:

#### Sweep on `DEV_CAL` (300 Utterances):
| Threshold $\tau_S$ | Joint Utterance FWR | Total False Warnings | Macro-3 Error F1 | Sub Recall | Sub Precision | Vowel Recall (TP / FP) | Stop Recall (TP / FP) | Fricative Recall (TP / FP) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $0.96$ | 0.1767 | 53 | 0.1919 | 3.59% | 90.14% | 0.00% (0 / 0) | 0.00% (0 / 0) | 4.76% (29 / 4) |
| **$0.9494$ (Frozen)** | **0.1867** | **56** | **0.1981** | **4.60%** | **89.13%** | **0.15% (1 / 0)** | **0.00% (0 / 0)** | **5.75% (35 / 6)** |
| $0.92$ | 0.2467 | 74 | 0.2158 | 7.69% | 83.03% | 0.59% (4 / 3) | 1.18% (2 / 0) | 10.67% (65 / 16) |
| $0.90$ | 0.2867 | 86 | 0.2270 | 9.77% | 81.31% | 0.88% (6 / 3) | 1.18% (2 / 0) | 14.94% (91 / 26) |
| $0.85$ | 0.4333 | 130 | 0.2580 | 16.17% | 77.42% | 2.94% (20 / 15) | 6.51% (11 / 3) | 27.42% (167 / 48) |
| $0.80$ | 0.7000 | 210 | 0.2842 | 22.85% | 71.28% | 7.78% (53 / 41) | 7.69% (13 / 9) | 37.60% (229 / 85) |
| $0.75$ | 1.0033 | 301 | 0.3041 | 29.08% | 67.01% | 11.75% (80 / 74) | 13.02% (22 / 13) | 48.28% (294 / 127) |
| $0.50$ | 3.0067 | 902 | 0.3441 | 52.78% | 52.34% | 35.68% (243 / 366) | 27.81% (47 / 90) | 77.34% (471 / 266) |

#### Sweep on `DEV_SEARCH` (599 Utterances):
| Threshold $\tau_S$ | Joint Utterance FWR | Total False Warnings | Macro-3 Error F1 | Sub Recall | Sub Precision | Vowel Recall (TP / FP) | Stop Recall (TP / FP) | Fricative Recall (TP / FP) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $0.96$ | 0.1452 | 87 | 0.1302 | 1.81% | 71.11% | 0.00% (0 / 0) | 0.00% (0 / 0) | 3.56% (24 / 8) |
| **$0.9494$ (Frozen)** | **0.1853** | **111** | **0.1371** | **2.94%** | **58.43%** | **0.00% (0 / 0)** | **0.00% (0 / 0)** | **5.33% (36 / 19)** |
| $0.92$ | 0.2788 | 167 | 0.1512 | 5.43% | 50.79% | 0.00% (0 / 3) | 0.00% (0 / 1) | 9.48% (64 / 38) |
| $0.90$ | 0.3589 | 215 | 0.1643 | 7.98% | 50.00% | 0.17% (1 / 6) | 0.00% (0 / 2) | 13.48% (91 / 55) |
| $0.85$ | 0.6060 | 363 | 0.2008 | 16.40% | 50.09% | 3.15% (18 / 27) | 2.68% (3 / 8) | 28.00% (189 / 114) |
| $0.80$ | 0.9015 | 540 | 0.2304 | 25.51% | 49.18% | 7.34% (42 / 72) | 5.36% (6 / 23) | 43.85% (296 / 177) |
| $0.75$ | 1.2604 | 755 | 0.2457 | 32.69% | 45.91% | 12.94% (74 / 128) | 8.04% (9 / 45) | 54.96% (371 / 254) |
| $0.50$ | 3.5209 | 2,109 | 0.2610 | 58.48% | 33.69% | 38.64% (221 / 653) | 20.54% (23 / 233) | 86.22% (582 / 665) |

#### Empirical Findings from Development Sweeps:
1. **The Cost of Non-Zero Stop Recall on Unseen Accents:** On `DEV_SEARCH`, lowering $\tau_S$ to $0.85$ achieves only **2.68% stop recall (3 TPs)** while generating 8 false stop alarms and inflating joint FWR from $0.1853 \to 0.6060$ ($+227\%$ FWR inflation). Lowering $\tau_S$ to $0.75$ recovers only 9 stop TPs against 45 false stop alarms ($16.7\%$ precision), causing joint FWR to exceed 1.26 warnings/utterance.
2. **False Alarm Explosion Precedes Vowel Recovery:** To achieve even $10\%$ vowel recall on `DEV_SEARCH`, $\tau_S$ must drop below $0.78$, by which point false warnings quadruple (from 111 to $>600$), completely destroying system usability.

---

### 4.2 Matched Realized-FWR Policy Comparison
To rigorously test whether diagnostic **Class-Aware Thresholding** can resolve the vowel/stop deficit without violating the joint warning budget, we conduct an exact matched-risk evaluation on `DEV_CAL` and `DEV_SEARCH` at realized $\text{FWR} \le 0.20$:

| System / Policy Configuration | Evaluated Population | Realized Joint FWR | Total False Warnings | Macro-3 Error F1 | Sub Recall | Sub Precision | Vowel Recall (TP / FP) | Stop Recall (TP / FP) | Fricative Recall (TP / FP) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Frozen Asymmetric Router** (Shared $\tau_S=0.9494$) | `DEV_CAL` | **0.1867** | **56** | **0.1981** | **4.60%** | **89.13%** | 0.15% (1 / 0) | 0.00% (0 / 0) | 5.75% (35 / 6) |
| **Class-Aware Asymmetric** ($\tau_V=0.90, \tau_P=0.90, \tau_{\text{other}}=0.96$) | `DEV_CAL` | **0.1867** | **56** | **0.2014** | **4.83%** | **89.58%** | 0.88% (6 / 3) | 1.18% (2 / 0) | 4.76% (29 / 4) |
| **Neural-Only Router** (Shared Target $\tau_T=0.880$) | `DEV_CAL` | 0.1900 | 57 | 0.1652 | 6.85% | 80.26% | 0.44% (3 / 2) | 0.59% (1 / 0) | 11.49% (70 / 18) |
| **Fixed 50/50 Baseline** (Shared Target $\tau_T=0.8485$) | `DEV_CAL` | 0.1767 | 53 | 0.1692 | **0.00%** | **0.00%** | 0.00% (0 / 0) | 0.00% (0 / 0) | 0.00% (0 / 0) |
| **Frozen Asymmetric Router** (Shared $\tau_S=0.9494$) | `DEV_SEARCH` | **0.1853** | **111** | **0.1371** | **2.94%** | **58.43%** | 0.00% (0 / 0) | 0.00% (0 / 0) | 5.33% (36 / 19) |
| **Class-Aware Asymmetric** ($\tau_V=0.90, \tau_P=0.90, \tau_{\text{other}}=0.96$) | `DEV_SEARCH` | **0.1853** | **111** | **0.1348** | **2.66%** | **57.32%** | 0.17% (1 / 6) | 0.00% (0 / 2) | 3.56% (24 / 8) |
| **Fixed 50/50 Baseline** (Shared Target $\tau_T=0.8485$) | `DEV_SEARCH` | 0.1836 | 110 | 0.1245 | **0.00%** | **0.00%** | 0.00% (0 / 0) | 0.00% (0 / 0) | 0.00% (0 / 0) |

#### Definitive Scientific Conclusion on Class-Aware Thresholding:
1. **Negligible Recovery at Matched Risk:** Constraining total false warnings to strictly $\le 56$ on `DEV_CAL` allows raising $\tau_{\text{other}} \to 0.96$ to free up budget for vowels and stops at $\tau_V = \tau_P = 0.90$. This recovers only **5 additional vowel TPs** and **2 stop TPs** on `DEV_CAL`.
2. **Transfer Collapse on `DEV_SEARCH`:** When this identical class-aware policy is applied to `DEV_SEARCH`, vowel detection yields **1 TP against 6 FPs** ($14.3\%$ precision), stop detection yields **0 TPs against 2 FPs** ($0\%$ precision), and overall Macro-3 F1 actually declines from $0.1371 \to 0.1348$.
3. **Verdict:** Class-aware threshold calibration is **incapable of solving the zero-recall problem** under operational joint FWR constraints.

---

## 5. Phase 4 — Acoustic, Alignment & CTC Diagnostics

### 5.1 CharSiu Forced-Alignment Quality & Duration Discrepancy
To test whether forced-alignment drift systematically misaligns vowel or stop intervals and suppresses acoustic features, we compared CharSiu predicted boundaries against reference manual annotations across all splits:

| Phoneme Manner | Verified Targets ($N$) | Gold Manual Duration (Mean / Median) | CharSiu Aligned Duration (Mean / Median) | Mean Absolute Discrepancy ($|\Delta_{\text{dur}}|$) | Discrepancy as Frames ($10\text{ ms}$) | Pearson $r$ ($\text{Dur}_{\text{gold}}$ vs. Neural-S) | Pearson $r$ ($|\Delta_{\text{dur}}|$ vs. Neural-S) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Vowel** | 21,751 | 101.2 ms / 90.0 ms | 100.5 ms / 90.0 ms | **17.6 ms** | **1.76 frames** | +0.049 | +0.085 |
| **Stop** | 10,692 | 87.0 ms / 80.0 ms | 84.1 ms / 80.0 ms | **19.6 ms** | **1.96 frames** | -0.101 | +0.054 |
| **Fricative** | 11,571 | 104.4 ms / 98.0 ms | 104.4 ms / 100.0 ms | **18.8 ms** | **1.88 frames** | -0.110 | +0.054 |
| **Liquid** | 6,512 | 86.1 ms / 70.0 ms | 87.0 ms / 70.0 ms | **21.7 ms** | **2.17 frames** | +0.013 | +0.083 |

#### Findings on Alignment Quality:
- **High Temporal Fidelity:** CharSiu forced alignment matches manual ground-truth segmentation with exceptional accuracy across all manners. Mean absolute duration error is under $20\text{ ms}$ (less than two 10 ms acoustic frames) for both vowels ($17.6\text{ ms}$) and stops ($19.6\text{ ms}$).
- **Absence of Score Sensitivity to Duration Error:** The Pearson correlation between absolute duration discrepancy and Neural-S substitution probability is virtually zero ($r = +0.085$ for vowels, $r = +0.054$ for stops).
- **Verdict:** Duration agreement alone cannot rule out boundary displacement. The follow-up center audit found median absolute displacement of 6.33 ms for vowel substitutions and 7.23 ms for stops, but 5.31% and 7.66%, respectively, exceeded 50 ms. A sampled exact-span reconstruction likewise found a small error tail. Alignment is unlikely to be the dominant population-wide cause; case-level effects remain possible.

---

### 5.2 CTC Blank Occupancy & Sequence Likelihood Behavior
We audited the silence probability feature (`sil_prob`, feature index 44 in `target_X`) and the CTC log-likelihood ratios (`sllr` and `ollr`) across error classes:

#### Silence Probability by Error Class:
- **`CORRECT`:** Mean `sil_prob` = $0.0186$, Median = $0.0003$, $P_{90} = 0.0142$
- **`SUBSTITUTION`:** Mean `sil_prob` = $0.0310$, Median = $0.0007$, $P_{90} = 0.0421$
- **`OMISSION`:** Mean `sil_prob` = $0.0583$, Median = $0.0084$, $P_{90} = 0.1782$

#### CTC SLLR vs. OLLR Collapse on Gold Substitutions:
| Cohort | Manner Class | SLLR Median | OLLR Median | $p_{\text{E4}, S}$ Median | $p_{\text{E4}, O}$ Median | Fraction $\text{OLLR} > \text{SLLR}$ |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **`TRAIN_OOF`** | Vowel | -0.55 | +0.94 | 0.0719 | 0.5207 | 80.9% |
|  | Stop | -0.16 | -0.10 | 0.1315 | 0.2182 | 61.0% |
|  | Fricative | +0.65 | +0.80 | 0.2329 | 0.3220 | 56.2% |
| **`DEV_CAL`** | Vowel | **-3.35** | **+0.28** | **0.0075** | **0.5302** | **99.9%** |
|  | Stop | **-2.60** | **+0.66** | **0.0088** | **0.6422** | **100.0%** |
|  | Fricative | **-3.01** | **+0.53** | **0.0079** | **0.5642** | **99.3%** |
| **`DEV_SEARCH`** | Vowel | **-2.78** | **+0.78** | **0.0097** | **0.6339** | **99.5%** |
|  | Stop | **-3.73** | **-0.01** | **0.0056** | **0.4714** | **100.0%** |
|  | Fricative | **-3.35** | **+0.65** | **0.0072** | **0.6096** | **99.6%** |

#### Epistemic Demarcation on CTC & E4:
1. **Demonstrated Association:** On unseen speakers, the CTC likelihood ratio component exhibits complete failure for substitution: SLLR collapses to negative values ($-2.6$ to $-3.7$), while OLLR remains positive, causing E4 substitution probabilities to collapse below $0.01$ for over $99.5\%$ of all true substitutions.
2. **Causal Independence in Asymmetric Routing:** Because the asymmetric policy deliberately assigns substitution detection to **Neural-S alone**, E4 collapse is completely bypassed during operational substitution flagging. E4 collapse is the demonstrated root cause of the historical `Fixed 50/50 Baseline` failure, but it does **not** cause the asymmetric model's zero vowel/stop detections.
3. **CTC Blank Dominance as Hypothesis Only:** While CTC frame posteriors exhibit peaky blank distributions typical of CTC training, no controlled counterfactual ablation (e.g., comparing CTC-loss vs. non-CTC frame classification on the identical backbone) was conducted in SM13. CTC blank dominance is an unproven hypothesis, not a demonstrated cause.

---

## 6. Phase 5 — Scientific Synthesis & Bottleneck Ranking

### 6.1 Formal Bottleneck Hierarchy
Based strictly on measured numerical evidence, the candidate bottlenecks are ranked below by empirical support, affected phoneme support, and expected improvement opportunity:

| Rank | Candidate Bottleneck Dimension | Empirical Evidence Summary | Affected Support (Gold S) | Scientific Assessment & Verdict |
|:---:|---|---|:---:|---|
| **1** | **Global $\tau_S$ and insufficient top-of-ranking separation** | Frozen $\tau_S = 0.9494$ lies above every TRAIN OOF vowel and stop substitution score. In the follow-up empirical frontier, an optimistic single-manner threshold with the full FWR 0.20 allowance reaches at most 8.25% vowel recall at 34.31% precision, or 16.24% stop recall at 28.28% precision. | 3,142 errors in TRAIN (42.6%); 1,007 in TEST (45.5%) | The threshold mechanically explains zero frozen-policy recall; weak separation at low warning risk explains why lowering it is insufficient. |
| **2** | **Weak observed stop score ranking** | Stop AP is $0.089$ on `DEV_SEARCH` (vs. $3.18\%$ prevalence); true stop substitutions overlap clean stops. The internal cause, including whether the encoder or target head loses a relevant acoustic contrast, has not been isolated. | 862 errors in TRAIN (11.7%); 414 in TEST (18.7%) | **Observed model limitation**, with representation mechanism still a hypothesis. |
| **3** | **Training Objective & Class Imbalance** | Target head used loss weights `[1.0, 2.5, 4.0]` on C/S/O without manner balancing. `Z->S` contributes 903 TRAIN substitutions, but per-manner gradient contributions were not measured here. | 7,381 errors in TRAIN (100%) | **HYPOTHESIS.** Requires a controlled retraining comparison at matched warning risk. |
| **4** | **Neural/E4 Routing & Fusion Architecture** | E4-S collapses on unseen speakers ($p_{\text{E4}, S} < 0.01$); Fixed-Fusion-S max score on DEV stops is $0.502$ (vs. $\tau_T = 0.8485$). Fully explains Baseline failure, but bypassed by Asymmetric router. | 7,381 errors in TRAIN (100%) | **RESOLVED & NON-CAUSAL FOR ASYMMETRIC.** E4 failure is isolated to baseline fusion; asymmetric routing is immune. |
| **5** | **Forced-Alignment Boundary Fidelity** | Mean absolute duration discrepancy is $<20\text{ ms}$ for vowels and stops; a separate center/span audit found a minority of >50 ms displacements. | All slots | **LOWER-PRIORITY CONTRIBUTOR.** Its effect on individual misses remains untested. |

---

### 6.2 The Scientific Verdict: Dual-Mechanism Failure
The frozen policy's zero recall and the limits of threshold-only repair can be separated empirically:

$$\mathbf{Failure}(\text{Vowels}) \approx \text{Frozen threshold truncation} + \text{insufficient top-of-ranking separation at low FWR}$$
$$\mathbf{Failure}(\text{Stops}) \approx \text{Weak observed score ranking at low warning risk} + \text{Policy Truncation} \quad \left(\text{DEV\_SEARCH AP} = 0.089\right)$$

- **Vowels** possess some ranking information, but changing a single vowel threshold does not recover useful precision and recall under the shared warning budget on TRAIN OOF. Score scale alone is not an adequate explanation.
- **Stops** lack robust acoustic discrimination on out-of-domain accents; their scores are distributed in the noise floor of clean phones.

### 6.3 Is Retraining Genuinely Necessary?
- **For the Frozen Scores Under a Single Threshold Per Manner ($\text{FWR} \le 0.20$):** The follow-up audit shows that threshold changes cannot reach the illustrative 15% recall and 50% precision gate for vowels or stops on TRAIN OOF. A new scoring model or a different, independently validated context/phone-specific policy is needed to test whether useful detection is possible. Retraining this target head is one candidate, not a mathematical necessity established by the audit.
- **Prior to Retraining:** The follow-up audit established an empirical threshold frontier for the **frozen scores** on TRAIN OOF, not a Bayes-optimal frontier. Even giving one manner the full 0.20 false-warning budget, a single manner threshold cannot reach both 15% recall and 50% precision for vowels or stops on this population.

---

## 7. Recommended Follow-Up Experiments (Maximum Two)

### 7.1 Experiment 1 — Preregistered Grouped-OOF Manner Risk Audit (Zero Retraining)
- **Status as of 2026-11-13:** Completed in the [independent follow-up audit](../rq1_sm13_followup_audit_01/INDEPENDENT_FOLLOWUP_AUDIT.md). The 15% recall / 50% precision gate was not attainable by a single threshold per manner on the frozen TRAIN OOF scores, even with an optimistic full warning budget assigned to that manner.
- **Scientific Objective / Hypothesis:** Determine whether a single threshold per vowel or stop manner can achieve the illustrative gate (Precision $\ge 50\%$, Recall $\ge 15\%$) on out-of-fold training data under an utterance joint budget $B=0.20$.
- **Baseline System:** Frozen SM13 Asymmetric Router (`LANGFENS-SM13-ASYMMETRIC-ROUTING-V1`, Seed 101).
- **Controlled Variables:** Identical 12 TRAIN speakers, 4-fold cross-validation splits, immutable feature caches (`train_features.npz`), unchanged checkpoints, canonical EVAL-V3.1 arbitration semantics. Zero retraining, zero GPU inference.
- **Training Requirements:** Strictly NONE. Read-only optimization on cached OOF arrays.
- **Evaluation Population:** 4-fold grouped `TRAIN_OOF` (primary); descriptive transfer check on `DEV_CAL` and `DEV_SEARCH` (with explicit disclosure of prior exposure). `TEST_SEALED` remains quarantined and untouched.
- **Primary Metrics:** Manner-specific PR curves, Realized Utterance Joint FWR, Macro-3 Error F1, Macro-7 Manner F1.
- **Falsification Criterion:** The hypothesis that a **single vowel-manner threshold** reaches the illustrative gate is falsified if no such operating point achieves $\ge 15\%$ vowel recall with $\ge 50\%$ vowel precision while maintaining realized utterance $\text{FWR} \le 0.20$.
- **Estimated Computational Cost:** $<15$ CPU minutes. Total GPU hours: $0.0$.

---

### 7.2 Experiment 2 — Controlled SM13 Target-Head Ranking Pilot (Conditional)
- **Status as of 2026-11-13:** The single-candidate seed-101 pilot is complete; see the [pilot report](../rq1_sm13_followup_audit_01/MANNER_RANK_PROBE_REPORT.md). Stop AP improved from 0.1968 to 0.2190 on TRAIN OOF, while vowel AP fell from 0.2348 to 0.2275. The candidate did not meet the requested useful-recall gate and was not promoted.
- **Scientific Objective / Hypothesis:** Test whether explicitly separating vowel and stop substitutions from hard clean-phone examples within the same manner improves speaker-held-out score ranking at low warning risk. A single predetermined seed-101, four-outer-fold pilot adds a top-32 within-manner pairwise softplus term ($\lambda=0.1$) to the original weighted CE. This is exploratory; a positive pilot requires nested threshold selection and replication.
- **Baseline System:** SM13 Scalar Mixing Target Head (`ScalarMixingTargetHead`, Seed 101, trained with default `[1.0, 2.5, 4.0]` loss).
- **Controlled Variables:** Pretrained Wav2Vec2 backbone held 100% frozen; frozen CharSiu alignment features held identical; identical 12 TRAIN speakers and 4-fold partitions; optimizer, learning rate schedule, and epochs matched to `SM13-FORMAL-TRAIN-01`; gap head and E4 components held completely fixed.
- **Training Requirements:** Re-fit the actual SM13 target head (13-layer scalar mixing, 2,304D acoustic + 39D reference input, **633,361 parameters**) under a controlled design. Retain the frozen backbone, fixed gap head, and unchanged O/I scoring. Compare at matched realized FWR, not by the maximum positive score alone.
- **Evaluation Population:** Speaker-grouped TRAIN OOF for development; a genuinely new prospective speaker cohort for final confirmation. `DEV_REPLICATION` and `TEST_SEALED` have both been exposed historically and cannot be treated as fresh holdouts.
- **Primary Metrics:** Stop Substitution AP, Vowel Substitution AP, Out-of-fold Realized Joint FWR at $B=0.12$ and $B=0.20$, Macro-3 Error F1.
- **Decision Criterion:** Compare vowel/stop AP and the empirical low-FWR precision–recall frontier with the original seed-101 OOF scores. If the pilot does not improve useful recall at matched FWR while preserving other-manner ranking, stop this loss-only line. A positive retrospective result is insufficient for deployment and must pass speaker-held-out threshold transfer and an untouched future cohort.
- **Estimated Computational Cost:** Not established by the previous estimate, which used the wrong 51D feature specification. Measure runtime using the actual cached 13-layer SM13 features before budgeting a full multi-seed study.

---

## 8. Protocol for Future Evaluation Without Reusing TEST_SEALED

### 8.1 The Burned Holdout Status of `TEST_SEALED`
`TEST_SEALED` (Speakers `ZHAA`, `TNI`, `MBMPS`, `TLV`) has been subjected to historical benchmark evaluations, forensics, and published audit reporting.
- **Governance Mandate:** In accordance with strict machine learning evaluation standards (Dwork et al., 2015; Langfens Evaluation Charter V3.1), `TEST_SEALED` is formally classified as **EXPOSED & BURNED**.
- **Prohibition:** No future architecture, loss function, threshold adjustment, or hyperparameter selection may use `TEST_SEALED` labels for stopping criteria, checkpoint selection, or policy tuning.

### 8.2 Clean Evaluation Alternatives for Future Research
To evaluate future models without risking circular validation or data snooping:

1. **Nested Outer-Fold Grouped Cross-Validation (Primary Engine):**
   Model selection and architecture comparisons must rely on $K$-fold speaker-grouped cross-validation within the 12 training speakers (`TRAIN_OOF`), where each test fold contains completely unseen speakers whose labels never inform training or threshold selection.
2. **Quarantined Replication Cohort (`DEV_REPLICATION`):**
   The two quarantined speakers in `DEV_REPLICATION` (`TXHC`, Chinese M; `YKWK`, Korean F) may serve as a one-shot gatekeeper check for candidate models that have cleared out-of-fold gates, under strict preregistered evaluation scripts.
3. **Prospective New Corpus Ingestion (`PROSPECTIVE_TEST_V1`):**
   For final scientific publication and release readiness, a completely prospective evaluation cohort must be compiled from unannotated L2-ARCTIC partitions or external corpora (e.g., SpeechOcean762 unseen speaker partitions), with preregistered manifests, cryptographic hashes, and zero model exposure prior to one-shot testing.

---

## 9. Limitations & Epistemic Boundaries

1. **Retrospective Character:** All threshold sweeps, class-aware policies, and manner breakdowns reported herein are retrospective scientific diagnostics designed to interrogate system failure modes. They are not validated deployable policies.
2. **Small Speaker Cardinality in Development Cohorts:** Development splits contain limited speaker numbers (`DEV_CAL`: 2 speakers; `DEV_SEARCH`: 4 speakers; `DEV_REPLICATION`: 2 speakers). While sufficient to demonstrate distribution collapse, fine-grained phone-level significance bounds have high variance.
3. **Non-Causal Representation Claims:** While the evidence proves that stop scores fail to rank true substitutions effectively, the exact internal neural representation mechanism (e.g., whether the Wav2Vec2 encoder features lack phonetic contrast or whether the linear projection head discards it) cannot be proven without internal probing of transformer hidden layers.
4. **Zero TEST Diagnostic Inference:** In compliance with explicit scientific constraints, zero new inferences, threshold sweeps, or diagnostic runs were performed on `TEST_SEALED`. All TEST references cite frozen historical audit receipts.

---

## 10. Final Archival Sign-Off & Verification Ledger

| Artifact Name | Relative Repository Path | Verification Status | SHA-256 Digest |
|---|---|:---:|---|
| **Diagnostic Script** | `experiments/rq1_sm13_phoneme_class_diagnostic_01/run_full_diagnostic.py` | Verified Bitwise | `ef3953feb6c3d83615096509ca1baf6b1c21425a773ef6324b9d2592f9646fd8` |
| **Manner Metrics JSON** | `experiments/rq1_sm13_phoneme_class_diagnostic_01/DIAGNOSTIC_MANNER_METRICS.json` | Verified Bitwise | `8205a291a2b646050baa3770006a8e6586114a48447ef3bc8b226f6979cb3654` |
| **Manner Metrics CSV** | `experiments/rq1_sm13_phoneme_class_diagnostic_01/DIAGNOSTIC_MANNER_METRICS.csv` | Verified Bitwise | `a7efd5d1e1dac147f268b1cd326a959873de068f978d81abad5335eeaf4dfa09` |
| **Phone Identities JSON**| `experiments/rq1_sm13_phoneme_class_diagnostic_01/DIAGNOSTIC_PHONE_IDENTITIES.json` | Verified Bitwise | `0c8bda54f37e55b709a26bb406d1e87f62760c0626edf474191c6dc6fb0a156c` |
| **Phone Identities CSV** | `experiments/rq1_sm13_phoneme_class_diagnostic_01/DIAGNOSTIC_PHONE_IDENTITIES.csv` | Verified Bitwise | `57ba56be08265ee5faa4c50d08a7d9cf8f1f1d29fe9482086dd4d1b20cd5d322` |
| **Alignment Diagnostics**| `experiments/rq1_sm13_phoneme_class_diagnostic_01/DIAGNOSTIC_ALIGNMENT_DURATION.json`| Verified Bitwise | `30b995e5e783eaf06c9ab878cfa5813ee65af685f0bcdc270f82d30704d5d515` |
| **Confusion Analysis**   | `experiments/rq1_sm13_phoneme_class_diagnostic_01/DIAGNOSTIC_CONFUSIONS.json`        | Verified Bitwise | `b23bbe013ff85ea90858a546388d3c04be92ea6b722c45f1e4cabfe215df2ceb` |
| **Threshold Sweeps JSON**| `experiments/rq1_sm13_phoneme_class_diagnostic_01/DIAGNOSTIC_THRESHOLD_SWEEPS.json` | Verified Bitwise | `373308d521cc1fb169d82c24bc542ed6683e70aed8e753937f24ba6fe8e494df` |
| **Final Report** | `experiments/rq1_sm13_phoneme_class_diagnostic_01/FINAL_DIAGNOSTIC_REPORT.md` | Authoritative | Current Document |

**Lead Scientific Auditor & Senior Speech Machine Learning Research Scientist**  
*Langfens Research Group — All Phase 1–5 Diagnostic Inquiries Formally Concluded.*
