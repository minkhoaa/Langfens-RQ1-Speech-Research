# SCORE DISTRIBUTION & CONFIDENCE SUPPRESSION ANALYSIS
## LANGFENS SM13-FUSION-DIAGNOSTIC-01

**Scope:** Quantitative forensic analysis of target class posterior distributions, raw log-likelihood ratios, ranking correlations, and cross-component suppression mechanics across TRAIN OOF, DEV_CAL, and DEV_SEARCH.  
**Auditor:** Senior Speech ML Research Scientist & Scientific Auditor  
**Primary Artifact:** `experiments/rq1_sm13_fusion_diagnostic_01/SCORE_DISTRIBUTIONS.csv`  

---

## 1. Executive Summary of Distribution Findings

This document presents a granular empirical dissection of the three scoring configurations in `LANGFENS-SM13-FUSION-V1`:
1. **SM13 Neural-only:** 13-layer scalar mixing head + 39D canonical phone reference.
2. **CTC/E4-only:** Layer 6 CTC projection evaluated under 5-hypothesis sequence likelihood ratios (SLLR, OLLR).
3. **Fixed 50/50 Fusion:** Arithmetic mean simplex $0.5 \cdot p_{\text{neural}} + 0.5 \cdot p_{\text{E4}}$.

### Key Discoveries:
1. **The Asymmetric Score Compression of E4:** On both out-of-domain development splits (`DEV_CAL` and `DEV_SEARCH`), E4's substitution probability $p_{\text{E4}, S}$ collapses into near-zero values (median on gold Substitution = **0.0081** on DEV_CAL, **0.0097** on DEV_SEARCH), with an absolute upper bound of **0.6508**. In contrast, E4's omission probability $p_{\text{E4}, O}$ is severely elevated across the entire target space (median on gold Substitution = **0.6351**, median on gold Omission = **0.9745**).
2. **The Fused Simplex Trap:** Because $p_{\text{fixed}, S} = 0.5 \cdot p_{\text{neural}, S} + 0.5 \cdot p_{\text{E4}, S}$, even when the Neural head achieves maximum confidence on a genuine substitution ($p_{\text{neural}, S} = 0.9957$), E4 averages it down to $\le 0.6539$.
3. **Operating Threshold Mismatch:** The shared operating threshold $\tau_T$ calibrated under nominal $B=0.20$ is selected in the range $[0.818, 0.849]$ to control false alarms from high $p_O$. Because fused $p_S \le 0.6539$, the operating threshold is mathematically unreachable for any substitution event on DEV_CAL, resulting in **exactly 0% Substitution recall**.
4. **Neural-Only Independence:** The Neural head alone suffers from **zero score compression**. It reaches $p_S = 0.9957$ on DEV_CAL and $0.9937$ on DEV_SEARCH, achieving healthy non-zero Substitution recall (**9.55%** on DEV_CAL, **7.75%** on DEV_SEARCH) under identical operating rules.

---

## 2. Quantile & Distribution Analysis

### 2.1 DEV_CAL Target Event Distributions ($N = 10,043$ events)

Breakdown by gold target class:
- Gold Correct ($y_t = 0$, $N = 7,800$)
- Gold Substitution ($y_t = 1$, $N = 1,781$)
- Gold Omission ($y_t = 2$, $N = 462$)

#### Table 2.1: Substitution Probabilities ($p_S$) on DEV_CAL
| Target Class | Scoring Config | Mean | Std | P10 | P25 | P50 (Med) | P75 | P90 | P95 | P99 | Max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Gold Substitution** | **Neural-only** | **0.5045** | 0.3081 | 0.0805 | 0.2096 | **0.5446** | **0.7811** | **0.8961** | **0.9467** | **0.9793** | **0.9957** |
| Gold Substitution | E4-only | 0.0296 | 0.0571 | 0.0017 | 0.0035 | 0.0081 | 0.0278 | 0.0817 | 0.1381 | 0.2583 | 0.6508 |
| Gold Substitution | Fixed 50/50 | 0.2670 | 0.1568 | 0.0572 | 0.1264 | 0.2880 | 0.4034 | 0.4675 | 0.4903 | 0.5453 | **0.6539** |
| Gold Correct | Neural-only | 0.1555 | 0.2003 | 0.0016 | 0.0089 | 0.0601 | 0.2393 | 0.4947 | 0.6660 | 0.8467 | 0.9910 |
| Gold Correct | E4-only | 0.0152 | 0.0336 | 0.0010 | 0.0022 | 0.0048 | 0.0125 | 0.0303 | 0.0711 | 0.1591 | 0.4852 |
| Gold Correct | Fixed 50/50 | 0.0853 | 0.1030 | 0.0029 | 0.0099 | 0.0435 | 0.1311 | 0.2564 | 0.3441 | 0.4337 | 0.6699 |
| Gold Omission | Neural-only | 0.2554 | 0.2593 | 0.0108 | 0.0381 | 0.1444 | 0.4116 | 0.6807 | 0.7855 | 0.9031 | 0.9836 |
| Gold Omission | E4-only | 0.0156 | 0.0260 | 0.0012 | 0.0024 | 0.0057 | 0.0144 | 0.0493 | 0.0801 | 0.1242 | 0.1903 |
| Gold Omission | Fixed 50/50 | 0.1355 | 0.1291 | 0.0118 | 0.0303 | 0.0832 | 0.2108 | 0.3465 | 0.3964 | 0.4573 | 0.4955 |

#### Table 2.2: Omission Probabilities ($p_O$) on DEV_CAL
| Target Class | Scoring Config | Mean | Std | P10 | P25 | P50 (Med) | P75 | P90 | P95 | P99 | Max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gold Substitution | Neural-only | 0.0304 | 0.1046 | 0.0001 | 0.0003 | 0.0010 | 0.0079 | 0.0571 | 0.1595 | 0.6143 | 0.9641 |
| **Gold Substitution** | **E4-only** | **0.5589** | 0.3425 | 0.0658 | 0.2335 | **0.6351** | **0.8859** | **0.9678** | **0.9839** | **0.9956** | **0.9998** |
| Gold Substitution | Fixed 50/50 | 0.2947 | 0.1875 | 0.0396 | 0.1378 | 0.3244 | 0.4511 | 0.4936 | 0.5308 | 0.7476 | 0.9588 |
| **Gold Omission** | **Neural-only** | **0.4060** | 0.3310 | 0.0232 | 0.0898 | **0.3518** | **0.6958** | **0.9027** | **0.9492** | **0.9832** | **0.9939** |
| **Gold Omission** | **E4-only** | **0.8886** | 0.2051 | 0.6279 | 0.8711 | **0.9745** | **0.9897** | **0.9954** | **0.9978** | **0.9989** | **0.9993** |
| **Gold Omission** | **Fixed 50/50** | **0.6473** | 0.2050 | 0.3700 | 0.5133 | **0.6271** | **0.8164** | **0.9240** | **0.9587** | **0.9846** | **0.9923** |
| Gold Correct | Neural-only | 0.0298 | 0.0877 | 0.0001 | 0.0003 | 0.0013 | 0.0102 | 0.0530 | 0.1439 | 0.6483 | 0.9707 |
| Gold Correct | E4-only | 0.2743 | 0.3340 | 0.0104 | 0.0267 | 0.0934 | 0.4907 | 0.8108 | 0.9219 | 0.9876 | 0.9999 |
| Gold Correct | Fixed 50/50 | 0.1521 | 0.1843 | 0.0093 | 0.0195 | 0.0619 | 0.2520 | 0.4254 | 0.4823 | 0.6994 | 0.9727 |

---

### 2.2 Deep Forensic Insights from the Distributions

1. **The Inversion of S and O Confidences by E4 on Genuine Substitutions:**
   - Look at row 1 vs row 2 of Table 2.2:
     - On genuine substitutions ($y_t = 1$), the Neural head correctly determines that omission probability is near zero (mean = **0.0304**, median = **0.0010**).
     - However, E4 assigns a mean omission probability of **0.5589** and median of **0.6351** to those exact same substitution events!
     - In fact, over 90% of genuine substitutions have $p_{\text{E4}, O} > 0.0658$, and over 25% have $p_{\text{E4}, O} > 0.8859$!
   - Consequently, when fused 50/50:
     - On genuine substitutions, the median fused omission probability is **0.3244**, which is *higher* than the median fused substitution probability (**0.2880**)!
     - Fused 50/50 actively converts Substitution candidates into Omission candidates or Correct abstentions!

2. **The 0.6539 Glass Ceiling on DEV_CAL:**
   - The absolute maximum $p_S$ generated by the fixed fusion on genuine substitutions in DEV_CAL is **0.653875**.
   - Not a single event among 1,781 substitutions reaches 0.6600.
   - 95% of all substitutions fall below **0.4903**.
   - Meanwhile, the frozen DEV_CAL threshold is $\tau_T = \mathbf{0.848521}$.
   - The probability that a substitution event exceeds $\tau_T$ is identically zero.

---

## 3. Analysis of Raw CTC Log-Likelihood Ratios (SLLR & OLLR)

To understand why E4's substitution probability collapses on DEV splits, we inspect the unnormalized log-likelihood ratios:
$$\text{SLLR}_i = \log P(\text{Sub Hypotheses}) - \log P(\text{Canonical Sequence})$$
$$\text{OLLR}_i = \log P(\text{Deletion Hypothesis}) - \log P(\text{Canonical Sequence})$$
$$p_{\text{E4}, S} = \frac{e^{\text{SLLR}_i}}{1 + e^{\text{SLLR}_i} + e^{\text{OLLR}_i}}$$

#### Table 3.1: Raw Log-Likelihood Ratio Quantiles on Genuine Errors
| Split | Target Event Type | Metric | Min | P10 | P25 | P50 (Med) | P75 | P90 | P95 | P99 | Max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **TRAIN OOF** | Gold Substitution | SLLR | -39.47 | -4.68 | -1.77 | -0.07 | +1.44 | +2.82 | +4.13 | +7.14 | **+17.58** |
| TRAIN OOF | Gold Substitution | OLLR | -35.82 | -3.53 | -1.41 | +0.43 | +2.18 | +3.73 | +4.70 | +6.29 | +10.23 |
| **DEV_CAL** | Gold Substitution | SLLR | -47.14 | -8.11 | -5.19 | **-2.85** | **-1.21** | **-0.05** | **+0.34** | **+1.79** | **+3.37** |
| DEV_CAL | Gold Substitution | OLLR | -44.82 | -3.85 | -1.45 | **+0.71** | **+2.33** | **+3.88** | **+4.71** | **+6.31** | **+11.58** |
| **DEV_SEARCH** | Gold Substitution | SLLR | -35.65 | -8.66 | -5.58 | **-3.17** | **-1.29** | **-0.08** | **+0.29** | **+1.37** | **+3.85** |
| DEV_SEARCH | Gold Substitution | OLLR | -34.68 | -4.42 | -1.89 | **+0.32** | **+2.02** | **+3.70** | **+4.54** | **+6.19** | **+8.02** |
| **DEV_CAL** | Gold Omission | OLLR | -21.43 | +0.52 | +2.05 | **+4.14** | **+5.25** | **+6.27** | **+7.07** | **+7.69** | **+8.21** |

### 3.1 The Algorithmic Cause of SLLR Collapse
Comparing TRAIN OOF to DEV_CAL reveals the smoking gun:
1. On **TRAIN OOF**, SLLR on gold substitutions reaches **+17.58** (P99 = +7.14). The exponential $e^{+17.58} = 4.3 \times 10^7$ completely dominates the softmax denominator ($1 + e^{\text{SLLR}} + e^{\text{OLLR}}$), allowing $p_{\text{E4}, S}$ to reach $1.0000$.
2. On **DEV_CAL**, the maximum SLLR across all 1,781 substitutions is only **+3.37**, and over 90% of substitutions have **negative SLLR** (median = **-2.85**).
3. Meanwhile, on those same substitutions in DEV_CAL, OLLR reaches **+11.58** (median = **+0.71**).
4. Because $\text{OLLR} > \text{SLLR}$ on the vast majority of substitution events, the deletion hypothesis exponentially suppresses the substitution probability:
   $$e^{\text{SLLR}} \ll e^{\text{OLLR}} \implies p_{\text{E4}, S} \to 0$$
   At the single best substitution event on DEV_CAL (Event 8637), $\text{SLLR} = 3.297$ and $\text{OLLR} = 2.603$:
   $$e^{3.297} \approx 27.03, \quad e^{2.603} \approx 13.50, \quad e^0 = 1.0$$
   $$p_{\text{E4}, S} = \frac{27.03}{1.0 + 27.03 + 13.50} = \frac{27.03}{41.53} = \mathbf{0.6508}$$
   Thus, E4's substitution probability cannot exceed 0.6508 even on its strongest candidate.

---

## 4. Cross-Component Ranking Correlations

We measure Pearson correlation ($r$) and Spearman rank correlation ($\rho$) between Neural probabilities and E4 probabilities across all splits:

#### Table 4.1: Neural vs E4 Correlation Matrix
| Dataset | Evaluation Subset | Score Type | Pearson $r$ | Spearman $\rho$ | Interpretation |
|---|---|---|---|---|---|
| TRAIN OOF (S101) | Full Population ($N=59,562$) | Substitution | +0.3957 | +0.4705 | Moderate ranking agreement |
| TRAIN OOF (S101) | Full Population ($N=59,562$) | Omission | +0.4079 | +0.4190 | Moderate ranking agreement |
| TRAIN OOF (S101) | Gold Substitution ($N=7,381$) | Substitution | +0.2235 | +0.2801 | Weak in-class correlation |
| TRAIN OOF (S101) | Gold Omission ($N=1,825$) | Omission | +0.0640 | +0.0414 | Virtually uncorrelated |
| **DEV_CAL** | **Full Population ($N=10,043$)** | **Substitution** | **+0.2185** | **+0.4137** | Moderate population ranking |
| **DEV_CAL** | **Gold Substitution ($N=1,781$)** | **Substitution** | **+0.0043** | **+0.0884** | **ZERO CORRELATION on true errors** |
| DEV_CAL | Full Population ($N=10,043$) | Omission | +0.5056 | +0.3475 | Moderate-to-strong agreement |
| DEV_CAL | Gold Omission ($N=462$) | Omission | +0.1220 | +0.0113 | Uncorrelated on true omissions |
| **DEV_SEARCH** | **Gold Substitution ($N=1,768$)** | **Substitution** | **+0.0441** | **+0.0570** | **ZERO CORRELATION on true errors** |
| DEV_SEARCH | Full Population ($N=19,755$) | Substitution | +0.2285 | +0.3695 | Moderate population ranking |

### 4.1 Interpretation
Within the subset of genuine substitution errors on unseen speakers (`DEV_CAL` and `DEV_SEARCH`), Pearson $r$ between the Neural head and E4 is **+0.0043** and **+0.0441**. There is **zero linear agreement** and near-zero monotonic agreement ($\rho \le 0.088$). The two heads are scoring completely orthogonal features on unseen speakers.

When two orthogonal models are combined via an unweighted, static 50/50 average, an uncalibrated head with suppressed dynamic range (E4) acts as pure noise/dampener on the calibrated head (Neural).

---

## 5. Comparative Warning Performance (Prospective Transfer)

Evaluating the three configurations using thresholds frozen strictly from Seed 101 TRAIN OOF ($B=0.20$, nominal $B_T=0.16$, $B_G=0.04$):
- **Neural-only:** $\tau_T = 0.901480, \tau_G = 0.842381$
- **E4-only:** $\tau_T = 0.988110, \tau_G = 0.842381$
- **Fixed 50/50 Fusion:** $\tau_T = 0.818188, \tau_G = 0.842381$

#### Table 5.1: Comparative Target Warning Metrics
| Split | Configuration | Realized FWR | Total FW | Wrong Type | Macro-3 F1 | AP-S | Recall-S | Precision-S | F1-S | AP-O | Recall-O | Precision-O | F1-O |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **TRAIN OOF** | Neural-only | 0.2056 | 370 | **33** | 0.1495 | 0.4136 | **5.89%** | 70.16% | **0.1087** | 0.3232 | 9.04% | 60.44% | 0.1573 |
| TRAIN OOF | E4-only | 0.2122 | 382 | 74 | 0.1264 | 0.2853 | 1.48% | 55.05% | 0.0288 | 0.2436 | 10.25% | 46.40% | 0.1679 |
| TRAIN OOF | Fixed 50/50 | 0.2017 | 363 | 52 | **0.1679** | **0.4222** | 3.36% | **75.15%** | 0.0643 | **0.3824** | **16.38%** | 59.44% | **0.2569** |
| **DEV_CAL** | **Neural-only** | 0.2333 | 70 | **8** | 0.1983 | 0.5472 | **9.55%** | **81.34%** | **0.1709** | 0.4692 | 10.17% | **71.21%** | 0.1780 |
| DEV_CAL | E4-only | 0.4867 | 146 | 58 | 0.2059 | 0.3221 | **0.00%** | 0.00% | 0.0000 | 0.3948 | **29.44%** | 50.37% | 0.3716 |
| DEV_CAL | Fixed 50/50 | 0.2333 | 70 | 10 | **0.2011** | **0.5513** | **0.00%** | 0.00% | 0.0000 | **0.5721** | 24.46% | 66.08% | **0.3570** |
| **DEV_SEARCH** | **Neural-only** | 0.3339 | 200 | **6** | **0.1595** | 0.3711 | **7.75%** | 49.82% | **0.1341** | 0.1436 | 6.22% | **34.21%** | 0.1053 |
| DEV_SEARCH | E4-only | 0.3139 | 188 | 40 | 0.1494 | 0.1861 | **0.00%** | 0.00% | 0.0000 | 0.1474 | **20.10%** | 21.76% | 0.2090 |
| DEV_SEARCH | Fixed 50/50 | **0.1636** | 98 | **4** | 0.1403 | **0.3814** | **0.00%** | 0.00% | 0.0000 | **0.1999** | 12.92% | 30.68% | 0.1818 |

---

## 6. Synthesis: The Exact Pathology of Fixed 50/50 Fusion

From the empirical evidence, the interaction between components can be summarized in three mechanical stages:

1. **Stage 1 (Underlying Representations):**
   - The Neural head has learned robust, high-AP representations for both Substitution (AP = 0.5472 on DEV_CAL) and Omission (AP = 0.4692).
   - The CTC/E4 head has acceptable AP for Omission (AP = 0.3948), but poor AP for Substitution (AP = 0.3221). More critically, on out-of-domain speakers, E4's substitution log-likelihood ratio (SLLR) is systematically biased negative (mean = -3.30).
2. **Stage 2 (Fusion Interaction):**
   - On genuine omissions ($y_t = 2$), both heads agree ($p_{\text{neural}, O} \approx 0.41, p_{\text{E4}, O} \approx 0.89$), yielding high fused scores ($p_{\text{fixed}, O} \approx 0.65 - 0.99$).
   - On genuine substitutions ($y_t = 1$), the heads violently disagree: Neural predicts $p_S = 0.70 - 0.99$, while E4 predicts $p_S = 0.01 - 0.08$ and $p_O = 0.60 - 0.95$.
   - Fused $p_S$ is cut in half ($0.25 - 0.49$).
3. **Stage 3 (Decision Boundary Decapitation):**
   - The system sets a single threshold $\tau_T \approx 0.82 - 0.85$ based on the upper tail of target alerts.
   - Because $p_{\text{fixed}, O}$ easily reaches 0.95, Omissions cross $\tau_T$ and produce warnings (Recall-O = 19 - 25%).
   - Because $p_{\text{fixed}, S}$ is capped at 0.6539, **Substitutions can never cross $\tau_T$**.
   - Result: 100% of substitutions are either silenced (classified as Correct) or misclassified as Omissions.

**Conclusion:** E4 is an **Omission specialist** that actively destroys Substitution recall when blindly combined via fixed 50/50 arithmetic fusion.
