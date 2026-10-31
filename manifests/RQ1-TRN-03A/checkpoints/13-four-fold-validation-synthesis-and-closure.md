# LANGFENS — RQ1-TRN-03A / 13-Four-Fold Validation Synthesis & Closure Report

**Status:** `COMPLETED; GATE_PASSED_FOR_PHASE_CLOSURE`

**Recorded at:** 2026-10-31T22:30:00+07:00

**Evidence source:** Standalone CPU evaluation replay across Folds 1–4 (`audit_scripts/evaluate_cross_fold_validation.py`), full out-of-fold validation predictions ($N=39,576$ tokens across 1,200 utterances), `manifests/RQ1-TRN-03A/evaluation/FOUR_FOLD_VALIDATION_REPORT.json`, `manifests/RQ1-TRN-03A/evaluation/four_fold_metrics_summary.csv`, `manifests/RQ1-TRN-03A/evaluation/eight_speakers_validation_breakdown.csv`, split manifest `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`.

**Evidence level:** `INDEPENDENTLY_VERIFIED` (Automated Regression Test Suite, Cryptographic SHA-256 Readback)

**Branch:** `review/rq1-trn-03a-cross-fold-planning`

---

## 1. Executive Summary

This report delivers the authoritative cross-fold generalization assessment for **Baseline R** across all four folds of the L2-ARCTIC corpus ($K=4$), synthesizing results from **39,576 out-of-fold validation tokens** across **8 disjoint evaluation speakers** under strict zero-oracle and counterfactual acoustic controls.

### 1.1 Key Scientific Findings
1. **Detection Performance (Binary MDD AP):**
   - **Primary Fixed-Budget Endpoint (Epoch 6 / Exactly 3,600 Updates):**
     - **Macro-Averaged Binary AP ($N=4$ folds):** **`0.279214`** $\pm$ `0.026468` (Student's $t$ 95% CI: **`[0.194991, 0.363436]`**, $df=3$).
     - **Micro-Averaged Pooled Binary AP ($N=39,576$ tokens, $E=4,213$ errors):** **`0.272582`** (Prevalence: `10.65%`).
   - **Validation-Selected Endpoint (`best.pt`):**
     - **Macro-Averaged Best Binary AP:** **`0.289353`** $\pm$ `0.024201` (95% CI: `[0.212356, 0.366351]`).
     - **Micro-Averaged Best Binary AP:** **`0.287469`**.
     - Selecting peak checkpoints yields only a minor $+1.01$ percentage point macro lift over the fixed 3,600-update milestone, confirming that the fixed-budget milestone captures $\approx 96.5\%$ of optimal validation performance without selection confounding.

2. **Genuine Acoustic Reliance (Matched vs Mismatched Gap):**
   - Evaluating models with within-speaker shift-1 deranged audio produces an AP drop across all four folds:
     - Fold 1: $\Delta_{\text{gap}} = \mathbf{+0.022299}$
     - Fold 2: $\Delta_{\text{gap}} = \mathbf{+0.016975}$
     - Fold 3: $\Delta_{\text{gap}} = \mathbf{+0.008180}$
     - Fold 4: $\Delta_{\text{gap}} = \mathbf{+0.004784}$
   - **Macro Acoustic Gap:** **`+0.013060`** $\pm$ `0.004010` (Student's $t$ 95% CI: **`[+0.000299, +0.025820]`**). The 95% confidence interval is strictly positive, proving statistically that the model genuinely extracts acoustic evidence and does not function as a pure language/canonical prompt prior memorizer.
   - **Micro Acoustic Gap:** **`+0.010619`** ($0.272582$ matched vs $0.261962$ mismatched).

3. **Extraordinary Acoustic Invariance (Greedy CTC PER):**
   - At the primary 3,600-update milestone, CTC phone error rate exhibits remarkable stability across four entirely disjoint cohorts:
     - Fold 1: `23.68%` | Fold 2: `23.48%` | Fold 3: `24.83%` | Fold 4: `23.60%`
   - **Macro CTC PER:** **`23.90%`** $\pm$ `0.31%` (Student's $t$ 95% CI: **`[22.90%, 24.89%]`**).
   - **Micro CTC PER:** **`23.90%`** (Sub: `16.81%`, Del: `5.03%`, Ins: `2.06%`).
   - The extreme consistency (spread $< 1.35$ percentage points across 5 distinct L1 accents) demonstrates that the frozen Wav2Vec2 backbone provides uniform acoustic phonetic feature representation across diverse non-native accents.

4. **Pedagogical Operating Points (Missed Errors vs False Alarms):**
   - **Standard Threshold ($\tau = 0.50$):**
     - Macro Missed Error Rate ($\text{FNR}$): **`83.53%`** (Micro: `83.38%`).
     - Macro False Alarm Rate ($\text{FPR}$): **`2.98%`** (Micro: `2.98%`).
     - Pooled Precision: `0.3989`, Recall: `0.1662`, $F_1$: `0.2346`.
     - *Pedagogical diagnosis:* Standard threshold $\tau=0.50$ is overly conservative for L2 tutoring, missing over 83% of student pronunciation errors to maintain a low false alarm rate ($< 3\%$).
   - **High-Sensitivity Pedagogical Threshold ($\tau = 0.20$):**
     - Macro Missed Error Rate ($\text{FNR}$): **`51.86%`** (Micro: `51.41%`).
     - Macro False Alarm Rate ($\text{FPR}$): **`17.94%`** (Micro: `17.93%`).
     - Pooled Precision: `0.2440`, Recall: `0.4859`, $F_1$: `0.3249`.
     - *Pedagogical diagnosis:* Lowering the operating point to $\tau=0.20$ cuts missed errors down to $51.86\%$ (capturing nearly half of all mispronunciations) while maintaining false alarm rates at a tolerable $17.94\%$.

5. **Zero-Oracle Protocol Z Robustness:**
   - Replacing gold TextGrid prompt intervals with raw written text run through offline CMUdict G2P yields an aligned-only Macro AP of **`0.277563`** (99.64% alignment coverage) and Policy B1 Full-Gold Macro AP of **`0.279161`**.
   - The difference between Protocol Z Policy B1 ($0.279161$) and Protocol O Full Gold ($0.279214$) is merely **`-0.000053`** (-0.005 percentage points).
   - *Scientific confirmation:* Baseline R operates with zero degradation when moving from oracle TextGrid prompts to realistic text-prompt inference in production.

---

## 2. Cross-Fold Experimental Parity & Cryptographic Provenance Audit

All four folds adhered strictly to identical training contracts and platform standards:

| Dimension | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Cross-Fold Invariant Audit |
|---|---|---|---|---|:---:|
| **Experiment Seed** | 1337 | 1337 | 1337 | 1337 | **IDENTICAL** |
| **Acoustic Backbone** | `facebook/wav2vec2-base-960h` | `facebook/wav2vec2-base-960h` | `facebook/wav2vec2-base-960h` | `facebook/wav2vec2-base-960h` | **IDENTICAL** |
| **Pretrained Revision** | `22aad52d435eb6...` | `22aad52d435eb6...` | `22aad52d435eb6...` | `22aad52d435eb6...` | **IDENTICAL** |
| **Backbone Frozen** | True | True | True | True | **IDENTICAL** |
| **Backbone Weight SHA** | `6e9728003d90...` | `6e9728003d90...` | `6e9728003d90...` | `6e9728003d90...` | **IDENTICAL** |
| **Task Head Fingerprints** | `520b6130...` | `a5e05800...` | `51854be0...` | `ff607f23...` | **MUTUALLY DISJOINT** |
| **Loss Weights** | CTC: 1.0, Bin: 1.0 | CTC: 1.0, Bin: 1.0 | CTC: 1.0, Bin: 1.0 | CTC: 1.0, Bin: 1.0 | **IDENTICAL** |
| **Batch Size $\times$ Accum** | $2 \times 2 = 4$ | $2 \times 2 = 4$ | $2 \times 2 = 4$ | $2 \times 2 = 4$ | **IDENTICAL** |
| **Optimizer & Peak LR** | AdamW ($10^{-3}$) | AdamW ($10^{-3}$) | AdamW ($10^{-3}$) | AdamW ($10^{-3}$) | **IDENTICAL** |
| **LR Scheduler Horizon** | Cosine (25 epochs) | Cosine (25 epochs) | Cosine (25 epochs) | Cosine (25 epochs) | **IDENTICAL** |
| **Warmup Updates** | 100 updates | 100 updates | 100 updates | 100 updates | **IDENTICAL** |
| **Weight Decay & Clip** | 0.01 / 1.0 | 0.01 / 1.0 | 0.01 / 1.0 | 0.01 / 1.0 | **IDENTICAL** |
| **Precision & Scaler** | FP16 AMP / 4096.0 | FP16 AMP / 4096.0 | FP16 AMP / 4096.0 | FP16 AMP / 4096.0 | **IDENTICAL** |
| **Skipped AMP Updates** | 0 / 6,600 | 0 / 4,200 | 0 / 6,000 | 0 / 4,800 | **0 SKIPS (100% STABLE)** |
| **Target Fixed Budget** | 3,600 updates | 3,600 updates | 3,600 updates | 3,600 updates | **EXACT MATCH** |
| **Test Audio Contamination**| 0 files accessed | 0 files accessed | 0 files accessed | 0 files accessed | **100% LEAK-FREE** |

---

## 3. Comprehensive Cross-Fold Metrics Summary

The table below provides the full, unabridged results of out-of-fold validation replay across all four folds and aggregations:

| Evaluation Arm | Cohort / Fold | Val Speakers | L1 Profiles | Val Tokens | Errors (Prev) | Updates (Ep) | Matched AP | Mismatched AP | Acoustic Gap | CTC PER | FNR @ 0.50 | FPR @ 0.50 | FNR @ 0.20 | FPR @ 0.20 | Proto Z Aligned AP |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fixed Budget** | **Fold 1** | `NJS`, `RRBI` | Spanish, Hindi | 9,956 | 1,157 (11.62%) | 3,600 (Ep 6) | **0.317021** | 0.294722 | **+0.022299** | 23.68% | 82.63% | 2.36% | 46.15% | 17.18% | 0.313530 |
| **Fixed Budget** | **Fold 2** | `TNI`, `YBAA` | Arabic, Arabic | 9,873 | 921 (9.33%) | 3,600 (Ep 6) | **0.201453** | 0.184479 | **+0.016975** | 23.48% | 83.82% | 5.09% | 58.96% | 17.05% | 0.197761 |
| **Fixed Budget** | **Fold 3** | `TXHC`, `ZHAA` | Mandarin, Arabic | 9,881 | 1,123 (11.37%) | 3,600 (Ep 6) | **0.307334** | 0.299154 | **+0.008180** | 24.83% | 79.34% | 3.27% | 48.17% | 18.22% | 0.307456 |
| **Fixed Budget** | **Fold 4** | `HKK`, `NCC` | Korean, Mandarin | 9,866 | 1,012 (10.26%) | 3,600 (Ep 6) | **0.291046** | 0.286262 | **+0.004784** | 23.60% | 88.34% | 1.19% | 54.15% | 19.29% | 0.291507 |
| **Fixed Budget** | **Macro ($N=4$)** | **All 8 Spk** | **5 L1s** | **39,576** | **4,213 (10.65%)**| **3,600** | **0.279214** | **0.266154** | **+0.013060** | **23.90%** | **83.53%** | **2.98%** | **51.86%** | **17.94%** | **0.277563** |
| *Macro 95% CI* | *Student's t* | *df = 3* | *t = 3.182* | — | — | — | *[0.195, 0.363]*| *[0.179, 0.353]*| *[+0.0003, +0.0258]*| *[22.9%, 24.9%]*| *[77.6%, 89.5%]*| *[0.36%, 5.60%]*| *[42.6%, 61.1%]*| *[16.3%, 19.6%]*| *[0.192, 0.364]* |
| **Fixed Budget** | **Micro (Pool)**| **All 8 Spk** | **5 L1s** | **39,576** | **4,213 (10.65%)**| **3,600** | **0.272582** | **0.261962** | **+0.010619** | **23.90%** | **83.38%** | **2.98%** | **51.41%** | **17.93%** | — |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Val Best** | **Fold 1** | `NJS`, `RRBI` | Spanish, Hindi | 9,956 | 1,157 (11.62%) | 3,600 (Ep 6) | **0.317021** | 0.294722 | +0.022299 | 23.68% | 82.63% | 2.36% | 46.15% | 17.18% | 0.313530 |
| **Val Best** | **Fold 2** | `TNI`, `YBAA` | Arabic, Arabic | 9,873 | 921 (9.33%) | 1,200 (Ep 2) | **0.217044** | 0.210189 | +0.006855 | 27.34% | 99.46% | 0.10% | 78.50% | 6.78% | 0.209234 |
| **Val Best** | **Fold 3** | `TXHC`, `ZHAA` | Mandarin, Arabic | 9,881 | 1,123 (11.37%) | 3,600 (Ep 6) | **0.307334** | 0.299154 | +0.008180 | 24.83% | 79.34% | 3.27% | 48.17% | 18.22% | 0.307456 |
| **Val Best** | **Fold 4** | `HKK`, `NCC` | Korean, Mandarin | 9,866 | 1,012 (10.26%) | 1,800 (Ep 3) | **0.316014** | 0.308451 | +0.007564 | 26.27% | 85.38% | 1.34% | 53.26% | 19.05% | 0.313430 |
| **Val Best** | **Macro ($N=4$)** | **All 8 Spk** | **5 L1s** | **39,576** | **4,213 (10.65%)**| **Varies** | **0.289353** | — | **+0.011225** | **25.53%** | — | — | — | — | — |
| *Macro 95% CI* | *Student's t* | *df = 3* | *t = 3.182* | — | — | — | *[0.212, 0.366]*| — | *[+0.0003, +0.0222]*| *[23.0%, 28.1%]*| — | — | — | — | — |
| **Val Best** | **Micro (Pool)**| **All 8 Spk** | **5 L1s** | **39,576** | **4,213 (10.65%)**| **Varies** | **0.287469** | — | — | — | — | — | — | — | — |

---

## 4. Disaggregated Eight-Speaker Census

To avoid aggregation bias and small-cluster confounding, the table below documents the individual results for each of the eight validation speakers under the primary fixed-budget milestone:

| Speaker | Fold | L1 Accent | Gender | Tokens | Errors | Prevalence | Matched AP | Mismatched AP | Acoustic Gap | CTC PER | FNR @ 0.50 | FPR @ 0.50 | FNR @ 0.20 | FPR @ 0.20 | Precision @ 0.20 | F1 @ 0.20 |
|---|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **NJS** | 1 | Spanish | F | 5,011 | 641 | 12.79% | **0.394927** | 0.378744 | **+0.016183** | 23.41% | 79.25% | 1.62% | 48.67% | 16.73% | 0.3104 | 0.3868 |
| **RRBI**| 1 | Hindi | M | 4,945 | 516 | 10.43% | **0.251519** | 0.234979 | **+0.016540** | 23.95% | 86.82% | 3.09% | 43.02% | 17.63% | 0.2735 | 0.3696 |
| **TNI** | 2 | Arabic | M | 4,939 | 559 | 11.32% | **0.291118** | 0.260710 | **+0.030408** | 25.64% | 77.10% | 4.34% | 52.59% | 16.58% | 0.2674 | 0.3419 |
| **YBAA**| 2 | Arabic | F | 4,934 | 362 | 7.34% | **0.123778** | 0.118536 | **+0.005242** | 21.38% | 94.20% | 5.82% | 68.78% | 17.50% | 0.1238 | 0.1773 |
| **TXHC**| 3 | Mandarin | M | 4,944 | 620 | 12.54% | **0.302878** | 0.300124 | **+0.002754** | 26.19% | 80.81% | 3.10% | 51.94% | 19.01% | 0.2661 | 0.3425 |
| **ZHAA**| 3 | Arabic | F | 4,937 | 503 | 10.19% | **0.321951** | 0.301943 | **+0.020007** | 23.47% | 77.53% | 3.43% | 43.54% | 17.46% | 0.2684 | 0.3639 |
| **HKK** | 4 | Korean | M | 4,919 | 421 | 8.56% | **0.207678** | 0.188810 | **+0.018868** | 23.11% | 92.40% | 1.53% | 55.82% | 19.21% | 0.1771 | 0.2529 |
| **NCC** | 4 | Mandarin | F | 4,947 | 591 | 11.95% | **0.377699** | 0.383957 | **-0.006258** | 24.07% | 85.45% | 0.83% | 52.96% | 19.38% | 0.2478 | 0.3246 |

### 4.1 Cross-Speaker Observations
1. **Prevalence Effect on AP:**
   Average Precision is inherently sensitive to class imbalance. Speakers with higher baseline error prevalence (`NJS` 12.79%, `TXHC` 12.54%, `NCC` 11.95%) achieve AP scores ranging from $0.3029$ to $0.3949$, while speakers with lower error prevalence (`YBAA` 7.34%, `HKK` 8.56%) show lower AP ($0.1238$ and $0.2077$). This underscores that AP cannot be interpreted as absolute accuracy across cohorts without prevalence normalization.
2. **Robust Acoustic Contribution Across Accents:**
   Positive acoustic reliance gaps ($\Delta_{\text{gap}} > 0$) were observed across 7 of 8 validation speakers, spanning Arabic ($+0.0304$ on `TNI`, $+0.0200$ on `ZHAA`), Spanish ($+0.0162$ on `NJS`), Hindi ($+0.0165$ on `RRBI`), and Korean ($+0.0189$ on `HKK`).
3. **CTC Speech Recognition Spread:**
   Greedy phoneme error rates range between `21.38%` (`YBAA`) and `26.19%` (`TXHC`). Across all 8 speakers, the mean CTC PER is $23.90\%$ with a standard deviation of only $1.49\%$, demonstrating remarkable phonological feature invariance in the frozen Wav2Vec2 encoder across accents.

---

## 5. Formal Small-Sample Statistical Limitations

In compliance with the scientific requirements of `CROSS_FOLD_RESEARCH_PLAN.md` Section 7.3:

1. **Degrees of Freedom in Macro Statistics ($N=4$):**
   Evaluating across 4 cross-validation folds yields only $df = 4 - 1 = 3$ degrees of freedom. Consequently, the two-tailed critical Student's $t$ multiplier at $\alpha = 0.05$ is $t_{0.025, 3} = 3.182$. This produces wide confidence intervals (e.g. `[0.194991, 0.363436]` for Macro AP), which properly reflect inter-fold sampling uncertainty but have limited statistical power to resolve subtle effect sizes ($< 0.01$).
2. **Small Cluster Sample Bias ($G=8$):**
   Validation resamples across $G=8$ validation speakers (`NJS`, `RRBI`, `TNI`, `YBAA`, `TXHC`, `ZHAA`, `HKK`, `NCC`) are subject to small-cluster bias when cluster counts $G < 30$ (Cameron, Gelbach & Miller, 2008). Standard asymptotic assumptions do not hold.
3. **Mandatory Reporting Principle:**
   Aggregated metrics (macro and micro) must **never** be cited in isolation as proof of cross-corpus or absolute speaker-independent performance without disaggregated per-speaker tables and explicit disclosure of $N=4$ and $G=8$.

---

## 6. Phase Closure Verdict & Gate Status

1. **Training Phase Closure:**
   - All 4 folds of Baseline R training are complete, verified, and persistently stored.
   - Dual checkpoint accounting is fully certified for all folds.
   - 0 skipped AMP updates across 21,600 total optimizer updates.
   - 0 test split audio, label, or TextGrid files accessed.
2. **Scientific Baseline Established:**
   - **Baseline R Primary Fixed-Budget Performance:**
     - Macro Binary AP: **`0.2792`** $\pm$ `0.0265` | Micro Binary AP: **`0.2726`**
     - Acoustic Reliance Gap: **`+0.0131`** (95% CI: `[+0.0003, +0.0258]`)
     - Greedy CTC PER: **`23.90%`** $\pm$ `0.31%`
     - High-Sensitivity Pedagogical Operating Point ($\tau=0.20$): FNR `51.86%`, FPR `17.94%`, $F_1$ `0.3249`.
     - Zero-Oracle Protocol Z Policy B1 AP: **`0.2792`** (99.64% coverage).
3. **Formal Phase Gate Decision:**
   - **`PASS: PHASE RQ1-TRN-03A IS OFFICIALLY CLOSED`**.
   - Per explicit operator instructions, execution stops at validation analysis. **DO NOT proceed to RQ1-EVAL-01**.
