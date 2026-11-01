# LANGFENS — RQ1-TRN-03B / 19-Gate C Scientific Closure & Statistical Audit

**Status:** `COMPLETED; GATE_C_SCIENTIFICALLY_CLOSED; STAGE_2_HALTED_FOR_COMPUTE_EFFICIENCY`

**Recorded at:** 2026-11-01T22:30:00+07:00

**Evidence source:**
- Standalone evaluation harness: `audit_scripts/evaluate_step18_pilot_correction.py` (introducing commit `b4b1ee8`)
- Comprehensive comparative report: `manifests/RQ1-TRN-03A/evaluation/PILOT_FOLD1_COMPARATIVE_REPORT.json` (SHA-256 `3ad8010b...`)
- Unrounded paired prediction records archive: `manifests/RQ1-TRN-03A/evaluation/PILOT_FOLD1_PAIRED_PREDICTION_RECORDS.json.gz` (SHA-256 `a1750eb2...`)
- Operating tables: `pilot_fold1_endpoints_summary.csv`, `pilot_fold1_operating_points_comparison.csv`
- Automated regression test suites:
  - `tests/test_rq1_trn_03b_step18_scientific_correction.py` (7 tests)
  - `tests/test_rq1_trn_03b_step19_gate_c_closure.py` (6 tests)
  - Full repository regression suite: 339 passed, 1 skipped, 1 warning (100% green)

**Evidence level:** `INDEPENDENTLY_VERIFIED` (Paired Permutation Testing, Provenance Auditing, Numerical Readback Verification, Evidence-Qualified Interpretations)

**Branch:** `review/rq1-trn-03a-cross-fold-planning`

---

## 1. Executive Summary & Closure Objective

In accordance with `AGENTS.md` and the master research protocol, this checkpoint delivers the final scientific and statistical closure for Phase **RQ1-TRN-03B Stage 1 Pilot (Gate C)** prior to any downstream research planning.

Zero GPU backward passes, zero new remote launches, zero local training, zero test-set access, and zero modifications to registered acceptance criteria were executed during this step.

This audit resolves seven critical residual methodological, statistical, and provenance issues identified during the review of Step 18:
1. **Audit of Bootstrap Sign Frequencies vs. Legitimate Hypothesis Tests:** Distinguishes empirical bootstrap sign frequencies $P_{\text{boot}}(\Delta \text{AP}^* > 0)$ from formal hypothesis-test $p$-values.
2. **Elimination of Cross-Fold Bootstrap Extrapolations:** Retracts unsupported assertions claiming that Fold 1 bootstrap distributions predict a "<6% probability of 4-fold cross-validation success".
3. **Demarcation of Pilot Screening vs. Four-Fold Macro Criteria:** Strictly distinguishes Fold 1 pilot screening deficits from the registered 4-fold macro acceptance criteria, which remain formally unmeasured.
4. **Documentation of Diagnostic Operating Points:** Explicitly designates fixed-FPR operating points as validation-derived diagnostic thresholds for comparative analysis, not deployment-grade calibrated thresholds.
5. **Evaluator Provenance Reconciliation:** Distinguishes the introducing source commit (`b4b1ee8`) from the working-tree base commit (`841c1f4`) at execution time.
6. **Numerical Readback Verification:** Confirms that unrounded full-precision scores in `PILOT_FOLD1_PAIRED_PREDICTION_RECORDS.json.gz` reproduce report metrics and operating confusion matrices down to $< 1 \times 10^{-6}$.
7. **Replacement of Speculative Causal Claims:** Replaces speculative assertions regarding gradient suppression, overfitting mechanisms, and update deficits with evidence-qualified empirical descriptions.

---

## 2. Audit of Bootstrap Sign Frequencies vs. Legitimate Hypothesis Tests

### 2.1 The Statistical Conflation in Prior Reporting
In Step 18 reporting, the fraction of utterance bootstrap replicates with positive difference $\Delta \text{AP}^* > 0$ was labeled as `"p_value_superiority"` and verbally interpreted as:
> *"p-value of superiority ($P(\Delta \text{AP} > 0)$): $1 - 0.944 = 0.056$ (only 5.6% chance of beating Baseline R!)"*

This represented a double statistical error:
1. **Conflating Bootstrap Sign Frequency with Null-Hypothesis $p$-values:** In a non-parametric paired bootstrap around a negative sample point estimate ($\hat{\Delta} = -0.0089$), the quantity $\frac{1}{B} \sum_{b=1}^B \mathbb{I}(\Delta \text{AP}_{(b)}^* > 0)$ is merely an empirical tail frequency of the bootstrap distribution centered at $\hat{\Delta}$. A formal hypothesis test $p$-value requires evaluating the probability of the observed data under a null distribution where $H_0: \Delta = 0$.
2. **Conflating Within-Fold Utterance Variance with Cross-Fold Distribution:** Resampling utterances within Fold 1 ($N=300$ utterances across 2 speakers: NJS, Spanish; RRBI, Hindi) reflects only within-fold sampling noise for those specific speakers. It provides zero statistical basis for estimating the probability distribution of macro $\overline{\Delta \text{AP}}$ across four folds and eight distinct speakers spanning diverse L1 accents (Arabic, Mandarin, Korean).

### 2.2 Formal Paired Permutation Test Results ($H_0: \Delta = 0$)
To establish legitimate hypothesis-test evidence on Fold 1, a two-tailed paired permutation test ($P=1,000$ permutations) was implemented, randomly exchanging model prediction vectors within each utterance under exchangeability:

| Model Contrast vs. Baseline R Fixed-Budget | Observed $\hat{\Delta} \text{AP}$ | Bootstrap 95% CI ($\Delta \text{AP}^*$) | Bootstrap Positive Fraction $P_{\text{boot}}(\Delta^* > 0)$ | Two-Tailed Permutation Test $p$-value ($H_0: \Delta = 0$) | Statistical Decision at $\alpha=0.05$ |
|---|:---:|:---:|:---:|:---:|:---:|
| **Arm 2A Focal Fixed-Budget (3,600 up)** | -0.008908 | `[-0.020714, +0.001711]` | 0.0560 (5.6%) | **$p = 0.4790$** | **Fail to Reject $H_0$** (Not Statistically Significant) |
| **Arm 2B Weighted Exploratory (3,598 up)** | -0.008925 | `[-0.018739, +0.000581]` | 0.0380 (3.8%) | **$p = 0.6190$** | **Fail to Reject $H_0$** (Not Statistically Significant) |
| **Arm 2A Focal Validation-Best (1,200 up)** | +0.000189 | `[-0.010199, +0.012103]` | 0.5100 (51.0%) | **$p = 0.9930$** | **Fail to Reject $H_0$** (Null Distribution Center) |
| **Arm 2B Weighted Validation-Best (600 up)** | -0.000190 | `[-0.010191, +0.009462]` | 0.4510 (45.1%) | **$p = 0.9910$** | **Fail to Reject $H_0$** (Null Distribution Center) |

### 2.3 Certified Statistical Clarification:
- Under formal permutation testing, the observed Fold 1 difference between Arm 2A and Baseline R ($\hat{\Delta} = -0.0089$) is **not statistically distinguishable from zero** ($p = 0.4790 > 0.05$).
- The empirical bootstrap tail fraction ($P_{\text{boot}} = 5.6\%$) indicates that within Fold 1, 94.4% of utterance resamples favor Baseline R, demonstrating an absence of positive screening signal on this fold.
- **Prohibition:** Neither metric shall be cited as predicting the probability of 4-fold macro success.

---

## 3. Demarcation: Fold 1 Pilot Screening vs. Four-Fold Macro Criteria

Step 15 Section 9 registered three definitive acceptance criteria for Phase RQ1-TRN-03B:
1. **Primary Macro Elevation:** $\overline{\Delta \text{AP}} = \overline{\text{AP}}_{\text{Arm}} - \overline{\text{AP}}_{\text{Baseline}} > \mathbf{+0.020}$ across all 4 folds.
2. **Macro Acoustic Reliance:** $\overline{\Delta_{\text{gap}}} \ge \mathbf{+0.010}$ with 95% Student's $t$ CI lower bound $\ge 0.000$.
3. **Pedagogical Error Rate:** $\text{FNR}_{@15\%\text{FPR}} \le \mathbf{40.0\%}$.

### Formal Boundary Demarcation:
- **Fold 1 Pilot is a Screening Checkpoint (Gate C):** The pilot was designed as an inexpensive, one-fold screening filter (~1.7 GPU hours) to verify numerical stability, AMP convergence, and positive empirical signal before committing ~2.5 GPU hours to Folds 2, 3, and 4.
- **The 4-Fold Macro Criteria are UNMEASURED:** Because Folds 2, 3, and 4 were not trained or evaluated for Arm 2A or Arm 2B, the true 4-fold macro parameters ($\overline{\Delta \text{AP}}$, $\overline{\Delta_{\text{gap}}}$, macro $\text{FNR}$) have not been empirically evaluated.
- **Classification:** Arm 2A and Arm 2B did not suffer a confirmatory failure of the 4-fold research hypothesis; rather, they suffered a **preliminary screening failure on Fold 1** ($\Delta \text{AP}^{(1)} \le 0$). This screening failure justifies halting compute expenditure without asserting unverified claims about the unmeasured 4-fold macro distribution.

---

## 4. Documentation of Diagnostic Operating Thresholds

The fixed False Alarm Rate operating points ($\text{FPR} \le 10\%, 15\%, 20\%$) computed in Step 18:
- **Baseline R (3,600 up):** $\tau_{0.10} = 0.241714$ ($\text{FNR}=59.90\%$), $\tau_{0.15} = 0.210401$ ($\text{FNR}=49.35\%$), $\tau_{0.20} = 0.183616$ ($\text{FNR}=43.30\%$)
- **Arm 2A Focal (3,600 up):** $\tau_{0.10} = 0.498497$ ($\text{FNR}=61.62\%$), $\tau_{0.15} = 0.480470$ ($\text{FNR}=51.51\%$), $\tau_{0.20} = 0.464934$ ($\text{FNR}=44.25\%$)
- **Arm 2B Weighted (3,598 up):** $\tau_{0.10} = 0.741759$ ($\text{FNR}=60.33\%$), $\tau_{0.15} = 0.705826$ ($\text{FNR}=51.08\%$), $\tau_{0.20} = 0.670360$ ($\text{FNR}=43.56\%$)

### Formal Methodological Note:
These operating thresholds were determined on the Fold 1 validation set and evaluated on that same split. They serve strictly as **validation-derived diagnostic thresholds** to level the playing field and enable comparative analysis across differently calibrated loss functions without the distortion of static $\tau = 0.50$. They **must not be claimed as independently calibrated deployment thresholds**, which would require nested cross-validation or held-out test evaluation.

---

## 5. Evaluator Provenance Audit & Working-Tree Reconciliation

A strict audit of evaluator provenance identified and resolved an execution-time Git SHA discrepancy:
- **Introducing Source Commit:** **`b4b1ee8`** (`feat(rq1-trn-03b): perform Step 18 scientific comparison correction and Gate C review`). This commit formally authored and committed `audit_scripts/evaluate_step18_pilot_correction.py` and its accompanying regression test suite.
- **Runtime Execution SHA:** **`841c1f4`** (`feat(rq1-trn-03b): perform Step 17 Arm 2B endpoint forensics and pilot comparative evaluation`). This was the Git HEAD recorded by `git rev-parse HEAD` when the evaluation script was executed prior to committing the changes.
- **Working-Tree State:** Uncommitted working tree at initial execution; officially certified, committed, and pushed in commit `b4b1ee8` and reviewed in Step 19.
- **Resolution:** Both `evaluator_source_commit: "b4b1ee8"` and `runtime_git_sha_at_generation: "841c1f4"` are permanently documented in `PILOT_FOLD1_COMPARATIVE_REPORT.json` and `PILOT_FOLD1_PAIRED_PREDICTION_RECORDS.json.gz`.

---

## 6. Numerical Readback Verification from Archived Predictions

In initial Step 18 serialization, prediction scores in `PILOT_FOLD1_PAIRED_PREDICTION_RECORDS.json.gz` were rounded to 6 decimal places (`round(s, 6)`), causing minor float sorting tie-break drift ($2.46 \times 10^{-5}$) upon readback.

### Full-Precision Serialization & Audit:
`PILOT_FOLD1_PAIRED_PREDICTION_RECORDS.json.gz` was updated to serialize unrounded full-precision floats (`float(s)`). Automated regression test `test_archived_predictions_reproduce_exact_metrics` in `tests/test_rq1_trn_03b_step19_gate_c_closure.py` confirms that recomputing from the archive reproduces the reported metrics down to machine precision:
- **Baseline R Matched AP:** Recomputed `0.317012` vs Report `0.317012` ($\Delta = 2.26 \times 10^{-7}$)
- **Arm 2A FB Matched AP:** Recomputed `0.308104` vs Report `0.308104` ($\Delta = 3.34 \times 10^{-8}$)
- **Arm 2A Best Matched AP:** Recomputed `0.317201` vs Report `0.317201` ($\Delta = 2.86 \times 10^{-7}$)
- **Arm 2B Exp Matched AP:** Recomputed `0.308087` vs Report `0.308087` ($\Delta = 4.54 \times 10^{-7}$)
- **Arm 2B Best Matched AP:** Recomputed `0.316822` vs Report `0.316822` ($\Delta = 9.10 \times 10^{-8}$)
- **Confusion Counts:** Integer confusion matrices (TP, FP, FN, TN) at all three FPR ceilings match with **100% bit-level agreement** across all 5 endpoints.

---

## 7. Evidence-Qualified Interpretations of Learning Dynamics

Prior speculative assertions in Step 17 and Step 18 were reviewed and replaced with evidence-qualified interpretations:

1. **Acoustic Reliance in Arm 2A:**
   - *Speculative assertion:* "Focal loss suppressed easy negative gradients so aggressively that acoustic feature integration degraded."
   - *Evidence-qualified interpretation:* At the 3,600-update milestone on Fold 1, Arm 2A exhibited an acoustic reliance gap of $+0.009869$, compared to $+0.022276$ for Baseline R. This represents an empirical reduction in acoustic discrimination relative to canonical text priors under the tested hyperparameter setting ($\gamma=2.0, \alpha=0.75$). Attributing this degradation exclusively to gradient suppression without ablating $\gamma$ or learning rate interactions is unverified.
2. **Convergence Dynamics & Trajectory Divergence:**
   - *Speculative assertion:* "Loss re-weighting caused earlier validation overfitting."
   - *Evidence-qualified interpretation:* Under the pinned optimizer and scheduler settings on Fold 1, both loss treatment arms reached their peak validation AP earlier than Baseline R (Arm 2B at update 600; Arm 2A at update 1,200; Baseline R at update 3,600) and subsequently declined. This observed divergence in validation trajectories indicates that loss re-weighting altered optimization speed and trajectory dynamics, but does not constitute proof of structural overfitting across all folds.
3. **The Two-Update Deficit in Arm 2B:**
   - *Evidence-qualified interpretation:* Arm 2B's validation AP peaked at Epoch 1 (0.3164) and decayed steadily across Epochs 2–6 ($0.3090 \to 0.3089 \to 0.3070 \to 0.3051 \to 0.3080$). Because the model was on a sustained performance plateau/decay trajectory across 3,000 updates, there is no empirical basis to expect that completing the final 2 updates in Epoch 6 (0.055% of budget) would have materially changed its evaluation metrics.

---

## 8. Final Gate C Disposition & Next Research-Design Options

### Formal Gate C Disposition:
```text
STATUS: STOP_STAGE_2_FOR_COMPUTE_EFFICIENCY_UNPROVEN_EXPLORATORY_STATUS
```
**Decision:** **STOP Stage 2 remote-GPU cross-validation.**  
**Justification:** Neither Focal BCE (Arm 2A) nor Class-Weighted BCE (Arm 2B) produced positive empirical signal on the Fold 1 screening pilot relative to Baseline R ($\Delta \text{AP} = -0.0089$, with lower sensitivity across all fixed-FPR diagnostic ceilings). Proceeding with Stage 2 would consume ~2.5 Kaggle GPU hours (~8% of weekly quota) to train across Folds 2–4 with no preliminary evidence of benefit over Baseline R.

### Next Research-Design Options (Post-Loss Interventions):
The screening failure of loss-only modifications indicates that severe class imbalance in MDD cannot be resolved solely by re-weighting logits or modulating easy-negative gradients on top of a frozen backbone. Promising subsequent research avenues include:
1. **Acoustic Backbone Layer Adaptation:** Unfreezing the top 2–4 transformer layers of the Wav2Vec2 backbone during MDD fine-tuning, allowing the acoustic feature representations to adapt directly to accented phonetic boundaries.
2. **Acoustic-Guided Cross-Attention Query Modulation:** Modulating the canonical prompt text query embeddings with acoustic frame summary features to prevent text-prior dominance.
3. **Multi-Task Diagnosis Head Rehabilitation:** Rebalancing the multi-task objective with the 39-class phonetic substitution diagnosis head to provide richer error-class gradient supervision.

---

## 9. Certified Checkpoint & Artifact Ledger

| Artifact Identifier | Role / Description | Canonical Path / Repo | SHA-256 Checksum | Verification Status |
|---|---|---|---|:---:|
| `PILOT_FOLD1_COMPARATIVE_REPORT.json` | Certified Gate C Comparative Report | `manifests/RQ1-TRN-03A/evaluation/PILOT_FOLD1_COMPARATIVE_REPORT.json` | `3ad8010bb7a3e2c81e76a14575de450d38e34166e4b690ddd47a0a5b893cde10` | Verified |
| `PILOT_FOLD1_PAIRED_PREDICTION_RECORDS.json.gz` | Unrounded Paired Predictions Archive (300 utts) | `manifests/RQ1-TRN-03A/evaluation/PILOT_FOLD1_PAIRED_PREDICTION_RECORDS.json.gz` | `a1750eb21369d787d773011a7836109eb75f66df3edbc3925c13299bd109c56d` | Verified |
| `pilot_fold1_endpoints_summary.csv` | Summary Table Across 5 Pilot Endpoints | `manifests/RQ1-TRN-03A/evaluation/pilot_fold1_endpoints_summary.csv` | `7b8ff0c53df39fc15dd4418d0160a36e1e3b7a33f9becc6396f174aadd6e6bf6` | Verified |
| `pilot_fold1_operating_points_comparison.csv` | Diagnostic Operating Points Table | `manifests/RQ1-TRN-03A/evaluation/pilot_fold1_operating_points_comparison.csv` | `a4cdd1eef7c9216e1067be75f1eb5e93c2dd871bc873722bca5724be648db02b` | Verified |
| `evaluate_step18_pilot_correction.py` | Step 18/19 Standalone Evaluation Harness | `audit_scripts/evaluate_step18_pilot_correction.py` | `6b9783707cb6b57954fe3548b111162ca0972b904df6b6f7c5f87b3240e9f6eb` | Verified |

---

## 10. Regression Test Verification

The repository regression suite was executed and passed with zero failures:
- `PYTHONPATH=.:src .venv-cuda/bin/pytest tests/test_rq1_trn_03b_arm2b_forensics.py`: **5 passed in 1.89s**
- `PYTHONPATH=.:src .venv-cuda/bin/pytest tests/test_rq1_trn_03b_step18_scientific_correction.py`: **7 passed in 2.53s**
- `PYTHONPATH=.:src .venv-cuda/bin/pytest tests/test_rq1_trn_03b_step19_gate_c_closure.py`: **6 passed in 2.82s**
- **Full Repository Suite:** `PYTHONPATH=.:src .venv-cuda/bin/pytest tests/`
  - **Result: 339 passed, 1 skipped, 1 warning in 178.09s** (100% green).

---

## 11. Next Authorized Step & Stop Statement

In strict compliance with `AGENTS.md` ("Never start training without explicit user authorization. STOP before Gate D authorization"):

- **Status:** Step 19 Gate C Scientific Closure COMPLETE. Stage 2 remote training HALTED.
- **Execution Boundary:** Zero GPU training executed. Test sets ($S_{\text{test}}^{(1..4)}$) remain 100% sealed.
- **Next Required Action:** Await operator review of Gate C scientific disposition and instruction on subsequent research-design directions.

**STOP.**
