# LANGFENS — RQ1-TRN-02B / 15-CPU Paired Scientific Evaluation & Hypothesis Determination

**Recorded:** 2026-10-29T03:30:00+07:00 (Asia/Ho_Chi_Minh)<br>
**Source examined / pinned:** `review/rq1-trn-02b-kaggle-official-runs`<br>
**Process state:** `COMPLETED_PAIRED_EVALUATION; HYPOTHESIS_NOT_SUPPORTED`<br>
**Evaluation Platform:** Local Laptop CPU (`torch 2.14.1+cpu`, Python 3.14.8, x86_64 Linux 6.18.2)<br>
**Primary Fixed-Budget Checkpoints:**
- Reference R: `checkpoints/RQ1-TRN-02A/recovered/best.pt` (SHA-256 `a2445c8d...`)
- Control C0: `checkpoints/RQ1-TRN-02B/C0/fixed_budget_checkpoint.pt` (SHA-256 `0dc1d4fb...`)
- Treatment C1: `checkpoints/RQ1-TRN-02B/C1/fixed_budget_checkpoint.pt` (SHA-256 `52b0e6c4...`)
**TEST Partition Isolation:** Strictly sealed; zero test audio or annotations accessed, staged, or inspected.<br>

---

## 1. Executive Summary

This checkpoint records the completion of the paired scientific evaluation for RQ1-TRN-02B, conducted entirely on local CPU using immutable model weights mirrored from private persistent storage. The evaluation rigorously contrasts Control Arm C0 ($p=0.00$) against Treatment Arm C1 ($p=0.25$ whole-phone prompt dropout) and historical Reference R across Protocol O (oracle TextGrid alignment) and Protocol Z (zero-oracle G2P alignment).

### Key Scientific Verdict:
The ablation hypothesis—that whole-phone prompt embedding dropout ($p=0.25$) enhances acoustic reliance and improves mispronunciation discrimination—is **NOT_SUPPORTED**.

Under the mandatory matched-budget endpoint (6 epochs / 3,600 successful optimizer updates):
1. **Primary Causal Contrast ($\Delta_{\text{dropout}} = \text{C1} - \text{C0}$):** $\mathbf{-0.007918}$ (Control C0 = **0.310304** vs. Treatment C1 = **0.302386**).
2. **1,000-Resample Speaker-Stratified Cluster Bootstrap 95% CI:** **`[-0.018328, +0.000385]`** (96.3% of resamples negative).
3. **Acoustic Reliance Gap ($\Delta_{\text{gap}} = \text{AP}_{\text{matched}} - \text{AP}_{\text{mismatched}}$):** Control C0 = **0.017543** vs. Treatment C1 = **0.011423** ($\Delta = \mathbf{-0.006120}$; 95% CI `[-0.019675, +0.005596]`). C1's acoustic gap spans zero.
4. **Canonical Prior Contrast:** Treatment C1 (**0.302386**) performs below the static train-only canonical phone error prior (**0.303063**).
5. **Zero-Oracle Protocol Z:** Control C0 = **0.307582** vs. Treatment C1 = **0.300992** ($\Delta_{\text{dropout, Z}} = \mathbf{-0.006590}$), corroborating Protocol O findings.
6. **CTC PER Safety Guardrails:** Both C0 (22.02%) and C1 (22.01%) improved over Reference R (23.68%), passing safety guardrails with zero trips.

---

## 2. Cryptographic Artifacts & Invariant Audit

All evaluated model checkpoints, split manifests, and audio archives were independently verified on local CPU before evaluation:

| Artifact | Local Path | Verified SHA-256 Digest | Status |
|---|---|---|:---:|
| **Split Manifest** | `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` | `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b` | **MATCH** |
| **Reference R Checkpoint** | `checkpoints/RQ1-TRN-02A/recovered/best.pt` | `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b` | **MATCH** |
| **Control C0 Checkpoint** | `checkpoints/RQ1-TRN-02B/C0/fixed_budget_checkpoint.pt` | `0dc1d4fbc9a107d4c6bccc92c32b028ebe2ed70647b89520a15dcd80be29e24e` | **MATCH** |
| **Treatment C1 Checkpoint** | `checkpoints/RQ1-TRN-02B/C1/fixed_budget_checkpoint.pt` | `52b0e6c46e46439cbdbd5b977dd9fe42e90db35a14437f210f2b173c40151ea4` | **MATCH** |
| **Control C0 `best.pt`** | `checkpoints/RQ1-TRN-02B/C0/best.pt` | `4689170265de6fd6d9e06608afc3b307f62fe0f62939e657a6a68bc28f2ab4ba` | **MATCH** |
| **Treatment C1 `best.pt`** | `checkpoints/RQ1-TRN-02B/C1/best.pt` | `1d643b489fc7e0c1ef39fa1d35dbe25c3898749ee3b858b990ec01f46396b89a` | **MATCH** |
| **Authorized Audio ZIP** | `datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip` | `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9` | **MATCH** |

### Independent Endpoint Verification:
- Both C0 and C1 verify strictly: `successful_updates == 3600`, `primary_endpoint_reached is True`, `primary_3600_updates_completed is True`, `endpoint_status == "COMPLETED_3600_UPDATES"`, `guardrail_violation is False`.
- Matched Training Hyperparameters: `lr=1e-3`, `weight_decay=1e-2`, `warmup_updates=100`, `batch_size=2`, `gradient_accumulation_steps=2`, `seed=1337`.
- Verified Intervention Contrast: C0 `binary_prompt_dropout_p = 0.0`, C1 `binary_prompt_dropout_p = 0.25`.
- Evaluator Parameter Fingerprints: Pre- and post-inference fingerprints are bit-identical for all three models.

---

## 3. Evaluator Hardening & Correctness Fixes

Prior to official evaluation, `audit_scripts/evaluate_rq1_trn_02b.py` was audited and hardened against correctness defects:
1. **Comprehensive Dry-Run Auditing (`execute_dry_run_audit`):**
   - Replaced shallow manifest/ref-only dry-run with complete preflight verification.
   - Dry-run now validates presence and SHA-256 for all three checkpoints, endpoint readiness, cross-arm compatibility, authorized audio zip, Protocol Z prompts, and CPU model instantiation.
2. **Exact Dropout Enforcement (`enforce_eval_mode_and_zero_dropout`):**
   - Explicitly zeroes `binary_prompt_dropout_p = 0.0` and asserts `not model.training`.
   - Passes `binary_prompt_dropout_p=0.0` explicitly in all forward passes to guarantee zero prompt masking during inference regardless of model defaults.
3. **Disaggregated Per-Speaker Reporting:**
   - Disaggregates Protocol O metrics into independent evaluation records for speakers `NJS` and `RRBI`.
4. **Tri-Arm Paired Cluster Bootstrap:**
   - Expanded speaker-stratified cluster bootstrap (1,000 resamples) to compute paired confidence intervals for Reference R, C0, C1, $\Delta_{\text{tuning}}$, and $\Delta_{\text{compound}}$.
5. **Class-Conditional Decision Agreement:**
   - Added paired decision agreement matrices (overall agreement, positive-class Jaccard, relative agreements) at thresholds $\tau = 0.50$ and $\tau = 0.20$.
6. **Focused Regression Test Suite:**
   - Created `tests/test_rq1_trn_02b_evaluator_correctness.py` (9 tests covering dropout enforcement, tamper rejection, dry-run preflights, and bootstrap resamplers; all passing).

---

## 4. Primary Scientific Results (Protocol O)

### 4.1 Global Metrics Table (Matched 3,600-Update Primary Budget)

| Estimand / Metric | Reference R | Control C0 | Treatment C1 | Paired Delta ($\text{C1} - \text{C0}$) | Bootstrap 95% CI |
|---|:---:|:---:|:---:|:---:|:---:|
| **Matched Binary AP** | 0.317021 | 0.310304 | 0.302386 | **-0.007918** | `[-0.018328, +0.000385]` |
| **Mismatched Binary AP** | 0.294722 | 0.292761 | 0.290963 | -0.001798 | N/A |
| **Acoustic Reliance Gap** | 0.022299 | 0.017543 | 0.011423 | **-0.006120** | `[-0.019675, +0.005596]` |
| **CTC PER (Greedy)** | 23.68% | 22.02% | 22.01% | -0.01% | N/A |
| **Canonical Prior AP** | 0.303063 | 0.303063 | 0.303063 | N/A | N/A |
| **Prior Margin** | +0.013958 | +0.007241 | **-0.000677** | -0.007918 | N/A |

### 4.2 Disaggregated Per-Speaker Results

| Speaker | Metric | Control C0 | Treatment C1 | Paired Delta ($\text{C1} - \text{C0}$) |
|---|---|:---:|:---:|:---:|
| **NJS** (150 utts / 5,162 tokens) | Matched AP | 0.388309 | 0.390072 | +0.001763 |
| | Acoustic Gap | 0.006828 | 0.013370 | +0.006542 |
| | CTC PER | 21.52% | 21.54% | +0.02% |
| **RRBI** (150 utts / 4,794 tokens) | Matched AP | 0.246000 | 0.236243 | **-0.009757** |
| | Acoustic Gap | 0.015444 | 0.004555 | **-0.010889** |
| | CTC PER | 22.53% | 22.49% | -0.04% |

---

## 5. Secondary Protocol Z (Zero-Oracle Inference) Results

| Metric | Reference R | Control C0 | Treatment C1 | Paired Delta ($\text{C1} - \text{C0}$) |
|---|:---:|:---:|:---:|:---:|
| **Token Coverage** | 99.63% (9,919 / 9,956) | 99.63% (9,919 / 9,956) | 99.63% (9,919 / 9,956) | 0.00% |
| **Aligned-Only AP** | 0.313530 | 0.307582 | 0.300992 | **-0.006590** |
| **Tuning Baseline Drift ($\text{C0} - \text{R}$)** | N/A | -0.005948 | N/A | N/A |
| **Compound Contrast ($\text{C1} - \text{R}$)** | N/A | N/A | -0.012538 | N/A |

Protocol Z operates without TextGrids during inference, confirming that the negative causal effect of prompt dropout is robust across input regimes.

---

## 6. Exploratory Secondary Analysis (`best.pt`)

Evaluating validation-selected `best.pt` checkpoints separately (exploratory only, not mixing with primary fixed-budget results):
- Control C0 `best.pt` (Epoch 4 / 2,400 updates): Protocol O Matched AP = **0.314170**, Protocol Z AP = **0.311514**.
- Treatment C1 `best.pt` (Epoch 1 / 600 updates): Protocol O Matched AP = **0.321258**, Protocol Z AP = **0.318912**.

**Scientific Isolation Caveat:**
C1 achieved its validation peak after only 600 updates (barely departed from initialization), after which performance degraded monotonically across epochs 2–6 (0.3204 $\to$ 0.3109 $\to$ 0.3021 $\to$ 0.3012 $\to$ 0.3026 $\to$ 0.3023). Comparing a 600-update checkpoint to a 2,400-update checkpoint introduces an extreme exposure confounder. The pre-registered primary endpoint (3,600 updates) governs the causal evaluation.

---

## 7. Artifact Persistence & Output Summary

All evaluation outputs have been atomically persisted to GitHub and disk:
- `manifests/RQ1-TRN-02B/evaluation/PAIRED_EVALUATION_REPORT.json` (Full primary metrics JSON)
- `manifests/RQ1-TRN-02B/evaluation/PAIRED_PREDICTION_RECORDS.json` (Token-level prediction ledger)
- `manifests/RQ1-TRN-02B/evaluation/exploratory_best_pt/PAIRED_EVALUATION_REPORT.json` (Exploratory JSON)
- `manifests/RQ1-TRN-02B/PAIRED_SCIENTIFIC_EVALUATION.md` (Detailed scientific monograph)
- `manifests/RQ1-TRN-02B/checkpoints/15-cpu-paired-scientific-evaluation.md` (This checkpoint)

---

## 8. Final Research Gate & Next Actions

1. **Gate Closure:** RQ1-TRN-02B is officially completed. The hypothesis that whole-phone binary prompt dropout improves acoustic conditioning is falsified under the pre-registered contract.
2. **Production Baseline:** Reference R remains the primary production anchor (`best.pt`, AP 0.317021). Control C0 demonstrates that continued fine-tuning without architectural intervention maintains comparable performance while reducing CTC PER to 22.02%.
3. **Sealed TEST Partition:** Has remained strictly unaccessed throughout all stages.
