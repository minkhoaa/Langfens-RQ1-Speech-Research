# LANGFENS — RQ1-TRN-03A / 02-Cross-Fold Protocol Parity & Isolation Corrections

**Status:** `COMPLETED`

**Recorded at:** 2026-10-29T16:05:00+07:00

**Evidence source:** Commits `05796e25b1f745ddb7772e0d5bda2edd5d0a7f1c`, `configs/rq1_trn_02a_fold1.yaml`, `audit_scripts/train_fold1_ctc_binary.py`, `tests/test_rq1_trn_03a_protocol_parity_and_isolation.py`

**Evidence level:** `INDEPENDENTLY_VERIFIED`

**Branch:** `review/rq1-trn-03a-cross-fold-planning`

---

## 1. Objective
Execute a rigorous protocol audit and correction for Phase RQ1-TRN-03A before any code implementation or GPU training occurs. Restore exact hyperparameter parity with Fold 1 Baseline R, establish precise per-fold test isolation while removing contradictory global sealing claims, distinguish fixed-budget from validation-selected checkpoints, preserve the deterministic within-speaker shift-1 mismatched acoustic protocol, clarify confusion metrics and small-sample bootstrap limitations, correct thesis scope boundaries, and add automated regression tests.

---

## 2. Inputs & Dependencies
- Base commit: `05796e25b1f745ddb7772e0d5bda2edd5d0a7f1c` on branch `review/rq1-trn-03a-cross-fold-planning`.
- Pinned Fold 1 Baseline R config: `configs/rq1_trn_02a_fold1.yaml`.
- Pinned Fold 1 Baseline R training driver: `audit_scripts/train_fold1_ctc_binary.py`.
- Master split manifest: `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` (SHA-256 `7d6bfdf6...`).
- Master policy: `AGENTS.md`.

---

## 3. Audit Findings & Implemented Corrections

### 3.1 Baseline R Hyperparameter Parity Audit
An exhaustive audit of the initial draft against `configs/rq1_trn_02a_fold1.yaml` and `train_fold1_ctc_binary.py` identified and corrected four major discrepancies:
1. **Master Seed:** Corrected from `seed: 42` to `seed: 1337` to guarantee identical pseudorandom parameter initialization behavior.
2. **Micro Batch Size & Accumulation:** Corrected from `batch_size: 4, accum: 1` to `batch_size: 2, gradient_accumulation_steps: 2` (effective batch size of 4 utterances per optimizer update), preserving the exact gradient variance dynamics.
3. **Warmup Updates:** Corrected from "10% of steps" to strictly `warmup_updates: 100` fixed updates.
4. **Scheduler Horizon:** Corrected total updates from 3,600 to `total_updates = math.ceil(len(train_loader)/accum) * max_epochs = 600 * 25 = 15,000 updates`, matching the learning rate decay rate of Baseline R.

### 3.2 Dual Checkpoint Accounting Semantics
To prevent confounding training exposure with model performance:
- **Fixed-Budget Checkpoint:** Captured at strictly Epoch 6 / 3,600 updates, directly matching C0/C1 fixed exposure and Fold 1 peak selection point.
- **Validation-Selected Checkpoint (`best.pt`):** Selected at peak validation AP prior to early stopping (patience 5 on validation AP with `blank_fraction < 0.99`).
- Both checkpoints will be saved, tracked, and reported independently.

### 3.3 Precise Per-Fold Test Isolation & Refutation of Global Sealing
- Refuted erroneous claims that all 4 test sets are globally sealed from staging archives. In standard 4-fold cross-validation, the test speakers of Fold 1 ($S_{\text{test}}^{(1)}$) legitimately and necessarily serve as training speakers for Folds 2, 3, and 4.
- Formally defined per-fold isolation: For model $M_k$, $S_{\text{test}}^{(k)}$ is strictly excluded from $M_k$'s training set, validation set, early stopping / checkpoint selection, threshold calibration, and remote staging archive (`authorized_RQ1-TRN_fold{k}_train_val_only.zip`).
- All test sets ($S_{\text{test}}^{(1..4)}$) remain 100% un-evaluated during the cross-fold validation phase, preventing model selection leakage.

### 3.4 Preservation of Deterministic Shift-1 Acoustic Protocol
- Reaffirmed `method="within_speaker_cyclic"` via `generate_derangement_indices()`.
- Proved that within each speaker segment, utterance $i$ is permuted deterministically to $(i+1) \bmod L$ within the *same* speaker, maintaining speaker identity, vocal tract geometry, and recording channel while destroying prompt phonetic alignment.

### 3.5 Disambiguation of Confusion Metrics & Small-Sample Limitations
- Formally defined $\text{Recall} = \frac{TP}{TP + FN}$ and $\text{FNR} = \frac{FN}{TP + FN} = 1 - \text{Recall}$ (missed error rate).
- Formally defined $\text{Specificity} = \frac{TN}{TN + FP}$ and $\text{FPR} = \frac{FP}{TN + FP} = 1 - \text{Specificity}$ (false alarm rate).
- Documented that Macro Student's $t$ CI with $df=3$ produces wide intervals due to small $N=4$ fold sample size.
- Documented that cluster bootstrapping over $G=8$ validation speakers has known small-cluster bias ($G < 30$).
- Retracted unsubstantiated significance claims ($p < 10^{-5}$) on CTC PER differences.

### 3.6 Correction of Thesis Scope Boundaries
- Confirmed that SpeechOcean762 cross-corpus evaluation and pedagogical pronunciation feedback belong to RQ1.
- Confirmed that RQ2 is reserved exclusively for Multi-Agent Automated Test Generation.

---

## 4. Results & Verification
- **New Regression Test Suite:** `tests/test_rq1_trn_03a_protocol_parity_and_isolation.py` passes 8 tests in 5.78s:
  - `test_per_fold_partition_disjointness`: PASS
  - `test_cross_fold_speaker_rotation`: PASS
  - `test_demographic_stratification`: PASS
  - `test_baseline_r_exact_hyperparameter_contract`: PASS
  - `test_scheduler_horizon_computation`: PASS
  - `test_within_speaker_cyclic_derangement`: PASS
  - `test_fnr_fpr_precision_recall_relationships`: PASS
  - `test_checkpoint_distinction_semantics`: PASS
- **Repository Regression Suite:** `PYTHONPATH=. .venv/bin/pytest tests/` passed 244 tests with 0 failures in 148.07s.
- **Physical Test Isolation Maintained:** Zero test files accessed, extracted, or staged.

---

## 5. Output Artifacts
- Corrected Research Plan: `manifests/RQ1-TRN-03A/CROSS_FOLD_RESEARCH_PLAN.md`
- Phase Overview Checkpoint: `manifests/RQ1-TRN-03A/PROCESS_CHECKPOINT.md`
- Step Checkpoint: `manifests/RQ1-TRN-03A/checkpoints/02-cross-fold-protocol-parity-and-isolation-corrections.md`
- Policy Document: `AGENTS.md` (Per-fold test sealing & baseline contract updates)
- Regression Test Suite: `tests/test_rq1_trn_03a_protocol_parity_and_isolation.py`
- Global Registry: `docs/process-checkpoints/INDEX.md`

---

## 6. Restart / Recovery Instructions
- Review corrected `CROSS_FOLD_RESEARCH_PLAN.md`.
- Static protocol correction complete; all regression tests passing.
- Training remains blocked pending explicit operator approval of Section 10 action packet.

---

## 7. Remaining Work & Safety Boundaries
- **Prohibited Actions:** No local training, no test set extraction, no GPU job provisioning without operator authorization.
- **Next Step:** Obtain operator approval on the 5-item action packet; implement `audit_scripts/build_fold_packages.py` and fold training configs.
