# LANGFENS — RQ1-TRN-03B / 16B-Pilot Commit Pinning Hotfix

**Status:** `COMPLETED; GATE_B_HOTFIX_VERIFIED; EXECUTION_HALTED_PENDING_OPERATOR_AUTHORIZATION`

**Recorded at:** 2026-11-01T17:00:00+07:00

**Evidence source:** 
- Git commit ancestry (`git show 051a870c...` vs `git show 56bf7953...`)
- Reviewed base commit: `56bf79535e4f697b2eea05594d5d81153e04dbff`
- Corrected pilot notebooks in `deploy/kaggle_pilot_arm2{a,b}_{preflight,smoke,train}/`
- Automated regression suite (`tests/test_rq1_trn_03b_kaggle_pilot_deployment.py`, 321/321 passed repository tests)

**Evidence level:** `INDEPENDENTLY_VERIFIED` (Automated Git Ancestry Audits, Notebook Invariant Inspections & Fail-Closed Regression Suites)

**Branch:** `review/rq1-trn-03a-cross-fold-planning`

---

## 1. Executive Summary & Defect Identification

During preflight audit of the Step 16 deployment assets prior to remote Kaggle GPU dispatch, a critical commit pinning mismatch was identified:

- **Defect:** All six Kaggle pilot execution notebooks (`kaggle_preflight.ipynb`, `kaggle_smoke.ipynb`, `kaggle_train.ipynb` for both Arm 2A and Arm 2B) were authored pinning commit `051a87071279a1f2662fa9c6c449339281e05063` (Step 15 experimental design).
- **Impact:** The dedicated pilot execution modes (`train_pilot_arm2a`, `train_pilot_arm2b`, `verify_pilot_arm2a`, `verify_pilot_arm2b`) in `deploy/kaggle/kaggle_entrypoint.py` and Fold 1 pilot routing in `audit_scripts/kaggle_fold_train_exec.py` were introduced only in commit **`56bf79535e4f697b2eea05594d5d81153e04dbff`** (Step 16).
- **Consequence If Uncorrected:** If deployed to Kaggle, Phase 3 of the notebook would check out commit `051a870`, where `kaggle_entrypoint.py` only accepted `train_c0..1` and `train_fold2..4`. The subsequent Phase 4 preflight command would fail closed immediately with:
  `argparse: error: argument --mode: invalid choice: 'verify_pilot_arm2a'`.

This hotfix updates all six notebooks to pin the authoritative, reviewed training commit **`56bf79535e4f697b2eea05594d5d81153e04dbff`**, adds automated regression tests detecting stale commit pinning, and certifies Gate B readiness.

---

## 2. Historical Git Commit Verification

The commit disparity was verified using Git revision inspections:

1. **Commit `051a870c2388e852b628fae9814056b55bce5051` (Step 15):**
   - Inspection: `git show 051a870c2388e852b628fae9814056b55bce5051:deploy/kaggle/kaggle_entrypoint.py`
   - Result: Contained strictly legacy modes (`train_c0`, `train_c1`, `train_fold2..4`, `verify_fold2..4`). Zero references to `train_pilot` or `verify_pilot`.
2. **Commit `56bf79535e4f697b2eea05594d5d81153e04dbff` (Step 16):**
   - Inspection: `git show 56bf79535e4f697b2eea05594d5d81153e04dbff:deploy/kaggle/kaggle_entrypoint.py`
   - Result: Fully implements `train_pilot_arm2a`, `train_pilot_arm2b`, `verify_pilot_arm2a`, `verify_pilot_arm2b`, `train_fold1`, and `verify_fold1`.
   - Inspection: `git show 56bf79535e4f697b2eea05594d5d81153e04dbff:audit_scripts/kaggle_fold_train_exec.py`
   - Result: Accepts `--fold 1` with `--arm Arm2A_Focal` and `--arm Arm2B_Weighted`, routes outputs to `checkpoints/RQ1-TRN-03B/` and HF mirror prefix `pilot/`, and supports Fold 1 train/val archive `authorized_RQ1-TRN-02A_fold1_train_val_only.zip`.

---

## 3. Remediation Performed

All six Kaggle deployment notebooks were updated using targeted text replacements:

| Package Directory | Notebook File | Old Pinned Commit | Corrected Pinned Commit | Verification Status |
|---|---|---|---|:---:|
| `deploy/kaggle_pilot_arm2a_preflight/` | `kaggle_preflight.ipynb` | `051a870...` | `56bf79535e4f697b2eea05594d5d81153e04dbff` | Corrected |
| `deploy/kaggle_pilot_arm2a_smoke/` | `kaggle_smoke.ipynb` | `051a870...` | `56bf79535e4f697b2eea05594d5d81153e04dbff` | Corrected |
| `deploy/kaggle_pilot_arm2a_train/` | `kaggle_train.ipynb` | `051a870...` | `56bf79535e4f697b2eea05594d5d81153e04dbff` | Corrected |
| `deploy/kaggle_pilot_arm2b_preflight/` | `kaggle_preflight.ipynb` | `051a870...` | `56bf79535e4f697b2eea05594d5d81153e04dbff` | Corrected |
| `deploy/kaggle_pilot_arm2b_smoke/` | `kaggle_smoke.ipynb` | `051a870...` | `56bf79535e4f697b2eea05594d5d81153e04dbff` | Corrected |
| `deploy/kaggle_pilot_arm2b_train/` | `kaggle_train.ipynb` | `051a870...` | `56bf79535e4f697b2eea05594d5d81153e04dbff` | Corrected |

Both markdown overview text and Phase 3 code cell variables (`PINNED_COMMIT_SHA`) were updated synchronously.

---

## 4. Execution Paths & Safety Defenses Re-Verification

1. **Preflight Path (`verify_pilot_arm2a` / `verify_pilot_arm2b`):**
   - Validates Tesla T4 accelerator, resolves `HF_TOKEN` from Kaggle Secrets, checks out `56bf79535e4f697b2eea05594d5d81153e04dbff`, verifies clean working tree.
   - Stages Fold 1 archive (`authorized_RQ1-TRN-02A_fold1_train_val_only.zip`, SHA-256 `b750eb3b...`).
   - Verifies 0 test speaker leakage (`ABA`, `ASI`, `EBVS`, `HJK`, `LXC`, `PNV`).
   - Executes live write-readback-delete probe to `minkhoaa/langfens-rq1-cross-fold-checkpoints`.
   - Exits safely with 0 updates or weights created.
2. **Smoke Path (`pilot_arm2a_focal_smoke_1epoch` / `pilot_arm2b_weighted_smoke_1epoch`):**
   - Executes mandatory preflight gate first.
   - Executes strictly 1 completed epoch (~600 updates) under `--execution-cap-epochs 1`.
   - Verifies `last.pt`, `best.pt`, `history.json` (1 entry) in output directory.
3. **Official Training Path (`pilot_arm2a_focal` / `pilot_arm2b_weighted`):**
   - Executes mandatory preflight gate first.
   - Executes up to 10 epochs with primary fixed-budget checkpoint locked at Epoch 6 / 3,600 updates.
   - Evaluates `classify_fold_training_endpoint` verifying `COMPLETED_3600_UPDATES`, fold index 1, manifest SHA `7d6bfdf6...`, and arm-specific loss parameters.

---

## 5. Automated Regression Test Suite

A dedicated regression test was implemented in `tests/test_rq1_trn_03b_kaggle_pilot_deployment.py`:

- **Test Name:** `test_notebook_commit_pinning_integrity_and_stale_commit_detection`
- **Verifications:**
  1. Target commit `56bf79535e4f697b2eea05594d5d81153e04dbff` exists in git.
  2. Pinned entrypoint at `56bf7953` contains all four pilot modes.
  3. Negative control: proves that stale commit `051a870c` lacks pilot modes.
  4. All six notebooks contain target commit `56bf79535e4f697b2eea05594d5d81153e04dbff` and 0 references to stale commit `051a870c`.
- **Results:**
  - `tests/test_rq1_trn_03b_kaggle_pilot_deployment.py`: **11 passed in 7.66s**.
  - Full repository suite: **321 passed, 1 skipped, 1 warning in 168.72s** (100% green).

---

## 6. Updated Operator Deployment Commands for Gate B

```bash
# Push Arm 2A Focal Pilot Packages (Pinned to 56bf79535e4f697b2eea05594d5d81153e04dbff)
kaggle kernels push -p deploy/kaggle_pilot_arm2a_preflight --no-run
kaggle kernels push -p deploy/kaggle_pilot_arm2a_smoke --no-run
kaggle kernels push -p deploy/kaggle_pilot_arm2a_train --no-run

# Push Arm 2B Weighted Pilot Packages (Pinned to 56bf79535e4f697b2eea05594d5d81153e04dbff)
kaggle kernels push -p deploy/kaggle_pilot_arm2b_preflight --no-run
kaggle kernels push -p deploy/kaggle_pilot_arm2b_smoke --no-run
kaggle kernels push -p deploy/kaggle_pilot_arm2b_train --no-run
```

---

## 7. Next Authorized Step & Stop Statement

- **Hotfix Verdict:** **`PASS: COMMIT PINNING MISMATCH FULLY REMEDIATED`**.
- **Gate B Readiness:** **`CERTIFIED READY FOR LAUNCH UNDER COMMIT 56bf79535e4f697b2eea05594d5d81153e04dbff`**.
- **Execution Boundary:** No remote GPU runs launched. No local training attempted. No test partitions unsealed.
- **Action Required:** Operator review and explicit sign-off to proceed with Kaggle CLI deployment.

**STOP.**
