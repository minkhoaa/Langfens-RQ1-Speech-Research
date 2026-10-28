# LANGFENS — RQ1-TRN-02B / 05-Colab Auth R2 Launcher Safety Fixes

**Status:** `PASS_SOFTWARE` (Operational Execution: `BLOCKED` on quota and runtime verification)<br>
**Recorded at:** 2026-10-28T12:00:00+07:00<br>
**Evidence Source:** Branch `fix/rq1-trn-02b-colab-auth-r2`, `manifests/RQ1-TRN-02B/`<br>
**Audited Starting Commit:** `a497ec3f0b6dd4c760c4e40296481f61b6426d8d`<br>
**Evidence Level:** `INDEPENDENTLY_VERIFIED`

---

## 1. Objective
Remediate the seven operational defects identified in the independent source audit of `fix/rq1-trn-02b-colab-auth-r1` (commit `a497ec3f0b6dd4c760c4e40296481f61b6426d8d`):
1. Resolve fresh VM bootstrap failure when `/content/repo` does not exist.
2. Resolve Colab notebook kernel execution incompatibilities (eliminate `__file__` dependencies and handle ipykernel arguments).
3. Eliminate duplicate run-ID nesting in checkpoint output paths (`.../<run_id>/<run_id>`).
4. Replace shallow zip inspection with safe, bounded internal path and speaker directory verification.
5. Implement fail-closed non-zero exit codes for executable readiness queries.
6. Enforce authentic Google Drive mount and baseline asset cryptographic identity checks.
7. Implement rigorous post-training primary endpoint classification (`COMPLETED_3600_UPDATES`, `INCOMPLETE`, `HALTED_GUARDRAIL`, `FAILED`).

---

## 2. Verified Starting Git Commit
- **Starting Commit SHA:** `a497ec3f0b6dd4c760c4e40296481f61b6426d8d`
- **Starting Branch:** `fix/rq1-trn-02b-colab-auth-r1`
- **Review Branch Created:** `fix/rq1-trn-02b-colab-auth-r2`
- **Execution Boundary:** Pre-training task. Strictly no GPU runtime allocation, Colab provisioning, backward passes, or optimizer updates occurred.

---

## 3. Work Completed & Exact Corrections Made

### A. Fresh VM Deterministic Bootstrap
- Authored standalone `audit_scripts/colab_bootstrap.py` and integrated `bootstrap_repository()` into `audit_scripts/colab_stage_and_verify.py`.
- If `/content/repo` is absent on a fresh VM, it automatically clones `https://github.com/minkhoaa/Langfens-RQ1-Speech-Research.git` and checks out the approved immutable commit SHA.
- If `/content/repo` already exists, it verifies origin identity (`Langfens-RQ1-Speech-Research`) and fails closed with `RuntimeError` if the directory is non-git, points to an unrelated origin, or has a dirty working tree.
- Tested against fresh-VM simulation where `/content/repo` is absent.

### B. Colab Notebook Kernel Execution Compatibility
- Audited `colab_train_exec.py`, `colab_stage_and_verify.py`, and `colab_bootstrap.py` to eliminate dependence on `__file__` (which is undefined in a fresh Jupyter kernel namespace).
- In `main()` of all three scripts, replaced standard `parse_args()` with `parse_known_args()` to cleanly ignore ipykernel launcher arguments (e.g. `['-f', '/root/.../kernel-uuid.json']`).
- Added regression test `test_kernel_execution_compatibility_without_dunder_file` compiling and running the code in an isolated namespace without `__file__`.

### C. Duplicate Run-ID Elimination
- Resolved path duplication where `--run-id` was appended both by the caller wrapper and by the underlying `train_rq1_trn_02b.py`.
- In `audit_scripts/train_rq1_trn_02b.py`: verified that `outdir` and `drive_out` only append `run_id` if their leaf directory does not already match `run_id`.
- In `colab_stage_and_verify.py` and `colab_train_exec.py`: codified `resolve_drive_output_dir(drive_root, arm, run_id)` ensuring idempotent, single-directory output construction for fresh runs and resumptions across both arms C0 and C1.

### D. Safe Bounded Dataset Package Inspection
- Replaced the shallow `.zip` filename check with `test_authorized_zip_bounded_speaker_inspection`.
- Reads outer packages and streams inner zip entries in-memory via `io.BytesIO` without writing to disk.
- Extracts every internal file and directory path, asserting that all paths strictly belong to approved TRAIN and VAL speakers (`BWC, ERMS, HKK, HQTV, MBMPS, NCC, NJS, RRBI, SKA, SVBI, THV, TLV, TNI, TXHC, YBAA, YDCK, YKWK, ZHAA`).
- Strictly asserts zero test speaker leakage ($S_{\text{test}} \cap S_{\text{observed}} = \emptyset$).
- Marks test `SKIPPED/UNVERIFIED` if the authorized archive is unavailable locally.

### E. Fail-Closed Quota and Layered Readiness
- Updated `audit_scripts/colab_launch_workflow.py` to separate operational readiness into 5 distinct layers:
  1. `STATIC_SOFTWARE`: Validates code, git tree cleanliness, YAML configs, and split manifest. Exits with status `PASS_SOFTWARE`.
  2. `COMPUTE_QUOTA`: Inspects read-only `colab usage`. Status is `BLOCKED` if balance < 15.0 units.
  3. `DRIVE_MOUNT`: Status is `PENDING_OPERATOR_ACTION` (requires `colab drivemount -s <session>`).
  4. `VM_STAGING_WRITE`: Status is `PENDING_RUNTIME_VERIFICATION`.
  5. `GPU_AUTHORIZATION`: Status is `BLOCKED` until all upstream layers pass.
- When `--check-only` or `--check-readiness` is invoked and prerequisites fail, `colab_launch_workflow.py` exits with non-zero exit code (`sys.exit(1)`).
- Added explicit notice: **`PASS_SOFTWARE does not authorize GPU execution.`**

### F. Authentic Google Drive Persistence Verification
- Enforced mountpoint verification: On Colab VM (`/content`), confirms `/content/drive` is an active FUSE mountpoint (`os.path.ismount('/content/drive')` or `/proc/mounts`).
- Enforced cryptographic identity verification: Under `KLTN-Dataset` (Folder ID: `1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK`), verifies that `manifests/RQ1-TRN-02A/checkpoints/best.pt` exists and matches SHA-256 `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`.
- Rejects any arbitrary local directory named `KLTN-Dataset`.

### G. Primary Endpoint Verification and Classification
- Implemented `classify_training_endpoint(drive_output_dir, arm)` in `colab_train_exec.py`.
- Formally classifies training runs into four distinct states based on checkpoint payload metadata:
  - `COMPLETED_3600_UPDATES`: Verified `fixed_budget_checkpoint.pt`, strictly $\ge 3,600$ updates, zero guardrail violations.
  - `INCOMPLETE`: Terminated before 3,600 updates without guardrail violation.
  - `HALTED_GUARDRAIL`: CTC PER degradation exceeded anchor + 1.0% (`halted_guardrail.pt` preserved).
  - `FAILED`: Checkpoints missing, corrupted, or process returned non-zero code.
- Does not label a run complete merely because `last.pt` exists.

---

## 4. Tests and Verification Evidence
- **Launch & Staging Test Suite:**
  `PYTHONPATH=src:. .venv/bin/pytest tests/test_rq1_trn_02b_colab_launch.py`
  Result: **25 passed in 1.32s** (0 failures).
- **Full Repository Regression Suite:**
  `PYTHONPATH=src:. .venv/bin/pytest tests/`
  Result: **182 passed, 4 skipped in 101.74s** (0 failures).
- **Local Workflow Orchestrator (Fail-Closed Execution):**
  `python3 audit_scripts/colab_launch_workflow.py --check-only`
  Exits with code 1 when quota is 0.00 units or prerequisites fail.

---

## 5. Remaining Operational Blockers
1. **Compute Quota Blocker:** Measured balance is `0.00 compute units`. Operator must assign ~15–20 compute units before provisioning.
2. **Interactive Drive Mount Prerequisite:** `colab drivemount -s <session>` requires human operator terminal interaction.
3. **Live VM Staging & Drive Write Verification:** Live VM disk copy and remote Drive write verification remain outstanding until an authorized Colab session is provisioned.

---

## 6. Restart / Recovery Instructions for Next Agent
1. Verify working branch and clean tree:
   ```bash
   git status
   git log -n 1 --oneline
   ```
2. Verify all tests pass on CPU:
   ```bash
   PYTHONPATH=src:. .venv/bin/pytest tests/test_rq1_trn_02b_colab_launch.py
   ```
3. Run software check:
   ```bash
   python3 audit_scripts/colab_launch_workflow.py --check-software-only
   ```
4. If quota is 0.00, do not attempt to run `colab new` or allocate resources; report blocker to operator.

---

## 7. Explicit Authorization Boundary for Future GPU Execution
- **Strict Prohibition:** Never execute training backward passes, fine-tuning, or optimizer updates on local CPU or GPU.
- **Strict Non-Interactive Rule:** Never execute `colab drivemount` or `colab auth` automatically from scripts.
- **PASS_SOFTWARE does not authorize GPU execution.**
- Launching C0 and C1 training requires operator compute unit assignment, manual Drive mounting, and explicit execution authorization.
