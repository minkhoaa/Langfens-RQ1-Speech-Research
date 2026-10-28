# LANGFENS — RQ1-TRN-02B / 12-Kaggle Direct C0 Launch Gate and Benchmark Waiver

**Recorded:** 2026-10-28 (Asia/Ho_Chi_Minh)
**Source examined:** `96f7997b85e2fc96c612ef62b8d813e5599ef572` on `feat/rq1-trn-02b-kaggle-t4x2-migration` (remote branch HEAD verified)
**Process state:** `STATIC_SOURCE_REVIEW_COMPLETED; EXECUTION_BLOCKED_IMPLEMENTATION_AND_STAGING`
**Benchmark decision:** `WAIVED_BY_OPERATOR_FOR_THIS_EXECUTION_PLAN` (user explicitly requested skipping disposable 50-update benchmark). **Not benchmark PASS**.
**Official C0:** `NOT_STARTED`, 0/3600 successful optimizer updates.
**Official C1:** `NOT_STARTED`, 0/3600 successful optimizer updates.
**Kaggle actual execution:** `NOT_ATTEMPTED` in this review; no authenticated Kaggle CLI runtime tool was available to the reviewing assistant.
**Dataset:** user-provided Step 11 report says private Kaggle Dataset has not yet been populated; not independently checked in Kaggle.
**TEST:** remained sealed; no sealed TEST material accessed, staged, or inspected in this process.

## 1. Decision

Operator explicitly prefers **starting the next official Kaggle GPU phase without running the 50-successful-update feasibility benchmark**. Accept this as a documented waiver of an optional resource-performance measurement, **not a waiver of software correctness, data integrity, storage durability, account quota, device identity, scientific protocol or per-epoch validation/guardrails**.

A normal `kaggle kernels push -p deploy/kaggle` launches the saved Python script, while `--no-run` only saves a version (official Kaggle CLI documentation: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md). However, current entrypoint defaults to `--mode benchmark`; **pushing without changing it would not run official C0**.

**Decision gate:** `BLOCKED` pending listed fixes and real TRAIN/VAL-only staging. Do not submit the current code as a production C0 run.

## 2. Critical confirmed defects at the pinned source

1. **False remote persistence:** `audit_scripts/kaggle_checkpoint_sync.py` `KaggleRemoteMirror.mirror_file()` performs a real file copy/hash only for `local_dir`. For backends `gdrive`, `huggingface`, and `kaggle_dataset`, its final unconditional return reports `SYNCED` and `remote_sha256 == local_sha256` without any remote upload or readback. The non-local readiness checks only test presence of credentials, not authenticated remote write-read-delete. **This is not implemented durable mirroring.**
2. **No per-epoch remote sync:** `audit_scripts/kaggle_train_exec.py` only calls `mirror.mirror_file()` **after** `subprocess.run(cmd)` finishes. The declared mandatory per-epoch durability guarantee is therefore false, regardless of the selected backend.
3. **Broken original trainer invocation:** `audit_scripts/kaggle_train_exec.py` constructs `train_rq1_trn_02b.py --arm ... --output-dir ...`, yet the original `train_rq1_trn_02b.py` argparse does not define `--arm` or `--output-dir`. Resume also passes a bare `--resume` although the original trainer requires a checkpoint path.
4. **Misrouted Baseline R input:** Staging finds/verifies Kaggle's `best.pt`, but the training command does not pass the verified `--initial-checkpoint`; original config points to a relative `checkpoints/RQ1-TRN-02A/recovered/best.pt`, which a fresh Kaggle clone will not have.
5. **Entrypoint not pinned end-to-end:** `deploy/kaggle/kaggle_entrypoint.py` defaults to benchmark mode; `--commit-sha` is optional and checkout occurs only in its fresh-clone branch. A reused checkout receives no source SHA verification. No production execution may depend on a floating default branch.
6. **Mock-GPU bypass still exposed:** `audit_scripts/kaggle_stage_and_verify.py` and `audit_scripts/kaggle_train_exec.py` expose `--allow-mock-gpu` in production CLIs; notebook setup uses it. The previous R3 GPU guard intentionally removed this class of flag; ensure no production path can bypass real CUDA T4 proof.
7. **Ephemeral `local_dir` accepted as durable:** `KaggleRemoteMirror.verify_readiness()` accepts a local directory, including one under `/kaggle/working`, based on local file readback only. Such a path does not satisfy the `AGENTS.md` persistent-remote requirement.
8. **Input dataset missing per Step 11:** No evidence a private Kaggle Dataset with authorized TRAIN+VAL archive and exact Baseline R was populated. Do not fabricate that staging is ready or assume private Kaggle metadata alone ensures licensed storage.

These findings are from source reading, not a new pytest run or an executed Kaggle failure. Historical Step 11 test census `222 passed / 4 skipped` is preserved as agent-reported and does not establish the correctness of real external persistence.

## 3. Minimal next implementation and execution contract

- Fix the above defects test-first on a review branch, preserving existing Colab R3 implementation unchanged. Implement **real** approved private remote checkpoint upload, authenticated readback SHA-256 and durability verification immediately after each epoch. Fail closed and stop on any mirror error. Do **not** replace remote storage with local `/kaggle/working` or mark successful by copying the digest from the local file.
- Wire Kaggle staging to original trainer using supported arguments; ensure exact staged paths for Baseline R and authorized TRAIN/VAL-only ZIP; independent run IDs and complete state (optimizer/scheduler/AMP scaler/RNG), resumption by explicit path, and actual endpoint classification.
- Pin immutable execution source SHA in a private Kaggle kernel/script that explicitly selects `train_c0`, not default benchmark; verify Git HEAD and clean source inside actual remote runtime. Do not change model architecture, data split, inference protocol, optimizer budget, or precision.
- Before upload of licensed data to Kaggle storage, confirm authority to store L2-ARCTIC restricted TRAIN/VAL-only copies in that private service; private visibility by itself is insufficient to establish licensing permission. No sealed TEST.
- Confirm current authenticated Kaggle quota and T4 capability, exact datasets/weights SHA-256, unique C0 run directory, successful genuine remote write/readback, and run authorization. Skip only the disposable 50-update benchmark. A staging-only, non-training verification is still mandatory.
- Once and only once these gates pass, execute official C0 `p=0.00` (up to 6 epochs / 3600 successful updates), log telemetry continuously, and persist every epoch. On interruption document actual successful/skipped update count, last remotely verified full-state checkpoint and restart command. Then independently cold-initialize C1 `p=0.25` from the SAME Baseline R with identical environment and budget, never from C0.
- If any guardrail trip, incomplete updates or source mismatch occurs, mark incomplete/blocked as appropriate; never present `COMPLETED_3600_UPDATES` without the full independent endpoint invariants.
- Publish next numbered research/process Markdown checkpoint (13) and update phase overview/global INDEX based on **real** Kaggle execution evidence, not inferred performance. Keep official TEST sealed.

## 4. Execution ledger for this review

| Activity | Status |
| --- | --- |
| Pinned commit/branch, Kaggle code, AGENTS.md and Step 11 read | Source inspection complete |
| Operator decision to skip disposable 50-step benchmark | Recorded as waiver, not PASS |
| Production Kaggle run safety | **BLOCKED** (critical implementation defects) |
| Live Kaggle account, GPU, storage staging and remote mirror | **NOT RE-QUERIED / NOT VERIFIED in current session** |
| Benchmark 50 successful updates | **NOT_EXECUTED** (explicitly waived) |
| Official C0 | **0/3600, NOT_STARTED** |
| Official C1 | **0/3600, NOT_STARTED** |
| Official sealed TEST | **No reads or staging in this review** |
| Regression tests | **NOT_RERUN** |
| New model or GPU session | **NONE** |

**Safe restart point:** apply test-first remediation; obtain true authorized data staging and remote mirror verification; execute private Kaggle T4 official C0 via documented CLI after guarded sign-off; independently run C1 only after C0 evidence. Preserve all historical checkpoint evidence without silent corrections.
