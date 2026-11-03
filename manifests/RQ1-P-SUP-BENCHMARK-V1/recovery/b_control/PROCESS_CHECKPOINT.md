# Process Checkpoint — P-SUP B-CONTROL Artifact Recovery & Verification

## State
- **Process ID:** `psup-recover-b-control-v1`
- **Subagent Identity:** Agent B (Independent Recovery Specialist for B-Control Matched Control)
- **Status:** COMPLETED — Bit-for-bit cryptographic recovery and deep payload verification successful across Kaggle, Hugging Face, and local candidates.
- **Timestamp:** 2026-11-04T01:26:00+07:00 (Asia/Ho_Chi_Minh)
- **Repository Worktree:** `/home/khoa/Projects/KLTN/psup-recover-b-control-v1`
- **Branch:** `review/psup-recover-b-control-v1`
- **Base Commit:** `40bc3770776ee1c99462d44101fdd145602b76f5`
- **Training Status in THIS Process:** NOT_STARTED. Optimizer updates = 0, backward passes = 0, forward training steps = 0, model mutations = 0. Strictly read-only forensic inspection and artifact recovery.

## Inputs
- **Kaggle Kernel:** `minkhoaa/langfens-rq1-psup-b-control-t4x2` (Machine shape: `NvidiaTeslaT4` T4 x 2, Status: `KernelWorkerStatus.COMPLETE`).
- **Private Hugging Face Mirror:** `minkhoaa/langfens-rq1-cross-fold-checkpoints` (Prefix: `psup/b_control/`, Repo Type: `model`, Privacy: `is_private=True`).
  - *Pinned b_control revision:* `4aa9f2a68f0709889eacacf170b6446d3c45c73d`
  - *Current repository HEAD:* `6d1ec3217c94730cc590b935c45e523dc5259f39` (Advanced due to parent orchestrator mirroring archival benchmark predictions under `benchmarks/psup-v1/`; `psup/b_control/` verified bit-identical and unchanged from pinned revision `4aa9f2a68f0709889eacacf170b6446d3c45c73d`).
- **Pre-existing Local Candidate Directory:** `/home/khoa/Projects/KLTN/Dataset/checkpoints/RQ1-P-SUP/b_control` (PRESERVED completely untouched).
- **Split Manifest:** `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` (SHA-256 `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`).
- **Authorized Audio Archive:** `authorized_RQ1-TRN-02A_fold1_train_val_only.bin` / zip (SHA-256 `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`).
- **Evaluation Contract V1:** SHA-256 `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`.
- **Pretrained Backbone:** `facebook/wav2vec2-base-960h` (rev `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`).

## Outputs & Recovered Artifacts
- **Scratch Directory:** `/home/khoa/.cache/langfens/psup-benchmark-v1/b_control/recovered/` (Kept strictly outside Git and remote staging).
- **Recovered Checkpoints & Logs:**
  1. `fixed_budget_3600.pt`:
     - Path: `/home/khoa/.cache/langfens/psup-benchmark-v1/b_control/recovered/fixed_budget_3600.pt`
     - Size: 3,241,295 bytes
     - SHA-256: `e121a7b965747b49608263f1e3ec447f2a631f09a36e326e20fe5f2cfe619201`
     - In-payload updates: 3,600 successful updates, 0 skipped, epoch 7, patience 2, `is_fixed_budget_endpoint=True`.
  2. `best.pt`:
     - Path: `/home/khoa/.cache/langfens/psup-benchmark-v1/b_control/recovered/best.pt`
     - Size: 3,239,450 bytes
     - SHA-256: `2dbad5a0160224f9d936e564a6c5d52dffe6a8efdbde8e409c50a8cc393be93f`
     - In-payload updates: 2,250 successful updates (epoch 4), patience 0, best selection AP 0.467607, `is_fixed_budget_endpoint=False`.
  3. `last.pt`:
     - Path: `/home/khoa/.cache/langfens/psup-benchmark-v1/b_control/recovered/last.pt`
     - Size: 3,239,898 bytes
     - SHA-256: `aa35c14dbf6827050efa779b419b3c4887988fa073bc693c81dcb41275afa1b5`
     - In-payload updates: 3,600 successful updates (epoch 7), patience 2, `is_fixed_budget_endpoint=False`.
     - *Note:* Missing from pre-existing local candidates; successfully recovered from Hugging Face Hub and Kaggle run.
  4. `training_summary.json`:
     - Path: `/home/khoa/.cache/langfens/psup-benchmark-v1/b_control/recovered/training_summary.json`
     - Size: 3,339 bytes
     - SHA-256: `fb8917eda607e2cffdc3e276507a3924b68d418a01a3bd0302a296a24b174b4f`
  5. `history.json`:
     - Path: `/home/khoa/.cache/langfens/psup-benchmark-v1/b_control/recovered/history.json`
     - Size: 2,896 bytes
     - SHA-256: `dc61bbff9b58f804712b49ba4e5c1998a2b137b58d53c544910e7fa31e0ce2c6`
  6. `langfens-rq1-psup-b-control-t4x2.log`:
     - Path: `/home/khoa/.cache/langfens/psup-benchmark-v1/b_control/recovered/langfens-rq1-psup-b-control-t4x2.log`
     - Size: 380,708 bytes
     - SHA-256: `24f1a2a30211b29d1590fba830a8c314c3a46657db57bef50cf333afc25d8936`
  7. `langfens-rq1-psup-b-control-t4x2.ipynb`:
     - Source notebook pinned to git commit `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f`.
- **Receipts Stored in Tree:**
  - `manifests/RQ1-P-SUP-BENCHMARK-V1/recovery/b_control/RECOVERY_RECEIPT.json`
  - `manifests/RQ1-P-SUP-BENCHMARK-V1/recovery/b_control/CHECKPOINT_PAYLOAD_AUDIT.json`
  - `manifests/RQ1-P-SUP-BENCHMARK-V1/recovery/b_control/SHA256SUMS`

## Verification Evidence & Operational Accounting
1. **Bit-Identical Cryptographic Parity:**
   - HF LFS blob hashes, Kaggle output files, pre-existing local candidates, and scratch files are 100% bit-identical.
2. **Kaggle Output Retrieval & Download Operational Accounting:**
   - An early command `kaggle kernels output` began downloading the full cloned git tree (`repo/`).
   - The parent orchestrator issued SIGINT to terminate PID 183900 to prevent unnecessary bulk downloading.
   - Bounded retrieval (`--file-pattern ".*\.(pt|json)$"` and pinned HF downloads) was subsequently used.
   - This interrupted download is an accounted operational download boundary filter and NOT an artifact integrity failure. All target payloads are intact, complete, and verified.
3. **Safe CPU Tensor Inspection (`weights_only=True` allowlist):**
   - Head architecture: `CompactOperationHead` (266,667 parameters across 16 tensors).
   - Finiteness: 100% finite (0 NaN, 0 Inf).
   - Optimizer: AdamW complete state dict with 16 parameter state tensors.
   - Scheduler: Cosine schedule with 100 warmup updates, horizon 15,000 updates, step counter intact.
   - Scaler: GradScaler complete (scale=8192.0, growth_factor=2.0, interval=2000).
   - RNG: Full 4-tuple preserved (Python, NumPy, PyTorch CPU, PyTorch CUDA [2 devices]).
4. **Historical Updates vs Process Updates:**
   - Historical remote training completed 3,600 updates / 0 skipped in 561.20s on Tesla T4.
   - In-payload micro_steps = 7,200 (effective batch size 4 = 2 batch_size * 2 gradient_accumulation).
   - Best checkpoint occurred at epoch 4, update 2,250 (AP = 0.467607 on SELECT).
   - Fixed budget endpoint saved at epoch 7, update 3,600 (AP = 0.464117 on SELECT).
   - This process performed 0 updates, 0 backward passes.
5. **Data Isolation & Sealing:**
   - Partition isolation strictly preserved: FIT (1,799 utts, 12 spk), SELECT (300 utts, 2 spk: MBMPS, TLV), CAL (300 utts, 2 spk: NCC, THV), VAL (300 utts, 2 spk: NJS, RRBI).
   - TEST partition (900 utts, 6 spk: ABA, ASI, EBVS, HJK, LXC, PNV) sealed; 0 test utterances staged, trained, or leaked.
6. **Subtype Score Semantics Disclosure:**
   - Under `binary_collapsed`, the loss collapses classes 1 (substitution) and 2 (omission) via `logsumexp(z1, z2) - z0` against binary label `(y != 0)`.
   - Classes 1 and 2 receive identical gradient signals; the model never learns to separate substitutions from deletions.
   - In runtime rows and offline evaluator:
     - `s_error` = $p_1 + p_2$
     - `s_omit` is `null` (unidentified)
     - `s_sub` is `null` (unidentified)
     - `probabilities` is `null`
     - `omission_ap_dedicated` is `null` (unsupported).
   - Any omission AP or substitution AP reported for B-CONTROL is strictly legacy-mode (evaluating general binary error score `s_error` on the subset). Dedicated omission detection is mathematically unsupported.

## Boundaries & Governance
- Sole writer in own worktree (`/home/khoa/Projects/KLTN/psup-recover-b-control-v1`).
- No commits or pushes made.
- Global process index `docs/process-checkpoints/INDEX.md` left untouched (owned by coordinator).
- Local candidate directory `/home/khoa/Projects/KLTN/Dataset/checkpoints/RQ1-P-SUP/b_control` preserved without modification.

## Safe Restart & Next Steps
1. The coordinator may ingest `manifests/RQ1-P-SUP-BENCHMARK-V1/recovery/b_control/RECOVERY_RECEIPT.json`.
2. Recovered checkpoints in scratch (`/home/khoa/.cache/langfens/psup-benchmark-v1/b_control/recovered/`) are ready for authorized CPU offline evaluation or comparative synthesis against T-OP.
3. No training or parameter re-initialization is required; historical artifacts are fully verified and reproducible.
