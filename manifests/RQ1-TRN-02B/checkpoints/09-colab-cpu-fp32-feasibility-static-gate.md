# LANGFENS — RQ1-TRN-02B / 09-Colab CPU FP32 Feasibility: Static Decision Gate

**Date:** 2026-10-28 (Asia/Ho_Chi_Minh)
**Process status:** `STATIC_ASSESSMENT_RECORDED`
**CPU benchmark status:** `NOT_EXECUTED`
**CPU FP32 variant status:** `PROPOSED_ONLY / NOT_AUTHORIZED / NOT_IMPLEMENTED`
**C0 and C1 official training:** `NOT_STARTED`; 0 successful optimizer updates for each arm
**Evidence level:** `GITHUB_SOURCE_INSPECTED; LIVE_RUNTIME_NOT_QUERIED`
**Immutable source reviewed:** `2caa2bfc4208a16a63fe8f1d5a4d72effccb7ba6`
**Original frozen R3 implementation:** `ba7be0dc407313a4f5e31577051e6746278adeed`
**Last experimental runtime evidence:** [Step 08](08-live-gpu-runtime-and-staging-verification.md) (historical, agent-reported)

## 1. Objective and scope

Re-establish research state from the 28-Oct handoff; assess what must be measured to decide Colab CPU versus GPU; prepare a reproducible **CPU FP32-only feasibility benchmark protocol**, without changing production GPU FP16 training or running any optimizer update.

This is a **process checkpoint**, not a PyTorch model checkpoint, benchmark result, official protocol amendment, or training authorization. The repository was inspected using the connected GitHub API. This step did NOT access Colab, Drive binary artifacts, or sealed TEST.

## 2. Source-grounded findings

1. `AGENTS.md` restricts training/backward/optimizer updates to **Google Colab GPU**, and prohibits local CPU/GPU fallback. No CPU training may be launched under current policy.
2. Both original arm configs (`configs/rq1_trn_02b_c0.yaml`, `configs/rq1_trn_02b_c1.yaml`) are `amp: fp16`. They differ in the intended intervention only: binary-only whole-phone prompt embedding dropout `p=0.00` versus `p=0.25`.
3. `audit_scripts/train_rq1_trn_02b.py` has an explicit CUDA guard for non-dry-run training. `audit_scripts/colab_stage_and_verify.py` and `audit_scripts/colab_train_exec.py` also reject non-CUDA runtimes. CPU measurement needs a separately reviewed and expressly authorized route, not a silent fallback or a mock-GPU flag.
4. The production trainer makes a full epoch pass before validation and its normal endpoint depends on 3,600 successful updates; it does **not** currently expose a disposable 50-successful-update CPU benchmark endpoint.
5. Original C0/C1 scientific constraints: same authenticated Baseline R weights SHA-256 `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`; split manifest SHA-256 `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`; authorized TRAIN/VAL ZIP SHA-256 `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`. TRAIN 2,399 utterances/16 speakers, VAL 300/NJS and RRBI; TEST 900/6 speakers remains sealed.
6. Historical Step 08 reports that a free-tier T4 was provisioned once, but Drive was unmounted and interactive OAuth blocked staging. These observations do not prove any current CPU uptime, memory, staging readiness, or training throughput.

## 3. Resource feasibility: bounds, NOT measured performance

The UI message, *"At your current usage level, this runtime may last up to 36 hours 20 minutes"*, corresponds to an **indicative** upper limit of 130,800 seconds, **not** guaranteed uninterrupted uptime and **not** a CPU training estimate.

- One arm, 3,600 successful updates in one hypothetical 130,800-second session: **36.33 seconds per successful update MAX**, ignoring setup, validation, logging, checkpoint transfer, preemption, and safety margin.
- Both arms, 7,200 successful updates in the same hypothetical session: **18.17 seconds per successful update MAX** under the same unrealistic zero-overhead assumptions.
- With batch size 2, gradient accumulation 2, and 2,399 TRAIN utterances, a normal epoch has `ceil(2399/2)=1200` microbatches and `ceil(1200/2)=600` optimizer attempts; six epochs nominally reach 3,600 successful updates if no attempts are skipped, with no early halt.
- No reliable throughput estimate can be inferred from the earlier *data ingestion* speed; full Wav2Vec2 forward, head backward, optimizer updates, and validation need measurement.

Estimated per-arm elapsed time must account for:
`T_arm = 3600*t_successful_update + 6*T_val_epoch + T_staging + T_checkpoint_sync + T_other`.
Actual extrapolation must adjust for skipped attempts, variable utterance lengths, warmup/runtime variability, early-stop/guardrail behavior, session loss risk, and whether C0 and C1 run in independent sessions. If the measured confidence range or overhead makes either arm unrealistic within safe runtime limits, select GPU or request a separate compute decision; **do not reduce the frozen experimental budget to make CPU fit**.

## 4. Proposed CPU FP32 feasibility benchmark — requires separate approval

Before ANY CPU backward/optimizer step:

1. Record operator approval for **Colab CPU, FP32-only, disposable 50-successful-update benchmark**; amend `AGENTS.md` narrowly on a review branch for that variant. Retain original GPU-only production path, all existing fail-closed Drive/root checks, and local-device training prohibition.
2. Add a separately named, explicitly gated CPU benchmark driver/config. Use `benchmark_only=true`, `device=cpu`, `precision=fp32`, `amp=disabled`; never expose a generic `--mock-gpu` or lower-trust CUDA bypass. Pin source commit and require authentic Colab execution context and operator sign-off. Keep CPU artifacts outside C0/C1 official run directories.
3. Authenticate Drive access and prove the canonical project root and persistent writable output. Stage the authorized TRAIN/VAL-only package and the exact Baseline R weights onto **Colab VM local disk**, verify SHA-256, verify split/hash, and explicitly enforce TRAIN-only for the measured optimizer steps. No sealed TEST access or staging.
4. Run **50 genuinely successful, disposable optimizer updates** with 2 real microbatches/update, actual Wav2Vec2 frozen forward, CTC+binary objective, backward, AdamW and scheduler behavior. Preserve seed 1337, batch 2, accumulation 2, frozen backbone, diagnosis-head exclusion and existing model head architecture. Use original C0 intervention semantics for the diagnostic benchmark, but tag the run **BENCHMARK_CPU_FP32**, not official C0; optionally repeat a comparable C1-style sample if its cost is materially different. A failure, skip or partial update does not count as a successful update. No benchmark update may be credited toward official C0/C1.
5. Time model initialization, data loading, forward, backward, optimizer and successful update latency with a warmup-aware distribution (e.g. p50/p95 and both early/late segments). Record actual machine CPU specification, vCPU count, `torch.get_num_threads()`, total and peak RAM/RSS, available disk, batch acoustic lengths, package versions, elapsed wall time and thermal/resource effects where observable. Include representative length variation; do not extrapolate solely from unusually short utterances.
6. Separately measure a real VAL-only evaluation pass and representative atomic checkpoint + Drive sync/readback cost **without labeling a benchmark checkpoint as a primary endpoint**. Benchmark-only artifacts must not overwrite Baseline R or arm output folders.
7. Save machine-readable benchmark evidence (timings, count of attempted/successful/skipped updates, max RSS, measurements and SHA-256) and a new Markdown step checkpoint (Step 10) with `PASS_MEASURED`, `INFEASIBLE_MEASURED`, `INCONCLUSIVE`, or `BLOCKED` as actually supported. Update overview + INDEX. Do not represent a static readiness review as a CPU benchmark result.

## 5. Scientific protocol and reporting separation

**Unchanged scientific experiment:** C0 `p=0.00`; C1 `p=0.25`, applied only to binary prompt embedding after cross-attention; CTC and canonical query paths unchanged; independent cold starts from the same Baseline R; identical seed=1337, split, model, optimizer AdamW lr=0.001/weight_decay=0.01, cosine/100 warmup, batch=2, accumulation=2, max 6 epochs/3,600 successful updates per arm; frozen backbone, CTC+binary training only, same evaluation protocols and CTC PER guardrail.

**Device/numerics deviation requiring formal approval:** Original `GPU_CUDA_FP16_AMP` versus proposed `COLAB_CPU_FP32_NO_AMP`. If CPU is approved for official runs, C0 and C1 must use the SAME variant, be labeled and checkpointed as CPU FP32, and not be presented as numerically equivalent to original GPU FP16 results. Preserve original GPU configs and launchers. Early halt or missing successful updates must be reported as INCOMPLETE, never as completed 3,600-update endpoint.

**Validation only** for checkpoint selection and inference threshold; preserve distinct Protocol O versus Z denominators, matched versus mismatched acoustic evidence, and CTC PER. Diagnosis head remains untrained. **Official TEST remains sealed**.

## 6. Execution status, blocker, restart point

| Action | Observed state | Supporting evidence |
| --- | --- | --- |
| Handoff, INDEX, AGENTS and Step 08 read | COMPLETE (source read) | Pinned GitHub commit and files |
| Production CPU training capability | NOT SUPPORTED | Explicit three-layer GPU guards and `amp: fp16` |
| Approved CPU FP32 variant | NOT AUTHORIZED / NOT IMPLEMENTED | No reviewed policy amendment or CPU runner |
| Live Colab CPU environment and quota | NOT QUERIED | No direct runtime-control capability in this session |
| Authorized Drive mount, staging, read/write | NOT VERIFIED for current session | Historical Step 08 failure only |
| 50 successful CPU FP32 benchmark updates | **0 / 50, NOT EXECUTED** | No runtime executed |
| Official C0 and C1 | **0 / 3,600 each, NOT STARTED** | No training evidence |
| Sealed TEST | UNTOUCHED IN THIS PROCESS | No TEST reads, extraction, or staging performed |
| Regression tests | NOT EXECUTED IN THIS PROCESS | Historical 190 passed / 4 skipped is agent-reported |

**Safe restart:** Review/approve a narrowly scoped Colab CPU FP32 benchmark protocol and implement/test its isolated runner (no official training); establish current Colab runtime and authenticated TRAIN/VAL-only Drive staging; then run disposable 50-update benchmark with live evidence and Step 10 checkpoint. If approval/staging is blocked, record BLOCKED and stop. Do not provision or train on GPU as an automatic fallback.

**Unmodified by this process:** scientific configs, model implementation, production launcher guards, original Baseline R weights, dataset package, and TEST sealing.
