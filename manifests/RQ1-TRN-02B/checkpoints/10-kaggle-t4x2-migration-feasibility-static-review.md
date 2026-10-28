# LANGFENS — RQ1-TRN-02B / 10-Kaggle T4×2 GPU Migration Feasibility (Static Review)

**Date:** 2026-10-28 (Asia/Ho_Chi_Minh)
**Process status:** `STATIC_PLATFORM_REVIEW_RECORDED`
**Kaggle authorization:** `NOT_AUTHORIZED` (current AGENTS.md allows training on Colab GPU only)
**Kaggle runtime/quota:** `NOT_QUERIED` / `NOT_PROVISIONED`
**Training (C0, C1):** `NOT_STARTED`, each 0/3600 successful optimizer updates
**CPU FP32 benchmark:** `NOT_EXECUTED`, 0/50 successful updates
**Evidence:** GitHub source inspection + public Kaggle documentation; no remote runtime execution
**Source baseline:** `72e38bb1c4b9d66311aec43cf9cfeabed5b96bbd` (Step 09 documentation)
**Original frozen production GPU R3 implementation:** `ba7be0dc407313a4f5e31577051e6746278adeed`

## 1. Objective

Assess whether **Kaggle GPU T4×2** is a suitable alternative to Colab GPU and Colab CPU for official LANGFENS Fold 1 RQ1-TRN-02B acoustic conditioning C0/C1. Preserve experiment semantics, original GPU CUDA FP16 AMP numerical regime, separate per-arm training state, and sealed TEST isolation. No protocol, training or runtime changes are authorized by this document.

## 2. Public platform facts checked on 2026-10-28

- Kaggle staff announced the Tesla P100 retirement effective **2026-09-15**. T4×2 (2 Tesla T4 devices, 16 GiB each) remains offered for standard notebooks. A single process does **not** get 32 GiB combined VRAM.
  Source: https://www.kaggle.com/product-announcements/735239
- Kaggle Notebook documentation states GPU sessions **up to 12h**, roughly **29 GB host RAM** for T4×2, and `/kaggle/working` saved-output limit **20 GB**; actual accelerator availability and individual quota remain account-specific.
  Source: https://www.kaggle.com/docs/notebooks
- Kaggle's official GPU tips describe **~30h/week** floating GPU allocation; actual hours may change with demand.
  Source: https://www.kaggle.com/docs/efficient-gpu-usage
- Kaggle supports **private Datasets** and private Notebooks; Datasets and Notebooks have distinct sharing permissions. An uploaded ZIP may be **automatically extracted**, so a previously pinned archive-level SHA-256 cannot simply be assumed to remain directly available within mounted `/kaggle/input`.
  Source: https://www.kaggle.com/docs/datasets
- Kaggle CLI supports `kernel-metadata.json` with `enable_gpu`, `machine_shape: NvidiaTeslaT4`, `is_private` and `dataset_sources`; `kaggle kernels push --no-run` can save without executing. This is not evidence the user's account is currently entitled to an accelerator.
  Sources: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md and https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md

## 3. Project-specific runtime evidence and estimate (NOT Kaggle benchmark)

- Historical Baseline R **Colab Tesla T4** end-to-end reported 71–91 seconds per epoch, 11 epochs and 6600 successful updates in approximately 855 seconds; frozen Wav2Vec2, CTC+binary only, batch 2, accumulation 2, CUDA FP16 AMP.
  Repository source: `manifests/RQ1-TRN-02A/GPU_FEASIBILITY_REPORT.md` and `TRAINING_HISTORY.json`.
- C0 and C1 require 6 epochs / max 3600 successful updates **each**, total 12 epochs / 7200 updates if endpoint reached.
- If a Kaggle T4 matches the **historical** 71–91 s/epoch baseline, pure sequential epoch-time extrapolation is approximately 14–18 minutes for 12 epochs, excluding new overhead and code/runtime differences. **This is not measured C0/C1 speed or a promise**. Conservative setup, staging, upload, dependency and checkpoint time may materially increase total time; account quota could block allocation entirely.
- Two separate T4 GPUs could technically each host one independent arm process, but parallelism must not be enabled without CPU/RAM/I/O and deterministic isolation tests. **Default feasibility strategy: sequential C0 then C1 on one GPU**. The second T4 need not be used.

## 4. Migration boundary: execution platform, NOT scientific intervention

This is a **platform port** `KAGGLE_T4_CUDA_FP16_AMP` of the original `COLAB_T4_CUDA_FP16_AMP` regime, **not** the unapproved `COLAB_CPU_FP32_NO_AMP` numerical variant.

Preserve immutable: C0 binary-only full-phone prompt embedding dropout `p=0.00`; C1 `p=0.25` (post-attention, CTC and query paths unaffected), independent cold starts from the SAME Baseline R, seed 1337, batch 2, accumulation 2, AdamW learning rate 1e-3/weight decay 0.01, cosine LR with 100 warmup updates, maximum six epochs/3600 successful updates per arm, frozen Wav2Vec2, CTC+binary two-head only, diagnosis head untrained, independent optimizer/scheduler/scaler/RNG/run IDs, matched training exposure and evaluation protocol.

Pin training source SHA, PyTorch/CUDA/driver/Transformers environment and record algorithm/numerics differences. Same FP16 precision and Tesla T4 model do NOT guarantee bitwise replay across infrastructure; both official arms must use identical Kaggle software and hardware. Never silently present CPU FP32 as equivalent to GPU FP16 or mix arms across regimes.

## 5. Required safe implementation before any Kaggle backward pass

1. **Explicit operator approval** for Kaggle GPU execution and a narrow `AGENTS.md` amendment; keep Colab launcher guardrails intact. Do not enable a generic bypass/force flag. Review source changes and regression tests on a review branch.
2. Verify current Kaggle GPU quota, account eligibility, T4×2 provisioning, actual GPU UUIDs, available RAM/disk, and all runtime package versions using a real Kaggle session, after approval.
3. Stage **only authorized TRAIN+VAL** dataset (split SHA-256 `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`, authorized archive SHA-256 `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`) and Baseline R `best.pt` SHA-256 `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`. Never upload/stage/test official SEALED TEST audio or annotation.
4. Before uploading any L2-ARCTIC audio/annotations to private Kaggle storage, verify its **CC BY-NC 4.0** license and third-party storage redistribution permissions; `private` is an access setting, not evidence of licensing permission. Alternative: authenticated download from authorized Drive into ephemeral Kaggle runtime, with verified checksum and secrets hygiene. Never publish a public Kaggle dataset or notebook containing restricted material.
5. Kaggle datasets mount read-only under `/kaggle/input`; output writes go to `/kaggle/working`. Add Kaggle-specific authenticated staging and persistence adapter. An auto-extracted ZIP needs equivalent documented member-level integrity verification **in addition to** preserving canonical archive provenance; never skip checksum gates because archive paths changed.
6. Existing `audit_scripts/colab_stage_and_verify.py` and `audit_scripts/colab_train_exec.py` hardcode Colab/Drive trust boundaries; do NOT use these directly or remove guards globally. Build a reviewed Kaggle-specific runner that provides equivalent strict source/hash checks, per-arm output uniqueness, checkpoint synchronization, and fail-closed endpoint classification.
7. The original protocol requires durable checkpoint mirroring after **every epoch**, with remote SHA/readback. Kaggle interactive `/kaggle/working` alone and Save Version at the end are **not proof of durable mid-run recovery**. Implement and test an authorized secure remote mirror to Drive or other explicitly approved private storage; account for 20 GB output quota and one run ID per arm.
8. After approval and staging verification, run a disposable, real-data **50-successful-update Kaggle T4 FP16** preflight (not counted toward C0). Measure p50/p95 update latency, GPU peak reserved memory, epoch-validation and checkpoint-transfer overhead, failed/skipped update counts, and derive estimated full-arm time. Do not claim pass until actually executed.
9. If resource and integrity gates pass, obtain explicit official run sign-off, train **C0 and C1 independently**, validate only TRAIN/VAL, write numbered Markdown checkpoint after each process and update phase overview + global INDEX. STOP on failure, no CPU fallback or silent change in experimental budget.

## 6. Verdict and safe restart

**Verdict: `KAGGLE_T4X2_FEASIBLE_IN_PRINCIPLE; EXECUTION_NOT_AUTHORIZED; BENCHMARK_NOT_EXECUTED`.**

Unlike CPU FP32, Kaggle T4 CUDA FP16 preserves the original model precision regime and already has strong historical Colab T4 throughput evidence, but availability, checkpoint durability, data licensing and GPU execution authorization remain unresolved.

**Next step:** Obtain explicit scope approval for Kaggle GPU platform port; implement reviewable Kaggle-specific adapter and tests; authenticate TRAIN/VAL storage; perform genuine 50-successful-update benchmark and record a new Step 11 execution checkpoint. Keep original frozen GPU R3 and Step 09 CPU FP32 proposal unchanged.

**Scientific status:** C0=0/3600; C1=0/3600; Kaggle T4 preflight=0/50; CPU FP32 preflight=0/50. No user TEST data accessed; no actual training or Kaggle runtime started.
