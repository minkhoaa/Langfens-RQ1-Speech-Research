# Research environment notes

## Current runtime inventory

`research-runtime-20261114.txt` lists105effectiveuniquepackageversions captured from the owner's working CROTTC/SM13 Python3.11.17 runtime. Core: torch/torchaudio2.6.0+cu124, transformers4.57.0, speechbrain1.0.3.

This is **an inventory, not a portable lockfile with wheel hashes**. Public-index availability and clean installation on a teammate's machine have not been established. PyTorch CUDA12.4 builds use the corresponding official PyTorch index. The root `requirements-lock.txt` is historical and differs from current runtime.

Do not copy owner-specific `.pth` files. The owner `.venv-crottc-official` borrows packages from `.venv`; it is not a standalone environment. For collaboration use an independent Python3.11 environment, preserve exact relevant versions, and run the scoped fixtures in [ONBOARDING](../ONBOARDING.md) before touching private assets or production.

## Cleanup/restore incident

During authorized cleanup on2026-11-14, the main/recovery venv paths disappeared before this task's deletion precheck. The official env's borrowed `torch` stopped resolving. Exact package inventories had already been saved outside Git.

The main dependency env was restored **offline from existing UV extracted wheel payloads**, with generated console entrypoints.94main package versions and105effective official versions matched the saved inventory; CPU imports and85SM13targetedtests passed afterward. The original main payload was not hashed before disappearance; **whole-environment bitwise parity is not claimed**. No CROTTC process was restarted and no model weights/data/checkpoints were deleted by the repair.

The historical Recovery env had separately passed a real offline rebuild and26,712payload-file byte comparisons. That does not imply every venv in the repo is safe to delete. Temporary rebuild scratch and maintenance receipts live outside Git and are not project training assets.

See [SM13 dependency declaration](../docs/SM13_DEPENDENCIES.md) for the planning→data→backbone→training→benchmark chain, and `reports/TEAM_ONBOARDING_20261114/VALIDATION.md` (local evidence; not included in docs-only push) for scoped checks actually run during publication.
