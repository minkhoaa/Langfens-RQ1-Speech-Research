#!/usr/bin/env python3
"""
LANGFENS — RQ1-TRN-03A: Generator for Simplified Kaggle CLI Deployment Packages.

Creates dedicated, independent preflight, smoke, and training deployment packages
for cross-fold execution (Folds 2, 3, etc.):
  - deploy/kaggle_fold{k}_staging/
  - deploy/kaggle_fold{k}_preflight/
  - deploy/kaggle_fold{k}_smoke/
  - deploy/kaggle_fold{k}_train/

Replaces notebook execution flags with separate entrypoints independently
runnable via `kaggle kernels push --no-run`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PINNED_COMMIT_SHA = "cc56b82ca1a9a6ceeca85929e07db4cac8537348"

FOLD_SPECS = {
    2: {
        "dataset_slug": "minkhoaa/langfens-rq1-fold2-staging",
        "dataset_title": "LANGFENS RQ1 Fold 2 Train/Val Staging Package",
        "dataset_sha256": "112cc9e0cc5e37e5fa034df504eca019dd6ad438c96cb642335ecf10a802f0d9",
        "test_speakers": "BWC, HKK, HQTV, MBMPS, SKA, SVBI",
        "preflight_slug": "minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2-preflight",
        "smoke_slug": "minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2-smoke",
        "train_slug": "minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2",
    },
    3: {
        "dataset_slug": "minkhoaa/langfens-rq1-fold3-staging",
        "dataset_title": "LANGFENS RQ1 Fold 3 Train/Val Staging Package",
        "dataset_sha256": "e81ac9e5d95a68d7f02c7b2ea8e5f1e9fd89b047926ffeeb4d096b0a3a5378a2",
        "test_speakers": "ERMS, NCC, RRBI, THV, YBAA, YDCK",
        "preflight_slug": "minkhoaa/langfens-rq1-fold-3-baseline-r-t4x2-preflight",
        "smoke_slug": "minkhoaa/langfens-rq1-fold-3-baseline-r-t4x2-smoke",
        "train_slug": "minkhoaa/langfens-rq1-fold-3-baseline-r-t4x2",
    },
}


def make_preflight_notebook(fold_idx: int = 2) -> dict:
    spec = FOLD_SPECS[fold_idx]
    test_speakers_fmt = ", ".join(f"`{s.strip()}`" for s in spec["test_speakers"].split(","))
    cells = [
        {
            "cell_type": "markdown",
            "id": "markdown-overview",
            "metadata": {},
            "source": [
                f"# LANGFENS — RQ1-TRN-03A: Dedicated Fold {fold_idx} Preflight Verification on Kaggle Tesla T4x2\n",
                f"Official preflight verification notebook for Fold {fold_idx} Baseline R (`KAGGLE_T4_CUDA_FP16_AMP`).\n",
                "\n",
                "## Operational Role & Safety Invariants\n",
                "- **Role:** Independent, standalone preflight staging and persistence verification.\n",
                "- **Preflight Only:** Zero training steps, zero optimizer updates, zero model weights generated.\n",
                "- **Hardware Target:** Tesla T4 × 2 GPU (`Driver: >=580`, `CUDA: >=13.0`, 15,360 MiB VRAM per GPU).\n",
                f"- **Dataset Staging:** Authorized Fold {fold_idx} Train+Val package (`{spec['dataset_slug']}`).\n",
                f"- **Sealed Test Isolation:** Active test speakers ({test_speakers_fmt}) are 100% physically absent.\n",
                "- **Manifest Integrity:** Verifies split manifest SHA-256 (`7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`).\n",
                f"- **Dataset Integrity:** Verifies fold package SHA-256 (`{spec['dataset_sha256']}`).\n",
                "- **Remote Persistence Probe:** Validates live authenticated write-readback-delete probe to private Hugging Face model repository `minkhoaa/langfens-rq1-cross-fold-checkpoints`.\n",
                "- **Credential Hygiene:** Resolves `HF_TOKEN` strictly via Kaggle Secrets (`UserSecretsClient`). Zero token files on disk, zero CLI token arguments.\n",
                f"- **Reviewed Training Base Commit:** `{PINNED_COMMIT_SHA}`.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "code-hardware-check",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Phase 1: Hardware & Environment Verification\n",
                "import subprocess, sys\n",
                "\n",
                "# Checked execution of nvidia-smi\n",
                "subprocess.run([\"nvidia-smi\"], check=True)\n",
                "\n",
                "import torch\n",
                "print(f\"[GPU] CUDA Available: {torch.cuda.is_available()}\")\n",
                "if not torch.cuda.is_available():\n",
                "    raise RuntimeError(\"FAIL-CLOSED: CUDA GPU is not available on this runtime!\")\n",
                "print(f\"[GPU] Device Count: {torch.cuda.device_count()}\")\n",
                "for i in range(torch.cuda.device_count()):\n",
                "    print(f\"[GPU {i}]: {torch.cuda.get_device_name(i)}\")\n",
                "assert \"Tesla T4\" in torch.cuda.get_device_name(0), (\n",
                "    f\"FAIL-CLOSED: Expected Tesla T4 accelerator, got {torch.cuda.get_device_name(0)}\"\n",
                ")\n",
                "\n",
                "# Verify required core dependencies\n",
                "import yaml, transformers, huggingface_hub\n",
                "print(f\"[ENV] PyTorch version: {torch.__version__}\")\n",
                "print(f\"[ENV] Transformers version: {transformers.__version__}\")\n",
                "print(f\"[ENV] Hugging Face Hub version: {huggingface_hub.__version__}\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "code-secrets-resolution",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Phase 2: Credentials & Secrets Resolution (Strict Kaggle Secrets Path)\n",
                "# In accordance with AGENTS.md, tokens are never read from disk files or passed via CLI.\n",
                "import os\n",
                "\n",
                "try:\n",
                "    from kaggle_secrets import UserSecretsClient\n",
                "    token = UserSecretsClient().get_secret(\"HF_TOKEN\")\n",
                "    if token and token.strip():\n",
                "        os.environ[\"HF_TOKEN\"] = token.strip()\n",
                "        print(\"[AUTH] Successfully loaded HF_TOKEN from Kaggle Secrets.\")\n",
                "except Exception as e:\n",
                "    print(f\"[AUTH NOTE] Kaggle Secrets UserSecretsClient query: {e}\")\n",
                "\n",
                "if \"HF_TOKEN\" not in os.environ or not os.environ[\"HF_TOKEN\"].strip():\n",
                "    raise RuntimeError(\n",
                "        \"FAIL-CLOSED: HF_TOKEN is required in Kaggle Secrets (or environment) for private remote mirroring. \"\n",
                "        \"Attach HF_TOKEN under Add-ons -> Secrets in the Kaggle notebook.\"\n",
                "    )\n",
                "print(\"[AUTH] Verified HF_TOKEN presence in environment for private remote mirroring.\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "code-checkout-commit",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Phase 3: Clone, Checkout Reviewed Training Commit & Verify Tree\n",
                "import os, subprocess, sys\n",
                "from pathlib import Path\n",
                "\n",
                "repo_dir = Path(\"/kaggle/working/repo\")\n",
                "if not repo_dir.exists():\n",
                "    print(\"[GIT] Cloning repository into /kaggle/working/repo...\")\n",
                "    subprocess.run(\n",
                "        [\"git\", \"clone\", \"https://github.com/minkhoaa/Langfens-RQ1-Speech-Research.git\", str(repo_dir)],\n",
                "        check=True\n",
                "    )\n",
                "\n",
                "os.chdir(str(repo_dir))\n",
                "subprocess.run([\"git\", \"fetch\", \"--all\"], cwd=repo_dir, check=True)\n",
                "\n",
                "# Pinned immutable training source commit (closing Step 04C launch-gate corrections)\n",
                f'PINNED_COMMIT_SHA = "{PINNED_COMMIT_SHA}"\n',
                "print(f\"[GIT] Checking out pinned training commit: {PINNED_COMMIT_SHA}...\")\n",
                "subprocess.run([\"git\", \"checkout\", \"--detach\", PINNED_COMMIT_SHA], cwd=repo_dir, check=True)\n",
                "\n",
                "actual_head = subprocess.check_output([\"git\", \"rev-parse\", \"HEAD\"], cwd=repo_dir, text=True).strip().lower()\n",
                "print(f\"[GIT] Verified Training Commit HEAD: {actual_head}\")\n",
                "assert actual_head == PINNED_COMMIT_SHA.lower(), (\n",
                "    f\"FAIL-CLOSED: Commit SHA mismatch: actual {actual_head} != expected {PINNED_COMMIT_SHA}\"\n",
                ")\n",
                "\n",
                "status = subprocess.check_output([\"git\", \"status\", \"--porcelain\"], cwd=repo_dir, text=True).strip()\n",
                "assert len(status) == 0, f\"Dirty tree: {status}\"\n",
                "print(\"[GIT] Repository verified clean at pinned training commit.\")\n",
                "\n",
                "sys.path.insert(0, str(repo_dir / \"src\"))\n",
                "sys.path.insert(0, str(repo_dir))\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "code-preflight-verify",
            "metadata": {},
            "outputs": [],
            "source": [
                f"# Phase 4: Staging Audit & Preflight Verification (verify_fold{fold_idx})\n",
                "# Executes preflight staging check, validates split manifest SHA, verifies 0 test speaker\n",
                "# leakage, checks auto-extracted outer zip reconstruction, and executes live Hugging Face\n",
                "# write-readback-delete probe. Zero model weights or training steps are executed in this entrypoint.\n",
                "import subprocess, sys\n",
                "from pathlib import Path\n",
                "\n",
                f"staging_input = Path(\"/kaggle/input/{spec['dataset_slug'].split('/')[-1]}\")\n",
                "print(f\"[INPUT AUDIT] Checking dataset input directory: {staging_input}\")\n",
                "if staging_input.is_dir():\n",
                "    print(f\"[INPUT AUDIT] Directory contents: {[p.name for p in staging_input.iterdir()]}\")\n",
                "else:\n",
                "    print(f\"[INPUT AUDIT WARNING] {staging_input} does not exist yet (requires dataset attachment).\")\n",
                "\n",
                "preflight_cmd = [\n",
                "    sys.executable,\n",
                "    \"deploy/kaggle/kaggle_entrypoint.py\",\n",
                f"    \"--mode\", \"verify_fold{fold_idx}\",\n",
                "    \"--commit-sha\", PINNED_COMMIT_SHA,\n",
                "    \"--kaggle-input-dir\", str(staging_input),\n",
                "    \"--remote-mirror-backend\", \"huggingface\",\n",
                "    \"--remote-mirror-target\", \"minkhoaa/langfens-rq1-cross-fold-checkpoints\",\n",
                "]\n",
                "print(\"[PREFLIGHT] Executing checked command:\")\n",
                "print(\"  \" + \" \".join(preflight_cmd))\n",
                "subprocess.run(preflight_cmd, check=True)\n",
                "\n",
                "print(\"=\" * 80)\n",
                "print(\"[PREFLIGHT] Preflight verification PASSED successfully.\")\n",
                "print(\"[PREFLIGHT] Zero training, optimizer updates, or checkpoint generation occurred.\")\n",
                "print(\"=\" * 80)\n"
            ]
        }
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 5
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def make_smoke_notebook(fold_idx: int = 2) -> dict:
    spec = FOLD_SPECS[fold_idx]
    cells = [
        {
            "cell_type": "markdown",
            "id": "markdown-overview",
            "metadata": {},
            "source": [
                f"# LANGFENS — RQ1-TRN-03A: Dedicated Fold {fold_idx} Smoke Training on Kaggle Tesla T4x2\n",
                f"Official 1-epoch smoke execution notebook for Fold {fold_idx} Baseline R (`KAGGLE_T4_CUDA_FP16_AMP`).\n",
                "\n",
                "## Smoke Protocol & Operational Invariants\n",
                "- **Scope:** Strictly 1 completed epoch (max ~600 updates) for execution and persistence verification.\n",
                f"- **Run ID:** `fold{fold_idx}_smoke_1epoch` (isolated checkpoint hierarchy; zero interference with official Baseline R).\n",
                "- **Architecture:** Two-head Wav2Vec2 + Binary MDD (frozen backbone `facebook/wav2vec2-base-960h`, rev `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`)\n",
                "- **Initialization:** Strictly fresh pretrained initialization; zero warm-start from Fold 1.\n",
                "- **Seeding & Precision:** Seed 1337, CUDA FP16 AMP (`init_scale: 4096.0`), AdamW (`lr: 1e-3`, `weight_decay: 0.01`).\n",
                "- **Batching & Step Dynamics:** `batch_size: 2`, `gradient_accumulation_steps: 2` (effective batch size = 4 utterances per update).\n",
                "- **Scheduler Horizon:** 25 epochs (~15,000 updates, 100 warmup updates) preserving Fold 1 Baseline R cosine decay rate.\n",
                "- **Execution Cap:** Strictly 1 completed epoch (~600 updates).\n",
                f"- **Data & Test Isolation:** Authorized Fold {fold_idx} Train+Val package (18 speakers; 6 test speakers strictly excluded).\n",
                f"- **Remote Mirroring:** Per-epoch atomic mirror to private Hugging Face repo `minkhoaa/langfens-rq1-cross-fold-checkpoints` under prefix `fold{fold_idx}/fold{fold_idx}_smoke_1epoch/` with SHA-256 readback.\n",
                "- **Mandatory Preflight Gate:** Runs preflight first and fails closed on any error before training begins.\n",
                f"- **Reviewed Training Base Commit:** `{PINNED_COMMIT_SHA}`.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "code-hardware-check",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Phase 1: Hardware & Environment Verification\n",
                "import subprocess, sys\n",
                "\n",
                "# Checked execution of nvidia-smi\n",
                "subprocess.run([\"nvidia-smi\"], check=True)\n",
                "\n",
                "import torch\n",
                "print(f\"[GPU] CUDA Available: {torch.cuda.is_available()}\")\n",
                "if not torch.cuda.is_available():\n",
                "    raise RuntimeError(\"FAIL-CLOSED: CUDA GPU is not available on this runtime!\")\n",
                "print(f\"[GPU] Device Count: {torch.cuda.device_count()}\")\n",
                "for i in range(torch.cuda.device_count()):\n",
                "    print(f\"[GPU {i}]: {torch.cuda.get_device_name(i)}\")\n",
                "assert \"Tesla T4\" in torch.cuda.get_device_name(0), (\n",
                "    f\"FAIL-CLOSED: Expected Tesla T4 accelerator, got {torch.cuda.get_device_name(0)}\"\n",
                ")\n",
                "\n",
                "# Verify required core dependencies\n",
                "import yaml, transformers, huggingface_hub\n",
                "print(f\"[ENV] PyTorch version: {torch.__version__}\")\n",
                "print(f\"[ENV] Transformers version: {transformers.__version__}\")\n",
                "print(f\"[ENV] Hugging Face Hub version: {huggingface_hub.__version__}\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "code-secrets-resolution",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Phase 2: Credentials & Secrets Resolution (Strict Kaggle Secrets Path)\n",
                "# In accordance with AGENTS.md, tokens are never read from disk files or passed via CLI.\n",
                "import os\n",
                "\n",
                "try:\n",
                "    from kaggle_secrets import UserSecretsClient\n",
                "    token = UserSecretsClient().get_secret(\"HF_TOKEN\")\n",
                "    if token and token.strip():\n",
                "        os.environ[\"HF_TOKEN\"] = token.strip()\n",
                "        print(\"[AUTH] Successfully loaded HF_TOKEN from Kaggle Secrets.\")\n",
                "except Exception as e:\n",
                "    print(f\"[AUTH NOTE] Kaggle Secrets UserSecretsClient query: {e}\")\n",
                "\n",
                "if \"HF_TOKEN\" not in os.environ or not os.environ[\"HF_TOKEN\"].strip():\n",
                "    raise RuntimeError(\n",
                "        \"FAIL-CLOSED: HF_TOKEN is required in Kaggle Secrets (or environment) for private remote mirroring. \"\n",
                "        \"Attach HF_TOKEN under Add-ons -> Secrets in the Kaggle notebook.\"\n",
                "    )\n",
                "print(\"[AUTH] Verified HF_TOKEN presence in environment for private remote mirroring.\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "code-checkout-commit",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Phase 3: Clone, Checkout Reviewed Training Commit & Verify Tree\n",
                "import os, subprocess, sys\n",
                "from pathlib import Path\n",
                "\n",
                "repo_dir = Path(\"/kaggle/working/repo\")\n",
                "if not repo_dir.exists():\n",
                "    print(\"[GIT] Cloning repository into /kaggle/working/repo...\")\n",
                "    subprocess.run(\n",
                "        [\"git\", \"clone\", \"https://github.com/minkhoaa/Langfens-RQ1-Speech-Research.git\", str(repo_dir)],\n",
                "        check=True\n",
                "    )\n",
                "\n",
                "os.chdir(str(repo_dir))\n",
                "subprocess.run([\"git\", \"fetch\", \"--all\"], cwd=repo_dir, check=True)\n",
                "\n",
                f'PINNED_COMMIT_SHA = "{PINNED_COMMIT_SHA}"\n',
                "print(f\"[GIT] Checking out pinned training commit: {PINNED_COMMIT_SHA}...\")\n",
                "subprocess.run([\"git\", \"checkout\", \"--detach\", PINNED_COMMIT_SHA], cwd=repo_dir, check=True)\n",
                "\n",
                "actual_head = subprocess.check_output([\"git\", \"rev-parse\", \"HEAD\"], cwd=repo_dir, text=True).strip().lower()\n",
                "print(f\"[GIT] Verified Training Commit HEAD: {actual_head}\")\n",
                "assert actual_head == PINNED_COMMIT_SHA.lower(), (\n",
                "    f\"FAIL-CLOSED: Commit SHA mismatch: actual {actual_head} != expected {PINNED_COMMIT_SHA}\"\n",
                ")\n",
                "\n",
                "status = subprocess.check_output([\"git\", \"status\", \"--porcelain\"], cwd=repo_dir, text=True).strip()\n",
                "assert len(status) == 0, f\"Dirty tree: {status}\"\n",
                "print(\"[GIT] Repository verified clean at pinned training commit.\")\n",
                "\n",
                "sys.path.insert(0, str(repo_dir / \"src\"))\n",
                "sys.path.insert(0, str(repo_dir))\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "code-mandatory-preflight-gate",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Phase 4: Mandatory Preflight Check (Fail-Closed Gate)\n",
                "# Training entrypoint executes preflight checks first; fails closed immediately on any error.\n",
                "import subprocess, sys\n",
                "from pathlib import Path\n",
                "\n",
                f"staging_input = Path(\"/kaggle/input/{spec['dataset_slug'].split('/')[-1]}\")\n",
                "print(f\"[PREFLIGHT-GATE] Auditing dataset input directory: {staging_input}\")\n",
                "if staging_input.is_dir():\n",
                "    print(f\"[PREFLIGHT-GATE] Directory contents: {[p.name for p in staging_input.iterdir()]}\")\n",
                "else:\n",
                "    print(f\"[PREFLIGHT-GATE WARNING] {staging_input} does not exist yet (requires dataset attachment).\")\n",
                "\n",
                "preflight_cmd = [\n",
                "    sys.executable,\n",
                "    \"deploy/kaggle/kaggle_entrypoint.py\",\n",
                f"    \"--mode\", \"verify_fold{fold_idx}\",\n",
                "    \"--commit-sha\", PINNED_COMMIT_SHA,\n",
                "    \"--kaggle-input-dir\", str(staging_input),\n",
                "    \"--remote-mirror-backend\", \"huggingface\",\n",
                "    \"--remote-mirror-target\", \"minkhoaa/langfens-rq1-cross-fold-checkpoints\",\n",
                "]\n",
                "print(\"[PREFLIGHT-GATE] Executing mandatory preflight verification:\")\n",
                "print(\"  \" + \" \".join(preflight_cmd))\n",
                "subprocess.run(preflight_cmd, check=True)\n",
                "print(\"=\" * 80)\n",
                "print(\"[PREFLIGHT] Preflight verification PASSED successfully. Proceeding to smoke training...\")\n",
                "print(\"=\" * 80)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "code-smoke-training-execution",
            "metadata": {},
            "outputs": [],
            "source": [
                f"# Phase 5: Isolated Smoke Training Execution (1 epoch, run ID fold{fold_idx}_smoke_1epoch)\n",
                "import subprocess, sys\n",
                "from pathlib import Path\n",
                "\n",
                "print(\"=\" * 80)\n",
                f"print(\"[SMOKE LAUNCH] Starting Fold {fold_idx} Smoke Training on Tesla T4...\")\n",
                f"print(f\"[SMOKE LAUNCH] Run ID: fold{fold_idx}_smoke_1epoch\")\n",
                "print(f\"[SMOKE LAUNCH] Execution Cap: 1 completed epoch (~600 updates)\")\n",
                "print(f\"[SMOKE LAUNCH] Pinned Commit SHA: {PINNED_COMMIT_SHA}\")\n",
                "print(\"=\" * 80)\n",
                "\n",
                "smoke_cmd = [\n",
                "    sys.executable,\n",
                "    \"deploy/kaggle/kaggle_entrypoint.py\",\n",
                f"    \"--mode\", \"train_fold{fold_idx}\",\n",
                "    \"--commit-sha\", PINNED_COMMIT_SHA,\n",
                "    \"--kaggle-input-dir\", str(staging_input),\n",
                f"    \"--run-id\", \"fold{fold_idx}_smoke_1epoch\",\n",
                "    \"--execution-cap-epochs\", \"1\",\n",
                "    \"--remote-mirror-backend\", \"huggingface\",\n",
                "    \"--remote-mirror-target\", \"minkhoaa/langfens-rq1-cross-fold-checkpoints\",\n",
                "]\n",
                "print(\"[SMOKE LAUNCH] Executing checked command:\")\n",
                "print(\"  \" + \" \".join(smoke_cmd))\n",
                "subprocess.run(smoke_cmd, check=True)\n",
                f"print(\"[SMOKE LAUNCH] Fold {fold_idx} Smoke training subprocess completed successfully.\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "code-smoke-artifact-verification",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Phase 6: Post-Smoke Artifact & Metric Verification\n",
                "import json\n",
                "from pathlib import Path\n",
                "\n",
                f"smoke_dir = Path(\"/kaggle/working/checkpoints/fold{fold_idx}/fold{fold_idx}_smoke_1epoch\")\n",
                "print(f\"[VERIFY SMOKE] Checking smoke directory: {smoke_dir}\")\n",
                "assert smoke_dir.is_dir(), f\"Smoke output directory missing: {smoke_dir}\"\n",
                "assert (smoke_dir / \"last.pt\").is_file(), f\"Missing last.pt in {smoke_dir}\"\n",
                "assert (smoke_dir / \"best.pt\").is_file(), f\"Missing best.pt in {smoke_dir}\"\n",
                "assert (smoke_dir / \"history.json\").is_file(), f\"Missing history.json in {smoke_dir}\"\n",
                "\n",
                "hist = json.loads((smoke_dir / \"history.json\").read_text())\n",
                "assert len(hist) == 1, f\"Expected exactly 1 completed epoch, got {len(hist)}\"\n",
                "ep1 = hist[0]\n",
                "print(\"=\" * 80)\n",
                "print(\"[SMOKE EXECUTION COMPLETED SUCCESSFULLY]\")\n",
                "print(f\"Epoch: {ep1.get('epoch')}\")\n",
                "print(f\"Updates: {ep1.get('successful_updates_total')}\")\n",
                "print(f\"Micro Steps: {ep1.get('micro_steps_total')}\")\n",
                "print(f\"Val Binary AP: {ep1.get('val_binary_ap')}\")\n",
                "print(f\"Val CTC PER: {ep1.get('val_ctc_per')}\")\n",
                "print(f\"Elapsed Time: {ep1.get('wall_time_seconds')}s\")\n",
                "print(\"=\" * 80)\n"
            ]
        }
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 5
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def make_train_notebook(fold_idx: int = 2) -> dict:
    spec = FOLD_SPECS[fold_idx]
    test_speakers_fmt = ", ".join(f"`{s.strip()}`" for s in spec["test_speakers"].split(","))
    cells = [
        {
            "cell_type": "markdown",
            "id": "markdown-overview",
            "metadata": {},
            "source": [
                f"# LANGFENS — RQ1-TRN-03A: Dedicated Fold {fold_idx} Baseline R Training on Kaggle Tesla T4x2\n",
                f"Official execution notebook for Fold {fold_idx} Baseline R training (`KAGGLE_T4_CUDA_FP16_AMP`).\n",
                "\n",
                "## Scientific Protocol & Provenance Invariants\n",
                "- **Architecture:** Two-head Wav2Vec2 + Binary MDD (frozen backbone `facebook/wav2vec2-base-960h`, revision `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`)\n",
                "- **Initialization:** Strictly fresh pretrained initialization; zero warm-start or checkpoint transfer from Fold 1.\n",
                "- **Seeding & Precision:** Seed 1337, CUDA FP16 AMP (`init_scale: 4096.0`), AdamW (`lr: 1e-3`, `weight_decay: 0.01`).\n",
                "- **Batching & Step Dynamics:** `batch_size: 2`, `gradient_accumulation_steps: 2` (effective batch size = 4 utterances per update).\n",
                "- **Decoupled Horizon & Cap:**\n",
                "  - Scheduler Horizon: 25 epochs (~15,000 updates, 100 warmup updates) preserving Fold 1 Baseline R cosine decay rate.\n",
                "  - Execution Cap: 10 completed epochs (max 6,000 updates), strictly bounding Kaggle GPU quota usage.\n",
                "- **Dual Checkpoint Accounting:**\n",
                "  - Fixed-Budget Primary Endpoint: Strictly captured and locked at Epoch 6 / 3,600 updates (`fixed_budget_checkpoint.pt`).\n",
                "  - Validation-Selected Endpoint: Captured at peak validation AP prior to early stopping (`best.pt`, patience 5).\n",
                "- **Data & Test Isolation:**\n",
                f"  - Dataset: Authorized Fold {fold_idx} Train+Val package (`{spec['dataset_slug']}`).\n",
                f"  - Sealed Test Exclusion: Active test speakers ({test_speakers_fmt}) are 100% physically absent.\n",
                "- **Credential & Durability Hygiene:**\n",
                "  - Authentication: strictly via Kaggle Secrets (`HF_TOKEN`) or environment variables; zero token file fallbacks.\n",
                f"  - Remote Mirroring: Per-epoch atomic mirror to private Hugging Face repo `minkhoaa/langfens-rq1-cross-fold-checkpoints` with SHA-256 readback.\n",
                "- **Run ID & Persistence Hierarchy:**\n",
                f"  - Run ID: `baseline_r_fold{fold_idx}`.\n",
                f"  - Local Checkpoints: `/kaggle/working/checkpoints/fold{fold_idx}/baseline_r_fold{fold_idx}`.\n",
                f"  - Remote Mirror: private Hugging Face repo `minkhoaa/langfens-rq1-cross-fold-checkpoints` under prefix `fold{fold_idx}/baseline_r_fold{fold_idx}/` with SHA-256 readback.\n",
                "- **Mandatory Preflight Gate:** Runs preflight first and fails closed on any error before training begins.\n",
                f"- **Reviewed Training Source Commit:** `{PINNED_COMMIT_SHA}` (pinned training codebase).\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "code-hardware-check",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Phase 1: Hardware & Environment Verification\n",
                "import subprocess, sys\n",
                "\n",
                "# Checked execution of nvidia-smi\n",
                "subprocess.run([\"nvidia-smi\"], check=True)\n",
                "\n",
                "import torch\n",
                "print(f\"[GPU] CUDA Available: {torch.cuda.is_available()}\")\n",
                "if not torch.cuda.is_available():\n",
                "    raise RuntimeError(\"FAIL-CLOSED: CUDA GPU is not available on this runtime!\")\n",
                "print(f\"[GPU] Device Count: {torch.cuda.device_count()}\")\n",
                "for i in range(torch.cuda.device_count()):\n",
                "    print(f\"[GPU {i}]: {torch.cuda.get_device_name(i)}\")\n",
                "assert \"Tesla T4\" in torch.cuda.get_device_name(0), (\n",
                "    f\"FAIL-CLOSED: Expected Tesla T4 accelerator, got {torch.cuda.get_device_name(0)}\"\n",
                ")\n",
                "\n",
                "# Verify required core dependencies\n",
                "import yaml, transformers, huggingface_hub\n",
                "print(f\"[ENV] PyTorch version: {torch.__version__}\")\n",
                "print(f\"[ENV] Transformers version: {transformers.__version__}\")\n",
                "print(f\"[ENV] Hugging Face Hub version: {huggingface_hub.__version__}\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "code-secrets-resolution",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Phase 2: Credentials & Secrets Resolution (Strict Kaggle Secrets Path)\n",
                "# In accordance with AGENTS.md, tokens are never read from disk files or passed via CLI.\n",
                "import os\n",
                "\n",
                "try:\n",
                "    from kaggle_secrets import UserSecretsClient\n",
                "    token = UserSecretsClient().get_secret(\"HF_TOKEN\")\n",
                "    if token and token.strip():\n",
                "        os.environ[\"HF_TOKEN\"] = token.strip()\n",
                "        print(\"[AUTH] Successfully loaded HF_TOKEN from Kaggle Secrets.\")\n",
                "except Exception as e:\n",
                "    print(f\"[AUTH NOTE] Kaggle Secrets UserSecretsClient query: {e}\")\n",
                "\n",
                "if \"HF_TOKEN\" not in os.environ or not os.environ[\"HF_TOKEN\"].strip():\n",
                "    raise RuntimeError(\n",
                "        \"FAIL-CLOSED: HF_TOKEN is required in Kaggle Secrets (or environment) for private remote mirroring. \"\n",
                "        \"Attach HF_TOKEN under Add-ons -> Secrets in the Kaggle notebook.\"\n",
                "    )\n",
                "print(\"[AUTH] Verified HF_TOKEN presence in environment for private remote mirroring.\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "code-checkout-commit",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Phase 3: Clone, Checkout Reviewed Training Commit & Verify Tree\n",
                "import os, subprocess, sys\n",
                "from pathlib import Path\n",
                "\n",
                "repo_dir = Path(\"/kaggle/working/repo\")\n",
                "if not repo_dir.exists():\n",
                "    print(\"[GIT] Cloning repository into /kaggle/working/repo...\")\n",
                "    subprocess.run(\n",
                "        [\"git\", \"clone\", \"https://github.com/minkhoaa/Langfens-RQ1-Speech-Research.git\", str(repo_dir)],\n",
                "        check=True\n",
                "    )\n",
                "\n",
                "os.chdir(str(repo_dir))\n",
                "subprocess.run([\"git\", \"fetch\", \"--all\"], cwd=repo_dir, check=True)\n",
                "\n",
                "# Pinned immutable training source commit (closing Step 04C launch-gate corrections)\n",
                f'PINNED_COMMIT_SHA = "{PINNED_COMMIT_SHA}"\n',
                "print(f\"[GIT] Checking out pinned training commit: {PINNED_COMMIT_SHA}...\")\n",
                "subprocess.run([\"git\", \"checkout\", \"--detach\", PINNED_COMMIT_SHA], cwd=repo_dir, check=True)\n",
                "\n",
                "actual_head = subprocess.check_output([\"git\", \"rev-parse\", \"HEAD\"], cwd=repo_dir, text=True).strip().lower()\n",
                "print(f\"[GIT] Verified Training Commit HEAD: {actual_head}\")\n",
                "assert actual_head == PINNED_COMMIT_SHA.lower(), (\n",
                "    f\"FAIL-CLOSED: Commit SHA mismatch: actual {actual_head} != expected {PINNED_COMMIT_SHA}\"\n",
                ")\n",
                "\n",
                "status = subprocess.check_output([\"git\", \"status\", \"--porcelain\"], cwd=repo_dir, text=True).strip()\n",
                "assert len(status) == 0, f\"Dirty tree: {status}\"\n",
                "print(\"[GIT] Repository verified clean at pinned training commit.\")\n",
                "\n",
                "sys.path.insert(0, str(repo_dir / \"src\"))\n",
                "sys.path.insert(0, str(repo_dir))\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "code-mandatory-preflight-gate",
            "metadata": {},
            "outputs": [],
            "source": [
                f"# Phase 4: Mandatory Preflight Check (Fail-Closed Gate)\n",
                "# Training entrypoint executes preflight checks first; fails closed immediately on any error.\n",
                "import subprocess, sys\n",
                "from pathlib import Path\n",
                "\n",
                f"staging_input = Path(\"/kaggle/input/{spec['dataset_slug'].split('/')[-1]}\")\n",
                "print(f\"[PREFLIGHT-GATE] Auditing dataset input directory: {staging_input}\")\n",
                "if staging_input.is_dir():\n",
                "    print(f\"[PREFLIGHT-GATE] Directory contents: {[p.name for p in staging_input.iterdir()]}\")\n",
                "else:\n",
                "    print(f\"[PREFLIGHT-GATE WARNING] {staging_input} does not exist yet (requires dataset attachment).\")\n",
                "\n",
                "preflight_cmd = [\n",
                "    sys.executable,\n",
                "    \"deploy/kaggle/kaggle_entrypoint.py\",\n",
                f"    \"--mode\", \"verify_fold{fold_idx}\",\n",
                "    \"--commit-sha\", PINNED_COMMIT_SHA,\n",
                "    \"--kaggle-input-dir\", str(staging_input),\n",
                "    \"--remote-mirror-backend\", \"huggingface\",\n",
                "    \"--remote-mirror-target\", \"minkhoaa/langfens-rq1-cross-fold-checkpoints\",\n",
                "]\n",
                "print(\"[PREFLIGHT-GATE] Executing mandatory preflight verification:\")\n",
                "print(\"  \" + \" \".join(preflight_cmd))\n",
                "subprocess.run(preflight_cmd, check=True)\n",
                "print(\"=\" * 80)\n",
                "print(\"[PREFLIGHT] Preflight verification PASSED successfully. Proceeding to official training...\")\n",
                "print(\"=\" * 80)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "code-training-execution",
            "metadata": {},
            "outputs": [],
            "source": [
                f"# Phase 5: Official Training Execution (train_fold{fold_idx})\n",
                f"# Preflight verified; launching official Fold {fold_idx} Baseline R training.\n",
                "print(\"=\" * 80)\n",
                f"print(\"[LAUNCH] Starting Official Fold {fold_idx} Baseline R Training on Tesla T4...\")\n",
                "print(f\"[LAUNCH] Pinned Commit SHA: {PINNED_COMMIT_SHA}\")\n",
                "print(\"=\" * 80)\n",
                "\n",
                "train_cmd = [\n",
                "    sys.executable,\n",
                "    \"deploy/kaggle/kaggle_entrypoint.py\",\n",
                f"    \"--mode\", \"train_fold{fold_idx}\",\n",
                "    \"--commit-sha\", PINNED_COMMIT_SHA,\n",
                "    \"--kaggle-input-dir\", str(staging_input),\n",
                f"    \"--run-id\", \"baseline_r_fold{fold_idx}\",\n",
                "    \"--remote-mirror-backend\", \"huggingface\",\n",
                "    \"--remote-mirror-target\", \"minkhoaa/langfens-rq1-cross-fold-checkpoints\",\n",
                "]\n",
                "print(\"[LAUNCH] Executing checked command:\")\n",
                "print(\"  \" + \" \".join(train_cmd))\n",
                "subprocess.run(train_cmd, check=True)\n",
                f"print(\"[LAUNCH] Fold {fold_idx} Baseline R training subprocess completed successfully.\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "code-endpoint-verification",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Phase 6: Post-Training Endpoint Classification\n",
                "# Enforces strict deliverable completion invariants:\n",
                "# 1. fixed_budget_checkpoint.pt with exactly 3,600 updates at Epoch 6.\n",
                "# 2. best.pt, last.pt, and history.json.\n",
                "# 3. Correct fold index and manifest SHA.\n",
                "# 4. Pinned configuration parity.\n",
                "from audit_scripts.kaggle_fold_train_exec import classify_fold_training_endpoint\n",
                "res = classify_fold_training_endpoint(\n",
                f"    \"/kaggle/working/checkpoints/fold{fold_idx}/baseline_r_fold{fold_idx}\",\n",
                f"    fold_idx={fold_idx},\n",
                "    expected_commit=PINNED_COMMIT_SHA\n",
                ")\n",
                "print(\"=\" * 80)\n",
                "print(\"[FINAL ENDPOINT STATUS]:\", res[\"status\"])\n",
                "print(\"[ENDPOINT DETAILS]:\", res)\n",
                "print(\"=\" * 80)\n",
                "assert res[\"status\"] == \"COMPLETED_3600_UPDATES\", f\"Training endpoint verification failed: {res}\"\n"
            ]
        }
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 5
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def make_kernel_metadata(kernel_id: str, code_file: str, title: str, dataset_slug: str) -> dict:
    return {
        "id": kernel_id,
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [
            dataset_slug
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }


def make_dataset_metadata(fold_idx: int = 2) -> dict:
    spec = FOLD_SPECS[fold_idx]
    return {
        "title": spec["dataset_title"],
        "id": spec["dataset_slug"],
        "licenses": [
            {
                "name": "CC-BY-NC-4.0"
            }
        ],
        "is_private": True,
        "description": (
            f"Authorized Fold {fold_idx} Train+Val audio archive (18 speakers) for LANGFENS RQ1-TRN-03A. "
            f"Contains NO active test partition audio or annotations ({spec['test_speakers']} strictly excluded). "
            f"Strictly for authorized private research under CC BY-NC 4.0."
        )
    }


def build_fold_packages(fold_idx: int):
    spec = FOLD_SPECS[fold_idx]
    prefix = f"kaggle_fold{fold_idx}"

    preflight_dir = REPO_ROOT / "deploy" / f"{prefix}_preflight"
    smoke_dir = REPO_ROOT / "deploy" / f"{prefix}_smoke"
    train_dir = REPO_ROOT / "deploy" / f"{prefix}_train"

    preflight_dir.mkdir(parents=True, exist_ok=True)
    smoke_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    dmeta = make_dataset_metadata(fold_idx)

    # 0. Staging package (dataset metadata for kaggle datasets create)
    if fold_idx >= 3:
        staging_dir = REPO_ROOT / "deploy" / f"{prefix}_staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        (staging_dir / "dataset-metadata.json").write_text(json.dumps(dmeta, indent=2) + "\n")
        print(f"[BUILD] Generated Staging Metadata at {staging_dir}")

    # 1. Preflight package
    (preflight_dir / "kernel-metadata.json").write_text(
        json.dumps(make_kernel_metadata(spec["preflight_slug"], "kaggle_preflight.ipynb", f"LANGFENS RQ1 Fold {fold_idx} Baseline R T4x2 (Preflight)", spec["dataset_slug"]), indent=2) + "\n"
    )
    (preflight_dir / "dataset-metadata.json").write_text(json.dumps(dmeta, indent=2) + "\n")
    (preflight_dir / "kaggle_preflight.ipynb").write_text(
        json.dumps(make_preflight_notebook(fold_idx), indent=2) + "\n"
    )
    print(f"[BUILD] Generated Fold {fold_idx} Preflight Package at {preflight_dir}")

    # 2. Smoke training package (1 epoch, run ID fold{k}_smoke_1epoch)
    (smoke_dir / "kernel-metadata.json").write_text(
        json.dumps(make_kernel_metadata(spec["smoke_slug"], "kaggle_smoke.ipynb", f"LANGFENS RQ1 Fold {fold_idx} Baseline R T4x2 (Smoke)", spec["dataset_slug"]), indent=2) + "\n"
    )
    (smoke_dir / "dataset-metadata.json").write_text(json.dumps(dmeta, indent=2) + "\n")
    (smoke_dir / "kaggle_smoke.ipynb").write_text(
        json.dumps(make_smoke_notebook(fold_idx), indent=2) + "\n"
    )
    print(f"[BUILD] Generated Fold {fold_idx} Smoke Package at {smoke_dir}")

    # 3. Official training package (10 epochs, Baseline R)
    (train_dir / "kernel-metadata.json").write_text(
        json.dumps(make_kernel_metadata(spec["train_slug"], "kaggle_train.ipynb", f"LANGFENS RQ1 Fold {fold_idx} Baseline R T4x2", spec["dataset_slug"]), indent=2) + "\n"
    )
    (train_dir / "dataset-metadata.json").write_text(json.dumps(dmeta, indent=2) + "\n")
    (train_dir / "kaggle_train.ipynb").write_text(
        json.dumps(make_train_notebook(fold_idx), indent=2) + "\n"
    )
    print(f"[BUILD] Generated Fold {fold_idx} Training Package at {train_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build Kaggle deployment packages for cross-fold baseline.")
    parser.add_argument("--folds", type=int, nargs="+", default=[2, 3], help="Folds to build (e.g. 2 3)")
    args = parser.parse_args()

    for f in args.folds:
        if f not in FOLD_SPECS:
            raise ValueError(f"Unsupported fold: {f}. Supported folds: {list(FOLD_SPECS.keys())}")
        build_fold_packages(f)


if __name__ == "__main__":
    main()
