#!/usr/bin/env python3
"""
LANGFENS — RQ1-TRN-03A: Generator for Simplified Kaggle CLI Deployment Packages.

Creates dedicated, independent preflight and training deployment packages:
  - deploy/kaggle_fold2_preflight/
  - deploy/kaggle_fold2_train/

Replaces notebook execution flags with separate entrypoints independently
runnable via `kaggle kernels push`.
"""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PINNED_COMMIT_SHA = "407cf04e83704d9e83953e9309e3c7dca8d7154e"


def make_preflight_notebook() -> dict:
    cells = [
        {
            "cell_type": "markdown",
            "id": "markdown-overview",
            "metadata": {},
            "source": [
                "# LANGFENS — RQ1-TRN-03A: Dedicated Fold 2 Preflight Verification on Kaggle Tesla T4x2\n",
                "Official preflight verification notebook for Fold 2 Baseline R (`KAGGLE_T4_CUDA_FP16_AMP`).\n",
                "\n",
                "## Operational Role & Safety Invariants\n",
                "- **Role:** Independent, standalone preflight staging and persistence verification.\n",
                "- **Preflight Only:** Zero training steps, zero optimizer updates, zero model weights generated.\n",
                "- **Hardware Target:** Tesla T4 × 2 GPU (`Driver: >=580`, `CUDA: >=13.0`, 15,360 MiB VRAM per GPU).\n",
                "- **Dataset Staging:** Authorized Fold 2 Train+Val package (`minkhoaa/langfens-rq1-fold2-staging`).\n",
                "- **Sealed Test Isolation:** Active test speakers (`BWC`, `HKK`, `HQTV`, `MBMPS`, `SKA`, `SVBI`) are 100% physically absent.\n",
                "- **Manifest Integrity:** Verifies split manifest SHA-256 (`7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`).\n",
                "- **Dataset Integrity:** Verifies fold package SHA-256 (`112cc9e0cc5e37e5fa034df504eca019dd6ad438c96cb642335ecf10a802f0d9`).\n",
                "- **Remote Persistence Probe:** Validates live authenticated write-readback-delete probe to private Hugging Face model repository `minkhoaa/langfens-rq1-cross-fold-checkpoints`.\n",
                "- **Credential Hygiene:** Resolves `HF_TOKEN` strictly via Kaggle Secrets (`UserSecretsClient`). Zero token files on disk, zero CLI token arguments.\n",
                "- **Reviewed Training Base Commit:** `407cf04e83704d9e83953e9309e3c7dca8d7154e`.\n"
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
                "# Phase 4: Staging Audit & Preflight Verification (verify_fold2)\n",
                "# Executes preflight staging check, validates split manifest SHA, verifies 0 test speaker\n",
                "# leakage, checks auto-extracted outer zip reconstruction, and executes live Hugging Face\n",
                "# write-readback-delete probe. Zero model weights or training steps are executed in this entrypoint.\n",
                "import subprocess, sys\n",
                "from pathlib import Path\n",
                "\n",
                "staging_input = Path(\"/kaggle/input/langfens-rq1-fold2-staging\")\n",
                "print(f\"[INPUT AUDIT] Checking dataset input directory: {staging_input}\")\n",
                "if staging_input.is_dir():\n",
                "    print(f\"[INPUT AUDIT] Directory contents: {[p.name for p in staging_input.iterdir()]}\")\n",
                "else:\n",
                "    print(f\"[INPUT AUDIT WARNING] {staging_input} does not exist yet (requires dataset attachment).\")\n",
                "\n",
                "preflight_cmd = [\n",
                "    sys.executable,\n",
                "    \"deploy/kaggle/kaggle_entrypoint.py\",\n",
                "    \"--mode\", \"verify_fold2\",\n",
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


def make_train_notebook() -> dict:
    cells = [
        {
            "cell_type": "markdown",
            "id": "markdown-overview",
            "metadata": {},
            "source": [
                "# LANGFENS — RQ1-TRN-03A: Dedicated Fold 2 Baseline R Training on Kaggle Tesla T4x2\n",
                "Official execution notebook for Fold 2 Baseline R training (`KAGGLE_T4_CUDA_FP16_AMP`).\n",
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
                "  - Dataset: Authorized Fold 2 Train+Val package (`minkhoaa/langfens-rq1-fold2-staging`).\n",
                "  - Sealed Test Exclusion: Active test speakers (`BWC`, `HKK`, `HQTV`, `MBMPS`, `SKA`, `SVBI`) are 100% physically absent.\n",
                "- **Credential & Durability Hygiene:**\n",
                "  - Authentication: strictly via Kaggle Secrets (`HF_TOKEN`) or environment variables; zero token file fallbacks.\n",
                "  - Remote Mirroring: Per-epoch atomic mirror to private Hugging Face repo `minkhoaa/langfens-rq1-cross-fold-checkpoints` with SHA-256 readback.\n",
                "- **Mandatory Preflight Gate:** Runs preflight first and fails closed on any error before training begins.\n",
                "- **Reviewed Training Source Commit:** `407cf04e83704d9e83953e9309e3c7dca8d7154e` (pinned training codebase).\n"
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
                "# Phase 4: Mandatory Preflight Check (Fail-Closed Gate)\n",
                "# Training entrypoint executes preflight checks first; fails closed immediately on any error.\n",
                "import subprocess, sys\n",
                "from pathlib import Path\n",
                "\n",
                "staging_input = Path(\"/kaggle/input/langfens-rq1-fold2-staging\")\n",
                "print(f\"[PREFLIGHT-GATE] Auditing dataset input directory: {staging_input}\")\n",
                "if staging_input.is_dir():\n",
                "    print(f\"[PREFLIGHT-GATE] Directory contents: {[p.name for p in staging_input.iterdir()]}\")\n",
                "else:\n",
                "    print(f\"[PREFLIGHT-GATE WARNING] {staging_input} does not exist yet (requires dataset attachment).\")\n",
                "\n",
                "preflight_cmd = [\n",
                "    sys.executable,\n",
                "    \"deploy/kaggle/kaggle_entrypoint.py\",\n",
                "    \"--mode\", \"verify_fold2\",\n",
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
                "# Phase 5: Official Training Execution (train_fold2)\n",
                "# Preflight verified; launching official Fold 2 Baseline R training.\n",
                "print(\"=\" * 80)\n",
                "print(\"[LAUNCH] Starting Official Fold 2 Baseline R Training on Tesla T4...\")\n",
                "print(f\"[LAUNCH] Pinned Commit SHA: {PINNED_COMMIT_SHA}\")\n",
                "print(\"=\" * 80)\n",
                "\n",
                "train_cmd = [\n",
                "    sys.executable,\n",
                "    \"deploy/kaggle/kaggle_entrypoint.py\",\n",
                "    \"--mode\", \"train_fold2\",\n",
                "    \"--commit-sha\", PINNED_COMMIT_SHA,\n",
                "    \"--kaggle-input-dir\", str(staging_input),\n",
                "    \"--remote-mirror-backend\", \"huggingface\",\n",
                "    \"--remote-mirror-target\", \"minkhoaa/langfens-rq1-cross-fold-checkpoints\",\n",
                "]\n",
                "print(\"[LAUNCH] Executing checked command:\")\n",
                "print(\"  \" + \" \".join(train_cmd))\n",
                "subprocess.run(train_cmd, check=True)\n",
                "print(\"[LAUNCH] Fold 2 Baseline R training subprocess completed successfully.\")\n"
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
                "    \"/kaggle/working/checkpoints/fold2/baseline_r_fold2\",\n",
                "    fold_idx=2,\n",
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


def make_kernel_metadata(kernel_id: str, code_file: str, title: str) -> dict:
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
            "minkhoaa/langfens-rq1-fold2-staging"
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }


def make_dataset_metadata() -> dict:
    return {
        "title": "LANGFENS RQ1 Fold 2 Train/Val Staging Package",
        "id": "minkhoaa/langfens-rq1-fold2-staging",
        "licenses": [
            {
                "name": "CC-BY-NC-4.0"
            }
        ],
        "is_private": True,
        "description": (
            "Authorized Fold 2 Train+Val audio archive (18 speakers) for LANGFENS RQ1-TRN-03A. "
            "Contains NO active test partition audio or annotations (BWC, HKK, HQTV, MBMPS, SKA, SVBI strictly excluded). "
            "Strictly for authorized private research under CC BY-NC 4.0."
        )
    }


def main():
    preflight_dir = REPO_ROOT / "deploy" / "kaggle_fold2_preflight"
    train_dir = REPO_ROOT / "deploy" / "kaggle_fold2_train"

    preflight_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    # 1. Preflight package
    (preflight_dir / "kernel-metadata.json").write_text(
        json.dumps(make_kernel_metadata("minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2-preflight", "kaggle_preflight.ipynb", "LANGFENS RQ1 Fold 2 Baseline R T4x2 (Preflight)"), indent=2) + "\n"
    )
    (preflight_dir / "dataset-metadata.json").write_text(
        json.dumps(make_dataset_metadata(), indent=2) + "\n"
    )
    (preflight_dir / "kaggle_preflight.ipynb").write_text(
        json.dumps(make_preflight_notebook(), indent=2) + "\n"
    )
    print(f"[BUILD] Generated Preflight Package at {preflight_dir}")

    # 2. Training package
    (train_dir / "kernel-metadata.json").write_text(
        json.dumps(make_kernel_metadata("minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2", "kaggle_train.ipynb", "LANGFENS RQ1 Fold 2 Baseline R T4x2"), indent=2) + "\n"
    )
    (train_dir / "dataset-metadata.json").write_text(
        json.dumps(make_dataset_metadata(), indent=2) + "\n"
    )
    (train_dir / "kaggle_train.ipynb").write_text(
        json.dumps(make_train_notebook(), indent=2) + "\n"
    )
    print(f"[BUILD] Generated Training Package at {train_dir}")


if __name__ == "__main__":
    main()
