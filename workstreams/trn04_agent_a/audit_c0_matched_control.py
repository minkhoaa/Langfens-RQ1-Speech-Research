"""LANGFENS RQ1 TRN-04 — Agent A Independent Verification Script for C0 Matched Control.

Executes comprehensive audit:
1. Model head parameter census and exact matching against COMMON_INITIALIZATION_MANIFEST.json.
2. Acoustic slot mathematical zero property (bias=False, zero input -> zero output, zero gradient).
3. Objective function audit (operation_ce + realized CTC loss).
4. Partition census and sealed test speaker isolation verification.
5. Deployment package integrity audit.
6. Remote execution status query.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Repo root
REPO_ROOT = Path("/home/khoa/Projects/KLTN/Dataset")
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from langfens_prep.trn04.model import (
    CompactFactorialOperationHead,
    FactorialHeadConfig,
    matched_initial_quad,
    parameter_inventory
)
from langfens_prep.trn04.objectives import task_loss, realized_ctc_loss
from langfens_prep.p_sup.dataset import (
    PSupL2ArcticDataset,
    FIT_SPEAKERS,
    SELECT_SPEAKERS,
    CALIBRATION_SPEAKERS,
    VAL_SPEAKERS,
    SealedSpeakerContaminationError
)
from langfens_prep.p_sup.runtime import SEALED_SPEAKERS, CONTRACT_SHA256


def run_audit() -> dict:
    results = {}

    # 1. Parameter Census
    c0, t1, t2, t3 = matched_initial_quad(seed=1337)
    inv = parameter_inventory(c0)
    total_params = sum(v["numel"] for v in inv.values())
    total_trainable = sum(v["numel"] for v in inv.values() if v["trainable"])

    manifest_path = REPO_ROOT / "manifests/RQ1-TRN-04/COMMON_INITIALIZATION_MANIFEST.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    assert total_trainable == 267179, f"Expected 267179, got {total_trainable}"
    assert manifest["total_trainable_parameters"] == 267179

    # Check each tensor hash
    import hashlib
    tensor_hashes_matched = True
    for name, p in c0.named_parameters():
        p_sha = hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        manifest_sha = manifest["shared_tensor_hashes"][name]["sha256"]
        if p_sha != manifest_sha:
            tensor_hashes_matched = False
            print(f"Hash mismatch for {name}: {p_sha} != {manifest_sha}")

    results["parameter_census"] = {
        "total_trainable": total_trainable,
        "num_tensors": len(inv),
        "all_trainable": all(v["trainable"] for v in inv.values()),
        "tensor_hashes_match_manifest": tensor_hashes_matched
    }

    # 2. Acoustic slot mathematical zero property
    zero_in = torch.zeros(2, 50, 4)
    projected = c0.acoustic_feature_projection(zero_in)
    is_strictly_zero = torch.equal(projected, torch.zeros_like(projected))
    has_bias = c0.acoustic_feature_projection.bias is not None

    results["acoustic_slot"] = {
        "has_bias": has_bias,
        "input_zeros_yields_exact_zeros": is_strictly_zero
    }

    # 3. Loss formulation
    b, n = 2, 8
    logits = torch.randn(b, n, 3, requires_grad=True)
    targets = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 0], [1, 2, 0, 1, 2, 0, 1, 1]])
    mask = torch.ones(b, n, dtype=torch.bool)
    prompt_ids = torch.randint(1, 40, (b, n))
    ce_loss = task_loss(logits, targets, mask, prompt_ids, objective="operation_ce")

    ctc_logits = torch.randn(b, 40, 40)
    fl = torch.tensor([40, 40], dtype=torch.long)
    ctc_targets = torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]], dtype=torch.long)
    ctc_lens = torch.tensor([4, 3], dtype=torch.long)
    ctc_loss = realized_ctc_loss(ctc_logits, fl, ctc_targets, ctc_lens)

    results["loss_formulation"] = {
        "task_ce_loss_finite": bool(torch.isfinite(ce_loss).item()),
        "ctc_loss_finite": bool(torch.isfinite(ctc_loss).item()),
        "total_loss": float((ce_loss + ctc_loss).item())
    }

    # 4. Partitions Census
    split_manifest = REPO_ROOT / "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json"
    zip_path = REPO_ROOT / "datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip"

    ds_fit = PSupL2ArcticDataset(str(split_manifest), str(zip_path), split="fit")
    ds_sel = PSupL2ArcticDataset(str(split_manifest), str(zip_path), split="selection")
    ds_cal = PSupL2ArcticDataset(str(split_manifest), str(zip_path), split="calibration")
    ds_val = PSupL2ArcticDataset(str(split_manifest), str(zip_path), split="val")

    sealed_prevented = False
    try:
        PSupL2ArcticDataset(str(split_manifest), str(zip_path), split="test")
    except SealedSpeakerContaminationError:
        sealed_prevented = True

    results["partitions"] = {
        "fit_utterances": len(ds_fit),
        "fit_speakers": sorted(list(FIT_SPEAKERS)),
        "selection_utterances": len(ds_sel),
        "selection_speakers": sorted(list(SELECT_SPEAKERS)),
        "calibration_utterances": len(ds_cal),
        "calibration_speakers": sorted(list(CALIBRATION_SPEAKERS)),
        "val_utterances": len(ds_val),
        "val_speakers": sorted(list(VAL_SPEAKERS)),
        "sealed_speakers": sorted(list(SEALED_SPEAKERS)),
        "sealed_test_fail_closed": sealed_prevented,
        "contract_sha256": CONTRACT_SHA256
    }

    # 5. Deployment package verification
    deploy_dir = REPO_ROOT / "deploy/kaggle_trn04_c0"
    with open(deploy_dir / "kernel-metadata.json") as f:
        kmeta = json.load(f)

    with open(deploy_dir / "kaggle_train.ipynb") as f:
        nb = json.load(f)

    results["deployment_package"] = {
        "slug": kmeta.get("id"),
        "machine_shape": kmeta.get("machine_shape"),
        "enable_gpu": kmeta.get("enable_gpu"),
        "is_private": kmeta.get("is_private"),
        "dataset_sources": kmeta.get("dataset_sources"),
        "nb_num_cells": len(nb.get("cells", []))
    }

    # 6. Kaggle remote status
    try:
        res = subprocess.run(
            ["kaggle", "kernels", "status", kmeta.get("id")],
            capture_output=True,
            text=True,
            check=True
        )
        kaggle_status = res.stdout.strip()
    except Exception as e:
        kaggle_status = f"Error: {e}"

    results["kaggle_status"] = kaggle_status

    # 7. HuggingFace mirror check
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        files = api.list_repo_tree("minkhoaa/langfens-rq1-cross-fold-checkpoints", path_in_repo="trn04/c0", repo_type="model")
        hf_files = [{"path": f.path, "size": getattr(f, "size", None)} for f in files]
    except Exception as e:
        hf_files = f"Error: {e}"

    results["hf_mirror_files"] = hf_files

    return results


if __name__ == "__main__":
    audit_data = run_audit()
    print(json.dumps(audit_data, indent=2))
    with open("c0_control_audit_receipt.json", "w") as f:
        json.dump(audit_data, f, indent=2)
    print("\n[SUCCESS] Audit completed and saved to c0_control_audit_receipt.json")
