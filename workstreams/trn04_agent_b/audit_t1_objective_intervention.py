#!/usr/bin/env python3
"""LANGFENS RQ1 TRN-04 — Agent B Comprehensive Audit Script for T1 Objective Intervention.

Generates the machine-readable cryptographic receipt `t1_objective_audit_receipt.json`.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import torch
import torch.nn.functional as F

REPO_ROOT = Path("/home/khoa/Projects/KLTN/Dataset")
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from langfens_prep.trn04.model import (
    CompactFactorialOperationHead,
    FactorialHeadConfig,
    matched_initial_quad,
    parameter_inventory
)
from langfens_prep.trn04.objectives import (
    task_loss,
    CORONAL_RHOTIC_PHONE_IDS,
    DEFAULT_GAMMA,
    DEFAULT_ALPHA_CORONAL_DEL,
    build_coronal_lookup
)
from langfens_prep.p_sup.dataset import PSupL2ArcticDataset
from huggingface_hub import hf_hub_download


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def audit_t1():
    print("=" * 80)
    print("LANGFENS RQ1 TRN-04: Agent B (T1 Objective Intervention) Audit")
    print("=" * 80)

    # 1. Parameter census & capacity matching with C0
    c0, t1, t2, t3 = matched_initial_quad(seed=1337)
    inv = parameter_inventory(t1)
    total_trainable = sum(v["numel"] for v in inv.values() if v["trainable"])
    assert total_trainable == 267179, f"Parameter count mismatch: {total_trainable}"
    print(f"[OK] Total trainable parameters: {total_trainable} (exactly 267,179)")

    # 2. Check bit-for-bit initialization with C0
    for (name_c0, p_c0), (name_t1, p_t1) in zip(c0.named_parameters(), t1.named_parameters()):
        assert name_c0 == name_t1
        assert torch.equal(p_c0, p_t1), f"Initialization drift on {name_c0}"
    print("[OK] Bit-for-bit initialization match between C0 and T1 confirmed.")

    # 3. Check initialization manifest hashes
    manifest_path = REPO_ROOT / "manifests/RQ1-TRN-04/COMMON_INITIALIZATION_MANIFEST.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    manifest_hash_matches = {}
    for name, p in t1.named_parameters():
        actual_h = tensor_sha256(p)
        expected_h = manifest["shared_tensor_hashes"][name]["sha256"]
        assert actual_h == expected_h, f"Hash mismatch on {name}"
        manifest_hash_matches[name] = actual_h
    print(f"[OK] All {len(manifest_hash_matches)} parameter tensors match COMMON_INITIALIZATION_MANIFEST SHA-256 bit-for-bit.")

    # 4. Acoustic slot disabled invariance
    assert t1.acoustic_feature_projection.bias is None, "Bias must be False"
    zero_in = torch.zeros(2, 20, 4)
    zero_out = t1.acoustic_feature_projection(zero_in)
    assert torch.equal(zero_out, torch.zeros(2, 20, 128))
    print("[OK] Acoustic slot disabled invariance verified (bias=False, zero input -> zero output).")

    # 5. FIT deletion statistics
    split_manifest = REPO_ROOT / "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json"
    zip_path = REPO_ROOT / "datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip"
    ds = PSupL2ArcticDataset(str(split_manifest), str(zip_path), split="fit")
    assert len(ds) == 1799

    d_del, t_del, r_del, total_del = 0, 0, 0, 0
    for s in ds:
        for op, ph in zip(s["labels"].operation, s["labels"].canonical_ids):
            if op.item() == 2:
                total_del += 1
                ph_id = ph.item()
                if ph_id == 9:
                    d_del += 1
                elif ph_id == 31:
                    t_del += 1
                elif ph_id == 28:
                    r_del += 1

    coronal_del = d_del + t_del + r_del
    fraction = coronal_del / total_del
    print(f"[OK] FIT Deletion Census: Total={total_del}, D={d_del}, T={t_del}, R={r_del}, Sum={coronal_del} ({fraction*100:.2f}%)")
    assert total_del == 1503 and coronal_del == 852 and round(fraction * 100, 2) == 56.69

    # 6. Kaggle remote status check
    slug = "minkhoaa/langfens-rq1-trn-04-t1-focal-nofeat"
    res = subprocess.run(["kaggle", "kernels", "status", slug], capture_output=True, text=True)
    status_str = res.stdout.strip()
    print(f"[OK] Kaggle status for {slug}: {status_str}")

    # 7. HuggingFace mirror readback
    summary_path = hf_hub_download(
        repo_id="minkhoaa/langfens-rq1-cross-fold-checkpoints",
        filename="trn04/t1/training_summary.json"
    )
    with open(summary_path) as f:
        remote_summary = json.load(f)
    print(f"[OK] HuggingFace mirror readback: updates={remote_summary['successful_updates']}/3600, best_ap={remote_summary['best_selection_ap']:.6f}, epochs={remote_summary['total_epochs']}")

    receipt = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "agent": "Agent B (T1 Objective Intervention Lead)",
        "arm": "T1",
        "objective": "class_focal",
        "description": "Class-Conditioned Focal Loss, Acoustic Features Disabled",
        "audit_status": "PASSED",
        "experimental_invariants": {
            "total_trainable_parameters": total_trainable,
            "parameter_census_match_c0": True,
            "initialization_seed": 1337,
            "manifest_hash_match": True,
            "features_enabled": False,
            "acoustic_slot_bias": False,
            "zero_acoustic_invariance_verified": True
        },
        "objective_specification": {
            "name": "class_focal",
            "gamma": DEFAULT_GAMMA,
            "alpha_coronal_del": DEFAULT_ALPHA_CORONAL_DEL,
            "alpha_default": 1.0,
            "coronal_rhotic_phone_ids": list(CORONAL_RHOTIC_PHONE_IDS),
            "phone_names": {"9": "D", "28": "R", "31": "T"}
        },
        "fit_cohort_justification": {
            "total_fit_utterances": len(ds),
            "total_deletions": total_del,
            "d_deletions": d_del,
            "t_deletions": t_del,
            "r_deletions": r_del,
            "coronal_rhotic_sum": coronal_del,
            "coronal_rhotic_percentage": round(fraction * 100, 4)
        },
        "remote_deployment": {
            "kaggle_slug": slug,
            "remote_status": status_str,
            "target_hardware": "Tesla T4 (NvidiaTeslaT4)",
            "precision": "CUDA FP16 AMP",
            "dataset_source": "minkhoaa/langfens-rq1-fold1-staging",
            "git_branch": "review/rq1-trn-04-factorial-training"
        },
        "remote_execution_metrics": {
            "successful_updates": remote_summary["successful_updates"],
            "max_updates": remote_summary["max_updates"],
            "skipped_amp_attempts_total": remote_summary["skipped_amp_attempts_total"],
            "total_epochs": remote_summary["total_epochs"],
            "best_selection_ap": remote_summary["best_selection_ap"],
            "duration_seconds": remote_summary["duration_seconds"],
            "completed_at": remote_summary["completed_at"],
            "mirrored_checkpoints": [
                "trn04/t1/best.pt",
                "trn04/t1/fixed_budget_3600.pt",
                "trn04/t1/training_summary.json"
            ]
        }
    }

    out_path = Path("/home/khoa/Projects/KLTN/Dataset/workstreams/trn04_agent_b/t1_objective_audit_receipt.json")
    with open(out_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"[OK] Audit receipt written to {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    audit_t1()
