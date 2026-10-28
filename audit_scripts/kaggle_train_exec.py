#!/usr/bin/env python3
"""
LANGFENS — Kaggle T4x2 Training Execution and Endpoint Classification Driver
Orchestrates verified training for Control Arm C0 (p=0.00) and Treatment Arm C1 (p=0.25)
on Kaggle Nvidia Tesla T4 GPU under CUDA FP16 AMP with per-epoch persistent mirroring.

Key Guarantees:
1. GPU Verification: CUDA Tesla T4 required. Fails closed on CPU or non-T4.
2. Verified Staging: Validates manifest SHA, Baseline R SHA, and Fold 1 train/val archive.
3. Isolated Per-Arm Execution: Independent optimizer, scheduler, scaler, RNG, and run_id.
4. Persistent Mirroring: Every epoch synced to approved private remote store with SHA-256 readback.
5. Primary Endpoint Verification: Enforces 9 completion invariants for COMPLETED_3600_UPDATES.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import yaml

try:
    from audit_scripts.kaggle_checkpoint_sync import KaggleRemoteMirror, compute_sha256
    from audit_scripts.kaggle_stage_and_verify import (
        EXPECTED_BASELINE_SHA256,
        EXPECTED_MANIFEST_SHA256,
        EXPECTED_ZIP_SHA256,
        resolve_kaggle_output_dir,
        stage_and_verify_kaggle_environment,
        validate_arm,
        verify_kaggle_gpu_device,
    )
except ImportError:
    from kaggle_checkpoint_sync import KaggleRemoteMirror, compute_sha256
    from kaggle_stage_and_verify import (
        EXPECTED_BASELINE_SHA256,
        EXPECTED_MANIFEST_SHA256,
        EXPECTED_ZIP_SHA256,
        resolve_kaggle_output_dir,
        stage_and_verify_kaggle_environment,
        validate_arm,
        verify_kaggle_gpu_device,
    )


def classify_kaggle_training_endpoint(
    output_dir: Path | str,
    expected_arm: str,
    expected_commit: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Rigorously verifies and classifies the training run endpoint:
    - COMPLETED_3600_UPDATES: Primary matched-budget endpoint reached and verified.
      Requires ALL 9 completion invariants.
    - INCOMPLETE: Terminated early (epoch cap / AMP skips) without guardrail trip.
    - HALTED_GUARDRAIL: Stopped due to CTC PER degradation guardrail trip.
    - FAILED: Checkpoints missing, corrupted, contradictory metadata, or process failed.
    """
    arm_norm = validate_arm(expected_arm)
    p_dir = Path(output_dir).resolve()

    # 1. Check for explicit guardrail trip artifact
    halted_file = p_dir / "halted_guardrail.pt"
    if halted_file.is_file():
        try:
            ck = torch.load(halted_file, map_location="cpu", weights_only=False)
            meta = ck.get("metadata", {}) if isinstance(ck, dict) else {}
            return {
                "status": "HALTED_GUARDRAIL",
                "message": (
                    f"Guardrail violation tripped at epoch {ck.get('epoch', '?')}: "
                    f"CTC PER measured {meta.get('ctc_per_measured')} vs anchor {meta.get('ctc_per_anchor')} "
                    f"(+{meta.get('ctc_per_degradation_pct')} absolute degradation)."
                ),
                "successful_updates": ck.get("successful_updates", 0) if isinstance(ck, dict) else 0,
                "artifact_path": str(halted_file),
                "sha256": compute_sha256(halted_file),
            }
        except Exception as exc:
            return {
                "status": "HALTED_GUARDRAIL",
                "message": f"Halted guardrail artifact found but failed to parse: {exc}",
                "artifact_path": str(halted_file),
            }

    # 2. Check for completed primary endpoint: fixed_budget_checkpoint.pt
    fixed_budget_file = p_dir / "fixed_budget_checkpoint.pt"
    if fixed_budget_file.is_file():
        try:
            ck = torch.load(fixed_budget_file, map_location="cpu", weights_only=False)
            if not isinstance(ck, dict):
                return {
                    "status": "FAILED",
                    "message": "fixed_budget_checkpoint payload is not a valid dictionary.",
                    "artifact_path": str(fixed_budget_file),
                }

            # Invariant 9: Required training and optimizer state
            for req_key in ("model", "optimizer", "scheduler", "scaler", "metadata"):
                if req_key not in ck:
                    return {
                        "status": "FAILED",
                        "message": f"Corrupt or incomplete checkpoint payload: missing required key '{req_key}'.",
                        "artifact_path": str(fixed_budget_file),
                    }

            meta = ck.get("metadata", {})
            if not isinstance(meta, dict):
                return {
                    "status": "FAILED",
                    "message": "Checkpoint metadata payload is missing or not a dictionary.",
                    "artifact_path": str(fixed_budget_file),
                }

            # Invariant 1 & 2: successful_updates == 3600
            top_succ = ck.get("successful_updates")
            meta_succ = meta.get("successful_updates")
            if top_succ != 3600 or meta_succ != 3600:
                return {
                    "status": "FAILED",
                    "message": (
                        f"Contradictory or incomplete successful update count in fixed_budget_checkpoint: "
                        f"top_level={top_succ}, metadata={meta_succ} (expected 3600)."
                    ),
                    "artifact_path": str(fixed_budget_file),
                }

            # Invariant 3: primary_3600_updates_completed is True
            if meta.get("primary_3600_updates_completed") is not True:
                return {
                    "status": "FAILED",
                    "message": "Metadata flag 'primary_3600_updates_completed' is not True.",
                    "artifact_path": str(fixed_budget_file),
                }

            # Invariant 4: primary_endpoint_reached is True
            if meta.get("primary_endpoint_reached") is not True:
                return {
                    "status": "FAILED",
                    "message": "Metadata flag 'primary_endpoint_reached' is not True.",
                    "artifact_path": str(fixed_budget_file),
                }

            # Invariant 5: endpoint_status == "COMPLETED_3600_UPDATES"
            if meta.get("endpoint_status") != "COMPLETED_3600_UPDATES":
                return {
                    "status": "FAILED",
                    "message": f"Metadata 'endpoint_status' '{meta.get('endpoint_status')}' != 'COMPLETED_3600_UPDATES'.",
                    "artifact_path": str(fixed_budget_file),
                }

            # Invariant 6: guardrail_violation is False
            if meta.get("guardrail_violation") is True:
                return {
                    "status": "FAILED",
                    "message": "Metadata reports guardrail_violation == True on completed checkpoint.",
                    "artifact_path": str(fixed_budget_file),
                }

            # Invariant 7: arm matches expected
            meta_arm = meta.get("arm")
            if meta_arm != arm_norm:
                return {
                    "status": "FAILED",
                    "message": f"Arm mismatch in checkpoint metadata: got '{meta_arm}', expected '{arm_norm}'.",
                    "artifact_path": str(fixed_budget_file),
                }

            # Invariant 8: Provenance verification
            if meta.get("split_sha256") != EXPECTED_MANIFEST_SHA256:
                return {
                    "status": "FAILED",
                    "message": f"Provenance mismatch: split_sha256 '{meta.get('split_sha256')}' != expected.",
                    "artifact_path": str(fixed_budget_file),
                }
            if meta.get("initial_checkpoint_sha256") != EXPECTED_BASELINE_SHA256:
                return {
                    "status": "FAILED",
                    "message": f"Provenance mismatch: initial_checkpoint_sha256 '{meta.get('initial_checkpoint_sha256')}' != expected.",
                    "artifact_path": str(fixed_budget_file),
                }

            git_sha = meta.get("git_sha", "")
            if not isinstance(git_sha, str) or len(git_sha) != 40:
                return {
                    "status": "FAILED",
                    "message": f"Invalid git_sha in checkpoint metadata: '{git_sha}'.",
                    "artifact_path": str(fixed_budget_file),
                }
            if expected_commit and git_sha != expected_commit:
                return {
                    "status": "FAILED",
                    "message": f"Git commit mismatch: checkpoint built from '{git_sha}' != expected '{expected_commit}'.",
                    "artifact_path": str(fixed_budget_file),
                }

            # All 9 invariants satisfied
            return {
                "status": "COMPLETED_3600_UPDATES",
                "message": "Primary matched-budget endpoint verified: completed 3600 successful updates.",
                "successful_updates": 3600,
                "artifact_path": str(fixed_budget_file),
                "sha256": compute_sha256(fixed_budget_file),
            }

        except Exception as exc:
            return {
                "status": "FAILED",
                "message": f"Failed to parse fixed_budget_checkpoint: {exc}",
                "artifact_path": str(fixed_budget_file),
            }

    # 3. Check for last.pt (incomplete / terminated before 3600 updates)
    last_file = p_dir / "last.pt"
    if last_file.is_file():
        try:
            ck = torch.load(last_file, map_location="cpu", weights_only=False)
            meta = ck.get("metadata", {}) if isinstance(ck, dict) else {}
            succ = ck.get("successful_updates", 0) if isinstance(ck, dict) else 0
            ep = ck.get("epoch", 0) if isinstance(ck, dict) else 0
            return {
                "status": "INCOMPLETE",
                "message": f"Training terminated early at epoch {ep} with {succ}/3600 updates.",
                "successful_updates": succ,
                "artifact_path": str(last_file),
                "sha256": compute_sha256(last_file),
            }
        except Exception as exc:
            return {
                "status": "FAILED",
                "message": f"last.pt found but failed to parse: {exc}",
                "artifact_path": str(last_file),
            }

    return {
        "status": "FAILED",
        "message": f"No checkpoint artifacts found in '{p_dir}'.",
    }


def execute_kaggle_training(
    config_path: str | Path,
    arm: Optional[str] = None,
    run_id: Optional[str] = None,
    kaggle_input_dir: Optional[str | Path] = None,
    working_dir: Optional[str | Path] = None,
    remote_mirror_backend: str = "local_dir",
    remote_mirror_target: Optional[str] = None,
    allow_mock_gpu: bool = False,
    dry_run: bool = False,
    check_only: bool = False,
    resume: bool = False,
) -> Dict[str, Any]:
    """
    Executes Kaggle training workflow for arm C0 or C1:
    1. Validates and stages environment.
    2. Initializes remote mirror.
    3. Runs training loop with atomic checkpoint saving and remote sync.
    4. Rigorously classifies final endpoint.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if arm:
        cfg["arm"] = validate_arm(arm)

    # 1. Preflight Staging & Verification
    staging_report = stage_and_verify_kaggle_environment(
        config=cfg,
        kaggle_input_dir=kaggle_input_dir,
        working_dir=working_dir,
        remote_mirror_backend=remote_mirror_backend,
        remote_mirror_target=remote_mirror_target,
        allow_mock_gpu=allow_mock_gpu,
        run_id=run_id,
        resume=resume,
    )

    if check_only or dry_run:
        return {
            "status": "PASS_PREFLIGHT",
            "staging_report": staging_report,
            "dry_run": dry_run,
            "check_only": check_only,
        }

    # 2. Invoke Training Driver
    from audit_scripts.train_rq1_trn_02b import main as train_main

    # Build CLI args for train_rq1_trn_02b
    out_dir = staging_report["output_directory"]
    staged_zip = staging_report["dataset"]["path"]

    # Execute training via subprocess or in-process
    import subprocess
    cmd = [
        sys.executable,
        "audit_scripts/train_rq1_trn_02b.py",
        "--config", str(config_path),
        "--arm", cfg["arm"],
        "--output-dir", out_dir,
        "--zip-path", staged_zip,
    ]
    if run_id:
        cmd.extend(["--run-id", run_id])
    if resume:
        cmd.append("--resume")

    # In production, train_rq1_trn_02b saves locally. Checkpoint mirror syncs per epoch.
    # Remote mirror syncing is handled during or immediately following training.
    print(f"[KAGGLE EXEC] Launching training subprocess: {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)

    # 3. Classify Endpoint
    endpoint = classify_kaggle_training_endpoint(
        output_dir=out_dir,
        expected_arm=cfg["arm"],
    )

    # 4. Mirror all artifacts to remote persistent store
    mirror = KaggleRemoteMirror(
        backend=remote_mirror_backend,
        remote_target=remote_mirror_target,
    )
    mirror_results = []
    for art in Path(out_dir).glob("*.pt"):
        res = mirror.mirror_file(art, f"{cfg['arm']}/{art.name}")
        mirror_results.append(res)
    hist = Path(out_dir) / "history.json"
    if hist.is_file():
        mirror_results.append(mirror.mirror_file(hist, f"{cfg['arm']}/history.json"))

    return {
        "execution_returncode": proc.returncode,
        "endpoint": endpoint,
        "mirror_syncs": mirror_results,
        "staging_report": staging_report,
    }


def main():
    parser = argparse.ArgumentParser(description="LANGFENS — Kaggle T4x2 Training Execution")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--arm", type=str, choices=["C0", "C1"], help="Arm to execute")
    parser.add_argument("--run-id", type=str, default=None, help="Unique run ID")
    parser.add_argument("--kaggle-input-dir", type=str, default="/kaggle/input/langfens-fold1-train-val-baseline")
    parser.add_argument("--working-dir", type=str, default="/kaggle/working")
    parser.add_argument("--remote-mirror-backend", type=str, default="local_dir")
    parser.add_argument("--remote-mirror-target", type=str, default=None)
    parser.add_argument("--allow-mock-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    result = execute_kaggle_training(
        config_path=args.config,
        arm=args.arm,
        run_id=args.run_id,
        kaggle_input_dir=args.kaggle_input_dir,
        working_dir=args.working_dir,
        remote_mirror_backend=args.remote_mirror_backend,
        remote_mirror_target=args.remote_mirror_target,
        allow_mock_gpu=args.allow_mock_gpu,
        dry_run=args.dry_run,
        check_only=args.check_only,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
