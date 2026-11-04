#!/usr/bin/env python3
"""LANGFENS — RQ1 TRN-04 Official 2x2 Factorial Training Entrypoint.

Supports all four matched factorial arms:
- C0: Cross-Entropy loss, acoustic features disabled (zeros)
- T1: Class-Conditioned Focal loss, acoustic features disabled (zeros)
- T2: Cross-Entropy loss, acoustic features enabled (waveform features)
- T3: Class-Conditioned Focal loss, acoustic features enabled (waveform features)

Enforces:
- Training on FIT only (1,799 utterances across 12 speakers)
- Checkpoint selection on SELECT only (300 utterances across 2 speakers: MBMPS, TLV)
- Calibration on CAL only (300 utterances across 2 speakers: NCC, THV)
- Preserves VAL strictly unseen (300 utterances across 2 speakers: NJS, RRBI)
- Sealed TEST partition strictly prohibited (fail-closed)
- Pinned seed: 1337
- Batch size 2, gradient accumulation 2 (effective batch 4)
- Fixed budget: exactly 3,600 successful updates
- Dual checkpointing: fixed_budget_3600.pt and best.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add repo root and src to sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))

import torch
from langfens_prep.trn04.trainer import train_factorial_arm, ARM_CONFIGS
from langfens_prep.p_sup.trainer import RemoteCheckpointMirror
from audit_scripts.kaggle_checkpoint_sync import resolve_auth_token


def main():
    parser = argparse.ArgumentParser(description="TRN-04 Factorial Training Entrypoint")
    parser.add_argument(
        "--arm",
        type=str,
        required=True,
        choices=list(ARM_CONFIGS.keys()),
        help="Arm to train: C0, T1, T2, or T3"
    )
    parser.add_argument(
        "--split-manifest",
        type=str,
        default="manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json",
        help="Path to splits manifest"
    )
    parser.add_argument(
        "--zip-path",
        type=str,
        default=None,
        help="Path to authorized Fold 1 train/val audio zip"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base directory for output checkpoints"
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--warmup-updates", type=int, default=100)
    parser.add_argument("--max-updates", type=int, default=3600)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument(
        "--remote-mirror-target",
        type=str,
        default="minkhoaa/langfens-rq1-cross-fold-checkpoints",
        help="Remote persistent mirror target repository"
    )
    parser.add_argument(
        "--remote-mirror-backend",
        type=str,
        default="huggingface",
        help="Remote mirror backend (huggingface, local_dir)"
    )
    args = parser.parse_args()

    arm = args.arm.upper()
    split_manifest = Path(args.split_manifest)
    if not split_manifest.is_absolute():
        split_manifest = repo_root / split_manifest

    # Robust dataset discovery
    zip_path = None
    if args.zip_path:
        zip_path = Path(args.zip_path)
    if not zip_path or not zip_path.is_file():
        candidates = [
            Path("/kaggle/input/langfens-rq1-fold1-staging/authorized_RQ1-TRN-02A_fold1_train_val_only.bin"),
            Path("/kaggle/input/langfens-rq1-fold1-staging/authorized_RQ1-TRN-02A_fold1_train_val_only.zip"),
            repo_root / "datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip"
        ]
        if Path("/kaggle/input").exists():
            candidates.extend(Path("/kaggle/input").rglob("authorized_RQ1-TRN-02A_fold1_train_val_only.*"))
        for c in candidates:
            if c.is_file():
                zip_path = c
                break

    if not zip_path or not zip_path.is_file():
        raise FileNotFoundError(f"FAIL-CLOSED: Authorized Fold 1 dataset archive not found!")

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("/kaggle/working/checkpoints/RQ1-TRN-04") / arm.lower()
        if Path("/kaggle/working").is_dir()
        else repo_root / "checkpoints/RQ1-TRN-04" / arm.lower()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remote mirroring setup
    remote_mirror = None
    mirror_prefix = f"trn04/{arm.lower()}"
    auth_token = resolve_auth_token()

    if args.remote_mirror_backend and auth_token:
        try:
            remote_mirror = RemoteCheckpointMirror(
                backend=args.remote_mirror_backend,
                remote_target=args.remote_mirror_target,
                auth_token=auth_token,
                require_verification=False,
                allow_local_test=True
            )
            print(f"[MIRROR] Remote persistent mirroring configured: {args.remote_mirror_target}:{mirror_prefix}")
        except Exception as e:
            print(f"[MIRROR NOTE] Remote mirror initialization: {e}")

    # Launch training
    summary = train_factorial_arm(
        arm=arm,
        split_manifest_path=split_manifest,
        zip_path=zip_path,
        output_dir=output_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_updates=args.warmup_updates,
        max_updates=args.max_updates,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        max_epochs=args.max_epochs,
        remote_mirror=remote_mirror,
        remote_mirror_prefix=mirror_prefix
    )

    print(f"\n[DONE] Arm {arm} execution finished successfully.")


if __name__ == "__main__":
    main()
