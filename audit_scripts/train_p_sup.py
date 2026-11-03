#!/usr/bin/env python3
"""LANGFENS — RQ1 P-SUP Official Training Entrypoint.

Supports both:
a) Subtype treatment: CompactOperationHead with objective="operation_ce"
b) Matched binary control: CompactOperationHead with objective="binary_collapsed"

Enforces exact contracts:
- Training dataset: FIT only (1,799 utterances across 12 speakers: BWC, ERMS, HKK, HQTV, SKA, SVBI, TNI, TXHC, YBAA, YDCK, YKWK, ZHAA)
- Selection dataset: SELECT (300 utterances across 2 speakers: MBMPS, TLV) for validation loss, AP tracking, early stopping, and saving best.pt during training
- Calibration dataset: CAL (300 utterances across 2 speakers: NCC, THV) for threshold/temperature calibration
- Fold-1 VAL: (300 utterances across 2 speakers: NJS, RRBI) preserved strictly unseen during training & selection; used ONLY for final candidate comparison under Evaluation Contract V1
- Sealed TEST partition: (900 utterances across 6 speakers: ABA, ASI, EBVS, HJK, LXC, PNV) strictly prohibited; fail-closed with SealedSpeakerContaminationError
- seed 1337
- batch_size 2, gradient_accumulation_steps 2 (effective batch 4)
- warmup_updates 100, cosine schedule over 15,000 horizon
- max 3,600 successful updates
- AdamW lr 0.001, weight decay 0.01, gradient clipping 1.0, CUDA FP16 AMP
- Dual checkpointing: fixed_budget_3600.pt and best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add repo root and src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from langfens_prep.p_sup.trainer import train_p_sup
from langfens_prep.p_sup.dataset import (
    PSupL2ArcticDataset,
    FIT_SPEAKERS,
    SELECT_SPEAKERS,
    CALIBRATION_SPEAKERS,
    VAL_SPEAKERS,
    PARTITION_SPEAKERS,
    SealedSpeakerContaminationError
)
from langfens_prep.p_sup.runtime import SEALED_SPEAKERS


def main():
    parser = argparse.ArgumentParser(description="P-SUP Training Entrypoint")
    parser.add_argument(
        "--arm",
        type=str,
        choices=["treatment", "control", "both"],
        default="both",
        help="Arm to train: treatment (operation_ce), control (binary_collapsed), or both"
    )
    parser.add_argument(
        "--objective",
        type=str,
        choices=["operation_ce", "binary_collapsed"],
        default=None,
        help="Direct objective name override"
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
        default="/home/khoa/Projects/KLTN/Dataset/datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip",
        help="Path to authorized Fold 1 train/val audio zip"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/kaggle/working/checkpoints/RQ1-P-SUP",
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
    parser.add_argument("--train-partition", type=str, default="fit", help="Train partition (default: fit)")
    parser.add_argument("--select-partition", type=str, default="selection", help="Select partition (default: selection)")
    parser.add_argument("--cal-partition", type=str, default="calibration", help="Calibration partition (default: calibration)")
    parser.add_argument("--val-partition", type=str, default="val", help="Validation partition (default: val)")
    parser.add_argument(
        "--remote-mirror-backend",
        type=str,
        default="huggingface",
        choices=["local_dir", "huggingface"],
        help="Remote persistent mirror backend: local_dir or huggingface"
    )
    parser.add_argument(
        "--remote-mirror-target",
        type=str,
        default=None,
        help="Remote persistent mirror target: directory or Hugging Face repo id"
    )
    parser.add_argument(
        "--remote-mirror-token",
        type=str,
        default=None,
        help="Auth token for remote mirror (e.g. HF_TOKEN)"
    )
    parser.add_argument(
        "--remote-execution",
        action="store_true",
        default=False,
        help="Enable strict remote execution fail-closed checks (CUDA, Tesla T4, FP16 AMP)"
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from"
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    split_manifest = Path(args.split_manifest)
    if not split_manifest.is_absolute():
        split_manifest = repo_root / split_manifest

    zip_path = Path(args.zip_path)
    if not zip_path.is_file():
        # Fallback check under /kaggle/input
        kaggle_candidate = Path("/kaggle/input/langfens-rq1-fold1-staging/authorized_RQ1-TRN-02A_fold1_train_val_only.zip")
        if kaggle_candidate.is_file():
            zip_path = kaggle_candidate
        else:
            local_rel = repo_root / "datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip"
            if local_rel.is_file():
                zip_path = local_rel

    base_out = Path(args.output_dir)

    objectives = []
    if args.objective:
        objectives = [args.objective]
    elif args.arm == "treatment":
        objectives = ["operation_ce"]
    elif args.arm == "control":
        objectives = ["binary_collapsed"]
    elif args.arm == "both":
        objectives = ["operation_ce", "binary_collapsed"]

    print("=" * 80)
    print("LANGFENS — Official P-SUP Training Launch")
    print(f"Arms to execute: {objectives}")
    print(f"Split manifest: {split_manifest}")
    print(f"Zip path: {zip_path}")
    print(f"Base output dir: {base_out}")
    print("Partition Contract:")
    print("  FIT (train): 1,799 utterances across 12 speakers")
    print("  SELECT (validation/best.pt): 300 utterances across 2 speakers (MBMPS, TLV)")
    print("  CAL (calibration): 300 utterances across 2 speakers (NCC, THV)")
    print("  VAL (evaluation): 300 utterances across 2 speakers (NJS, RRBI) [PRESERVED UNSEEN]")
    print("  TEST (sealed): 900 utterances across 6 speakers [STRICTLY PROHIBITED]")
    print("=" * 80)

    # Preflight partition verification
    fit_ds = PSupL2ArcticDataset(split_manifest, zip_path, partition="fit", verify_checksums=False)
    select_ds = PSupL2ArcticDataset(split_manifest, zip_path, partition="selection", verify_checksums=False)
    cal_ds = PSupL2ArcticDataset(split_manifest, zip_path, partition="calibration", verify_checksums=False)
    val_ds = PSupL2ArcticDataset(split_manifest, zip_path, partition="val", verify_checksums=False)

    if len(fit_ds) != 1799 or fit_ds.speakers != FIT_SPEAKERS:
        raise ValueError(f"FIT partition verification failed: len={len(fit_ds)}, speakers={fit_ds.speakers}")
    if len(select_ds) != 300 or select_ds.speakers != SELECT_SPEAKERS:
        raise ValueError(f"SELECT partition verification failed: len={len(select_ds)}, speakers={select_ds.speakers}")
    if len(cal_ds) != 300 or cal_ds.speakers != CALIBRATION_SPEAKERS:
        raise ValueError(f"CAL partition verification failed: len={len(cal_ds)}, speakers={cal_ds.speakers}")
    if len(val_ds) != 300 or val_ds.speakers != VAL_SPEAKERS:
        raise ValueError(f"VAL partition verification failed: len={len(val_ds)}, speakers={val_ds.speakers}")

    # Strict disjointness check
    partitions = [fit_ds.speakers, select_ds.speakers, cal_ds.speakers, val_ds.speakers, SEALED_SPEAKERS]
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            overlap = partitions[i] & partitions[j]
            if overlap:
                raise ValueError(f"Partition speaker overlap detected: {overlap}")

    # Sealed test access check
    try:
        PSupL2ArcticDataset(split_manifest, zip_path, partition="test", verify_checksums=False)
        raise RuntimeError("FAIL-CLOSED CHECK FAILED: test partition did not raise SealedSpeakerContaminationError!")
    except SealedSpeakerContaminationError:
        pass

    print("[PREFLIGHT] All partition sizes, memberships, disjointness, and sealed-test checks passed.")

    # Remote Persistent Mirror Setup
    remote_mirror = None
    mirror_target = (
        args.remote_mirror_target
        or os.environ.get("REMOTE_MIRROR_TARGET")
        or os.environ.get("KAGGLE_REMOTE_MIRROR_TARGET")
    )
    if mirror_target:
        from langfens_prep.p_sup.trainer import RemoteCheckpointMirror
        remote_mirror = RemoteCheckpointMirror(
            backend=args.remote_mirror_backend,
            remote_target=mirror_target,
            auth_token=args.remote_mirror_token
        )
        ok, msg = remote_mirror.verify_readiness()
        if not ok:
            raise RuntimeError(f"FAIL-CLOSED: Remote mirror verification failed: {msg}")
        print(f"[REMOTE MIRROR] Verified remote persistent mirror: {args.remote_mirror_backend}:{mirror_target}")

    results = {}
    for obj in objectives:
        arm_name = "T-OP_treatment" if obj == "operation_ce" else "B-CONTROL_matched_binary"
        arm_out = base_out / arm_name
        arm_out.mkdir(parents=True, exist_ok=True)
        arm_prefix = "psup/t_op" if obj == "operation_ce" else "psup/b_control"

        res = train_p_sup(
            objective=obj,
            split_manifest_path=split_manifest,
            zip_path=zip_path,
            output_dir=arm_out,
            seed=args.seed,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_updates=args.warmup_updates,
            cosine_horizon_updates=15000,
            max_successful_updates=args.max_updates,
            lr=args.lr,
            weight_decay=args.weight_decay,
            gradient_clip=args.gradient_clip,
            max_epochs=args.max_epochs,
            train_partition=args.train_partition,
            select_partition=args.select_partition,
            cal_partition=args.cal_partition,
            val_partition=args.val_partition,
            remote_mirror=remote_mirror,
            remote_rel_prefix=arm_prefix,
            remote_execution=args.remote_execution,
            resume_checkpoint_path=args.resume_checkpoint,
        )
        results[obj] = res

    print("=" * 80)
    print("P-SUP Training Completed Successfully.")
    print(json.dumps(results, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
