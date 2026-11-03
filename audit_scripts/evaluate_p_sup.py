#!/usr/bin/env python3
"""LANGFENS — Strict Evaluation Contract V1 Evaluator for P-SUP Checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add repo root and src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langfens_prep.p_sup.evaluator import evaluate_checkpoint_v1


def main():
    parser = argparse.ArgumentParser(description="P-SUP Evaluation under Contract V1")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt")
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
        help="Path to authorized Fold 1 audio zip"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="manifests/RQ1-P-SUP/evaluation",
        help="Output directory for predictions and metrics"
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    split_manifest = Path(args.split_manifest)
    if not split_manifest.is_absolute():
        split_manifest = repo_root / split_manifest

    zip_path = Path(args.zip_path)
    if not zip_path.is_file():
        local_rel = repo_root / "datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip"
        if local_rel.is_file():
            zip_path = local_rel

    res = evaluate_checkpoint_v1(
        checkpoint_path=args.checkpoint,
        split_manifest_path=split_manifest,
        zip_path=zip_path,
        output_dir=args.output_dir
    )

    print("=" * 80)
    print("P-SUP V1 Evaluation Completed:")
    print(json.dumps(res, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
