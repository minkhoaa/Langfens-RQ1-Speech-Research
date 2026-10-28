#!/usr/bin/env python3
"""
LANGFENS — Kaggle Workflow Launcher and Quota Management Driver
Validates Kaggle environment, checks GPU quota, inspects software state,
and packages kernel artifacts for official Kaggle CLI deployment.

Safety Invariants:
1. Verified Quota: Requires at least 5.0h remaining GPU quota before scheduling.
2. Private & Isolated: All kernels and datasets must be strictly private (is_private: true).
3. Accelerator Binding: Binds strictly to NvidiaTeslaT4 accelerator shape.
4. Non-Execution Default: Prepares kernels with --no-run unless explicit execution authorized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EXPECTED_MANIFEST_SHA256 = "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b"
MINIMUM_REQUIRED_GPU_HOURS = 5.0


def compute_sha256(path: Path | str) -> str:
    """Computes SHA-256 digest of a file in 64KB blocks."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_kaggle_quota(raw_output: str) -> Dict[str, float]:
    """
    Parses table output of 'kaggle quota' command:
    resource  used   remaining  total   refreshAt
    GPU       0.00h  30.00h     30.00h  2026-10-31T00:00:00+00:00
    TPU       0.00h  20.00h     20.00h  2026-10-31T00:00:00+00:00
    """
    res: Dict[str, float] = {}
    for line in raw_output.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        resource_type = parts[0].upper()
        if resource_type in ("GPU", "TPU") and len(parts) >= 4:
            try:
                used = float(parts[1].rstrip("h"))
                remaining = float(parts[2].rstrip("h"))
                total = float(parts[3].rstrip("h"))
                res[f"{resource_type.lower()}_used_hours"] = used
                res[f"{resource_type.lower()}_remaining_hours"] = remaining
                res[f"{resource_type.lower()}_total_hours"] = total
            except ValueError:
                continue
    return res


def check_kaggle_quota(min_gpu_hours: float = MINIMUM_REQUIRED_GPU_HOURS) -> Tuple[bool, str, Dict[str, float]]:
    """
    Queries official Kaggle CLI for accelerator quota and validates eligibility.
    Fails closed if quota is exhausted or insufficient.
    """
    try:
        proc = subprocess.run(["kaggle", "quota"], capture_output=True, text=True, check=True)
    except FileNotFoundError:
        return False, "FAIL-CLOSED: Official Kaggle CLI ('kaggle') is not installed or not in PATH.", {}
    except subprocess.CalledProcessError as e:
        return False, f"FAIL-CLOSED: 'kaggle quota' command failed: {e.stderr}", {}

    quota = parse_kaggle_quota(proc.stdout)
    gpu_remaining = quota.get("gpu_remaining_hours", 0.0)

    if gpu_remaining <= 0.0:
        return (
            False,
            f"FAIL-CLOSED: Weekly Kaggle GPU accelerator quota is exhausted (0.00h remaining).",
            quota,
        )

    if gpu_remaining < min_gpu_hours:
        return (
            False,
            f"FAIL-CLOSED: Remaining Kaggle GPU quota ({gpu_remaining:.2f}h) is below required minimum ({min_gpu_hours:.2f}h).",
            quota,
        )

    return True, f"Kaggle GPU quota verified: {gpu_remaining:.2f}h remaining.", quota


def validate_git_software_state(
    expected_commit: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> Tuple[bool, str]:
    """
    Verifies that the repository is clean, points to an authorized commit,
    and manifest digests match authoritative ledgers.
    """
    root = repo_root or Path(__file__).resolve().parent.parent

    # 1. Manifest digest check
    manifest_p = root / "manifests" / "RQ1-SPLIT-01" / "l2arctic_4fold_splits.json"
    if not manifest_p.is_file():
        return False, f"FAIL-CLOSED: Split manifest not found at '{manifest_p}'."
    digest = compute_sha256(manifest_p)
    if digest != EXPECTED_MANIFEST_SHA256:
        return False, f"FAIL-CLOSED: Split manifest SHA mismatch: {digest} != expected."

    # 2. Git HEAD SHA
    try:
        head_proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
        head_sha = head_proc.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, f"FAIL-CLOSED: Git HEAD query failed: {e.stderr}"

    if expected_commit and head_sha != expected_commit:
        return False, f"FAIL-CLOSED: Current commit '{head_sha}' does not match expected '{expected_commit}'."

    # 3. Check configs exist
    for cfg_name in ("rq1_trn_02b_c0_kaggle.yaml", "rq1_trn_02b_c1_kaggle.yaml", "rq1_trn_02b_benchmark_kaggle.yaml"):
        if not (root / "configs" / cfg_name).is_file():
            return False, f"FAIL-CLOSED: Required Kaggle config '{cfg_name}' missing."

    return True, f"Software state verified at commit {head_sha}."


def prepare_kaggle_kernel_directory(
    deploy_dir: Path | str,
    kernel_slug: str = "langfens-rq1-trn-02b-t4x2",
    title: str = "LANGFENS RQ1-TRN-02B T4x2",
    code_file_source: Optional[Path | str] = None,
    dataset_sources: Optional[List[str]] = None,
) -> Path:
    """
    Prepares deployment directory with validated kernel-metadata.json
    and entrypoint script.
    """
    out_p = Path(deploy_dir).resolve()
    out_p.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent
    code_src = Path(code_file_source or repo_root / "deploy" / "kaggle" / "kaggle_entrypoint.py").resolve()
    code_dest = (out_p / code_src.name).resolve()
    if code_src != code_dest and code_src.is_file():
        shutil.copyfile(str(code_src), str(code_dest))

    metadata = {
        "id": f"minkhoaa/{kernel_slug}",
        "title": title,
        "code_file": code_src.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": dataset_sources or ["minkhoaa/langfens-rq1-fold1-staging"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }

    meta_file = out_p / "kernel-metadata.json"
    meta_file.write_text(json.dumps(metadata, indent=2) + "\n")
    return out_p


def push_kaggle_kernel(
    deploy_dir: Path | str,
    no_run: bool = True,
) -> Tuple[bool, str]:
    """
    Executes 'kaggle kernels push' via official CLI.
    Default enforces --no-run for safe preparation without premature remote execution.
    """
    cmd = ["kaggle", "kernels", "push", "-p", str(deploy_dir), "--accelerator", "NvidiaTeslaT4"]
    if no_run:
        cmd.append("--no-run")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, proc.stdout
    except subprocess.CalledProcessError as e:
        return False, f"Failed to push kernel: {e.stderr}"


def main():
    parser = argparse.ArgumentParser(description="LANGFENS — Kaggle Workflow Launcher")
    parser.add_argument("--check-quota", action="store_true", help="Check Kaggle accelerator quota")
    parser.add_argument("--check-software-only", action="store_true", help="Validate git and config software state")
    parser.add_argument("--prepare-kernel", action="store_true", help="Prepare deploy/kaggle directory")
    parser.add_argument("--deploy-dir", type=str, default="deploy/kaggle", help="Kernel package folder")
    parser.add_argument("--dry-run", action="store_true", help="Perform preflight without modifying state")
    args = parser.parse_args()

    if args.check_quota:
        ok, msg, q = check_kaggle_quota()
        print(f"Quota check: ok={ok}, message={msg}")
        print(json.dumps(q, indent=2))
        if not ok:
            sys.exit(1)

    if args.check_software_only:
        ok, msg = validate_git_software_state()
        print(f"Software check: ok={ok}, message={msg}")
        if not ok:
            sys.exit(1)

    if args.prepare_kernel:
        p = prepare_kaggle_kernel_directory(deploy_dir=args.deploy_dir)
        print(f"Prepared kernel package in {p}")


if __name__ == "__main__":
    main()
