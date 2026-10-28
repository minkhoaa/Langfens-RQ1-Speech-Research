#!/usr/bin/env python3
"""
LANGFENS — RQ1-TRN-02B: Colab VM Local Staging & Environment Verification Driver
Designed for execution on Google Colab VM via:
    colab exec -s <session> -f audit_scripts/colab_stage_and_verify.py --timeout 1800

Key Contracts & Invariants:
1. Validates Arm Identifier: Strictly C0 (p=0.00) or C1 (p=0.25).
2. Verifies Commit Provenance: Enforces exact target Git commit SHA (fails closed if mismatch).
3. Verifies Hardware: Requires CUDA GPU (strictly fails closed if GPU unavailable).
4. Google Drive Mount Check: Locates KLTN-Dataset root (fails closed with instructions if unmounted).
5. VM Local Staging: Copies authorized dataset archive and baseline weights to VM local disk (/content/...).
6. Cryptographic Verification: Computes SHA-256 of staged files and verifies exact matches.
7. Output Directory Preparation: Ensures persistent Google Drive output path is writable and clean.
8. No Fallback: Strictly prohibits falling back to CPU or local training.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Cryptographic Digests Ledger
EXPECTED_MANIFEST_SHA256 = "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b"
EXPECTED_BASELINE_SHA256 = "a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b"
EXPECTED_DATASET_SHA256 = "b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9"

PROJECT_DRIVE_NAME = "KLTN-Dataset"
PROJECT_DRIVE_FOLDER_ID = "1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK"


def compute_sha256(path: Path | str) -> str:
    """Computes SHA-256 digest of a file in 64KB blocks."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for hash computation: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_arm(arm: str) -> str:
    """Validates arm identifier strictly against C0 or C1."""
    arm_norm = arm.strip().upper()
    if arm_norm not in ("C0", "C1"):
        raise ValueError(f"Invalid experimental arm '{arm}'. Strictly 'C0' or 'C1' allowed.")
    return arm_norm


def validate_commit_sha(sha: str) -> str:
    """Validates 40-character hexadecimal Git commit SHA."""
    sha_clean = sha.strip()
    if len(sha_clean) != 40 or not all(c in "0123456789abcdefABCDEF" for c in sha_clean):
        raise ValueError(f"Invalid Git commit SHA '{sha}'. Expected 40-character hexadecimal string.")
    return sha_clean.lower()


def find_drive_root(custom_path: Optional[str] = None) -> Path:
    """Locates KLTN-Dataset root directory under /content/drive/ or custom mount."""
    if custom_path:
        p = Path(custom_path)
        if p.is_dir():
            return p
        raise FileNotFoundError(f"Specified custom Drive root not found: {custom_path}")

    # Standard Google Drive mount candidates
    candidates = [
        Path("/content/drive/MyDrive") / PROJECT_DRIVE_NAME,
        Path("/content/drive/Shareddrives") / PROJECT_DRIVE_NAME,
        Path("/content/drive") / PROJECT_DRIVE_NAME,
    ]

    for cand in candidates:
        if cand.is_dir():
            return cand

    # Deep search under /content/drive if mounted
    drive_base = Path("/content/drive")
    if drive_base.is_dir():
        for root, dirs, _ in os.walk(str(drive_base)):
            if PROJECT_DRIVE_NAME in dirs:
                return Path(root) / PROJECT_DRIVE_NAME

    raise RuntimeError(
        f"Google Drive root '{PROJECT_DRIVE_NAME}' not found under /content/drive/.\n"
        f"Drive mounting requires interactive human authorization.\n"
        f"Please run 'colab drivemount -s <session>' in an interactive terminal\n"
        f"or authorize Drive mount in the Colab web UI before running this staging script."
    )


def verify_git_commit(repo_root: Path, target_commit: str) -> str:
    """Verifies that the repository HEAD matches the immutable target commit."""
    if not (repo_root / ".git").is_dir():
        raise FileNotFoundError(f"Git repository not found at {repo_root}")

    head_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
    ).strip().lower()

    if head_sha != target_commit.lower():
        raise RuntimeError(
            f"Repository commit mismatch: HEAD is {head_sha}, expected target {target_commit}.\n"
            f"Please run 'git checkout {target_commit}' in {repo_root}."
        )

    # Check for uncommitted working tree modifications
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=str(repo_root), text=True
    ).strip()
    if status:
        raise RuntimeError(f"Repository working tree is not clean at {repo_root}:\n{status}")

    return head_sha


def stage_file_with_hash(
    src_path: Path,
    dest_path: Path,
    expected_sha256: str,
    desc: str,
) -> str:
    """Copies file to local destination if needed and validates SHA-256."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if destination file already exists and matches hash
    if dest_path.is_file():
        local_sha = compute_sha256(dest_path)
        if local_sha == expected_sha256:
            print(f"[STAGING CACHE HIT] {desc} already exists at {dest_path} with verified SHA-256.", flush=True)
            return local_sha
        else:
            print(f"[STAGING MISMATCH] Existing file at {dest_path} corrupt ({local_sha} != {expected_sha256}). Re-copying...", flush=True)
            dest_path.unlink()

    if not src_path.is_file():
        raise FileNotFoundError(f"Source file for {desc} not found at {src_path}")

    print(f"[STAGING COPY] Copying {desc} from {src_path} to {dest_path}...", flush=True)
    shutil.copyfile(str(src_path), str(dest_path))

    local_sha = compute_sha256(dest_path)
    if local_sha != expected_sha256:
        dest_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Cryptographic hash mismatch for {desc}!\n"
            f"Computed: {local_sha}\n"
            f"Expected: {expected_sha256}"
        )

    print(f"[STAGING VERIFIED] {desc} successfully verified ({local_sha[:12]}...).", flush=True)
    return local_sha


def check_conflicting_checkpoints(outdir: Path, resume: bool = False):
    """Ensures destination does not contain prior conflicting checkpoints unless resuming."""
    guard_files = ["last.pt", "best.pt", "fixed_budget_checkpoint.pt", "halted_guardrail.pt"]
    found = [f for f in guard_files if (outdir / f).is_file()]
    if found and not resume:
        raise FileExistsError(
            f"Conflicting checkpoint artifacts already exist in destination {outdir}: {found}.\n"
            f"To prevent accidental clobbering, fresh training is blocked.\n"
            f"Specify a unique --run-id or supply --resume to continue from an existing checkpoint."
        )


def verify_write_access(target_dir: Path):
    """Verifies write permissions to persistent storage by writing and removing a test file."""
    target_dir.mkdir(parents=True, exist_ok=True)
    test_file = target_dir / ".write_verification.tmp"
    try:
        test_file.write_text("LANGFENS WRITE VERIFICATION\n")
        read_back = test_file.read_text()
        if "LANGFENS WRITE VERIFICATION" not in read_back:
            raise IOError("Write verification read-back mismatch.")
    finally:
        if test_file.exists():
            test_file.unlink()


def stage_and_verify(
    arm: str,
    target_commit: str,
    repo_root: Path,
    local_data_dir: Path,
    local_checkpoint_dir: Path,
    drive_root: Path,
    run_id: Optional[str] = None,
    resume: bool = False,
    mock_gpu: bool = False,
) -> Dict[str, Any]:
    """Executes full staging and environment verification protocol."""
    arm_norm = validate_arm(arm)
    commit_norm = validate_commit_sha(target_commit)

    print("=" * 80)
    print(f"LANGFENS — RQ1-TRN-02B Colab Staging & Environment Verification")
    print(f"Arm: {arm_norm} | Target Commit: {commit_norm}")
    print(f"Drive Root: {drive_root}")
    print("=" * 80, flush=True)

    # 1. Hardware & CUDA Check
    if not mock_gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is NOT available on this runtime!\n"
                    "Per AGENTS.md execution boundaries, training without a Colab GPU is strictly prohibited.\n"
                    "Do NOT fall back to CPU training. Provision a GPU runtime via 'colab new -s <name> --gpu T4'."
                )
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[HARDWARE] GPU verified: {gpu_name} (CUDA {torch.version.cuda})", flush=True)
        except ImportError:
            raise RuntimeError("PyTorch is not installed in the current environment.")

    # 2. Repository and Commit Verification
    verified_head = verify_git_commit(repo_root, commit_norm)
    print(f"[GIT PROVENANCE] HEAD verified: {verified_head} (matches target {commit_norm})", flush=True)

    # 3. Locate Source Artifacts in Google Drive
    dataset_candidates = [
        drive_root / "manifests/RQ1-TRN-02A/authorized_data/fold1_train_val_only.zip",
        drive_root / "datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip",
    ]
    dataset_src: Optional[Path] = None
    for cand in dataset_candidates:
        if cand.is_file():
            dataset_src = cand
            break
    if not dataset_src:
        raise FileNotFoundError(
            f"Authorized dataset archive not found under Drive root {drive_root}.\n"
            f"Searched: {[str(c) for c in dataset_candidates]}"
        )

    baseline_candidates = [
        drive_root / "manifests/RQ1-TRN-02A/checkpoints/best.pt",
        drive_root / "checkpoints/RQ1-TRN-02A/recovered/best.pt",
    ]
    baseline_src: Optional[Path] = None
    for cand in baseline_candidates:
        if cand.is_file():
            baseline_src = cand
            break
    if not baseline_src:
        raise FileNotFoundError(
            f"Baseline R checkpoint (best.pt) not found under Drive root {drive_root}.\n"
            f"Searched: {[str(c) for c in baseline_candidates]}"
        )

    # 4. Stage to VM Local Storage & Cryptographic Verification
    local_data_dir.mkdir(parents=True, exist_ok=True)
    local_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    local_zip = local_data_dir / "fold1_train_val_only.zip"
    staged_zip_sha = stage_file_with_hash(
        dataset_src, local_zip, EXPECTED_DATASET_SHA256, "Fold 1 Authorized Dataset Archive"
    )

    local_baseline = local_checkpoint_dir / "best.pt"
    staged_baseline_sha = stage_file_with_hash(
        baseline_src, local_baseline, EXPECTED_BASELINE_SHA256, "Baseline R Model Weights (best.pt)"
    )

    # Also verify local split manifest inside repo
    local_manifest = repo_root / "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json"
    if not local_manifest.is_file():
        raise FileNotFoundError(f"Split manifest not found at {local_manifest}")
    manifest_sha = compute_sha256(local_manifest)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            f"Local split manifest hash mismatch: {manifest_sha} != {EXPECTED_MANIFEST_SHA256}"
        )
    print(f"[MANIFEST] Split manifest verified: {manifest_sha[:12]}...", flush=True)

    # 5. Persistent Output Directory Verification
    drive_out_dir = drive_root / f"manifests/RQ1-TRN-02B/checkpoints/{arm_norm}"
    if run_id:
        drive_out_dir = drive_out_dir / run_id

    verify_write_access(drive_out_dir)
    print(f"[STORAGE ACCESS] Verified persistent write access to: {drive_out_dir}", flush=True)

    check_conflicting_checkpoints(drive_out_dir, resume=resume)
    print(f"[CONFLICT CHECK] Output directory clean (no conflicting prior checkpoints).", flush=True)

    summary = {
        "status": "READY_FOR_TRAINING",
        "arm": arm_norm,
        "verified_commit": verified_head,
        "staged_zip_path": str(local_zip),
        "staged_zip_sha256": staged_zip_sha,
        "staged_baseline_path": str(local_baseline),
        "staged_baseline_sha256": staged_baseline_sha,
        "manifest_sha256": manifest_sha,
        "drive_output_dir": str(drive_out_dir),
    }

    print("=" * 80)
    print("STAGING & PREFLIGHT VERIFICATION PASSED: VM is ready for training launch.")
    print("=" * 80, flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description="RQ1-TRN-02B Colab Staging & Verification Driver")
    ap.add_argument("--arm", default=os.environ.get("ARM", "C0"), help="Arm identifier (C0 or C1)")
    ap.add_argument("--target-commit", default=os.environ.get("TARGET_COMMIT", ""), help="Target Git commit SHA")
    ap.add_argument("--repo-root", default=os.environ.get("REPO_ROOT", "/content/repo"), help="Path to cloned repository")
    ap.add_argument("--drive-root", default=os.environ.get("DRIVE_ROOT", ""), help="Path to KLTN-Dataset root in Google Drive")
    ap.add_argument("--local-data-dir", default=os.environ.get("LOCAL_DATA_DIR", "/content/data"), help="VM-local data directory")
    ap.add_argument("--local-checkpoint-dir", default=os.environ.get("LOCAL_CHECKPOINT_DIR", "/content/checkpoints/RQ1-TRN-02A/recovered"), help="VM-local checkpoint directory")
    ap.add_argument("--run-id", default=os.environ.get("RUN_ID", None), help="Optional execution run identifier")
    ap.add_argument("--resume", action="store_true", help="Allow resumption if prior checkpoints exist")
    ap.add_argument("--mock-gpu", action="store_true", help="Skip CUDA requirement for unit testing only")
    args = ap.parse_args()

    if not args.target_commit:
        print("ERROR: --target-commit or TARGET_COMMIT environment variable is required.", file=sys.stderr)
        sys.exit(1)

    repo_path = Path(args.repo_root)
    drive_path = find_drive_root(args.drive_root if args.drive_root else None)

    stage_and_verify(
        arm=args.arm,
        target_commit=args.target_commit,
        repo_root=repo_path,
        local_data_dir=Path(args.local_data_dir),
        local_checkpoint_dir=Path(args.local_checkpoint_dir),
        drive_root=drive_path,
        run_id=args.run_id,
        resume=args.resume,
        mock_gpu=args.mock_gpu,
    )


if __name__ == "__main__":
    main()
