#!/usr/bin/env python3
"""
LANGFENS — RQ1-TRN-02B: Colab VM Training Execution Driver
Designed for execution on Google Colab VM via:
    colab exec -s <session> -f audit_scripts/colab_train_exec.py --timeout 28800

Key Contracts & Invariants:
1. Strict Hardware Enforcement: Prohibits fallback to CPU. Fails closed if CUDA is unavailable.
2. Verified Git State: Enforces that repository HEAD exactly matches target commit SHA.
3. Local Staged Artifacts Verification: Verifies local VM audio zip and baseline weights exist and match SHA-256.
4. Isolated Arm Configuration: Correctly routes between Control Arm C0 and Treatment Arm C1.
5. Persistent Google Drive Logging: Captures full training execution stdout/stderr to local and remote logs.
6. Post-Run Verification: Verifies completed remote checkpoints and history on Google Drive.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Ensure src is importable
repo_candidate = Path(__file__).resolve().parent.parent
if (repo_candidate / "src").is_dir():
    sys.path.insert(0, str(repo_candidate / "src"))

EXPECTED_BASELINE_SHA256 = "a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b"
EXPECTED_DATASET_SHA256 = "b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9"
PROJECT_DRIVE_NAME = "KLTN-Dataset"


def compute_sha256(path: Path | str) -> str:
    """Computes SHA-256 digest in 64KB blocks."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_drive_root(custom_path: Optional[str] = None) -> Path:
    """Locates KLTN-Dataset root directory under /content/drive/ or custom mount."""
    if custom_path:
        p = Path(custom_path)
        if p.is_dir():
            return p
        raise FileNotFoundError(f"Specified custom Drive root not found: {custom_path}")

    candidates = [
        Path("/content/drive/MyDrive") / PROJECT_DRIVE_NAME,
        Path("/content/drive/Shareddrives") / PROJECT_DRIVE_NAME,
        Path("/content/drive") / PROJECT_DRIVE_NAME,
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand

    drive_base = Path("/content/drive")
    if drive_base.is_dir():
        for root, dirs, _ in os.walk(str(drive_base)):
            if PROJECT_DRIVE_NAME in dirs:
                return Path(root) / PROJECT_DRIVE_NAME

    raise RuntimeError(
        f"Google Drive root '{PROJECT_DRIVE_NAME}' not found under /content/drive/.\n"
        f"Please verify that Google Drive is mounted before running training."
    )


def execute_training(
    arm: str,
    target_commit: str,
    repo_root: Path,
    drive_root: Path,
    local_zip_path: Path,
    local_baseline_path: Path,
    run_id: Optional[str] = None,
    resume: Optional[str] = None,
    eval_only: bool = False,
    mock_gpu: bool = False,
) -> int:
    """Executes the verified training workflow with full error capture and remote sync."""
    arm_norm = arm.strip().upper()
    if arm_norm not in ("C0", "C1"):
        raise ValueError(f"Invalid experimental arm: {arm}. Strictly C0 or C1 allowed.")

    print("=" * 80)
    print(f"LANGFENS — RQ1-TRN-02B Colab Training Execution")
    print(f"Arm: {arm_norm} | Target Commit: {target_commit}")
    print(f"Repo Root: {repo_root}")
    print(f"Drive Root: {drive_root}")
    print("=" * 80, flush=True)

    # 1. Strict GPU Check
    if not mock_gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is NOT available on this runtime!\n"
                    "Execution on local CPU is strictly forbidden by AGENTS.md.\n"
                    "Stop immediately and report the absence of GPU resources."
                )
            print(f"[GPU] CUDA Active: {torch.cuda.get_device_name(0)} (Device count: {torch.cuda.device_count()})", flush=True)
        except ImportError:
            raise RuntimeError("PyTorch is not installed in the execution environment.")

    # 2. Strict Git Commit Verification
    head_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
    ).strip().lower()
    if head_sha != target_commit.strip().lower():
        raise RuntimeError(
            f"Git commit mismatch before training: HEAD is {head_sha}, expected {target_commit}."
        )
    print(f"[GIT] Repository HEAD verified: {head_sha}", flush=True)

    # 3. Verify Local Staged Files
    if not local_zip_path.is_file():
        raise FileNotFoundError(f"Local staged dataset zip not found: {local_zip_path}")
    zip_sha = compute_sha256(local_zip_path)
    if zip_sha != EXPECTED_DATASET_SHA256:
        raise RuntimeError(f"Local staged dataset zip checksum corrupted: {zip_sha} != {EXPECTED_DATASET_SHA256}")
    print(f"[DATA] Local dataset verified: {local_zip_path} ({zip_sha[:12]}...)", flush=True)

    if not local_baseline_path.is_file():
        raise FileNotFoundError(f"Local staged baseline checkpoint not found: {local_baseline_path}")
    baseline_sha = compute_sha256(local_baseline_path)
    if baseline_sha != EXPECTED_BASELINE_SHA256:
        raise RuntimeError(f"Local baseline checkpoint checksum corrupted: {baseline_sha} != {EXPECTED_BASELINE_SHA256}")
    print(f"[MODEL] Local baseline weights verified: {local_baseline_path} ({baseline_sha[:12]}...)", flush=True)

    # 4. Determine Config and Remote Output Paths
    config_file = repo_root / f"configs/rq1_trn_02b_{arm_norm.lower()}.yaml"
    if not config_file.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    drive_out_dir = drive_root / f"manifests/RQ1-TRN-02B/checkpoints/{arm_norm}"
    if run_id:
        drive_out_dir = drive_out_dir / run_id
    drive_out_dir.mkdir(parents=True, exist_ok=True)

    # 5. Build Execution Command
    train_script = repo_root / "audit_scripts/train_rq1_trn_02b.py"
    cmd = [
        sys.executable,
        str(train_script),
        "--config", str(config_file),
        "--zip-path", str(local_zip_path),
        "--initial-checkpoint", str(local_baseline_path),
        "--drive-output-dir", str(drive_out_dir),
    ]
    if run_id:
        cmd.extend(["--run-id", run_id])
    if resume:
        cmd.extend(["--resume", resume])
    if eval_only:
        cmd.append("--eval-only")

    log_file_local = repo_root / f"training_{arm_norm.lower()}_{int(time.time())}.log"
    remote_log = drive_out_dir / f"training_{arm_norm.lower()}.log"

    print(f"[LAUNCH] Executing training driver: {' '.join(cmd)}", flush=True)
    print(f"[LOGGING] Local log: {log_file_local} | Remote log: {remote_log}", flush=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo_root}/src:{repo_root}"

    start_time = time.time()
    with open(log_file_local, "w", buffering=1) as log_fp:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(repo_root),
            env=env,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_fp.write(line)
            log_fp.flush()

        ret = proc.wait()

    elapsed = time.time() - start_time
    print(f"[FINISHED] Process returned code {ret} in {elapsed / 60.0:.2f} minutes.", flush=True)

    # Sync local log to Drive
    if log_file_local.is_file():
        shutil.copyfile(str(log_file_local), str(remote_log))
        print(f"[LOG SYNC] Training log synced to Google Drive: {remote_log}", flush=True)

    if ret != 0:
        raise RuntimeError(f"Training run failed with exit code {ret}. Inspect log at {remote_log}.")

    # 6. Post-Run Remote Artifact Verification
    last_pt = drive_out_dir / "last.pt"
    if not last_pt.is_file():
        raise FileNotFoundError(f"Expected final checkpoint last.pt not found at {last_pt}")
    last_sha = compute_sha256(last_pt)
    print(f"[REMOTE ARTIFACT] Verified last.pt on Google Drive: {last_sha}", flush=True)

    history_file = drive_out_dir / "history.json"
    if history_file.is_file():
        print(f"[REMOTE ARTIFACT] Verified history.json on Google Drive.", flush=True)

    endpoint_pt = drive_out_dir / "fixed_budget_checkpoint.pt"
    if endpoint_pt.is_file():
        endpoint_sha = compute_sha256(endpoint_pt)
        print(f"[REMOTE ARTIFACT] Verified fixed_budget_checkpoint.pt on Google Drive: {endpoint_sha}", flush=True)

    print("=" * 80)
    print(f"TRAINING EXECUTION COMPLETE & PERSISTED FOR ARM {arm_norm}.")
    print("=" * 80, flush=True)
    return ret


def main():
    ap = argparse.ArgumentParser(description="RQ1-TRN-02B Colab Training Execution Driver")
    ap.add_argument("--arm", default=os.environ.get("ARM", "C0"), help="Arm identifier (C0 or C1)")
    ap.add_argument("--target-commit", default=os.environ.get("TARGET_COMMIT", ""), help="Target Git commit SHA")
    ap.add_argument("--repo-root", default=os.environ.get("REPO_ROOT", "/content/repo"), help="Path to cloned repository")
    ap.add_argument("--drive-root", default=os.environ.get("DRIVE_ROOT", ""), help="Path to KLTN-Dataset root in Google Drive")
    ap.add_argument("--local-zip", default=os.environ.get("LOCAL_ZIP", "/content/data/fold1_train_val_only.zip"), help="Local staged dataset zip path")
    ap.add_argument("--local-baseline", default=os.environ.get("LOCAL_BASELINE", "/content/checkpoints/RQ1-TRN-02A/recovered/best.pt"), help="Local staged baseline checkpoint path")
    ap.add_argument("--run-id", default=os.environ.get("RUN_ID", None), help="Optional execution run identifier")
    ap.add_argument("--resume", default=os.environ.get("RESUME", None), help="Path to checkpoint for resumption")
    ap.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    ap.add_argument("--mock-gpu", action="store_true", help="Skip CUDA requirement for unit testing only")
    args = ap.parse_args()

    if not args.target_commit:
        print("ERROR: --target-commit or TARGET_COMMIT environment variable is required.", file=sys.stderr)
        sys.exit(1)

    repo_path = Path(args.repo_root)
    drive_path = find_drive_root(args.drive_root if args.drive_root else None)

    execute_training(
        arm=args.arm,
        target_commit=args.target_commit,
        repo_root=repo_path,
        drive_root=drive_path,
        local_zip_path=Path(args.local_zip),
        local_baseline_path=Path(args.local_baseline),
        run_id=args.run_id,
        resume=args.resume,
        eval_only=args.eval_only,
        mock_gpu=args.mock_gpu,
    )


if __name__ == "__main__":
    main()
