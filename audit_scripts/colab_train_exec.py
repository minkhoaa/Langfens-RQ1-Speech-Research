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
5. Single Predictable Output Directory: Avoids duplicate run-ID path nesting.
6. Authentic Drive Persistence Check: Validates genuine Google Drive target before execution.
7. Primary Endpoint Verification: Classifies and validates completed endpoints (COMPLETED_3600_UPDATES,
   INCOMPLETE, HALTED_GUARDRAIL, FAILED). Does NOT label run complete merely because last.pt exists.
8. Kernel Execution Compatibility: Does not assume __file__ is defined; ignores ipykernel arguments.
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
from typing import Any, Dict, List, Optional

# Kernel execution compatibility: avoid relying on __file__
_repo_env = os.environ.get("REPO_ROOT", "/content/repo")
_repo_path = Path(_repo_env)
if (_repo_path / "src").is_dir():
    sys.path.insert(0, str(_repo_path / "src"))
if _repo_path.is_dir():
    sys.path.insert(0, str(_repo_path))

EXPECTED_BASELINE_SHA256 = "a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b"
EXPECTED_DATASET_SHA256 = "b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9"
PROJECT_DRIVE_NAME = "KLTN-Dataset"
PROJECT_DRIVE_FOLDER_ID = "1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK"


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


def validate_arm(arm: str) -> str:
    """Validates arm identifier strictly against C0 or C1."""
    norm = arm.strip().upper()
    if norm not in ("C0", "C1"):
        raise ValueError(f"Invalid experimental arm: {arm}. Strictly C0 or C1 allowed.")
    return norm


def resolve_drive_output_dir(drive_root: Path, arm: str, run_id: Optional[str] = None) -> Path:
    """Constructs single predictable output directory without duplicate run-ID nesting."""
    arm_norm = validate_arm(arm)
    p = Path(drive_root)
    if "manifests/RQ1-TRN-02B/checkpoints" in str(p):
        base = p
    else:
        base = p / f"manifests/RQ1-TRN-02B/checkpoints/{arm_norm}"

    if run_id:
        if base.name == run_id:
            return base
        return base / run_id
    return base


def verify_drive_authenticity(drive_root: Path, require_mount: bool = True) -> bool:
    """Validates that drive_root is an authentic Google Drive persistence target."""
    is_colab_env = Path("/content").is_dir()
    if is_colab_env and require_mount:
        drive_mount = Path("/content/drive")
        is_mounted = os.path.ismount(str(drive_mount))
        if not is_mounted and Path("/proc/mounts").is_file():
            mounts_text = Path("/proc/mounts").read_text()
            is_mounted = any("/content/drive" in line and ("fuse" in line or "drivefs" in line) for line in mounts_text.splitlines())

        if not is_mounted:
            raise RuntimeError(
                "FAIL-CLOSED: /content/drive is NOT an active Google Drive mountpoint.\n"
                "Please run 'colab drivemount -s <session>' before executing training."
            )

    # Identity check
    baseline_chk = drive_root / "manifests/RQ1-TRN-02A/checkpoints/best.pt"
    if not baseline_chk.is_file():
        alt_chk = drive_root / "checkpoints/RQ1-TRN-02A/recovered/best.pt"
        if alt_chk.is_file():
            baseline_chk = alt_chk
        else:
            raise FileNotFoundError(
                f"FAIL-CLOSED: Directory '{drive_root}' lacks authentic Baseline R weights.\n"
                f"Cannot verify authentic identity for Google Drive Root ID: {PROJECT_DRIVE_FOLDER_ID}."
            )
    return True


def find_drive_root(custom_path: Optional[str] = None, require_mount: bool = True) -> Path:
    """Locates and validates authentic KLTN-Dataset root directory."""
    if custom_path:
        p = Path(custom_path)
        if p.is_dir():
            verify_drive_authenticity(p, require_mount=require_mount)
            return p
        raise FileNotFoundError(f"Specified custom Drive root not found: {custom_path}")

    candidates = [
        Path("/content/drive/MyDrive") / PROJECT_DRIVE_NAME,
        Path("/content/drive/Shareddrives") / PROJECT_DRIVE_NAME,
        Path("/content/drive") / PROJECT_DRIVE_NAME,
    ]
    for cand in candidates:
        if cand.is_dir():
            try:
                verify_drive_authenticity(cand, require_mount=require_mount)
                return cand
            except Exception:
                continue

    drive_base = Path("/content/drive")
    if drive_base.is_dir():
        for root, dirs, _ in os.walk(str(drive_base)):
            if PROJECT_DRIVE_NAME in dirs:
                cand = Path(root) / PROJECT_DRIVE_NAME
                try:
                    verify_drive_authenticity(cand, require_mount=require_mount)
                    return cand
                except Exception:
                    continue

    raise RuntimeError(
        f"Google Drive root '{PROJECT_DRIVE_NAME}' not found under /content/drive/.\n"
        f"Please verify that Google Drive is mounted before running training."
    )


def classify_training_endpoint(
    drive_output_dir: Path,
    expected_arm: str,
) -> Dict[str, Any]:
    """
    Rigorously verifies and classifies the training run endpoint:
    - COMPLETED_3600_UPDATES: Primary matched-budget endpoint reached and verified.
    - INCOMPLETE: Terminated early (epoch cap / AMP skips) without guardrail violation.
    - HALTED_GUARDRAIL: Stopped due to CTC PER degradation guardrail trip.
    - FAILED: Checkpoints missing, corrupted, or process failed.
    """
    arm_norm = validate_arm(expected_arm)
    p_dir = Path(drive_output_dir)

    # 1. Check for guardrail trip artifact
    halted_file = p_dir / "halted_guardrail.pt"
    if halted_file.is_file():
        try:
            import torch
            ck = torch.load(halted_file, map_location="cpu", weights_only=False)
            meta = ck.get("metadata", {})
            return {
                "status": "HALTED_GUARDRAIL",
                "message": (
                    f"Guardrail violation tripped at epoch {ck.get('epoch', '?')}: "
                    f"CTC PER measured {meta.get('ctc_per_measured')} vs anchor {meta.get('ctc_per_anchor')} "
                    f"(+{meta.get('ctc_per_degradation_pct')} absolute degradation)."
                ),
                "successful_updates": ck.get("successful_updates", 0),
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
            import torch
            ck = torch.load(fixed_budget_file, map_location="cpu", weights_only=False)
            meta = ck.get("metadata", {})

            # Check arm
            actual_arm = meta.get("arm")
            if actual_arm != arm_norm:
                return {
                    "status": "FAILED",
                    "message": f"Arm mismatch in fixed_budget_checkpoint: declared '{actual_arm}', expected '{arm_norm}'.",
                }

            # Reject guardrail violation in metadata
            if meta.get("guardrail_violation") is True:
                return {
                    "status": "HALTED_GUARDRAIL",
                    "message": "Checkpoint marked with guardrail violation in metadata.",
                }

            # Check successful updates count
            updates = ck.get("successful_updates", meta.get("successful_updates", 0))
            if updates < 3600 and not meta.get("primary_3600_updates_completed", False):
                return {
                    "status": "INCOMPLETE",
                    "message": f"fixed_budget_checkpoint has only {updates} / 3600 updates.",
                    "successful_updates": updates,
                }

            return {
                "status": "COMPLETED_3600_UPDATES",
                "message": f"Primary matched-budget endpoint verified: {updates} successful updates.",
                "successful_updates": updates,
                "artifact_path": str(fixed_budget_file),
                "sha256": compute_sha256(fixed_budget_file),
            }
        except Exception as exc:
            return {
                "status": "FAILED",
                "message": f"Failed to load or parse fixed_budget_checkpoint.pt: {exc}",
            }

    # 3. Check for last.pt without completed fixed_budget_checkpoint.pt
    last_file = p_dir / "last.pt"
    if last_file.is_file():
        try:
            import torch
            ck = torch.load(last_file, map_location="cpu", weights_only=False)
            meta = ck.get("metadata", {})

            if meta.get("guardrail_violation") is True:
                return {
                    "status": "HALTED_GUARDRAIL",
                    "message": "last.pt marked with guardrail violation.",
                }

            updates = ck.get("successful_updates", meta.get("successful_updates", 0))
            return {
                "status": "INCOMPLETE",
                "message": (
                    f"Training terminated before completing 3,600 updates ({updates} / 3600 updates). "
                    f"last.pt exists but fixed_budget_checkpoint.pt was not produced."
                ),
                "successful_updates": updates,
                "artifact_path": str(last_file),
                "sha256": compute_sha256(last_file),
            }
        except Exception as exc:
            return {
                "status": "FAILED",
                "message": f"Failed to parse last.pt: {exc}",
            }

    return {
        "status": "FAILED",
        "message": f"No valid checkpoint artifacts found in {drive_output_dir}.",
    }


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
    require_mount: bool = True,
) -> int:
    """Executes verified training workflow with full error capture and remote sync."""
    arm_norm = validate_arm(arm)

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

    # 4. Authentic Drive & Output Directory Construction (Single predictable path)
    verify_drive_authenticity(drive_root, require_mount=require_mount)
    drive_out_dir = resolve_drive_output_dir(drive_root, arm_norm, run_id=run_id)
    drive_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[STORAGE] Verified persistent output destination: {drive_out_dir}", flush=True)

    # 5. Build Execution Command
    config_file = repo_root / f"configs/rq1_trn_02b_{arm_norm.lower()}.yaml"
    if not config_file.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

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

    # 6. Post-Run Endpoint Verification & Rigorous Classification
    endpoint_info = classify_training_endpoint(drive_out_dir, arm_norm)
    status = endpoint_info["status"]
    msg = endpoint_info.get("message", "")
    print(f"[ENDPOINT AUDIT] Classification: {status} | Detail: {msg}", flush=True)

    if status == "COMPLETED_3600_UPDATES":
        print(f"[PRIMARY ENDPOINT] Verified: {endpoint_info['artifact_path']} (SHA: {endpoint_info.get('sha256', '')[:12]}...)", flush=True)
    elif status == "HALTED_GUARDRAIL":
        raise RuntimeError(f"FAIL-CLOSED: Training halted due to CTC PER guardrail violation: {msg}")
    elif status == "INCOMPLETE":
        raise RuntimeError(f"FAIL-CLOSED: Training ended before primary endpoint: {msg}")
    else:
        raise RuntimeError(f"FAIL-CLOSED: Training verification failed: {msg}")

    print("=" * 80)
    print(f"TRAINING EXECUTION COMPLETE & PRIMARY ENDPOINT VERIFIED FOR ARM {arm_norm}.")
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
    # Kernel execution compatibility: parse_known_args ignores Jupyter launcher arguments
    args, _ = ap.parse_known_args()

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
