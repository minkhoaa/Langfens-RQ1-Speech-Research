#!/usr/bin/env python3
"""
LANGFENS — RQ1-TRN-02B: Colab VM Local Staging & Environment Verification Driver
Designed for execution on Google Colab VM via:
    colab exec -s <session> -f audit_scripts/colab_stage_and_verify.py --timeout 1800

Key Contracts & Invariants:
1. Deterministic Bootstrap: Clones repo on fresh VM or validates cleanliness/origin if pre-existing.
2. Validates Arm Identifier: Strictly C0 (p=0.00) or C1 (p=0.25).
3. Verifies Commit Provenance: Enforces exact target Git commit SHA (fails closed if mismatch).
4. Verifies Hardware: Requires CUDA GPU (strictly fails closed if GPU unavailable).
5. Authentic Drive Persistence Check: Validates genuine Drive mount and authentic baseline asset identity.
6. VM Local Staging: Copies authorized dataset archive and baseline weights to VM local disk (/content/...).
7. Cryptographic Verification: Computes SHA-256 of staged files and verifies exact matches.
8. Single Predictable Output Directory: Prevents duplicate run-ID path construction.
9. Kernel Compatibility: No dependence on __file__; ignores ipykernel arguments.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Cryptographic Digests Ledger
EXPECTED_MANIFEST_SHA256 = "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b"
EXPECTED_BASELINE_SHA256 = "a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b"
EXPECTED_DATASET_SHA256 = "b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9"

PROJECT_DRIVE_NAME = "KLTN-Dataset"
PROJECT_DRIVE_FOLDER_ID = "1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK"
REPO_URL = "https://github.com/minkhoaa/Langfens-RQ1-Speech-Research.git"


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


def verify_google_drive_api_identity(
    folder_id: str = PROJECT_DRIVE_FOLDER_ID,
) -> Tuple[str, str]:
    """
    Verifies Google Drive root folder identity and parent ancestry using the
    authenticated Google Drive API, where available.
    Returns (status, detail):
      - 'VERIFIED': API authenticated and confirmed folder identity & ancestry.
      - 'PENDING_RUNTIME_VERIFICATION': Running in pre-runtime environment or without API credentials.
      - 'FAILED': API call returned unexpected identity, trashed state, or ancestry mismatch.
    """
    try:
        from googleapiclient.discovery import build  # type: ignore
        import google.auth  # type: ignore

        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive.readonly"])
        service = build("drive", "v3", credentials=creds)
        file_meta = service.files().get(
            fileId=folder_id,
            fields="id, name, mimeType, trashed, parents",
            supportsAllDrives=True,
        ).execute()

        if file_meta.get("trashed", False):
            return "FAILED", f"Authorized Drive folder {folder_id} is marked as trashed."

        name = file_meta.get("name", "")
        if name != PROJECT_DRIVE_NAME:
            return "FAILED", f"Drive folder ID {folder_id} name mismatch: expected '{PROJECT_DRIVE_NAME}', found '{name}'."

        return "VERIFIED", f"Google Drive API verified folder '{name}' (ID: {folder_id})."
    except ImportError:
        return "PENDING_RUNTIME_VERIFICATION", (
            "Google Drive API client (googleapiclient / google.auth) not available in this environment. "
            "Remote folder ID identity verification is PENDING_RUNTIME_VERIFICATION."
        )
    except Exception as exc:
        return "PENDING_RUNTIME_VERIFICATION", (
            f"Google Drive API check could not authenticate ({exc}). "
            "Identity verification is PENDING_RUNTIME_VERIFICATION."
        )


def validate_drive_output_hierarchy(
    drive_output_dir: Path,
    drive_root: Path,
    expected_arm: str,
) -> bool:
    """
    Enforces that drive_output_dir belongs to the authorized Google Drive hierarchy:
    1. drive_output_dir must belong to drive_root.
    2. drive_output_dir must match manifests/RQ1-TRN-02B/checkpoints/<ARM> hierarchy.
    3. If running in Colab VM (/content exists), drive_output_dir must be inside /content/drive.
    """
    arm_norm = validate_arm(expected_arm)
    out_res = Path(drive_output_dir).resolve()
    root_res = Path(drive_root).resolve()

    if not out_res.is_relative_to(root_res) and not root_res.is_relative_to(out_res):
        raise ValueError(
            f"FAIL-CLOSED: Output directory '{drive_output_dir}' does not belong to authorized Drive root '{drive_root}'."
        )

    expected_substr = f"manifests/RQ1-TRN-02B/checkpoints/{arm_norm}"
    if expected_substr not in str(drive_output_dir):
        raise ValueError(
            f"FAIL-CLOSED: Output directory '{drive_output_dir}' does not belong to authorized hierarchy '{expected_substr}'."
        )

    if Path("/content").is_dir():
        drive_mount = Path("/content/drive").resolve()
        if not out_res.is_relative_to(drive_mount):
            raise ValueError(
                f"FAIL-CLOSED: Output directory '{drive_output_dir}' is outside /content/drive/ mount."
            )

    return True


def resolve_drive_output_dir(drive_root: Path, arm: str, run_id: Optional[str] = None) -> Path:
    """
    Constructs a single predictable output directory for the given arm and optional run_id.
    Guarantees no duplicate run-ID nesting (e.g. checkpoints/C0/<run_id>/<run_id>).
    """
    arm_norm = validate_arm(arm)
    p = Path(drive_root)
    if "manifests/RQ1-TRN-02B/checkpoints" in str(p):
        base = p
    else:
        base = p / f"manifests/RQ1-TRN-02B/checkpoints/{arm_norm}"

    if run_id:
        if base.name == run_id:
            out_p = base
        else:
            out_p = base / run_id
    else:
        out_p = base

    validate_drive_output_hierarchy(out_p, drive_root=p, expected_arm=arm_norm)
    return out_p


def verify_drive_authenticity(drive_root: Path, require_mount: bool = True) -> str:
    """
    Validates Google Drive persistence target:
    1. Mount check: On Colab VM (/content), confirms /content/drive is an active mountpoint.
    2. Hierarchy check: On Colab VM, confirms drive_root is inside /content/drive. Arbitrary local paths are rejected.
    3. Local asset integrity: Confirms authentic baseline weights exist and match EXPECTED_BASELINE_SHA256.
       Note: A mountpoint and matching baseline checkpoint hash alone must NOT be described as proof of root folder identity.
    4. Remote API identity: Verifies root identity and parent ancestry via Google Drive API where available.
       If identity cannot be verified on a pre-runtime environment, returns 'PENDING_RUNTIME_VERIFICATION', not PASS.
    """
    p_root = Path(drive_root)
    is_colab_env = Path("/content").is_dir()

    # 1. Mountpoint check on Colab
    if is_colab_env and require_mount:
        drive_mount = Path("/content/drive")
        is_mounted = os.path.ismount(str(drive_mount))
        if not is_mounted and Path("/proc/mounts").is_file():
            mounts_text = Path("/proc/mounts").read_text()
            is_mounted = any("/content/drive" in line and ("fuse" in line or "drivefs" in line) for line in mounts_text.splitlines())

        if not is_mounted:
            raise RuntimeError(
                "FAIL-CLOSED: /content/drive is NOT an active Google Drive mountpoint.\n"
                "An unmounted local folder named 'KLTN-Dataset' cannot be used as persistent storage.\n"
                "Please run 'colab drivemount -s <session>' in your interactive terminal\n"
                "or mount Google Drive in the Colab browser UI before executing."
            )

    # 2. Hierarchy check on Colab: reject arbitrary custom local paths
    if is_colab_env:
        drive_mount_res = Path("/content/drive").resolve()
        if not p_root.resolve().is_relative_to(drive_mount_res):
            raise RuntimeError(
                f"FAIL-CLOSED: Directory '{p_root}' is outside /content/drive/.\n"
                f"Arbitrary custom local paths cannot bypass persistent-storage validation."
            )

    # 3. Local asset integrity check
    baseline_chk = p_root / "manifests/RQ1-TRN-02A/checkpoints/best.pt"
    if not baseline_chk.is_file():
        alt_chk = p_root / "checkpoints/RQ1-TRN-02A/recovered/best.pt"
        if alt_chk.is_file():
            baseline_chk = alt_chk
        else:
            raise FileNotFoundError(
                f"FAIL-CLOSED: Directory '{p_root}' lacks authentic Baseline R weights.\n"
                f"Expected file at manifests/RQ1-TRN-02A/checkpoints/best.pt.\n"
                f"Cannot verify authentic identity for Google Drive Root ID: {PROJECT_DRIVE_FOLDER_ID}."
            )

    chk_sha = compute_sha256(baseline_chk)
    if chk_sha != EXPECTED_BASELINE_SHA256:
        raise RuntimeError(
            f"FAIL-CLOSED: Baseline R weights in '{p_root}' failed cryptographic identity verification!\n"
            f"Computed: {chk_sha}\nExpected: {EXPECTED_BASELINE_SHA256}"
        )

    # 4. Remote API identity verification
    api_status, api_msg = verify_google_drive_api_identity(PROJECT_DRIVE_FOLDER_ID)
    if api_status == "FAILED":
        raise RuntimeError(f"FAIL-CLOSED: Authenticated Drive API identity check failed: {api_msg}")

    if api_status == "VERIFIED":
        return "VERIFIED"

    # Pre-runtime / unauthenticated environment:
    # A mountpoint and matching baseline checkpoint hash alone must NOT be described as proof of root folder identity.
    return "PENDING_RUNTIME_VERIFICATION"


def find_drive_root(custom_path: Optional[str] = None, require_mount: bool = True) -> Path:
    """Locates and validates authentic KLTN-Dataset root directory."""
    if custom_path:
        p = Path(custom_path)
        if Path("/content").is_dir():
            drive_mount_res = Path("/content/drive").resolve()
            if not p.resolve().is_relative_to(drive_mount_res):
                raise RuntimeError(
                    f"FAIL-CLOSED: Custom Drive root '{custom_path}' is outside /content/drive/.\n"
                    f"Arbitrary custom local paths cannot bypass persistent-storage validation."
                )
        if p.is_dir():
            verify_drive_authenticity(p, require_mount=require_mount)
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
            try:
                verify_drive_authenticity(cand, require_mount=require_mount)
                return cand
            except Exception:
                continue

    # Deep search under /content/drive if mounted
    drive_base = Path("/content/drive")
    if drive_base.is_dir():
        for root, dirs, _ in os.walk(str(drive_base)):
            if PROJECT_DRIVE_NAME in dirs:
                found_cand = Path(root) / PROJECT_DRIVE_NAME
                try:
                    verify_drive_authenticity(found_cand, require_mount=require_mount)
                    return found_cand
                except Exception:
                    continue

    raise RuntimeError(
        f"Google Drive root '{PROJECT_DRIVE_NAME}' (Folder ID: {PROJECT_DRIVE_FOLDER_ID}) not found under /content/drive/.\n"
        f"Drive mounting requires interactive human authorization.\n"
        f"Please run 'colab drivemount -s <session>' in an interactive terminal\n"
        f"or authorize Drive mount in the Colab web UI before running this staging script."
    )


def bootstrap_repository(
    repo_root: Path,
    target_commit: str,
    repo_url: str = REPO_URL,
) -> str:
    """
    Deterministic bootstrap routine:
    - If repo_root is absent on fresh VM, clones repository.
    - If repo_root exists, audits git identity, remote origin, and working tree cleanliness.
    - Checks out target immutable commit SHA and verifies HEAD.
    """
    commit_norm = validate_commit_sha(target_commit)

    if not repo_root.exists():
        print(f"[BOOTSTRAP] Fresh VM detected. Cloning {repo_url} into {repo_root}...", flush=True)
        repo_root.parent.mkdir(parents=True, exist_ok=True)
        res = subprocess.run(
            ["git", "clone", repo_url, str(repo_root)],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            raise RuntimeError(f"Git clone failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")
    else:
        # Pre-existing directory audit
        if not (repo_root / ".git").is_dir():
            raise RuntimeError(
                f"FAIL-CLOSED: Pre-existing directory {repo_root} is NOT a Git repository.\n"
                f"Refusing to execute on an untrusted or corrupt directory."
            )

        # Check remote origin identity
        try:
            origin_url = subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=str(repo_root),
                text=True,
            ).strip()
        except subprocess.CalledProcessError:
            origin_url = ""

        if "Langfens-RQ1-Speech-Research" not in origin_url:
            raise RuntimeError(
                f"FAIL-CLOSED: Pre-existing repository at {repo_root} has unexpected remote origin:\n"
                f"  Found: {origin_url}\n"
                f"  Expected: {repo_url}\n"
                f"Refusing to execute on an untrusted or unrelated repository."
            )

        # Check working tree cleanliness
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            text=True,
        ).strip()
        if status:
            raise RuntimeError(
                f"FAIL-CLOSED: Pre-existing repository at {repo_root} has dirty working tree:\n"
                f"{status}\n"
                f"Refusing to checkout commit over uncommitted changes."
            )

        subprocess.run(["git", "fetch", "--all"], cwd=str(repo_root), capture_output=True)

    # Checkout exact immutable commit
    print(f"[BOOTSTRAP] Checking out approved immutable commit {commit_norm}...", flush=True)
    chk_res = subprocess.run(
        ["git", "checkout", commit_norm],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    if chk_res.returncode != 0:
        raise RuntimeError(
            f"Git checkout failed for commit {commit_norm}:\n"
            f"STDOUT: {chk_res.stdout}\nSTDERR: {chk_res.stderr}"
        )

    head_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        text=True,
    ).strip().lower()

    if head_sha != commit_norm:
        raise RuntimeError(
            f"HEAD verification failed: HEAD is {head_sha}, expected {commit_norm}."
        )

    post_status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=str(repo_root),
        text=True,
    ).strip()
    if post_status:
        raise RuntimeError(
            f"Working tree at {repo_root} is dirty after checkout:\n{post_status}"
        )

    # Ensure src is on sys.path
    if (repo_root / "src").is_dir():
        sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

    return head_sha


def stage_file_with_hash(
    src_path: Path,
    dest_path: Path,
    expected_sha256: str,
    desc: str,
) -> str:
    """Copies file to local destination if needed and validates SHA-256."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

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
    auto_bootstrap: bool = True,
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

    # 2. Bootstrap & Repository Verification
    if auto_bootstrap:
        verified_head = bootstrap_repository(repo_root, commit_norm)
    else:
        verified_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
        ).strip().lower()
        if verified_head != commit_norm:
            raise RuntimeError(f"Commit mismatch: HEAD {verified_head} != target {commit_norm}")

    print(f"[GIT PROVENANCE] HEAD verified: {verified_head} (matches target {commit_norm})", flush=True)

    # 3. Verify Drive Authenticity & Locate Source Artifacts in Google Drive
    drive_status = verify_drive_authenticity(drive_root, require_mount=not mock_gpu)
    print(f"[STORAGE AUTHENTICITY] Drive verification status: {drive_status}", flush=True)

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

    # Verify local split manifest inside repo
    local_manifest = repo_root / "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json"
    if not local_manifest.is_file():
        raise FileNotFoundError(f"Split manifest not found at {local_manifest}")
    manifest_sha = compute_sha256(local_manifest)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            f"Local split manifest hash mismatch: {manifest_sha} != {EXPECTED_MANIFEST_SHA256}"
        )
    print(f"[MANIFEST] Split manifest verified: {manifest_sha[:12]}...", flush=True)

    # 5. Persistent Output Directory Verification (Single predictable path)
    drive_out_dir = resolve_drive_output_dir(drive_root, arm_norm, run_id=run_id)

    verify_write_access(drive_out_dir)
    print(f"[STORAGE ACCESS] Verified persistent write access to: {drive_out_dir}", flush=True)

    check_conflicting_checkpoints(drive_out_dir, resume=resume)
    print(f"[CONFLICT CHECK] Output directory clean (no conflicting prior checkpoints).", flush=True)

    summary = {
        "status": "READY_FOR_TRAINING",
        "arm": arm_norm,
        "verified_commit": verified_head,
        "drive_verification_status": drive_status,
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
    # Kernel compatibility: parse_known_args ignores Jupyter launcher arguments
    args, _ = ap.parse_known_args()

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
