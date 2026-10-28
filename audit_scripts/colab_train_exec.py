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
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Kernel execution compatibility: avoid relying on __file__
_repo_env = os.environ.get("REPO_ROOT", "/content/repo")
_repo_path = Path(_repo_env)
if (_repo_path / "src").is_dir():
    sys.path.insert(0, str(_repo_path / "src"))
if _repo_path.is_dir():
    sys.path.insert(0, str(_repo_path))

EXPECTED_MANIFEST_SHA256 = "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b"
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


def verify_google_drive_api_identity(
    folder_id: str = PROJECT_DRIVE_FOLDER_ID,
    drive_root_path: Optional[Path] = None,
) -> Tuple[str, str]:
    """
    Verifies Google Drive root folder identity, expected MIME type, parent ancestry,
    and establishes mount-to-folder binding where available.
    Returns (status, detail):
      - 'VERIFIED': API authenticated, folder identity, MIME type, ancestry verified,
                    and mount-to-folder binding confirmed.
      - 'BLOCKED_IDENTITY_MISMATCH': API returned unexpected folder identity, MIME type, or trashed folder.
      - 'BLOCKED_UNBOUND_MOUNT': API authenticated but cannot bind local mount to remote folder.
      - 'PENDING_HUMAN_RUNTIME_VERIFICATION': Running in pre-runtime environment or without API credentials.
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
            return "BLOCKED_IDENTITY_MISMATCH", f"Authorized Drive folder {folder_id} is marked as trashed."

        name = file_meta.get("name", "")
        if name != PROJECT_DRIVE_NAME:
            return "BLOCKED_IDENTITY_MISMATCH", f"Drive folder ID {folder_id} name mismatch: expected '{PROJECT_DRIVE_NAME}', found '{name}'."

        mime = file_meta.get("mimeType", "")
        if mime != "application/vnd.google-apps.folder":
            return "BLOCKED_IDENTITY_MISMATCH", f"Drive item ID {folder_id} is not a folder (mimeType='{mime}')."

        parents = file_meta.get("parents", [])
        if not isinstance(parents, list):
            return "BLOCKED_IDENTITY_MISMATCH", f"Drive folder ID {folder_id} ancestry is invalid or missing."

        # Establish mount-to-folder binding if local path provided
        if drive_root_path is not None:
            p_root = Path(drive_root_path).resolve()
            children_res = service.files().list(
                q=f"'{folder_id}' in parents and trashed=false",
                fields="files(id, name, mimeType)",
                supportsAllDrives=True,
            ).execute()
            remote_child_names = {f["name"] for f in children_res.get("files", [])}

            expected_children = {"manifests"}
            if not expected_children.issubset(remote_child_names):
                return "BLOCKED_UNBOUND_MOUNT", (
                    f"Remote folder {folder_id} does not contain expected children {expected_children}. "
                    f"Found: {remote_child_names}."
                )
            for child in expected_children:
                if not (p_root / child).exists():
                    return "BLOCKED_UNBOUND_MOUNT", (
                        f"Local mount path '{p_root}' does not contain expected child '{child}'. "
                        f"Mount-to-folder binding failed."
                    )

            return "VERIFIED", f"Google Drive API verified folder '{name}' (ID: {folder_id}) and confirmed mount binding."
        else:
            return "VERIFIED", f"Google Drive API verified folder '{name}' (ID: {folder_id})."

    except ImportError:
        human_proc = (
            "Google Drive API client (googleapiclient / google.auth) not available in this environment. "
            "FAIL-CLOSED: Mount-to-folder binding cannot be established automatically. "
            "Human Verification Procedure: "
            "1. Run 'colab drivemount -s <session>' in an interactive terminal. "
            "2. Confirm Google account mounted at /content/drive owns folder ID '1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK'. "
            "3. Verify local path /content/drive/MyDrive/KLTN-Dataset contains manifests/RQ1-TRN-02A/checkpoints/best.pt."
        )
        return "PENDING_HUMAN_RUNTIME_VERIFICATION", human_proc
    except Exception as exc:
        human_proc = (
            f"Google Drive API check could not authenticate ({exc}). "
            "FAIL-CLOSED: Mount-to-folder binding cannot be established automatically. "
            "Human Verification Procedure: "
            "1. Run 'colab drivemount -s <session>' in an interactive terminal. "
            "2. Confirm Google account mounted at /content/drive owns folder ID '1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK'. "
            "3. Verify local path /content/drive/MyDrive/KLTN-Dataset contains manifests/RQ1-TRN-02A/checkpoints/best.pt."
        )
        return "PENDING_HUMAN_RUNTIME_VERIFICATION", human_proc


def validate_drive_output_hierarchy(
    drive_output_dir: Path | str,
    drive_root: Path | str,
    expected_arm: str,
    expected_run_id: Optional[str] = None,
) -> bool:
    """
    Enforces strict descendant containment and exact relative output components:
    1. Strict descendant containment: resolved_output.is_relative_to(resolved_root).
       Does NOT accept reverse ancestry or equality (resolved_root.is_relative_to(resolved_output) prohibited).
    2. Check exact relative output components:
       manifests/RQ1-TRN-02B/checkpoints/{C0|C1} OR manifests/RQ1-TRN-02B/checkpoints/{C0|C1}/[run_id]
    3. Reject path traversal, symlink escapes, and arbitrary custom roots.
    4. On Colab VM (/content exists), both drive_root and drive_output_dir must reside within /content/drive.
    """
    arm_norm = validate_arm(expected_arm)
    out_res = Path(drive_output_dir).resolve()
    root_res = Path(drive_root).resolve()

    # 1. Strict descendant containment (reject equality and reverse ancestry)
    if not out_res.is_relative_to(root_res) or out_res == root_res:
        raise ValueError(
            f"FAIL-CLOSED: Output directory '{drive_output_dir}' (resolved: '{out_res}') "
            f"is NOT a strict descendant of authorized Drive root '{drive_root}' (resolved: '{root_res}'). "
            f"Reverse ancestry or identical path is strictly prohibited."
        )

    # 2. Check exact relative components
    rel_path = out_res.relative_to(root_res)
    parts = rel_path.parts
    expected_prefix = ("manifests", "RQ1-TRN-02B", "checkpoints", arm_norm)

    if len(parts) == 4:
        if parts != expected_prefix:
            raise ValueError(
                f"FAIL-CLOSED: Output path '{rel_path}' parts do not match exact hierarchy: {expected_prefix}."
            )
    elif len(parts) == 5:
        if parts[:4] != expected_prefix:
            raise ValueError(
                f"FAIL-CLOSED: Output path prefix '{parts[:4]}' does not match expected hierarchy: {expected_prefix}."
            )
        run_comp = parts[4]
        # Reject path traversal or illegal characters in run_id
        if not re.match(r"^[A-Za-z0-9_-]+$", run_comp) or run_comp in (".", ".."):
            raise ValueError(
                f"FAIL-CLOSED: Illegal run ID component '{run_comp}' in output path '{rel_path}'."
            )
        if expected_run_id and run_comp != expected_run_id:
            raise ValueError(
                f"FAIL-CLOSED: Run ID in output path '{run_comp}' does not match expected run ID '{expected_run_id}'."
            )
    else:
        raise ValueError(
            f"FAIL-CLOSED: Output path '{rel_path}' has {len(parts)} components; "
            f"strictly 4 ('manifests/RQ1-TRN-02B/checkpoints/{arm_norm}') or "
            f"5 ('manifests/RQ1-TRN-02B/checkpoints/{arm_norm}/<run_id>') required."
        )

    # 3. Colab VM containment
    if Path("/content").is_dir():
        drive_mount = Path("/content/drive").resolve()
        if not root_res.is_relative_to(drive_mount):
            raise ValueError(
                f"FAIL-CLOSED: Authorized Drive root '{root_res}' is outside /content/drive/ mount."
            )
        if not out_res.is_relative_to(drive_mount):
            raise ValueError(
                f"FAIL-CLOSED: Output directory '{out_res}' is outside /content/drive/ mount."
            )

    return True


def resolve_drive_output_dir(drive_root: Path | str, arm: str, run_id: Optional[str] = None) -> Path:
    """Constructs single predictable output directory without duplicate run-ID nesting."""
    arm_norm = validate_arm(arm)
    p = Path(drive_root).resolve()

    base = p / "manifests" / "RQ1-TRN-02B" / "checkpoints" / arm_norm
    if run_id:
        out_p = base / run_id
    else:
        out_p = base

    validate_drive_output_hierarchy(out_p, drive_root=p, expected_arm=arm_norm, expected_run_id=run_id)
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
                "Please run 'colab drivemount -s <session>' before executing training."
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
    api_status, api_msg = verify_google_drive_api_identity(PROJECT_DRIVE_FOLDER_ID, drive_root_path=p_root)
    if api_status in ("BLOCKED_IDENTITY_MISMATCH", "BLOCKED_UNBOUND_MOUNT"):
        raise RuntimeError(f"FAIL-CLOSED: Drive API identity/binding check failed: {api_msg}")

    if api_status == "VERIFIED":
        return "VERIFIED"

    # Pre-runtime / unauthenticated environment or pending human verification:
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
    expected_commit: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Rigorously verifies and classifies the training run endpoint:
    - COMPLETED_3600_UPDATES: Primary matched-budget endpoint reached and verified.
      Requires ALL of:
      1. successful_updates == 3600 (top-level)
      2. metadata["successful_updates"] == 3600
      3. metadata["primary_3600_updates_completed"] is True
      4. metadata["primary_endpoint_reached"] is True
      5. metadata["endpoint_status"] == "COMPLETED_3600_UPDATES"
      6. metadata["guardrail_violation"] is False
      7. metadata["arm"] matches expected_arm
      8. Critical provenance fields match approved run contract:
         - split_sha256 == EXPECTED_MANIFEST_SHA256
         - initial_checkpoint_sha256 == EXPECTED_BASELINE_SHA256
         - git_sha is valid 40-char hex (matches expected_commit if supplied)
      9. Checkpoint payload contains required training and optimizer state:
         - 'model' dict (non-empty)
         - 'optimizer' dict (non-empty)
         - 'scheduler' dict
         - 'scaler' dict
    - INCOMPLETE: Terminated early (epoch cap / AMP skips) without guardrail violation.
    - HALTED_GUARDRAIL: Stopped due to CTC PER degradation guardrail trip.
    - FAILED: Checkpoints missing, corrupted, contradictory metadata, or process failed.
    """
    arm_norm = validate_arm(expected_arm)
    p_dir = Path(drive_output_dir)

    # 1. Check for explicit guardrail trip artifact
    halted_file = p_dir / "halted_guardrail.pt"
    if halted_file.is_file():
        try:
            import torch
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
            import torch
            ck = torch.load(fixed_budget_file, map_location="cpu", weights_only=False)
            if not isinstance(ck, dict):
                return {
                    "status": "FAILED",
                    "message": "fixed_budget_checkpoint payload is not a valid dictionary.",
                    "artifact_path": str(fixed_budget_file),
                }

            # Check 9: Required training and optimizer state
            for req_key in ("model", "optimizer", "scheduler", "scaler", "metadata"):
                if req_key not in ck:
                    return {
                        "status": "FAILED",
                        "message": f"Corrupt or incomplete checkpoint payload: missing required key '{req_key}'.",
                        "artifact_path": str(fixed_budget_file),
                    }

            if not isinstance(ck["model"], dict) or len(ck["model"]) == 0:
                return {
                    "status": "FAILED",
                    "message": "Corrupt or incomplete checkpoint payload: 'model' state dict is missing or empty.",
                    "artifact_path": str(fixed_budget_file),
                }
            if not isinstance(ck["optimizer"], dict) or len(ck["optimizer"]) == 0:
                return {
                    "status": "FAILED",
                    "message": "Corrupt or incomplete checkpoint payload: 'optimizer' state dict is missing or empty.",
                    "artifact_path": str(fixed_budget_file),
                }
            if not isinstance(ck["scheduler"], dict):
                return {
                    "status": "FAILED",
                    "message": "Corrupt or incomplete checkpoint payload: 'scheduler' state dict must be a dict.",
                    "artifact_path": str(fixed_budget_file),
                }
            if not isinstance(ck["scaler"], dict):
                return {
                    "status": "FAILED",
                    "message": "Corrupt or incomplete checkpoint payload: 'scaler' state dict must be a dict.",
                    "artifact_path": str(fixed_budget_file),
                }

            meta = ck.get("metadata")
            if not isinstance(meta, dict):
                return {
                    "status": "FAILED",
                    "message": "Corrupt checkpoint payload: 'metadata' is missing or not a dictionary.",
                    "artifact_path": str(fixed_budget_file),
                }

            # Check 6: Guardrail violation check
            if meta.get("guardrail_violation") is True:
                return {
                    "status": "HALTED_GUARDRAIL",
                    "message": "fixed_budget_checkpoint marked with guardrail violation in metadata.",
                    "artifact_path": str(fixed_budget_file),
                }
            if meta.get("guardrail_violation") is not False:
                return {
                    "status": "FAILED",
                    "message": "Missing mandatory metadata: 'guardrail_violation' must be explicitly False.",
                    "artifact_path": str(fixed_budget_file),
                }

            # Check 7: Arm matching
            actual_arm = meta.get("arm")
            if actual_arm != arm_norm:
                return {
                    "status": "FAILED",
                    "message": f"Arm mismatch in fixed_budget_checkpoint: declared '{actual_arm}', expected '{arm_norm}'.",
                    "artifact_path": str(fixed_budget_file),
                }

            # Check 1, 2, 3, 4, 5: Successful updates and completion metadata
            top_updates = ck.get("successful_updates")
            meta_updates = meta.get("successful_updates")
            p3600 = meta.get("primary_3600_updates_completed")
            p_endpoint = meta.get("primary_endpoint_reached")
            endpoint_status = meta.get("endpoint_status")

            # Check for contradictory completion flags with non-3600 updates
            if top_updates != 3600 or meta_updates != 3600:
                if p3600 is True or p_endpoint is True or endpoint_status == "COMPLETED_3600_UPDATES":
                    return {
                        "status": "FAILED",
                        "message": (
                            f"Contradictory metadata: successful_updates is top={top_updates}, meta={meta_updates} "
                            f"but completion flags indicate completion (p3600={p3600}, p_endpoint={p_endpoint}, endpoint_status='{endpoint_status}')."
                        ),
                        "successful_updates": top_updates,
                        "artifact_path": str(fixed_budget_file),
                    }
                else:
                    return {
                        "status": "INCOMPLETE",
                        "message": f"fixed_budget_checkpoint has only {top_updates} / 3600 updates.",
                        "successful_updates": top_updates,
                        "artifact_path": str(fixed_budget_file),
                    }

            # Top and meta updates both equal 3600; check required completion flags
            if p3600 is not True:
                return {
                    "status": "FAILED",
                    "message": f"Contradictory metadata: successful_updates is 3600 but primary_3600_updates_completed is {p3600}.",
                    "successful_updates": 3600,
                    "artifact_path": str(fixed_budget_file),
                }
            if p_endpoint is not True:
                return {
                    "status": "FAILED",
                    "message": f"Contradictory metadata: successful_updates is 3600 but primary_endpoint_reached is {p_endpoint}.",
                    "successful_updates": 3600,
                    "artifact_path": str(fixed_budget_file),
                }
            if endpoint_status != "COMPLETED_3600_UPDATES":
                return {
                    "status": "FAILED",
                    "message": f"Missing mandatory metadata: 'endpoint_status' is '{endpoint_status}', expected 'COMPLETED_3600_UPDATES'.",
                    "successful_updates": 3600,
                    "artifact_path": str(fixed_budget_file),
                }

            # Check 8: Critical provenance fields match approved run contract
            split_sha = meta.get("split_sha256")
            if not split_sha or split_sha.strip().lower() != EXPECTED_MANIFEST_SHA256:
                return {
                    "status": "FAILED",
                    "message": f"Provenance mismatch: split_sha256 '{split_sha}' != expected '{EXPECTED_MANIFEST_SHA256}'.",
                    "artifact_path": str(fixed_budget_file),
                }

            init_sha = meta.get("initial_checkpoint_sha256")
            if not init_sha or init_sha.strip().lower() != EXPECTED_BASELINE_SHA256:
                return {
                    "status": "FAILED",
                    "message": f"Provenance mismatch: initial_checkpoint_sha256 '{init_sha}' != expected '{EXPECTED_BASELINE_SHA256}'.",
                    "artifact_path": str(fixed_budget_file),
                }

            git_sha = meta.get("git_sha")
            if not git_sha or len(git_sha.strip()) != 40 or not all(c in "0123456789abcdefABCDEF" for c in git_sha.strip()):
                return {
                    "status": "FAILED",
                    "message": f"Provenance mismatch: invalid or missing git_sha '{git_sha}' in metadata.",
                    "artifact_path": str(fixed_budget_file),
                }
            if expected_commit and git_sha.strip().lower() != expected_commit.strip().lower():
                return {
                    "status": "FAILED",
                    "message": f"Provenance mismatch: git_sha '{git_sha}' != expected '{expected_commit}'.",
                    "artifact_path": str(fixed_budget_file),
                }

            # ALL MANDATORY INVARIANTS SATISFIED!
            return {
                "status": "COMPLETED_3600_UPDATES",
                "message": f"Primary matched-budget endpoint verified: 3600 successful updates.",
                "successful_updates": 3600,
                "artifact_path": str(fixed_budget_file),
                "sha256": compute_sha256(fixed_budget_file),
            }
        except Exception as exc:
            return {
                "status": "FAILED",
                "message": f"Failed to load or parse fixed_budget_checkpoint.pt: {exc}",
                "artifact_path": str(fixed_budget_file),
            }

    # 3. Check for last.pt without completed fixed_budget_checkpoint.pt
    last_file = p_dir / "last.pt"
    if last_file.is_file():
        try:
            import torch
            ck = torch.load(last_file, map_location="cpu", weights_only=False)
            meta = ck.get("metadata", {}) if isinstance(ck, dict) else {}

            if meta.get("guardrail_violation") is True:
                return {
                    "status": "HALTED_GUARDRAIL",
                    "message": "last.pt marked with guardrail violation.",
                    "artifact_path": str(last_file),
                }

            updates = ck.get("successful_updates", meta.get("successful_updates", 0)) if isinstance(ck, dict) else 0

            # Reject contradictory metadata in last.pt
            if updates < 3600 and meta.get("primary_3600_updates_completed") is True:
                return {
                    "status": "FAILED",
                    "message": f"Contradictory metadata in last.pt: successful_updates is {updates} but primary_3600_updates_completed is True.",
                    "successful_updates": updates,
                    "artifact_path": str(last_file),
                }

            return {
                "status": "INCOMPLETE",
                "message": (
                    f"Training terminated before producing verified fixed_budget_checkpoint.pt ({updates} / 3600 updates). "
                    f"last.pt exists but primary matched-budget endpoint artifact was not verified."
                ),
                "successful_updates": updates,
                "artifact_path": str(last_file),
                "sha256": compute_sha256(last_file),
            }
        except Exception as exc:
            return {
                "status": "FAILED",
                "message": f"Failed to parse last.pt: {exc}",
                "artifact_path": str(last_file),
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

    # 1. Strict GPU Check: strictly required; cannot be bypassed
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
    drive_status = verify_drive_authenticity(drive_root, require_mount=require_mount)
    print(f"[STORAGE AUTHENTICITY] Drive verification status: {drive_status}", flush=True)

    # Section A Invariant: Require drive_verification_status == "VERIFIED" before launching training
    if drive_status != "VERIFIED":
        raise RuntimeError(
            f"FAIL-CLOSED: Google Drive identity verification status is '{drive_status}', NOT 'VERIFIED'.\n"
            f"Training launch is strictly BLOCKED until Drive identity (Folder ID: {PROJECT_DRIVE_FOLDER_ID}) "
            f"is authenticated and verified via Google Drive API.\n"
            f"Under no circumstances may training launch when storage identity is pending or unverified."
        )

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
    endpoint_info = classify_training_endpoint(drive_out_dir, arm_norm, expected_commit=target_commit)
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
    # Kernel execution compatibility: parse_known_args ignores Jupyter launcher arguments
    args, unknown = ap.parse_known_args()
    # Reject unknown CLI flags (only allowing ipykernel connection arguments)
    disallowed_unknown = [
        u for u in unknown
        if not (u.startswith("-f") or u.endswith(".json") or "kernel" in u or "ipykernel" in u)
    ]
    if disallowed_unknown:
        ap.error(f"unrecognized arguments: {' '.join(disallowed_unknown)}")

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
    )


if __name__ == "__main__":
    main()
