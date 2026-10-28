#!/usr/bin/env python3
"""
LANGFENS — RQ1-TRN-02B: Colab Launch Workflow Orchestrator & Multi-Layer Readiness Validator
Local tool to validate preflight readiness, check compute quota, verify Git state,
and generate verified Colab CLI execution sequences for Control C0 and Treatment C1.

Readiness Layer Separation:
1. Static Software Validation (PASS_SOFTWARE): Validates code, git tree, configs, manifests.
   * PASS_SOFTWARE does NOT authorize GPU execution.
2. Quota Readiness: Read-only query of 'colab usage' (requires >= 15.0 units).
3. Authenticated Drive Mount Readiness: Operator interactive terminal execution of 'colab drivemount'.
4. Real VM Staging / Storage Write Readiness: Verified execution of staging on Colab VM disk.
5. Final GPU Training Authorization: Formal operator gate sign-off.

Key Guarantees:
- Fail-Closed: Executable readiness check exits with non-zero status if any prerequisite is blocked.
- Non-Interactive CLI Syntax: Uses 'colab exec -s SESSION -f SCRIPT --env K=V --timeout SEC'.
- Strict Fallback Prohibition: Never executes training on local CPU/GPU.
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
from typing import Dict, List, Optional, Tuple

REPO_URL = "https://github.com/minkhoaa/Langfens-RQ1-Speech-Research.git"
DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/KLTN-Dataset"
PROJECT_DRIVE_FOLDER_ID = "1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK"
REQUIRED_COMPUTE_UNITS_MIN = 15.0
EXPECTED_MANIFEST_SHA256 = "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b"
EXPECTED_BASELINE_SHA256 = "a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b"
EXPECTED_DATASET_SHA256 = "b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9"


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
    """Strictly validates arm identifier."""
    norm = arm.strip().upper()
    if norm not in ("C0", "C1"):
        raise ValueError(f"Invalid arm '{arm}'. Strictly 'C0' or 'C1' allowed.")
    return norm


def validate_commit_sha(sha: str) -> str:
    """Strictly validates 40-character hexadecimal Git commit SHA."""
    cleaned = sha.strip().lower()
    if len(cleaned) != 40 or not all(c in "0123456789abcdef" for c in cleaned):
        raise ValueError(f"Invalid Git commit SHA '{sha}'. Expected 40-character hex string.")
    return cleaned


def parse_colab_usage(raw_output: str) -> Dict[str, float | int]:
    """Parses output of 'colab usage' into structured numbers."""
    balance_match = re.search(r"Current balance:\s*([0-9.]+)\s*compute units", raw_output, re.IGNORECASE)
    usage_rate_match = re.search(r"Usage rate:\s*([0-9.]+)/hr", raw_output, re.IGNORECASE)
    assignments_match = re.search(r"Active assignments:\s*([0-9]+)", raw_output, re.IGNORECASE)

    balance = float(balance_match.group(1)) if balance_match else 0.0
    usage_rate = float(usage_rate_match.group(1)) if usage_rate_match else 0.0
    assignments = int(assignments_match.group(1)) if assignments_match else 0

    return {
        "balance": balance,
        "usage_rate": usage_rate,
        "active_assignments": assignments,
    }


def check_quota_status(balance: float, min_required: float = REQUIRED_COMPUTE_UNITS_MIN) -> Tuple[bool, str]:
    """Determines whether Colab compute units are sufficient."""
    if balance >= min_required:
        return True, f"SUFFICIENT ({balance:.2f} units >= {min_required:.1f} required)"
    if balance > 0.0:
        return False, f"LOW_QUOTA ({balance:.2f} units < {min_required:.1f} required; risk of mid-run termination)"
    return False, f"ZERO_QUOTA ({balance:.2f} units; cannot provision GPU runtime)"


def get_current_git_head() -> str:
    """Retrieves current local Git commit SHA."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip().lower()


def check_git_clean() -> bool:
    """Checks if git working directory is clean."""
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    return len(status) == 0


def validate_static_software(commit_sha: str, repo_root: Optional[Path] = None) -> Tuple[bool, str]:
    """
    Validates local static software:
    1. Working tree cleanliness.
    2. Actual Git HEAD equals the requested commit SHA (rejects wrong commits even if tree is clean).
    3. Required configurations exist.
    4. Frozen split manifest SHA-256 matches the ledger.
    """
    root = repo_root or Path(".")

    # 1. Check working directory cleanliness
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            text=True,
        ).strip()
        if len(status) > 0:
            return False, f"FAIL: Git working directory is not clean:\n{status}"
    except Exception as exc:
        return False, f"FAIL: Failed to check git status: {exc}"

    # 2. Check actual Git HEAD equals requested commit SHA
    try:
        actual_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            text=True,
        ).strip().lower()
        requested = commit_sha.strip().lower()
        if actual_head != requested:
            return False, f"FAIL: Git HEAD mismatch: actual {actual_head} != requested {requested}"
    except Exception as exc:
        return False, f"FAIL: Failed to resolve Git HEAD: {exc}"

    # 3. Check required configurations exist
    for arm in ("c0", "c1"):
        cfg_file = root / f"configs/rq1_trn_02b_{arm}.yaml"
        if not cfg_file.is_file():
            return False, f"FAIL: Config file missing: {cfg_file}"

    # 4. Check split manifest existence and cryptographic SHA-256 against ledger
    manifest_file = root / "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json"
    if not manifest_file.is_file():
        return False, f"FAIL: Split manifest missing: {manifest_file}"

    manifest_sha = compute_sha256(manifest_file)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        return False, (
            f"FAIL: Split manifest SHA-256 mismatch!\n"
            f"Computed: {manifest_sha}\nExpected: {EXPECTED_MANIFEST_SHA256}"
        )

    return True, "PASS_SOFTWARE (Note: PASS_SOFTWARE does not authorize GPU execution)"


def generate_colab_arm_plan(
    arm: str,
    commit_sha: str,
    session_name: Optional[str] = None,
    run_id: Optional[str] = None,
) -> List[str]:
    """Generates the verified non-interactive Colab CLI command sequence for one arm."""
    arm_norm = validate_arm(arm)
    sha_norm = validate_commit_sha(commit_sha)
    session = session_name or f"langfens-rq1-{arm_norm.lower()}"

    env_flags = f"--env ARM={arm_norm} --env TARGET_COMMIT={sha_norm}"
    if run_id:
        env_flags += f" --env RUN_ID={run_id}"

    commands = [
        f"# --- Phase 1: Provision fresh Colab T4 GPU Session ---",
        f"colab new -s {session} --gpu T4",
        f"",
        f"# --- Phase 2: Operator Google Drive Mount (Interactive Prerequisite) ---",
        f"# Note: colab drivemount requires terminal interaction. Run once per session:",
        f"colab drivemount -s {session}",
        f"",
        f"# --- Phase 3: Install Required Dependencies on Colab VM ---",
        f"colab install -s {session} -r requirements.txt",
        f"",
        f"# --- Phase 4: Deterministic Bootstrap, Staging & Preflight Verification ---",
        f"# Automatically clones repo on fresh VM, verifies commit, stages data to VM disk, validates SHA-256:",
        f"colab exec -s {session} -f audit_scripts/colab_stage_and_verify.py {env_flags} --timeout 1800",
        f"",
        f"# --- Phase 5: Execute Matched-Budget Training & Primary Endpoint Verification (Timeout: 8 hours) ---",
        f"colab exec -s {session} -f audit_scripts/colab_train_exec.py {env_flags} --timeout 28800",
        f"",
        f"# --- Phase 6: Release Colab VM Runtime Promptly ---",
        f"colab stop -s {session}",
    ]
    return commands


def verify_mount_attestation(
    attestation_path: Path | str,
    expected_session: str,
) -> Tuple[bool, str]:
    """
    Verifies authentic Google Drive mount attestation JSON artifact bound to expected session.
    """
    p = Path(attestation_path)
    if not p.is_file():
        return False, f"Mount attestation artifact not found: {p}"
    try:
        data = json.loads(p.read_text())
    except Exception as exc:
        return False, f"Failed to parse mount attestation JSON: {exc}"

    if not isinstance(data, dict):
        return False, "Mount attestation must be a JSON dictionary."

    session = data.get("session")
    if not session or session != expected_session:
        return False, f"Mismatched session ID in mount attestation: found '{session}', expected '{expected_session}'."

    if data.get("mount_active") is not True:
        return False, "Mount attestation declares mount_active is not True."

    folder_id = data.get("drive_folder_id")
    if folder_id != PROJECT_DRIVE_FOLDER_ID:
        return False, f"Mismatched Google Drive folder ID: found '{folder_id}', expected '{PROJECT_DRIVE_FOLDER_ID}'."

    drive_status = data.get("drive_verification_status")
    if drive_status != "VERIFIED":
        return False, f"Drive verification status in attestation is '{drive_status}', expected 'VERIFIED'."

    return True, f"Google Drive mount and genuine identity (ID: {PROJECT_DRIVE_FOLDER_ID}) verified for session '{expected_session}'."


def verify_staging_report(
    staging_report_path: Path | str,
    expected_session: str,
    expected_commit: str,
    expected_arm: str,
    expected_run_id: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Verifies authentic VM staging report JSON artifact bound to session, run ID, and immutable Git SHA.
    """
    p = Path(staging_report_path)
    if not p.is_file():
        return False, f"Staging report artifact not found: {p}"
    try:
        data = json.loads(p.read_text())
    except Exception as exc:
        return False, f"Failed to parse staging report JSON: {exc}"

    if not isinstance(data, dict):
        return False, "Staging report must be a JSON dictionary."

    if data.get("status") != "READY_FOR_TRAINING":
        return False, f"Staging report status is '{data.get('status')}', expected 'READY_FOR_TRAINING'."

    if data.get("drive_verification_status") != "VERIFIED":
        return False, f"Drive verification status in staging report is '{data.get('drive_verification_status')}', expected 'VERIFIED'."

    session = data.get("session")
    if not session or session != expected_session:
        return False, f"Mismatched session ID in staging report: found '{session}', expected '{expected_session}'."

    commit = data.get("verified_commit")
    if not commit or commit.strip().lower() != expected_commit.strip().lower():
        return False, f"Mismatched commit SHA in staging report: found '{commit}', expected '{expected_commit}'."

    arm = data.get("arm")
    if expected_arm != "both":
        arm_norm = validate_arm(expected_arm)
        if arm != arm_norm:
            return False, f"Mismatched arm in staging report: found '{arm}', expected '{arm_norm}'."
    else:
        if arm not in ("C0", "C1"):
            return False, f"Invalid arm in staging report: found '{arm}'."

    if expected_run_id:
        run_id = data.get("run_id")
        if run_id != expected_run_id:
            return False, f"Mismatched run ID in staging report: found '{run_id}', expected '{expected_run_id}'."

    if data.get("staged_zip_sha256") != EXPECTED_DATASET_SHA256:
        return False, "Staged dataset archive SHA-256 in report does not match ledger."

    if data.get("staged_baseline_sha256") != EXPECTED_BASELINE_SHA256:
        return False, "Staged baseline checkpoint SHA-256 in report does not match ledger."

    if data.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        return False, "Split manifest SHA-256 in report does not match ledger."

    return True, f"Verified staging report bound to session '{expected_session}', commit '{expected_commit[:10]}...', and arm '{arm}'."


def verify_operator_approval(
    approval_path: Path | str,
    expected_session: str,
    expected_commit: str,
    expected_arm: str,
    expected_run_id: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Verifies explicit operator authorization JSON record bound independently of command-line flags.
    """
    p = Path(approval_path)
    if not p.is_file():
        return False, f"Operator authorization record not found: {p}"
    try:
        data = json.loads(p.read_text())
    except Exception as exc:
        return False, f"Failed to parse operator authorization JSON: {exc}"

    if not isinstance(data, dict):
        return False, "Operator authorization record must be a JSON dictionary."

    if data.get("status") not in ("AUTHORIZED", "APPROVED"):
        return False, f"Operator authorization status is '{data.get('status')}', expected 'AUTHORIZED' or 'APPROVED'."

    commit = data.get("approved_commit")
    if not commit or commit.strip().lower() != expected_commit.strip().lower():
        return False, f"Mismatched commit SHA in operator approval: found '{commit}', expected '{expected_commit}'."

    session = data.get("approved_session")
    if not session or session != expected_session:
        return False, f"Mismatched session in operator approval: found '{session}', expected '{expected_session}'."

    approved_arm = data.get("approved_arm")
    if expected_arm != "both":
        arm_norm = validate_arm(expected_arm)
        if approved_arm not in (arm_norm, "both"):
            return False, f"Mismatched arm in operator approval: found '{approved_arm}', expected '{arm_norm}'."

    if expected_run_id:
        run_id = data.get("approved_run_id")
        if run_id != expected_run_id:
            return False, f"Mismatched run ID in operator approval: found '{run_id}', expected '{expected_run_id}'."

    op_id = data.get("operator_id") or data.get("approved_by")
    if not op_id or not str(op_id).strip():
        return False, "Operator authorization record missing operator identifier ('operator_id' or 'approved_by')."

    timestamp = data.get("timestamp")
    if not timestamp or not str(timestamp).strip():
        return False, "Operator authorization record missing 'timestamp'."

    return True, f"Operator authorization confirmed by '{op_id}' for session '{expected_session}' at {timestamp}."


def evaluate_readiness_layers(
    commit_sha: str,
    session_name: str = "langfens-rq1",
    arm: str = "both",
    run_id: Optional[str] = None,
    raw_usage: Optional[str] = None,
    attestation_path: Optional[Path | str] = None,
    staging_report_path: Optional[Path | str] = None,
    operator_approval_path: Optional[Path | str] = None,
    repo_root: Optional[Path] = None,
) -> Dict[str, Tuple[str, str]]:
    """
    Evaluates all 5 operational readiness layers using verifiable evidence:
    1. Static Software Validation: verified local repo state against requested commit.
    2. Colab Quota Readiness: verified read-only Colab usage balance.
    3. Authenticated Drive Mount Readiness: verified mount attestation artifact bound to session.
    4. Real VM Staging & Storage Write Readiness: verified staging report artifact bound to session, run ID, and commit.
    5. Final GPU Training Authorization: explicit operator authorization record bound to session, run ID, and commit.
    """
    layers: Dict[str, Tuple[str, str]] = {}

    # Layer 1: Software
    soft_ok, soft_msg = validate_static_software(commit_sha, repo_root=repo_root)
    layers["1_STATIC_SOFTWARE"] = ("PASS_SOFTWARE" if soft_ok else "FAIL", soft_msg)

    # Layer 2: Quota
    if raw_usage is not None:
        usage = parse_colab_usage(raw_usage)
        quota_ok, quota_msg = check_quota_status(usage["balance"])
        layers["2_COMPUTE_QUOTA"] = ("PASS" if quota_ok else "BLOCKED", quota_msg)
    else:
        layers["2_COMPUTE_QUOTA"] = ("BLOCKED", "colab usage query not available")

    # Layer 3: Interactive Mount & Storage Identity (requires verifiable attestation)
    if attestation_path:
        m_ok, m_msg = verify_mount_attestation(attestation_path, session_name)
        layers["3_DRIVE_MOUNT"] = ("PASS" if m_ok else "BLOCKED", m_msg)
    else:
        layers["3_DRIVE_MOUNT"] = (
            "PENDING_OPERATOR_ACTION",
            "Interactive terminal execution of 'colab drivemount' required; mount attestation not found.",
        )

    # Layer 4: VM Staging & Write Test (requires verifiable staging report)
    if staging_report_path:
        s_ok, s_msg = verify_staging_report(
            staging_report_path,
            expected_session=session_name,
            expected_commit=commit_sha,
            expected_arm=arm,
            expected_run_id=run_id,
        )
        layers["4_VM_STAGING_WRITE"] = ("PASS" if s_ok else "BLOCKED", s_msg)
    else:
        layers["4_VM_STAGING_WRITE"] = (
            "PENDING_RUNTIME_VERIFICATION",
            "Live VM copy & staging report bound to session and commit not found.",
        )

    # Layer 5: Explicit Operator Training Authorization (requires verifiable approval record)
    if operator_approval_path:
        o_ok, o_msg = verify_operator_approval(
            operator_approval_path,
            expected_session=session_name,
            expected_commit=commit_sha,
            expected_arm=arm,
            expected_run_id=run_id,
        )
        layers["5_OPERATOR_AUTHORIZATION"] = ("PASS" if o_ok else "BLOCKED", o_msg)
    else:
        layers["5_OPERATOR_AUTHORIZATION"] = (
            "PENDING_OPERATOR_SIGN_OFF",
            "Operator sign-off file required; PASS_SOFTWARE does not authorize GPU execution.",
        )

    return layers


def is_executable_ready(layers: Dict[str, Tuple[str, str]]) -> Tuple[bool, str]:
    """
    Evaluates whether all 5 mandatory operational layers are ready for execution.
    Executable readiness requires:
    - Layer 1: PASS_SOFTWARE
    - Layer 2: PASS
    - Layer 3: PASS
    - Layer 4: PASS
    - Layer 5: PASS
    Any blocked, pending, or unverified layer prevents execution readiness.
    """
    for layer_name, (status, detail) in layers.items():
        if layer_name == "1_STATIC_SOFTWARE":
            if status != "PASS_SOFTWARE":
                return False, f"NOT READY: {layer_name} is {status} ({detail})"
        else:
            if status != "PASS":
                return False, f"NOT READY: {layer_name} is {status} ({detail})"
    return True, "READY: All mandatory operational layers verified and authorized."


def main():
    ap = argparse.ArgumentParser(description="RQ1-TRN-02B Colab Launch Workflow Orchestrator")
    ap.add_argument("--arm", choices=["C0", "C1", "both"], default="both", help="Experimental arm")
    ap.add_argument("--commit", default=None, help="Target immutable Git commit SHA (defaults to HEAD)")
    ap.add_argument("--session-prefix", default="langfens-rq1", help="Colab session prefix")
    ap.add_argument("--run-id", default=None, help="Optional execution run identifier")
    ap.add_argument("--check-only", action="store_true", help="Perform preflight readiness checks and exit (fails closed with code 1 if blocked or unverified)")
    ap.add_argument("--check-software-only", action="store_true", help="Validate static software only (exits 0 if software is valid)")
    ap.add_argument("--print-plan", action="store_true", help="Print verified execution command sequence")
    ap.add_argument("--attestation-file", default=None, help="Path to verified mount attestation JSON artifact")
    ap.add_argument("--staging-report", default=None, help="Path to verified staging report JSON artifact")
    ap.add_argument("--operator-approval-file", default=None, help="Path to operator execution approval JSON record")
    args = ap.parse_args()

    commit = validate_commit_sha(args.commit if args.commit else get_current_git_head())

    print("=" * 80)
    print("LANGFENS — RQ1-TRN-02B Colab Launch Workflow Validator")
    print(f"Target Git Commit: {commit}")
    print("=" * 80)

    # Software-only check mode: exits 0 if software validation passes
    if args.check_software_only:
        soft_ok, soft_msg = validate_static_software(commit)
        print(f"[SOFTWARE VALIDATION] {soft_msg}")
        if soft_ok:
            print("\nRESULT: PASS_SOFTWARE")
            print("Notice: PASS_SOFTWARE does not authorize GPU execution.")
            sys.exit(0)
        else:
            print("\nRESULT: FAIL")
            sys.exit(1)

    # Query Colab CLI usage (read-only real execution)
    raw_usage = None
    colab_bin = shutil.which("colab")
    if colab_bin:
        try:
            raw_usage = subprocess.check_output([colab_bin, "usage"], text=True)
        except Exception as exc:
            print(f"[COLAB QUOTA] Failed to query colab usage: {exc}", file=sys.stderr)
    else:
        print("[COLAB CLI] Binary 'colab' not found in PATH.", file=sys.stderr)

    session_name = f"{args.session_prefix}-{args.arm.lower() if args.arm != 'both' else 'c0'}"

    # Multi-layer readiness assessment based on verifiable evidence artifacts
    layers = evaluate_readiness_layers(
        commit_sha=commit,
        session_name=session_name,
        arm=args.arm,
        run_id=args.run_id,
        raw_usage=raw_usage,
        attestation_path=args.attestation_file,
        staging_report_path=args.staging_report,
        operator_approval_path=args.operator_approval_file,
    )

    print("\n--- OPERATIONAL READINESS MATRIX ---")
    for name, (status, detail) in layers.items():
        print(f"  [{status:<26}] {name}: {detail}")

    # Display execution plans if requested
    if args.print_plan:
        arms = ["C0", "C1"] if args.arm == "both" else [args.arm]
        for arm in arms:
            print("\n" + "=" * 80)
            print(f"VERIFIED EXECUTION PLAN: ARM {arm}")
            print("=" * 80)
            session_name = f"{args.session_prefix}-{arm.lower()}"
            plan = generate_colab_arm_plan(arm, commit, session_name=session_name, run_id=args.run_id)
            print("\n".join(plan))

    print("\n" + "=" * 80)
    is_ready, ready_msg = is_executable_ready(layers)

    if not is_ready:
        print("DECISION GATE: NOT READY FOR EXECUTION")
        print(f"Detail: {ready_msg}")
        print("Notice: PASS_SOFTWARE does not authorize GPU execution.")
        # Fail closed on executable check
        if args.check_only:
            sys.exit(1)
        # Even without check-only, if software failed, exit 1
        if layers["1_STATIC_SOFTWARE"][0] != "PASS_SOFTWARE":
            sys.exit(1)
    else:
        print("DECISION GATE: FULLY READY FOR EXECUTION")
        print(f"Detail: {ready_msg}")
        if args.check_only:
            sys.exit(0)

    print("=" * 80)


if __name__ == "__main__":
    main()
