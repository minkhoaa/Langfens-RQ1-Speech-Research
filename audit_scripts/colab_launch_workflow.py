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


def evaluate_readiness_layers(
    commit_sha: str,
    raw_usage: Optional[str] = None,
    drive_mounted: Optional[bool] = None,
    staging_verified: Optional[bool] = None,
    operator_authorized: Optional[bool] = None,
    repo_root: Optional[Path] = None,
) -> Dict[str, Tuple[str, str]]:
    """
    Evaluates all 5 operational readiness layers independently:
    1. Static Software Validation
    2. Colab Quota Readiness
    3. Authenticated Drive Mount Readiness
    4. Real VM Staging & Storage Write Readiness
    5. Final GPU Training Authorization
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

    # Layer 3: Interactive Mount & Storage Identity
    if drive_mounted is True:
        layers["3_DRIVE_MOUNT"] = ("PASS", "Google Drive mount verified on active runtime.")
    elif drive_mounted is False:
        layers["3_DRIVE_MOUNT"] = ("BLOCKED", "Google Drive is NOT mounted.")
    else:
        layers["3_DRIVE_MOUNT"] = (
            "PENDING_RUNTIME_VERIFICATION",
            "Interactive terminal execution of 'colab drivemount' required; storage identity PENDING_RUNTIME_VERIFICATION.",
        )

    # Layer 4: VM Staging & Write Test
    if staging_verified is True:
        layers["4_VM_STAGING_WRITE"] = ("PASS", "VM local staging & persistent Drive write access verified.")
    elif staging_verified is False:
        layers["4_VM_STAGING_WRITE"] = ("BLOCKED", "VM staging or write verification failed.")
    else:
        layers["4_VM_STAGING_WRITE"] = (
            "PENDING_RUNTIME_VERIFICATION",
            "Live VM copy & persistent Drive write test outstanding.",
        )

    # Layer 5: Explicit Operator Training Authorization
    if operator_authorized is True:
        layers["5_OPERATOR_AUTHORIZATION"] = ("PASS", "Explicit operator execution authorization confirmed.")
    elif operator_authorized is False:
        layers["5_OPERATOR_AUTHORIZATION"] = ("BLOCKED", "Operator explicitly withheld authorization.")
    else:
        layers["5_OPERATOR_AUTHORIZATION"] = (
            "PENDING_OPERATOR_SIGN_OFF",
            "Operator sign-off required; PASS_SOFTWARE does not authorize GPU execution.",
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
    ap.add_argument("--mock-usage", default=None, help="Mock output string for colab usage (for testing)")
    ap.add_argument("--drive-mounted", action="store_true", help="Assert Drive is mounted (for runtime check)")
    ap.add_argument("--staging-verified", action="store_true", help="Assert staging has been verified")
    ap.add_argument("--operator-authorized", action="store_true", help="Assert operator sign-off granted")
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

    # Query Colab CLI usage (read-only)
    raw_usage = args.mock_usage
    if raw_usage is None:
        colab_bin = shutil.which("colab")
        if colab_bin:
            try:
                raw_usage = subprocess.check_output([colab_bin, "usage"], text=True)
            except Exception as exc:
                print(f"[COLAB QUOTA] Failed to query colab usage: {exc}", file=sys.stderr)
        else:
            print("[COLAB CLI] Binary 'colab' not found in PATH.", file=sys.stderr)

    # Multi-layer readiness assessment
    layers = evaluate_readiness_layers(
        commit_sha=commit,
        raw_usage=raw_usage,
        drive_mounted=True if args.drive_mounted else None,
        staging_verified=True if args.staging_verified else None,
        operator_authorized=True if args.operator_authorized else None,
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
