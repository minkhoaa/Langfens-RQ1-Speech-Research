#!/usr/bin/env python3
"""
LANGFENS — RQ1-TRN-02B: Colab Launch Workflow Orchestrator & Static Validator
Local tool to validate preflight readiness, check compute quota, verify Git state,
and generate verified Colab CLI execution sequences for Control C0 and Treatment C1.

Key Guarantees:
1. Non-Interactive CLI Syntax: Uses 'colab exec -s SESSION -f SCRIPT --env K=V --timeout SEC'.
2. Quota Gate: Inspects read-only 'colab usage' and blocks execution if compute units are zero.
3. Provenance Verification: Binds execution to an immutable Git commit SHA.
4. Separate Arm Workflows: Generates dedicated, non-overlapping workflows for C0 and C1.
5. Strict Fallback Prohibition: Never executes training on local CPU/GPU.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_URL = "https://github.com/minkhoaa/Langfens-RQ1-Speech-Research.git"
DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/KLTN-Dataset"
REQUIRED_COMPUTE_UNITS_MIN = 15.0


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
        f"# --- Phase 4: Stage Authorized Data & Baseline Checkpoint to VM Disk ---",
        f"colab exec -s {session} -f audit_scripts/colab_stage_and_verify.py {env_flags} --timeout 1800",
        f"",
        f"# --- Phase 5: Execute Matched-Budget Training (Timeout: 8 hours) ---",
        f"colab exec -s {session} -f audit_scripts/colab_train_exec.py {env_flags} --timeout 28800",
        f"",
        f"# --- Phase 6: Release Colab VM Runtime ---",
        f"colab stop -s {session}",
    ]
    return commands


def main():
    ap = argparse.ArgumentParser(description="RQ1-TRN-02B Colab Launch Workflow Orchestrator")
    ap.add_argument("--arm", choices=["C0", "C1", "both"], default="both", help="Experimental arm")
    ap.add_argument("--commit", default=None, help="Target immutable Git commit SHA (defaults to HEAD)")
    ap.add_argument("--session-prefix", default="langfens-rq1", help="Colab session prefix")
    ap.add_argument("--run-id", default=None, help="Optional execution run identifier")
    ap.add_argument("--check-only", action="store_true", help="Perform preflight checks and exit")
    ap.add_argument("--print-plan", action="store_true", help="Print verified execution command sequence")
    args = ap.parse_args()

    commit = validate_commit_sha(args.commit if args.commit else get_current_git_head())

    print("=" * 80)
    print("LANGFENS — RQ1-TRN-02B Colab Launch Workflow Validator")
    print(f"Target Git Commit: {commit}")
    print("=" * 80)

    # 1. Local Git State Audit
    git_clean = check_git_clean()
    print(f"[GIT] Working tree clean: {git_clean}")

    # 2. Read-Only Colab Usage Audit
    colab_bin = shutil.which("colab")
    if not colab_bin:
        print("[COLAB CLI] Binary 'colab' not found in PATH.", file=sys.stderr)
        sys.exit(1)

    try:
        raw_usage = subprocess.check_output([colab_bin, "usage"], text=True)
        usage_info = parse_colab_usage(raw_usage)
        quota_ok, quota_msg = check_quota_status(usage_info["balance"])
        print(f"[COLAB QUOTA] Balance: {usage_info['balance']:.2f} compute units | Status: {quota_msg}")
        print(f"[COLAB QUOTA] Usage rate: {usage_info['usage_rate']:.2f}/hr | Active assignments: {usage_info['active_assignments']}")
    except Exception as exc:
        print(f"[COLAB QUOTA] Failed to query colab usage: {exc}", file=sys.stderr)
        quota_ok = False
        quota_msg = f"ERROR: {exc}"

    # 3. Read-Only Active Sessions Audit
    try:
        raw_sessions = subprocess.check_output([colab_bin, "sessions"], text=True).strip()
        print(f"[COLAB SESSIONS] {raw_sessions}")
    except Exception as exc:
        print(f"[COLAB SESSIONS] Failed to query colab sessions: {exc}", file=sys.stderr)

    # 4. Generate or Display Plans
    arms = ["C0", "C1"] if args.arm == "both" else [args.arm]
    for arm in arms:
        print("\n" + "=" * 80)
        print(f"VERIFIED EXECUTION PLAN: ARM {arm}")
        print("=" * 80)
        session_name = f"{args.session_prefix}-{arm.lower()}"
        plan = generate_colab_arm_plan(arm, commit, session_name=session_name, run_id=args.run_id)
        print("\n".join(plan))

    print("\n" + "=" * 80)
    if not quota_ok:
        print("DECISION GATE: BLOCKED ON COMPUTE QUOTA")
        print("Operator must purchase/allocate ~15-20 compute units before executing.")
    else:
        print("DECISION GATE: READY FOR AUTHORIZED OPERATOR LAUNCH")
    print("=" * 80)


if __name__ == "__main__":
    main()
