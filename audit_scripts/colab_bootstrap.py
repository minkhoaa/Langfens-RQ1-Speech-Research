#!/usr/bin/env python3
"""
LANGFENS — RQ1-TRN-02B: Colab VM Deterministic Repository Bootstrap Driver
Designed for execution on Google Colab VM via:
    colab exec -s <session> -f audit_scripts/colab_bootstrap.py --timeout 300

Key Contracts:
1. Deterministic Clone: Clones the official repository into /content/repo if not present.
2. Commit Binding: Checks out the approved immutable Git commit SHA.
3. Identity & Cleanliness:
   - Validates that origin URL matches 'Langfens-RQ1-Speech-Research'.
   - Refuses pre-existing directories that are not valid Git repositories.
   - Refuses repositories with dirty working trees.
4. Kernel Execution Compatibility:
   - Does not assume __file__ is defined.
   - Uses parse_known_args() to ignore Jupyter kernel launcher arguments.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_URL = "https://github.com/minkhoaa/Langfens-RQ1-Speech-Research.git"
DEFAULT_REPO_ROOT = "/content/repo"


def validate_commit_sha(sha: str) -> str:
    """Validates 40-character hexadecimal Git commit SHA."""
    cleaned = sha.strip().lower()
    if len(cleaned) != 40 or not all(c in "0123456789abcdef" for c in cleaned):
        raise ValueError(f"Invalid Git commit SHA '{sha}'. Expected 40-character hex string.")
    return cleaned


def bootstrap_repository(
    repo_root: Path | str,
    target_commit: str,
    repo_url: str = REPO_URL,
) -> str:
    """
    Deterministic bootstrap routine:
    - Clones repository if absent.
    - Validates origin URL and cleanliness if present.
    - Checks out target immutable commit SHA.
    - Verifies HEAD matches target commit.
    """
    repo_path = Path(repo_root)
    commit_norm = validate_commit_sha(target_commit)

    print("=" * 80)
    print(f"LANGFENS — Colab Repository Bootstrap")
    print(f"Target Path: {repo_path}")
    print(f"Target Commit: {commit_norm}")
    print("=" * 80, flush=True)

    if not repo_path.exists():
        print(f"[BOOTSTRAP] Cloning {repo_url} into {repo_path}...", flush=True)
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        res = subprocess.run(
            ["git", "clone", repo_url, str(repo_path)],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            raise RuntimeError(f"Git clone failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")
    else:
        # Pre-existing directory audit
        git_dir = repo_path / ".git"
        if not git_dir.is_dir():
            raise RuntimeError(
                f"FAIL-CLOSED: Pre-existing directory {repo_path} is NOT a Git repository.\n"
                f"Refusing to use an unrelated or corrupt directory."
            )

        # Check remote origin identity
        try:
            origin_url = subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=str(repo_path),
                text=True,
            ).strip()
        except subprocess.CalledProcessError:
            origin_url = ""

        if "Langfens-RQ1-Speech-Research" not in origin_url:
            raise RuntimeError(
                f"FAIL-CLOSED: Pre-existing repository at {repo_path} has unexpected remote origin:\n"
                f"  Found: {origin_url}\n"
                f"  Expected: {repo_url}\n"
                f"Refusing to execute on an untrusted or unrelated repository."
            )

        # Check working tree cleanliness
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_path),
            text=True,
        ).strip()
        if status:
            raise RuntimeError(
                f"FAIL-CLOSED: Pre-existing repository at {repo_path} has dirty working tree:\n"
                f"{status}\n"
                f"Refusing to checkout commit over uncommitted changes."
            )

        # Fetch latest if needed
        subprocess.run(["git", "fetch", "--all"], cwd=str(repo_path), capture_output=True)

    # Checkout exact immutable commit
    print(f"[BOOTSTRAP] Checking out commit {commit_norm}...", flush=True)
    chk_res = subprocess.run(
        ["git", "checkout", commit_norm],
        cwd=str(repo_path),
        capture_output=True,
        text=True,
    )
    if chk_res.returncode != 0:
        raise RuntimeError(
            f"Git checkout failed for commit {commit_norm}:\n"
            f"STDOUT: {chk_res.stdout}\nSTDERR: {chk_res.stderr}"
        )

    # Verify verified HEAD
    head_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_path),
        text=True,
    ).strip().lower()

    if head_sha != commit_norm:
        raise RuntimeError(
            f"HEAD verification failed after checkout: HEAD is {head_sha}, expected {commit_norm}."
        )

    # Re-verify clean tree
    post_status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=str(repo_path),
        text=True,
    ).strip()
    if post_status:
        raise RuntimeError(
            f"Working tree at {repo_path} is dirty after checking out {commit_norm}:\n{post_status}"
        )

    print(f"[BOOTSTRAP VERIFIED] Repository ready at {repo_path} at commit {head_sha}.", flush=True)
    return head_sha


def main():
    ap = argparse.ArgumentParser(description="RQ1-TRN-02B Colab Repository Bootstrap Driver")
    ap.add_argument("--repo-root", default=os.environ.get("REPO_ROOT", DEFAULT_REPO_ROOT), help="Path to clone/verify repository")
    ap.add_argument("--target-commit", default=os.environ.get("TARGET_COMMIT", ""), help="Target immutable Git commit SHA")
    ap.add_argument("--repo-url", default=os.environ.get("REPO_URL", REPO_URL), help="Git repository URL")
    args, _ = ap.parse_known_args()

    if not args.target_commit:
        print("ERROR: --target-commit or TARGET_COMMIT environment variable is required.", file=sys.stderr)
        sys.exit(1)

    bootstrap_repository(
        repo_root=args.repo_root,
        target_commit=args.target_commit,
        repo_url=args.repo_url,
    )


if __name__ == "__main__":
    main()
