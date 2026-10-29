#!/usr/bin/env python3
"""
LANGFENS — Kaggle Runtime Entrypoint Script
Executable entrypoint designed for execution inside the Kaggle T4x2 environment.

Orchestration:
1. Environment verification (Tesla T4 GPU, CUDA FP16 AMP).
2. Clones/verifies pinned repository commit.
3. Stages and verifies immutable dataset and baseline weights from /kaggle/input.
4. Executes benchmark or training driver.
5. Verifies and exports durable artifacts with SHA-256 readback.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description="LANGFENS Kaggle Runtime Entrypoint")
    parser.add_argument(
        "--mode",
        choices=["train_c0", "train_c1", "verify_only", "benchmark", "train_fold", "train_fold2", "train_fold3", "train_fold4"],
        default="train_c0"
    )
    parser.add_argument("--fold", type=int, choices=[2, 3, 4], default=None, help="Fold index for cross-fold baseline training (2, 3, 4)")
    parser.add_argument("--commit-sha", type=str, required=True, help="Pinned immutable Git commit SHA to checkout and verify")
    parser.add_argument("--remote-mirror-backend", type=str, default="huggingface")
    parser.add_argument("--remote-mirror-target", type=str, default=None, help="Remote persistent mirror destination")
    parser.add_argument("--remote-mirror-token", type=str, default=None)
    parser.add_argument("--kaggle-input-dir", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    repo_dir = Path("/kaggle/working/repo")
    if not repo_dir.exists():
        print("[SETUP] Cloning repository into /kaggle/working/repo...")
        run_command(["git", "clone", "https://github.com/minkhoaa/Langfens-RQ1-Speech-Research.git", str(repo_dir)])

    # Strictly checkout and verify target commit
    print(f"[SETUP] Verifying and checking out target commit {args.commit_sha}...")
    run_command(["git", "fetch", "--all"], cwd=repo_dir)
    run_command(["git", "checkout", args.commit_sha], cwd=repo_dir)

    # Verify HEAD matches commit_sha exactly
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True).strip().lower()
    if head != args.commit_sha.strip().lower():
        raise RuntimeError(f"FAIL-CLOSED: Git HEAD mismatch on Kaggle VM: actual {head} != expected {args.commit_sha}")

    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_dir, text=True).strip()
    if status:
        raise RuntimeError(f"FAIL-CLOSED: Repository working tree is dirty on Kaggle VM:\n{status}")
    print(f"[SETUP] Repository verified clean at target commit: {head}")

    # Set PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_dir}/src:{repo_dir}"

    # Search for staged dataset directory under /kaggle/input
    kaggle_input = None
    if args.kaggle_input_dir:
        candidate = Path(args.kaggle_input_dir)
        if candidate.is_dir():
            kaggle_input = candidate
    if not kaggle_input:
        candidates = [
            Path("/kaggle/input/langfens-rq1-fold1-staging"),
            Path("/kaggle/input/langfens-fold1-train-val-baseline"),
        ]
        for c in candidates:
            if c.is_dir():
                kaggle_input = c
                break
    if not kaggle_input and Path("/kaggle/input").is_dir():
        for sub in Path("/kaggle/input").iterdir():
            if sub.is_dir() and ((sub / "best.pt").is_file() or any(sub.glob("*.zip"))):
                kaggle_input = sub
                break
    print(f"[INPUT] Using Kaggle dataset input path: {kaggle_input}")

    # Set default remote target based on mode
    remote_target = args.remote_mirror_target
    if not remote_target:
        if args.mode in ("train_fold", "train_fold2", "train_fold3", "train_fold4"):
            remote_target = "minkhoaa/langfens-rq1-cross-fold-checkpoints"
        else:
            remote_target = "minkhoaa/langfens-rq1-trn-02b-checkpoints"

    if args.mode in ("train_fold", "train_fold2", "train_fold3", "train_fold4"):
        if args.mode == "train_fold2":
            f_idx = 2
        elif args.mode == "train_fold3":
            f_idx = 3
        elif args.mode == "train_fold4":
            f_idx = 4
        else:
            f_idx = args.fold or 2

        print(f"[TRAIN] Running Baseline R training for Fold {f_idx}...")
        cmd = [
            sys.executable,
            str(repo_dir / "audit_scripts" / "kaggle_fold_train_exec.py"),
            "--fold", str(f_idx),
            "--commit-sha", args.commit_sha,
            "--working-dir", "/kaggle/working",
            "--remote-mirror-backend", args.remote_mirror_backend,
            "--remote-mirror-target", remote_target,
        ]
        if kaggle_input:
            cmd.extend(["--kaggle-input-dir", str(kaggle_input)])
        if args.remote_mirror_token:
            cmd.extend(["--remote-mirror-token", args.remote_mirror_token])
        if args.run_id:
            cmd.extend(["--run-id", args.run_id])
        subprocess.run(cmd, env=env, check=True)
        print(f"[TRAIN] Training completed for Fold {f_idx}.")

    elif args.mode == "verify_only":
        print("[VERIFY] Running preflight staging check...")
        cmd = [
            sys.executable,
            str(repo_dir / "audit_scripts" / "kaggle_stage_and_verify.py"),
            "--config", str(repo_dir / "configs" / "rq1_trn_02b_c0_kaggle.yaml"),
            "--kaggle-input-dir", str(kaggle_input),
            "--working-dir", "/kaggle/working",
            "--remote-mirror-backend", args.remote_mirror_backend,
            "--remote-mirror-target", remote_target,
        ]
        subprocess.run(cmd, env=env, check=True)
        print("[VERIFY] Preflight staging check passed.")

    elif args.mode == "benchmark":
        print("[BENCHMARK] Running disposable 50-update Kaggle T4 FP16 benchmark...")
        cmd = [
            sys.executable,
            str(repo_dir / "audit_scripts" / "kaggle_benchmark_fp16.py"),
            "--config", str(repo_dir / "configs" / "rq1_trn_02b_benchmark_kaggle.yaml"),
            "--kaggle-input-dir", str(kaggle_input),
            "--output-dir", "/kaggle/working/benchmark",
            "--max-updates", "50",
        ]
        subprocess.run(cmd, env=env, check=True)
        print("[BENCHMARK] Benchmark completed successfully.")

    elif args.mode in ("train_c0", "train_c1"):
        arm = "C0" if args.mode == "train_c0" else "C1"
        cfg_name = f"rq1_trn_02b_{arm.lower()}_kaggle.yaml"
        print(f"[TRAIN] Running training for Arm {arm}...")
        cmd = [
            sys.executable,
            str(repo_dir / "audit_scripts" / "kaggle_train_exec.py"),
            "--config", str(repo_dir / "configs" / cfg_name),
            "--arm", arm,
            "--commit-sha", args.commit_sha,
            "--kaggle-input-dir", str(kaggle_input),
            "--working-dir", "/kaggle/working",
            "--remote-mirror-backend", args.remote_mirror_backend,
            "--remote-mirror-target", remote_target,
        ]
        if args.remote_mirror_token:
            cmd.extend(["--remote-mirror-token", args.remote_mirror_token])
        if args.run_id:
            cmd.extend(["--run-id", args.run_id])
        subprocess.run(cmd, env=env, check=True)
        print(f"[TRAIN] Training completed for Arm {arm}.")


if __name__ == "__main__":
    main()
