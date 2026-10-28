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
    parser.add_argument("--mode", choices=["benchmark", "train_c0", "train_c1", "verify_only"], default="benchmark")
    parser.add_argument("--commit-sha", type=str, default=None, help="Pinned commit SHA to checkout")
    parser.add_argument("--remote-mirror-backend", type=str, default="local_dir")
    parser.add_argument("--remote-mirror-target", type=str, default="/kaggle/working/persistent_mirror")
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    repo_dir = Path("/kaggle/working/repo")
    if not repo_dir.exists():
        print("[SETUP] Cloning repository into /kaggle/working/repo...")
        run_command(["git", "clone", "https://github.com/minkhoaa/Langfens-RQ1-Speech-Research.git", str(repo_dir)])
        if args.commit_sha:
            run_command(["git", "checkout", args.commit_sha], cwd=repo_dir)
    else:
        print(f"[SETUP] Using existing repository at {repo_dir}")

    # Set PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_dir}/src:{repo_dir}"

    kaggle_input = Path("/kaggle/input/langfens-fold1-train-val-baseline")

    if args.mode == "verify_only":
        print("[VERIFY] Running preflight staging check...")
        cmd = [
            sys.executable,
            str(repo_dir / "audit_scripts" / "kaggle_stage_and_verify.py"),
            "--config", str(repo_dir / "configs" / "rq1_trn_02b_benchmark_kaggle.yaml"),
            "--kaggle-input-dir", str(kaggle_input),
            "--working-dir", "/kaggle/working",
            "--remote-mirror-backend", args.remote_mirror_backend,
            "--remote-mirror-target", args.remote_mirror_target,
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
            "--kaggle-input-dir", str(kaggle_input),
            "--working-dir", "/kaggle/working",
            "--remote-mirror-backend", args.remote_mirror_backend,
            "--remote-mirror-target", args.remote_mirror_target,
        ]
        if args.run_id:
            cmd.extend(["--run-id", args.run_id])
        subprocess.run(cmd, env=env, check=True)
        print(f"[TRAIN] Training completed for Arm {arm}.")


if __name__ == "__main__":
    main()
