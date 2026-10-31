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
        choices=[
            "train_c0", "train_c1", "verify_only", "benchmark",
            "train_fold", "train_fold1", "train_fold2", "train_fold3", "train_fold4",
            "verify_fold", "verify_fold1", "verify_fold2", "verify_fold3", "verify_fold4",
            "train_pilot_arm2a", "train_pilot_arm2b",
            "verify_pilot_arm2a", "verify_pilot_arm2b"
        ],
        default="train_c0"
    )
    parser.add_argument("--fold", type=int, choices=[1, 2, 3, 4], default=None, help="Fold index for cross-fold baseline or pilot training (1, 2, 3, 4)")
    parser.add_argument("--arm", type=str, default=None, help="Experiment arm (e.g. Arm2A_Focal, Arm2B_Weighted)")
    parser.add_argument("--config", type=str, default=None, help="Path to training config YAML")
    parser.add_argument("--commit-sha", type=str, required=True, help="Pinned immutable Git commit SHA to checkout and verify")
    parser.add_argument("--remote-mirror-backend", type=str, default="huggingface")
    parser.add_argument("--remote-mirror-target", type=str, default=None, help="Remote persistent mirror destination")
    parser.add_argument("--remote-mirror-token", type=str, default=None)
    parser.add_argument("--kaggle-input-dir", type=str, default=None)
    parser.add_argument("--repo-dir", type=str, default="/kaggle/working/repo", help="Repository directory")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--execution-cap-epochs", type=int, default=None, help="Capped execution epochs")
    args = parser.parse_args()

    # Strict token hygiene: reject CLI tokens for cross-fold baseline and pilot runs early
    is_fold_or_pilot = (
        args.mode.startswith("train_fold") or
        args.mode.startswith("verify_fold") or
        args.mode.startswith("train_pilot") or
        args.mode.startswith("verify_pilot") or
        (args.fold in (1, 2, 3, 4))
    )
    if is_fold_or_pilot and args.remote_mirror_token:
        raise ValueError(
            "FAIL-CLOSED: CLI token argument '--remote-mirror-token' is strictly prohibited for cross-fold and pilot execution. "
            "Pass authentication token strictly via HF_TOKEN environment variable or Kaggle Secrets."
        )

    repo_dir = Path(args.repo_dir)
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
        if is_fold_or_pilot:
            remote_target = "minkhoaa/langfens-rq1-cross-fold-checkpoints"
        else:
            remote_target = "minkhoaa/langfens-rq1-trn-02b-checkpoints"

    if args.mode in ("train_pilot_arm2a", "train_pilot_arm2b"):
        arm_name = "Arm2A_Focal" if args.mode == "train_pilot_arm2a" else "Arm2B_Weighted"
        cfg_name = "rq1_trn_03b_pilot_arm2a_focal.yaml" if args.mode == "train_pilot_arm2a" else "rq1_trn_03b_pilot_arm2b_weighted.yaml"
        run_id = args.run_id or ("pilot_arm2a_focal" if args.mode == "train_pilot_arm2a" else "pilot_arm2b_weighted")
        cfg_path = Path(args.config) if args.config else (repo_dir / "configs" / cfg_name)

        print(f"[TRAIN] Running RQ1-TRN-03B Pilot training for Arm {arm_name} on Fold 1...")
        cmd = [
            sys.executable,
            str(repo_dir / "audit_scripts" / "kaggle_fold_train_exec.py"),
            "--fold", "1",
            "--arm", arm_name,
            "--config", str(cfg_path),
            "--commit-sha", args.commit_sha,
            "--working-dir", "/kaggle/working",
            "--remote-mirror-backend", args.remote_mirror_backend,
            "--remote-mirror-target", remote_target,
            "--run-id", run_id,
        ]
        if kaggle_input:
            cmd.extend(["--kaggle-input-dir", str(kaggle_input)])
        if args.execution_cap_epochs is not None:
            cmd.extend(["--execution-cap-epochs", str(args.execution_cap_epochs)])
        subprocess.run(cmd, env=env, check=True)
        print(f"[TRAIN] Pilot training completed for Arm {arm_name}.")

    elif args.mode in ("verify_pilot_arm2a", "verify_pilot_arm2b"):
        arm_name = "Arm2A_Focal" if args.mode == "verify_pilot_arm2a" else "Arm2B_Weighted"
        cfg_name = "rq1_trn_03b_pilot_arm2a_focal.yaml" if args.mode == "verify_pilot_arm2a" else "rq1_trn_03b_pilot_arm2b_weighted.yaml"
        run_id = args.run_id or ("pilot_arm2a_focal" if args.mode == "verify_pilot_arm2a" else "pilot_arm2b_weighted")
        cfg_path = Path(args.config) if args.config else (repo_dir / "configs" / cfg_name)

        print(f"[VERIFY] Running preflight check-only staging and probe for Arm {arm_name} on Fold 1...")
        cmd = [
            sys.executable,
            str(repo_dir / "audit_scripts" / "kaggle_fold_train_exec.py"),
            "--fold", "1",
            "--arm", arm_name,
            "--config", str(cfg_path),
            "--commit-sha", args.commit_sha,
            "--working-dir", "/kaggle/working",
            "--check-only",
            "--remote-mirror-backend", args.remote_mirror_backend,
            "--remote-mirror-target", remote_target,
            "--run-id", run_id,
        ]
        if kaggle_input:
            cmd.extend(["--kaggle-input-dir", str(kaggle_input)])
        subprocess.run(cmd, env=env, check=True)
        print(f"[VERIFY] Preflight staging check passed for Arm {arm_name}.")

    elif args.mode in ("train_fold", "train_fold1", "train_fold2", "train_fold3", "train_fold4"):
        if args.mode == "train_fold1":
            f_idx = 1
        elif args.mode == "train_fold2":
            f_idx = 2
        elif args.mode == "train_fold3":
            f_idx = 3
        elif args.mode == "train_fold4":
            f_idx = 4
        else:
            f_idx = args.fold or 2

        print(f"[TRAIN] Running training for Fold {f_idx}...")
        cmd = [
            sys.executable,
            str(repo_dir / "audit_scripts" / "kaggle_fold_train_exec.py"),
            "--fold", str(f_idx),
            "--commit-sha", args.commit_sha,
            "--working-dir", "/kaggle/working",
            "--remote-mirror-backend", args.remote_mirror_backend,
            "--remote-mirror-target", remote_target,
        ]
        if args.arm:
            cmd.extend(["--arm", args.arm])
        if args.config:
            cmd.extend(["--config", args.config])
        if kaggle_input:
            cmd.extend(["--kaggle-input-dir", str(kaggle_input)])
        if args.run_id:
            cmd.extend(["--run-id", args.run_id])
        if args.execution_cap_epochs is not None:
            cmd.extend(["--execution-cap-epochs", str(args.execution_cap_epochs)])
        subprocess.run(cmd, env=env, check=True)
        print(f"[TRAIN] Training completed for Fold {f_idx}.")

    elif args.mode in ("verify_fold", "verify_fold1", "verify_fold2", "verify_fold3", "verify_fold4") or (args.mode == "verify_only" and args.fold in (1, 2, 3, 4)):
        if args.mode == "verify_fold1":
            f_idx = 1
        elif args.mode == "verify_fold2":
            f_idx = 2
        elif args.mode == "verify_fold3":
            f_idx = 3
        elif args.mode == "verify_fold4":
            f_idx = 4
        else:
            f_idx = args.fold or 2

        print(f"[VERIFY] Running preflight check-only staging and probe for Fold {f_idx}...")
        cmd = [
            sys.executable,
            str(repo_dir / "audit_scripts" / "kaggle_fold_train_exec.py"),
            "--fold", str(f_idx),
            "--commit-sha", args.commit_sha,
            "--working-dir", "/kaggle/working",
            "--check-only",
            "--remote-mirror-backend", args.remote_mirror_backend,
            "--remote-mirror-target", remote_target,
        ]
        if args.arm:
            cmd.extend(["--arm", args.arm])
        if args.config:
            cmd.extend(["--config", args.config])
        if kaggle_input:
            cmd.extend(["--kaggle-input-dir", str(kaggle_input)])
        if args.run_id:
            cmd.extend(["--run-id", args.run_id])
        subprocess.run(cmd, env=env, check=True)
        print(f"[VERIFY] Preflight staging check passed for Fold {f_idx}.")

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
