#!/usr/bin/env python3
"""
LANGFENS — Kaggle T4x2 Training Execution and Endpoint Classification Driver (Folds 2–4).
Orchestrates verified, leak-free Baseline R training from fresh pretrained initialization
on Kaggle Nvidia Tesla T4 GPU under CUDA FP16 AMP with per-epoch persistent mirroring.

Strict Scientific & Operational Invariants:
1. Fresh Initialization Guarantee: Model must initialize strictly from pretrained Wav2Vec2
   (facebook/wav2vec2-base-960h); loading or fine-tuning from Fold 1 checkpoints is prohibited.
2. Verified Hardware: CUDA Tesla T4 required. Fails closed on CPU or non-T4 hardware.
3. Verified Staging & Per-Fold Isolation: Validates split manifest SHA-256 and the fold's
   authorized train/val archive. Re-verifies that test partition S_test^(k) is 100% physically absent.
4. Persistent Cloud Mirroring: Every epoch checkpoint (last.pt, best.pt, fixed_budget_checkpoint.pt)
   is mirrored to private Hugging Face persistent storage with SHA-256 readback.
5. Primary Endpoint Verification: Enforces invariant completion for COMPLETED_3600_UPDATES
   while preserving validation-selected best.pt under early stopping.
6. Credential Hygiene: Resolves tokens from environment / Kaggle Secrets. Never leaks tokens in output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure repo root and src are on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

try:
    from audit_scripts.kaggle_checkpoint_sync import KaggleRemoteMirror, compute_sha256, resolve_hf_token
    from audit_scripts.kaggle_stage_and_verify import verify_kaggle_gpu_device
except ImportError:
    from kaggle_checkpoint_sync import KaggleRemoteMirror, compute_sha256, resolve_hf_token
    from kaggle_stage_and_verify import verify_kaggle_gpu_device


EXPECTED_MANIFEST_SHA256 = "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b"

EXPECTED_FOLD_PACKAGE_SHA256: Dict[int, str] = {
    1: "b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9",
    2: "112cc9e0cc5e37e5fa034df504eca019dd6ad438c96cb642335ecf10a802f0d9",
    3: "e81ac9e5d95a68d7f02c7b2ea8e5f1e9fd89b047926ffeeb4d096b0a3a5378a2",
    4: "653bdad079ee951d6b6d6a3f19964143aa5f4290a3f8c232e9b2524abdbe7f94",
}

SEALED_TEST_SPEAKERS_BY_FOLD: Dict[int, Set[str]] = {
    1: {"ABA", "ASI", "EBVS", "HJK", "LXC", "PNV"},
    2: {"BWC", "HKK", "HQTV", "MBMPS", "SKA", "SVBI"},
    3: {"ERMS", "NCC", "RRBI", "THV", "YBAA", "YDCK"},
    4: {"NJS", "TLV", "TNI", "TXHC", "YKWK", "ZHAA"},
}


def verify_manifest(manifest_path: Path) -> str:
    """Verifies SHA-256 digest of immutable split manifest."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"FAIL-CLOSED: Split manifest not found: {manifest_path}")
    digest = compute_sha256(manifest_path)
    if digest != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            f"FAIL-CLOSED: Manifest SHA-256 mismatch: actual={digest} != expected={EXPECTED_MANIFEST_SHA256}"
        )
    return digest


def find_and_stage_fold_package(
    fold_idx: int,
    kaggle_input_dir: Optional[Path | str],
    working_dir: Path,
    expected_sha256: str
) -> Path:
    """
    Locates the authorized fold package in /kaggle/input or local datasets directory,
    verifies SHA-256 integrity, copies to working staging directory, and verifies test exclusion.
    """
    target_name = f"authorized_RQ1-TRN_fold{fold_idx}_train_val_only.zip"
    staging_dir = working_dir / "staged"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_zip = staging_dir / target_name

    # If already staged and verified, reuse
    if staged_zip.is_file() and compute_sha256(staged_zip) == expected_sha256:
        print(f"[STAGING] Reusing verified staged package on VM disk: {staged_zip}")
        return staged_zip

    # Search candidates
    search_dirs = []
    if kaggle_input_dir:
        search_dirs.append(Path(kaggle_input_dir))
    search_dirs.extend([
        Path("/kaggle/input"),
        Path("datasets/L2-ARCTIC"),
        Path.cwd() / "datasets" / "L2-ARCTIC",
    ])

    source_path = None
    for s_dir in search_dirs:
        if not s_dir.is_dir():
            continue
        # Direct file check
        candidate = s_dir / target_name
        if candidate.is_file():
            source_path = candidate
            break
        # Search nested within staging dataset directory
        matches = list(s_dir.rglob(target_name))
        if matches:
            source_path = matches[0]
            break
        # Also check for .bin or .dat archives if Kaggle renamed it
        bin_matches = list(s_dir.rglob(f"*fold{fold_idx}*.zip")) + list(s_dir.rglob(f"*fold{fold_idx}*.bin"))
        for bm in bin_matches:
            if bm.is_file() and bm.stat().st_size > 500_000_000:
                source_path = bm
                break

    if not source_path or not source_path.is_file():
        raise FileNotFoundError(
            f"FAIL-CLOSED: Authorized package for Fold {fold_idx} ('{target_name}') not found in search paths: {search_dirs}"
        )

    print(f"[STAGING] Found source package: {source_path} ({source_path.stat().st_size:,} bytes)")
    src_sha = compute_sha256(source_path)
    if src_sha != expected_sha256:
        raise ValueError(
            f"FAIL-CLOSED: Source package SHA-256 mismatch for Fold {fold_idx}: "
            f"actual={src_sha} != expected={expected_sha256}"
        )

    # Copy to fast local VM disk
    tmp_dest = staged_zip.with_suffix(".tmp")
    shutil.copyfile(str(source_path), str(tmp_dest))
    os.replace(tmp_dest, staged_zip)

    staged_sha = compute_sha256(staged_zip)
    if staged_sha != expected_sha256:
        raise ValueError(
            f"FAIL-CLOSED: Staged package copy SHA-256 corrupted: {staged_sha} != {expected_sha256}"
        )

    # Verify physical test speaker exclusion inside the archive
    with zipfile.ZipFile(staged_zip, "r") as zf:
        members = set(zf.namelist())
        packaged_spks = {m.replace(".zip", "") for m in members if m.endswith(".zip")}
        assert len(packaged_spks) == 18, f"Expected 18 inner speaker archives, got {len(packaged_spks)}"
        for test_spk in SEALED_TEST_SPEAKERS_BY_FOLD[fold_idx]:
            if test_spk in packaged_spks or f"{test_spk}.zip" in members:
                raise PermissionError(
                    f"CRITICAL SAFETY VIOLATION: Test speaker '{test_spk}' found in staged archive for Fold {fold_idx}!"
                )

    print(f"[STAGING] Staged and verified package for Fold {fold_idx}: {staged_zip} (18 speakers, 0 test speakers)")
    return staged_zip


def verify_fresh_pretrained_initialization(config: Dict[str, Any]) -> None:
    """Guarantees that no Fold 1 weights or prior fine-tuned checkpoints are loaded."""
    forbidden_keys = ["initial_checkpoint", "baseline_checkpoint", "resume_checkpoint", "finetune_from"]
    for k in forbidden_keys:
        if k in config and config[k]:
            raise ValueError(
                f"FAIL-CLOSED: Configuration contains forbidden initial checkpoint key '{k}': {config[k]}. "
                f"Folds 2–4 must be trained strictly from fresh pretrained Wav2Vec2 initialization."
            )
    model_name = config.get("pretrained_model")
    if model_name != "facebook/wav2vec2-base-960h":
        raise ValueError(
            f"FAIL-CLOSED: Pretrained model must be 'facebook/wav2vec2-base-960h', got '{model_name}'"
        )


def classify_fold_training_endpoint(
    output_dir: Path | str,
    fold_idx: int,
    expected_updates: int = 3600
) -> Dict[str, Any]:
    """
    Verifies that the training run produced the required deliverable checkpoints:
    1. fixed_budget_checkpoint.pt with exactly expected_updates (3,600)
    2. best.pt
    3. last.pt
    4. history.json
    """
    p_dir = Path(output_dir).resolve()
    fixed_file = p_dir / "fixed_budget_checkpoint.pt"
    best_file = p_dir / "best.pt"
    last_file = p_dir / "last.pt"
    hist_file = p_dir / "history.json"

    if not fixed_file.is_file():
        return {
            "status": "INCOMPLETE",
            "message": f"fixed_budget_checkpoint.pt missing in {p_dir}.",
            "fold": fold_idx
        }

    try:
        ck_fixed = torch.load(fixed_file, map_location="cpu", weights_only=False)
    except Exception as e:
        return {
            "status": "FAILED",
            "message": f"Failed to load fixed_budget_checkpoint.pt: {e}",
            "fold": fold_idx
        }

    if not isinstance(ck_fixed, dict):
        return {"status": "FAILED", "message": "fixed_budget_checkpoint is not a dictionary.", "fold": fold_idx}

    updates = ck_fixed.get("successful_updates", 0)
    if updates < expected_updates:
        return {
            "status": "INCOMPLETE",
            "message": f"fixed_budget_checkpoint has only {updates} / {expected_updates} updates.",
            "fold": fold_idx
        }

    for req_key in ("model", "optimizer", "scheduler", "scaler", "metadata", "rng"):
        if req_key not in ck_fixed:
            return {
                "status": "FAILED",
                "message": f"fixed_budget_checkpoint missing required key '{req_key}'.",
                "fold": fold_idx
            }

    if not best_file.is_file():
        return {"status": "FAILED", "message": f"best.pt missing in {p_dir}.", "fold": fold_idx}

    if not last_file.is_file():
        return {"status": "FAILED", "message": f"last.pt missing in {p_dir}.", "fold": fold_idx}

    if not hist_file.is_file():
        return {"status": "FAILED", "message": f"history.json missing in {p_dir}.", "fold": fold_idx}

    return {
        "status": "COMPLETED_3600_UPDATES",
        "fold": fold_idx,
        "successful_updates": updates,
        "epoch": ck_fixed.get("epoch"),
        "best_val_ap": ck_fixed.get("best_ap"),
        "fixed_budget_sha256": compute_sha256(fixed_file),
        "best_sha256": compute_sha256(best_file),
        "last_sha256": compute_sha256(last_file),
        "history_path": str(hist_file)
    }


def main():
    parser = argparse.ArgumentParser(description="LANGFENS Kaggle Fold Training Driver (Folds 2–4)")
    parser.add_argument("--fold", type=int, choices=[2, 3, 4], required=True, help="Fold index (2, 3, or 4)")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML (defaults to configs/rq1_trn_03a_fold{k}.yaml)")
    parser.add_argument("--commit-sha", type=str, default=None, help="Expected immutable Git commit SHA")
    parser.add_argument("--kaggle-input-dir", type=str, default=None, help="Path to Kaggle input staging directory")
    parser.add_argument("--working-dir", type=str, default="/kaggle/working")
    parser.add_argument("--remote-mirror-backend", type=str, default="huggingface")
    parser.add_argument("--remote-mirror-target", type=str, default="minkhoaa/langfens-rq1-cross-fold-checkpoints")
    parser.add_argument("--remote-mirror-token", type=str, default=None, help="Token (optional; defaults to HF_TOKEN env / Kaggle Secrets)")
    parser.add_argument("--run-id", type=str, default=None, help="Unique run ID (defaults to baseline_r_fold{k})")
    parser.add_argument("--check-only", action="store_true", help="Run preflight staging and remote probe without training")
    parser.add_argument("--dry-run", action="store_true", help="Simulate execution without GPU requirement")
    parser.add_argument("--allow-mock-gpu", action="store_true", help="Allow CPU execution for test harnesses")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    args = parser.parse_args()

    fold_idx = args.fold
    cfg_path = Path(args.config or f"configs/rq1_trn_03a_fold{fold_idx}.yaml")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as cf:
        cfg = yaml.safe_load(cf)

    print("=" * 80)
    print(f"LANGFENS — Kaggle T4 Baseline R Runner for Fold {fold_idx}")
    print(f"Configuration: {cfg_path}")
    print("=" * 80)

    # 1. Commit verification
    if args.commit_sha:
        try:
            head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            if head_sha.lower() != args.commit_sha.lower():
                raise RuntimeError(
                    f"FAIL-CLOSED: Git commit SHA mismatch: HEAD={head_sha} != expected={args.commit_sha}"
                )
            print(f"[GIT] Verified immutable commit SHA: {head_sha}")
        except Exception as e:
            if not args.dry_run and not args.allow_mock_gpu:
                raise e
            print(f"[GIT WARNING] Git verification skipped in dry-run/mock mode: {e}")

    # 2. Hardware verification
    gpu_info = verify_kaggle_gpu_device(allow_mock=(args.dry_run or args.allow_mock_gpu))
    print(f"[DEVICE] Verified accelerator: {gpu_info.get('device_name', 'MOCK')}")

    # 3. Manifest integrity
    manifest_path = Path(cfg["manifest"])
    manifest_sha = verify_manifest(manifest_path)
    print(f"[MANIFEST] Verified split manifest SHA-256: {manifest_sha}")

    # 4. Fresh pretrained initialization check
    verify_fresh_pretrained_initialization(cfg)
    print(f"[INITIALIZATION] Verified fresh pretrained initialization from {cfg['pretrained_model']} (zero Fold 1 checkpoint contamination)")

    # 5. Output directory setup
    raw_working_dir = Path(args.working_dir)
    try:
        raw_working_dir.mkdir(parents=True, exist_ok=True)
        working_dir = raw_working_dir.resolve()
    except (PermissionError, OSError):
        # Local development / CI test fallback when /kaggle is not writable
        working_dir = (Path.cwd() / "checkpoints" / "kaggle_working").resolve()
        working_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id or f"baseline_r_fold{fold_idx}"
    outdir = working_dir / "checkpoints" / f"fold{fold_idx}" / run_id

    # Conflict detection
    if not args.resume and not args.check_only and not args.dry_run and outdir.is_dir():
        conflicts = [str(outdir / chk) for chk in ("last.pt", "best.pt", "fixed_budget_checkpoint.pt") if (outdir / chk).exists()]
        if conflicts:
            raise FileExistsError(
                f"FAIL-CLOSED: Pre-existing checkpoint artifacts detected in output directory:\n"
                f"  {conflicts}\n"
                f"Fresh execution cannot overwrite previous runs. Specify a unique --run-id or pass --resume."
            )
    outdir.mkdir(parents=True, exist_ok=True)

    # 6. Dataset staging and test exclusion
    expected_pkg_sha = EXPECTED_FOLD_PACKAGE_SHA256[fold_idx]
    staged_zip = find_and_stage_fold_package(
        fold_idx=fold_idx,
        kaggle_input_dir=args.kaggle_input_dir,
        working_dir=working_dir,
        expected_sha256=expected_pkg_sha
    )

    # 7. Remote mirror setup & readiness probe
    remote_mirror = None
    if args.remote_mirror_backend:
        remote_mirror = KaggleRemoteMirror(
            backend=args.remote_mirror_backend,
            remote_target=args.remote_mirror_target,
            auth_token=args.remote_mirror_token,
            require_verification=not args.dry_run,
        )
        if not args.dry_run and not args.allow_mock_gpu:
            ok, msg = remote_mirror.verify_readiness()
            if not ok:
                raise RuntimeError(f"FAIL-CLOSED: Remote mirror readiness verification failed: {msg}")
            print(f"[REMOTE MIRROR] Verified remote persistent mirror to {args.remote_mirror_backend}:{args.remote_mirror_target}")
        else:
            print(f"[REMOTE MIRROR] Remote mirror configured ({args.remote_mirror_backend}:{args.remote_mirror_target}); live probe skipped in dry-run/mock mode.")

    # Check-only exit
    if args.check_only:
        print(f"\n[CHECK-ONLY] Preflight checks passed for Fold {fold_idx}. Ready for Kaggle GPU launch.")
        return 0

    if args.dry_run:
        print(f"\n[DRY-RUN] Preflight checks passed. Training simulation complete.")
        return 0

    # 8. Launch training
    repo_dir = Path(__file__).resolve().parent.parent
    trainer_script = repo_dir / "audit_scripts" / "train_fold1_ctc_binary.py"
    cmd = [
        sys.executable,
        str(trainer_script),
        "--config", str(cfg_path),
        "--fold", str(fold_idx),
        "--zip-path", str(staged_zip),
        "--output-dir", str(outdir),
        "--remote-mirror-backend", args.remote_mirror_backend,
        "--remote-mirror-target", args.remote_mirror_target,
        "--remote-rel-prefix", f"fold{fold_idx}/{run_id}",
    ]
    if args.remote_mirror_token:
        cmd.extend(["--remote-mirror-token", args.remote_mirror_token])
    if args.resume:
        cmd.extend(["--resume", str(args.resume)])

    print(f"\n[LAUNCH] Starting Baseline R training for Fold {fold_idx}...")
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_dir}/src:{repo_dir}:{env.get('PYTHONPATH', '')}"

    subprocess.run(cmd, env=env, check=True)
    elapsed = time.time() - t0
    print(f"[LAUNCH] Training subprocess finished in {elapsed:.1f}s.")

    # 9. Verify and classify endpoint
    endpoint_status = classify_fold_training_endpoint(outdir, fold_idx)
    report_path = outdir / "training_endpoint_report.json"
    report_path.write_text(json.dumps(endpoint_status, indent=2) + "\n")

    if remote_mirror is not None:
        remote_mirror.mirror_file(report_path, f"fold{fold_idx}/{run_id}/training_endpoint_report.json", verify_sha=True)

    print(f"[ENDPOINT] Final endpoint classification: {endpoint_status['status']}")
    if endpoint_status["status"] != "COMPLETED_3600_UPDATES":
        raise RuntimeError(f"Training did not achieve COMPLETED_3600_UPDATES: {endpoint_status}")

    print("=" * 80)
    print(f"FOLD {fold_idx} TRAINING EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
