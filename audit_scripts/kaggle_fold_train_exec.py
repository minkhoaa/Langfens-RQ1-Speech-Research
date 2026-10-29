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
    from audit_scripts.kaggle_checkpoint_sync import KaggleRemoteMirror, compute_sha256, resolve_cross_fold_hf_token
    from audit_scripts.kaggle_stage_and_verify import verify_kaggle_gpu_device
except ImportError:
    from kaggle_checkpoint_sync import KaggleRemoteMirror, compute_sha256, resolve_cross_fold_hf_token
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

ALL_SPEAKERS: Set[str] = {
    "ABA", "ASI", "BWC", "EBVS", "ERMS", "HKK", "HJK", "HQTV",
    "LXC", "MBMPS", "NCC", "NJS", "PNV", "RRBI", "SKA", "SVBI",
    "THV", "TLV", "TNI", "TXHC", "YBAA", "YDCK", "YKWK", "ZHAA"
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


def audit_archive_for_test_speaker_leak(zip_path: Path | str, fold_idx: int) -> None:
    """
    Rigorously inspects an archive's members to detect and reject active test speaker leakage.
    Operates independently of SHA-256 verification.
    """
    p = Path(zip_path)
    if not p.is_file():
        raise FileNotFoundError(f"Archive not found: {p}")
    test_spks = SEALED_TEST_SPEAKERS_BY_FOLD.get(fold_idx, set())
    try:
        with zipfile.ZipFile(p, "r") as zf:
            members = set(zf.namelist())
            packaged_spks = {m.replace(".zip", "") for m in members if m.endswith(".zip")}
            for test_spk in test_spks:
                if test_spk in packaged_spks or f"{test_spk}.zip" in members:
                    raise PermissionError(
                        f"CRITICAL SAFETY VIOLATION: Test speaker '{test_spk}' found in staged archive for Fold {fold_idx}!"
                    )
    except zipfile.BadZipFile as e:
        raise ValueError(f"Corrupted archive: {p} is not a valid zip file ({e})")


def find_and_stage_fold_package(
    fold_idx: int,
    kaggle_input_dir: Optional[Path | str],
    working_dir: Path,
    expected_sha256: str
) -> Path:
    """
    Locates the authorized fold package in /kaggle/input or local datasets directory,
    verifies SHA-256 integrity, copies to working staging directory, and verifies test exclusion.
    Gracefully handles Kaggle auto-extracted datasets with bit-exact reconstruction and verification.
    """
    target_name = f"authorized_RQ1-TRN_fold{fold_idx}_train_val_only.zip"
    staging_dir = working_dir / "staged"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_zip = staging_dir / target_name

    # If already staged and verified, reuse
    if staged_zip.is_file() and compute_sha256(staged_zip) == expected_sha256:
        audit_archive_for_test_speaker_leak(staged_zip, fold_idx)
        print(f"[STAGING] Reusing verified staged package on VM disk: {staged_zip}")
        return staged_zip

    test_spks = SEALED_TEST_SPEAKERS_BY_FOLD.get(fold_idx, set())
    auth_spks = sorted(ALL_SPEAKERS - test_spks)

    # 1. Primary path: if kaggle_input_dir is provided, inspect it with strict priority
    if kaggle_input_dir:
        input_p = Path(kaggle_input_dir)
        if input_p.is_dir():
            # A. Immediate test speaker exclusion check on input directory
            for test_spk in test_spks:
                test_candidates = list(input_p.rglob(f"{test_spk}.zip"))
                for sub in input_p.iterdir():
                    if sub.name == test_spk:
                        test_candidates.append(sub)
                if any(tc.exists() for tc in test_candidates):
                    raise PermissionError(
                        f"CRITICAL SAFETY VIOLATION: Test speaker '{test_spk}' found in staged dataset directory '{input_p}' for Fold {fold_idx}!"
                    )

            # B. Check for outer package in input directory
            source_path = None
            if (input_p / target_name).is_file():
                source_path = input_p / target_name
            else:
                matches = list(input_p.rglob(target_name))
                if matches:
                    source_path = matches[0]
                else:
                    bin_matches = list(input_p.rglob(f"*fold{fold_idx}*.zip")) + list(input_p.rglob(f"*fold{fold_idx}*.bin")) + list(input_p.rglob(f"*fold{fold_idx}*.dat"))
                    outer_bins = [bm for bm in bin_matches if bm.is_file() and bm.stat().st_size > 500_000_000]
                    if outer_bins:
                        source_path = outer_bins[0]

            if source_path and source_path.is_file():
                audit_archive_for_test_speaker_leak(source_path, fold_idx)
                print(f"[STAGING] Found source package in input directory: {source_path} ({source_path.stat().st_size:,} bytes)")
                src_sha = compute_sha256(source_path)
                if src_sha != expected_sha256:
                    raise ValueError(
                        f"FAIL-CLOSED: Source package SHA-256 mismatch for Fold {fold_idx}: "
                        f"actual={src_sha} != expected={expected_sha256}"
                    )
                tmp_dest = staged_zip.with_suffix(".tmp")
                shutil.copyfile(str(source_path), str(tmp_dest))
                os.replace(tmp_dest, staged_zip)
                audit_archive_for_test_speaker_leak(staged_zip, fold_idx)
                print(f"[STAGING] Staged and verified package for Fold {fold_idx}: {staged_zip}")
                return staged_zip

            # C. Check for auto-extracted speaker archives in input directory
            spk_files = list(input_p.rglob("*.zip"))
            spk_file_map = {sf.stem: sf for sf in spk_files if sf.is_file()}
            if set(auth_spks).issubset(spk_file_map.keys()):
                print(f"[STAGING] Detected auto-extracted dataset in '{input_p}' with all 18 authorized speakers.")
                tmp_dest = staged_zip.with_suffix(".rebuild_tmp")
                with zipfile.ZipFile(tmp_dest, "w", zipfile.ZIP_STORED) as out_zf:
                    for spk in auth_spks:
                        sf_path = spk_file_map[spk]
                        zinfo = zipfile.ZipInfo(f"{spk}.zip", (2026, 1, 1, 0, 0, 0))
                        zinfo.compress_type = zipfile.ZIP_STORED
                        out_zf.writestr(zinfo, sf_path.read_bytes())
                os.replace(tmp_dest, staged_zip)
                staged_sha = compute_sha256(staged_zip)
                if staged_sha != expected_sha256:
                    raise ValueError(
                        f"FAIL-CLOSED: Reconstructed package SHA-256 mismatch for Fold {fold_idx}: "
                        f"actual={staged_sha} != expected={expected_sha256}"
                    )
                audit_archive_for_test_speaker_leak(staged_zip, fold_idx)
                print(f"[STAGING] Successfully reconstructed and verified outer zip package from auto-extracted speaker archives: {staged_zip}")
                return staged_zip

    # 2. Fallback search candidates
    search_dirs = [
        Path("/kaggle/input"),
        Path("datasets/L2-ARCTIC"),
        Path.cwd() / "datasets" / "L2-ARCTIC",
    ]

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
        bin_matches = list(s_dir.rglob(f"*fold{fold_idx}*.zip")) + list(s_dir.rglob(f"*fold{fold_idx}*.bin")) + list(s_dir.rglob(f"*fold{fold_idx}*.dat"))
        for bm in bin_matches:
            if bm.is_file() and bm.stat().st_size > 500_000_000:
                source_path = bm
                break

    if source_path and source_path.is_file():
        # Pre-copy test leakage audit
        audit_archive_for_test_speaker_leak(source_path, fold_idx)

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
    else:
        # Check for auto-extracted speaker archives across fallback search_dirs
        print(f"[STAGING] Outer archive '{target_name}' not found. Checking for Kaggle auto-extracted dataset in search dirs...")
        reconstructed = False

        for s_dir in search_dirs:
            if not s_dir.is_dir():
                continue

            # First, check for active test speaker leak in auto-extracted directory
            for test_spk in test_spks:
                test_candidates = list(s_dir.rglob(f"{test_spk}.zip"))
                for sub in s_dir.iterdir():
                    if sub.name == test_spk:
                        test_candidates.append(sub)
                if any(tc.exists() for tc in test_candidates):
                    raise PermissionError(
                        f"CRITICAL SAFETY VIOLATION: Test speaker '{test_spk}' found in staged dataset directory '{s_dir}' for Fold {fold_idx}!"
                    )

            # Look for inner speaker zip files
            spk_files = list(s_dir.rglob("*.zip"))
            spk_file_map = {sf.stem: sf for sf in spk_files if sf.is_file()}

            if set(auth_spks).issubset(spk_file_map.keys()):
                print(f"[STAGING] Detected auto-extracted dataset in '{s_dir}' with all 18 authorized speakers.")
                tmp_dest = staged_zip.with_suffix(".rebuild_tmp")
                with zipfile.ZipFile(tmp_dest, "w", zipfile.ZIP_STORED) as out_zf:
                    for spk in auth_spks:
                        sf_path = spk_file_map[spk]
                        zinfo = zipfile.ZipInfo(f"{spk}.zip", (2026, 1, 1, 0, 0, 0))
                        zinfo.compress_type = zipfile.ZIP_STORED
                        out_zf.writestr(zinfo, sf_path.read_bytes())

                os.replace(tmp_dest, staged_zip)
                reconstructed = True
                print(f"[STAGING] Successfully reconstructed outer zip package from auto-extracted speaker archives.")
                break

        if not reconstructed:
            raise FileNotFoundError(
                f"FAIL-CLOSED: Authorized package for Fold {fold_idx} ('{target_name}') not found in search paths: {search_dirs}"
            )

    staged_sha = compute_sha256(staged_zip)
    if staged_sha != expected_sha256:
        raise ValueError(
            f"FAIL-CLOSED: Staged package copy SHA-256 corrupted: {staged_sha} != {expected_sha256}"
        )

    # Post-copy test speaker exclusion audit
    audit_archive_for_test_speaker_leak(staged_zip, fold_idx)

    with zipfile.ZipFile(staged_zip, "r") as zf:
        members = set(zf.namelist())
        packaged_spks = {m.replace(".zip", "") for m in members if m.endswith(".zip")}
        assert len(packaged_spks) == 18, f"Expected 18 inner speaker archives, got {len(packaged_spks)}"

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
    expected_updates: int = 3600,
    expected_commit: Optional[str] = None
) -> Dict[str, Any]:
    """
    Rigorously verifies that the training run produced the required deliverable checkpoints:
    1. fixed_budget_checkpoint.pt with EXACTLY expected_updates (3,600) and epoch == 5 (6th completed epoch).
    2. best.pt
    3. last.pt
    4. history.json (at least 6 completed epochs)
    5. Correct fold in metadata (meta['fold'] == fold_idx).
    6. Immutable manifest SHA-256 match.
    7. Pinned configuration hyperparameter invariants.
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

    updates = ck_fixed.get("successful_updates")
    epoch = ck_fixed.get("epoch")

    # Invariant 1: updates == 3600
    if updates != expected_updates:
        return {
            "status": "INCOMPLETE" if (updates is not None and updates < expected_updates) else "FAILED",
            "message": f"fixed_budget_checkpoint has updates={updates}, expected exactly {expected_updates}.",
            "fold": fold_idx
        }

    # Invariant 2: epoch == 5 (0-indexed, meaning 6th completed epoch)
    if epoch != 5:
        return {
            "status": "FAILED",
            "message": f"fixed_budget_checkpoint has epoch={epoch}, expected exactly 5 (6th epoch).",
            "fold": fold_idx
        }

    # Invariant 3: Required top-level payload keys
    for req_key in ("model", "optimizer", "scheduler", "scaler", "metadata", "rng", "history"):
        if req_key not in ck_fixed:
            return {
                "status": "FAILED",
                "message": f"fixed_budget_checkpoint missing required key '{req_key}'.",
                "fold": fold_idx
            }
    if not isinstance(ck_fixed["model"], dict) or len(ck_fixed["model"]) == 0:
        return {"status": "FAILED", "message": "fixed_budget_checkpoint model state dict missing or empty.", "fold": fold_idx}
    if not isinstance(ck_fixed["optimizer"], dict) or len(ck_fixed["optimizer"]) == 0:
        return {"status": "FAILED", "message": "fixed_budget_checkpoint optimizer state dict missing or empty.", "fold": fold_idx}

    meta = ck_fixed.get("metadata", {})
    if not isinstance(meta, dict):
        return {"status": "FAILED", "message": "fixed_budget_checkpoint metadata is not a dict.", "fold": fold_idx}

    # Invariant 4: Correct fold identity
    ck_fold = meta.get("fold")
    if ck_fold is None:
        ck_fold = meta.get("config", {}).get("fold")
    if ck_fold is None or int(ck_fold) != fold_idx:
        return {
            "status": "FAILED",
            "message": f"Fold identity mismatch in checkpoint: found fold {ck_fold}, expected {fold_idx}.",
            "fold": fold_idx
        }

    # Invariant 5: Immutable split manifest SHA-256
    if meta.get("split_sha256") != EXPECTED_MANIFEST_SHA256:
        return {
            "status": "FAILED",
            "message": f"Manifest SHA-256 mismatch in checkpoint: {meta.get('split_sha256')} != {EXPECTED_MANIFEST_SHA256}.",
            "fold": fold_idx
        }

    # Invariant 6: Pinned configuration provenance
    ck_cfg = meta.get("config", {})
    if not isinstance(ck_cfg, dict):
        return {"status": "FAILED", "message": "Checkpoint metadata missing config dict.", "fold": fold_idx}

    cfg_invariants = {
        "seed": 1337,
        "batch_size": 2,
        "gradient_accumulation_steps": 2,
        "warmup_updates": 100,
        "max_epochs": 25,
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "optimizer": "AdamW",
        "scheduler": "cosine",
        "loss_weights": {"ctc": 1.0, "binary": 1.0},
        "backbone_frozen": True,
        "amp": "fp16",
        "grad_scaler_initial_scale": 4096,
        "early_stopping_patience": 5,
    }
    for k_inv, expected_val in cfg_invariants.items():
        if ck_cfg.get(k_inv) != expected_val:
            return {
                "status": "FAILED",
                "message": f"Configuration invariant violated in checkpoint: {k_inv}={ck_cfg.get(k_inv)}, expected {expected_val}.",
                "fold": fold_idx
            }

    # Invariant 7: Git commit verification if provided
    if expected_commit and meta.get("git_sha"):
        if meta.get("git_sha").lower() != expected_commit.lower():
            return {
                "status": "FAILED",
                "message": f"Git commit SHA mismatch in checkpoint: {meta.get('git_sha')} != {expected_commit}.",
                "fold": fold_idx
            }

    # Invariant 8: best.pt, last.pt, and history.json
    if not best_file.is_file():
        return {"status": "FAILED", "message": f"best.pt missing in {p_dir}.", "fold": fold_idx}

    if not last_file.is_file():
        return {"status": "FAILED", "message": f"last.pt missing in {p_dir}.", "fold": fold_idx}

    if not hist_file.is_file():
        return {"status": "FAILED", "message": f"history.json missing in {p_dir}.", "fold": fold_idx}

    try:
        hist_data = json.loads(hist_file.read_text())
        if not isinstance(hist_data, list) or len(hist_data) < 6:
            return {
                "status": "INCOMPLETE",
                "message": f"history.json has only {len(hist_data) if isinstance(hist_data, list) else 0} epochs, expected >= 6.",
                "fold": fold_idx
            }
        if len(hist_data) > 10:
            return {
                "status": "FAILED",
                "message": f"history.json has {len(hist_data)} epochs, exceeding execution cap of 10 epochs.",
                "fold": fold_idx
            }
    except Exception as e:
        return {"status": "FAILED", "message": f"Failed to parse history.json: {e}", "fold": fold_idx}

    return {
        "status": "COMPLETED_3600_UPDATES",
        "fold": fold_idx,
        "successful_updates": updates,
        "epoch": epoch,
        "best_val_ap": ck_fixed.get("best_ap"),
        "fixed_budget_sha256": compute_sha256(fixed_file),
        "best_sha256": compute_sha256(best_file),
        "last_sha256": compute_sha256(last_file),
        "history_sha256": compute_sha256(hist_file),
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
    if not args.dry_run:
        if not args.remote_mirror_backend or args.remote_mirror_backend != "huggingface":
            raise RuntimeError(
                "FAIL-CLOSED: Production Kaggle runs strictly require 'huggingface' remote persistence backend. "
                f"Got '{args.remote_mirror_backend}'."
            )
        if not args.remote_mirror_target or not args.remote_mirror_target.strip():
            raise RuntimeError("FAIL-CLOSED: Production Kaggle runs require explicit remote mirror target repository.")

        # Require authentication token from environment or Kaggle Secrets only
        token = resolve_cross_fold_hf_token()
        if not token:
            raise RuntimeError(
                "FAIL-CLOSED: Hugging Face remote persistence requires valid HF_TOKEN in environment or Kaggle Secrets."
            )

        remote_mirror = KaggleRemoteMirror(
            backend=args.remote_mirror_backend,
            remote_target=args.remote_mirror_target,
            auth_token=token,
            require_verification=not args.allow_mock_gpu,
            strict_secrets_only=True
        )
        if not args.allow_mock_gpu:
            ok, msg = remote_mirror.verify_readiness()
            if not ok:
                raise RuntimeError(f"FAIL-CLOSED: Remote mirror readiness verification failed: {msg}")
            print(f"[REMOTE MIRROR] Verified private remote persistent mirror to {args.remote_mirror_backend}:{args.remote_mirror_target}")
        else:
            print(f"[REMOTE MIRROR] Mock GPU mode: HF_TOKEN verified; live network probe skipped.")
    else:
        token = resolve_cross_fold_hf_token()
        remote_mirror = KaggleRemoteMirror(
            backend=args.remote_mirror_backend,
            remote_target=args.remote_mirror_target,
            auth_token=token,
            require_verification=False,
            strict_secrets_only=True
        )
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
        "--execution-cap-epochs", "10",
        "--remote-mirror-backend", args.remote_mirror_backend,
        "--remote-mirror-target", args.remote_mirror_target,
        "--remote-rel-prefix", f"fold{fold_idx}/{run_id}",
    ]
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
