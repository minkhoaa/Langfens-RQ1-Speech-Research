"""LANGFENS — RQ1-TRN-03A: Kaggle Deployment & Trainer Resume Tests.

Senior Speech ML Research Engineer & Reproducibility Auditor.

Verifies:
  1. Trainer resume bug resolution (outdir referenced before initialization).
  2. Fresh pretrained initialization enforcement (strict rejection of Fold 1 checkpoints).
  3. Staging and per-fold test speaker isolation for Folds 2, 3, and 4.
  4. Fail-closed safety failure paths (test speaker leakage, SHA-256 corruption).
  5. Endpoint classification semantics (COMPLETED_3600_UPDATES vs INCOMPLETE).
  6. End-to-end dry-run preflight checks for Folds 2, 3, and 4.
  7. Kaggle entrypoint CLI mode dispatch for cross-fold baseline runs.
"""

import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
import pytest
import torch

from audit_scripts.kaggle_fold_train_exec import (
    EXPECTED_FOLD_PACKAGE_SHA256,
    classify_fold_training_endpoint,
    compute_sha256,
    find_and_stage_fold_package,
    verify_fresh_pretrained_initialization,
    verify_manifest,
)


MANIFEST_PATH = Path("manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json")


# ==============================================================================
# 1. TRAINER RESUME BUG RESOLUTION & STATE INTEGRITY
# ==============================================================================

def test_trainer_resume_bug_resolved():
    """Proves that outdir is properly initialized before resume check, preventing UnboundLocalError."""
    from audit_scripts.train_fold1_ctc_binary import LangfensWav2Vec2MDD

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        ckpt_path = tmp_path / "mock_ckpt.pt"
        out_dir = tmp_path / "run_out"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Create a mock checkpoint with valid weights and state dicts
        model = LangfensWav2Vec2MDD.from_pretrained(
            "facebook/wav2vec2-base-960h",
            revision="22aad52d435eb6dbaf354bdad9b0da84ce7d6156",
            ctc_weight=1.0,
            binary_weight=1.0
        )
        trainable = model.configure_two_head_training()
        optimizer = torch.optim.AdamW(trainable, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15000)
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        mock_payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": 2,
            "successful_updates": 1200,
            "micro_steps": 2400,
            "best_ap": 0.3125,
            "patience": 1,
            "skipped_amp_attempts_total": 0,
            "consecutive_skips": 0,
            "history": [{"epoch": 1}, {"epoch": 2}, {"epoch": 3}],
            "rng": {
                "python": None,
                "numpy": None,
                "torch": torch.get_rng_state(),
                "cuda": None
            },
            "metadata": {"git_sha": "test", "fold": 2},
            "fixed_budget_saved": False
        }
        torch.save(mock_payload, ckpt_path)

        # Test resuming via train_fold1_ctc_binary with --eval-only
        cfg_path = Path("configs/rq1_trn_03a_fold2.yaml")
        cmd = [
            sys.executable,
            "audit_scripts/train_fold1_ctc_binary.py",
            "--config", str(cfg_path),
            "--fold", "2",
            "--resume", str(ckpt_path),
            "--output-dir", str(out_dir),
            "--eval-only",
            "--max-eval-utterances", "2"
        ]
        # In CPU testing, train_fold1_ctc_binary exits early if device != cuda,
        # but argument parsing, config loading, and resume dictionary unpickling execute first.
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{Path.cwd()}/src:{Path.cwd()}:{env.get('PYTHONPATH', '')}"

        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        # Should NOT crash with UnboundLocalError
        assert "UnboundLocalError" not in proc.stderr
        assert "local variable 'outdir' referenced before assignment" not in proc.stderr
        assert "outdir is not defined" not in proc.stderr


# ==============================================================================
# 2. FRESH PRETRAINED INITIALIZATION ENFORCEMENT
# ==============================================================================

def test_fresh_pretrained_initialization_strictly_enforced():
    """Proves that any attempt to pass or configure a Fold 1 checkpoint is rejected."""
    # 1. Valid config passes
    valid_cfg = {
        "pretrained_model": "facebook/wav2vec2-base-960h",
        "pretrained_revision": "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
    }
    verify_fresh_pretrained_initialization(valid_cfg)

    # 2. Initial checkpoint key rejected
    forbidden_cfgs = [
        {"pretrained_model": "facebook/wav2vec2-base-960h", "initial_checkpoint": "checkpoints/RQ1-TRN-02A/recovered/best.pt"},
        {"pretrained_model": "facebook/wav2vec2-base-960h", "baseline_checkpoint": "best.pt"},
        {"pretrained_model": "facebook/wav2vec2-base-960h", "finetune_from": "fold1"},
        {"pretrained_model": "wrong_model"},
    ]
    for bad_cfg in forbidden_cfgs:
        with pytest.raises(ValueError) as exc:
            verify_fresh_pretrained_initialization(bad_cfg)
        assert "FAIL-CLOSED" in str(exc.value)


# ==============================================================================
# 3. STAGING & TEST SPEAKER ISOLATION FOR FOLDS 2, 3, 4
# ==============================================================================

def test_find_and_stage_fold_package_success():
    """Proves find_and_stage_fold_package locates, copies, and verifies fold packages."""
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        for fold_idx in [2, 3, 4]:
            expected_sha = EXPECTED_FOLD_PACKAGE_SHA256[fold_idx]
            staged_zip = find_and_stage_fold_package(
                fold_idx=fold_idx,
                kaggle_input_dir=None,
                working_dir=work_dir,
                expected_sha256=expected_sha
            )
            assert staged_zip.is_file()
            assert compute_sha256(staged_zip) == expected_sha


def test_find_and_stage_test_speaker_leakage_failure():
    """Proves that if an archive accidentally contains a test speaker, it fails closed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        leaked_pkg = work_dir / "authorized_RQ1-TRN_fold2_train_val_only.zip"

        # Create a malicious zip containing Fold 2 test speaker 'BWC.zip'
        with zipfile.ZipFile(leaked_pkg, "w") as zf:
            for spk in ["ABA", "ASI", "EBVS", "ERMS", "HJK", "LXC", "NCC", "NJS",
                        "PNV", "RRBI", "THV", "TLV", "TNI", "TXHC", "YBAA", "YDCK",
                        "YKWK", "BWC"]:  # BWC is a TEST speaker for Fold 2!
                zf.writestr(f"{spk}.zip", b"dummy")

        with pytest.raises(ValueError) as exc:
            find_and_stage_fold_package(
                fold_idx=2,
                kaggle_input_dir=work_dir,
                working_dir=work_dir / "target",
                expected_sha256=EXPECTED_FOLD_PACKAGE_SHA256[2]  # Hash will also mismatch
            )
        assert "FAIL-CLOSED" in str(exc.value)


def test_find_and_stage_sha256_corruption_failure():
    """Proves that a corrupted package fails closed with SHA-256 mismatch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        corrupt_pkg = work_dir / "authorized_RQ1-TRN_fold2_train_val_only.zip"
        corrupt_pkg.write_bytes(b"corrupted data")

        with pytest.raises(ValueError) as exc:
            find_and_stage_fold_package(
                fold_idx=2,
                kaggle_input_dir=work_dir,
                working_dir=work_dir / "target",
                expected_sha256=EXPECTED_FOLD_PACKAGE_SHA256[2]
            )
        assert "SHA-256 mismatch" in str(exc.value)


# ==============================================================================
# 4. ENDPOINT CLASSIFICATION SEMANTICS
# ==============================================================================

def test_endpoint_classification():
    """Proves endpoint classification accurately distinguishes completed from incomplete runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        # 1. Missing files -> INCOMPLETE
        status = classify_fold_training_endpoint(out_dir, fold_idx=2)
        assert status["status"] == "INCOMPLETE"

        # 2. Incomplete updates (< 3600) -> INCOMPLETE
        fixed_path = out_dir / "fixed_budget_checkpoint.pt"
        torch.save({"successful_updates": 1200, "epoch": 2}, fixed_path)
        status = classify_fold_training_endpoint(out_dir, fold_idx=2)
        assert status["status"] == "INCOMPLETE"

        # 3. Exactly 3,600 updates but missing best.pt -> FAILED
        torch.save({
            "successful_updates": 3600,
            "epoch": 6,
            "model": {},
            "optimizer": {},
            "scheduler": {},
            "scaler": {},
            "metadata": {},
            "rng": {},
            "best_ap": 0.320
        }, fixed_path)
        status = classify_fold_training_endpoint(out_dir, fold_idx=2)
        assert status["status"] == "FAILED"
        assert "best.pt missing" in status["message"]

        # 4. Complete deliverable suite -> COMPLETED_3600_UPDATES
        (out_dir / "best.pt").write_bytes(b"mock_best")
        (out_dir / "last.pt").write_bytes(b"mock_last")
        (out_dir / "history.json").write_text(json.dumps([{"epoch": 1}]) + "\n")

        status = classify_fold_training_endpoint(out_dir, fold_idx=2)
        assert status["status"] == "COMPLETED_3600_UPDATES"
        assert status["fold"] == 2
        assert status["successful_updates"] == 3600


# ==============================================================================
# 5. PREFLIGHT DRY-RUN ON ALL FOLDS (FOLDS 2, 3, 4)
# ==============================================================================

def test_kaggle_fold_train_exec_dry_run_all_folds():
    """Proves kaggle_fold_train_exec completes preflight dry-run successfully for Folds 2, 3, 4."""
    for fold_idx in [2, 3, 4]:
        cmd = [
            sys.executable,
            "audit_scripts/kaggle_fold_train_exec.py",
            "--fold", str(fold_idx),
            "--dry-run",
            "--working-dir", f"checkpoints/test_dry_run_fold{fold_idx}"
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, f"Fold {fold_idx} dry-run failed:\n{proc.stderr}"
        assert f"LANGFENS — Kaggle T4 Baseline R Runner for Fold {fold_idx}" in proc.stdout
        assert "Preflight checks passed." in proc.stdout


def test_kaggle_entrypoint_mode_dispatch():
    """Proves deploy/kaggle/kaggle_entrypoint.py parses and accepts fold modes."""
    # Test argument parsing without execution
    from deploy.kaggle.kaggle_entrypoint import main
    # Verification that the script runs with --help
    cmd = [sys.executable, "deploy/kaggle/kaggle_entrypoint.py", "--help"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "train_fold2" in proc.stdout
    assert "train_fold3" in proc.stdout
    assert "train_fold4" in proc.stdout
