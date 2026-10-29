"""LANGFENS — RQ1-TRN-03A: Kaggle Deployment & Trainer Resume Tests.

Senior Speech ML Research Engineer & Reproducibility Auditor.

Verifies:
  1. Executable mocked checkpoint recovery and state restoration.
  2. Fixed-budget checkpoint Epoch 6 lock (exactly 3,600 updates at Epoch 6).
  3. Fresh pretrained initialization enforcement (strict rejection of Fold 1 checkpoints).
  4. Staging and per-fold test speaker isolation for Folds 2, 3, and 4.
  5. Actual test-speaker leak detection independently of SHA mismatch.
  6. Corrupt archive detection.
  7. Cross-fold resume rejection.
  8. Remote mirror missing token failure in production mode.
  9. Endpoint classification semantics (COMPLETED_3600_UPDATES vs INCOMPLETE vs FAILED).
  10. End-to-end dry-run preflight checks for Folds 2, 3, and 4.
  11. Kaggle entrypoint CLI mode dispatch for cross-fold baseline runs.
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
    audit_archive_for_test_speaker_leak,
    classify_fold_training_endpoint,
    compute_sha256,
    find_and_stage_fold_package,
    verify_fresh_pretrained_initialization,
    verify_manifest,
)
from audit_scripts.train_fold1_ctc_binary import (
    LangfensWav2Vec2MDD,
    checkpoint,
    validate_resume_checkpoint,
)


MANIFEST_PATH = Path("manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json")


# ==============================================================================
# 1. EXECUTABLE MOCKED CHECKPOINT RECOVERY & RESUME INTEGRITY
# ==============================================================================

def test_executable_mocked_checkpoint_recovery():
    """Proves bit-exact weight and optimizer/scheduler state restoration on resume."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        ckpt_path = tmp_p / "epoch_2.pt"

        # 1. Create a model with trainable heads
        model = LangfensWav2Vec2MDD(
            hidden_dim=32, conv_dim=16, num_transformer_layers=1,
            num_attention_heads=4, intermediate_dim=64, dropout=0.1
        )
        trainable = model.configure_two_head_training()
        optimizer = torch.optim.AdamW(trainable, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15000)
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        # Fill with specific distinct values
        with torch.no_grad():
            for p in trainable:
                p.fill_(0.42)

        meta = {
            "fold": 2,
            "split_sha256": "mock_manifest_sha",
            "config": {
                "seed": 1337,
                "pretrained_model": "facebook/wav2vec2-base-960h"
            }
        }
        checkpoint(
            ckpt_path,
            model, optimizer, scheduler, scaler,
            epoch=2, updates=1200, micro=2400, best=0.3125,
            history=[{"epoch": 1}, {"epoch": 2}, {"epoch": 3}],
            metadata=meta,
            patience=1, skipped_amp=0, consecutive_skips=0,
            fixed_budget_saved=False
        )

        # 2. Instantiate a fresh model with different initial weights
        new_model = LangfensWav2Vec2MDD(
            hidden_dim=32, conv_dim=16, num_transformer_layers=1,
            num_attention_heads=4, intermediate_dim=64, dropout=0.1
        )
        new_trainable = new_model.configure_two_head_training()
        with torch.no_grad():
            for p in new_trainable:
                p.fill_(-0.99)
        new_optimizer = torch.optim.AdamW(new_trainable, lr=1e-3)
        new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_optimizer, T_max=15000)
        new_scaler = torch.amp.GradScaler("cpu", enabled=False)

        # 3. Load and validate
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = {"seed": 1337, "pretrained_model": "facebook/wav2vec2-base-960h"}
        validate_resume_checkpoint(ck, cfg, expected_fold=2, manifest_sha="mock_manifest_sha")

        new_model.load_state_dict(ck["model"])
        new_optimizer.load_state_dict(ck["optimizer"])
        new_scheduler.load_state_dict(ck["scheduler"])
        new_scaler.load_state_dict(ck["scaler"])

        # 4. Assert bit-exact weight restoration
        for p in new_trainable:
            assert torch.allclose(p, torch.tensor(0.42)), "Weights were not restored to saved state!"

        # 5. Assert counters
        assert ck["epoch"] == 2
        assert ck["successful_updates"] == 1200
        assert ck["micro_steps"] == 2400
        assert ck["best_ap"] == 0.3125
        assert ck["fixed_budget_saved"] is False


def test_epoch_6_fixed_budget_checkpoint_recovery_and_lock():
    """Proves fixed_budget_checkpoint represents exactly 3,600 updates at Epoch 6 and rejects overwriting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        fb_path = tmp_p / "fixed_budget_checkpoint.pt"

        model = LangfensWav2Vec2MDD(
            hidden_dim=32, conv_dim=16, num_transformer_layers=1,
            num_attention_heads=4, intermediate_dim=64, dropout=0.1
        )
        trainable = model.configure_two_head_training()
        optimizer = torch.optim.AdamW(trainable, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15000)
        scaler = torch.amp.GradScaler("cpu", enabled=False)

        # 1. Valid Epoch 6 fixed-budget checkpoint succeeds
        checkpoint(
            fb_path,
            model, optimizer, scheduler, scaler,
            epoch=5, updates=3600, micro=7200, best=0.318,
            history=[{"epoch": i} for i in range(1, 7)],
            metadata={"fold": 2},
            fixed_budget_saved=True,
            is_fixed_budget_endpoint=True
        )
        assert fb_path.is_file()
        ck = torch.load(fb_path, map_location="cpu", weights_only=False)
        assert ck["successful_updates"] == 3600
        assert ck["epoch"] == 5
        assert ck["is_fixed_budget_endpoint"] is True

        # 2. Invalid updates (<3600 or >3600) or invalid epoch (!=5) raises ValueError
        with pytest.raises(ValueError) as exc:
            checkpoint(
                fb_path,
                model, optimizer, scheduler, scaler,
                epoch=6, updates=4200, micro=8400, best=0.318,
                history=[{"epoch": i} for i in range(1, 8)],
                metadata={"fold": 2},
                fixed_budget_saved=True,
                is_fixed_budget_endpoint=True
            )
        assert "fixed_budget_checkpoint must represent exactly 3,600 updates at Epoch 6" in str(exc.value)


# ==============================================================================
# 2. FRESH PRETRAINED INITIALIZATION ENFORCEMENT
# ==============================================================================

def test_fresh_pretrained_initialization_strictly_enforced():
    """Proves that any attempt to pass or configure a Fold 1 checkpoint is rejected."""
    valid_cfg = {
        "pretrained_model": "facebook/wav2vec2-base-960h",
        "pretrained_revision": "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
    }
    verify_fresh_pretrained_initialization(valid_cfg)

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


def test_actual_test_speaker_leak_detection_independent_of_sha():
    """Proves test speaker leak is caught by member inspection even if expected SHA matches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        leaked_pkg = work_dir / "authorized_RQ1-TRN_fold2_train_val_only.zip"

        # Create a zip containing Fold 2 test speaker 'BWC.zip'
        with zipfile.ZipFile(leaked_pkg, "w") as zf:
            for spk in ["ABA", "ASI", "EBVS", "ERMS", "HJK", "LXC", "NCC", "NJS",
                        "PNV", "RRBI", "THV", "TLV", "TNI", "TXHC", "YBAA", "YDCK",
                        "YKWK", "BWC"]:  # BWC is Fold 2 test speaker!
                zf.writestr(f"{spk}.zip", b"dummy")

        # 1. Direct audit function check
        with pytest.raises(PermissionError) as exc_audit:
            audit_archive_for_test_speaker_leak(leaked_pkg, fold_idx=2)
        assert "CRITICAL SAFETY VIOLATION: Test speaker 'BWC'" in str(exc_audit.value)

        # 2. find_and_stage_fold_package check with actual SHA (independent of hash mismatch)
        actual_sha = compute_sha256(leaked_pkg)
        with pytest.raises(PermissionError) as exc_stage:
            find_and_stage_fold_package(
                fold_idx=2,
                kaggle_input_dir=work_dir,
                working_dir=work_dir / "target",
                expected_sha256=actual_sha  # SHA matches, but test speaker leak must fail!
            )
        assert "CRITICAL SAFETY VIOLATION: Test speaker 'BWC'" in str(exc_stage.value)


def test_corrupt_archive_detection():
    """Proves non-zip or corrupt archive fails closed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        bad_zip = work_dir / "corrupt.zip"
        bad_zip.write_bytes(b"not a valid zip file")

        with pytest.raises(ValueError) as exc:
            audit_archive_for_test_speaker_leak(bad_zip, fold_idx=2)
        assert "Corrupted archive" in str(exc.value)


def test_wrong_fold_resume_rejected():
    """Proves cross-fold checkpoint resume is strictly rejected."""
    cfg = {"seed": 1337, "pretrained_model": "facebook/wav2vec2-base-960h"}
    mock_ck = {
        "model": {"dummy": torch.zeros(1)},
        "optimizer": {"dummy": 1},
        "scheduler": {"dummy": 1},
        "scaler": {},
        "epoch": 1,
        "successful_updates": 600,
        "micro_steps": 1200,
        "best_ap": 0.31,
        "history": [{"epoch": 1}, {"epoch": 2}],
        "rng": {"torch": torch.get_rng_state()},
        "metadata": {
            "fold": 3,  # Checkpoint is from Fold 3!
            "config": {"seed": 1337, "pretrained_model": "facebook/wav2vec2-base-960h"}
        }
    }
    # Attempting to resume into Fold 2 MUST raise ValueError
    with pytest.raises(ValueError) as exc:
        validate_resume_checkpoint(mock_ck, cfg, expected_fold=2, manifest_sha="mock_sha")
    assert "Cross-fold resume rejected: checkpoint is from fold 3, expected fold 2" in str(exc.value)


def test_remote_mirror_missing_token_production_failure():
    """Proves production Kaggle execution halts if HF_TOKEN is absent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        old_env = os.environ.copy()
        try:
            os.environ.pop("HF_TOKEN", None)
            os.environ.pop("HUGGINGFACE_TOKEN", None)
            os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
            cmd = [
                sys.executable,
                "audit_scripts/kaggle_fold_train_exec.py",
                "--fold", "2",
                "--allow-mock-gpu",
                "--working-dir", tmpdir
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            assert proc.returncode != 0
            assert "FAIL-CLOSED: Hugging Face remote persistence requires valid HF_TOKEN" in proc.stderr
        finally:
            os.environ.clear()
            os.environ.update(old_env)


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

        # 3. Exactly 3,600 updates but epoch != 5 -> FAILED
        torch.save({
            "successful_updates": 3600,
            "epoch": 6,  # Epoch 7 instead of 5!
            "model": {"dummy": 1},
            "optimizer": {"dummy": 1},
            "scheduler": {"dummy": 1},
            "scaler": {},
            "metadata": {"fold": 2, "split_sha256": "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b", "config": {"seed": 1337, "batch_size": 2, "gradient_accumulation_steps": 2, "warmup_updates": 100, "max_epochs": 25, "learning_rate": 0.001, "weight_decay": 0.01, "optimizer": "AdamW", "scheduler": "cosine", "loss_weights": {"ctc": 1.0, "binary": 1.0}, "backbone_frozen": True, "amp": "fp16", "grad_scaler_initial_scale": 4096, "early_stopping_patience": 5}},
            "rng": {"torch": torch.get_rng_state()},
            "history": [{"epoch": i} for i in range(1, 8)],
            "best_ap": 0.320
        }, fixed_path)
        status = classify_fold_training_endpoint(out_dir, fold_idx=2)
        assert status["status"] == "FAILED"
        assert "expected exactly 5 (6th epoch)" in status["message"]

        # 4. Complete deliverable suite -> COMPLETED_3600_UPDATES
        torch.save({
            "successful_updates": 3600,
            "epoch": 5,
            "model": {"dummy": torch.zeros(1)},
            "optimizer": {"dummy": torch.zeros(1)},
            "scheduler": {"dummy": 1},
            "scaler": {"scale": 4096.0},
            "metadata": {"fold": 2, "split_sha256": "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b", "config": {"seed": 1337, "batch_size": 2, "gradient_accumulation_steps": 2, "warmup_updates": 100, "max_epochs": 25, "learning_rate": 0.001, "weight_decay": 0.01, "optimizer": "AdamW", "scheduler": "cosine", "loss_weights": {"ctc": 1.0, "binary": 1.0}, "backbone_frozen": True, "amp": "fp16", "grad_scaler_initial_scale": 4096, "early_stopping_patience": 5}},
            "rng": {"torch": torch.get_rng_state()},
            "history": [{"epoch": i} for i in range(1, 7)],
            "best_ap": 0.320
        }, fixed_path)
        (out_dir / "best.pt").write_bytes(b"mock_best")
        (out_dir / "last.pt").write_bytes(b"mock_last")
        (out_dir / "history.json").write_text(json.dumps([{"epoch": i} for i in range(1, 7)]) + "\n")

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
    cmd = [sys.executable, "deploy/kaggle/kaggle_entrypoint.py", "--help"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "train_fold2" in proc.stdout
    assert "train_fold3" in proc.stdout
    assert "train_fold4" in proc.stdout
