"""
LANGFENS — RQ1-TRN-03B: Kaggle Pilot Deployment & Execution Safety Tests.
Senior Speech ML Research Engineer & Reproducibility Auditor.

Verifies:
  1. Complete config -> trainer -> loss -> checkpoint execution path for Arm 2A and Arm 2B.
  2. Kaggle entrypoint dispatch for pilot modes (train_pilot_arm2a/b, verify_pilot_arm2a/b)
     and strict CLI token argument rejection.
  3. Kaggle fold runner (kaggle_fold_train_exec.py) acceptance of Fold 1 pilot modes.
  4. Fold 1 train/val archive SHA integrity, 18-speaker census, and zero test leakage.
  5. Arm-specific loss parameter verification and wrong-arm checkpoint classification rejection.
  6. Resume safety and wrong-arm checkpoint resumption rejection in trainer.
  7. Missing Kaggle Secrets / HF_TOKEN fail-closed safety.
  8. GPU device verification and non-Tesla T4 rejection.
  9. Metadata and notebook integrity across all 6 pilot deployment packages.
  10. Audit of pos_weight=8.40 derivation: distinguishing validation-informed operating choices
      from empirical training set class balance statistics.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
import pytest
import torch
import yaml

from audit_scripts.kaggle_checkpoint_sync import resolve_cross_fold_hf_token
from audit_scripts.kaggle_fold_train_exec import (
    EXPECTED_FOLD_PACKAGE_SHA256,
    SEALED_TEST_SPEAKERS_BY_FOLD,
    audit_archive_for_test_speaker_leak,
    classify_fold_training_endpoint,
    compute_sha256,
    find_and_stage_fold_package,
    verify_fresh_pretrained_initialization,
    verify_manifest,
)
from audit_scripts.kaggle_stage_and_verify import verify_kaggle_gpu_device
from audit_scripts.train_fold1_ctc_binary import (
    LangfensWav2Vec2MDD,
    checkpoint,
    rng_state,
    validate_resume_checkpoint,
)


MANIFEST_PATH = Path("manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json")
EXPECTED_MANIFEST_SHA = "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b"
EXPECTED_FOLD1_SHA = "b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9"


# ==============================================================================
# 1. KAGGLE ENTRYPOINT DISPATCH & TOKEN HYGIENE TESTS
# ==============================================================================

def test_kaggle_pilot_entrypoint_mode_dispatch():
    """Proves deploy/kaggle/kaggle_entrypoint.py parses and accepts all pilot modes."""
    cmd = [sys.executable, "deploy/kaggle/kaggle_entrypoint.py", "--help"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "train_pilot_arm2a" in proc.stdout
    assert "train_pilot_arm2b" in proc.stdout
    assert "verify_pilot_arm2a" in proc.stdout
    assert "verify_pilot_arm2b" in proc.stdout
    assert "train_fold1" in proc.stdout
    assert "verify_fold1" in proc.stdout


def test_kaggle_entrypoint_rejects_cli_token_for_pilot():
    """Proves deploy/kaggle/kaggle_entrypoint.py rejects CLI tokens for pilot runs."""
    for mode in ["train_pilot_arm2a", "train_pilot_arm2b", "verify_pilot_arm2a", "verify_pilot_arm2b"]:
        cmd = [
            sys.executable,
            "deploy/kaggle/kaggle_entrypoint.py",
            "--mode", mode,
            "--commit-sha", "56bf79535e4f697b2eea05594d5d81153e04dbff",
            "--remote-mirror-token", "hf_forbidden_cli_token_12345",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode != 0
        assert "FAIL-CLOSED: CLI token argument '--remote-mirror-token' is strictly prohibited" in proc.stderr


# ==============================================================================
# 2. KAGGLE FOLD RUNNER PILOT SUPPORT (FOLD 1)
# ==============================================================================

def test_kaggle_fold_train_exec_dry_run_pilot_arm2a():
    """Proves kaggle_fold_train_exec.py successfully dry-runs Arm 2A Focal pilot on Fold 1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable,
            "audit_scripts/kaggle_fold_train_exec.py",
            "--fold", "1",
            "--arm", "Arm2A_Focal",
            "--dry-run",
            "--working-dir", tmpdir
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, f"Arm 2A pilot dry-run failed:\n{proc.stderr}"
        assert "LANGFENS — Kaggle T4 Runner for Fold 1 (Arm: Arm2A_Focal)" in proc.stdout
        assert "Verified fresh pretrained initialization from facebook/wav2vec2-base-960h" in proc.stdout
        assert "[DRY-RUN] Preflight checks passed." in proc.stdout


def test_kaggle_fold_train_exec_dry_run_pilot_arm2b():
    """Proves kaggle_fold_train_exec.py successfully dry-runs Arm 2B Weighted pilot on Fold 1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable,
            "audit_scripts/kaggle_fold_train_exec.py",
            "--fold", "1",
            "--arm", "Arm2B_Weighted",
            "--dry-run",
            "--working-dir", tmpdir
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, f"Arm 2B pilot dry-run failed:\n{proc.stderr}"
        assert "LANGFENS — Kaggle T4 Runner for Fold 1 (Arm: Arm2B_Weighted)" in proc.stdout
        assert "Verified fresh pretrained initialization from facebook/wav2vec2-base-960h" in proc.stdout
        assert "[DRY-RUN] Preflight checks passed." in proc.stdout


# ==============================================================================
# 3. FOLD 1 ARCHIVE INTEGRITY & ZERO ACTIVE TEST LEAKAGE
# ==============================================================================

def test_fold1_archive_sha_and_test_speaker_isolation():
    """Proves Fold 1 archive matches pinned SHA-256 and has zero active test speaker leakage."""
    archive_path = Path("datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip")
    assert archive_path.is_file(), f"Missing Fold 1 archive: {archive_path}"

    # 1. Cryptographic SHA-256 check
    actual_sha = compute_sha256(archive_path)
    assert actual_sha == EXPECTED_FOLD1_SHA, f"Fold 1 SHA mismatch: {actual_sha} != {EXPECTED_FOLD1_SHA}"

    # 2. Archive structure and test exclusion
    audit_archive_for_test_speaker_leak(archive_path, fold_idx=1)

    with zipfile.ZipFile(archive_path, "r") as zf:
        members = zf.namelist()
        packaged_spks = sorted([m.replace(".zip", "") for m in members if m.endswith(".zip")])

    assert len(packaged_spks) == 18, f"Expected 18 inner speaker archives, got {len(packaged_spks)}"

    test_spks = SEALED_TEST_SPEAKERS_BY_FOLD[1]
    assert test_spks == {"ABA", "ASI", "EBVS", "HJK", "LXC", "PNV"}
    overlap = set(packaged_spks) & test_spks
    assert len(overlap) == 0, f"CRITICAL LEAK: Test speakers found in Fold 1 archive: {overlap}"


# ==============================================================================
# 4. ARM-SPECIFIC LOSS VERIFICATION & ENDPOINT CLASSIFIER SAFETY
# ==============================================================================

def test_endpoint_classifier_verifies_arm2a_and_arm2b_loss_contracts():
    """Proves classify_fold_training_endpoint validates arm-specific loss parameters and rejects mismatches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_p = Path(tmpdir)
        fixed_pt = out_p / "fixed_budget_checkpoint.pt"
        best_pt = out_p / "best.pt"
        last_pt = out_p / "last.pt"
        hist_file = out_p / "history.json"

        # Touch dummy companion files
        best_pt.write_bytes(b"dummy_best")
        last_pt.write_bytes(b"dummy_last")
        hist_data = [{"epoch": i, "val_ap": 0.3} for i in range(6)]
        hist_file.write_text(json.dumps(hist_data))

        # Base valid payload
        base_cfg = {
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

        # 1. Valid Arm 2A Checkpoint
        cfg_arm2a = dict(base_cfg)
        cfg_arm2a.update({
            "arm": "Arm2A_Focal",
            "binary_loss_type": "focal",
            "focal_gamma": 2.0,
            "focal_alpha": 0.75,
        })
        payload_arm2a = {
            "model": {"dummy": torch.tensor([1.0])},
            "optimizer": {"dummy": torch.tensor([1.0])},
            "scheduler": {"dummy": torch.tensor([1.0])},
            "scaler": {"scale": 4096.0},
            "epoch": 5,
            "successful_updates": 3600,
            "metadata": {"fold": 1, "split_sha256": EXPECTED_MANIFEST_SHA, "config": cfg_arm2a},
            "rng": {},
            "history": hist_data,
        }
        torch.save(payload_arm2a, fixed_pt)

        # Classification as Arm 2A succeeds
        res_a = classify_fold_training_endpoint(out_p, fold_idx=1, expected_arm="Arm2A_Focal")
        assert res_a["status"] == "COMPLETED_3600_UPDATES"

        # Classification of Arm 2A as Arm 2B fails closed
        res_wrong_arm = classify_fold_training_endpoint(out_p, fold_idx=1, expected_arm="Arm2B_Weighted")
        assert res_wrong_arm["status"] == "FAILED"
        assert "Arm mismatch" in res_wrong_arm["message"]

        # 2. Corrupted Focal Gamma fails closed
        cfg_arm2a_bad = dict(cfg_arm2a)
        cfg_arm2a_bad["focal_gamma"] = 1.0
        payload_arm2a_bad = dict(payload_arm2a)
        payload_arm2a_bad["metadata"] = {"fold": 1, "split_sha256": EXPECTED_MANIFEST_SHA, "config": cfg_arm2a_bad}
        torch.save(payload_arm2a_bad, fixed_pt)
        res_bad_gamma = classify_fold_training_endpoint(out_p, fold_idx=1, expected_arm="Arm2A_Focal")
        assert res_bad_gamma["status"] == "FAILED"
        assert "focal loss parameter violation" in res_bad_gamma["message"]

        # 3. Valid Arm 2B Checkpoint
        cfg_arm2b = dict(base_cfg)
        cfg_arm2b.update({
            "arm": "Arm2B_Weighted",
            "binary_loss_type": "bce",
            "binary_pos_weight": 8.40,
        })
        payload_arm2b = {
            "model": {"dummy": torch.tensor([1.0])},
            "optimizer": {"dummy": torch.tensor([1.0])},
            "scheduler": {"dummy": torch.tensor([1.0])},
            "scaler": {"scale": 4096.0},
            "epoch": 5,
            "successful_updates": 3600,
            "metadata": {"fold": 1, "split_sha256": EXPECTED_MANIFEST_SHA, "config": cfg_arm2b},
            "rng": {},
            "history": hist_data,
        }
        torch.save(payload_arm2b, fixed_pt)
        res_b = classify_fold_training_endpoint(out_p, fold_idx=1, expected_arm="Arm2B_Weighted")
        assert res_b["status"] == "COMPLETED_3600_UPDATES"

        # 4. Corrupted pos_weight fails closed
        cfg_arm2b_bad = dict(cfg_arm2b)
        cfg_arm2b_bad["binary_pos_weight"] = 5.0
        payload_arm2b_bad = dict(payload_arm2b)
        payload_arm2b_bad["metadata"] = {"fold": 1, "split_sha256": EXPECTED_MANIFEST_SHA, "config": cfg_arm2b_bad}
        torch.save(payload_arm2b_bad, fixed_pt)
        res_bad_pos = classify_fold_training_endpoint(out_p, fold_idx=1, expected_arm="Arm2B_Weighted")
        assert res_bad_pos["status"] == "FAILED"
        assert "class-weight parameter violation" in res_bad_pos["message"]


# ==============================================================================
# 5. RESUME INTEGRITY & WRONG-ARM RESUME REJECTION
# ==============================================================================

def test_validate_resume_checkpoint_rejects_wrong_arm_checkpoints():
    """Proves validate_resume_checkpoint strictly rejects resuming across different loss arms."""
    base_cfg = {
        "fold": 1,
        "seed": 1337,
        "batch_size": 2,
        "gradient_accumulation_steps": 2,
        "warmup_updates": 100,
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "loss_weights": {"ctc": 1.0, "binary": 1.0},
        "backbone_frozen": True,
        "amp": "fp16",
        "early_stopping_patience": 5,
        "grad_scaler_initial_scale": 4096,
        "pretrained_model": "facebook/wav2vec2-base-960h",
        "pretrained_revision": "22aad52d435eb6dbaf354bdad9b0da84ce7d6156",
    }

    # Arm 2A expected config
    cfg_expected_arm2a = dict(base_cfg)
    cfg_expected_arm2a.update({
        "arm": "Arm2A_Focal",
        "binary_loss_type": "focal",
        "focal_gamma": 2.0,
        "focal_alpha": 0.75,
    })

    # Checkpoint from Arm 2B
    cfg_ck_arm2b = dict(base_cfg)
    cfg_ck_arm2b.update({
        "arm": "Arm2B_Weighted",
        "binary_loss_type": "bce",
        "binary_pos_weight": 8.40,
    })

    hist = [
        {"epoch": 1, "successful_updates_total": 600},
        {"epoch": 2, "successful_updates_total": 1200},
        {"epoch": 3, "successful_updates_total": 1200},
    ]
    ck_payload = {
        "model": {"dummy": torch.tensor([1.0])},
        "optimizer": {"dummy": torch.tensor([1.0])},
        "scheduler": {"dummy": torch.tensor([1.0])},
        "scaler": {},
        "epoch": 2,
        "successful_updates": 1200,
        "micro_steps": 2400,
        "history": hist,
        "rng": rng_state(),
        "metadata": {
            "fold": 1,
            "split_sha256": EXPECTED_MANIFEST_SHA,
            "git_sha": "56bf79535e4f697b2eea05594d5d81153e04dbff",
            "config": cfg_ck_arm2b,
        }
    }

    # Resuming Arm 2A from Arm 2B checkpoint MUST fail closed
    with pytest.raises(ValueError, match="Hyperparameter mismatch on resume for 'binary_loss_type'"):
        validate_resume_checkpoint(ck_payload, cfg_expected_arm2a, expected_fold=1, manifest_sha=EXPECTED_MANIFEST_SHA)

    # Resuming Arm 2A from valid Arm 2A checkpoint succeeds
    ck_payload["metadata"]["config"] = cfg_expected_arm2a
    validate_resume_checkpoint(ck_payload, cfg_expected_arm2a, expected_fold=1, manifest_sha=EXPECTED_MANIFEST_SHA)


# ==============================================================================
# 6. MISSING SECRETS & GPU HARDWARE FAILURE SAFETY
# ==============================================================================

def test_missing_secrets_and_gpu_failure_safety(monkeypatch):
    """Proves missing HF_TOKEN and missing/non-T4 GPU fail closed as required by AGENTS.md."""
    # 1. Missing HF_TOKEN
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert resolve_cross_fold_hf_token() is None

    # 2. Non-Tesla T4 GPU rejection
    with pytest.raises(RuntimeError, match="FAIL-CLOSED"):
        # Without allow_mock, on CPU laptop verify_kaggle_gpu_device fails closed
        verify_kaggle_gpu_device(allow_mock=False)


# ==============================================================================
# 7. ALL SIX PILOT DEPLOYMENT PACKAGES INTEGRITY
# ==============================================================================

def test_all_six_pilot_packages_exist_and_conform_to_standards():
    """Verifies all 6 pilot deployment packages have valid metadata, notebooks, and isolation."""
    expected_packages = [
        ("deploy/kaggle_pilot_arm2a_preflight", "minkhoaa/langfens-rq1-pilot-arm2a-focal-t4x2-preflight", "kaggle_preflight.ipynb"),
        ("deploy/kaggle_pilot_arm2a_smoke", "minkhoaa/langfens-rq1-pilot-arm2a-focal-t4x2-smoke", "kaggle_smoke.ipynb"),
        ("deploy/kaggle_pilot_arm2a_train", "minkhoaa/langfens-rq1-pilot-arm2a-focal-t4x2", "kaggle_train.ipynb"),
        ("deploy/kaggle_pilot_arm2b_preflight", "minkhoaa/langfens-rq1-pilot-arm2b-weighted-t4x2-preflight", "kaggle_preflight.ipynb"),
        ("deploy/kaggle_pilot_arm2b_smoke", "minkhoaa/langfens-rq1-pilot-arm2b-weighted-t4x2-smoke", "kaggle_smoke.ipynb"),
        ("deploy/kaggle_pilot_arm2b_train", "minkhoaa/langfens-rq1-pilot-arm2b-weighted-t4x2", "kaggle_train.ipynb"),
    ]

    seen_slugs = set()
    for pkg_dir_str, expected_slug, nb_filename in expected_packages:
        pkg_dir = Path(pkg_dir_str)
        assert pkg_dir.is_dir(), f"Missing package directory: {pkg_dir}"

        # 1. kernel-metadata.json
        k_meta_file = pkg_dir / "kernel-metadata.json"
        assert k_meta_file.is_file(), f"Missing {k_meta_file}"
        with open(k_meta_file) as f:
            k_meta = json.load(f)

        assert k_meta["id"] == expected_slug
        assert k_meta["id"] not in seen_slugs, f"Duplicate slug: {expected_slug}"
        seen_slugs.add(k_meta["id"])

        assert k_meta["is_private"] is True
        assert k_meta["enable_gpu"] is True
        assert k_meta["enable_internet"] is True
        assert k_meta["machine_shape"] == "NvidiaTeslaT4"
        assert k_meta["dataset_sources"] == ["minkhoaa/langfens-rq1-fold1-staging"]
        assert k_meta["code_file"] == nb_filename

        # 2. dataset-metadata.json
        d_meta_file = pkg_dir / "dataset-metadata.json"
        assert d_meta_file.is_file(), f"Missing {d_meta_file}"
        with open(d_meta_file) as f:
            d_meta = json.load(f)
        assert d_meta["id"] == "minkhoaa/langfens-rq1-fold1-staging"
        assert d_meta["is_private"] is True

        # 3. Notebook JSON validity and hygiene
        nb_file = pkg_dir / nb_filename
        assert nb_file.is_file(), f"Missing {nb_file}"
        with open(nb_file) as f:
            nb = json.load(f)

        assert "cells" in nb and len(nb["cells"]) >= 4
        all_code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")

        # Zero disk search for tokens
        assert ".hf_token" not in all_code
        assert "hf_token.txt" not in all_code
        assert "--remote-mirror-token" not in all_code

        # Verified Kaggle Secrets acquisition
        assert "UserSecretsClient().get_secret(\"HF_TOKEN\")" in all_code
        # Verified Pinned commit SHA present and matches reviewed training commit
        assert 'PINNED_COMMIT_SHA = "56bf79535e4f697b2eea05594d5d81153e04dbff"' in all_code


def test_notebook_commit_pinning_integrity_and_stale_commit_detection():
    """
    Rigorously verifies that all 6 pilot notebooks pin the reviewed commit that introduced
    pilot execution modes (56bf795), and proves detection of stale commits (e.g. 051a870).
    """
    expected_packages = [
        ("deploy/kaggle_pilot_arm2a_preflight", "kaggle_preflight.ipynb"),
        ("deploy/kaggle_pilot_arm2a_smoke", "kaggle_smoke.ipynb"),
        ("deploy/kaggle_pilot_arm2a_train", "kaggle_train.ipynb"),
        ("deploy/kaggle_pilot_arm2b_preflight", "kaggle_preflight.ipynb"),
        ("deploy/kaggle_pilot_arm2b_smoke", "kaggle_smoke.ipynb"),
        ("deploy/kaggle_pilot_arm2b_train", "kaggle_train.ipynb"),
    ]

    target_commit = "56bf79535e4f697b2eea05594d5d81153e04dbff"
    stale_commit = "051a870c2388e852b628fae9814056b55bce5051"

    # 1. Verify target commit exists in git
    rev = subprocess.check_output(["git", "rev-parse", target_commit], text=True).strip()
    assert rev == target_commit

    # 2. Verify target commit contains pilot modes in deploy/kaggle/kaggle_entrypoint.py
    entry_code = subprocess.check_output(["git", "show", f"{target_commit}:deploy/kaggle/kaggle_entrypoint.py"], text=True)
    assert "train_pilot_arm2a" in entry_code
    assert "train_pilot_arm2b" in entry_code
    assert "verify_pilot_arm2a" in entry_code
    assert "verify_pilot_arm2b" in entry_code

    # 3. Negative control: Prove stale commit lacks pilot modes
    stale_entry_code = subprocess.check_output(["git", "show", f"{stale_commit}:deploy/kaggle/kaggle_entrypoint.py"], text=True)
    assert "train_pilot_arm2a" not in stale_entry_code
    assert "train_pilot_arm2b" not in stale_entry_code

    # 4. Check every notebook pins the target commit and NOT the stale commit
    for pkg_dir_str, nb_filename in expected_packages:
        nb_path = Path(pkg_dir_str) / nb_filename
        with open(nb_path) as f:
            nb = json.load(f)

        all_text = "".join("".join(c["source"]) for c in nb["cells"])
        assert target_commit in all_text, f"{nb_path} missing target commit {target_commit}"
        assert stale_commit not in all_text, f"{nb_path} contains stale commit {stale_commit}"


# ==============================================================================
# 8. POS_WEIGHT=8.40 DERIVATION & OPERATING CHOICE AUDIT
# ==============================================================================

def test_pos_weight_derivation_audit():
    """
    Rigorously verifies the origin of pos_weight = 8.40:
    - Validation Macro Prevalence: 10.6430% -> (1 - 0.10643)/0.10643 = 8.396 -> 8.40
    - Train Macro Prevalence: 15.2074% -> (1 - 0.15207)/0.15207 = 5.576
    - Fold 1 Train Prevalence: 14.9082% -> (1 - 0.14908)/0.14908 = 5.707
    Proves that 8.40 is an intentional 1.47x over-weighting targeting Recall elevation.
    """
    with open(MANIFEST_PATH) as f:
        splits = json.load(f)

    # 1. Validation statistics across 4 folds
    val_rates = [f["val"]["error_rate_pct"] / 100.0 for f in splits["folds"]]
    macro_val_prev = sum(val_rates) / len(val_rates)
    assert pytest.approx(macro_val_prev, abs=1e-5) == 0.106430

    w_val_derived = (1.0 - macro_val_prev) / macro_val_prev
    assert pytest.approx(w_val_derived, abs=1e-2) == 8.40

    # 2. Train statistics across 4 folds
    train_rates = [f["train"]["error_rate_pct"] / 100.0 for f in splits["folds"]]
    macro_train_prev = sum(train_rates) / len(train_rates)
    assert pytest.approx(macro_train_prev, abs=1e-4) == 0.152074

    f1_train_prev = train_rates[0]
    assert pytest.approx(f1_train_prev, abs=1e-4) == 0.149082

    w_train_f1 = (1.0 - f1_train_prev) / f1_train_prev
    assert pytest.approx(w_train_f1, abs=1e-2) == 5.71

    # Overweighting ratio
    overweight_ratio = 8.40 / w_train_f1
    assert pytest.approx(overweight_ratio, abs=1e-2) == 1.47
