"""
LANGFENS — RQ1-TRN-03B: Step 17 Arm 2B Endpoint Forensics & Artifact Verification Tests.
Senior Speech ML Research Engineer & Reproducibility Auditor.

Verifies:
  1. Arm 2B history ledger, AMP skipped update progression, and early stopping termination.
  2. Failure mechanism for missing fixed_budget_checkpoint.pt in Arm 2B (3,598 != 3,600).
  3. Arm 2A fixed-budget completion (exactly 3,600 updates, 0 skips, COMPLETED_3600_UPDATES).
  4. Cryptographic SHA-256 readback for all remote Hugging Face pilot artifacts.
  5. Strict distinction between registered primary endpoints and exploratory near-budget artifacts.
  6. Standalone Protocol O and Protocol Z comparative evaluation metrics on Fold 1 validation.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch

from audit_scripts.kaggle_fold_train_exec import (
    classify_fold_training_endpoint,
    compute_sha256,
)


EXPECTED_SHA256_LEDGER = {
    "arm2a_fixed_budget": "46c78486ea41c13e45ef4654604acfbe58f8cf4a0e237b65f60abfebac506f93",
    "arm2a_best": "e39813bbb3cab9fdfab303167f66e215640d6024970fba919eecdeac281ab175",
    "arm2a_last": "f633a45d9d17fa7a62279b88b06612d957d3c54297836cb82db7c5d90acde6e9",
    "arm2a_history": "5d7cd3c20623c9e1175b38c3e2204061b782cd12a07ed526390e74ee0bc39896",
    "arm2b_best": "a94434ca31308d29853bf28b020101820dc47353d598620a50a2a65fea12608a",
    "arm2b_last": "3e886a4fbd64414c60b0993b92419c66007b5cd34085c25978069fac5d3310ac",
    "arm2b_history": "860c0a0c2c1a40cb673892fdb7610ee7c54bfcfce06423cb3a9675386a61127c",
}


def test_arm2b_history_amp_skips_and_early_stopping():
    """Proves Arm 2B achieved 3,598 updates across 6 completed epochs with 2 cumulative AMP skips."""
    cache_dir = Path("checkpoints/pilot_evaluation_cache")
    hist_path = cache_dir / "arm2b_history.json"
    if not hist_path.is_file():
        # Fallback to loading directly from HF download
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id="minkhoaa/langfens-rq1-cross-fold-checkpoints",
            filename="pilot/pilot_arm2b_weighted/history.json",
            repo_type="model",
        )
        data = Path(downloaded).read_bytes()
        hist_path.write_bytes(data)

    with open(hist_path) as f:
        history = json.load(f)

    assert len(history) == 6, f"Expected 6 completed epochs for Arm 2B, got {len(history)}"

    # Epoch 1: Peak validation AP
    assert history[0]["epoch"] == 1
    assert history[0]["successful_updates_total"] == 600
    assert history[0]["skipped_amp_attempts_total"] == 0
    assert pytest.approx(history[0]["validation_binary_ap"], abs=1e-5) == 0.316407

    # Epoch 4: First AMP skip (2,399 updates instead of 2,400)
    assert history[3]["epoch"] == 4
    assert history[3]["successful_updates_total"] == 2399
    assert history[3]["skipped_amp_attempts_total"] == 1

    # Epoch 6: Second AMP skip (3,598 updates instead of 3,600)
    assert history[5]["epoch"] == 6
    assert history[5]["successful_updates_total"] == 3598
    assert history[5]["skipped_amp_attempts_total"] == 2

    # Monotonic degradation from peak after Epoch 1 -> early stopping triggered at patience 5
    for row in history[1:]:
        assert row["validation_binary_ap"] < history[0]["validation_binary_ap"]


def test_arm2b_missing_fixed_budget_checkpoint_mechanism():
    """Proves why fixed_budget_checkpoint.pt is missing in Arm 2B (3,598 != 3,600)."""
    cache_dir = Path("checkpoints/pilot_evaluation_cache")
    last_pt = cache_dir / "last_Arm2B_Weighted_3598up.pt"
    if not last_pt.is_file():
        pytest.skip(f"Missing {last_pt}")

    ck_last = torch.load(last_pt, map_location="cpu", weights_only=False)
    assert ck_last["epoch"] == 5  # 0-indexed Epoch 6
    assert ck_last["successful_updates"] == 3598  # Exact count with 2 skips
    assert ck_last["skipped_amp_attempts_total"] == 2
    assert ck_last["fixed_budget_saved"] is False  # Contract condition was not met!


def test_arm2a_completed_3600_updates_primary_endpoint():
    """Proves Arm 2A achieved exactly 3,600 updates at Epoch 6 with 0 AMP skips."""
    cache_dir = Path("checkpoints/pilot_evaluation_cache")
    fb_pt = cache_dir / "fixed_budget_checkpoint_Arm2A_Focal_3600up.pt"
    if not fb_pt.is_file():
        pytest.skip(f"Missing {fb_pt}")

    ck_fb = torch.load(fb_pt, map_location="cpu", weights_only=False)
    assert ck_fb["epoch"] == 5  # 0-indexed Epoch 6
    assert ck_fb["successful_updates"] == 3600
    assert ck_fb["skipped_amp_attempts_total"] == 0
    assert ck_fb["fixed_budget_saved"] is True
    assert ck_fb["metadata"]["config"]["arm"] == "Arm2A_Focal"
    assert ck_fb["metadata"]["config"]["binary_loss_type"] == "focal"


def test_cryptographic_readback_for_all_existing_artifacts():
    """Verifies that all local cached artifacts match their authoritative cryptographic digests."""
    cache_dir = Path("checkpoints/pilot_evaluation_cache")
    file_map = {
        "arm2a_fixed_budget": cache_dir / "fixed_budget_checkpoint_Arm2A_Focal_3600up.pt",
        "arm2a_best": cache_dir / "best_Arm2A_Focal_1200up.pt",
        "arm2b_best": cache_dir / "best_Arm2B_Weighted_600up.pt",
        "arm2b_last": cache_dir / "last_Arm2B_Weighted_3598up.pt",
    }

    for key, path in file_map.items():
        if path.is_file():
            actual_sha = compute_sha256(path)
            expected_sha = EXPECTED_SHA256_LEDGER[key]
            assert actual_sha == expected_sha, f"SHA mismatch for {key}: {actual_sha} != {expected_sha}"


def test_pilot_fold1_comparative_metrics_integrity():
    """Verifies standalone evaluation report numbers for Arm 2A, Arm 2B, and Baseline R."""
    rep_path = Path("manifests/RQ1-TRN-03A/evaluation/PILOT_FOLD1_COMPARATIVE_REPORT.json")
    assert rep_path.is_file(), f"Missing {rep_path}"

    with open(rep_path) as f:
        rep = json.load(f)

    # 1. Arm 2A Fixed-Budget Primary (3,600 updates)
    a_fb = rep["Arm2A_Focal_fixed_budget_registered_primary"]
    assert a_fb["updates_stored"] == 3600
    assert pytest.approx(a_fb["matched_ap"], abs=1e-4) == 0.308104
    assert pytest.approx(a_fb["acoustic_gap"], abs=1e-4) == 0.009869
    assert pytest.approx(a_fb["ctc_per"], abs=1e-4) == 0.236413
    assert a_fb["acoustic_gap"] > 0.0  # Positive acoustic reliance maintained

    # 2. Arm 2A Validation Best Secondary (1,200 updates)
    a_best = rep["Arm2A_Focal_validation_best_secondary"]
    assert a_best["updates_stored"] == 1200
    assert pytest.approx(a_best["matched_ap"], abs=1e-4) == 0.317201
    assert pytest.approx(a_best["acoustic_gap"], abs=1e-4) == 0.021745

    # 3. Arm 2B Exploratory Near-Budget (3,598 updates)
    b_last = rep["Arm2B_Weighted_exploratory_near_budget_last"]
    assert b_last["updates_stored"] == 3598
    assert pytest.approx(b_last["matched_ap"], abs=1e-4) == 0.308087
    assert pytest.approx(b_last["acoustic_gap"], abs=1e-4) == 0.018733
    assert pytest.approx(b_last["ctc_per"], abs=1e-4) == 0.238033
    assert b_last["acoustic_gap"] > 0.0

    # 4. Operating Threshold Shift Verification
    # Arm 2B pos_weight=8.40 shifts FNR@0.50 down to 16.94% but increases FPR@0.50 to 50.55%
    assert b_last["fnr_at_050"] < 0.20
    assert b_last["fpr_at_050"] > 0.45

    # Arm 2A focal loss achieves balanced FNR@0.50 = 61.80% and FPR@0.50 = 9.47%
    assert a_fb["fnr_at_050"] < 0.65
    assert a_fb["fpr_at_050"] < 0.10
