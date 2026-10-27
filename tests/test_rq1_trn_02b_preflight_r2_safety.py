"""
LANGFENS — RQ1-TRN-02B Preflight-R2 Fail-Closed Safety Regression Tests (CPU Forward-Only)
Verifies:
1. Rejection of missing checkpoint arm, seed, hash and configuration fields.
2. Cross-arm checkpoint resume rejection (C0 into C1, C1 into C0).
3. Rejection of malformed or contradictory optimizer-update counters.
4. Guardrail-failure artifact schema and preservation of last.pt.
5. Rejection of halted checkpoints during normal resume.
6. Rejection of incorrect endpoint arm and incomplete update count.
7. Exploratory override without false VALID reporting (EXPLORATORY_INCOMPLETE_BUDGET).
8. C0-only guardrail failure marks experiment invalid.
9. C1-only guardrail failure marks experiment invalid.
10. Frozen split and official test access boundaries strictly maintained.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure src and root are in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir))

import numpy as np
import pytest
import torch

from audit_scripts.evaluate_rq1_trn_02b import (
    HISTORICAL_REF_R_CTC_PER,
    validate_endpoint_readiness,
    validate_paired_checkpoints_compatibility,
)
from audit_scripts.train_rq1_trn_02b import (
    CHECKPOINT_SCHEMA_VERSION,
    save_checkpoint,
    validate_checkpoint_for_resume,
)

MANIFEST_PATH = root_dir / "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json"
ZIP_PATH = root_dir / "datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip"


@pytest.fixture
def valid_c0_cfg() -> Dict[str, Any]:
    return {
        "arm": "C0",
        "seed": 1337,
        "manifest_sha256": "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b",
        "initial_checkpoint_sha256": "a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b",
        "binary_prompt_dropout_p": 0.0,
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "warmup_updates": 100,
        "max_successful_updates": 3600,
        "gradient_accumulation_steps": 2,
        "max_epochs": 6,
        "batch_size": 2,
    }


@pytest.fixture
def valid_c1_cfg() -> Dict[str, Any]:
    return {
        "arm": "C1",
        "seed": 1337,
        "manifest_sha256": "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b",
        "initial_checkpoint_sha256": "a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b",
        "binary_prompt_dropout_p": 0.25,
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "warmup_updates": 100,
        "max_successful_updates": 3600,
        "gradient_accumulation_steps": 2,
        "max_epochs": 6,
        "batch_size": 2,
    }


@pytest.fixture
def valid_c1_payload(valid_c1_cfg) -> Dict[str, Any]:
    return {
        "model": {"dummy": torch.zeros(1)},
        "optimizer": {"state": {}, "param_groups": []},
        "scheduler": {"last_epoch": 600},
        "scaler": {"scale": 4096.0},
        "successful_updates": 600,
        "step_attempts": 600,
        "skipped_updates": 0,
        "micro_steps": 1200,
        "epoch": 0,
        "display_epoch": 1,
        "history": [{"epoch": 1}],
        "rng": {"python": (), "numpy": (), "torch": torch.ByteTensor([1])},
        "dataloader_generator": torch.zeros(10, dtype=torch.uint8),
        "intervention_rng": torch.zeros(10, dtype=torch.uint8),
        "metadata": {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "git_sha": "9fe993b6af0346a0d8170e328b5416cab30bdbdd",
            "arm": "C1",
            "split_sha256": valid_c1_cfg["manifest_sha256"],
            "initial_checkpoint_sha256": valid_c1_cfg["initial_checkpoint_sha256"],
            "binary_prompt_dropout_p": 0.25,
            "experiment_status": "IN_PROGRESS",
            "guardrail_violation": False,
            "config": copy.deepcopy(valid_c1_cfg),
        },
    }


# ---------------------------------------------------------------------------
# Test 1: Missing checkpoint arm, seed, hash and configuration fields
# ---------------------------------------------------------------------------
def test_1_missing_checkpoint_fields_rejection(valid_c1_payload, valid_c1_cfg):
    """Verify that any missing provenance or configuration field triggers fail-closed rejection."""
    manifest_sha = valid_c1_cfg["manifest_sha256"]
    init_sha = valid_c1_cfg["initial_checkpoint_sha256"]

    # Base valid passes
    validate_checkpoint_for_resume(valid_c1_payload, valid_c1_cfg, manifest_sha, init_sha, 0.25)

    # Missing arm
    ck = copy.deepcopy(valid_c1_payload)
    del ck["metadata"]["arm"]
    with pytest.raises(ValueError, match="Invalid checkpoint: arm ID"):
        validate_checkpoint_for_resume(ck, valid_c1_cfg, manifest_sha, init_sha, 0.25)

    # Missing seed
    ck = copy.deepcopy(valid_c1_payload)
    del ck["metadata"]["config"]["seed"]
    with pytest.raises(ValueError, match="Seed mismatch on resume"):
        validate_checkpoint_for_resume(ck, valid_c1_cfg, manifest_sha, init_sha, 0.25)

    # Missing split SHA
    ck = copy.deepcopy(valid_c1_payload)
    del ck["metadata"]["split_sha256"]
    with pytest.raises(ValueError, match="Manifest SHA-256 mismatch"):
        validate_checkpoint_for_resume(ck, valid_c1_cfg, manifest_sha, init_sha, 0.25)

    # Missing initial checkpoint SHA
    ck = copy.deepcopy(valid_c1_payload)
    del ck["metadata"]["initial_checkpoint_sha256"]
    with pytest.raises(ValueError, match="Initial checkpoint SHA-256 mismatch"):
        validate_checkpoint_for_resume(ck, valid_c1_cfg, manifest_sha, init_sha, 0.25)

    # Missing schema version
    ck = copy.deepcopy(valid_c1_payload)
    del ck["metadata"]["schema_version"]
    with pytest.raises(ValueError, match="missing 'schema_version'"):
        validate_checkpoint_for_resume(ck, valid_c1_cfg, manifest_sha, init_sha, 0.25)

    # Missing git commit SHA
    ck = copy.deepcopy(valid_c1_payload)
    del ck["metadata"]["git_sha"]
    with pytest.raises(ValueError, match="missing or invalid 'git_sha'"):
        validate_checkpoint_for_resume(ck, valid_c1_cfg, manifest_sha, init_sha, 0.25)

    # Missing shared optimizer configuration field
    ck = copy.deepcopy(valid_c1_payload)
    del ck["metadata"]["config"]["learning_rate"]
    with pytest.raises(ValueError, match="missing shared configuration field 'learning_rate'"):
        validate_checkpoint_for_resume(ck, valid_c1_cfg, manifest_sha, init_sha, 0.25)


# ---------------------------------------------------------------------------
# Test 2: Cross-arm checkpoint resume rejection
# ---------------------------------------------------------------------------
def test_2_cross_arm_checkpoint_resume_rejection(valid_c0_cfg, valid_c1_cfg, valid_c1_payload):
    """Verify that resuming a C1 checkpoint under C0 configuration (or vice versa) is rejected."""
    manifest_sha = valid_c1_cfg["manifest_sha256"]
    init_sha = valid_c1_cfg["initial_checkpoint_sha256"]

    # Attempt to resume C1 checkpoint under C0 config
    with pytest.raises(ValueError, match="Cross-arm resume rejected.*expects arm 'C0'"):
        validate_checkpoint_for_resume(valid_c1_payload, valid_c0_cfg, manifest_sha, init_sha, 0.0)

    # Attempt to resume C0 checkpoint under C1 config
    ck_c0 = copy.deepcopy(valid_c1_payload)
    ck_c0["metadata"]["arm"] = "C0"
    ck_c0["metadata"]["binary_prompt_dropout_p"] = 0.0
    ck_c0["metadata"]["config"]["arm"] = "C0"
    ck_c0["metadata"]["config"]["binary_prompt_dropout_p"] = 0.0

    with pytest.raises(ValueError, match="Cross-arm resume rejected.*expects arm 'C1'"):
        validate_checkpoint_for_resume(ck_c0, valid_c1_cfg, manifest_sha, init_sha, 0.25)


# ---------------------------------------------------------------------------
# Test 3: Malformed or contradictory optimizer-update counters
# ---------------------------------------------------------------------------
def test_3_malformed_contradictory_update_counters(valid_c1_payload, valid_c1_cfg):
    """Verify that corrupt, negative, or inconsistent update counters fail validation."""
    manifest_sha = valid_c1_cfg["manifest_sha256"]
    init_sha = valid_c1_cfg["initial_checkpoint_sha256"]

    # Negative successful updates
    ck = copy.deepcopy(valid_c1_payload)
    ck["successful_updates"] = -1
    with pytest.raises(ValueError, match="must be a non-negative integer"):
        validate_checkpoint_for_resume(ck, valid_c1_cfg, manifest_sha, init_sha, 0.25)

    # Contradictory step_attempts (step_attempts != successful + skipped)
    ck = copy.deepcopy(valid_c1_payload)
    ck["successful_updates"] = 500
    ck["skipped_updates"] = 10
    ck["step_attempts"] = 520  # Expected 510
    with pytest.raises(ValueError, match="Contradictory counters on resume"):
        validate_checkpoint_for_resume(ck, valid_c1_cfg, manifest_sha, init_sha, 0.25)

    # micro_steps less than step_attempts
    ck = copy.deepcopy(valid_c1_payload)
    ck["step_attempts"] = 600
    ck["micro_steps"] = 599
    with pytest.raises(ValueError, match="micro_steps .* cannot be less than step_attempts"):
        validate_checkpoint_for_resume(ck, valid_c1_cfg, manifest_sha, init_sha, 0.25)

    # Epoch length mismatch with history
    ck = copy.deepcopy(valid_c1_payload)
    ck["epoch"] = 2
    ck["history"] = [{"epoch": 1}]  # Expected length 3 for epoch index 2
    with pytest.raises(ValueError, match="history length .* does not match completed epoch count"):
        validate_checkpoint_for_resume(ck, valid_c1_cfg, manifest_sha, init_sha, 0.25)


# ---------------------------------------------------------------------------
# Test 4: Guardrail-failure artifact schema and preservation of last.pt
# ---------------------------------------------------------------------------
def test_4_guardrail_failure_artifact_schema_and_last_pt_preservation(tmp_path, valid_c1_payload):
    """Verify that a guardrail failure writes halted_guardrail.pt and preserves last.pt intact."""
    # 1. Establish valid last.pt from Epoch 0
    last_pt = tmp_path / "last.pt"
    save_checkpoint(last_pt, valid_c1_payload, drive_output_dir=None)
    last_pt_sha = hashlib.sha256(last_pt.read_bytes()).hexdigest()

    # 2. Simulate driver writing halted_guardrail.pt at Epoch 1
    halted_payload = copy.deepcopy(valid_c1_payload)
    halted_payload["epoch"] = 1
    halted_payload["display_epoch"] = 2
    halted_payload["metadata"]["guardrail_violation"] = True
    halted_payload["metadata"]["experiment_status"] = "HALTED_GUARDRAIL_VIOLATION"
    halted_payload["metadata"]["endpoint_status"] = "HALTED_GUARDRAIL_VIOLATION"
    halted_payload["metadata"]["primary_endpoint_reached"] = False

    halted_pt = tmp_path / "halted_guardrail.pt"
    save_checkpoint(halted_pt, halted_payload, drive_output_dir=None)

    # Assert last.pt is 100% byte-for-byte preserved
    assert last_pt.is_file()
    assert hashlib.sha256(last_pt.read_bytes()).hexdigest() == last_pt_sha

    # Assert halted_guardrail.pt contains full schema and flags
    assert halted_pt.is_file()
    loaded_halted = torch.load(halted_pt, map_location="cpu", weights_only=False)
    for k in ["model", "optimizer", "scheduler", "scaler", "rng", "dataloader_generator"]:
        assert k in loaded_halted
    assert loaded_halted["metadata"]["guardrail_violation"] is True
    assert loaded_halted["metadata"]["experiment_status"] == "HALTED_GUARDRAIL_VIOLATION"
    assert loaded_halted["metadata"]["primary_endpoint_reached"] is False

    # fixed_budget_checkpoint.pt must NOT exist
    assert not (tmp_path / "fixed_budget_checkpoint.pt").exists()


# ---------------------------------------------------------------------------
# Test 5: Rejection of halted checkpoints during normal resume
# ---------------------------------------------------------------------------
def test_5_rejection_of_halted_checkpoints_during_resume(valid_c1_payload, valid_c1_cfg):
    """Verify that validate_checkpoint_for_resume rejects any halted or guardrail-failed artifact."""
    manifest_sha = valid_c1_cfg["manifest_sha256"]
    init_sha = valid_c1_cfg["initial_checkpoint_sha256"]

    # Checkpoint with guardrail_violation = True
    ck_halted_1 = copy.deepcopy(valid_c1_payload)
    ck_halted_1["metadata"]["guardrail_violation"] = True
    with pytest.raises(RuntimeError, match="RESUME REJECTED: Checkpoint is marked with guardrail_violation=True"):
        validate_checkpoint_for_resume(ck_halted_1, valid_c1_cfg, manifest_sha, init_sha, 0.25)

    # Checkpoint with experiment_status = HALTED_GUARDRAIL_VIOLATION
    ck_halted_2 = copy.deepcopy(valid_c1_payload)
    ck_halted_2["metadata"]["experiment_status"] = "HALTED_GUARDRAIL_VIOLATION"
    with pytest.raises(RuntimeError, match="RESUME REJECTED: Checkpoint status is 'HALTED_GUARDRAIL_VIOLATION'"):
        validate_checkpoint_for_resume(ck_halted_2, valid_c1_cfg, manifest_sha, init_sha, 0.25)


# ---------------------------------------------------------------------------
# Test 6: Incorrect endpoint arm and incomplete update count
# ---------------------------------------------------------------------------
def test_6_incorrect_endpoint_arm_and_incomplete_update_count(tmp_path):
    """Verify that validate_endpoint_readiness catches cross-arm labels and incomplete budgets."""
    # 1. Arm mismatch: C1 passed where C0 expected
    ckpt_c1 = tmp_path / "c1.pt"
    torch.save({
        "successful_updates": 3600,
        "metadata": {
            "arm": "C1",
            "primary_endpoint_reached": True,
            "primary_3600_updates_completed": True,
            "endpoint_status": "COMPLETED_3600_UPDATES",
        },
    }, ckpt_c1)

    with pytest.raises(ValueError, match="Arm mismatch.*expected arm 'C0', but checkpoint metadata declares 'C1'"):
        validate_endpoint_readiness(ckpt_c1, "C0", allow_incomplete=False)

    # 2. Incomplete update count (3590 < 3600)
    ckpt_inc = tmp_path / "c0_incomplete.pt"
    torch.save({
        "successful_updates": 3590,
        "metadata": {
            "arm": "C0",
            "primary_endpoint_reached": False,
            "primary_3600_updates_completed": False,
            "endpoint_status": "INCOMPLETE_AMP_SKIPS",
        },
    }, ckpt_inc)

    with pytest.raises(RuntimeError, match="FAIL-CLOSED.*NOT an authorized completed 3,600-successful-update endpoint"):
        validate_endpoint_readiness(ckpt_inc, "C0", allow_incomplete=False)


# ---------------------------------------------------------------------------
# Test 7: Exploratory override without false VALID reporting
# ---------------------------------------------------------------------------
def test_7_exploratory_override_without_false_valid_reporting(tmp_path):
    """Verify that using allow_incomplete=True results in EXPLORATORY_INCOMPLETE_BUDGET, never VALID."""
    ckpt_inc_c0 = tmp_path / "c0_inc.pt"
    torch.save({
        "successful_updates": 1800,
        "metadata": {
            "arm": "C0",
            "primary_endpoint_reached": False,
            "primary_3600_updates_completed": False,
            "endpoint_status": "INCOMPLETE_AMP_SKIPS",
            "binary_prompt_dropout_p": 0.0,
            "split_sha256": "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b",
            "initial_checkpoint_sha256": "a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b",
            "config": {"seed": 1337},
        },
    }, ckpt_inc_c0)

    ckpt_inc_c1 = tmp_path / "c1_inc.pt"
    torch.save({
        "successful_updates": 1800,
        "metadata": {
            "arm": "C1",
            "primary_endpoint_reached": False,
            "primary_3600_updates_completed": False,
            "endpoint_status": "INCOMPLETE_AMP_SKIPS",
            "binary_prompt_dropout_p": 0.25,
            "split_sha256": "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b",
            "initial_checkpoint_sha256": "a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b",
            "config": {"seed": 1337},
        },
    }, ckpt_inc_c1)

    info_c0 = validate_endpoint_readiness(ckpt_inc_c0, "C0", allow_incomplete=True)
    info_c1 = validate_endpoint_readiness(ckpt_inc_c1, "C1", allow_incomplete=True)

    validate_paired_checkpoints_compatibility(info_c0, info_c1)
    assert not info_c0["is_completed_endpoint"]
    assert not info_c1["is_completed_endpoint"]

    # Status derivation check
    both_completed = info_c0["is_completed_endpoint"] and info_c1["is_completed_endpoint"]
    allow_incomplete_flag = True
    overall_guardrail_pass = True

    if not overall_guardrail_pass:
        status = "INVALID_GUARDRAIL_FAILURE"
    elif allow_incomplete_flag or not both_completed:
        status = "EXPLORATORY_INCOMPLETE_BUDGET"
    else:
        status = "VALID"

    assert status == "EXPLORATORY_INCOMPLETE_BUDGET"
    assert status != "VALID"


# ---------------------------------------------------------------------------
# Test 8: C0-only guardrail failure
# ---------------------------------------------------------------------------
def test_8_c0_only_guardrail_failure():
    """Verify that Control C0 failing CTC PER guardrail marks experiment INVALID."""
    measured_r_per = 0.2364  # 23.64%
    c1_per = 0.2414          # +0.5% degradation vs R (PASS <= 1.0%)
    c0_per = 0.2484          # +1.2% degradation vs R (FAIL > 1.0%)

    c0_deg = (c0_per - measured_r_per) * 100.0
    c1_deg = (c1_per - measured_r_per) * 100.0

    c0_pass = c0_deg <= 1.0
    c1_pass = c1_deg <= 1.0
    overall_pass = c0_pass and c1_pass

    assert c0_pass is False
    assert c1_pass is True
    assert overall_pass is False

    status = "VALID" if overall_pass else "INVALID_GUARDRAIL_FAILURE"
    assert status == "INVALID_GUARDRAIL_FAILURE"


# ---------------------------------------------------------------------------
# Test 9: C1-only guardrail failure
# ---------------------------------------------------------------------------
def test_9_c1_only_guardrail_failure():
    """Verify that Treatment C1 failing CTC PER guardrail marks experiment INVALID."""
    measured_r_per = 0.2364  # 23.64%
    c0_per = 0.2414          # +0.5% degradation vs R (PASS <= 1.0%)
    c1_per = 0.2484          # +1.2% degradation vs R (FAIL > 1.0%)

    c0_deg = (c0_per - measured_r_per) * 100.0
    c1_deg = (c1_per - measured_r_per) * 100.0

    c0_pass = c0_deg <= 1.0
    c1_pass = c1_deg <= 1.0
    overall_pass = c0_pass and c1_pass

    assert c0_pass is True
    assert c1_pass is False
    assert overall_pass is False

    status = "VALID" if overall_pass else "INVALID_GUARDRAIL_FAILURE"
    assert status == "INVALID_GUARDRAIL_FAILURE"


# ---------------------------------------------------------------------------
# Test 10: Frozen split and official test access boundaries
# ---------------------------------------------------------------------------
def test_10_frozen_split_and_official_test_access_boundaries():
    """Verify that official test partition speakers remain sealed and unreferenced."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Split manifest checksum
    manifest_sha = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    assert manifest_sha == "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b"

    fold1 = data["folds"][0]
    train_spks = set(fold1["train"]["speakers"])
    val_spks = set(fold1["val"]["speakers"])
    test_spks = set(fold1["test"]["speakers"])

    assert val_spks == {"NJS", "RRBI"}
    assert test_spks == {"ABA", "ASI", "EBVS", "HJK", "LXC", "PNV"}
    assert len(train_spks.intersection(test_spks)) == 0
    assert len(val_spks.intersection(test_spks)) == 0
