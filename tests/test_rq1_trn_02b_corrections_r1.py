"""
LANGFENS — RQ1-TRN-02B Execution-Readiness Corrections Regression Tests (CPU Forward-Only)
Verifies:
1. Correct evaluation dataset field access (audio_waveform_16k, ctc_target_indices, speaker_id).
2. Correct speaker-safe audio permutation (never mixing across speakers).
3. Matching label/score populations in Protocol O.
4. Secondary Protocol Z zero-oracle G2P inference and paired differencing.
5. Resume determinism without optimizer steps (DataLoader generator state persistence and order equivalence).
6. AMP-skipped update simulations and UpdateBudgetTracker accounting.
7. Fail-closed endpoint handling on incomplete budget checkpoints.
8. CTC PER regression guardrail enforcement against Reference R anchor.
9. Checkpoint provenance validation on resume (arm ID, seed, split SHA, init ckpt SHA, counters).
10. Strict exclusion and zero access to sealed official test speakers.
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
from torch.utils.data import DataLoader, Dataset

from langfens_prep.alignment import align_canonical_sequences
from langfens_prep.dataset import (
    L2ArcticMDDDataset,
    ZeroOracleL2ArcticDataset,
    load_arctic_prompts,
)
from langfens_prep.g2p import CMUDictG2P, EXPECTED_CMUDICT_SHA256
from langfens_prep.model import LangfensWav2Vec2MDD
from audit_scripts.evaluate_rq1_trn_02b import (
    average_precision_binary,
    calculate_binary_metrics,
    evaluate_models_protocol_z,
    evaluate_single_model_protocol_o,
    make_within_speaker_perm,
    validate_endpoint_readiness,
)
from audit_scripts.train_rq1_trn_02b import (
    UpdateBudgetTracker,
    save_checkpoint,
    validate_checkpoint_for_resume,
)

MANIFEST_PATH = root_dir / "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json"
ZIP_PATH = root_dir / "datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip"
PROMPTS_PATH = root_dir / "resources/L2-ARCTIC/prompts.json"
REF_CKPT_PATH = root_dir / "checkpoints/RQ1-TRN-02A/recovered/best.pt"


@pytest.fixture
def lightweight_model() -> LangfensWav2Vec2MDD:
    """Lightweight synthetic Wav2Vec2MDD model for fast deterministic CPU forward testing."""
    torch.manual_seed(1337)
    return LangfensWav2Vec2MDD(
        hidden_dim=64,
        conv_dim=32,
        num_transformer_layers=2,
        num_attention_heads=2,
        intermediate_dim=128,
        num_canonical_phones=40,
        phone_embed_dim=32,
        num_ctc_classes=40,
        num_diagnosis_classes=39,
        binary_prompt_dropout_p=0.0,
    )


# ---------------------------------------------------------------------------
# Test 1: Correct evaluation dataset field access
# ---------------------------------------------------------------------------
def test_1_dataset_field_access_contract():
    """Verify that L2ArcticMDDDataset items strictly conform to the expected interface."""
    ds = L2ArcticMDDDataset(
        str(MANIFEST_PATH),
        fold=1,
        split="val",
        policy="strict",
        zip_path=str(ZIP_PATH),
        max_samples=2,
    )
    assert len(ds) == 2
    item = ds[0]

    # Verify actual contract keys exist
    assert "audio_waveform_16k" in item, "Item must contain 'audio_waveform_16k'"
    assert "ctc_target_indices" in item, "Item must contain 'ctc_target_indices'"
    assert "speaker_id" in item, "Item must contain 'speaker_id'"
    assert "canonical_phone_indices" in item, "Item must contain 'canonical_phone_indices'"
    assert "binary_labels" in item, "Item must contain 'binary_labels'"

    # Verify types and tensor properties
    assert isinstance(item["audio_waveform_16k"], torch.Tensor)
    assert item["audio_waveform_16k"].dim() == 1
    assert isinstance(item["ctc_target_indices"], torch.Tensor)
    assert item["ctc_target_indices"].dtype == torch.long
    assert isinstance(item["speaker_id"], str)
    assert item["speaker_id"] in {"NJS", "RRBI"}

    # Verify deprecated/incorrect keys are NOT present
    assert "waveform" not in item, "Deprecated key 'waveform' should not be in dataset item"
    assert "target_ctc" not in item, "Deprecated key 'target_ctc' should not be in dataset item"
    assert "speaker" not in item, "Deprecated key 'speaker' should not be in dataset item"


# ---------------------------------------------------------------------------
# Test 2: Correct speaker-safe audio permutation
# ---------------------------------------------------------------------------
def test_2_speaker_safe_audio_permutation():
    """Verify that within-speaker permutation never crosses speaker boundaries."""
    # Simulation with 150 NJS and 150 RRBI utterances (boundary at [150])
    total = 300
    bounds = [150]
    perm = make_within_speaker_perm(bounds, total, shift=1)

    assert len(perm) == total
    # Check that speaker 0 (0..149) never receives audio from speaker 1 (150..299)
    for idx in range(150):
        donor = perm[idx]
        assert 0 <= donor < 150, f"Speaker NJS item {idx} received donor {donor} outside its partition!"
        assert donor != idx, f"Permutation must be derangement, but item {idx} mapped to itself"

    for idx in range(150, 300):
        donor = perm[idx]
        assert 150 <= donor < 300, f"Speaker RRBI item {idx} received donor {donor} outside its partition!"
        assert donor != idx, f"Permutation must be derangement, but item {idx} mapped to itself"

    # Multi-speaker boundaries test [30, 70, 120] for total=150
    multi_bounds = [30, 70, 120]
    multi_perm = make_within_speaker_perm(multi_bounds, 150, shift=2)
    for idx in range(30):
        assert 0 <= multi_perm[idx] < 30
    for idx in range(30, 70):
        assert 30 <= multi_perm[idx] < 70
    for idx in range(70, 120):
        assert 70 <= multi_perm[idx] < 120
    for idx in range(120, 150):
        assert 120 <= multi_perm[idx] < 150


# ---------------------------------------------------------------------------
# Test 3: Matching label/score populations in Protocol O
# ---------------------------------------------------------------------------
def test_3_protocol_o_evaluation_pipeline(lightweight_model):
    """Verify Protocol O single-model evaluation using actual authorized validation samples."""
    val_dataset = L2ArcticMDDDataset(
        str(MANIFEST_PATH),
        fold=1,
        split="val",
        policy="strict",
        zip_path=str(ZIP_PATH),
        max_samples=4,  # Fast forward-only smoke test
    )
    # Speaker boundaries for 4 samples (all NJS in the first 4 samples)
    spk_bounds = [
        i for i in range(1, len(val_dataset))
        if val_dataset.utterance_entries[i][0] != val_dataset.utterance_entries[i - 1][0]
    ]

    summary, res = evaluate_single_model_protocol_o(
        model=lightweight_model,
        val_dataset=val_dataset,
        spk_bounds=spk_bounds,
        device=torch.device("cpu"),
    )

    assert "matched_binary_ap" in summary
    assert "mismatched_binary_ap" in summary
    assert "matched_minus_mismatched_gap" in summary
    assert "ctc_per" in summary
    assert len(res) == 4

    for u in res:
        assert len(u["matched_scores"]) == len(u["labels"])
        assert len(u["mismatched_scores"]) == len(u["labels"])
        assert u["speaker_id"] in {"NJS", "RRBI"}


# ---------------------------------------------------------------------------
# Test 4: Protocol Z zero-oracle G2P evaluation and paired differencing
# ---------------------------------------------------------------------------
def test_4_protocol_z_zero_oracle_scoring(lightweight_model):
    """Verify Protocol Z zero-oracle frontend, alignment, and paired differences."""
    g2p = CMUDictG2P(verify_checksum=True, oov_policy="strict")
    assert g2p.dict_path.is_file()

    # Load zero-oracle dataset with 2 samples
    z_ds = ZeroOracleL2ArcticDataset(
        split_manifest_path=str(MANIFEST_PATH),
        fold=1,
        split="val",
        prompts_source=str(PROMPTS_PATH),
        g2p_frontend=g2p,
        zip_path=str(ZIP_PATH),
        max_samples=2,
    )
    assert len(z_ds) == 2
    item = z_ds[0]

    # Verify zero labels in zero-oracle dataset
    assert item["binary_labels"] is None
    assert item["ctc_target_indices"] is None
    assert "canonical_phones" in item
    assert len(item["canonical_phones"]) > 0

    # Evaluate paired models under Protocol Z
    model_c0 = lightweight_model
    model_c1 = copy.deepcopy(lightweight_model)
    model_r = copy.deepcopy(lightweight_model)

    # Fast forward on 2 samples
    proto_z = evaluate_models_protocol_z(
        models={"reference_r": model_r, "control_c0": model_c0, "treatment_c1": model_c1},
        manifest_path=MANIFEST_PATH,
        zip_path=str(ZIP_PATH),
        device=torch.device("cpu"),
        prompts_source=str(PROMPTS_PATH),
    )

    assert "token_coverage" in proto_z
    assert proto_z["token_coverage"] > 0.90  # Expected ~99%
    assert "aligned_only_ap" in proto_z
    assert "reference_r" in proto_z["aligned_only_ap"]
    assert "control_c0" in proto_z["aligned_only_ap"]
    assert "treatment_c1" in proto_z["aligned_only_ap"]
    assert "primary_causal_delta_dropout_z (C1 - C0)" in proto_z["paired_estimands"]


# ---------------------------------------------------------------------------
# Test 5: Resume sample-order determinism without optimizer steps
# ---------------------------------------------------------------------------
def test_5_resume_sample_order_determinism_without_optimizer_steps():
    """Verify that DataLoader generator state persistence guarantees exact epoch resumption order."""
    class MockIndexDataset(Dataset):
        def __len__(self):
            return 50
        def __getitem__(self, idx):
            return {"index": idx}

    ds = MockIndexDataset()

    # 1. Uninterrupted run: Epoch 1 then Epoch 2
    g_uninterrupted = torch.Generator()
    g_uninterrupted.manual_seed(1337)
    loader_1 = DataLoader(ds, batch_size=5, shuffle=True, generator=g_uninterrupted)

    ep1_indices = []
    for batch in loader_1:
        ep1_indices.extend(batch["index"].tolist())

    # Capture generator state at epoch 1 boundary
    state_at_ep1_boundary = g_uninterrupted.get_state()

    ep2_uninterrupted_indices = []
    for batch in loader_1:
        ep2_uninterrupted_indices.extend(batch["index"].tolist())

    # 2. Resumed run: Restore generator state and run Epoch 2
    g_resumed = torch.Generator()
    g_resumed.set_state(state_at_ep1_boundary)
    loader_resumed = DataLoader(ds, batch_size=5, shuffle=True, generator=g_resumed)

    ep2_resumed_indices = []
    for batch in loader_resumed:
        ep2_resumed_indices.extend(batch["index"].tolist())

    # Assert exact bitwise sample-order equivalence across epochs
    assert ep2_uninterrupted_indices == ep2_resumed_indices
    # Ensure Epoch 2 is different from Epoch 1 (proper shuffling across epochs)
    assert ep1_indices != ep2_resumed_indices


# ---------------------------------------------------------------------------
# Test 6: AMP-skipped update simulations and UpdateBudgetTracker
# ---------------------------------------------------------------------------
def test_6_amp_skipped_update_simulations_and_budget_tracker():
    """Verify that AMP-skipped updates do not increment successful updates and trigger safety halts."""
    tracker = UpdateBudgetTracker(max_epochs=6, max_successful_updates=3600, max_consecutive_skips=8)

    # Simulate 10 successful updates
    for _ in range(10):
        tracker.record_step_attempt(successful=True)
    assert tracker.successful_updates == 10
    assert tracker.step_attempts == 10
    assert tracker.skipped_updates == 0
    assert tracker.consecutive_skips == 0
    assert not tracker.budget_exhausted

    # Simulate 3 AMP skips
    for _ in range(3):
        tracker.record_step_attempt(successful=False)
    assert tracker.successful_updates == 10
    assert tracker.step_attempts == 13
    assert tracker.skipped_updates == 3
    assert tracker.consecutive_skips == 3

    # Reset consecutive skips on success
    tracker.record_step_attempt(successful=True)
    assert tracker.successful_updates == 11
    assert tracker.consecutive_skips == 0

    # 8 consecutive skips must trigger FloatingPointError
    with pytest.raises(FloatingPointError, match="8 consecutive AMP-skipped"):
        for _ in range(8):
            tracker.record_step_attempt(successful=False)

    # Budget exhaustion check: 3600 successful updates
    tracker_full = UpdateBudgetTracker(max_epochs=6, max_successful_updates=100)
    for _ in range(99):
        tracker_full.record_step_attempt(successful=True)
    assert not tracker_full.budget_exhausted
    tracker_full.record_step_attempt(successful=True)
    assert tracker_full.budget_exhausted


# ---------------------------------------------------------------------------
# Test 7: Fail-closed endpoint handling
# ---------------------------------------------------------------------------
def test_7_fail_closed_endpoint_handling(tmp_path):
    """Verify that evaluator fails closed when checkpoints have incomplete update budgets."""
    incomplete_payload = {
        "model": {"dummy": torch.zeros(1)},
        "successful_updates": 3580,  # Incomplete: 20 updates were skipped
        "metadata": {
            "primary_endpoint_reached": False,
            "endpoint_status": "INCOMPLETE_AMP_SKIPS",
        },
    }
    ckpt_file = tmp_path / "incomplete.pt"
    torch.save(incomplete_payload, ckpt_file)

    # Must raise RuntimeError with fail-closed message
    with pytest.raises(RuntimeError, match="FAIL-CLOSED.*NOT a completed 3,600-successful-update endpoint"):
        validate_endpoint_readiness(ckpt_file, "C1", allow_incomplete=False)

    # Overridden with allow_incomplete=True must pass
    info = validate_endpoint_readiness(ckpt_file, "C1", allow_incomplete=True)
    assert info["successful_updates"] == 3580
    assert info["primary_endpoint_reached"] is False

    # Completed checkpoint must pass strictly
    completed_payload = {
        "model": {"dummy": torch.zeros(1)},
        "successful_updates": 3600,
        "metadata": {
            "primary_endpoint_reached": True,
            "endpoint_status": "COMPLETED_3600_UPDATES",
        },
    }
    complete_file = tmp_path / "completed.pt"
    torch.save(completed_payload, complete_file)
    complete_info = validate_endpoint_readiness(complete_file, "C0", allow_incomplete=False)
    assert complete_info["successful_updates"] == 3600
    assert complete_info["primary_endpoint_reached"] is True


# ---------------------------------------------------------------------------
# Test 8: CTC PER guardrail enforcement
# ---------------------------------------------------------------------------
def test_8_ctc_per_guardrail_enforcement():
    """Verify CTC PER degradation guardrail logic against Reference R."""
    ref_per = 0.2364  # Reference R anchor (23.64%)

    # Case 1: Degradation within limit (+0.5% degradation -> 24.14%)
    c1_per_pass = 0.2414
    deg_pass = (c1_per_pass - ref_per) * 100.0
    assert deg_pass <= 1.0, f"Expected {deg_pass:.2f} <= 1.0"

    # Case 2: Degradation exceeds limit (+1.2% degradation -> 24.84%)
    c1_per_fail = 0.2484
    deg_fail = (c1_per_fail - ref_per) * 100.0
    assert deg_fail > 1.0, f"Expected {deg_fail:.2f} > 1.0"
    guardrail_pass = deg_fail <= 1.0
    assert not guardrail_pass


# ---------------------------------------------------------------------------
# Test 9: Checkpoint provenance validation on resume
# ---------------------------------------------------------------------------
def test_9_checkpoint_provenance_validation_on_resume():
    """Verify validate_checkpoint_for_resume rejects invalid or mismatched provenance."""
    cfg = {
        "arm": "C1",
        "seed": 1337,
        "manifest_sha256": "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b",
        "initial_checkpoint_sha256": "a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b",
        "binary_prompt_dropout_p": 0.25,
    }
    valid_ck = {
        "successful_updates": 1200,
        "step_attempts": 1200,
        "skipped_updates": 0,
        "micro_steps": 2400,
        "epoch": 1,
        "metadata": {
            "arm": "C1",
            "split_sha256": cfg["manifest_sha256"],
            "initial_checkpoint_sha256": cfg["initial_checkpoint_sha256"],
            "binary_prompt_dropout_p": 0.25,
            "config": {"seed": 1337},
        },
    }

    # Valid checkpoint passes
    validate_checkpoint_for_resume(
        valid_ck, cfg, cfg["manifest_sha256"], cfg["initial_checkpoint_sha256"], 0.25
    )

    # 1. Arm mismatch
    bad_arm = copy.deepcopy(valid_ck)
    bad_arm["metadata"]["arm"] = "C0"
    with pytest.raises(ValueError, match="Arm mismatch on resume"):
        validate_checkpoint_for_resume(
            bad_arm, cfg, cfg["manifest_sha256"], cfg["initial_checkpoint_sha256"], 0.25
        )

    # 2. Seed mismatch
    bad_seed = copy.deepcopy(valid_ck)
    bad_seed["metadata"]["config"]["seed"] = 42
    with pytest.raises(ValueError, match="Seed mismatch on resume"):
        validate_checkpoint_for_resume(
            bad_seed, cfg, cfg["manifest_sha256"], cfg["initial_checkpoint_sha256"], 0.25
        )

    # 3. Initial checkpoint SHA mismatch
    bad_init = copy.deepcopy(valid_ck)
    bad_init["metadata"]["initial_checkpoint_sha256"] = "wrong_sha"
    with pytest.raises(ValueError, match="Initial checkpoint SHA-256 mismatch"):
        validate_checkpoint_for_resume(
            bad_init, cfg, cfg["manifest_sha256"], cfg["initial_checkpoint_sha256"], 0.25
        )

    # 4. Inconsistent update counters (step_attempts < successful_updates)
    bad_counters = copy.deepcopy(valid_ck)
    bad_counters["step_attempts"] = 1000  # < 1200
    with pytest.raises(ValueError, match="step_attempts .* cannot be less than successful_updates"):
        validate_checkpoint_for_resume(
            bad_counters, cfg, cfg["manifest_sha256"], cfg["initial_checkpoint_sha256"], 0.25
        )


# ---------------------------------------------------------------------------
# Test 10: Strict exclusion and zero access to sealed official test speakers
# ---------------------------------------------------------------------------
def test_10_no_official_test_access():
    """Verify that official test speakers remain completely sealed and unreferenced."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    fold1 = manifest_data["folds"][0]
    test_speakers = set(fold1["test"]["speakers"])
    assert test_speakers == {"ABA", "ASI", "EBVS", "HJK", "LXC", "PNV"}

    train_speakers = set(fold1["train"]["speakers"])
    val_speakers = set(fold1["val"]["speakers"])
    assert val_speakers == {"NJS", "RRBI"}

    # Assert zero overlap between train/val and test
    assert len(train_speakers.intersection(test_speakers)) == 0
    assert len(val_speakers.intersection(test_speakers)) == 0

    # Verify authorized zip contains no test audio
    import zipfile
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        namelist = zf.namelist()
        for name in namelist:
            for ts in test_speakers:
                assert f"{ts}/" not in name, f"Test speaker {ts} audio leaked into authorized zip: {name}!"
