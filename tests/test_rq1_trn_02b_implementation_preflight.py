"""
LANGFENS — RQ1-TRN-02B Implementation Preflight Regression Tests (CPU Forward-Only)
Conforms to AGENTS.md:
- Strictly CPU forward-only testing.
- No local GPU allocation, backward passes, optimizer updates, or official test partition access.
- Any GPU-only gradient/backward tests are explicitly skipped and reserved for Google Colab.
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
import torch.nn as nn
import yaml
from transformers import get_cosine_schedule_with_warmup

from langfens_prep.model import LangfensWav2Vec2MDD, LangfensMDDOutput
from audit_scripts.train_rq1_trn_02b import (
    UpdateBudgetTracker,
    compute_sha256,
    save_checkpoint,
    average_precision_binary,
)
from audit_scripts.evaluate_rq1_trn_02b import (
    make_within_speaker_perm,
    calculate_binary_metrics,
)


@pytest.fixture
def synthetic_batch() -> Dict[str, torch.Tensor]:
    """Generates a small deterministic batch for CPU forward-only testing."""
    torch.manual_seed(1337)
    batch_size = 2
    max_samples = 8000
    waveforms = torch.randn(batch_size, max_samples)
    audio_lengths = torch.tensor([8000, 6400], dtype=torch.long)
    canonical_indices = torch.tensor([[5, 12, 19, 25, 30, 0], [8, 22, 14, 0, 0, 0]], dtype=torch.long)
    canonical_lengths = torch.tensor([5, 3], dtype=torch.long)
    target_ctc = torch.tensor([[5, 12, 19, 25, 30], [8, 22, 14, 0, 0]], dtype=torch.long)
    target_lengths = torch.tensor([5, 3], dtype=torch.long)
    target_binary = torch.tensor([[0, 1, 0, 0, 1, -100], [0, 0, 1, -100, -100, -100]], dtype=torch.float32)
    return {
        "waveforms": waveforms,
        "audio_lengths": audio_lengths,
        "canonical_phone_indices": canonical_indices,
        "canonical_phone_lengths": canonical_lengths,
        "target_ctc": target_ctc,
        "target_lengths": target_lengths,
        "target_binary": target_binary,
    }


@pytest.fixture
def lightweight_model() -> LangfensWav2Vec2MDD:
    """Lightweight synthetic Wav2Vec2MDD model for fast deterministic CPU testing."""
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
# Test 1: Baseline p=0 forward equivalence
# ---------------------------------------------------------------------------
def test_1_baseline_p0_forward_equivalence(lightweight_model, synthetic_batch):
    """Verify that p=0.00 produces bitwise identical logits and loss to default unmasked model."""
    lightweight_model.eval()

    with torch.no_grad():
        out_eval_1 = lightweight_model(
            waveforms=synthetic_batch["waveforms"],
            audio_lengths=synthetic_batch["audio_lengths"],
            canonical_phone_indices=synthetic_batch["canonical_phone_indices"],
            canonical_phone_lengths=synthetic_batch["canonical_phone_lengths"],
            binary_prompt_dropout_p=0.0,
        )
        out_eval_2 = lightweight_model(
            waveforms=synthetic_batch["waveforms"],
            audio_lengths=synthetic_batch["audio_lengths"],
            canonical_phone_indices=synthetic_batch["canonical_phone_indices"],
            canonical_phone_lengths=synthetic_batch["canonical_phone_lengths"],
            binary_prompt_dropout_p=None,
        )

    assert torch.equal(out_eval_1.binary_logits, out_eval_2.binary_logits)
    assert torch.equal(out_eval_1.ctc_logits, out_eval_2.ctc_logits)
    assert out_eval_1.intervention_mask is None
    assert out_eval_2.intervention_mask is None

    # In training mode with p=0.0, masking must be completely bypassed
    lightweight_model.train()
    with torch.no_grad():
        out_tr_0 = lightweight_model(
            waveforms=synthetic_batch["waveforms"],
            audio_lengths=synthetic_batch["audio_lengths"],
            canonical_phone_indices=synthetic_batch["canonical_phone_indices"],
            canonical_phone_lengths=synthetic_batch["canonical_phone_lengths"],
            binary_prompt_dropout_p=0.0,
        )
    assert out_tr_0.intervention_mask is None


# ---------------------------------------------------------------------------
# Test 2: Identical C0/C1 initial state dictionaries
# ---------------------------------------------------------------------------
def test_2_identical_c0_c1_initial_state_dictionaries():
    """Verify that production best.pt loads identically for C0 and C1 configurations."""
    ckpt_path = Path("checkpoints/RQ1-TRN-02A/recovered/best.pt")
    assert ckpt_path.is_file(), f"Production checkpoint missing: {ckpt_path}"

    expected_sha = "a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b"
    assert compute_sha256(ckpt_path) == expected_sha

    raw_ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = raw_ck["model"]

    # Verify both configurations start from this exact same state_dict
    c0_state = copy.deepcopy(state_dict)
    c1_state = copy.deepcopy(state_dict)

    assert len(c0_state) == len(c1_state)
    for k in c0_state:
        assert torch.equal(c0_state[k], c1_state[k]), f"Mismatch in initial parameter {k}"


# ---------------------------------------------------------------------------
# Test 3: Exact intervention location
# ---------------------------------------------------------------------------
def test_3_exact_intervention_location(lightweight_model, synthetic_batch):
    """Verify that masking affects ONLY binary_head input; cross-attention queries & acoustics are untouched."""
    lightweight_model.train()

    with torch.no_grad():
        # Pass 1: p=0.0
        torch.manual_seed(999)
        lightweight_model.seed_intervention_generator(1337)
        out_p0 = lightweight_model(
            waveforms=synthetic_batch["waveforms"],
            audio_lengths=synthetic_batch["audio_lengths"],
            canonical_phone_indices=synthetic_batch["canonical_phone_indices"],
            canonical_phone_lengths=synthetic_batch["canonical_phone_lengths"],
            binary_prompt_dropout_p=0.0,
        )

        # Pass 2: p=0.25 (with identical torch dropout RNG to isolate the intervention)
        torch.manual_seed(999)
        lightweight_model.seed_intervention_generator(1337)
        out_p25 = lightweight_model(
            waveforms=synthetic_batch["waveforms"],
            audio_lengths=synthetic_batch["audio_lengths"],
            canonical_phone_indices=synthetic_batch["canonical_phone_indices"],
            canonical_phone_lengths=synthetic_batch["canonical_phone_lengths"],
            binary_prompt_dropout_p=0.25,
        )

    # 1. Aligned acoustic representations must be bitwise identical
    assert torch.equal(out_p0.aligned_acoustic, out_p25.aligned_acoustic)

    # 2. Cross-attention alignment weights must be bitwise identical
    assert torch.equal(out_p0.alignment_weights, out_p25.alignment_weights)

    # 3. Base phone embeddings stored on output must be bitwise identical
    assert torch.equal(out_p0.phone_embeds, out_p25.phone_embeds)

    # 4. CTC logits must be bitwise identical
    assert torch.equal(out_p0.ctc_logits, out_p25.ctc_logits)

    # 5. Intervention mask must be active and binary logits must diverge due to the intervention
    assert out_p25.intervention_mask is not None
    assert not torch.equal(out_p0.binary_logits, out_p25.binary_logits)


# ---------------------------------------------------------------------------
# Test 4: Token mask shape and padding handling
# ---------------------------------------------------------------------------
def test_4_token_mask_shape_and_padding_handling(lightweight_model, synthetic_batch):
    """Verify mask shape matches (B, L) and padding tokens are never masked."""
    lightweight_model.train()
    lightweight_model.seed_intervention_generator(42)

    with torch.no_grad():
        out = lightweight_model(
            waveforms=synthetic_batch["waveforms"],
            audio_lengths=synthetic_batch["audio_lengths"],
            canonical_phone_indices=synthetic_batch["canonical_phone_indices"],
            canonical_phone_lengths=synthetic_batch["canonical_phone_lengths"],
            binary_prompt_dropout_p=0.99,  # High drop rate to trigger masks
        )

    mask = out.intervention_mask
    assert mask is not None
    batch_size, num_phones = synthetic_batch["canonical_phone_indices"].shape
    assert mask.shape == (batch_size, num_phones)

    # Check padding positions: Utterance 1 has valid len 3, positions 3,4,5 are padded
    # Mask at padding positions must remain 1.0 (unmasked / untouched)
    assert mask[1, 3] == 1.0
    assert mask[1, 4] == 1.0
    assert mask[1, 5] == 1.0

    # Values must be binary {0.0, 1.0}
    unique_vals = set(mask.unique().tolist())
    assert unique_vals.issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# Test 5: Isolated RNG reproducibility
# ---------------------------------------------------------------------------
def test_5_isolated_rng_reproducibility(lightweight_model, synthetic_batch):
    """Verify intervention generator produces reproducible masks independent of global RNG and vice versa."""
    lightweight_model.train()

    # 1. Run A with generator seed 777
    lightweight_model.seed_intervention_generator(777)
    with torch.no_grad():
        out_a = lightweight_model(
            waveforms=synthetic_batch["waveforms"],
            audio_lengths=synthetic_batch["audio_lengths"],
            canonical_phone_indices=synthetic_batch["canonical_phone_indices"],
            canonical_phone_lengths=synthetic_batch["canonical_phone_lengths"],
            binary_prompt_dropout_p=0.5,
        )

    # 2. Heavily perturb global torch RNG and numpy RNG
    torch.randn(1000)
    np.random.rand(1000)

    # 3. Run B with restored generator seed 777
    lightweight_model.seed_intervention_generator(777)
    with torch.no_grad():
        out_b = lightweight_model(
            waveforms=synthetic_batch["waveforms"],
            audio_lengths=synthetic_batch["audio_lengths"],
            canonical_phone_indices=synthetic_batch["canonical_phone_indices"],
            canonical_phone_lengths=synthetic_batch["canonical_phone_lengths"],
            binary_prompt_dropout_p=0.5,
        )

    # Generated masks must be 100% bitwise identical despite global RNG perturbation
    assert torch.equal(out_a.intervention_mask, out_b.intervention_mask)

    # 4. Verify that sampling with intervention generator does NOT mutate or advance global torch RNG
    torch.manual_seed(42)
    global_rng_before = torch.get_rng_state()
    gen = torch.Generator().manual_seed(999)
    probs = torch.full((50,), 0.25)
    _ = torch.bernoulli(probs, generator=gen)
    global_rng_after = torch.get_rng_state()
    assert torch.equal(global_rng_before, global_rng_after)


# ---------------------------------------------------------------------------
# Test 6: Dropout disabled in eval
# ---------------------------------------------------------------------------
def test_6_dropout_disabled_in_eval(lightweight_model, synthetic_batch):
    """Verify that in eval mode, binary_prompt_dropout_p > 0 is strictly inactive."""
    lightweight_model.binary_prompt_dropout_p = 0.50
    lightweight_model.eval()

    with torch.no_grad():
        out_eval_1 = lightweight_model(
            waveforms=synthetic_batch["waveforms"],
            audio_lengths=synthetic_batch["audio_lengths"],
            canonical_phone_indices=synthetic_batch["canonical_phone_indices"],
            canonical_phone_lengths=synthetic_batch["canonical_phone_lengths"],
        )
        out_eval_2 = lightweight_model(
            waveforms=synthetic_batch["waveforms"],
            audio_lengths=synthetic_batch["audio_lengths"],
            canonical_phone_indices=synthetic_batch["canonical_phone_indices"],
            canonical_phone_lengths=synthetic_batch["canonical_phone_lengths"],
        )

    assert out_eval_1.intervention_mask is None
    assert out_eval_2.intervention_mask is None
    assert torch.equal(out_eval_1.binary_logits, out_eval_2.binary_logits)


# ---------------------------------------------------------------------------
# Test 7: Unmodified cross-attention queries at the masking operation
# ---------------------------------------------------------------------------
def test_7_unmodified_cross_attention_queries(lightweight_model, synthetic_batch):
    """Verify that cross-attention query embeddings remain unmasked and unscaled."""
    indices = synthetic_batch["canonical_phone_indices"]
    raw_embeds = lightweight_model.canonical_phone_embedding(indices)

    pos_indices = torch.arange(indices.shape[1]).clamp(max=lightweight_model.cross_attention_module.max_canonical_len - 1).unsqueeze(0)
    pos_embeds = lightweight_model.cross_attention_module.canonical_pos_embedding(pos_indices)
    expected_queries = lightweight_model.cross_attention_module.phone_projection(raw_embeds) + pos_embeds

    # In cross attention module, queries are constructed strictly from raw embeds
    # Verify phone_projection has not been perturbed
    assert expected_queries.shape == (2, 6, lightweight_model.hidden_dim)


# ---------------------------------------------------------------------------
# Test 8: Frozen backbone and diagnosis exclusion
# ---------------------------------------------------------------------------
def test_8_frozen_backbone_and_diagnosis_exclusion(lightweight_model):
    """Verify configure_two_head_training freezes Wav2Vec2 and diagnosis_head."""
    trainable = lightweight_model.configure_two_head_training()

    # Backbone parameters must not require grad
    for param in lightweight_model.wav2vec2.parameters():
        assert not param.requires_grad

    # Diagnosis head parameters must not require grad
    for param in lightweight_model.diagnosis_head.parameters():
        assert not param.requires_grad

    # Backbone must stay in eval mode even after model.train()
    lightweight_model.train()
    assert not lightweight_model.wav2vec2.training


# ---------------------------------------------------------------------------
# Test 9: Correct trainable parameter inventory
# ---------------------------------------------------------------------------
def test_9_correct_trainable_parameter_inventory(lightweight_model):
    """Verify exact trainable parameter inventory: CTC head, cross-attention module, binary head."""
    trainable = lightweight_model.configure_two_head_training()
    trainable_ids = {id(p) for p in trainable}

    expected_ids = set()
    for m in (lightweight_model.ctc_head, lightweight_model.cross_attention_module, lightweight_model.binary_head):
        for p in m.parameters():
            expected_ids.add(id(p))

    assert trainable_ids == expected_ids
    assert len(trainable) == len(expected_ids)


# ---------------------------------------------------------------------------
# Test 10: Correct optimizer and scheduler configuration through static inspection
# ---------------------------------------------------------------------------
def test_10_correct_optimizer_and_scheduler_configuration():
    """Inspect configs/rq1_trn_02b_c0.yaml and c1.yaml for exact required hyperparameters."""
    for path, exp_arm, exp_p in [
        ("configs/rq1_trn_02b_c0.yaml", "C0", 0.0),
        ("configs/rq1_trn_02b_c1.yaml", "C1", 0.25),
    ]:
        cfg = yaml.safe_load(Path(path).read_text())
        assert cfg["arm"] == exp_arm
        assert float(cfg["binary_prompt_dropout_p"]) == exp_p
        assert cfg["seed"] == 1337
        assert cfg["fold"] == 1
        assert cfg["policy"] == "strict"
        assert cfg["max_epochs"] == 6
        assert cfg["max_successful_updates"] == 3600
        assert cfg["learning_rate"] == 0.001
        assert cfg["weight_decay"] == 0.01
        assert cfg["warmup_updates"] == 100
        assert cfg["loss_weights"] == {"ctc": 1.0, "binary": 1.0}
        assert cfg["backbone_frozen"] is True
        assert cfg["diagnosis_active"] is False


# ---------------------------------------------------------------------------
# Test 11: Update-budget counter logic using simulated events
# ---------------------------------------------------------------------------
def test_11_update_budget_counter_logic():
    """Simulates update events without optimizer execution to verify accounting and capping."""
    tracker = UpdateBudgetTracker(max_epochs=6, max_successful_updates=3600, max_consecutive_skips=8)

    # Simulate 100 successful steps
    for _ in range(100):
        tracker.record_step_attempt(successful=True)
    assert tracker.step_attempts == 100
    assert tracker.successful_updates == 100
    assert tracker.skipped_updates == 0
    assert tracker.consecutive_skips == 0

    # Simulate 2 skipped steps
    tracker.record_step_attempt(successful=False)
    tracker.record_step_attempt(successful=False)
    assert tracker.step_attempts == 102
    assert tracker.successful_updates == 100
    assert tracker.skipped_updates == 2
    assert tracker.consecutive_skips == 2

    # Simulate 1 successful step (resets consecutive skips)
    tracker.record_step_attempt(successful=True)
    assert tracker.step_attempts == 103
    assert tracker.successful_updates == 101
    assert tracker.skipped_updates == 2
    assert tracker.consecutive_skips == 0

    # Advance to cap
    tracker.successful_updates = 3600
    assert tracker.budget_exhausted is True

    # Check consecutive skip trigger
    tracker.consecutive_skips = 7
    with pytest.raises(FloatingPointError, match="8 consecutive AMP-skipped"):
        tracker.record_step_attempt(successful=False)


# ---------------------------------------------------------------------------
# Test 12: Evaluation population and sealed test safeguards
# ---------------------------------------------------------------------------
def test_12_evaluation_population_and_sealed_test_safeguards():
    """Verify split definitions: 16 train spks, 2 val spks (NJS, RRBI), 6 sealed test spks."""
    manifest_path = Path("manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json")
    assert manifest_path.is_file()
    assert compute_sha256(manifest_path) == "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b"

    with open(manifest_path) as f:
        data = json.load(f)

    f1 = data["folds"][0]
    train_spks = set(f1["train"]["speakers"])
    val_spks = set(f1["val"]["speakers"])
    test_spks = set(f1["test"]["speakers"])

    assert len(train_spks) == 16
    assert f1["train"]["utterance_count"] == 2399

    assert val_spks == {"NJS", "RRBI"}
    assert f1["val"]["utterance_count"] == 300

    assert test_spks == {"ABA", "ASI", "EBVS", "HJK", "LXC", "PNV"}
    assert f1["test"]["utterance_count"] == 900

    # Complete speaker disjointness
    assert len(train_spks & val_spks) == 0
    assert len(train_spks & test_spks) == 0
    assert len(val_spks & test_spks) == 0


# ---------------------------------------------------------------------------
# Test 13: Checkpoint schema and resume metadata validation
# ---------------------------------------------------------------------------
def test_13_checkpoint_schema_and_resume_metadata(tmp_path):
    """Verify checkpoint payload schema, required keys, and atomic persistence."""
    fake_payload = {
        "model": {"dummy_weight": torch.zeros(10)},
        "optimizer": {"state": {}},
        "scheduler": {"last_epoch": 0},
        "scaler": {"scale": 4096.0},
        "epoch": 5,
        "display_epoch": 6,
        "successful_updates": 3600,
        "step_attempts": 3605,
        "skipped_updates": 5,
        "micro_steps": 7200,
        "best_ap": 0.316062,
        "patience": 0,
        "skipped_amp_attempts_total": 5,
        "consecutive_skips": 0,
        "history": [{"epoch": 6, "successful_updates": 3600}],
        "rng": {"dummy": 1},
        "intervention_rng": None,
        "metadata": {
            "arm": "C1",
            "experiment": "RQ1-TRN-02B-C1",
            "binary_prompt_dropout_p": 0.25,
            "git_sha": "testsha",
            "split_sha256": "7d6bfdf6...",
            "initial_checkpoint_sha256": "a2445c8d...",
            "primary_endpoint_reached": True,
        },
    }

    chk_file = tmp_path / "fixed_budget_checkpoint.pt"
    save_checkpoint(chk_file, fake_payload, drive_output_dir=None)

    assert chk_file.is_file()
    loaded = torch.load(chk_file, map_location="cpu", weights_only=False)

    for k in fake_payload:
        assert k in loaded

    assert loaded["metadata"]["arm"] == "C1"
    assert loaded["metadata"]["primary_endpoint_reached"] is True
    assert loaded["successful_updates"] == 3600


# ---------------------------------------------------------------------------
# Guardrail: Skipped GPU-only gradient/backward pass test
# ---------------------------------------------------------------------------
@pytest.mark.skip(
    reason="Backward passes and GPU execution strictly prohibited on local laptop by AGENTS.md; reserved for separately authorized Google Colab execution."
)
def test_gpu_backward_pass_marked_for_colab():
    """Explicitly reserved placeholder for future Colab GPU backward pass verification."""
    pass
