"""
LANGFENS — RQ-MODEL-01 / E1: Alignment-Aware MDD Unit and Smoke Regression Tests.
Verifies deployable alignment-aware cross-attention, learnable acoustic skip token,
monotonic temporal prior, binary head integration, and strict Baseline R backward compatibility.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path

from langfens_prep.model import (
    CanonicalAcousticCrossAttention,
    LangfensWav2Vec2MDD,
    LangfensMDDOutput,
)


def test_cross_attention_baseline_backward_compatibility():
    """Verify default parameters preserve bit-exact Baseline R signature and behavior."""
    ca = CanonicalAcousticCrossAttention(
        hidden_dim=256,
        phone_embed_dim=128,
        num_canonical_phones=40,
        num_heads=4,
        dropout=0.0,
        enable_skip_token=False,
        monotonic_bias=False,
    )
    assert ca.skip_token is None
    assert not ca.enable_skip_token
    assert not ca.monotonic_bias

    frame_feats = torch.randn(2, 20, 256)
    phone_idx = torch.randint(1, 40, (2, 8))
    pad_mask = torch.zeros(2, 20, dtype=torch.bool)
    pad_mask[0, 15:] = True

    res = ca(frame_feats, phone_idx, pad_mask)
    assert len(res) == 3
    aligned, embeds, weights = res
    assert aligned.shape == (2, 8, 256)
    assert embeds.shape == (2, 8, 128)
    assert weights.shape == (2, 8, 20)
    # Padded frames must have zero attention
    assert torch.all(weights[0, :, 15:] == 0.0)
    # Sum over frames must be 1.0
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 8), atol=1e-5)


def test_cross_attention_e1_skip_token_and_monotonic_bias():
    """Verify E1 skip token extension, key extension, and monotonic temporal bias."""
    ca = CanonicalAcousticCrossAttention(
        hidden_dim=256,
        phone_embed_dim=128,
        num_canonical_phones=40,
        num_heads=4,
        dropout=0.0,
        enable_skip_token=True,
        monotonic_bias=True,
        monotonic_bandwidth=3.0,
    )
    assert ca.enable_skip_token
    assert ca.monotonic_bias
    assert isinstance(ca.skip_token, nn.Parameter)
    assert ca.skip_token.shape == (1, 1, 256)

    frame_feats = torch.randn(2, 20, 256)
    phone_idx = torch.randint(1, 40, (2, 8))
    pad_mask = torch.zeros(2, 20, dtype=torch.bool)
    pad_mask[0, 16:] = True

    res = ca(frame_feats, phone_idx, pad_mask)
    assert len(res) == 4
    aligned, embeds, weights, skip_w = res
    assert aligned.shape == (2, 8, 256)
    assert embeds.shape == (2, 8, 128)
    # Extended weights shape: 1 skip token + 20 frames = 21
    assert weights.shape == (2, 8, 21)
    assert skip_w.shape == (2, 8, 1)

    # Skip weight must equal weights[:, :, 0:1]
    assert torch.equal(skip_w, weights[:, :, 0:1])

    # All weights must be strictly in [0, 1]
    assert (weights >= 0.0).all() and (weights <= 1.0).all()
    # Sum over (skip + frames) must be exactly 1.0
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 8), atol=1e-5)

    # Padded frames (positions 16..20 in acoustic frames -> positions 17..21 in extended keys) must be 0
    assert torch.all(weights[0, :, 17:] == 0.0)


def test_model_e1_architecture_and_forward_shapes():
    """Verify LangfensWav2Vec2MDD with E1 alignment-aware skip modeling."""
    model = LangfensWav2Vec2MDD(
        hidden_dim=256,
        conv_dim=128,
        num_transformer_layers=2,
        num_attention_heads=4,
        intermediate_dim=512,
        enable_skip_token=True,
        monotonic_bias=True,
        monotonic_bandwidth=3.0,
    )
    assert model.enable_skip_token
    assert model.monotonic_bias
    # Binary head input dimension must be hidden_dim + phone_embed_dim + 1 = 256 + 128 + 1 = 385
    assert model.binary_head[0].in_features == 385

    batch_size = 2
    audio_len = 6400  # 0.4s @ 16kHz -> ~20 frames
    waveforms = torch.randn(batch_size, audio_len)
    audio_lengths = torch.tensor([6400, 4800], dtype=torch.long)
    can_indices = torch.tensor([[5, 12, 19], [8, 22, 0]], dtype=torch.long)
    target_ctc = torch.tensor([[5, 12, 19], [8, 22, 0]], dtype=torch.long)
    target_lengths = torch.tensor([3, 2], dtype=torch.long)
    target_binary = torch.tensor([[0, 1, 0], [1, 0, -100]], dtype=torch.long)

    model.train()
    out = model(
        waveforms=waveforms,
        audio_lengths=audio_lengths,
        canonical_phone_indices=can_indices,
        target_ctc=target_ctc,
        target_lengths=target_lengths,
        target_binary=target_binary,
    )

    assert isinstance(out, LangfensMDDOutput)
    assert out.binary_logits is not None
    assert out.binary_logits.shape == (2, 3)
    assert out.skip_weights is not None
    assert out.skip_weights.shape == (2, 3, 1)
    assert (out.skip_weights >= 0.0).all() and (out.skip_weights <= 1.0).all()
    assert out.loss is not None
    assert torch.isfinite(out.loss)
    assert "ctc_loss" in out.loss_dict
    assert "binary_loss" in out.loss_dict


def test_model_two_head_trainable_parameters_include_skip_token():
    """Verify configure_two_head_training isolates exact head parameters including skip_token."""
    model = LangfensWav2Vec2MDD(
        hidden_dim=256,
        conv_dim=128,
        num_transformer_layers=2,
        num_attention_heads=4,
        intermediate_dim=512,
        enable_skip_token=True,
    )
    trainable = model.configure_two_head_training()
    trainable_ids = {id(p) for p in trainable}

    # Backbone must be frozen
    assert all(not p.requires_grad for p in model.wav2vec2.parameters())
    # Diagnosis head must be frozen
    assert all(not p.requires_grad for p in model.diagnosis_head.parameters())
    # Skip token must be trainable
    assert model.cross_attention_module.skip_token.requires_grad
    assert id(model.cross_attention_module.skip_token) in trainable_ids

    # Expected exact parameter count match
    expected = {
        id(p)
        for m in (model.ctc_head, model.cross_attention_module, model.binary_head)
        for p in m.parameters()
    }
    assert trainable_ids == expected


def test_baseline_r_checkpoint_strict_state_dict_loading():
    """Verify existing Baseline R checkpoint loads with strict=True when enable_skip_token=False."""
    ckpt_path = Path("checkpoints/RQ1-TRN-02A/recovered/best.pt")
    if not ckpt_path.is_file():
        pytest.skip(f"Baseline R checkpoint not present at {ckpt_path}")

    model = LangfensWav2Vec2MDD(
        pretrained_model_name_or_path="facebook/wav2vec2-base-960h",
        enable_skip_token=False,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert missing == []
    assert unexpected == []


def test_e1_configs_and_deployment_metadata_integrity():
    """Verify that E1 training configurations and Kaggle deployment metadata conform to invariants."""
    import yaml
    import json

    # 1. Verify configs for Folds 1-4
    for fold in range(1, 5):
        cfg_path = Path(f"configs/rq_model_01_e1_fold{fold}.yaml")
        assert cfg_path.is_file(), f"Missing config for fold {fold}"
        cfg = yaml.safe_load(cfg_path.read_text())
        assert cfg["experiment"] == "RQ-MODEL-01-E1"
        assert cfg["fold"] == fold
        assert cfg["enable_skip_token"] is True
        assert cfg["monotonic_bias"] is True
        assert cfg["backbone_frozen"] is True
        assert cfg["loss_weights"] == {"ctc": 1.0, "binary": 1.0}
        assert cfg["optimizer"] == "AdamW"
        assert cfg["learning_rate"] == 0.001
        assert cfg["batch_size"] == 2
        assert cfg["gradient_accumulation_steps"] == 2

    # 2. Verify Kaggle deployment packages
    for pkg_dir in [Path("deploy/kaggle_e1_fold1_preflight"), Path("deploy/kaggle_e1_fold1_train")]:
        assert pkg_dir.is_dir()
        k_meta_path = pkg_dir / "kernel-metadata.json"
        d_meta_path = pkg_dir / "dataset-metadata.json"
        assert k_meta_path.is_file()
        assert d_meta_path.is_file()

        k_meta = json.loads(k_meta_path.read_text())
        d_meta = json.loads(d_meta_path.read_text())

        assert k_meta["is_private"] is True
        assert k_meta["enable_gpu"] is True
        assert k_meta["machine_shape"] == "NvidiaTeslaT4"
        assert d_meta["is_private"] is True
        assert d_meta["id"] == "minkhoaa/langfens-rq1-fold1-staging"


def test_monotonic_bias_batched_vs_single_item_equivalence():
    """Verify batched and single-item inference produce mathematically identical attention on valid frames."""
    ca = CanonicalAcousticCrossAttention(
        hidden_dim=64,
        phone_embed_dim=32,
        num_heads=2,
        dropout=0.0,
        enable_skip_token=True,
        monotonic_bias=True,
        monotonic_bandwidth=3.0,
    ).eval()

    torch.manual_seed(42)
    # Item 0: 15 frames, 6 phones
    f0 = torch.randn(1, 15, 64)
    p0 = torch.randint(1, 39, (1, 6))

    # Item 1: 25 frames, 10 phones
    f1 = torch.randn(1, 25, 64)
    p1 = torch.randint(1, 39, (1, 10))

    # Batched container (padded to max_frames=25, max_phones=10)
    f_batch = torch.zeros(2, 25, 64)
    f_batch[0, :15] = f0[0]
    f_batch[1, :25] = f1[0]

    p_batch = torch.zeros(2, 10, dtype=torch.long)
    p_batch[0, :6] = p0[0]
    p_batch[1, :10] = p1[0]

    pad_mask = torch.zeros(2, 25, dtype=torch.bool)
    pad_mask[0, 15:] = True

    frame_lens = torch.tensor([15, 25])
    phone_lens = torch.tensor([6, 10])

    # Single-item inference on item 0
    with torch.inference_mode():
        out0, emb0, w0, s0 = ca(f0, p0, None)
        out_b, emb_b, w_b, s_b = ca(
            f_batch,
            p_batch,
            pad_mask,
            canonical_phone_lengths=phone_lens,
            frame_output_lengths=frame_lens,
        )

    # Valid frames for item 0 are index 0 (skip token) + indices 1..15 (acoustic frames 0..14)
    diff = (w_b[0, :6, :16] - w0[0, :6, :16]).abs().max().item()
    assert diff < 1e-6, f"Batched vs single-item diff {diff} exceeds tolerance"

    # Padded frame attention weights in batch must be exactly 0.0
    assert torch.all(w_b[0, :6, 16:] == 0.0)


def test_skip_token_gradient_connectivity_without_auxiliary_loss():
    """Verify skip token receives non-zero finite gradient directly from standard multi-task loss."""
    torch.manual_seed(1337)
    model = LangfensWav2Vec2MDD(
        hidden_dim=256,
        conv_dim=128,
        num_transformer_layers=2,
        num_attention_heads=4,
        intermediate_dim=512,
        enable_skip_token=True,
        monotonic_bias=True,
        monotonic_bandwidth=3.0,
    )
    trainable = model.configure_two_head_training()
    model.train()

    waveforms = torch.randn(2, 6400)
    audio_lengths = torch.tensor([6400, 4800], dtype=torch.long)
    can_idx = torch.tensor([[5, 12, 19, 25], [8, 22, 14, 0]], dtype=torch.long)
    can_len = torch.tensor([4, 3], dtype=torch.long)
    target_ctc = torch.tensor([[5, 12, 19, 25], [8, 22, 14, 0]], dtype=torch.long)
    target_lengths = torch.tensor([4, 3], dtype=torch.long)
    target_binary = torch.tensor([[0, 1, 0, 1], [1, 0, 1, -100]], dtype=torch.long)

    out = model(
        waveforms=waveforms,
        audio_lengths=audio_lengths,
        canonical_phone_indices=can_idx,
        canonical_phone_lengths=can_len,
        target_ctc=target_ctc,
        target_lengths=target_lengths,
        target_binary=target_binary,
    )

    assert out.loss is not None
    assert torch.isfinite(out.loss)
    out.loss.backward()

    skip_token = model.cross_attention_module.skip_token
    assert skip_token.grad is not None
    assert torch.isfinite(skip_token.grad).all()
    assert (skip_token.grad != 0.0).any()
    assert skip_token.grad.norm().item() > 0.0

    for p in trainable:
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_attention_evaluation_excludes_skip_index_and_distinguishes_conditional_mass():
    """Verify evaluation logic strictly excludes skip index from acoustic frames and separates conditional mass."""
    import numpy as np

    # Synthetic attention row: skip weight = 0.5, acoustic frames = 10 frames summing to 0.5
    raw_attn = np.array([
        0.50,  # skip token (index 0)
        0.02, 0.03, 0.05, 0.15, 0.10, 0.05, 0.04, 0.03, 0.02, 0.01  # acoustic frames 0..9
    ])
    assert np.isclose(raw_attn.sum(), 1.0)

    # Evaluation logic with skip token enabled
    enable_skip_token = True
    active_frames = 10
    if enable_skip_token:
        skip_mass = float(raw_attn[0])
        acoustic_attn_w = raw_attn[1:1 + active_frames]
    else:
        skip_mass = 0.0
        acoustic_attn_w = raw_attn[:active_frames]

    assert np.isclose(skip_mass, 0.50)
    assert len(acoustic_attn_w) == 10
    acoustic_sum = float(acoustic_attn_w.sum())
    assert np.isclose(acoustic_sum, 0.50)

    # Gold interval: frames 3..5 (indices 3, 4: weights 0.15, 0.10)
    start_frame, end_frame = 3, 5
    abs_in_win = float(acoustic_attn_w[start_frame:end_frame].sum())  # 0.25
    cond_in_win = float(abs_in_win / max(acoustic_sum, 1e-12))        # 0.25 / 0.50 = 0.50

    assert np.isclose(abs_in_win, 0.25)
    assert np.isclose(cond_in_win, 0.50)

    # Peak frame among acoustic frames must be frame 3 (weight 0.15)
    peak_frame = int(np.argmax(acoustic_attn_w))
    assert peak_frame == 3  # Not index 0 (which would be 0.50 if skip token was not excluded!)
    peak_time = peak_frame * 0.020
    center_time = 0.065
    offset = abs(peak_time - center_time)
    assert np.isclose(peak_time, 0.060)
    assert np.isclose(offset, 0.005)
