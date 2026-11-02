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

