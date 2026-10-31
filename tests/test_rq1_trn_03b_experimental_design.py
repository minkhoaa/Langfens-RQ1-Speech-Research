"""
LANGFENS — RQ1-TRN-03B: Experimental Design & Scientific Readiness Regression Tests
Validates the one-variable-at-a-time (OVAT) experimental design for Phase RQ1-TRN-03B,
verifies mathematical loss contracts (Focal BCE and Class-Weighted BCE), confirms
closure of residual Step 14 issues, and enforces preservation of Baseline R benchmarks
and per-fold test sealing invariants.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import yaml

from langfens_prep.losses import MaskedBCEWithLogitsLoss, MaskedFocalBCEWithLogitsLoss
from langfens_prep.model import LangfensWav2Vec2MDD, CanonicalAcousticCrossAttention
from langfens_prep.phoneme_normalizer import STANDARD_39_ARPABET, PHONE_TO_CTC_INDEX, PHONE_TO_INDEX


def compute_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 1. Closure of Residual Step 14 Issues
# ---------------------------------------------------------------------------

def test_actual_1199_validation_utterances_across_all_four_folds():
    """Confirms exact 1,199 total validation utterances (F1: 300, F2: 299, F3: 300, F4: 300)."""
    manifest_path = Path("manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json")
    with open(manifest_path) as f:
        splits = json.load(f)

    folds = splits["folds"]
    assert len(folds) == 4
    val_counts = [len(f["val"]["utterances"]) for f in folds]
    assert val_counts == [300, 299, 300, 300], f"Expected [300, 299, 300, 300], got {val_counts}"
    assert sum(val_counts) == 1199, f"Expected 1,199 total validation utterances, got {sum(val_counts)}"

    # Check report metadata reflects 1,199
    rep_path = Path("manifests/RQ1-TRN-03A/evaluation/FOUR_FOLD_VALIDATION_REPORT.json")
    with open(rep_path) as f:
        rep = json.load(f)
    assert rep["metadata"]["total_out_of_fold_validation_utterances"] == 1199

    # Check token-level predictions archive metadata reflects 1,199
    archive_path = Path("manifests/RQ1-TRN-03A/evaluation/FOUR_FOLD_PAIRED_PREDICTION_RECORDS.json.gz")
    with gzip.open(archive_path, "rt", encoding="utf-8") as f:
        pred_data = json.load(f)
    assert pred_data["metadata"]["total_utterances"] == 1199
    assert len(pred_data["utterances"]) == 1199


def test_evaluator_runtime_provenance_and_introducing_commit():
    """Confirms evaluator metadata records introducing commit 618e86d alongside runtime SHA."""
    rep_path = Path("manifests/RQ1-TRN-03A/evaluation/FOUR_FOLD_VALIDATION_REPORT.json")
    with open(rep_path) as f:
        rep = json.load(f)
    meta = rep["metadata"]
    assert meta["evaluator_source_commit"] == "618e86d"
    assert "evaluator_git_sha" in meta and len(meta["evaluator_git_sha"]) > 0

    archive_path = Path("manifests/RQ1-TRN-03A/evaluation/FOUR_FOLD_PAIRED_PREDICTION_RECORDS.json.gz")
    with gzip.open(archive_path, "rt", encoding="utf-8") as f:
        pred_data = json.load(f)
    assert pred_data["metadata"]["evaluator_source_commit"] == "618e86d"
    assert "evaluator_git_sha" in pred_data["metadata"]


def test_cross_attention_query_key_value_mapping():
    """
    Confirms exact cross-attention mapping:
    Query (Q) = Canonical Prompt Phoneme Sequence
    Key (K) & Value (V) = Acoustic Frame Features from Wav2Vec2.
    """
    hidden_dim = 64
    num_heads = 2
    phone_embed_dim = 32
    max_canonical_len = 16

    attn_mod = CanonicalAcousticCrossAttention(
        hidden_dim=hidden_dim,
        phone_embed_dim=phone_embed_dim,
        num_canonical_phones=40,
        num_heads=num_heads,
        max_canonical_len=max_canonical_len,
        dropout=0.0
    )

    batch_size = 2
    num_frames = 20
    num_phones = 8

    # Acoustic frames from backbone: (B, T, hidden_dim)
    frame_features = torch.randn(batch_size, num_frames, hidden_dim)
    # Canonical prompt phone indices: (B, N)
    canonical_phones = torch.randint(1, 40, (batch_size, num_phones))

    aligned_acoustic, phone_embeds, attn_weights = attn_mod(frame_features, canonical_phones)

    # Output aligned acoustic representation must match (B, N, hidden_dim) - phone query positions
    assert aligned_acoustic.shape == (batch_size, num_phones, hidden_dim)
    # Attention weights must have shape (B, num_queries, num_keys) = (B, num_phones, num_frames)
    assert attn_weights.shape == (batch_size, num_phones, num_frames)


def test_vocabulary_cardinality_and_head_separation():
    """
    Confirms exact vocabulary separation:
    - Realized CTC Head: 40 classes (1 blank at index 0 + 39 ARPAbet phonemes).
    - Substitution Diagnosis Head: 39 classes (39 ARPAbet phonemes, no blank).
    """
    assert len(STANDARD_39_ARPABET) == 39
    assert len(PHONE_TO_CTC_INDEX) == 39  # 39 phones mapped to 1..39
    assert len(PHONE_TO_INDEX) == 39      # 39 phones mapped to 0..38

    # Model instantiation check with lightweight synthetic config
    model = LangfensWav2Vec2MDD(
        hidden_dim=64,
        num_transformer_layers=1,
        num_attention_heads=2,
        intermediate_dim=128,
        conv_dim=64,
        num_ctc_classes=40,
        num_diagnosis_classes=39
    )
    assert model.ctc_head.out_features == 40
    assert model.diagnosis_head[-1].out_features == 39


# ---------------------------------------------------------------------------
# 2. Mathematical Contracts for Loss Interventions (Arm 2A & Arm 2B)
# ---------------------------------------------------------------------------

def test_masked_focal_bce_loss_mathematical_contract():
    """Verifies MaskedFocalBCEWithLogitsLoss properties: masking, modulation, zero-fallback."""
    loss_fn = MaskedFocalBCEWithLogitsLoss(ignore_index=-100, gamma=2.0, alpha=0.75, reduction="mean")

    # 1. Zero-loss fallback on all-ignored batch
    logits = torch.randn(2, 5, requires_grad=True)
    all_ignored = torch.full((2, 5), -100, dtype=torch.long)
    loss_zero = loss_fn(logits, all_ignored)
    assert loss_zero.item() == 0.0
    loss_zero.backward()
    assert logits.grad is not None
    assert (logits.grad == 0.0).all()

    # 2. Invalid target rejection
    invalid_targets = torch.tensor([[0, 2], [1, -100]])
    with pytest.raises(ValueError, match="outside {0, 1}"):
        loss_fn(torch.randn(2, 2), invalid_targets)

    # 3. Easy vs. Hard Example Downweighting
    # Confident correct negative (z = -5.0, y = 0 -> p = 0.0067)
    # Uncertain correct negative (z = 0.0, y = 0 -> p = 0.5)
    logits_comp = torch.tensor([-5.0, 0.0])
    targets_comp = torch.tensor([0.0, 0.0])
    loss_none = MaskedFocalBCEWithLogitsLoss(ignore_index=-100, gamma=2.0, alpha=0.75, reduction="none")(
        logits_comp, targets_comp
    )
    # The confident prediction must have massively downweighted loss compared to uncertain
    assert loss_none[0] < loss_none[1] * 0.01

    # 4. Asymmetric alpha weighting: positive error vs negative correct at z=0 (p=0.5)
    # For z=0, p=0.5, (1-pt)^2 = 0.25, BCE = ln(2)
    # For y=1: alpha_t = 0.75 -> loss = 0.75 * 0.25 * ln(2)
    # For y=0: alpha_t = 0.25 -> loss = 0.25 * 0.25 * ln(2)
    # Ratio must be exactly 0.75 / 0.25 = 3.0
    logits_sym = torch.tensor([0.0, 0.0])
    targets_sym = torch.tensor([1.0, 0.0])
    loss_sym = MaskedFocalBCEWithLogitsLoss(ignore_index=-100, gamma=2.0, alpha=0.75, reduction="none")(
        logits_sym, targets_sym
    )
    ratio = (loss_sym[0] / loss_sym[1]).item()
    assert pytest.approx(ratio, rel=1e-4) == 3.0


def test_masked_bce_with_pos_weight_buffer_and_weighting():
    """Verifies MaskedBCEWithLogitsLoss properly weights positive class and handles buffers."""
    pos_w = 8.40
    loss_fn = MaskedBCEWithLogitsLoss(ignore_index=-100, pos_weight=pos_w, reduction="none")

    # pos_weight must be a registered buffer
    assert hasattr(loss_fn, "pos_weight")
    assert isinstance(loss_fn.pos_weight, torch.Tensor)
    assert pytest.approx(loss_fn.pos_weight.item(), rel=1e-4) == 8.40

    # At z=0 (p=0.5), BCE for y=1 is pos_w * ln(2), for y=0 is ln(2)
    logits = torch.tensor([0.0, 0.0])
    targets = torch.tensor([1.0, 0.0])
    loss = loss_fn(logits, targets)
    ratio = (loss[0] / loss[1]).item()
    assert pytest.approx(ratio, rel=1e-4) == 8.40


def test_langfens_wav2vec2_mdd_loss_type_configuration():
    """Verifies LangfensWav2Vec2MDD supports configuring binary loss type without breaking defaults."""
    # 1. Default: standard MaskedBCEWithLogitsLoss (Baseline R parity)
    m_default = LangfensWav2Vec2MDD(
        hidden_dim=64, num_transformer_layers=1, num_attention_heads=2, intermediate_dim=128, conv_dim=64
    )
    assert isinstance(m_default.binary_loss_fn, MaskedBCEWithLogitsLoss)
    assert m_default.binary_loss_fn.pos_weight is None
    assert m_default.binary_loss_type == "bce"

    # 2. Arm 2A: Focal BCE
    m_focal = LangfensWav2Vec2MDD(
        hidden_dim=64, num_transformer_layers=1, num_attention_heads=2, intermediate_dim=128, conv_dim=64,
        binary_loss_type="focal", focal_gamma=2.0, focal_alpha=0.75
    )
    assert isinstance(m_focal.binary_loss_fn, MaskedFocalBCEWithLogitsLoss)
    assert m_focal.binary_loss_type == "focal"
    assert m_focal.binary_loss_fn.gamma == 2.0
    assert m_focal.binary_loss_fn.alpha == 0.75

    # 3. Arm 2B: Class-Weighted BCE
    m_weighted = LangfensWav2Vec2MDD(
        hidden_dim=64, num_transformer_layers=1, num_attention_heads=2, intermediate_dim=128, conv_dim=64,
        binary_loss_type="bce", binary_pos_weight=8.40
    )
    assert isinstance(m_weighted.binary_loss_fn, MaskedBCEWithLogitsLoss)
    assert m_weighted.binary_loss_type == "bce"
    assert m_weighted.binary_loss_fn.pos_weight is not None
    assert pytest.approx(m_weighted.binary_loss_fn.pos_weight.item(), rel=1e-4) == 8.40

    # 4. Invalid loss type rejection
    with pytest.raises(ValueError, match="Unknown binary_loss_type"):
        LangfensWav2Vec2MDD(
            hidden_dim=64, num_transformer_layers=1, num_attention_heads=2, intermediate_dim=128, conv_dim=64,
            binary_loss_type="unsupported_loss"
        )


# ---------------------------------------------------------------------------
# 3. OVAT Configuration Parity & Invariance
# ---------------------------------------------------------------------------

def test_ovat_config_parity_across_arm2a_and_arm2b():
    """Verifies that all Arm 2A and Arm 2B configs preserve bit-level parity with Baseline R."""
    config_dir = Path("configs")
    arm2a_configs = [
        config_dir / "rq1_trn_03b_pilot_arm2a_focal.yaml",
        config_dir / "rq1_trn_03b_arm2a_fold1.yaml",
        config_dir / "rq1_trn_03b_arm2a_fold2.yaml",
        config_dir / "rq1_trn_03b_arm2a_fold3.yaml",
        config_dir / "rq1_trn_03b_arm2a_fold4.yaml",
    ]
    arm2b_configs = [
        config_dir / "rq1_trn_03b_pilot_arm2b_weighted.yaml",
        config_dir / "rq1_trn_03b_arm2b_fold1.yaml",
        config_dir / "rq1_trn_03b_arm2b_fold2.yaml",
        config_dir / "rq1_trn_03b_arm2b_fold3.yaml",
        config_dir / "rq1_trn_03b_arm2b_fold4.yaml",
    ]

    all_configs = arm2a_configs + arm2b_configs
    for p in all_configs:
        assert p.is_file(), f"Missing config: {p}"
        with open(p) as f:
            cfg = yaml.safe_load(f)

        assert cfg["experiment"] == "RQ1-TRN-03B"
        assert cfg["seed"] == 1337
        assert cfg["policy"] == "strict"
        assert cfg["manifest_sha256"] == "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b"
        assert cfg["pretrained_model"] == "facebook/wav2vec2-base-960h"
        assert cfg["pretrained_revision"] == "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
        assert cfg["loss_weights"] == {"ctc": 1.0, "binary": 1.0}
        assert cfg["backbone_frozen"] is True
        assert cfg["batch_size"] == 2
        assert cfg["gradient_accumulation_steps"] == 2
        assert cfg["learning_rate"] == 0.001
        assert cfg["weight_decay"] == 0.01
        assert cfg["scheduler"] == "cosine"
        assert cfg["warmup_updates"] == 100
        assert cfg["max_epochs"] == 25
        assert cfg["scheduler_horizon_epochs"] == 25
        assert cfg["execution_cap_epochs"] == 10
        assert cfg["early_stopping_patience"] == 5
        assert cfg["amp"] == "fp16"
        assert cfg["grad_scaler_initial_scale"] == 4096

    # Arm-specific checks
    for p in arm2a_configs:
        with open(p) as f:
            cfg = yaml.safe_load(f)
        assert cfg["arm"] == "Arm2A_Focal"
        assert cfg["binary_loss_type"] == "focal"
        assert cfg["focal_gamma"] == 2.0
        assert cfg["focal_alpha"] == 0.75

    for p in arm2b_configs:
        with open(p) as f:
            cfg = yaml.safe_load(f)
        assert cfg["arm"] == "Arm2B_Weighted"
        assert cfg["binary_loss_type"] == "bce"
        assert pytest.approx(cfg["binary_pos_weight"], rel=1e-4) == 8.40


# ---------------------------------------------------------------------------
# 4. Invariant Preservation: Baseline R Benchmarks & Test Sealing
# ---------------------------------------------------------------------------

def test_per_fold_test_isolation_invariants_preserved():
    """Verifies that across all 4 folds, train, val, and test partitions remain strictly disjoint."""
    manifest_path = Path("manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json")
    with open(manifest_path) as f:
        splits = json.load(f)

    for i, fold in enumerate(splits["folds"]):
        train_spk = set(fold["train"]["speakers"])
        val_spk = set(fold["val"]["speakers"])
        test_spk = set(fold["test"]["speakers"])

        assert len(train_spk) == 16
        assert len(val_spk) == 2
        assert len(test_spk) == 6

        assert train_spk.isdisjoint(test_spk), f"Fold {i+1}: Train and Test speakers overlap!"
        assert val_spk.isdisjoint(test_spk), f"Fold {i+1}: Val and Test speakers overlap!"
        assert train_spk.isdisjoint(val_spk), f"Fold {i+1}: Train and Val speakers overlap!"


def test_baseline_r_artifacts_and_hashes_unaltered():
    """Verifies that all immutable Baseline R archives, splits, and records remain bit-exact."""
    # Split manifest
    assert compute_sha256("manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json") == (
        "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b"
    )

    # Train/Val staging archives
    expected_archives = {
        "datasets/L2-ARCTIC/authorized_RQ1-TRN_fold2_train_val_only.zip": "112cc9e0cc5e37e5fa034df504eca019dd6ad438c96cb642335ecf10a802f0d9",
        "datasets/L2-ARCTIC/authorized_RQ1-TRN_fold3_train_val_only.zip": "e81ac9e5d95a68d7f02c7b2ea8e5f1e9fd89b047926ffeeb4d096b0a3a5378a2",
        "datasets/L2-ARCTIC/authorized_RQ1-TRN_fold4_train_val_only.zip": "653bdad079ee951d6b6d6a3f19964143aa5f4290a3f8c232e9b2524abdbe7f94",
    }
    for path_str, expected_hash in expected_archives.items():
        p = Path(path_str)
        if p.is_file():
            assert compute_sha256(p) == expected_hash, f"Hash mismatch for {path_str}"

    # Prediction records archive
    pred_path = Path("manifests/RQ1-TRN-03A/evaluation/FOUR_FOLD_PAIRED_PREDICTION_RECORDS.json.gz")
    assert compute_sha256(pred_path) == "355b6e23bcb172d20b99ba3f9f4e07dc15453846ac312eca847471ad476e5892"
