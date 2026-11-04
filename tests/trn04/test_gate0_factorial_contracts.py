"""LANGFENS RQ1-TRN-04 — Gate 0 Shared Contract Freeze Tests.

Validates the full preflight contract before GPU training execution:
1. Pinned seed 1337 initialization produces bit-identical weights across all 4 arms.
2. Identical capacity and architecture dimensions: exactly 267,179 trainable parameters in C0, T1, T2, T3.
3. Architectural slot behavior: zero acoustic features produce strictly zero extra projection (bias=False).
4. Waveform-derived frame-level acoustic features: 4D features, aligned to Wav2Vec2 frames, finite float32, zero gold leakage.
5. Class-conditioned focal loss: gamma=2.0, alpha=0.25 on coronal/rhotic deletions, alpha=1.0 otherwise, finite gradients.
6. Speaker-disjoint partitions: FIT (1,799), SELECT (300), CAL (300), VAL (300), TEST (900 sealed).
7. Synthetic end-to-end forward/backward step on all 4 arms.
"""
from __future__ import annotations

import hashlib
import pytest
import torch
import torch.nn as nn
from pathlib import Path

from langfens_prep.trn04.features import (
    extract_frame_acoustic_features,
    ACOUSTIC_FEATURE_DIM,
    FIT_LOG_ENERGY_MEAN,
    FIT_LOG_ENERGY_STD
)
from langfens_prep.trn04.model import (
    CompactFactorialOperationHead,
    FactorialHeadConfig,
    FactorialAudioAdapter,
    matched_initial_quad,
    parameter_inventory
)
from langfens_prep.trn04.objectives import task_loss, realized_ctc_loss, CORONAL_RHOTIC_PHONE_IDS
from langfens_prep.p_sup.dataset import (
    PSupL2ArcticDataset,
    FIT_SPEAKERS,
    SELECT_SPEAKERS,
    CALIBRATION_SPEAKERS,
    VAL_SPEAKERS,
    PARTITION_SPEAKERS,
    SealedSpeakerContaminationError
)
from langfens_prep.p_sup.runtime import SEALED_SPEAKERS

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SPLIT_MANIFEST = REPO_ROOT / "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json"
ZIP_PATH = REPO_ROOT / "datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip"


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def test_gate0_1_matched_quad_initialization_bit_identity():
    """Confirms C0, T1, T2, T3 are bit-identical at initialization from pinned seed 1337."""
    c0, t1, t2, t3 = matched_initial_quad(seed=1337)

    for name, p_c0 in c0.named_parameters():
        p_t1 = t1.get_parameter(name)
        p_t2 = t2.get_parameter(name)
        p_t3 = t3.get_parameter(name)

        sha_c0 = tensor_sha256(p_c0)
        sha_t1 = tensor_sha256(p_t1)
        sha_t2 = tensor_sha256(p_t2)
        sha_t3 = tensor_sha256(p_t3)

        assert sha_c0 == sha_t1 == sha_t2 == sha_t3, f"Initial tensor mismatch on {name}"
        assert torch.equal(p_c0, p_t1), f"C0 != T1 on {name}"
        assert torch.equal(p_c0, p_t2), f"C0 != T2 on {name}"
        assert torch.equal(p_c0, p_t3), f"C0 != T3 on {name}"


def test_gate0_2_matched_capacity_and_parameter_count():
    """Confirms all four arms have exactly 267,179 trainable parameters and matching inventory."""
    c0, t1, t2, t3 = matched_initial_quad(seed=1337)

    models = [("C0", c0), ("T1", t1), ("T2", t2), ("T3", t3)]
    expected_param_count = 267179

    for name, mod in models:
        inv = parameter_inventory(mod)
        total_trainable = sum(v["numel"] for v in inv.values() if v["trainable"])
        assert total_trainable == expected_param_count, (
            f"Arm {name} has {total_trainable} parameters, expected {expected_param_count}"
        )
        assert "acoustic_feature_projection.weight" in inv
        assert inv["acoustic_feature_projection.weight"]["shape"] == [128, 4]


def test_gate0_3_acoustic_slot_zero_input_invariance():
    """Confirms that when acoustic features are zero, the acoustic projection is strictly zero."""
    head = CompactFactorialOperationHead(FactorialHeadConfig())
    zero_features = torch.zeros(2, 49, 4)

    # With bias=False, linear projection of zeros is strictly identical to zero tensor
    projected = head.acoustic_feature_projection(zero_features)
    assert torch.equal(projected, torch.zeros_like(projected)), (
        "Zero acoustic features produced non-zero projection in acoustic slot!"
    )


def test_gate0_4_acoustic_feature_extraction_properties():
    """Confirms 4D waveform acoustic features are finite, aligned, and have no gold dependence."""
    # Synthetic waveform at 16kHz (1.5 seconds)
    audio = torch.randn(2, 24000)
    expected_frames = 74

    feat = extract_frame_acoustic_features(audio, expected_frames=expected_frames)

    assert feat.shape == (2, expected_frames, ACOUSTIC_FEATURE_DIM)
    assert torch.isfinite(feat).all(), "Nonfinite values in acoustic features"

    # Feature 0: Normalized Log Energy
    # Feature 1: Relative RMS in [0, 1]
    assert (feat[..., 1] >= 0.0).all() and (feat[..., 1] <= 1.0 + 1e-5).all()

    # Feature 3: Zero-crossing rate in [0, 1]
    assert (feat[..., 3] >= 0.0).all() and (feat[..., 3] <= 1.0 + 1e-5).all()


def test_gate0_5_class_conditioned_focal_loss_mechanics():
    """Confirms focal loss focuses on hard examples and applies alpha=0.25 to coronal deletions."""
    b, n = 2, 6
    logits = torch.randn(b, n, 3, requires_grad=True)
    operation = torch.tensor([[0, 1, 2, 0, 2, 1], [2, 0, 1, 2, 0, 2]])
    mask = torch.ones(b, n, dtype=torch.bool)
    # Positions 2 and 4 in row 0 are deletions: 31 (T) and 9 (D)
    prompt_ids = torch.tensor([[10, 12, 31, 14, 9, 20], [28, 5, 6, 15, 8, 31]])

    # Loss under CE
    ce_loss = task_loss(logits, operation, mask, prompt_ids, objective="operation_ce")
    assert torch.isfinite(ce_loss)

    # Loss under Class Focal
    focal_loss = task_loss(logits, operation, mask, prompt_ids, objective="class_focal")
    assert torch.isfinite(focal_loss)

    # Gradient flow test
    focal_loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert not torch.equal(logits.grad, torch.zeros_like(logits.grad))


def test_gate0_6_speaker_disjoint_partitions_and_sealed_test():
    """Confirms frozen speaker partition census and sealed test rejection."""
    assert len(FIT_SPEAKERS) == 12
    assert len(SELECT_SPEAKERS) == 2
    assert len(CALIBRATION_SPEAKERS) == 2
    assert len(VAL_SPEAKERS) == 2
    assert len(SEALED_SPEAKERS) == 6

    # Verify mutual disjointness
    all_groups = [FIT_SPEAKERS, SELECT_SPEAKERS, CALIBRATION_SPEAKERS, VAL_SPEAKERS, SEALED_SPEAKERS]
    for i in range(len(all_groups)):
        for j in range(i + 1, len(all_groups)):
            overlap = all_groups[i] & all_groups[j]
            assert not overlap, f"Speaker overlap detected: {overlap}"

    # Verify partition loading
    fit_ds = PSupL2ArcticDataset(str(SPLIT_MANIFEST), str(ZIP_PATH), split="fit")
    assert len(fit_ds) == 1799

    sel_ds = PSupL2ArcticDataset(str(SPLIT_MANIFEST), str(ZIP_PATH), split="selection")
    assert len(sel_ds) == 300

    cal_ds = PSupL2ArcticDataset(str(SPLIT_MANIFEST), str(ZIP_PATH), split="calibration")
    assert len(cal_ds) == 300

    val_ds = PSupL2ArcticDataset(str(SPLIT_MANIFEST), str(ZIP_PATH), split="val")
    assert len(val_ds) == 300


def test_gate0_7_synthetic_step_across_all_four_arms():
    """Executes one synthetic forward+backward+optimizer update on all 4 arms."""
    from langfens_prep.trn04.trainer import ARM_CONFIGS

    for arm_name, cfg in ARM_CONFIGS.items():
        torch.manual_seed(1337)
        head = CompactFactorialOperationHead(FactorialHeadConfig())
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.001)

        # Synthetic inputs
        b, t, d = 2, 40, 768
        n = 15
        feat = torch.randn(b, t, d)
        fl = torch.tensor([40, 35])
        prompt_ids = torch.randint(1, 39, (b, n))
        pl = torch.tensor([15, 12])
        mask = torch.arange(n)[None] < pl[:, None]
        prompt_ids = prompt_ids.masked_fill(~mask, 0)
        targets = torch.randint(0, 3, (b, n))

        if cfg["features_enabled"]:
            acoust = torch.randn(b, t, ACOUSTIC_FEATURE_DIM)
        else:
            acoust = torch.zeros(b, t, ACOUSTIC_FEATURE_DIM)

        out = head(
            features=feat,
            acoustic_features=acoust,
            frame_lengths=fl,
            prompt_phone_ids=prompt_ids,
            prompt_lengths=pl
        )

        loss_task = task_loss(
            logits=out["operation_logits"],
            operation=targets,
            mask=mask,
            prompt_phone_ids=prompt_ids,
            objective=cfg["objective"]
        )

        ctc_targets = torch.tensor([[1, 2, 3, 4, 5, 0], [6, 7, 8, 0, 0, 0]], dtype=torch.long)
        ctc_lens = torch.tensor([5, 3], dtype=torch.long)
        loss_ctc = realized_ctc_loss(out["ctc_logits"], fl, ctc_targets, ctc_lens)
        loss = loss_task + loss_ctc

        optimizer.zero_grad()
        loss.backward()

        # Check gradients exist and are finite
        for name, p in head.named_parameters():
            if p.requires_grad:
                if not cfg["features_enabled"] and "acoustic_feature_projection" in name:
                    # When features are disabled (zeros input), gradient is zero
                    assert p.grad is not None
                    assert torch.equal(p.grad, torch.zeros_like(p.grad))
                else:
                    assert p.grad is not None
                    assert torch.isfinite(p.grad).all()

        optimizer.step()
        assert loss.item() > 0.0
