"""Comprehensive unit and integration tests for P-SUP data, trainer, and evaluator pipeline."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from langfens_prep.p_sup.model import CompactOperationHead, HeadConfig, FrozenAudioAdapter, matched_initial_pair
from langfens_prep.p_sup.objectives import task_loss, realized_ctc_loss, collapsed_binary_logit
from langfens_prep.p_sup.planning import UpdateLedger, learning_rate_multiplier, accumulation_groups
from langfens_prep.p_sup.dataset import (
    PSupL2ArcticDataset,
    collate_psup_batch,
    SealedSpeakerContaminationError,
    FOLD1_TRAIN_SPEAKERS,
    FOLD1_VAL_SPEAKERS,
    PARTITION_SPEAKERS
)
from langfens_prep.p_sup.labels import OperationLabels
from langfens_prep.p_sup.runtime import (
    SEALED_SPEAKERS,
    CONTRACT_SHA256,
    operation_runtime_rows,
    validate_complete_run,
    validate_runtime_row
)
from langfens_prep.p_sup.trainer import (
    average_precision_binary,
    greedy_ctc_decode,
    compute_levenshtein_edits,
    save_checkpoint
)


def test_partition_speaker_counts_and_disjointness():
    """Verify exact partition speaker memberships and disjointness from sealed test speakers."""
    assert len(PARTITION_SPEAKERS["fit"]) == 12
    assert len(PARTITION_SPEAKERS["selection"]) == 2
    assert len(PARTITION_SPEAKERS["calibration"]) == 2
    assert len(PARTITION_SPEAKERS["val"]) == 2

    # Verify fit, selection, calibration partition TRAIN exactly
    union_train = PARTITION_SPEAKERS["fit"] | PARTITION_SPEAKERS["selection"] | PARTITION_SPEAKERS["calibration"]
    assert union_train == FOLD1_TRAIN_SPEAKERS
    assert len(PARTITION_SPEAKERS["fit"] & PARTITION_SPEAKERS["selection"]) == 0
    assert len(PARTITION_SPEAKERS["fit"] & PARTITION_SPEAKERS["calibration"]) == 0
    assert len(PARTITION_SPEAKERS["selection"] & PARTITION_SPEAKERS["calibration"]) == 0

    # Strict isolation from sealed test speakers
    for partition_name, speakers in PARTITION_SPEAKERS.items():
        assert not (speakers & SEALED_SPEAKERS), f"Partition {partition_name} contains sealed test speakers!"


def test_dataset_sealed_speaker_rejection():
    """Dataset must fail-closed if any sealed test speaker is accessed."""
    manifest_path = Path("manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json")
    zip_path = Path("datasets/fake.zip")
    ds = PSupL2ArcticDataset(manifest_path, zip_path, split="train", verify_checksums=False)

    for sealed_spk in SEALED_SPEAKERS:
        ds.entries = [(sealed_spk, "arctic_a0001")]
        with pytest.raises(SealedSpeakerContaminationError):
            _ = ds[0]


def test_collate_psup_batch_padding_and_masks():
    """Verify collator output shapes, padding sentinels, and mask boolean types."""
    labels_a = OperationLabels(
        canonical_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        operation=torch.tensor([0, 1, 2], dtype=torch.long),
        binary=torch.tensor([0, 1, 1], dtype=torch.long),
        realization=torch.tensor([0, 5, -100], dtype=torch.long),
        ctc=torch.tensor([1, 6], dtype=torch.long),
        addition_gaps=(),
        unknown_realized_intervals=()
    )
    labels_b = OperationLabels(
        canonical_ids=torch.tensor([4, 5], dtype=torch.long),
        operation=torch.tensor([0, 0], dtype=torch.long),
        binary=torch.tensor([0, 0], dtype=torch.long),
        realization=torch.tensor([3, 4], dtype=torch.long),
        ctc=torch.tensor([4, 5], dtype=torch.long),
        addition_gaps=(),
        unknown_realized_intervals=()
    )

    item_a = {
        "speaker_id": "BWC",
        "utterance_id": "arctic_a0001",
        "waveform": torch.randn(1600),
        "labels": labels_a
    }
    item_b = {
        "speaker_id": "ERMS",
        "utterance_id": "arctic_a0002",
        "waveform": torch.randn(3200),
        "labels": labels_b
    }

    batch = collate_psup_batch([item_a, item_b])
    assert batch["waveforms"].shape == (2, 3200)
    assert batch["audio_lengths"].tolist() == [1600, 3200]
    assert batch["prompt_phone_ids"].shape == (2, 3)
    assert batch["prompt_lengths"].tolist() == [3, 2]
    assert batch["operation"].shape == (2, 3)
    assert batch["operation"][1, 2].item() == -100
    assert batch["operation_mask"].dtype == torch.bool
    assert batch["operation_mask"][1, 2].item() is False
    assert batch["binary"].shape == (2, 3)
    assert batch["binary"][1, 2].item() == -100
    assert batch["ctc"].shape == (2, 2)


def test_collate_rejects_sealed_speaker():
    """Collator must reject sealed test speaker in batch."""
    dummy_labels = OperationLabels(
        canonical_ids=torch.tensor([1], dtype=torch.long),
        operation=torch.tensor([0], dtype=torch.long),
        binary=torch.tensor([0], dtype=torch.long),
        realization=torch.tensor([0], dtype=torch.long),
        ctc=torch.tensor([1], dtype=torch.long),
        addition_gaps=(),
        unknown_realized_intervals=()
    )
    bad_item = {
        "speaker_id": "ABA",  # Sealed test speaker!
        "utterance_id": "arctic_a0001",
        "waveform": torch.randn(1600),
        "labels": dummy_labels
    }
    with pytest.raises(SealedSpeakerContaminationError):
        collate_psup_batch([bad_item])


def test_treatment_and_control_objectives_and_score_identity():
    """Verify treatment 3-way CE, control collapsed BCE, and mathematical score equivalence."""
    head = CompactOperationHead(HeadConfig())
    features = torch.randn(2, 20, 768)
    frame_lengths = torch.tensor([15, 20], dtype=torch.long)
    prompt_ids = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
    prompt_lengths = torch.tensor([3, 2], dtype=torch.long)

    out = head(
        features=features,
        frame_lengths=frame_lengths,
        prompt_phone_ids=prompt_ids,
        prompt_lengths=prompt_lengths
    )

    op_logits = out["operation_logits"]
    ctc_logits = out["ctc_logits"]
    assert op_logits.shape == (2, 3, 3)
    assert ctc_logits.shape == (2, 20, 40)

    operation_targets = torch.tensor([[0, 1, 2], [0, 0, -100]], dtype=torch.long)
    operation_mask = operation_targets != -100

    loss_treatment = task_loss(op_logits, operation_targets, operation_mask, objective="operation_ce")
    assert loss_treatment.ndim == 0 and torch.isfinite(loss_treatment) and loss_treatment > 0

    loss_control = task_loss(op_logits, operation_targets, operation_mask, objective="binary_collapsed")
    assert loss_control.ndim == 0 and torch.isfinite(loss_control) and loss_control > 0

    # CTC loss
    ctc_targets = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
    ctc_lengths = torch.tensor([3, 2], dtype=torch.long)
    loss_ctc = realized_ctc_loss(ctc_logits, frame_lengths, ctc_targets, ctc_lengths)
    assert loss_ctc.ndim == 0 and torch.isfinite(loss_ctc) and loss_ctc > 0

    # Mathematical identity check:
    # sigma(LSE(z1, z2) - z0) == 1 - softmax(z)[0]
    z = torch.randn(10, 3)
    collapsed = collapsed_binary_logit(z)
    s_error_collapsed = torch.sigmoid(collapsed)
    probs = z.softmax(dim=-1)
    s_error_softmax = probs[:, 1] + probs[:, 2]
    assert torch.allclose(s_error_collapsed, s_error_softmax, atol=1e-6)


def test_accumulation_groups_and_ledger():
    """Verify microbatch grouping and UpdateLedger overflow counting."""
    groups = accumulation_groups(5, accumulation=2)
    assert groups == (2, 2, 1)

    groups_even = accumulation_groups(6, accumulation=2)
    assert groups_even == (2, 2, 2)

    ledger = UpdateLedger()
    # Non-overflow step
    assert ledger.record_attempt(overflow=False) is True
    assert ledger.successful == 1 and ledger.skipped == 0 and ledger.attempts == 1

    # Overflow step
    assert ledger.record_attempt(overflow=True) is False
    assert ledger.successful == 1 and ledger.skipped == 1 and ledger.attempts == 2

    # Advance to fixed budget
    for _ in range(3599):
        ledger.record_attempt(overflow=False)
    assert ledger.successful == 3600
    assert ledger.fixed_budget_reached is True

    # Cannot exceed 3,600
    with pytest.raises(ValueError, match="budget"):
        ledger.record_attempt(overflow=False)


def test_learning_rate_schedule():
    """Verify warmup and cosine decay curve."""
    assert learning_rate_multiplier(0) == 0.0
    assert learning_rate_multiplier(50) == 0.5
    assert learning_rate_multiplier(100) == 1.0
    assert 0.0 < learning_rate_multiplier(3600) < 1.0


def test_checkpoint_roundtrip_and_sha256(tmp_path):
    """Verify dual checkpoint payload and SHA-256 verification."""
    head = CompactOperationHead(HeadConfig())
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1.0)
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)

    ckpt_file = tmp_path / "fixed_budget_3600.pt"
    sha = save_checkpoint(
        path=ckpt_file,
        head=head,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=5,
        successful_updates=3600,
        micro_steps=7200,
        best_ap=0.35,
        patience=0,
        skipped_amp=0,
        history=[],
        metadata={"objective": "operation_ce"},
        is_fixed_budget_endpoint=True
    )

    assert ckpt_file.is_file()
    computed_sha = hashlib.sha256(ckpt_file.read_bytes()).hexdigest()
    assert sha == computed_sha

    loaded = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    assert loaded["successful_updates"] == 3600
    assert loaded["is_fixed_budget_endpoint"] is True
    assert "head" in loaded and "optimizer" in loaded and "rng" in loaded


def test_runtime_rows_validation_for_treatment_and_control():
    """Verify runtime row outputs comply with rq1-runtime-v1 schema."""
    logits = torch.randn(3, 3)
    prompt_ids = [1, 2, 3]

    treatment_rows = operation_runtime_rows(
        logits=logits,
        prompt_phone_ids=prompt_ids,
        run_id="test_run",
        condition="matched",
        speaker_id="BWC",
        utterance_id="arctic_a0001",
        objective="operation_ce"
    )
    assert len(treatment_rows) == 3
    for r in treatment_rows:
        validate_runtime_row(r)
        assert r["scores"]["s_omit"] is not None
        assert r["probabilities"] is not None

    control_rows = operation_runtime_rows(
        logits=logits,
        prompt_phone_ids=prompt_ids,
        run_id="test_run",
        condition="matched",
        speaker_id="BWC",
        utterance_id="arctic_a0001",
        objective="binary_collapsed"
    )
    assert len(control_rows) == 3
    for r in control_rows:
        validate_runtime_row(r)
        assert r["scores"]["s_omit"] is None
        assert r["scores"]["s_sub"] is None
        assert r["probabilities"] is None
