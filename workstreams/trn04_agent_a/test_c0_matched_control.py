"""LANGFENS RQ1 TRN-04 — Unit and Regression Tests for C0 Matched Control Lead (Agent A).

Validates:
1. Architectural parameter census: exactly 267,179 trainable parameters across 17 tensors.
2. Initial seed 1337 tensor identity against COMMON_INITIALIZATION_MANIFEST.json.
3. Acoustic feature slot null invariance (bias=False, zero input -> zero output, zero gradient).
4. Operation CE loss formulation and gradient properties.
5. Speaker partition disjointness, counts, and sealed test isolation.
6. Kaggle deployment configuration invariants.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path("/home/khoa/Projects/KLTN/Dataset")
import sys
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from langfens_prep.trn04.model import (
    CompactFactorialOperationHead,
    FactorialHeadConfig,
    matched_initial_quad,
    parameter_inventory
)
from langfens_prep.trn04.objectives import task_loss, realized_ctc_loss
from langfens_prep.p_sup.dataset import (
    PSupL2ArcticDataset,
    FIT_SPEAKERS,
    SELECT_SPEAKERS,
    CALIBRATION_SPEAKERS,
    VAL_SPEAKERS,
    SealedSpeakerContaminationError
)
from langfens_prep.p_sup.runtime import SEALED_SPEAKERS, CONTRACT_SHA256


def test_c0_parameter_count_and_manifest_identity():
    """Validates C0 head parameter census and exact byte-level match to initialization manifest."""
    c0, t1, t2, t3 = matched_initial_quad(seed=1337)
    inv = parameter_inventory(c0)

    total_trainable = sum(v["numel"] for v in inv.values() if v["trainable"])
    assert total_trainable == 267179, f"C0 parameter count mismatch: {total_trainable} != 267179"
    assert len(inv) == 17, f"Expected 17 parameter tensors, got {len(inv)}"

    manifest_path = REPO_ROOT / "manifests/RQ1-TRN-04/COMMON_INITIALIZATION_MANIFEST.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    assert manifest["total_trainable_parameters"] == 267179
    assert manifest["seed"] == 1337

    for name, p in c0.named_parameters():
        assert name in manifest["shared_tensor_hashes"], f"Tensor {name} missing from manifest"
        expected_meta = manifest["shared_tensor_hashes"][name]
        assert list(p.shape) == expected_meta["shape"], f"Shape mismatch for {name}"
        assert p.numel() == expected_meta["numel"], f"Numel mismatch for {name}"
        assert p.requires_grad == expected_meta["trainable"], f"Trainable status mismatch for {name}"

        actual_sha = hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        assert actual_sha == expected_meta["sha256"], f"SHA-256 hash mismatch for tensor {name}"


def test_c0_acoustic_slot_zero_input_and_gradient_invariance():
    """Validates that C0 acoustic slot has bias=False, outputs zero on zero inputs, and receives zero gradient."""
    head = CompactFactorialOperationHead(FactorialHeadConfig())

    # Verify bias=False
    assert head.acoustic_feature_projection.bias is None, "acoustic_feature_projection must have bias=False"

    # Forward zero features
    b, t = 2, 45
    feat = torch.randn(b, t, 768)
    fl = torch.tensor([45, 40])
    zero_acoust = torch.zeros(b, t, 4)
    prompt_ids = torch.randint(1, 39, (b, 10))
    pl = torch.tensor([10, 8])
    mask = torch.arange(10)[None] < pl[:, None]
    prompt_ids = prompt_ids.masked_fill(~mask, 0)
    targets = torch.randint(0, 3, (b, 10))

    # Test standalone projection
    proj = head.acoustic_feature_projection(zero_acoust)
    assert torch.equal(proj, torch.zeros_like(proj)), "Projection of zero acoustic features must be strictly zero"

    # Forward full head
    out = head(
        features=feat,
        acoustic_features=zero_acoust,
        frame_lengths=fl,
        prompt_phone_ids=prompt_ids,
        prompt_lengths=pl
    )

    loss = task_loss(
        logits=out["operation_logits"],
        operation=targets,
        mask=mask,
        prompt_phone_ids=prompt_ids,
        objective="operation_ce"
    )

    loss.backward()

    # Weight gradient for acoustic_feature_projection must be strictly zero
    w_grad = head.acoustic_feature_projection.weight.grad
    assert w_grad is not None
    assert torch.equal(w_grad, torch.zeros_like(w_grad)), "Gradient for acoustic projection must be strictly zero when features are zero"


def test_c0_loss_formulation_equivalence():
    """Validates that operation_ce matches standard PyTorch cross entropy and respects masking."""
    b, n = 3, 12
    logits = torch.randn(b, n, 3, requires_grad=True)
    targets = torch.randint(0, 3, (b, n))
    prompt_lengths = torch.tensor([12, 8, 5])
    mask = torch.arange(n)[None] < prompt_lengths[:, None]
    prompt_ids = torch.randint(1, 39, (b, n)).masked_fill(~mask, 0)

    loss_custom = task_loss(logits, targets, mask, prompt_ids, objective="operation_ce")

    valid_logits = logits[mask].float()
    valid_targets = targets[mask]
    loss_expected = F.cross_entropy(valid_logits, valid_targets)

    assert torch.isclose(loss_custom, loss_expected, atol=1e-6), "operation_ce does not match F.cross_entropy"


def test_c0_speaker_partitions_and_sealed_test_isolation():
    """Validates frozen speaker partition sets, counts, and fail-closed test sealing."""
    # Disjointness
    assert len(FIT_SPEAKERS) == 12
    assert len(SELECT_SPEAKERS) == 2
    assert len(CALIBRATION_SPEAKERS) == 2
    assert len(VAL_SPEAKERS) == 2
    assert len(SEALED_SPEAKERS) == 6

    partitions = [FIT_SPEAKERS, SELECT_SPEAKERS, CALIBRATION_SPEAKERS, VAL_SPEAKERS, SEALED_SPEAKERS]
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            assert not (partitions[i] & partitions[j]), f"Speaker overlap between partition {i} and {j}"

    # Partition counts from dataset
    split_manifest = REPO_ROOT / "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json"
    zip_path = REPO_ROOT / "datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip"

    ds_fit = PSupL2ArcticDataset(str(split_manifest), str(zip_path), split="fit")
    assert len(ds_fit) == 1799

    ds_sel = PSupL2ArcticDataset(str(split_manifest), str(zip_path), split="selection")
    assert len(ds_sel) == 300

    ds_cal = PSupL2ArcticDataset(str(split_manifest), str(zip_path), split="calibration")
    assert len(ds_cal) == 300

    ds_val = PSupL2ArcticDataset(str(split_manifest), str(zip_path), split="val")
    assert len(ds_val) == 300

    # Fail-closed test access
    with pytest.raises(SealedSpeakerContaminationError):
        PSupL2ArcticDataset(str(split_manifest), str(zip_path), split="test")


def test_c0_deployment_package_integrity():
    """Validates kernel-metadata.json and kaggle_train.ipynb configuration."""
    deploy_dir = REPO_ROOT / "deploy/kaggle_trn04_c0"
    with open(deploy_dir / "kernel-metadata.json") as f:
        meta = json.load(f)

    assert meta["id"] == "minkhoaa/langfens-rq1-trn-04-c0-ce-nofeat"
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["enable_gpu"] is True
    assert meta["is_private"] is True
    assert "minkhoaa/langfens-rq1-fold1-staging" in meta["dataset_sources"]

    with open(deploy_dir / "kaggle_train.ipynb") as f:
        nb = json.load(f)

    source_text = " ".join("".join(c.get("source", [])) for c in nb["cells"])
    assert "EXPECTED_SHA = \"b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9\"" in source_text
    assert "--arm" in source_text and "arm_name = \"C0\"" in source_text
    assert "--max-updates" in source_text and "\"3600\"" in source_text
    assert "--seed" in source_text and "\"1337\"" in source_text
