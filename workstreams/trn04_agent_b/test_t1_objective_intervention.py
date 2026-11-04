"""LANGFENS RQ1 TRN-04 — Unit and Regression Tests for T1 Objective Intervention Lead (Agent B).

Validates:
1. Architectural parameter census: exactly 267,179 trainable parameters across 17 tensors.
2. Initial seed 1337 tensor identity against C0 and COMMON_INITIALIZATION_MANIFEST.json.
3. Acoustic feature slot disabled invariance (bias=False, zero input -> zero output, zero gradient).
4. FIT cohort deletion distribution: 852 coronal/rhotic deletions out of 1,503 total deletions (56.69%).
5. Class-Conditioned Focal Loss mathematical formulation: gamma=2.0, alpha=0.25 on coronal/rhotic deletions, alpha=1.0 otherwise.
6. Gradient dynamics comparison: Focal vs CE (easy-example attenuation and coronal deletion gradient scaling).
7. Kaggle deployment package configuration invariants.
8. HuggingFace mirror receipt and training integrity (3,600 updates, selection AP).
"""
from __future__ import annotations

import hashlib
import json
import math
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
from langfens_prep.trn04.objectives import (
    task_loss,
    realized_ctc_loss,
    CORONAL_RHOTIC_PHONE_IDS,
    DEFAULT_GAMMA,
    DEFAULT_ALPHA_CORONAL_DEL,
    build_coronal_lookup
)
from langfens_prep.p_sup.dataset import PSupL2ArcticDataset


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def test_t1_parameter_count_and_c0_bit_identity():
    """Validates T1 head parameter census (267,179), bit-for-bit equality with C0, and manifest hash match."""
    c0, t1, t2, t3 = matched_initial_quad(seed=1337)
    inv = parameter_inventory(t1)

    total_trainable = sum(v["numel"] for v in inv.values() if v["trainable"])
    assert total_trainable == 267179, f"T1 parameter count mismatch: {total_trainable} != 267179"
    assert len(inv) == 17, f"Expected 17 parameter tensors, got {len(inv)}"

    # Bit-for-bit match with C0
    for (name_c0, p_c0), (name_t1, p_t1) in zip(c0.named_parameters(), t1.named_parameters()):
        assert name_c0 == name_t1
        assert torch.equal(p_c0, p_t1), f"Initialization difference between C0 and T1 in {name_c0}"

    # Byte-level match with COMMON_INITIALIZATION_MANIFEST.json
    manifest_path = REPO_ROOT / "manifests/RQ1-TRN-04/COMMON_INITIALIZATION_MANIFEST.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    assert manifest["total_trainable_parameters"] == 267179
    assert manifest["seed"] == 1337

    for name, p in t1.named_parameters():
        assert name in manifest["shared_tensor_hashes"], f"Tensor {name} missing from manifest"
        expected_meta = manifest["shared_tensor_hashes"][name]
        assert list(p.shape) == expected_meta["shape"], f"Shape mismatch for {name}"
        assert p.numel() == expected_meta["numel"], f"Numel mismatch for {name}"
        assert p.requires_grad == expected_meta["trainable"], f"Trainable mismatch for {name}"

        actual_sha = tensor_sha256(p)
        assert actual_sha == expected_meta["sha256"], f"SHA-256 hash mismatch for tensor {name}"


def test_t1_acoustic_slot_disabled_invariance():
    """Validates that in T1, acoustic features are disabled (bias=False, zero input -> zero projection & zero gradient)."""
    head = CompactFactorialOperationHead(FactorialHeadConfig())

    # Verify bias=False
    assert head.acoustic_feature_projection.bias is None, "acoustic_feature_projection must have bias=False"

    # Forward zero features through standalone projection
    zero_acoust = torch.zeros(2, 40, 4)
    proj = head.acoustic_feature_projection(zero_acoust)
    assert torch.equal(proj, torch.zeros(2, 40, 128)), "Zero features must yield strictly zero projection"

    # Forward full head and verify gradient into acoustic_feature_projection.weight is identically zero
    b, t = 2, 40
    feat = torch.randn(b, t, 768)
    fl = torch.tensor([40, 36])
    prompt_ids = torch.tensor([[14, 9, 31, 28, 10], [5, 6, 9, 31, 0]])
    pl = torch.tensor([5, 4])
    mask = torch.arange(5)[None] < pl[:, None]
    operation = torch.tensor([[0, 2, 2, 1, 0], [1, 0, 2, 2, 0]])

    head.zero_grad()
    out = head(
        features=feat,
        acoustic_features=zero_acoust,
        frame_lengths=fl,
        prompt_phone_ids=prompt_ids,
        prompt_lengths=pl
    )
    loss = task_loss(
        out["operation_logits"],
        operation,
        out["prompt_mask"],
        prompt_ids,
        objective="class_focal"
    )
    loss.backward()

    # Gradient into acoustic_feature_projection.weight must be identically zero
    assert head.acoustic_feature_projection.weight.grad is not None
    assert torch.equal(head.acoustic_feature_projection.weight.grad, torch.zeros(128, 4)), (
        "Non-zero gradient leaked into acoustic_feature_projection.weight in disabled slot mode!"
    )


def test_t1_fit_deletion_distribution_and_coronal_fraction():
    """Validates the empirical justification for T1: D, T, R account for 56.69% of FIT deletions."""
    split_manifest = REPO_ROOT / "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json"
    zip_path = REPO_ROOT / "datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip"

    ds = PSupL2ArcticDataset(str(split_manifest), str(zip_path), split="fit")
    assert len(ds) == 1799, f"Expected 1799 FIT utterances, got {len(ds)}"

    d_del = 0  # D: phone 9
    t_del = 0  # T: phone 31
    r_del = 0  # R: phone 28
    total_deletions = 0

    for sample in ds:
        ops = sample["labels"].operation
        phones = sample["labels"].canonical_ids
        for op, ph in zip(ops, phones):
            if op.item() == 2:  # deletion
                total_deletions += 1
                ph_id = ph.item()
                if ph_id == 9:
                    d_del += 1
                elif ph_id == 31:
                    t_del += 1
                elif ph_id == 28:
                    r_del += 1

    assert total_deletions == 1503, f"Expected 1503 total deletions in FIT, got {total_deletions}"
    assert d_del == 323, f"Expected 323 D deletions in FIT, got {d_del}"
    assert t_del == 273, f"Expected 273 T deletions in FIT, got {t_del}"
    assert r_del == 256, f"Expected 256 R deletions in FIT, got {r_del}"

    coronal_rhotic_sum = d_del + t_del + r_del
    assert coronal_rhotic_sum == 852, f"Expected 852 coronal/rhotic deletions, got {coronal_rhotic_sum}"

    percentage = (coronal_rhotic_sum / total_deletions) * 100.0
    assert abs(percentage - 56.6866) < 0.01, f"Expected 56.69%, got {percentage:.4f}%"


def test_t1_class_focal_loss_formulation_and_weights():
    """Validates class focal loss mathematical behavior on individual phoneme operations."""
    lookup = build_coronal_lookup(torch.device("cpu"))
    for ph in CORONAL_RHOTIC_PHONE_IDS:
        assert lookup[ph].item() is True

    # Case 1: Coronal deletion (D [9], y=2) -> alpha=0.25
    logits = torch.tensor([[[0.0, 0.0, 2.0]]], requires_grad=True)  # (1, 1, 3)
    operation = torch.tensor([[2]])
    mask = torch.tensor([[True]])
    prompt_ids = torch.tensor([[9]])  # D

    loss = task_loss(logits, operation, mask, prompt_ids, objective="class_focal", gamma=2.0, alpha_coronal_del=0.25)

    probs = F.softmax(logits[0, 0].float(), dim=-1)
    p_t = probs[2].item()
    expected_focal = - 0.25 * ((1.0 - p_t) ** 2.0) * math.log(p_t)
    assert abs(loss.item() - expected_focal) < 1e-6, f"Mismatch: {loss.item()} vs {expected_focal}"

    # Case 2: Non-coronal deletion (e.g., S [29], y=2) -> alpha=1.0
    prompt_ids_non_coronal = torch.tensor([[29]])
    loss_non_coronal = task_loss(logits, operation, mask, prompt_ids_non_coronal, objective="class_focal", gamma=2.0, alpha_coronal_del=0.25)
    expected_non_coronal = - 1.0 * ((1.0 - p_t) ** 2.0) * math.log(p_t)
    assert abs(loss_non_coronal.item() - expected_non_coronal) < 1e-6

    # Ratio must be exactly 0.25
    assert abs(loss.item() / loss_non_coronal.item() - 0.25) < 1e-6

    # Case 3: Substitution (y=1) on coronal phone (D [9]) -> alpha=1.0 (only deletions are down-weighted)
    operation_sub = torch.tensor([[1]])
    loss_sub = task_loss(logits, operation_sub, mask, prompt_ids, objective="class_focal", gamma=2.0, alpha_coronal_del=0.25)
    p_t_sub = probs[1].item()
    expected_sub = - 1.0 * ((1.0 - p_t_sub) ** 2.0) * math.log(p_t_sub)
    assert abs(loss_sub.item() - expected_sub) < 1e-6


def test_t1_gradient_dynamics_focal_vs_ce():
    """Validates gradient dynamics: easy-example attenuation and coronal deletion gradient scaling."""
    # Simulate an easy example: p_t ≈ 0.9
    z_easy = torch.tensor([[0.0, 0.0, 3.5]], requires_grad=True)
    op = torch.tensor([2])
    mask = torch.tensor([[True]])
    p_non_coronal = torch.tensor([[15]])  # non-coronal
    p_coronal = torch.tensor([[9]])      # coronal (D)

    loss_ce = F.cross_entropy(z_easy, op)
    loss_ce.backward()
    grad_ce = z_easy.grad[0, 2].item()

    # Focal on easy non-coronal
    z_focal_easy = torch.tensor([[0.0, 0.0, 3.5]], requires_grad=True)
    loss_focal_easy = task_loss(z_focal_easy.unsqueeze(0), op.unsqueeze(0), mask, p_non_coronal, objective="class_focal", gamma=2.0)
    loss_focal_easy.backward()
    grad_focal_easy = z_focal_easy.grad[0, 2].item()

    # Focal gradient on easy example must be strongly attenuated compared to CE
    prob_t = F.softmax(torch.tensor([0.0, 0.0, 3.5]), dim=-1)[2].item()
    assert prob_t > 0.9
    attenuation_ratio = abs(grad_focal_easy / grad_ce)
    assert attenuation_ratio < 0.05, f"Expected heavy attenuation on easy example, got {attenuation_ratio}"

    # Focal on easy coronal: down-weighted further by 0.25
    z_focal_coronal = torch.tensor([[0.0, 0.0, 3.5]], requires_grad=True)
    loss_focal_coronal = task_loss(z_focal_coronal.unsqueeze(0), op.unsqueeze(0), mask, p_coronal, objective="class_focal", gamma=2.0, alpha_coronal_del=0.25)
    loss_focal_coronal.backward()
    grad_focal_coronal = z_focal_coronal.grad[0, 2].item()

    assert abs(grad_focal_coronal / grad_focal_easy - 0.25) < 1e-4, "Coronal gradient must be scaled by 0.25"


def test_t1_deployment_package_integrity():
    """Validates Kaggle deployment metadata and training notebook for Arm T1."""
    deploy_dir = REPO_ROOT / "deploy/kaggle_trn04_t1"
    meta_path = deploy_dir / "kernel-metadata.json"
    nb_path = deploy_dir / "kaggle_train.ipynb"

    assert meta_path.is_file(), f"Missing metadata at {meta_path}"
    assert nb_path.is_file(), f"Missing notebook at {nb_path}"

    with open(meta_path) as f:
        meta = json.load(f)

    assert meta["id"] == "minkhoaa/langfens-rq1-trn-04-t1-focal-nofeat"
    assert meta["title"] == "langfens-rq1-trn-04-t1-focal-nofeat"
    assert meta["code_file"] == "kaggle_train.ipynb"
    assert meta["is_private"] is True
    assert meta["enable_gpu"] is True
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert "minkhoaa/langfens-rq1-fold1-staging" in meta["dataset_sources"]

    with open(nb_path) as f:
        nb = json.load(f)

    full_nb_text = "".join("".join(c["source"]) for c in nb["cells"])
    assert 'arm_name = "T1"' in full_nb_text
    assert "--seed 1337" in full_nb_text or '"--seed", "1337"' in full_nb_text
    assert "--max-updates 3600" in full_nb_text or '"--max-updates", "3600"' in full_nb_text
    assert "b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9" in full_nb_text


def test_t1_remote_execution_and_hf_mirror_receipt():
    """Validates T1 remote execution receipt mirrored on HuggingFace."""
    from huggingface_hub import hf_hub_download

    summary_file = hf_hub_download(
        repo_id="minkhoaa/langfens-rq1-cross-fold-checkpoints",
        filename="trn04/t1/training_summary.json"
    )
    with open(summary_file) as f:
        receipt = json.load(f)

    assert receipt["arm"] == "T1"
    assert receipt["objective"] == "class_focal"
    assert receipt["features_enabled"] is False
    assert receipt["seed"] == 1337
    assert receipt["max_updates"] == 3600
    assert receipt["successful_updates"] == 3600
    assert receipt["total_epochs"] == 9
    assert abs(receipt["best_selection_ap"] - 0.46191769787543335) < 1e-6
    assert len(receipt["history"]) == 9
    assert receipt["skipped_amp_attempts_total"] == 3
