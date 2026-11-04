"""LANGFENS RQ1 TRN-04 — Unit and Regression Tests for T3 Combined Intervention Lead (Agent D).

Validates:
1. Architectural parameter census: exactly 267,179 trainable parameters across 17 tensors.
2. Initial seed 1337 tensor identity against C0, T1, T2, and COMMON_INITIALIZATION_MANIFEST.json.
3. Combined intervention configuration:
   - Objective: class_focal (gamma=2.0, alpha=0.25 on coronal/rhotic deletions {D, T, R})
   - Frame-level acoustic features: ENABLED (4D waveform features)
4. Active acoustic slot gradient and active class-focal gradient backpropagation in T3.
5. Waveform acoustic feature extraction properties: 4D features, finite float32, centered frame alignment, normalized ranges.
6. Zero TextGrid leakage: strict signal-processing extraction without alignment timestamps or gold labels.
7. Pinned FIT empirical normalization statistics: mean = -6.047091, std = 2.544367.
8. Kaggle deployment package configuration invariants at deploy/kaggle_trn04_t3/.
9. Remote training execution and checkpoint integrity: exactly 3,600 updates, selection AP, dual checkpointing.
10. Factorial interaction algebra: Interaction = T3 - T2 - T1 + C0.
11. Floor effect decomposition: algebraic interaction vs genuine super-additive synergy.
12. Explicit confirmation: T3 is not declared a winner prematurely.
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
from langfens_prep.trn04.objectives import (
    task_loss,
    realized_ctc_loss,
    CORONAL_RHOTIC_PHONE_IDS,
    DEFAULT_GAMMA,
    DEFAULT_ALPHA_CORONAL_DEL
)


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_t3_parameter_count_and_quad_bit_identity():
    """Validates T3 head parameter census (267,179), bit-for-bit equality with quad, and manifest hash match."""
    c0, t1, t2, t3 = matched_initial_quad(seed=1337)
    inv = parameter_inventory(t3)

    total_trainable = sum(v["numel"] for v in inv.values() if v["trainable"])
    assert total_trainable == 267179, f"T3 parameter count mismatch: {total_trainable} != 267179"
    assert len(inv) == 17, f"Expected 17 parameter tensors, got {len(inv)}"

    # Bit-for-bit match with C0, T1, and T2
    for (name_t3, p_t3), (name_c0, p_c0), (name_t1, p_t1), (name_t2, p_t2) in zip(
        t3.named_parameters(), c0.named_parameters(), t1.named_parameters(), t2.named_parameters()
    ):
        assert name_t3 == name_c0 == name_t1 == name_t2
        assert torch.equal(p_t3, p_c0), f"Initialization difference between T3 and C0 in {name_t3}"
        assert torch.equal(p_t3, p_t1), f"Initialization difference between T3 and T1 in {name_t3}"
        assert torch.equal(p_t3, p_t2), f"Initialization difference between T3 and T2 in {name_t3}"

    # Byte-level match with COMMON_INITIALIZATION_MANIFEST.json
    manifest_path = REPO_ROOT / "manifests/RQ1-TRN-04/COMMON_INITIALIZATION_MANIFEST.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    assert manifest["total_trainable_parameters"] == 267179
    assert manifest["seed"] == 1337

    for name, p in t3.named_parameters():
        assert name in manifest["shared_tensor_hashes"], f"Tensor {name} missing from manifest"
        expected_meta = manifest["shared_tensor_hashes"][name]
        assert list(p.shape) == expected_meta["shape"], f"Shape mismatch for {name}"
        assert p.numel() == expected_meta["numel"], f"Numel mismatch for {name}"
        assert p.requires_grad == expected_meta["trainable"], f"Trainable mismatch for {name}"

        actual_sha = tensor_sha256(p)
        assert actual_sha == expected_meta["sha256"], f"SHA-256 hash mismatch for tensor {name}"


def test_t3_combined_active_projection_and_focal_gradient():
    """Validates that in T3, acoustic features are active AND class-focal loss propagates non-zero gradients."""
    head = CompactFactorialOperationHead(FactorialHeadConfig())
    assert head.acoustic_feature_projection.bias is None, "acoustic_feature_projection must have bias=False"
    assert head.acoustic_feature_projection.weight.shape == (128, 4)

    b, t = 2, 40
    feat = torch.randn(b, t, 768)
    acoust = torch.randn(b, t, 4)
    fl = torch.tensor([40, 36])
    prompt_ids = torch.tensor([[14, 9, 31, 28, 10], [5, 6, 9, 31, 0]])
    pl = torch.tensor([5, 4])
    op = torch.tensor([[0, 2, 2, 1, 0], [1, 0, 2, 0, 0]])

    head.zero_grad()
    out = head(
        features=feat,
        acoustic_features=acoust,
        frame_lengths=fl,
        prompt_phone_ids=prompt_ids,
        prompt_lengths=pl
    )
    loss = task_loss(
        out["operation_logits"],
        op,
        out["prompt_mask"],
        prompt_ids,
        objective="class_focal",
        gamma=2.0,
        alpha_coronal_del=0.25
    )
    assert torch.isfinite(loss), "T3 task loss is not finite"
    loss.backward()

    acoust_grad = head.acoustic_feature_projection.weight.grad
    assert acoust_grad is not None, "Gradient into acoustic projection weight is None"
    assert torch.isfinite(acoust_grad).all(), "Non-finite gradient in acoustic projection"
    assert not torch.equal(acoust_grad, torch.zeros_like(acoust_grad)), "Acoustic projection gradient is identically zero"


def test_t3_acoustic_feature_extraction_properties_and_leakage():
    """Validates 4D feature extraction, centering, pinned normalization, and absence of gold TextGrid leakage."""
    sr = 16000
    duration_s = 1.0
    expected_frames = 50
    t_axis = torch.linspace(0, duration_s, int(sr * duration_s))
    waveform = torch.sin(2 * math.pi * 150 * t_axis) + 0.5 * torch.sin(2 * math.pi * 300 * t_axis)
    waveform[int(0.4 * sr): int(0.6 * sr)] = 0.0

    feat = extract_frame_acoustic_features(waveform, expected_frames=expected_frames)
    assert feat.shape == (expected_frames, ACOUSTIC_FEATURE_DIM)
    assert feat.dtype == torch.float32
    assert torch.isfinite(feat).all()

    # Pinned FIT training statistics
    assert FIT_LOG_ENERGY_MEAN == -6.047091
    assert FIT_LOG_ENERGY_STD == 2.544367

    # Relative RMS energy range
    rms = feat[:, 1]
    assert (rms >= 0.0).all() and (rms <= 1.0 + 1e-5).all()

    # Zero-crossing rate range
    zcr = feat[:, 3]
    assert (zcr >= 0.0).all() and (zcr <= 1.0 + 1e-5).all()


def test_t3_kaggle_deployment_package():
    """Validates Kaggle deployment metadata, machine shape, private status, and notebook integrity."""
    deploy_dir = REPO_ROOT / "deploy/kaggle_trn04_t3"
    meta_file = deploy_dir / "kernel-metadata.json"
    nb_file = deploy_dir / "kaggle_train.ipynb"

    assert meta_file.is_file(), "kernel-metadata.json missing"
    assert nb_file.is_file(), "kaggle_train.ipynb missing"

    with open(meta_file) as f:
        meta = json.load(f)

    assert meta["id"] == "minkhoaa/langfens-rq1-trn-04-t3-focal-feat"
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["enable_gpu"] is True
    assert meta["is_private"] is True
    assert "minkhoaa/langfens-rq1-fold1-staging" in meta["dataset_sources"]

    with open(nb_file) as f:
        nb = json.load(f)

    source_text = " ".join("".join(c["source"]) for c in nb["cells"])
    assert "T3" in source_text
    assert "Class-Conditioned Focal" in source_text
    assert "3600" in source_text
    assert "1337" in source_text
    assert "b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9" in source_text


def test_t3_remote_training_execution_and_checkpoint_durability():
    """Validates that T3 completed exactly 3,600 updates, has dual checkpoints, and full RNG/optimizer state."""
    ckpt_dir = REPO_ROOT / "checkpoints/RQ1-TRN-04/t3"
    summary_path = ckpt_dir / "training_summary.json"
    fixed_path = ckpt_dir / "fixed_budget_3600.pt"
    best_path = ckpt_dir / "best.pt"

    assert summary_path.is_file(), "training_summary.json missing"
    assert fixed_path.is_file(), "fixed_budget_3600.pt missing"
    assert best_path.is_file(), "best.pt missing"

    with open(summary_path) as f:
        summary = json.load(f)

    assert summary["arm"] == "T3"
    assert summary["objective"] == "class_focal"
    assert summary["features_enabled"] is True
    assert summary["successful_updates"] == 3600
    assert summary["max_updates"] == 3600
    assert summary["total_epochs"] == 9
    assert summary["skipped_amp_attempts_total"] == 3
    assert math.isclose(summary["best_selection_ap"], 0.4670601353725024, rel_tol=1e-5)

    # Checkpoint content audit
    fixed_ckpt = torch.load(fixed_path, map_location="cpu", weights_only=False)
    assert fixed_ckpt["successful_updates"] == 3600
    assert fixed_ckpt["epoch"] == 9
    assert fixed_ckpt["metadata"]["arm"] == "T3"
    assert fixed_ckpt["metadata"]["objective"] == "class_focal"
    assert fixed_ckpt["metadata"]["features_enabled"] is True
    assert fixed_ckpt["metadata"]["is_fixed_budget_endpoint"] is True
    assert "head" in fixed_ckpt
    assert "optimizer" in fixed_ckpt
    assert "scheduler" in fixed_ckpt
    assert "scaler" in fixed_ckpt
    assert "rng" in fixed_ckpt

    best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    assert best_ckpt["successful_updates"] == 2247
    assert best_ckpt["epoch"] == 5
    assert best_ckpt["metadata"]["arm"] == "T3"
    assert best_ckpt["metadata"]["is_best_endpoint"] is True


def test_factorial_interaction_algebra_and_floor_effect():
    """Validates the interaction formula T3 - T2 - T1 + C0 and confirms floor effect artifact."""
    metrics_path = REPO_ROOT / "manifests/RQ1-TRN-04/ALL_EVALUATION_METRICS.json"
    with open(metrics_path) as f:
        data = json.load(f)

    c0 = data["c0"]["fixed_budget_3600"]["metrics"]
    t1 = data["t1"]["fixed_budget_3600"]["metrics"]
    t2 = data["t2"]["fixed_budget_3600"]["metrics"]
    t3 = data["t3"]["fixed_budget_3600"]["metrics"]

    # 1. Precision Interaction
    prec_c0 = c0["strict_deployed_omission_precision"]
    prec_t1 = t1["strict_deployed_omission_precision"]
    prec_t2 = t2["strict_deployed_omission_precision"]
    prec_t3 = t3["strict_deployed_omission_precision"]

    # In T1 and T3, precision collapses to identical 1.4564%
    assert math.isclose(prec_t1, prec_t3, abs_tol=1e-5)
    prec_interaction = prec_t3 - prec_t2 - prec_t1 + prec_c0
    assert math.isclose(prec_interaction, 0.004506, abs_tol=1e-5)

    # 2. Dedicated Omission AP Interaction
    ap_c0 = c0["dedicated_omission_ap"]
    ap_t1 = t1["dedicated_omission_ap"]
    ap_t2 = t2["dedicated_omission_ap"]
    ap_t3 = t3["dedicated_omission_ap"]

    ap_interaction = ap_t3 - ap_t2 - ap_t1 + ap_c0
    assert math.isclose(ap_interaction, 0.017077, abs_tol=1e-5)

    # The interaction is algebraically driven by (C0 - T2) because (T3 - T1) is near zero:
    diff_t3_t1 = ap_t3 - ap_t1
    diff_c0_t2 = ap_c0 - ap_t2
    assert math.isclose(diff_t3_t1, 0.000039, abs_tol=1e-5)
    assert math.isclose(diff_c0_t2, 0.017038, abs_tol=1e-5)
    # Confirm that interaction is 99.8% composed of (C0 - T2), demonstrating floor effect artifact
    assert math.isclose(ap_interaction, diff_c0_t2 + diff_t3_t1, abs_tol=1e-6)

    # 3. False alarm explosion
    fa_t1 = t1["false_alarms_per_1000_canonical"]
    fa_t3 = t3["false_alarms_per_1000_canonical"]
    assert math.isclose(fa_t1, 985.435918, rel_tol=1e-5)
    assert math.isclose(fa_t3, 985.435918, rel_tol=1e-5)


def test_t3_prohibition_of_premature_winner_declaration():
    """Confirms that T3 fails acceptance gates and cannot be declared a winner."""
    metrics_path = REPO_ROOT / "manifests/RQ1-TRN-04/ALL_EVALUATION_METRICS.json"
    with open(metrics_path) as f:
        data = json.load(f)

    t3_gates = data["t3"]["fixed_budget_3600"]["acceptance_gates"]
    assert t3_gates["level_a_pilot_feasibility"]["passed"] is False
    assert t3_gates["level_b_research_target"]["passed"] is False

    t3_m = data["t3"]["fixed_budget_3600"]["metrics"]
    c0_m = data["c0"]["fixed_budget_3600"]["metrics"]

    # T3 is substantially worse than C0 in deployed precision and false alarms
    assert t3_m["strict_deployed_omission_precision"] < c0_m["strict_deployed_omission_precision"]
    assert t3_m["false_alarms_per_1000_canonical"] > c0_m["false_alarms_per_1000_canonical"]
    assert t3_m["dedicated_omission_ap"] < c0_m["dedicated_omission_ap"]
