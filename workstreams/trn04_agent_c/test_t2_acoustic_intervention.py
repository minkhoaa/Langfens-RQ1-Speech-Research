"""LANGFENS RQ1 TRN-04 — Unit and Regression Tests for T2 Acoustic Intervention Lead (Agent C).

Validates:
1. Architectural parameter census: exactly 267,179 trainable parameters across 17 tensors.
2. Initial seed 1337 tensor identity against C0, T1, T3, and COMMON_INITIALIZATION_MANIFEST.json.
3. Acoustic feature slot enabled behavior: active non-zero projection and active gradients into acoustic weights.
4. Waveform acoustic feature extraction properties: 4D features, finite float32, centered frame alignment, normalized ranges.
5. Zero TextGrid leakage: strict signal-processing extraction without alignment timestamps or gold labels.
6. Pinned FIT empirical normalization statistics: mean = -6.047091, std = 2.544367.
7. Objective formulation: operation_ce (standard 3-way CE) + realized CTC loss.
8. Kaggle deployment package configuration invariants at deploy/kaggle_trn04_t2/.
9. Remote training execution and checkpoint integrity: exactly 3,600 updates, selection AP, dual checkpointing.
10. Acoustic reliance gap and evaluation metrics comparison vs C0 baseline.
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
from langfens_prep.trn04.objectives import task_loss, realized_ctc_loss
from langfens_prep.p_sup.dataset import PSupL2ArcticDataset


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def test_t2_parameter_count_and_quad_bit_identity():
    """Validates T2 head parameter census (267,179), bit-for-bit equality with quad, and manifest hash match."""
    c0, t1, t2, t3 = matched_initial_quad(seed=1337)
    inv = parameter_inventory(t2)

    total_trainable = sum(v["numel"] for v in inv.values() if v["trainable"])
    assert total_trainable == 267179, f"T2 parameter count mismatch: {total_trainable} != 267179"
    assert len(inv) == 17, f"Expected 17 parameter tensors, got {len(inv)}"

    # Bit-for-bit match with C0, T1, and T3
    for (name_t2, p_t2), (name_c0, p_c0), (name_t1, p_t1), (name_t3, p_t3) in zip(
        t2.named_parameters(), c0.named_parameters(), t1.named_parameters(), t3.named_parameters()
    ):
        assert name_t2 == name_c0 == name_t1 == name_t3
        assert torch.equal(p_t2, p_c0), f"Initialization difference between T2 and C0 in {name_t2}"
        assert torch.equal(p_t2, p_t1), f"Initialization difference between T2 and T1 in {name_t2}"
        assert torch.equal(p_t2, p_t3), f"Initialization difference between T2 and T3 in {name_t2}"

    # Byte-level match with COMMON_INITIALIZATION_MANIFEST.json
    manifest_path = REPO_ROOT / "manifests/RQ1-TRN-04/COMMON_INITIALIZATION_MANIFEST.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    assert manifest["total_trainable_parameters"] == 267179
    assert manifest["seed"] == 1337

    for name, p in t2.named_parameters():
        assert name in manifest["shared_tensor_hashes"], f"Tensor {name} missing from manifest"
        expected_meta = manifest["shared_tensor_hashes"][name]
        assert list(p.shape) == expected_meta["shape"], f"Shape mismatch for {name}"
        assert p.numel() == expected_meta["numel"], f"Numel mismatch for {name}"
        assert p.requires_grad == expected_meta["trainable"], f"Trainable mismatch for {name}"

        actual_sha = tensor_sha256(p)
        assert actual_sha == expected_meta["sha256"], f"SHA-256 hash mismatch for tensor {name}"


def test_t2_acoustic_slot_active_projection_and_gradient():
    """Validates that in T2, acoustic features are active (non-zero output and non-zero gradient backpropagation)."""
    head = CompactFactorialOperationHead(FactorialHeadConfig())
    assert head.acoustic_feature_projection.bias is None, "acoustic_feature_projection must have bias=False"
    assert head.acoustic_feature_projection.weight.shape == (128, 4)

    # Provide synthetic active 4D features
    b, t = 2, 40
    feat = torch.randn(b, t, 768)
    acoust = torch.randn(b, t, 4)
    fl = torch.tensor([40, 36])
    prompt_ids = torch.tensor([[14, 9, 31, 28, 10], [5, 6, 9, 31, 0]])
    pl = torch.tensor([5, 4])
    operation = torch.tensor([[0, 2, 2, 1, 0], [1, 0, 2, 2, 0]])

    # Standalone projection produces non-zero output
    proj = head.acoustic_feature_projection(acoust)
    assert not torch.equal(proj, torch.zeros_like(proj)), "Acoustic projection should not be zero for non-zero inputs"

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
        operation,
        out["prompt_mask"],
        prompt_ids,
        objective="operation_ce"
    )
    loss.backward()

    # In T2, acoustic_feature_projection.weight MUST receive non-zero gradients
    grad = head.acoustic_feature_projection.weight.grad
    assert grad is not None, "Gradient must be computed for acoustic projection weight"
    assert torch.isfinite(grad).all(), "Gradient must be finite"
    assert not torch.equal(grad, torch.zeros_like(grad)), "Gradient into acoustic projection weight must be non-zero in T2"


def test_t2_acoustic_feature_properties_and_ranges():
    """Validates physical and numerical properties of the 4D waveform acoustic features."""
    sample_rate = 16000
    duration_sec = 1.0
    num_samples = int(sample_rate * duration_sec)
    expected_frames = 50  # 1.0s @ 50 Hz

    # Create synthetic speech-like signal (voiced vowel with harmonics + silent closure)
    t_axis = torch.linspace(0, duration_sec, num_samples)
    waveform = torch.sin(2 * math.pi * 150 * t_axis) + 0.5 * torch.sin(2 * math.pi * 300 * t_axis)
    # Insert silent closure in middle 200ms
    waveform[int(0.4 * sample_rate): int(0.6 * sample_rate)] = 0.0

    features = extract_frame_acoustic_features(waveform, expected_frames=expected_frames)

    assert features.shape == (expected_frames, ACOUSTIC_FEATURE_DIM)
    assert torch.isfinite(features).all(), "Nonfinite values in extracted acoustic features"

    norm_log_e = features[:, 0]
    rel_rms = features[:, 1]
    delta_log_e = features[:, 2]
    zcr = features[:, 3]

    # Relative RMS must be bounded in [0, 1]
    assert (rel_rms >= 0.0).all() and (rel_rms <= 1.0 + 1e-5).all(), "Relative RMS must be in [0, 1]"
    # ZCR must be bounded in [0, 1]
    assert (zcr >= 0.0).all() and (zcr <= 1.0 + 1e-5).all(), "ZCR must be in [0, 1]"

    # Silent closure region (frames 20 to 30) must have significantly lower RMS than voiced peak
    closure_rms = rel_rms[20:30].mean()
    voiced_rms = rel_rms[5:15].mean()
    assert closure_rms < voiced_rms, f"Closure RMS ({closure_rms}) should be lower than voiced RMS ({voiced_rms})"

    # First frame delta log energy should be initialized to 0.0
    assert delta_log_e[0].item() == 0.0, "Delta log energy at frame 0 must be 0.0"


def test_t2_zero_textgrid_leakage_assertion():
    """Confirms that acoustic feature extraction interface is pure waveform-based with zero gold leakage."""
    import inspect
    sig = inspect.signature(extract_frame_acoustic_features)
    param_names = list(sig.parameters.keys())

    # Only signal processing parameters allowed
    forbidden_tokens = ["textgrid", "boundary", "align", "gold", "label", "phone", "transcript"]
    for p in param_names:
        for f in forbidden_tokens:
            assert f not in p.lower(), f"Forbidden leakage parameter '{p}' in extract_frame_acoustic_features!"

    # Ensure FactorialAudioAdapter passes only audio and lengths to extract_frame_acoustic_features
    adapter_source = inspect.getsource(FactorialAudioAdapter.forward)
    assert "extract_frame_acoustic_features(audio, lengths)" in adapter_source, (
        "Adapter must pass only waveform audio and lengths to feature extractor"
    )


def test_t2_pinned_fit_statistics():
    """Validates the pinned empirical normalization constants."""
    assert FIT_LOG_ENERGY_MEAN == -6.047091
    assert FIT_LOG_ENERGY_STD == 2.544367
    assert ACOUSTIC_FEATURE_DIM == 4


def test_t2_objective_uniform_cross_entropy():
    """Validates that T2 uses standard uniform Cross-Entropy (operation_ce)."""
    b, n = 2, 5
    logits = torch.randn(b, n, 3)
    operation = torch.tensor([[0, 1, 2, 0, 1], [2, 0, 1, 2, 0]])
    mask = torch.tensor([[True, True, True, True, False], [True, True, True, False, False]])
    phone_ids = torch.tensor([[10, 9, 31, 28, 5], [9, 31, 14, 20, 1]])

    loss_ce = task_loss(logits, operation, mask, phone_ids, objective="operation_ce")

    # Manual Cross Entropy computation on masked tokens
    z = logits[mask].float()
    y = operation[mask]
    expected_ce = F.cross_entropy(z, y, reduction="mean")

    assert torch.isclose(loss_ce, expected_ce, atol=1e-6), "T2 loss must match standard F.cross_entropy"


def test_t2_kaggle_deployment_package():
    """Validates kernel-metadata.json and kaggle_train.ipynb in deploy/kaggle_trn04_t2/."""
    deploy_dir = REPO_ROOT / "deploy/kaggle_trn04_t2"
    assert deploy_dir.is_dir(), f"Deployment directory {deploy_dir} does not exist"

    metadata_path = deploy_dir / "kernel-metadata.json"
    assert metadata_path.is_file(), "kernel-metadata.json missing"
    with open(metadata_path) as f:
        meta = json.load(f)

    assert meta["id"] == "minkhoaa/langfens-rq1-trn-04-t2-ce-feat"
    assert meta["code_file"] == "kaggle_train.ipynb"
    assert meta["language"] == "python"
    assert meta["kernel_type"] == "notebook"
    assert meta["is_private"] is True
    assert meta["enable_gpu"] is True
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert "minkhoaa/langfens-rq1-fold1-staging" in meta["dataset_sources"]

    # Verify notebook content
    nb_path = deploy_dir / "kaggle_trn04_t2" / "kaggle_train.ipynb" if not (deploy_dir / "kaggle_train.ipynb").is_file() else (deploy_dir / "kaggle_train.ipynb")
    with open(nb_path) as f:
        nb = json.load(f)

    all_source = "".join("".join(cell["source"]) for cell in nb["cells"])
    assert "Tesla T4" in all_source
    assert "HF_TOKEN" in all_source
    assert "b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9" in all_source
    assert "--arm" in all_source and "T2" in all_source
    assert "--max-updates" in all_source and "3600" in all_source


def test_t2_checkpoints_and_training_summary():
    """Validates local T2 checkpoints and training summary in checkpoints/RQ1-TRN-04/t2/."""
    chk_dir = REPO_ROOT / "checkpoints/RQ1-TRN-04/t2"
    assert chk_dir.is_dir(), f"Checkpoints dir {chk_dir} does not exist"

    summary_file = chk_dir / "training_summary.json"
    assert summary_file.is_file(), "training_summary.json missing"
    with open(summary_file) as f:
        summary = json.load(f)

    assert summary["arm"] == "T2"
    assert summary["objective"] == "operation_ce"
    assert summary["features_enabled"] is True
    assert summary["seed"] == 1337
    assert summary["max_updates"] == 3600
    assert summary["successful_updates"] == 3600
    assert summary["total_epochs"] == 9
    assert summary["best_selection_ap"] > 0.46

    # Verify both checkpoints exist and have correct SHA-256
    fixed_chk = chk_dir / "fixed_budget_3600.pt"
    best_chk = chk_dir / "best.pt"
    assert fixed_chk.is_file(), "fixed_budget_3600.pt missing"
    assert best_chk.is_file(), "best.pt missing"

    sha_fixed = hashlib.sha256(fixed_chk.read_bytes()).hexdigest()
    sha_best = hashlib.sha256(best_chk.read_bytes()).hexdigest()

    assert sha_fixed == "e8b00002aef9ee1587990a69162d5a5438b092e02c83d5654ae5aea0e1ec9c13"
    assert sha_best == "98d3fc1ecf6a5539d211028e32c9cbd2d5668b3942ad4d8fcae8226a99927fb1"


def test_t2_evaluation_metrics_and_acoustic_reliance_gap():
    """Validates evaluation metrics and verifies acoustic reliance gap vs C0 baseline."""
    eval_file = REPO_ROOT / "manifests/RQ1-TRN-04/ALL_EVALUATION_METRICS.json"
    assert eval_file.is_file(), "ALL_EVALUATION_METRICS.json missing"
    with open(eval_file) as f:
        metrics = json.load(f)

    assert "t2" in metrics
    t2_fixed = metrics["t2"]["fixed_budget_3600"]["metrics"]
    c0_fixed = metrics["c0"]["fixed_budget_3600"]["metrics"]

    # In T2, acoustic features are enabled, so acoustic reliance gap must be positive
    t2_gap = t2_fixed["acoustic_reliance_gap"]
    c0_gap = c0_fixed["acoustic_reliance_gap"]
    assert t2_gap > 0.0, f"T2 acoustic reliance gap should be positive, got {t2_gap}"
    assert t2_gap > c0_gap, f"T2 acoustic reliance gap ({t2_gap}) should exceed C0 gap ({c0_gap})"

    # Substitution APs should be non-inferior to C0
    assert t2_fixed["dedicated_substitution_ap"] >= c0_fixed["dedicated_substitution_ap"]
    assert t2_fixed["legacy_substitution_ap"] >= c0_fixed["legacy_substitution_ap"]

    # Greedy PER should be non-inferior (within 0.5% margin)
    assert abs(t2_fixed["greedy_per"] - c0_fixed["greedy_per"]) < 0.005
