"""
LANGFENS — RQ1-TRN-03A Steps 12-13: Four-Fold Closure & Validation Synthesis Regression Tests
Certifies Fold 4 training closure, cross-fold parity, speaker isolation, and 4-fold synthesis.
"""

import csv
import json
import math
from pathlib import Path
import pytest
import torch
import yaml

from audit_scripts.evaluate_fold1_ctc_binary import compute_sha256, average_precision_binary
from audit_scripts.evaluate_cross_fold_validation import (
    FOLD_DEFINITIONS,
    compute_student_t_ci,
    CRITICAL_T_3_DF,
)


def test_fold4_training_closure_and_sha256_digests():
    """Verifies that Fold 4 training artifacts match verified cryptographic digests."""
    fold4_dir = Path("checkpoints/fold4/baseline_r_fold4")
    if not fold4_dir.is_dir():
        pytest.skip("Fold 4 checkpoint directory not found locally")

    expected_shas = {
        "fixed_budget_checkpoint.pt": "c502271feba097781a6b0bdfe413085271d3a03647353f272dccd29bbb1499f6",
        "best.pt": "3daa8ca0db8a632050e3f965c705df2d271d07a450cb7110a9780537d623274d",
        "last.pt": "894a12487c6d2c43bcbf154fa3617e66bca739f23fdee855b13b720606f9c47f",
        "history.json": "4b75d13eee70c7e72b5938e0d8a1148a9bc41f4149f82f9ecc9e997b3ad7ee79",
        "training_endpoint_report.json": "34e06481168c2226ec00862721ed8c36d0e0c0855104b7882cd5ffbd953a5dcb",
    }

    for fname, exp_sha in expected_shas.items():
        fpath = fold4_dir / fname
        assert fpath.is_file(), f"Missing Fold 4 artifact: {fpath}"
        actual_sha = compute_sha256(fpath)
        assert actual_sha == exp_sha, f"SHA-256 mismatch for {fname}: got {actual_sha}, expected {exp_sha}"


def test_fold4_endpoint_classification_and_history():
    """Verifies Fold 4 endpoint classification and training history invariants."""
    fold4_dir = Path("checkpoints/fold4/baseline_r_fold4")
    if not fold4_dir.is_dir():
        pytest.skip("Fold 4 checkpoint directory not found locally")

    report_path = fold4_dir / "training_endpoint_report.json"
    with open(report_path) as f:
        report = json.load(f)

    assert report["status"] == "COMPLETED_3600_UPDATES"
    assert report["fold"] == 4
    assert report["successful_updates"] == 3600
    assert report["epoch"] == 5  # 0-indexed Epoch 6

    hist_path = fold4_dir / "history.json"
    with open(hist_path) as f:
        hist = json.load(f)

    assert len(hist) == 8, f"Expected 8 completed epochs in Fold 4, got {len(hist)}"
    assert all(r["skipped_amp_attempts_total"] == 0 for r in hist), "AMP updates were skipped in Fold 4!"
    assert hist[5]["successful_updates_total"] == 3600
    assert pytest.approx(hist[5]["validation_ctc_per"], abs=1e-4) == 0.236263


def test_four_fold_config_and_hyperparameter_parity():
    """Verifies 100% hyperparameter parity across all four fold configuration files."""
    configs = {}
    for fold in [1, 2, 3, 4]:
        cpath = f"configs/rq1_trn_02a_fold1.yaml" if fold == 1 else f"configs/rq1_trn_03a_fold{fold}.yaml"
        with open(cpath) as f:
            configs[fold] = yaml.safe_load(f)

    shared_invariants = {
        "seed": 1337,
        "pretrained_model": "facebook/wav2vec2-base-960h",
        "pretrained_revision": "22aad52d435eb6dbaf354bdad9b0da84ce7d6156",
        "loss_weights": {"ctc": 1.0, "binary": 1.0},
        "backbone_frozen": True,
        "batch_size": 2,
        "gradient_accumulation_steps": 2,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "scheduler": "cosine",
        "warmup_updates": 100,
        "max_epochs": 25,
        "gradient_clip_norm": 1.0,
        "amp": "fp16",
        "grad_scaler_initial_scale": 4096,
        "early_stopping_patience": 5,
    }

    for fold in [1, 2, 3, 4]:
        cfg = configs[fold]
        for key, val in shared_invariants.items():
            assert cfg[key] == val, f"Fold {fold} configuration mismatch on {key}: got {cfg[key]}, expected {val}"


def test_four_fold_speaker_isolation_and_no_test_leakage():
    """Verifies that all 4 folds enforce disjoint speaker partitions and test sets remain sealed."""
    manifest_path = Path("manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json")
    with open(manifest_path) as f:
        splits = json.load(f)

    val_speakers_all = []
    for f in splits["folds"]:
        fold_idx = f["fold_index"]
        train_spk = set(f["train"]["speakers"])
        val_spk = set(f["val"]["speakers"])
        test_spk = set(f["test"]["speakers"])

        assert len(train_spk) == 16
        assert len(val_spk) == 2
        assert len(test_spk) == 6

        assert train_spk.isdisjoint(val_spk), f"Fold {fold_idx} train/val leak: {train_spk & val_spk}"
        assert train_spk.isdisjoint(test_spk), f"Fold {fold_idx} train/test leak: {train_spk & test_spk}"
        assert val_spk.isdisjoint(test_spk), f"Fold {fold_idx} val/test leak: {val_spk & test_spk}"

        val_speakers_all.extend(list(val_spk))

    assert len(val_speakers_all) == 8
    assert len(set(val_speakers_all)) == 8, f"Duplicate validation speakers across folds: {val_speakers_all}"
    assert set(val_speakers_all) == {"NJS", "RRBI", "TNI", "YBAA", "TXHC", "ZHAA", "HKK", "NCC"}


def test_four_fold_synthesis_report_integrity():
    """Verifies that FOUR_FOLD_VALIDATION_REPORT.json is valid and contains exact aggregate metrics."""
    report_path = Path("manifests/RQ1-TRN-03A/evaluation/FOUR_FOLD_VALIDATION_REPORT.json")
    if not report_path.is_file():
        pytest.skip("FOUR_FOLD_VALIDATION_REPORT.json not found")

    with open(report_path) as f:
        rep = json.load(f)

    meta = rep["metadata"]
    assert meta["total_out_of_fold_validation_utterances"] == 1200
    assert meta["total_out_of_fold_validation_tokens"] == 39576
    assert meta["total_out_of_fold_validation_errors"] == 4213
    assert meta["validation_speakers_count"] == 8

    fb_macro = rep["primary_fixed_budget_3600_updates"]["macro_statistics"]
    assert pytest.approx(fb_macro["matched_binary_ap"]["mean"], abs=1e-5) == 0.279214
    assert pytest.approx(fb_macro["acoustic_reliance_gap"]["mean"], abs=1e-5) == 0.013060
    assert pytest.approx(fb_macro["ctc_per"]["mean"], abs=1e-4) == 0.238971

    fb_micro = rep["primary_fixed_budget_3600_updates"]["pooled_micro_statistics"]
    assert pytest.approx(fb_micro["matched_binary_ap"], abs=1e-5) == 0.272582
    assert pytest.approx(fb_micro["acoustic_reliance_gap"], abs=1e-5) == 0.010619
    assert pytest.approx(fb_micro["ctc_per"], abs=1e-4) == 0.238981


def test_four_fold_summary_csv_and_eight_speakers_csv_exist():
    """Verifies that summary and 8-speaker CSV tables exist and have correct row counts."""
    summary_path = Path("manifests/RQ1-TRN-03A/evaluation/four_fold_metrics_summary.csv")
    assert summary_path.is_file(), f"Missing {summary_path}"

    with open(summary_path) as f:
        reader = csv.DictReader(f)
        summary_rows = list(reader)

    # 4 folds + macro + micro for fixed budget (6), plus 4 folds + macro + micro for best (6) = 12 rows
    assert len(summary_rows) == 12, f"Expected 12 summary rows, got {len(summary_rows)}"

    spk_path = Path("manifests/RQ1-TRN-03A/evaluation/eight_speakers_validation_breakdown.csv")
    assert spk_path.is_file(), f"Missing {spk_path}"

    with open(spk_path) as f:
        reader = csv.DictReader(f)
        spk_rows = list(reader)

    assert len(spk_rows) == 8, f"Expected 8 speaker rows, got {len(spk_rows)}"
    spk_names = [r["Speaker"] for r in spk_rows]
    assert set(spk_names) == {"NJS", "RRBI", "TNI", "YBAA", "TXHC", "ZHAA", "HKK", "NCC"}
