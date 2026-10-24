"""
Tests for LANGFENS RQ1-TRN-01B1: CTC Stabilization & Training Readiness Pilot.
Verifies:
1. Checkpoint continuity determinism on CPU (bit-identical, diff = 0.0).
2. Variant D frozen backbone prevents blank collapse and beats PER baseline.
3. Multi-task substitution diagnosis beats uniform-chance and majority baselines on N=124 eligible substitutions.
4. Pilot metrics JSON completeness and contract validity.
5. Zero leakage of train-only subset with Fold 1 val/test splits.
"""

import copy
import json
import os
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from langfens_prep.model import (
    LangfensWav2Vec2MDD,
    DEFAULT_PRETRAINED_CHECKPOINT,
    DEFAULT_PRETRAINED_REVISION
)
from langfens_prep.collate import LangfensMDDCollate
from audit_scripts.run_training_sanity import TrainSubsetDataset


def test_ctc_pilot_metrics_structure_and_constraints():
    """Verify that CTC_PILOT_METRICS.json is present, valid, and contains all 4 variants."""
    metrics_path = Path("manifests/RQ1-TRN-01B1/CTC_PILOT_METRICS.json")
    assert metrics_path.exists(), f"Missing required artifact: {metrics_path}"

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    expected_variants = ["Variant_A", "Variant_B", "Variant_C", "Variant_D"]
    for v in expected_variants:
        assert v in data, f"Missing variant {v} in pilot metrics"

        vm = data[v]
        assert "final_metrics" in vm
        fm = vm["final_metrics"]
        assert "ctc_metrics" in fm
        assert "substitution_diagnosis" in fm
        assert "binary_mdd" in fm

        # Verify exact substitution denominator N=124
        assert fm["substitution_diagnosis"]["eligible_denominator"] == 124


def test_checkpoint_continuity_cpu_determinism():
    """Verify that under CPU deterministic execution, resume continuation is bit-identical (diff = 0.0)."""
    orig_det = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        random.seed(1337)
        np.random.seed(1337)
        torch.manual_seed(1337)

        ds = TrainSubsetDataset()
        collate_fn = LangfensMDDCollate(pad_token_id=-100, ctc_blank_id=0)
        loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
        batches = list(loader)

        # Short test: run uninterrupted 6 steps, save at 4
        model_a = LangfensWav2Vec2MDD.from_pretrained(
            DEFAULT_PRETRAINED_CHECKPOINT,
            revision=DEFAULT_PRETRAINED_REVISION,
            alignment_mode="cross_attention",
            ctc_zero_infinity_policy="raise"
        )
        model_a.freeze_feature_encoder()
        model_a.train()

        opt_a = torch.optim.AdamW(filter(lambda p: p.requires_grad, model_a.parameters()), lr=1e-4)

        ckpt_4 = None
        losses_a = []

        for s in range(1, 7):
            batch = batches[s % len(batches)]
            out = model_a(
                **batch["net_input"],
                target_binary=batch["target_binary"],
                target_ctc=batch["target_ctc"],
                target_lengths=batch["target_lengths"],
                target_diagnosis=batch["target_diagnosis"]
            )
            out.loss.backward()
            opt_a.step()
            opt_a.zero_grad()
            losses_a.append(out.loss.item())

            if s == 4:
                ckpt_4 = {
                    "model": copy.deepcopy(model_a.state_dict()),
                    "opt": copy.deepcopy(opt_a.state_dict()),
                    "torch_rng": torch.get_rng_state()
                }

        # Model B: resume from 4
        model_b = LangfensWav2Vec2MDD.from_pretrained(
            DEFAULT_PRETRAINED_CHECKPOINT,
            revision=DEFAULT_PRETRAINED_REVISION,
            alignment_mode="cross_attention",
            ctc_zero_infinity_policy="raise"
        )
        model_b.freeze_feature_encoder()
        model_b.load_state_dict(ckpt_4["model"])
        model_b.train()

        opt_b = torch.optim.AdamW(filter(lambda p: p.requires_grad, model_b.parameters()), lr=1e-4)
        opt_b.load_state_dict(ckpt_4["opt"])
        torch.set_rng_state(ckpt_4["torch_rng"])

        losses_b = []
        for s in range(5, 7):
            batch = batches[s % len(batches)]
            out = model_b(
                **batch["net_input"],
                target_binary=batch["target_binary"],
                target_ctc=batch["target_ctc"],
                target_lengths=batch["target_lengths"],
                target_diagnosis=batch["target_diagnosis"]
            )
            out.loss.backward()
            opt_b.step()
            opt_b.zero_grad()
            losses_b.append(out.loss.item())

        # Assert bit-identical loss values at step 5 and step 6
        assert losses_a[4] == losses_b[0], f"CPU Step 5 mismatch: A={losses_a[4]}, B={losses_b[0]}"
        assert losses_a[5] == losses_b[1], f"CPU Step 6 mismatch: A={losses_a[5]}, B={losses_b[1]}"
    finally:
        torch.use_deterministic_algorithms(orig_det)


def test_variant_d_frozen_backbone_prevents_blank_collapse():
    """Verify that Variant D achieves non-collapsed CTC emissions and beats the baseline."""
    metrics_path = Path("manifests/RQ1-TRN-01B1/CTC_PILOT_METRICS.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vd = data["Variant_D"]["final_metrics"]["ctc_metrics"]
    va = data["Variant_A"]["final_metrics"]["ctc_metrics"]

    # Variant A collapsed
    assert va["is_blank_collapsed"] is True
    assert va["blank_fraction"] == 1.0000
    assert va["hyp_tokens_total"] == 0
    assert va["phone_error_rate"] == 1.0000

    # Variant D non-collapsed
    assert vd["is_blank_collapsed"] is False
    assert vd["blank_fraction"] < 0.99
    assert vd["hyp_tokens_total"] > 100
    assert vd["phone_error_rate"] < 1.0000
    assert vd["non_empty_utterance_fraction"] > 0.90


def test_multitask_diagnosis_majority_and_chance_baselines():
    """Verify that Variant D substitution diagnosis exceeds both chance and dynamic majority baselines on N=124."""
    metrics_path = Path("manifests/RQ1-TRN-01B1/CTC_PILOT_METRICS.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vd_diag = data["Variant_D"]["final_metrics"]["substitution_diagnosis"]
    acc = vd_diag["top1_accuracy"]
    chance_baseline = vd_diag["uniform_chance_baseline"]    # 2.56%

    # Reconciled dynamic majority baseline from confusion matrix
    cm = np.array(vd_diag["confusion_matrix_39x39"])
    assert cm.sum() == 124, f"Confusion matrix total {cm.sum()} must equal eligible denominator 124"
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)

    # Assert exactly 29 classes have gold support
    supported_classes = np.where(row_sums > 0)[0]
    assert len(supported_classes) == 29, f"Supported classes count {len(supported_classes)} must be 29"

    # Assert majority class is D (index 8) with count 20, and S (index 28) has count 18
    maj_idx = int(np.argmax(row_sums))
    assert maj_idx == 8, f"Majority class index must be 8 (D), got {maj_idx}"
    assert row_sums[8] == 20, f"Gold count for D must be 20, got {row_sums[8]}"
    assert row_sums[28] == 18, f"Gold count for S must be 18, got {row_sums[28]}"

    # True dynamic majority baseline is 20 / 124 = 0.16129
    dynamic_majority = float(row_sums[maj_idx] / cm.sum())
    assert abs(dynamic_majority - 0.16129) < 1e-4

    assert vd_diag["eligible_denominator"] == 124
    assert acc > dynamic_majority, f"Variant D accuracy ({acc}) must exceed dynamic majority baseline ({dynamic_majority})"
    assert acc > chance_baseline, f"Variant D accuracy ({acc}) must exceed chance baseline ({chance_baseline})"
    assert vd_diag["classes_with_correct_hits"] >= 10, "Should have hits across multiple phoneme classes"


def test_variant_a_majority_collapse_verification():
    """Verify that Variant A collapsed completely to predicting the majority class D (20/124)."""
    metrics_path = Path("manifests/RQ1-TRN-01B1/CTC_PILOT_METRICS.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    va_diag = data["Variant_A"]["final_metrics"]["substitution_diagnosis"]
    cm_a = np.array(va_diag["confusion_matrix_39x39"])
    col_sums = cm_a.sum(axis=0)

    # Column 8 (D) should have all 124 predictions
    assert col_sums[8] == 124, f"Variant A must predict class D for all 124 tokens, got {col_sums[8]}"
    # Top1 accuracy is exactly 20 / 124 = 0.1613
    assert va_diag["top1_accuracy"] == 0.1613
    assert va_diag["correct_predictions"] == 20


def test_train_subset_zero_leakage_assertion():
    """Verify that the 32 train utterances have zero overlap with Fold 1 val or test sets."""
    with open("manifests/RQ1-TRN-01B0/TRAIN_ONLY_SUBSET_MANIFEST.json", "r", encoding="utf-8") as f:
        train_manifest = json.load(f)

    with open("manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)

    fold1 = splits["folds"][0]
    val_set = set(fold1["val"]["utterances"])
    test_set = set(fold1["test"]["utterances"])
    train_utts = set(f"{u['speaker_id']}/{u['utterance_id']}" for u in train_manifest["selected_utterances"])

    assert len(train_utts) == 32
    assert len(train_utts.intersection(val_set)) == 0, "Data leakage into Fold 1 validation!"
    assert len(train_utts.intersection(test_set)) == 0, "Data leakage into Fold 1 test!"
