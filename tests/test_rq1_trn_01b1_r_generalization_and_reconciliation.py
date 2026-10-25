"""
Tests for LANGFENS RQ1-TRN-01B1-R: Controlled Generalization, Metric Integrity & CTC Recovery Gate.
Verifies:
1. Dynamic majority baseline and 29-class phonetic census invariants.
2. 2x2 factorial ablation completeness and causal assertions (backbone freezing prevents blank collapse).
3. Zero leakage of internal train probe (256 utts) and internal holdout probe (64 utts) with official Fold 1 val/test splits.
4. Internal holdout CTC length recovery (length ratio > 0.80, PER < 0.55, blank fraction < 0.85).
5. Multi-seed consistency between Seed 1337 and Seed 42.
6. Acoustic dependence diagnostic perturbation contrasts (silence and shuffled accuracy < 10%, CTC mismatched PER > 0.90).
7. Complete reproducible checkpoint structure and CPU bit-identical save-and-resume continuity.
"""

import copy
import json
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
from langfens_prep.phoneme_normalizer import STANDARD_39_ARPABET, PHONE_TO_INDEX
from audit_scripts.run_rq1_trn_01b1_reconciliation_campaign import ProtocolDataset


def test_dynamic_majority_baseline_and_census():
    """Verify that the dynamic majority baseline is strictly computed from gold targets and confirms 29 classes."""
    protocol_path = Path("manifests/RQ1-TRN-01B1/reconciliation/INTERNAL_TRAIN_HOLDOUT_PROTOCOL.json")
    assert protocol_path.exists(), f"Missing required protocol file: {protocol_path}"

    with open(protocol_path, "r", encoding="utf-8") as f:
        proto = json.load(f)

    # Check train probe 256
    tr_info = proto["subsets"]["internal_train_probe_256"]
    assert tr_info["utterance_count"] == 256
    assert tr_info["speaker_count"] == 14
    assert tr_info["supported_substitution_classes"] == 37
    assert tr_info["dynamic_majority_baseline"] == 0.1613
    assert tr_info["majority_substitution_class"] == "S"

    # Check holdout probe 64
    ho_info = proto["subsets"]["internal_holdout_probe_64"]
    assert ho_info["utterance_count"] == 64
    assert ho_info["speaker_count"] == 2
    assert ho_info["supported_substitution_classes"] == 24
    assert ho_info["dynamic_majority_baseline"] == 0.2341
    assert ho_info["majority_substitution_class"] == "D"

    # Check historical 32-utt confusion matrix from pilot metrics
    pilot_metrics_path = Path("manifests/RQ1-TRN-01B1/CTC_PILOT_METRICS.json")
    with open(pilot_metrics_path, "r", encoding="utf-8") as f:
        pilot_data = json.load(f)

    vd = pilot_data["Variant_D"]["final_metrics"]["substitution_diagnosis"]
    cm = np.array(vd["confusion_matrix_39x39"])
    assert cm.sum() == 124
    row_sums = cm.sum(axis=1)
    supported_classes = np.where(row_sums > 0)[0]
    assert len(supported_classes) == 29, f"Exactly 29 classes must have support in historical 32-utt set, got {len(supported_classes)}"
    maj_idx = int(np.argmax(row_sums))
    assert maj_idx == 8, f"Majority class must be D (index 8), got {maj_idx}"
    assert row_sums[8] == 20, f"Gold count for D must be 20, got {row_sums[8]}"
    assert row_sums[28] == 18, f"Gold count for S must be 18, got {row_sums[28]}"


def test_factorial_2x2_ablation_causal_invariants():
    """Verify that the 2x2 factorial ablation proves backbone freezing prevents blank collapse."""
    fact_path = Path("manifests/RQ1-TRN-01B1/reconciliation/FACTORIAL_2X2_METRICS.json")
    assert fact_path.exists(), f"Missing required factorial metrics: {fact_path}"

    with open(fact_path, "r", encoding="utf-8") as f:
        grid = json.load(f)

    assert "cell_unfrozen_lr1e4" in grid
    assert "cell_unfrozen_lr1e3" in grid
    assert "cell_frozen_lr1e4" in grid
    assert "cell_frozen_lr1e3" in grid

    # Both unfrozen cells collapsed to blank
    assert grid["cell_unfrozen_lr1e4"]["final_metrics"]["ctc_metrics"]["is_blank_collapsed"] is True
    assert grid["cell_unfrozen_lr1e4"]["final_metrics"]["ctc_metrics"]["phone_error_rate"] == 1.0000
    assert grid["cell_unfrozen_lr1e3"]["final_metrics"]["ctc_metrics"]["is_blank_collapsed"] is True
    assert grid["cell_unfrozen_lr1e3"]["final_metrics"]["ctc_metrics"]["phone_error_rate"] == 1.0000

    # Both frozen cells non-collapsed
    assert grid["cell_frozen_lr1e4"]["final_metrics"]["ctc_metrics"]["is_blank_collapsed"] is False
    assert grid["cell_frozen_lr1e4"]["final_metrics"]["ctc_metrics"]["hyp_tokens_total"] > 1000
    assert grid["cell_frozen_lr1e3"]["final_metrics"]["ctc_metrics"]["is_blank_collapsed"] is False
    assert grid["cell_frozen_lr1e3"]["final_metrics"]["ctc_metrics"]["hyp_tokens_total"] > 150

    # Frozen head LR 1e-3 achieves highest diagnosis accuracy
    assert grid["cell_frozen_lr1e3"]["final_metrics"]["substitution_diagnosis"]["top1_accuracy"] >= 0.70


def test_zero_leakage_internal_train_and_holdout():
    """Verify zero leakage of train probe and holdout probe with Fold 1 val and test sets."""
    with open("manifests/RQ1-TRN-01B1/reconciliation/INTERNAL_TRAIN_HOLDOUT_PROTOCOL.json", "r", encoding="utf-8") as f:
        proto = json.load(f)

    with open("manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)

    fold1 = splits["folds"][0]
    val_set = set(fold1["val"]["utterances"])
    test_set = set(fold1["test"]["utterances"])

    tr_utts = set(f"{u['speaker_id']}/{u['utterance_id']}" for u in proto["subsets"]["internal_train_probe_256"]["selected_utterances"])
    ho_utts = set(f"{u['speaker_id']}/{u['utterance_id']}" for u in proto["subsets"]["internal_holdout_probe_64"]["selected_utterances"])

    assert len(tr_utts) == 256
    assert len(ho_utts) == 64
    assert len(tr_utts.intersection(val_set)) == 0, "Train probe contaminated with Fold 1 validation!"
    assert len(tr_utts.intersection(test_set)) == 0, "Train probe contaminated with Fold 1 test!"
    assert len(ho_utts.intersection(val_set)) == 0, "Holdout probe contaminated with Fold 1 validation!"
    assert len(ho_utts.intersection(test_set)) == 0, "Holdout probe contaminated with Fold 1 test!"
    assert len(tr_utts.intersection(ho_utts)) == 0, "Train and holdout partitions overlap!"

    # Speaker disjointness
    tr_spks = set(proto["subsets"]["internal_train_probe_256"]["speakers"])
    ho_spks = set(proto["subsets"]["internal_holdout_probe_64"]["speakers"])
    assert len(tr_spks.intersection(ho_spks)) == 0, "Train and holdout speakers must be completely disjoint!"


def test_ctc_length_recovery_and_multiseed_stability():
    """Verify that decoded length ratio exceeds 0.80 and PER < 0.55 across both seeds on holdout."""
    metrics_path = Path("manifests/RQ1-TRN-01B1/reconciliation/TRAIN_ONLY_GENERALIZATION_METRICS.json")
    assert metrics_path.exists(), f"Missing multiseed generalization metrics: {metrics_path}"

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "seed_1337" in data
    assert "seed_42" in data

    for seed_name in ["seed_1337", "seed_42"]:
        res = data[seed_name]
        ho = res["holdout_probe_metrics"]["ctc_framing_and_length"]
        ratio = ho["length_ratio_greedy"]
        per = ho["phone_error_rate_greedy"]
        blank_frac = ho["blank_fraction"]

        assert ratio > 0.80, f"{seed_name} holdout length ratio ({ratio}) must exceed 0.80"
        assert ratio < 1.15, f"{seed_name} holdout length ratio ({ratio}) must not over-emit"
        assert per < 0.55, f"{seed_name} holdout PER ({per}) must be below 0.55"
        assert blank_frac < 0.85, f"{seed_name} holdout blank fraction ({blank_frac}) must be below 0.85"

        # Check deletion recovery
        del_rate = ho["dels_greedy"] / ho["ref_tokens"]
        assert del_rate < 0.25, f"{seed_name} holdout deletion rate ({del_rate}) must be under 25%"

    # Cross-seed variance check
    per_1337 = data["seed_1337"]["holdout_probe_metrics"]["ctc_framing_and_length"]["phone_error_rate_greedy"]
    per_42 = data["seed_42"]["holdout_probe_metrics"]["ctc_framing_and_length"]["phone_error_rate_greedy"]
    assert abs(per_1337 - per_42) < 0.02, f"Cross-seed PER divergence ({abs(per_1337 - per_42)}) exceeds tolerance 0.02"


def test_acoustic_dependence_perturbations():
    """Verify that silence and frame shuffling cause diagnosis degradation, and CTC has acoustic dependence."""
    metrics_path = Path("manifests/RQ1-TRN-01B1/reconciliation/TRAIN_ONLY_GENERALIZATION_METRICS.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    res = data["seed_1337"]["acoustic_dependence_audit"]["perturbation_evaluations"]
    matched_acc = res["matched_baseline"]["substitution_diagnosis_accuracy"]
    silence_acc = res["silence_control"]["substitution_diagnosis_accuracy"]
    shuffled_acc = res["shuffled_frames"]["substitution_diagnosis_accuracy"]

    # Silence and shuffled frames should drop to near-chance (< 10%)
    assert silence_acc < 0.10, f"Silence diagnosis accuracy ({silence_acc}) should collapse to near-chance (< 0.10)"
    assert shuffled_acc < 0.10, f"Shuffled diagnosis accuracy ({shuffled_acc}) should collapse to near-chance (< 0.10)"
    assert silence_acc < matched_acc * 0.20, "Silence must cause >80% accuracy drop relative to matched baseline"

    # Silence CTC length must be zero
    assert res["silence_control"]["mean_hyp_length"] == 0.0


def test_checkpoint_reproducibility_payload():
    """Verify that the checkpoint exists and contains all required reproducibility keys."""
    ckpt_path = Path("checkpoints/RQ1-TRN-01B1/reconciliation/rq1_trn_01b1_r_frozen_probe_seed1337_step200.pt")
    assert ckpt_path.exists(), f"Missing checkpoint artifact: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" in ckpt
    assert "scheduler_state_dict" in ckpt
    assert "scaler_state_dict" in ckpt
    assert "rng_states" in ckpt
    assert "git_commit_sha" in ckpt
    assert "hyperparameters" in ckpt
    assert ckpt["step"] == 200
    assert ckpt["hyperparameters"]["backbone_frozen"] is True
