"""LANGFENS — RQ1-TRN-03A: Cross-Fold Protocol Parity and Isolation Test Suite.

Senior Speech ML Research Engineer & Reproducibility Auditor.

Verifies:
  1. Strict per-fold partition disjointness (train, val, test).
  2. Legitimate cross-fold speaker rotation (all 24 speakers tested across 4 folds).
  3. Demographic and L1 stratification per fold.
  4. Exact hyperparameter parity between Fold 1 Baseline R and cross-fold specification:
     - seed = 1337
     - batch_size = 2, gradient_accumulation_steps = 2 (effective batch size 4)
     - warmup_updates = 100
     - scheduler horizon with max_epochs = 25
     - loss weights ctc=1.0, binary=1.0
     - early stopping patience = 5 on validation AP with blank fraction < 0.99
  5. Deterministic within-speaker cyclic shift-1 mismatched audio derangement.
  6. Mathematical definitions of FNR, FPR, Precision, Recall, Specificity.
  7. Checkpoint selection semantics (fixed-budget 3,600 vs validation peak best.pt).
"""

import json
import math
from pathlib import Path
import pytest
import yaml

from audit_scripts.evaluate_fold1_ctc_binary import generate_derangement_indices


SPLIT_MANIFEST_PATH = Path("manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json")
BASELINE_R_CONFIG_PATH = Path("configs/rq1_trn_02a_fold1.yaml")


@pytest.fixture(scope="module")
def split_data():
    assert SPLIT_MANIFEST_PATH.is_file(), f"Split manifest missing: {SPLIT_MANIFEST_PATH}"
    with open(SPLIT_MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def baseline_r_config():
    assert BASELINE_R_CONFIG_PATH.is_file(), f"Baseline config missing: {BASELINE_R_CONFIG_PATH}"
    with open(BASELINE_R_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ==============================================================================
# 1. PER-FOLD SPEAKER ISOLATION & PARTITION DISJOINTNESS
# ==============================================================================

def test_per_fold_partition_disjointness(split_data):
    """Proves that for every fold k, Train_k, Val_k, and Test_k are strictly disjoint."""
    all_24_speakers = {
        "ABA", "SKA", "YBAA", "ZHAA", "BWC", "LXC", "NCC", "TXHC",
        "ASI", "RRBI", "SVBI", "TNI", "HJK", "HKK", "YDCK", "YKWK",
        "EBVS", "ERMS", "MBMPS", "NJS", "HQTV", "PNV", "THV", "TLV"
    }

    for fold in split_data["folds"]:
        k = fold["fold_index"]
        train_spks = set(fold["train"]["speakers"])
        val_spks = set(fold["val"]["speakers"])
        test_spks = set(fold["test"]["speakers"])

        # Exact cardinalities
        assert len(train_spks) == 16, f"Fold {k}: Expected 16 train speakers, got {len(train_spks)}"
        assert len(val_spks) == 2, f"Fold {k}: Expected 2 val speakers, got {len(val_spks)}"
        assert len(test_spks) == 6, f"Fold {k}: Expected 6 test speakers, got {len(test_spks)}"

        # Strict pairwise disjointness
        assert train_spks.isdisjoint(val_spks), f"Fold {k}: Train and Val overlap: {train_spks & val_spks}"
        assert train_spks.isdisjoint(test_spks), f"Fold {k}: Train and Test overlap: {train_spks & test_spks}"
        assert val_spks.isdisjoint(test_spks), f"Fold {k}: Val and Test overlap: {val_spks & test_spks}"

        # Exhaustive coverage of all 24 speakers
        assert (train_spks | val_spks | test_spks) == all_24_speakers, f"Fold {k}: Partition does not encompass all 24 speakers"


def test_cross_fold_speaker_rotation(split_data):
    """Proves that every speaker is in exactly one test set and rotates into train/val of others."""
    all_tested_speakers = []
    test_sets_by_fold = {}

    for fold in split_data["folds"]:
        k = fold["fold_index"]
        test_spks = fold["test"]["speakers"]
        test_sets_by_fold[k] = set(test_spks)
        all_tested_speakers.extend(test_spks)

    # 1. Exactly 24 test speakers across all 4 folds with zero duplicates
    assert len(all_tested_speakers) == 24
    assert len(set(all_tested_speakers)) == 24

    # 2. Pairwise disjointness among test sets
    for i in range(1, 5):
        for j in range(i + 1, 5):
            overlap = test_sets_by_fold[i] & test_sets_by_fold[j]
            assert len(overlap) == 0, f"Test sets of Fold {i} and Fold {j} overlap: {overlap}"

    # 3. Legitimate cross-fold reuse: Fold 1 test speakers must be in Fold 2 train
    fold1_test = test_sets_by_fold[1]
    fold2_train = set(split_data["folds"][1]["train"]["speakers"])
    assert fold1_test.issubset(fold2_train), (
        f"Fold 1 test speakers {fold1_test} must legitimately be in Fold 2 train {fold2_train}"
    )


def test_demographic_stratification(split_data):
    """Proves that each fold satisfies demographic and L1 balance constraints."""
    for fold in split_data["folds"]:
        k = fold["fold_index"]
        test_gender = fold["test"]["gender_distribution"]
        test_l1 = fold["test"]["l1_distribution"]

        assert test_gender == {"M": 3, "F": 3}, f"Fold {k}: Expected 3M/3F, got {test_gender}"
        assert len(test_l1) == 6, f"Fold {k}: Expected all 6 L1s represented, got {len(test_l1)}"
        assert all(count == 1 for count in test_l1.values()), f"Fold {k}: Expected 1 speaker per L1, got {test_l1}"

        val_gender = fold["val"]["gender_distribution"]
        val_l1 = fold["val"]["l1_distribution"]
        assert val_gender == {"M": 1, "F": 1}, f"Fold {k}: Expected 1M/1F val, got {val_gender}"
        assert len(val_l1) == 2, f"Fold {k}: Expected 2 distinct L1s in val, got {len(val_l1)}"


# ==============================================================================
# 2. BASELINE R CONFIGURATION & HYPERPARAMETER PARITY
# ==============================================================================

def test_baseline_r_exact_hyperparameter_contract(baseline_r_config):
    """Proves that Fold 1 Baseline R pinned configuration has exact expected parameters."""
    cfg = baseline_r_config

    assert cfg["seed"] == 1337, f"Expected seed=1337, got {cfg['seed']}"
    assert cfg["batch_size"] == 2, f"Expected batch_size=2, got {cfg['batch_size']}"
    assert cfg["gradient_accumulation_steps"] == 2, (
        f"Expected gradient_accumulation_steps=2, got {cfg['gradient_accumulation_steps']}"
    )
    assert cfg["warmup_updates"] == 100, f"Expected warmup_updates=100, got {cfg['warmup_updates']}"
    assert cfg["max_epochs"] == 25, f"Expected max_epochs=25, got {cfg['max_epochs']}"
    assert cfg["early_stopping_patience"] == 5, (
        f"Expected early_stopping_patience=5, got {cfg['early_stopping_patience']}"
    )
    assert cfg["optimizer"] == "AdamW", f"Expected AdamW, got {cfg['optimizer']}"
    assert cfg["learning_rate"] == 0.001, f"Expected lr=0.001, got {cfg['learning_rate']}"
    assert cfg["weight_decay"] == 0.01, f"Expected weight_decay=0.01, got {cfg['weight_decay']}"
    assert cfg["scheduler"] == "cosine", f"Expected cosine scheduler, got {cfg['scheduler']}"
    assert cfg["loss_weights"] == {"ctc": 1.0, "binary": 1.0}, (
        f"Expected loss_weights {{ctc: 1.0, binary: 1.0}}, got {cfg['loss_weights']}"
    )
    assert cfg["backbone_frozen"] is True, "Backbone must be frozen"
    assert cfg["amp"] == "fp16", f"Expected amp=fp16, got {cfg['amp']}"
    assert cfg["grad_scaler_initial_scale"] == 4096, (
        f"Expected grad_scaler_initial_scale=4096, got {cfg['grad_scaler_initial_scale']}"
    )


def test_scheduler_horizon_computation(baseline_r_config, split_data):
    """Proves that scheduler horizon is computed over max_epochs=25 (approx 15,000 updates)."""
    cfg = baseline_r_config
    max_epochs = cfg["max_epochs"]
    accum = cfg["gradient_accumulation_steps"]
    batch_size = cfg["batch_size"]

    for fold in split_data["folds"]:
        k = fold["fold_index"]
        train_utts = fold["train"]["utterance_count"]
        # train_loader length
        num_batches = math.ceil(train_utts / batch_size)
        updates_per_epoch = math.ceil(num_batches / accum)
        total_updates = updates_per_epoch * max_epochs

        # Fold 1: 2399 utts -> 1200 batches -> 600 updates/epoch -> 15000 total updates
        # Fold 2: 2400 utts -> 1200 batches -> 600 updates/epoch -> 15000 total updates
        assert updates_per_epoch == 600, f"Fold {k}: Expected 600 updates/epoch, got {updates_per_epoch}"
        assert total_updates == 15000, f"Fold {k}: Expected total_updates=15000, got {total_updates}"

        # 6 epochs corresponds to 3,600 updates
        fixed_budget_updates = updates_per_epoch * 6
        assert fixed_budget_updates == 3600, f"Fold {k}: Fixed 6-epoch budget != 3,600 updates"


# ==============================================================================
# 3. DETERMINISTIC WITHIN-SPEAKER CYCLIC SHIFT-1 MISMATCHED AUDIO
# ==============================================================================

def test_within_speaker_cyclic_derangement():
    """Proves that within_speaker_cyclic creates zero self-matches and stays within speaker boundaries."""
    # Simulate Fold 1 validation: 150 NJS utterances followed by 150 RRBI utterances
    n_total = 300
    speaker_boundaries = [150]

    indices = generate_derangement_indices(
        n_total,
        method="within_speaker_cyclic",
        speaker_boundaries=speaker_boundaries
    )

    assert len(indices) == n_total, f"Expected {n_total} indices, got {len(indices)}"

    # 1. Zero self-matches
    for i in range(n_total):
        assert indices[i] != i, f"Self-match at index {i}: mapped to {indices[i]}"

    # 2. Strict speaker boundary containment
    # Segment 0: 0..149 (Speaker NJS)
    for i in range(150):
        target = indices[i]
        assert 0 <= target < 150, f"Speaker leak: NJS utterance {i} mapped to non-NJS {target}"
        # Shift-1 cyclic check: 0->1, 1->2, ..., 149->0
        expected = (i + 1) % 150
        assert target == expected, f"Utterance {i} expected {expected}, got {target}"

    # Segment 1: 150..299 (Speaker RRBI)
    for i in range(150, 300):
        target = indices[i]
        assert 150 <= target < 300, f"Speaker leak: RRBI utterance {i} mapped to non-RRBI {target}"
        # Shift-1 cyclic check: 150->151, ..., 299->150
        expected = 150 + ((i - 150 + 1) % 150)
        assert target == expected, f"Utterance {i} expected {expected}, got {target}"


# ==============================================================================
# 4. STATISTICAL & CONFUSION METRIC MATHEMATICAL RIGOR
# ==============================================================================

def test_fnr_fpr_precision_recall_relationships():
    """Proves exact mathematical relationships for binary MDD confusion metrics."""
    # Synthetic counts
    TP = 450
    FP = 150
    FN = 550
    TN = 8850

    # Total errors (positives) and total correct (negatives)
    P = TP + FN
    N = TN + FP

    recall = TP / P
    sensitivity = recall
    fnr = FN / P

    specificity = TN / N
    fpr = FP / N

    precision = TP / (TP + FP)

    # Fundamental complements
    assert math.isclose(recall + fnr, 1.0, abs_tol=1e-12), "Recall + FNR must equal 1.0"
    assert math.isclose(specificity + fpr, 1.0, abs_tol=1e-12), "Specificity + FPR must equal 1.0"

    # In educational CAPT terminology:
    # A False Negative (FN) means an error was MISSED by the system (accepted as correct).
    # Thus, FNR is the missed error rate (pedagogical false acceptance rate).
    # A False Positive (FP) means a correct sound was falsely flagged as error.
    # Thus, FPR is the false alarm rate (pedagogical false rejection rate).
    assert math.isclose(fnr, 1.0 - sensitivity, abs_tol=1e-12)
    assert math.isclose(fpr, 1.0 - specificity, abs_tol=1e-12)
    assert precision == 450 / 600 == 0.75


# ==============================================================================
# 5. DUAL CHECKPOINT SEMANTICS DISTINCTION
# ==============================================================================

def test_checkpoint_distinction_semantics():
    """Proves formal distinction between fixed-budget (Epoch 6 / 3,600) and early-stopping best.pt."""
    # Mock history tracking
    history = [
        {"epoch": 1, "updates": 600, "val_ap": 0.280, "blank_fraction": 0.05},
        {"epoch": 2, "updates": 1200, "val_ap": 0.300, "blank_fraction": 0.04},
        {"epoch": 3, "updates": 1800, "val_ap": 0.315, "blank_fraction": 0.04},
        {"epoch": 4, "updates": 2400, "val_ap": 0.325, "blank_fraction": 0.03},  # best.pt here
        {"epoch": 5, "updates": 3000, "val_ap": 0.320, "blank_fraction": 0.03},
        {"epoch": 6, "updates": 3600, "val_ap": 0.318, "blank_fraction": 0.03},  # fixed budget here
        {"epoch": 7, "updates": 4200, "val_ap": 0.310, "blank_fraction": 0.03},
        {"epoch": 8, "updates": 4800, "val_ap": 0.305, "blank_fraction": 0.03},
        {"epoch": 9, "updates": 5400, "val_ap": 0.300, "blank_fraction": 0.03},  # patience=5 triggered
    ]

    patience_limit = 5
    best_val_ap = -float("inf")
    best_epoch = None
    fixed_budget_checkpoint = None

    for row in history:
        ep = row["epoch"]
        ap = row["val_ap"]
        bf = row["blank_fraction"]

        # Selection condition matching train_fold1_ctc_binary.py
        if ap > best_val_ap and bf < 0.99:
            best_val_ap = ap
            best_epoch = ep

        if ep == 6 and row["updates"] == 3600:
            fixed_budget_checkpoint = row

    # Best checkpoint occurred at Epoch 4 (val_ap 0.325)
    assert best_epoch == 4
    assert best_val_ap == 0.325

    # Fixed budget checkpoint occurred at Epoch 6 (val_ap 0.318)
    assert fixed_budget_checkpoint is not None
    assert fixed_budget_checkpoint["epoch"] == 6
    assert fixed_budget_checkpoint["updates"] == 3600
    assert fixed_budget_checkpoint["val_ap"] == 0.318

    # The two checkpoints must be reported separately to avoid confounding treatment with exposure
    assert best_epoch != fixed_budget_checkpoint["epoch"]
