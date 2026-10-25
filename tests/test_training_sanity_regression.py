import json
import os
from pathlib import Path
import pytest
import torch

from langfens_prep.model import DEFAULT_PRETRAINED_CHECKPOINT, DEFAULT_PRETRAINED_REVISION


def test_train_subset_manifest_integrity():
    manifest_path = Path("manifests/RQ1-TRN-01B0/TRAIN_ONLY_SUBSET_MANIFEST.json")
    assert manifest_path.exists(), "TRAIN_ONLY_SUBSET_MANIFEST.json does not exist"

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Total utterances and speakers
    selected = data["selected_utterances"]
    assert len(selected) == 32, f"Expected 32 utterances, got {len(selected)}"
    assert data["metadata"]["total_selected_utterances"] == 32

    # Check 16 Fold 1 train speakers
    with open("manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)
    fold1 = next(fold for fold in splits["folds"] if fold["fold_index"] == 1)
    train_speakers = set(fold1["train"]["speakers"])
    val_speakers = set(fold1["val"]["speakers"])
    test_speakers = set(fold1["test"]["speakers"])
    train_utts = set(fold1["train"]["utterances"])
    val_utts = set(fold1["val"]["utterances"])
    test_utts = set(fold1["test"]["utterances"])

    selected_speakers = set(x["speaker_id"] for x in selected)
    assert selected_speakers == train_speakers, "Selected speakers do not match Fold 1 train speakers exactly"
    assert selected_speakers.isdisjoint(val_speakers), "Speaker leakage into val!"
    assert selected_speakers.isdisjoint(test_speakers), "Speaker leakage into test!"

    # Exactly 2 utterances per speaker
    from collections import Counter
    spk_counts = Counter(x["speaker_id"] for x in selected)
    assert all(c == 2 for c in spk_counts.values()), f"Expected exactly 2 utts per speaker: {spk_counts}"

    # Check zero utterance leakage
    selected_uids = set(x["full_id"] for x in selected)
    assert selected_uids.issubset(train_utts), "Found utterance outside Fold 1 train!"
    assert selected_uids.isdisjoint(val_utts), "Utterance leakage into val!"
    assert selected_uids.isdisjoint(test_utts), "Utterance leakage into test!"

    # 2. Token counts and representations
    tot_can = sum(x["canonical_phone_count"] for x in selected)
    tot_cor = sum(x["correct_count"] for x in selected)
    tot_sub = sum(x["substitution_count"] for x in selected)
    tot_del = sum(x["deletion_count"] for x in selected)

    assert tot_can == 1054
    assert tot_cor == 876
    assert tot_sub == 126
    assert tot_del == 52
    assert tot_cor + tot_sub + tot_del == tot_can

    # Check phone diversity
    all_can_phones = set()
    for x in selected:
        all_can_phones.update(x["canonical_phones"])
    assert len(all_can_phones) == 39, f"Expected 39 ARPAbet phones, got {len(all_can_phones)}"


def test_gpu_runtime_measurements_validity():
    metrics_path = Path("manifests/RQ1-TRN-01B0/GPU_RUNTIME_MEASUREMENTS.json")
    assert metrics_path.exists(), "GPU_RUNTIME_MEASUREMENTS.json does not exist"

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    assert meta["experiment_id"] == "RQ1-TRN-01B0"
    assert meta["status"] == "PASS"

    # Backbone and revision
    hp = data["hyperparameters"]
    assert hp["pretrained_backbone"] == DEFAULT_PRETRAINED_CHECKPOINT
    assert hp["backbone_revision"] == DEFAULT_PRETRAINED_REVISION
    assert hp["total_optimizer_updates"] == 100
    assert hp["micro_batch_size"] == 2
    assert hp["gradient_accumulation_steps"] == 2
    assert hp["effective_batch_size"] == 4

    # Runtime performance
    rp = data["runtime_performance"]
    assert rp["total_optimizer_updates"] == 100
    assert rp["total_micro_steps"] == 216
    assert rp["skipped_amp_steps"] == 8
    assert rp["wall_clock_time_seconds"] > 0.0

    # VRAM utilization
    vram = data["vram_utilization"]
    assert vram["peak_allocated_mb"] > 0.0
    assert vram["peak_reserved_mb"] > 0.0
    assert vram["available_headroom_mb"] > 1000.0  # At least 1 GB headroom

    # Evidence of learning
    delta = data["evidence_of_learning"]["loss_delta"]
    assert delta["total_loss_final"] < delta["total_loss_initial"]
    assert delta["ctc_loss_final"] < delta["ctc_loss_initial"]
    assert delta["binary_loss_final"] < delta["binary_loss_initial"]
    assert delta["diagnosis_loss_final"] < delta["diagnosis_loss_initial"]

    # All three loss components decreased
    assert delta["total_loss_relative_reduction_pct"] > 50.0  # >70% actual
    assert delta["ctc_loss_relative_reduction_pct"] > 50.0    # >80% actual
    assert delta["binary_loss_relative_reduction_pct"] > 20.0 # >30% actual
    assert delta["diagnosis_loss_relative_reduction_pct"] > 5.0 # ~8% actual

    # Task performance improved
    task_delta = data["evidence_of_learning"]["task_performance_delta"]
    assert task_delta["binary_pr_auc_final"] > task_delta["binary_pr_auc_initial"]
    assert task_delta["binary_f1_calibrated_final"] > task_delta["binary_f1_calibrated_initial"]
    assert task_delta["diagnosis_top1_accuracy_final"] > task_delta["diagnosis_top1_accuracy_initial"]
    assert task_delta["ctc_phone_error_rate_final"] < task_delta["ctc_phone_error_rate_initial"]

    # Checkpoint verification
    assert data["checkpoint"]["verification_status"] == "PASS"


def test_checkpoint_payload_integrity():
    ckpt_path = Path("checkpoints/RQ1-TRN-01B0/rq1_trn_01b0_checkpoint_step_100.pt")
    assert ckpt_path.exists(), "Checkpoint file missing"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    required_keys = [
        "model_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
        "scheduler_state_dict",
        "random_seed",
        "git_commit_sha",
        "split_manifest_checksum_sha256",
        "train_subset_manifest_path",
        "selected_training_ids",
        "hyperparameters",
        "training_step",
        "micro_step",
        "loss_gradient_history",
        "initial_metrics",
        "final_metrics"
    ]
    for k in required_keys:
        assert k in ckpt, f"Key {k} missing from checkpoint"

    assert ckpt["training_step"] == 100
    assert ckpt["random_seed"] == 42
    assert len(ckpt["selected_training_ids"]) == 32
    assert len(ckpt["loss_gradient_history"]) == 108
    assert sum(1 for s in ckpt["loss_gradient_history"] if not s["step_skipped"]) == 100
    assert sum(1 for s in ckpt["loss_gradient_history"] if s["step_skipped"]) == 8
    assert ckpt["split_manifest_checksum_sha256"] == "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b"
