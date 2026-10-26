import pytest
import tempfile
import torch
import numpy as np
from pathlib import Path

from audit_scripts.evaluate_fold1_ctc_binary import (
    compute_sha256,
    verify_checkpoint_integrity,
    get_model_weights_fingerprint,
    generate_derangement_indices,
    calculate_binary_metrics,
    calculate_ctc_metrics,
    fit_canonical_empirical_prior,
    evaluate_canonical_prior,
    build_eval_model
)
from langfens_prep.model import LangfensWav2Vec2MDD


def test_compute_sha256(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_bytes(b"hello world\n")
    digest = compute_sha256(test_file)
    import hashlib
    expected = hashlib.sha256(b"hello world\n").hexdigest()
    assert digest == expected


def test_verify_checkpoint_integrity_rejects_missing(tmp_path):
    missing_path = tmp_path / "non_existent.pt"
    with pytest.raises(FileNotFoundError):
        verify_checkpoint_integrity(missing_path)


def test_verify_checkpoint_integrity_rejects_invalid_format(tmp_path):
    bad_ckpt = tmp_path / "bad.pt"
    torch.save(["not", "a", "dict"], bad_ckpt)
    with pytest.raises(ValueError, match="must deserialize to a dictionary"):
        verify_checkpoint_integrity(bad_ckpt)

    bad_dict = tmp_path / "missing_model.pt"
    torch.save({"epoch": 1}, bad_dict)
    with pytest.raises(KeyError, match="missing required 'model' state dict"):
        verify_checkpoint_integrity(bad_dict)


def test_generate_derangement_indices():
    # Test for various sequence lengths
    for n in [2, 5, 10, 150, 300]:
        indices = generate_derangement_indices(n, method="cyclic")
        assert len(indices) == n
        # Must be a permutation of 0..n-1
        assert sorted(indices) == list(range(n))
        # No fixed points (no self-matches)
        for i, j in enumerate(indices):
            assert i != j, f"Self-match found at index {i} == {j}"


def test_generate_derangement_within_speakers():
    speaker_counts = [150, 150]
    total_n = sum(speaker_counts)
    indices = generate_derangement_indices(total_n, method="within_speaker_cyclic", speaker_boundaries=[150])
    assert len(indices) == total_n
    assert sorted(indices) == list(range(total_n))
    for i, j in enumerate(indices):
        assert i != j
        # Verify within speaker 1
        if i < 150:
            assert j < 150
        else:
            assert j >= 150


def test_get_model_weights_fingerprint_and_immutability():
    model = LangfensWav2Vec2MDD(
        hidden_dim=32, conv_dim=16, num_transformer_layers=1,
        num_attention_heads=4, intermediate_dim=64, dropout=0.1
    )
    fp_before = get_model_weights_fingerprint(model)
    assert isinstance(fp_before, str) and len(fp_before) == 64

    # Run dummy forward pass under inference mode
    model.eval()
    with torch.inference_mode():
        dummy_audio = torch.randn(1, 16000)
        dummy_can = torch.tensor([[1, 2, 3]], dtype=torch.long)
        _ = model(
            source=dummy_audio,
            audio_lengths=torch.tensor([16000]),
            canonical_phone_indices=dummy_can,
            canonical_phone_lengths=torch.tensor([3])
        )

    fp_after = get_model_weights_fingerprint(model)
    assert fp_before == fp_after, "Model weights changed after inference pass!"


def test_calculate_binary_metrics():
    # Simple synthetic test
    # 4 tokens: 2 positives, 2 negatives
    labels = [1, 0, 1, 0]
    scores = [0.9, 0.1, 0.8, 0.2]
    metrics = calculate_binary_metrics(labels, scores, thresholds=[0.5])
    assert metrics["prevalence"] == 0.5
    assert metrics["average_precision"] == 1.0  # perfect separation
    op05 = metrics["operating_point_standard_05"]
    assert op05["tp"] == 2
    assert op05["fp"] == 0
    assert op05["fn"] == 0
    assert op05["tn"] == 2
    assert op05["precision"] == 1.0
    assert op05["recall"] == 1.0
    assert op05["f1"] == 1.0


def test_calculate_ctc_metrics():
    refs = [[1, 2, 3], [4, 5]]
    hyps = [[1, 2, 6], [4]]  # 1 sub in first, 1 del in second
    blank_frames = 100
    total_frames = 120
    metrics = calculate_ctc_metrics(refs, hyps, blank_frames, total_frames)
    assert metrics["total_ref_tokens"] == 5
    assert metrics["total_hyp_tokens"] == 4
    assert metrics["substitutions"] == 1
    assert metrics["deletions"] == 1
    assert metrics["insertions"] == 0
    assert metrics["total_edits"] == 2
    assert pytest.approx(metrics["greedy_per"], abs=1e-5) == 2 / 5
    assert pytest.approx(metrics["blank_fraction"], abs=1e-5) == 100 / 120


def test_canonical_empirical_prior_synthetic():
    class DummyItem:
        def __init__(self, can, y):
            self.data = {
                "canonical_phone_indices": torch.tensor(can, dtype=torch.long),
                "binary_labels": torch.tensor(y, dtype=torch.long)
            }
        def __getitem__(self, k):
            return self.data[k]

    # Phone 1 has 1 error out of 2 (p=0.5)
    # Phone 2 has 0 errors out of 2 (p=0.0)
    train_items = [
        DummyItem([1, 2], [1, 0]),
        DummyItem([1, 2], [0, 0])
    ]
    prior_data = fit_canonical_empirical_prior(train_items)
    assert prior_data["overall_train_rate"] == 0.25
    assert prior_data["phone_priors"][1] == 0.5
    assert prior_data["phone_priors"][2] == 0.0

    val_items = [
        DummyItem([1, 2], [1, 0])
    ]
    val_res = evaluate_canonical_prior(val_items, prior_data)
    assert val_res["total_tokens"] == 2
    assert val_res["scores"] == [0.5, 0.0]
    assert val_res["labels"] == [1, 0]
    assert val_res["average_precision"] == 1.0
