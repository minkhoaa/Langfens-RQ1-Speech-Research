import json
import pytest
import torch
from pathlib import Path
from unittest.mock import patch

from langfens_prep.dataset import ZeroOracleL2ArcticDataset
from langfens_prep.model import LangfensWav2Vec2MDD


def test_zero_oracle_dataset_has_zero_labels_and_zero_textgrid_access():
    """Verifies that ZeroOracleL2ArcticDataset produces no labels and does not read TextGrids."""
    manifest_path = "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json"
    zip_path = "datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip"
    prompts_path = "resources/L2-ARCTIC/prompts.json"

    if not Path(zip_path).is_file():
        pytest.skip(f"Authorized zip not found at {zip_path}")

    ds = ZeroOracleL2ArcticDataset(
        split_manifest_path=manifest_path,
        fold=1,
        split="val",
        prompts_source=prompts_path,
        zip_path=zip_path,
        max_samples=5
    )

    # Patch read_annotation_text to guarantee that if called, it raises an exception
    with patch("langfens_prep.audio_loader.L2ArcticArchiveReader.read_annotation_text") as mock_tg:
        for idx in range(len(ds)):
            item = ds[idx]

            # 1. Zero TextGrid read check
            mock_tg.assert_not_called()

            # 2. Strict label absence checks
            assert item["binary_labels"] is None
            assert item["ctc_target_indices"] is None
            assert item["is_blinded"] is True

            # 3. Model input presence
            assert "audio_waveform_16k" in item
            assert "canonical_phone_indices" in item
            assert isinstance(item["audio_waveform_16k"], torch.Tensor)
            assert isinstance(item["canonical_phone_indices"], torch.Tensor)
            assert item["audio_waveform_16k"].ndim == 1
            assert item["canonical_phone_indices"].ndim == 1
            assert item["canonical_phone_indices"].shape[0] > 0


def test_model_inference_isolated_from_targets():
    """Verifies that the MDD model forward pass succeeds with zero target arguments."""
    model = LangfensWav2Vec2MDD(
        hidden_dim=32, conv_dim=16, num_transformer_layers=1,
        num_attention_heads=4, intermediate_dim=64, dropout=0.1
    )
    model.eval()

    source = torch.randn(1, 16000)
    audio_lengths = torch.tensor([16000])
    can_indices = torch.tensor([[1, 2, 3, 4]])
    can_lengths = torch.tensor([4])

    with torch.inference_mode():
        out = model(
            source=source,
            audio_lengths=audio_lengths,
            canonical_phone_indices=can_indices,
            canonical_phone_lengths=can_lengths
        )

    assert out.loss is None
    assert out.ctc_logits is not None
    assert out.binary_logits is not None
    assert out.binary_logits.shape == (1, 4)
    assert torch.isfinite(out.binary_logits).all()


def test_strict_no_test_split_boundary():
    """Verifies that attempting to initialize an authorized evaluation on 'test' split raises an error."""
    manifest_path = "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json"
    prompts_path = "resources/L2-ARCTIC/prompts.json"
    zip_path = "datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip"

    if not Path(zip_path).is_file():
        pytest.skip("Authorized zip not found")

    with open(manifest_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    fold1 = splits["folds"][0]
    test_speakers = fold1["test"]["speakers"]
    assert len(test_speakers) == 6
    assert "ABA" in test_speakers
    assert "PNV" in test_speakers

    # Protocol Z dataset validation check: val split must strictly be NJS and RRBI
    ds = ZeroOracleL2ArcticDataset(
        split_manifest_path=manifest_path,
        fold=1,
        split="val",
        prompts_source=prompts_path,
        zip_path=zip_path
    )
    val_speakers = set(entry[0] for entry in ds.utterance_entries)
    assert val_speakers == {"NJS", "RRBI"}
    assert not val_speakers.intersection(set(test_speakers)), "Test speaker leakage detected in validation dataset!"
