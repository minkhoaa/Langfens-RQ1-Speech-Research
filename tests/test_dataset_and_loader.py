import pytest
import torch
from torch.utils.data import DataLoader

def test_l2arctic_dataset_loading_and_policy():
    from langfens_prep.dataset import L2ArcticMDDDataset
    
    manifest_path = "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json"
    
    # Load fold 1 validation split with strict policy
    ds_strict = L2ArcticMDDDataset(
        split_manifest_path=manifest_path,
        fold=1,
        split="val",
        policy="strict",
        max_samples=5
    )
    assert len(ds_strict) == 5
    sample = ds_strict[0]
    assert "audio_waveform_16k" in sample
    assert "canonical_phones_norm39" in sample
    assert "binary_labels" in sample
    assert "record_id" in sample
    assert sample["binary_labels"].dtype == torch.long
    
    # Load fold 1 validation split with lenient policy
    ds_lenient = L2ArcticMDDDataset(
        split_manifest_path=manifest_path,
        fold=1,
        split="val",
        policy="lenient",
        max_samples=5
    )
    sample_len = ds_lenient[0]
    assert sample_len["binary_labels"].shape == sample["binary_labels"].shape

def test_l2arctic_blinded_test_isolation():
    from langfens_prep.dataset import L2ArcticMDDDataset
    
    manifest_path = "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json"
    
    ds_test_blinded = L2ArcticMDDDataset(
        split_manifest_path=manifest_path,
        fold=1,
        split="test",
        blind_labels=True,
        max_samples=3
    )
    sample = ds_test_blinded[0]
    assert sample["binary_labels"] is None
    assert sample["ctc_target_indices"] is None
    assert sample["is_blinded"] is True

def test_mdd_batch_collation():
    from langfens_prep.dataset import L2ArcticMDDDataset
    from langfens_prep.collate import LangfensMDDCollate
    
    manifest_path = "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json"
    ds = L2ArcticMDDDataset(
        split_manifest_path=manifest_path,
        fold=1,
        split="train",
        max_samples=4
    )
    
    loader = DataLoader(ds, batch_size=4, collate_fn=LangfensMDDCollate())
    batch = next(iter(loader))
    
    assert "net_input" in batch
    assert "source" in batch["net_input"]
    assert "padding_mask" in batch["net_input"]
    assert "audio_lengths" in batch["net_input"]
    assert "target_binary" in batch
    assert "target_ctc" in batch
    assert "target_lengths" in batch
    
    assert batch["net_input"]["source"].shape[0] == 4
    assert batch["target_binary"].shape[0] == 4
    # Check padding value for binary targets is -100
    assert (batch["target_binary"] == -100).any()

def test_speechocean_score_dataset():
    from langfens_prep.dataset import SpeechOcean762ScoreDataset
    
    manifest_path = "manifests/RQ1-SPLIT-01/speechocean762_official_splits.json"
    ds = SpeechOcean762ScoreDataset(
        split_manifest_path=manifest_path,
        split="val",
        max_samples=4
    )
    assert len(ds) == 4
    sample = ds[0]
    assert "audio_waveform_16k" in sample
    assert "accuracy_score" in sample
    assert "completeness_score" in sample
    assert "fluency_score" in sample
    assert "prosodic_score" in sample
    assert "total_score" in sample
    assert "words" in sample
