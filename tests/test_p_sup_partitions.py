"""Regression tests for P-SUP data partitioning invariants.

Verifies:
1. FIT partition has exactly 1,799 utterances and 12 speakers.
2. SELECT partition has exactly 300 utterances and 2 speakers.
3. CAL partition has exactly 300 utterances and 2 speakers.
4. VAL partition has exactly 300 utterances and 2 speakers (preserved strictly unseen for evaluation).
5. Strict speaker and utterance disjointness across all partitions.
6. Sealed test speaker fail-closed rejection with SealedSpeakerContaminationError.
7. Configuration reflection in configs/rq1_p_sup_v1.json.
8. Bit-exact provenance hash alignment with SHARED_PROVENANCE.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from langfens_prep.p_sup.dataset import (
    PSupL2ArcticDataset,
    collate_psup_batch,
    FIT_SPEAKERS,
    SELECT_SPEAKERS,
    CALIBRATION_SPEAKERS,
    VAL_SPEAKERS,
    FOLD1_TRAIN_SPEAKERS,
    FOLD1_VAL_SPEAKERS,
    PARTITION_SPEAKERS,
    PARTITION_UTTERANCE_COUNTS,
    SealedSpeakerContaminationError
)
from langfens_prep.p_sup.labels import OperationLabels
from langfens_prep.p_sup.runtime import SEALED_SPEAKERS

MANIFEST_PATH = Path("manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json")
FAKE_ZIP_PATH = Path("datasets/fake.zip")
CONFIG_PATH = Path("configs/rq1_p_sup_v1.json")
PROVENANCE_PATH = Path("manifests/RQ1-WORKSPACE-PREP-01/SHARED_PROVENANCE.json")


def test_fit_partition_exact_counts_and_speakers():
    """Verify FIT partition has exactly 1,799 utterances across 12 speakers."""
    ds = PSupL2ArcticDataset(MANIFEST_PATH, FAKE_ZIP_PATH, partition="fit", verify_checksums=False)

    assert len(ds) == 1799
    assert ds.speakers == FIT_SPEAKERS
    assert len(ds.speakers) == 12

    # Check expected 12 speakers explicitly
    expected_fit_speakers = frozenset({
        "BWC", "ERMS", "HKK", "HQTV", "SKA", "SVBI",
        "TNI", "TXHC", "YBAA", "YDCK", "YKWK", "ZHAA"
    })
    assert ds.speakers == expected_fit_speakers

    # Per-speaker census: YBAA has 149 utterances, all other 11 speakers have 150 each
    counts = ds.speaker_counts()
    assert counts["YBAA"] == 149
    for spk in expected_fit_speakers:
        if spk != "YBAA":
            assert counts[spk] == 150, f"Speaker {spk} has {counts[spk]} utterances, expected 150"
    assert sum(counts.values()) == 1799

    # Also verify split="fit" alias
    ds_split = PSupL2ArcticDataset(MANIFEST_PATH, FAKE_ZIP_PATH, split="fit", verify_checksums=False)
    assert len(ds_split) == 1799
    assert ds_split.speakers == expected_fit_speakers


def test_selection_partition_exact_counts_and_speakers():
    """Verify SELECT partition has exactly 300 utterances across 2 speakers (MBMPS, TLV)."""
    ds = PSupL2ArcticDataset(MANIFEST_PATH, FAKE_ZIP_PATH, partition="selection", verify_checksums=False)

    assert len(ds) == 300
    assert ds.speakers == SELECT_SPEAKERS
    assert len(ds.speakers) == 2

    expected_select_speakers = frozenset({"MBMPS", "TLV"})
    assert ds.speakers == expected_select_speakers

    counts = ds.speaker_counts()
    assert counts["MBMPS"] == 150
    assert counts["TLV"] == 150
    assert sum(counts.values()) == 300

    # Also test partition="select" alias
    ds_alias = PSupL2ArcticDataset(MANIFEST_PATH, FAKE_ZIP_PATH, partition="select", verify_checksums=False)
    assert len(ds_alias) == 300
    assert ds_alias.speakers == expected_select_speakers


def test_calibration_partition_exact_counts_and_speakers():
    """Verify CAL partition has exactly 300 utterances across 2 speakers (NCC, THV)."""
    ds = PSupL2ArcticDataset(MANIFEST_PATH, FAKE_ZIP_PATH, partition="calibration", verify_checksums=False)

    assert len(ds) == 300
    assert ds.speakers == CALIBRATION_SPEAKERS
    assert len(ds.speakers) == 2

    expected_cal_speakers = frozenset({"NCC", "THV"})
    assert ds.speakers == expected_cal_speakers

    counts = ds.speaker_counts()
    assert counts["NCC"] == 150
    assert counts["THV"] == 150
    assert sum(counts.values()) == 300

    # Also test partition="cal" alias
    ds_alias = PSupL2ArcticDataset(MANIFEST_PATH, FAKE_ZIP_PATH, partition="cal", verify_checksums=False)
    assert len(ds_alias) == 300
    assert ds_alias.speakers == expected_cal_speakers


def test_val_partition_exact_counts_and_speakers():
    """Verify VAL partition has exactly 300 utterances across 2 speakers (NJS, RRBI)."""
    ds = PSupL2ArcticDataset(MANIFEST_PATH, FAKE_ZIP_PATH, partition="val", verify_checksums=False)

    assert len(ds) == 300
    assert ds.speakers == VAL_SPEAKERS
    assert len(ds.speakers) == 2

    expected_val_speakers = frozenset({"NJS", "RRBI"})
    assert ds.speakers == expected_val_speakers

    counts = ds.speaker_counts()
    assert counts["NJS"] == 150
    assert counts["RRBI"] == 150
    assert sum(counts.values()) == 300

    # Also test partition="evaluation" and split="val" aliases
    ds_eval = PSupL2ArcticDataset(MANIFEST_PATH, FAKE_ZIP_PATH, partition="evaluation", verify_checksums=False)
    assert len(ds_eval) == 300
    assert ds_eval.speakers == expected_val_speakers

    ds_split = PSupL2ArcticDataset(MANIFEST_PATH, FAKE_ZIP_PATH, split="val", verify_checksums=False)
    assert len(ds_split) == 300
    assert ds_split.speakers == expected_val_speakers


def test_strict_speaker_disjointness_across_all_partitions():
    """Verify strict pairwise speaker disjointness across FIT, SELECT, CAL, VAL, and SEALED."""
    partitions = {
        "fit": FIT_SPEAKERS,
        "selection": SELECT_SPEAKERS,
        "calibration": CALIBRATION_SPEAKERS,
        "val": VAL_SPEAKERS,
        "sealed_test": SEALED_SPEAKERS
    }

    names = list(partitions.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a, name_b = names[i], names[j]
            overlap = partitions[name_a] & partitions[name_b]
            assert not overlap, f"Speaker overlap between {name_a} and {name_b}: {overlap}"

    # Verify FIT + SELECT + CAL reconstructs full Fold-1 train set
    train_reconstructed = FIT_SPEAKERS | SELECT_SPEAKERS | CALIBRATION_SPEAKERS
    assert train_reconstructed == FOLD1_TRAIN_SPEAKERS
    assert len(train_reconstructed) == 16

    # Verify all 24 speakers accounted for
    all_speakers = train_reconstructed | VAL_SPEAKERS | SEALED_SPEAKERS
    assert len(all_speakers) == 24


def test_strict_utterance_disjointness_across_all_partitions():
    """Verify utterance ID sets across partitions are strictly pairwise disjoint."""
    fit_ds = PSupL2ArcticDataset(MANIFEST_PATH, FAKE_ZIP_PATH, partition="fit", verify_checksums=False)
    sel_ds = PSupL2ArcticDataset(MANIFEST_PATH, FAKE_ZIP_PATH, partition="selection", verify_checksums=False)
    cal_ds = PSupL2ArcticDataset(MANIFEST_PATH, FAKE_ZIP_PATH, partition="calibration", verify_checksums=False)
    val_ds = PSupL2ArcticDataset(MANIFEST_PATH, FAKE_ZIP_PATH, partition="val", verify_checksums=False)

    u_fit = set(fit_ds.entries)
    u_sel = set(sel_ds.entries)
    u_cal = set(cal_ds.entries)
    u_val = set(val_ds.entries)

    assert len(u_fit & u_sel) == 0, "Utterance overlap between FIT and SELECT"
    assert len(u_fit & u_cal) == 0, "Utterance overlap between FIT and CAL"
    assert len(u_fit & u_val) == 0, "Utterance overlap between FIT and VAL"
    assert len(u_sel & u_cal) == 0, "Utterance overlap between SELECT and CAL"
    assert len(u_sel & u_val) == 0, "Utterance overlap between SELECT and VAL"
    assert len(u_cal & u_val) == 0, "Utterance overlap between CAL and VAL"

    assert len(u_fit) + len(u_sel) + len(u_cal) == 2399  # Full train split
    assert len(u_val) == 300                             # Full val split


def test_val_partition_preserved_unseen_during_training():
    """Verify Fold-1 VAL speakers (NJS, RRBI) are strictly absent from training, selection, and calibration."""
    fit_ds = PSupL2ArcticDataset(MANIFEST_PATH, FAKE_ZIP_PATH, partition="fit", verify_checksums=False)
    sel_ds = PSupL2ArcticDataset(MANIFEST_PATH, FAKE_ZIP_PATH, partition="selection", verify_checksums=False)
    cal_ds = PSupL2ArcticDataset(MANIFEST_PATH, FAKE_ZIP_PATH, partition="calibration", verify_checksums=False)

    for ds_name, ds in [("FIT", fit_ds), ("SELECT", sel_ds), ("CAL", cal_ds)]:
        overlap = ds.speakers & VAL_SPEAKERS
        assert not overlap, f"VAL speaker(s) leaked into {ds_name} partition: {overlap}"


def test_sealed_test_partition_fail_closed_rejection():
    """Verify sealed test partition or speaker access fail-closed with SealedSpeakerContaminationError."""
    # 1. Reject split="test"
    with pytest.raises(SealedSpeakerContaminationError):
        PSupL2ArcticDataset(MANIFEST_PATH, FAKE_ZIP_PATH, split="test", verify_checksums=False)

    # 2. Reject partition="test"
    with pytest.raises(SealedSpeakerContaminationError):
        PSupL2ArcticDataset(MANIFEST_PATH, FAKE_ZIP_PATH, partition="test", verify_checksums=False)

    # 3. Reject partition="sealed_test"
    with pytest.raises(SealedSpeakerContaminationError):
        PSupL2ArcticDataset(MANIFEST_PATH, FAKE_ZIP_PATH, partition="sealed_test", verify_checksums=False)

    # 4. Accessing any sealed test speaker at runtime raises SealedSpeakerContaminationError
    ds = PSupL2ArcticDataset(MANIFEST_PATH, FAKE_ZIP_PATH, partition="fit", verify_checksums=False)
    for test_spk in SEALED_SPEAKERS:
        ds.entries = [(test_spk, "arctic_a0001")]
        with pytest.raises(SealedSpeakerContaminationError):
            _ = ds[0]

    # 5. Collator rejects batch containing any sealed test speaker
    dummy_labels = OperationLabels(
        canonical_ids=torch.tensor([1], dtype=torch.long),
        operation=torch.tensor([0], dtype=torch.long),
        binary=torch.tensor([0], dtype=torch.long),
        realization=torch.tensor([0], dtype=torch.long),
        ctc=torch.tensor([1], dtype=torch.long),
        addition_gaps=(),
        unknown_realized_intervals=()
    )
    for test_spk in SEALED_SPEAKERS:
        bad_item = {
            "speaker_id": test_spk,
            "utterance_id": "arctic_a0001",
            "waveform": torch.randn(1600),
            "labels": dummy_labels
        }
        with pytest.raises(SealedSpeakerContaminationError):
            collate_psup_batch([bad_item])


def test_config_rq1_p_sup_v1_partition_reflection():
    """Verify configs/rq1_p_sup_v1.json accurately reflects the partition specifications."""
    assert CONFIG_PATH.is_file(), f"Missing config file: {CONFIG_PATH}"
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    # Utterance counts
    assert cfg["fit_utterances"] == 1799
    assert cfg["selection_utterances"] == 300
    assert cfg["calibration_utterances"] == 300
    assert cfg["evaluation_utterances"] == 300

    # Speaker memberships
    assert set(cfg["fit_speakers"]) == FIT_SPEAKERS
    assert len(cfg["fit_speakers"]) == 12

    assert set(cfg["selection_speakers"]) == SELECT_SPEAKERS
    assert len(cfg["selection_speakers"]) == 2

    assert set(cfg["calibration_speakers"]) == CALIBRATION_SPEAKERS
    assert len(cfg["calibration_speakers"]) == 2

    assert set(cfg["evaluation_speakers"]) == VAL_SPEAKERS
    assert len(cfg["evaluation_speakers"]) == 2

    # Disjointness in config
    all_cfg_spks = (
        set(cfg["fit_speakers"]) |
        set(cfg["selection_speakers"]) |
        set(cfg["calibration_speakers"]) |
        set(cfg["evaluation_speakers"])
    )
    assert len(all_cfg_spks) == 18
    assert not (all_cfg_spks & SEALED_SPEAKERS), "Config contains sealed test speakers!"


def test_shared_provenance_ordered_hash_verification():
    """Verify ordered utterance ID SHA-256 digests align bit-exactly with SHARED_PROVENANCE.json."""
    splits = json.loads(MANIFEST_PATH.read_text())
    fold1 = next(f for f in splits["folds"] if f["fold_index"] == 1)
    prov = json.loads(PROVENANCE_PATH.read_text())
    registered_partitions = prov["fold1_partitions"]

    for part_name, reg in registered_partitions.items():
        envelope = fold1["val" if part_name == "val" else "train"]["utterances"]
        ids = [u for u in envelope if u.split("/")[0] in reg["speakers"]]
        digest = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
        assert len(ids) == reg["utterances"], f"Count mismatch for {part_name}"
        assert digest == reg["ordered_id_sha256"], f"Digest mismatch for {part_name}"
