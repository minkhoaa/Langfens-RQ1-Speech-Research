"""Data allowlist, loader, and collation for Fold 1 P-SUP training and evaluation.

Strictly enforces:
- Fold 1 TRAIN (2,399 utterances across 16 speakers)
- Fold 1 VAL (300 utterances across NJS, RRBI)
- Physical absence and fail-closed rejection of sealed TEST speakers (ABA, ASI, EBVS, HJK, LXC, PNV)
- Verified SHA-256 for split manifest and fold zip archive
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from langfens_prep.audio_loader import load_l2arctic_audio_from_zip, get_l2arctic_reader
from langfens_prep.textgrid_parser import parse_gold_textgrid_content
from langfens_prep.p_sup.labels import build_operation_labels, OperationLabels
from langfens_prep.p_sup.runtime import SEALED_SPEAKERS

SPLIT_MANIFEST_SHA256 = "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b"
AUTHORIZED_ZIP_SHA256 = "b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9"

FOLD1_TRAIN_SPEAKERS = frozenset({
    "BWC", "ERMS", "HKK", "HQTV", "MBMPS", "NCC",
    "SKA", "SVBI", "THV", "TLV", "TNI", "TXHC",
    "YBAA", "YDCK", "YKWK", "ZHAA"
})
FOLD1_VAL_SPEAKERS = frozenset({"NJS", "RRBI"})

PARTITION_SPEAKERS = {
    "fit": frozenset({"BWC", "ERMS", "HKK", "HQTV", "SKA", "SVBI", "TNI", "TXHC", "YBAA", "YDCK", "YKWK", "ZHAA"}),
    "selection": frozenset({"MBMPS", "TLV"}),
    "calibration": frozenset({"NCC", "THV"}),
    "val": frozenset({"NJS", "RRBI"}),
    "train": FOLD1_TRAIN_SPEAKERS
}


class SealedSpeakerContaminationError(RuntimeError):
    """Raised when an active test speaker is requested or detected."""
    pass


def verify_file_sha256(path: Path | str, expected_sha256: str) -> None:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {p}: actual {actual} != expected {expected_sha256}")


class PSupL2ArcticDataset(Dataset):
    """Dataset connecting Fold 1 Train / Val utterances with operation labels and audio."""

    def __init__(
        self,
        split_manifest_path: str | Path,
        zip_path: str | Path,
        split: str = "train",
        partition: Optional[str] = None,
        verify_checksums: bool = True,
        max_samples: Optional[int] = None
    ):
        super().__init__()
        self.split_manifest_path = Path(split_manifest_path)
        self.zip_path = Path(zip_path)
        self.split = split
        self.partition = partition

        if split not in ("train", "val"):
            raise ValueError(f"P-SUP Dataset strictly accepts 'train' or 'val', got {split!r}")

        if verify_checksums:
            verify_file_sha256(self.split_manifest_path, SPLIT_MANIFEST_SHA256)
            if self.zip_path.is_file():
                # Verify zip hash only if path exists locally
                verify_file_sha256(self.zip_path, AUTHORIZED_ZIP_SHA256)

        manifest = json.loads(self.split_manifest_path.read_text(encoding="utf-8"))
        fold_entry = next(f for f in manifest["folds"] if f["fold_index"] == 1)

        raw_utterances: List[str] = fold_entry[split]["utterances"]

        # Filter by partition if specified
        target_speakers = PARTITION_SPEAKERS[partition] if partition else (
            FOLD1_TRAIN_SPEAKERS if split == "train" else FOLD1_VAL_SPEAKERS
        )

        self.entries: List[Tuple[str, str]] = []
        for u in raw_utterances:
            spk, uid = u.split("/")
            if spk in SEALED_SPEAKERS:
                raise SealedSpeakerContaminationError(
                    f"FAIL-CLOSED: Sealed test speaker {spk} encountered in manifest partition {split}!"
                )
            if spk in target_speakers:
                self.entries.append((spk, uid))

        if max_samples is not None:
            self.entries = self.entries[:max_samples]

        if not self.entries:
            raise ValueError(f"No utterances found for split={split}, partition={partition}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        spk, uid = self.entries[idx]
        if spk in SEALED_SPEAKERS:
            raise SealedSpeakerContaminationError(f"Sealed test speaker {spk} accessed at runtime!")

        # 1. Load audio
        waveform, sr, _ = load_l2arctic_audio_from_zip(
            zip_path=str(self.zip_path),
            speaker=spk,
            utterance_id=uid,
            target_sr=16000
        )

        # 2. Load annotation and build operation labels
        reader = get_l2arctic_reader(str(self.zip_path))
        tg_text = reader.read_annotation_text(spk, uid)
        annotation = parse_gold_textgrid_content(tg_text, f"{uid}.TextGrid")
        labels = build_operation_labels(annotation)

        return {
            "speaker_id": spk,
            "utterance_id": uid,
            "waveform": waveform,
            "labels": labels
        }


def collate_psup_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate waveforms and OperationLabels into a batch suitable for CompactOperationHead training."""
    if not batch:
        raise ValueError("empty batch")

    speaker_ids = [item["speaker_id"] for item in batch]
    utterance_ids = [item["utterance_id"] for item in batch]

    for spk in speaker_ids:
        if spk in SEALED_SPEAKERS:
            raise SealedSpeakerContaminationError(f"Sealed speaker {spk} detected in collator!")

    waveforms = [item["waveform"] for item in batch]
    audio_lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)
    padded_waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0.0)

    labels_list = [item["labels"] for item in batch]
    prompt_ids = [lab.canonical_ids for lab in labels_list]
    prompt_lengths = torch.tensor([p.numel() for p in prompt_ids], dtype=torch.long)
    padded_prompts = pad_sequence(prompt_ids, batch_first=True, padding_value=0)

    padded_operations = pad_sequence([lab.operation for lab in labels_list], batch_first=True, padding_value=-100)
    padded_binaries = pad_sequence([lab.binary for lab in labels_list], batch_first=True, padding_value=-100)
    padded_realizations = pad_sequence([lab.realization for lab in labels_list], batch_first=True, padding_value=-100)

    ctc_targets = [lab.ctc for lab in labels_list]
    ctc_lengths = torch.tensor([c.numel() for c in ctc_targets], dtype=torch.long)
    padded_ctc = pad_sequence(ctc_targets, batch_first=True, padding_value=0)

    return {
        "speaker_ids": speaker_ids,
        "utterance_ids": utterance_ids,
        "waveforms": padded_waveforms,
        "audio_lengths": audio_lengths,
        "prompt_phone_ids": padded_prompts,
        "prompt_lengths": prompt_lengths,
        "operation": padded_operations,
        "operation_mask": padded_operations != -100,
        "binary": padded_binaries,
        "binary_mask": padded_binaries != -100,
        "realization": padded_realizations,
        "ctc": padded_ctc,
        "ctc_lengths": ctc_lengths
    }
