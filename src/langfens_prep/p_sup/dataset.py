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

FIT_SPEAKERS = frozenset({
    "BWC", "ERMS", "HKK", "HQTV", "SKA", "SVBI",
    "TNI", "TXHC", "YBAA", "YDCK", "YKWK", "ZHAA"
})
SELECT_SPEAKERS = frozenset({"MBMPS", "TLV"})
CALIBRATION_SPEAKERS = frozenset({"NCC", "THV"})
VAL_SPEAKERS = frozenset({"NJS", "RRBI"})
EVALUATION_SPEAKERS = VAL_SPEAKERS

FOLD1_TRAIN_SPEAKERS = frozenset(FIT_SPEAKERS | SELECT_SPEAKERS | CALIBRATION_SPEAKERS)
FOLD1_VAL_SPEAKERS = VAL_SPEAKERS

PARTITION_SPEAKERS = {
    "fit": FIT_SPEAKERS,
    "selection": SELECT_SPEAKERS,
    "select": SELECT_SPEAKERS,
    "calibration": CALIBRATION_SPEAKERS,
    "cal": CALIBRATION_SPEAKERS,
    "val": VAL_SPEAKERS,
    "evaluation": EVALUATION_SPEAKERS,
    "train": FOLD1_TRAIN_SPEAKERS
}

PARTITION_UTTERANCE_COUNTS = {
    "fit": 1799,
    "selection": 300,
    "select": 300,
    "calibration": 300,
    "cal": 300,
    "val": 300,
    "evaluation": 300,
    "train": 2399,
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
    """Dataset connecting Fold 1 Train / Val utterances with operation labels and audio.

    Partitions:
    - fit: 1,799 utterances across 12 speakers (BWC, ERMS, HKK, HQTV, SKA, SVBI, TNI, TXHC, YBAA, YDCK, YKWK, ZHAA)
    - selection / select: 300 utterances across 2 speakers (MBMPS, TLV)
    - calibration / cal: 300 utterances across 2 speakers (NCC, THV)
    - val / evaluation: 300 utterances across 2 speakers (NJS, RRBI) [Preserved strictly unseen for evaluation]
    - train: 2,399 utterances across 16 speakers (fit + selection + calibration)
    - test: STRICTLY PROHIBITED (fail-closed with SealedSpeakerContaminationError)
    """

    def __init__(
        self,
        split_manifest_path: str | Path,
        zip_path: str | Path,
        split: Optional[str] = None,
        partition: Optional[str] = None,
        verify_checksums: bool = True,
        max_samples: Optional[int] = None
    ):
        super().__init__()
        self.split_manifest_path = Path(split_manifest_path)
        self.zip_path = Path(zip_path)

        # Fail-closed check for sealed test split or partition
        if split is not None and split.lower() in ("test", "sealed_test"):
            raise SealedSpeakerContaminationError("FAIL-CLOSED: Sealed test split requested!")
        if partition is not None and partition.lower() in ("test", "sealed_test"):
            raise SealedSpeakerContaminationError("FAIL-CLOSED: Sealed test partition requested!")

        # Resolve partition and split
        if partition is not None:
            resolved_partition = partition.lower()
            if resolved_partition not in PARTITION_SPEAKERS:
                raise ValueError(f"Unrecognized partition: {partition!r}")
            if resolved_partition in ("val", "evaluation"):
                resolved_split = "val"
                if split is not None and split.lower() not in ("val", "evaluation"):
                    raise ValueError(f"Contradictory split={split!r} and partition={partition!r}")
            else:
                resolved_split = "train"
                if split is not None and split.lower() not in ("train", resolved_partition):
                    raise ValueError(f"Contradictory split={split!r} and partition={partition!r}")
        elif split is not None:
            norm_split = split.lower()
            if norm_split in ("val", "evaluation"):
                resolved_partition = "val"
                resolved_split = "val"
            elif norm_split in ("fit", "selection", "select", "calibration", "cal"):
                resolved_partition = norm_split
                resolved_split = "train"
            elif norm_split == "train":
                resolved_partition = "train"
                resolved_split = "train"
            else:
                raise ValueError(f"Unrecognized split: {split!r}")
        else:
            resolved_partition = "fit"
            resolved_split = "train"

        self.partition = resolved_partition
        self.split = resolved_split

        if verify_checksums:
            verify_file_sha256(self.split_manifest_path, SPLIT_MANIFEST_SHA256)
            if self.zip_path.is_file():
                # Verify zip hash only if path exists locally
                verify_file_sha256(self.zip_path, AUTHORIZED_ZIP_SHA256)

        manifest = json.loads(self.split_manifest_path.read_text(encoding="utf-8"))
        fold_entry = next(f for f in manifest["folds"] if f["fold_index"] == 1)

        raw_utterances: List[str] = fold_entry[self.split]["utterances"]
        target_speakers = PARTITION_SPEAKERS[self.partition]

        # Fail-closed check: target_speakers must NEVER intersect SEALED_SPEAKERS
        if target_speakers & SEALED_SPEAKERS:
            raise SealedSpeakerContaminationError(
                f"FAIL-CLOSED: Target speakers contain sealed test speaker(s): {target_speakers & SEALED_SPEAKERS}"
            )

        self.entries: List[Tuple[str, str]] = []
        for u in raw_utterances:
            spk, uid = u.split("/")
            if spk in SEALED_SPEAKERS:
                raise SealedSpeakerContaminationError(
                    f"FAIL-CLOSED: Sealed test speaker {spk} encountered in manifest partition {self.split}!"
                )
            if spk in target_speakers:
                self.entries.append((spk, uid))

        if max_samples is not None:
            self.entries = self.entries[:max_samples]

        if not self.entries:
            raise ValueError(f"No utterances found for split={self.split}, partition={self.partition}")

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def speakers(self) -> frozenset[str]:
        return frozenset(spk for spk, _ in self.entries)

    def speaker_counts(self) -> Dict[str, int]:
        from collections import Counter
        return dict(Counter(spk for spk, _ in self.entries))

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
