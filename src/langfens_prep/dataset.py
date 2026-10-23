import json
import os
import tarfile
from typing import Optional, List, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset

from langfens_prep.audio_loader import (
    load_l2arctic_audio_from_zip,
    load_speechocean_audio_from_tar,
    get_l2arctic_reader
)
from langfens_prep.textgrid_parser import parse_gold_textgrid_content
from langfens_prep.phoneme_normalizer import PHONE_TO_CTC_INDEX

class L2ArcticMDDDataset(Dataset):
    """
    Lightweight, manifest-driven PyTorch Dataset for L2-ARCTIC Mispronunciation
    Detection & Diagnosis (MDD). Strictly consumes frozen split manifests and guarantees
    zero test-label contamination.
    """
    def __init__(
        self,
        split_manifest_path: str,
        fold: int,
        split: str,
        policy: str = "strict",
        blind_labels: bool = False,
        target_sr: int = 16000,
        zip_path: str = "datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip",
        max_samples: Optional[int] = None
    ):
        super().__init__()
        assert policy in ("strict", "lenient"), f"Invalid policy: {policy}"
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        
        self.split_manifest_path = split_manifest_path
        self.fold = fold
        self.split = split
        self.policy = policy
        self.blind_labels = blind_labels
        self.target_sr = target_sr
        self.zip_path = zip_path
        
        with open(split_manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            
        # Locate fold by fold_index (1-indexed)
        target_fold = None
        for f_entry in manifest_data["folds"]:
            if f_entry["fold_index"] == fold:
                target_fold = f_entry
                break
                
        if target_fold is None:
            raise KeyError(f"Fold with fold_index={fold} not found in manifest {split_manifest_path}")
            
        # Utterances in format "<SPEAKER>/<UTT_ID>"
        raw_utts: List[str] = target_fold[split]["utterances"]
        self.utterance_entries: List[Tuple[str, str]] = []
        for item in raw_utts:
            parts = item.split("/")
            if len(parts) == 2:
                self.utterance_entries.append((parts[0], parts[1]))
            else:
                self.utterance_entries.append((parts[0], parts[0]))
        
        if max_samples is not None:
            self.utterance_entries = self.utterance_entries[:max_samples]

    def __len__(self) -> int:
        return len(self.utterance_entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        speaker_id, utt_id = self.utterance_entries[idx]
        reader = get_l2arctic_reader(self.zip_path)
        
        # 1. Load audio safely from nested zip
        waveform, sr, audio_metrics = load_l2arctic_audio_from_zip(
            zip_path=self.zip_path,
            speaker=speaker_id,
            utterance_id=utt_id,
            target_sr=self.target_sr
        )
        
        # 2. Load and parse TextGrid
        tg_text = reader.read_annotation_text(speaker_id, utt_id)
        annotation = parse_gold_textgrid_content(tg_text, f"{utt_id}.TextGrid")
        
        canonical_intervals = annotation.canonical_intervals
        canonical_phones = [i.canonical_phone for i in canonical_intervals]
        realized_phones = [i.realized_phone for i in canonical_intervals]
        
        if self.blind_labels:
            binary_labels = None
            ctc_target_indices = None
        else:
            if self.policy == "strict":
                bin_vals = [i.binary_label_strict for i in canonical_intervals]
            else:
                bin_vals = [i.binary_label_lenient for i in canonical_intervals]
            binary_labels = torch.tensor(bin_vals, dtype=torch.long)
            
            ctc_indices = [PHONE_TO_CTC_INDEX.get(p, 0) for p in realized_phones if p in PHONE_TO_CTC_INDEX]
            ctc_target_indices = torch.tensor(ctc_indices, dtype=torch.long)
            
        return {
            "record_id": f"l2arctic_{speaker_id}_{utt_id}",
            "utterance_id": utt_id,
            "speaker_id": speaker_id,
            "audio_waveform_16k": waveform,
            "canonical_phones_norm39": canonical_phones,
            "realized_phones_norm39": realized_phones,
            "binary_labels": binary_labels,
            "ctc_target_indices": ctc_target_indices,
            "is_blinded": self.blind_labels,
            "policy": self.policy,
            "duration_seconds": audio_metrics.duration_seconds
        }

class SpeechOcean762ScoreDataset(Dataset):
    """
    Lightweight PyTorch Dataset for SpeechOcean762 multi-dimensional pronunciation scoring.
    Excludes the 245 unmanifested audio files and preserves raw continuous scores.
    """
    def __init__(
        self,
        split_manifest_path: str,
        split: str,
        tar_path: str = "datasets/SpeechOcean762/original/speechocean762.tar.gz",
        target_sr: int = 16000,
        max_samples: Optional[int] = None
    ):
        super().__init__()
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        self.split_manifest_path = split_manifest_path
        self.split = split
        self.tar_path = tar_path
        self.target_sr = target_sr
        
        with open(split_manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            
        self.allowed_utts = set(manifest_data["partitions"][split]["utterances"])
        
        with tarfile.open(self.tar_path, "r:gz") as tf:
            scores_file = tf.extractfile("speechocean762/resource/scores.json")
            all_scores = json.loads(scores_file.read().decode("utf-8"))
            
            utt2spk = {}
            utt2path = {}
            for s in ["train", "test"]:
                m = tf.extractfile(f"speechocean762/{s}/utt2spk")
                if m:
                    for line in m.read().decode("utf-8").splitlines():
                        p = line.strip().split()
                        if len(p) >= 2:
                            utt2spk[p[0]] = p[1]
                w = tf.extractfile(f"speechocean762/{s}/wav.scp")
                if w:
                    for line in w.read().decode("utf-8").splitlines():
                        p = line.strip().split()
                        if len(p) >= 2:
                            utt2path[p[0]] = p[1]

        self.samples = []
        for utt_id in sorted(list(self.allowed_utts)):
            if utt_id in all_scores and utt_id in utt2path:
                self.samples.append({
                    "utterance_id": utt_id,
                    "speaker_id": utt2spk.get(utt_id, "unknown"),
                    "member_path": f"speechocean762/{utt2path[utt_id]}",
                    "data": all_scores[utt_id]
                })
                
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        utt_id = item["utterance_id"]
        speaker_id = item["speaker_id"]
        member_path = item["member_path"]
        score_data = item["data"]
        
        waveform, sr, audio_metrics = load_speechocean_audio_from_tar(
            tar_path=self.tar_path,
            member_path=member_path,
            target_sr=self.target_sr
        )
        
        return {
            "record_id": f"speechocean762_{speaker_id}_{utt_id}",
            "utterance_id": utt_id,
            "speaker_id": speaker_id,
            "audio_waveform_16k": waveform,
            "accuracy_score": float(score_data.get("accuracy", 0.0)),
            "completeness_score": float(score_data.get("completeness", 0.0)),
            "fluency_score": float(score_data.get("fluency", 0.0)),
            "prosodic_score": float(score_data.get("prosodic", 0.0)),
            "total_score": float(score_data.get("total", 0.0)),
            "words": score_data.get("words", []),
            "duration_seconds": audio_metrics.duration_seconds
        }
