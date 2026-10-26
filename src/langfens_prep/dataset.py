import json
import os
import tarfile
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import re

import torch
from torch.utils.data import Dataset

from langfens_prep.audio_loader import (
    load_l2arctic_audio_from_zip,
    load_speechocean_audio_from_tar,
    get_l2arctic_reader
)
from langfens_prep.textgrid_parser import parse_gold_textgrid_content
from langfens_prep.phoneme_normalizer import (
    PHONE_TO_CTC_INDEX,
    PHONE_TO_INDEX,
    STANDARD_39_ARPABET
)

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
        
        # 3. Canonical prompt tokens (available at test/deployment time)
        canonical_intervals = annotation.canonical_intervals
        canonical_phones = [i.canonical_phone for i in canonical_intervals]
        canonical_phone_indices = torch.tensor(
            [PHONE_TO_CTC_INDEX[p] for p in canonical_phones if p in PHONE_TO_CTC_INDEX],
            dtype=torch.long
        )

        # 4. Chronological Realized Phone Reconstruction (Task C)
        # Includes: valid 39-ARPAbet correct productions, substitutions, and epenthetic additions.
        # Excludes: physical silence, acoustic deletions, unaligned 'err' tokens, non-standard symbols.
        chronological_realized_phones = []
        for i in annotation.intervals:
            if i.error_type == "silence":
                continue
            elif i.error_type == "deletion":
                continue
            elif i.has_err_tag:
                continue
            p = i.realized_phone
            if p and p in STANDARD_39_ARPABET:
                chronological_realized_phones.append(p)
        
        if self.blind_labels:
            binary_labels = None
            ctc_target_indices = None
            diagnosis_target_indices = None
            aux_phone_target_indices = None
        else:
            if self.policy == "strict":
                bin_vals = [i.binary_label_strict for i in canonical_intervals]
            else:
                bin_vals = [i.binary_label_lenient for i in canonical_intervals]
            binary_labels = torch.tensor(bin_vals, dtype=torch.long)
            
            ctc_indices = [PHONE_TO_CTC_INDEX[p] for p in chronological_realized_phones]
            ctc_target_indices = torch.tensor(ctc_indices, dtype=torch.long)
            
            # Primary substitution-only diagnosis target (correct phones, deletions, and 'err' tags masked as -100)
            diag_indices = [
                PHONE_TO_INDEX[i.diagnosis_target] if (i.diagnosis_target is not None and i.diagnosis_target in PHONE_TO_INDEX)
                else -100
                for i in canonical_intervals
            ]
            diagnosis_target_indices = torch.tensor(diag_indices, dtype=torch.long)

            # Auxiliary realized-phone classification target (evaluated over all valid canonical phones, including correct)
            aux_indices = [
                PHONE_TO_INDEX[i.realized_phone] if (i.error_type in ("correct", "substitution") and not i.has_err_tag and i.realized_phone in PHONE_TO_INDEX)
                else -100
                for i in canonical_intervals
            ]
            aux_phone_target_indices = torch.tensor(aux_indices, dtype=torch.long)
            
        return {
            "record_id": f"l2arctic_{speaker_id}_{utt_id}",
            "utterance_id": utt_id,
            "speaker_id": speaker_id,
            "audio_waveform_16k": waveform,
            "canonical_phones_norm39": canonical_phones,
            "canonical_phone_indices": canonical_phone_indices,
            "realized_phones_norm39": chronological_realized_phones,
            "chronological_realized_phones": chronological_realized_phones,
            "binary_labels": binary_labels,
            "ctc_target_indices": ctc_target_indices,
            "diagnosis_target_indices": diagnosis_target_indices,
            "aux_phone_target_indices": aux_phone_target_indices,
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


def load_arctic_prompts(prompts_source: str | Dict[str, str] = "resources/L2-ARCTIC/prompts.json") -> Dict[str, str]:
    """
    Loads CMU ARCTIC written prompts mapping utterance_id -> prompt_text.
    Supports JSON prompt mapping, raw PROMPTS text file, or master ZIP archive.
    """
    if isinstance(prompts_source, dict):
        return dict(prompts_source)
    
    p = Path(prompts_source)
    if not p.is_file():
        raise FileNotFoundError(f"Prompts source not found: {p}")
        
    if p.suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
            
    if p.suffix == ".zip":
        import zipfile
        with zipfile.ZipFile(p, "r") as zf:
            raw = zf.read("PROMPTS").decode("utf-8")
    else:
        with open(p, "r", encoding="utf-8") as f:
            raw = f.read()
            
    prompts = {}
    for line in raw.splitlines():
        m = re.match(r'^\(\s*(\S+)\s+"([^"]+)"\s*\)$', line.strip())
        if m:
            prompts[m.group(1)] = m.group(2)
    return prompts


class ZeroOracleL2ArcticDataset(Dataset):
    """
    Zero-Oracle Dataset for Protocol Z evaluation.
    Guarantees:
    1. Zero TextGrid access. TextGrids are NEVER opened, read, or referenced.
    2. Model inputs are derived strictly from raw audio waveform and written English prompt text.
    3. Pinned CMUdict G2P frontend produces canonical phone indices.
    4. Target/reference labels are strictly absent (set to None).
    """
    def __init__(
        self,
        split_manifest_path: str,
        fold: int,
        split: str,
        prompts_source: str | Dict[str, str] = "resources/L2-ARCTIC/prompts.json",
        g2p_frontend: Optional[Any] = None,
        target_sr: int = 16000,
        zip_path: str = "datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip",
        max_samples: Optional[int] = None
    ):
        super().__init__()
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        
        self.split_manifest_path = split_manifest_path
        self.fold = fold
        self.split = split
        self.target_sr = target_sr
        self.zip_path = zip_path
        
        # Load prompts
        self.prompts = load_arctic_prompts(prompts_source)
        
        # Load G2P frontend
        if g2p_frontend is None:
            from langfens_prep.g2p import CMUDictG2P
            self.g2p = CMUDictG2P(oov_policy="strict")
        else:
            self.g2p = g2p_frontend
            
        with open(split_manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            
        target_fold = None
        for f_entry in manifest_data["folds"]:
            if f_entry["fold_index"] == fold:
                target_fold = f_entry
                break
                
        if target_fold is None:
            raise KeyError(f"Fold with fold_index={fold} not found in manifest {split_manifest_path}")
            
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
        
        # 1. Load audio ONLY from zip (zero TextGrid read)
        waveform, sr, audio_metrics = load_l2arctic_audio_from_zip(
            zip_path=self.zip_path,
            speaker=speaker_id,
            utterance_id=utt_id,
            target_sr=self.target_sr
        )
        
        # 2. Derive canonical phone sequence from written prompt text via G2P
        if utt_id not in self.prompts:
            raise KeyError(f"Utterance '{utt_id}' not found in prompts dictionary.")
        prompt_text = self.prompts[utt_id]
        canonical_phones = self.g2p.phonemize(prompt_text)
        canonical_phone_indices = torch.tensor(
            [PHONE_TO_CTC_INDEX[p] for p in canonical_phones if p in PHONE_TO_CTC_INDEX],
            dtype=torch.long
        )
        
        # 3. Protocol Z guarantees: zero labels, zero TextGrid intervals
        return {
            "record_id": f"l2arctic_{speaker_id}_{utt_id}",
            "speaker_id": speaker_id,
            "utterance_id": utt_id,
            "prompt_text": prompt_text,
            "audio_waveform_16k": waveform,
            "audio_lengths": torch.tensor(waveform.shape[0], dtype=torch.long),
            "canonical_phones": canonical_phones,
            "canonical_phone_indices": canonical_phone_indices,
            "canonical_phone_lengths": torch.tensor(canonical_phone_indices.shape[0], dtype=torch.long),
            "binary_labels": None,
            "ctc_target_indices": None,
            "is_blinded": True
        }

