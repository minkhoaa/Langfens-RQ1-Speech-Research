import json
import tarfile
from dataclasses import dataclass
from typing import List, Dict, Optional, Set

@dataclass
class SpeechOceanWordRecord:
    text: str
    accuracy: float
    stress: float
    total: float
    phones: List[str]
    phones_accuracy: List[float]

@dataclass
class SpeechOceanUtteranceRecord:
    utterance_id: str
    speaker_id: str
    gender: str
    age: str
    overall_accuracy: float
    completeness: float
    fluency: float
    prosodic: float
    total: float
    words: List[SpeechOceanWordRecord]

def parse_speechocean_manifested(
    archive_path: str,
    split_manifest_path: str,
    limit: Optional[int] = None
) -> Dict[str, SpeechOceanUtteranceRecord]:
    """
    Parses official SpeechOcean762 score records strictly for manifested utterances,
    guaranteeing that unmanifested recordings are completely excluded from supervised samples.
    """
    with open(split_manifest_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)
        
    allowed_utts: Set[str] = set()
    for part in ["train", "val", "test"]:
        allowed_utts.update(split_data["partitions"][part]["utterances"])
        
    records: Dict[str, SpeechOceanUtteranceRecord] = {}
    
    with tarfile.open(archive_path, "r:gz") as tf:
        # Load scores
        scores_file = tf.extractfile("speechocean762/resource/scores.json")
        if scores_file is None:
            raise FileNotFoundError("speechocean762/resource/scores.json not found in archive")
        scores = json.loads(scores_file.read().decode("utf-8"))
        
        # Load utt2spk, spk2gender, spk2age
        utt2spk: Dict[str, str] = {}
        for split in ["train", "test"]:
            member = tf.extractfile(f"speechocean762/{split}/utt2spk")
            if member:
                for line in member.read().decode("utf-8").splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        utt2spk[parts[0]] = parts[1]
                        
        spk2gender: Dict[str, str] = {}
        for split in ["train", "test"]:
            member = tf.extractfile(f"speechocean762/{split}/spk2gender")
            if member:
                for line in member.read().decode("utf-8").splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        spk2gender[parts[0]] = parts[1]
                        
        spk2age: Dict[str, str] = {}
        for split in ["train", "test"]:
            member = tf.extractfile(f"speechocean762/{split}/spk2age")
            if member:
                for line in member.read().decode("utf-8").splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        spk2age[parts[0]] = parts[1]

        count = 0
        for utt_id, data in scores.items():
            if utt_id not in allowed_utts:
                continue
                
            speaker_id = utt2spk.get(utt_id, "unknown")
            gender = spk2gender.get(speaker_id, "unknown")
            age = spk2age.get(speaker_id, "unknown")
            
            words_list: List[SpeechOceanWordRecord] = []
            for w in data.get("words", []):
                words_list.append(SpeechOceanWordRecord(
                    text=w.get("text", ""),
                    accuracy=float(w.get("accuracy", 0.0)),
                    stress=float(w.get("stress", 0.0)),
                    total=float(w.get("total", 0.0)),
                    phones=w.get("phones", []),
                    phones_accuracy=[float(p) for p in w.get("phones-accuracy", [])]
                ))
                
            rec = SpeechOceanUtteranceRecord(
                utterance_id=utt_id,
                speaker_id=speaker_id,
                gender=gender,
                age=age,
                overall_accuracy=float(data.get("accuracy", 0.0)),
                completeness=float(data.get("completeness", 0.0)),
                fluency=float(data.get("fluency", 0.0)),
                prosodic=float(data.get("prosodic", 0.0)),
                total=float(data.get("total", 0.0)),
                words=words_list
            )
            records[utt_id] = rec
            count += 1
            if limit is not None and count >= limit:
                break
                
    return records
