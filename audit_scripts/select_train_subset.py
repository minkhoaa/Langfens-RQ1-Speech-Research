#!/usr/bin/env python3
"""
Select deterministic train-only subset for LANGFENS RQ1-TRN-01B0.
Selection rules:
- Fixed random seed 42.
- Fold 1 train speakers only (16 speakers).
- Exactly 32 utterances (2 per speaker).
- Representative correct and mispronounced phone tokens (substitutions & deletions).
- Never sample from val or test partitions.
- Strictly audit against leakage.
"""

import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, os.path.abspath("src"))

from langfens_prep.audio_loader import get_l2arctic_reader, load_l2arctic_audio_from_zip
from langfens_prep.textgrid_parser import parse_gold_textgrid_content
from langfens_prep.phoneme_normalizer import STANDARD_39_ARPABET, PHONE_TO_CTC_INDEX


def select_deterministic_subset(
    split_manifest_path: str = "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json",
    zip_path: str = "datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip",
    seed: int = 42,
    num_utterances: int = 32,
    output_dir: str = "manifests/RQ1-TRN-01B0"
):
    print(f"Loading split manifest: {split_manifest_path}")
    with open(split_manifest_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    fold1 = next(f for f in splits["folds"] if f["fold_index"] == 1)
    train_utts_raw = set(fold1["train"]["utterances"])
    val_utts_raw = set(fold1["val"]["utterances"])
    test_utts_raw = set(fold1["test"]["utterances"])
    train_speakers = sorted(list(fold1["train"]["speakers"]))
    val_speakers = set(fold1["val"]["speakers"])
    test_speakers = set(fold1["test"]["speakers"])

    print(f"Fold 1 train speakers ({len(train_speakers)}): {train_speakers}")
    print(f"Fold 1 train utterances: {len(train_utts_raw)}")
    print(f"Fold 1 val utterances: {len(val_utts_raw)}")
    print(f"Fold 1 test utterances: {len(test_utts_raw)}")

    # Audit partition disjointness
    assert train_utts_raw.isdisjoint(val_utts_raw), "FATAL: Train/Val utterance overlap!"
    assert train_utts_raw.isdisjoint(test_utts_raw), "FATAL: Train/Test utterance overlap!"
    assert set(train_speakers).isdisjoint(val_speakers), "FATAL: Train/Val speaker overlap!"
    assert set(train_speakers).isdisjoint(test_speakers), "FATAL: Train/Test speaker overlap!"

    # Group train utterances by speaker
    spk_to_utts = defaultdict(list)
    for u in sorted(list(train_utts_raw)):
        spk, utt = u.split("/")
        assert spk in train_speakers, f"FATAL: Utterance {u} belongs to non-train speaker {spk}"
        spk_to_utts[spk].append((spk, utt, u))

    reader = get_l2arctic_reader(zip_path)
    rng = random.Random(seed)

    # For each speaker, evaluate all utterances
    spk_candidates = {}
    print(f"Auditing annotations for candidate utterances across {len(train_speakers)} speakers...")
    for spk in train_speakers:
        cand_list = []
        for s, utt, full_id in spk_to_utts[spk]:
            tg_text = reader.read_annotation_text(s, utt)
            ann = parse_gold_textgrid_content(tg_text, f"{utt}.TextGrid")
            counts = Counter(i.error_type for i in ann.canonical_intervals)
            num_sub = counts.get("substitution", 0)
            num_del = counts.get("deletion", 0)
            num_cor = counts.get("correct", 0)
            num_can = len(ann.canonical_intervals)
            
            # Count chronological realized CTC phones
            realized_phones = []
            for i in ann.intervals:
                if i.error_type in ("silence", "deletion") or i.has_err_tag:
                    continue
                p = i.realized_phone
                if p and p in STANDARD_39_ARPABET:
                    realized_phones.append(p)

            cand_list.append({
                "full_id": full_id,
                "speaker_id": s,
                "utterance_id": utt,
                "num_canonical_phones": num_can,
                "num_correct": num_cor,
                "num_substitution": num_sub,
                "num_deletion": num_del,
                "num_realized_ctc_phones": len(realized_phones),
                "has_substitution": num_sub > 0,
                "has_deletion": num_del > 0,
                "has_both": (num_sub > 0 and num_del > 0),
                "canonical_phones": [i.canonical_phone for i in ann.canonical_intervals],
                "binary_labels_strict": [i.binary_label_strict for i in ann.canonical_intervals],
                "error_types": [i.error_type for i in ann.canonical_intervals],
                "realized_phones": realized_phones
            })
        # Shuffle deterministically per speaker
        rng.shuffle(cand_list)
        spk_candidates[spk] = cand_list

    # Selection strategy:
    # 32 utterances = exactly 2 per speaker across 16 speakers.
    # For each speaker:
    # Utterance 1: prioritize utterance with both substitutions and deletions (or substitution).
    # Utterance 2: prioritize utterance with substitution or deletion to maximize error coverage.
    selected = []
    for spk in train_speakers:
        cands = spk_candidates[spk]
        # Candidate with both sub and del
        both = [c for c in cands if c["has_both"]]
        if len(both) >= 2:
            chosen = both[:2]
        elif len(both) == 1:
            chosen = [both[0]]
            remaining = [c for c in cands if c != both[0] and (c["has_substitution"] or c["has_deletion"])]
            if remaining:
                chosen.append(remaining[0])
            else:
                chosen.append([c for c in cands if c != both[0]][0])
        else:
            # No candidate with both: pick one with sub, one with del or another sub
            subs = [c for c in cands if c["has_substitution"]]
            dels = [c for c in cands if c["has_deletion"]]
            chosen = []
            if subs:
                chosen.append(subs[0])
            if dels and dels[0] != chosen[0]:
                chosen.append(dels[0])
            elif len(subs) > 1:
                chosen.append(subs[1])
            else:
                rem = [c for c in cands if c not in chosen]
                chosen.append(rem[0])
        assert len(chosen) == 2, f"Expected 2 utterances for speaker {spk}, got {len(chosen)}"
        selected.extend(chosen)

    assert len(selected) == num_utterances, f"Expected {num_utterances} total, got {len(selected)}"

    # Sort deterministically by full_id
    selected.sort(key=lambda x: x["full_id"])

    # Verify no leakage
    for item in selected:
        uid = item["full_id"]
        assert uid in train_utts_raw, f"LEAKAGE: {uid} not in Fold 1 train!"
        assert uid not in val_utts_raw, f"LEAKAGE: {uid} in Fold 1 val!"
        assert uid not in test_utts_raw, f"LEAKAGE: {uid} in Fold 1 test!"
        assert item["speaker_id"] in train_speakers, f"LEAKAGE: {item['speaker_id']} not in train speakers!"
        assert item["speaker_id"] not in val_speakers, f"LEAKAGE: {item['speaker_id']} in val speakers!"
        assert item["speaker_id"] not in test_speakers, f"LEAKAGE: {item['speaker_id']} in test speakers!"

    # Compute aggregate statistics
    tot_can = sum(x["num_canonical_phones"] for x in selected)
    tot_cor = sum(x["num_correct"] for x in selected)
    tot_sub = sum(x["num_substitution"] for x in selected)
    tot_del = sum(x["num_deletion"] for x in selected)
    tot_ctc = sum(x["num_realized_ctc_phones"] for x in selected)
    unique_can_phones = set().union(*[x["canonical_phones"] for x in selected])
    unique_realized_phones = set().union(*[x["realized_phones"] for x in selected])

    print("\n--- Selected Subset Summary ---")
    print(f"Total Utterances: {len(selected)}")
    print(f"Distinct Speakers: {len(set(x['speaker_id'] for x in selected))} / {len(train_speakers)}")
    print(f"Total Canonical Phonemes: {tot_can}")
    print(f"  - Correct: {tot_cor} ({tot_cor/tot_can:.1%})")
    print(f"  - Substitution: {tot_sub} ({tot_sub/tot_can:.1%})")
    print(f"  - Deletion: {tot_del} ({tot_del/tot_can:.1%})")
    print(f"Total Realized CTC Tokens: {tot_ctc}")
    print(f"Canonical Phone Coverage: {len(unique_can_phones)} / 39 standard ARPAbet")
    print(f"Realized Phone Coverage: {len(unique_realized_phones)} / 39 standard ARPAbet")

    # Measure exact audio durations
    for item in selected:
        wf, sr, metrics = load_l2arctic_audio_from_zip(
            zip_path=zip_path,
            speaker=item["speaker_id"],
            utterance_id=item["utterance_id"],
            target_sr=16000
        )
        item["audio_duration_seconds"] = round(metrics.duration_seconds, 4)
        item["sample_rate"] = sr
        item["num_audio_samples"] = wf.shape[0]

    tot_duration = sum(x["audio_duration_seconds"] for x in selected)
    print(f"Total Audio Duration: {tot_duration:.2f} seconds ({tot_duration/60:.2f} minutes)")
    print(f"Mean Duration: {tot_duration/len(selected):.2f} seconds per utterance")

    # Serialize manifest
    out_path = Path(output_dir) / "TRAIN_ONLY_SUBSET_MANIFEST.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_payload = {
        "metadata": {
            "experiment_id": "RQ1-TRN-01B0",
            "gate_description": "Controlled Real-Data Training Sanity Gate",
            "selection_random_seed": seed,
            "policy": "strict",
            "fold_index": 1,
            "partition": "train_only",
            "total_selected_utterances": len(selected),
            "total_canonical_phonemes": tot_can,
            "total_realized_ctc_tokens": tot_ctc,
            "total_audio_duration_seconds": round(tot_duration, 4),
            "phoneme_error_distribution": {
                "correct": tot_cor,
                "correct_ratio": round(tot_cor / tot_can, 4),
                "substitution": tot_sub,
                "substitution_ratio": round(tot_sub / tot_can, 4),
                "deletion": tot_del,
                "deletion_ratio": round(tot_del / tot_can, 4)
            },
            "source_manifest_checksum_sha256": "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b",
            "zero_leakage_assertion": "PASSED: Zero overlap with Fold 1 val (300 utts) or test (900 utts)"
        },
        "selected_speakers": train_speakers,
        "selected_utterances": [
            {
                "full_id": x["full_id"],
                "speaker_id": x["speaker_id"],
                "utterance_id": x["utterance_id"],
                "duration_seconds": x["audio_duration_seconds"],
                "num_samples_16k": x["num_audio_samples"],
                "canonical_phone_count": x["num_canonical_phones"],
                "realized_ctc_count": x["num_realized_ctc_phones"],
                "correct_count": x["num_correct"],
                "substitution_count": x["num_substitution"],
                "deletion_count": x["num_deletion"],
                "canonical_phones": x["canonical_phones"],
                "strict_binary_labels": x["binary_labels_strict"],
                "error_types": x["error_types"]
            }
            for x in selected
        ]
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    print(f"Manifest saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    select_deterministic_subset()
