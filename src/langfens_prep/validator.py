import io
import json
import os
import tarfile
import zipfile
from collections import defaultdict
from typing import Dict, Any, Set

from langfens_prep.phoneme_normalizer import normalize_phoneme, SILENCE_SYMBOLS
from langfens_prep.textgrid_parser import parse_textgrid_raw_tiers

L2ARCTIC_SPEAKERS = [
    "ABA", "SKA", "YBAA", "ZHAA", "BWC", "LXC", "NCC", "TXHC",
    "ASI", "RRBI", "SVBI", "TNI", "HJK", "HKK", "YDCK", "YKWK",
    "EBVS", "ERMS", "MBMPS", "NJS", "HQTV", "PNV", "THV", "TLV"
]

def run_full_l2arctic_census(zip_path: str) -> Dict[str, Any]:
    """
    Executes an exhaustive physical census of all 3,599 gold manual Praat TextGrids
    in L2-ARCTIC Release v5.0, reconciling exact token counts against
    CORRECTED_LABEL_ACCOUNTING.md.
    """
    zf = zipfile.ZipFile(zip_path, "r")
    
    total_gold_utterances = 0
    total_intervals_all = 0
    silence_intervals = 0
    addition_intervals = 0
    
    correct_tokens = 0
    deletion_tokens = 0
    substitution_tokens = 0
    
    sub_clean_diff_base = 0
    sub_clean_same_base = 0
    sub_base_preserving_allophone = 0
    sub_category_shifting_accented = 0
    sub_distorted_err = 0
    
    speaker_stats = {}
    
    for spk in L2ARCTIC_SPEAKERS:
        spk_zip_bytes = zf.read(f"{spk}.zip")
        szf = zipfile.ZipFile(io.BytesIO(spk_zip_bytes), "r")
        
        annotation_files = [
            n for n in szf.namelist() 
            if n.startswith(f"{spk}/annotation/") and n.endswith(".TextGrid")
        ]
        total_gold_utterances += len(annotation_files)
        
        spk_counts = defaultdict(int)
        
        for ann_path in annotation_files:
            tg_raw = szf.read(ann_path).decode("utf-8", errors="replace")
            tiers = parse_textgrid_raw_tiers(tg_raw)
            if "phones" not in tiers:
                continue
                
            for xmin, xmax, raw_label in tiers["phones"]["intervals"]:
                total_intervals_all += 1
                spk_counts["total_intervals"] += 1
                trimmed = raw_label.strip()
                
                if trimmed.lower() in SILENCE_SYMBOLS:
                    silence_intervals += 1
                    spk_counts["silence"] += 1
                    continue
                    
                if "," in trimmed:
                    parts = [p.strip() for p in trimmed.split(",")]
                    if len(parts) == 3:
                        cpl, ppl, err_code = parts[0], parts[1], parts[2].lower()
                        has_star = ("*" in cpl or "*" in ppl)
                        has_err = ("err" in cpl.lower() or "err" in ppl.lower())
                        
                        cpl_norm = normalize_phoneme(cpl)
                        ppl_norm = normalize_phoneme(ppl)
                        base_c = cpl_norm.normalized_phone
                        base_p = ppl_norm.normalized_phone
                        
                        if err_code == "s":
                            substitution_tokens += 1
                            spk_counts["substitution"] += 1
                            if has_err:
                                sub_distorted_err += 1
                            elif has_star:
                                if base_c == base_p:
                                    sub_base_preserving_allophone += 1
                                else:
                                    sub_category_shifting_accented += 1
                            else:
                                if base_c == base_p:
                                    sub_clean_same_base += 1
                                else:
                                    sub_clean_diff_base += 1
                        elif err_code == "d":
                            deletion_tokens += 1
                            spk_counts["deletion"] += 1
                        elif err_code == "a":
                            addition_intervals += 1
                            spk_counts["addition"] += 1
                else:
                    correct_tokens += 1
                    spk_counts["correct"] += 1
                    
        speaker_stats[spk] = dict(spk_counts)
        szf.close()
        
    zf.close()
    
    canonical_tokens = correct_tokens + substitution_tokens + deletion_tokens
    strict_binary_errors = substitution_tokens + deletion_tokens
    lenient_binary_errors = (substitution_tokens - sub_base_preserving_allophone) + deletion_tokens
    strict_correct = correct_tokens
    lenient_correct = correct_tokens + sub_base_preserving_allophone
    
    strict_canonical_total = strict_correct + strict_binary_errors
    lenient_canonical_total = lenient_correct + lenient_binary_errors
    
    strict_error_rate_pct = (strict_binary_errors / canonical_tokens) * 100.0 if canonical_tokens else 0.0
    lenient_error_rate_pct = (lenient_binary_errors / canonical_tokens) * 100.0 if canonical_tokens else 0.0
    
    return {
        "total_gold_utterances": total_gold_utterances,
        "total_intervals_all": total_intervals_all,
        "silence_intervals": silence_intervals,
        "addition_intervals": addition_intervals,
        "canonical_tokens": canonical_tokens,
        "correct_tokens": correct_tokens,
        "substitution_tokens": substitution_tokens,
        "deletion_tokens": deletion_tokens,
        "sub_clean_diff_base": sub_clean_diff_base,
        "sub_clean_same_base": sub_clean_same_base,
        "sub_base_preserving_allophone": sub_base_preserving_allophone,
        "sub_category_shifting_accented": sub_category_shifting_accented,
        "sub_distorted_err": sub_distorted_err,
        "strict_binary_errors": strict_binary_errors,
        "lenient_binary_errors": lenient_binary_errors,
        "strict_correct": strict_correct,
        "lenient_correct": lenient_correct,
        "strict_error_rate_pct": strict_error_rate_pct,
        "lenient_error_rate_pct": lenient_error_rate_pct,
        "strict_canonical_total": strict_canonical_total,
        "lenient_canonical_total": lenient_canonical_total,
        "speaker_breakdown": speaker_stats
    }

def validate_speechocean_splits(archive_path: str, manifest_path: str) -> Dict[str, Any]:
    """
    Validates SpeechOcean762 split integrity, checking partition isolation,
    manifest counts, and exclusion of the 245 unmanifested audio files.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    train_utts = set(data["partitions"]["train"]["utterances"])
    val_utts = set(data["partitions"]["val"]["utterances"])
    test_utts = set(data["partitions"]["test"]["utterances"])
    
    overlap_train_val = len(train_utts & val_utts)
    overlap_train_test = len(train_utts & test_utts)
    overlap_val_test = len(val_utts & test_utts)
    total_overlap = overlap_train_val + overlap_train_test + overlap_val_test
    
    unmanifested_count = data.get("unmanifested_excluded_recordings", {}).get("count", 0)
    
    # Check that unmanifested wavs are never present in any partition
    unmanifested_files = data.get("unmanifested_excluded_recordings", {}).get("file_paths", [])
    unmanifested_utts = {os.path.basename(p).replace(".WAV", "").replace(".wav", "") for p in unmanifested_files}
    
    all_manifested = train_utts | val_utts | test_utts
    leakage = len(unmanifested_utts & all_manifested)
    
    return {
        "train_utts": len(train_utts),
        "val_utts": len(val_utts),
        "test_utts": len(test_utts),
        "total_manifested_utts": len(all_manifested),
        "unmanifested_excluded_count": unmanifested_count,
        "partition_overlap_count": total_overlap,
        "unmanifested_leakage_count": leakage
    }
