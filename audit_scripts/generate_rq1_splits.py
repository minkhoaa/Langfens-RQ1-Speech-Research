#!/usr/bin/env python3
"""
LANGFENS — RQ1-SPLIT-01: Deterministic Speaker-Independent Split Manifest Generator
Engineering Lead: Senior Speech ML Research Engineer & Research Data Engineer
Target Corpora:
  1. L2-ARCTIC Release v5.0 (24 Speakers across 6 L1s; 3,599 Gold Annotations)
  2. SpeechOcean762 (250 Speakers across Train/Test; 5,000 Scored Utterances)
"""

import os
import sys
import io
import json
import hashlib
import zipfile
import tarfile
import random
from datetime import datetime, timezone
from collections import defaultdict, Counter

# Ensure local imports work
sys.path.insert(0, os.path.dirname(__file__))
from textgrid_parser import parse_textgrid

# Fixed configuration
MASTER_SEED = 42
SPLIT_PROTOCOL_VERSION = "RQ1-SPLIT-01"
DEFAULT_TIMESTAMP_UTC = "2026-10-23T13:30:00Z"
L2ARCTIC_ARCHIVE_PATH = "datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip"
SPEECHOCEAN_ARCHIVE_PATH = "datasets/SpeechOcean762/original/speechocean762.tar.gz"
OUTPUT_DIR = "manifests/RQ1-SPLIT-01"

L2ARCTIC_ARCHIVE_SHA256 = "490d5a43e48b0af84f6bce8e766ad453e8cd670521d5876bb4d0741cca304fda"
SPEECHOCEAN_ARCHIVE_SHA256 = "876c34b828c2ebcc35ba2ff1ba5947d50e6b0162fc4a4af91db0942422b3bdd5"

L2ARCTIC_LICENSE = "CC BY-NC 4.0"
SPEECHOCEAN_LICENSE = "CC BY 4.0"

# L2-ARCTIC Speaker Metadata
L2ARCTIC_SPEAKERS = {
    "ABA":   {"gender": "M", "l1": "Arabic"},
    "SKA":   {"gender": "F", "l1": "Arabic"},
    "YBAA":  {"gender": "M", "l1": "Arabic"},
    "ZHAA":  {"gender": "F", "l1": "Arabic"},
    "BWC":   {"gender": "M", "l1": "Chinese"},
    "LXC":   {"gender": "F", "l1": "Chinese"},
    "NCC":   {"gender": "F", "l1": "Chinese"},
    "TXHC":  {"gender": "M", "l1": "Chinese"},
    "ASI":   {"gender": "M", "l1": "Hindi"},
    "RRBI":  {"gender": "M", "l1": "Hindi"},
    "SVBI":  {"gender": "F", "l1": "Hindi"},
    "TNI":   {"gender": "F", "l1": "Hindi"},
    "HJK":   {"gender": "F", "l1": "Korean"},
    "HKK":   {"gender": "M", "l1": "Korean"},
    "YDCK":  {"gender": "F", "l1": "Korean"},
    "YKWK":  {"gender": "M", "l1": "Korean"},
    "EBVS":  {"gender": "M", "l1": "Spanish"},
    "ERMS":  {"gender": "M", "l1": "Spanish"},
    "MBMPS": {"gender": "F", "l1": "Spanish"},
    "NJS":   {"gender": "F", "l1": "Spanish"},
    "HQTV":  {"gender": "M", "l1": "Vietnamese"},
    "PNV":   {"gender": "F", "l1": "Vietnamese"},
    "THV":   {"gender": "F", "l1": "Vietnamese"},
    "TLV":   {"gender": "M", "l1": "Vietnamese"},
}

def compute_sha256(data_bytes: bytes) -> str:
    h = hashlib.sha256()
    h.update(data_bytes)
    return h.hexdigest()

def compute_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def parse_l2arctic_corpus():
    """Extracts all gold annotations and calculates canonical tokens and errors per utterance."""
    print("-> Parsing L2-ARCTIC v5.0 gold annotations from raw archive...")
    zf = zipfile.ZipFile(L2ARCTIC_ARCHIVE_PATH, "r")
    spk_zips = [n for n in zf.namelist() if n.endswith(".zip") and n != "suitcase_corpus.zip"]
    
    utterance_db = {}
    speaker_utts = defaultdict(list)
    speaker_stats = defaultdict(lambda: {"N": 0, "C": 0, "S": 0, "D": 0, "E": 0, "A": 0, "sil": 0, "utts": 0})
    
    for sz_name in sorted(spk_zips):
        spk = sz_name.replace(".zip", "")
        with zipfile.ZipFile(io.BytesIO(zf.read(sz_name))) as sz:
            ann_names = [n for n in sz.namelist() if n.startswith(f"{spk}/annotation/") and n.endswith(".TextGrid")]
            for an in sorted(ann_names):
                utt_id = an.split("/")[-1].replace(".TextGrid", "")
                full_utt_id = f"{spk}/{utt_id}"
                
                tg_text = sz.read(an).decode("utf-8", errors="replace")
                tiers = parse_textgrid(tg_text)
                
                c_cnt, s_cnt, d_cnt, a_cnt, sil_cnt = 0, 0, 0, 0, 0
                for xmin, xmax, text in tiers["phones"]["intervals"]:
                    t = text.strip()
                    if not t or t in ["sil", "sp", "spn"]:
                        sil_cnt += 1
                    elif "," in t:
                        parts = t.split(",")
                        if len(parts) == 3:
                            err = parts[2].strip().lower()
                            if err == "s":
                                s_cnt += 1
                            elif err == "d":
                                d_cnt += 1
                            elif err == "a":
                                a_cnt += 1
                    else:
                        c_cnt += 1
                
                n_canon = c_cnt + s_cnt + d_cnt
                e_canon = s_cnt + d_cnt
                
                utterance_db[full_utt_id] = {
                    "speaker_id": spk,
                    "utterance_id": utt_id,
                    "full_utterance_id": full_utt_id,
                    "canonical_tokens": n_canon,
                    "correct_tokens": c_cnt,
                    "substitution_tokens": s_cnt,
                    "deletion_tokens": d_cnt,
                    "error_tokens": e_canon,
                    "addition_tokens": a_cnt,
                    "silence_intervals": sil_cnt
                }
                speaker_utts[spk].append(full_utt_id)
                
                s_stat = speaker_stats[spk]
                s_stat["utts"] += 1
                s_stat["N"] += n_canon
                s_stat["C"] += c_cnt
                s_stat["S"] += s_cnt
                s_stat["D"] += d_cnt
                s_stat["E"] += e_canon
                s_stat["A"] += a_cnt
                s_stat["sil"] += sil_cnt
                
    total_utts = len(utterance_db)
    total_canon_tokens = sum(s["N"] for s in speaker_stats.values())
    total_canon_errors = sum(s["E"] for s in speaker_stats.values())
    print(f"   Parsed {len(speaker_stats)} speakers, {total_utts} utterances, {total_canon_tokens} canonical tokens, {total_canon_errors} errors.")
    
    assert total_utts == 3599, f"Expected 3,599 utterances, got {total_utts}"
    assert total_canon_tokens == 119151, f"Expected 119,151 tokens, got {total_canon_tokens}"
    assert total_canon_errors == 17518, f"Expected 17,518 errors, got {total_canon_errors}"
    
    return utterance_db, speaker_utts, speaker_stats

def generate_l2arctic_4fold(utterance_db, speaker_utts, speaker_stats, timestamp_utc):
    """Generates the validated 4-fold cross-validation split manifest."""
    print("-> Generating L2-ARCTIC 4-Fold CV manifest...")
    
    # Validated allocation from VALIDATED_SPLIT_PROTOCOL_ADDENDUM.md
    fold_defs = [
        {
            "fold_index": 1,
            "test_speakers": ["ABA", "LXC", "ASI", "HJK", "EBVS", "PNV"],
            "val_speakers": ["RRBI", "NJS"]
        },
        {
            "fold_index": 2,
            "test_speakers": ["SKA", "BWC", "SVBI", "HKK", "MBMPS", "HQTV"],
            "val_speakers": ["YBAA", "TNI"]
        },
        {
            "fold_index": 3,
            "test_speakers": ["YBAA", "NCC", "RRBI", "YDCK", "ERMS", "THV"],
            "val_speakers": ["TXHC", "ZHAA"]
        },
        {
            "fold_index": 4,
            "test_speakers": ["ZHAA", "TXHC", "TNI", "YKWK", "NJS", "TLV"],
            "val_speakers": ["HKK", "NCC"]
        }
    ]
    
    all_speakers = sorted(list(L2ARCTIC_SPEAKERS.keys()))
    tested_speakers_union = set()
    folds_output = []
    
    for f in fold_defs:
        k = f["fold_index"]
        test_spks = sorted(f["test_speakers"])
        val_spks = sorted(f["val_speakers"])
        train_spks = sorted([s for s in all_speakers if s not in test_spks and s not in val_spks])
        
        # Validations
        assert len(test_spks) == 6, f"Fold {k} test spks count != 6"
        assert len(val_spks) == 2, f"Fold {k} val spks count != 2"
        assert len(train_spks) == 16, f"Fold {k} train spks count != 16"
        
        # Pairwise disjoint
        assert set(test_spks).isdisjoint(set(val_spks)), f"Fold {k} test-val leak"
        assert set(test_spks).isdisjoint(set(train_spks)), f"Fold {k} test-train leak"
        assert set(val_spks).isdisjoint(set(train_spks)), f"Fold {k} val-train leak"
        assert set(test_spks) | set(val_spks) | set(train_spks) == set(all_speakers), f"Fold {k} incomplete coverage"
        
        # Demographic checks on test
        test_genders = Counter(L2ARCTIC_SPEAKERS[s]["gender"] for s in test_spks)
        test_l1s = Counter(L2ARCTIC_SPEAKERS[s]["l1"] for s in test_spks)
        assert test_genders == {"M": 3, "F": 3}, f"Fold {k} gender skew: {test_genders}"
        assert len(test_l1s) == 6 and all(v == 1 for v in test_l1s.values()), f"Fold {k} L1 imbalance: {test_l1s}"
        
        # Demographic checks on val
        val_genders = Counter(L2ARCTIC_SPEAKERS[s]["gender"] for s in val_spks)
        assert val_genders == {"M": 1, "F": 1}, f"Fold {k} val gender skew: {val_genders}"
        
        # Demographic checks on train
        train_genders = Counter(L2ARCTIC_SPEAKERS[s]["gender"] for s in train_spks)
        assert train_genders == {"M": 8, "F": 8}, f"Fold {k} train gender skew: {train_genders}"
        
        # Accumulate union
        tested_speakers_union.update(test_spks)
        
        def aggregate_partition(spk_list):
            utts = []
            for s in spk_list:
                utts.extend(speaker_utts[s])
            utts.sort()
            
            n_tokens = sum(utterance_db[u]["canonical_tokens"] for u in utts)
            c_tokens = sum(utterance_db[u]["correct_tokens"] for u in utts)
            s_tokens = sum(utterance_db[u]["substitution_tokens"] for u in utts)
            d_tokens = sum(utterance_db[u]["deletion_tokens"] for u in utts)
            e_tokens = sum(utterance_db[u]["error_tokens"] for u in utts)
            err_rate = round((e_tokens / n_tokens) * 100, 4) if n_tokens else 0.0
            
            gender_dist = Counter(L2ARCTIC_SPEAKERS[s]["gender"] for s in spk_list)
            l1_dist = Counter(L2ARCTIC_SPEAKERS[s]["l1"] for s in spk_list)
            
            return {
                "speaker_count": len(spk_list),
                "utterance_count": len(utts),
                "canonical_tokens": n_tokens,
                "correct_tokens": c_tokens,
                "substitution_tokens": s_tokens,
                "deletion_tokens": d_tokens,
                "error_tokens": e_tokens,
                "error_rate_pct": err_rate,
                "gender_distribution": dict(gender_dist),
                "l1_distribution": dict(l1_dist),
                "speakers": spk_list,
                "utterances": utts
            }
        
        train_part = aggregate_partition(train_spks)
        val_part = aggregate_partition(val_spks)
        test_part = aggregate_partition(test_spks)
        
        # Check conservation
        assert train_part["utterance_count"] + val_part["utterance_count"] + test_part["utterance_count"] == 3599
        assert train_part["canonical_tokens"] + val_part["canonical_tokens"] + test_part["canonical_tokens"] == 119151
        assert train_part["error_tokens"] + val_part["error_tokens"] + test_part["error_tokens"] == 17518
        
        folds_output.append({
            "split_id": f"l2arctic_k4_fold{k}",
            "fold_index": k,
            "train": train_part,
            "val": val_part,
            "test": test_part
        })
        
    assert len(tested_speakers_union) == 24, "Not all 24 speakers were tested across 4 folds"
    
    manifest = {
        "dataset_name": "L2-ARCTIC",
        "dataset_version": "Release v5.0 (March 2020)",
        "archive_filename": os.path.basename(L2ARCTIC_ARCHIVE_PATH),
        "archive_sha256": L2ARCTIC_ARCHIVE_SHA256,
        "license": L2ARCTIC_LICENSE,
        "license_source": "https://psi.engr.tamu.edu/l2-arctic-corpus/",
        "split_protocol": "Balanced 4-Fold Speaker-Independent Cross-Validation",
        "split_protocol_version": SPLIT_PROTOCOL_VERSION,
        "master_seed": MASTER_SEED,
        "generation_algorithm": "Validated Stratified Disjoint 4-Fold Combinatorial Allocation",
        "generation_timestamp_utc": timestamp_utc,
        "corpus_totals": {
            "speakers": 24,
            "utterances": 3599,
            "canonical_tokens": 119151,
            "canonical_errors": 17518,
            "mean_error_rate_pct": 14.7024
        },
        "folds": folds_output
    }
    
    # Calculate checksum of the payload
    payload_bytes = json.dumps(folds_output, sort_keys=True).encode("utf-8")
    manifest["payload_sha256"] = compute_sha256(payload_bytes)
    
    return manifest

def generate_l2arctic_vietnamese_splits(utterance_db, speaker_utts, speaker_stats, timestamp_utc):
    """Generates Regime A (Cross-Vietnamese LOSO, 4 rounds) and Regime B (Zero-Shot Transfer)."""
    print("-> Generating L2-ARCTIC Vietnamese Evaluation manifest (Regimes A & B)...")
    
    all_speakers = sorted(list(L2ARCTIC_SPEAKERS.keys()))
    viet_speakers = ["HQTV", "PNV", "THV", "TLV"]
    val_speakers = ["RRBI", "NJS"] # RRBI (M, Hindi), NJS (F, Spanish)
    base_non_viet_train = sorted([s for s in all_speakers if s not in viet_speakers and s not in val_speakers])
    
    assert len(base_non_viet_train) == 18, f"Expected 18 base non-Viet train speakers, got {len(base_non_viet_train)}"
    assert len(val_speakers) == 2
    assert len(viet_speakers) == 4
    
    def aggregate_partition(spk_list):
        utts = []
        for s in spk_list:
            utts.extend(speaker_utts[s])
        utts.sort()
        
        n_tokens = sum(utterance_db[u]["canonical_tokens"] for u in utts)
        c_tokens = sum(utterance_db[u]["correct_tokens"] for u in utts)
        s_tokens = sum(utterance_db[u]["substitution_tokens"] for u in utts)
        d_tokens = sum(utterance_db[u]["deletion_tokens"] for u in utts)
        e_tokens = sum(utterance_db[u]["error_tokens"] for u in utts)
        err_rate = round((e_tokens / n_tokens) * 100, 4) if n_tokens else 0.0
        
        gender_dist = Counter(L2ARCTIC_SPEAKERS[s]["gender"] for s in spk_list)
        l1_dist = Counter(L2ARCTIC_SPEAKERS[s]["l1"] for s in spk_list)
        
        return {
            "speaker_count": len(spk_list),
            "utterance_count": len(utts),
            "canonical_tokens": n_tokens,
            "correct_tokens": c_tokens,
            "substitution_tokens": s_tokens,
            "deletion_tokens": d_tokens,
            "error_tokens": e_tokens,
            "error_rate_pct": err_rate,
            "gender_distribution": dict(gender_dist),
            "l1_distribution": dict(l1_dist),
            "speakers": spk_list,
            "utterances": utts
        }
        
    val_part = aggregate_partition(val_speakers)
    
    # Regime A: 4 rounds holding out one Vietnamese speaker
    regime_a_rounds = []
    for r_idx, held_out in enumerate(viet_speakers, start=1):
        other_viets = [s for s in viet_speakers if s != held_out]
        round_train_spks = sorted(base_non_viet_train + other_viets)
        round_test_spks = [held_out]
        
        assert len(round_train_spks) == 21, f"Regime A Round {r_idx} train != 21"
        assert len(round_test_spks) == 1, f"Regime A Round {r_idx} test != 1"
        assert set(round_train_spks).isdisjoint(set(val_speakers))
        assert set(round_train_spks).isdisjoint(set(round_test_spks))
        assert set(val_speakers).isdisjoint(set(round_test_spks))
        assert set(round_train_spks) | set(val_speakers) | set(round_test_spks) == set(all_speakers)
        
        train_part = aggregate_partition(round_train_spks)
        test_part = aggregate_partition(round_test_spks)
        
        assert train_part["utterance_count"] + val_part["utterance_count"] + test_part["utterance_count"] == 3599
        assert train_part["canonical_tokens"] + val_part["canonical_tokens"] + test_part["canonical_tokens"] == 119151
        assert train_part["error_tokens"] + val_part["error_tokens"] + test_part["error_tokens"] == 17518
        
        regime_a_rounds.append({
            "round_id": f"vietnamese_loso_round{r_idx}",
            "round_index": r_idx,
            "held_out_speaker": held_out,
            "train": train_part,
            "val": val_part,
            "test": test_part
        })
        
    # Regime B: Zero-shot transfer (18 non-Viet train, 2 non-Viet val, 4 Viet test)
    regime_b_train_spks = base_non_viet_train
    regime_b_test_spks = sorted(viet_speakers)
    
    assert len(regime_b_train_spks) == 18
    assert len(regime_b_test_spks) == 4
    assert set(regime_b_train_spks).isdisjoint(set(val_speakers))
    assert set(regime_b_train_spks).isdisjoint(set(regime_b_test_spks))
    assert set(val_speakers).isdisjoint(set(regime_b_test_spks))
    assert set(regime_b_train_spks) | set(val_speakers) | set(regime_b_test_spks) == set(all_speakers)
    
    regime_b_train = aggregate_partition(regime_b_train_spks)
    regime_b_test = aggregate_partition(regime_b_test_spks)
    
    assert regime_b_train["utterance_count"] + val_part["utterance_count"] + regime_b_test["utterance_count"] == 3599
    assert regime_b_train["canonical_tokens"] + val_part["canonical_tokens"] + regime_b_test["canonical_tokens"] == 119151
    assert regime_b_train["error_tokens"] + val_part["error_tokens"] + regime_b_test["error_tokens"] == 17518
    
    manifest = {
        "dataset_name": "L2-ARCTIC",
        "dataset_version": "Release v5.0 (March 2020)",
        "archive_filename": os.path.basename(L2ARCTIC_ARCHIVE_PATH),
        "archive_sha256": L2ARCTIC_ARCHIVE_SHA256,
        "license": L2ARCTIC_LICENSE,
        "license_source": "https://psi.engr.tamu.edu/l2-arctic-corpus/",
        "split_protocol": "Vietnamese Evaluation Regimes (Cross-Vietnamese LOSO & Zero-Shot Transfer)",
        "split_protocol_version": SPLIT_PROTOCOL_VERSION,
        "master_seed": MASTER_SEED,
        "generation_timestamp_utc": timestamp_utc,
        "corpus_totals": {
            "speakers": 24,
            "utterances": 3599,
            "canonical_tokens": 119151,
            "canonical_errors": 17518,
            "vietnamese_cohort_speakers": viet_speakers
        },
        "regime_a_cross_vietnamese_loso": {
            "protocol_description": "Cross-Vietnamese Generalization (Intra-L1 Speaker LOSO, 4 rounds). Validation set invariant: {RRBI, NJS}.",
            "rounds": regime_a_rounds
        },
        "regime_b_zero_shot_vietnamese_transfer": {
            "protocol_description": "Zero-Shot Vietnamese-L1 Transfer (18 non-Viet train, 2 non-Viet val, 4 Viet test).",
            "train": regime_b_train,
            "val": val_part,
            "test": regime_b_test
        }
    }
    
    payload_bytes = json.dumps({
        "regime_a": regime_a_rounds,
        "regime_b": {"train": regime_b_train, "val": val_part, "test": regime_b_test}
    }, sort_keys=True).encode("utf-8")
    manifest["payload_sha256"] = compute_sha256(payload_bytes)
    
    return manifest

def generate_speechocean762_splits(timestamp_utc):
    """Parses SpeechOcean762, preserves official test set, carves out 25-speaker stratified validation subset."""
    print("-> Parsing SpeechOcean762 from raw archive and generating official splits...")
    tf = tarfile.open(SPEECHOCEAN_ARCHIVE_PATH, "r:gz")
    
    train_utts_list = [l.split()[0] for l in tf.extractfile("speechocean762/train/wav.scp").read().decode().splitlines() if l.strip()]
    test_utts_list = [l.split()[0] for l in tf.extractfile("speechocean762/test/wav.scp").read().decode().splitlines() if l.strip()]
    
    train_u2s = dict(l.split()[:2] for l in tf.extractfile("speechocean762/train/utt2spk").read().decode().splitlines() if l.strip())
    test_u2s = dict(l.split()[:2] for l in tf.extractfile("speechocean762/test/utt2spk").read().decode().splitlines() if l.strip())
    
    train_s2g = dict(l.split()[:2] for l in tf.extractfile("speechocean762/train/spk2gender").read().decode().splitlines() if l.strip())
    test_s2g = dict(l.split()[:2] for l in tf.extractfile("speechocean762/test/spk2gender").read().decode().splitlines() if l.strip())
    
    train_s2a = dict(l.split()[:2] for l in tf.extractfile("speechocean762/train/spk2age").read().decode().splitlines() if l.strip())
    test_s2a = dict(l.split()[:2] for l in tf.extractfile("speechocean762/test/spk2age").read().decode().splitlines() if l.strip())
    
    scores = json.loads(tf.extractfile("speechocean762/resource/scores.json").read().decode())
    
    # Identify the 245 unmanifested extra WAVs
    all_wav_members = [m.name for m in tf.getmembers() if m.name.endswith(".WAV")]
    unmanifested_wavs = []
    manifested_set = set(train_utts_list) | set(test_utts_list)
    for m in all_wav_members:
        base = m.split("/")[-1].replace(".WAV", "")
        if base not in manifested_set:
            unmanifested_wavs.append(m)
    unmanifested_wavs.sort()
    
    assert len(unmanifested_wavs) == 245, f"Expected 245 unmanifested WAVs, found {len(unmanifested_wavs)}"
    
    # 25-speaker stratified validation carve-out from 125 train speakers
    strata = defaultdict(list)
    for spk in sorted(train_s2g.keys()):
        g = train_s2g[spk]
        age = int(train_s2a[spk])
        age_grp = "child" if age <= 15 else "adult"
        strata[(g, age_grp)].append(spk)
        
    rng = random.Random(MASTER_SEED)
    val_spks = []
    # Stratified target counts (20% of 125 = 25):
    # ('f', 'adult'): 41 -> 8
    # ('f', 'child'): 26 -> 5
    # ('m', 'adult'): 26 -> 5
    # ('m', 'child'): 32 -> 7
    # Total: 8 + 5 + 5 + 7 = 25 (13 Female, 12 Male; 13 Adult, 12 Child)
    target_strata_counts = {
        ("f", "adult"): 8,
        ("f", "child"): 5,
        ("m", "adult"): 5,
        ("m", "child"): 7
    }
    
    for grp in [("f", "adult"), ("f", "child"), ("m", "adult"), ("m", "child")]:
        spks = sorted(strata[grp])
        n_sample = target_strata_counts[grp]
        sampled = rng.sample(spks, n_sample)
        val_spks.extend(sampled)
        
    val_spks = sorted(val_spks)
    val_spk_set = set(val_spks)
    
    train_supervised_spks = sorted([s for s in train_s2g.keys() if s not in val_spk_set])
    official_test_spks = sorted(list(test_s2g.keys()))
    
    assert len(val_spks) == 25, f"Expected 25 val speakers, got {len(val_spks)}"
    assert len(train_supervised_spks) == 100, f"Expected 100 train speakers, got {len(train_supervised_spks)}"
    assert len(official_test_spks) == 125, f"Expected 125 test speakers, got {len(official_test_spks)}"
    
    # Disjointness checks
    assert val_spk_set.isdisjoint(set(train_supervised_spks))
    assert val_spk_set.isdisjoint(set(official_test_spks))
    assert set(train_supervised_spks).isdisjoint(set(official_test_spks))
    
    train_utts = [u for u in train_utts_list if train_u2s[u] in set(train_supervised_spks)]
    val_utts = [u for u in train_utts_list if train_u2s[u] in val_spk_set]
    test_utts = test_utts_list
    
    assert len(train_utts) == 2000, f"Expected 2,000 train utts, got {len(train_utts)}"
    assert len(val_utts) == 500, f"Expected 500 val utts, got {len(val_utts)}"
    assert len(test_utts) == 2500, f"Expected 2,500 test utts, got {len(test_utts)}"
    
    def aggregate_so_partition(spk_list, utt_list, s2g, s2a):
        n_words = sum(len(scores[u]["words"]) for u in utt_list)
        n_phones = sum(len(w["phones-accuracy"]) for u in utt_list for w in scores[u]["words"])
        
        mean_acc = round(sum(scores[u]["accuracy"] for u in utt_list) / len(utt_list), 4)
        mean_comp = round(sum(scores[u]["completeness"] for u in utt_list) / len(utt_list), 4)
        mean_flue = round(sum(scores[u]["fluency"] for u in utt_list) / len(utt_list), 4)
        mean_pros = round(sum(scores[u]["prosodic"] for u in utt_list) / len(utt_list), 4)
        mean_tot = round(sum(scores[u]["total"] for u in utt_list) / len(utt_list), 4)
        
        gender_dist = Counter(s2g[s] for s in spk_list)
        age_group_dist = Counter("child" if int(s2a[s]) <= 15 else "adult" for s in spk_list)
        age_dist = Counter(int(s2a[s]) for s in spk_list)
        
        return {
            "speaker_count": len(spk_list),
            "utterance_count": len(utt_list),
            "word_token_count": n_words,
            "phone_token_count": n_phones,
            "score_means": {
                "accuracy": mean_acc,
                "completeness": mean_comp,
                "fluency": mean_flue,
                "prosodic": mean_pros,
                "total": mean_tot
            },
            "demographics": {
                "gender_distribution": dict(gender_dist),
                "age_group_distribution": dict(age_group_dist),
                "age_distribution": {str(k): v for k, v in sorted(age_dist.items())}
            },
            "speakers": spk_list,
            "utterances": utt_list
        }
        
    part_train = aggregate_so_partition(train_supervised_spks, train_utts, train_s2g, train_s2a)
    part_val = aggregate_so_partition(val_spks, val_utts, train_s2g, train_s2a)
    part_test = aggregate_so_partition(official_test_spks, test_utts, test_s2g, test_s2a)
    
    assert part_train["word_token_count"] + part_val["word_token_count"] + part_test["word_token_count"] == 31816
    assert part_train["phone_token_count"] + part_val["phone_token_count"] + part_test["phone_token_count"] == 94445
    
    manifest = {
        "dataset_name": "SpeechOcean762",
        "dataset_version": "SLR101 / commit 613968e3b0b7",
        "archive_filename": os.path.basename(SPEECHOCEAN_ARCHIVE_PATH),
        "archive_sha256": SPEECHOCEAN_ARCHIVE_SHA256,
        "license": SPEECHOCEAN_LICENSE,
        "license_source": "https://openslr.org/101/",
        "split_protocol": "Preserved Official Kaldi Splits with 25-Speaker Stratified Train-Validation Sub-Split",
        "split_protocol_version": SPLIT_PROTOCOL_VERSION,
        "master_seed": MASTER_SEED,
        "generation_timestamp_utc": timestamp_utc,
        "corpus_totals": {
            "total_speakers": 250,
            "total_manifested_utterances": 5000,
            "total_word_tokens": 31816,
            "total_phone_tokens": 94445,
            "unmanifested_extra_recordings": len(unmanifested_wavs)
        },
        "unmanifested_excluded_recordings": {
            "count": len(unmanifested_wavs),
            "policy": "Strict exclusion from supervised training/validation/test manifests; preserved in raw archive only.",
            "file_paths": unmanifested_wavs
        },
        "partitions": {
            "train": part_train,
            "val": part_val,
            "test": part_test
        }
    }
    
    payload_bytes = json.dumps({
        "train": part_train, "val": part_val, "test": part_test
    }, sort_keys=True).encode("utf-8")
    manifest["payload_sha256"] = compute_sha256(payload_bytes)
    
    return manifest

def main():
    print("=== LANGFENS RQ1-SPLIT-01: Manifest Generator ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Allow override via environment variable if requested, else fixed reproducible timestamp
    timestamp_utc = os.environ.get("LANGFENS_SPLIT_TIMESTAMP", DEFAULT_TIMESTAMP_UTC)
    print(f"-> Using generation timestamp: {timestamp_utc}")
    
    # 1. Parse L2-ARCTIC
    utterance_db, speaker_utts, speaker_stats = parse_l2arctic_corpus()
    
    # 2. Generate 4-fold
    l2_4fold = generate_l2arctic_4fold(utterance_db, speaker_utts, speaker_stats, timestamp_utc)
    path_4fold = os.path.join(OUTPUT_DIR, "l2arctic_4fold_splits.json")
    with open(path_4fold, "w", encoding="utf-8") as f:
        json.dump(l2_4fold, f, indent=2, sort_keys=True)
    print(f"   Wrote {path_4fold} ({os.path.getsize(path_4fold):,} bytes)")
    
    # 3. Generate Vietnamese LOSO
    l2_viet = generate_l2arctic_vietnamese_splits(utterance_db, speaker_utts, speaker_stats, timestamp_utc)
    path_viet = os.path.join(OUTPUT_DIR, "l2arctic_loso_vietnamese_splits.json")
    with open(path_viet, "w", encoding="utf-8") as f:
        json.dump(l2_viet, f, indent=2, sort_keys=True)
    print(f"   Wrote {path_viet} ({os.path.getsize(path_viet):,} bytes)")
    
    # 4. Generate SpeechOcean762
    so_splits = generate_speechocean762_splits(timestamp_utc)
    path_so = os.path.join(OUTPUT_DIR, "speechocean762_official_splits.json")
    with open(path_so, "w", encoding="utf-8") as f:
        json.dump(so_splits, f, indent=2, sort_keys=True)
    print(f"   Wrote {path_so} ({os.path.getsize(path_so):,} bytes)")
    
    print("-> Split JSON manifests successfully written.")

if __name__ == "__main__":
    main()
