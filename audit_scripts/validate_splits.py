#!/usr/bin/env python3
"""
LANGFENS — RQ1-SPLIT-01: Automated Split Validation Test Suite
Senior Speech ML Research Engineer & Research Data Engineer
Verifies:
  1. Zero speaker leakage across train, val, and test partitions
  2. Duplicate utterance assignments within each fold
  3. Missing or unknown speaker IDs
  4. Exact partition cardinalities
  5. 4-fold exhaustive speaker coverage (every speaker in exactly one test fold)
  6. L1 and biological sex balance (3M/3F, 1 per L1)
  7. Canonical token and error-count conservation
  8. Repeat-run deterministic output equality
"""

import os
import sys
import json
import hashlib
import tempfile
import subprocess

OUTPUT_DIR = "manifests/RQ1-SPLIT-01"

def load_json(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def test_l2arctic_4fold():
    print("[TEST 1/8] Verifying L2-ARCTIC 4-Fold CV Manifest...")
    data = load_json("l2arctic_4fold_splits.json")
    
    # Metadata
    assert data["dataset_name"] == "L2-ARCTIC"
    assert data["dataset_version"] == "Release v5.0 (March 2020)"
    assert data["archive_sha256"] == "490d5a43e48b0af84f6bce8e766ad453e8cd670521d5876bb4d0741cca304fda"
    assert data["license"] == "CC BY-NC 4.0"
    assert data["master_seed"] == 42
    assert len(data["folds"]) == 4
    
    all_24_speakers = {
        "ABA", "SKA", "YBAA", "ZHAA", "BWC", "LXC", "NCC", "TXHC",
        "ASI", "RRBI", "SVBI", "TNI", "HJK", "HKK", "YDCK", "YKWK",
        "EBVS", "ERMS", "MBMPS", "NJS", "HQTV", "PNV", "THV", "TLV"
    }
    
    tested_speakers = []
    
    for f in data["folds"]:
        k = f["fold_index"]
        train_spks = set(f["train"]["speakers"])
        val_spks = set(f["val"]["speakers"])
        test_spks = set(f["test"]["speakers"])
        
        # 1. Leakage checks
        assert train_spks.isdisjoint(val_spks), f"Fold {k}: Train and Val overlap: {train_spks & val_spks}"
        assert train_spks.isdisjoint(test_spks), f"Fold {k}: Train and Test overlap: {train_spks & test_spks}"
        assert val_spks.isdisjoint(test_spks), f"Fold {k}: Val and Test overlap: {val_spks & test_spks}"
        assert (train_spks | val_spks | test_spks) == all_24_speakers, f"Fold {k} speaker set does not equal all 24 speakers"
        
        # 2. Cardinalities
        assert len(train_spks) == 16, f"Fold {k}: Expected 16 train speakers, got {len(train_spks)}"
        assert len(val_spks) == 2, f"Fold {k}: Expected 2 val speakers, got {len(val_spks)}"
        assert len(test_spks) == 6, f"Fold {k}: Expected 6 test speakers, got {len(test_spks)}"
        
        # 3. Utterance duplicate and leakage checks
        train_utts = f["train"]["utterances"]
        val_utts = f["val"]["utterances"]
        test_utts = f["test"]["utterances"]
        
        assert len(train_utts) == len(set(train_utts)), f"Fold {k}: Duplicate utterances in train"
        assert len(val_utts) == len(set(val_utts)), f"Fold {k}: Duplicate utterances in val"
        assert len(test_utts) == len(set(test_utts)), f"Fold {k}: Duplicate utterances in test"
        
        set_train_utts = set(train_utts)
        set_val_utts = set(val_utts)
        set_test_utts = set(test_utts)
        
        assert set_train_utts.isdisjoint(set_val_utts), f"Fold {k}: Utterance leak train-val"
        assert set_train_utts.isdisjoint(set_test_utts), f"Fold {k}: Utterance leak train-test"
        assert set_val_utts.isdisjoint(set_test_utts), f"Fold {k}: Utterance leak val-test"
        assert len(set_train_utts | set_val_utts | set_test_utts) == 3599, f"Fold {k}: Total utterances != 3,599"
        
        # 4. Demographic balance
        test_sex = f["test"]["gender_distribution"]
        test_l1 = f["test"]["l1_distribution"]
        assert test_sex == {"M": 3, "F": 3}, f"Fold {k}: Test sex imbalance: {test_sex}"
        assert len(test_l1) == 6 and all(v == 1 for v in test_l1.values()), f"Fold {k}: Test L1 imbalance: {test_l1}"
        
        val_sex = f["val"]["gender_distribution"]
        assert val_sex == {"M": 1, "F": 1}, f"Fold {k}: Val sex imbalance: {val_sex}"
        
        train_sex = f["train"]["gender_distribution"]
        assert train_sex == {"M": 8, "F": 8}, f"Fold {k}: Train sex imbalance: {train_sex}"
        
        # 5. Token and error conservation
        tot_tok = f["train"]["canonical_tokens"] + f["val"]["canonical_tokens"] + f["test"]["canonical_tokens"]
        tot_err = f["train"]["error_tokens"] + f["val"]["error_tokens"] + f["test"]["error_tokens"]
        assert tot_tok == 119151, f"Fold {k}: Token conservation failure: {tot_tok} != 119,151"
        assert tot_err == 17518, f"Fold {k}: Error conservation failure: {tot_err} != 17,518"
        
        tested_speakers.extend(f["test"]["speakers"])
        
    # 6. Exhaustive 4-fold test coverage
    assert len(tested_speakers) == 24
    assert set(tested_speakers) == all_24_speakers
    print("       -> PASS: All 4 folds strictly disjoint, demographic parity confirmed, token conservation verified.")

def test_l2arctic_vietnamese_regimes():
    print("[TEST 2/8] Verifying L2-ARCTIC Vietnamese Evaluation Manifest (Regimes A & B)...")
    data = load_json("l2arctic_loso_vietnamese_splits.json")
    
    assert data["dataset_name"] == "L2-ARCTIC"
    assert data["license"] == "CC BY-NC 4.0"
    all_24_speakers = {
        "ABA", "SKA", "YBAA", "ZHAA", "BWC", "LXC", "NCC", "TXHC",
        "ASI", "RRBI", "SVBI", "TNI", "HJK", "HKK", "YDCK", "YKWK",
        "EBVS", "ERMS", "MBMPS", "NJS", "HQTV", "PNV", "THV", "TLV"
    }
    viet_speakers = ["HQTV", "PNV", "THV", "TLV"]
    
    # Regime A
    rounds = data["regime_a_cross_vietnamese_loso"]["rounds"]
    assert len(rounds) == 4
    for r in rounds:
        r_id = r["round_id"]
        held_out = r["held_out_speaker"]
        train_spks = set(r["train"]["speakers"])
        val_spks = set(r["val"]["speakers"])
        test_spks = set(r["test"]["speakers"])
        
        assert len(train_spks) == 21
        assert len(val_spks) == 2
        assert len(test_spks) == 1
        assert test_spks == {held_out}
        assert val_spks == {"RRBI", "NJS"}
        
        assert train_spks.isdisjoint(val_spks)
        assert train_spks.isdisjoint(test_spks)
        assert val_spks.isdisjoint(test_spks)
        assert (train_spks | val_spks | test_spks) == all_24_speakers
        
        # Token conservation
        tot_tok = r["train"]["canonical_tokens"] + r["val"]["canonical_tokens"] + r["test"]["canonical_tokens"]
        tot_err = r["train"]["error_tokens"] + r["val"]["error_tokens"] + r["test"]["error_tokens"]
        assert tot_tok == 119151
        assert tot_err == 17518
        
    # Regime B
    reg_b = data["regime_b_zero_shot_vietnamese_transfer"]
    b_train = set(reg_b["train"]["speakers"])
    b_val = set(reg_b["val"]["speakers"])
    b_test = set(reg_b["test"]["speakers"])
    
    assert len(b_train) == 18
    assert len(b_val) == 2
    assert len(b_test) == 4
    assert b_val == {"RRBI", "NJS"}
    assert b_test == set(viet_speakers)
    assert b_train.isdisjoint(b_val)
    assert b_train.isdisjoint(b_test)
    assert b_val.isdisjoint(b_test)
    assert (b_train | b_val | b_test) == all_24_speakers
    
    # Zero Vietnamese speakers in train or val
    for s in viet_speakers:
        assert s not in b_train, f"Vietnamese speaker {s} leaked into Regime B train!"
        assert s not in b_val, f"Vietnamese speaker {s} leaked into Regime B val!"
        
    tot_tok_b = reg_b["train"]["canonical_tokens"] + reg_b["val"]["canonical_tokens"] + reg_b["test"]["canonical_tokens"]
    tot_err_b = reg_b["train"]["error_tokens"] + reg_b["val"]["error_tokens"] + reg_b["test"]["error_tokens"]
    assert tot_tok_b == 119151
    assert tot_err_b == 17518
    print("       -> PASS: Vietnamese Regimes A & B leak-free, zero-shot transfer strictly isolated.")

def test_speechocean762_splits():
    print("[TEST 3/8] Verifying SpeechOcean762 Official Split Manifest...")
    data = load_json("speechocean762_official_splits.json")
    
    assert data["dataset_name"] == "SpeechOcean762"
    assert data["license"] == "CC BY 4.0"
    assert data["corpus_totals"]["total_speakers"] == 250
    assert data["corpus_totals"]["total_manifested_utterances"] == 5000
    assert data["corpus_totals"]["unmanifested_extra_recordings"] == 245
    
    parts = data["partitions"]
    train_spks = set(parts["train"]["speakers"])
    val_spks = set(parts["val"]["speakers"])
    test_spks = set(parts["test"]["speakers"])
    
    # Speaker cardinalities
    assert len(train_spks) == 100, f"Expected 100 train speakers, got {len(train_spks)}"
    assert len(val_spks) == 25, f"Expected 25 val speakers, got {len(val_spks)}"
    assert len(test_spks) == 125, f"Expected 125 test speakers, got {len(test_spks)}"
    assert len(train_spks | val_spks | test_spks) == 250
    
    # Disjointness
    assert train_spks.isdisjoint(val_spks), "SpeechOcean train and val speakers overlap"
    assert train_spks.isdisjoint(test_spks), "SpeechOcean train and test speakers overlap"
    assert val_spks.isdisjoint(test_spks), "SpeechOcean val and test speakers overlap"
    
    # Utterance cardinalities
    train_utts = set(parts["train"]["utterances"])
    val_utts = set(parts["val"]["utterances"])
    test_utts = set(parts["test"]["utterances"])
    assert len(train_utts) == 2000
    assert len(val_utts) == 500
    assert len(test_utts) == 2500
    assert len(train_utts | val_utts | test_utts) == 5000
    
    # Official test partition preserved:
    # Official train side is train | val
    official_train_spks = train_spks | val_spks
    assert len(official_train_spks) == 125
    assert len(official_train_spks | test_spks) == 250
    
    # Demographic balance on val
    val_g = parts["val"]["demographics"]["gender_distribution"]
    val_a = parts["val"]["demographics"]["age_group_distribution"]
    assert val_g == {"f": 13, "m": 12}, f"Val gender imbalance: {val_g}"
    assert val_a == {"adult": 13, "child": 12}, f"Val age imbalance: {val_a}"
    
    # Unmanifested WAV check
    unman = data["unmanifested_excluded_recordings"]["file_paths"]
    assert len(unman) == 245
    for p in unman:
        base = p.split("/")[-1].replace(".WAV", "")
        assert base not in train_utts
        assert base not in val_utts
        assert base not in test_utts
        
    # Word and phone tokens conservation
    tot_words = parts["train"]["word_token_count"] + parts["val"]["word_token_count"] + parts["test"]["word_token_count"]
    tot_phones = parts["train"]["phone_token_count"] + parts["val"]["phone_token_count"] + parts["test"]["phone_token_count"]
    assert tot_words == 31816, f"Word token count != 31,816: {tot_words}"
    assert tot_phones == 94445, f"Phone token count != 94,445: {tot_phones}"
    print("       -> PASS: SpeechOcean762 official test set preserved, 25-speaker stratified val validated, 245 orphan WAVs excluded.")

def test_determinism_and_idempotence():
    print("[TEST 4/8] Verifying Repeat-Run Deterministic Output Equality...")
    with tempfile.TemporaryDirectory() as tmpdir:
        env = dict(os.environ)
        env["LANGFENS_SPLIT_TIMESTAMP"] = "2026-10-23T13:30:00Z"
        env["PYTHONPATH"] = os.path.abspath("audit_scripts") + ":" + env.get("PYTHONPATH", "")
        
        # Run generator script into tempdir by copying script and modifying OUTPUT_DIR
        script_path = "audit_scripts/generate_rq1_splits.py"
        with open(script_path, "r", encoding="utf-8") as f:
            code = f.read()
        
        temp_script = os.path.join(tmpdir, "gen_temp.py")
        code_mod = code.replace('OUTPUT_DIR = "manifests/RQ1-SPLIT-01"', f'OUTPUT_DIR = "{tmpdir}"')
        with open(temp_script, "w", encoding="utf-8") as f:
            f.write(code_mod)
            
        res = subprocess.run([sys.executable, temp_script], env=env, capture_output=True, text=True)
        assert res.returncode == 0, f"Generator run failed: {res.stderr}"
        
        # Compare checksums of each manifest
        for fname in ["l2arctic_4fold_splits.json", "l2arctic_loso_vietnamese_splits.json", "speechocean762_official_splits.json"]:
            orig_p = os.path.join(OUTPUT_DIR, fname)
            temp_p = os.path.join(tmpdir, fname)
            
            with open(orig_p, "rb") as f1, open(temp_p, "rb") as f2:
                h1 = hashlib.sha256(f1.read()).hexdigest()
                h2 = hashlib.sha256(f2.read()).hexdigest()
                assert h1 == h2, f"Determinism mismatch for {fname}: {h1} != {h2}"
                
    print("       -> PASS: Generation script is 100% deterministic and bit-for-bit idempotent.")

def main():
    print("=== Running LANGFENS RQ1-SPLIT-01 Automated Validation Suite ===")
    test_l2arctic_4fold()
    test_l2arctic_vietnamese_regimes()
    test_speechocean762_splits()
    test_determinism_and_idempotence()
    print("=== ALL 4 CORE AUTOMATED VALIDATION CHECKS PASSED ===")

if __name__ == "__main__":
    main()
