import json
import pytest
import numpy as np
from pathlib import Path

from langfens_prep.alignment import align_canonical_sequences
from audit_scripts.evaluate_protocol_z import average_precision_binary, calculate_binary_metrics
from langfens_prep.phoneme_normalizer import PHONE_TO_CTC_INDEX

RECORDS_PATH = Path("manifests/RQ1-TRN-02A/evaluation_reconciliation/zero_oracle_verification/PROTOCOL_Z_PREDICTION_RECORDS.json")
REPORT_PATH = Path("manifests/RQ1-TRN-02A/evaluation_reconciliation/zero_oracle_verification/PROTOCOL_Z_VERIFICATION_REPORT.json")


def test_alignment_correctness_synthetic_cases():
    """Tests that dynamic programming alignment correctly decomposes edits into S, D, I, and matches."""
    # 1. Exact match
    res_match = align_canonical_sequences(["B", "AE", "T"], ["B", "AE", "T"])
    assert res_match["edits_summary"]["total_edits"] == 0
    assert len(res_match["aligned_pairs"]) == 3
    assert len(res_match["unaligned_gold"]) == 0
    assert len(res_match["unaligned_g2p"]) == 0

    # 2. Pure substitution
    res_sub = align_canonical_sequences(["DH", "AH"], ["DH", "IY"])
    assert res_sub["edits_summary"]["total_edits"] == 1
    assert res_sub["edits_summary"]["substitutions"] == 1
    assert len(res_sub["aligned_pairs"]) == 2
    assert res_sub["aligned_pairs"][1] == (1, 1, 1)  # substitution op_code == 1

    # 3. Pure deletion from gold (omitted in G2P)
    res_del = align_canonical_sequences(["HH", "W", "AH", "T"], ["W", "AH", "T"])
    assert res_del["edits_summary"]["total_edits"] == 1
    assert res_del["edits_summary"]["deletions"] == 1
    assert len(res_del["unaligned_gold"]) == 1
    assert res_del["unaligned_gold"][0] == (0, "HH")

    # 4. Pure insertion into G2P (extra in G2P)
    res_ins = align_canonical_sequences(["B", "L", "EH", "S"], ["B", "L", "EH", "S", "AH", "M"])
    assert res_ins["edits_summary"]["total_edits"] == 2
    assert res_ins["edits_summary"]["insertions"] == 2
    assert len(res_ins["unaligned_g2p"]) == 2
    assert res_ins["unaligned_g2p"][0] == (4, "AH")
    assert res_ins["unaligned_g2p"][1] == (5, "M")

    # 5. Tie-breaking ordering
    for tie in ["sub_first", "del_first", "ins_first"]:
        r = align_canonical_sequences(["AH", "K"], ["IH", "K"], tie_break=tie)
        assert r["edits_summary"]["total_edits"] == 1


def test_reconstruction_of_protocol_z_ap_from_exported_records():
    """Reconstructs Protocol Z AP directly from PROTOCOL_Z_PREDICTION_RECORDS.json."""
    if not RECORDS_PATH.is_file():
        pytest.skip(f"Prediction records artifact not found at {RECORDS_PATH}")

    with open(RECORDS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    assert meta["total_utterances"] == 300
    assert meta["total_gold_tokens"] == 9956
    assert meta["checkpoint_sha256"] == "a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b"
    assert meta["cmudict_sha256"] == "81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22"

    utterances = data["utterances"]
    assert len(utterances) == 300

    aligned_z_scores = []
    aligned_gold_labels = []

    full_labels = []
    audit_imputed_scores = []
    policy_b1_scores = []

    exact_z_scores = []
    exact_labels = []

    # Load train priors for policy B1
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    for u in utterances:
        g_lbls = u["gold_labels"]
        g_ph = u["gold_phones"]
        z_sc = u["g2p_scores"]
        z_ph = u["g2p_phones"]

        if u["is_exact_match"]:
            exact_z_scores.extend(z_sc)
            exact_labels.extend(g_lbls)

        # Trace aligned pairs
        for g_idx, z_idx, op in u["aligned_pairs"]:
            aligned_z_scores.append(z_sc[z_idx])
            aligned_gold_labels.append(g_lbls[g_idx])

            full_labels.append(g_lbls[g_idx])
            audit_imputed_scores.append(z_sc[z_idx])

        # Trace unaligned gold tokens (deletions)
        for g_idx, g_phone in u["unaligned_gold"]:
            full_labels.append(g_lbls[g_idx])
            audit_imputed_scores.append(0.116211)  # validation prevalence constant

    # 1. Verify Aligned-Only Population ($N=9,919$)
    assert len(aligned_gold_labels) == 9919
    assert sum(aligned_gold_labels) == 1145
    aligned_ap = average_precision_binary(aligned_gold_labels, aligned_z_scores)
    assert pytest.approx(aligned_ap, abs=1e-5) == 0.313530

    # 2. Verify Full Gold Population under Audit Validation Prevalence Imputation ($N=9,956$)
    assert len(full_labels) == 9956
    assert sum(full_labels) == 1157
    audit_ap = average_precision_binary(full_labels, audit_imputed_scores)
    # Verifies reproduction of reported 0.312467 to within 0.000004
    assert pytest.approx(audit_ap, abs=1e-5) == 0.312471
    assert abs(audit_ap - 0.312467) < 5e-6

    # 3. Verify Exact Sequence Utterances ($N=2,338$ across 82 utts)
    assert len(exact_labels) == 2338
    assert sum(exact_labels) == 259
    exact_ap = average_precision_binary(exact_labels, exact_z_scores)
    assert pytest.approx(exact_ap, abs=1e-5) == 0.305362


def test_scoring_denominator_integrity():
    """Verifies that scoring population denominators are explicitly accounted for without truncation."""
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    census = report["frontend_token_census"]
    total_gold = census["total_gold_phone_tokens"]
    total_g2p = census["total_g2p_phone_tokens"]
    unaligned_gold = census["alignment_edits"]["unaligned_gold_tokens (deletions)"]
    unaligned_g2p = census["alignment_edits"]["unaligned_g2p_tokens (insertions)"]
    subs = census["alignment_edits"]["substitutions"]

    assert total_gold == 9956
    assert total_g2p == 9956
    assert unaligned_gold == 37
    assert unaligned_g2p == 37
    assert subs == 345

    # Identity check: Aligned tokens + unaligned gold == total gold
    pop1_tokens = report["scoring_populations"]["population_1_aligned_only"]["evaluated_tokens"]
    assert pop1_tokens + unaligned_gold == total_gold

    # Denominator check: Full gold population must strictly evaluate on 9,956 tokens
    pop2_tokens = report["scoring_populations"]["population_2_full_gold_benchmark"]["total_gold_tokens"]
    assert pop2_tokens == total_gold


def test_ambiguous_alignment_tie_break_divergence():
    """
    Verifies that ambiguous alignments produce different aligned and unaligned token positions
    under sub_first vs del_first policies, confirming the necessity of independent matched-pair construction.
    """
    # Synthetic minimal example where DP paths tie:
    # Option A (sub_first): substitute R->AO, substitute ER->R (cost 2; 3 aligned pairs, 0 unaligned)
    # Option B (del_first): match F, insert AO, match R==R, delete ER (cost 2; 2 aligned pairs, 1 unaligned gold, 1 unaligned g2p)
    gold = ["F", "R", "ER"]
    g2p = ["F", "AO", "R"]

    align_sub = align_canonical_sequences(gold, g2p, tie_break="sub_first")
    align_del = align_canonical_sequences(gold, g2p, tie_break="del_first")

    # 1. Verify that alignment outcomes differ under the two tie-break policies
    assert align_sub["aligned_pairs"] != align_del["aligned_pairs"]
    assert align_sub["unaligned_gold"] != align_del["unaligned_gold"]

    # 2. sub_first prefers substitution
    assert len(align_sub["aligned_pairs"]) == 3
    assert len(align_sub["unaligned_gold"]) == 0
    assert len(align_sub["unaligned_g2p"]) == 0
    assert align_sub["edits_summary"]["substitutions"] == 2
    assert align_sub["edits_summary"]["deletions"] == 0
    assert align_sub["edits_summary"]["insertions"] == 0

    # 3. del_first prefers deletion + insertion to preserve the exact match on R
    assert len(align_del["aligned_pairs"]) == 2
    assert len(align_del["unaligned_gold"]) == 1
    assert len(align_del["unaligned_g2p"]) == 1
    assert align_del["unaligned_gold"] == [(2, "ER")]
    assert align_del["unaligned_g2p"] == [(1, "AO")]
    assert align_del["aligned_pairs"] == [(0, 0, 0), (1, 2, 0)]
    assert align_del["edits_summary"]["substitutions"] == 0
    assert align_del["edits_summary"]["deletions"] == 1
    assert align_del["edits_summary"]["insertions"] == 1


def test_independent_matched_pairs_vs_vector_reuse_bug():
    """
    Demonstrates that reusing the sub_first gold label vector on del_first scores causes
    catastrophic vector misalignment, dropping AP to ~0.306, while reconstructing matched pairs
    independently preserves genuine AP stability (~0.3145).
    """
    if not RECORDS_PATH.is_file():
        pytest.skip("Prediction records artifact not found")

    with open(RECORDS_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)["utterances"]

    # Load train priors
    from langfens_prep.dataset import L2ArcticMDDDataset
    from audit_scripts.evaluate_protocol_z import fit_train_canonical_prior

    train_ds = L2ArcticMDDDataset(
        split_manifest_path="manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json",
        fold=1,
        split="train",
        zip_path="datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip"
    )
    train_prior = fit_train_canonical_prior(train_ds)

    # 1. Independent matched-pair reconstruction for sub_first
    sub_labels, sub_b1 = [], []
    for rec in records:
        al = align_canonical_sequences(rec["gold_phones"], rec["g2p_phones"], tie_break="sub_first")
        for g_i, z_i, op in al["aligned_pairs"]:
            sub_labels.append(rec["gold_labels"][g_i])
            sub_b1.append(rec["g2p_scores"][z_i])
        for g_i, g_p in al["unaligned_gold"]:
            sub_labels.append(rec["gold_labels"][g_i])
            g_c = PHONE_TO_CTC_INDEX.get(g_p, 0)
            sub_b1.append(train_prior["phone_priors"].get(g_c, train_prior["overall_train_rate"]))

    # 2. Independent matched-pair reconstruction for del_first
    del_labels, del_b1 = [], []
    for rec in records:
        al = align_canonical_sequences(rec["gold_phones"], rec["g2p_phones"], tie_break="del_first")
        for g_i, z_i, op in al["aligned_pairs"]:
            del_labels.append(rec["gold_labels"][g_i])
            del_b1.append(rec["g2p_scores"][z_i])
        for g_i, g_p in al["unaligned_gold"]:
            del_labels.append(rec["gold_labels"][g_i])
            g_c = PHONE_TO_CTC_INDEX.get(g_p, 0)
            del_b1.append(train_prior["phone_priors"].get(g_c, train_prior["overall_train_rate"]))

    # Genuine APs with matched pairs
    genuine_sub_ap = average_precision_binary(sub_labels, sub_b1)
    genuine_del_ap = average_precision_binary(del_labels, del_b1)

    assert pytest.approx(genuine_sub_ap, abs=1e-5) == 0.314265
    assert pytest.approx(genuine_del_ap, abs=1e-5) == 0.314519
    # The genuine variation between sub_first and del_first is negligible (< 0.0003)
    assert abs(genuine_sub_ap - genuine_del_ap) < 0.0005

    # 3. Simulate the vector reuse bug: evaluate del_b1 scores against sub_labels!
    buggy_del_ap = average_precision_binary(sub_labels, del_b1)
    assert pytest.approx(buggy_del_ap, abs=1e-5) == 0.306266
    # The bug artificially drops AP by ~0.008
    assert genuine_del_ap - buggy_del_ap > 0.007


def test_tie_break_sweep_recomputed_metrics():
    """Verifies that PROTOCOL_Z_VERIFICATION_REPORT.json contains genuinely computed tie-break metrics."""
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        rep = json.load(f)

    sweep = rep["tie_break_sensitivity_sweep"]
    assert "sub_first" in sweep
    assert "del_first" in sweep
    assert "ins_first" in sweep

    # All three policies must include Policy B2 and genuine APs
    for policy, vals in sweep.items():
        assert "aligned_only_ap" in vals
        assert "policy_b1_ap" in vals
        assert "policy_b2_ap" in vals
        assert "audit_val_prev_ap" in vals

    # Numerical regression assertions for genuine metrics
    assert pytest.approx(sweep["sub_first"]["aligned_only_ap"], abs=1e-5) == 0.313530
    assert pytest.approx(sweep["sub_first"]["policy_b1_ap"], abs=1e-5) == 0.314265
    assert pytest.approx(sweep["sub_first"]["policy_b2_ap"], abs=1e-5) == 0.313006
    assert pytest.approx(sweep["sub_first"]["audit_val_prev_ap"], abs=1e-5) == 0.312471

    assert pytest.approx(sweep["del_first"]["aligned_only_ap"], abs=1e-5) == 0.313670
    assert pytest.approx(sweep["del_first"]["policy_b1_ap"], abs=1e-5) == 0.314519
    assert pytest.approx(sweep["del_first"]["policy_b2_ap"], abs=1e-5) == 0.312997
    assert pytest.approx(sweep["del_first"]["audit_val_prev_ap"], abs=1e-5) == 0.312270

    assert pytest.approx(sweep["ins_first"]["aligned_only_ap"], abs=1e-5) == 0.313863
    assert pytest.approx(sweep["ins_first"]["policy_b1_ap"], abs=1e-5) == 0.314790
    assert pytest.approx(sweep["ins_first"]["policy_b2_ap"], abs=1e-5) == 0.313319
    assert pytest.approx(sweep["ins_first"]["audit_val_prev_ap"], abs=1e-5) == 0.312725


def test_production_checkpoint_provenance_and_metadata_integrity():
    """Inspects recovered best.pt and asserts true stored provenance, rejecting fabricated defaults."""
    import torch
    from audit_scripts.evaluate_protocol_z import load_production_model

    ckpt_path = Path("checkpoints/RQ1-TRN-02A/recovered/best.pt")
    if not ckpt_path.is_file():
        pytest.skip("Checkpoint best.pt not found")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 1. Assert true metadata stored in production checkpoint
    assert ckpt["epoch"] == 5, f"Expected epoch 5, got {ckpt['epoch']}"
    assert ckpt["successful_updates"] == 3600, f"Expected 3600 updates, got {ckpt['successful_updates']}"
    assert pytest.approx(float(ckpt["best_ap"]), abs=1e-5) == 0.316062
    assert ckpt["metadata"]["git_sha"] == "caa56783113d40eac438e5f458cdb313b8932218"

    # 2. Verify that fabricated defaults (18, 14300, 0.316139) are NOT used
    assert ckpt["epoch"] != 18
    assert ckpt["successful_updates"] != 14300
    assert abs(float(ckpt["best_ap"]) - 0.316139) > 5e-5

    # 3. Verify load_production_model returns true metadata
    device = torch.device("cpu")
    model, meta = load_production_model(ckpt_path, device)
    assert meta["epoch"] == 5
    assert meta["successful_updates"] == 3600
    assert meta["training_git_sha"] == "caa56783113d40eac438e5f458cdb313b8932218"
    assert pytest.approx(meta["best_ap"], abs=1e-5) == 0.316062


def test_frozen_manifest_split_and_speaker_boundaries():
    """Verifies that frozen manifest Fold 1 test speakers are ABA, ASI, EBVS, HJK, LXC, PNV, not SKA, YDCK, ZHAA."""
    manifest_path = Path("manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    fold1 = splits["folds"][0]
    val_spks = sorted(fold1["val"]["speakers"])
    test_spks = sorted(fold1["test"]["speakers"])
    train_spks = sorted(fold1["train"]["speakers"])

    # Official Fold 1 test partition: 6 speakers (3M, 3F; 1 per L1)
    assert test_spks == ["ABA", "ASI", "EBVS", "HJK", "LXC", "PNV"]
    assert val_spks == ["NJS", "RRBI"]

    # SKA, YDCK, and ZHAA must be in train partition, NEVER in test partition
    assert "SKA" in train_spks
    assert "YDCK" in train_spks
    assert "ZHAA" in train_spks

    assert "SKA" not in test_spks
    assert "YDCK" not in test_spks
    assert "ZHAA" not in test_spks


def test_evaluation_entrypoint_split_safeguard():
    """Verifies that Protocol Z evaluator entrypoint rejects attempts to evaluate on official test split."""
    import subprocess
    import sys
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = f"src:{env.get('PYTHONPATH', '')}"
    cmd = [
        sys.executable, "audit_scripts/evaluate_protocol_z.py",
        "--split", "test"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    # Must exit with non-zero code due to argparse choices or security guard
    assert res.returncode != 0
    assert "invalid choice" in res.stderr or "CRITICAL SAFETY VIOLATION" in res.stderr
