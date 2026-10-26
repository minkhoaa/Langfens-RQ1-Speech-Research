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
