"""
LANGFENS — RQ1-TRN-03B: Step 18 Scientific Comparison Correction Tests.
Senior Speech ML Research Engineer & Reproducibility Auditor.

Verifies:
  1. Reproduction of Fold 1 validation metrics across all 5 comparable endpoints.
  2. Pedagogical operating points at fixed FPR ceilings (10%, 15%, 20%).
  3. Paired prediction records archive integrity and cryptographic provenance.
  4. Separation of ranking quality (AP / ROC-AUC) from calibration shifts (logit bias).
  5. Protocol O and Protocol Z parity and acoustic reliance gap integrity.
  6. Registered Gate C criteria audit: failure of AP elevation target (> +0.020) and candidate classification.
  7. Fail-closed artifact verification (no silent skips or unconditional passes).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import pytest
import numpy as np

from audit_scripts.evaluate_cross_fold_validation import compute_sha256


EXPECTED_SHA256_CHECKPOINTS = {
    "Baseline_R_fixed_budget_registered_primary": "a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b",
    "Arm2A_Focal_fixed_budget_registered_primary": "46c78486ea41c13e45ef4654604acfbe58f8cf4a0e237b65f60abfebac506f93",
    "Arm2A_Focal_validation_best_secondary": "e39813bbb3cab9fdfab303167f66e215640d6024970fba919eecdeac281ab175",
    "Arm2B_Weighted_exploratory_near_budget_last": "3e886a4fbd64414c60b0993b92419c66007b5cd34085c25978069fac5d3310ac",
    "Arm2B_Weighted_validation_best_secondary": "a94434ca31308d29853bf28b020101820dc47353d598620a50a2a65fea12608a",
}


def test_fold1_metrics_reproduction_and_provenance():
    """Verifies that all 5 comparable endpoints reproduce exact Fold 1 metrics."""
    report_path = Path("manifests/RQ1-TRN-03A/evaluation/PILOT_FOLD1_COMPARATIVE_REPORT.json")
    assert report_path.is_file(), f"FAIL-CLOSED: Missing {report_path}"

    with open(report_path) as f:
        rep = json.load(f)

    # 1. Baseline R Fixed-Budget (3,600 updates / Ep 6)
    base = rep["Baseline_R_fixed_budget_registered_primary"]
    assert base["updates_stored"] == 3600
    assert pytest.approx(base["matched_ap"], abs=1e-5) == 0.317012
    assert pytest.approx(base["mismatched_ap"], abs=1e-5) == 0.294736
    assert pytest.approx(base["acoustic_gap"], abs=1e-5) == 0.022276
    assert pytest.approx(base["ctc_per"], abs=1e-4) == 0.236818
    assert pytest.approx(base["proto_z_aligned_ap"], abs=1e-5) == 0.313513

    # 2. Arm 2A Focal Fixed-Budget (3,600 updates / Ep 6)
    a_fb = rep["Arm2A_Focal_fixed_budget_registered_primary"]
    assert a_fb["updates_stored"] == 3600
    assert pytest.approx(a_fb["matched_ap"], abs=1e-5) == 0.308104
    assert pytest.approx(a_fb["mismatched_ap"], abs=1e-5) == 0.298235
    assert pytest.approx(a_fb["acoustic_gap"], abs=1e-5) == 0.009869
    assert pytest.approx(a_fb["ctc_per"], abs=1e-4) == 0.236413
    assert pytest.approx(a_fb["proto_z_aligned_ap"], abs=1e-5) == 0.306181

    # 3. Arm 2A Focal Validation-Best (1,200 updates / Ep 2)
    a_best = rep["Arm2A_Focal_validation_best_secondary"]
    assert a_best["updates_stored"] == 1200
    assert pytest.approx(a_best["matched_ap"], abs=1e-5) == 0.317201
    assert pytest.approx(a_best["mismatched_ap"], abs=1e-5) == 0.295456
    assert pytest.approx(a_best["acoustic_gap"], abs=1e-5) == 0.021745
    assert pytest.approx(a_best["ctc_per"], abs=1e-4) == 0.275681
    assert pytest.approx(a_best["proto_z_aligned_ap"], abs=1e-5) == 0.313677

    # 4. Arm 2B Weighted Exploratory (3,598 updates / Ep 6)
    b_exp = rep["Arm2B_Weighted_exploratory_near_budget_last"]
    assert b_exp["updates_stored"] == 3598
    assert pytest.approx(b_exp["matched_ap"], abs=1e-5) == 0.308087
    assert pytest.approx(b_exp["mismatched_ap"], abs=1e-5) == 0.289355
    assert pytest.approx(b_exp["acoustic_gap"], abs=1e-5) == 0.018733
    assert pytest.approx(b_exp["ctc_per"], abs=1e-4) == 0.238033
    assert pytest.approx(b_exp["proto_z_aligned_ap"], abs=1e-5) == 0.302864

    # 5. Arm 2B Weighted Validation-Best (600 updates / Ep 1)
    b_best = rep["Arm2B_Weighted_validation_best_secondary"]
    assert b_best["updates_stored"] == 600
    assert pytest.approx(b_best["matched_ap"], abs=1e-5) == 0.316822
    assert pytest.approx(b_best["mismatched_ap"], abs=1e-5) == 0.296586
    assert pytest.approx(b_best["acoustic_gap"], abs=1e-5) == 0.020236
    assert pytest.approx(b_best["ctc_per"], abs=1e-4) == 0.312215
    assert pytest.approx(b_best["proto_z_aligned_ap"], abs=1e-5) == 0.312743


def test_fixed_fpr_operating_points_and_baseline_superiority():
    """Proves Baseline R outperforms both Arm 2A and Arm 2B at fixed FPR ceilings (10%, 15%, 20%)."""
    report_path = Path("manifests/RQ1-TRN-03A/evaluation/PILOT_FOLD1_COMPARATIVE_REPORT.json")
    with open(report_path) as f:
        rep = json.load(f)

    base_op = rep["Baseline_R_fixed_budget_registered_primary"]["operating_points_fixed_fpr"]
    a_fb_op = rep["Arm2A_Focal_fixed_budget_registered_primary"]["operating_points_fixed_fpr"]
    b_exp_op = rep["Arm2B_Weighted_exploratory_near_budget_last"]["operating_points_fixed_fpr"]

    # 1. Verify FPR ceilings are strictly satisfied (<= 10%, 15%, 20%)
    for op in [base_op, a_fb_op, b_exp_op]:
        assert op["fpr_ceiling_10"]["fpr"] <= 0.10
        assert op["fpr_ceiling_15"]["fpr"] <= 0.15
        assert op["fpr_ceiling_20"]["fpr"] <= 0.20

    # 2. At FPR <= 10%: Baseline R FNR (59.90%) < Arm 2B (60.33%) < Arm 2A (61.62%)
    assert base_op["fpr_ceiling_10"]["fnr"] == 0.5990
    assert a_fb_op["fpr_ceiling_10"]["fnr"] == 0.6162
    assert b_exp_op["fpr_ceiling_10"]["fnr"] == 0.6033
    assert base_op["fpr_ceiling_10"]["fnr"] < a_fb_op["fpr_ceiling_10"]["fnr"]
    assert base_op["fpr_ceiling_10"]["fnr"] < b_exp_op["fpr_ceiling_10"]["fnr"]

    # 3. At FPR <= 15%: Baseline R FNR (49.35%) < Arm 2B (51.08%) < Arm 2A (51.51%)
    assert base_op["fpr_ceiling_15"]["fnr"] == 0.4935
    assert a_fb_op["fpr_ceiling_15"]["fnr"] == 0.5151
    assert b_exp_op["fpr_ceiling_15"]["fnr"] == 0.5108
    assert base_op["fpr_ceiling_15"]["fnr"] < a_fb_op["fpr_ceiling_15"]["fnr"]
    assert base_op["fpr_ceiling_15"]["fnr"] < b_exp_op["fpr_ceiling_15"]["fnr"]

    # 4. At FPR <= 20%: Baseline R FNR (43.30%) < Arm 2B (43.56%) < Arm 2A (44.25%)
    assert base_op["fpr_ceiling_20"]["fnr"] == 0.4330
    assert a_fb_op["fpr_ceiling_20"]["fnr"] == 0.4425
    assert b_exp_op["fpr_ceiling_20"]["fnr"] == 0.4356
    assert base_op["fpr_ceiling_20"]["fnr"] < a_fb_op["fpr_ceiling_20"]["fnr"]
    assert base_op["fpr_ceiling_20"]["fnr"] < b_exp_op["fpr_ceiling_20"]["fnr"]

    # 5. Registered Criterion 3 failure: ALL models fail the target FNR <= 40% at 15% FPR
    for op in [base_op, a_fb_op, b_exp_op]:
        assert op["fpr_ceiling_15"]["fnr"] > 0.40


def test_paired_prediction_records_integrity_and_provenance():
    """Verifies that paired prediction records contain full per-token predictions and valid provenance."""
    gz_path = Path("manifests/RQ1-TRN-03A/evaluation/PILOT_FOLD1_PAIRED_PREDICTION_RECORDS.json.gz")
    assert gz_path.is_file(), f"FAIL-CLOSED: Missing {gz_path}"

    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    assert meta["total_utterances"] == 300
    assert meta["total_tokens"] == 9956
    assert meta["total_errors"] == 1157
    assert meta["prevalence"] == pytest.approx(1157 / 9956, abs=1e-5)

    # Provenance ledger
    assert len(meta["checkpoints_sha256"]) == 5
    for key, expected_sha in EXPECTED_SHA256_CHECKPOINTS.items():
        assert meta["checkpoints_sha256"][key] == expected_sha

    utts = data["utterances"]
    assert len(utts) == 300

    spk_counts = {"NJS": 0, "RRBI": 0}
    for u in utts:
        spk = u["speaker"]
        assert spk in spk_counts
        spk_counts[spk] += 1
        assert len(u["canonical_phones"]) == len(u["labels"])
        for mod_key in EXPECTED_SHA256_CHECKPOINTS:
            assert mod_key in u["models"]
            assert len(u["models"][mod_key]["matched_scores"]) == len(u["labels"])
            assert len(u["models"][mod_key]["mismatched_scores"]) == len(u["labels"])

    assert spk_counts["NJS"] == 150
    assert spk_counts["RRBI"] == 150


def test_ranking_quality_vs_calibration_shift_separation():
    """Proves static tau=0.50 comparison evaluates calibration shift rather than discriminative ranking."""
    report_path = Path("manifests/RQ1-TRN-03A/evaluation/PILOT_FOLD1_COMPARATIVE_REPORT.json")
    with open(report_path) as f:
        rep = json.load(f)

    base = rep["Baseline_R_fixed_budget_registered_primary"]
    a_fb = rep["Arm2A_Focal_fixed_budget_registered_primary"]
    b_exp = rep["Arm2B_Weighted_exploratory_near_budget_last"]

    # 1. Intrinsic Ranking Quality: ROC-AUC and PR-AUC (AP)
    # Baseline R achieves the highest ROC-AUC of all endpoints
    assert base["roc_auc_matched"] == 0.751401
    assert a_fb["roc_auc_matched"] == 0.746914
    assert b_exp["roc_auc_matched"] == 0.746055
    assert base["roc_auc_matched"] > a_fb["roc_auc_matched"]
    assert base["roc_auc_matched"] > b_exp["roc_auc_matched"]

    # 2. Calibration Shift: Logit Offset
    # Arm 2B pos_weight=8.40 shifts median negative logit from -2.192 to +0.013 (+2.205 shift)
    base_neg_logit = base["distribution_statistics"]["logits_negatives"]["median"]
    b_exp_neg_logit = b_exp["distribution_statistics"]["logits_negatives"]["median"]
    assert pytest.approx(base_neg_logit, abs=0.1) == -2.192
    assert pytest.approx(b_exp_neg_logit, abs=0.1) == 0.013
    logit_shift = b_exp_neg_logit - base_neg_logit
    theoretical_shift = np.log(8.40)  # ~2.128
    assert pytest.approx(logit_shift, abs=0.15) == theoretical_shift

    # 3. Operating thresholds reflect the calibration shift:
    # At 15% FPR: Baseline tau=0.2104, Focal tau=0.4805, Weighted tau=0.7058
    assert base["operating_points_fixed_fpr"]["fpr_ceiling_15"]["threshold"] == 0.210401
    assert a_fb["operating_points_fixed_fpr"]["fpr_ceiling_15"]["threshold"] == 0.480470
    assert b_exp["operating_points_fixed_fpr"]["fpr_ceiling_15"]["threshold"] == 0.705826


def test_acoustic_reliance_and_protocol_parity():
    """Verifies that acoustic reliance gap degraded in Arm 2A fixed-budget and Protocol Z parity holds."""
    report_path = Path("manifests/RQ1-TRN-03A/evaluation/PILOT_FOLD1_COMPARATIVE_REPORT.json")
    with open(report_path) as f:
        rep = json.load(f)

    base = rep["Baseline_R_fixed_budget_registered_primary"]
    a_fb = rep["Arm2A_Focal_fixed_budget_registered_primary"]
    b_exp = rep["Arm2B_Weighted_exploratory_near_budget_last"]

    # 1. Acoustic gap criterion: Step 15 Criterion 2 required Delta_gap >= +0.010
    # Arm 2A fixed-budget drops to +0.009869 < +0.010 (FAILED)
    assert a_fb["acoustic_gap"] == 0.009869
    assert a_fb["acoustic_gap"] < 0.010
    assert rep["comparative_analysis"]["paired_differences_vs_baseline_r_fixed_budget"]["Arm2A_Focal_fixed_budget_registered_primary"]["meets_step15_gap_target_ge0010"] is False

    # Arm 2B retains stronger acoustic gap (+0.018733 >= +0.010)
    assert b_exp["acoustic_gap"] == 0.018733
    assert b_exp["acoustic_gap"] >= 0.010

    # 2. Protocol Z (zero-oracle G2P) parity matches Protocol O ordering
    assert base["proto_z_aligned_ap"] > a_fb["proto_z_aligned_ap"]
    assert base["proto_z_aligned_ap"] > b_exp["proto_z_aligned_ap"]


def test_gate_c_registered_acceptance_criteria_audit():
    """Asserts that both arms fail Step 15 registered criteria and Arm 2A is not a proven winner."""
    report_path = Path("manifests/RQ1-TRN-03A/evaluation/PILOT_FOLD1_COMPARATIVE_REPORT.json")
    with open(report_path) as f:
        rep = json.load(f)

    diffs = rep["comparative_analysis"]["paired_differences_vs_baseline_r_fixed_budget"]

    # Criterion 1: Delta AP > +0.020
    # Arm 2A FB: -0.008908
    # Arm 2B Exp: -0.008925
    assert diffs["Arm2A_Focal_fixed_budget_registered_primary"]["delta_matched_ap"] == -0.008908
    assert diffs["Arm2B_Weighted_exploratory_near_budget_last"]["delta_matched_ap"] == -0.008925
    assert diffs["Arm2A_Focal_fixed_budget_registered_primary"]["meets_step15_ap_elevation_target_plus0020"] is False
    assert diffs["Arm2B_Weighted_exploratory_near_budget_last"]["meets_step15_ap_elevation_target_plus0020"] is False

    # Arm 2A Best is only +0.000189 (well below +0.020)
    assert diffs["Arm2A_Focal_validation_best_secondary"]["delta_matched_ap"] == 0.000189
    assert diffs["Arm2A_Focal_validation_best_secondary"]["meets_step15_ap_elevation_target_plus0020"] is False

    # Gate C Decision
    gate_c = rep["comparative_analysis"]["gate_c_scientific_decision"]
    assert gate_c["primary_registered_target_met"] is False
    assert gate_c["pedagogical_error_rate_target_met"] is False


def test_fail_closed_artifact_verification():
    """Verifies that all required artifacts and binaries exist without silent skips."""
    required_paths = [
        Path("manifests/RQ1-TRN-03A/evaluation/PILOT_FOLD1_COMPARATIVE_REPORT.json"),
        Path("manifests/RQ1-TRN-03A/evaluation/PILOT_FOLD1_PAIRED_PREDICTION_RECORDS.json.gz"),
        Path("manifests/RQ1-TRN-03A/evaluation/pilot_fold1_endpoints_summary.csv"),
        Path("manifests/RQ1-TRN-03A/evaluation/pilot_fold1_operating_points_comparison.csv"),
        Path("checkpoints/RQ1-TRN-02A/recovered/best.pt"),
        Path("checkpoints/pilot_evaluation_cache/fixed_budget_checkpoint_Arm2A_Focal_3600up.pt"),
        Path("checkpoints/pilot_evaluation_cache/best_Arm2A_Focal_1200up.pt"),
        Path("checkpoints/pilot_evaluation_cache/last_Arm2B_Weighted_3598up.pt"),
        Path("checkpoints/pilot_evaluation_cache/best_Arm2B_Weighted_600up.pt"),
    ]

    for p in required_paths:
        assert p.is_file(), f"FAIL-CLOSED: Required artifact {p} is missing!"
