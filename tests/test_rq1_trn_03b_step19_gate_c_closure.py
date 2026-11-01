"""
LANGFENS — RQ1-TRN-03B: Step 19 Gate C Scientific Closure Tests.
Senior Speech ML Research Engineer & Reproducibility Auditor.

Verifies:
  1. Distinction between bootstrap empirical sign frequency P_boot(ΔAP > 0) and permutation hypothesis test p-value.
  2. Demarcation between Fold 1 pilot screening failure and unmeasured 4-fold macro acceptance criteria.
  3. Formal disclosure of fixed-FPR operating points as validation-derived diagnostic thresholds.
  4. Exact evaluator provenance: introducing source commit b4b1ee8 vs runtime working-tree base 841c1f4.
  5. Exact numerical reproduction of AP, ROC-AUC, and operating points from archived prediction records (< 1e-5).
  6. Formal Gate C decision disposition: STOP loss-only Stage 2 for compute efficiency and delineate next research options.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import pytest
import numpy as np
from scipy.stats import rankdata

from audit_scripts.evaluate_fold1_ctc_binary import average_precision_binary
from audit_scripts.evaluate_step18_pilot_correction import compute_operating_points_fixed_fpr


REPORT_PATH = Path("manifests/RQ1-TRN-03A/evaluation/PILOT_FOLD1_COMPARATIVE_REPORT.json")
PRED_GZ_PATH = Path("manifests/RQ1-TRN-03A/evaluation/PILOT_FOLD1_PAIRED_PREDICTION_RECORDS.json.gz")


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=bool)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = int((~y_true).sum())
    ranks = rankdata(y_score)
    r_pos = ranks[y_true].sum()
    u = r_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def test_bootstrap_sign_frequency_vs_permutation_p_value():
    """Proves P_boot(Delta > 0) is a bootstrap sign frequency and permutation test confirms non-significance under H0."""
    assert REPORT_PATH.is_file(), f"Missing {REPORT_PATH}"
    with open(REPORT_PATH) as f:
        rep = json.load(f)

    bs_dict = rep["comparative_analysis"]["utterance_level_bootstrap_uncertainty"]

    # Arm 2A Fixed-Budget
    a_bs = bs_dict["Arm2A_Focal_fixed_budget_registered_primary"]
    assert "bootstrap_positive_fraction" in a_bs
    assert "permutation_test_two_tailed_p" in a_bs
    assert pytest.approx(a_bs["bootstrap_positive_fraction"], abs=1e-3) == 0.0560
    # Two-tailed permutation test p-value under H0: Delta = 0 is non-significant (p = 0.4790 > 0.05)
    assert pytest.approx(a_bs["permutation_test_two_tailed_p"], abs=1e-3) == 0.4790
    assert a_bs["permutation_test_two_tailed_p"] > 0.05

    # Arm 2A Validation-Best
    a_best_bs = bs_dict["Arm2A_Focal_validation_best_secondary"]
    assert pytest.approx(a_best_bs["bootstrap_positive_fraction"], abs=1e-3) == 0.5100
    assert pytest.approx(a_best_bs["permutation_test_two_tailed_p"], abs=1e-3) == 0.9930

    # Arm 2B Exploratory
    b_bs = bs_dict["Arm2B_Weighted_exploratory_near_budget_last"]
    assert pytest.approx(b_bs["bootstrap_positive_fraction"], abs=1e-3) == 0.0380
    assert pytest.approx(b_bs["permutation_test_two_tailed_p"], abs=1e-3) == 0.6190

    # Arm 2B Validation-Best
    b_best_bs = bs_dict["Arm2B_Weighted_validation_best_secondary"]
    assert pytest.approx(b_best_bs["bootstrap_positive_fraction"], abs=1e-3) == 0.4510
    assert pytest.approx(b_best_bs["permutation_test_two_tailed_p"], abs=1e-3) == 0.9910


def test_four_fold_macro_criteria_status_unmeasured():
    """Verifies that 4-fold macro criteria are explicitly designated as UNMEASURED rather than confirmed failed."""
    with open(REPORT_PATH) as f:
        rep = json.load(f)

    gate_c = rep["comparative_analysis"]["gate_c_scientific_decision"]
    assert gate_c["four_fold_macro_criteria_status"] == "UNMEASURED_DUE_TO_PILOT_SCREENING_HALT"
    assert "STOP" in gate_c["decision"]


def test_validation_derived_diagnostic_thresholds_disclosure():
    """Verifies that fixed-FPR operating points are formally documented as diagnostic thresholds."""
    with open(REPORT_PATH) as f:
        rep = json.load(f)

    meta = rep["comparative_analysis"]["metadata"]
    assert "diagnostic_thresholds_note" in meta
    note = meta["diagnostic_thresholds_note"]
    assert "validation-derived diagnostic thresholds" in note
    assert "NOT as independently calibrated deployment thresholds" in note


def test_evaluator_source_commit_and_runtime_provenance():
    """Verifies evaluator introducing commit b4b1ee8 and runtime execution SHA 841c1f4."""
    with open(REPORT_PATH) as f:
        rep = json.load(f)

    meta = rep["comparative_analysis"]["metadata"]
    assert meta["evaluator_source_commit"] == "b4b1ee8"
    assert meta["runtime_git_sha_at_generation"].startswith("841c1f4")
    assert meta["evaluator_script_path"] == "audit_scripts/evaluate_step18_pilot_correction.py"
    assert "dirty_uncommitted_at_execution; certified_in_commit_b4b1ee8" in meta["working_tree_state"]

    with gzip.open(PRED_GZ_PATH, "rt") as f:
        gz_data = json.load(f)
    gz_meta = gz_data["metadata"]
    assert gz_meta["evaluator_source_commit"] == "b4b1ee8"
    assert gz_meta["runtime_git_sha_at_generation"].startswith("841c1f4")


def test_archived_predictions_reproduce_exact_metrics():
    """Verifies that unrounded scores in the prediction archive reproduce report metrics within 1e-5."""
    with open(REPORT_PATH) as f:
        report = json.load(f)

    with gzip.open(PRED_GZ_PATH, "rt") as f:
        gz_data = json.load(f)

    utts = gz_data["utterances"]
    labels = np.array([lbl for u in utts for lbl in u["labels"]])

    targets = [
        "Baseline_R_fixed_budget_registered_primary",
        "Arm2A_Focal_fixed_budget_registered_primary",
        "Arm2A_Focal_validation_best_secondary",
        "Arm2B_Weighted_exploratory_near_budget_last",
        "Arm2B_Weighted_validation_best_secondary",
    ]

    for t in targets:
        m_scores = np.array([sc for u in utts for sc in u["models"][t]["matched_scores"]])
        mm_scores = np.array([sc for u in utts for sc in u["models"][t]["mismatched_scores"]])

        calc_ap_m = average_precision_binary(labels, m_scores)
        calc_ap_mm = average_precision_binary(labels, mm_scores)
        calc_gap = calc_ap_m - calc_ap_mm
        calc_roc = roc_auc(labels, m_scores)

        rep_ap_m = report[t]["matched_ap"]
        rep_ap_mm = report[t]["mismatched_ap"]
        rep_gap = report[t]["acoustic_gap"]
        rep_roc = report[t]["roc_auc_matched"]

        assert abs(calc_ap_m - rep_ap_m) < 1e-5, f"Matched AP mismatch for {t}"
        assert abs(calc_ap_mm - rep_ap_mm) < 1e-5, f"Mismatched AP mismatch for {t}"
        assert abs(calc_gap - rep_gap) < 1e-5, f"Acoustic gap mismatch for {t}"
        assert abs(calc_roc - rep_roc) < 1e-5, f"ROC-AUC mismatch for {t}"

        # Operating points integer confusion count check
        calc_op = compute_operating_points_fixed_fpr(labels, m_scores, [0.10, 0.15, 0.20])
        rep_op = report[t]["operating_points_fixed_fpr"]
        for c in ["fpr_ceiling_10", "fpr_ceiling_15", "fpr_ceiling_20"]:
            assert calc_op[c]["tp"] == rep_op[c]["tp"]
            assert calc_op[c]["fp"] == rep_op[c]["fp"]
            assert calc_op[c]["fn"] == rep_op[c]["fn"]
            assert calc_op[c]["tn"] == rep_op[c]["tn"]


def test_gate_c_scientific_disposition_stop_recommendation():
    """Verifies that Gate C formally recommends STOP of Stage 2 and records next research options."""
    with open(REPORT_PATH) as f:
        rep = json.load(f)

    gate_c = rep["comparative_analysis"]["gate_c_scientific_decision"]
    assert gate_c["decision"] == "STOP_STAGE_2_FOR_COMPUTE_EFFICIENCY_UNPROVEN_EXPLORATORY_STATUS"
    assert len(gate_c["next_research_design_options"]) == 3
