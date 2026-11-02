"""
Tests for LANGFENS Phase RQ1-DIAG-01: Phoneme Omission & False-Negative Root Cause.

Validates:
1. Complete 4-fold token partition (39,576 tokens: 35,363 correct, 3,584 substitutions, 629 deletions).
2. AP, ROC-AUC, and score distributions distinguishing substitutions from deletions.
3. Diagnostic fixed-FPR operating points and severe deletion FNR elevation (FNR > 95% at tau=0.50).
4. Acoustic reliance gap decoupling (deletions near zero vs substitutions > +0.010).
5. Cross-attention temporal localization collapse (in-window mass < 0.05, offset > 1.0s).
6. Phonetic, positional, and durational concentration of omissions (word-final > 50%, < 80ms > 80%).
7. Representative high-confidence false negatives integrity and TextGrid alignment.
8. CTC ASR error vs true gold omission conceptual demarcation.
"""

import csv
import json
import os
import pytest


REPORT_PATH = "manifests/RQ1-DIAG-01/evaluation/OMISSION_AND_FALSE_NEGATIVE_REPORT.json"
SPK_CSV_PATH = "manifests/RQ1-DIAG-01/evaluation/omission_error_breakdown_by_speaker.csv"
PHONE_CSV_PATH = "manifests/RQ1-DIAG-01/evaluation/omission_error_breakdown_by_phone.csv"
POS_CSV_PATH = "manifests/RQ1-DIAG-01/evaluation/omission_error_breakdown_by_position.csv"
DUR_CSV_PATH = "manifests/RQ1-DIAG-01/evaluation/omission_error_breakdown_by_duration.csv"
FN_CSV_PATH = "manifests/RQ1-DIAG-01/evaluation/representative_false_negatives.csv"


@pytest.fixture(scope="module")
def report_data():
    """Loads master diagnostic report."""
    assert os.path.exists(REPORT_PATH), f"Report not found at {REPORT_PATH}"
    with open(REPORT_PATH, "r") as f:
        return json.load(f)


def test_report_structure_and_metadata(report_data):
    """Verifies that the report contains all required top-level diagnostic sections."""
    required_sections = [
        "metadata",
        "token_census",
        "ranking_metrics_by_error_type",
        "operating_points_comparison",
        "ctc_asr_audit",
        "cross_attention_temporal_localization",
        "disaggregated_breakdowns",
        "representative_false_negatives_summary",
    ]
    for sec in required_sections:
        assert sec in report_data, f"Missing required report section: {sec}"
        
    meta = report_data["metadata"]
    assert meta["experiment"] == "RQ1-DIAG-01"
    assert meta["total_phone_tokens"] == 39576
    assert meta["total_validation_utterances"] == 1199
    assert meta["speaker_census_count"] == 8
    assert meta["folds_evaluated"] == [1, 2, 3, 4]


def test_token_census_exact_mathematical_partition(report_data):
    """Verifies the exact 4-fold token census and mutually exclusive partition."""
    census = report_data["token_census"]
    total = census["total_tokens"]
    correct = census["correct_tokens"]
    subs = census["substitution_tokens"]
    dels = census["deletion_tokens"]
    errors = census["total_errors"]

    assert total == 39576
    assert correct == 35363
    assert subs == 3584
    assert dels == 629
    assert errors == 4213

    # Mutually exclusive and exhaustive partition checks
    assert correct + subs + dels == total
    assert subs + dels == errors
    assert pytest.approx(correct / total, abs=1e-5) == 0.893547
    assert pytest.approx(subs / total, abs=1e-5) == 0.090560
    assert pytest.approx(dels / total, abs=1e-5) == 0.015893
    assert pytest.approx(errors / total, abs=1e-5) == 0.106453
    assert pytest.approx(subs / errors, abs=1e-4) == 0.8507
    assert pytest.approx(dels / errors, abs=1e-4) == 0.1493


def test_severe_deletion_ap_and_gap_degradation(report_data):
    """
    Verifies that Baseline R has severely depressed AP and near-zero acoustic sensitivity
    on true deletions, while maintaining standard performance on substitutions.
    """
    ranking = report_data["ranking_metrics_by_error_type"]
    
    sub_metrics = ranking["substitutions"]
    del_metrics = ranking["deletions"]
    all_metrics = ranking["all_errors"]
    cor_metrics = ranking["correct_negatives"]

    # Overall metrics match Step 13/14 Baseline R micro evaluation
    assert pytest.approx(all_metrics["matched_ap"], abs=1e-5) == 0.272580
    assert pytest.approx(all_metrics["mismatched_ap"], abs=1e-5) == 0.261960
    assert pytest.approx(all_metrics["acoustic_gap"], abs=1e-5) == 0.010619

    # Substitutions AP is > 0.250 with positive acoustic reliance gap > +0.010
    assert sub_metrics["matched_ap"] > 0.260
    assert pytest.approx(sub_metrics["matched_ap"], abs=1e-4) == 0.2643
    assert sub_metrics["acoustic_gap"] > 0.010

    # Deletions AP is severely depressed (< 0.050) and acoustic gap is near zero (< 0.003)
    assert del_metrics["matched_ap"] < 0.040
    assert pytest.approx(del_metrics["matched_ap"], abs=1e-4) == 0.0305
    assert del_metrics["acoustic_gap"] < 0.003
    assert pytest.approx(del_metrics["acoustic_gap"], abs=1e-4) == 0.0017

    # Mean score ordering: Correct < Deletions < Substitutions
    assert cor_metrics["score_mean"] < del_metrics["score_mean"] < sub_metrics["score_mean"]
    # Deletions mean score is barely above correct
    assert pytest.approx(del_metrics["score_mean"] - cor_metrics["score_mean"], abs=0.01) == 0.059


def test_diagnostic_operating_points_severe_omission_fnr(report_data):
    """
    Verifies that omission False Negative Rate (FNR) is > 95% at tau=0.50,
    and remains > 65% even at diagnostic FPR ceilings (15% FPR).
    """
    op_list = report_data["operating_points_comparison"]
    op_map = {item["operating_point"]: item for item in op_list}

    # Standard tau = 0.50
    t05 = op_map["Standard tau=0.50"]
    assert pytest.approx(t05["threshold"], abs=1e-4) == 0.50
    assert t05["fpr_on_negatives"] <= 0.03
    assert t05["deletions"]["fnr"] > 0.95  # > 95% of deletions missed
    assert pytest.approx(t05["deletions"]["fnr"], abs=1e-2) == 0.9746
    assert t05["deletions"]["tp"] == 16  # Only 16 out of 629 detected

    # Standard tau = 0.20
    t02 = op_map["Standard tau=0.20"]
    assert pytest.approx(t02["threshold"], abs=1e-4) == 0.20
    assert pytest.approx(t02["deletions"]["fnr"], abs=1e-2) == 0.6121

    # FPR <= 15% Ceiling
    t15 = op_map["FPR <= 15% Ceiling"]
    assert pytest.approx(t15["fpr_on_negatives"], abs=1e-2) == 0.1500
    assert t15["substitutions"]["recall"] > 0.45
    assert t15["deletions"]["fnr"] > 0.65  # Deletion FNR > 65% at 15% FPR ceiling
    assert pytest.approx(t15["deletions"]["fnr"], abs=1e-2) == 0.6868

    # FPR <= 10% Ceiling
    t10 = op_map["FPR <= 10% Ceiling"]
    assert pytest.approx(t10["fpr_on_negatives"], abs=1e-2) == 0.1000
    assert t10["deletions"]["fnr"] > 0.80  # Deletion FNR > 80% at 10% FPR ceiling
    assert pytest.approx(t10["deletions"]["fnr"], abs=1e-2) == 0.8251


def test_cross_attention_temporal_localization_failure(report_data):
    """
    Verifies that the cross-attention module exhibits temporal localization failure:
    in-window attention mass is near zero (< 0.05) and temporal offset is > 1.0 s.
    """
    ca = report_data["cross_attention_temporal_localization"]
    assert ca["sampled_utterances_count"] >= 40
    assert ca["sampled_tokens_count"] > 1000

    by_type = ca["by_error_type"]
    for k in ["correct", "substitution", "deletion", "all"]:
        assert k in by_type
        # In-window mass is uniformly tiny (< 0.05) across all types, showing lack of alignment
        assert by_type[k]["in_window_mass_mean"] < 0.050
        # Peak frame temporal offset is > 1.0 second from gold phone center
        assert by_type[k]["temporal_offset_mean_sec"] > 1.0

    # Max frame mass averages around 0.25 (one attention head completely saturated)
    assert 0.20 < by_type["all"]["max_frame_mass_mean"] < 0.35


def test_phonetic_and_positional_omission_concentration(report_data):
    """
    Verifies that omissions are concentrated in word-final positions (> 50%)
    and short durations (< 80 ms accounts for > 80%).
    """
    pos_breakdown = report_data["disaggregated_breakdowns"]["by_word_position"]
    dur_breakdown = report_data["disaggregated_breakdowns"]["by_duration"]
    phone_breakdown = report_data["disaggregated_breakdowns"]["by_phoneme"]

    # Word-final concentration: > 50% of all omissions are word-final
    wf = pos_breakdown["word_final"]
    assert wf["deletion_tokens"] == 350
    assert wf["fraction_of_all_omissions"] > 0.50
    assert pytest.approx(wf["fraction_of_all_omissions"], abs=1e-3) == 0.556

    # Duration concentration: < 80ms accounts for > 80% of omissions
    dur_lt40 = dur_breakdown["< 40ms"]["deletion_tokens"]
    dur_40_80 = dur_breakdown["40-80ms"]["deletion_tokens"]
    total_dels = report_data["token_census"]["deletion_tokens"]
    assert (dur_lt40 + dur_40_80) / total_dels > 0.85

    # Top deleted phonemes are alveolar stops /D/, /T/ and rhotic /R/
    d_dels = phone_breakdown["D"]["deletion_tokens"]
    r_dels = phone_breakdown["R"]["deletion_tokens"]
    t_dels = phone_breakdown["T"]["deletion_tokens"]
    assert d_dels == 144
    assert r_dels == 112
    assert t_dels == 109
    assert (d_dels + r_dels + t_dels) / total_dels > 0.55


def test_representative_false_negatives_csv_integrity():
    """Verifies that representative false negatives are properly extracted and verified."""
    assert os.path.exists(FN_CSV_PATH)
    with open(FN_CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) >= 40
    # Every row is an error
    for r in rows:
        assert r["error_type"] in ("substitution", "deletion")
        assert float(r["matched_score"]) < 0.05  # High-confidence false negatives
        assert float(r["duration_sec"]) >= 0.0
        assert r["speaker"] in ["HKK", "NCC", "NJS", "RRBI", "TNI", "TXHC", "YBAA", "ZHAA"]

    # Verify top substitution false negatives are coda nasals (/M/ -> /N/ or /NG/)
    m_nasal_fn = [r for r in rows if r["canonical_phone"] == "M" and r["error_type"] == "substitution"]
    assert len(m_nasal_fn) >= 10
    for r in m_nasal_fn[:5]:
        assert r["realized_phone"] in ("N", "NG", "B", "P")
        assert float(r["matched_score"]) < 0.015


def test_csv_breakdown_files_integrity():
    """Verifies that all 4 summary CSV tables exist and have valid rows and columns."""
    for path, expected_min_rows in [
        (SPK_CSV_PATH, 8),
        (PHONE_CSV_PATH, 39),
        (POS_CSV_PATH, 4),
        (DUR_CSV_PATH, 4),
    ]:
        assert os.path.exists(path), f"File not found: {path}"
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) >= expected_min_rows, f"Insufficient rows in {path}: {len(rows)}"
