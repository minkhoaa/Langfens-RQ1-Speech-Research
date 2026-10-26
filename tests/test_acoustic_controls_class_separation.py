import json
import pytest
import numpy as np
from pathlib import Path

RECORDS_PATH = Path("manifests/RQ1-TRN-02A/evaluation_reconciliation/corrections/PAIRED_PREDICTION_RECORDS.json")
REPORT_PATH = Path("manifests/RQ1-TRN-02A/evaluation_reconciliation/INDEPENDENT_VALIDATION_REPLAY.json")


def test_acoustic_controls_class_separated_metrics():
    """Verifies that decision agreement and score shifts are disaggregated by ground truth class."""
    if not RECORDS_PATH.is_file():
        pytest.skip(f"Prediction records artifact not found at {RECORDS_PATH}")

    with open(RECORDS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_labels = []
    all_matched_scores = []
    all_mismatched_scores = []

    for u in data["utterances"]:
        all_labels.extend(u["binary_reference_labels"])
        all_matched_scores.extend(u["matched_binary_scores"])
        all_mismatched_scores.extend(u["mismatched_binary_scores"])

    y = np.array(all_labels, dtype=np.int64)
    s_m = np.array(all_matched_scores, dtype=np.float64)
    s_mm = np.array(all_mismatched_scores, dtype=np.float64)

    assert len(y) == 9956
    pos_mask = (y == 1)
    neg_mask = (y == 0)

    assert pos_mask.sum() == 1157
    assert neg_mask.sum() == 8799

    # 1. Positive Class (True Errors)
    agree_05_pos = float(((s_m[pos_mask] >= 0.5) == (s_mm[pos_mask] >= 0.5)).mean())
    agree_02_pos = float(((s_m[pos_mask] >= 0.2) == (s_mm[pos_mask] >= 0.2)).mean())
    assert pytest.approx(agree_05_pos, abs=1e-4) == 0.8660
    assert pytest.approx(agree_02_pos, abs=1e-4) == 0.9118

    tp_matched = int((s_m[pos_mask] >= 0.5).sum())
    tp_mismatched = int((s_mm[pos_mask] >= 0.5).sum())
    assert tp_matched == 201
    assert tp_mismatched == 172
    # Matched audio detects 29 more true errors (+16.86%) than mismatched audio
    assert (tp_matched - tp_mismatched) == 29

    flips_pos = int(((s_m[pos_mask] >= 0.5) != (s_mm[pos_mask] >= 0.5)).sum())
    assert flips_pos == 155

    # 2. Negative Class (True Corrects)
    agree_05_neg = float(((s_m[neg_mask] >= 0.5) == (s_mm[neg_mask] >= 0.5)).mean())
    agree_02_neg = float(((s_m[neg_mask] >= 0.2) == (s_mm[neg_mask] >= 0.2)).mean())
    assert pytest.approx(agree_05_neg, abs=1e-4) == 0.9809
    assert pytest.approx(agree_02_neg, abs=1e-4) == 0.9244

    fp_matched = int((s_m[neg_mask] >= 0.5).sum())
    fp_mismatched = int((s_mm[neg_mask] >= 0.5).sum())
    assert fp_matched == 208
    assert fp_mismatched == 250
    # Mismatched audio produces 42 additional false positives (+20.19%)
    assert (fp_mismatched - fp_matched) == 42

    # 3. Overall vs Class-Separated Comparison
    agree_05_overall = float(((s_m >= 0.5) == (s_mm >= 0.5)).mean())
    assert pytest.approx(agree_05_overall, abs=1e-4) == 0.9676

    # Verify that overall agreement is dominated by the negative class
    weighted_agreement = (agree_05_pos * 1157 + agree_05_neg * 8799) / 9956
    assert pytest.approx(agree_05_overall, abs=1e-5) == weighted_agreement


def test_four_permutation_results_preserved():
    """Verifies that the four distinct permutation AP values are permanently preserved."""
    if not REPORT_PATH.is_file():
        pytest.skip(f"Report artifact not found at {REPORT_PATH}")

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    perms = data["acoustic_contribution_controls"]["permutation_sensitivity"]
    assert "shift_1" in perms
    assert "shift_2" in perms
    assert "shift_75" in perms
    assert "reverse" in perms

    assert perms["shift_1"]["overall_ap"] == 0.294722
    assert perms["shift_2"]["overall_ap"] == 0.302581
    assert perms["shift_75"]["overall_ap"] == 0.300112
    assert perms["reverse"]["overall_ap"] == 0.314003

    assert perms["shift_1"]["delta_ap_over_matched"] == 0.022299
    assert perms["shift_2"]["delta_ap_over_matched"] == 0.01444
    assert perms["shift_75"]["delta_ap_over_matched"] == 0.016909
    assert perms["reverse"]["delta_ap_over_matched"] == 0.003018

    # Reverse anomaly on NJS: mismatched AP exceeds matched AP
    assert perms["reverse"]["njs_ap"] == 0.404422
    matched_njs_ap = data["per_speaker_validation_breakdown"]["NJS"]["binary_ap"]
    assert perms["reverse"]["njs_ap"] > matched_njs_ap
