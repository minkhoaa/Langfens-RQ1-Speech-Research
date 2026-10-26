import json
import pytest
import numpy as np
from pathlib import Path
from audit_scripts.evaluate_fold1_ctc_binary import (
    average_precision_binary,
    calculate_binary_metrics,
    calculate_ctc_metrics,
    compute_levenshtein_edits
)

PRED_RECORDS_PATH = Path("manifests/RQ1-TRN-02A/evaluation_reconciliation/corrections/PAIRED_PREDICTION_RECORDS.json")


def test_prediction_records_schema_and_reproducibility():
    if not PRED_RECORDS_PATH.is_file():
        pytest.skip(f"Prediction records artifact not found at {PRED_RECORDS_PATH}")

    with open(PRED_RECORDS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Metadata schema checks
    assert "metadata" in data
    meta = data["metadata"]
    assert meta["split"] == "val"
    assert meta["fold"] == 1
    assert meta["total_utterances"] == 300
    assert meta["checkpoint_sha256"] == "a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b"

    utterances = data["utterances"]
    assert len(utterances) == 300

    # 2. Reconstruct Binary Metrics
    all_labels = []
    all_matched_scores = []
    all_mismatched_scores = []
    all_ctc_refs = []
    all_ctc_hyps = []

    njs_labels, njs_matched_scores, njs_mismatched_scores = [], [], []
    rrbi_labels, rrbi_matched_scores, rrbi_mismatched_scores = [], [], []

    for u in utterances:
        # Check required fields per utterance
        assert "record_id" in u
        assert "speaker" in u
        assert "utterance_id" in u
        assert "mismatched_audio_record_id" in u
        assert u["record_id"] != u["mismatched_audio_record_id"], "Self-match detected in permutation partner!"

        lbls = u["binary_reference_labels"]
        m_sc = u["matched_binary_scores"]
        mm_sc = u["mismatched_binary_scores"]

        assert len(lbls) == len(m_sc) == len(mm_sc), "Length mismatch across labels and scores"

        all_labels.extend(lbls)
        all_matched_scores.extend(m_sc)
        all_mismatched_scores.extend(mm_sc)

        if u["speaker"] == "NJS":
            njs_labels.extend(lbls)
            njs_matched_scores.extend(m_sc)
            njs_mismatched_scores.extend(mm_sc)
        elif u["speaker"] == "RRBI":
            rrbi_labels.extend(lbls)
            rrbi_matched_scores.extend(m_sc)
            rrbi_mismatched_scores.extend(mm_sc)

        all_ctc_refs.append(u["matched_ctc_reference"])
        all_ctc_hyps.append(u["matched_ctc_hypothesis"])

    # Verify overall token count: exactly 9,956 valid tokens
    assert len(all_labels) == 9956
    assert sum(all_labels) == 1157

    # 3. Verify AP Reconstructions
    matched_ap = average_precision_binary(all_labels, all_matched_scores)
    mismatched_ap = average_precision_binary(all_labels, all_mismatched_scores)
    assert pytest.approx(matched_ap, abs=1e-5) == 0.317021
    assert pytest.approx(mismatched_ap, abs=1e-5) == 0.294722

    # Per-speaker AP
    njs_ap = average_precision_binary(njs_labels, njs_matched_scores)
    njs_mm_ap = average_precision_binary(njs_labels, njs_mismatched_scores)
    rrbi_ap = average_precision_binary(rrbi_labels, rrbi_matched_scores)
    rrbi_mm_ap = average_precision_binary(rrbi_labels, rrbi_mismatched_scores)

    assert pytest.approx(njs_ap, abs=1e-5) == 0.394927
    assert pytest.approx(njs_mm_ap, abs=1e-5) == 0.378744
    assert pytest.approx(rrbi_ap, abs=1e-5) == 0.251519
    assert pytest.approx(rrbi_mm_ap, abs=1e-5) == 0.234979

    # 4. Verify Standard 0.50 Confusion Matrix
    bin_metrics = calculate_binary_metrics(all_labels, all_matched_scores, thresholds=[0.50])
    op05 = bin_metrics["operating_point_standard_05"]
    assert op05["tp"] == 201
    assert op05["fp"] == 208
    assert op05["fn"] == 956
    assert op05["tn"] == 8591

    # 5. Verify CTC Recognition Metrics
    total_ref = sum(len(r) for r in all_ctc_refs)
    total_hyp = sum(len(h) for h in all_ctc_hyps)
    assert total_ref == 9881
    assert total_hyp == 9652

    tot_edits, subs, dels, inss = 0, 0, 0, 0
    for r, h in zip(all_ctc_refs, all_ctc_hyps):
        e, s, d, i = compute_levenshtein_edits(r, h)
        tot_edits += e
        subs += s
        dels += d
        inss += i

    assert tot_edits == 2340
    assert subs == 1707
    assert dels == 431
    assert inss == 202
    assert pytest.approx(tot_edits / total_ref, abs=1e-5) == 0.236818

    # 6. Verify Paired Difference and Agreement Statistics
    s_m = np.array(all_matched_scores)
    s_mm = np.array(all_mismatched_scores)
    diff = s_m - s_mm
    abs_diff = np.abs(diff)

    assert pytest.approx(float(np.mean(diff)), abs=1e-5) == -0.002053
    assert pytest.approx(float(np.median(diff)), abs=1e-5) == -0.001932
    assert pytest.approx(float(np.mean(abs_diff)), abs=1e-5) == 0.029648

    agree_05 = float(((s_m >= 0.5) == (s_mm >= 0.5)).mean())
    agree_02 = float(((s_m >= 0.2) == (s_mm >= 0.2)).mean())
    assert pytest.approx(agree_05, abs=1e-4) == 0.9676
    assert pytest.approx(agree_02, abs=1e-4) == 0.9230
