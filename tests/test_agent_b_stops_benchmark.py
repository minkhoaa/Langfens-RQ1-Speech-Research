"""
Unit and regression test suite for Agent B — Stop-Consonant Acoustic Verification.
Validates the structural integrity, numerical invariants, error case taxonomy,
and byte-identical mirroring of STOP_VERIFIER_RESULTS.json, STOP_ERROR_CASES.jsonl,
and STOP_ACOUSTIC_VERIFICATION.md across workstream and root locations.
"""

import json
import os
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORKSTREAM_DIR = os.path.join(ROOT_DIR, 'workstreams', 'agent_b_stops')

RESULTS_JSON_WS = os.path.join(WORKSTREAM_DIR, 'STOP_VERIFIER_RESULTS.json')
RESULTS_JSON_ROOT = os.path.join(ROOT_DIR, 'STOP_VERIFIER_RESULTS.json')

ERROR_CASES_WS = os.path.join(WORKSTREAM_DIR, 'STOP_ERROR_CASES.jsonl')
ERROR_CASES_ROOT = os.path.join(ROOT_DIR, 'STOP_ERROR_CASES.jsonl')

REPORT_MD_WS = os.path.join(WORKSTREAM_DIR, 'STOP_ACOUSTIC_VERIFICATION.md')
REPORT_MD_ROOT = os.path.join(ROOT_DIR, 'STOP_ACOUSTIC_VERIFICATION.md')


def test_deliverable_files_exist_and_match():
    """Verify that all Agent B deliverables exist in both workstream and root and are byte-identical."""
    for ws_path, root_path in [
        (RESULTS_JSON_WS, RESULTS_JSON_ROOT),
        (ERROR_CASES_WS, ERROR_CASES_ROOT),
        (REPORT_MD_WS, REPORT_MD_ROOT),
    ]:
        assert os.path.exists(ws_path), f"Missing workstream artifact: {ws_path}"
        assert os.path.exists(root_path), f"Missing root artifact: {root_path}"
        assert os.path.getsize(ws_path) > 0, f"Artifact is empty: {ws_path}"
        with open(ws_path, 'rb') as f1, open(root_path, 'rb') as f2:
            assert f1.read() == f2.read(), f"File mismatch between {ws_path} and {root_path}"


def test_stop_verifier_results_schema_and_candidates():
    """Verify exact candidate counts and names in STOP_VERIFIER_RESULTS.json."""
    with open(RESULTS_JSON_WS) as f:
        results = json.load(f)

    assert len(results) == 8, f"Expected 8 candidates, found {len(results)}"
    
    cand_names = [r['candidate_name'] for r in results]
    assert "Original T-OP Baseline" in cand_names[0]
    assert "Global Conservative Threshold" in cand_names[1]
    assert "Phoneme-Identity Suppression" in cand_names[2]
    assert "Candidate B1: Burst Detector Only" in cand_names[3]
    assert "Candidate B2: Duration & Energy Gate" in cand_names[4]
    assert "Candidate B3: High-Energy Voiced Closure Gate" in cand_names[5]
    assert "Candidate B4: Multi-Feature Stop Verifier (Primary)" in cand_names[6]
    assert "Candidate B5: Conservative Stop Verifier (High Specificity)" in cand_names[7]


def test_stop_verifier_baseline_invariants():
    """Verify numerical invariants for Original T-OP Baseline."""
    with open(RESULTS_JSON_WS) as f:
        results = json.load(f)

    base = results[0]
    assert base['stop_cohort_TP'] == 65
    assert base['stop_cohort_FP_cor'] == 673
    assert base['stop_cohort_FP_sub'] == 18
    assert base['stop_cohort_FN'] == 16
    assert base['stop_total_alarms'] == 756
    assert base['stop_filtered_FP_cor'] == 0
    assert base['stop_filtered_true_deletions'] == 0
    assert abs(base['stop_deployed_precision'] - 0.085979) < 1e-5
    assert abs(base['stop_recall'] - 0.802469) < 1e-5
    assert abs(base['stop_f1'] - 0.155317) < 1e-5


def test_stop_verifier_candidate_b4_invariants():
    """Verify numerical invariants for Candidate B4 (Multi-Feature Stop Verifier)."""
    with open(RESULTS_JSON_WS) as f:
        results = json.load(f)

    b4 = results[6]
    assert b4['candidate_name'] == "Candidate B4: Multi-Feature Stop Verifier (Primary)"
    assert b4['stop_cohort_TP'] == 41
    assert b4['stop_cohort_FP_cor'] == 476
    assert b4['stop_cohort_FP_sub'] == 9
    assert b4['stop_cohort_FN'] == 40
    assert b4['stop_total_alarms'] == 526
    assert b4['stop_filtered_FP_cor'] == 197
    assert b4['stop_filtered_true_deletions'] == 24
    assert abs(b4['stop_deployed_precision'] - 0.077947) < 1e-5
    assert abs(b4['stop_recall'] - 0.506173) < 1e-5
    assert abs(b4['stop_f1'] - 0.135091) < 1e-5
    assert abs(b4['stop_fp_reduction_pct'] - 29.27) < 1e-2

    # Precision destruction check: B4 precision < Baseline precision
    base = results[0]
    assert b4['stop_deployed_precision'] < base['stop_deployed_precision']
    # Recall loss check: B4 recall < Baseline recall
    assert b4['stop_recall'] < base['stop_recall']


def test_stop_error_cases_taxonomy_and_counts():
    """Verify exact counts and acoustic attributes in STOP_ERROR_CASES.jsonl."""
    with open(ERROR_CASES_WS) as f:
        cases = [json.loads(line) for line in f]

    assert len(cases) == 200, f"Expected exactly 200 exported error cases, found {len(cases)}"

    surv_fp = [c for c in cases if c['type'] == 'surviving_false_positive_correct']
    supp_tp = [c for c in cases if c['type'] == 'mistakenly_suppressed_true_deletion']

    assert len(surv_fp) == 177
    assert len(supp_tp) == 23

    # Check phone breakdown in surviving FP
    assert sum(1 for c in surv_fp if c['phone'] == 'D') == 104
    assert sum(1 for c in surv_fp if c['phone'] == 'T') == 73
    # Surviving FPs must have no detected burst (otherwise B4 would have suppressed them)
    assert all(not c['features']['has_burst'] for c in surv_fp)

    # Check phone breakdown in suppressed TP
    assert sum(1 for c in supp_tp if c['phone'] == 'D') == 15
    assert sum(1 for c in supp_tp if c['phone'] == 'T') == 8
    # Suppressed TPs have burst detected in 9 cases
    burst_count = sum(1 for c in supp_tp if c['features']['has_burst'])
    assert burst_count == 9


def test_stop_acoustic_verification_report_structure():
    """Verify that STOP_ACOUSTIC_VERIFICATION.md contains all required scientific sections."""
    with open(REPORT_MD_WS) as f:
        content = f.read()

    required_sections = [
        "Executive Summary & Core Scientific Question",
        "Theoretical Acoustics of Coronal Stops in L2 Speech",
        "Quantitative Results Across All 8 Stop Verifier Candidates",
        "Physical Acoustic Feature Definitions & Measurement Methodology",
        "Quantitative Acoustic Distributions Across Validation Cohorts",
        "Detailed Failure Mode Forensics: Why Stop Verification Fails",
        "Failure Mode 1: The Acoustic Bleeding / Coarticulation Paradox",
        "Failure Mode 2: The Unreleased Closure Stealth Paradox",
        "Failure Mode 3: The Asymmetric Precision-Destruction Math",
        "Forensic Case Dossier from `STOP_ERROR_CASES.jsonl`",
        "Breakdown by Phoneme Identity, Speaker Accent, and Word Position",
        "Architectural & Theoretical Implications for MDD Research",
        "Deliverable Artifact Registry & Verification Signatures"
    ]

    for sec in required_sections:
        assert sec in content, f"Missing section '{sec}' in STOP_ACOUSTIC_VERIFICATION.md"
