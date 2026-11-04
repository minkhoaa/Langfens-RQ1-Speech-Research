"""
Unit and regression test suite for Agent A — Precision & Decision-Policy Benchmark.
Validates the structural integrity, numerical invariants, policy taxonomy,
screening target boundaries, and byte-identical mirroring of POLICY_COMPARISON.csv,
PRECISION_RECALL_FRONTIER.json, and DECISION_POLICY_BENCHMARK.md across workstream and root.
"""

import csv
import json
import os
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORKSTREAM_DIR = os.path.join(ROOT_DIR, 'workstreams', 'agent_a_policy')

CSV_WS = os.path.join(WORKSTREAM_DIR, 'POLICY_COMPARISON.csv')
CSV_ROOT = os.path.join(ROOT_DIR, 'POLICY_COMPARISON.csv')

JSON_WS = os.path.join(WORKSTREAM_DIR, 'policy_summary.json')

FRONTIER_WS = os.path.join(WORKSTREAM_DIR, 'PRECISION_RECALL_FRONTIER.json')
FRONTIER_ROOT = os.path.join(ROOT_DIR, 'PRECISION_RECALL_FRONTIER.json')

REPORT_MD_WS = os.path.join(WORKSTREAM_DIR, 'DECISION_POLICY_BENCHMARK.md')
REPORT_MD_ROOT = os.path.join(ROOT_DIR, 'DECISION_POLICY_BENCHMARK.md')


def test_deliverable_files_exist_and_match():
    """Verify that all Agent A deliverables exist in both workstream and root and are byte-identical."""
    for ws_path, root_path in [
        (CSV_WS, CSV_ROOT),
        (FRONTIER_WS, FRONTIER_ROOT),
        (REPORT_MD_WS, REPORT_MD_ROOT),
    ]:
        assert os.path.exists(ws_path), f"Missing workstream artifact: {ws_path}"
        assert os.path.exists(root_path), f"Missing root artifact: {root_path}"
        assert os.path.getsize(ws_path) > 0, f"Artifact is empty: {ws_path}"
        with open(ws_path, 'rb') as f1, open(root_path, 'rb') as f2:
            assert f1.read() == f2.read(), f"File mismatch between {ws_path} and {root_path}"


def test_policy_comparison_schema_and_counts():
    """Verify exact count of 25 policies and standard column schema in POLICY_COMPARISON.csv."""
    with open(CSV_WS) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 25, f"Expected exactly 25 policies, got {len(rows)}"

    expected_fields = [
        'policy_name', 'policy_type', 'threshold_desc', 'TP', 'FP_cor', 'FP_sub', 'FN',
        'total_alarms', 'abstentions', 'covered', 'deployed_precision', 'subset_precision',
        'all_gold_recall', 'aligned_recall', 'deployed_f1', 'false_alarms_per_1000',
        'correct_token_fpr', 'NJS_precision', 'NJS_recall', 'RRBI_precision', 'RRBI_recall',
        'd_alarms', 't_alarms', 'r_alarms'
    ]
    assert reader.fieldnames == expected_fields


def test_policy_summary_json_alignment():
    """Verify that POLICY_COMPARISON.csv and policy_summary.json match on every row and value."""
    with open(CSV_WS) as f:
        csv_rows = list(csv.DictReader(f))
    with open(JSON_WS) as f:
        json_rows = json.load(f)

    assert len(csv_rows) == len(json_rows) == 25
    for i, (r_csv, r_json) in enumerate(zip(csv_rows, json_rows)):
        assert r_csv['policy_name'] == r_json['policy_name']
        assert int(r_csv['TP']) == r_json['TP']
        assert int(r_csv['FP_cor']) == r_json['FP_cor']
        assert int(r_csv['FP_sub']) == r_json['FP_sub']
        assert int(r_csv['FN']) == r_json['FN']
        assert int(r_csv['total_alarms']) == r_json['total_alarms']
        assert abs(float(r_csv['deployed_precision']) - r_json['deployed_precision']) < 1e-6
        assert abs(float(r_csv['all_gold_recall']) - r_json['all_gold_recall']) < 1e-6
        assert abs(float(r_csv['false_alarms_per_1000']) - r_json['false_alarms_per_1000']) < 1e-2


def test_baseline_operating_point_invariants():
    """Verify numerical invariants for the primary calibrated baseline (15% CAL ceiling)."""
    with open(CSV_WS) as f:
        rows = {r['policy_name']: r for r in csv.DictReader(f)}

    base = rows['Global CAL 15.0% Ceiling']
    assert int(base['TP']) == 82
    assert int(base['FP_cor']) == 1068
    assert int(base['FP_sub']) == 59
    assert int(base['FN']) == 63
    assert int(base['total_alarms']) == 1209
    assert abs(float(base['deployed_precision']) - 0.067825) < 1e-6
    assert abs(float(base['subset_precision']) - 0.071304) < 1e-6
    assert abs(float(base['all_gold_recall']) - 0.565517) < 1e-6
    assert abs(float(base['aligned_recall']) - 0.598540) < 1e-6
    assert abs(float(base['false_alarms_per_1000']) - 113.2) < 0.1
    assert int(base['d_alarms']) == 436
    assert int(base['t_alarms']) == 320
    assert int(base['r_alarms']) == 377


def test_phoneme_muting_sensitivity_collapse():
    """Verify that muting {D, T, R} causes catastrophic sensitivity collapse."""
    with open(CSV_WS) as f:
        rows = {r['policy_name']: r for r in csv.DictReader(f)}

    mute_dtr = rows['Baseline + Mute {D, T, R}']
    # True Positives collapse from 82 to 2 (97.6% sensitivity loss)
    assert int(mute_dtr['TP']) == 2
    assert int(mute_dtr['FN']) == 143
    assert int(mute_dtr['FP_cor']) == 64
    assert int(mute_dtr['FP_sub']) == 10
    assert int(mute_dtr['total_alarms']) == 76
    assert abs(float(mute_dtr['all_gold_recall']) - 0.013793) < 1e-5
    assert abs(float(mute_dtr['deployed_precision']) - 0.026316) < 1e-5
    assert int(mute_dtr['d_alarms']) == 0
    assert int(mute_dtr['t_alarms']) == 0
    assert int(mute_dtr['r_alarms']) == 0


def test_margin_gating_rhotic_suppression():
    """Verify that margin gating (Delta >= 0.02) successfully purges rhotic alarms."""
    with open(CSV_WS) as f:
        rows = {r['policy_name']: r for r in csv.DictReader(f)}

    margin_02 = rows['Margin Gating: Delta >= 0.02']
    assert int(margin_02['TP']) == 46
    assert int(margin_02['FP_cor']) == 390
    assert int(margin_02['FP_sub']) == 7
    assert int(margin_02['r_alarms']) == 0
    assert int(margin_02['d_alarms']) == 260
    assert int(margin_02['t_alarms']) == 173
    assert abs(float(margin_02['deployed_precision']) - 0.103837) < 1e-5
    assert abs(float(margin_02['all_gold_recall']) - 0.317241) < 1e-5


def test_screening_targets_empirical_boundaries():
    """Verify all screening targets against empirical results."""
    with open(CSV_WS) as f:
        rows = list(csv.DictReader(f))

    # Target 1: Deployed precision >= 30% with recall >= 10% (FAILED across all policies)
    p30_r10 = [r for r in rows if float(r['deployed_precision']) >= 0.30 and float(r['all_gold_recall']) >= 0.10]
    assert len(p30_r10) == 0

    # Target 2: Deployed precision >= 50% (FAILED across all policies)
    p50 = [r for r in rows if float(r['deployed_precision']) >= 0.50]
    assert len(p50) == 0

    # Target 3: Strict precision >= 60% and >= 70% in retrospective frontier (UNATTAINABLE)
    with open(FRONTIER_WS) as f:
        frontier = json.load(f)

    t_op_z = frontier['configurations']['t_op_fixed']['protocol_z']['omission_dedicated']
    targets = t_op_z['retrospective_targets']
    assert targets['0.3']['attainable'] is True
    assert targets['0.3']['tp'] == 4
    assert targets['0.3']['fp'] == 8
    assert targets['0.5']['attainable'] is True
    assert targets['0.5']['tp'] == 1
    assert targets['0.5']['fp'] == 1
    assert targets['0.6']['attainable'] is False
    assert targets['0.7']['attainable'] is False

    # Max retrospective precision for T-OP fixed
    max_p = t_op_z['max_retrospective_precision']
    assert abs(max_p['precision'] - 0.5) < 1e-5
    assert max_p['tp'] == 1
    assert max_p['fp'] == 1


def test_decision_policy_benchmark_report_structure():
    """Verify that DECISION_POLICY_BENCHMARK.md contains all required scientific sections."""
    with open(REPORT_MD_WS) as f:
        content = f.read()

    required_sections = [
        "Executive Summary & Screening Target Verdicts",
        "Evaluation Framework, Full Cohort Accounting & Protocol Closure",
        "The 25 Decision Policies — Taxonomy, Design & Benchmark Table",
        "Deep-Dive Analysis of the 7 Policy Paradigms",
        "Precision–Recall Frontier & Screening Targets Verification",
        "Granular Phonetic & Demographic Forensics",
        "Synthesis, Forward Integration & Multi-Workstream Interface",
        "Provenance, Checksums & Reproducibility Ledger"
    ]

    for sec in required_sections:
        assert sec in content, f"Missing section '{sec}' in DECISION_POLICY_BENCHMARK.md"
