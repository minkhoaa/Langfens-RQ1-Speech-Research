"""
LANGFENS — RQ1: Precision Recovery Experiment 01 Regression Test Suite.
Verifies the existence, structural validity, numerical consistency, and
evaluation contract invariants for all 10 master deliverables and associated artifacts.
"""

import csv
import json
import os
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

MASTER_DELIVERABLES = [
    'DECISION_POLICY_BENCHMARK.md',
    'STOP_ACOUSTIC_VERIFICATION.md',
    'RHOTIC_ERROR_ANALYSIS.md',
    'ANNOTATION_RELIABILITY_RECONCILIATION.md',
    'ACOUSTIC_VERIFIER_BENCHMARK.md',
    'SYSTEM_LEVEL_COMPARISON.json',
    'PRECISION_RECOVERY_COMPARISON.md',
    'PARETO_AND_NONINFERIORITY_ANALYSIS.md',
    'NEXT_RESEARCH_DECISION.md',
    'PROCESS_CHECKPOINT.md'
]

SUBAGENT_ARTIFACTS = [
    'PRECISION_RECALL_FRONTIER.json',
    'POLICY_COMPARISON.csv',
    'STOP_VERIFIER_RESULTS.json',
    'STOP_ERROR_CASES.jsonl',
    'ADJUDICATED_METRICS.md',
    'VERIFIER_ABLATION_REPORT.md',
    'SHARED_EVIDENCE_MANIFEST.json'
]

def test_all_ten_master_deliverables_exist_and_non_empty():
    for f in MASTER_DELIVERABLES:
        p = os.path.join(ROOT_DIR, f)
        assert os.path.exists(p), f"Missing master deliverable: {f}"
        assert os.path.getsize(p) > 1000, f"Master deliverable unexpectedly small: {f} ({os.path.getsize(p)} bytes)"

def test_subagent_artifacts_exist_and_non_empty():
    for f in SUBAGENT_ARTIFACTS:
        p = os.path.join(ROOT_DIR, f)
        assert os.path.exists(p), f"Missing subagent artifact: {f}"
        assert os.path.getsize(p) > 0, f"Artifact is empty: {f}"

def test_shared_evidence_manifest_integrity():
    p = os.path.join(ROOT_DIR, 'SHARED_EVIDENCE_MANIFEST.json')
    with open(p) as f:
        manifest = json.load(f)
    assert manifest['schema_version'] == 'rq1-precision-recovery-v1'
    assert manifest['experiment_id'] == 'LANGFENS-RQ1-PRECISION-RECOVERY-01'
    base = manifest['registered_baseline']
    assert base['dedicated_omission_ap'] == 0.098560
    assert base['deployed_omission_precision'] == 0.067825
    assert base['all_gold_omission_recall'] == 0.565517
    assert base['binary_ap'] == 0.318619
    assert base['legacy_substitution_ap'] == 0.308974
    assert base['greedy_per'] == 0.241980

def test_policy_comparison_csv_structure_and_closure():
    p = os.path.join(ROOT_DIR, 'POLICY_COMPARISON.csv')
    with open(p) as f:
        reader = list(csv.DictReader(f))
    assert len(reader) >= 20, f"Expected at least 20 policy evaluations, got {len(reader)}"
    
    # Check baseline row
    baseline_row = next(r for r in reader if '15.0% Ceiling' in r['policy_name'])
    assert int(baseline_row['TP']) == 82
    assert int(baseline_row['FP_cor']) == 1068
    assert int(baseline_row['FP_sub']) == 59
    assert float(baseline_row['deployed_precision']) == pytest.approx(0.067825, abs=1e-4)
    assert float(baseline_row['all_gold_recall']) == pytest.approx(0.565517, abs=1e-4)

def test_system_level_comparison_json_closure():
    p = os.path.join(ROOT_DIR, 'SYSTEM_LEVEL_COMPARISON.json')
    with open(p) as f:
        data = json.load(f)
    assert 'primary_configurations' in data
    assert 'ablation_suite' in data
    cfgs = data['primary_configurations']
    assert len(cfgs) == 5
    
    # Invariance check: Upstream Multi-Class MDD metrics must be invariant across all configurations
    for cfg in cfgs:
        assert cfg['binary_ap'] == pytest.approx(0.318619, abs=1e-6)
        assert cfg['substitution_ap'] == pytest.approx(0.308974, abs=1e-6)
        assert cfg['greedy_per'] == pytest.approx(0.241980, abs=1e-6)

def test_stop_verifier_results_integrity():
    p = os.path.join(ROOT_DIR, 'STOP_VERIFIER_RESULTS.json')
    with open(p) as f:
        data = json.load(f)
    assert len(data) >= 8
    
    # Check baseline candidate
    base_cand = next(c for c in data if 'Baseline' in c['candidate_name'])
    assert base_cand['stop_cohort_TP'] == 65
    assert base_cand['stop_cohort_FP_cor'] == 673

def test_pareto_dominance_of_acoustic_verifier():
    """Acoustic verifier must be strictly Pareto-dominated by Global CAL 10% on original gold."""
    p_csv = os.path.join(ROOT_DIR, 'POLICY_COMPARISON.csv')
    with open(p_csv) as f:
        policies = list(csv.DictReader(f))
    cal_10 = next(r for r in policies if '10.0% Ceiling' in r['policy_name'])
    
    p_sys = os.path.join(ROOT_DIR, 'SYSTEM_LEVEL_COMPARISON.json')
    with open(p_sys) as f:
        sys_data = json.load(f)
    verifier = next(c for c in sys_data['primary_configurations'] if 'Acoustic Verification Candidate' in c['config_name'])
    
    # CAL 10% must dominate Verifier
    assert float(cal_10['deployed_precision']) > verifier['deployed_precision'] # 9.10% > 6.31%
    assert float(cal_10['all_gold_recall']) > verifier['all_gold_recall']       # 41.38% > 39.31%
    assert float(cal_10['false_alarms_per_1000']) < verifier['false_alarms_per_1000'] # 60.16 < 85.07
