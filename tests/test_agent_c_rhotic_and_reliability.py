"""
LANGFENS RQ1 — Agent C: Rhotic & Annotation Reliability Regression Tests.
Verifies the existence, numerical closure, and Horvitz-Thompson statistical
consistency of Agent C artifacts.
"""

import json
import os
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORKSTREAM_DIR = os.path.join(ROOT_DIR, 'workstreams', 'agent_c_rhotic')

RHOTIC_JSON = os.path.join(WORKSTREAM_DIR, 'rhotic_summary.json')
ADJUDICATED_JSON = os.path.join(WORKSTREAM_DIR, 'adjudicated_metrics_summary.json')

FILES_TO_CHECK = [
    os.path.join(WORKSTREAM_DIR, 'RHOTIC_ERROR_ANALYSIS.md'),
    os.path.join(ROOT_DIR, 'RHOTIC_ERROR_ANALYSIS.md'),
    os.path.join(WORKSTREAM_DIR, 'ANNOTATION_RELIABILITY_RECONCILIATION.md'),
    os.path.join(ROOT_DIR, 'ANNOTATION_RELIABILITY_RECONCILIATION.md'),
    os.path.join(WORKSTREAM_DIR, 'ADJUDICATED_METRICS.md'),
    os.path.join(ROOT_DIR, 'ADJUDICATED_METRICS.md'),
    RHOTIC_JSON,
    ADJUDICATED_JSON,
    os.path.join(WORKSTREAM_DIR, 'run_rhotic_and_reliability_analysis.py')
]

def test_agent_c_deliverables_exist_and_non_empty():
    for f in FILES_TO_CHECK:
        assert os.path.exists(f), f"Missing Agent C deliverable: {f}"
        assert os.path.getsize(f) > 0, f"Empty Agent C deliverable: {f}"

def test_rhotic_summary_numerical_invariants():
    with open(RHOTIC_JSON) as f:
        data = json.load(f)
        
    assert data['total_rhotic_tokens'] == 421
    assert data['gold_omissions'] == 26
    assert data['gold_correct'] == 363
    assert data['gold_substitutions'] == 32
    assert data['gold_omissions'] + data['gold_correct'] + data['gold_substitutions'] == 421
    
    assert data['omission_alarms'] == 377
    assert data['alarm_rate_pct'] == 89.55
    assert data['TP'] == 15
    assert data['FP_cor'] == 331
    assert data['FP_sub'] == 31
    assert data['FN'] == 11
    assert data['TP'] + data['FN'] == data['gold_omissions']
    assert data['TP'] + data['FP_cor'] + data['FP_sub'] == data['omission_alarms']
    
    # Speaker NJS (Spanish L1)
    njs = data['NJS']
    assert njs['tokens'] == 212
    assert njs['alarms'] == 190
    assert njs['TP'] == 2
    assert njs['FP_cor'] == 179
    assert njs['FP_sub'] == 9
    assert njs['deployed_precision'] == 0.0105
    assert njs['recall'] == 0.6667
    
    # Speaker RRBI (Hindi L1)
    rrbi = data['RRBI']
    assert rrbi['tokens'] == 209
    assert rrbi['alarms'] == 187
    assert rrbi['TP'] == 13
    assert rrbi['FP_cor'] == 152
    assert rrbi['FP_sub'] == 22
    assert rrbi['deployed_precision'] == 0.0695
    assert rrbi['recall'] == 0.5652
    
    # Cross-speaker conservation
    assert njs['tokens'] + rrbi['tokens'] == data['total_rhotic_tokens']
    assert njs['alarms'] + rrbi['alarms'] == data['omission_alarms']
    assert njs['TP'] + rrbi['TP'] == data['TP']
    assert njs['FP_cor'] + rrbi['FP_cor'] == data['FP_cor']
    assert njs['FP_sub'] + rrbi['FP_sub'] == data['FP_sub']

def test_adjudicated_metrics_summary_invariants():
    with open(ADJUDICATED_JSON) as f:
        data = json.load(f)
        
    assert data['total_deployed_alarms'] == 1209
    assert data['original_tp'] == 82
    assert data['original_deployed_precision_pct'] == 6.78
    
    tri = data['tripartite_decomposition']
    assert tri['confirmed_model_false_positives']['count'] == 820.9
    assert tri['confirmed_model_false_positives']['pct'] == 67.9
    assert tri['confirmed_model_false_positives']['std_error'] == 43.8
    
    assert tri['confirmed_annotation_disagreements']['count'] == 285.5
    assert tri['confirmed_annotation_disagreements']['pct'] == 23.62
    assert tri['confirmed_annotation_disagreements']['std_error'] == 43.7
    
    assert tri['confirmed_true_omissions']['count'] == 82.0
    assert tri['confirmed_true_omissions']['pct'] == 6.78
    
    assert tri['acoustic_ambiguity']['count'] == 20.6
    assert tri['acoustic_ambiguity']['pct'] == 1.7
    
    total_decomposed = (
        tri['confirmed_model_false_positives']['count'] +
        tri['confirmed_annotation_disagreements']['count'] +
        tri['confirmed_true_omissions']['count'] +
        tri['acoustic_ambiguity']['count']
    )
    assert abs(total_decomposed - 1209.0) < 1e-4
    
    broad = data['broad_pronunciation_event_horizon']
    assert broad['total_event_detections'] == 382.3
    assert broad['deployed_precision_pct'] == 31.62
    assert broad['ci_95_pct'] == [24.5, 38.74]
