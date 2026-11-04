"""
LANGFENS — RQ1: Precision & Dataset Reliability Investigation Regression Test Suite.
Verifies the integrity, blindness constraints, full cohort accounting,
and numerical consistency of all 11 deliverables.
"""

import json
import os
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

MANIFEST_PATH = os.path.join(ROOT_DIR, 'CASE_MANIFEST.json')
FRONTIER_PATH = os.path.join(ROOT_DIR, 'PRECISION_RECALL_FRONTIER.json')
LEDGER_PATH = os.path.join(ROOT_DIR, 'ANNOTATION_REVIEW_LEDGER.jsonl')

REQUIRED_DELIVERABLES = [
    'PRECISION_BENCHMARK.md',
    'PRECISION_RECALL_FRONTIER.json',
    'FALSE_POSITIVE_FORENSICS.md',
    'CASE_MANIFEST.json',
    'BLINDED_REANNOTATION_PROTOCOL.md',
    'ANNOTATION_REVIEW_LEDGER.jsonl',
    'GOLD_LABEL_DISAGREEMENT_REPORT.md',
    'ACOUSTIC_EVIDENCE_ANALYSIS.md',
    'ORIGINAL_VS_ADJUDICATED_METRICS.md',
    'NEXT_EXPERIMENT_DECISION.md',
    'PROCESS_CHECKPOINT.md'
]

def test_all_eleven_deliverables_exist():
    for f in REQUIRED_DELIVERABLES:
        p = os.path.join(ROOT_DIR, f)
        assert os.path.exists(p), f"Missing required deliverable: {f}"
        assert os.path.getsize(p) > 0, f"Deliverable is empty: {f}"

def test_case_manifest_structure_and_closure():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
        
    assert manifest['manifest_version'] == '1.0.0'
    cases = manifest['cases']
    assert len(cases) == 9993, f"Expected 9993 total records, got {len(cases)}"
    
    aligned = [c for c in cases if c['mapping_status'] == 'aligned']
    abstentions = [c for c in cases if c['mapping_status'] == 'unmapped_gold_abstention']
    extras = [c for c in cases if c['mapping_status'] == 'prompt_extra']
    
    assert len(aligned) == 9919
    assert len(abstentions) == 37
    assert len(extras) == 37
    assert len(aligned) + len(abstentions) == 9956, "Total gold positions must equal 9956"

def test_blinded_presentation_invariants():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
        
    for c in manifest['cases']:
        b = c['blinded_case']
        # Must NOT contain model scores or outcomes
        assert 's_omit' not in b
        assert 's_error' not in b
        assert 's_sub' not in b
        assert 'probabilities' not in b
        assert 'outcome' not in b
        assert 'gold' not in b
        assert 'operation_target' not in b
        assert 'binary_target' not in b
        # Must contain prompt context
        assert 'canonical_phone' in b
        assert 'word_text' in b
        assert 'prompt_sentence' in b

def test_sampling_design_and_ledger_alignment():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    with open(LEDGER_PATH) as f:
        ledger = [json.loads(line) for line in f]
        
    assert len(ledger) == 230
    sampled_manifest_cases = [c for c in manifest['cases'] if c['review_sample']['is_sampled']]
    assert len(sampled_manifest_cases) == 230
    
    ledger_case_ids = {r['case_id'] for r in ledger}
    manifest_case_ids = {c['case_id'] for c in sampled_manifest_cases}
    assert ledger_case_ids == manifest_case_ids

def test_precision_recall_frontier_structure():
    with open(FRONTIER_PATH) as f:
        frontier = json.load(f)
        
    cfgs = frontier['configurations']
    assert 't_op_fixed' in cfgs
    assert 'b_control_fixed' in cfgs
    
    for cfg in ['t_op_fixed', 'b_control_fixed']:
        for prot in ['protocol_z', 'protocol_o']:
            assert prot in cfgs[cfg]
            for fam in cfgs[cfg][prot]:
                ceilings = cfgs[cfg][prot][fam]['calibrated_ceilings']
                assert len(ceilings) == 7
                for c in ['0.001', '0.005', '0.01', '0.02', '0.05', '0.1', '0.15']:
                    assert c in ceilings
                    pt = ceilings[c]
                    assert 'TP' in pt
                    assert 'FP_cor' in pt
                    assert 'FP_sub' in pt
                    assert 'recall' in pt
                    assert 'subset_precision' in pt
                    assert 'deployed_precision' in pt

def test_retrospective_precision_target_boundary():
    with open(FRONTIER_PATH) as f:
        frontier = json.load(f)
        
    t_op_z = frontier['configurations']['t_op_fixed']['protocol_z']['omission_dedicated']
    targets = t_op_z['retrospective_targets']
    
    # Target 30% is attainable at small recall
    assert targets['0.3']['attainable'] is True
    assert targets['0.3']['max_recall'] < 0.05
    
    # Target 50% is attainable only for <= 1% recall
    assert targets['0.5']['attainable'] is True
    assert targets['0.5']['max_recall'] < 0.01
    
    # Target 60% and 70% are NOT attainable
    assert targets['0.6']['attainable'] is False
    assert targets['0.7']['attainable'] is False
