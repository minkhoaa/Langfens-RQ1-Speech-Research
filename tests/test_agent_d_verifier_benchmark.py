"""
Unit and regression test suite for Agent D — Controlled Acoustic Verifier & System-Level Benchmark.
Validates the structural integrity, numerical closure, and multi-class invariants of
SYSTEM_LEVEL_COMPARISON.json, ACOUSTIC_VERIFIER_BENCHMARK.md, and VERIFIER_ABLATION_REPORT.md.
"""

import json
import os
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORKSTREAM_DIR = os.path.join(ROOT_DIR, 'workstreams', 'agent_d_verifier')

JSON_WORKSTREAM = os.path.join(WORKSTREAM_DIR, 'SYSTEM_LEVEL_COMPARISON.json')
JSON_ROOT = os.path.join(ROOT_DIR, 'SYSTEM_LEVEL_COMPARISON.json')

BENCHMARK_MD_WORKSTREAM = os.path.join(WORKSTREAM_DIR, 'ACOUSTIC_VERIFIER_BENCHMARK.md')
BENCHMARK_MD_ROOT = os.path.join(ROOT_DIR, 'ACOUSTIC_VERIFIER_BENCHMARK.md')

ABLATION_MD_WORKSTREAM = os.path.join(WORKSTREAM_DIR, 'VERIFIER_ABLATION_REPORT.md')
ABLATION_MD_ROOT = os.path.join(ROOT_DIR, 'VERIFIER_ABLATION_REPORT.md')


def test_deliverable_files_exist_and_match():
    """Verify that all Agent D deliverables exist in both workstream and root and are byte-identical."""
    for ws_path, root_path in [
        (JSON_WORKSTREAM, JSON_ROOT),
        (BENCHMARK_MD_WORKSTREAM, BENCHMARK_MD_ROOT),
        (ABLATION_MD_WORKSTREAM, ABLATION_MD_ROOT),
    ]:
        assert os.path.exists(ws_path), f"Missing workstream artifact: {ws_path}"
        assert os.path.exists(root_path), f"Missing root artifact: {root_path}"
        assert os.path.getsize(ws_path) > 0, f"Artifact is empty: {ws_path}"
        with open(ws_path, 'rb') as f1, open(root_path, 'rb') as f2:
            assert f1.read() == f2.read(), f"File mismatch between {ws_path} and {root_path}"


def test_system_level_comparison_schema_and_metadata():
    """Verify evaluation metadata and schema in SYSTEM_LEVEL_COMPARISON.json."""
    with open(JSON_WORKSTREAM) as f:
        data = json.load(f)

    meta = data['evaluation_metadata']
    assert meta['protocol'] == 'protocol_z'
    assert meta['model'] == 't_op_fixed_budget_3600'
    assert meta['total_gold_positions'] == 9956
    assert meta['total_gold_omissions'] == 145
    assert meta['feature_extraction_overhead_ms_per_utt'] == 13.84

    assert len(data['primary_configurations']) == 5
    assert len(data['ablation_suite']) == 4


def test_primary_configurations_numerical_integrity():
    """Verify exact metrics across all 5 primary configurations."""
    with open(JSON_WORKSTREAM) as f:
        data = json.load(f)

    configs = {c['config_name']: c for c in data['primary_configurations']}

    # Config 1: Original T-OP Baseline
    c1 = configs['1. Original T-OP Baseline']
    assert c1['TP'] == 82
    assert c1['FP_cor'] == 1068
    assert c1['FP_sub'] == 59
    assert c1['FN'] == 63
    assert c1['total_alarms'] == 1209
    assert c1['alarms_filtered'] == 0
    assert abs(c1['deployed_precision'] - 0.067825) < 1e-6
    assert abs(c1['all_gold_recall'] - 0.565517) < 1e-6
    assert c1['false_alarms_per_1000'] == 113.20

    # Config 2: Conservative T-OP Threshold
    c2 = configs['2. Conservative T-OP Threshold']
    assert c2['TP'] == 26
    assert c2['FP_cor'] == 182
    assert c2['FP_sub'] == 7
    assert c2['total_alarms'] == 215
    assert c2['alarms_filtered'] == 994
    assert abs(c2['deployed_precision'] - 0.120930) < 1e-6
    assert abs(c2['all_gold_recall'] - 0.179310) < 1e-6

    # Config 3: Phoneme-Only Filter
    c3 = configs['3. Phoneme-Only Filter']
    assert c3['TP'] == 2
    assert c3['FP_cor'] == 64
    assert c3['FP_sub'] == 10
    assert c3['total_alarms'] == 76
    assert c3['alarms_filtered'] == 1133
    assert abs(c3['deployed_precision'] - 0.026316) < 1e-6
    assert abs(c3['all_gold_recall'] - 0.013793) < 1e-6

    # Config 4: Acoustic Verification Candidate
    c4 = configs['4. Acoustic Verification Candidate']
    assert c4['TP'] == 57
    assert c4['FP_cor'] == 802
    assert c4['FP_sub'] == 45
    assert c4['FN'] == 88
    assert c4['total_alarms'] == 904
    assert c4['alarms_filtered'] == 305
    assert abs(c4['deployed_precision'] - 0.063053) < 1e-6
    assert abs(c4['all_gold_recall'] - 0.393103) < 1e-6
    assert c4['false_alarms_per_1000'] == 85.07

    # Config 5: Combined Candidate
    c5 = configs['5. Combined Candidate (Conservative + Acoustic)']
    assert c5['TP'] == 18
    assert c5['FP_cor'] == 194
    assert c5['FP_sub'] == 14
    assert c5['total_alarms'] == 226
    assert c5['alarms_filtered'] == 983
    assert abs(c5['deployed_precision'] - 0.079646) < 1e-6
    assert abs(c5['all_gold_recall'] - 0.124138) < 1e-6


def test_ablation_suite_numerical_integrity():
    """Verify exact metrics across all 4 ablation variants."""
    with open(JSON_WORKSTREAM) as f:
        data = json.load(f)

    ablations = {a['config_name']: a for a in data['ablation_suite']}

    # Candidate 4 (Full)
    a_full = ablations['4. Acoustic Verification Candidate']
    assert a_full['TP'] == 57
    assert a_full['total_alarms'] == 904
    assert a_full['alarms_filtered'] == 305

    # Ablation: No-Burst Gate
    a_no_burst = ablations['Ablation: No-Burst Gate']
    assert a_no_burst['TP'] == 64
    assert a_no_burst['FP_cor'] == 838
    assert a_no_burst['FP_sub'] == 47
    assert a_no_burst['total_alarms'] == 949
    assert a_no_burst['alarms_filtered'] == 260
    assert abs(a_no_burst['deployed_precision'] - 0.067439) < 1e-6
    assert abs(a_no_burst['all_gold_recall'] - 0.441379) < 1e-6

    # Ablation: No-Duration Gate
    a_no_dur = ablations['Ablation: No-Duration Gate']
    assert a_no_dur['TP'] == 72
    assert a_no_dur['FP_cor'] == 958
    assert a_no_dur['FP_sub'] == 52
    assert a_no_dur['total_alarms'] == 1082
    assert a_no_dur['alarms_filtered'] == 127
    assert abs(a_no_dur['deployed_precision'] - 0.066543) < 1e-6
    assert abs(a_no_dur['all_gold_recall'] - 0.496552) < 1e-6

    # Ablation: No-Rhotic Gate
    a_no_rhotic = ablations['Ablation: No-Rhotic Gate']
    assert a_no_rhotic['TP'] == 58
    assert a_no_rhotic['FP_cor'] == 871
    assert a_no_rhotic['FP_sub'] == 50
    assert a_no_rhotic['total_alarms'] == 979
    assert a_no_rhotic['alarms_filtered'] == 230
    assert abs(a_no_rhotic['deployed_precision'] - 0.059244) < 1e-6
    assert abs(a_no_rhotic['all_gold_recall'] - 0.400000) < 1e-6


def test_multi_class_mdd_invariants():
    """Verify that multi-class MDD AP and PER metrics are preserved exactly across all configs."""
    with open(JSON_WORKSTREAM) as f:
        data = json.load(f)

    for cfg in data['primary_configurations'] + data['ablation_suite']:
        assert cfg['binary_ap'] == 0.318619, f"Binary AP drifted in {cfg['config_name']}"
        assert cfg['substitution_ap'] == 0.308974, f"Substitution AP drifted in {cfg['config_name']}"
        assert cfg['greedy_per'] == 0.241980, f"Greedy PER drifted in {cfg['config_name']}"


def test_markdown_reports_contain_key_findings_and_sections():
    """Verify that both markdown reports contain the required analyses and section headers."""
    with open(BENCHMARK_MD_WORKSTREAM) as f:
        bench_text = f.read()

    assert "LANGFENS RQ1 — Controlled Acoustic Verifier & System-Level Benchmark Report (Agent D)" in bench_text
    assert "Executive Summary & Definitive Benchmark Outcome" in bench_text
    assert "Master System-Level Comparison Table" in bench_text
    assert "Multi-Class MDD Invariants & Upstream Decoupling" in bench_text
    assert "0.318619" in bench_text  # Binary AP invariant
    assert "0.308974" in bench_text  # Substitution AP invariant
    assert "0.241980" in bench_text  # PER invariant
    assert "13.84" in bench_text     # Overhead ms/utt
    assert "Speaker Breakdown & Phonetic Asymmetry (NJS vs. RRBI)" in bench_text
    assert "Deep Scientific Synthesis: Why Acoustic Verification Alone Fails" in bench_text

    with open(ABLATION_MD_WORKSTREAM) as f:
        abl_text = f.read()

    assert "LANGFENS RQ1 — Verifier Feature Ablation & Component Analysis (Agent D)" in abl_text
    assert "Executive Summary & Purpose of the Ablation Suite" in abl_text
    assert "Burst Prominence & Spectral Centroid Gate" in abl_text
    assert "Closure Duration & Relative RMS Energy Gate" in abl_text
    assert "Rhotic Vocalization & Duration Gate" in abl_text
    assert "The Non-Native Tap/Flap Dilemma" in abl_text
    assert "Ablation: No-Burst Gate" in abl_text
    assert "Ablation: No-Duration Gate" in abl_text
    assert "Ablation: No-Rhotic Gate" in abl_text
