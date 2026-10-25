"""
LANGFENS — RQ1-TRN-01B1-R2: Acoustic-Conditioned MDD & Final Source Provenance Gate Regression Suite.

Verifies:
1. GitHub provenance and machine-readable metadata consistency.
2. Canonical-only baseline superiority (prompt prior >= 55% vs matched diagnosis <= 52%).
3. Matched vs mismatched token agreement (>90% top-1 agreement on substitution tokens).
4. Acoustic Sensitivity Index (ASI near zero, confirming prompt prior dominance).
5. Deployable detection-diagnosis pipeline separation (oracle vs cascaded vs E2E).
6. Preserved CTC recovery (greedy PER < 0.50, length ratio > 0.85 on matched audio).
7. Strategy ablation completeness across all 4 bounded architectures.
"""

import json
from pathlib import Path
import pytest
import numpy as np

GATE_DIR = Path("manifests/RQ1-TRN-01B1/reconciliation/final_acoustic_gate")

@pytest.fixture(scope="module")
def task_b_report():
    path = GATE_DIR / "task_b_acoustic_contribution.json"
    assert path.exists(), f"Task B report missing: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture(scope="module")
def baselines_report():
    path = GATE_DIR / "canonical_only_baselines.json"
    assert path.exists(), f"Baselines report missing: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture(scope="module")
def task_c_report():
    path = GATE_DIR / "task_c_strategy_ablations.json"
    assert path.exists(), f"Task C report missing: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture(scope="module")
def task_d_report():
    path = GATE_DIR / "task_d_pipeline_evaluation.json"
    assert path.exists(), f"Task D report missing: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def test_provenance_and_git_commit():
    """Verifies that parent reconciliation commit 493b7ab exists in local git history."""
    import subprocess
    res = subprocess.run(
        ["git", "cat-file", "-t", "493b7ab875174c7a6dd9812e3526a50717ecb5dc"],
        capture_output=True,
        text=True
    )
    assert res.returncode == 0
    assert res.stdout.strip() == "commit"

def test_canonical_only_baseline_superiority(baselines_report, task_b_report):
    """
    Verifies that canonical prompt-only baselines achieve higher accuracy than the
    multi-task model with matched audio, proving prompt prior dominance.
    """
    prior_acc = baselines_report["substitution_baselines"]["canonical_empirical_prior"]["holdout_accuracy"]
    learned_mlp_acc = baselines_report["substitution_baselines"]["canonical_learned_mlp"]["holdout_accuracy"]
    matched_acc = task_b_report["condition_performances"]["matched"]["substitution_accuracy"]
    mismatched_acc = task_b_report["condition_performances"]["mismatched"]["substitution_accuracy"]

    assert prior_acc >= 0.55, f"Empirical prior accuracy {prior_acc} unexpectedly low"
    assert learned_mlp_acc >= 0.55, f"Learned MLP accuracy {learned_mlp_acc} unexpectedly low"
    assert matched_acc <= 0.52, f"Matched accuracy {matched_acc} unexpectedly high"
    # Prior strictly exceeds matched audio accuracy
    assert prior_acc > matched_acc
    assert learned_mlp_acc > matched_acc

    # Matched and mismatched audio accuracy are within 1 percentage point
    assert abs(matched_acc - mismatched_acc) < 0.01

def test_matched_vs_mismatched_token_agreement(task_b_report):
    """
    Verifies that token-level prediction agreement between matched and mismatched audio
    is extremely high (>85% all tokens, >90% substitution tokens), and cosine similarity > 0.98.
    """
    all_top1 = task_b_report["matched_vs_mismatched_token_agreement"]["all_tokens"]["top1_agreement_rate"]
    sub_top1 = task_b_report["matched_vs_mismatched_token_agreement"]["substitution_tokens"]["top1_agreement_rate"]
    cos_sim = task_b_report["matched_vs_mismatched_logit_distances"]["cosine_similarity"]["mean"]
    aligned_cos = task_b_report["cross_attention_pathway_and_invariance_audit"]["aligned_acoustic_cosine_similarity"]["mean"]

    assert all_top1 >= 0.85, f"All tokens top-1 agreement {all_top1} < 0.85"
    assert sub_top1 >= 0.90, f"Substitutions top-1 agreement {sub_top1} < 0.90"
    assert cos_sim >= 0.98, f"Logit cosine similarity {cos_sim} < 0.98"
    assert aligned_cos >= 0.95, f"Aligned acoustic cosine similarity {aligned_cos} < 0.95"

def test_acoustic_sensitivity_index_near_zero(task_b_report):
    """
    Verifies that Acoustic Sensitivity Index (ASI) is virtually zero (|ASI| < 0.02),
    proving that diagnosis outputs do not respond when the spoken phone changes.
    """
    asi = task_b_report["controlled_spoken_phone_sensitivity_census"]["acoustic_sensitivity_index_overall"]
    assert abs(asi) < 0.02, f"ASI {asi} unexpectedly large; expected near zero"

def test_deployable_pipeline_separation(task_d_report):
    """
    Verifies that oracle diagnosis accuracy is separated from cascaded pipeline performance,
    and documents severe false-negative error propagation at standard threshold 0.5.
    """
    cond_matched = task_d_report["conditions"]["matched"]
    oracle_acc = cond_matched["oracle_diagnosis"]["accuracy"]
    assert 0.50 <= oracle_acc <= 0.52

    # At standard tau = 0.50, binary detector recall is low (< 0.25)
    std_rec = cond_matched["operating_point_standard_05"]["binary_recall"]
    assert std_rec < 0.25, f"Binary recall {std_rec} unexpectedly high at tau=0.5"

    # End-to-end recall at 0.5 is bounded by binary recall
    e2e_rec = cond_matched["operating_point_standard_05"]["e2e_recall"]
    assert e2e_rec < 0.25

    # Optimal F1 threshold recovers recall at the cost of precision
    opt_f1 = cond_matched["operating_point_optimal_f1"]["binary_f1"]
    assert opt_f1 >= 0.45

    # Mismatched audio produces comparable E2E F1 to matched audio (difference < 0.02)
    m_e2e_f1 = task_d_report["key_pipeline_comparisons"]["end_to_end_f1_at_optimal_tau"]["matched"]
    mis_e2e_f1 = task_d_report["key_pipeline_comparisons"]["end_to_end_f1_at_optimal_tau"]["mismatched"]
    assert abs(m_e2e_f1 - mis_e2e_f1) < 0.02

def test_preserved_ctc_recovery(task_c_report):
    """
    Verifies that the verified frozen-backbone CTC recovery is preserved:
    Holdout greedy PER < 0.50, decoded length ratio > 0.85, no blank collapse.
    """
    strat1 = task_c_report["detailed_strategy_reports"]["Strategy_1_Canonical_Cross_Attention_Baseline"]["evaluation"]["matched"]
    per = strat1["ctc_greedy_per"]
    lr = strat1["ctc_length_ratio"]

    assert per < 0.50, f"Greedy PER {per} exceeded 0.50"
    assert lr > 0.85, f"Length ratio {lr} below 0.85"

def test_strategy_ablation_completeness(task_c_report):
    """
    Verifies that all 4 strategies are present in the comparative ablation matrix.
    """
    matrix = task_c_report["comparative_ablation_matrix"]
    assert len(matrix) == 4
    expected_strategies = [
        "Strategy_1_Canonical_Cross_Attention_Baseline",
        "Strategy_2_Audio_Only_Representation",
        "Strategy_3_Joint_Gated_Residual_Fusion",
        "Strategy_4_CTC_Derived_Acoustic_Evidence"
    ]
    for strat in expected_strategies:
        assert strat in matrix, f"Strategy {strat} missing from ablation matrix"
        assert "matched_substitution_accuracy" in matrix[strat]
        assert "mismatched_substitution_accuracy" in matrix[strat]
        assert "delta_matched_mismatched" in matrix[strat]
