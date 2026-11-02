"""
Tests for LANGFENS Phase RQ1-DIAG-02: Temporal Alignment Root-Cause Validation.

Validates:
1. Exact reproduction of RQ1-DIAG-01 token census and omission metrics.
2. TextGrid timestamp, short-duration interval, and transition span congruence.
3. Attention enrichment ratios, normalized entropy, and temporal offset baselines.
4. Individual attention head saturation and static frame cycling across Heads 0-3.
5. Counterfactual acoustic masking proving near-zero local sensitivity (< 0.005).
6. Forced attention counterfactual proving downstream MLP out-of-distribution drift.
7. CTC omission acoustic evidence proving > 80% blank emission on true omissions.
8. Speaker-aware paired uncertainty across 8 disjoint validation speakers (df=7).
9. Scientific GO/NO-GO recommendation for alignment-focused experimentation.
"""

import csv
import json
import os
import pytest


REPORT_PATH = "manifests/RQ1-DIAG-02/evaluation/TEMPORAL_ALIGNMENT_VALIDATION_REPORT.json"
ENRICH_CSV_PATH = "manifests/RQ1-DIAG-02/evaluation/attention_enrichment_by_token_type.csv"
HEAD_CSV_PATH = "manifests/RQ1-DIAG-02/evaluation/head_attention_breakdown.csv"
CF_CSV_PATH = "manifests/RQ1-DIAG-02/evaluation/omission_counterfactual_masking_results.csv"
CTC_CSV_PATH = "manifests/RQ1-DIAG-02/evaluation/ctc_omission_boundary_audit.csv"
SPK_CSV_PATH = "manifests/RQ1-DIAG-02/evaluation/speaker_paired_uncertainty_summary.csv"


@pytest.fixture(scope="module")
def report_data():
    """Loads master temporal alignment validation report."""
    assert os.path.exists(REPORT_PATH), f"Report not found at {REPORT_PATH}"
    with open(REPORT_PATH, "r") as f:
        return json.load(f)


def test_report_structure_and_metadata(report_data):
    """Verifies that the report contains all required top-level diagnostic sections."""
    required_sections = [
        "metadata",
        "step01_metrics_reproduction",
        "textgrid_timestamp_and_span_audit",
        "attention_enrichment_and_baselines",
        "individual_head_inspection",
        "counterfactual_masking_experiments",
        "ctc_omission_boundary_audit",
        "speaker_aware_paired_uncertainty",
        "scientific_go_no_go_recommendation",
    ]
    for sec in required_sections:
        assert sec in report_data, f"Missing required report section: {sec}"

    meta = report_data["metadata"]
    assert meta["experiment"] == "RQ1-DIAG-02"
    assert meta["total_tokens_audited"] == 39576
    assert meta["sampled_tokens"] > 2500
    assert meta["random_seed"] == 1337


def test_step01_metrics_exact_reproduction(report_data):
    """Verifies exact reproduction of Step 01 token census, AP, and acoustic gap metrics."""
    repro = report_data["step01_metrics_reproduction"]
    assert repro["status"] == "EXACT_REPRODUCTION_VERIFIED"
    assert repro["total_tokens"] == 39576
    assert repro["correct_tokens"] == 35363
    assert repro["substitution_tokens"] == 3584
    assert repro["deletion_tokens"] == 629

    assert pytest.approx(repro["overall_matched_ap"], abs=1e-5) == 0.272580
    assert pytest.approx(repro["substitution_matched_ap"], abs=1e-5) == 0.264270
    assert pytest.approx(repro["deletion_matched_ap"], abs=1e-5) == 0.030506
    assert pytest.approx(repro["substitution_acoustic_gap"], abs=1e-5) == 0.010715
    assert pytest.approx(repro["deletion_acoustic_gap"], abs=1e-5) == 0.001677


def test_textgrid_timestamp_and_span_audit(report_data):
    """Verifies Praat TextGrid timestamps and transition span congruence."""
    tg = report_data["textgrid_timestamp_and_span_audit"]
    assert tg["total_omissions_audited"] == 629
    assert tg["zero_duration_count"] == 0

    # Duration stats
    dur_stats = tg["gold_duration_stats"]
    assert dur_stats["min_sec"] > 0.0
    assert pytest.approx(dur_stats["median_sec"], abs=0.01) == 0.0400
    assert pytest.approx(dur_stats["mean_sec"], abs=0.01) == 0.0486

    # Expected transition span stats
    span_stats = tg["expected_transition_span_stats"]
    assert pytest.approx(span_stats["median_sec"], abs=0.01) == 0.0500
    assert pytest.approx(span_stats["mean_sec"], abs=0.01) == 0.0687

    # Congruence
    cong = tg["transition_span_congruence"]
    assert cong["exact_match_count"] >= 500
    assert cong["exact_match_fraction"] > 0.75
    assert pytest.approx(cong["exact_match_fraction"], abs=1e-3) == 0.795


def test_individual_head_saturation_and_static_peaks(report_data):
    """
    Verifies that all 4 individual attention heads saturate to near-one-hot spikes (> 0.95)
    and cycle across fewer than 10 unique frames per utterance.
    """
    heads = report_data["individual_head_inspection"]
    for h in range(4):
        h_data = heads[f"head_{h}"]
        assert h_data["evaluated_tokens"] > 2500
        # Every head saturates to near 1.0
        assert h_data["max_weight_mean"] > 0.95
        assert pytest.approx(h_data["max_weight_mean"], abs=0.02) == 0.99
        # Every head cycles between fewer than 10 unique frames per utterance
        assert h_data["mean_unique_peaks_per_utterance"] < 10.0
        # In-window mass for each head is near zero (< 0.05)
        assert h_data["in_window_mass_mean"] < 0.050


def test_counterfactual_local_masking_insensitivity(report_data):
    """
    Verifies that locally masking the acoustic window produces negligible score change (< 0.005),
    proving that the baseline binary head is locally acoustically insensitive.
    """
    cf = report_data["counterfactual_masking_experiments"]
    for etype in ["correct", "substitution", "deletion", "all"]:
        sub = cf[etype]
        # Mean absolute delta on local masking is < 0.005
        assert sub["mean_absolute_score_delta_on_masking"] < 0.005
        assert pytest.approx(sub["mean_absolute_score_delta_on_masking"], abs=0.003) == 0.001


def test_forced_attention_counterfactual_distribution_drift(report_data):
    """
    Verifies that forcing attention into the local window causes massive upward score drift (> +0.15),
    proving that the downstream MLP cannot interpret localized representations without retraining.
    """
    cf = report_data["counterfactual_masking_experiments"]
    for etype in ["correct", "substitution", "deletion"]:
        sub = cf[etype]
        assert sub["mean_score_delta_on_forced_attention"] > 0.15
        assert sub["forced_local_attention_score_mean"] > 0.35


def test_ctc_omission_boundary_acoustic_evidence(report_data):
    """
    Verifies that the CTC acoustic branch possesses decisive omission evidence (> 80% blank frames),
    proving that acoustic omission information exists in Wav2Vec2 frames but is bypassed by cross-attention.
    """
    ctc = report_data["ctc_omission_boundary_audit"]
    assert ctc["evaluated_omissions_count"] >= 100
    assert ctc["blank_argmax_frame_fraction"] > 0.80
    assert pytest.approx(ctc["blank_argmax_frame_fraction"], abs=0.05) == 0.86
    assert ctc["mean_blank_posterior_probability"] > 0.80
    assert pytest.approx(ctc["mean_blank_posterior_probability"], abs=0.05) == 0.84
    assert ctc["mean_canonical_posterior_probability"] < 0.15
    assert ctc["omissions_with_majority_blank_fraction"] > 0.90


def test_speaker_aware_paired_uncertainty_and_disclosure(report_data):
    """Verifies 8-speaker paired uncertainty quantification and small-cluster disclosure."""
    spk_data = report_data["speaker_aware_paired_uncertainty"]
    macro = spk_data["macro_8speaker_statistics"]
    assert macro["speaker_count"] == 8
    assert macro["degrees_of_freedom"] == 7
    assert pytest.approx(macro["critical_t_95"], abs=0.01) == 2.3646
    assert "small-cluster" in macro["small_sample_disclosure"].lower()

    # Per-speaker metrics
    per_spk = spk_data["per_speaker"]
    assert len(per_spk) == 8
    for spk, s_metrics in per_spk.items():
        assert s_metrics["ctc_omission_blank_prob"] > 0.70
        assert s_metrics["deletion_local_mask_delta"] < 0.005


def test_scientific_go_no_go_recommendation(report_data):
    """Verifies that the scientific recommendation clearly distinguishes joint training from inference clipping."""
    rec = report_data["scientific_go_no_go_recommendation"]
    assert rec["alignment_focused_experimentation_recommendation"] == "GO_FOR_JOINT_TRAINING_ALIGNMENT_REGULARIZATION"
    assert rec["inference_only_window_restriction_recommendation"] == "NO_GO_FOR_INFERENCE_ONLY_WINDOW_CLIPPING"
    assert "cross-attention temporal localization has completely collapsed" in rec["causal_mechanism_disposition"]


def test_csv_artifact_files_exist_and_non_empty():
    """Verifies all 5 CSV artifact tables exist and have valid rows and columns."""
    for path, min_rows in [
        (ENRICH_CSV_PATH, 4),
        (HEAD_CSV_PATH, 4),
        (CF_CSV_PATH, 4),
        (CTC_CSV_PATH, 100),
        (SPK_CSV_PATH, 8),
    ]:
        assert os.path.exists(path), f"File missing: {path}"
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) >= min_rows, f"Insufficient rows in {path}: {len(rows)}"
