import pytest

def test_full_l2arctic_gold_census():
    from langfens_prep.validator import run_full_l2arctic_census
    
    zip_path = "datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip"
    census = run_full_l2arctic_census(zip_path)
    
    assert census["total_gold_utterances"] == 3599
    assert census["total_intervals_all"] == 135890
    assert census["silence_intervals"] == 15647
    assert census["addition_intervals"] == 1092
    
    # Canonical denominator
    assert census["canonical_tokens"] == 119151
    assert census["correct_tokens"] == 101633
    assert census["substitution_tokens"] == 14098
    assert census["deletion_tokens"] == 3420
    
    # Substitution subcategories
    assert census["sub_clean_diff_base"] == 12364
    assert census["sub_clean_same_base"] == 4
    assert census["sub_base_preserving_allophone"] == 1112
    assert census["sub_category_shifting_accented"] == 360
    assert census["sub_distorted_err"] == 258
    assert (
        census["sub_clean_diff_base"] +
        census["sub_clean_same_base"] +
        census["sub_base_preserving_allophone"] +
        census["sub_category_shifting_accented"] +
        census["sub_distorted_err"]
    ) == 14098
    
    # Binary policies
    assert census["strict_binary_errors"] == 17518
    assert census["lenient_binary_errors"] == 16406
    assert census["strict_error_rate_pct"] == pytest.approx(14.7024, abs=0.001)
    assert census["lenient_error_rate_pct"] == pytest.approx(13.7691, abs=0.001)
    
    # Invariant denominator
    assert census["strict_canonical_total"] == 119151
    assert census["lenient_canonical_total"] == 119151

def test_speechocean762_validation():
    from langfens_prep.validator import validate_speechocean_splits
    
    archive_path = "datasets/SpeechOcean762/original/speechocean762.tar.gz"
    manifest_path = "manifests/RQ1-SPLIT-01/speechocean762_official_splits.json"
    
    val_report = validate_speechocean_splits(archive_path, manifest_path)
    assert val_report["train_utts"] == 2000
    assert val_report["val_utts"] == 500
    assert val_report["test_utts"] == 2500
    assert val_report["total_manifested_utts"] == 5000
    assert val_report["unmanifested_excluded_count"] == 245
    assert val_report["partition_overlap_count"] == 0
