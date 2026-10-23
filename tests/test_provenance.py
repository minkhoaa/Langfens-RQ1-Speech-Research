import json
import os
import pytest

def test_verify_split_manifest_hashes():
    from langfens_prep.provenance import verify_split_manifest_hashes
    
    results = verify_split_manifest_hashes()
    assert len(results) == 3
    assert results["l2arctic_4fold_splits.json"] is True
    assert results["l2arctic_loso_vietnamese_splits.json"] is True
    assert results["speechocean762_official_splits.json"] is True

def test_canonical_payload_sha256():
    from langfens_prep.provenance import compute_canonical_payload_sha256
    
    with open("manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    computed = compute_canonical_payload_sha256(data["folds"])
    assert computed == data["payload_sha256"]
    assert len(computed) == 64

def test_provenance_timestamp_investigation():
    from langfens_prep.provenance import investigate_provenance_timestamp
    
    report = investigate_provenance_timestamp()
    assert "host_timezone" in report
    assert "recorded_timestamp_utc" in report
    assert report["recorded_timestamp_utc"] == "2026-10-23T13:30:00Z"
    assert report["status"] == "NOT_VERIFIED"
