"""
LANGFENS — RQ1-TRN-02B: Colab Launch Workflow & Operational Safety Regression Tests
Verifies:
1. Arm identifier validation (C0 / C1 strictly).
2. Git commit SHA validation (verifiable 40-character hex).
3. Colab CLI usage parsing & quota gate evaluation.
4. Verified command syntax: colab exec -s SESSION -f SCRIPT --env K=V --timeout SEC.
5. Strict GPU enforcement: prohibits fallback to CPU training.
6. Fail-closed staging: detects missing source files, corrupted hashes, conflicting artifacts.
7. Scientific provenance: Fold 1 speaker disjointness, zero test speaker leakage.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict

import pytest

from audit_scripts.colab_launch_workflow import (
    check_quota_status,
    generate_colab_arm_plan,
    parse_colab_usage,
    validate_arm,
    validate_commit_sha,
)
from audit_scripts.colab_stage_and_verify import (
    check_conflicting_checkpoints,
    compute_sha256,
    stage_file_with_hash,
    verify_write_access,
)
import audit_scripts.colab_stage_and_verify as stage_mod
import audit_scripts.colab_train_exec as train_mod


# --- 1. Arm Identifier Validation ---

def test_validate_arm_valid():
    assert validate_arm("C0") == "C0"
    assert validate_arm("c0") == "C0"
    assert validate_arm("C1") == "C1"
    assert validate_arm("c1") == "C1"


def test_validate_arm_invalid():
    with pytest.raises(ValueError, match="Strictly 'C0' or 'C1'"):
        validate_arm("C2")
    with pytest.raises(ValueError, match="Strictly 'C0' or 'C1'"):
        validate_arm("R")
    with pytest.raises(ValueError, match="Strictly 'C0' or 'C1'"):
        validate_arm("")


# --- 2. Commit SHA Validation ---

def test_validate_commit_sha_valid():
    sha = "aa6415a8d3a073a3550f209a53e1baa44461ce97"
    assert validate_commit_sha(sha) == sha
    assert validate_commit_sha(sha.upper()) == sha


def test_validate_commit_sha_invalid():
    with pytest.raises(ValueError, match="Expected 40-character"):
        validate_commit_sha("aa6415a")  # short sha
    with pytest.raises(ValueError, match="Expected 40-character"):
        validate_commit_sha("g" * 40)  # non-hex
    with pytest.raises(ValueError, match="Expected 40-character"):
        validate_commit_sha("")


# --- 3. Colab CLI Usage Parsing & Quota Gate ---

def test_parse_colab_usage_normal():
    sample = """
Current balance: 0.00 compute units
Usage rate: 0.00/hr
Active assignments: 0
"""
    res = parse_colab_usage(sample)
    assert res["balance"] == 0.0
    assert res["usage_rate"] == 0.0
    assert res["active_assignments"] == 0


def test_parse_colab_usage_funded():
    sample = """
Current balance: 25.50 compute units
Usage rate: 1.96/hr
Active assignments: 1
"""
    res = parse_colab_usage(sample)
    assert res["balance"] == 25.5
    assert res["usage_rate"] == 1.96
    assert res["active_assignments"] == 1


def test_quota_gate_evaluation():
    ok, msg = check_quota_status(0.0)
    assert not ok
    assert "ZERO_QUOTA" in msg

    ok, msg = check_quota_status(5.0)
    assert not ok
    assert "LOW_QUOTA" in msg

    ok, msg = check_quota_status(20.0)
    assert ok
    assert "SUFFICIENT" in msg


# --- 4. Verified Command Syntax & Structure ---

def test_generate_colab_arm_plan_syntax():
    commit = "aa6415a8d3a073a3550f209a53e1baa44461ce97"
    plan_c0 = generate_colab_arm_plan("C0", commit, session_name="langfens-c0-test")
    full_text = "\n".join(plan_c0)

    # Must contain verified colab exec -s ... -f ... syntax
    assert "colab exec -s langfens-c0-test -f audit_scripts/colab_stage_and_verify.py" in full_text
    assert "--env ARM=C0" in full_text
    assert f"--env TARGET_COMMIT={commit}" in full_text
    assert "--timeout 1800" in full_text

    assert "colab exec -s langfens-c0-test -f audit_scripts/colab_train_exec.py" in full_text
    assert "--timeout 28800" in full_text

    # Must NOT contain old broken shell pipe pattern
    assert 'colab exec -s langfens-c0-test -- "' not in full_text
    assert "&&" not in full_text


def test_generate_colab_arm_plan_arm_c1_isolated():
    commit = "aa6415a8d3a073a3550f209a53e1baa44461ce97"
    plan_c1 = generate_colab_arm_plan("C1", commit, session_name="langfens-c1-test")
    full_text = "\n".join(plan_c1)

    assert "--env ARM=C1" in full_text
    assert "--env ARM=C0" not in full_text
    assert "colab stop -s langfens-c1-test" in full_text


# --- 5. Strict GPU Enforcement & Fallback Prohibition ---

def test_strict_gpu_enforcement_in_staging(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is NOT available"):
        stage_mod.stage_and_verify(
            arm="C0",
            target_commit="aa6415a8d3a073a3550f209a53e1baa44461ce97",
            repo_root=Path("."),
            local_data_dir=Path("/tmp"),
            local_checkpoint_dir=Path("/tmp"),
            drive_root=Path("/tmp"),
            mock_gpu=False,
        )


def test_strict_gpu_enforcement_in_training(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is NOT available"):
        train_mod.execute_training(
            arm="C0",
            target_commit="aa6415a8d3a073a3550f209a53e1baa44461ce97",
            repo_root=Path("."),
            drive_root=Path("/tmp"),
            local_zip_path=Path("/tmp/fake.zip"),
            local_baseline_path=Path("/tmp/fake.pt"),
            mock_gpu=False,
        )


# --- 6. Staging, Hashing & Conflict Guards ---

def test_stage_file_with_hash_success(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"HELLO LANGFENS VERIFICATION")
    expected_sha = hashlib.sha256(b"HELLO LANGFENS VERIFICATION").hexdigest()

    dest = tmp_path / "local" / "dest.bin"
    out_sha = stage_file_with_hash(src, dest, expected_sha, "test file")
    assert out_sha == expected_sha
    assert dest.is_file()


def test_stage_file_with_hash_mismatch_fails_closed(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"CORRUPTED CONTENT")

    dest = tmp_path / "local" / "dest.bin"
    with pytest.raises(RuntimeError, match="Cryptographic hash mismatch"):
        stage_file_with_hash(src, dest, "0000000000000000000000000000000000000000000000000000000000000000", "test file")

    # Corrupt destination must be cleaned up
    assert not dest.exists()


def test_stage_file_missing_source_fails_closed(tmp_path):
    src = tmp_path / "nonexistent.bin"
    dest = tmp_path / "dest.bin"
    with pytest.raises(FileNotFoundError):
        stage_file_with_hash(src, dest, "dummy", "test file")


def test_check_conflicting_checkpoints(tmp_path):
    outdir = tmp_path / "checkpoints"
    outdir.mkdir()

    # Empty dir should pass
    check_conflicting_checkpoints(outdir, resume=False)

    # Creating last.pt must fail closed if resume=False
    (outdir / "last.pt").write_text("checkpoint")
    with pytest.raises(FileExistsError, match="Conflicting checkpoint artifacts already exist"):
        check_conflicting_checkpoints(outdir, resume=False)

    # With resume=True, should pass
    check_conflicting_checkpoints(outdir, resume=True)


def test_verify_write_access(tmp_path):
    target = tmp_path / "output_test"
    verify_write_access(target)
    assert target.is_dir()
    assert not (target / ".write_verification.tmp").exists()


# --- 7. Scientific Provenance: Split Boundaries & Speaker Disjointness ---

def test_fold1_speaker_disjointness_and_counts():
    manifest_path = Path("manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json")
    assert manifest_path.is_file()

    with open(manifest_path) as f:
        data = json.load(f)

    fold1 = data["folds"][0]
    assert fold1["fold_index"] == 1

    train_speakers = sorted(fold1["train"]["speakers"])
    val_speakers = sorted(fold1["val"]["speakers"])
    test_speakers = sorted(fold1["test"]["speakers"])

    expected_train = sorted([
        "BWC", "ERMS", "HKK", "HQTV", "MBMPS", "NCC", "SKA", "SVBI",
        "THV", "TLV", "TNI", "TXHC", "YBAA", "YDCK", "YKWK", "ZHAA",
    ])
    expected_val = sorted(["NJS", "RRBI"])
    expected_test = sorted(["ABA", "ASI", "EBVS", "HJK", "LXC", "PNV"])

    assert train_speakers == expected_train
    assert val_speakers == expected_val
    assert test_speakers == expected_test

    # Strict disjointness
    set_train = set(train_speakers)
    set_val = set(val_speakers)
    set_test = set(test_speakers)

    assert set_train.isdisjoint(set_val), f"Train/Val overlap: {set_train & set_val}"
    assert set_train.isdisjoint(set_test), f"Train/Test overlap: {set_train & set_test}"
    assert set_val.isdisjoint(set_test), f"Val/Test overlap: {set_val & set_test}"

    # Utterance counts
    train_utts = len(fold1["train"]["utterances"])
    val_utts = len(fold1["val"]["utterances"])
    test_utts = len(fold1["test"]["utterances"])

    assert train_utts == 2399, f"Expected 2399 train utts, got {train_utts}"
    assert val_utts == 300, f"Expected 300 val utts, got {val_utts}"
    assert test_utts == 900, f"Expected 900 test utts, got {test_utts}"
    assert train_utts + val_utts + test_utts == 3599


def test_authorized_zip_contains_zero_test_speakers():
    zip_path = Path("datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip")
    if not zip_path.is_file():
        pytest.skip("Authorized zip not present on local machine.")

    with zipfile.ZipFile(zip_path) as z:
        entries = [n.replace(".zip", "") for n in z.namelist() if n.endswith(".zip")]

    test_speakers = {"ABA", "ASI", "EBVS", "HJK", "LXC", "PNV"}
    leaked = set(entries) & test_speakers
    assert len(leaked) == 0, f"Test speaker leakage in authorized zip! Leaked: {leaked}"
