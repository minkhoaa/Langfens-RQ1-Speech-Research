#!/usr/bin/env python3
"""
LANGFENS — RQ1-TRN-02B: Kaggle T4x2 Platform Port Regression Test Suite
Exhaustively verifies:
1. Kaggle Device Checks: Tesla T4 detection, CUDA active, CPU/non-T4 rejection.
2. Immutable SHA Verification: Baseline R best.pt, split manifest, Fold 1 archive.
3. Train/Val-Only Staging & Test Sealing: 18 approved Fold 1 speakers, sealed test exclusion.
4. Auto-extracted ZIP Handling: Cryptographic member ledger verification & bit-identical reconstruction.
5. Per-Arm Output Isolation: C0 vs C1 isolation, run_id validation, conflict detection.
6. Missing Credentials & Fail-Closed Persistence: Ephemeral output rejection without verified mirror.
7. Checkpoint Synchronization: Atomic save, remote mirror, and SHA-256 readback.
8. Fail-Closed Endpoint Semantics: All 9 completion invariants and negative defect regressions.
9. Quota Parsing & Accelerator Gates: Kaggle quota parser and threshold enforcement.
10. Kernel Metadata Invariants: Private visibility, Tesla T4 machine shape, and licensing.
11. Disposable Benchmark Contract: 50 updates cap, counts_toward_c0=False isolation.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import zipfile
import zlib
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
import yaml

from audit_scripts.kaggle_checkpoint_sync import KaggleRemoteMirror, compute_sha256
from audit_scripts.kaggle_launch_workflow import (
    MINIMUM_REQUIRED_GPU_HOURS,
    check_kaggle_quota,
    parse_kaggle_quota,
    prepare_kaggle_kernel_directory,
    validate_git_software_state,
)
from audit_scripts.kaggle_stage_and_verify import (
    APPROVED_FOLD1_SPEAKERS,
    APPROVED_FOLD1_TRAIN_SPEAKERS,
    APPROVED_FOLD1_VAL_SPEAKERS,
    AUTHORIZED_SPEAKER_LEDGER,
    EXPECTED_BASELINE_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_ZIP_SHA256,
    SEALED_TEST_SPEAKERS,
    reconstruct_outer_zip_from_speaker_files,
    resolve_kaggle_output_dir,
    stage_and_verify_kaggle_environment,
    validate_arm,
    verify_and_stage_fold1_data,
    verify_baseline_weights,
    verify_bounded_zip_speakers,
    verify_kaggle_gpu_device,
    verify_split_manifest,
)
from audit_scripts.kaggle_train_exec import classify_kaggle_training_endpoint


# --- 1. Kaggle Device Checks ---

def test_kaggle_device_verification_mock():
    info = verify_kaggle_gpu_device(allow_mock=True)
    assert info["status"] == "MOCK_VERIFIED"
    assert info["device_count"] == 2
    assert "Tesla T4" in info["device_name"]


def test_kaggle_device_verification_rejects_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        verify_kaggle_gpu_device(allow_mock=False)


def test_kaggle_device_verification_rejects_non_t4_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "NVIDIA A100-SXM4-40GB")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda idx: (8, 0))

    with pytest.raises(RuntimeError, match="does not match required Nvidia Tesla T4"):
        verify_kaggle_gpu_device(allow_mock=False)


def test_kaggle_device_verification_rejects_incorrect_capability(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "Tesla T4")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda idx: (7, 0))

    with pytest.raises(RuntimeError, match="Unexpected compute capability"):
        verify_kaggle_gpu_device(allow_mock=False)


# --- 2. Immutable SHA Verification ---

def test_verify_split_manifest_success():
    manifest_p = Path("manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json")
    digest = verify_split_manifest(manifest_p)
    assert digest == EXPECTED_MANIFEST_SHA256


def test_verify_split_manifest_corrupted_fails_closed(tmp_path):
    bad_manifest = tmp_path / "bad_splits.json"
    bad_manifest.write_text('{"tampered": true}')
    with pytest.raises(ValueError, match="Split manifest SHA-256 mismatch"):
        verify_split_manifest(bad_manifest)


def test_verify_baseline_weights_success():
    baseline_p = Path("checkpoints/RQ1-TRN-02A/recovered/best.pt")
    if not baseline_p.is_file():
        pytest.skip("Baseline R best.pt unavailable locally")
    digest = verify_baseline_weights(baseline_p)
    assert digest == EXPECTED_BASELINE_SHA256


def test_verify_baseline_weights_tampered_fails_closed(tmp_path):
    bad_weights = tmp_path / "tampered_best.pt"
    bad_weights.write_text("corrupted weights")
    with pytest.raises(ValueError, match="Baseline R SHA-256 mismatch"):
        verify_baseline_weights(bad_weights)


# --- 3. Train/Val-Only Staging & Test Sealing ---

def test_speaker_partition_membership():
    assert len(APPROVED_FOLD1_TRAIN_SPEAKERS) == 16
    assert len(APPROVED_FOLD1_VAL_SPEAKERS) == 2
    assert len(APPROVED_FOLD1_SPEAKERS) == 18
    assert len(SEALED_TEST_SPEAKERS) == 6
    assert APPROVED_FOLD1_SPEAKERS.isdisjoint(SEALED_TEST_SPEAKERS)


def test_verify_bounded_zip_speakers_rejects_sealed_test(tmp_path):
    leak_zip = tmp_path / "leaked_test.zip"
    with zipfile.ZipFile(leak_zip, "w") as zf:
        # Include an approved speaker and a sealed test speaker
        zf.writestr("BWC.zip", b"dummy")
        zf.writestr("ABA.zip", b"leaked_test")

    with pytest.raises(ValueError, match="Sealed test speaker 'ABA' detected"):
        verify_bounded_zip_speakers(leak_zip)


def test_verify_bounded_zip_speakers_rejects_unapproved_speaker(tmp_path):
    unapproved_zip = tmp_path / "unapproved.zip"
    with zipfile.ZipFile(unapproved_zip, "w") as zf:
        zf.writestr("BWC.zip", b"dummy")
        zf.writestr("UNKNOWN_SPK.zip", b"unapproved")

    with pytest.raises(ValueError, match="Unapproved speaker 'UNKNOWN_SPK' detected"):
        verify_bounded_zip_speakers(unapproved_zip)


# --- 4. Auto-extracted ZIP Handling & Bit-Identical Reconstruction ---

def test_auto_extracted_zip_reconstruction(tmp_path):
    orig_zip = Path("datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip")
    if not orig_zip.is_file():
        pytest.skip("Authorized Fold 1 zip unavailable locally")

    # Extract all 18 speaker packages to simulated Kaggle extracted directory
    extracted_dir = tmp_path / "kaggle_extracted_input"
    extracted_dir.mkdir()
    with zipfile.ZipFile(orig_zip) as zf:
        zf.extractall(extracted_dir)

    # Reconstruct outer zip
    rebuilt_zip = tmp_path / "reconstructed.zip"
    computed_sha = reconstruct_outer_zip_from_speaker_files(extracted_dir, rebuilt_zip)

    assert computed_sha == EXPECTED_ZIP_SHA256
    assert compute_sha256(rebuilt_zip) == EXPECTED_ZIP_SHA256


def test_auto_extracted_zip_rejects_corrupted_crc(tmp_path):
    # Create fake directory with corrupted speaker package
    fake_dir = tmp_path / "corrupted_extracted"
    fake_dir.mkdir()
    for spk_name in AUTHORIZED_SPEAKER_LEDGER:
        (fake_dir / spk_name).write_bytes(b"dummy corrupted data")

    with pytest.raises(ValueError, match="size mismatch"):
        reconstruct_outer_zip_from_speaker_files(fake_dir, tmp_path / "out.zip")


# --- 5. Per-Arm Output Isolation ---

def test_resolve_kaggle_output_dir_valid(tmp_path):
    c0_dir = resolve_kaggle_output_dir(tmp_path, "C0", run_id="run_001")
    expected_c0 = tmp_path / "checkpoints" / "RQ1-TRN-02B" / "C0" / "run_001"
    assert c0_dir == expected_c0
    assert c0_dir.is_dir()

    c1_dir = resolve_kaggle_output_dir(tmp_path, "C1", run_id="run_002")
    expected_c1 = tmp_path / "checkpoints" / "RQ1-TRN-02B" / "C1" / "run_002"
    assert c1_dir == expected_c1
    assert c1_dir.is_dir()
    assert c0_dir != c1_dir


def test_resolve_kaggle_output_dir_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError, match="Illegal run ID"):
        resolve_kaggle_output_dir(tmp_path, "C0", run_id="../escaped")

    with pytest.raises(ValueError, match="Illegal run ID"):
        resolve_kaggle_output_dir(tmp_path, "C0", run_id="invalid/slash")


def test_resolve_kaggle_output_dir_conflicts(tmp_path):
    out_dir = resolve_kaggle_output_dir(tmp_path, "C0", run_id="run_test")
    (out_dir / "last.pt").write_text("existing_checkpoint")

    # Without resume: fails closed
    with pytest.raises(FileExistsError, match="Conflicting checkpoints detected"):
        resolve_kaggle_output_dir(tmp_path, "C0", run_id="run_test", allow_conflicts=False)

    # With resume: allowed
    re_dir = resolve_kaggle_output_dir(tmp_path, "C0", run_id="run_test", allow_conflicts=True)
    assert re_dir == out_dir


# --- 6. Missing Credentials & Fail-Closed Persistence ---

def test_kaggle_remote_mirror_missing_target():
    mirror = KaggleRemoteMirror(backend="local_dir", remote_target=None)
    ok, msg = mirror.verify_readiness()
    assert not ok
    assert "FAIL-CLOSED: No remote persistent mirror target configured" in msg


def test_kaggle_remote_mirror_missing_gdrive_credentials(monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GDRIVE_SA_KEY", raising=False)
    mirror = KaggleRemoteMirror(backend="gdrive", remote_target="some/remote/path")
    ok, msg = mirror.verify_readiness()
    assert not ok
    assert "requires valid credentials" in msg


def test_kaggle_remote_mirror_missing_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    mirror = KaggleRemoteMirror(backend="huggingface", remote_target="minkhoaa/repo")
    ok, msg = mirror.verify_readiness()
    assert not ok
    assert "requires valid HF_TOKEN" in msg


# --- 7. Checkpoint Synchronization & SHA-256 Readback ---

def test_kaggle_remote_mirror_sync_and_readback(tmp_path):
    remote_dir = tmp_path / "remote_persistent"
    mirror = KaggleRemoteMirror(backend="local_dir", remote_target=remote_dir)

    ok, msg = mirror.verify_readiness()
    assert ok

    local_file = tmp_path / "checkpoint.pt"
    local_file.write_bytes(b"model_weights_bytes_12345")

    res = mirror.mirror_file(local_file, "C0/checkpoint.pt", verify_sha=True)
    assert res["status"] == "SYNCED"
    assert res["local_sha256"] == res["remote_sha256"]
    assert (remote_dir / "C0" / "checkpoint.pt").is_file()


def test_kaggle_remote_mirror_detects_corruption(tmp_path, monkeypatch):
    remote_dir = tmp_path / "remote_persistent"
    mirror = KaggleRemoteMirror(backend="local_dir", remote_target=remote_dir)

    local_file = tmp_path / "checkpoint.pt"
    local_file.write_bytes(b"original_model_bytes")

    # Monkeypatch compute_sha256 to simulate corruption during remote transfer
    orig_compute = compute_sha256

    def corrupt_remote_sha(p):
        if "remote_persistent" in str(p):
            return "corrupted_remote_digest_0000000000000000"
        return orig_compute(p)

    monkeypatch.setattr("audit_scripts.kaggle_checkpoint_sync.compute_sha256", corrupt_remote_sha)

    with pytest.raises(IOError, match="Remote mirror SHA-256 mismatch"):
        mirror.mirror_file(local_file, "C0/checkpoint.pt", verify_sha=True)


# --- 8. Fail-Closed Endpoint Classification Semantics ---

def _build_valid_completed_checkpoint(arm: str = "C0", commit: str = "a" * 40) -> Dict[str, Any]:
    return {
        "model": {"dummy": torch.zeros(2)},
        "optimizer": {"state": {}},
        "scheduler": {"state": {}},
        "scaler": {"scale": 4096.0},
        "epoch": 5,
        "successful_updates": 3600,
        "step_attempts": 3600,
        "skipped_updates": 0,
        "metadata": {
            "arm": arm,
            "successful_updates": 3600,
            "primary_3600_updates_completed": True,
            "primary_endpoint_reached": True,
            "endpoint_status": "COMPLETED_3600_UPDATES",
            "guardrail_violation": False,
            "split_sha256": EXPECTED_MANIFEST_SHA256,
            "initial_checkpoint_sha256": EXPECTED_BASELINE_SHA256,
            "git_sha": commit,
        },
    }


def test_endpoint_classification_completed_success(tmp_path):
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    commit_sha = "c91aad8c1f61c2fc962553b8f595658d7ef421de"
    payload = _build_valid_completed_checkpoint("C0", commit_sha)
    torch.save(payload, out_dir / "fixed_budget_checkpoint.pt")

    result = classify_kaggle_training_endpoint(out_dir, expected_arm="C0", expected_commit=commit_sha)
    assert result["status"] == "COMPLETED_3600_UPDATES"
    assert result["successful_updates"] == 3600


def test_endpoint_classification_rejects_under_budget(tmp_path):
    out_dir = tmp_path / "output_under"
    out_dir.mkdir()
    payload = _build_valid_completed_checkpoint("C0")
    payload["successful_updates"] = 3599  # 1 update short
    payload["metadata"]["successful_updates"] = 3599
    torch.save(payload, out_dir / "fixed_budget_checkpoint.pt")

    result = classify_kaggle_training_endpoint(out_dir, expected_arm="C0")
    assert result["status"] == "FAILED"
    assert "Contradictory or incomplete" in result["message"]


def test_endpoint_classification_rejects_arm_mismatch(tmp_path):
    out_dir = tmp_path / "output_mismatch"
    out_dir.mkdir()
    payload = _build_valid_completed_checkpoint("C1")  # Checkpoint is C1
    torch.save(payload, out_dir / "fixed_budget_checkpoint.pt")

    # Classifying for C0 must fail closed
    result = classify_kaggle_training_endpoint(out_dir, expected_arm="C0")
    assert result["status"] == "FAILED"
    assert "Arm mismatch" in result["message"]


def test_endpoint_classification_rejects_guardrail_trip(tmp_path):
    out_dir = tmp_path / "output_guardrail"
    out_dir.mkdir()
    halted_payload = {
        "epoch": 2,
        "successful_updates": 1200,
        "metadata": {
            "ctc_per_measured": 0.2500,
            "ctc_per_anchor": 0.2364,
            "ctc_per_degradation_pct": 1.36,
        },
    }
    torch.save(halted_payload, out_dir / "halted_guardrail.pt")

    result = classify_kaggle_training_endpoint(out_dir, expected_arm="C0")
    assert result["status"] == "HALTED_GUARDRAIL"
    assert result["successful_updates"] == 1200


def test_endpoint_classification_incomplete_last_pt(tmp_path):
    out_dir = tmp_path / "output_incomplete"
    out_dir.mkdir()
    last_payload = {
        "epoch": 3,
        "successful_updates": 2400,
        "metadata": {"endpoint_status": "INCOMPLETE"},
    }
    torch.save(last_payload, out_dir / "last.pt")

    result = classify_kaggle_training_endpoint(out_dir, expected_arm="C0")
    assert result["status"] == "INCOMPLETE"
    assert result["successful_updates"] == 2400


# --- 9. Quota Parsing & Accelerator Gates ---

def test_parse_kaggle_quota_valid():
    sample_table = """
resource  used   remaining  total   refreshAt                  
--------  -----  ---------  ------  -------------------------  
GPU       0.00h  30.00h     30.00h  2026-10-31T00:00:00+00:00  
TPU       0.00h  20.00h     20.00h  2026-10-31T00:00:00+00:00  
"""
    quota = parse_kaggle_quota(sample_table)
    assert quota["gpu_used_hours"] == 0.0
    assert quota["gpu_remaining_hours"] == 30.0
    assert quota["gpu_total_hours"] == 30.0
    assert quota["tpu_remaining_hours"] == 20.0


def test_check_kaggle_quota_fails_on_exhaustion(monkeypatch):
    import subprocess
    exhausted_output = """
resource  used    remaining  total   refreshAt                  
GPU       30.00h  0.00h      30.00h  2026-10-31T00:00:00+00:00  
"""
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout=exhausted_output, stderr=""),
    )
    ok, msg, q = check_kaggle_quota(min_gpu_hours=5.0)
    assert not ok
    assert "exhausted" in msg


# --- 10. Kernel Metadata Invariants & Licensing ---

def test_kernel_metadata_invariants():
    meta_path = Path("deploy/kaggle/kernel-metadata.json")
    assert meta_path.is_file(), "deploy/kaggle/kernel-metadata.json must exist"

    with open(meta_path) as f:
        meta = json.load(f)

    # Invariants
    assert meta["is_private"] in (True, "true"), "Kernel must be strictly private"
    assert meta["enable_gpu"] in (True, "true"), "Kernel must enable GPU accelerator"
    assert meta["machine_shape"] == "NvidiaTeslaT4", "Machine shape must be NvidiaTeslaT4"
    assert meta["enable_internet"] in (True, "true"), "Kernel must enable internet for remote sync"


def test_dataset_metadata_invariants():
    meta_path = Path("deploy/kaggle/dataset-metadata.json")
    assert meta_path.is_file(), "deploy/kaggle/dataset-metadata.json must exist"

    with open(meta_path) as f:
        meta = json.load(f)

    assert meta["is_private"] is True, "Staged dataset must be strictly private"
    assert meta["licenses"][0]["name"] == "CC-BY-NC-4.0", "Dataset must carry CC-BY-NC-4.0 license"


# --- 11. Disposable Benchmark Contract ---

def test_stage_and_verify_kaggle_environment_full_flow(tmp_path):
    config = {
        "arm": "C0",
        "manifest": "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json",
        "initial_checkpoint": "checkpoints/RQ1-TRN-02A/recovered/best.pt",
        "zip_path": "datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip",
    }
    baseline_p = Path(config["initial_checkpoint"])
    zip_p = Path(config["zip_path"])
    if not (baseline_p.is_file() and zip_p.is_file()):
        pytest.skip("Local baseline or dataset unavailable")

    working_dir = tmp_path / "working"
    remote_mirror = tmp_path / "remote_mirror"

    report = stage_and_verify_kaggle_environment(
        config=config,
        working_dir=working_dir,
        remote_mirror_backend="local_dir",
        remote_mirror_target=str(remote_mirror),
        allow_mock_gpu=True,
        run_id="flow_test_01",
    )

    assert report["status"] == "PASS_STAGING"
    assert report["arm"] == "C0"
    assert report["run_id"] == "flow_test_01"
    assert report["dataset"]["speakers_count"] == 18
    assert report["dataset"]["test_speakers_count"] == 0
    assert report["remote_mirror"]["status"] == "VERIFIED"


def test_benchmark_config_invariants():
    bench_cfg_path = Path("configs/rq1_trn_02b_benchmark_kaggle.yaml")
    assert bench_cfg_path.is_file()

    with open(bench_cfg_path) as f:
        cfg = yaml.safe_load(f)

    assert cfg["max_successful_updates"] == 50
    assert cfg["disposable_benchmark"] is True
    assert cfg["counts_toward_c0"] is False
    assert cfg["arm"] == "BENCHMARK"
    assert cfg["amp"] == "fp16"
    assert cfg["batch_size"] == 2
    assert cfg["gradient_accumulation_steps"] == 2

