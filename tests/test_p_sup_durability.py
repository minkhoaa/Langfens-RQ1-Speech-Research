"""Tests for P-SUP Training Durability and Checkpoint Management.

Covers:
1. Remote Persistent Mirroring:
   - Atomic upload with cryptographic SHA-256 readback probe (local_dir & mock huggingface).
   - Detection of remote corruption or mismatch with fail-closed halt.
   - Missing credentials / targets fail-closed.
   - Rejection of ephemeral /kaggle/working as remote target.
2. Checkpoint Save and Resume Roundtrip:
   - Full payload completeness: head, optimizer, scheduler, scaler, RNG, counters, metadata.
   - Exact weight and optimizer state restoration.
   - Deterministic RNG restoration.
   - Recovery of missing local checkpoint from remote mirror.
3. Fail-Closed Durability Checks:
   - Prohibition of ephemeral storage (/kaggle/working) as sole durable store without verified remote mirror.
   - Checkpoint payload corruption, truncated keys, and counter boundary validations.
4. Fail-Closed CUDA / Tesla T4 / FP16 AMP Checks:
   - Verification of torch.cuda.is_available() in remote execution mode.
   - Device type validation (CUDA only).
   - Tesla T4 identity verification.
   - FP16 AMP capability verification.
5. Exact Accounting:
   - Optimizer advances ONLY on unskipped steps without overflow.
   - 3,600 maximum update budget ceiling.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from langfens_prep.p_sup.model import CompactOperationHead, HeadConfig
from langfens_prep.p_sup.planning import UpdateLedger, learning_rate_multiplier
from langfens_prep.p_sup.trainer import (
    REQUIRED_CHECKPOINT_KEYS,
    RemoteCheckpointMirror,
    compute_file_sha256,
    is_ephemeral_path,
    resume_from_checkpoint,
    save_checkpoint,
    validate_checkpoint_payload,
    verify_cuda_t4_environment,
)


# ==============================================================================
# 1. Remote Mirror & SHA-256 Readback Probe Tests
# ==============================================================================

def test_remote_mirror_local_dir_probe_and_upload(tmp_path):
    """Verify local_dir backend performs write-readback-delete probe and genuine atomic mirroring."""
    remote_dir = tmp_path / "persistent_remote"
    mirror = RemoteCheckpointMirror(
        backend="local_dir",
        remote_target=remote_dir,
        allow_local_test=True,
    )

    # 1. Readiness probe (write-readback-delete)
    ok, msg = mirror.verify_readiness()
    assert ok, f"Readiness probe failed: {msg}"
    assert "verified" in msg

    # 2. Atomic upload & SHA-256 readback
    local_file = tmp_path / "epoch_1_last.pt"
    content = b"LANGFENS_CHECKPOINT_BYTES_EPOCH_1"
    local_file.write_bytes(content)
    expected_sha = hashlib.sha256(content).hexdigest()

    res = mirror.mirror_file(local_file, "T-OP/last.pt", verify_sha=True)
    assert res["status"] == "SYNCED"
    assert res["backend"] == "local_dir"
    assert res["local_sha256"] == expected_sha
    assert res["remote_sha256"] == expected_sha
    assert (remote_dir / "T-OP" / "last.pt").is_file()
    assert compute_file_sha256(remote_dir / "T-OP" / "last.pt") == expected_sha

    # 3. Download and readback
    download_dest = tmp_path / "downloaded_last.pt"
    out_p = mirror.download_file("T-OP/last.pt", download_dest, verify_sha=True)
    assert out_p.is_file()
    assert compute_file_sha256(out_p) == expected_sha


def test_remote_mirror_local_dir_readback_corruption_fails_closed(tmp_path, monkeypatch):
    """If remote file is corrupted during sync, mirror_file must raise IOError."""
    remote_dir = tmp_path / "persistent_remote"
    mirror = RemoteCheckpointMirror(
        backend="local_dir",
        remote_target=remote_dir,
        allow_local_test=True,
    )
    ok, msg = mirror.verify_readiness()
    assert ok

    local_file = tmp_path / "corrupt_test.pt"
    local_file.write_bytes(b"ORIGINAL_BYTES")

    orig_compute = compute_file_sha256

    def corrupt_compute(p):
        if "persistent_remote" in str(p):
            return "0000000000000000000000000000000000000000000000000000000000000000"
        return orig_compute(p)

    monkeypatch.setattr("langfens_prep.p_sup.trainer.compute_file_sha256", corrupt_compute)

    with pytest.raises(IOError, match="Remote mirror SHA-256 mismatch"):
        mirror.mirror_file(local_file, "T-OP/corrupt_test.pt", verify_sha=True)


def test_remote_mirror_huggingface_mock_probe_and_upload(tmp_path, monkeypatch):
    """Mock Hugging Face Hub HfApi upload and hf_hub_download readback."""
    storage = {}

    class MockHfApi:
        def __init__(self, token=None):
            self.token = token

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type="model"):
            p = Path(path_or_fileobj)
            storage[(repo_id, path_in_repo)] = p.read_bytes()

        def delete_file(self, path_in_repo, repo_id, repo_type="model"):
            storage.pop((repo_id, path_in_repo), None)

        def list_repo_files(self, repo_id, repo_type="model"):
            return [path for (r, path) in storage.keys() if r == repo_id]

    def mock_hf_hub_download(repo_id, filename, repo_type="model", token=None, local_dir=None, force_download=True):
        key = (repo_id, filename)
        if key not in storage:
            raise FileNotFoundError(f"File {filename} not in mock repo {repo_id}")
        dest_dir = Path(local_dir) if local_dir else tmp_path / "hf_cache"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / Path(filename).name
        dest_file.write_bytes(storage[key])
        return str(dest_file)

    monkeypatch.setattr("huggingface_hub.HfApi", MockHfApi)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", mock_hf_hub_download)

    mirror = RemoteCheckpointMirror(
        backend="huggingface",
        remote_target="minkhoaa/test-rq1-checkpoints",
        auth_token="hf_mock_valid_token_secret",
    )

    # 1. Readiness probe
    ok, msg = mirror.verify_readiness()
    assert ok, f"HF readiness probe failed: {msg}"
    assert "verified" in msg

    # 2. Upload file
    local_file = tmp_path / "best.pt"
    local_bytes = b"BEST_CHECKPOINT_DATA_AP_0.42"
    local_file.write_bytes(local_bytes)
    local_sha = hashlib.sha256(local_bytes).hexdigest()

    res = mirror.mirror_file(local_file, "T-OP/best.pt", verify_sha=True)
    assert res["status"] == "SYNCED"
    assert res["backend"] == "huggingface"
    assert res["local_sha256"] == local_sha
    assert res["remote_sha256"] == local_sha
    assert ("minkhoaa/test-rq1-checkpoints", "T-OP/best.pt") in storage
    assert mirror.file_exists("T-OP/best.pt") is True

    # 3. Download file
    download_dest = tmp_path / "downloaded_best.pt"
    downloaded = mirror.download_file("T-OP/best.pt", download_dest, verify_sha=True)
    assert downloaded.is_file()
    assert compute_file_sha256(downloaded) == local_sha


def test_remote_mirror_huggingface_readback_sha_mismatch_fails_closed(tmp_path, monkeypatch):
    """HF Hub readback returning mismatched digest must raise IOError."""
    class MockHfApi:
        def __init__(self, token=None): pass
        def upload_file(self, *args, **kwargs): pass
        def delete_file(self, *args, **kwargs): pass

    def mock_hf_hub_download_corrupt(*args, **kwargs):
        dest_file = tmp_path / "corrupted_download.tmp"
        dest_file.write_bytes(b"CORRUPTED_READBACK_BYTES")
        return str(dest_file)

    monkeypatch.setattr("huggingface_hub.HfApi", MockHfApi)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", mock_hf_hub_download_corrupt)

    mirror = RemoteCheckpointMirror(
        backend="huggingface",
        remote_target="minkhoaa/test-rq1-checkpoints",
        auth_token="hf_mock_token",
        require_verification=False,
    )

    local_file = tmp_path / "last.pt"
    local_file.write_bytes(b"GENUINE_LOCAL_BYTES")

    with pytest.raises(IOError, match="Hugging Face remote mirror SHA-256 readback mismatch"):
        mirror.mirror_file(local_file, "T-OP/last.pt", verify_sha=True)


def test_remote_mirror_missing_target_and_token(monkeypatch):
    """Remote mirror must fail-closed when persistent target or HF token is missing."""
    # Missing target
    mirror_no_target = RemoteCheckpointMirror(backend="local_dir", remote_target="")
    ok, msg = mirror_no_target.verify_readiness()
    assert not ok
    assert "No remote persistent mirror target configured" in msg

    # Missing HF token
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.delenv("KAGGLE_REMOTE_MIRROR_TOKEN", raising=False)

    mirror_no_token = RemoteCheckpointMirror(
        backend="huggingface",
        remote_target="minkhoaa/repo",
        auth_token=None,
    )
    ok, msg = mirror_no_token.verify_readiness()
    assert not ok
    assert "requires valid HF_TOKEN" in msg

    with pytest.raises(RuntimeError, match=r"requires valid HF_TOKEN|Hugging Face token missing"):
        tmp_dummy = Path(tempfile.gettempdir()) / "dummy.pt"
        tmp_dummy.write_bytes(b"dummy")
        mirror_no_token.mirror_file(tmp_dummy, "dummy.pt")


def test_remote_mirror_rejects_ephemeral_kaggle_path(monkeypatch):
    """Remote mirror target must fail-closed if pointed to ephemeral /kaggle path."""
    mirror = RemoteCheckpointMirror(
        backend="local_dir",
        remote_target="/kaggle/working/mirror",
        allow_local_test=False,
    )
    ok, msg = mirror.verify_readiness()
    assert not ok
    assert "Ephemeral /kaggle path" in msg


# ==============================================================================
# 2. Checkpoint Save and Resume Roundtrip Tests
# ==============================================================================

def test_checkpoint_save_and_resume_exact_roundtrip(tmp_path):
    """Full roundtrip restoring model weights, optimizer state, scheduler, scaler, RNG, and update counters."""
    head = CompactOperationHead(HeadConfig())
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate_multiplier)
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)

    # Advance optimizer state with one synthetic update
    feat = torch.randn(2, 5, 768)
    prompt = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    out = head(
        features=feat,
        frame_lengths=torch.tensor([5, 5]),
        prompt_phone_ids=prompt,
        prompt_lengths=torch.tensor([2, 2]),
    )
    loss = out["operation_logits"].sum() + out["ctc_logits"].sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    # Capture original weights and optimizer states
    orig_head_state = {k: v.clone() for k, v in head.state_dict().items()}
    orig_opt_state = optimizer.state_dict()

    ckpt_path = tmp_path / "last.pt"
    metadata = {
        "registration": "P-SUP-COMPACT-OP-V1",
        "objective": "operation_ce",
        "seed": 1337,
    }
    history = [{"epoch": 0, "val_ap": 0.32}, {"epoch": 1, "val_ap": 0.35}]

    sha = save_checkpoint(
        path=ckpt_path,
        head=head,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=1,
        successful_updates=150,
        micro_steps=300,
        best_ap=0.35,
        patience=0,
        skipped_amp=2,
        history=history,
        metadata=metadata,
        is_fixed_budget_endpoint=False,
    )
    assert ckpt_path.is_file()
    assert compute_file_sha256(ckpt_path) == sha

    # Resume from checkpoint without pre-instantiated head or optimizer
    resumed = resume_from_checkpoint(ckpt_path)

    # 1. Update counters and scalar state
    assert resumed["epoch"] == 1
    assert resumed["successful_updates"] == 150
    assert resumed["micro_steps"] == 300
    assert resumed["best_ap"] == pytest.approx(0.35)
    assert resumed["patience"] == 0
    assert resumed["skipped_amp_attempts_total"] == 2
    assert resumed["is_fixed_budget_endpoint"] is False
    assert resumed["metadata"]["objective"] == "operation_ce"
    assert len(resumed["history"]) == 2

    # 2. Model weights bitwise parity
    res_head_state = resumed["head"].state_dict()
    for k, v in orig_head_state.items():
        assert k in res_head_state, f"Missing key {k} in resumed head state dict"
        assert torch.equal(v, res_head_state[k]), f"Weight mismatch for parameter {k}"

    # 3. Optimizer state parity
    res_opt_state = resumed["optimizer"].state_dict()
    assert res_opt_state["param_groups"] == orig_opt_state["param_groups"]
    assert set(res_opt_state["state"].keys()) == set(orig_opt_state["state"].keys())
    for p_id in orig_opt_state["state"]:
        for state_key in orig_opt_state["state"][p_id]:
            v_orig = orig_opt_state["state"][p_id][state_key]
            v_res = res_opt_state["state"][p_id][state_key]
            if isinstance(v_orig, torch.Tensor):
                assert torch.equal(v_orig, v_res), f"Optimizer state mismatch for {state_key}"
            else:
                assert v_orig == v_res, f"Optimizer state scalar mismatch for {state_key}"


def test_checkpoint_resume_into_existing_instances(tmp_path):
    """Resume correctly restores weights and states into pre-existing head and optimizer instances."""
    head = CompactOperationHead(HeadConfig())
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate_multiplier)
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)

    ckpt_path = tmp_path / "resume_in_place.pt"
    save_checkpoint(
        path=ckpt_path,
        head=head,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=2,
        successful_updates=300,
        micro_steps=600,
        best_ap=0.39,
        patience=1,
        skipped_amp=0,
        history=[],
        metadata={"objective": "binary_collapsed"},
    )

    new_head = CompactOperationHead(HeadConfig())
    new_opt = torch.optim.AdamW(new_head.parameters(), lr=0.001)
    new_sched = torch.optim.lr_scheduler.LambdaLR(new_opt, lr_lambda=learning_rate_multiplier)

    resumed = resume_from_checkpoint(
        ckpt_path,
        head=new_head,
        optimizer=new_opt,
        scheduler=new_sched,
    )
    assert resumed["head"] is new_head
    assert resumed["optimizer"] is new_opt
    assert resumed["scheduler"] is new_sched
    assert resumed["successful_updates"] == 300


def test_checkpoint_rng_restoration_determinism(tmp_path):
    """RNG state restored from checkpoint produces identical random sequences across random, numpy, torch."""
    random.seed(9999)
    np.random.seed(9999)
    torch.manual_seed(9999)

    head = CompactOperationHead(HeadConfig())
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate_multiplier)
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)

    ckpt_path = tmp_path / "rng_test.pt"
    save_checkpoint(
        path=ckpt_path,
        head=head,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=0,
        successful_updates=0,
        micro_steps=0,
        best_ap=-1.0,
        patience=0,
        skipped_amp=0,
        history=[],
        metadata={},
    )

    # Sample random stream immediately after saving
    py_expected = [random.random() for _ in range(5)]
    np_expected = np.random.rand(5).tolist()
    torch_expected = torch.randn(5).tolist()

    # Alter RNG state
    _ = [random.random() for _ in range(100)]
    _ = np.random.rand(100)
    _ = torch.randn(100)

    # Resume and verify exact restoration
    resume_from_checkpoint(ckpt_path)
    py_actual = [random.random() for _ in range(5)]
    np_actual = np.random.rand(5).tolist()
    torch_actual = torch.randn(5).tolist()

    assert py_actual == py_expected
    assert np.allclose(np_actual, np_expected)
    assert np.allclose(torch_actual, torch_expected)


def test_checkpoint_fixed_budget_3600_metadata_lock(tmp_path):
    """Fixed-budget endpoint checkpoint locks updates=3600 and is_fixed_budget_endpoint=True."""
    head = CompactOperationHead(HeadConfig())
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate_multiplier)
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)

    fb_path = tmp_path / "fixed_budget_3600.pt"
    save_checkpoint(
        path=fb_path,
        head=head,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=5,
        successful_updates=3600,
        micro_steps=7200,
        best_ap=0.38,
        patience=0,
        skipped_amp=1,
        history=[],
        metadata={"objective": "operation_ce"},
        is_fixed_budget_endpoint=True,
    )

    res = resume_from_checkpoint(fb_path)
    assert res["successful_updates"] == 3600
    assert res["is_fixed_budget_endpoint"] is True


# ==============================================================================
# 3. Fail-Closed Durability & Ephemeral Storage Checks
# ==============================================================================

def test_fail_closed_ephemeral_storage_without_mirror(tmp_path):
    """Saving to ephemeral container path (/kaggle/working) without remote mirror fails closed."""
    head = CompactOperationHead(HeadConfig())
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate_multiplier)
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)

    ephemeral_path = Path("/kaggle/working/checkpoints/last.pt")

    with pytest.raises(RuntimeError, match="FAIL-CLOSED: Checkpoint path .* is on ephemeral storage"):
        save_checkpoint(
            path=ephemeral_path,
            head=head,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=0,
            successful_updates=0,
            micro_steps=0,
            best_ap=-1.0,
            patience=0,
            skipped_amp=0,
            history=[],
            metadata={},
            remote_mirror=None,
        )


def test_fail_closed_require_remote_mirror_flag(tmp_path):
    """When require_remote_mirror=True, missing mirror raises RuntimeError even on non-ephemeral path."""
    head = CompactOperationHead(HeadConfig())
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate_multiplier)
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)

    with pytest.raises(RuntimeError, match="FAIL-CLOSED: Checkpoint path .* without a verified remote persistent mirror"):
        save_checkpoint(
            path=tmp_path / "last.pt",
            head=head,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=0,
            successful_updates=0,
            micro_steps=0,
            best_ap=-1.0,
            patience=0,
            skipped_amp=0,
            history=[],
            metadata={},
            remote_mirror=None,
            require_remote_mirror=True,
        )


def test_fail_closed_save_checkpoint_detects_remote_mirror_mismatch(tmp_path):
    """save_checkpoint fails closed if remote mirror reports SHA-256 mismatch."""
    head = CompactOperationHead(HeadConfig())
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate_multiplier)
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)

    class CorruptMockMirror:
        def mirror_file(self, path, remote_rel_path, verify_sha=True):
            return {
                "status": "SYNCED",
                "backend": "mock_corrupt",
                "local_sha256": "correct_digest",
                "remote_sha256": "corrupted_digest_000000",
            }

    with pytest.raises(IOError, match="Remote mirror SHA-256 mismatch"):
        save_checkpoint(
            path=tmp_path / "last.pt",
            head=head,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=0,
            successful_updates=0,
            micro_steps=0,
            best_ap=-1.0,
            patience=0,
            skipped_amp=0,
            history=[],
            metadata={},
            remote_mirror=CorruptMockMirror(),
        )


def test_fail_closed_resume_missing_file_without_mirror(tmp_path):
    """Resuming non-existent file without remote mirror raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="FAIL-CLOSED: Checkpoint not found"):
        resume_from_checkpoint(tmp_path / "non_existent.pt")


def test_resume_recovers_missing_checkpoint_from_remote_mirror(tmp_path):
    """Resuming non-existent local file recovers it from remote mirror if available."""
    remote_dir = tmp_path / "remote_store"
    remote_dir.mkdir(parents=True, exist_ok=True)

    # Create valid checkpoint directly in remote store
    head = CompactOperationHead(HeadConfig())
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate_multiplier)
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)

    remote_file = remote_dir / "last.pt"
    save_checkpoint(
        path=remote_file,
        head=head,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=3,
        successful_updates=200,
        micro_steps=400,
        best_ap=0.36,
        patience=0,
        skipped_amp=0,
        history=[],
        metadata={"objective": "operation_ce"},
    )

    mirror = RemoteCheckpointMirror(
        backend="local_dir",
        remote_target=remote_dir,
        allow_local_test=True,
    )

    # Local file does not exist yet
    local_target = tmp_path / "recovered_last.pt"
    assert not local_target.is_file()

    res = resume_from_checkpoint(
        local_target,
        remote_mirror=mirror,
        remote_rel_path="last.pt",
    )
    assert local_target.is_file()
    assert res["successful_updates"] == 200
    assert res["epoch"] == 3


@pytest.mark.parametrize("missing_key", [
    "head", "optimizer", "scheduler", "scaler", "epoch", "micro_steps",
    "successful_updates", "skipped_amp_attempts_total", "best_ap", "history", "rng", "metadata"
])
def test_fail_closed_resume_missing_required_keys(tmp_path, missing_key):
    """Payload missing any required top-level key must fail closed with ValueError."""
    payload = {
        "head": {}, "optimizer": {}, "scheduler": {}, "scaler": {},
        "epoch": 0, "micro_steps": 0, "successful_updates": 0,
        "skipped_amp_attempts_total": 0, "best_ap": -1.0, "history": [],
        "rng": {"python": (), "numpy": (), "torch": torch.ByteTensor()},
        "metadata": {},
    }
    del payload[missing_key]
    corrupt_path = tmp_path / f"corrupt_{missing_key}.pt"
    torch.save(payload, corrupt_path)

    with pytest.raises(ValueError, match="missing required key"):
        resume_from_checkpoint(corrupt_path)


@pytest.mark.parametrize("invalid_update", [-1, 3601, "3600", None])
def test_fail_closed_resume_invalid_update_counter(tmp_path, invalid_update):
    """Out-of-bound update counters must fail closed with ValueError."""
    payload = {
        "head": {}, "optimizer": {}, "scheduler": {}, "scaler": {},
        "epoch": 0, "micro_steps": 0, "successful_updates": invalid_update,
        "skipped_amp_attempts_total": 0, "best_ap": -1.0, "history": [],
        "rng": {"python": (), "numpy": (), "torch": torch.ByteTensor()},
        "metadata": {},
    }
    corrupt_path = tmp_path / "corrupt_update.pt"
    torch.save(payload, corrupt_path)

    with pytest.raises(ValueError, match="successful_updates must be in"):
        resume_from_checkpoint(corrupt_path)


# ==============================================================================
# 4. Fail-Closed CUDA / Tesla T4 / FP16 AMP Checks
# ==============================================================================

def test_cuda_t4_checks_fail_closed_when_cuda_missing(monkeypatch):
    """Remote execution without CUDA availability must fail closed immediately."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="FAIL-CLOSED: CUDA is not available"):
        verify_cuda_t4_environment(remote_execution=True)


def test_cuda_t4_checks_fail_closed_when_device_not_cuda(monkeypatch):
    """Remote execution with non-CUDA device must fail closed."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="FAIL-CLOSED: Expected CUDA device in remote execution mode"):
        verify_cuda_t4_environment(remote_execution=True, device="cpu")


def test_cuda_t4_checks_fail_closed_when_non_t4_gpu(monkeypatch):
    """Remote execution on unauthorized GPU (e.g. RTX 4090, A100) must fail closed."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "NVIDIA RTX 4090")

    with pytest.raises(RuntimeError, match="does not match required Nvidia Tesla T4"):
        verify_cuda_t4_environment(remote_execution=True, require_t4=True)


def test_cuda_t4_checks_fail_closed_when_fp16_amp_unsupported(monkeypatch):
    """Old GPU architecture without FP16 AMP support must fail closed."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "Tesla T4")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda idx: (3, 5))

    with pytest.raises(RuntimeError, match="Compute capability .* does not support FP16 AMP"):
        verify_cuda_t4_environment(remote_execution=True, require_t4=True)


def test_cuda_t4_checks_pass_in_local_testing_mode():
    """In local mode (remote_execution=False), absence of CUDA must not raise."""
    info = verify_cuda_t4_environment(remote_execution=False)
    assert info["remote_execution"] is False
    assert info["status"] == "LOCAL_OK"


def test_cuda_t4_checks_mock_t4_success(monkeypatch):
    """Tesla T4 with capability 7.5 and working AMP probe passes verification."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "Tesla T4")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda idx: (7, 5))

    # Mock working autocast context manager and tensor probe
    class MockAutocast:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

    monkeypatch.setattr(torch.amp, "autocast", MockAutocast)
    orig_ones = torch.ones
    def mock_ones(shape, device=None, dtype=None):
        return orig_ones(shape, dtype=dtype)
    monkeypatch.setattr(torch, "ones", mock_ones)

    info = verify_cuda_t4_environment(remote_execution=True, require_t4=True)
    assert info["status"] == "VERIFIED_TESLA_T4"
    assert info["fp16_amp_supported"] is True


# ==============================================================================
# 5. Exact Accounting Tests
# ==============================================================================

def test_exact_accounting_advances_only_on_unskipped_steps():
    """Ledger advances successful_updates strictly on non-overflow steps and caps at 3,600."""
    ledger = UpdateLedger()

    # Step 1: Overflow (skipped)
    unskipped = ledger.record_attempt(overflow=True)
    assert unskipped is False
    assert ledger.successful == 0
    assert ledger.skipped == 1
    assert ledger.attempts == 1

    # Step 2: Clean step (successful)
    unskipped = ledger.record_attempt(overflow=False)
    assert unskipped is True
    assert ledger.successful == 1
    assert ledger.skipped == 1
    assert ledger.attempts == 2

    # Advance to 3,600
    for _ in range(3599):
        ledger.record_attempt(overflow=False)

    assert ledger.successful == 3600
    assert ledger.fixed_budget_reached is True

    # Any attempt after 3,600 must raise ValueError
    with pytest.raises(ValueError, match="invalid attempt/budget exhausted"):
        ledger.record_attempt(overflow=False)
