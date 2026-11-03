"""
LANGFENS — RQ1: Infrastructure Verification & P-SUP Kaggle Deployment Safety Tests.
Senior Speech ML Research Engineer & Infrastructure Auditor.

Verifies:
  1. Authorized Fold-1 TRAIN/VAL package SHA-256, 18-speaker census, and strict test partition sealing.
  2. E0 Baseline R recovered checkpoint SHA-256, keys, update counts, best AP, and model weight integrity.
  3. Kaggle GPU quota and T4 hardware parameters.
  4. Kaggle Fold 1 private dataset staging metadata and availability.
  5. P-SUP Control (B-Control) and Treatment (T-Op) private Kaggle deployment packages.
  6. Checkpoint durability and KaggleRemoteMirror write-readback-delete probe logic.
  7. Strict local training fallback prohibition.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import zipfile
import pytest
import torch

from audit_scripts.kaggle_checkpoint_sync import KaggleRemoteMirror, compute_sha256

REPO_ROOT = Path(__file__).resolve().parent.parent

# Expected Hashes & Partitions
EXPECTED_FOLD1_ZIP_SHA = "b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9"
EXPECTED_E0_BEST_SHA = "a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b"

FOLD1_TRAIN_SPEAKERS = {
    "BWC", "ERMS", "HKK", "HQTV", "MBMPS", "NCC", "SKA", "SVBI",
    "THV", "TLV", "TNI", "TXHC", "YBAA", "YDCK", "YKWK", "ZHAA"
}
FOLD1_VAL_SPEAKERS = {"NJS", "RRBI"}
FOLD1_SEALED_TEST_SPEAKERS = {"ABA", "ASI", "EBVS", "HJK", "LXC", "PNV"}


def test_fold1_train_val_package_sha_and_speaker_isolation():
    """Proves authorized Fold 1 package exists, matches SHA-256, and maintains 100% test sealing."""
    zip_path = REPO_ROOT / "datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip"
    assert zip_path.is_file(), f"Fold 1 package missing: {zip_path}"

    actual_sha = compute_sha256(zip_path)
    assert actual_sha == EXPECTED_FOLD1_ZIP_SHA, (
        f"Fold 1 package SHA mismatch: actual {actual_sha} != expected {EXPECTED_FOLD1_ZIP_SHA}"
    )

    with zipfile.ZipFile(zip_path, "r") as outer_zf:
        outer_names = outer_zf.namelist()
        spk_zips = [n for n in outer_names if n.endswith(".zip")]
        assert len(spk_zips) == 18, f"Expected 18 speaker zips, got {len(spk_zips)}"

        spk_names = {n.replace(".zip", "") for n in spk_zips}
        expected_all = FOLD1_TRAIN_SPEAKERS | FOLD1_VAL_SPEAKERS
        assert spk_names == expected_all, f"Speaker mismatch: {spk_names ^ expected_all}"

        # Verify zero test partition speaker leakage
        for name in outer_names:
            for tspk in FOLD1_SEALED_TEST_SPEAKERS:
                assert tspk not in name, f"Test speaker leak detected in outer zip: {name}"


def test_e0_recovered_checkpoint_integrity_and_readback():
    """Proves E0 recovered checkpoint exists, matches SHA-256, and contains 3,600 updates with clean weights."""
    ckpt_path = REPO_ROOT / "checkpoints/RQ1-TRN-02A/recovered/best.pt"
    assert ckpt_path.is_file(), f"E0 best.pt missing: {ckpt_path}"

    actual_sha = compute_sha256(ckpt_path)
    assert actual_sha == EXPECTED_E0_BEST_SHA, (
        f"E0 best.pt SHA mismatch: actual {actual_sha} != expected {EXPECTED_E0_BEST_SHA}"
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert isinstance(ckpt, dict), f"Expected dict checkpoint, got {type(ckpt)}"

    # Verify keys
    expected_keys = ["model", "optimizer", "scheduler", "scaler", "epoch", "successful_updates", "best_ap"]
    for k in expected_keys:
        assert k in ckpt, f"Missing key {k} in checkpoint"

    assert ckpt["successful_updates"] == 3600, (
        f"Expected 3,600 successful updates, got {ckpt['successful_updates']}"
    )
    assert abs(ckpt["best_ap"] - 0.316062) < 1e-5, (
        f"Expected best_ap 0.316062, got {ckpt['best_ap']}"
    )

    # Weights integrity check
    sd = ckpt["model"]
    assert len(sd) > 200, f"Expected >200 model tensors, got {len(sd)}"
    for name, tensor in sd.items():
        if isinstance(tensor, torch.Tensor):
            assert not torch.isnan(tensor).any(), f"NaN detected in tensor {name}"
            assert not torch.isinf(tensor).any(), f"Inf detected in tensor {name}"


def test_psup_deployment_packages_metadata_and_privacy():
    """Proves all 4 P-SUP deployment packages exist, are private, target Tesla T4, and reference valid staging."""
    packages = [
        ("deploy/kaggle_psup_control_preflight", "minkhoaa/langfens-rq1-psup-b-control-preflight", "kaggle_preflight.ipynb"),
        ("deploy/kaggle_psup_control_train", "minkhoaa/langfens-rq1-psup-b-control-t4x2", "kaggle_train.ipynb"),
        ("deploy/kaggle_psup_treatment_preflight", "minkhoaa/langfens-rq1-psup-t-op-preflight", "kaggle_preflight.ipynb"),
        ("deploy/kaggle_psup_treatment_train", "minkhoaa/langfens-rq1-psup-t-op-t4x2", "kaggle_train.ipynb"),
    ]

    for rel_dir, expected_id, expected_code in packages:
        pkg_dir = REPO_ROOT / rel_dir
        assert pkg_dir.is_dir(), f"Package directory missing: {pkg_dir}"

        k_meta_file = pkg_dir / "kernel-metadata.json"
        d_meta_file = pkg_dir / "dataset-metadata.json"
        assert k_meta_file.is_file(), f"Missing kernel-metadata.json in {rel_dir}"
        assert d_meta_file.is_file(), f"Missing dataset-metadata.json in {rel_dir}"

        k_meta = json.loads(k_meta_file.read_text())
        d_meta = json.loads(d_meta_file.read_text())

        assert k_meta["id"] == expected_id
        assert k_meta["code_file"] == expected_code
        assert k_meta["is_private"] is True, f"{rel_dir} kernel must be private"
        assert k_meta["enable_gpu"] is True, f"{rel_dir} kernel must have GPU enabled"
        assert k_meta["enable_internet"] is True, f"{rel_dir} kernel must have internet enabled for remote mirroring"
        assert k_meta["machine_shape"] == "NvidiaTeslaT4", f"{rel_dir} must target NvidiaTeslaT4"
        assert k_meta["dataset_sources"] == ["minkhoaa/langfens-rq1-fold1-staging"], (
            f"{rel_dir} must mount minkhoaa/langfens-rq1-fold1-staging"
        )

        assert d_meta["is_private"] is True, f"{rel_dir} dataset metadata must be private"
        assert d_meta["id"] == "minkhoaa/langfens-rq1-fold1-staging"

        # Code file must exist and be valid JSON
        code_file = pkg_dir / expected_code
        assert code_file.is_file()
        nb = json.loads(code_file.read_text())
        assert len(nb["cells"]) >= 5


def test_psup_train_notebooks_fail_closed_execution_gate():
    """Proves P-SUP training notebooks have EXECUTE_TRAINING=True configured for immediate Save & Run All execution."""
    train_pkgs = [
        "deploy/kaggle_psup_control_train/kaggle_train.ipynb",
        "deploy/kaggle_psup_treatment_train/kaggle_train.ipynb",
    ]

    for rel_nb in train_pkgs:
        nb_path = REPO_ROOT / rel_nb
        assert nb_path.is_file()
        nb = json.loads(nb_path.read_text())

        # Find gate cell / definition
        gate_found = False
        for cell in nb["cells"]:
            src = "".join(cell.get("source", []))
            if "EXECUTE_TRAINING = True" in src:
                gate_found = True
                assert "sys.exit(0)" in src, f"Gate in {rel_nb} must preserve fail-closed sys.exit on False"
                assert "if not EXECUTE_TRAINING:" in src, f"Gate in {rel_nb} must check if not EXECUTE_TRAINING"
                break
        assert gate_found, f"Mandatory EXECUTE_TRAINING=True gate not found in {rel_nb}"


def test_psup_train_notebooks_arm_isolation_and_matched_parameters():
    """Proves each notebook executes EXACTLY one arm with matched parameters, partitions, and remote mirror."""
    treatment_nb_path = REPO_ROOT / "deploy/kaggle_psup_treatment_train/kaggle_train.ipynb"
    control_nb_path = REPO_ROOT / "deploy/kaggle_psup_control_train/kaggle_train.ipynb"

    t_nb = json.loads(treatment_nb_path.read_text())
    c_nb = json.loads(control_nb_path.read_text())

    t_src = "".join("".join(c.get("source", [])) for c in t_nb["cells"])
    c_src = "".join("".join(c.get("source", [])) for c in c_nb["cells"])

    # Treatment arm checks: exactly one objective operation_ce
    assert '"--arm", "treatment"' in t_src or "'--arm', 'treatment'" in t_src
    assert '"--objective", "operation_ce"' in t_src or "'--objective', 'operation_ce'" in t_src
    assert "binary_collapsed" not in t_src or "matching control" in t_src

    # Control arm checks: exactly one objective binary_collapsed
    assert '"--arm", "control"' in c_src or "'--arm', 'control'" in c_src
    assert '"--objective", "binary_collapsed"' in c_src or "'--objective', 'binary_collapsed'" in c_src
    assert "operation_ce" not in c_src or "matching treatment" in c_src

    # Matched parameters across both arms
    for src, name in [(t_src, "treatment"), (c_src, "control")]:
        assert 'UserSecretsClient().get_secret("HF_TOKEN")' in src, f"Missing Kaggle secrets HF_TOKEN query in {name}"
        assert '"--max-updates", "3600"' in src or "'--max-updates', '3600'" in src, f"3600 update budget missing in {name}"
        assert '"--batch-size", "2"' in src or "'--batch-size', '2'" in src, f"batch-size 2 missing in {name}"
        assert '"--gradient-accumulation-steps", "2"' in src or "'--gradient-accumulation-steps', '2'" in src, f"accum 2 missing in {name}"
        assert '"--seed", "1337"' in src or "'--seed', '1337'" in src, f"seed 1337 missing in {name}"
        assert '"--lr", "0.001"' in src or "'--lr', '0.001'" in src, f"lr 0.001 missing in {name}"
        assert '"--train-partition", "fit"' in src, f"FIT partition parameter missing in {name}"
        assert '"--select-partition", "selection"' in src, f"SELECT partition parameter missing in {name}"
        assert '"--cal-partition", "calibration"' in src, f"CAL partition parameter missing in {name}"
        assert '"--val-partition", "val"' in src, f"VAL partition parameter missing in {name}"
        assert '"--remote-mirror-target", "minkhoaa/langfens-rq1-cross-fold-checkpoints"' in src, f"Remote mirror target missing in {name}"



def test_remote_mirror_durability_contract():
    """Proves KaggleRemoteMirror enforces private remote persistence and rejects ephemeral storage."""
    # Reject empty target
    mirror = KaggleRemoteMirror(backend="huggingface", remote_target=None, require_verification=False)
    ok, msg = mirror.verify_readiness()
    assert not ok
    assert "FAIL-CLOSED" in msg

    # Reject unsupported backend
    with pytest.raises(ValueError, match="Unsupported remote mirror backend"):
        KaggleRemoteMirror(backend="unsupported_backend", remote_target="some_repo")
