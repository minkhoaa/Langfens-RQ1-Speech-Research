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
8. Safe bounded dataset package inspection (inner paths, no extraction of sealed TEST).
9. Deterministic bootstrap on fresh VM (clones repo, checks out target commit).
10. Bootstrap refusal on dirty or unrelated pre-existing repositories.
11. Jupyter notebook kernel execution compatibility (no dependence on __file__, ignores ipykernel args).
12. Single predictable output directory (no duplicate run-ID nesting).
13. Authentic Drive persistence vs fake local directory detection.
14. Primary endpoint classification across all four states (COMPLETED_3600_UPDATES, INCOMPLETE, HALTED_GUARDRAIL, FAILED).
15. Fail-closed readiness checks returning non-zero exit code when blocked.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict

import pytest

from audit_scripts.colab_bootstrap import (
    bootstrap_repository,
    validate_commit_sha as bootstrap_validate_sha,
)
from audit_scripts.colab_launch_workflow import (
    check_quota_status,
    evaluate_readiness_layers,
    generate_colab_arm_plan,
    parse_colab_usage,
    validate_arm,
    validate_commit_sha,
    validate_static_software,
)
from audit_scripts.colab_stage_and_verify import (
    check_conflicting_checkpoints,
    compute_sha256,
    resolve_drive_output_dir,
    stage_file_with_hash,
    verify_drive_authenticity,
    verify_write_access,
)
import audit_scripts.colab_stage_and_verify as stage_mod
import audit_scripts.colab_train_exec as train_mod
from audit_scripts.colab_train_exec import classify_training_endpoint


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
    assert bootstrap_validate_sha(sha) == sha


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
    commit = "a497ec3f0b6dd4c760c4e40296481f61b6426d8d"
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
    commit = "a497ec3f0b6dd4c760c4e40296481f61b6426d8d"
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
            target_commit="a497ec3f0b6dd4c760c4e40296481f61b6426d8d",
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
            target_commit="a497ec3f0b6dd4c760c4e40296481f61b6426d8d",
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


# --- 8. Safe Bounded Dataset Package Inspection (Defect D) ---

def test_authorized_zip_bounded_speaker_inspection():
    zip_path = Path("datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip")
    if not zip_path.is_file():
        pytest.skip("Authorized ZIP unavailable on local environment - SKIPPED/UNVERIFIED")

    manifest_path = Path("manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json")
    assert manifest_path.is_file(), "Split manifest missing"

    with open(manifest_path) as f:
        splits = json.load(f)["folds"][0]

    train_speakers = set(splits["train"]["speakers"])
    val_speakers = set(splits["val"]["speakers"])
    approved_speakers = train_speakers | val_speakers
    test_speakers = set(splits["test"]["speakers"])

    observed_speakers = set()
    observed_audio_paths = []

    with zipfile.ZipFile(zip_path) as outer:
        outer_names = outer.namelist()
        assert len(outer_names) == 18, f"Expected 18 speaker packages, found {len(outer_names)}"

        for name in outer_names:
            spk = name.replace(".zip", "")
            assert spk in approved_speakers, f"Outer zip entry '{name}' not in approved speakers!"
            observed_speakers.add(spk)

            # Bounded inner entry path inspection without extracting to disk
            inner_bytes = outer.read(name)
            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
                for iname in inner.namelist():
                    parts = iname.strip("/").split("/")
                    if parts and parts[0]:
                        inner_spk = parts[0]
                        assert inner_spk in approved_speakers, (
                            f"Unapproved speaker directory '{inner_spk}' found inside '{name}' entry '{iname}'"
                        )
                        observed_speakers.add(inner_spk)
                        if len(parts) >= 2 and parts[1] == "wav":
                            observed_audio_paths.append(iname)

    # 1. Exact match with approved Fold 1 partition speakers
    assert observed_speakers == approved_speakers, (
        f"Observed speakers do not match approved speakers: {observed_speakers ^ approved_speakers}"
    )

    # 2. Strict isolation: Zero test speakers
    leakage = observed_speakers & test_speakers
    assert len(leakage) == 0, f"SEALED TEST speaker leakage detected: {leakage}"

    # 3. Bounded check on audio files
    assert len(observed_audio_paths) > 0, "No audio paths detected inside archive packages."


# --- 9. Deterministic Bootstrap on Fresh VM & Refusal of Dirty Repos (Defect A) ---

def test_fresh_vm_bootstrap_scenario(tmp_path):
    """Simulates fresh VM where /content/repo does not exist."""
    fake_remote = tmp_path / "remote_repo.git"
    target_repo = tmp_path / "content" / "repo"

    # Create a bare git repo with a commit
    src_repo = tmp_path / "src_repo"
    src_repo.mkdir()
    subprocess.run(["git", "init"], cwd=str(src_repo), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "audit@langfens.local"], cwd=str(src_repo), check=True)
    subprocess.run(["git", "config", "user.name", "Audit Runner"], cwd=str(src_repo), check=True)
    (src_repo / "README.md").write_text("Langfens-RQ1-Speech-Research test")
    subprocess.run(["git", "add", "."], cwd=str(src_repo), check=True)
    subprocess.run(["git", "commit", "-m", "init commit"], cwd=str(src_repo), check=True, capture_output=True)
    commit_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(src_repo), text=True).strip()

    # Clone bare to simulate remote URL with expected name
    subprocess.run(["git", "clone", "--bare", str(src_repo), str(fake_remote)], check=True, capture_output=True)
    fake_url = f"file://{fake_remote.resolve()}"
    # Ensure URL contains Langfens-RQ1-Speech-Research
    fake_url_named = f"file://{fake_remote.parent}/Langfens-RQ1-Speech-Research.git"
    shutil.move(str(fake_remote), str(fake_remote.parent / "Langfens-RQ1-Speech-Research.git"))

    # Test fresh VM clone
    assert not target_repo.exists()
    head = bootstrap_repository(target_repo, commit_sha, repo_url=fake_url_named)
    assert target_repo.exists()
    assert (target_repo / ".git").is_dir()
    assert head == commit_sha.lower()


def test_bootstrap_refuses_unrelated_or_dirty_repo(tmp_path):
    target_repo = tmp_path / "dirty_repo"
    target_repo.mkdir()

    # 1. Non-git dir must be rejected
    with pytest.raises(RuntimeError, match="is NOT a Git repository"):
        bootstrap_repository(target_repo, "a" * 40, repo_url="dummy")

    # 2. Repo with wrong origin must be rejected
    subprocess.run(["git", "init"], cwd=str(target_repo), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(target_repo), check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=str(target_repo), check=True)
    (target_repo / "file.txt").write_text("data")
    subprocess.run(["git", "add", "."], cwd=str(target_repo), check=True)
    subprocess.run(["git", "commit", "-m", "first"], cwd=str(target_repo), check=True, capture_output=True)
    subprocess.run(["git", "remote", "add", "origin", "https://github.com/unrelated/other.git"], cwd=str(target_repo), check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(target_repo), text=True).strip()

    with pytest.raises(RuntimeError, match="unexpected remote origin"):
        bootstrap_repository(target_repo, commit, repo_url="https://github.com/minkhoaa/Langfens-RQ1-Speech-Research.git")

    # 3. Dirty tree must be rejected
    subprocess.run(["git", "remote", "set-url", "origin", "https://github.com/minkhoaa/Langfens-RQ1-Speech-Research.git"], cwd=str(target_repo), check=True)
    (target_repo / "uncommitted.txt").write_text("dirty")
    with pytest.raises(RuntimeError, match="dirty working tree"):
        bootstrap_repository(target_repo, commit, repo_url="https://github.com/minkhoaa/Langfens-RQ1-Speech-Research.git")


# --- 10. Jupyter Notebook Kernel Execution Compatibility (Defect B) ---

def test_kernel_execution_compatibility_without_dunder_file():
    """
    Executes source code of staging, training, and bootstrap drivers in an isolated
    namespace that mimics a Jupyter/IPython kernel:
    - '__file__' is explicitly NOT present in globals.
    - sys.argv contains typical ipykernel launcher arguments ('-f', '/tmp/fake-kernel.json').
    """
    scripts_to_test = [
        "audit_scripts/colab_bootstrap.py",
        "audit_scripts/colab_stage_and_verify.py",
        "audit_scripts/colab_train_exec.py",
    ]

    orig_argv = list(sys.argv)
    try:
        sys.argv = ["ipykernel_launcher.py", "-f", "/root/.local/share/jupyter/runtime/kernel-fake.json"]

        for rel_path in scripts_to_test:
            path = Path(rel_path)
            assert path.is_file(), f"Script missing: {rel_path}"
            code = path.read_text()

            # Clean namespace without __file__
            kernel_globals = {
                "__name__": "kernel_test_namespace",
                "__doc__": "simulated notebook kernel",
                "__builtins__": __builtins__,
            }
            assert "__file__" not in kernel_globals

            # Compile and execute in simulated notebook kernel
            compiled = compile(code, str(path), "exec")
            exec(compiled, kernel_globals)

            # Ensure functions are exported successfully
            assert "main" in kernel_globals
            print(f"[KERNEL COMPATIBILITY PASS] {rel_path} executed in notebook namespace without __file__.")
    finally:
        sys.argv = orig_argv


# --- 11. Single Predictable Output Directory (Defect C) ---

def test_duplicate_run_id_resolution(tmp_path):
    drive_root = tmp_path / "KLTN-Dataset"

    # Case 1: Fresh run with run-id
    out_c0 = resolve_drive_output_dir(drive_root, "C0", run_id="run_20261028_01")
    assert str(out_c0) == str(drive_root / "manifests/RQ1-TRN-02B/checkpoints/C0/run_20261028_01")

    # Case 2: Resume / caller already nested run-id
    already_nested = drive_root / "manifests/RQ1-TRN-02B/checkpoints/C0/run_20261028_01"
    resolved_idempotent = resolve_drive_output_dir(already_nested, "C0", run_id="run_20261028_01")
    assert str(resolved_idempotent) == str(drive_root / "manifests/RQ1-TRN-02B/checkpoints/C0/run_20261028_01")
    # Verify no double nesting
    assert "run_20261028_01/run_20261028_01" not in str(resolved_idempotent)

    # Case 3: Fresh run without run-id
    out_c1 = resolve_drive_output_dir(drive_root, "C1", run_id=None)
    assert str(out_c1) == str(drive_root / "manifests/RQ1-TRN-02B/checkpoints/C1")


# --- 12. Authentic Drive Persistence Verification (Defect F) ---

def test_authentic_drive_persistence_verification(tmp_path):
    # Case 1: Empty arbitrary local directory called KLTN-Dataset must be rejected
    fake_drive = tmp_path / "fake_local" / "KLTN-Dataset"
    fake_drive.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="lacks authentic Baseline R weights"):
        verify_drive_authenticity(fake_drive, require_mount=False)

    # Case 2: Directory with corrupted baseline hash must be rejected
    chk_dir = fake_drive / "manifests/RQ1-TRN-02A/checkpoints"
    chk_dir.mkdir(parents=True)
    (chk_dir / "best.pt").write_bytes(b"corrupt fake weights")
    with pytest.raises(RuntimeError, match="failed cryptographic identity verification"):
        verify_drive_authenticity(fake_drive, require_mount=False)

    # Case 3: Directory with authentic baseline hash must pass
    real_baseline = Path("checkpoints/RQ1-TRN-02A/recovered/best.pt")
    if real_baseline.is_file():
        shutil.copyfile(str(real_baseline), str(chk_dir / "best.pt"))
        assert verify_drive_authenticity(fake_drive, require_mount=False) is True


# --- 13. Primary Endpoint Verification & Classification (Defect G) ---

def test_primary_endpoint_classification_states(tmp_path):
    import torch
    out_dir = tmp_path / "checkpoints_eval"
    out_dir.mkdir()

    # State 1: FAILED (No checkpoints)
    info_fail = classify_training_endpoint(out_dir, "C0")
    assert info_fail["status"] == "FAILED"

    # State 2: HALTED_GUARDRAIL (halted_guardrail.pt exists)
    guardrail_payload = {
        "metadata": {
            "arm": "C0",
            "guardrail_violation": True,
            "experiment_status": "HALTED_GUARDRAIL_VIOLATION",
            "ctc_per_measured": 25.1,
            "ctc_per_anchor": 23.64,
            "ctc_per_degradation_pct": 1.46,
        },
        "successful_updates": 1200,
        "epoch": 2,
    }
    torch.save(guardrail_payload, out_dir / "halted_guardrail.pt")
    info_guard = classify_training_endpoint(out_dir, "C0")
    assert info_guard["status"] == "HALTED_GUARDRAIL"
    assert "Guardrail violation" in info_guard["message"]
    (out_dir / "halted_guardrail.pt").unlink()

    # State 3: INCOMPLETE (last.pt exists with < 3600 updates, no guardrail)
    incomplete_payload = {
        "metadata": {
            "arm": "C0",
            "guardrail_violation": False,
            "primary_3600_updates_completed": False,
            "successful_updates": 1800,
        },
        "successful_updates": 1800,
        "epoch": 3,
    }
    torch.save(incomplete_payload, out_dir / "last.pt")
    info_incomp = classify_training_endpoint(out_dir, "C0")
    assert info_incomp["status"] == "INCOMPLETE"
    assert info_incomp["successful_updates"] == 1800

    # State 4: COMPLETED_3600_UPDATES (fixed_budget_checkpoint.pt with 3600 updates)
    completed_payload = {
        "metadata": {
            "arm": "C0",
            "guardrail_violation": False,
            "primary_3600_updates_completed": True,
            "successful_updates": 3600,
        },
        "successful_updates": 3600,
        "epoch": 5,
    }
    torch.save(completed_payload, out_dir / "fixed_budget_checkpoint.pt")
    info_comp = classify_training_endpoint(out_dir, "C0")
    assert info_comp["status"] == "COMPLETED_3600_UPDATES"
    assert info_comp["successful_updates"] == 3600


# --- 14. Fail-Closed Readiness Checks (Defect E) ---

def test_fail_closed_readiness_checks():
    raw_usage_zero = "Current balance: 0.00 compute units\nUsage rate: 0.00/hr\nActive assignments: 0"
    commit = "a497ec3f0b6dd4c760c4e40296481f61b6426d8d"
    layers = evaluate_readiness_layers(commit, raw_usage_zero)

    assert layers["2_COMPUTE_QUOTA"][0] == "BLOCKED"
    assert "ZERO_QUOTA" in layers["2_COMPUTE_QUOTA"][1]
    assert layers["5_GPU_AUTHORIZATION"][0] == "BLOCKED"

    # With funded quota
    raw_usage_funded = "Current balance: 20.00 compute units\nUsage rate: 1.00/hr\nActive assignments: 0"
    layers_funded = evaluate_readiness_layers(commit, raw_usage_funded)
    assert layers_funded["2_COMPUTE_QUOTA"][0] == "PASS"
