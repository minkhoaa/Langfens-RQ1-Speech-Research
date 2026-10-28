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
    is_executable_ready,
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
    validate_drive_output_hierarchy,
    verify_drive_authenticity,
    verify_google_drive_api_identity,
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

    # Case 3: Directory with authentic baseline hash in pre-runtime environment
    # A mountpoint and matching baseline hash alone must NOT be described as proof of root folder identity.
    # Must return PENDING_RUNTIME_VERIFICATION, not PASS.
    real_baseline = Path("checkpoints/RQ1-TRN-02A/recovered/best.pt")
    if real_baseline.is_file():
        shutil.copyfile(str(real_baseline), str(chk_dir / "best.pt"))
        res = verify_drive_authenticity(fake_drive, require_mount=False)
        assert res == "PENDING_RUNTIME_VERIFICATION"


def test_drive_output_hierarchy_and_containment(tmp_path):
    drive_root = tmp_path / "drive" / "KLTN-Dataset"
    drive_root.mkdir(parents=True)

    # Valid output path
    valid_out = drive_root / "manifests/RQ1-TRN-02B/checkpoints/C0/run_01"
    assert validate_drive_output_hierarchy(valid_out, drive_root, "C0") is True

    # Arm mismatch in hierarchy
    with pytest.raises(ValueError, match="does not belong to authorized hierarchy"):
        validate_drive_output_hierarchy(valid_out, drive_root, "C1")

    # Outside Drive root
    outside_out = tmp_path / "outside_dir" / "manifests/RQ1-TRN-02B/checkpoints/C0"
    with pytest.raises(ValueError, match="does not belong to authorized Drive root"):
        validate_drive_output_hierarchy(outside_out, drive_root, "C0")


# --- 13. Primary Endpoint Verification & Classification (Defect G) ---

def _make_checkpoint_payload(
    arm: str = "C0",
    successful_updates: int = 3600,
    primary_3600_updates_completed: bool = True,
    primary_endpoint_reached: bool = True,
    endpoint_status: str = "COMPLETED_3600_UPDATES",
    guardrail_violation: bool = False,
    git_sha: str = "ea8762079735cfb5520de49eaca4528a458a1f38",
    split_sha256: str = "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b",
    initial_checkpoint_sha256: str = "a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b",
    include_state_dicts: bool = True,
) -> dict:
    import torch
    ck = {
        "successful_updates": successful_updates,
        "epoch": 5,
        "metadata": {
            "arm": arm,
            "successful_updates": successful_updates,
            "primary_3600_updates_completed": primary_3600_updates_completed,
            "primary_endpoint_reached": primary_endpoint_reached,
            "endpoint_status": endpoint_status,
            "guardrail_violation": guardrail_violation,
            "git_sha": git_sha,
            "split_sha256": split_sha256,
            "initial_checkpoint_sha256": initial_checkpoint_sha256,
            "schema_version": "rq1_trn_02b_v1",
        },
    }
    if include_state_dicts:
        ck["model"] = {"encoder.weight": torch.tensor([1.0])}
        ck["optimizer"] = {"param_groups": [{"lr": 1e-4}]}
        ck["scheduler"] = {"last_epoch": 5}
        ck["scaler"] = {"scale": 1.0}
    return ck


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

    # State 4: COMPLETED_3600_UPDATES (fixed_budget_checkpoint.pt with all 9 invariants)
    completed_payload = _make_checkpoint_payload(arm="C0", successful_updates=3600)
    torch.save(completed_payload, out_dir / "fixed_budget_checkpoint.pt")
    info_comp = classify_training_endpoint(out_dir, "C0")
    assert info_comp["status"] == "COMPLETED_3600_UPDATES"
    assert info_comp["successful_updates"] == 3600


# --- Task A Negative Endpoint Regressions ---

def test_task_a_negative_endpoint_regressions(tmp_path):
    import torch
    out_dir = tmp_path / "neg_endpoints"
    out_dir.mkdir()
    chk_file = out_dir / "fixed_budget_checkpoint.pt"

    # 1. 1,800 updates with completion flag True
    ck1 = _make_checkpoint_payload(
        successful_updates=1800,
        primary_3600_updates_completed=True,
    )
    torch.save(ck1, chk_file)
    res1 = classify_training_endpoint(out_dir, "C0")
    assert res1["status"] == "FAILED"
    assert "Contradictory metadata" in res1["message"]

    # 2. 3,600 updates with completion flag False
    ck2 = _make_checkpoint_payload(
        successful_updates=3600,
        primary_3600_updates_completed=False,
    )
    torch.save(ck2, chk_file)
    res2 = classify_training_endpoint(out_dir, "C0")
    assert res2["status"] == "FAILED"
    assert "Contradictory metadata" in res2["message"]

    # 3. 3,600 updates with missing endpoint status
    ck3 = _make_checkpoint_payload(successful_updates=3600)
    del ck3["metadata"]["endpoint_status"]
    torch.save(ck3, chk_file)
    res3 = classify_training_endpoint(out_dir, "C0")
    assert res3["status"] == "FAILED"
    assert "endpoint_status" in res3["message"]

    # 4a. Wrong arm
    ck4a = _make_checkpoint_payload(arm="C1")
    torch.save(ck4a, chk_file)
    res4a = classify_training_endpoint(out_dir, "C0")
    assert res4a["status"] == "FAILED"
    assert "Arm mismatch" in res4a["message"]

    # 4b. Incompatible provenance (corrupted split SHA)
    ck4b = _make_checkpoint_payload(split_sha256="corrupt_hash" + "0" * 28)
    torch.save(ck4b, chk_file)
    res4b = classify_training_endpoint(out_dir, "C0")
    assert res4b["status"] == "FAILED"
    assert "Provenance mismatch" in res4b["message"]

    # 5. Guardrail violation despite completion flags
    ck5 = _make_checkpoint_payload(
        successful_updates=3600,
        primary_3600_updates_completed=True,
        guardrail_violation=True,
    )
    torch.save(ck5, chk_file)
    res5 = classify_training_endpoint(out_dir, "C0")
    assert res5["status"] == "HALTED_GUARDRAIL"
    assert "guardrail violation" in res5["message"]

    # 6a. Incomplete checkpoint payload (missing model state dict)
    ck6a = _make_checkpoint_payload(include_state_dicts=False)
    torch.save(ck6a, chk_file)
    res6a = classify_training_endpoint(out_dir, "C0")
    assert res6a["status"] == "FAILED"
    assert "Corrupt or incomplete checkpoint payload" in res6a["message"]

    # 6b. Corrupt checkpoint file (truncated bytes)
    chk_file.write_bytes(b"PK\x03\x04not_a_valid_torch_checkpoint")
    res6b = classify_training_endpoint(out_dir, "C0")
    assert res6b["status"] == "FAILED"
    assert "Failed to load or parse" in res6b["message"]


# --- 14. Fail-Closed Readiness Checks (Task B) ---

def test_fail_closed_readiness_checks():
    raw_usage_zero = "Current balance: 0.00 compute units\nUsage rate: 0.00/hr\nActive assignments: 0"
    commit = "ea8762079735cfb5520de49eaca4528a458a1f38"
    layers = evaluate_readiness_layers(commit, raw_usage_zero)

    assert layers["2_COMPUTE_QUOTA"][0] == "BLOCKED"
    assert "ZERO_QUOTA" in layers["2_COMPUTE_QUOTA"][1]
    assert layers["5_OPERATOR_AUTHORIZATION"][0] != "PASS"

    # With funded quota
    raw_usage_funded = "Current balance: 20.00 compute units\nUsage rate: 1.00/hr\nActive assignments: 0"
    layers_funded = evaluate_readiness_layers(commit, raw_usage_funded)
    assert layers_funded["2_COMPUTE_QUOTA"][0] == "PASS"


# --- Task B Strict Executable Readiness Tests ---

def _setup_clean_test_repo(tmp_path: Path) -> Tuple[Path, str]:
    repo_dir = tmp_path / "clean_repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "audit@langfens.local"], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "config", "user.name", "Auditor"], cwd=str(repo_dir), check=True)

    # Configs
    (repo_dir / "configs").mkdir(parents=True, exist_ok=True)
    (repo_dir / "configs" / "rq1_trn_02b_c0.yaml").write_text("experiment: c0\n")
    (repo_dir / "configs" / "rq1_trn_02b_c1.yaml").write_text("experiment: c1\n")

    # Real manifest
    manifest_dir = repo_dir / "manifests" / "RQ1-SPLIT-01"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    real_manifest = Path("manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json")
    shutil.copyfile(str(real_manifest), str(manifest_dir / "l2arctic_4fold_splits.json"))

    subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo_dir), check=True, capture_output=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), text=True).strip()
    return repo_dir, head


def test_task_b_strict_executable_readiness(tmp_path):
    repo_dir, commit = _setup_clean_test_repo(tmp_path)
    raw_funded = "Current balance: 20.00 compute units\nUsage rate: 1.00/hr\nActive assignments: 0"

    # 1. Zero quota -> NOT READY
    raw_zero = "Current balance: 0.00 compute units\nUsage rate: 0.00/hr\nActive assignments: 0"
    layers_zero = evaluate_readiness_layers(commit, raw_usage=raw_zero, repo_root=repo_dir)
    ready_zero, reason_zero = is_executable_ready(layers_zero)
    assert not ready_zero
    assert layers_zero["1_STATIC_SOFTWARE"][0] == "PASS_SOFTWARE"
    assert layers_zero["2_COMPUTE_QUOTA"][0] == "BLOCKED"
    assert "2_COMPUTE_QUOTA" in reason_zero

    # 2. Sufficient quota but missing Drive mount -> NOT READY
    layers_no_drive = evaluate_readiness_layers(commit, raw_usage=raw_funded, drive_mounted=None, repo_root=repo_dir)
    ready_no_drive, reason_no_drive = is_executable_ready(layers_no_drive)
    assert not ready_no_drive
    assert layers_no_drive["2_COMPUTE_QUOTA"][0] == "PASS"
    assert layers_no_drive["3_DRIVE_MOUNT"][0] != "PASS"
    assert "3_DRIVE_MOUNT" in reason_no_drive

    # 3. Sufficient quota & Drive mount, but unverified staging -> NOT READY
    layers_no_staging = evaluate_readiness_layers(
        commit, raw_usage=raw_funded, drive_mounted=True, staging_verified=None, repo_root=repo_dir
    )
    ready_no_staging, reason_no_staging = is_executable_ready(layers_no_staging)
    assert not ready_no_staging
    assert layers_no_staging["3_DRIVE_MOUNT"][0] == "PASS"
    assert layers_no_staging["4_VM_STAGING_WRITE"][0] != "PASS"
    assert "4_VM_STAGING_WRITE" in reason_no_staging

    # 4. Sufficient quota, Drive mount, staging verified, but missing operator authorization -> NOT READY
    layers_no_auth = evaluate_readiness_layers(
        commit, raw_usage=raw_funded, drive_mounted=True, staging_verified=True, operator_authorized=None, repo_root=repo_dir
    )
    ready_no_auth, reason_no_auth = is_executable_ready(layers_no_auth)
    assert not ready_no_auth
    assert layers_no_auth["4_VM_STAGING_WRITE"][0] == "PASS"
    assert layers_no_auth["5_OPERATOR_AUTHORIZATION"][0] != "PASS"
    assert "5_OPERATOR_AUTHORIZATION" in reason_no_auth

    # All 5 layers satisfied -> READY
    layers_all = evaluate_readiness_layers(
        commit, raw_usage=raw_funded, drive_mounted=True, staging_verified=True, operator_authorized=True, repo_root=repo_dir
    )
    ready_all, reason_all = is_executable_ready(layers_all)
    assert ready_all is True
    assert "READY" in reason_all


# --- Task C Immutable Source Validation Tests ---

def test_task_c_immutable_source_validation(tmp_path):
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "audit@langfens.local"], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "config", "user.name", "Auditor"], cwd=str(repo_dir), check=True)

    # Configs
    (repo_dir / "configs").mkdir()
    (repo_dir / "configs" / "rq1_trn_02b_c0.yaml").write_text("experiment: c0\n")
    (repo_dir / "configs" / "rq1_trn_02b_c1.yaml").write_text("experiment: c1\n")

    # Real manifest
    manifest_dir = repo_dir / "manifests" / "RQ1-SPLIT-01"
    manifest_dir.mkdir(parents=True)
    real_manifest = Path("manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json")
    shutil.copyfile(str(real_manifest), str(manifest_dir / "l2arctic_4fold_splits.json"))

    subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo_dir), check=True, capture_output=True)
    real_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), text=True).strip()

    # 1. Matching commit SHA on clean repo -> PASS_SOFTWARE
    pass_ok, pass_msg = validate_static_software(real_head, repo_root=repo_dir)
    assert pass_ok is True
    assert "PASS_SOFTWARE" in pass_msg

    # 2. Wrong commit SHA (even if clean worktree) -> FAIL
    wrong_sha = "0" * 40
    wrong_ok, wrong_msg = validate_static_software(wrong_sha, repo_root=repo_dir)
    assert wrong_ok is False
    assert "Git HEAD mismatch" in wrong_msg

    # 3. Dirty working tree -> FAIL
    (repo_dir / "dirty_file.txt").write_text("uncommitted")
    dirty_ok, dirty_msg = validate_static_software(real_head, repo_root=repo_dir)
    assert dirty_ok is False
    assert "not clean" in dirty_msg
    (repo_dir / "dirty_file.txt").unlink()

    # 4. Corrupted manifest SHA-256 -> FAIL
    (manifest_dir / "l2arctic_4fold_splits.json").write_text('{"corrupted": true}')
    subprocess.run(["git", "commit", "-am", "corrupt manifest"], cwd=str(repo_dir), check=True, capture_output=True)
    new_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), text=True).strip()
    corrupt_ok, corrupt_msg = validate_static_software(new_head, repo_root=repo_dir)
    assert corrupt_ok is False
    assert "Split manifest SHA-256 mismatch" in corrupt_msg


def test_cli_invocation_exit_codes(tmp_path):
    repo_dir, commit = _setup_clean_test_repo(tmp_path)
    script_path = str(Path("audit_scripts/colab_launch_workflow.py").resolve())

    # 1. check-software-only exits 0 on clean valid repo
    res_soft = subprocess.run(
        [sys.executable, script_path, "--commit", commit, "--check-software-only"],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
    )
    assert res_soft.returncode == 0
    assert "PASS_SOFTWARE" in res_soft.stdout

    # 2. check-software-only exits 1 on wrong commit
    res_wrong_commit = subprocess.run(
        [sys.executable, script_path, "--commit", "0" * 40, "--check-software-only"],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
    )
    assert res_wrong_commit.returncode == 1
    assert "FAIL" in res_wrong_commit.stdout

    # 3. check-only with zero quota exits 1
    res_check_zero = subprocess.run(
        [
            sys.executable, script_path,
            "--commit", commit, "--check-only",
            "--mock-usage", "Current balance: 0.00 compute units\nUsage rate: 0.00/hr\nActive assignments: 0",
        ],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
    )
    assert res_check_zero.returncode == 1
    assert "NOT READY FOR EXECUTION" in res_check_zero.stdout

    # 4. check-only with funded quota but missing drive mount exits 1 (NOT READY)
    res_check_funded = subprocess.run(
        [
            sys.executable, script_path,
            "--commit", commit, "--check-only",
            "--mock-usage", "Current balance: 25.00 compute units\nUsage rate: 1.00/hr\nActive assignments: 0",
        ],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
    )
    assert res_check_funded.returncode == 1
    assert "NOT READY FOR EXECUTION" in res_check_funded.stdout
    assert "3_DRIVE_MOUNT" in res_check_funded.stdout

    # 5. check-only with funded quota and drive mount, but unverified staging exits 1
    res_check_staging = subprocess.run(
        [
            sys.executable, script_path,
            "--commit", commit, "--check-only",
            "--mock-usage", "Current balance: 25.00 compute units\nUsage rate: 1.00/hr\nActive assignments: 0",
            "--drive-mounted",
        ],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
    )
    assert res_check_staging.returncode == 1
    assert "4_VM_STAGING_WRITE" in res_check_staging.stdout

    # 6. check-only with funded quota, drive mount, staging verified, but missing auth exits 1
    res_check_auth = subprocess.run(
        [
            sys.executable, script_path,
            "--commit", commit, "--check-only",
            "--mock-usage", "Current balance: 25.00 compute units\nUsage rate: 1.00/hr\nActive assignments: 0",
            "--drive-mounted", "--staging-verified",
        ],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
    )
    assert res_check_auth.returncode == 1
    assert "5_OPERATOR_AUTHORIZATION" in res_check_auth.stdout

    # 7. check-only with all 5 layers satisfied exits 0
    res_check_all = subprocess.run(
        [
            sys.executable, script_path,
            "--commit", commit, "--check-only",
            "--mock-usage", "Current balance: 25.00 compute units\nUsage rate: 1.00/hr\nActive assignments: 0",
            "--drive-mounted", "--staging-verified", "--operator-authorized",
        ],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
    )
    assert res_check_all.returncode == 0
    assert "FULLY READY FOR EXECUTION" in res_check_all.stdout
