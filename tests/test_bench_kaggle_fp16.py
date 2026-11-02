"""Launcher-only tests: fake runtime delegation is not CUDA execution evidence."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "deploy/bench_kaggle_fp16"


@pytest.fixture
def launcher():
    spec = importlib.util.spec_from_file_location("bench_kaggle_fp16", PACKAGE / "bench_kaggle_fp16.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def staged(tmp_path, monkeypatch, launcher):
    input_root = tmp_path / "input"
    working_root = tmp_path / "working"
    source = input_root / "source/repo"
    artifacts = input_root / "artifacts"
    cache = input_root / "cache/huggingface"
    for directory in (source, artifacts, cache, working_root):
        directory.mkdir(parents=True)
    contract = source / "manifests/RQ1-BENCH-01/contract.json"
    contract.parent.mkdir(parents=True)
    contract.write_text("{}")
    runtime_file = source / "src/langfens_bench/runtime.py"
    runtime_file.parent.mkdir(parents=True)
    runtime_file.write_text("# fake runtime; never evaluates\n")
    runtime = SimpleNamespace(__file__=str(runtime_file), main=Mock(return_value=0))
    importer = Mock(return_value=runtime)
    monkeypatch.setattr(launcher, "INPUT_ROOT", input_root)
    monkeypatch.setattr(launcher, "WORKING_ROOT", working_root)
    monkeypatch.setattr(launcher, "importlib", SimpleNamespace(import_module=importer))
    monkeypatch.setattr(launcher.sys, "path", list(launcher.sys.path))
    for key in (
        "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HOME",
        "HF_HUB_CACHE", "MKL_CBWR", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    ):
        monkeypatch.setenv(key, "incorrect-existing-value")
    outdir = working_root / "fresh-run"
    args = [
        "--source-root", str(source), "--artifact-root", str(artifacts),
        "--hf-cache", str(cache), "--outdir", str(outdir),
    ]
    return SimpleNamespace(
        args=args, source=source, artifacts=artifacts, cache=cache, contract=contract,
        runtime_file=runtime_file, runtime=runtime, importer=importer, outdir=outdir,
        input_root=input_root, working_root=working_root,
    )


def test_default_preflight_delegates_fixed_fp16_without_reading_data(launcher, staged):
    before = set(staged.input_root.rglob("*"))
    assert launcher.main(staged.args) == 0
    staged.importer.assert_called_once_with("langfens_bench.runtime")
    staged.runtime.main.assert_called_once_with([
        "preflight", "--profile", "KAGGLE_FP16", "--contract", str(staged.contract),
        "--artifact-root", str(staged.artifacts), "--outdir", str(staged.outdir),
    ])
    assert not staged.outdir.exists()
    assert set(staged.input_root.rglob("*")) == before
    assert staged.contract.read_text() == "{}"


def test_evaluate_requires_explicit_authorization_before_import(launcher, staged, capsys):
    with pytest.raises(SystemExit) as exc:
        launcher.main(["evaluate", *staged.args])
    assert exc.value.code == 2
    assert "requires --authorize-remote-inference" in capsys.readouterr().err
    staged.importer.assert_not_called()


def test_authorized_evaluate_delegates_flag_and_fixed_profile(launcher, staged):
    launcher.main(["evaluate", *staged.args, "--authorize-remote-inference"])
    argv = staged.runtime.main.call_args.args[0]
    assert argv[0] == "evaluate"
    assert argv[argv.index("--profile") + 1] == "KAGGLE_FP16"
    assert argv[-1] == "--authorize-remote-inference"


def test_authorization_alone_never_selects_evaluation(launcher, staged):
    launcher.main([*staged.args, "--authorize-remote-inference"])
    assert staged.runtime.main.call_args.args[0][0] == "preflight"


@pytest.mark.parametrize("blocker", ["CUDA unavailable", "verified Kaggle T4 required", "offline cache incomplete"])
def test_mocked_common_runtime_blockers_propagate_without_fallback(launcher, staged, blocker):
    staged.runtime.main.side_effect = RuntimeError(blocker)
    with pytest.raises(RuntimeError, match=blocker):
        launcher.main(["evaluate", *staged.args, "--authorize-remote-inference"])
    assert staged.runtime.main.call_count == 1
    assert not staged.outdir.exists()


def test_offline_and_precision_environment_precede_runtime_import(launcher, staged):
    def check_import(name):
        assert name == "langfens_bench.runtime"
        env = launcher.os.environ
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            assert env[key] == "1"
        assert env["HF_HOME"] == str(staged.cache)
        assert env["HF_HUB_CACHE"] == str(staged.cache / "hub")
        assert env["MKL_CBWR"] == "COMPATIBLE"
        assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert launcher.sys.path[:2] == [str(staged.source / "src"), str(staged.source)]
        return staged.runtime
    staged.importer.side_effect = check_import
    launcher.main(staged.args)


@pytest.mark.parametrize("override", [["--profile", "LOCAL_FP32"], ["--device", "cpu"], ["--precision", "fp32"], ["--allow-fallback"]])
def test_profile_precision_and_fallback_overrides_rejected(launcher, staged, override):
    with pytest.raises(SystemExit):
        launcher.main([*staged.args, *override])
    staged.importer.assert_not_called()


@pytest.mark.parametrize("option", ["--source-root", "--artifact-root", "--hf-cache", "--outdir"])
def test_paths_outside_kaggle_roots_rejected(launcher, staged, tmp_path, option):
    with pytest.raises(SystemExit):
        launcher.main([*staged.args, option, str(tmp_path / "escape")])
    staged.importer.assert_not_called()


def test_symlink_output_escape_rejected(launcher, staged, tmp_path):
    link = staged.working_root / "escape"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(SystemExit):
        launcher.main([*staged.args, "--outdir", str(link / "new")])
    staged.importer.assert_not_called()


def test_input_symlink_escape_rejected(launcher, staged, tmp_path):
    link = staged.input_root / "escape"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(SystemExit):
        launcher.main([*staged.args, "--source-root", str(link)])
    staged.importer.assert_not_called()


@pytest.mark.parametrize("member", ["contract", "runtime_file", "cache"])
def test_missing_staged_inputs_fail_before_import(launcher, staged, member):
    path = getattr(staged, member)
    path.rmdir() if path.is_dir() else path.unlink()
    with pytest.raises(SystemExit):
        launcher.main(staged.args)
    staged.importer.assert_not_called()


def test_existing_output_never_overwritten(launcher, staged):
    staged.outdir.mkdir()
    marker = staged.outdir / "marker"
    marker.write_text("preserve")
    with pytest.raises(SystemExit):
        launcher.main(staged.args)
    assert marker.read_text() == "preserve"
    staged.importer.assert_not_called()


def test_dangling_output_symlink_is_not_a_new_output_path(launcher, staged):
    staged.outdir.symlink_to(staged.working_root / "missing-target", target_is_directory=True)
    with pytest.raises(SystemExit):
        launcher.main(staged.args)
    staged.importer.assert_not_called()


def test_runtime_must_come_from_prestaged_source(launcher, staged):
    staged.runtime.__file__ = __file__
    with pytest.raises(SystemExit):
        launcher.main(staged.args)
    staged.runtime.main.assert_not_called()


def test_common_runtime_nonzero_status_is_not_reported_as_success(launcher, staged):
    staged.runtime.main.return_value = 7
    assert launcher.main(staged.args) == 7


def test_private_t4_offline_metadata_has_no_public_sources():
    metadata = json.loads((PACKAGE / "kernel-metadata.json").read_text())
    assert metadata["id"] == "minkhoaa/langfens-rq1-bench-fp16-private"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["enable_gpu"] is True
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    assert metadata["kernel_type"] == "script"
    assert (PACKAGE / metadata["code_file"]).is_file()
    for key in ("dataset_sources", "competition_sources", "kernel_sources", "model_sources"):
        assert metadata[key] == []  # Intentionally blocked until private staging is approved.
