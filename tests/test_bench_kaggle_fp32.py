"""KAGGLE_FP32 boundary contracts; no model loads, remote calls or CUDA work."""
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import zipfile

import pytest

from langfens_bench import kaggle


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "deploy/bench_kaggle_fp32/kernel.py"


@pytest.fixture
def kaggle_mounts(monkeypatch):
    original = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", lambda p: True if str(p) in ("/kaggle/input", "/kaggle/working") else original(p))


def fake_torch(name="Tesla T4", available=True, capability=(7, 5)):
    return SimpleNamespace(
        version=SimpleNamespace(cuda="12.4"),
        cuda=SimpleNamespace(is_available=lambda: available, device_count=lambda: 2,
                             get_device_name=lambda index: name,
                             get_device_capability=lambda index: capability,
                             get_device_properties=lambda index: SimpleNamespace(total_memory=15_000_000_000)))


def test_fp32_private_metadata_has_no_attached_sources():
    metadata = json.loads((KERNEL.parent / "kernel-metadata.json").read_text())
    assert metadata["id"] == "minkhoaa/langfens-rq1-bench-fp32-private"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["enable_gpu"] is True and metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    assert metadata["code_file"] == "kernel.py"
    for key in ("dataset_sources", "model_sources", "kernel_sources", "competition_sources"):
        assert metadata[key] == []


@pytest.mark.parametrize("name", ["Tesla T4", "NVIDIA Tesla T4", "NVIDIA T4", "T4"])
def test_only_cuda_zero_t4_is_used(kaggle_mounts, name):
    torch = fake_torch(name=name)
    called = []
    torch.cuda.get_device_name = lambda index: called.append(index) or name
    report = kaggle.validate_kaggle_hardware(torch)
    assert report["device"] == "cuda:0" and report["visible_device_count"] == 2
    assert called == [0]


@pytest.mark.parametrize("torch", [fake_torch(name="NVIDIA A100"), fake_torch(available=False), fake_torch(capability=(8, 0))])
def test_wrong_hardware_fails_closed(kaggle_mounts, torch):
    with pytest.raises(ValueError):
        kaggle.validate_kaggle_hardware(torch)


def test_no_local_cuda_fallback(monkeypatch):
    monkeypatch.setattr(Path, "is_dir", lambda p: False)
    with pytest.raises(ValueError, match="no local CUDA fallback"):
        kaggle.validate_kaggle_hardware(fake_torch())


def test_dependency_missing_fails_without_install(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "transformers" else object())
    with pytest.raises(ValueError, match="transformers"):
        kaggle.validate_dependencies()


@pytest.fixture
def mocked_runtime(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "langfens_bench.runtime", SimpleNamespace(main=lambda argv: calls.append(argv) or 0))
    monkeypatch.setattr(kaggle, "validate_mount_paths", lambda a, o: (a, o))
    monkeypatch.setattr(kaggle, "validate_dependencies", lambda: None)
    monkeypatch.setattr(kaggle, "validate_kaggle_hardware", lambda: {})
    return calls


@pytest.mark.parametrize("profile", kaggle.PROFILES)
def test_both_wrappers_default_to_same_preflight_gate(mocked_runtime, profile, monkeypatch):
    # Do not leak deterministic/offline environment mutations to other tests.
    monkeypatch.setattr(kaggle.os, "environ", dict(kaggle.os.environ))
    assert kaggle.main(profile, ["--fold", "1"]) == 0
    argv = mocked_runtime[0]
    assert argv[:5] == ["preflight", "--profile", profile, "--fold", "1"]
    assert "--authorize-remote-inference" not in argv
    assert kaggle.os.environ["HF_HUB_OFFLINE"] == "1"
    assert kaggle.os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert kaggle.os.environ["MKL_CBWR"] == "COMPATIBLE"


@pytest.mark.parametrize("profile", kaggle.PROFILES)
def test_evaluate_requires_explicit_authorization_before_runtime(mocked_runtime, profile):
    with pytest.raises(SystemExit) as error:
        kaggle.main(profile, ["evaluate", "--fold", "1"])
    assert error.value.code == 2
    assert mocked_runtime == []


@pytest.mark.parametrize("profile", kaggle.PROFILES)
@pytest.mark.parametrize("fold", [1, 2, 3, 4])
def test_each_remote_fold_is_forwarded_to_common_runtime(mocked_runtime, monkeypatch, profile, fold):
    monkeypatch.setattr(kaggle.os, "environ", dict(kaggle.os.environ))
    assert kaggle.main(profile, ["--fold", str(fold)]) == 0
    argv = mocked_runtime[0]
    assert argv[argv.index("--fold") + 1] == str(fold)
    assert argv[0] == "preflight"


@pytest.mark.parametrize("profile", kaggle.PROFILES)
@pytest.mark.parametrize("fold_args", [[], ["--fold", "0"], ["--fold", "5"], ["--fold", "all"]])
def test_remote_fold_required_and_bounded_before_runtime(mocked_runtime, profile, fold_args):
    with pytest.raises(SystemExit) as error:
        kaggle.main(profile, fold_args)
    assert error.value.code == 2
    assert mocked_runtime == []


def test_authorized_evaluation_calls_common_runtime_only(mocked_runtime, monkeypatch):
    monkeypatch.setattr(kaggle.os, "environ", dict(kaggle.os.environ))
    assert kaggle.main("KAGGLE_FP32", ["evaluate", "--fold", "1", "--authorize-remote-inference"]) == 0
    assert mocked_runtime[0][0] == "evaluate"
    assert mocked_runtime[0][-1] == "--authorize-remote-inference"


def test_profile_cannot_be_overridden(mocked_runtime):
    with pytest.raises(SystemExit):
        kaggle.main("KAGGLE_FP32", ["--fold", "1", "--profile", "KAGGLE_FP16"])
    assert mocked_runtime == []


def test_symlink_and_outside_paths_refused(tmp_path):
    boundary = tmp_path / "input"
    boundary.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (boundary / "escape").symlink_to(outside, target_is_directory=True)
    for path in (boundary, boundary / "escape", outside):
        with pytest.raises(ValueError):
            kaggle._contained(path, boundary, "input")


def test_existing_output_refused(kaggle_mounts, tmp_path, monkeypatch):
    artifacts, output = tmp_path / "artifacts", tmp_path / "output"
    artifacts.mkdir()
    output.mkdir()
    monkeypatch.setattr(kaggle, "_contained", lambda path, root, description: path)
    with pytest.raises(ValueError, match="already exists"):
        kaggle.validate_mount_paths(artifacts, output)


def minimal_source(repo):
    files = {
        "LICENSE": "Project license",
        "resources/CMUDict/LICENSE": "CMUdict license",
        "src/langfens_bench/__init__.py": "",
        "src/langfens_bench/runtime.py": "def main(argv=None): return 0\n",
        "src/langfens_bench/kaggle.py": "",
        "manifests/RQ1-BENCH-01/contract.json": "{}",
        "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json": "{}",
        "resources/CMUDict/cmudict.dict": "WORD W ER D\n",
        "resources/L2-ARCTIC/prompts.json": "{}",
        "datasets/sealed-test.wav": "NEVER PACKAGE",
        "checkpoints/weights.pt": "NEVER PACKAGE",
        ".env": "NEVER PACKAGE",
    }
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def test_source_package_deterministic_source_only_and_no_overwrite(tmp_path):
    repo = tmp_path / "repo"
    minimal_source(repo)
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    digest = kaggle.build_source_package(repo, first)
    assert digest == kaggle.build_source_package(repo, second)
    with zipfile.ZipFile(first) as archive:
        assert "src/langfens_bench/runtime.py" in archive.namelist()
        assert {"LICENSE", "resources/CMUDict/LICENSE"}.issubset(archive.namelist())
        assert not any(name.startswith(("datasets/", "checkpoints/", ".env")) for name in archive.namelist())
    with pytest.raises(FileExistsError):
        kaggle.build_source_package(repo, first)


def test_source_package_missing_runtime_and_symlinks_refused(tmp_path):
    minimal_source(tmp_path / "repo")
    runtime = tmp_path / "repo/src/langfens_bench/runtime.py"
    runtime.unlink()
    with pytest.raises(ValueError, match="Missing or unsafe"):
        kaggle.build_source_package(tmp_path / "repo", tmp_path / "bad.zip")
    runtime.symlink_to(KERNEL)
    with pytest.raises(ValueError, match="Missing or unsafe"):
        kaggle.build_source_package(tmp_path / "repo", tmp_path / "bad.zip")
    assert not (tmp_path / "bad.zip").exists()


@pytest.fixture
def bootstrap_setup(tmp_path, kaggle_mounts, monkeypatch):
    minimal_source(tmp_path / "repo")
    source = tmp_path / "source.zip"
    digest = kaggle.build_source_package(tmp_path / "repo", source)
    cache = tmp_path / "cache"
    cache.mkdir()
    output = tmp_path / "extracted"
    namespace = runpy.run_path(str(KERNEL))
    namespace["bootstrap"].__globals__["_inside"] = lambda path, root: path
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(kaggle.os, "environ", dict(kaggle.os.environ))
    calls = []
    monkeypatch.setattr(kaggle, "main", lambda profile, argv: calls.append((profile, argv)) or 0)
    args = ["--fold", "1", "--source-archive", str(source), "--source-sha256", digest,
            "--source-root", str(output), "--hf-home", str(cache)]
    return namespace["bootstrap"], args, source, output, calls


def test_bootstrap_calls_fixed_profile_and_leaves_default_preflight(bootstrap_setup):
    bootstrap, args, source, output, calls = bootstrap_setup
    assert bootstrap(args) == 0
    assert calls == [("KAGGLE_FP32", ["--fold", "1"])]
    assert (output / "src/langfens_bench/runtime.py").is_file()


def test_bootstrap_hash_mismatch_before_extraction(bootstrap_setup):
    bootstrap, args, source, output, calls = bootstrap_setup
    source.write_bytes(b"corrupt")
    with pytest.raises(SystemExit):
        bootstrap(args)
    assert not output.exists() and calls == []


@pytest.mark.parametrize("member", ["../escape.py", "/absolute.py", "src/link.py", "datasets/TEST.wav", "src/duplicate.py"])
def test_bootstrap_rejects_unsafe_inventory_before_any_extraction(bootstrap_setup, member):
    bootstrap, args, source, output, calls = bootstrap_setup
    with zipfile.ZipFile(source, "a") as archive:
        info = zipfile.ZipInfo(member)
        if member == "src/link.py":
            info.external_attr = 0o120777 << 16
        archive.writestr(info, b"unsafe")
        if member == "src/duplicate.py":
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr(info, b"unsafe")
    args[args.index("--source-sha256") + 1] = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(SystemExit):
        bootstrap(args)
    assert not output.exists() and calls == []


@pytest.mark.parametrize("fold_args", [[], ["--fold", "0"], ["--fold", "5"]])
def test_bootstrap_requires_single_valid_fold_before_staging(bootstrap_setup, fold_args):
    bootstrap, args, source, output, calls = bootstrap_setup
    with pytest.raises(SystemExit) as error:
        bootstrap([*fold_args, *args[2:]])
    assert error.value.code == 2
    assert not output.exists() and calls == []


def test_bootstrap_authorization_gate_precedes_staging(bootstrap_setup):
    bootstrap, args, source, output, calls = bootstrap_setup
    with pytest.raises(SystemExit):
        bootstrap(args + ["evaluate"])
    assert not output.exists() and calls == []
