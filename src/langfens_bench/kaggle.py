"""Offline Kaggle packaging and launch boundary shared by both precision lanes.

Scientific validation, per-fold archive inventory/staging and inference belong to
``runtime``. This module never evaluates, extracts audio, or contacts a service.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
import re
import zipfile


PROFILES = ("KAGGLE_FP32", "KAGGLE_FP16")
SOURCE_LIMIT_BYTES = 64 * 1024 * 1024
DEPENDENCIES = ("torch", "torchaudio", "transformers", "numpy", "scipy", "soundfile", "psutil", "yaml")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contained(path: Path, root: Path, description: str) -> Path:
    resolved, boundary = Path(path).resolve(), Path(root).resolve()
    if resolved == boundary or not resolved.is_relative_to(boundary):
        raise ValueError(f"{description} must be strictly below {boundary}: {resolved}")
    return resolved


def validate_mount_paths(artifact_root: Path, outdir: Path) -> tuple[Path, Path]:
    """Reject symlink escapes, mutable inputs, overwrites and non-Kaggle mounts."""
    inputs = Path("/kaggle/input")
    working = Path("/kaggle/working")
    if not inputs.is_dir() or not working.is_dir():
        raise ValueError("Kaggle /kaggle/input and /kaggle/working mounts are required")
    artifacts = _contained(artifact_root, inputs, "artifact-root")
    output = _contained(outdir, working, "outdir")
    if not artifacts.is_dir():
        raise ValueError(f"Missing immutable artifact mount: {artifacts}")
    if output.exists() or output.is_symlink():
        raise ValueError(f"Output already exists; choose a new outdir: {output}")
    return artifacts, output


def configure_offline_environment() -> None:
    """Must execute before importing torch or the common evaluator."""
    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "MKL_CBWR": "COMPATIBLE",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }.items():
        os.environ[key] = value


def validate_dependencies() -> None:
    missing = [name for name in DEPENDENCIES if importlib.util.find_spec(name) is None]
    if missing:
        raise ValueError("Missing offline dependencies (no automatic installation): " + ", ".join(missing))


def validate_kaggle_hardware(torch_module=None) -> dict:
    """Verify cuda:0 is a T4; a second allocated T4 is deliberately unused."""
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise ValueError("Verified Kaggle mounts required; no local CUDA fallback")
    torch = torch_module if torch_module is not None else importlib.import_module("torch")
    if not torch.cuda.is_available() or not torch.version.cuda or torch.cuda.device_count() < 1:
        raise ValueError("CUDA-enabled PyTorch and an allocated T4 are required")
    name = torch.cuda.get_device_name(0)
    if re.fullmatch(r"(?:(?:NVIDIA )?Tesla |NVIDIA )?T4", name.strip(), re.IGNORECASE) is None:
        raise ValueError(f"cuda:0 must be NVIDIA Tesla T4, found {name!r}")
    capability = tuple(torch.cuda.get_device_capability(0))
    if capability != (7, 5):
        raise ValueError(f"T4 compute capability must be (7, 5), found {capability}")
    return {
        "device": "cuda:0",
        "name": name,
        "capability": list(capability),
        "visible_device_count": torch.cuda.device_count(),
        "cuda_version": torch.version.cuda,
        "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
    }


def build_source_package(repo_root: Path, destination: Path) -> str:
    """Build source/resources ONLY, never datasets, weights, cache or credentials.

    Call after integrating all lanes. The returned SHA must be supplied separately
    to the bootstrap; a digest bundled with an untrusted ZIP is not a trust anchor.
    """
    root, output = Path(repo_root).resolve(), Path(destination)
    required = [
        "LICENSE",
        "resources/CMUDict/LICENSE",
        "src/langfens_bench/__init__.py",
        "src/langfens_bench/runtime.py",
        "src/langfens_bench/kaggle.py",
        "manifests/RQ1-BENCH-01/contract.json",
        "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json",
        "resources/CMUDict/cmudict.dict",
        "resources/L2-ARCTIC/prompts.json",
    ]
    paths = {Path(name) for name in required}
    for directory, pattern in (("src", "*.py"), ("audit_scripts", "*.py"), ("configs", "*.yaml")):
        paths.update(path.relative_to(root) for path in (root / directory).rglob(pattern))
    total = 0
    for relative in sorted(paths):
        path = root / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Missing or unsafe source input: {relative}")
        total += path.stat().st_size
    if total > SOURCE_LIMIT_BYTES:
        raise ValueError("Source package exceeds 64 MiB source-only limit")
    # Exclusive creation keeps measured/reviewed source packages immutable.
    with output.open("xb") as handle:
        with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for relative in sorted(paths):
                info = zipfile.ZipInfo(relative.as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, (root / relative).read_bytes())
    return sha256_file(output)


def main(profile: str, argv=None) -> int:
    """Fixed-profile wrapper: default preflight, never automatic evaluation."""
    if profile not in PROFILES:
        raise ValueError(f"Unsupported Kaggle profile: {profile}")
    parser = argparse.ArgumentParser(description=f"Offline {profile}; default is read-only preflight")
    parser.add_argument("command", choices=("preflight", "evaluate"), nargs="?", default="preflight")
    parser.add_argument("--contract", type=Path, default=Path(__file__).resolve().parents[2] / "manifests/RQ1-BENCH-01/contract.json")
    parser.add_argument("--artifact-root", type=Path, default=Path("/kaggle/input/langfens-bench-artifacts"))
    parser.add_argument("--outdir", type=Path, default=Path(f"/kaggle/working/bench-{profile.lower()}"))
    parser.add_argument("--authorize-remote-inference", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "evaluate" and not args.authorize_remote_inference:
        parser.error("evaluate requires explicit --authorize-remote-inference")
    try:
        artifacts, output = validate_mount_paths(args.artifact_root, args.outdir)
        contract = _contained(args.contract, Path(__file__).resolve().parents[2], "contract")
        if not contract.is_file():
            raise ValueError(f"Missing frozen contract: {contract}")
        configure_offline_environment()
        validate_dependencies()
        validate_kaggle_hardware()
    except (ValueError, OSError, ImportError) as exc:
        parser.error(f"BLOCKED: {exc}")
    forwarded = [args.command, "--profile", profile, "--contract", str(contract),
                 "--artifact-root", str(artifacts), "--outdir", str(output)]
    if args.authorize_remote_inference:
        forwarded.append("--authorize-remote-inference")
    # Import late so environment controls apply before torch; runtime repeats all
    # scientific/hardware/hash/cache gates even if invoked without this wrapper.
    from langfens_bench.runtime import main as runtime_main
    return runtime_main(forwarded)
