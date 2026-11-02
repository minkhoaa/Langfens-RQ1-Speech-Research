"""Offline Kaggle launcher; all evaluation and hardware checks belong to the runtime."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys


INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
PROFILE = "KAGGLE_FP16"


def _contained_path(value: str, root: Path, parser: argparse.ArgumentParser) -> Path:
    path = Path(value).resolve()
    if path == root.resolve() or not path.is_relative_to(root.resolve()):
        parser.error(f"path must be strictly under {root}: {value}")
    return path


def main(argv: list[str] | None = None) -> int | None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("preflight", "evaluate"), default="preflight")
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--source-root", default="/kaggle/input/langfens-rq1-bench-source/repo")
    parser.add_argument("--artifact-root")
    parser.add_argument("--hf-cache", default="/kaggle/input/langfens-rq1-bench-cache/huggingface")
    parser.add_argument("--outdir")
    parser.add_argument("--authorize-remote-inference", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "evaluate" and not args.authorize_remote_inference:
        parser.error("evaluate requires --authorize-remote-inference; no inference started")

    source = _contained_path(args.source_root, INPUT_ROOT, parser)
    artifacts = _contained_path(
        args.artifact_root or f"/kaggle/input/langfens-rq1-bench-fold{args.fold}-artifacts",
        INPUT_ROOT, parser,
    )
    cache = _contained_path(args.hf_cache, INPUT_ROOT, parser)
    output_value = args.outdir or f"/kaggle/working/rq1-bench-fp16-fold{args.fold}-{args.mode}"
    if Path(output_value).is_symlink():
        parser.error(f"output path is an existing symlink; choose a new --outdir: {output_value}")
    outdir = _contained_path(output_value, WORKING_ROOT, parser)
    for path in (source, artifacts, cache):
        if not path.is_dir():
            parser.error(f"missing pre-staged private input directory: {path}")
    contract = source / "manifests/RQ1-BENCH-01/contract.json"
    runtime_file = source / "src/langfens_bench/runtime.py"
    for path in (contract, runtime_file):
        if not path.is_file() or not path.resolve().is_relative_to(source):
            parser.error(f"missing or escaped pre-staged source file: {path}")
    if outdir.exists() or outdir.is_symlink():
        parser.error(f"output path already exists; choose a new --outdir: {outdir}")

    # Establish deterministic/offline settings before importing the common runtime or torch.
    os.environ.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": str(cache),
        "HF_HUB_CACHE": str(cache / "hub"),
        "MKL_CBWR": "COMPATIBLE",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    sys.path[:0] = [str(source / "src"), str(source)]
    runtime = importlib.import_module("langfens_bench.runtime")
    if Path(runtime.__file__).resolve() != runtime_file.resolve():
        parser.error("common runtime was not imported from the pre-staged source root")
    runtime_argv = [
        args.mode, "--profile", PROFILE, "--fold", str(args.fold), "--contract", str(contract),
        "--artifact-root", str(artifacts), "--outdir", str(outdir),
    ]
    if args.authorize_remote_inference:
        runtime_argv.append("--authorize-remote-inference")
    return runtime.main(runtime_argv)


if __name__ == "__main__":
    raise SystemExit(main())
