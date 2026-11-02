"""Private offline bootstrap. No install, download, evaluation default, or training."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import zipfile


PROFILE = "KAGGLE_FP32"
MAX_SOURCE_BYTES = 64 * 1024 * 1024


def _inside(path: Path, root: Path) -> Path:
    path, root = path.resolve(), root.resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Path must be below {root}: {path}")
    return path


def bootstrap(argv=None) -> int:
    parser = argparse.ArgumentParser(description=f"{PROFILE}: offline source bootstrap; default preflight")
    parser.add_argument("--source-archive", type=Path, default=Path("/kaggle/input/langfens-bench-source/langfens-bench-source.bin"))
    parser.add_argument("--source-sha256", required=True, help="SHA-256 from separately reviewed local package build")
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), required=True,
                        help="Active fold; attach only this fold's private artifact input")
    parser.add_argument("--source-root", type=Path, default=Path("/kaggle/working/langfens-bench-source"))
    parser.add_argument("--hf-home", type=Path, default=Path("/kaggle/input/langfens-bench-cache"))
    args, remaining = parser.parse_known_args(argv)
    # Refuse evaluation before staging/importing anything, including model code.
    if "evaluate" in remaining and "--authorize-remote-inference" not in remaining:
        parser.error("evaluate requires explicit --authorize-remote-inference")
    try:
        if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
            raise ValueError("Kaggle immutable input and working mounts are required")
        source = _inside(args.source_archive, Path("/kaggle/input"))
        cache = _inside(args.hf_home, Path("/kaggle/input"))
        destination = _inside(args.source_root, Path("/kaggle/working"))
        if not source.is_file() or not cache.is_dir():
            raise ValueError("Missing offline source ZIP or pinned Hugging Face cache mount")
        if destination.exists() or destination.is_symlink():
            raise ValueError("Source root already exists; choose a new --source-root")
        if re.fullmatch(r"[0-9a-f]{64}", args.source_sha256) is None:
            raise ValueError("--source-sha256 must be a lowercase SHA-256 digest")
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != args.source_sha256:
            raise ValueError("Offline source SHA-256 mismatch")
        with zipfile.ZipFile(source) as archive:
            infos = archive.infolist()
            names = set()
            total = 0
            for info in infos:
                path = PurePosixPath(info.filename)
                if (not path.parts or info.is_dir() or path.is_absolute() or ".." in path.parts or
                        "\\" in info.filename or path.as_posix() != info.filename or
                        info.filename in names or stat.S_ISLNK(info.external_attr >> 16)):
                    raise ValueError(f"Unsafe/duplicate source member: {info.filename}")
                allowed = ((len(path.parts) == 3 and
                            path.parts[:2] in (("src", "langfens_prep"), ("src", "langfens_bench")) and
                            path.suffix == ".py") or
                           info.filename in {"LICENSE", "resources/CMUDict/LICENSE",
                                             "audit_scripts/evaluate_fold1_ctc_binary.py",
                                             "audit_scripts/diagnose_omissions_and_false_negatives.py",
                                             "configs/bench_parity_thresholds.json",
                                             "configs/bench_dependency_contract.json",
                                             "manifests/RQ1-BENCH-01/contract.json",
                                             "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json",
                                             "resources/CMUDict/cmudict.dict",
                                             "resources/L2-ARCTIC/prompts.json"})
                if not allowed or info.flag_bits & 1:
                    raise ValueError(f"Non-source/encrypted member refused: {info.filename}")
                names.add(info.filename)
                total += info.file_size
            required = {"src/langfens_bench/__init__.py", "src/langfens_bench/runtime.py",
                        "src/langfens_bench/kaggle.py", "manifests/RQ1-BENCH-01/contract.json",
                        "audit_scripts/evaluate_fold1_ctc_binary.py",
                        "audit_scripts/diagnose_omissions_and_false_negatives.py",
                        "configs/bench_parity_thresholds.json", "configs/bench_dependency_contract.json",
                        "LICENSE", "resources/CMUDict/LICENSE",
                        "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json",
                        "resources/CMUDict/cmudict.dict", "resources/L2-ARCTIC/prompts.json"}
            if not required.issubset(names) or total > MAX_SOURCE_BYTES:
                raise ValueError("Incomplete or oversized source-only package")
            destination.mkdir(parents=True, exist_ok=False)
            # Inventory checked in full before extraction; never extract data ZIPs.
            for info in infos:
                target = destination / info.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source_handle, target.open("xb") as target_handle:
                    shutil.copyfileobj(source_handle, target_handle)
        os.environ["HF_HOME"] = str(cache)
        os.environ["HF_HUB_CACHE"] = str(cache / "hub")
        # Source package must win over notebook cwd or an unrelated installed copy.
        sys.path[:0] = [str(destination / "src"), str(destination)]
        from langfens_bench.kaggle import main
        return main(PROFILE, ["--fold", str(args.fold), *remaining])
    except (ValueError, OSError, ImportError, zipfile.BadZipFile) as exc:
        parser.error(f"BLOCKED: {exc}")


if __name__ == "__main__":
    raise SystemExit(bootstrap())
