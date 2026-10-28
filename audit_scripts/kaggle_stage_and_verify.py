#!/usr/bin/env python3
"""
LANGFENS — Kaggle T4x2 Staging, Input Verification, and Containment Driver
Rigorously verifies Kaggle execution environment, immutable input integrity,
Fold 1 train/val-only dataset isolation, sealed test exclusion, auto-extracted ZIP handling,
per-arm output isolation, and persistent checkpoint mirror readiness.

Scientific & Operational Invariants:
1. Nvidia Tesla T4 GPU required (FP16 AMP precision).
2. Immutable SHA-256 checks for Baseline R best.pt, split manifest, and Fold 1 archive.
3. Strict Fold 1 Train+Val isolation (16 train + 2 val = 18 speakers). Sealed test strictly forbidden.
4. Kaggle auto-extracted ZIPs handled explicitly: cryptographic member ledger verification
   and bit-identical reconstruction if needed.
5. Per-arm output isolation under /kaggle/working/checkpoints/RQ1-TRN-02B/{C0,C1}/<RUN_ID>.
6. Fail-closed persistence: Ephemeral /kaggle/working alone rejected as durable storage.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from audit_scripts.kaggle_checkpoint_sync import KaggleRemoteMirror
except ImportError:
    from kaggle_checkpoint_sync import KaggleRemoteMirror

EXPECTED_BASELINE_SHA256 = "a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b"
EXPECTED_ZIP_SHA256 = "b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9"
EXPECTED_MANIFEST_SHA256 = "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b"

APPROVED_FOLD1_TRAIN_SPEAKERS = {
    "BWC", "ERMS", "HKK", "HQTV", "MBMPS", "NCC", "SKA", "SVBI",
    "THV", "TLV", "TNI", "TXHC", "YBAA", "YDCK", "YKWK", "ZHAA",
}
APPROVED_FOLD1_VAL_SPEAKERS = {"NJS", "RRBI"}
APPROVED_FOLD1_SPEAKERS = APPROVED_FOLD1_TRAIN_SPEAKERS | APPROVED_FOLD1_VAL_SPEAKERS
SEALED_TEST_SPEAKERS = {"ABA", "ASI", "EBVS", "HJK", "LXC", "PNV"}

# Cryptographic member ledger for the 18 speaker packages in the authorized fold 1 archive
AUTHORIZED_SPEAKER_LEDGER: Dict[str, Dict[str, Any]] = {
    "BWC.zip": {"size": 48454909, "crc32": 2592346794, "date_time": (2026, 10, 25, 20, 48, 26)},
    "ERMS.zip": {"size": 46559440, "crc32": 2004756136, "date_time": (2026, 10, 25, 20, 48, 28)},
    "HKK.zip": {"size": 41345077, "crc32": 815718313, "date_time": (2026, 10, 25, 20, 48, 30)},
    "HQTV.zip": {"size": 43168355, "crc32": 1834085992, "date_time": (2026, 10, 25, 20, 48, 32)},
    "MBMPS.zip": {"size": 51918846, "crc32": 1005469331, "date_time": (2026, 10, 25, 20, 48, 34)},
    "NCC.zip": {"size": 42849663, "crc32": 2137294805, "date_time": (2026, 10, 25, 20, 48, 34)},
    "NJS.zip": {"size": 37321454, "crc32": 4299928, "date_time": (2026, 10, 25, 20, 48, 36)},
    "RRBI.zip": {"size": 40071148, "crc32": 1006727322, "date_time": (2026, 10, 25, 20, 48, 38)},
    "SKA.zip": {"size": 38156487, "crc32": 768672849, "date_time": (2026, 10, 25, 20, 48, 40)},
    "SVBI.zip": {"size": 32925294, "crc32": 1826285903, "date_time": (2026, 10, 25, 20, 48, 40)},
    "THV.zip": {"size": 44276965, "crc32": 1723859683, "date_time": (2026, 10, 25, 20, 48, 42)},
    "TLV.zip": {"size": 46519350, "crc32": 3123989594, "date_time": (2026, 10, 25, 20, 48, 44)},
    "TNI.zip": {"size": 37778272, "crc32": 3054346310, "date_time": (2026, 10, 25, 20, 48, 46)},
    "TXHC.zip": {"size": 41027615, "crc32": 358386652, "date_time": (2026, 10, 25, 20, 48, 50)},
    "YBAA.zip": {"size": 42199544, "crc32": 3661122291, "date_time": (2026, 10, 25, 20, 48, 52)},
    "YDCK.zip": {"size": 48264848, "crc32": 725414037, "date_time": (2026, 10, 25, 20, 48, 54)},
    "YKWK.zip": {"size": 41188671, "crc32": 354416425, "date_time": (2026, 10, 25, 20, 48, 56)},
    "ZHAA.zip": {"size": 37895316, "crc32": 2242469834, "date_time": (2026, 10, 25, 20, 48, 58)},
}


def compute_sha256(path: Path | str) -> str:
    """Computes SHA-256 digest of a file in 64KB blocks."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_arm(arm: str) -> str:
    """Validates experimental arm identifier (C0, C1, or BENCHMARK)."""
    norm = arm.strip().upper()
    if norm not in ("C0", "C1", "BENCHMARK"):
        raise ValueError(f"Invalid experimental arm '{arm}'. Strictly 'C0', 'C1', or 'BENCHMARK' required.")
    return norm


def verify_kaggle_gpu_device(device_idx: int = 0, allow_mock: bool = False) -> Dict[str, Any]:
    """
    Verifies that the execution device is an authorized Nvidia Tesla T4 GPU
    and that CUDA is active. Fails closed on CPU or non-T4 accelerators.
    """
    if allow_mock:
        return {
            "status": "MOCK_VERIFIED",
            "device_name": "Nvidia Tesla T4 (Mock)",
            "device_count": 2,
            "compute_capability": (7, 5),
            "cuda_available": True,
        }

    try:
        import torch
    except ImportError as e:
        raise RuntimeError("PyTorch is required for device verification.") from e

    if not torch.cuda.is_available():
        raise RuntimeError(
            "FAIL-CLOSED: CUDA is not available. Execution requires Nvidia Tesla T4 on Kaggle GPU. "
            "Local CPU or unaccelerated execution is strictly prohibited by AGENTS.md."
        )

    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("FAIL-CLOSED: No CUDA devices found.")

    device_name = torch.cuda.get_device_name(device_idx)
    major, minor = torch.cuda.get_device_capability(device_idx)

    # Validate Tesla T4 identity
    if "T4" not in device_name and "Tesla T4" not in device_name:
        raise RuntimeError(
            f"FAIL-CLOSED: Accelerator '{device_name}' does not match required Nvidia Tesla T4 hardware. "
            f"Official LANGFENS RQ1-TRN-02B must execute on Tesla T4 to match baseline precision and regime."
        )

    if (major, minor) != (7, 5):
        raise RuntimeError(
            f"FAIL-CLOSED: Unexpected compute capability ({major}, {minor}) for Tesla T4 (expected 7.5)."
        )

    return {
        "status": "VERIFIED",
        "device_name": device_name,
        "device_count": device_count,
        "compute_capability": (major, minor),
        "cuda_available": True,
    }


def verify_split_manifest(manifest_path: Path | str) -> str:
    """Verifies existence and exact cryptographic SHA-256 of the 4-fold split manifest."""
    p = Path(manifest_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"FAIL-CLOSED: Split manifest missing at '{p}'.")
    digest = compute_sha256(p)
    if digest != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            f"FAIL-CLOSED: Split manifest SHA-256 mismatch: computed {digest} != expected {EXPECTED_MANIFEST_SHA256}."
        )
    return digest


def verify_baseline_weights(baseline_path: Path | str) -> str:
    """Verifies existence and exact cryptographic SHA-256 of Baseline R best.pt."""
    p = Path(baseline_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"FAIL-CLOSED: Baseline R checkpoint missing at '{p}'.")
    digest = compute_sha256(p)
    if digest != EXPECTED_BASELINE_SHA256:
        raise ValueError(
            f"FAIL-CLOSED: Baseline R SHA-256 mismatch: computed {digest} != expected {EXPECTED_BASELINE_SHA256}."
        )
    return digest


def verify_bounded_zip_speakers(zip_path: Path) -> Set[str]:
    """
    Performs bounded inspection of the zip archive:
    Confirms all 18 approved Fold 1 speakers are present and no sealed test speakers exist.
    """
    observed_speakers: Set[str] = set()
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        for name in names:
            spk = name.replace(".zip", "")
            if spk in SEALED_TEST_SPEAKERS:
                raise ValueError(
                    f"FAIL-CLOSED: Sealed test speaker '{spk}' detected inside authorized archive '{zip_path}'. "
                    f"Test partition isolation strictly violated!"
                )
            if spk not in APPROVED_FOLD1_SPEAKERS:
                raise ValueError(
                    f"FAIL-CLOSED: Unapproved speaker '{spk}' detected inside archive '{zip_path}'."
                )
            observed_speakers.add(spk)

    if observed_speakers != APPROVED_FOLD1_SPEAKERS:
        missing = APPROVED_FOLD1_SPEAKERS - observed_speakers
        raise ValueError(f"FAIL-CLOSED: Archive missing required Fold 1 speakers: {missing}")

    return observed_speakers


def reconstruct_outer_zip_from_speaker_files(
    speaker_files_dir: Path,
    output_zip_path: Path,
) -> str:
    """
    Reconstructs the bit-identical outer zip archive from the 18 verified inner speaker zip files
    when Kaggle auto-extracts the archive during dataset ingestion.
    Verifies that the rebuilt archive matches EXPECTED_ZIP_SHA256 exactly.
    """
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = output_zip_path.with_suffix(output_zip_path.suffix + ".rebuild_tmp")

    # Verify each speaker package against AUTHORIZED_SPEAKER_LEDGER
    for spk_zip, meta in AUTHORIZED_SPEAKER_LEDGER.items():
        spk_path = speaker_files_dir / spk_zip
        if not spk_path.is_file():
            # Check if nested in a subfolder with archive name
            nested = list(speaker_files_dir.glob(f"**/{spk_zip}"))
            if nested and nested[0].is_file():
                spk_path = nested[0]
            else:
                raise FileNotFoundError(
                    f"FAIL-CLOSED: Auto-extracted package missing required speaker archive '{spk_zip}' in '{speaker_files_dir}'."
                )

        # Check size
        actual_size = spk_path.stat().st_size
        if actual_size != meta["size"]:
            raise ValueError(
                f"FAIL-CLOSED: Speaker archive '{spk_zip}' size mismatch: "
                f"actual={actual_size} != expected={meta['size']}."
            )

        # Check CRC-32
        with open(spk_path, "rb") as sf:
            import zlib
            crc = zlib.crc32(sf.read())
            if crc != meta["crc32"]:
                raise ValueError(
                    f"FAIL-CLOSED: Speaker archive '{spk_zip}' CRC-32 mismatch: "
                    f"actual={crc} ({hex(crc)}) != expected={meta['crc32']} ({hex(meta['crc32'])})."
                )

    # Check that no sealed test speaker zip exists in directory
    for item in speaker_files_dir.rglob("*.zip"):
        spk_stem = item.stem
        if spk_stem in SEALED_TEST_SPEAKERS:
            raise ValueError(
                f"FAIL-CLOSED: Sealed test speaker archive '{item.name}' detected in staging directory! Test leakage."
            )

    # Deterministic reconstruction in exact canonical order with exact timestamps and attributes
    with zipfile.ZipFile(tmp_out, "w", compression=zipfile.ZIP_STORED) as out_zf:
        for spk_zip, meta in AUTHORIZED_SPEAKER_LEDGER.items():
            spk_path = speaker_files_dir / spk_zip
            if not spk_path.is_file():
                nested = list(speaker_files_dir.glob(f"**/{spk_zip}"))
                spk_path = nested[0]

            data = spk_path.read_bytes()
            zinfo = zipfile.ZipInfo(filename=spk_zip, date_time=meta["date_time"])
            zinfo.compress_type = zipfile.ZIP_STORED
            zinfo.external_attr = 0x1800000
            zinfo.create_system = 3
            zinfo.create_version = 20
            zinfo.extract_version = 20
            zinfo.flag_bits = 0
            out_zf.writestr(zinfo, data)

    os.replace(tmp_out, output_zip_path)
    computed_sha = compute_sha256(output_zip_path)
    if computed_sha != EXPECTED_ZIP_SHA256:
        raise ValueError(
            f"FAIL-CLOSED: Reconstructed outer zip SHA-256 mismatch: "
            f"computed={computed_sha} != expected={EXPECTED_ZIP_SHA256}."
        )
    return computed_sha


def verify_and_stage_fold1_data(
    input_source: Path | str,
    target_staging_dir: Optional[Path | str] = None,
) -> Path:
    """
    Verifies authorized Fold 1 Train+Val data from input_source:
    - If input_source is a zip file: verifies SHA-256 and bounded speaker set.
    - If input_source is a directory:
      a) If the outer zip is inside the directory, verifies it directly.
      b) If Kaggle auto-extracted the zip into speaker packages, verifies all 18 speaker packages
         by size & CRC-32, confirms sealed test exclusion, and reconstructs the verified outer zip
         in target_staging_dir with SHA-256 verification.
    """
    src_p = Path(input_source).resolve()
    if not src_p.exists():
        raise FileNotFoundError(f"FAIL-CLOSED: Input source '{src_p}' does not exist.")

    # Case 1: Direct Zip File
    if src_p.is_file() and src_p.suffix == ".zip":
        sha = compute_sha256(src_p)
        if sha != EXPECTED_ZIP_SHA256:
            raise ValueError(
                f"FAIL-CLOSED: Fold 1 archive SHA-256 mismatch: "
                f"computed={sha} != expected={EXPECTED_ZIP_SHA256}."
            )
        verify_bounded_zip_speakers(src_p)
        return src_p

    # Case 2: Directory (Kaggle mountpoint under /kaggle/input/...)
    if src_p.is_dir():
        staging_dir = Path(target_staging_dir or "/kaggle/working/staged").resolve()
        staging_dir.mkdir(parents=True, exist_ok=True)

        # Check if direct outer zip exists inside directory
        candidates = list(src_p.glob("*.zip"))
        outer_candidates = [c for c in candidates if c.stat().st_size > 500_000_000]
        if outer_candidates:
            outer = outer_candidates[0]
            sha = compute_sha256(outer)
            if sha == EXPECTED_ZIP_SHA256:
                verify_bounded_zip_speakers(outer)
                return outer

        # Check for uncompressed binary archive (*.bin or *.dat) uploaded to bypass auto-extraction
        bin_candidates = list(src_p.glob("*.bin")) + list(src_p.glob("*.dat"))
        outer_bins = [b for b in bin_candidates if b.stat().st_size > 500_000_000]
        if outer_bins:
            outer_bin = outer_bins[0]
            sha = compute_sha256(outer_bin)
            if sha == EXPECTED_ZIP_SHA256:
                staged_zip = staging_dir / "authorized_RQ1-TRN-02A_fold1_train_val_only.zip"
                if not staged_zip.exists():
                    try:
                        staged_zip.symlink_to(outer_bin)
                    except OSError:
                        shutil.copyfile(str(outer_bin), str(staged_zip))
                verify_bounded_zip_speakers(staged_zip)
                return staged_zip

        # Check for auto-extracted speaker packages
        rebuilt_zip = staging_dir / "authorized_RQ1-TRN-02A_fold1_train_val_only.zip"
        reconstruct_outer_zip_from_speaker_files(src_p, rebuilt_zip)
        verify_bounded_zip_speakers(rebuilt_zip)
        return rebuilt_zip

    raise ValueError(f"FAIL-CLOSED: Unsupported input source path '{src_p}'.")


def resolve_kaggle_output_dir(
    working_root: Path | str,
    arm: str,
    run_id: Optional[str] = None,
    allow_conflicts: bool = False,
) -> Path:
    """
    Resolves and verifies per-arm output directory in /kaggle/working.
    Enforces strict path containment, traversal protection, and checks
    conflicting checkpoint files unless explicitly resuming.
    """
    arm_norm = validate_arm(arm)
    root_p = Path(working_root).resolve()

    if run_id:
        if not re.match(r"^[A-Za-z0-9_-]+$", run_id) or run_id in (".", ".."):
            raise ValueError(f"FAIL-CLOSED: Illegal run ID '{run_id}' contains unsafe path characters.")
        out_dir = root_p / "checkpoints" / "RQ1-TRN-02B" / arm_norm / run_id
    else:
        out_dir = root_p / "checkpoints" / "RQ1-TRN-02B" / arm_norm

    # Strict descendant containment check: must be strictly inside root_p (not equal)
    if not out_dir.resolve().is_relative_to(root_p) or out_dir.resolve() == root_p:
        raise ValueError(f"FAIL-CLOSED: Output directory '{out_dir}' is not a strict descendant of working root '{root_p}'.")

    # Conflict detection
    if not allow_conflicts and out_dir.is_dir():
        conflicts = []
        for chk in ("last.pt", "best.pt", "fixed_budget_checkpoint.pt", "halted_guardrail.pt"):
            if (out_dir / chk).exists():
                conflicts.append(str(out_dir / chk))
        if conflicts:
            raise FileExistsError(
                f"FAIL-CLOSED: Conflicting checkpoints detected in output directory:\n"
                f"  {conflicts}\n"
                f"Fresh run cannot overwrite existing experiments. Specify a unique --run-id or pass --resume."
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def stage_and_verify_kaggle_environment(
    config: Dict[str, Any],
    kaggle_input_dir: Optional[Path | str] = None,
    working_dir: Optional[Path | str] = None,
    remote_mirror_backend: str = "huggingface",
    remote_mirror_target: Optional[str] = None,
    allow_mock_gpu: bool = False,
    run_id: Optional[str] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    """
    Complete preflight staging and verification workflow for Kaggle T4x2 environment:
    1. Device verification (Tesla T4 GPU, CUDA active, FP16 AMP support).
    2. Immutable split manifest SHA-256 check.
    3. Baseline R best.pt weights verification.
    4. Fold 1 train/val-only dataset staging (with auto-extracted ZIP support and test exclusion).
    5. Per-arm output isolation and conflict check.
    6. Remote persistent storage readiness verification.
    """
    arm = validate_arm(config.get("arm", "C0"))
    work_root = Path(working_dir or "/kaggle/working").resolve()

    # Search for valid input_root
    input_root = None
    if kaggle_input_dir:
        candidate = Path(kaggle_input_dir).resolve()
        if candidate.is_dir():
            input_root = candidate
    if not input_root:
        for cand in [
            Path("/kaggle/input/langfens-rq1-fold1-staging"),
            Path("/kaggle/input/langfens-fold1-train-val-baseline"),
            Path(config.get("kaggle_input_dir", "/kaggle/input/langfens-fold1-train-val-baseline")),
        ]:
            if cand.is_dir():
                input_root = cand.resolve()
                break
    if not input_root and Path("/kaggle/input").is_dir():
        for sub in Path("/kaggle/input").iterdir():
            if sub.is_dir() and ((sub / "best.pt").is_file() or any(sub.glob("*.zip"))):
                input_root = sub.resolve()
                break
    if not input_root:
        input_root = Path(kaggle_input_dir or "/kaggle/input").resolve()

    # 1. Device check
    gpu_info = verify_kaggle_gpu_device(allow_mock=allow_mock_gpu)

    # 2. Manifest check
    manifest_path = Path(config.get("manifest", "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json"))
    manifest_sha = verify_split_manifest(manifest_path)

    # 3. Baseline weights check
    init_ckpt = config.get("initial_checkpoint")
    baseline_path = Path(init_ckpt) if init_ckpt else None
    if (not baseline_path or not baseline_path.is_file()) and input_root.is_dir():
        candidate = input_root / "best.pt"
        if candidate.is_file():
            baseline_path = candidate
        else:
            found = list(input_root.rglob("best.pt"))
            if found:
                baseline_path = found[0]
    if not baseline_path:
        baseline_path = Path("checkpoints/RQ1-TRN-02A/recovered/best.pt")
    baseline_sha = verify_baseline_weights(baseline_path)

    # 4. Fold 1 dataset staging & test sealing
    zip_source = config.get("zip_path")
    zip_path = Path(zip_source) if zip_source and Path(zip_source).exists() else None
    if not zip_path and input_root.is_dir():
        # Check for zip file inside input_root
        zips = list(input_root.glob("*.zip"))
        outer_zips = [z for z in zips if z.stat().st_size > 500_000_000]
        if outer_zips:
            zip_path = outer_zips[0]
        else:
            zip_path = input_root
    if not zip_path:
        zip_path = Path("datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip")

    staged_zip = verify_and_stage_fold1_data(
        input_source=zip_path,
        target_staging_dir=work_root / "staged",
    )
    staged_sha = compute_sha256(staged_zip)

    # 5. Output directory & conflict resolution
    out_dir = resolve_kaggle_output_dir(
        working_root=work_root,
        arm=arm,
        run_id=run_id,
        allow_conflicts=resume,
    )

    # 6. Remote mirror readiness check (fail-closed if unverified)
    mirror = KaggleRemoteMirror(
        backend=remote_mirror_backend,
        remote_target=remote_mirror_target,
        require_verification=True,
    )
    mirror_ok, mirror_msg = mirror.verify_readiness()
    if not mirror_ok:
        raise RuntimeError(f"FAIL-CLOSED: {mirror_msg}")

    report = {
        "status": "PASS_STAGING",
        "arm": arm,
        "run_id": run_id,
        "platform": "KAGGLE_T4_CUDA_FP16_AMP",
        "gpu": gpu_info,
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha,
        },
        "baseline_weights": {
            "path": str(baseline_path),
            "sha256": baseline_sha,
        },
        "dataset": {
            "path": str(staged_zip),
            "sha256": staged_sha,
            "speakers_count": len(APPROVED_FOLD1_SPEAKERS),
            "test_speakers_count": 0,
        },
        "output_directory": str(out_dir),
        "remote_mirror": {
            "backend": remote_mirror_backend,
            "target": str(remote_mirror_target),
            "status": "VERIFIED",
            "message": mirror_msg,
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="LANGFENS — Kaggle T4x2 Staging and Verification")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--arm", type=str, choices=["C0", "C1", "BENCHMARK"], help="Override arm")
    parser.add_argument("--kaggle-input-dir", type=str, default=None, help="Path to /kaggle/input")
    parser.add_argument("--working-dir", type=str, default="/kaggle/working", help="Path to /kaggle/working")
    parser.add_argument("--remote-mirror-backend", type=str, default="huggingface", choices=list(KaggleRemoteMirror.SUPPORTED_BACKENDS))
    parser.add_argument("--remote-mirror-target", type=str, default="minkhoaa/langfens-rq1-trn-02b-checkpoints", help="Remote persistent mirror destination")
    parser.add_argument("--run-id", type=str, default=None, help="Unique run identifier")
    parser.add_argument("--resume", action="store_true", help="Resumption mode (allows pre-existing checkpoints)")
    args, unknown = parser.parse_known_args()

    # Reject unexpected flags (preserve Jupyter ipykernel connection arguments)
    disallowed = [u for u in unknown if not (u.startswith("-f") or u.endswith(".json") or "kernel" in u)]
    if disallowed:
        parser.error(f"unrecognized arguments: {' '.join(disallowed)}")

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.arm:
        cfg["arm"] = args.arm

    report = stage_and_verify_kaggle_environment(
        config=cfg,
        kaggle_input_dir=args.kaggle_input_dir,
        working_dir=args.working_dir,
        remote_mirror_backend=args.remote_mirror_backend,
        remote_mirror_target=args.remote_mirror_target,
        run_id=args.run_id,
        resume=args.resume,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
