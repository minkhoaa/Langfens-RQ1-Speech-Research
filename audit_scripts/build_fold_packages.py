#!/usr/bin/env python3
"""
LANGFENS — Reusable Deterministic Fold Package Builder.
Author: Senior Speech ML Research Engineer & Reproducibility Auditor.

Builds train-and-validation-only zip packages for each L2-ARCTIC cross-validation fold:
  `authorized_RQ1-TRN_fold{k}_train_val_only.zip`

Strict Invariants:
  1. The active test partition S_test^(k) is 100% physically excluded.
  2. Non-gold / orphan audio files are excluded.
  3. Only the exact gold train and val utterances listed in `l2arctic_4fold_splits.json`
     are included ({speaker}/wav/{utt}.wav and {speaker}/annotation/{utt}.TextGrid).
  4. Packaging is deterministic: sorted speaker entries, fixed ZipInfo timestamps,
     and stored outer compression matching the Fold 1 baseline artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile
from typing import Dict, List, Set, Tuple


DEFAULT_MANIFEST = Path("manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json")
DEFAULT_SOURCE_ZIP = Path("datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip")
DEFAULT_OUTPUT_DIR = Path("datasets/L2-ARCTIC")
FIXED_ZIP_DATETIME = (2026, 1, 1, 0, 0, 0)


def compute_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_fold_plan(
    manifest_path: Path,
    fold_idx: int
) -> Dict[str, any]:
    """Extracts authorized speakers and utterance lists for fold_idx."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        mdata = json.load(f)

    fold_obj = next((f for f in mdata["folds"] if f["fold_index"] == fold_idx), None)
    if fold_obj is None:
        raise ValueError(f"Fold {fold_idx} not found in manifest: {manifest_path}")

    train_spks = set(fold_obj["train"]["speakers"])
    val_spks = set(fold_obj["val"]["speakers"])
    test_spks = set(fold_obj["test"]["speakers"])

    assert train_spks.isdisjoint(val_spks), f"Fold {fold_idx}: Train/Val overlap"
    assert train_spks.isdisjoint(test_spks), f"Fold {fold_idx}: Train/Test overlap"
    assert val_spks.isdisjoint(test_spks), f"Fold {fold_idx}: Val/Test overlap"

    authorized_spks = sorted(train_spks | val_spks)
    assert len(authorized_spks) == 18, f"Expected 18 authorized speakers, got {len(authorized_spks)}"
    assert len(test_spks) == 6, f"Expected 6 test speakers, got {len(test_spks)}"

    # Map authorized utterances per speaker
    utterances_by_speaker: Dict[str, List[str]] = {spk: [] for spk in authorized_spks}
    for utt in fold_obj["train"]["utterances"] + fold_obj["val"]["utterances"]:
        parts = utt.split("/")
        spk, utt_id = parts[0], parts[1]
        if spk in utterances_by_speaker:
            utterances_by_speaker[spk].append(utt_id)
        else:
            raise ValueError(f"Utterance {utt} belongs to non-authorized speaker {spk}")

    # Sort utterance lists for determinism
    for spk in utterances_by_speaker:
        utterances_by_speaker[spk].sort()

    return {
        "fold_index": fold_idx,
        "train_speakers": sorted(train_spks),
        "val_speakers": sorted(val_spks),
        "test_speakers": sorted(test_spks),
        "authorized_speakers": authorized_spks,
        "utterances_by_speaker": utterances_by_speaker,
        "train_utterance_count": fold_obj["train"]["utterance_count"],
        "val_utterance_count": fold_obj["val"]["utterance_count"],
        "test_utterance_count": fold_obj["test"]["utterance_count"]
    }


def build_fold_package(
    source_zip_path: Path,
    manifest_path: Path,
    fold_idx: int,
    output_zip_path: Path,
    dry_run: bool = False
) -> Dict[str, any]:
    """Deterministically builds or dry-runs an authorized fold package."""
    plan = get_fold_plan(manifest_path, fold_idx)
    auth_spks = plan["authorized_speakers"]
    test_spks = set(plan["test_speakers"])

    print(f"\n[BUILD PLAN] Fold {fold_idx}:")
    print(f"  Authorized speakers (18): {auth_spks}")
    print(f"  Sealed test speakers (6, EXCLUDED): {sorted(test_spks)}")
    print(f"  Train utterances: {plan['train_utterance_count']}")
    print(f"  Val utterances: {plan['val_utterance_count']}")

    if dry_run:
        print(f"  [DRY-RUN] Verified isolation and file counts. Output file NOT written: {output_zip_path}")
        return {
            "fold_index": fold_idx,
            "status": "DRY_RUN_PASSED",
            "authorized_speakers": auth_spks,
            "test_speakers_excluded": sorted(test_spks),
            "output_path": str(output_zip_path)
        }

    output_zip_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_zip_path.with_suffix(".tmp")

    with zipfile.ZipFile(source_zip_path, "r") as source_master:
        available_files = set(source_master.namelist())

        # Verify that all authorized speaker zips exist in master
        for spk in auth_spks:
            spk_zip_name = f"{spk}.zip"
            if spk_zip_name not in available_files:
                raise FileNotFoundError(f"Speaker archive {spk_zip_name} missing from source {source_zip_path}")

        with zipfile.ZipFile(temp_output_path, "w", zipfile.ZIP_STORED) as out_master:
            for spk in auth_spks:
                spk_zip_name = f"{spk}.zip"
                # Never write a test speaker!
                assert spk not in test_spks, f"CRITICAL: Attempted to package test speaker {spk}!"

                print(f"  Packaging speaker {spk} ({len(plan['utterances_by_speaker'][spk])} utts)...")
                raw_spk_zip_bytes = source_master.read(spk_zip_name)
                with zipfile.ZipFile(io.BytesIO(raw_spk_zip_bytes), "r") as in_spk_zf:
                    inner_buf = io.BytesIO()
                    with zipfile.ZipFile(inner_buf, "w", zipfile.ZIP_DEFLATED) as out_spk_zf:
                        for utt_id in plan["utterances_by_speaker"][spk]:
                            wav_path = f"{spk}/wav/{utt_id}.wav"
                            tg_path = f"{spk}/annotation/{utt_id}.TextGrid"

                            wav_data = in_spk_zf.read(wav_path)
                            tg_data = in_spk_zf.read(tg_path)

                            # Deterministic ZipInfo for wav
                            wav_info = zipfile.ZipInfo(wav_path, date_time=FIXED_ZIP_DATETIME)
                            wav_info.compress_type = zipfile.ZIP_DEFLATED
                            out_spk_zf.writestr(wav_info, wav_data)

                            # Deterministic ZipInfo for TextGrid
                            tg_info = zipfile.ZipInfo(tg_path, date_time=FIXED_ZIP_DATETIME)
                            tg_info.compress_type = zipfile.ZIP_DEFLATED
                            out_spk_zf.writestr(tg_info, tg_data)

                    inner_bytes = inner_buf.getvalue()

                # Add outer speaker zip entry with fixed metadata
                outer_info = zipfile.ZipInfo(spk_zip_name, date_time=FIXED_ZIP_DATETIME)
                outer_info.compress_type = zipfile.ZIP_STORED
                out_master.writestr(outer_info, inner_bytes)

    # Atomic move
    os.replace(temp_output_path, output_zip_path)
    pkg_sha256 = compute_sha256(output_zip_path)
    pkg_size = output_zip_path.stat().st_size

    print(f"  [SUCCESS] Built package: {output_zip_path}")
    print(f"  Size: {pkg_size:,} bytes")
    print(f"  SHA-256: {pkg_sha256}")

    # Post-build integrity verification
    with zipfile.ZipFile(output_zip_path, "r") as verify_zf:
        packaged_names = set(verify_zf.namelist())
        expected_names = {f"{spk}.zip" for spk in auth_spks}
        assert packaged_names == expected_names, f"Package names mismatch: {packaged_names ^ expected_names}"
        for test_spk in test_spks:
            assert f"{test_spk}.zip" not in packaged_names, f"LEAKAGE: {test_spk}.zip found in {output_zip_path}!"

    return {
        "fold_index": fold_idx,
        "status": "BUILT",
        "output_path": str(output_zip_path),
        "size_bytes": pkg_size,
        "sha256": pkg_sha256,
        "authorized_speakers": auth_spks,
        "test_speakers_excluded": sorted(test_spks)
    }


def main():
    parser = argparse.ArgumentParser(description="LANGFENS Fold Package Builder")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-zip", type=Path, default=DEFAULT_SOURCE_ZIP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", type=int, nargs="+", default=[2, 3, 4], help="Folds to build (e.g. 2 3 4)")
    parser.add_argument("--dry-run", action="store_true", help="Audit without writing zip files")
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if not args.source_zip.is_file():
        raise FileNotFoundError(f"Source raw zip not found: {args.source_zip}")

    results = []
    for f_idx in args.folds:
        out_name = f"authorized_RQ1-TRN_fold{f_idx}_train_val_only.zip"
        out_path = args.output_dir / out_name
        res = build_fold_package(
            source_zip_path=args.source_zip,
            manifest_path=args.manifest,
            fold_idx=f_idx,
            output_zip_path=out_path,
            dry_run=args.dry_run
        )
        results.append(res)

    print("\n" + "=" * 80)
    print("PACKAGE BUILD SUMMARY:")
    for r in results:
        print(f"  Fold {r['fold_index']}: status={r['status']} path={r['output_path']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
