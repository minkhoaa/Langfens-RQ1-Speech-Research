#!/usr/bin/env python3
"""Independent model-free recomputation from coordinator-frozen TRN-04R rows.

Expected layout: INPUT/{C0,T1,T2,T3}/ with the seven named JSONL files.
No model/audio/label sources are loaded. Use a new output directory; raw inputs
and resulting records must remain in the approved private store.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from langfens_prep.trn04r.metrics import summarize_cell, factorial_contrasts

CELLS = ("C0", "T1", "T2", "T3")
FILES = ("runtime_Z.jsonl", "tokens_Z.jsonl", "cal_runtime_Z.jsonl", "cal_tokens_Z.jsonl",
         "mismatched_runtime_Z.jsonl", "mismatched_tokens_Z.jsonl", "recognition_Z.jsonl")
EXPECTED_GOLD = {0: 8799, 1: 1012, 2: 145}
EXPECTED_ALIGNED = {0: 8774, 1: 1008, 2: 137}


def load(path: Path):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite {value}")))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid JSONL {path}:{number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{number}")
            rows.append(row)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_z_contract(summary: dict, cell: str) -> None:
    if summary["gold_rows"] != 9956:
        raise ValueError(f"{cell}: expected 9,956 Z gold rows, got {summary['gold_rows']}")
    labels = {int(k): v for k, v in summary["actual_label_counts"].items()}
    if labels != EXPECTED_GOLD:
        raise ValueError(f"{cell}: gold label census mismatch: {labels}")
    aligned = summary["coverage"]["answered_s_error"]
    # Score absence is an abstention, so prompt mapping coverage must be determined
    # from the independent mapping key, not inferred from score availability.
    tokens = summary.pop("_tokens")
    mapped = [r for r in tokens if r.get("gold_canonical_index") is not None and r.get("prompt_index") is not None]
    mapped_labels = {op: sum(r["operation_target"] == op for r in mapped) for op in (0, 1, 2)}
    if mapped_labels != EXPECTED_ALIGNED:
        raise ValueError(f"{cell}: mapped Z label census mismatch: {mapped_labels}")
    extras = [r for r in tokens if r.get("gold_canonical_index") is None]
    if len(extras) != 37 or summary["coverage"]["prompt_extras"] != 37:
        raise ValueError(f"{cell}: expected 37 prompt extras; got {len(extras)}")
    if len(mapped) != 9919:
        raise ValueError(f"{cell}: expected 9,919 mapped canonical positions; got {len(mapped)}")
    omission = summary["metrics"]["omission_legacy"]
    if omission["status"] != "computed":
        return
    for ceiling, point in omission["cal_operating_points"].items():
        counts = point.get("counts")
        if counts is None:
            continue
        tp = counts["TP"]
        expected_fn = 145 - tp
        if tp > 137 or counts["FN"] != expected_fn:
            raise ValueError(f"{cell}: omission accounting invariant failed at CAL FPR {ceiling}: TP={tp}, FN={counts['FN']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {args.output}")
    all_results = {}
    input_hashes = {}
    for cell in CELLS:
        directory = args.input_root / cell
        missing = [filename for filename in FILES if not (directory / filename).is_file()]
        if missing:
            raise FileNotFoundError(f"{cell}: missing frozen exchange files: {missing}")
        loaded = {name: load(directory / name) for name in FILES}
        input_hashes.update({f"{cell}/{name}": sha256(directory / name) for name in FILES})
        runtime, tokens, cal_runtime, cal_tokens, mismatch_runtime, mismatch_tokens, recognition = (loaded[name] for name in FILES)
        # Validate calibration runtime presence/key uniqueness as well as its offline join.
        for label, records in (("CAL runtime", cal_runtime), ("mismatched runtime", mismatch_runtime)):
            keys = [(r.get("speaker_id"), r.get("utterance_id"), r.get("prompt_index")) for r in records]
            if len(keys) != len(set(keys)):
                raise ValueError(f"{cell}: duplicate key in {label}")
        # C0 is the binary B-Control; T1-T3 are operation-score treatments.
        is_binary_control = cell == "C0"
        primary = summarize_cell(runtime, tokens, cal_tokens, recognition, control=is_binary_control, cell_id=cell,
                                 cal_runtime=cal_runtime)
        mismatch = summarize_cell(mismatch_runtime, mismatch_tokens, cal_tokens, (), control=is_binary_control,
                                  cell_id=cell + "_mismatched", cal_runtime=cal_runtime)
        primary["_tokens"] = tokens
        verify_z_contract(primary, cell)
        primary["mismatched_acoustic_control"] = mismatch
        # Control alarm counts are separately evaluated on prompt extras; extras never
        # enter canonical gold AP/denominators.
        all_results[cell] = primary
    # The pre-registered factorial contrasts are computed only where comparable
    # scalar endpoints and the same threshold definition exist in all cells.
    scalar = {}
    for cell, result in all_results.items():
        om = result["metrics"]["omission_legacy"]
        points = om.get("cal_operating_points", {})
        point = points.get("0.1", {}).get("counts", {})
        scalar[cell] = {
            "binary_ap": result["metrics"]["binary"].get("ap"),
            "omission_dedicated_ap": result["metrics"]["omission_dedicated"].get("ap"),
            "omission_recall_all_gold": point.get("recall_all_gold"),
            "correct_fpr_all_gold": point.get("correct_fpr_all_gold_lower"),
        }
    report = {"status": "RECOMPUTED_FROM_FROZEN_ROWS", "execution_scope": "model-free CPU metric accounting only",
              "source_hashes": input_hashes, "gold_population": 9956,
              "registered_gold_label_count_expectations": EXPECTED_GOLD,
              "registered_aligned_label_count_expectations": EXPECTED_ALIGNED,
              "cell_results": all_results,
              "factorial_contrasts": factorial_contrasts(scalar),
              "registration_gate_status": "NOT_EVALUATED: numerical registered margins/external reference required from frozen protocol",
              "noninferiority_status": "NOT_EVALUATED: predeclared margin/reference/uncertainty specification required",
              "notes": ["T-OP dedicated s_omit/s_sub and s_error are distinct score semantics.",
                        "B-Control dedicated subtype AP is null/unavailable, never imputed.",
                        "Extras are excluded from gold metrics and reported in separate alarm ledger.",
                        "Validation thresholds/metrics are distinct from CAL-selected operating points.",
                        "Matched contrasts are T1-C0, T2-C0, T3-C0; interaction is T3-T2-T1+C0."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"wrote {args.output}; cells={len(all_results)}; source_files={len(input_hashes)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
