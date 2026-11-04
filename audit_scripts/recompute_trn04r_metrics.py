#!/usr/bin/env python3
"""Independent model-free recomputation from coordinator-frozen TRN-04R rows.

Accepted layouts: INPUT/C0_endpoint, INPUT/arms/C0/C0_endpoint, or INPUT/C0/C0_endpoint (likewise other cells), with eight named JSONL files. Symlinks and ambiguous layouts are rejected.
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
from langfens_prep.trn04r.metrics import (summarize_cell, factorial_contrasts,
                                           validate_speaker_census, VAL_SPEAKERS, CAL_SPEAKERS)

CELLS = ("C0", "T1", "T2", "T3")
FILES = ("runtime_Z.jsonl", "tokens_Z.jsonl", "cal_runtime_Z.jsonl", "cal_tokens_Z.jsonl",
         "mismatched_runtime_Z.jsonl", "mismatched_tokens_Z.jsonl", "recognition_Z.jsonl",
         "recognition_runtime_Z.jsonl")
EXPECTED_GOLD = {0: 8799, 1: 1012, 2: 145}
EXPECTED_ALIGNED = {0: 8774, 1: 1008, 2: 137}
OBJECTIVES = {"C0": "operation_ce", "T1": "class_focal",
              "T2": "operation_ce", "T3": "class_focal"}


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


def _reject_symlink_components(root: Path, path: Path) -> None:
    root = root.absolute()
    path = path.absolute()
    if root.is_symlink():
        raise ValueError(f"input root must not be a symlink: {root}")
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"resolved input escaped input root: {path}") from exc
    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"input path must not contain symlinks: {current}")


def resolve_cell_directory(input_root: Path, cell: str, endpoint: str) -> Path:
    """Resolve only the three authorized layouts; refuse ambiguity and symlinks."""
    root = input_root.absolute()
    if not root.is_dir():
        raise FileNotFoundError(f"input root is not a directory: {input_root}")
    if root.is_symlink():
        raise ValueError(f"input root must not be a symlink: {root}")
    leaf = f"{cell}_{endpoint}"
    candidates = (root / leaf, root / "arms" / cell / leaf, root / cell / leaf)
    for candidate in candidates:
        _reject_symlink_components(root, candidate)
    matches = [candidate for candidate in candidates if candidate.is_dir()]
    if not matches:
        raise FileNotFoundError(f"no authorized input layout for {cell}/{endpoint}; tried {list(map(str, candidates))}")
    if len(matches) != 1:
        raise ValueError(f"ambiguous input layout for {cell}/{endpoint}: {list(map(str, matches))}")
    return matches[0]


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


def validate_recognition_pair(runtime_rows, offline_rows):
    """Keep gold-free saved decoding separate from offline references/edit counts."""
    forbidden = {"ref_phones", "reference_phones", "edit_counts", "gold_canonical_index",
                 "operation_target", "binary_target", "canonical_phone", "gold_phone"}
    runtime = {}
    for row in runtime_rows:
        key = (row.get("speaker_id"), row.get("utterance_id"))
        if key in runtime:
            raise ValueError(f"duplicate recognition runtime utterance {key}")
        if forbidden.intersection(row):
            raise ValueError(f"gold/offline fields in recognition runtime row {key}")
        if not isinstance(row.get("hyp_phones"), list):
            raise ValueError(f"recognition runtime row lacks saved hyp_phones: {key}")
        runtime[key] = row
    offline = {}
    for row in offline_rows:
        key = (row.get("speaker_id"), row.get("utterance_id"))
        if key in offline:
            raise ValueError(f"duplicate offline recognition utterance {key}")
        offline[key] = row
    if set(runtime) != set(offline):
        raise ValueError("recognition runtime/offline utterance keys differ")
    if any(not isinstance(row.get("ref_phones"), list) or not isinstance(row.get("hyp_phones"), list)
           or not isinstance(row.get("edit_counts"), list) for row in offline_rows):
        raise ValueError("offline recognition rows require ref_phones, hyp_phones and edit_counts")
    for key, row in runtime.items():
        if "hyp_phones" in row and row["hyp_phones"] != offline[key].get("hyp_phones"):
            raise ValueError(f"recognition hypothesis changed across offline join: {key}")


def primary_scalar_values(result):
    """Registered matched-cell scalar endpoints; omission uses dedicated CAL-15."""
    omission = result["metrics"]["omission_dedicated"]
    operating = omission.get("cal_operating_points", {}).get("0.15", {}).get("counts", {})
    binary_ap = result["metrics"]["binary"].get("ap")
    mismatch_ap = (result.get("mismatched_acoustic_control", {}).get("metrics", {})
                   .get("binary", {}).get("ap"))
    return {
        "binary_ap": binary_ap,
        "omission_dedicated_ap": omission.get("ap"),
        "omission_precision_cal15": operating.get("precision"),
        "omission_recall_all_gold_cal15": operating.get("recall_all_gold"),
        "omission_fa_per_1000_cal15": operating.get("false_alarms_per_1000_full_gold"),
        "substitution_legacy_ap": result["metrics"]["substitution_legacy"].get("ap"),
        "substitution_dedicated_ap": result["metrics"]["substitution_dedicated"].get("ap"),
        "PER": result["recognition"].get("PER"),
        "acoustic_gap_binary_ap": binary_ap - mismatch_ap
            if binary_ap is not None and mismatch_ap is not None else None,
    }


def evaluate_registered_gates(scalar, endpoint):
    """Apply frozen Protocol Z gates only to the registered fixed-budget endpoint."""
    if endpoint != "fixed_budget_3600":
        return {cell: {"Level_A": {"status": "not_applicable", "reason": "registered absolute gates target fixed_budget_3600"},
                       "Level_B": {"status": "not_applicable", "reason": "registered absolute gates target fixed_budget_3600"}}
                for cell in scalar}
    levels = {
        "Level_A": {"omission_precision_cal15": (">=", .25),
                    "omission_recall_all_gold_cal15": (">=", .30),
                    "omission_fa_per_1000_cal15": ("<=", 30),
                    "binary_ap": (">=", .313619),
                    "substitution_legacy_ap": (">=", .303974),
                    "PER": ("<=", .246980)},
        "Level_B": {"omission_precision_cal15": (">=", .30),
                    "omission_recall_all_gold_cal15": (">=", .40),
                    "omission_fa_per_1000_cal15": ("<=", 15),
                    "binary_ap": (">=", .313619),
                    "substitution_legacy_ap": (">=", .303974),
                    "PER": ("<=", .246980),
                    "acoustic_gap_binary_ap": (">=", .010)},
    }
    result = {}
    for cell, values in scalar.items():
        result[cell] = {}
        for level, criteria in levels.items():
            checks = {}
            for metric, (operator, threshold) in criteria.items():
                value = values.get(metric)
                if value is None or not math.isfinite(value):
                    checks[metric] = {"status": "unavailable", "value": value,
                                      "operator": operator, "threshold": threshold}
                else:
                    passed = value >= threshold if operator == ">=" else value <= threshold
                    checks[metric] = {"status": "pass" if passed else "fail", "value": value,
                                      "operator": operator, "threshold": threshold}
            statuses = {entry["status"] for entry in checks.values()}
            result[cell][level] = {"status": "pass" if statuses == {"pass"} else
                                   "fail" if "fail" in statuses else "unavailable",
                                   "checks": checks}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--endpoint", choices=("fixed_budget_3600", "best"), default="fixed_budget_3600")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {args.output}")
    all_results = {}
    input_hashes = {}
    for cell in CELLS:
        directory = resolve_cell_directory(args.input_root, cell, args.endpoint)
        symlink_files = [filename for filename in FILES if (directory / filename).is_symlink()]
        if symlink_files:
            raise ValueError(f"{cell}: input files must not be symlinks: {symlink_files}")
        missing = [filename for filename in FILES if not (directory / filename).is_file()]
        if missing:
            raise FileNotFoundError(f"{cell}: missing frozen exchange files: {missing}")
        loaded = {name: load(directory / name) for name in FILES}
        input_hashes.update({f"{cell}_{args.endpoint}/{name}": sha256(directory / name) for name in FILES})
        (runtime, tokens, cal_runtime, cal_tokens, mismatch_runtime, mismatch_tokens,
         recognition, recognition_runtime) = (loaded[name] for name in FILES)
        for records, speakers, label in (
            (runtime, VAL_SPEAKERS, "VAL runtime"), (tokens, VAL_SPEAKERS, "VAL tokens"),
            (mismatch_runtime, VAL_SPEAKERS, "mismatched VAL runtime"),
            (mismatch_tokens, VAL_SPEAKERS, "mismatched VAL tokens"),
            (recognition, VAL_SPEAKERS, "offline recognition"),
            (recognition_runtime, VAL_SPEAKERS, "runtime recognition"),
            (cal_runtime, CAL_SPEAKERS, "CAL runtime"), (cal_tokens, CAL_SPEAKERS, "CAL tokens")):
            validate_speaker_census(records, speakers, f"{cell} {label}")
        validate_recognition_pair(recognition_runtime, recognition)
        # Validate calibration runtime presence/key uniqueness as well as its offline join.
        for label, records in (("CAL runtime", cal_runtime), ("mismatched runtime", mismatch_runtime)):
            keys = [(r.get("speaker_id"), r.get("utterance_id"), r.get("prompt_index")) for r in records]
            if len(keys) != len(set(keys)):
                raise ValueError(f"{cell}: duplicate key in {label}")
        objective = OBJECTIVES[cell]
        primary = summarize_cell(runtime, tokens, cal_tokens, recognition, objective=objective, cell_id=cell,
                                 cal_runtime=cal_runtime)
        mismatch = summarize_cell(mismatch_runtime, mismatch_tokens, cal_tokens, (), objective=objective,
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
        scalar[cell] = primary_scalar_values(result)
    report = {"status": "RECOMPUTED_FROM_FROZEN_ROWS", "execution_scope": "model-free CPU metric accounting only",
              "endpoint": args.endpoint, "source_hashes": input_hashes, "gold_population": 9956,
              "registered_gold_label_count_expectations": EXPECTED_GOLD,
              "registered_aligned_label_count_expectations": EXPECTED_ALIGNED,
              "cell_results": all_results,
              "factorial_contrasts": factorial_contrasts(scalar),
              "registered_absolute_gates": evaluate_registered_gates(scalar, args.endpoint),
              "gate_interpretation": "Absolute external-reference gates are separate from internal matched C0 factorial contrasts; descriptive VAL results do not establish population superiority.",
              "notes": ["T-OP dedicated s_omit/s_sub and s_error are distinct score semantics.",
                        "Only binary_collapsed B-Control dedicated subtype AP is null/unavailable, never imputed.",
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
