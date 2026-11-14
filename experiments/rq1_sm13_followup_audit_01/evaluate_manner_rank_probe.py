#!/usr/bin/env python3
"""Recompute pilot ranking and descriptive threshold-transfer summaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

import run_audit as audit


HERE = Path(__file__).resolve().parent


def main() -> None:
    data = audit.load_data()
    with np.load(HERE / "MANNER_RANK_PROBE_OOF.npz") as pred:
        score = pred["score"].copy()
        if not np.array_equal(pred["gold_target"], data["y"]):
            raise ValueError("pilot gold mismatch")
        if not np.array_equal(pred["target_keys"], [r["key"] for r in data["rows"]]):
            raise ValueError("pilot event order mismatch")
    if len(score) != 59562 or not np.isfinite(score).all():
        raise ValueError("pilot score coverage invalid")
    candidate = dict(data)
    candidate["s"] = score
    results_path = HERE / "MANNER_RANK_PROBE_RESULTS.json"
    results = json.loads(results_path.read_text())
    results["per_outer_fold_ap"] = []
    for fold, speakers in enumerate((
        ("ABA", "ASI", "BWC"), ("EBVS", "HJK", "HKK"),
        ("HQTV", "LXC", "NJS"), ("PNV", "SKA", "SVBI"),
    )):
        row = {"fold": fold, "held_speakers": list(speakers)}
        for manner in ("Vowel", "Stop", "Other"):
            mask = np.isin(data["speaker"], speakers) & (data["manner"] == manner)
            y = data["y"][mask] == 1
            row[manner] = {
                "support": int(y.sum()),
                "baseline_ap": float(average_precision_score(y, data["s"][mask])),
                "candidate_ap": float(average_precision_score(y, score[mask])),
            }
        results["per_outer_fold_ap"].append(row)
    results["descriptive_threshold_transfer"] = {}
    for manner in ("Vowel", "Stop"):
        results["descriptive_threshold_transfer"][manner] = {}
        for label, minimum in (("fwr_only", None), ("fwr_and_train_p50", 0.5)):
            results["descriptive_threshold_transfer"][manner][label] = {
                "baseline": audit.crossfit(data, manner, minimum)["pooled"],
                "candidate": audit.crossfit(candidate, manner, minimum)["pooled"],
            }
    results["threshold_transfer_caveat"] = (
        "Thresholds selected on other TRAIN OOF speaker scores, but their fitted SM13 models "
        "may have seen the target held-out speakers. This is a descriptive transfer check, "
        "not fully nested independent calibration or deployment evidence."
    )
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    print("Updated", results_path)


if __name__ == "__main__":
    main()
