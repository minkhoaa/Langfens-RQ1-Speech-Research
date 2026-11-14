#!/usr/bin/env python3
"""Verify exact-order specialist OOF scores and add per-fold AP details."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

import run_audit as audit


HERE = Path(__file__).resolve().parent
GROUPS = (("ABA", "ASI", "BWC"), ("EBVS", "HJK", "HKK"),
          ("HQTV", "LXC", "NJS"), ("PNV", "SKA", "SVBI"))


def main() -> None:
    data = audit.load_data()
    with np.load(HERE / "MANNER_SPECIALIST_OOF.npz") as pred:
        if not np.array_equal(pred["target_keys"], [row["key"] for row in data["rows"]]):
            raise ValueError("specialist event order mismatch")
        if not np.array_equal(pred["gold_target"], data["y"]):
            raise ValueError("specialist gold mismatch")
        score = pred["score"].copy()
    if len(score) != 59562 or not np.isfinite(score).all():
        raise ValueError("specialist OOF coverage invalid")
    path = HERE / "MANNER_SPECIALIST_RESULTS.json"
    result = json.loads(path.read_text())
    result["per_outer_fold_ap"] = []
    for fold, speakers in enumerate(GROUPS):
        row = {"fold": fold, "held_speakers": list(speakers)}
        for manner in ("Vowel", "Stop"):
            mask = np.isin(data["speaker"], speakers) & (data["manner"] == manner)
            y = data["y"][mask] == 1
            row[manner] = {"support": int(y.sum()),
                           "baseline_ap": float(average_precision_score(y, data["s"][mask])),
                           "specialist_ap": float(average_precision_score(y, score[mask]))}
        result["per_outer_fold_ap"].append(row)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print("Updated", path)


if __name__ == "__main__":
    main()
