#!/usr/bin/env python3
"""Describe TRAIN OOF vowel/stop misses and false alerts at matched-risk points."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

import run_audit as audit


HERE = Path(__file__).resolve().parent


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    d = audit.load_data()
    all_speakers = set(d["speaker"])
    with (HERE / "RAW_SAME_PHONE_LABELS.csv").open(newline="") as f:
        starred = {r["event_key"] for r in csv.DictReader(f) if r["realized_has_deviation_star"] == "True"}
    phone = np.asarray([r["phone"] for r in d["rows"]])
    realized = np.asarray([r["realized"] for r in d["rows"]])
    star = np.asarray([r["key"] in starred for r in d["rows"]])
    by_phone, by_speaker = [], []
    for manner in ("Vowel", "Stop"):
        threshold = audit.select(d, all_speakers, manner, None)["threshold"]
        group = d["manner"] == manner
        s_trigger = d["s"] >= threshold
        o_trigger = d["o"] >= audit.TAU_O
        flag = group & s_trigger & (~o_trigger | (d["s"] - threshold > d["o"] - audit.TAU_O))
        for p in sorted(set(phone[group])):
            subset = group & (phone == p)
            y = d["y"][subset] == 1
            if not len(y) or not y.any():
                continue
            positive = subset & (d["y"] == 1)
            misses = positive & ~flag
            pair_counts = Counter(realized[misses])
            top_realized, top_n = pair_counts.most_common(1)[0] if pair_counts else ("", 0)
            by_phone.append({
                "manner": manner, "target_phone": p, "target_slots": int(subset.sum()),
                "gold_substitutions": int(positive.sum()),
                "gold_starred_deviations": int((positive & star).sum()),
                "ap": float(average_precision_score(y, d["s"][subset])),
                "tp_at_manner_risk_point": int((positive & flag).sum()),
                "fp_at_manner_risk_point": int((subset & (d["y"] != 1) & flag).sum()),
                "missed_substitutions": int(misses.sum()),
                "most_common_missed_realized": top_realized,
                "most_common_missed_pair_count": top_n,
                "retrospective_manner_threshold": threshold,
            })
        for speaker in sorted(set(d["speaker"])):
            subset = group & (d["speaker"] == speaker)
            positive = subset & (d["y"] == 1)
            if not subset.any():
                continue
            by_speaker.append({
                "manner": manner, "speaker": speaker,
                "target_slots": int(subset.sum()), "gold_substitutions": int(positive.sum()),
                "gold_starred_deviations": int((positive & star).sum()),
                "tp_at_manner_risk_point": int((positive & flag).sum()),
                "fp_at_manner_risk_point": int((subset & (d["y"] != 1) & flag).sum()),
                "missed_substitutions": int((positive & ~flag).sum()),
                "retrospective_manner_threshold": threshold,
            })
    write_csv(HERE / "ERROR_REGION_BY_PHONE.csv", by_phone)
    write_csv(HERE / "ERROR_REGION_BY_SPEAKER.csv", by_speaker)
    for manner in ("Vowel", "Stop"):
        rows = sorted((r for r in by_phone if r["manner"] == manner),
                      key=lambda r: r["missed_substitutions"], reverse=True)
        print(manner, [(r["target_phone"], r["missed_substitutions"],
                        r["tp_at_manner_risk_point"], r["fp_at_manner_risk_point"],
                        r["gold_starred_deviations"]) for r in rows[:7]])


if __name__ == "__main__":
    main()
