#!/usr/bin/env python3
"""Inner-speaker threshold selection and outer-speaker transfer for 51D specialist."""

from __future__ import annotations

import bisect
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import run_audit as audit
from run_temporal_probe import GROUPS, model


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments/rq1_encoder_bench_01/common"))
from dataset_util import DatasetManager  # noqa: E402


def admissible(rows: list[dict], cap: int) -> list[dict]:
    return [r for r in rows if r["delta_fp"] <= cap and
            (r["tp"] == 0 or (r["precision"] is not None and r["precision"] >= 0.5))]


def choose_joint(data: dict, speakers: set[str], utterances: int) -> dict:
    cap = math.floor(0.08 * utterances + 1e-9)
    vowels = admissible(audit.candidates(data, speakers, "Vowel"), cap)
    stops = admissible(audit.candidates(data, speakers, "Stop"), cap)
    if not vowels or not stops:
        raise ValueError("no admissible no-warning point")
    by_cost = {}
    for row in stops:
        cost = row["delta_fp"]
        key = (row["tp"] / row["support"], row["tp"], -row["fp"])
        if cost not in by_cost or key > by_cost[cost][0]:
            by_cost[cost] = (key, row)
    costs = sorted(by_cost)
    best_stop = []
    chosen = None
    for cost in costs:
        entry = by_cost[cost]
        if chosen is None or entry[0] > chosen[0]:
            chosen = entry
        best_stop.append(chosen[1])
    best = None
    for vowel in vowels:
        remaining = cap - vowel["delta_fp"]
        index = bisect.bisect_right(costs, remaining) - 1
        if index < 0:
            continue
        stop = best_stop[index]
        key = (min(vowel["recall"], stop["recall"]),
               vowel["tp"] + stop["tp"],
               -(vowel["delta_fp"] + stop["delta_fp"]))
        if best is None or key > best[0]:
            best = (key, vowel, stop)
    if best is None:
        raise ValueError("no feasible joint threshold pair")
    return {"cap_extra_fp": cap, "vowel": best[1], "stop": best[2],
            "joint_extra_fp": best[1]["delta_fp"] + best[2]["delta_fp"]}


def evaluate(data: dict, speakers: set[str], score: np.ndarray,
             tau_v: float, tau_s: float) -> dict:
    target = np.isin(data["speaker"], list(speakers))
    gap = np.isin(data["gap_speaker"], list(speakers))
    tau = np.full(len(score), audit.TAU_S, dtype=np.float64)
    tau[data["manner"] == "Vowel"] = tau_v
    tau[data["manner"] == "Stop"] = tau_s
    trigger_s = target & (score >= tau)
    trigger_o = target & (data["o"] >= audit.TAU_O)
    flag_s = trigger_s & (~trigger_o | (score - tau > data["o"] - audit.TAU_O))
    flag_o = trigger_o & ~flag_s
    flag_i = gap & (data["gap_s"] >= audit.TAU_I)
    fp_s = int(np.sum(flag_s & (data["y"] != 1)))
    fp_o = int(np.sum(flag_o & (data["y"] != 2)))
    fp_i = int(np.sum(flag_i & (data["gap_y"] == 0)))
    n_utt = len({r["utterance"] for r in data["rows"] if r["speaker"] in speakers})
    out = {"utterances": n_utt, "total_fp": fp_s + fp_o + fp_i,
           "fwr": (fp_s + fp_o + fp_i) / n_utt,
           "fp_s": fp_s, "fp_o": fp_o, "fp_i": fp_i}
    for manner in ("Vowel", "Stop"):
        mask = target & (data["manner"] == manner)
        tp = int(np.sum(mask & flag_s & (data["y"] == 1)))
        fp = int(np.sum(mask & flag_s & (data["y"] != 1)))
        support = int(np.sum(mask & (data["y"] == 1)))
        out[manner] = {"tp": tp, "fp": fp, "support": support,
                       "recall": tp / support if support else None,
                       "precision": tp / (tp + fp) if tp + fp else None}
    return out


def main() -> None:
    t0 = time.perf_counter()
    dm = DatasetManager()
    data = audit.load_data()
    with np.load(ROOT / "data/derived/l2arctic/features/train_features.npz") as f:
        X = f["target_X"].astype(np.float32)
        if not np.array_equal(f["target_Y"], data["y"]):
            raise ValueError("static feature labels mismatch")
    with np.load(HERE / "TEMPORAL_PROBE_OOF.npz") as p:
        if not np.array_equal(p["target_keys"], dm.target_keys):
            raise ValueError("outer OOF event order mismatch")
        outer_score = p["static"].copy()
    baseline = evaluate(data, set(dm.target_speaker), data["s"], audit.TAU_S, audit.TAU_S)
    if baseline["total_fp"] != 216 or abs(baseline["fwr"] - 0.12) > 1e-10:
        raise ValueError(f"frozen-policy accounting mismatch: {baseline}")
    folds = []
    inner_scores = np.full((4, len(X)), np.nan, dtype=np.float32)
    for outer_fold, held_group in enumerate(GROUPS):
        outer = dm.outer_masks(outer_fold)
        train_speakers = set(outer["fit_speakers"])
        for manner in ("Vowel", "Stop"):
            for inner_fold in range(3):
                masks = dm.inner_masks(outer_fold, inner_fold)
                fit = np.flatnonzero(masks["t_fit"] & (data["manner"] == manner))
                held = np.flatnonzero(masks["t_held"] & (data["manner"] == manner))
                if set(masks["fit_speakers"]) & set(held_group):
                    raise ValueError("outer-held speaker leaked into inner fit")
                y = (data["y"][fit] == 1).astype(int)
                pos = int(y.sum())
                neg = len(y) - pos
                if pos == 0 or neg == 0:
                    raise ValueError("inner class empty")
                weight = np.where(y == 1, np.sqrt(neg / pos), 1.0).astype(np.float32)
                clf = model(101 + outer_fold)
                with threadpool_limits(limits=4):
                    clf.fit(X[fit], y, sample_weight=weight)
                    inner_scores[outer_fold, held] = clf.predict_proba(X[held])[:, 1]
        inner_mask = np.isin(data["speaker"], list(train_speakers)) & np.isin(data["manner"], ("Vowel", "Stop"))
        if not np.isfinite(inner_scores[outer_fold, inner_mask]).all():
            raise ValueError("inner OOF coverage incomplete")
        inner_data = dict(data)
        inner_data["s"] = data["s"].copy()
        inner_data["s"][inner_mask] = inner_scores[outer_fold, inner_mask]
        n_inner_utterances = len({r["utterance"] for r in data["rows"] if r["speaker"] in train_speakers})
        chosen = choose_joint(inner_data, train_speakers, n_inner_utterances)
        observed = evaluate(data, set(held_group), outer_score,
                            chosen["vowel"]["threshold"], chosen["stop"]["threshold"])
        fold = {"outer_fold": outer_fold, "held_speakers": list(held_group),
                "inner_train_speakers": sorted(train_speakers),
                "inner_utterances": n_inner_utterances, "selected": chosen,
                "outer": observed}
        folds.append(fold)
        print(f"fold={outer_fold} tauV={chosen['vowel']['threshold']:.4f} "
              f"tauS={chosen['stop']['threshold']:.4f} V={observed['Vowel']['tp']}/"
              f"{observed['Vowel']['fp']} S={observed['Stop']['tp']}/"
              f"{observed['Stop']['fp']} FWR={observed['fwr']:.4f}", flush=True)
    pooled = {"utterances": sum(f["outer"]["utterances"] for f in folds),
              "total_fp": sum(f["outer"]["total_fp"] for f in folds)}
    pooled["fwr"] = pooled["total_fp"] / pooled["utterances"]
    for manner in ("Vowel", "Stop"):
        tp = sum(f["outer"][manner]["tp"] for f in folds)
        fp = sum(f["outer"][manner]["fp"] for f in folds)
        support = sum(f["outer"][manner]["support"] for f in folds)
        pooled[manner] = {"tp": tp, "fp": fp, "support": support,
                          "recall": tp / support,
                          "precision": tp / (tp + fp) if tp + fp else None}
    np.savez_compressed(HERE / "NESTED_STATIC_INNER_OOF.npz", target_keys=dm.target_keys,
                        inner_scores=inner_scores)
    result = {"protocol": "STATIC_SPECIALIST_NESTED_PREREGISTRATION.md",
              "development_exposed": True, "baseline_frozen": baseline,
              "folds": folds, "pooled": pooled, "seconds": time.perf_counter() - t0,
              "caveat": "Candidate family selected after seeing prior TRAIN OOF. Inner-fitted threshold transfers to outer speakers without direct held-label use, but this is not a fresh independent confirmation."}
    (HERE / "NESTED_STATIC_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pooled": pooled, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
