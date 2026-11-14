#!/usr/bin/env python3
"""Read-only follow-up audit of SM13 TRAIN speaker-OOF scores.

Writes only into this experiment directory. TEST_SEALED is never opened.
The frontier is an empirical property of frozen scores, not a Bayes bound.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PRED = ROOT / "experiments/rq1_method_select_02r/agent_a/predictions/predictions_sm13_seed101.npz"
MANIFEST = ROOT / "data/derived/l2arctic/manifests/train.jsonl"
FEATURES = ROOT / "data/derived/l2arctic/features/train_features.npz"
ALIGNMENT = ROOT / "experiments/rq1_encoder_bench_01/features/alignment_centers.npz"
TAU_S = 0.9493744373321533
TAU_O = 0.8574540615081787
TAU_I = 0.8866584300994873
BUDGET = 0.20

VOWELS = set("AA AE AH AO AW AX AY EH EY IH IX IY OW OY UH UW".split())
STOPS = set("P B T D K G".split())


def manner(phone: str) -> str:
    if phone in VOWELS:
        return "Vowel"
    if phone in STOPS:
        return "Stop"
    return "Other"


def load_data() -> dict:
    rows = []
    with MANIFEST.open() as f, np.load(ALIGNMENT) as align:
        for line in f:
            rec = json.loads(line)
            assert rec["num_gaps"] == rec["num_target_phones"] + 1
            uid = rec["utterance_id"]
            centers = align[f"centers__{uid}"]
            nframes = int(align[f"nframes__{uid}"])
            assert len(centers) == rec["num_target_phones"] and nframes > 0
            audio_path = (ROOT / rec["audio_path"]).resolve()
            audio_duration = sf.info(audio_path).duration
            frame_seconds = audio_duration / nframes
            for tp in rec["target_phones"]:
                key = f'L2_ARCTIC|{rec["speaker_id"]}|{rec["utterance_id"]}|target|{tp["slot_idx"]}'
                rows.append({
                    "key": key,
                    "speaker": rec["speaker_id"],
                    "utterance": rec["utterance_id"],
                    "slot": tp["slot_idx"],
                    "phone": tp["target_phone"],
                    "realized": tp.get("realized_phone") or "",
                    "label": tp["label"],
                    "xmin": float(tp.get("xmin") or 0.0),
                    "xmax": float(tp.get("xmax") or 0.0),
                    "aligned_center_s": float((centers[tp["slot_idx"]] + 0.5) * frame_seconds),
                    "audio": str(audio_path),
                    "prompt": rec["prompt_text"],
                })
    with np.load(PRED) as p, np.load(FEATURES) as f:
        keys = p["target_keys"]
        assert len(rows) == len(keys) == len(f["target_X"])
        assert all(row["key"] == key for row, key in zip(rows, keys)), "manifest/prediction order mismatch"
        label_map = {"CORRECT": 0, "SUBSTITUTION": 1, "OMISSION": 2}
        assert all(label_map[row["label"]] == y for row, y in zip(rows, p["gold_target"]))
        assert len(p["gap_keys"]) == len(p["gold_gap"]) == len(p["probs_gap"])
        assert set(p["target_speakers"]) == set(p["gap_speakers"])
        return {
            "rows": rows,
            "speaker": np.asarray([r["speaker"] for r in rows]),
            "manner": np.asarray([manner(r["phone"]) for r in rows]),
            "y": p["gold_target"].copy(),
            "s": p["probs_target_neural"][:, 1].copy(),
            "o": p["probs_target_fixed"][:, 2].copy(),
            "gap_y": p["gold_gap"].copy(),
            "gap_s": p["probs_gap"].copy(),
            "gap_speaker": p["gap_speakers"].copy(),
            "duration": f["target_X"][:, 48].copy() * 0.01,
        }


def base_counts(d: dict, speakers: set[str], target_manner: str) -> dict:
    t = np.isin(d["speaker"], list(speakers))
    g = np.isin(d["gap_speaker"], list(speakers))
    other_s = t & (d["manner"] != target_manner) & (d["s"] >= TAU_S)
    omi = t & (d["o"] >= TAU_O)
    gap = g & (d["gap_s"] >= TAU_I)
    assert not np.any(other_s & omi), "frozen S/O conflict requires arbitration"
    other_s_fp = int(np.sum(other_s & (d["y"] != 1)))
    omi_fp = int(np.sum(omi & (d["y"] != 2)))
    gap_fp = int(np.sum(gap & (d["gap_y"] == 0)))
    return {"other_s_fp": other_s_fp, "omi_fp": omi_fp, "gap_fp": gap_fp,
            "total_other_fp": other_s_fp + omi_fp + gap_fp,
            "n_utterances": len({r["utterance"] for r in d["rows"] if r["speaker"] in speakers})}


def candidates(d: dict, speakers: set[str], target_manner: str) -> list[dict]:
    mask = np.isin(d["speaker"], list(speakers)) & (d["manner"] == target_manner)
    scores = d["s"][mask].astype(float)
    labels = d["y"][mask]
    omission = d["o"][mask] >= TAU_O
    # With score-margin arbitration, a candidate S warning displaces O only
    # after the S margin exceeds the O margin. This activation score makes the
    # whole per-manner empirical risk curve computable with one sort.
    activation = scores - np.where(omission, d["o"][mask].astype(float) - TAU_O, 0.0)
    support = int(np.sum(labels == 1))
    order = np.argsort(-activation, kind="stable")
    activation, labels, omission = activation[order], labels[order], omission[order]
    count = int(np.sum(activation >= 0.0))
    activation, labels, omission = activation[:count], labels[:count], omission[:count]
    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels != 1)
    displaced_omi_fp = np.cumsum(omission & (labels != 2))
    delta_fp = fp - displaced_omi_fp
    ends = np.r_[np.flatnonzero(np.diff(activation) != 0), len(activation) - 1] if len(activation) else []
    result = [{"threshold": 1.0, "tp": 0, "fp": 0, "delta_fp": 0, "support": support,
               "recall": 0.0, "precision": None}]
    for j in ends:
        a, b = int(tp[j]), int(fp[j])
        result.append({"threshold": float(np.nextafter(activation[j], -np.inf)),
                       "tp": a, "fp": b, "delta_fp": int(delta_fp[j]),
                       "support": support, "recall": a / support if support else None,
                       "precision": a / (a + b)})
    return result


def select(d: dict, speakers: set[str], target_manner: str, min_precision: float | None) -> dict:
    base = base_counts(d, speakers, target_manner)
    cap = math.floor(BUDGET * base["n_utterances"] + 1e-9)
    possible = []
    for c in candidates(d, speakers, target_manner):
        if base["total_other_fp"] + c["delta_fp"] > cap:
            continue
        if min_precision is not None and c["tp"] and c["precision"] < min_precision:
            continue
        possible.append(c)
    best = max(possible, key=lambda x: (x["tp"], -x["fp"], x["threshold"]))
    check = evaluate_threshold(d, speakers, target_manner, best["threshold"])
    assert (check["tp"], check["fp"], check["delta_fp"]) == (best["tp"], best["fp"], best["delta_fp"])
    return {**best, "base_other_fp": base["total_other_fp"],
            "total_fp": base["total_other_fp"] + best["delta_fp"],
            "fwr": (base["total_other_fp"] + best["delta_fp"]) / base["n_utterances"],
            "n_utterances": base["n_utterances"], "fp_cap": cap}


def evaluate_threshold(d: dict, speakers: set[str], target_manner: str, threshold: float) -> dict:
    base = base_counts(d, speakers, target_manner)
    mask = np.isin(d["speaker"], list(speakers)) & (d["manner"] == target_manner)
    trig_s = mask & (d["s"] >= threshold)
    trig_o = d["o"] >= TAU_O
    positive = trig_s & (~trig_o | ((d["s"].astype(float) - threshold) >
                                  (d["o"].astype(float) - TAU_O)))
    tp = int(np.sum(positive & (d["y"] == 1)))
    fp = int(np.sum(positive & (d["y"] != 1)))
    displaced_omi_fp = int(np.sum(positive & trig_o & (d["y"] != 2)))
    delta_fp = fp - displaced_omi_fp
    support = int(np.sum(mask & (d["y"] == 1)))
    return {"threshold": threshold, "tp": tp, "fp": fp, "support": support,
            "recall": tp / support if support else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "delta_fp": delta_fp, "base_other_fp": base["total_other_fp"],
            "total_fp": base["total_other_fp"] + delta_fp,
            "fwr": (base["total_other_fp"] + delta_fp) / base["n_utterances"],
            "n_utterances": base["n_utterances"]}


def crossfit(d: dict, target_manner: str, min_precision: float | None) -> dict:
    groups = d["speaker"]
    unique = np.unique(groups)
    fold_results = []
    for train_index, test_index in GroupKFold(n_splits=4).split(unique, groups=unique):
        train_speakers, test_speakers = set(unique[train_index]), set(unique[test_index])
        chosen = select(d, train_speakers, target_manner, min_precision)
        actual = evaluate_threshold(d, test_speakers, target_manner, chosen["threshold"])
        fold_results.append({"train_speakers": sorted(train_speakers),
                             "test_speakers": sorted(test_speakers),
                             "train": chosen, "test": actual})
    tp = sum(x["test"]["tp"] for x in fold_results)
    fp = sum(x["test"]["fp"] for x in fold_results)
    support = sum(x["test"]["support"] for x in fold_results)
    total_fp = sum(x["test"]["total_fp"] for x in fold_results)
    utt = sum(x["test"]["n_utterances"] for x in fold_results)
    return {"folds": fold_results, "pooled": {"tp": tp, "fp": fp,
            "support": support, "recall": tp / support,
            "precision": tp / (tp + fp) if tp + fp else None,
            "total_fp": total_fp, "fwr": total_fp / utt, "utterances": utt}}


def single_stream_optimistic(d: dict, target_manner: str, min_precision: float | None) -> dict:
    """Grant this manner every false alert; switch every other alert stream off."""
    mask = d["manner"] == target_manner
    scores = d["s"][mask]
    labels = d["y"][mask] == 1
    order = np.argsort(-scores, kind="stable")
    scores, labels = scores[order], labels[order]
    tp, fp = np.cumsum(labels), np.cumsum(~labels)
    ends = np.r_[np.flatnonzero(np.diff(scores) != 0), len(scores) - 1]
    candidates_idx = [j for j in ends if fp[j] <= 360 and
                      (min_precision is None or tp[j] / (tp[j] + fp[j]) >= min_precision)]
    if not candidates_idx:
        return {"tp": 0, "fp": 0, "recall": 0.0, "precision": None, "threshold": 1.0}
    j = max(candidates_idx, key=lambda i: (tp[i], -fp[i], scores[i]))
    return {"tp": int(tp[j]), "fp": int(fp[j]), "support": int(sum(labels)),
            "recall": float(tp[j] / sum(labels)),
            "precision": float(tp[j] / (tp[j] + fp[j])), "threshold": float(scores[j]),
            "fwr": float(fp[j] / 1800)}


def choose_diverse(indices: list[int], d: dict, count: int) -> list[int]:
    by_speaker = defaultdict(list)
    for i in indices:
        by_speaker[d["rows"][i]["speaker"]].append(i)
    out = []
    while len(out) < count and any(by_speaker.values()):
        for speaker in sorted(by_speaker):
            if by_speaker[speaker] and len(out) < count:
                out.append(by_speaker[speaker].pop(0))
    return out


def audit_labels(d: dict) -> dict:
    y = d["y"]
    same = np.asarray([r["phone"] == r["realized"] for r in d["rows"]]) & (y == 1)
    counts = Counter((d["rows"][i]["phone"], d["rows"][i]["realized"])
                     for i in np.flatnonzero(same))
    manner_counts = Counter(d["manner"][same])
    duration_gold = np.asarray([max(0.0, r["xmax"] - r["xmin"]) for r in d["rows"]])
    abs_diff = np.abs(d["duration"] - duration_gold)
    center_abs = np.asarray([abs(r["aligned_center_s"] - (r["xmin"] + r["xmax"]) / 2)
                             for r in d["rows"]])
    center_inside = np.asarray([r["xmin"] <= r["aligned_center_s"] <= r["xmax"]
                                for r in d["rows"]])
    per_manner = {}
    for m in ("Vowel", "Stop"):
        mask = (d["manner"] == m) & (y == 1)
        clean = (d["manner"] == m) & (y == 0)
        distinct = mask & ~same
        retained = clean | distinct
        per_manner[m] = {
            "gold_substitutions": int(np.sum(mask)),
            "same_phone_substitutions": int(np.sum(mask & same)),
            "median_duration_difference_ms": float(np.median(abs_diff[mask]) * 1000),
            "over_50ms_duration_difference": int(np.sum(mask & (abs_diff > 0.05))),
            "over_50ms_fraction": float(np.mean(abs_diff[mask] > 0.05)),
            "median_center_error_ms": float(np.median(center_abs[mask]) * 1000),
            "p90_center_error_ms": float(np.percentile(center_abs[mask], 90) * 1000),
            "center_outside_gold_interval_fraction": float(np.mean(~center_inside[mask])),
            "center_over_50ms_fraction": float(np.mean(center_abs[mask] > 0.05)),
            "center_over_100ms_fraction": float(np.mean(center_abs[mask] > 0.1)),
            "small_duration_error_large_center_error_count": int(np.sum(mask & (abs_diff <= 0.02) & (center_abs > 0.05))),
            "median_center_error_ms_same_phone": float(np.median(center_abs[mask & same]) * 1000) if np.any(mask & same) else None,
            "median_center_error_ms_distinct_phone": float(np.median(center_abs[distinct]) * 1000),
            "median_score_same_phone": float(np.median(d["s"][mask & same])) if np.any(mask & same) else None,
            "median_score_distinct_phone": float(np.median(d["s"][distinct])),
            "ap_all_labels": float(average_precision_score(mask[d["manner"] == m], d["s"][d["manner"] == m])),
            "ap_exclude_same_phone": float(average_precision_score(distinct[retained], d["s"][retained])),
            "auroc_all_labels": float(roc_auc_score(mask[d["manner"] == m], d["s"][d["manner"] == m])),
            "auroc_exclude_same_phone": float(roc_auc_score(distinct[retained], d["s"][retained])),
        }
    return {"same_phone_substitutions": int(np.sum(same)),
            "same_phone_by_manner": dict(manner_counts),
            "same_phone_top_pairs": [{"target": a, "realized": b, "count": n}
                                     for (a, b), n in counts.most_common(20)],
            "substitution_duration": per_manner,
            "alignment_center_available": True,
            "alignment_boundary_available": False,
            "note": "SM13 alignment_centers cache permits center-vs-manual interval checks. Predicted start/end were not cached, so boundary errors remain unmeasured."}


def review_queue(d: dict) -> list[dict]:
    cases = []
    for m, n in (("Vowel", 8), ("Stop", 24), ("Other", 8)):
        same = [i for i, r in enumerate(d["rows"]) if d["y"][i] == 1
                and d["manner"][i] == m and r["phone"] == r["realized"]]
        cases += [(f"same_phone_{m.lower()}", i) for i in choose_diverse(same, d, n)]
    for m in ("Vowel", "Stop"):
        idx = np.flatnonzero(d["manner"] == m)
        clean = sorted((i for i in idx if d["y"][i] == 0), key=lambda i: -float(d["s"][i]))
        subst = sorted((i for i in idx if d["y"][i] == 1), key=lambda i: float(d["s"][i]))
        cases += [(f"{m.lower()}_high_score_clean", i) for i in choose_diverse(clean, d, 18)]
        cases += [(f"{m.lower()}_low_score_substitution", i) for i in choose_diverse(subst, d, 18)]
        cases += [(f"{m.lower()}_high_score_substitution", i)
                  for i in choose_diverse(list(reversed(subst)), d, 12)]
    output = []
    for reason, i in cases:
        r = d["rows"][i]
        output.append({"reason": reason, "event_key": r["key"], "speaker": r["speaker"],
                       "utterance": r["utterance"], "slot": r["slot"],
                       "target_phone": r["phone"], "realized_phone": r["realized"],
                       "gold_label": r["label"], "neural_s_score": float(d["s"][i]),
                       "audio_path": r["audio"], "audio_exists": Path(r["audio"]).is_file(),
                       "crop_start_s": max(0.0, r["xmin"] - 0.1),
                       "crop_end_s": r["xmax"] + 0.1, "prompt": r["prompt"],
                       "aligned_center_s": r["aligned_center_s"],
                       "reviewer_1": "", "reviewer_2": "", "adjudication": ""})
    return output


def main() -> None:
    d = load_data()
    speakers = set(d["speaker"])
    assert len(speakers) == 12
    assert len(d["rows"]) == 59562
    assert len(d["gap_y"]) == 61362
    result = {"scope": "TRAIN_OOF only; no TEST_SEALED access",
              "method": "Frozen S/O/I scores; O/I and other manner S fixed; exact attainable target-manner S thresholds in [0,1] with score-margin arbitration; 4-fold GroupKFold policy crossfit by speaker",
              "n_speakers": len(speakers), "n_targets": len(d["y"]), "n_gaps": len(d["gap_y"]),
              "manner": {}, "label_audit": audit_labels(d)}
    for m in ("Vowel", "Stop"):
        result["manner"][m] = {
            "frozen": evaluate_threshold(d, speakers, m, TAU_S),
            "in_sample_risk_only": select(d, speakers, m, None),
            "in_sample_risk_and_50pct_precision": select(d, speakers, m, 0.50),
            "optimistic_all_budget_risk_only": single_stream_optimistic(d, m, None),
            "optimistic_all_budget_50pct_precision": single_stream_optimistic(d, m, 0.50),
            "crossfit_risk_only": crossfit(d, m, None),
            "crossfit_risk_and_50pct_precision": crossfit(d, m, 0.50),
        }
    (OUT / "AUDIT_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    queue = review_queue(d)
    with (OUT / "AUDIO_REVIEW_QUEUE.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(queue[0]))
        writer.writeheader()
        writer.writerows(queue)
    print(json.dumps({"manner": {m: {k: v for k, v in result["manner"][m].items()
                                         if k in ("frozen", "in_sample_risk_only", "in_sample_risk_and_50pct_precision", "optimistic_all_budget_risk_only", "optimistic_all_budget_50pct_precision")}
                                   for m in result["manner"]},
                      "crossfit": {m: {k: result["manner"][m][k]["pooled"]
                                      for k in ("crossfit_risk_only", "crossfit_risk_and_50pct_precision")}
                                   for m in result["manner"]},
                      "label_audit": result["label_audit"],
                      "review_queue_rows": len(queue),
                      "review_queue_audio_exists": sum(x["audio_exists"] for x in queue)}, indent=2))


if __name__ == "__main__":
    main()
