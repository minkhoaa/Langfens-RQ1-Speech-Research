#!/usr/bin/env python3
"""Aggregate exact held OOF predictions for the matched encoder-adapter study."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/khoa/Projects/New-Method-Research")
EXP = ROOT / "experiments/sm13_three_directions_20261113"
HERE = EXP / "encoder_adapter"
FOLDS = HERE / "folds"
BASELINE = EXP / "temporal/baseline_predictions"
sys.path.insert(0, str(ROOT))
from research.evaluation_v3.evaluator import UnifiedEvaluatorV3  # noqa: E402


def read_npz(kind: str, seed: int, fold: int):
    if kind == "lora":
        path = FOLDS / f"encoder_adapter_lora_seed{seed}_fold{fold}.npz"
    elif kind == "control":
        path = FOLDS / f"encoder_adapter_control_prefix_seed{seed}_fold{fold}.npz"
    elif kind == "baseline":
        path = BASELINE / f"sm13_scalar_mix_seed{seed}_fold{fold}.npz"
    else:
        raise ValueError(kind)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def eval_rows(data):
    tg, tp, tpr = data["target_gold"], data["target_pred"], data["target_probabilities"]
    gg, gp, gpr = data["gap_gold"], data["gap_pred"], data["gap_probability"]
    target_gt, target_pr, gap_gt, gap_pr = [], [], [], []
    for key, gold, pred, prob in zip(data["target_event_id"], tg, tp, tpr):
        ds, speaker, uid, _, pos = str(key).split("|")
        target_gt.append({"dataset_id": ds, "speaker_id": speaker, "utterance_id": uid,
                          "position": int(pos), "gold_label": str(gold)})
        target_pr.append({"dataset_id": ds, "speaker_id": speaker, "utterance_id": uid,
                          "position": int(pos), "pred_label": str(pred),
                          "prob_correct": float(prob[0]), "prob_substitution": float(prob[1]),
                          "prob_omission": float(prob[2])})
    for key, gold, pred, prob in zip(data["gap_event_id"], gg, gp, gpr):
        ds, speaker, uid, _, pos = str(key).split("|")
        gap_gt.append({"dataset_id": ds, "speaker_id": speaker, "utterance_id": uid,
                       "gap_position": int(pos), "gold_has_insertion": bool(gold)})
        gap_pr.append({"dataset_id": ds, "speaker_id": speaker, "utterance_id": uid,
                       "gap_position": int(pos), "pred_has_insertion": bool(pred),
                       "prob_insertion": float(prob)})
    return target_pr, gap_pr, target_gt, gap_gt


def load_oof(kind: str, seed: int):
    all_data = [read_npz(kind, seed, fold) for fold in range(4)]
    target_ids = np.concatenate([d["target_event_id"] for d in all_data]).astype(str)
    gap_ids = np.concatenate([d["gap_event_id"] for d in all_data]).astype(str)
    if len(set(target_ids.tolist())) != len(target_ids) or len(set(gap_ids.tolist())) != len(gap_ids):
        raise AssertionError(f"duplicate OOF event keys for {kind} seed={seed}")
    order_t = np.argsort(target_ids, kind="stable")
    order_g = np.argsort(gap_ids, kind="stable")
    merged = {
        "target_event_id": target_ids[order_t],
        "target_gold": np.concatenate([d["target_gold"] for d in all_data])[order_t],
        "target_pred": np.concatenate([d["target_pred"] for d in all_data])[order_t],
        "target_probabilities": np.concatenate([d["target_probabilities"] for d in all_data])[order_t],
        "gap_event_id": gap_ids[order_g],
        "gap_gold": np.concatenate([d["gap_gold"] for d in all_data])[order_g],
        "gap_pred": np.concatenate([d["gap_pred"] for d in all_data])[order_g],
        "gap_probability": np.concatenate([d["gap_probability"] for d in all_data])[order_g],
    }
    return merged


def save_oof_eval(kind, seed, data):
    tp, gp, gt, gg = eval_rows(data)
    result = UnifiedEvaluatorV3().evaluate(tp, gp, gt, gg)
    path = FOLDS / f"oof_seed{seed}_{kind}_metrics.json"
    # Event predictions are already preserved as compact keyed NPZ files; keep
    # the metric JSON focused on the evaluator summaries and classwise results.
    summary_only = {k: v for k, v in result.items()
                    if k not in ("target_predictions", "gap_predictions")}
    path.write_text(json.dumps(summary_only, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(FOLDS / f"oof_seed{seed}_{kind}.npz", **data)
    return result


def main():
    table = []
    for seed in (101, 202):
        loaded = {kind: load_oof(kind, seed) for kind in ("baseline", "control", "lora")}
        base_keys = set(loaded["baseline"]["target_event_id"].astype(str))
        base_gap_keys = set(loaded["baseline"]["gap_event_id"].astype(str))
        for kind in ("control", "lora"):
            if set(loaded[kind]["target_event_id"].astype(str)) != base_keys:
                raise AssertionError(f"target OOF keys differ from baseline for {kind} seed={seed}")
            if set(loaded[kind]["gap_event_id"].astype(str)) != base_gap_keys:
                raise AssertionError(f"gap OOF keys differ from baseline for {kind} seed={seed}")
            if not np.array_equal(loaded[kind]["gap_probability"], loaded["baseline"]["gap_probability"]):
                raise AssertionError(f"gap probabilities not exactly shared baseline for {kind} seed={seed}")
            if not np.array_equal(loaded[kind]["gap_pred"], loaded["baseline"]["gap_pred"]):
                raise AssertionError(f"gap decisions not exactly shared baseline for {kind} seed={seed}")
        results = {kind: save_oof_eval(kind, seed, loaded[kind])
                   for kind in ("baseline", "control", "lora")}
        for fold in range(4):
            pair_path = FOLDS / f"matched_pair_seed{seed}_fold{fold}.json"
            if not pair_path.exists():
                raise FileNotFoundError(pair_path)
            pair = json.loads(pair_path.read_text())
            c, l = pair["control_prefix"]["metrics"], pair["lora"]["metrics"]
            c_classes, l_classes = pair["control_prefix"]["classes"], pair["lora"]["classes"]
            row = {"seed": seed, "fold": fold}
            for m in ("macro3_error_ap", "macro3_error_f1", "joint_fwr", "ap_substitution",
                      "ap_omission", "ap_insertion_exact_gap"):
                row[f"control_{m}"] = c[m]; row[f"lora_{m}"] = l[m]
                row[f"delta_{m}"] = l[m] - c[m]
            for cls in ("SUBSTITUTION", "OMISSION", "INSERTION"):
                row[f"control_{cls.lower()}_precision"] = c_classes[cls]["precision"]
                row[f"control_{cls.lower()}_recall"] = c_classes[cls]["recall"]
                row[f"control_{cls.lower()}_f1"] = c_classes[cls]["f1"]
                row[f"control_{cls.lower()}_ap"] = c_classes[cls]["ap"]
                row[f"lora_{cls.lower()}_precision"] = l_classes[cls]["precision"]
                row[f"lora_{cls.lower()}_recall"] = l_classes[cls]["recall"]
                row[f"lora_{cls.lower()}_f1"] = l_classes[cls]["f1"]
                row[f"lora_{cls.lower()}_ap"] = l_classes[cls]["ap"]
                row[f"delta_{cls.lower()}_ap"] = l_classes[cls]["ap"] - c_classes[cls]["ap"]
            table.append(row)
        for kind, result in results.items():
            m = result["summary"]
            cls = result["classes"]
            oof_row = {"seed": seed, "model": kind,
                       "macro3_error_ap": m["macro3_error_ap"],
                       "macro3_error_f1": m["macro3_error_f1"],
                       "joint_fwr": m["joint_fwr"],
                       "ap_substitution": m["ap_substitution"],
                       "ap_omission": m["ap_omission"],
                       "ap_insertion_exact_gap": m["ap_insertion_exact_gap"],
                       "macro3_target_f1": m["macro3_target_f1"],
                       "macro4_mixed_domain_f1": m["macro4_mixed_domain_f1"]}
            for name in ("SUBSTITUTION", "OMISSION", "INSERTION"):
                n = name.lower()
                for metric in ("precision", "recall", "f1", "ap"):
                    oof_row[f"{n}_{metric}"] = cls[name][metric]
            for metric in ("binary_md_precision", "binary_md_recall", "binary_md_f1", "binary_md_ap"):
                oof_row[metric] = m[metric]
            # Store separate per-seed OOF rows as a JSON alongside per-class metrics.
            out_json = FOLDS / f"oof_seed{seed}_{kind}_metrics.json"
            out_json.with_suffix(".summary.json").write_text(json.dumps(oof_row, indent=2) + "\n")

    csv_path = FOLDS / "matched_foldseed_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader(); w.writerows(table)
    # Descriptive paired mean and range over the eight fold/seed pairs.
    paired = {}
    for metric in ("macro3_error_ap", "macro3_error_f1", "joint_fwr", "ap_substitution",
                   "ap_omission", "ap_insertion_exact_gap", "substitution_ap", "omission_ap", "insertion_ap"):
        vals = np.asarray([r[f"delta_{metric}"] for r in table], dtype=float)
        paired[metric] = {"mean_delta": float(vals.mean()), "median_delta": float(np.median(vals)),
                          "min_delta": float(vals.min()), "max_delta": float(vals.max()),
                          "n_fold_seed_pairs": int(len(vals))}
    (FOLDS / "paired_delta_summary.json").write_text(json.dumps(paired, indent=2) + "\n")
    print(json.dumps({"foldseed_csv": str(csv_path), "paired_deltas": paired}, indent=2))


if __name__ == "__main__": main()
