#!/usr/bin/env python3
"""
LANGFENS — SM13-FUSION-DIAGNOSTIC-01
Comprehensive Forensic Diagnostic of Score Compression, Fusion Mechanics,
and Classwise Suppression in LANGFENS-SM13-FUSION-V1.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path("/home/khoa/Projects/New-Method-Research")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/rq1_encoder_bench_01/common"))

from dataset_util import DatasetManager, GROUPS
from research.evaluation_v3.evaluator import UnifiedEvaluatorV3

EXP_DIR = ROOT / "experiments/rq1_sm13_fusion_diagnostic_01"
EXP_DIR.mkdir(parents=True, exist_ok=True)

PRED_DIR = ROOT / "experiments/rq1_method_select_02r/agent_a/predictions"
DEV_CAL_NPZ = ROOT / "experiments/rq1_sm13_dev_inference_01/predictions_sm13_seed101_dev_cal.npz"
DEV_SEARCH_NPZ = ROOT / "experiments/rq1_sm13_dev_inference_01/predictions_sm13_seed101_dev_search.npz"

TIE_PREFIX = "LANGFENS-MB07-warning-tie-v1:"


def tie_hash(event_id: str) -> int:
    return int.from_bytes(hashlib.sha256((TIE_PREFIX + str(event_id)).encode()).digest()[:8], "big")


def compute_ap(gold: np.ndarray, scores: np.ndarray) -> float:
    if len(gold) == 0 or np.sum(gold) == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = gold[order]
    cum_tp = np.cumsum(y_sorted)
    cum_fp = np.cumsum(1 - y_sorted)
    precision = cum_tp / (cum_tp + cum_fp)
    recall_step = y_sorted / np.sum(gold)
    return float(np.sum(precision * recall_step))


def freeze_threshold(
    event_ids: np.ndarray,
    scores: np.ndarray,
    false_flags: np.ndarray,
    budget_per_utt: float,
    n_utts: int
) -> Tuple[float, int, int, int]:
    limit = int(math.floor(budget_per_utt * n_utts))
    n = len(event_ids)
    order = sorted(range(n), key=lambda i: (-float(scores[i]), -tie_hash(event_ids[i])))
    chosen = 0
    fp = 0
    for i in order:
        nf = fp + int(bool(false_flags[i]))
        if nf > limit:
            break
        chosen += 1
        fp = nf
    if chosen:
        last = order[chosen - 1]
        threshold = float(scores[last])
        cutoff = tie_hash(event_ids[last])
    else:
        threshold = 1.0
        cutoff = 2**64
    return threshold, cutoff, chosen, fp


def apply_threshold(
    event_ids: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    cutoff: int
) -> np.ndarray:
    out = np.zeros(len(scores), dtype=bool)
    for i, (eid, s) in enumerate(zip(event_ids, scores)):
        if s > threshold:
            out[i] = True
        elif s < threshold:
            out[i] = False
        else:
            out[i] = (tie_hash(eid) >= cutoff)
    return out


def compute_metrics_fast(
    yt: np.ndarray,
    dt: np.ndarray,
    yg: np.ndarray,
    dg: np.ndarray,
    n_utts: int,
    ap_s: float,
    ap_o: float,
    ap_i: float
) -> Dict[str, Any]:
    tp_s = int(np.sum((dt == 1) & (yt == 1)))
    fp_s = int(np.sum((dt == 1) & (yt != 1)))
    fn_s = int(np.sum((dt != 1) & (yt == 1)))

    tp_o = int(np.sum((dt == 2) & (yt == 2)))
    fp_o = int(np.sum((dt == 2) & (yt != 2)))
    fn_o = int(np.sum((dt != 2) & (yt == 2)))

    tp_i = int(np.sum((dg == True) & (yg == 1)))
    fp_i = int(np.sum((dg == True) & (yg == 0)))
    fn_i = int(np.sum((dg == False) & (yg == 1)))

    p_s = tp_s / (tp_s + fp_s) if tp_s + fp_s else 0.0
    r_s = tp_s / (tp_s + fn_s) if tp_s + fn_s else 0.0
    f1_s = 2 * p_s * r_s / (p_s + r_s) if p_s + r_s else 0.0

    p_o = tp_o / (tp_o + fp_o) if tp_o + fp_o else 0.0
    r_o = tp_o / (tp_o + fn_o) if tp_o + fn_o else 0.0
    f1_o = 2 * p_o * r_o / (p_o + r_o) if p_o + r_o else 0.0

    p_i = tp_i / (tp_i + fp_i) if tp_i + fp_i else 0.0
    r_i = tp_i / (tp_i + fn_i) if tp_i + fn_i else 0.0
    f1_i = 2 * p_i * r_i / (p_i + r_i) if p_i + r_i else 0.0

    m3_f1 = (f1_s + f1_o + f1_i) / 3.0
    m3_ap = (ap_s + ap_o + ap_i) / 3.0

    gold_err = (yt != 0)
    pred_err = (dt == 1) | (dt == 2)
    false_alarms_correct = int(np.sum((yt == 0) & pred_err))
    wrong_type_warnings = int(np.sum(gold_err & pred_err & (yt != dt)))
    total_false_warnings = fp_s + fp_o + fp_i
    joint_fwr = total_false_warnings / n_utts if n_utts else 0.0

    return {
        "macro3_error_f1": m3_f1,
        "macro3_error_ap": m3_ap,
        "ap_substitution": ap_s,
        "ap_omission": ap_o,
        "ap_insertion": ap_i,
        "joint_fwr": joint_fwr,
        "false_warning_count": total_false_warnings,
        "false_alarms_gold_correct": false_alarms_correct,
        "wrong_type_warnings": wrong_type_warnings,
        "sub_tp": tp_s,
        "sub_fp": fp_s,
        "sub_fn": fn_s,
        "sub_precision": p_s,
        "sub_recall": r_s,
        "sub_f1": f1_s,
        "omi_tp": tp_o,
        "omi_fp": fp_o,
        "omi_fn": fn_o,
        "omi_precision": p_o,
        "omi_recall": r_o,
        "omi_f1": f1_o,
        "ins_tp": tp_i,
        "ins_fp": fp_i,
        "ins_fn": fn_i,
        "ins_precision": p_i,
        "ins_recall": r_i,
        "ins_f1": f1_i,
    }


def compute_distribution_stats(arr: np.ndarray) -> Dict[str, float]:
    if len(arr) == 0:
        return {
            "count": 0, "mean": 0.0, "std": 0.0, "min": 0.0,
            "p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0,
            "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0
        }
    p = np.percentile(arr, [10, 25, 50, 75, 90, 95, 99])
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p10": float(p[0]),
        "p25": float(p[1]),
        "p50": float(p[2]),
        "p75": float(p[3]),
        "p90": float(p[4]),
        "p95": float(p[5]),
        "p99": float(p[6]),
        "max": float(np.max(arr))
    }


def main():
    print("================================================================================")
    print("LANGFENS SM13-FUSION-DIAGNOSTIC-01: Component Comparison & Root Cause Diagnostic")
    print("================================================================================")

    # 1. Load datasets
    datasets = {}
    for s in [101, 202, 303]:
        datasets[f"TRAIN_OOF_S{s}"] = np.load(PRED_DIR / f"predictions_sm13_seed{s}.npz")
    datasets["DEV_CAL"] = np.load(DEV_CAL_NPZ)
    datasets["DEV_SEARCH"] = np.load(DEV_SEARCH_NPZ)

    component_records = []
    distribution_records = []
    correlation_records = []

    configs = ["NEURAL_ONLY", "E4_ONLY", "FIXED_50_50_FUSION"]

    # 2. Detailed Distribution Analysis across Splits and Gold Classes
    for ds_name, d in datasets.items():
        yt = d["gold_target"]
        p_neu = d["probs_target_neural"]
        p_e4 = d["e4_probs"]
        p_fix = d["probs_target_fixed"]
        sllr = d["sllr"]
        ollr = d["ollr"]

        for class_label, class_idx in [("ALL", None), ("GOLD_CORRECT", 0), ("GOLD_SUBSTITUTION", 1), ("GOLD_OMISSION", 2)]:
            mask = np.ones(len(yt), dtype=bool) if class_idx is None else (yt == class_idx)

            # Record stats for S and O across the three configurations
            for cfg_name, p_arr in [("NEURAL_ONLY", p_neu), ("E4_ONLY", p_e4), ("FIXED_50_50_FUSION", p_fix)]:
                stats_s = compute_distribution_stats(p_arr[mask, 1])
                stats_o = compute_distribution_stats(p_arr[mask, 2])

                distribution_records.append({
                    "dataset": ds_name,
                    "target_class": class_label,
                    "scoring_configuration": cfg_name,
                    "score_type": "PROB_SUBSTITUTION",
                    **stats_s
                })
                distribution_records.append({
                    "dataset": ds_name,
                    "target_class": class_label,
                    "scoring_configuration": cfg_name,
                    "score_type": "PROB_OMISSION",
                    **stats_o
                })

            # Record raw LLR stats
            stats_sllr = compute_distribution_stats(sllr[mask])
            stats_ollr = compute_distribution_stats(ollr[mask])
            distribution_records.append({
                "dataset": ds_name,
                "target_class": class_label,
                "scoring_configuration": "RAW_CTC_LLR",
                "score_type": "SLLR",
                **stats_sllr
            })
            distribution_records.append({
                "dataset": ds_name,
                "target_class": class_label,
                "scoring_configuration": "RAW_CTC_LLR",
                "score_type": "OLLR",
                **stats_ollr
            })

        # Correlations between Neural and E4
        for class_label, class_idx in [("ALL", None), ("GOLD_CORRECT", 0), ("GOLD_SUBSTITUTION", 1), ("GOLD_OMISSION", 2)]:
            mask = np.ones(len(yt), dtype=bool) if class_idx is None else (yt == class_idx)
            if np.sum(mask) > 1:
                p_r_s, _ = pearsonr(p_neu[mask, 1], p_e4[mask, 1])
                s_r_s, _ = spearmanr(p_neu[mask, 1], p_e4[mask, 1])
                p_r_o, _ = pearsonr(p_neu[mask, 2], p_e4[mask, 2])
                s_r_o, _ = spearmanr(p_neu[mask, 2], p_e4[mask, 2])

                correlation_records.append({
                    "dataset": ds_name,
                    "target_class": class_label,
                    "score_type": "SUBSTITUTION",
                    "pearson_r": float(p_r_s),
                    "spearman_rho": float(s_r_s),
                    "count": int(np.sum(mask))
                })
                correlation_records.append({
                    "dataset": ds_name,
                    "target_class": class_label,
                    "score_type": "OMISSION",
                    "pearson_r": float(p_r_o),
                    "spearman_rho": float(s_r_o),
                    "count": int(np.sum(mask))
                })

    # 3. Warning Performance Evaluation with Provenance-Checked Inner OOF Thresholds
    # Evaluate across seeds 101, 202, 303 for TRAIN OOF
    train_frozen_th = {}
    for s in [101, 202, 303]:
        d = datasets[f"TRAIN_OOF_S{s}"]
        tkeys = d["target_keys"]
        gkeys = d["gap_keys"]
        speakers_t = d["target_speakers"]
        speakers_g = d["gap_speakers"]
        yt = d["gold_target"]
        yg = d["gold_gap"]
        pg = d["probs_gap"]

        train_frozen_th[s] = {}

        for cfg in configs:
            pt = d["probs_target_neural"] if cfg == "NEURAL_ONLY" else (
                d["e4_probs"] if cfg == "E4_ONLY" else d["probs_target_fixed"]
            )

            ap_s = compute_ap((yt == 1).astype(int), pt[:, 1])
            ap_o = compute_ap((yt == 2).astype(int), pt[:, 2])
            ap_i = compute_ap(yg.astype(int), pg)

            # Outer 4-fold CV
            all_pred_t = np.zeros(len(yt), dtype=np.int64)
            all_pred_g = np.zeros(len(yg), dtype=bool)
            fold_ths = []

            for f_idx, grp in enumerate(GROUPS):
                held_m_t = np.isin(speakers_t, grp)
                held_m_g = np.isin(speakers_g, grp)
                fit_m_t = ~held_m_t
                fit_m_g = ~held_m_g

                # Select on inner 9 speakers
                conf_fit = pt[fit_m_t, 1:].max(axis=1)
                kind_fit = pt[fit_m_t, 1:].argmax(axis=1) + 1
                false_t_fit = (kind_fit != yt[fit_m_t])
                false_g_fit = (yg[fit_m_g] == 0)

                th_t, cut_t, _, _ = freeze_threshold(tkeys[fit_m_t], conf_fit, false_t_fit, 0.16, 1350)
                th_g, cut_g, _, _ = freeze_threshold(gkeys[fit_m_g], pg[fit_m_g], false_g_fit, 0.04, 1350)

                # Apply to held-out outer fold
                conf_held = pt[held_m_t, 1:].max(axis=1)
                kind_held = pt[held_m_t, 1:].argmax(axis=1) + 1
                warn_t = (conf_held > th_t) | ((conf_held == th_t) & (np.array([tie_hash(k) for k in tkeys[held_m_t]]) >= cut_t))
                warn_g = apply_threshold(gkeys[held_m_g], pg[held_m_g], th_g, cut_g)

                all_pred_t[held_m_t] = np.where(warn_t, kind_held, 0)
                all_pred_g[held_m_g] = warn_g

                fold_ths.append((th_t, th_g))

                # Per-fold recording
                ap_s_f = compute_ap((yt[held_m_t] == 1).astype(int), pt[held_m_t, 1])
                ap_o_f = compute_ap((yt[held_m_t] == 2).astype(int), pt[held_m_t, 2])
                ap_i_f = compute_ap(yg[held_m_g].astype(int), pg[held_m_g])
                m_fold = compute_metrics_fast(yt[held_m_t], all_pred_t[held_m_t], yg[held_m_g], all_pred_g[held_m_g], 450, ap_s_f, ap_o_f, ap_i_f)

                component_records.append({
                    "seed": s,
                    "dataset": "TRAIN_OOF",
                    "evaluation_scope": f"FOLD_{f_idx}",
                    "scoring_configuration": cfg,
                    "policy_type": "SHARED_TARGET_NOMINAL_0.2",
                    "nominal_budget": 0.20,
                    "tau_target": th_t,
                    "tau_gap": th_g,
                    **m_fold
                })

            # Pooled recording
            m_pooled = compute_metrics_fast(yt, all_pred_t, yg, all_pred_g, 1800, ap_s, ap_o, ap_i)
            mean_th_t = float(np.mean([x[0] for x in fold_ths]))
            mean_th_g = float(np.mean([x[1] for x in fold_ths]))

            component_records.append({
                "seed": s,
                "dataset": "TRAIN_OOF",
                "evaluation_scope": "TRAIN_OOF_POOLED",
                "scoring_configuration": cfg,
                "policy_type": "SHARED_TARGET_NOMINAL_0.2",
                "nominal_budget": 0.20,
                "tau_target": mean_th_t,
                "tau_gap": mean_th_g,
                **m_pooled
            })

            # Per-speaker recording (TRAIN OOF)
            all_spk = sorted(list(set(str(x) for x in speakers_t)))
            for spk in all_spk:
                spk_m_t = (speakers_t == spk)
                spk_m_g = (speakers_g == spk)
                ap_s_spk = compute_ap((yt[spk_m_t] == 1).astype(int), pt[spk_m_t, 1])
                ap_o_spk = compute_ap((yt[spk_m_t] == 2).astype(int), pt[spk_m_t, 2])
                ap_i_spk = compute_ap(yg[spk_m_g].astype(int), pg[spk_m_g])
                m_spk = compute_metrics_fast(yt[spk_m_t], all_pred_t[spk_m_t], yg[spk_m_g], all_pred_g[spk_m_g], 150, ap_s_spk, ap_o_spk, ap_i_spk)

                component_records.append({
                    "seed": s,
                    "dataset": "TRAIN_OOF",
                    "evaluation_scope": f"SPEAKER_{spk}",
                    "scoring_configuration": cfg,
                    "policy_type": "SHARED_TARGET_NOMINAL_0.2",
                    "nominal_budget": 0.20,
                    "tau_target": mean_th_t,
                    "tau_gap": mean_th_g,
                    **m_spk
                })

            # Full TRAIN OOF freeze for prospective DEV transfer (Seed 101)
            conf_full = pt[:, 1:].max(axis=1)
            kind_full = pt[:, 1:].argmax(axis=1) + 1
            false_t_full = (kind_full != yt)
            false_g_full = (yg == 0)
            th_t_full, cut_t_full, _, _ = freeze_threshold(tkeys, conf_full, false_t_full, 0.16, 1800)
            th_g_full, cut_g_full, _, _ = freeze_threshold(gkeys, pg, false_g_full, 0.04, 1800)
            train_frozen_th[s][cfg] = (th_t_full, cut_t_full, th_g_full, cut_g_full)

    # 4. Prospective Diagnostic Evaluation on DEV_CAL and DEV_SEARCH (Seed 101 thresholds)
    for dev_name in ["DEV_CAL", "DEV_SEARCH"]:
        d_dev = datasets[dev_name]
        dev_tkeys = d_dev["target_keys"]
        dev_gkeys = d_dev["gap_keys"]
        dev_speakers_t = d_dev["target_speakers"]
        dev_speakers_g = d_dev["gap_speakers"]
        dev_yt = d_dev["gold_target"]
        dev_yg = d_dev["gold_gap"]
        dev_pg = d_dev["probs_gap"]
        n_dev_utts = 300 if dev_name == "DEV_CAL" else 599

        for cfg in configs:
            pt_dev = d_dev["probs_target_neural"] if cfg == "NEURAL_ONLY" else (
                d_dev["e4_probs"] if cfg == "E4_ONLY" else d_dev["probs_target_fixed"]
            )

            ap_s_dev = compute_ap((dev_yt == 1).astype(int), pt_dev[:, 1])
            ap_o_dev = compute_ap((dev_yt == 2).astype(int), pt_dev[:, 2])
            ap_i_dev = compute_ap(dev_yg.astype(int), dev_pg)

            # Apply Seed 101 thresholds frozen on Full TRAIN OOF
            th_t, cut_t, th_g, cut_g = train_frozen_th[101][cfg]
            conf_dev = pt_dev[:, 1:].max(axis=1)
            kind_dev = pt_dev[:, 1:].argmax(axis=1) + 1

            warn_t_dev = (conf_dev > th_t) | ((conf_dev == th_t) & (np.array([tie_hash(k) for k in dev_tkeys]) >= cut_t))
            warn_g_dev = apply_threshold(dev_gkeys, dev_pg, th_g, cut_g)

            dec_t_dev = np.where(warn_t_dev, kind_dev, 0)
            m_dev = compute_metrics_fast(dev_yt, dec_t_dev, dev_yg, warn_g_dev, n_dev_utts, ap_s_dev, ap_o_dev, ap_i_dev)

            component_records.append({
                "seed": 101,
                "dataset": dev_name,
                "evaluation_scope": f"{dev_name}_POOLED",
                "scoring_configuration": cfg,
                "policy_type": "FIXED_TRAIN_OOF_TRANSFER",
                "nominal_budget": 0.20,
                "tau_target": th_t,
                "tau_gap": th_g,
                **m_dev
            })

            # Per-speaker evaluation on DEV
            dev_spks = sorted(list(set(str(x) for x in dev_speakers_t)))
            for spk in dev_spks:
                spk_m_t = (dev_speakers_t == spk)
                spk_m_g = (dev_speakers_g == spk)
                n_spk_u = 150  # 150 utts per speaker
                ap_s_spk = compute_ap((dev_yt[spk_m_t] == 1).astype(int), pt_dev[spk_m_t, 1])
                ap_o_spk = compute_ap((dev_yt[spk_m_t] == 2).astype(int), pt_dev[spk_m_t, 2])
                ap_i_spk = compute_ap(dev_yg[spk_m_g].astype(int), dev_pg[spk_m_g])
                m_spk = compute_metrics_fast(dev_yt[spk_m_t], dec_t_dev[spk_m_t], dev_yg[spk_m_g], warn_g_dev[spk_m_g], n_spk_u, ap_s_spk, ap_o_spk, ap_i_spk)

                component_records.append({
                    "seed": 101,
                    "dataset": dev_name,
                    "evaluation_scope": f"SPEAKER_{spk}",
                    "scoring_configuration": cfg,
                    "policy_type": "FIXED_TRAIN_OOF_TRANSFER",
                    "nominal_budget": 0.20,
                    "tau_target": th_t,
                    "tau_gap": th_g,
                    **m_spk
                })

    # Convert to DataFrames and export
    df_components = pd.DataFrame(component_records)
    df_distributions = pd.DataFrame(distribution_records)
    df_correlations = pd.DataFrame(correlation_records)

    comp_csv_path = EXP_DIR / "COMPONENT_COMPARISON.csv"
    dist_csv_path = EXP_DIR / "SCORE_DISTRIBUTIONS.csv"
    corr_csv_path = EXP_DIR / "SCORE_CORRELATIONS.csv"

    df_components.to_csv(comp_csv_path, index=False)
    df_distributions.to_csv(dist_csv_path, index=False)
    df_correlations.to_csv(corr_csv_path, index=False)

    print(f"\nSaved {len(df_components)} component comparison rows to: {comp_csv_path.name}")
    print(f"Saved {len(df_distributions)} distribution stat rows to: {dist_csv_path.name}")
    print(f"Saved {len(df_correlations)} correlation rows to: {corr_csv_path.name}")

    return df_components, df_distributions, df_correlations


if __name__ == "__main__":
    main()
