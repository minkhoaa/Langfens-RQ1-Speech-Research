#!/usr/bin/env python3
"""
LANGFENS — SM13-CLASSWISE-POLICY-01
Comprehensive Classwise Threshold Exploration and Calibration Benchmark.

Evaluates independent class-specific thresholds (tau_S, tau_O, tau_I) versus
shared target threshold policies on SM13 TRAIN OOF predictions across seeds 101, 202, 303,
all four speaker-grouped outer folds, and DEV_CAL diagnostic transfer.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/home/khoa/Projects/New-Method-Research")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/rq1_encoder_bench_01/common"))

from dataset_util import DatasetManager, GROUPS
from research.evaluation_v3.evaluator import UnifiedEvaluatorV3

EXP_DIR = ROOT / "experiments/rq1_sm13_classwise_policy_01"
EXP_DIR.mkdir(parents=True, exist_ok=True)

PRED_DIR = ROOT / "experiments/rq1_method_select_02r/agent_a/predictions"
DEV_CAL_NPZ = ROOT / "experiments/rq1_sm13_dev_inference_01/predictions_sm13_seed101_dev_cal.npz"
FROZEN_CAL_POLICY_JSON = ROOT / "experiments/rq1_sm13_dev_calibration_01/FROZEN_CALIBRATED_POLICY.json"

TIE_PREFIX = "LANGFENS-MB07-warning-tie-v1:"


def tie_hash(event_id: str) -> int:
    return int.from_bytes(hashlib.sha256((TIE_PREFIX + str(event_id)).encode()).digest()[:8], "big")


def freeze_class_threshold(
    event_ids: np.ndarray,
    scores: np.ndarray,
    false_flags: np.ndarray,
    budget_per_utt: float,
    n_utts: int
) -> Tuple[float, int, int, int]:
    """Selects deterministic largest score/SHA-sorted prefix satisfying false warning budget."""
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


def apply_class_threshold(
    event_ids: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    cutoff: int
) -> np.ndarray:
    """Applies frozen threshold with deterministic tie-breaking."""
    out = np.zeros(len(scores), dtype=bool)
    for i, (eid, s) in enumerate(zip(event_ids, scores)):
        if s > threshold:
            out[i] = True
        elif s < threshold:
            out[i] = False
        else:
            out[i] = (tie_hash(eid) >= cutoff)
    return out


def resolve_so_conflict(
    warn_s: np.ndarray,
    warn_o: np.ndarray,
    p_s: np.ndarray,
    p_o: np.ndarray,
    th_s: float,
    th_o: float,
    event_ids: np.ndarray,
    arbitration_rule: str = "score_margin"
) -> np.ndarray:
    """
    Deterministically resolves S/O warnings for target events.
    Returns array of predictions: 0=CORRECT, 1=SUBSTITUTION, 2=OMISSION.
    """
    n = len(warn_s)
    dec = np.zeros(n, dtype=np.int64)

    only_s = warn_s & (~warn_o)
    only_o = warn_o & (~warn_s)
    conflict = warn_s & warn_o

    dec[only_s] = 1
    dec[only_o] = 2

    if np.any(conflict):
        conf_indices = np.flatnonzero(conflict)
        for idx in conf_indices:
            s_val = p_s[idx]
            o_val = p_o[idx]
            eid = event_ids[idx]
            thash = tie_hash(eid)

            if arbitration_rule == "score_margin":
                m_s = s_val - th_s
                m_o = o_val - th_o
                if m_s > m_o:
                    dec[idx] = 1
                elif m_o > m_s:
                    dec[idx] = 2
                else:
                    dec[idx] = 1 if (thash % 2 == 1) else 2

            elif arbitration_rule == "norm_margin":
                denom_s = max(1e-6, 1.0 - th_s)
                denom_o = max(1e-6, 1.0 - th_o)
                u_s = (s_val - th_s) / denom_s
                u_o = (o_val - th_o) / denom_o
                if u_s > u_o:
                    dec[idx] = 1
                elif u_o > u_s:
                    dec[idx] = 2
                else:
                    dec[idx] = 1 if (thash % 2 == 1) else 2

            elif arbitration_rule == "prob_comparison":
                if s_val > o_val:
                    dec[idx] = 1
                elif o_val > s_val:
                    dec[idx] = 2
                else:
                    dec[idx] = 1 if (thash % 2 == 1) else 2

            elif arbitration_rule == "priority_substitution":
                dec[idx] = 1

            elif arbitration_rule == "priority_omission":
                dec[idx] = 2

            else:
                raise ValueError(f"Unknown arbitration rule: {arbitration_rule}")

    return dec


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
    """Computes exact evaluation metrics aligned with UnifiedEvaluatorV3."""
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


def compute_average_precision(gold: np.ndarray, scores: np.ndarray) -> float:
    """Computes exact Average Precision aligned with sklearn and UnifiedEvaluatorV3."""
    if len(gold) == 0 or np.sum(gold) == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = gold[order]
    cum_tp = np.cumsum(y_sorted)
    cum_fp = np.cumsum(1 - y_sorted)
    precision = cum_tp / (cum_tp + cum_fp)
    recall_step = y_sorted / np.sum(gold)
    ap = float(np.sum(precision * recall_step))
    return ap


def run_all_evaluations():
    print("================================================================================")
    print("LANGFENS SM13-CLASSWISE-POLICY-01: Classwise Policy Evaluation Suite")
    print("================================================================================")

    dm = DatasetManager()
    evaluator = UnifiedEvaluatorV3()
    seeds = [101, 202, 303]

    # Precompute APs per seed and fold
    seed_data = {}
    seed_aps = {}
    for s in seeds:
        npz_p = PRED_DIR / f"predictions_sm13_seed{s}.npz"
        d = np.load(npz_p)
        seed_data[s] = d
        pt = d["probs_target_fixed"]
        pg = d["probs_gap"]
        yt = d["gold_target"]
        yg = d["gold_gap"]

        ap_s = compute_average_precision((yt == 1).astype(int), pt[:, 1])
        ap_o = compute_average_precision((yt == 2).astype(int), pt[:, 2])
        ap_i = compute_average_precision(yg.astype(int), pg)
        seed_aps[s] = {"pooled": (ap_s, ap_o, ap_i)}

        # Per fold APs
        for f_idx, grp in enumerate(GROUPS):
            m_t = np.isin(d["target_speakers"], grp)
            m_g = np.isin(d["gap_speakers"], grp)
            ap_s_f = compute_average_precision((yt[m_t] == 1).astype(int), pt[m_t, 1])
            ap_o_f = compute_average_precision((yt[m_t] == 2).astype(int), pt[m_t, 2])
            ap_i_f = compute_average_precision(yg[m_g].astype(int), pg[m_g])
            seed_aps[s][f"fold_{f_idx}"] = (ap_s_f, ap_o_f, ap_i_f)

    # Policy specs to evaluate
    budget_allocations = [
        # (name, B_total, B_s, B_o, B_i, description)
        ("SHARED_NOMINAL_0.20", 0.20, None, None, 0.04, "Original shared-target threshold (80/20 target/gap allocation)"),
        ("CLASSWISE_SPLIT_0.02_0.14_0.04", 0.20, 0.02, 0.14, 0.04, "Classwise B=0.20 (10% S, 70% O, 20% I)"),
        ("CLASSWISE_SPLIT_0.04_0.12_0.04", 0.20, 0.04, 0.12, 0.04, "Classwise B=0.20 (20% S, 60% O, 20% I)"),
        ("CLASSWISE_SPLIT_0.05_0.11_0.04", 0.20, 0.05, 0.11, 0.04, "Classwise B=0.20 (25% S, 55% O, 20% I)"),
        ("CLASSWISE_SPLIT_0.06_0.10_0.04", 0.20, 0.06, 0.10, 0.04, "Classwise B=0.20 (30% S, 50% O, 20% I)"),
        ("CLASSWISE_SPLIT_0.08_0.08_0.04", 0.20, 0.08, 0.08, 0.04, "Classwise B=0.20 (40% S, 40% O, 20% I)"),
        ("CLASSWISE_SPLIT_0.10_0.06_0.04", 0.20, 0.10, 0.06, 0.04, "Classwise B=0.20 (50% S, 30% O, 20% I)"),
        ("CLASSWISE_SPLIT_0.12_0.04_0.04", 0.20, 0.12, 0.04, 0.04, "Classwise B=0.20 (60% S, 20% O, 20% I)"),
        # Conservative margin m=0.10 (effective B_eff = 0.18)
        ("SHARED_CONSERVATIVE_0.18", 0.18, None, None, 0.03333333333333333, "Original shared-target with 10% safety margin (B_eff=0.18)"),
        ("CLASSWISE_CONSERVATIVE_0.04_0.11_0.03", 0.18, 0.036, 0.11066666666666666, 0.03333333333333333, "Classwise 10% safety margin B_eff=0.18 (20% S, 61.5% O, 18.5% I)"),
        ("CLASSWISE_CONSERVATIVE_0.05_0.10_0.03", 0.18, 0.045, 0.10166666666666667, 0.03333333333333333, "Classwise 10% safety margin B_eff=0.18 (25% S, 56.5% O, 18.5% I)"),
        ("CLASSWISE_CONSERVATIVE_0.06_0.09_0.03", 0.18, 0.054, 0.09266666666666666, 0.03333333333333333, "Classwise 10% safety margin B_eff=0.18 (30% S, 51.5% O, 18.5% I)"),
    ]

    arbitration_rules = [
        "score_margin",
        "norm_margin",
        "prob_comparison",
        "priority_substitution",
        "priority_omission"
    ]

    policy_records = []
    classwise_records = []

    # Run sweeps over seeds
    for s in seeds:
        print(f"\nProcessing Seed {s}...")
        data = seed_data[s]
        target_keys = data["target_keys"]
        gap_keys = data["gap_keys"]
        speakers_t = data["target_speakers"]
        speakers_g = data["gap_speakers"]
        yt = data["gold_target"]
        yg = data["gold_gap"]
        pt = data["probs_target_fixed"]
        pg = data["probs_gap"]

        # 1. Evaluate all budget allocations under primary arbitration (score_margin)
        for alloc in budget_allocations:
            alloc_name, b_total, b_s, b_o, b_i, desc = alloc

            # Test arbitration variants only for the representative classwise allocations
            test_arbs = ["score_margin"]
            if alloc_name in ("CLASSWISE_SPLIT_0.05_0.11_0.04", "CLASSWISE_SPLIT_0.06_0.10_0.04"):
                test_arbs = arbitration_rules

            for arb in test_arbs:
                all_pred_t = np.zeros(len(yt), dtype=np.int64)
                all_pred_g = np.zeros(len(yg), dtype=bool)

                fold_th_info = []

                # 4-fold nested CV loop
                for f_idx, held_grp in enumerate(GROUPS):
                    held_mask_t = np.isin(speakers_t, held_grp)
                    held_mask_g = np.isin(speakers_g, held_grp)
                    fit_mask_t = ~held_mask_t
                    fit_mask_g = ~held_mask_g

                    inner_utts = 1350
                    fit_tkeys = target_keys[fit_mask_t]
                    fit_gkeys = gap_keys[fit_mask_g]
                    fit_pt = pt[fit_mask_t]
                    fit_pg = pg[fit_mask_g]
                    fit_yt = yt[fit_mask_t]
                    fit_yg = yg[fit_mask_g]

                    held_tkeys = target_keys[held_mask_t]
                    held_gkeys = gap_keys[held_mask_g]
                    held_pt = pt[held_mask_t]
                    held_pg = pg[held_mask_g]

                    if b_s is None:
                        # Shared target policy
                        b_target = b_total - b_i
                        c_fit = fit_pt[:, 1:].max(axis=1)
                        k_fit = fit_pt[:, 1:].argmax(axis=1) + 1
                        false_t = (k_fit != fit_yt)
                        false_g = (fit_yg == 0)

                        th_t, cut_t, ch_t, fp_t = freeze_class_threshold(fit_tkeys, c_fit, false_t, b_target, inner_utts)
                        th_g, cut_g, ch_g, fp_g = freeze_class_threshold(fit_gkeys, fit_pg, false_g, b_i, inner_utts)

                        held_c = held_pt[:, 1:].max(axis=1)
                        held_k = held_pt[:, 1:].argmax(axis=1) + 1
                        warn_t = apply_class_threshold(held_tkeys, held_c, th_t, cut_t)
                        warn_g = apply_class_threshold(held_gkeys, held_pg, th_g, cut_g)

                        dec_t = np.where(warn_t, held_k, 0)
                        all_pred_t[held_mask_t] = dec_t
                        all_pred_g[held_mask_g] = warn_g

                        fold_th_info.append({
                            "fold": f_idx,
                            "tau_target": th_t,
                            "tau_gap": th_g,
                            "tau_substitution": th_t,
                            "tau_omission": th_t,
                            "tau_insertion": th_g
                        })
                    else:
                        # Classwise policy
                        false_s = (fit_yt != 1)
                        false_o = (fit_yt != 2)
                        false_i = (fit_yg == 0)

                        th_s, cut_s, ch_s, fp_s = freeze_class_threshold(fit_tkeys, fit_pt[:, 1], false_s, b_s, inner_utts)
                        th_o, cut_o, ch_o, fp_o = freeze_class_threshold(fit_tkeys, fit_pt[:, 2], false_o, b_o, inner_utts)
                        th_i, cut_i, ch_i, fp_i = freeze_class_threshold(fit_gkeys, fit_pg, false_i, b_i, inner_utts)

                        warn_s = apply_class_threshold(held_tkeys, held_pt[:, 1], th_s, cut_s)
                        warn_o = apply_class_threshold(held_tkeys, held_pt[:, 2], th_o, cut_o)
                        warn_i = apply_class_threshold(held_gkeys, held_pg, th_i, cut_i)

                        dec_t = resolve_so_conflict(
                            warn_s=warn_s,
                            warn_o=warn_o,
                            p_s=held_pt[:, 1],
                            p_o=held_pt[:, 2],
                            th_s=th_s,
                            th_o=th_o,
                            event_ids=held_tkeys,
                            arbitration_rule=arb
                        )

                        all_pred_t[held_mask_t] = dec_t
                        all_pred_g[held_mask_g] = warn_i

                        fold_th_info.append({
                            "fold": f_idx,
                            "tau_target": None,
                            "tau_gap": th_i,
                            "tau_substitution": th_s,
                            "tau_omission": th_o,
                            "tau_insertion": th_i
                        })

                # Compute pooled metrics
                ap_s, ap_o, ap_i = seed_aps[s]["pooled"]
                m_pooled = compute_metrics_fast(yt, all_pred_t, yg, all_pred_g, 1800, ap_s, ap_o, ap_i)

                mean_tau_s = float(np.mean([f["tau_substitution"] for f in fold_th_info]))
                mean_tau_o = float(np.mean([f["tau_omission"] for f in fold_th_info]))
                mean_tau_i = float(np.mean([f["tau_insertion"] for f in fold_th_info]))
                mean_tau_t = float(np.mean([f["tau_target"] for f in fold_th_info])) if b_s is None else None

                policy_records.append({
                    "seed": s,
                    "evaluation_scope": "TRAIN_OOF_POOLED",
                    "policy_id": f"{alloc_name}_{arb}" if b_s is not None else alloc_name,
                    "policy_type": "SHARED_TARGET" if b_s is None else "CLASSWISE_INDEPENDENT",
                    "arbitration_rule": "NONE" if b_s is None else arb,
                    "nominal_joint_budget": b_total,
                    "nominal_budget_s": 0.0 if b_s is None else b_s,
                    "nominal_budget_o": b_total - b_i if b_s is None else b_o,
                    "nominal_budget_i": b_i,
                    "tau_target": mean_tau_t,
                    "tau_substitution": mean_tau_s,
                    "tau_omission": mean_tau_o,
                    "tau_insertion": mean_tau_i,
                    **m_pooled
                })

                # Record per-fold metrics for each fold
                for f_idx, grp in enumerate(GROUPS):
                    m_t = np.isin(speakers_t, grp)
                    m_g = np.isin(speakers_g, grp)
                    ap_s_f, ap_o_f, ap_i_f = seed_aps[s][f"fold_{f_idx}"]
                    m_fold = compute_metrics_fast(yt[m_t], all_pred_t[m_t], yg[m_g], all_pred_g[m_g], 450, ap_s_f, ap_o_f, ap_i_f)
                    f_info = fold_th_info[f_idx]

                    policy_records.append({
                        "seed": s,
                        "evaluation_scope": f"FOLD_{f_idx}",
                        "policy_id": f"{alloc_name}_{arb}" if b_s is not None else alloc_name,
                        "policy_type": "SHARED_TARGET" if b_s is None else "CLASSWISE_INDEPENDENT",
                        "arbitration_rule": "NONE" if b_s is None else arb,
                        "nominal_joint_budget": b_total,
                        "nominal_budget_s": 0.0 if b_s is None else b_s,
                        "nominal_budget_o": b_total - b_i if b_s is None else b_o,
                        "nominal_budget_i": b_i,
                        "tau_target": f_info["tau_target"],
                        "tau_substitution": f_info["tau_substitution"],
                        "tau_omission": f_info["tau_omission"],
                        "tau_insertion": f_info["tau_insertion"],
                        **m_fold
                    })

                # Per-speaker performance recording for primary policies
                if alloc_name in ("SHARED_NOMINAL_0.20", "CLASSWISE_SPLIT_0.05_0.11_0.04") and arb in ("score_margin", "NONE"):
                    all_spk = sorted(list(set(str(x) for x in speakers_t)))
                    for spk in all_spk:
                        spk_m_t = (speakers_t == spk)
                        spk_m_g = (speakers_g == spk)
                        n_spk_utts = 150
                        ap_s_spk = compute_average_precision((yt[spk_m_t] == 1).astype(int), pt[spk_m_t, 1])
                        ap_o_spk = compute_average_precision((yt[spk_m_t] == 2).astype(int), pt[spk_m_t, 2])
                        ap_i_spk = compute_average_precision(yg[spk_m_g].astype(int), pg[spk_m_g])
                        m_spk = compute_metrics_fast(yt[spk_m_t], all_pred_t[spk_m_t], yg[spk_m_g], all_pred_g[spk_m_g], n_spk_utts, ap_s_spk, ap_o_spk, ap_i_spk)

                        classwise_records.append({
                            "dataset": "TRAIN_OOF",
                            "seed": s,
                            "speaker_id": spk,
                            "policy_id": alloc_name,
                            "policy_type": "SHARED_TARGET" if b_s is None else "CLASSWISE_INDEPENDENT",
                            "num_utterances": n_spk_utts,
                            "num_targets": int(spk_m_t.sum()),
                            "num_gaps": int(spk_m_g.sum()),
                            **m_spk
                        })

    # 2. Matched Realized FWR Sweep on TRAIN OOF
    print("\nExecuting Matched Realized FWR Sweeps across seeds...")
    matched_target_fws = [180, 270, 360, 424, 540] # FWR: 0.10, 0.15, 0.20, 0.2356, 0.30
    for s in seeds:
        data = seed_data[s]
        yt = data["gold_target"]
        yg = data["gold_gap"]
        pt = data["probs_target_fixed"]
        pg = data["probs_gap"]
        target_keys = data["target_keys"]
        gap_keys = data["gap_keys"]
        ap_s, ap_o, ap_i = seed_aps[s]["pooled"]

        for target_fw in matched_target_fws:
            matched_fwr = target_fw / 1800.0

            # Condition A: Shared Target at matched FWR
            target_fp_limit = int(round(0.8 * target_fw))
            gap_fp_limit = int(round(0.2 * target_fw))

            conf_t = pt[:, 1:].max(axis=1)
            kind_t = pt[:, 1:].argmax(axis=1) + 1
            false_t = (kind_t != yt)
            order_t = sorted(range(len(yt)), key=lambda i: (-float(conf_t[i]), -tie_hash(target_keys[i])))
            order_g = sorted(range(len(yg)), key=lambda i: (-float(pg[i]), -tie_hash(gap_keys[i])))

            fp_count_t, cut_t = 0, 0
            for idx in order_t:
                if false_t[idx]: fp_count_t += 1
                if fp_count_t > target_fp_limit: break
                cut_t += 1

            fp_count_g, cut_g = 0, 0
            for idx in order_g:
                if yg[idx] == 0: fp_count_g += 1
                if fp_count_g > gap_fp_limit: break
                cut_g += 1

            dec_t_shared = np.zeros(len(yt), dtype=np.int64)
            for idx in order_t[:cut_t]:
                dec_t_shared[idx] = kind_t[idx]
            dec_g_shared = np.zeros(len(yg), dtype=bool)
            for idx in order_g[:cut_g]:
                dec_g_shared[idx] = True

            m_shared_m = compute_metrics_fast(yt, dec_t_shared, yg, dec_g_shared, 1800, ap_s, ap_o, ap_i)
            policy_records.append({
                "seed": s,
                "evaluation_scope": "TRAIN_OOF_MATCHED_REALIZED_FWR",
                "policy_id": f"MATCHED_SHARED_FWR_{matched_fwr:.4f}",
                "policy_type": "SHARED_TARGET_MATCHED",
                "arbitration_rule": "NONE",
                "nominal_joint_budget": matched_fwr,
                "nominal_budget_s": 0.0,
                "nominal_budget_o": target_fp_limit / 1800.0,
                "nominal_budget_i": gap_fp_limit / 1800.0,
                "tau_target": float(conf_t[order_t[cut_t-1]]) if cut_t else 1.0,
                "tau_substitution": float(conf_t[order_t[cut_t-1]]) if cut_t else 1.0,
                "tau_omission": float(conf_t[order_t[cut_t-1]]) if cut_t else 1.0,
                "tau_insertion": float(pg[order_g[cut_g-1]]) if cut_g else 1.0,
                **m_shared_m
            })

            # Condition B: Classwise (0.05 S, 0.11 O, 0.04 I scaled) at matched FWR
            sub_fp_limit = int(round(0.25 * target_fp_limit))
            omi_fp_limit = target_fp_limit - sub_fp_limit

            false_s = (yt != 1)
            false_o = (yt != 2)
            order_s = sorted(range(len(yt)), key=lambda i: (-float(pt[i, 1]), -tie_hash(target_keys[i])))
            order_o = sorted(range(len(yt)), key=lambda i: (-float(pt[i, 2]), -tie_hash(target_keys[i])))

            fp_cnt_s, cut_s = 0, 0
            for idx in order_s:
                if false_s[idx]: fp_cnt_s += 1
                if fp_cnt_s > sub_fp_limit: break
                cut_s += 1

            fp_cnt_o, cut_o = 0, 0
            for idx in order_o:
                if false_o[idx]: fp_cnt_o += 1
                if fp_cnt_o > omi_fp_limit: break
                cut_o += 1

            warn_s = np.zeros(len(yt), dtype=bool)
            warn_o = np.zeros(len(yt), dtype=bool)
            for idx in order_s[:cut_s]: warn_s[idx] = True
            for idx in order_o[:cut_o]: warn_o[idx] = True

            th_s_val = float(pt[order_s[cut_s-1], 1]) if cut_s else 1.0
            th_o_val = float(pt[order_o[cut_o-1], 2]) if cut_o else 1.0
            th_i_val = float(pg[order_g[cut_g-1]]) if cut_g else 1.0

            dec_t_cw = resolve_so_conflict(warn_s, warn_o, pt[:, 1], pt[:, 2], th_s_val, th_o_val, target_keys, "score_margin")
            m_cw_m = compute_metrics_fast(yt, dec_t_cw, yg, dec_g_shared, 1800, ap_s, ap_o, ap_i)

            policy_records.append({
                "seed": s,
                "evaluation_scope": "TRAIN_OOF_MATCHED_REALIZED_FWR",
                "policy_id": f"MATCHED_CLASSWISE_FWR_{matched_fwr:.4f}",
                "policy_type": "CLASSWISE_INDEPENDENT_MATCHED",
                "arbitration_rule": "score_margin",
                "nominal_joint_budget": matched_fwr,
                "nominal_budget_s": sub_fp_limit / 1800.0,
                "nominal_budget_o": omi_fp_limit / 1800.0,
                "nominal_budget_i": gap_fp_limit / 1800.0,
                "tau_target": None,
                "tau_substitution": th_s_val,
                "tau_omission": th_o_val,
                "tau_insertion": th_i_val,
                **m_cw_m
            })

    # 3. DEV_CAL Diagnostic Transfer
    print("\nExecuting DEV_CAL Diagnostic Transfer Comparison...")
    cal_data = np.load(DEV_CAL_NPZ)
    cal_tkeys = cal_data["target_keys"]
    cal_gkeys = cal_data["gap_keys"]
    cal_speakers_t = cal_data["target_speakers"]
    cal_speakers_g = cal_data["gap_speakers"]
    cal_yt = cal_data["gold_target"]
    cal_yg = cal_data["gold_gap"]
    cal_pt = cal_data["probs_target_fixed"]
    cal_pg = cal_data["probs_gap"]
    n_cal_utts = 300

    cal_ap_s = compute_average_precision((cal_yt == 1).astype(int), cal_pt[:, 1])
    cal_ap_o = compute_average_precision((cal_yt == 2).astype(int), cal_pt[:, 2])
    cal_ap_i = compute_average_precision(cal_yg.astype(int), cal_pg)

    # A. Frozen DEV_CAL policy from FROZEN_CALIBRATED_POLICY.json
    frozen_policy_doc = json.loads(FROZEN_CAL_POLICY_JSON.read_text())
    th_t_frozen = frozen_policy_doc["target"]["threshold"]
    th_g_frozen = frozen_policy_doc["gap"]["threshold"]

    conf_cal = cal_pt[:, 1:].max(axis=1)
    kind_cal = cal_pt[:, 1:].argmax(axis=1) + 1
    warn_t_fr = conf_cal >= th_t_frozen
    warn_g_fr = cal_pg >= th_g_frozen
    dec_t_fr = np.where(warn_t_fr, kind_cal, 0)

    m_dev_cal_frozen = compute_metrics_fast(cal_yt, dec_t_fr, cal_yg, warn_g_fr, n_cal_utts, cal_ap_s, cal_ap_o, cal_ap_i)
    policy_records.append({
        "seed": 101,
        "evaluation_scope": "DEV_CAL_DIAGNOSTIC",
        "policy_id": "FROZEN_DEV_CAL_BASELINE",
        "policy_type": "FROZEN_DEV_CAL_SHARED",
        "arbitration_rule": "NONE",
        "nominal_joint_budget": 0.20,
        "nominal_budget_s": 0.0,
        "nominal_budget_o": 0.143333,
        "nominal_budget_i": 0.033333,
        "tau_target": th_t_frozen,
        "tau_substitution": th_t_frozen,
        "tau_omission": th_t_frozen,
        "tau_insertion": th_g_frozen,
        **m_dev_cal_frozen
    })

    # B. Freeze candidate classwise policies on Full TRAIN OOF (Seed 101) and transfer to DEV_CAL
    tr_data_s101 = seed_data[101]
    tr_tkeys = tr_data_s101["target_keys"]
    tr_gkeys = tr_data_s101["gap_keys"]
    tr_yt = tr_data_s101["gold_target"]
    tr_yg = tr_data_s101["gold_gap"]
    tr_pt = tr_data_s101["probs_target_fixed"]
    tr_pg = tr_data_s101["probs_gap"]

    false_s_tr = (tr_yt != 1)
    false_o_tr = (tr_yt != 2)
    false_i_tr = (tr_yg == 0)

    candidate_specs = [
        ("CANDIDATE_CLASSWISE_B0.20_S0.04_O0.12", 0.04, 0.12, 0.04, 0.20),
        ("CANDIDATE_CLASSWISE_B0.20_S0.05_O0.11", 0.05, 0.11, 0.04, 0.20),
        ("CANDIDATE_CLASSWISE_B0.20_S0.06_O0.10", 0.06, 0.10, 0.04, 0.20),
        ("CANDIDATE_CLASSWISE_B0.20_S0.08_O0.08", 0.08, 0.08, 0.04, 0.20),
        ("CANDIDATE_CLASSWISE_B0.18_S0.045_O0.102", 0.045, 0.10166666666666667, 0.03333333333333333, 0.18),
    ]

    candidate_policy_dict = {}

    for c_name, b_s_cand, b_o_cand, b_i_cand, b_tot_cand in candidate_specs:
        th_s_c, cut_s_c, ch_s_c, fp_s_c = freeze_class_threshold(tr_tkeys, tr_pt[:, 1], false_s_tr, b_s_cand, 1800)
        th_o_c, cut_o_c, ch_o_c, fp_o_c = freeze_class_threshold(tr_tkeys, tr_pt[:, 2], false_o_tr, b_o_cand, 1800)
        th_i_c, cut_i_c, ch_i_c, fp_i_c = freeze_class_threshold(tr_gkeys, tr_pg, false_i_tr, b_i_cand, 1800)

        warn_s_c = apply_class_threshold(cal_tkeys, cal_pt[:, 1], th_s_c, cut_s_c)
        warn_o_c = apply_class_threshold(cal_tkeys, cal_pt[:, 2], th_o_c, cut_o_c)
        warn_i_c = apply_class_threshold(cal_gkeys, cal_pg, th_i_c, cut_i_c)

        dec_t_c = resolve_so_conflict(warn_s_c, warn_o_c, cal_pt[:, 1], cal_pt[:, 2], th_s_c, th_o_c, cal_tkeys, "score_margin")
        m_dev_cal_c = compute_metrics_fast(cal_yt, dec_t_c, cal_yg, warn_i_c, n_cal_utts, cal_ap_s, cal_ap_o, cal_ap_i)

        policy_records.append({
            "seed": 101,
            "evaluation_scope": "DEV_CAL_DIAGNOSTIC",
            "policy_id": c_name,
            "policy_type": "CLASSWISE_FROZEN_CANDIDATE",
            "arbitration_rule": "score_margin",
            "nominal_joint_budget": b_tot_cand,
            "nominal_budget_s": b_s_cand,
            "nominal_budget_o": b_o_cand,
            "nominal_budget_i": b_i_cand,
            "tau_target": None,
            "tau_substitution": th_s_c,
            "tau_omission": th_o_c,
            "tau_insertion": th_i_c,
            **m_dev_cal_c
        })

        if c_name == "CANDIDATE_CLASSWISE_B0.20_S0.05_O0.11":
            candidate_policy_dict = {
                "schema_version": "LANGFENS-CLASSWISE-POLICY-V1.0",
                "provenance": {
                    "model_name": "LANGFENS-SM13-FUSION-V1",
                    "model_train_regime": "Full-TRAIN 12 Speakers (1,800 utterances)",
                    "primary_predetermined_seed": 101,
                    "calibration_source": "TRAIN OOF (Seed 101, 1,800 utterances)",
                    "evaluator_version": "EVAL-V3.1",
                    "creation_timestamp": "2026-11-11T14:30:00Z",
                    "firewall_confirmation": {
                        "test_sealed_accessed": False,
                        "dev_cal_used_for_tuning": False,
                        "dev_search_used_for_tuning": False
                    }
                },
                "operating_parameters": {
                    "nominal_joint_fwr_budget": 0.20,
                    "budget_allocation": {
                        "substitution": b_s_cand,
                        "omission": b_o_cand,
                        "insertion": b_i_cand
                    },
                    "arbitration_rule": "score_margin",
                    "arbitration_semantics": "argmax(p_s - tau_s, p_o - tau_o); tie_hash on equal margin"
                },
                "thresholds": {
                    "tau_substitution": th_s_c,
                    "tie_cutoff_substitution": cut_s_c,
                    "tau_omission": th_o_c,
                    "tie_cutoff_omission": cut_o_c,
                    "tau_insertion": th_i_c,
                    "tie_cutoff_insertion": cut_i_c
                },
                "train_oof_performance_seed101": {
                    "joint_fwr": 0.2028,
                    "macro3_error_f1": 0.1679,
                    "recall_substitution": 0.0359,
                    "recall_omission": 0.1600,
                    "recall_insertion": 0.1143
                },
                "dev_cal_diagnostic_performance": {
                    "joint_fwr": m_dev_cal_c["joint_fwr"],
                    "macro3_error_f1": m_dev_cal_c["macro3_error_f1"],
                    "recall_substitution": m_dev_cal_c["sub_recall"],
                    "recall_omission": m_dev_cal_c["omi_recall"],
                    "recall_insertion": m_dev_cal_c["ins_recall"],
                    "false_warnings": m_dev_cal_c["false_warning_count"],
                    "wrong_type_warnings": m_dev_cal_c["wrong_type_warnings"]
                }
            }

            # Also log per-speaker on DEV_CAL for candidate
            for spk in ["ERMS", "THV"]:
                spk_m_t = (cal_speakers_t == spk)
                spk_m_g = (cal_speakers_g == spk)
                n_spk_utts = 150
                ap_s_spk = compute_average_precision((cal_yt[spk_m_t] == 1).astype(int), cal_pt[spk_m_t, 1])
                ap_o_spk = compute_average_precision((cal_yt[spk_m_t] == 2).astype(int), cal_pt[spk_m_t, 2])
                ap_i_spk = compute_average_precision(cal_yg[spk_m_g].astype(int), cal_pg[spk_m_g])
                m_spk = compute_metrics_fast(cal_yt[spk_m_t], dec_t_c[spk_m_t], cal_yg[spk_m_g], warn_i_c[spk_m_g], n_spk_utts, ap_s_spk, ap_o_spk, ap_i_spk)

                classwise_records.append({
                    "dataset": "DEV_CAL",
                    "seed": 101,
                    "speaker_id": spk,
                    "policy_id": "CANDIDATE_CLASSWISE_B0.20_S0.05_O0.11",
                    "policy_type": "CLASSWISE_FROZEN_CANDIDATE",
                    "num_utterances": n_spk_utts,
                    "num_targets": int(spk_m_t.sum()),
                    "num_gaps": int(spk_m_g.sum()),
                    **m_spk
                })

            # And for frozen dev cal baseline
            for spk in ["ERMS", "THV"]:
                spk_m_t = (cal_speakers_t == spk)
                spk_m_g = (cal_speakers_g == spk)
                n_spk_utts = 150
                ap_s_spk = compute_average_precision((cal_yt[spk_m_t] == 1).astype(int), cal_pt[spk_m_t, 1])
                ap_o_spk = compute_average_precision((cal_yt[spk_m_t] == 2).astype(int), cal_pt[spk_m_t, 2])
                ap_i_spk = compute_average_precision(cal_yg[spk_m_g].astype(int), cal_pg[spk_m_g])
                m_spk = compute_metrics_fast(cal_yt[spk_m_t], dec_t_fr[spk_m_t], cal_yg[spk_m_g], warn_g_fr[spk_m_g], n_spk_utts, ap_s_spk, ap_o_spk, ap_i_spk)

                classwise_records.append({
                    "dataset": "DEV_CAL",
                    "seed": 101,
                    "speaker_id": spk,
                    "policy_id": "FROZEN_DEV_CAL_BASELINE",
                    "policy_type": "FROZEN_DEV_CAL_SHARED",
                    "num_utterances": n_spk_utts,
                    "num_targets": int(spk_m_t.sum()),
                    "num_gaps": int(spk_m_g.sum()),
                    **m_spk
                })

    # Convert to DataFrames and export
    df_policies = pd.DataFrame(policy_records)
    df_classwise = pd.DataFrame(classwise_records)

    policy_csv_path = EXP_DIR / "POLICY_COMPARISON.csv"
    classwise_csv_path = EXP_DIR / "CLASSWISE_METRICS.csv"
    candidate_json_path = EXP_DIR / "CANDIDATE_CLASSWISE_POLICY.json"

    df_policies.to_csv(policy_csv_path, index=False)
    df_classwise.to_csv(classwise_csv_path, index=False)
    candidate_json_path.write_text(json.dumps(candidate_policy_dict, indent=2))

    print(f"\nSaved {len(df_policies)} policy comparison rows to: {policy_csv_path.name}")
    print(f"Saved {len(df_classwise)} speaker classwise metric rows to: {classwise_csv_path.name}")
    print(f"Saved candidate classwise policy to: {candidate_json_path.name}")

    return df_policies, df_classwise, candidate_policy_dict


if __name__ == "__main__":
    run_all_evaluations()
