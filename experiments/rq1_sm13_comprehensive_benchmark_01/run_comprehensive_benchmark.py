#!/usr/bin/env python3
"""
experiments/rq1_sm13_comprehensive_benchmark_01/run_comprehensive_benchmark.py

Master Comprehensive Benchmark Runner for SM13 Asymmetric Routing V1.
Executes Phases 3, 4, 5, and 6.
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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path("/home/khoa/Projects/New-Method-Research")
sys.path.insert(0, str(ROOT))

from research.evaluation_v3.evaluator import UnifiedEvaluatorV3

OUT_DIR = ROOT / "experiments/rq1_sm13_comprehensive_benchmark_01"
CURVES_DIR = OUT_DIR / "PR_FWR_CURVES"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CURVES_DIR.mkdir(parents=True, exist_ok=True)

# Frozen policy paths
POLICY_BASELINE = ROOT / "experiments/rq1_sm13_dev_calibration_01/FROZEN_CALIBRATED_POLICY.json"
POLICY_ASYMMETRIC = ROOT / "FROZEN_ASYMMETRIC_POLICY.json"

TIE_PREFIX = "LANGFENS-MB07-warning-tie-v1:"


def tie_hash(event_id: str) -> int:
    return int.from_bytes(hashlib.sha256((TIE_PREFIX + str(event_id)).encode()).digest()[:8], "big")


def apply_class_threshold(event_ids: np.ndarray, scores: np.ndarray, threshold: float, cutoff: int) -> np.ndarray:
    out = np.zeros(len(scores), dtype=bool)
    for i, (eid, s) in enumerate(zip(event_ids, scores)):
        if s > threshold:
            out[i] = True
        elif s < threshold:
            out[i] = False
        else:
            out[i] = (tie_hash(eid) >= cutoff)
    return out


def resolve_so_conflict(warn_s: np.ndarray, warn_o: np.ndarray, p_s: np.ndarray, p_o: np.ndarray,
                        th_s: float, th_o: float, event_ids: np.ndarray) -> np.ndarray:
    n = len(warn_s)
    dec = np.zeros(n, dtype=np.int64)
    only_s = warn_s & (~warn_o)
    only_o = warn_o & (~warn_s)
    conflict = warn_s & warn_o

    dec[only_s] = 1  # SUBSTITUTION
    dec[only_o] = 2  # OMISSION

    if np.any(conflict):
        for idx in np.flatnonzero(conflict):
            m_s = p_s[idx] - th_s
            m_o = p_o[idx] - th_o
            if m_s > m_o:
                dec[idx] = 1
            elif m_o > m_s:
                dec[idx] = 2
            else:
                dec[idx] = 1 if (tie_hash(event_ids[idx]) % 2 == 1) else 2
    return dec


def freeze_class_threshold(event_ids: np.ndarray, scores: np.ndarray, false_flags: np.ndarray,
                           budget_per_utt: float, n_utts: int) -> Tuple[float, int, int, int]:
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


def load_populations() -> Dict[str, Dict[str, Any]]:
    print("Loading all benchmark populations...")
    pops = {}

    # File paths
    paths = {
        "TRAIN_FULL": ROOT / "experiments/rq1_sm13_formal_train_01/predictions/predictions_sm13_seed101_fulltrain.npz",
        "TRAIN_OOF": ROOT / "experiments/rq1_method_select_02r/agent_a/predictions/predictions_sm13_seed101.npz",
        "DEV_CAL": ROOT / "experiments/rq1_sm13_dev_inference_01/predictions_sm13_seed101_dev_cal.npz",
        "DEV_SEARCH": ROOT / "experiments/rq1_sm13_dev_inference_01/predictions_sm13_seed101_dev_search.npz",
        "TEST_SEALED": ROOT / "experiments/rq1_sm13_test_sealed_evaluation_01/predictions_sm13_seed101_test_sealed.npz",
        "DEV_REPLICATION": OUT_DIR / "predictions_sm13_seed101_dev_replication.npz",
    }

    for name, p in paths.items():
        data = np.load(p)
        pops[name] = {k: data[k] for k in data.files}
        pops[name]["n_utts"] = len(data["gap_keys"]) - len(data["target_keys"])

    # Build DEV_COMBINED (CAL + SEARCH)
    c_cal = pops["DEV_CAL"]
    c_sea = pops["DEV_SEARCH"]
    pops["DEV_COMBINED"] = {
        "target_keys": np.concatenate([c_cal["target_keys"], c_sea["target_keys"]]),
        "gap_keys": np.concatenate([c_cal["gap_keys"], c_sea["gap_keys"]]),
        "target_speakers": np.concatenate([c_cal["target_speakers"], c_sea["target_speakers"]]),
        "gap_speakers": np.concatenate([c_cal["gap_speakers"], c_sea["gap_speakers"]]),
        "gold_target": np.concatenate([c_cal["gold_target"], c_sea["gold_target"]]),
        "gold_gap": np.concatenate([c_cal["gold_gap"], c_sea["gold_gap"]]),
        "probs_target_neural": np.concatenate([c_cal["probs_target_neural"], c_sea["probs_target_neural"]]),
        "probs_target_fixed": np.concatenate([c_cal["probs_target_fixed"], c_sea["probs_target_fixed"]]),
        "probs_gap": np.concatenate([c_cal["probs_gap"], c_sea["probs_gap"]]),
        "e4_probs": np.concatenate([c_cal["e4_probs"], c_sea["e4_probs"]]),
        "sllr": np.concatenate([c_cal["sllr"], c_sea["sllr"]]),
        "ollr": np.concatenate([c_cal["ollr"], c_sea["ollr"]]),
        "n_utts": c_cal["n_utts"] + c_sea["n_utts"]
    }

    # Load SpeechOcean
    so_p = OUT_DIR / "predictions_sm13_seed101_speechocean.npz"
    so_data = np.load(so_p)
    pops["SPEECHOCEAN"] = {k: so_data[k] for k in so_data.files}
    pops["SPEECHOCEAN"]["n_utts"] = len(so_data["gap_keys"]) - len(so_data["target_keys"])

    print(f"Loaded {len(pops)} populations.")
    return pops


def evaluate_system_on_pop(pop_name: str, pop_data: Dict[str, Any], sys_name: str,
                           dec_t: np.ndarray, dec_g: np.ndarray,
                           p_s: np.ndarray, p_o: np.ndarray, p_g: np.ndarray) -> Dict[str, Any]:
    tkeys = pop_data["target_keys"]
    gkeys = pop_data["gap_keys"]
    yt = pop_data["gold_target"]
    yg = pop_data["gold_gap"]
    n_utts = pop_data["n_utts"]

    labels_map = ("CORRECT", "SUBSTITUTION", "OMISSION")
    gt = [
        {"dataset_id": k.split("|")[0], "speaker_id": k.split("|")[1], "utterance_id": k.split("|")[2],
         "position": int(k.split("|")[4]), "gold_label": labels_map[y]}
        for k, y in zip(tkeys, yt)
    ]
    gg = [
        {"dataset_id": k.split("|")[0], "speaker_id": k.split("|")[1], "utterance_id": k.split("|")[2],
         "gap_position": int(k.split("|")[4]), "gold_has_insertion": bool(y)}
        for k, y in zip(gkeys, yg)
    ]
    pt = [
        {"dataset_id": k.split("|")[0], "speaker_id": k.split("|")[1], "utterance_id": k.split("|")[2],
         "position": int(k.split("|")[4]), "pred_label": labels_map[p],
         "score_substitution": float(pr_s), "score_omission": float(pr_o)}
        for k, p, pr_s, pr_o in zip(tkeys, dec_t, p_s, p_o)
    ]
    pg = [
        {"dataset_id": k.split("|")[0], "speaker_id": k.split("|")[1], "utterance_id": k.split("|")[2],
         "gap_position": int(k.split("|")[4]), "pred_has_insertion": bool(p),
         "score_insertion": float(pr)}
        for k, p, pr in zip(gkeys, dec_g, p_g)
    ]

    evaluator = UnifiedEvaluatorV3()
    res = evaluator.evaluate(pt, pg, gt, gg)
    summary = res["summary"]

    out = {
        "population": pop_name,
        "system": sys_name,
        "utterances": n_utts,
        "target_events": len(tkeys),
        "gap_events": len(gkeys),
        "joint_fwr": summary["joint_fwr"],
        "false_warnings": summary["false_warning_count"],
        "wrong_type_warnings": summary["wrong_type_error_false_warnings"],
        "macro3_f1": summary["macro3_error_f1"],
        "macro3_ap": summary["macro3_error_ap"],
        # Substitution
        "sub_tp": res["classes"]["SUBSTITUTION"]["TP"],
        "sub_fp": res["classes"]["SUBSTITUTION"]["FP"],
        "sub_fn": res["classes"]["SUBSTITUTION"]["FN"],
        "sub_precision": res["classes"]["SUBSTITUTION"]["precision"],
        "sub_recall": res["classes"]["SUBSTITUTION"]["recall"],
        "sub_f1": res["classes"]["SUBSTITUTION"]["f1"],
        "sub_ap": res["classes"]["SUBSTITUTION"]["ap"],
        # Omission
        "omi_tp": res["classes"]["OMISSION"]["TP"],
        "omi_fp": res["classes"]["OMISSION"]["FP"],
        "omi_fn": res["classes"]["OMISSION"]["FN"],
        "omi_precision": res["classes"]["OMISSION"]["precision"],
        "omi_recall": res["classes"]["OMISSION"]["recall"],
        "omi_f1": res["classes"]["OMISSION"]["f1"],
        "omi_ap": res["classes"]["OMISSION"]["ap"],
        # Insertion
        "ins_tp": res["classes"]["INSERTION"]["TP"],
        "ins_fp": res["classes"]["INSERTION"]["FP"],
        "ins_fn": res["classes"]["INSERTION"]["FN"],
        "ins_precision": res["classes"]["INSERTION"]["precision"],
        "ins_recall": res["classes"]["INSERTION"]["recall"],
        "ins_f1": res["classes"]["INSERTION"]["f1"],
        "ins_ap": res["classes"]["INSERTION"]["ap"],
        "raw_res": res
    }
    return out


def main():
    pops = load_populations()

    # Load frozen policies
    pol_base = json.loads(POLICY_BASELINE.read_text())
    tau_T_base = float(pol_base["target"]["threshold"])
    tau_G_base = float(pol_base["gap"]["threshold"])

    pol_asym = json.loads(POLICY_ASYMMETRIC.read_text())
    th_asym = pol_asym["operating_parameters"]["thresholds"]
    tau_S = float(th_asym["tau_substitution"])
    cut_S = int(th_asym["tie_cutoff_substitution"])
    tau_O = float(th_asym["tau_omission"])
    cut_O = int(th_asym["tie_cutoff_omission"])
    tau_I = float(th_asym["tau_insertion"])
    cut_I = int(th_asym["tie_cutoff_insertion"])

    print(f"Loaded frozen policies:\n  Baseline: tau_T={tau_T_base:.6f}, tau_G={tau_G_base:.6f}\n  Asymmetric: tau_S={tau_S:.6f}, tau_O={tau_O:.6f}, tau_I={tau_I:.6f}")

    # =========================================================================
    # Phase 3: L2-ARCTIC Supervised Benchmark
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 3: L2-ARCTIC Supervised Benchmark Across Populations")
    print("="*80)

    l2_pop_names = ["TRAIN_FULL", "TRAIN_OOF", "DEV_CAL", "DEV_SEARCH", "DEV_COMBINED", "TEST_SEALED", "DEV_REPLICATION"]
    l2_results = []
    speaker_results = []
    classwise_records = []

    for pname in l2_pop_names:
        pdata = pops[pname]
        tkeys = pdata["target_keys"]
        gkeys = pdata["gap_keys"]
        p_neu = pdata["probs_target_neural"]
        p_fix = pdata["probs_target_fixed"]
        pg = pdata["probs_gap"]

        # 1. Baseline decisions
        conf_base = p_fix[:, 1:].max(axis=1)
        kind_base = p_fix[:, 1:].argmax(axis=1) + 1
        dec_t_base = np.where(conf_base >= tau_T_base, kind_base, 0)
        dec_g_base = pg >= tau_G_base
        m_base = evaluate_system_on_pop(pname, pdata, "Fixed 50/50 Baseline",
                                       dec_t_base, dec_g_base, p_fix[:, 1], p_fix[:, 2], pg)

        # 2. Asymmetric decisions
        warn_s = apply_class_threshold(tkeys, p_neu[:, 1], tau_S, cut_S)
        warn_o = apply_class_threshold(tkeys, p_fix[:, 2], tau_O, cut_O)
        warn_i = apply_class_threshold(gkeys, pg, tau_I, cut_I)
        dec_t_asym = resolve_so_conflict(warn_s, warn_o, p_neu[:, 1], p_fix[:, 2], tau_S, tau_O, tkeys)
        dec_g_asym = warn_i
        m_asym = evaluate_system_on_pop(pname, pdata, "Asymmetric Routing (B=0.12)",
                                       dec_t_asym, dec_g_asym, p_neu[:, 1], p_fix[:, 2], pg)

        l2_results.append(m_base)
        l2_results.append(m_asym)

        print(f"\n--- Population: {pname} ({pdata['n_utts']} utts, {len(tkeys)} targets, {len(gkeys)} gaps) ---")
        print(f"  Fixed 50/50:   FWR={m_base['joint_fwr']:.4f}, Macro-3 F1={m_base['macro3_f1']:.4f}, S_rec={m_base['sub_recall']*100:.2f}%, S_prec={m_base['sub_precision']*100 if m_base['sub_precision'] else 0:.2f}%, O_rec={m_base['omi_recall']*100:.2f}%, I_rec={m_base['ins_recall']*100:.2f}%")
        print(f"  Asymmetric:    FWR={m_asym['joint_fwr']:.4f}, Macro-3 F1={m_asym['macro3_f1']:.4f}, S_rec={m_asym['sub_recall']*100:.2f}%, S_prec={m_asym['sub_precision']*100 if m_asym['sub_precision'] else 0:.2f}%, O_rec={m_asym['omi_recall']*100:.2f}%, I_rec={m_asym['ins_recall']*100:.2f}%")

        # Classwise records
        for sys_name, res_dict in [("Fixed 50/50 Baseline", m_base), ("Asymmetric Routing (B=0.12)", m_asym)]:
            for cname in ["SUBSTITUTION", "OMISSION", "INSERTION"]:
                prefix = {"SUBSTITUTION": "sub_", "OMISSION": "omi_", "INSERTION": "ins_"}[cname]
                raw_c = res_dict["raw_res"]["classes"][cname]
                classwise_records.append({
                    "population": pname,
                    "system": sys_name,
                    "class": cname,
                    "tp": res_dict[prefix + "tp"],
                    "fp": res_dict[prefix + "fp"],
                    "fn": res_dict[prefix + "fn"],
                    "precision": res_dict[prefix + "precision"],
                    "recall": res_dict[prefix + "recall"],
                    "f1": res_dict[prefix + "f1"],
                    "ap": res_dict[prefix + "ap"],
                    "roc_auc": raw_c.get("roc_auc"),
                    "prevalence": raw_c.get("prevalence"),
                    "normalized_lift": raw_c.get("normalized_lift"),
                    "support": raw_c.get("support")
                })

        # Per-speaker evaluation on this population
        spks = sorted(set(pdata["target_speakers"]))
        for spk in spks:
            mask_t = (pdata["target_speakers"] == spk)
            mask_g = (pdata["gap_speakers"] == spk)
            n_spk_utts = int(mask_g.sum() - mask_t.sum())

            # Evaluate Asymmetric for this speaker
            yt_spk = pdata["gold_target"][mask_t]
            yg_spk = pdata["gold_gap"][mask_g]
            dt_spk = dec_t_asym[mask_t]
            dg_spk = dec_g_asym[mask_g]

            # Fast metrics
            tp_s = int(np.sum((yt_spk == 1) & (dt_spk == 1)))
            fp_s = int(np.sum((yt_spk != 1) & (dt_spk == 1)))
            fn_s = int(np.sum((yt_spk == 1) & (dt_spk != 1)))

            tp_o = int(np.sum((yt_spk == 2) & (dt_spk == 2)))
            fp_o = int(np.sum((yt_spk != 2) & (dt_spk == 2)))
            fn_o = int(np.sum((yt_spk == 2) & (dt_spk != 2)))

            tp_i = int(np.sum((yg_spk == 1) & (dg_spk == 1)))
            fp_i = int(np.sum((yg_spk != 1) & (dg_spk == 1)))
            fn_i = int(np.sum((yg_spk == 1) & (dg_spk != 1)))

            f1_s = 2 * tp_s / (2 * tp_s + fp_s + fn_s) if (2 * tp_s + fp_s + fn_s) else 0.0
            f1_o = 2 * tp_o / (2 * tp_o + fp_o + fn_o) if (2 * tp_o + fp_o + fn_o) else 0.0
            f1_i = 2 * tp_i / (2 * tp_i + fp_i + fn_i) if (2 * tp_i + fp_i + fn_i) else 0.0

            fw_spk = fp_s + fp_o + fp_i
            fwr_spk = fw_spk / max(1, n_spk_utts)
            macro_f1_spk = (f1_s + f1_o + f1_i) / 3.0

            # Baseline metrics
            dt_base_spk = dec_t_base[mask_t]
            dg_base_spk = dec_g_base[mask_g]
            b_tp_s = int(np.sum((yt_spk == 1) & (dt_base_spk == 1)))
            b_fp_s = int(np.sum((yt_spk != 1) & (dt_base_spk == 1)))
            b_fn_s = int(np.sum((yt_spk == 1) & (dt_base_spk != 1)))
            b_tp_o = int(np.sum((yt_spk == 2) & (dt_base_spk == 2)))
            b_fp_o = int(np.sum((yt_spk != 2) & (dt_base_spk == 2)))
            b_fn_o = int(np.sum((yt_spk == 2) & (dt_base_spk != 2)))
            b_tp_i = int(np.sum((yg_spk == 1) & (dg_base_spk == 1)))
            b_fp_i = int(np.sum((yg_spk != 1) & (dg_base_spk == 1)))
            b_fn_i = int(np.sum((yg_spk == 1) & (dg_base_spk != 1)))

            b_f1_s = 2 * b_tp_s / (2 * b_tp_s + b_fp_s + b_fn_s) if (2 * b_tp_s + b_fp_s + b_fn_s) else 0.0
            b_f1_o = 2 * b_tp_o / (2 * b_tp_o + b_fp_o + b_fn_o) if (2 * b_tp_o + b_fp_o + b_fn_o) else 0.0
            b_f1_i = 2 * b_tp_i / (2 * b_tp_i + b_fp_i + b_fn_i) if (2 * b_tp_i + b_fp_i + b_fn_i) else 0.0
            b_fwr = (b_fp_s + b_fp_o + b_fp_i) / max(1, n_spk_utts)
            b_macro_f1 = (b_f1_s + b_f1_o + b_f1_i) / 3.0

            speaker_results.append({
                "population": pname,
                "speaker_id": spk,
                "utterances": n_spk_utts,
                "baseline_fwr": b_fwr,
                "baseline_macro3_f1": b_macro_f1,
                "baseline_sub_recall": b_tp_s / (b_tp_s + b_fn_s) if (b_tp_s + b_fn_s) else 0.0,
                "asym_fwr": fwr_spk,
                "asym_false_warnings": fw_spk,
                "asym_macro3_f1": macro_f1_spk,
                "asym_sub_tp": tp_s,
                "asym_sub_recall": tp_s / (tp_s + fn_s) if (tp_s + fn_s) else 0.0,
                "asym_sub_precision": tp_s / (tp_s + fp_s) if (tp_s + fp_s) else 0.0,
                "asym_omi_tp": tp_o,
                "asym_omi_recall": tp_o / (tp_o + fn_o) if (tp_o + fn_o) else 0.0,
                "asym_omi_precision": tp_o / (tp_o + fp_o) if (tp_o + fp_o) else 0.0,
                "asym_ins_tp": tp_i,
                "asym_ins_recall": tp_i / (tp_i + fn_i) if (tp_i + fn_i) else 0.0,
                "asym_ins_precision": tp_i / (tp_i + fp_i) if (tp_i + fp_i) else 0.0,
                "delta_fwr": fwr_spk - b_fwr,
                "delta_macro3_f1": macro_f1_spk - b_macro_f1
            })

    # Save L2_ARCTIC_BENCHMARK.csv
    l2_df_records = []
    for r in l2_results:
        rec = {k: v for k, v in r.items() if k != "raw_res"}
        l2_df_records.append(rec)
    df_l2 = pd.DataFrame(l2_df_records)
    df_l2.to_csv(OUT_DIR / "L2_ARCTIC_BENCHMARK.csv", index=False)
    print(f"\n[OK] Wrote L2_ARCTIC_BENCHMARK.csv ({len(df_l2)} rows).")

    # Save CLASSWISE_METRICS.csv
    df_cls = pd.DataFrame(classwise_records)
    df_cls.to_csv(OUT_DIR / "CLASSWISE_METRICS.csv", index=False)
    print(f"[OK] Wrote CLASSWISE_METRICS.csv ({len(df_cls)} rows).")

    # Save PER_SPEAKER_METRICS.csv
    df_spk = pd.DataFrame(speaker_results)
    df_spk.to_csv(OUT_DIR / "PER_SPEAKER_METRICS.csv", index=False)
    print(f"[OK] Wrote PER_SPEAKER_METRICS.csv ({len(df_spk)} rows).")

    # =========================================================================
    # Phase 4: Warning Budget Sensitivity & Unrestricted Sweeps
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 4: Warning Budget Sensitivity & Unrestricted Diagnostic Sweeps")
    print("="*80)

    train_oof = pops["TRAIN_OOF"]
    oof_tkeys = train_oof["target_keys"]
    oof_gkeys = train_oof["gap_keys"]
    oof_yt = train_oof["gold_target"]
    oof_yg = train_oof["gold_gap"]
    oof_p_neu = train_oof["probs_target_neural"]
    oof_p_fix = train_oof["probs_target_fixed"]
    oof_pg = train_oof["probs_gap"]
    n_oof_utts = train_oof["n_utts"]

    false_s_flag = (oof_yt != 1)
    false_o_flag = (oof_yt != 2)
    false_i_flag = (oof_yg != 1)

    budgets = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0, 2.0, 5.0]
    fwr_sweep_records = []

    # Map of nominal budgets to derived thresholds
    budget_thresholds = {}
    for B in budgets:
        if B == 0.0:
            th_s, cut_s = 1.0, 2**64
            th_o, cut_o = 1.0, 2**64
            th_i, cut_i = 1.0, 2**64
        else:
            # Canonical allocation: B_s = B / 6, B_o = 4B / 6, B_i = B / 6
            b_s = B * (2.0 / 12.0)
            b_o = B * (8.0 / 12.0)
            b_i = B * (2.0 / 12.0)
            th_s, cut_s, _, _ = freeze_class_threshold(oof_tkeys, oof_p_neu[:, 1], false_s_flag, b_s, n_oof_utts)
            th_o, cut_o, _, _ = freeze_class_threshold(oof_tkeys, oof_p_fix[:, 2], false_o_flag, b_o, n_oof_utts)
            th_i, cut_i, _, _ = freeze_class_threshold(oof_gkeys, oof_pg, false_i_flag, b_i, n_oof_utts)

        budget_thresholds[B] = (th_s, cut_s, th_o, cut_o, th_i, cut_i)

    # Evaluate across populations
    for pname in ["TRAIN_OOF", "DEV_CAL", "DEV_SEARCH", "DEV_COMBINED", "DEV_REPLICATION", "TEST_SEALED"]:
        pdata = pops[pname]
        tkeys = pdata["target_keys"]
        gkeys = pdata["gap_keys"]
        yt = pdata["gold_target"]
        yg = pdata["gold_gap"]
        p_neu = pdata["probs_target_neural"]
        p_fix = pdata["probs_target_fixed"]
        pg = pdata["probs_gap"]
        n_utts = pdata["n_utts"]

        for B in budgets:
            th_s, cut_s, th_o, cut_o, th_i, cut_i = budget_thresholds[B]
            warn_s = apply_class_threshold(tkeys, p_neu[:, 1], th_s, cut_s)
            warn_o = apply_class_threshold(tkeys, p_fix[:, 2], th_o, cut_o)
            warn_i = apply_class_threshold(gkeys, pg, th_i, cut_i)
            dec_t = resolve_so_conflict(warn_s, warn_o, p_neu[:, 1], p_fix[:, 2], th_s, th_o, tkeys)
            dec_g = warn_i

            tp_s = int(np.sum((yt == 1) & (dec_t == 1)))
            fp_s = int(np.sum((yt != 1) & (dec_t == 1)))
            fn_s = int(np.sum((yt == 1) & (dec_t != 1)))

            tp_o = int(np.sum((yt == 2) & (dec_t == 2)))
            fp_o = int(np.sum((yt != 2) & (dec_t == 2)))
            fn_o = int(np.sum((yt == 2) & (dec_t != 2)))

            tp_i = int(np.sum((yg == 1) & (dec_g == 1)))
            fp_i = int(np.sum((yg != 1) & (dec_g == 1)))
            fn_i = int(np.sum((yg == 1) & (dec_g != 1)))

            prec_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) else 0.0
            rec_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) else 0.0
            f1_s = 2 * tp_s / (2 * tp_s + fp_s + fn_s) if (2 * tp_s + fp_s + fn_s) else 0.0

            prec_o = tp_o / (tp_o + fp_o) if (tp_o + fp_o) else 0.0
            rec_o = tp_o / (tp_o + fn_o) if (tp_o + fn_o) else 0.0
            f1_o = 2 * tp_o / (2 * tp_o + fp_o + fn_o) if (2 * tp_o + fp_o + fn_o) else 0.0

            prec_i = tp_i / (tp_i + fp_i) if (tp_i + fp_i) else 0.0
            rec_i = tp_i / (tp_i + fn_i) if (tp_i + fn_i) else 0.0
            f1_i = 2 * tp_i / (2 * tp_i + fp_i + fn_i) if (2 * tp_i + fp_i + fn_i) else 0.0

            tot_fw = fp_s + fp_o + fp_i
            tot_correct = tp_s + tp_o + tp_i
            wrong_type = int(np.sum((yt == 2) & (dec_t == 1)) + np.sum((yt == 1) & (dec_t == 2)))
            realized_fwr = tot_fw / n_utts
            macro3_f1 = (f1_s + f1_o + f1_i) / 3.0

            fwr_sweep_records.append({
                "population": pname,
                "nominal_budget": B,
                "realized_fwr": realized_fwr,
                "macro3_f1": macro3_f1,
                "tau_substitution": th_s,
                "tau_omission": th_o,
                "tau_insertion": th_i,
                "correct_detections": tot_correct,
                "false_warnings": tot_fw,
                "wrong_type_warnings": wrong_type,
                "sub_tp": tp_s, "sub_fp": fp_s, "sub_fn": fn_s,
                "sub_precision": prec_s, "sub_recall": rec_s, "sub_f1": f1_s,
                "omi_tp": tp_o, "omi_fp": fp_o, "omi_fn": fn_o,
                "omi_precision": prec_o, "omi_recall": rec_o, "omi_f1": f1_o,
                "ins_tp": tp_i, "ins_fp": fp_i, "ins_fn": fn_i,
                "ins_precision": prec_i, "ins_recall": rec_i, "ins_f1": f1_i,
                "policy_status": "CALIBRATED_PROSPECTIVE_POLICY" if pname != "TEST_SEALED" else "HISTORICAL_SEALED_TEST_APPLICATION"
            })

    # Diagnostic Oracle Sweeps on DEV_COMBINED (Grid Search for peak Macro-3 F1)
    print("\nRunning Post-hoc Diagnostic Oracle Sweep on DEV_COMBINED...")
    dev_comb = pops["DEV_COMBINED"]
    d_tkeys = dev_comb["target_keys"]
    d_gkeys = dev_comb["gap_keys"]
    d_yt = dev_comb["gold_target"]
    d_yg = dev_comb["gold_gap"]
    d_neu = dev_comb["probs_target_neural"]
    d_fix = dev_comb["probs_target_fixed"]
    d_pg = dev_comb["probs_gap"]
    n_dev_utts = dev_comb["n_utts"]

    best_macro_f1 = -1.0
    best_oracle = None

    # Coarse-to-fine sweep over sensible probability ranges
    s_grid = np.linspace(0.10, 0.95, 18)
    o_grid = np.linspace(0.10, 0.90, 17)
    i_grid = np.linspace(0.10, 0.90, 17)

    for ts in s_grid:
        w_s = d_neu[:, 1] >= ts
        for to in o_grid:
            w_o = d_fix[:, 2] >= to
            # Resolve conflict
            dt = np.zeros(len(d_tkeys), dtype=np.int64)
            dt[w_s & (~w_o)] = 1
            dt[w_o & (~w_s)] = 2
            conf = w_s & w_o
            if np.any(conf):
                ms = d_neu[:, 1][conf] - ts
                mo = d_fix[:, 2][conf] - to
                dt[conf] = np.where(ms >= mo, 1, 2)

            tp_s = int(np.sum((d_yt == 1) & (dt == 1)))
            fp_s = int(np.sum((d_yt != 1) & (dt == 1)))
            fn_s = int(np.sum((d_yt == 1) & (dt != 1)))
            f1_s = 2 * tp_s / (2 * tp_s + fp_s + fn_s) if (2 * tp_s + fp_s + fn_s) else 0.0

            tp_o = int(np.sum((d_yt == 2) & (dt == 2)))
            fp_o = int(np.sum((d_yt != 2) & (dt == 2)))
            fn_o = int(np.sum((d_yt == 2) & (dt != 2)))
            f1_o = 2 * tp_o / (2 * tp_o + fp_o + fn_o) if (2 * tp_o + fp_o + fn_o) else 0.0

            for ti in i_grid:
                w_i = d_pg >= ti
                tp_i = int(np.sum((d_yg == 1) & (w_i == 1)))
                fp_i = int(np.sum((d_yg != 1) & (w_i == 1)))
                fn_i = int(np.sum((d_yg == 1) & (w_i != 1)))
                f1_i = 2 * tp_i / (2 * tp_i + fp_i + fn_i) if (2 * tp_i + fp_i + fn_i) else 0.0

                mf1 = (f1_s + f1_o + f1_i) / 3.0
                if mf1 > best_macro_f1:
                    best_macro_f1 = mf1
                    best_oracle = {
                        "tau_s": ts, "tau_o": to, "tau_i": ti,
                        "macro3_f1": mf1,
                        "realized_fwr": (fp_s + fp_o + fp_i) / n_dev_utts,
                        "sub_recall": tp_s / (tp_s + fn_s), "sub_prec": tp_s / (tp_s + fp_s), "sub_f1": f1_s,
                        "omi_recall": tp_o / (tp_o + fn_o), "omi_prec": tp_o / (tp_o + fp_o), "omi_f1": f1_o,
                        "ins_recall": tp_i / (tp_i + fn_i), "ins_prec": tp_i / (tp_i + fp_i), "ins_f1": f1_i,
                        "correct": tp_s + tp_o + tp_i, "false_warnings": fp_s + fp_o + fp_i
                    }

    print(f"  DEV_COMBINED Diagnostic Peak Macro-3 F1: {best_macro_f1:.4f} (Realized FWR: {best_oracle['realized_fwr']:.4f} warnings/utt)")
    print(f"  Oracle Thresholds: tau_S={best_oracle['tau_s']:.2f}, tau_O={best_oracle['tau_o']:.2f}, tau_I={best_oracle['tau_i']:.2f}")
    print(f"  Oracle Recalls: S={best_oracle['sub_recall']*100:.1f}%, O={best_oracle['omi_recall']*100:.1f}%, I={best_oracle['ins_recall']*100:.1f}%")
    print(f"  Oracle Precisions: S={best_oracle['sub_prec']*100:.1f}%, O={best_oracle['omi_prec']*100:.1f}%, I={best_oracle['ins_prec']*100:.1f}%")

    fwr_sweep_records.append({
        "population": "DEV_COMBINED",
        "nominal_budget": "UNRESTRICTED_ORACLE_PEAK",
        "realized_fwr": best_oracle["realized_fwr"],
        "macro3_f1": best_oracle["macro3_f1"],
        "tau_substitution": best_oracle["tau_s"],
        "tau_omission": best_oracle["tau_o"],
        "tau_insertion": best_oracle["tau_i"],
        "correct_detections": best_oracle["correct"],
        "false_warnings": best_oracle["false_warnings"],
        "wrong_type_warnings": "DIAGNOSTIC_ORACLE",
        "sub_tp": "ORACLE", "sub_fp": "ORACLE", "sub_fn": "ORACLE",
        "sub_precision": best_oracle["sub_prec"], "sub_recall": best_oracle["sub_recall"], "sub_f1": best_oracle["sub_f1"],
        "omi_tp": "ORACLE", "omi_fp": "ORACLE", "omi_fn": "ORACLE",
        "omi_precision": best_oracle["omi_prec"], "omi_recall": best_oracle["omi_recall"], "omi_f1": best_oracle["omi_f1"],
        "ins_tp": "ORACLE", "ins_fp": "ORACLE", "ins_fn": "ORACLE",
        "ins_precision": best_oracle["ins_prec"], "ins_recall": best_oracle["ins_recall"], "ins_f1": best_oracle["ins_f1"],
        "policy_status": "POSTHOC_ORACLE_DIAGNOSTIC_NOT_DEPLOYABLE"
    })

    # Save FWR_SWEEP_RESULTS.csv
    df_fwr = pd.DataFrame(fwr_sweep_records)
    df_fwr.to_csv(OUT_DIR / "FWR_SWEEP_RESULTS.csv", index=False)
    print(f"\n[OK] Wrote FWR_SWEEP_RESULTS.csv ({len(df_fwr)} rows).")

    # Generate continuous PR and FWR curves for TEST_SEALED and DEV_COMBINED
    for c_name, c_pop in [("TEST_SEALED", pops["TEST_SEALED"]), ("DEV_COMBINED", pops["DEV_COMBINED"])]:
        # S curves
        y_s = (c_pop["gold_target"] == 1).astype(int)
        sc_s = c_pop["probs_target_neural"][:, 1]
        p_s, r_s, th_s_arr = precision_recall_curve(y_s, sc_s)
        df_pr_s = pd.DataFrame({"threshold": np.append(th_s_arr, 1.0), "precision": p_s, "recall": r_s})
        df_pr_s.to_csv(CURVES_DIR / f"{c_name}_substitution_pr_curve.csv", index=False)

        # O curves
        y_o = (c_pop["gold_target"] == 2).astype(int)
        sc_o = c_pop["probs_target_fixed"][:, 2]
        p_o, r_o, th_o_arr = precision_recall_curve(y_o, sc_o)
        df_pr_o = pd.DataFrame({"threshold": np.append(th_o_arr, 1.0), "precision": p_o, "recall": r_o})
        df_pr_o.to_csv(CURVES_DIR / f"{c_name}_omission_pr_curve.csv", index=False)

        # I curves
        y_i = (c_pop["gold_gap"] == 1).astype(int)
        sc_i = c_pop["probs_gap"]
        p_i, r_i, th_i_arr = precision_recall_curve(y_i, sc_i)
        df_pr_i = pd.DataFrame({"threshold": np.append(th_i_arr, 1.0), "precision": p_i, "recall": r_i})
        df_pr_i.to_csv(CURVES_DIR / f"{c_name}_insertion_pr_curve.csv", index=False)

    print(f"[OK] Generated continuous PR curves in {CURVES_DIR}")

    # =========================================================================
    # Phase 5: SpeechOcean762 External Validation
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 5: SpeechOcean762 External Cross-Corpus Validation")
    print("="*80)

    so = pops["SPEECHOCEAN"]
    y_human = so["human_phone_scores"]
    p_neu = so["probs_target_neural"]
    p_fix = so["probs_target_fixed"]
    p_correct = p_neu[:, 0]
    p_err = p_neu[:, 1] + p_neu[:, 2]
    gold_gap = so["gold_gap"]
    pg = so["probs_gap"]

    # Stratified slices
    slices = {
        "ALL_5000_UTTERANCES": np.ones(len(y_human), dtype=bool),
        "TRAIN_SPLIT_2500": (so["target_splits"] == "TRAIN"),
        "TEST_SPLIT_2500": (so["target_splits"] == "TEST"),
        "CHILDREN_SPEAKERS": (so["target_ages"] > 0) & (so["target_ages"] < 18),
        "ADULT_SPEAKERS": (so["target_ages"] >= 18),
        "MALE_SPEAKERS": (so["target_genders"] == "M"),
        "FEMALE_SPEAKERS": (so["target_genders"] == "F")
    }

    gap_slices = {
        "ALL_5000_UTTERANCES": np.ones(len(gold_gap), dtype=bool),
        "TRAIN_SPLIT_2500": (so["gap_splits"] == "TRAIN"),
        "TEST_SPLIT_2500": (so["gap_splits"] == "TEST"),
        "CHILDREN_SPEAKERS": None,  # evaluated on targets
        "ADULT_SPEAKERS": None,
        "MALE_SPEAKERS": None,
        "FEMALE_SPEAKERS": None
    }

    so_benchmark_records = []

    for sname, mask_t in slices.items():
        y_h_sub = y_human[mask_t]
        p_c_sub = p_correct[mask_t]
        p_e_sub = p_err[mask_t]

        # Correlations
        pr_val, pr_pval = pearsonr(p_c_sub, y_h_sub)
        sr_val, sr_pval = spearmanr(p_c_sub, y_h_sub)

        # Binary Error: Score < 1.5
        y_1p5 = (y_h_sub < 1.5).astype(int)
        auc_1p5 = roc_auc_score(y_1p5, p_e_sub)
        ap_1p5 = average_precision_score(y_1p5, p_e_sub)

        # Binary Error: Score == 0.0 (Severe)
        y_0p0 = (y_h_sub == 0.0).astype(int)
        auc_0p0 = roc_auc_score(y_0p0, p_e_sub)
        ap_0p0 = average_precision_score(y_0p0, p_e_sub)

        # Binary Error: Score < 2.0 (Any Non-Perfect)
        y_2p0 = (y_h_sub < 2.0).astype(int)
        auc_2p0 = roc_auc_score(y_2p0, p_e_sub)
        ap_2p0 = average_precision_score(y_2p0, p_e_sub)

        # Gap insertion for full / train / test
        mask_g = gap_slices.get(sname)
        if mask_g is not None:
            yg_sub = gold_gap[mask_g]
            pg_sub = pg[mask_g]
            auc_ins = roc_auc_score(yg_sub, pg_sub)
            ap_ins = average_precision_score(yg_sub, pg_sub)
            ins_prev = yg_sub.mean()
        else:
            auc_ins, ap_ins, ins_prev = None, None, None

        so_benchmark_records.append({
            "cohort_slice": sname,
            "target_phonemes": int(mask_t.sum()),
            "pearson_r": pr_val,
            "pearson_p_value": pr_pval,
            "spearman_rho": sr_val,
            "spearman_p_value": sr_pval,
            # Threshold < 1.5
            "prevalence_lt_1p5": y_1p5.mean(),
            "auroc_lt_1p5": auc_1p5,
            "auprc_lt_1p5": ap_1p5,
            # Threshold == 0.0
            "prevalence_eq_0p0": y_0p0.mean(),
            "auroc_eq_0p0": auc_0p0,
            "auprc_eq_0p0": ap_0p0,
            # Threshold < 2.0
            "prevalence_lt_2p0": y_2p0.mean(),
            "auroc_lt_2p0": auc_2p0,
            "auprc_lt_2p0": ap_2p0,
            # Gap Insertion
            "insertion_prevalence": ins_prev,
            "insertion_auroc": auc_ins,
            "insertion_auprc": ap_ins,
            "so_soi_mapping_verdict": "UNAMBIGUOUS_SO_ABSENT_IN_GOLD (scores 0-2 combine S/O; gold EVAL-V3 FWR disallowed)"
        })

        print(f"  Slice {sname:20s}: N={int(mask_t.sum()):5d}, r={pr_val:.4f}, rho={sr_val:.4f}, AUROC(<1.5)={auc_1p5:.4f}, AUROC(Ins)={auc_ins if auc_ins else 0.0:.4f}")

    df_so = pd.DataFrame(so_benchmark_records)
    df_so.to_csv(OUT_DIR / "SPEECHOCEAN_BENCHMARK.csv", index=False)
    print(f"\n[OK] Wrote SPEECHOCEAN_BENCHMARK.csv ({len(df_so)} rows).")

    # =========================================================================
    # Phase 6: Statistical Confidence Intervals (Speaker-Clustered Bootstrap)
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 6: Statistical Bootstrap & Paired Significance Certification")
    print("="*80)

    # Perform on TEST_SEALED and DEV_COMBINED
    for pop_target in ["TEST_SEALED", "DEV_COMBINED"]:
        pop_data = pops[pop_target]
        spks = sorted(set(pop_data["target_speakers"]))
        K = len(spks)
        print(f"\nRunning 10,000 Speaker-Clustered Bootstrap Resamples on {pop_target} ({K} speakers: {spks})...")

        # Precompute per-speaker contingency tables for Asymmetric and Baseline
        spk_data_list = []
        for spk in spks:
            m_t = (pop_data["target_speakers"] == spk)
            m_g = (pop_data["gap_speakers"] == spk)
            n_spk_utts = int(m_g.sum() - m_t.sum())

            yt_s = pop_data["gold_target"][m_t]
            yg_s = pop_data["gold_gap"][m_g]

            # Asymmetric
            warn_s = apply_class_threshold(pop_data["target_keys"][m_t], pop_data["probs_target_neural"][m_t, 1], tau_S, cut_S)
            warn_o = apply_class_threshold(pop_data["target_keys"][m_t], pop_data["probs_target_fixed"][m_t, 2], tau_O, cut_O)
            warn_i = apply_class_threshold(pop_data["gap_keys"][m_g], pop_data["probs_gap"][m_g], tau_I, cut_I)
            dt_asym = resolve_so_conflict(warn_s, warn_o, pop_data["probs_target_neural"][m_t, 1], pop_data["probs_target_fixed"][m_t, 2], tau_S, tau_O, pop_data["target_keys"][m_t])
            dg_asym = warn_i

            asym_tp_s = int(np.sum((yt_s == 1) & (dt_asym == 1)))
            asym_fp_s = int(np.sum((yt_s != 1) & (dt_asym == 1)))
            asym_fn_s = int(np.sum((yt_s == 1) & (dt_asym != 1)))

            asym_tp_o = int(np.sum((yt_s == 2) & (dt_asym == 2)))
            asym_fp_o = int(np.sum((yt_s != 2) & (dt_asym == 2)))
            asym_fn_o = int(np.sum((yt_s == 2) & (dt_asym != 2)))

            asym_tp_i = int(np.sum((yg_s == 1) & (dg_asym == 1)))
            asym_fp_i = int(np.sum((yg_s != 1) & (dg_asym == 1)))
            asym_fn_i = int(np.sum((yg_s == 1) & (dg_asym != 1)))

            # Baseline
            conf_base = pop_data["probs_target_fixed"][m_t, 1:].max(axis=1)
            kind_base = pop_data["probs_target_fixed"][m_t, 1:].argmax(axis=1) + 1
            dt_base = np.where(conf_base >= tau_T_base, kind_base, 0)
            dg_base = pop_data["probs_gap"][m_g] >= tau_G_base

            base_tp_s = int(np.sum((yt_s == 1) & (dt_base == 1)))
            base_fp_s = int(np.sum((yt_s != 1) & (dt_base == 1)))
            base_fn_s = int(np.sum((yt_s == 1) & (dt_base != 1)))

            base_tp_o = int(np.sum((yt_s == 2) & (dt_base == 2)))
            base_fp_o = int(np.sum((yt_s != 2) & (dt_base == 2)))
            base_fn_o = int(np.sum((yt_s == 2) & (dt_base != 2)))

            base_tp_i = int(np.sum((yg_s == 1) & (dg_base == 1)))
            base_fp_i = int(np.sum((yg_s != 1) & (dg_base == 1)))
            base_fn_i = int(np.sum((yg_s == 1) & (dg_base != 1)))

            spk_data_list.append([
                # 0..8: Asymmetric (TP, FP, FN for S, O, I)
                asym_tp_s, asym_fp_s, asym_fn_s,
                asym_tp_o, asym_fp_o, asym_fn_o,
                asym_tp_i, asym_fp_i, asym_fn_i,
                # 9..17: Baseline (TP, FP, FN for S, O, I)
                base_tp_s, base_fp_s, base_fn_s,
                base_tp_o, base_fp_o, base_fn_o,
                base_tp_i, base_fp_i, base_fn_i,
                # 18: Utterance count
                n_spk_utts
            ])

        M = np.array(spk_data_list, dtype=np.float64)  # (K, 19)
        rng = np.random.default_rng(101)
        B_samples = 10000
        idx_resampled = rng.integers(0, K, size=(B_samples, K))  # (10000, K)
        M_resampled = np.take(M, idx_resampled, axis=0).sum(axis=1)  # (10000, 19)

        # Vectorized metrics calculation across 10,000 resamples
        # Asymmetric
        a_tp_s, a_fp_s, a_fn_s = M_resampled[:, 0], M_resampled[:, 1], M_resampled[:, 2]
        a_tp_o, a_fp_o, a_fn_o = M_resampled[:, 3], M_resampled[:, 4], M_resampled[:, 5]
        a_tp_i, a_fp_i, a_fn_i = M_resampled[:, 6], M_resampled[:, 7], M_resampled[:, 8]
        u_counts = M_resampled[:, 18]

        a_f1_s = np.where(2*a_tp_s + a_fp_s + a_fn_s > 0, 2*a_tp_s / (2*a_tp_s + a_fp_s + a_fn_s), 0.0)
        a_f1_o = np.where(2*a_tp_o + a_fp_o + a_fn_o > 0, 2*a_tp_o / (2*a_tp_o + a_fp_o + a_fn_o), 0.0)
        a_f1_i = np.where(2*a_tp_i + a_fp_i + a_fn_i > 0, 2*a_tp_i / (2*a_tp_i + a_fp_i + a_fn_i), 0.0)
        a_macro_f1 = (a_f1_s + a_f1_o + a_f1_i) / 3.0
        a_fwr = (a_fp_s + a_fp_o + a_fp_i) / u_counts
        a_rec_s = np.where(a_tp_s + a_fn_s > 0, a_tp_s / (a_tp_s + a_fn_s), 0.0)
        a_prec_s = np.where(a_tp_s + a_fp_s > 0, a_tp_s / (a_tp_s + a_fp_s), 0.0)

        # Baseline
        b_tp_s, b_fp_s, b_fn_s = M_resampled[:, 9], M_resampled[:, 10], M_resampled[:, 11]
        b_tp_o, b_fp_o, b_fn_o = M_resampled[:, 12], M_resampled[:, 13], M_resampled[:, 14]
        b_tp_i, b_fp_i, b_fn_i = M_resampled[:, 15], M_resampled[:, 16], M_resampled[:, 17]

        b_f1_s = np.where(2*b_tp_s + b_fp_s + b_fn_s > 0, 2*b_tp_s / (2*b_tp_s + b_fp_s + b_fn_s), 0.0)
        b_f1_o = np.where(2*b_tp_o + b_fp_o + b_fn_o > 0, 2*b_tp_o / (2*b_tp_o + b_fp_o + b_fn_o), 0.0)
        b_f1_i = np.where(2*b_tp_i + b_fp_i + b_fn_i > 0, 2*b_tp_i / (2*b_tp_i + b_fp_i + b_fn_i), 0.0)
        b_macro_f1 = (b_f1_s + b_f1_o + b_f1_i) / 3.0
        b_fwr = (b_fp_s + b_fp_o + b_fp_i) / u_counts

        delta_f1 = a_macro_f1 - b_macro_f1
        delta_fwr = a_fwr - b_fwr

        print(f"  --- {pop_target} Bootstrap 95% Confidence Intervals ---")
        print(f"  Asymmetric Macro-3 F1: {np.mean(a_macro_f1):.4f} [95% CI: {np.percentile(a_macro_f1, 2.5):.4f}, {np.percentile(a_macro_f1, 97.5):.4f}]")
        print(f"  Baseline Macro-3 F1:   {np.mean(b_macro_f1):.4f} [95% CI: {np.percentile(b_macro_f1, 2.5):.4f}, {np.percentile(b_macro_f1, 97.5):.4f}]")
        print(f"  Delta Macro-3 F1:      {np.mean(delta_f1):.4f} [95% CI: {np.percentile(delta_f1, 2.5):.4f}, {np.percentile(delta_f1, 97.5):.4f}]")
        pval_f1 = 2 * min(np.mean(delta_f1 <= 0), np.mean(delta_f1 >= 0))
        print(f"    Empirical Paired p-value (Macro F1): p={pval_f1:.4f}")

        print(f"  Asymmetric Joint FWR:  {np.mean(a_fwr):.4f} [95% CI: {np.percentile(a_fwr, 2.5):.4f}, {np.percentile(a_fwr, 97.5):.4f}]")
        print(f"  Baseline Joint FWR:    {np.mean(b_fwr):.4f} [95% CI: {np.percentile(b_fwr, 2.5):.4f}, {np.percentile(b_fwr, 97.5):.4f}]")
        print(f"  Delta Joint FWR:       {np.mean(delta_fwr):.4f} [95% CI: {np.percentile(delta_fwr, 2.5):.4f}, {np.percentile(delta_fwr, 97.5):.4f}]")
        print(f"  Asymmetric Sub Recall: {np.mean(a_rec_s)*100:.2f}% [95% CI: {np.percentile(a_rec_s, 2.5)*100:.2f}%, {np.percentile(a_rec_s, 97.5)*100:.2f}%]")
        print(f"  Asymmetric Sub Prec:   {np.mean(a_prec_s)*100:.2f}% [95% CI: {np.percentile(a_prec_s, 2.5)*100:.2f}%, {np.percentile(a_prec_s, 97.5)*100:.2f}%]")

    print("\nAll Benchmark Phases Successfully Computed!")


if __name__ == "__main__":
    main()
