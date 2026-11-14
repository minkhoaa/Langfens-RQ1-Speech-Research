#!/usr/bin/env python3
"""
LANGFENS RQ1-PHONEME-CLASS-DIAGNOSTIC-01: Comprehensive Full Diagnostic Execution
"""

import sys
import os
import json
import re
import hashlib
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("/home/khoa/Projects/New-Method-Research")
OUT_DIR = ROOT / "experiments/rq1_sm13_phoneme_class_diagnostic_01"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = {
    "Stop": {"P", "B", "T", "D", "K", "G"},
    "Fricative": {"F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"},
    "Affricate": {"CH", "JH"},
    "Nasal": {"M", "N", "NG"},
    "Liquid": {"L", "R", "ER", "AXR", "ERR"},
    "Glide": {"W", "Y"},
    "Vowel": {"AA", "AE", "AH", "AO", "AW", "AX", "AY", "EH", "EY", "IH", "IX", "IY", "OW", "OY", "UH", "UW"}
}

MANNER_ORDER = ["Vowel", "Stop", "Fricative", "Liquid", "Affricate", "Nasal", "Glide"]

FROZEN_POLICY = {
    "tau_S": 0.9493744373321533,
    "tau_O": 0.8574540615081787,
    "tau_I": 0.8866584300994873,
    "baseline_tau_T": 0.848521411895752,
    "baseline_tau_G": 0.8736369609832764
}

def clean_phone(p):
    if not p:
        return ""
    return re.sub(r"[\d\*\+\-\_\`\)\(\'\" ]+", "", str(p)).strip().upper()

def get_manner(p):
    cp = clean_phone(p)
    for cat, pset in CATEGORIES.items():
        if cp in pset:
            return cat
    return "Other"

def get_tie_hash(event_id):
    h = hashlib.sha256(f"LANGFENS-MB07-warning-tie-v1:{event_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big")

def load_cohort(name, manifest_path, feat_path, pred_path, num_utts):
    print(f"Loading {name}...")
    slots = []
    with open(manifest_path) as f:
        for line in f:
            rec = json.loads(line)
            u = rec["utterance_id"]
            spk = rec["speaker_id"]
            for tp in rec["target_phones"]:
                s = tp["slot_idx"]
                p = tp["target_phone"]
                rp = tp.get("realized_phone")
                lbl = tp["label"]
                xmin = tp.get("xmin", 0.0)
                xmax = tp.get("xmax", 0.0)
                dur = max(0.0, xmax - xmin)
                k = f"L2_ARCTIC|{spk}|{u}|target|{s}"
                manner = get_manner(p)
                realized_manner = get_manner(rp) if rp else "None"
                slots.append({
                    "target_key": k,
                    "utterance_id": u,
                    "speaker_id": spk,
                    "slot_idx": s,
                    "target_phone": clean_phone(p),
                    "realized_phone": clean_phone(rp) if rp else "",
                    "label": lbl,
                    "xmin": xmin,
                    "xmax": xmax,
                    "gold_duration": dur,
                    "manner": manner,
                    "realized_manner": realized_manner
                })
    df_manifest = pd.DataFrame(slots)
    
    feat = np.load(feat_path)
    pred = np.load(pred_path)
    
    tkeys = pred["target_keys"]
    assert len(df_manifest) == len(tkeys), f"Manifest vs tkeys mismatch: {len(df_manifest)} vs {len(tkeys)}"
    
    dur_frames = feat["target_X"][:, 48]
    sil_prob = feat["target_X"][:, 44]
    sil_logit = feat["target_X"][:, 45]
    mean_prob = feat["target_X"][:, 41]
    best_comp_prob = feat["target_X"][:, 43]
    charsiu_dur = dur_frames * 0.01 # seconds
    
    p_neural_S = pred["probs_target_neural"][:, 1]
    p_neural_O = pred["probs_target_neural"][:, 2]
    p_fixed_S = pred["probs_target_fixed"][:, 1]
    p_fixed_O = pred["probs_target_fixed"][:, 2]
    e4_S = pred["e4_probs"][:, 1]
    e4_O = pred["e4_probs"][:, 2]
    sllr = pred["sllr"]
    ollr = pred["ollr"]
    gold_target = pred["gold_target"]
    
    gold_gap = pred["gold_gap"] if "gold_gap" in pred else None
    p_gap_I = pred["probs_gap"] if "probs_gap" in pred else None
    
    df = df_manifest.copy()
    df["gold_target"] = gold_target
    df["p_neural_S"] = p_neural_S
    df["p_neural_O"] = p_neural_O
    df["p_fixed_S"] = p_fixed_S
    df["p_fixed_O"] = p_fixed_O
    df["e4_S"] = e4_S
    df["e4_O"] = e4_O
    df["sllr"] = sllr
    df["ollr"] = ollr
    df["dur_frames"] = dur_frames
    df["charsiu_dur"] = charsiu_dur
    df["sil_prob"] = sil_prob
    df["sil_logit"] = sil_logit
    df["mean_prob"] = mean_prob
    df["best_comp_prob"] = best_comp_prob
    
    # Evaluate frozen asymmetric router decisions
    tau_S = FROZEN_POLICY["tau_S"]
    tau_O = FROZEN_POLICY["tau_O"]
    asym_preds = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        trig_S = p_neural_S[i] >= tau_S
        trig_O = p_fixed_O[i] >= tau_O
        if trig_S and trig_O:
            m_S = p_neural_S[i] - tau_S
            m_O = p_fixed_O[i] - tau_O
            if m_S > m_O:
                asym_preds[i] = 1
            elif m_O > m_S:
                asym_preds[i] = 2
            else:
                asym_preds[i] = 1 if (get_tie_hash(i) % 2 == 1) else 2
        elif trig_S:
            asym_preds[i] = 1
        elif trig_O:
            asym_preds[i] = 2
    df["asym_pred_target"] = asym_preds
    
    return {
        "name": name,
        "df": df,
        "num_utts": num_utts,
        "gold_gap": gold_gap,
        "p_gap_I": p_gap_I
    }

def safe_ap_auroc(y_true, y_score):
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan
    ap = float(average_precision_score(y_true, y_score))
    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auroc = np.nan
    return ap, auroc

def get_quantiles(arr):
    if len(arr) == 0:
        return {"p10": np.nan, "p25": np.nan, "p50": np.nan, "p75": np.nan, "p90": np.nan, "p99": np.nan, "max": np.nan}
    return {
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr))
    }

def compute_manner_metrics(cohort):
    name = cohort["name"]
    df = cohort["df"]
    results = []
    
    tau_S = FROZEN_POLICY["tau_S"]
    tau_T_base = FROZEN_POLICY["baseline_tau_T"]
    
    for manner in MANNER_ORDER + ["ALL"]:
        if manner == "ALL":
            sub_df = df
        else:
            sub_df = df[df["manner"] == manner]
            
        n_targets = len(sub_df)
        if n_targets == 0:
            continue
            
        gold_c = int(np.sum(sub_df["gold_target"] == 0))
        gold_s = int(np.sum(sub_df["gold_target"] == 1))
        gold_o = int(np.sum(sub_df["gold_target"] == 2))
        prev_s = float(gold_s / n_targets) if n_targets > 0 else 0.0
        
        y_true_s = (sub_df["gold_target"] == 1).values.astype(int)
        
        # AP / AUROC
        ap_neural, auc_neural = safe_ap_auroc(y_true_s, sub_df["p_neural_S"].values)
        ap_e4, auc_e4 = safe_ap_auroc(y_true_s, sub_df["e4_S"].values)
        ap_fix, auc_fix = safe_ap_auroc(y_true_s, sub_df["p_fixed_S"].values)
        
        # Quantiles
        s_neural = sub_df[sub_df["gold_target"] == 1]["p_neural_S"].values
        c_neural = sub_df[sub_df["gold_target"] == 0]["p_neural_S"].values
        q_s_neural = get_quantiles(s_neural)
        q_c_neural = get_quantiles(c_neural)
        
        s_e4 = sub_df[sub_df["gold_target"] == 1]["e4_S"].values
        c_e4 = sub_df[sub_df["gold_target"] == 0]["e4_S"].values
        q_s_e4 = get_quantiles(s_e4)
        
        s_fix = sub_df[sub_df["gold_target"] == 1]["p_fixed_S"].values
        c_fix = sub_df[sub_df["gold_target"] == 0]["p_fixed_S"].values
        q_s_fix = get_quantiles(s_fix)
        q_c_fix = get_quantiles(c_fix)
        
        # Operational policy performance (asymmetric router)
        asym_preds = sub_df["asym_pred_target"].values
        tp_asym = int(np.sum((asym_preds == 1) & (y_true_s == 1)))
        fp_asym = int(np.sum((asym_preds == 1) & (y_true_s != 1)))
        fn_asym = int(np.sum((asym_preds != 1) & (y_true_s == 1)))
        rec_asym = float(tp_asym / gold_s) if gold_s > 0 else 0.0
        prec_asym = float(tp_asym / (tp_asym + fp_asym)) if (tp_asym + fp_asym) > 0 else 0.0
        f1_asym = float(2 * prec_asym * rec_asym / (prec_asym + rec_asym)) if (prec_asym + rec_asym) > 0 else 0.0
        
        # Direct cut at tau_S on Neural-S
        neural_cut = sub_df["p_neural_S"].values >= tau_S
        tp_ncut = int(np.sum(neural_cut & (y_true_s == 1)))
        fp_ncut = int(np.sum(neural_cut & (y_true_s != 1)))
        fn_ncut = int(np.sum((~neural_cut) & (y_true_s == 1)))
        rec_ncut = float(tp_ncut / gold_s) if gold_s > 0 else 0.0
        prec_ncut = float(tp_ncut / (tp_ncut + fp_ncut)) if (tp_ncut + fp_ncut) > 0 else 0.0
        f1_ncut = float(2 * prec_ncut * rec_ncut / (prec_ncut + rec_ncut)) if (prec_ncut + rec_ncut) > 0 else 0.0
        
        # Direct cut at tau_T on Fixed-Fusion-S
        fix_cut = sub_df["p_fixed_S"].values >= tau_T_base
        tp_fcut = int(np.sum(fix_cut & (y_true_s == 1)))
        fp_fcut = int(np.sum(fix_cut & (y_true_s != 1)))
        fn_fcut = int(np.sum((~fix_cut) & (y_true_s == 1)))
        rec_fcut = float(tp_fcut / gold_s) if gold_s > 0 else 0.0
        prec_fcut = float(tp_fcut / (tp_fcut + fp_fcut)) if (tp_fcut + fp_fcut) > 0 else 0.0
        f1_fcut = float(2 * prec_fcut * rec_fcut / (prec_fcut + rec_fcut)) if (prec_fcut + rec_fcut) > 0 else 0.0
        
        # Direct cut at tau_S on E4-S
        e4_cut = sub_df["e4_S"].values >= tau_S
        tp_ecut = int(np.sum(e4_cut & (y_true_s == 1)))
        fp_ecut = int(np.sum(e4_cut & (y_true_s != 1)))
        fn_ecut = int(np.sum((~e4_cut) & (y_true_s == 1)))
        rec_ecut = float(tp_ecut / gold_s) if gold_s > 0 else 0.0
        prec_ecut = float(tp_ecut / (tp_ecut + fp_ecut)) if (tp_ecut + fp_ecut) > 0 else 0.0
        f1_ecut = float(2 * prec_ecut * rec_ecut / (prec_ecut + rec_ecut)) if (prec_ecut + rec_ecut) > 0 else 0.0

        results.append({
            "cohort": name,
            "manner": manner,
            "n_targets": n_targets,
            "gold_c": gold_c,
            "gold_s": gold_s,
            "gold_o": gold_o,
            "prevalence_s": prev_s,
            "ap_neural": ap_neural,
            "auc_neural": auc_neural,
            "ap_e4": ap_e4,
            "auc_e4": auc_e4,
            "ap_fixed": ap_fix,
            "auc_fixed": auc_fix,
            "q_s_neural": q_s_neural,
            "q_c_neural": q_c_neural,
            "q_s_e4": q_s_e4,
            "q_s_fixed": q_s_fix,
            "q_c_fixed": q_c_fix,
            "asym": {"tp": tp_asym, "fp": fp_asym, "fn": fn_asym, "rec": rec_asym, "prec": prec_asym, "f1": f1_asym},
            "neural_cut": {"tp": tp_ncut, "fp": fp_ncut, "fn": fn_ncut, "rec": rec_ncut, "prec": prec_ncut, "f1": f1_ncut},
            "fixed_cut": {"tp": tp_fcut, "fp": fp_fcut, "fn": fn_fcut, "rec": rec_fcut, "prec": prec_fcut, "f1": f1_fcut},
            "e4_cut": {"tp": tp_ecut, "fp": fp_ecut, "fn": fn_ecut, "rec": rec_ecut, "prec": prec_ecut, "f1": f1_ecut}
        })
    return results

print("Manner metrics module ready.")

def compute_phone_identities(cohort):
    name = cohort["name"]
    df = cohort["df"]
    results = []
    
    target_phones_to_check = [
        ("Vowel", sorted(list(CATEGORIES["Vowel"]))),
        ("Stop", sorted(list(CATEGORIES["Stop"])))
    ]
    
    for manner, plist in target_phones_to_check:
        for p in plist:
            sub_df = df[df["target_phone"] == p]
            n_targets = len(sub_df)
            if n_targets == 0:
                continue
            gold_c = int(np.sum(sub_df["gold_target"] == 0))
            gold_s = int(np.sum(sub_df["gold_target"] == 1))
            gold_o = int(np.sum(sub_df["gold_target"] == 2))
            prev_s = float(gold_s / n_targets) if n_targets > 0 else 0.0
            
            y_true_s = (sub_df["gold_target"] == 1).values.astype(int)
            ap_neural, auc_neural = safe_ap_auroc(y_true_s, sub_df["p_neural_S"].values)
            
            s_neural = sub_df[sub_df["gold_target"] == 1]["p_neural_S"].values
            c_neural = sub_df[sub_df["gold_target"] == 0]["p_neural_S"].values
            q_s = get_quantiles(s_neural)
            q_c = get_quantiles(c_neural)
            
            asym_preds = sub_df["asym_pred_target"].values
            tp_asym = int(np.sum((asym_preds == 1) & (y_true_s == 1)))
            fp_asym = int(np.sum((asym_preds == 1) & (y_true_s != 1)))
            fn_asym = int(np.sum((asym_preds != 1) & (y_true_s == 1)))
            rec_asym = float(tp_asym / gold_s) if gold_s > 0 else 0.0
            prec_asym = float(tp_asym / (tp_asym + fp_asym)) if (tp_asym + fp_asym) > 0 else 0.0
            
            results.append({
                "cohort": name,
                "manner": manner,
                "phone": p,
                "n_targets": n_targets,
                "gold_c": gold_c,
                "gold_s": gold_s,
                "gold_o": gold_o,
                "prevalence_s": prev_s,
                "ap_neural": ap_neural,
                "auc_neural": auc_neural,
                "q_s_p50": q_s["p50"],
                "q_s_p90": q_s["p90"],
                "q_s_max": q_s["max"],
                "q_c_p50": q_c["p50"],
                "q_c_p90": q_c["p90"],
                "q_c_max": q_c["max"],
                "asym_tp": tp_asym,
                "asym_fp": fp_asym,
                "asym_fn": fn_asym,
                "asym_rec": rec_asym,
                "asym_prec": prec_asym
            })
    return results

def compute_confusions(cohort):
    name = cohort["name"]
    df = cohort["df"]
    subs = df[df["gold_target"] == 1]
    
    pair_counts = {}
    manner_transitions = {}
    same_phone_count = 0
    total_subs = len(subs)
    
    for _, row in subs.iterrows():
        tp = row["target_phone"]
        rp = row["realized_phone"]
        tm = row["manner"]
        rm = row["realized_manner"]
        pair = f"{tp}->{rp}"
        trans = f"{tm}->{rm}"
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        manner_transitions[trans] = manner_transitions.get(trans, 0) + 1
        if tp == rp:
            same_phone_count += 1
            
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_trans = sorted(manner_transitions.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "cohort": name,
        "total_subs": total_subs,
        "same_phone_anomalies": same_phone_count,
        "top_pairs": sorted_pairs[:25],
        "manner_transitions": sorted_trans
    }

def compute_alignment_diagnostics(cohort):
    name = cohort["name"]
    df = cohort["df"]
    
    df["abs_dur_diff"] = np.abs(df["charsiu_dur"] - df["gold_duration"])
    df["rel_dur_diff"] = df["abs_dur_diff"] / (df["gold_duration"] + 1e-4)
    
    manner_stats = {}
    for m in MANNER_ORDER:
        sub_df = df[df["manner"] == m]
        if len(sub_df) == 0:
            continue
        g_dur = sub_df["gold_duration"].values
        c_dur = sub_df["charsiu_dur"].values
        diff = sub_df["abs_dur_diff"].values
        
        # Correlations with p_neural_S
        p_neu = sub_df["p_neural_S"].values
        pear_g, _ = stats.pearsonr(g_dur, p_neu) if len(g_dur) > 1 else (np.nan, np.nan)
        spear_g, _ = stats.spearmanr(g_dur, p_neu) if len(g_dur) > 1 else (np.nan, np.nan)
        pear_diff, _ = stats.pearsonr(diff, p_neu) if len(diff) > 1 else (np.nan, np.nan)
        spear_diff, _ = stats.spearmanr(diff, p_neu) if len(diff) > 1 else (np.nan, np.nan)
        
        # Check Gold S specifically
        s_df = sub_df[sub_df["gold_target"] == 1]
        if len(s_df) > 1:
            s_pear_g, _ = stats.pearsonr(s_df["gold_duration"].values, s_df["p_neural_S"].values)
            s_spear_g, _ = stats.spearmanr(s_df["gold_duration"].values, s_df["p_neural_S"].values)
            s_pear_diff, _ = stats.pearsonr(s_df["abs_dur_diff"].values, s_df["p_neural_S"].values)
        else:
            s_pear_g, s_spear_g, s_pear_diff = np.nan, np.nan, np.nan
            
        manner_stats[m] = {
            "n_events": len(sub_df),
            "gold_dur_mean": float(np.mean(g_dur)),
            "gold_dur_median": float(np.median(g_dur)),
            "gold_dur_std": float(np.std(g_dur)),
            "charsiu_dur_mean": float(np.mean(c_dur)),
            "charsiu_dur_median": float(np.median(c_dur)),
            "abs_diff_mean": float(np.mean(diff)),
            "abs_diff_median": float(np.median(diff)),
            "corr_gold_dur_pneu_pearson": float(pear_g),
            "corr_gold_dur_pneu_spearman": float(spear_g),
            "corr_diff_pneu_pearson": float(pear_diff),
            "corr_diff_pneu_spearman": float(spear_diff),
            "s_corr_gold_dur_pneu_pearson": float(s_pear_g),
            "s_corr_diff_pneu_pearson": float(s_pear_diff)
        }
        
    # Error class duration stats
    error_class_stats = {}
    for lbl, code in [("CORRECT", 0), ("SUBSTITUTION", 1), ("OMISSION", 2)]:
        sub_df = df[df["gold_target"] == code]
        error_class_stats[lbl] = {
            "count": len(sub_df),
            "gold_dur_mean": float(np.mean(sub_df["gold_duration"])),
            "gold_dur_median": float(np.median(sub_df["gold_duration"])),
            "charsiu_dur_mean": float(np.mean(sub_df["charsiu_dur"])),
            "charsiu_dur_median": float(np.median(sub_df["charsiu_dur"])),
            "abs_diff_mean": float(np.mean(sub_df["abs_dur_diff"])),
            "sil_prob_mean": float(np.mean(sub_df["sil_prob"])),
            "sil_prob_median": float(np.median(sub_df["sil_prob"])),
            "sil_prob_p90": float(np.percentile(sub_df["sil_prob"], 90))
        }
        
    # SLLR and OLLR behavior
    ctc_stats = {}
    for m in MANNER_ORDER:
        sub_df = df[df["manner"] == m]
        s_df = sub_df[sub_df["gold_target"] == 1]
        if len(s_df) > 0:
            sllr_s = s_df["sllr"].values
            ollr_s = s_df["ollr"].values
            e4_s = s_df["e4_S"].values
            e4_o = s_df["e4_O"].values
            ollr_gt_sllr = float(np.mean(ollr_s > sllr_s))
            ctc_stats[m] = {
                "gold_s_count": len(s_df),
                "sllr_median": float(np.median(sllr_s)),
                "sllr_p10": float(np.percentile(sllr_s, 10)),
                "sllr_p90": float(np.percentile(sllr_s, 90)),
                "ollr_median": float(np.median(ollr_s)),
                "ollr_p10": float(np.percentile(ollr_s, 10)),
                "ollr_p90": float(np.percentile(ollr_s, 90)),
                "e4_s_median": float(np.median(e4_s)),
                "e4_o_median": float(np.median(e4_o)),
                "fraction_ollr_gt_sllr": ollr_gt_sllr
            }
            
    return {
        "cohort": name,
        "manner_duration": manner_stats,
        "error_class_duration": error_class_stats,
        "ctc_sllr_ollr": ctc_stats
    }

print("Diagnostics modules ready.")

def compute_threshold_sweeps(cohort):
    name = cohort["name"]
    df = cohort["df"]
    num_utts = cohort["num_utts"]
    gold_gap = cohort["gold_gap"]
    p_gap_I = cohort["p_gap_I"]
    
    tau_O = FROZEN_POLICY["tau_O"]
    tau_I = FROZEN_POLICY["tau_I"]
    
    # Gap insertions are fixed
    if gold_gap is not None and p_gap_I is not None:
        pred_gap = p_gap_I >= tau_I
        ins_fp = int(np.sum((pred_gap == 1) & (gold_gap == 0)))
        ins_tp = int(np.sum((pred_gap == 1) & (gold_gap == 1)))
        ins_fn = int(np.sum((pred_gap == 0) & (gold_gap == 1)))
    else:
        ins_fp, ins_tp, ins_fn = 0, 0, 0
        
    p_neural_S = df["p_neural_S"].values
    p_fixed_O = df["p_fixed_O"].values
    gold_target = df["gold_target"].values
    manners = df["manner"].values
    
    sweep_results = []
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.9493744, 0.96]
    
    for t_s in thresholds:
        pred_target = np.zeros(len(df), dtype=int)
        for i in range(len(df)):
            trig_S = p_neural_S[i] >= t_s
            trig_O = p_fixed_O[i] >= tau_O
            if trig_S and trig_O:
                m_S = p_neural_S[i] - t_s
                m_O = p_fixed_O[i] - tau_O
                if m_S > m_O:
                    pred_target[i] = 1
                elif m_O > m_S:
                    pred_target[i] = 2
                else:
                    pred_target[i] = 1 if (get_tie_hash(i) % 2 == 1) else 2
            elif trig_S:
                pred_target[i] = 1
            elif trig_O:
                pred_target[i] = 2
                
        sub_tp = int(np.sum((pred_target == 1) & (gold_target == 1)))
        sub_fp = int(np.sum((pred_target == 1) & (gold_target != 1)))
        sub_fn = int(np.sum((pred_target != 1) & (gold_target == 1)))
        sub_rec = float(sub_tp / (sub_tp + sub_fn)) if (sub_tp + sub_fn) > 0 else 0.0
        sub_prec = float(sub_tp / (sub_tp + sub_fp)) if (sub_tp + sub_fp) > 0 else 0.0
        sub_f1 = float(2 * sub_prec * sub_rec / (sub_prec + sub_rec)) if (sub_prec + sub_rec) > 0 else 0.0
        
        omi_tp = int(np.sum((pred_target == 2) & (gold_target == 2)))
        omi_fp = int(np.sum((pred_target == 2) & (gold_target != 2)))
        omi_fn = int(np.sum((pred_target != 2) & (gold_target == 2)))
        omi_rec = float(omi_tp / (omi_tp + omi_fn)) if (omi_tp + omi_fn) > 0 else 0.0
        omi_prec = float(omi_tp / (omi_tp + omi_fp)) if (omi_tp + omi_fp) > 0 else 0.0
        omi_f1 = float(2 * omi_prec * omi_rec / (omi_prec + omi_rec)) if (omi_prec + omi_rec) > 0 else 0.0
        
        ins_rec = float(ins_tp / (ins_tp + ins_fn)) if (ins_tp + ins_fn) > 0 else 0.0
        ins_prec = float(ins_tp / (ins_tp + ins_fp)) if (ins_tp + ins_fp) > 0 else 0.0
        ins_f1 = float(2 * ins_prec * ins_rec / (ins_prec + ins_rec)) if (ins_prec + ins_rec) > 0 else 0.0
        
        macro3_f1 = float((sub_f1 + omi_f1 + ins_f1) / 3.0)
        total_fw = sub_fp + omi_fp + ins_fp
        fwr = float(total_fw / num_utts)
        
        # Manner specific breakdown
        manner_rec = {}
        for m in MANNER_ORDER:
            m_mask = (manners == m)
            m_s_gold = int(np.sum(m_mask & (gold_target == 1)))
            m_s_tp = int(np.sum(m_mask & (pred_target == 1) & (gold_target == 1)))
            m_s_fp = int(np.sum(m_mask & (pred_target == 1) & (gold_target != 1)))
            m_rec = float(m_s_tp / m_s_gold) if m_s_gold > 0 else 0.0
            m_prec = float(m_s_tp / (m_s_tp + m_s_fp)) if (m_s_tp + m_s_fp) > 0 else 0.0
            manner_rec[m] = {
                "gold_s": m_s_gold,
                "tp": m_s_tp,
                "fp": m_s_fp,
                "rec": m_rec,
                "prec": m_prec
            }
            
        sweep_results.append({
            "tau_S": float(t_s),
            "joint_fwr": fwr,
            "total_fw": total_fw,
            "macro3_f1": macro3_f1,
            "sub_rec": sub_rec,
            "sub_prec": sub_prec,
            "sub_f1": sub_f1,
            "sub_tp": sub_tp,
            "sub_fp": sub_fp,
            "omi_f1": omi_f1,
            "manner_metrics": manner_rec
        })
        
    return {
        "cohort": name,
        "sweeps": sweep_results
    }

def main():
    print("Beginning comprehensive diagnostic execution...")
    
    cohorts_meta = [
        ("TRAIN_OOF", "data/derived/l2arctic/manifests/train.jsonl", "data/derived/l2arctic/features/train_features.npz", "experiments/rq1_method_select_02r/agent_a/predictions/predictions_sm13_seed101.npz", 1800),
        ("DEV_CAL", "data/derived/l2arctic/manifests/dev_cal.jsonl", "data/derived/l2arctic/features/dev_cal_features.npz", "experiments/rq1_sm13_dev_inference_01/predictions_sm13_seed101_dev_cal.npz", 300),
        ("DEV_SEARCH", "data/derived/l2arctic/manifests/dev_search.jsonl", "data/derived/l2arctic/features/dev_search_features.npz", "experiments/rq1_sm13_dev_inference_01/predictions_sm13_seed101_dev_search.npz", 599),
        ("DEV_REPLICATION", "data/derived/l2arctic/manifests/dev_replication.jsonl", "data/derived/l2arctic/features/dev_replication_features.npz", "experiments/rq1_sm13_comprehensive_benchmark_01/predictions_sm13_seed101_dev_replication.npz", 300)
    ]
    
    loaded_cohorts = []
    for name, m_path, f_path, p_path, n_utts in cohorts_meta:
        c = load_cohort(name, ROOT / m_path, ROOT / f_path, ROOT / p_path, n_utts)
        loaded_cohorts.append(c)
        
    all_manner_metrics = []
    all_phone_metrics = []
    all_confusions = []
    all_alignments = []
    all_sweeps = []
    
    for c in loaded_cohorts:
        print(f"Processing metrics for {c['name']}...")
        all_manner_metrics.extend(compute_manner_metrics(c))
        all_phone_metrics.extend(compute_phone_identities(c))
        all_confusions.append(compute_confusions(c))
        all_alignments.append(compute_alignment_diagnostics(c))
        if c["name"] in ["DEV_CAL", "DEV_SEARCH", "DEV_REPLICATION"]:
            all_sweeps.append(compute_threshold_sweeps(c))
            
    # Save results
    with open(OUT_DIR / "DIAGNOSTIC_MANNER_METRICS.json", "w") as f:
        json.dump(all_manner_metrics, f, indent=2)
    pd.DataFrame(all_manner_metrics).to_csv(OUT_DIR / "DIAGNOSTIC_MANNER_METRICS.csv", index=False)
    
    with open(OUT_DIR / "DIAGNOSTIC_PHONE_IDENTITIES.json", "w") as f:
        json.dump(all_phone_metrics, f, indent=2)
    pd.DataFrame(all_phone_metrics).to_csv(OUT_DIR / "DIAGNOSTIC_PHONE_IDENTITIES.csv", index=False)
    
    with open(OUT_DIR / "DIAGNOSTIC_CONFUSIONS.json", "w") as f:
        json.dump(all_confusions, f, indent=2)
        
    with open(OUT_DIR / "DIAGNOSTIC_ALIGNMENT_DURATION.json", "w") as f:
        json.dump(all_alignments, f, indent=2)
        
    with open(OUT_DIR / "DIAGNOSTIC_THRESHOLD_SWEEPS.json", "w") as f:
        json.dump(all_sweeps, f, indent=2)
        
    print("All diagnostic calculations complete! Artifacts saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
