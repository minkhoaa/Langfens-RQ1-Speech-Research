#!/usr/bin/env python3
"""
experiments/rq1_sm13_comprehensive_benchmark_01/generate_breakdowns_and_plots.py

Generates fine-grained breakdowns:
1. Accent / L1 breakdown
2. Phoneme manner / type breakdown
3. Utterance duration breakdown
4. Error prevalence breakdown
5. High-resolution PR, FWR, and Pareto frontier plots and CSV curves.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import precision_recall_curve

ROOT = Path("/home/khoa/Projects/New-Method-Research")
OUT_DIR = ROOT / "experiments/rq1_sm13_comprehensive_benchmark_01"
CURVES_DIR = OUT_DIR / "PR_FWR_CURVES"

# Categories
CATEGORIES = {
    "Stop": {"P", "B", "T", "D", "K", "G"},
    "Fricative": {"F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"},
    "Affricate": {"CH", "JH"},
    "Nasal": {"M", "N", "NG"},
    "Liquid": {"L", "R", "ER", "AXR", "ERR"},
    "Glide": {"W", "Y"},
    "Vowel": {"AA", "AE", "AH", "AO", "AW", "AX", "AY", "EH", "EY", "IH", "IX", "IY", "OW", "OY", "UH", "UW"}
}

L1_MAP = {
    "ABA": "Arabic", "SKA": "Arabic", "YBAA": "Arabic", "ZHAA": "Arabic",
    "BWC": "Chinese", "LXC": "Chinese", "NCC": "Chinese", "TXHC": "Chinese",
    "ASI": "Hindi", "SVBI": "Hindi", "RRBI": "Hindi", "TNI": "Hindi",
    "HKK": "Korean", "HJK": "Korean", "YDCK": "Korean", "YKWK": "Korean",
    "EBVS": "Spanish", "NJS": "Spanish", "ERMS": "Spanish", "MBMPS": "Spanish",
    "HQTV": "Vietnamese", "PNV": "Vietnamese", "THV": "Vietnamese", "TLV": "Vietnamese"
}


def clean_phone_token(p: str) -> str:
    p = re.sub(r"[\d\*\+\-\_\`\)\(\'\" ]+", "", p).strip().upper()
    return p


def get_phone_manner(p: str) -> str:
    cp = clean_phone_token(p)
    for cat, pset in CATEGORIES.items():
        if cp in pset:
            return cat
    return "Other"


def main():
    print("Loading predictions and manifests for subgroup breakdowns...")
    manifest_paths = [
        ROOT / "data/derived/l2arctic/manifests/train.jsonl",
        ROOT / "data/derived/l2arctic/manifests/dev_cal.jsonl",
        ROOT / "data/derived/l2arctic/manifests/dev_search.jsonl",
        ROOT / "data/derived/l2arctic/manifests/test_sealed.jsonl",
        ROOT / "data/derived/l2arctic/manifests/dev_replication.jsonl",
    ]

    key_meta = {}
    utt_meta = {}
    for mp in manifest_paths:
        with open(mp) as f:
            for line in f:
                r = json.loads(line)
                uid = r["utterance_id"]
                spk = r["speaker_id"]
                audio_p = ROOT / r["audio_path"]
                dur = sf.info(str(audio_p)).duration if audio_p.exists() else 3.5

                targets = r["target_phones"]
                err_count = sum(1 for tp in targets if tp["label"] in ("SUBSTITUTION", "OMISSION"))
                err_rate = err_count / max(1, len(targets))

                dur_cat = "Short (<3s)" if dur < 3.0 else ("Medium (3-4.5s)" if dur <= 4.5 else "Long (>4.5s)")
                prev_cat = "Low (<5%)" if err_rate < 0.05 else ("Medium (5-15%)" if err_rate <= 0.15 else "High (>15%)")

                utt_meta[uid] = {
                    "duration": dur,
                    "duration_cat": dur_cat,
                    "prevalence_cat": prev_cat,
                    "error_rate": err_rate
                }

                for i, tp in enumerate(targets):
                    k = f"L2_ARCTIC|{spk}|{uid}|target|{i}"
                    key_meta[k] = {
                        "phone": tp["target_phone"],
                        "manner": get_phone_manner(tp["target_phone"]),
                        "is_vowel": get_phone_manner(tp["target_phone"]) == "Vowel",
                        "speaker": spk,
                        "l1": L1_MAP.get(spk, "Unknown"),
                        "duration_cat": dur_cat,
                        "prevalence_cat": prev_cat
                    }

    # Load TEST_SEALED and DEV_COMBINED
    test_sealed = np.load(ROOT / "experiments/rq1_sm13_test_sealed_evaluation_01/predictions_sm13_seed101_test_sealed.npz")
    dev_cal = np.load(ROOT / "experiments/rq1_sm13_dev_inference_01/predictions_sm13_seed101_dev_cal.npz")
    dev_search = np.load(ROOT / "experiments/rq1_sm13_dev_inference_01/predictions_sm13_seed101_dev_search.npz")

    # Evaluate breakdowns on TEST_SEALED
    print("Computing subgroup breakdowns on TEST_SEALED...")
    tkeys = test_sealed["target_keys"]
    yt = test_sealed["gold_target"]
    dt_asym = test_sealed["decisions_target_asymmetric"]
    dt_base = test_sealed["decisions_target_baseline"]
    p_neu = test_sealed["probs_target_neural"]
    p_fix = test_sealed["probs_target_fixed"]

    breakdown_records = []

    def evaluate_subgroup(dim_name: str, group_val: str, mask: np.ndarray):
        sub_yt = yt[mask]
        sub_dt_a = dt_asym[mask]
        sub_dt_b = dt_base[mask]
        n_items = len(sub_yt)

        # Substitution
        s_tp_a = int(np.sum((sub_yt == 1) & (sub_dt_a == 1)))
        s_fp_a = int(np.sum((sub_yt != 1) & (sub_dt_a == 1)))
        s_fn_a = int(np.sum((sub_yt == 1) & (sub_dt_a != 1)))
        s_rec_a = s_tp_a / (s_tp_a + s_fn_a) if (s_tp_a + s_fn_a) else 0.0
        s_prec_a = s_tp_a / (s_tp_a + s_fp_a) if (s_tp_a + s_fp_a) else 0.0
        s_f1_a = 2 * s_tp_a / (2 * s_tp_a + s_fp_a + s_fn_a) if (2 * s_tp_a + s_fp_a + s_fn_a) else 0.0

        s_tp_b = int(np.sum((sub_yt == 1) & (sub_dt_b == 1)))
        s_fp_b = int(np.sum((sub_yt != 1) & (sub_dt_b == 1)))
        s_fn_b = int(np.sum((sub_yt == 1) & (sub_dt_b != 1)))
        s_rec_b = s_tp_b / (s_tp_b + s_fn_b) if (s_tp_b + s_fn_b) else 0.0
        s_prec_b = s_tp_b / (s_tp_b + s_fp_b) if (s_tp_b + s_fp_b) else 0.0
        s_f1_b = 2 * s_tp_b / (2 * s_tp_b + s_fp_b + s_fn_b) if (2 * s_tp_b + s_fp_b + s_fn_b) else 0.0

        # Omission
        o_tp_a = int(np.sum((sub_yt == 2) & (sub_dt_a == 2)))
        o_fp_a = int(np.sum((sub_yt != 2) & (sub_dt_a == 2)))
        o_fn_a = int(np.sum((sub_yt == 2) & (sub_dt_a != 2)))
        o_rec_a = o_tp_a / (o_tp_a + o_fn_a) if (o_tp_a + o_fn_a) else 0.0
        o_prec_a = o_tp_a / (o_tp_a + o_fp_a) if (o_tp_a + o_fp_a) else 0.0
        o_f1_a = 2 * o_tp_a / (2 * o_tp_a + o_fp_a + o_fn_a) if (2 * o_tp_a + o_fp_a + o_fn_a) else 0.0

        o_tp_b = int(np.sum((sub_yt == 2) & (sub_dt_b == 2)))
        o_fp_b = int(np.sum((sub_yt != 2) & (sub_dt_b == 2)))
        o_fn_b = int(np.sum((sub_yt == 2) & (sub_dt_b != 2)))
        o_rec_b = o_tp_b / (o_tp_b + o_fn_b) if (o_tp_b + o_fn_b) else 0.0
        o_prec_b = o_tp_b / (o_tp_b + o_fp_b) if (o_tp_b + o_fp_b) else 0.0
        o_f1_b = 2 * o_tp_b / (2 * o_tp_b + o_fp_b + o_fn_b) if (2 * o_tp_b + o_fp_b + o_fn_b) else 0.0

        breakdown_records.append({
            "dimension": dim_name,
            "group": group_val,
            "target_count": n_items,
            "gold_sub_count": int(np.sum(sub_yt == 1)),
            "gold_omi_count": int(np.sum(sub_yt == 2)),
            "asym_sub_recall": s_rec_a,
            "asym_sub_prec": s_prec_a,
            "asym_sub_f1": s_f1_a,
            "base_sub_recall": s_rec_b,
            "base_sub_prec": s_prec_b,
            "base_sub_f1": s_f1_b,
            "asym_omi_recall": o_rec_a,
            "asym_omi_prec": o_prec_a,
            "asym_omi_f1": o_f1_a,
            "base_omi_recall": o_rec_b,
            "base_omi_prec": o_prec_b,
            "base_omi_f1": o_f1_b,
        })

    # Groupings
    l1s = np.array([key_meta[k]["l1"] for k in tkeys])
    manners = np.array([key_meta[k]["manner"] for k in tkeys])
    vowels = np.array(["Vowel" if key_meta[k]["is_vowel"] else "Consonant" for k in tkeys])
    durs = np.array([key_meta[k]["duration_cat"] for k in tkeys])
    prevs = np.array([key_meta[k]["prevalence_cat"] for k in tkeys])

    # L1
    for l1 in sorted(set(l1s)):
        evaluate_subgroup("Accent_L1", l1, l1s == l1)

    # Vowel vs Consonant
    for vc in ["Vowel", "Consonant"]:
        evaluate_subgroup("Phoneme_Class", vc, vowels == vc)

    # Manners
    for m in sorted(set(manners)):
        evaluate_subgroup("Phoneme_Manner", m, manners == m)

    # Durations
    for d in ["Short (<3s)", "Medium (3-4.5s)", "Long (>4.5s)"]:
        evaluate_subgroup("Utterance_Duration", d, durs == d)

    # Prevalence
    for p in ["Low (<5%)", "Medium (5-15%)", "High (>15%)"]:
        evaluate_subgroup("Error_Prevalence", p, prevs == p)

    df_bd = pd.DataFrame(breakdown_records)
    df_bd.to_csv(OUT_DIR / "L2_ARCTIC_BREAKDOWNS.csv", index=False)
    print(f"[OK] Wrote L2_ARCTIC_BREAKDOWNS.csv ({len(df_bd)} rows).")

    # =========================================================================
    # High-Resolution Publication Plots
    # =========================================================================
    print("Rendering publication plots in PR_FWR_CURVES/...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. TEST_SEALED PR Curves
    plt.figure(figsize=(8, 6), dpi=300)
    for cname, label, col in [("substitution", "Substitution", "#1f77b4"),
                               ("omission", "Omission", "#ff7f0e"),
                               ("insertion", "Insertion", "#2ca02c")]:
        df_c = pd.read_csv(CURVES_DIR / f"TEST_SEALED_{cname}_pr_curve.csv")
        plt.plot(df_c["recall"], df_c["precision"], label=f"{label} (AP: {df_c['precision'].mean():.3f})", color=col, lw=2)

    # Operating point on TEST_SEALED
    plt.scatter([0.0434], [0.7559], color="#1f77b4", s=100, zorder=5, marker="o", label="Asymmetric Op. Point (S: R=4.3%, P=75.6%)")
    plt.scatter([0.1484], [0.6875], color="#ff7f0e", s=100, zorder=5, marker="s", label="Asymmetric Op. Point (O: R=14.8%, P=68.8%)")
    plt.scatter([0.1230], [0.4054], color="#2ca02c", s=100, zorder=5, marker="^", label="Asymmetric Op. Point (I: R=12.3%, P=40.5%)")

    plt.xlabel("Recall", fontsize=12, fontweight="bold")
    plt.ylabel("Precision", fontsize=12, fontweight="bold")
    plt.title("TEST_SEALED Classwise Precision-Recall Curves (LANGFENS-SM13)", fontsize=13, fontweight="bold")
    plt.legend(frameon=True, fontsize=10, loc="upper right")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(CURVES_DIR / "TEST_SEALED_pr_curves.png")
    plt.close()

    # 2. FWR Sensitivity: Macro-3 F1 and S/O/I Recall vs Realized FWR
    df_fwr = pd.read_csv(OUT_DIR / "FWR_SWEEP_RESULTS.csv")
    df_test_fwr = df_fwr[df_fwr["population"] == "TEST_SEALED"].sort_values("realized_fwr")

    plt.figure(figsize=(9, 6), dpi=300)
    plt.plot(df_test_fwr["realized_fwr"], df_test_fwr["macro3_f1"], "k-o", lw=2.5, label="Macro-3 Error F1")
    plt.plot(df_test_fwr["realized_fwr"], df_test_fwr["sub_recall"], "b--s", lw=2, label="Substitution Recall")
    plt.plot(df_test_fwr["realized_fwr"], df_test_fwr["omi_recall"], "r--^", lw=2, label="Omission Recall")
    plt.plot(df_test_fwr["realized_fwr"], df_test_fwr["ins_recall"], "g--d", lw=2, label="Insertion Recall")

    # Mark official ceiling
    plt.axvline(0.20, color="red", linestyle=":", lw=2, label="Safety Ceiling (FWR <= 0.20)")
    # Mark Asymmetric frozen point
    plt.scatter([0.1633], [0.1716], color="purple", s=150, zorder=6, marker="*", label="Frozen Candidate (B=0.12, FWR=0.163)")

    plt.xlabel("Realized Joint False Warning Rate (warnings / utterance)", fontsize=12, fontweight="bold")
    plt.ylabel("Score / Recall", fontsize=12, fontweight="bold")
    plt.title("FWR Sensitivity & Detection Recovery Frontier (TEST_SEALED)", fontsize=13, fontweight="bold")
    plt.legend(frameon=True, fontsize=10, loc="lower right")
    plt.xlim([0.0, 4.0])
    plt.ylim([0.0, 0.70])
    plt.tight_layout()
    plt.savefig(CURVES_DIR / "fwr_sensitivity_tradeoff.png")
    plt.close()

    # 3. Precision vs FWR Degradation
    plt.figure(figsize=(9, 6), dpi=300)
    plt.plot(df_test_fwr["realized_fwr"], df_test_fwr["sub_precision"], "b-o", lw=2, label="Substitution Precision")
    plt.plot(df_test_fwr["realized_fwr"], df_test_fwr["omi_precision"], "r-s", lw=2, label="Omission Precision")
    plt.plot(df_test_fwr["realized_fwr"], df_test_fwr["ins_precision"], "g-^", lw=2, label="Insertion Precision")
    plt.axvline(0.20, color="red", linestyle=":", lw=2, label="Safety Ceiling (FWR <= 0.20)")

    plt.xlabel("Realized Joint False Warning Rate (warnings / utterance)", fontsize=12, fontweight="bold")
    plt.ylabel("Precision", fontsize=12, fontweight="bold")
    plt.title("Precision Decay vs False Warning Budget (TEST_SEALED)", fontsize=13, fontweight="bold")
    plt.legend(frameon=True, fontsize=10, loc="upper right")
    plt.xlim([0.0, 4.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(CURVES_DIR / "precision_vs_fwr_decay.png")
    plt.close()

    print(f"[OK] Rendered 3 publication PNG figures in {CURVES_DIR}")


if __name__ == "__main__":
    main()
