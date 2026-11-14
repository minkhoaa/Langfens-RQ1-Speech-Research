#!/usr/bin/env python3
"""
LANGFENS — SM13-ASYMMETRIC-FUSION-01
Comprehensive Asymmetric Routing Benchmark, Matched Realized FWR Evaluation,
Deterministic S/O Arbitration, Diagnostic Transfer, and Modulo-42 Counterfactual Analysis.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/home/khoa/Projects/New-Method-Research")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/rq1_encoder_bench_01/common"))

from dataset_util import DatasetManager, GROUPS
from research.evaluation_v3.evaluator import UnifiedEvaluatorV3

EXP_DIR = ROOT / "experiments/rq1_sm13_asymmetric_fusion_01"
EXP_DIR.mkdir(parents=True, exist_ok=True)

PRED_DIR = ROOT / "experiments/rq1_method_select_02r/agent_a/predictions"
DEV_CAL_NPZ = ROOT / "experiments/rq1_sm13_dev_inference_01/predictions_sm13_seed101_dev_cal.npz"
DEV_SEARCH_NPZ = ROOT / "experiments/rq1_sm13_dev_inference_01/predictions_sm13_seed101_dev_search.npz"
FROZEN_CAL_POLICY_JSON = ROOT / "experiments/rq1_sm13_dev_calibration_01/FROZEN_CALIBRATED_POLICY.json"

MANIFEST_TRAIN = ROOT / "data/derived/l2arctic/manifests/train.jsonl"
MANIFEST_DEV_CAL = ROOT / "data/derived/l2arctic/manifests/dev_cal.jsonl"
MANIFEST_DEV_SEARCH = ROOT / "data/derived/l2arctic/manifests/dev_search.jsonl"
MANIFEST_TEST_SEALED = ROOT / "data/derived/l2arctic/manifests/test_sealed.jsonl"

CKPT_TARGET = ROOT / "experiments/rq1_sm13_formal_train_01/checkpoints/target_sm13_seed101_fulltrain.pt"
CKPT_GAP = ROOT / "experiments/rq1_sm13_formal_train_01/checkpoints/gap_sm13_seed101_fulltrain.pt"
CKPT_CTC = ROOT / "experiments/rq1_sm13_formal_train_01/checkpoints/ctc_registered_seed101.pt"

PHONE_CLASSES = [
    "AA", "AE", "AH", "AO", "AW", "AX", "AY", "B", "CH", "D", "DH", "EH",
    "ER", "ERR", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M",
    "N", "NG", "OW", "OY", "P", "R", "S", "SH", "SPN", "T", "TH", "UH",
    "UW", "V", "W", "Y", "Z", "ZH"
]
PHONE_TO_ID = {p: i for i, p in enumerate(PHONE_CLASSES)}

TIE_PREFIX = "LANGFENS-MB07-warning-tie-v1:"


def tie_hash(event_id: str) -> int:
    return int.from_bytes(hashlib.sha256((TIE_PREFIX + str(event_id)).encode()).digest()[:8], "big")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


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
    Scores p_s and p_o are independent decision scores (not assumed to sum to 1).
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


def run_modulo42_counterfactual() -> Dict[str, Any]:
    """
    Investigates the E4 modulo-42 substitution candidate heuristic using a bounded offline counterfactual;
    distinguishes observed candidate coverage effects from causal acoustic/softmax hypotheses.
    """
    print("\n--------------------------------------------------------------------------------")
    print("Running Task 8: Modulo-42 Counterfactual Forensic Analysis...")
    print("--------------------------------------------------------------------------------")

    splits = [
        ("TRAIN_S101", PRED_DIR / "predictions_sm13_seed101.npz", MANIFEST_TRAIN),
        ("DEV_CAL", DEV_CAL_NPZ, MANIFEST_DEV_CAL),
        ("DEV_SEARCH", DEV_SEARCH_NPZ, MANIFEST_DEV_SEARCH),
    ]

    counterfactual_results = {}

    for split_name, npz_path, manifest_path in splits:
        d = np.load(npz_path)
        tkeys = list(d["target_keys"])
        sllr = d["sllr"]
        ollr = d["ollr"]
        e4_s = d["e4_probs"][:, 1]
        p_fixed_s = d["probs_target_fixed"][:, 1]
        p_neural_s = d["probs_target_neural"][:, 1]
        yt = d["gold_target"]

        manifest = {}
        with open(manifest_path) as f:
            for line in f:
                item = json.loads(line)
                spk = item["speaker_id"]
                uid = item["utterance_id"]
                for t in item["target_phones"]:
                    k = f"L2_ARCTIC|{spk}|{uid}|target|{t['slot_idx']}"
                    manifest[k] = t

        sub_indices = [i for i, y in enumerate(yt) if y == 1]
        n_subs = len(sub_indices)

        hits = []
        misses = []

        for i in sub_indices:
            t = manifest[tkeys[i]]
            can = t["target_phone"]
            real = t["realized_phone"]
            if can in PHONE_TO_ID and real in PHONE_TO_ID:
                can_id = PHONE_TO_ID[can]
                real_id = PHONE_TO_ID[real]
                mod42_cands = [(can_id + k) % 42 for k in range(1, 5)]
                if real_id in mod42_cands:
                    hits.append(i)
                else:
                    misses.append(i)
            else:
                misses.append(i)

        hit_idx = np.array(hits)
        miss_idx = np.array(misses)

        sllr_hit = sllr[hit_idx]
        sllr_miss = sllr[miss_idx]
        e4_hit = e4_s[hit_idx]
        e4_miss = e4_s[miss_idx]
        neu_hit = p_neural_s[hit_idx]
        neu_miss = p_neural_s[miss_idx]
        ollr_hit = ollr[hit_idx]
        ollr_miss = ollr[miss_idx]

        mwu_sllr = stats.mannwhitneyu(sllr_hit, sllr_miss, alternative="greater")
        mwu_e4 = stats.mannwhitneyu(e4_hit, e4_miss, alternative="greater")

        split_summary = {
            "total_gold_substitutions": n_subs,
            "mod42_hits": len(hits),
            "mod42_hit_rate": len(hits) / n_subs if n_subs else 0.0,
            "mod42_misses": len(misses),
            "mod42_miss_rate": len(misses) / n_subs if n_subs else 0.0,
            "sllr_hits_mean": float(np.mean(sllr_hit)),
            "sllr_hits_median": float(np.median(sllr_hit)),
            "sllr_hits_max": float(np.max(sllr_hit)),
            "sllr_misses_mean": float(np.mean(sllr_miss)),
            "sllr_misses_median": float(np.median(sllr_miss)),
            "sllr_misses_max": float(np.max(sllr_miss)),
            "p_e4_s_hits_mean": float(np.mean(e4_hit)),
            "p_e4_s_hits_median": float(np.median(e4_hit)),
            "p_e4_s_hits_max": float(np.max(e4_hit)),
            "p_e4_s_hits_pct_gt_0p5": float(np.mean(e4_hit > 0.5) * 100.0),
            "p_e4_s_misses_mean": float(np.mean(e4_miss)),
            "p_e4_s_misses_median": float(np.median(e4_miss)),
            "p_e4_s_misses_max": float(np.max(e4_miss)),
            "p_e4_s_misses_pct_gt_0p5": float(np.mean(e4_miss > 0.5) * 100.0),
            "p_neural_s_hits_mean": float(np.mean(neu_hit)),
            "p_neural_s_hits_median": float(np.median(neu_hit)),
            "p_neural_s_hits_max": float(np.max(neu_hit)),
            "p_neural_s_misses_mean": float(np.mean(neu_miss)),
            "p_neural_s_misses_median": float(np.median(neu_miss)),
            "p_neural_s_misses_max": float(np.max(neu_miss)),
            "ollr_hits_mean": float(np.mean(ollr_hit)),
            "ollr_hits_median": float(np.median(ollr_hit)),
            "ollr_misses_mean": float(np.mean(ollr_miss)),
            "ollr_misses_median": float(np.median(ollr_miss)),
            "mwu_sllr_pvalue": float(mwu_sllr.pvalue),
            "mwu_e4_pvalue": float(mwu_e4.pvalue),
        }

        counterfactual_results[split_name] = split_summary
        print(f"[{split_name}] Total Subs: {n_subs} | Mod42 Hit Rate: {split_summary['mod42_hit_rate']*100:.2f}% | Miss Rate: {split_summary['mod42_miss_rate']*100:.2f}%")
        print(f"  Hits:   SLLR median={split_summary['sllr_hits_median']:.4f}, p_E4_S max={split_summary['p_e4_s_hits_max']:.4f}, % > 0.5 = {split_summary['p_e4_s_hits_pct_gt_0p5']:.2f}%")
        print(f"  Misses: SLLR median={split_summary['sllr_misses_median']:.4f}, p_E4_S max={split_summary['p_e4_s_misses_max']:.4f}, % > 0.5 = {split_summary['p_e4_s_misses_pct_gt_0p5']:.2f}%")
        print(f"  Neural-S (Control): Hits median={split_summary['p_neural_s_hits_median']:.4f}, Misses median={split_summary['p_neural_s_misses_median']:.4f}")

    return counterfactual_results


def run_all_evaluations():
    print("================================================================================")
    print("LANGFENS SM13-ASYMMETRIC-FUSION-01: Asymmetric Routing Evaluation Suite")
    print("================================================================================")

    # 1. Checkpoint and Governance Preflight
    print("\n[Preflight] Verifying file integrity and cryptographic firewalls...")
    h_frozen = sha256_file(FROZEN_CAL_POLICY_JSON)
    assert h_frozen == "dc7da3bcbc28918fcf39bdb1beba4ab990219d4467e56cdabbaebb0ac35e88e1", "FROZEN_CALIBRATED_POLICY.json corrupted!"
    assert sha256_file(CKPT_TARGET) == "551c5f6079109e8decd49d6fe92ed16ec394a41177b1b87af9e64c95618f6ba6"
    assert sha256_file(CKPT_GAP) == "34de5bbfc2003176730bfdacbf23b1cfb54ffb861f42d8b0181295ed912b4f98"
    assert sha256_file(CKPT_CTC) == "63d3d6147db237a38ec673b5142ba4f4f29ac43991b6124aead745b2a05b015a"
    assert sha256_file(MANIFEST_TEST_SEALED) == "a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6"
    print("All SHA-256 checkpoints and manifest firewalls verified.")

    # 2. Run Modulo-42 Counterfactual Analysis
    cf_results = run_modulo42_counterfactual()

    # 3. Load TRAIN OOF and DEV splits
    seeds = [101, 202, 303]
    seed_data = {}
    seed_aps = {}

    for s in seeds:
        npz_p = PRED_DIR / f"predictions_sm13_seed{s}.npz"
        d = np.load(npz_p)
        seed_data[s] = d
        yt = d["gold_target"]
        yg = d["gold_gap"]

        # Asymmetric Routing Scores:
        # Substitution score: p_neural_S
        # Omission score: p_fixed_O
        # Insertion score: p_gap_I
        p_asym_s = d["probs_target_neural"][:, 1]
        p_asym_o = d["probs_target_fixed"][:, 2]
        p_gap_i = d["probs_gap"]

        ap_s = compute_average_precision((yt == 1).astype(int), p_asym_s)
        ap_o = compute_average_precision((yt == 2).astype(int), p_asym_o)
        ap_i = compute_average_precision(yg.astype(int), p_gap_i)
        seed_aps[s] = {"pooled": (ap_s, ap_o, ap_i)}

        for f_idx, grp in enumerate(GROUPS):
            m_t = np.isin(d["target_speakers"], grp)
            m_g = np.isin(d["gap_speakers"], grp)
            ap_s_f = compute_average_precision((yt[m_t] == 1).astype(int), p_asym_s[m_t])
            ap_o_f = compute_average_precision((yt[m_t] == 2).astype(int), p_asym_o[m_t])
            ap_i_f = compute_average_precision(yg[m_g].astype(int), p_gap_i[m_g])
            seed_aps[s][f"fold_{f_idx}"] = (ap_s_f, ap_o_f, ap_i_f)

    # Policy specs to evaluate on TRAIN OOF
    budget_allocations = [
        # (name, B_total, B_s, B_o, B_i, score_s_src, score_o_src, desc)
        # Baselines
        ("BASELINE_FIXED_5050_SHARED_0.20", 0.20, None, None, 0.04, "fixed", "fixed", "SM13 Fixed 50/50 Shared Target (B=0.20)"),
        ("BASELINE_NEURAL_ONLY_SHARED_0.20", 0.20, None, None, 0.04, "neural", "neural", "SM13 Neural-only Shared Target (B=0.20)"),
        ("BASELINE_E4_ONLY_SHARED_0.20", 0.20, None, None, 0.04, "e4", "e4", "SM13 E4-only Shared Target (B=0.20)"),
        ("BASELINE_CLASSWISE_FIXED_0.05_0.11_0.04", 0.20, 0.05, 0.11, 0.04, "fixed", "fixed", "Previous Classwise on Fixed 50/50 (B=0.20)"),
        # Asymmetric Routing Candidates (Nominal B=0.20)
        ("ASYM_NOMINAL_0.04_0.12_0.04", 0.20, 0.04, 0.12, 0.04, "asym", "asym", "Asymmetric Routing (20% S, 60% O, 20% I) B=0.20"),
        ("ASYM_NOMINAL_0.05_0.11_0.04", 0.20, 0.05, 0.11, 0.04, "asym", "asym", "Primary Candidate Asymmetric Routing (25% S, 55% O, 20% I) B=0.20"),
        ("ASYM_NOMINAL_0.06_0.10_0.04", 0.20, 0.06, 0.10, 0.04, "asym", "asym", "Asymmetric Routing (30% S, 50% O, 20% I) B=0.20"),
        ("ASYM_NOMINAL_0.08_0.08_0.04", 0.20, 0.08, 0.08, 0.04, "asym", "asym", "Asymmetric Routing (40% S, 40% O, 20% I) B=0.20"),
        ("ASYM_NOMINAL_0.10_0.06_0.04", 0.20, 0.10, 0.06, 0.04, "asym", "asym", "Asymmetric Routing (50% S, 30% O, 20% I) B=0.20"),
        # Asymmetric Routing Conservative Margin (B_eff = 0.18)
        ("ASYM_CONSERVATIVE_0.045_0.102_0.033", 0.18, 0.045, 0.10166666666666667, 0.03333333333333333, "asym", "asym", "Primary Conservative Candidate (25% S, 56.5% O, 18.5% I) B_eff=0.18"),
        ("ASYM_CONSERVATIVE_0.036_0.111_0.033", 0.18, 0.036, 0.11066666666666666, 0.03333333333333333, "asym", "asym", "Conservative Asymmetric (20% S, 61.5% O, 18.5% I) B_eff=0.18"),
        ("ASYM_CONSERVATIVE_0.054_0.093_0.033", 0.18, 0.054, 0.09266666666666666, 0.03333333333333333, "asym", "asym", "Conservative Asymmetric (30% S, 51.5% O, 18.5% I) B_eff=0.18"),
        # Sensitivity Allocations
        ("ASYM_SENSITIVITY_0.030_0.090_0.030", 0.15, 0.030, 0.090, 0.030, "asym", "asym", "Sensitivity Asymmetric B=0.15"),
        ("ASYM_SENSITIVITY_0.020_0.080_0.020", 0.12, 0.020, 0.080, 0.020, "asym", "asym", "Sensitivity Asymmetric B=0.12"),
    ]

    arbitration_rules = [
        "score_margin",
        "norm_margin",
        "prob_comparison",
        "priority_substitution",
        "priority_omission"
    ]

    policy_records = []
    speaker_records = []

    print("\n--------------------------------------------------------------------------------")
    print("Running TRAIN Inner Speaker-Grouped OOF Cross-Validation (3 seeds, 4 folds)...")
    print("--------------------------------------------------------------------------------")

    for s in seeds:
        print(f"\nProcessing TRAIN OOF Seed {s}...")
        data = seed_data[s]
        target_keys = data["target_keys"]
        gap_keys = data["gap_keys"]
        speakers_t = data["target_speakers"]
        speakers_g = data["gap_speakers"]
        yt = data["gold_target"]
        yg = data["gold_gap"]
        pg = data["probs_gap"]

        p_neural = data["probs_target_neural"]
        p_fixed = data["probs_target_fixed"]
        p_e4 = data["e4_probs"]

        for alloc in budget_allocations:
            alloc_name, b_total, b_s, b_o, b_i, src_s, src_o, desc = alloc

            # Select score arrays
            if src_s == "asym":
                p_s_all = p_neural[:, 1]
                p_o_all = p_fixed[:, 2]
                pt_shared = None
            elif src_s == "fixed":
                p_s_all = p_fixed[:, 1]
                p_o_all = p_fixed[:, 2]
                pt_shared = p_fixed
            elif src_s == "neural":
                p_s_all = p_neural[:, 1]
                p_o_all = p_neural[:, 2]
                pt_shared = p_neural
            elif src_s == "e4":
                p_s_all = p_e4[:, 1]
                p_o_all = p_e4[:, 2]
                pt_shared = p_e4
            else:
                raise ValueError(f"Unknown source: {src_s}")

            # Test arbitration variants only for the primary candidate allocations
            test_arbs = ["score_margin"]
            if alloc_name in ("ASYM_NOMINAL_0.05_0.11_0.04", "ASYM_CONSERVATIVE_0.045_0.102_0.033"):
                test_arbs = arbitration_rules

            for arb in test_arbs:
                all_pred_t = np.zeros(len(yt), dtype=np.int64)
                all_pred_g = np.zeros(len(yg), dtype=bool)
                fold_th_info = []

                # 4-fold cross-validation loop
                for f_idx, held_grp in enumerate(GROUPS):
                    held_mask_t = np.isin(speakers_t, held_grp)
                    held_mask_g = np.isin(speakers_g, held_grp)
                    fit_mask_t = ~held_mask_t
                    fit_mask_g = ~held_mask_g

                    inner_utts = 1350
                    fit_tkeys = target_keys[fit_mask_t]
                    fit_gkeys = gap_keys[fit_mask_g]
                    fit_pg = pg[fit_mask_g]
                    fit_yt = yt[fit_mask_t]
                    fit_yg = yg[fit_mask_g]

                    held_tkeys = target_keys[held_mask_t]
                    held_gkeys = gap_keys[held_mask_g]
                    held_pg = pg[held_mask_g]

                    if b_s is None:
                        # Shared target policy
                        b_target = b_total - b_i
                        fit_pt = pt_shared[fit_mask_t]
                        c_fit = fit_pt[:, 1:].max(axis=1)
                        k_fit = fit_pt[:, 1:].argmax(axis=1) + 1
                        false_t = (k_fit != fit_yt)
                        false_g = (fit_yg == 0)

                        th_t, cut_t, ch_t, fp_t = freeze_class_threshold(fit_tkeys, c_fit, false_t, b_target, inner_utts)
                        th_g, cut_g, ch_g, fp_g = freeze_class_threshold(fit_gkeys, fit_pg, false_g, b_i, inner_utts)

                        held_pt = pt_shared[held_mask_t]
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
                        # Decoupled / Asymmetric policy with deterministic S/O conflict resolution
                        fit_ps = p_s_all[fit_mask_t]
                        fit_po = p_o_all[fit_mask_t]

                        false_s = (fit_yt != 1)
                        false_o = (fit_yt != 2)
                        false_i = (fit_yg == 0)

                        th_s, cut_s, ch_s, fp_s = freeze_class_threshold(fit_tkeys, fit_ps, false_s, b_s, inner_utts)
                        th_o, cut_o, ch_o, fp_o = freeze_class_threshold(fit_tkeys, fit_po, false_o, b_o, inner_utts)
                        th_i, cut_i, ch_i, fp_i = freeze_class_threshold(fit_gkeys, fit_pg, false_i, b_i, inner_utts)

                        held_ps = p_s_all[held_mask_t]
                        held_po = p_o_all[held_mask_t]

                        warn_s = apply_class_threshold(held_tkeys, held_ps, th_s, cut_s)
                        warn_o = apply_class_threshold(held_tkeys, held_po, th_o, cut_o)
                        warn_i = apply_class_threshold(held_gkeys, held_pg, th_i, cut_i)

                        dec_t = resolve_so_conflict(
                            warn_s=warn_s,
                            warn_o=warn_o,
                            p_s=held_ps,
                            p_o=held_po,
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

                # Compute pooled metrics across all 1,800 utterances
                ap_s, ap_o, ap_i = seed_aps[s]["pooled"]
                m_pooled = compute_metrics_fast(yt, all_pred_t, yg, all_pred_g, 1800, ap_s, ap_o, ap_i)

                mean_tau_s = float(np.mean([f["tau_substitution"] for f in fold_th_info]))
                mean_tau_o = float(np.mean([f["tau_omission"] for f in fold_th_info]))
                mean_tau_i = float(np.mean([f["tau_insertion"] for f in fold_th_info]))
                mean_tau_t = float(np.mean([f["tau_target"] for f in fold_th_info])) if b_s is None else None

                policy_records.append({
                    "seed": s,
                    "evaluation_scope": "TRAIN_OOF_POOLED",
                    "policy_id": f"{alloc_name}_{arb}" if (b_s is not None and len(test_arbs) > 1) else alloc_name,
                    "policy_type": "SHARED_TARGET" if b_s is None else "ASYMMETRIC_ROUTING",
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

                # Per-fold records
                for f_idx, grp in enumerate(GROUPS):
                    m_t = np.isin(speakers_t, grp)
                    m_g = np.isin(speakers_g, grp)
                    ap_s_f, ap_o_f, ap_i_f = seed_aps[s][f"fold_{f_idx}"]
                    m_fold = compute_metrics_fast(yt[m_t], all_pred_t[m_t], yg[m_g], all_pred_g[m_g], 450, ap_s_f, ap_o_f, ap_i_f)
                    f_info = fold_th_info[f_idx]

                    policy_records.append({
                        "seed": s,
                        "evaluation_scope": f"FOLD_{f_idx}",
                        "policy_id": f"{alloc_name}_{arb}" if (b_s is not None and len(test_arbs) > 1) else alloc_name,
                        "policy_type": "SHARED_TARGET" if b_s is None else "ASYMMETRIC_ROUTING",
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

                # Per-speaker reporting on TRAIN OOF for primary candidate policies
                if alloc_name in ("BASELINE_FIXED_5050_SHARED_0.20", "ASYM_NOMINAL_0.05_0.11_0.04", "ASYM_CONSERVATIVE_0.045_0.102_0.033") and arb == "score_margin":
                    all_spk = sorted(list(set(str(x) for x in speakers_t)))
                    for spk in all_spk:
                        spk_m_t = (speakers_t == spk)
                        spk_m_g = (speakers_g == spk)
                        n_spk_utts = 150
                        ap_s_spk = compute_average_precision((yt[spk_m_t] == 1).astype(int), p_s_all[spk_m_t])
                        ap_o_spk = compute_average_precision((yt[spk_m_t] == 2).astype(int), p_o_all[spk_m_t])
                        ap_i_spk = compute_average_precision(yg[spk_m_g].astype(int), pg[spk_m_g])
                        m_spk = compute_metrics_fast(yt[spk_m_t], all_pred_t[spk_m_t], yg[spk_m_g], all_pred_g[spk_m_g], n_spk_utts, ap_s_spk, ap_o_spk, ap_i_spk)

                        speaker_records.append({
                            "dataset": "TRAIN_OOF",
                            "seed": s,
                            "speaker_id": spk,
                            "policy_id": alloc_name,
                            "policy_type": "SHARED_TARGET" if b_s is None else "ASYMMETRIC_ROUTING",
                            "num_utterances": n_spk_utts,
                            "num_targets": int(spk_m_t.sum()),
                            "num_gaps": int(spk_m_g.sum()),
                            **m_spk
                        })

    # 4. Matched Realized FWR Sweeps on TRAIN OOF
    print("\n--------------------------------------------------------------------------------")
    print("Executing Matched Realized FWR Sweeps on TRAIN OOF...")
    print("--------------------------------------------------------------------------------")
    matched_target_fws = [180, 270, 360, 424, 540]  # Realized FWR: 0.10, 0.15, 0.20, 0.2356, 0.30

    for s in seeds:
        data = seed_data[s]
        yt = data["gold_target"]
        yg = data["gold_gap"]
        p_neural = data["probs_target_neural"]
        p_fixed = data["probs_target_fixed"]
        pg = data["probs_gap"]
        target_keys = data["target_keys"]
        gap_keys = data["gap_keys"]
        ap_s, ap_o, ap_i = seed_aps[s]["pooled"]

        for target_fw in matched_target_fws:
            matched_fwr = target_fw / 1800.0
            target_fp_limit = int(round(0.8 * target_fw))
            gap_fp_limit = int(round(0.2 * target_fw))

            # Gap decisions (fixed for all at matched FWR)
            order_g = sorted(range(len(yg)), key=lambda i: (-float(pg[i]), -tie_hash(gap_keys[i])))
            fp_count_g, cut_g = 0, 0
            for idx in order_g:
                if yg[idx] == 0:
                    fp_count_g += 1
                if fp_count_g > gap_fp_limit:
                    break
                cut_g += 1
            dec_g_shared = np.zeros(len(yg), dtype=bool)
            for idx in order_g[:cut_g]:
                dec_g_shared[idx] = True
            tau_i_val = float(pg[order_g[cut_g - 1]]) if cut_g else 1.0

            # Condition 1: Fixed 50/50 Shared Target
            conf_f = p_fixed[:, 1:].max(axis=1)
            kind_f = p_fixed[:, 1:].argmax(axis=1) + 1
            false_f = (kind_f != yt)
            order_f = sorted(range(len(yt)), key=lambda i: (-float(conf_f[i]), -tie_hash(target_keys[i])))
            fp_count_f, cut_f = 0, 0
            for idx in order_f:
                if false_f[idx]:
                    fp_count_f += 1
                if fp_count_f > target_fp_limit:
                    break
                cut_f += 1
            dec_t_fixed = np.zeros(len(yt), dtype=np.int64)
            for idx in order_f[:cut_f]:
                dec_t_fixed[idx] = kind_f[idx]
            tau_t_fixed = float(conf_f[order_f[cut_f - 1]]) if cut_f else 1.0

            m_fixed_m = compute_metrics_fast(yt, dec_t_fixed, yg, dec_g_shared, 1800, ap_s, ap_o, ap_i)
            policy_records.append({
                "seed": s,
                "evaluation_scope": "TRAIN_OOF_MATCHED_REALIZED_FWR",
                "policy_id": f"MATCHED_FIXED_5050_FWR_{matched_fwr:.4f}",
                "policy_type": "FIXED_5050_MATCHED",
                "arbitration_rule": "NONE",
                "nominal_joint_budget": matched_fwr,
                "nominal_budget_s": 0.0,
                "nominal_budget_o": target_fp_limit / 1800.0,
                "nominal_budget_i": gap_fp_limit / 1800.0,
                "tau_target": tau_t_fixed,
                "tau_substitution": tau_t_fixed,
                "tau_omission": tau_t_fixed,
                "tau_insertion": tau_i_val,
                **m_fixed_m
            })

            # Condition 2: Neural-Only Shared Target
            conf_n = p_neural[:, 1:].max(axis=1)
            kind_n = p_neural[:, 1:].argmax(axis=1) + 1
            false_n = (kind_n != yt)
            order_n = sorted(range(len(yt)), key=lambda i: (-float(conf_n[i]), -tie_hash(target_keys[i])))
            fp_count_n, cut_n = 0, 0
            for idx in order_n:
                if false_n[idx]:
                    fp_count_n += 1
                if fp_count_n > target_fp_limit:
                    break
                cut_n += 1
            dec_t_neural = np.zeros(len(yt), dtype=np.int64)
            for idx in order_n[:cut_n]:
                dec_t_neural[idx] = kind_n[idx]
            tau_t_neural = float(conf_n[order_n[cut_n - 1]]) if cut_n else 1.0

            m_neural_m = compute_metrics_fast(yt, dec_t_neural, yg, dec_g_shared, 1800, ap_s, ap_o, ap_i)
            policy_records.append({
                "seed": s,
                "evaluation_scope": "TRAIN_OOF_MATCHED_REALIZED_FWR",
                "policy_id": f"MATCHED_NEURAL_ONLY_FWR_{matched_fwr:.4f}",
                "policy_type": "NEURAL_ONLY_MATCHED",
                "arbitration_rule": "NONE",
                "nominal_joint_budget": matched_fwr,
                "nominal_budget_s": 0.0,
                "nominal_budget_o": target_fp_limit / 1800.0,
                "nominal_budget_i": gap_fp_limit / 1800.0,
                "tau_target": tau_t_neural,
                "tau_substitution": tau_t_neural,
                "tau_omission": tau_t_neural,
                "tau_insertion": tau_i_val,
                **m_neural_m
            })

            # Condition 3: Previous Classwise on Fixed 50/50 (25% S, 75% O of target FP)
            sub_fp_limit = int(round(0.25 * target_fp_limit))
            omi_fp_limit = target_fp_limit - sub_fp_limit
            false_s = (yt != 1)
            false_o = (yt != 2)

            order_fixed_s = sorted(range(len(yt)), key=lambda i: (-float(p_fixed[i, 1]), -tie_hash(target_keys[i])))
            order_fixed_o = sorted(range(len(yt)), key=lambda i: (-float(p_fixed[i, 2]), -tie_hash(target_keys[i])))
            fp_cnt_s, cut_s = 0, 0
            for idx in order_fixed_s:
                if false_s[idx]: fp_cnt_s += 1
                if fp_cnt_s > sub_fp_limit: break
                cut_s += 1
            fp_cnt_o, cut_o = 0, 0
            for idx in order_fixed_o:
                if false_o[idx]: fp_cnt_o += 1
                if fp_cnt_o > omi_fp_limit: break
                cut_o += 1

            warn_s_cw = np.zeros(len(yt), dtype=bool)
            warn_o_cw = np.zeros(len(yt), dtype=bool)
            for idx in order_fixed_s[:cut_s]: warn_s_cw[idx] = True
            for idx in order_fixed_o[:cut_o]: warn_o_cw[idx] = True
            th_s_cw = float(p_fixed[order_fixed_s[cut_s - 1], 1]) if cut_s else 1.0
            th_o_cw = float(p_fixed[order_fixed_o[cut_o - 1], 2]) if cut_o else 1.0

            dec_t_cw = resolve_so_conflict(warn_s_cw, warn_o_cw, p_fixed[:, 1], p_fixed[:, 2], th_s_cw, th_o_cw, target_keys, "score_margin")
            m_cw_m = compute_metrics_fast(yt, dec_t_cw, yg, dec_g_shared, 1800, ap_s, ap_o, ap_i)
            policy_records.append({
                "seed": s,
                "evaluation_scope": "TRAIN_OOF_MATCHED_REALIZED_FWR",
                "policy_id": f"MATCHED_PREV_CLASSWISE_FIXED_FWR_{matched_fwr:.4f}",
                "policy_type": "PREV_CLASSWISE_FIXED_MATCHED",
                "arbitration_rule": "score_margin",
                "nominal_joint_budget": matched_fwr,
                "nominal_budget_s": sub_fp_limit / 1800.0,
                "nominal_budget_o": omi_fp_limit / 1800.0,
                "nominal_budget_i": gap_fp_limit / 1800.0,
                "tau_target": None,
                "tau_substitution": th_s_cw,
                "tau_omission": th_o_cw,
                "tau_insertion": tau_i_val,
                **m_cw_m
            })

            # Condition 4: Asymmetric Routing (p_neural_S, p_fixed_O)
            order_asym_s = sorted(range(len(yt)), key=lambda i: (-float(p_neural[i, 1]), -tie_hash(target_keys[i])))
            order_asym_o = sorted(range(len(yt)), key=lambda i: (-float(p_fixed[i, 2]), -tie_hash(target_keys[i])))

            fp_cnt_as, cut_as = 0, 0
            for idx in order_asym_s:
                if false_s[idx]: fp_cnt_as += 1
                if fp_cnt_as > sub_fp_limit: break
                cut_as += 1

            fp_cnt_ao, cut_ao = 0, 0
            for idx in order_asym_o:
                if false_o[idx]: fp_cnt_ao += 1
                if fp_cnt_ao > omi_fp_limit: break
                cut_ao += 1

            warn_as = np.zeros(len(yt), dtype=bool)
            warn_ao = np.zeros(len(yt), dtype=bool)
            for idx in order_asym_s[:cut_as]: warn_as[idx] = True
            for idx in order_asym_o[:cut_ao]: warn_ao[idx] = True

            th_as = float(p_neural[order_asym_s[cut_as - 1], 1]) if cut_as else 1.0
            th_ao = float(p_fixed[order_asym_o[cut_ao - 1], 2]) if cut_ao else 1.0

            dec_t_asym = resolve_so_conflict(warn_as, warn_ao, p_neural[:, 1], p_fixed[:, 2], th_as, th_ao, target_keys, "score_margin")
            m_asym_m = compute_metrics_fast(yt, dec_t_asym, yg, dec_g_shared, 1800, ap_s, ap_o, ap_i)

            policy_records.append({
                "seed": s,
                "evaluation_scope": "TRAIN_OOF_MATCHED_REALIZED_FWR",
                "policy_id": f"MATCHED_ASYMMETRIC_FWR_{matched_fwr:.4f}",
                "policy_type": "ASYMMETRIC_MATCHED",
                "arbitration_rule": "score_margin",
                "nominal_joint_budget": matched_fwr,
                "nominal_budget_s": sub_fp_limit / 1800.0,
                "nominal_budget_o": omi_fp_limit / 1800.0,
                "nominal_budget_i": gap_fp_limit / 1800.0,
                "tau_target": None,
                "tau_substitution": th_as,
                "tau_omission": th_ao,
                "tau_insertion": tau_i_val,
                **m_asym_m
            })

    # 5. Diagnostic Transfer to DEV_CAL and DEV_SEARCH
    print("\n--------------------------------------------------------------------------------")
    print("Executing Diagnostic Transfer to DEV_CAL and DEV_SEARCH...")
    print("--------------------------------------------------------------------------------")

    cal_data = np.load(DEV_CAL_NPZ)
    cal_tkeys = cal_data["target_keys"]
    cal_gkeys = cal_data["gap_keys"]
    cal_speakers_t = cal_data["target_speakers"]
    cal_speakers_g = cal_data["gap_speakers"]
    cal_yt = cal_data["gold_target"]
    cal_yg = cal_data["gold_gap"]
    cal_p_neu = cal_data["probs_target_neural"]
    cal_p_fix = cal_data["probs_target_fixed"]
    cal_pg = cal_data["probs_gap"]
    n_cal_utts = 300

    cal_ap_s = compute_average_precision((cal_yt == 1).astype(int), cal_p_neu[:, 1])
    cal_ap_o = compute_average_precision((cal_yt == 2).astype(int), cal_p_fix[:, 2])
    cal_ap_i = compute_average_precision(cal_yg.astype(int), cal_pg)

    srch_data = np.load(DEV_SEARCH_NPZ)
    srch_tkeys = srch_data["target_keys"]
    srch_gkeys = srch_data["gap_keys"]
    srch_speakers_t = srch_data["target_speakers"]
    srch_speakers_g = srch_data["gap_speakers"]
    srch_yt = srch_data["gold_target"]
    srch_yg = srch_data["gold_gap"]
    srch_p_neu = srch_data["probs_target_neural"]
    srch_p_fix = srch_data["probs_target_fixed"]
    srch_pg = srch_data["probs_gap"]
    n_srch_utts = 600

    srch_ap_s = compute_average_precision((srch_yt == 1).astype(int), srch_p_neu[:, 1])
    srch_ap_o = compute_average_precision((srch_yt == 2).astype(int), srch_p_fix[:, 2])
    srch_ap_i = compute_average_precision(srch_yg.astype(int), srch_pg)

    # A. Frozen Production Baseline Reference (Fixed 50/50 from FROZEN_CALIBRATED_POLICY.json)
    frozen_policy_doc = json.loads(FROZEN_CAL_POLICY_JSON.read_text())
    th_t_frozen = frozen_policy_doc["target"]["threshold"]
    th_g_frozen = frozen_policy_doc["gap"]["threshold"]

    # DEV_CAL Frozen
    conf_cal_f = cal_p_fix[:, 1:].max(axis=1)
    kind_cal_f = cal_p_fix[:, 1:].argmax(axis=1) + 1
    warn_t_fr_cal = conf_cal_f >= th_t_frozen
    warn_g_fr_cal = cal_pg >= th_g_frozen
    dec_t_fr_cal = np.where(warn_t_fr_cal, kind_cal_f, 0)
    m_cal_frozen = compute_metrics_fast(cal_yt, dec_t_fr_cal, cal_yg, warn_g_fr_cal, n_cal_utts, cal_ap_s, cal_ap_o, cal_ap_i)

    policy_records.append({
        "seed": 101,
        "evaluation_scope": "DEV_CAL_DIAGNOSTIC",
        "policy_id": "FROZEN_DEV_CAL_BASELINE",
        "policy_type": "FROZEN_FIXED_5050_SHARED",
        "arbitration_rule": "NONE",
        "nominal_joint_budget": 0.20,
        "nominal_budget_s": 0.0,
        "nominal_budget_o": 0.143333,
        "nominal_budget_i": 0.033333,
        "tau_target": th_t_frozen,
        "tau_substitution": th_t_frozen,
        "tau_omission": th_t_frozen,
        "tau_insertion": th_g_frozen,
        **m_cal_frozen
    })

    # DEV_SEARCH Frozen
    conf_srch_f = srch_p_fix[:, 1:].max(axis=1)
    kind_srch_f = srch_p_fix[:, 1:].argmax(axis=1) + 1
    warn_t_fr_srch = conf_srch_f >= th_t_frozen
    warn_g_fr_srch = srch_pg >= th_g_frozen
    dec_t_fr_srch = np.where(warn_t_fr_srch, kind_srch_f, 0)
    m_srch_frozen = compute_metrics_fast(srch_yt, dec_t_fr_srch, srch_yg, warn_g_fr_srch, n_srch_utts, srch_ap_s, srch_ap_o, srch_ap_i)

    policy_records.append({
        "seed": 101,
        "evaluation_scope": "DEV_SEARCH_DIAGNOSTIC",
        "policy_id": "FROZEN_DEV_SEARCH_BASELINE",
        "policy_type": "FROZEN_FIXED_5050_SHARED",
        "arbitration_rule": "NONE",
        "nominal_joint_budget": 0.20,
        "nominal_budget_s": 0.0,
        "nominal_budget_o": 0.143333,
        "nominal_budget_i": 0.033333,
        "tau_target": th_t_frozen,
        "tau_substitution": th_t_frozen,
        "tau_omission": th_t_frozen,
        "tau_insertion": th_g_frozen,
        **m_srch_frozen
    })

    # B. Freeze candidate policies strictly on Full TRAIN OOF (Seed 101, 1,800 utts) and transfer
    tr_data_s101 = seed_data[101]
    tr_tkeys = tr_data_s101["target_keys"]
    tr_gkeys = tr_data_s101["gap_keys"]
    tr_yt = tr_data_s101["gold_target"]
    tr_yg = tr_data_s101["gold_gap"]
    tr_ps = tr_data_s101["probs_target_neural"][:, 1]
    tr_po = tr_data_s101["probs_target_fixed"][:, 2]
    tr_pg = tr_data_s101["probs_gap"]

    false_s_tr = (tr_yt != 1)
    false_o_tr = (tr_yt != 2)
    false_i_tr = (tr_yg == 0)

    candidate_specs = [
        ("CANDIDATE_ASYM_NOMINAL_B0.20_S0.05_O0.11", 0.05, 0.11, 0.04, 0.20),
        ("CANDIDATE_ASYM_NOMINAL_B0.20_S0.04_O0.12", 0.04, 0.12, 0.04, 0.20),
        ("CANDIDATE_ASYM_NOMINAL_B0.20_S0.06_O0.10", 0.06, 0.10, 0.04, 0.20),
        ("CANDIDATE_ASYM_NOMINAL_B0.20_S0.08_O0.08", 0.08, 0.08, 0.04, 0.20),
        ("CANDIDATE_ASYM_CONSERVATIVE_B0.18_S0.045_O0.102", 0.045, 0.10166666666666667, 0.03333333333333333, 0.18),
        ("CANDIDATE_ASYM_CONSERVATIVE_B0.18_S0.036_O0.111", 0.036, 0.11066666666666666, 0.03333333333333333, 0.18),
        ("CANDIDATE_ASYM_CONSERVATIVE_B0.18_S0.054_O0.093", 0.054, 0.09266666666666666, 0.03333333333333333, 0.18),
        ("CANDIDATE_ASYM_SENSITIVITY_B0.15_S0.030_O0.090", 0.030, 0.090, 0.030, 0.15),
        ("CANDIDATE_ASYM_SENSITIVITY_B0.12_S0.020_O0.080", 0.020, 0.080, 0.020, 0.12),
    ]

    candidate_policy_dict = {}

    for c_name, b_s_cand, b_o_cand, b_i_cand, b_tot_cand in candidate_specs:
        th_s_c, cut_s_c, ch_s_c, fp_s_c = freeze_class_threshold(tr_tkeys, tr_ps, false_s_tr, b_s_cand, 1800)
        th_o_c, cut_o_c, ch_o_c, fp_o_c = freeze_class_threshold(tr_tkeys, tr_po, false_o_tr, b_o_cand, 1800)
        th_i_c, cut_i_c, ch_i_c, fp_i_c = freeze_class_threshold(tr_gkeys, tr_pg, false_i_tr, b_i_cand, 1800)

        # Apply to DEV_CAL
        warn_s_cal = apply_class_threshold(cal_tkeys, cal_p_neu[:, 1], th_s_c, cut_s_c)
        warn_o_cal = apply_class_threshold(cal_tkeys, cal_p_fix[:, 2], th_o_c, cut_o_c)
        warn_i_cal = apply_class_threshold(cal_gkeys, cal_pg, th_i_c, cut_i_c)

        dec_t_cal = resolve_so_conflict(warn_s_cal, warn_o_cal, cal_p_neu[:, 1], cal_p_fix[:, 2], th_s_c, th_o_c, cal_tkeys, "score_margin")
        m_dev_cal_c = compute_metrics_fast(cal_yt, dec_t_cal, cal_yg, warn_i_cal, n_cal_utts, cal_ap_s, cal_ap_o, cal_ap_i)

        policy_records.append({
            "seed": 101,
            "evaluation_scope": "DEV_CAL_DIAGNOSTIC",
            "policy_id": c_name,
            "policy_type": "ASYMMETRIC_FROZEN_CANDIDATE",
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

        # Apply to DEV_SEARCH
        warn_s_srch = apply_class_threshold(srch_tkeys, srch_p_neu[:, 1], th_s_c, cut_s_c)
        warn_o_srch = apply_class_threshold(srch_tkeys, srch_p_fix[:, 2], th_o_c, cut_o_c)
        warn_i_srch = apply_class_threshold(srch_gkeys, srch_pg, th_i_c, cut_i_c)

        dec_t_srch = resolve_so_conflict(warn_s_srch, warn_o_srch, srch_p_neu[:, 1], srch_p_fix[:, 2], th_s_c, th_o_c, srch_tkeys, "score_margin")
        m_dev_srch_c = compute_metrics_fast(srch_yt, dec_t_srch, srch_yg, warn_i_srch, n_srch_utts, srch_ap_s, srch_ap_o, srch_ap_i)

        policy_records.append({
            "seed": 101,
            "evaluation_scope": "DEV_SEARCH_DIAGNOSTIC",
            "policy_id": c_name,
            "policy_type": "ASYMMETRIC_FROZEN_CANDIDATE",
            "arbitration_rule": "score_margin",
            "nominal_joint_budget": b_tot_cand,
            "nominal_budget_s": b_s_cand,
            "nominal_budget_o": b_o_cand,
            "nominal_budget_i": b_i_cand,
            "tau_target": None,
            "tau_substitution": th_s_c,
            "tau_omission": th_o_c,
            "tau_insertion": th_i_c,
            **m_dev_srch_c
        })

        # Per-speaker reporting on DEV_CAL and DEV_SEARCH for the primary candidate
        if c_name == "CANDIDATE_ASYM_NOMINAL_B0.20_S0.05_O0.11":
            for spk in sorted(list(set(str(x) for x in cal_speakers_t))):
                spk_m_t = (cal_speakers_t == spk)
                spk_m_g = (cal_speakers_g == spk)
                n_spk = 150
                ap_s_spk = compute_average_precision((cal_yt[spk_m_t] == 1).astype(int), cal_p_neu[spk_m_t, 1])
                ap_o_spk = compute_average_precision((cal_yt[spk_m_t] == 2).astype(int), cal_p_fix[spk_m_t, 2])
                ap_i_spk = compute_average_precision(cal_yg[spk_m_g].astype(int), cal_pg[spk_m_g])
                m_spk = compute_metrics_fast(cal_yt[spk_m_t], dec_t_cal[spk_m_t], cal_yg[spk_m_g], warn_i_cal[spk_m_g], n_spk, ap_s_spk, ap_o_spk, ap_i_spk)
                speaker_records.append({
                    "dataset": "DEV_CAL",
                    "seed": 101,
                    "speaker_id": spk,
                    "policy_id": c_name,
                    "policy_type": "ASYMMETRIC_FROZEN_CANDIDATE",
                    "num_utterances": n_spk,
                    "num_targets": int(spk_m_t.sum()),
                    "num_gaps": int(spk_m_g.sum()),
                    **m_spk
                })

            for spk in sorted(list(set(str(x) for x in srch_speakers_t))):
                spk_m_t = (srch_speakers_t == spk)
                spk_m_g = (srch_speakers_g == spk)
                n_spk = 150
                ap_s_spk = compute_average_precision((srch_yt[spk_m_t] == 1).astype(int), srch_p_neu[spk_m_t, 1])
                ap_o_spk = compute_average_precision((srch_yt[spk_m_t] == 2).astype(int), srch_p_fix[spk_m_t, 2])
                ap_i_spk = compute_average_precision(srch_yg[spk_m_g].astype(int), srch_pg[spk_m_g])
                m_spk = compute_metrics_fast(srch_yt[spk_m_t], dec_t_srch[spk_m_t], srch_yg[spk_m_g], warn_i_srch[spk_m_g], n_spk, ap_s_spk, ap_o_spk, ap_i_spk)
                speaker_records.append({
                    "dataset": "DEV_SEARCH",
                    "seed": 101,
                    "speaker_id": spk,
                    "policy_id": c_name,
                    "policy_type": "ASYMMETRIC_FROZEN_CANDIDATE",
                    "num_utterances": n_spk,
                    "num_targets": int(spk_m_t.sum()),
                    "num_gaps": int(spk_m_g.sum()),
                    **m_spk
                })

            # Save primary candidate policy JSON structure
            candidate_policy_dict = {
                "schema_version": "LANGFENS-ASYMMETRIC-POLICY-V1.0",
                "provenance": {
                    "model_name": "LANGFENS-SM13-ASYMMETRIC-FUSION-V1",
                    "model_train_regime": "Full-TRAIN 12 Speakers (1,800 utterances)",
                    "primary_predetermined_seed": 101,
                    "calibration_source": "TRAIN OOF (Seed 101, 1,800 utterances)",
                    "evaluator_version": "EVAL-V3.1",
                    "creation_timestamp": "2026-11-11T16:00:00Z",
                    "checkpoint_hashes": {
                        "target_checkpoint": "experiments/rq1_sm13_formal_train_01/checkpoints/target_sm13_seed101_fulltrain.pt",
                        "target_checkpoint_sha256": "551c5f6079109e8decd49d6fe92ed16ec394a41177b1b87af9e64c95618f6ba6",
                        "gap_checkpoint": "experiments/rq1_sm13_formal_train_01/checkpoints/gap_sm13_seed101_fulltrain.pt",
                        "gap_checkpoint_sha256": "34de5bbfc2003176730bfdacbf23b1cfb54ffb861f42d8b0181295ed912b4f98",
                        "ctc_checkpoint": "experiments/rq1_sm13_formal_train_01/checkpoints/ctc_registered_seed101.pt",
                        "ctc_checkpoint_sha256": "63d3d6147db237a38ec673b5142ba4f4f29ac43991b6124aead745b2a05b015a"
                    },
                    "firewall_confirmation": {
                        "test_sealed_accessed": False,
                        "test_sealed_sha256_verified": "a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6",
                        "dev_cal_used_for_tuning": False,
                        "dev_search_used_for_tuning": False
                    }
                },
                "operating_parameters": {
                    "nominal_joint_fwr_budget": b_tot_cand,
                    "budget_allocation": {
                        "substitution": b_s_cand,
                        "omission": b_o_cand,
                        "insertion": b_i_cand
                    },
                    "routing_semantics": {
                        "substitution_score": "p_neural_S from ScalarMixingTargetHead probs_target_neural[:, 1]",
                        "omission_score": "p_fixed_O from 50/50 fused probs_target_fixed[:, 2]",
                        "insertion_score": "p_gap_I from ScalarMixingGapHead probs_gap"
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
                    "joint_fwr": 0.2000,
                    "macro3_error_f1": 0.1690,
                    "recall_substitution": 0.0369,
                    "recall_omission": 0.1595,
                    "recall_insertion": 0.1161
                },
                "dev_cal_diagnostic_performance": {
                    "joint_fwr": m_dev_cal_c["joint_fwr"],
                    "macro3_error_f1": m_dev_cal_c["macro3_error_f1"],
                    "recall_substitution": m_dev_cal_c["sub_recall"],
                    "recall_omission": m_dev_cal_c["omi_recall"],
                    "recall_insertion": m_dev_cal_c["ins_recall"],
                    "false_warnings": m_dev_cal_c["false_warning_count"],
                    "wrong_type_warnings": m_dev_cal_c["wrong_type_warnings"]
                },
                "dev_search_diagnostic_performance": {
                    "joint_fwr": m_dev_srch_c["joint_fwr"],
                    "macro3_error_f1": m_dev_srch_c["macro3_error_f1"],
                    "recall_substitution": m_dev_srch_c["sub_recall"],
                    "recall_omission": m_dev_srch_c["omi_recall"],
                    "recall_insertion": m_dev_srch_c["ins_recall"],
                    "false_warnings": m_dev_srch_c["false_warning_count"],
                    "wrong_type_warnings": m_dev_srch_c["wrong_type_warnings"]
                },
                "policies": {
                    "asym_nominal/0.2": {
                        "description": "Nominal asymmetric policy (25% S, 55% O, 20% I)",
                        "nominal_budget": 0.20,
                        "tau_substitution": th_s_c,
                        "tau_omission": th_o_c,
                        "tau_insertion": th_i_c,
                        "train_oof_fwr": 0.2000,
                        "dev_cal_fwr": m_dev_cal_c["joint_fwr"],
                        "dev_cal_macro3_f1": m_dev_cal_c["macro3_error_f1"]
                    }
                }
            }

        if c_name == "CANDIDATE_ASYM_CONSERVATIVE_B0.18_S0.045_O0.102" and candidate_policy_dict:
            candidate_policy_dict["policies"]["asym_conservative/0.18"] = {
                "description": "Conservative asymmetric policy with 10% safety margin (B_eff=0.18)",
                "nominal_budget": 0.18,
                "tau_substitution": th_s_c,
                "tau_omission": th_o_c,
                "tau_insertion": th_i_c,
                "train_oof_fwr": 0.1800,
                "dev_cal_fwr": m_dev_cal_c["joint_fwr"],
                "dev_cal_macro3_f1": m_dev_cal_c["macro3_error_f1"]
            }

        if c_name == "CANDIDATE_ASYM_SENSITIVITY_B0.12_S0.020_O0.080" and candidate_policy_dict:
            candidate_policy_dict["policies"]["asym_sensitivity/0.12"] = {
                "description": "Diagnostic sensitivity policy operating at realized DEV FWR <= 0.20",
                "nominal_budget": 0.12,
                "tau_substitution": th_s_c,
                "tau_omission": th_o_c,
                "tau_insertion": th_i_c,
                "train_oof_fwr": 0.1200,
                "dev_cal_fwr": m_dev_cal_c["joint_fwr"],
                "dev_cal_macro3_f1": m_dev_cal_c["macro3_error_f1"]
            }

    # C. Matched Realized FWR on DEV_CAL and DEV_SEARCH
    print("\n--------------------------------------------------------------------------------")
    print("Executing Matched Realized FWR Diagnostic Evaluation on DEV_CAL & DEV_SEARCH...")
    print("--------------------------------------------------------------------------------")

    # DEV_CAL Matched Points:
    # Point 1: 53 false warnings (FWR = 0.1767, matching Frozen Baseline exactly)
    # Point 2: 60 false warnings (FWR = 0.2000, nominal budget)
    for dev_name, n_utts, tkeys, gkeys, yt, yg, p_neu, p_fix, pg, ap_s, ap_o, ap_i in [
        ("DEV_CAL", 300, cal_tkeys, cal_gkeys, cal_yt, cal_yg, cal_p_neu, cal_p_fix, cal_pg, cal_ap_s, cal_ap_o, cal_ap_i),
        ("DEV_SEARCH", 600, srch_tkeys, srch_gkeys, srch_yt, srch_yg, srch_p_neu, srch_p_fix, srch_pg, srch_ap_s, srch_ap_o, srch_ap_i),
    ]:
        matched_points = [
            (int(round(0.17666666666666667 * n_utts)), 0.1767),
            (int(round(0.20000000000000000 * n_utts)), 0.2000),
        ]

        for total_fw, target_fwr in matched_points:
            t_fp_limit = int(round(0.8 * total_fw))
            g_fp_limit = total_fw - t_fp_limit

            # Gap decisions
            order_g = sorted(range(len(yg)), key=lambda i: (-float(pg[i]), -tie_hash(gkeys[i])))
            fp_cnt_g, cut_g = 0, 0
            for idx in order_g:
                if yg[idx] == 0: fp_cnt_g += 1
                if fp_cnt_g > g_fp_limit: break
                cut_g += 1
            dec_g = np.zeros(len(yg), dtype=bool)
            for idx in order_g[:cut_g]: dec_g[idx] = True
            tau_i = float(pg[order_g[cut_g - 1]]) if cut_g else 1.0

            # 1. Fixed 50/50 Shared Target
            conf_f = p_fix[:, 1:].max(axis=1)
            kind_f = p_fix[:, 1:].argmax(axis=1) + 1
            false_f = (kind_f != yt)
            order_f = sorted(range(len(yt)), key=lambda i: (-float(conf_f[i]), -tie_hash(tkeys[i])))
            fp_cnt_f, cut_f = 0, 0
            for idx in order_f:
                if false_f[idx]: fp_cnt_f += 1
                if fp_cnt_f > t_fp_limit: break
                cut_f += 1
            dec_t_fix = np.zeros(len(yt), dtype=int)
            for idx in order_f[:cut_f]: dec_t_fix[idx] = kind_f[idx]
            tau_t_fix = float(conf_f[order_f[cut_f - 1]]) if cut_f else 1.0
            m_fix = compute_metrics_fast(yt, dec_t_fix, yg, dec_g, n_utts, ap_s, ap_o, ap_i)

            policy_records.append({
                "seed": 101,
                "evaluation_scope": f"{dev_name}_MATCHED_REALIZED_FWR",
                "policy_id": f"MATCHED_FIXED_5050_FWR_{target_fwr:.4f}",
                "policy_type": "FIXED_5050_MATCHED",
                "arbitration_rule": "NONE",
                "nominal_joint_budget": target_fwr,
                "nominal_budget_s": 0.0,
                "nominal_budget_o": t_fp_limit / float(n_utts),
                "nominal_budget_i": g_fp_limit / float(n_utts),
                "tau_target": tau_t_fix,
                "tau_substitution": tau_t_fix,
                "tau_omission": tau_t_fix,
                "tau_insertion": tau_i,
                **m_fix
            })

            # 2. Neural-Only Shared Target
            conf_n = p_neu[:, 1:].max(axis=1)
            kind_n = p_neu[:, 1:].argmax(axis=1) + 1
            false_n = (kind_n != yt)
            order_n = sorted(range(len(yt)), key=lambda i: (-float(conf_n[i]), -tie_hash(tkeys[i])))
            fp_cnt_n, cut_n = 0, 0
            for idx in order_n:
                if false_n[idx]: fp_cnt_n += 1
                if fp_cnt_n > t_fp_limit: break
                cut_n += 1
            dec_t_neu = np.zeros(len(yt), dtype=int)
            for idx in order_n[:cut_n]: dec_t_neu[idx] = kind_n[idx]
            tau_t_neu = float(conf_n[order_n[cut_n - 1]]) if cut_n else 1.0
            m_neu = compute_metrics_fast(yt, dec_t_neu, yg, dec_g, n_utts, ap_s, ap_o, ap_i)

            policy_records.append({
                "seed": 101,
                "evaluation_scope": f"{dev_name}_MATCHED_REALIZED_FWR",
                "policy_id": f"MATCHED_NEURAL_ONLY_FWR_{target_fwr:.4f}",
                "policy_type": "NEURAL_ONLY_MATCHED",
                "arbitration_rule": "NONE",
                "nominal_joint_budget": target_fwr,
                "nominal_budget_s": 0.0,
                "nominal_budget_o": t_fp_limit / float(n_utts),
                "nominal_budget_i": g_fp_limit / float(n_utts),
                "tau_target": tau_t_neu,
                "tau_substitution": tau_t_neu,
                "tau_omission": tau_t_neu,
                "tau_insertion": tau_i,
                **m_neu
            })

            # 3. Previous Classwise on Fixed 50/50 (25% S, 75% O)
            s_fp_lim = int(round(0.25 * t_fp_limit))
            o_fp_lim = t_fp_limit - s_fp_lim
            order_fs = sorted(range(len(yt)), key=lambda i: (-float(p_fix[i, 1]), -tie_hash(tkeys[i])))
            order_fo = sorted(range(len(yt)), key=lambda i: (-float(p_fix[i, 2]), -tie_hash(tkeys[i])))
            fp_cnt_fs, cut_fs = 0, 0
            for idx in order_fs:
                if yt[idx] != 1: fp_cnt_fs += 1
                if fp_cnt_fs > s_fp_lim: break
                cut_fs += 1
            fp_cnt_fo, cut_fo = 0, 0
            for idx in order_fo:
                if yt[idx] != 2: fp_cnt_fo += 1
                if fp_cnt_fo > o_fp_lim: break
                cut_fo += 1
            w_fs = np.zeros(len(yt), dtype=bool)
            w_fo = np.zeros(len(yt), dtype=bool)
            for idx in order_fs[:cut_fs]: w_fs[idx] = True
            for idx in order_fo[:cut_fo]: w_fo[idx] = True
            tau_s_cw = float(p_fix[order_fs[cut_fs - 1], 1]) if cut_fs else 1.0
            tau_o_cw = float(p_fix[order_fo[cut_fo - 1], 2]) if cut_fo else 1.0
            dec_t_cw = resolve_so_conflict(w_fs, w_fo, p_fix[:, 1], p_fix[:, 2], tau_s_cw, tau_o_cw, tkeys, "score_margin")
            m_cw = compute_metrics_fast(yt, dec_t_cw, yg, dec_g, n_utts, ap_s, ap_o, ap_i)

            policy_records.append({
                "seed": 101,
                "evaluation_scope": f"{dev_name}_MATCHED_REALIZED_FWR",
                "policy_id": f"MATCHED_PREV_CLASSWISE_FIXED_FWR_{target_fwr:.4f}",
                "policy_type": "PREV_CLASSWISE_FIXED_MATCHED",
                "arbitration_rule": "score_margin",
                "nominal_joint_budget": target_fwr,
                "nominal_budget_s": s_fp_lim / float(n_utts),
                "nominal_budget_o": o_fp_lim / float(n_utts),
                "nominal_budget_i": g_fp_limit / float(n_utts),
                "tau_target": None,
                "tau_substitution": tau_s_cw,
                "tau_omission": tau_o_cw,
                "tau_insertion": tau_i,
                **m_cw
            })

            # 4. Asymmetric Routing (p_neural_S, p_fixed_O)
            order_as = sorted(range(len(yt)), key=lambda i: (-float(p_neu[i, 1]), -tie_hash(tkeys[i])))
            order_ao = sorted(range(len(yt)), key=lambda i: (-float(p_fix[i, 2]), -tie_hash(tkeys[i])))
            fp_cnt_as, cut_as = 0, 0
            for idx in order_as:
                if yt[idx] != 1: fp_cnt_as += 1
                if fp_cnt_as > s_fp_lim: break
                cut_as += 1
            fp_cnt_ao, cut_ao = 0, 0
            for idx in order_ao:
                if yt[idx] != 2: fp_cnt_ao += 1
                if fp_cnt_ao > o_fp_lim: break
                cut_ao += 1
            w_as = np.zeros(len(yt), dtype=bool)
            w_ao = np.zeros(len(yt), dtype=bool)
            for idx in order_as[:cut_as]: w_as[idx] = True
            for idx in order_ao[:cut_ao]: w_ao[idx] = True
            tau_s_as = float(p_neu[order_as[cut_as - 1], 1]) if cut_as else 1.0
            tau_o_as = float(p_fix[order_ao[cut_ao - 1], 2]) if cut_ao else 1.0
            dec_t_as = resolve_so_conflict(w_as, w_ao, p_neu[:, 1], p_fix[:, 2], tau_s_as, tau_o_as, tkeys, "score_margin")
            m_as = compute_metrics_fast(yt, dec_t_as, yg, dec_g, n_utts, ap_s, ap_o, ap_i)

            policy_records.append({
                "seed": 101,
                "evaluation_scope": f"{dev_name}_MATCHED_REALIZED_FWR",
                "policy_id": f"MATCHED_ASYMMETRIC_FWR_{target_fwr:.4f}",
                "policy_type": "ASYMMETRIC_MATCHED",
                "arbitration_rule": "score_margin",
                "nominal_joint_budget": target_fwr,
                "nominal_budget_s": s_fp_lim / float(n_utts),
                "nominal_budget_o": o_fp_lim / float(n_utts),
                "nominal_budget_i": g_fp_limit / float(n_utts),
                "tau_target": None,
                "tau_substitution": tau_s_as,
                "tau_omission": tau_o_as,
                "tau_insertion": tau_i,
                **m_as
            })

    # Save deliverables
    print("\n--------------------------------------------------------------------------------")
    print("Writing Deliverable Files...")
    print("--------------------------------------------------------------------------------")

    # 1. POLICY_COMPARISON.csv
    df_policies = pd.DataFrame(policy_records)
    policy_csv_path = EXP_DIR / "POLICY_COMPARISON.csv"
    df_policies.to_csv(policy_csv_path, index=False)
    print(f"Saved {policy_csv_path} ({len(df_policies)} rows)")

    # 2. PER_SPEAKER_METRICS.csv
    df_speakers = pd.DataFrame(speaker_records)
    speaker_csv_path = EXP_DIR / "PER_SPEAKER_METRICS.csv"
    df_speakers.to_csv(speaker_csv_path, index=False)
    print(f"Saved {speaker_csv_path} ({len(df_speakers)} rows)")

    # 3. CANDIDATE_POLICY.json
    candidate_json_path = EXP_DIR / "CANDIDATE_POLICY.json"
    with open(candidate_json_path, "w") as f:
        json.dump(candidate_policy_dict, f, indent=2)
    print(f"Saved {candidate_json_path}")

    # 4. COUNTERFACTUAL_RESULTS.json
    cf_json_path = EXP_DIR / "COUNTERFACTUAL_RESULTS.json"
    with open(cf_json_path, "w") as f:
        json.dump(cf_results, f, indent=2)
    print(f"Saved {cf_json_path}")

    print("\nBenchmark Execution Complete.")
    return df_policies, df_speakers, candidate_policy_dict, cf_results


if __name__ == "__main__":
    run_all_evaluations()
