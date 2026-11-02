#!/usr/bin/env python3
"""
LANGFENS — RQ1-DIAG-01: Phoneme Omission & False-Negative Root Cause Diagnostic Harness.

Comprehensive offline diagnostic audit analyzing phoneme omissions (gold deletions),
CTC ASR deletions, and Binary MDD false negatives across all 4 folds and 8 validation speakers.

Adheres strictly to AGENTS.md:
- CPU-first; bounded local inference only.
- Validation split only; zero test set access.
- Read-only diagnostics on existing Baseline R checkpoints and predictions.
- No new architecture or loss implementations.
"""

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from scipy.stats import rankdata

import numpy as np
import torch

from langfens_prep.audio_loader import get_l2arctic_reader, load_l2arctic_audio_from_zip
from langfens_prep.model import LangfensWav2Vec2MDD
from langfens_prep.phoneme_normalizer import (
    CTC_INDEX_TO_PHONE,
    PHONE_TO_CTC_INDEX,
    STANDARD_39_ARPABET,
)
from langfens_prep.textgrid_parser import parse_gold_textgrid_content

VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH",
    "IY", "OW", "OY", "UH", "UW"
}
CONSONANTS = set(STANDARD_39_ARPABET) - VOWELS

FRAME_STRIDE_SEC = 0.020  # 20 ms per Wav2Vec2 convolutional stride

SPEAKER_DEMOGRAPHICS = {
    "HKK": {"l1": "Korean", "gender": "Female", "fold": 4},
    "NCC": {"l1": "Mandarin", "gender": "Female", "fold": 4},
    "NJS": {"l1": "Spanish", "gender": "Female", "fold": 1},
    "RRBI": {"l1": "Hindi", "gender": "Male", "fold": 1},
    "TNI": {"l1": "Spanish", "gender": "Male", "fold": 2},
    "TXHC": {"l1": "Mandarin", "gender": "Male", "fold": 3},
    "YBAA": {"l1": "Arabic", "gender": "Female", "fold": 2},
    "ZHAA": {"l1": "Arabic", "gender": "Female", "fold": 3},
}

FOLD_CHECKPOINTS = {
    1: "checkpoints/RQ1-TRN-02A/recovered/best.pt",
    2: "checkpoints/fold2/baseline_r_fold2/fixed_budget_checkpoint.pt",
    3: "checkpoints/fold3/baseline_r_fold3/fixed_budget_checkpoint.pt",
    4: "checkpoints/fold4/baseline_r_fold4/fixed_budget_checkpoint.pt",
}

FOLD_ZIPS = {
    1: "datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip",
    2: "datasets/L2-ARCTIC/authorized_RQ1-TRN_fold2_train_val_only.zip",
    3: "datasets/L2-ARCTIC/authorized_RQ1-TRN_fold3_train_val_only.zip",
    4: "datasets/L2-ARCTIC/authorized_RQ1-TRN_fold4_train_val_only.zip",
}


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def average_precision_binary(
    labels: List[int] | np.ndarray,
    scores: List[float] | np.ndarray,
) -> float:
    """
    Computes Average Precision with tied scores grouped, matching scikit-learn.
    """
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError("AP labels must be a non-empty binary vector")
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    s = s[order]
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp = np.cumsum(y)[ends]
    fp = (ends + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_auc_binary(
    labels: List[int] | np.ndarray,
    scores: List[float] | np.ndarray,
) -> float:
    """Computes exact ROC AUC via Mann-Whitney U statistic with tie handling."""
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)
    r_pos = ranks[y].sum()
    u = r_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def calculate_operating_point(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Computes binary classification metrics at a given scalar decision threshold."""
    preds = (scores >= threshold).astype(int)
    positives = int(y_true.sum())
    negatives = len(y_true) - positives
    
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))
    
    recall = tp / positives if positives > 0 else 0.0
    fnr = fn / positives if positives > 0 else 0.0
    fpr = fp / negatives if negatives > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "positives": positives,
        "negatives": negatives,
        "recall": float(recall),
        "fnr": float(fnr),
        "fpr": float(fpr),
        "precision": float(precision),
        "f1": float(f1),
    }


def find_threshold_at_fpr_ceiling(
    negative_scores: np.ndarray,
    target_fpr: float,
) -> float:
    """Finds the lowest score threshold such that actual FPR <= target_fpr."""
    # target_fpr is the top fraction of negative scores
    # quantile (1.0 - target_fpr) gives the threshold where target_fpr fraction exceed it
    return float(np.quantile(negative_scores, 1.0 - target_fpr))


def extract_word_positions(
    canonical_intervals: list,
    words_intervals: list,
) -> Dict[int, str]:
    """Maps canonical interval index to word position: word_initial, word_medial, word_final, single_phone_word."""
    words_map = defaultdict(list)
    for idx, iv in enumerate(canonical_intervals):
        phone_center = (iv.xmin + iv.xmax) / 2.0
        matching_word = None
        for w in words_intervals:
            if w["xmin"] <= phone_center <= w["xmax"]:
                matching_word = (w["xmin"], w["xmax"], w["text"])
                break
        w_key = matching_word if matching_word else ("unaligned", idx, "")
        words_map[w_key].append(idx)
        
    positions = {}
    for w_key, p_indices in words_map.items():
        if w_key[0] == "unaligned" or len(p_indices) == 1:
            for p_idx in p_indices:
                positions[p_idx] = "single_phone_word"
        else:
            for rank, p_idx in enumerate(p_indices):
                if rank == 0:
                    positions[p_idx] = "word_initial"
                elif rank == len(p_indices) - 1:
                    positions[p_idx] = "word_final"
                else:
                    positions[p_idx] = "word_medial"
    return positions


def diagnose_cross_attention_sample(
    fold_utts_map: Dict[int, List[Dict[str, Any]]],
    samples_per_speaker: int = 5,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Evaluates cross-attention temporal localization on a balanced sample of validation utterances
    containing deletions across all 8 speakers and 4 folds.
    """
    print(f"\n[DIAG-ATTN] Running Cross-Attention Temporal Localization on {samples_per_speaker} utts/speaker...")
    attn_records = []
    
    for fold_idx in [1, 2, 3, 4]:
        ckpt_path = FOLD_CHECKPOINTS[fold_idx]
        zip_path = FOLD_ZIPS[fold_idx]
        reader = get_l2arctic_reader(zip_path)
        
        # Load model for this fold
        t0 = time.time()
        model = LangfensWav2Vec2MDD(pretrained_model_name_or_path="facebook/wav2vec2-base-960h")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()
        print(f"  Fold {fold_idx} model loaded from {ckpt_path} ({time.time() - t0:.2f}s)")
        
        # Select balanced utterances containing deletions for each speaker in this fold
        fold_utts = fold_utts_map[fold_idx]
        spk_groups = defaultdict(list)
        for u in fold_utts:
            spk_groups[u["speaker"]].append(u)
            
        selected_utts = []
        for spk, utts in sorted(spk_groups.items()):
            del_utts = [u for u in utts if any(et == "deletion" for et in u["token_error_types"])]
            chosen = del_utts[:samples_per_speaker] if len(del_utts) >= samples_per_speaker else del_utts
            selected_utts.extend(chosen)
            print(f"    Speaker {spk}: selected {len(chosen)} utterances with deletions")
            
        with torch.no_grad():
            for u in selected_utts:
                spk = u["speaker"]
                uid = u["utterance_id"]
                wav, sr, _ = load_l2arctic_audio_from_zip(zip_path, spk, uid)
                c_phones = u["canonical_phones"]
                can_indices = torch.tensor(
                    [[PHONE_TO_CTC_INDEX[p] for p in c_phones]],
                    dtype=torch.long,
                    device=device,
                )
                wav_tensor = wav.unsqueeze(0).to(device)
                
                out = model(wav_tensor, canonical_phone_indices=can_indices)
                attn = out.alignment_weights[0].cpu().numpy()  # (num_phones, num_frames)
                num_phones, num_frames = attn.shape
                
                # Analyze per token
                for idx in range(num_phones):
                    et = u["token_error_types"][idx]
                    xmin = u["token_xmins"][idx]
                    xmax = u["token_xmaxs"][idx]
                    p = c_phones[idx]
                    
                    f_start = min(num_frames - 1, max(0, int(xmin / FRAME_STRIDE_SEC)))
                    f_end = min(num_frames, max(f_start + 1, int(xmax / FRAME_STRIDE_SEC)))
                    
                    in_window_mass = float(attn[idx, f_start:f_end].sum())
                    peak_f = int(np.argmax(attn[idx]))
                    peak_time = peak_f * FRAME_STRIDE_SEC
                    center_time = (xmin + xmax) / 2.0
                    temporal_offset = abs(peak_time - center_time)
                    max_head_mass = float(np.max(attn[idx]))
                    
                    # Attention entropy: -sum(p * log(p))
                    p_dist = attn[idx] / (attn[idx].sum() + 1e-12)
                    entropy = -float(np.sum(p_dist * np.log(p_dist + 1e-12)))
                    
                    attn_records.append({
                        "fold": fold_idx,
                        "speaker": spk,
                        "utterance_id": uid,
                        "token_idx": idx,
                        "phone": p,
                        "error_type": et,
                        "xmin": xmin,
                        "xmax": xmax,
                        "duration": xmax - xmin,
                        "in_window_mass": in_window_mass,
                        "peak_time": peak_time,
                        "center_time": center_time,
                        "temporal_offset_sec": temporal_offset,
                        "max_frame_mass": max_head_mass,
                        "entropy": entropy,
                    })

    # Aggregate attention metrics by error type
    attn_summary = {}
    for etype in ["correct", "substitution", "deletion", "all"]:
        sub = attn_records if etype == "all" else [r for r in attn_records if r["error_type"] == etype]
        if not sub:
            continue
        in_masses = [r["in_window_mass"] for r in sub]
        offsets = [r["temporal_offset_sec"] for r in sub]
        entropies = [r["entropy"] for r in sub]
        max_masses = [r["max_frame_mass"] for r in sub]
        
        attn_summary[etype] = {
            "token_count": len(sub),
            "in_window_mass_mean": round(float(np.mean(in_masses)), 6),
            "in_window_mass_std": round(float(np.std(in_masses)), 6),
            "in_window_mass_median": round(float(np.median(in_masses)), 6),
            "temporal_offset_mean_sec": round(float(np.mean(offsets)), 6),
            "temporal_offset_median_sec": round(float(np.median(offsets)), 6),
            "max_frame_mass_mean": round(float(np.mean(max_masses)), 6),
            "attention_entropy_mean": round(float(np.mean(entropies)), 6),
        }
        print(f"  [ATTN-SUMMARY] {etype.upper()} (N={len(sub)}): "
              f"in_window_mass={np.mean(in_masses):.4f}, offset={np.mean(offsets):.3f}s, max_mass={np.mean(max_masses):.3f}")
        
    return {
        "sampled_utterances_count": len(set((r["speaker"], r["utterance_id"]) for r in attn_records)),
        "sampled_tokens_count": len(attn_records),
        "by_error_type": attn_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="LANGFENS RQ1-DIAG-01 Diagnostic Harness")
    parser.add_argument(
        "--pred_file",
        default="manifests/RQ1-TRN-03A/evaluation/FOUR_FOLD_PAIRED_PREDICTION_RECORDS.json.gz",
        help="Path to four-fold predictions archive",
    )
    parser.add_argument(
        "--four_fold_report",
        default="manifests/RQ1-TRN-03A/evaluation/FOUR_FOLD_VALIDATION_REPORT.json",
        help="Path to 4-fold validation synthesis report",
    )
    parser.add_argument(
        "--outdir",
        default="manifests/RQ1-DIAG-01/evaluation",
        help="Output directory for reports and tables",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device for cross-attention audit (default: cpu)",
    )
    parser.add_argument(
        "--attn_samples_per_speaker",
        type=int,
        default=5,
        help="Number of utterances per speaker for cross-attention audit",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    t_start = time.time()

    print("=" * 80)
    print("LANGFENS — RQ1-DIAG-01: Phoneme Omission & False-Negative Root Cause Audit")
    print("=" * 80)

    # 1. Load predictions archive
    print(f"\n[LOAD] Loading four-fold predictions from {args.pred_file}...")
    with gzip.open(args.pred_file, "rt") as f:
        pred_data = json.load(f)
    print(f"  Loaded {len(pred_data['utterances'])} utterances, total tokens: {pred_data['metadata']['total_tokens']}")

    # 2. Enrich each token with TextGrid annotation
    print("\n[ENRICH] Joining token predictions with Praat TextGrids...")
    all_tokens = []
    fold_utts_map = defaultdict(list)
    
    for u in pred_data["utterances"]:
        f_idx = u["fold"]
        spk = u["speaker"]
        uid = u["utterance_id"]
        reader = get_l2arctic_reader(FOLD_ZIPS[f_idx])
        tg_text = reader.read_annotation_text(spk, uid)
        ann = parse_gold_textgrid_content(tg_text, f"{uid}.TextGrid")
        c_ivs = ann.canonical_intervals
        w_ivs = ann.words_intervals
        
        assert len(c_ivs) == len(u["canonical_phones"]) == len(u["labels"]) == len(u["matched_scores"]) == len(u["mismatched_scores"]), (
            f"Length mismatch in {uid}: {len(c_ivs)} vs {len(u['canonical_phones'])}"
        )
        
        # Word positions
        word_positions = extract_word_positions(c_ivs, w_ivs)
        
        # Track for sample attention audit
        u_record = {
            "fold": f_idx,
            "speaker": spk,
            "utterance_id": uid,
            "canonical_phones": u["canonical_phones"],
            "token_error_types": [iv.error_type for iv in c_ivs],
            "token_xmins": [iv.xmin for iv in c_ivs],
            "token_xmaxs": [iv.xmax for iv in c_ivs],
        }
        fold_utts_map[f_idx].append(u_record)
        
        # Word text lookup helper
        for idx, (iv, s_m, s_mm, lbl) in enumerate(zip(c_ivs, u["matched_scores"], u["mismatched_scores"], u["labels"])):
            dur = iv.xmax - iv.xmin
            pos = word_positions.get(idx, "unknown")
            
            # Find word text
            p_center = (iv.xmin + iv.xmax) / 2.0
            word_str = ""
            for w in w_ivs:
                if w["xmin"] <= p_center <= w["xmax"]:
                    word_str = w["text"]
                    break
                    
            # Context
            prev_p = c_ivs[idx - 1].canonical_phone if idx > 0 else "SIL"
            next_p = c_ivs[idx + 1].canonical_phone if idx < len(c_ivs) - 1 else "SIL"
            prev_type = "V" if prev_p in VOWELS else ("C" if prev_p in CONSONANTS else "SIL")
            curr_type = "V" if iv.canonical_phone in VOWELS else "C"
            next_type = "V" if next_p in VOWELS else ("C" if next_p in CONSONANTS else "SIL")
            ctx_str = f"{prev_type}-{curr_type}-{next_type}"
            
            # Binary label verification: 1 for substitution or deletion; 0 for correct
            expected_bin = 1 if iv.error_type in ("substitution", "deletion") else 0
            assert lbl == expected_bin, f"Label mismatch in {uid} token {idx}: {lbl} != {expected_bin}"
            
            all_tokens.append({
                "fold": f_idx,
                "speaker": spk,
                "l1": SPEAKER_DEMOGRAPHICS[spk]["l1"],
                "gender": SPEAKER_DEMOGRAPHICS[spk]["gender"],
                "utterance_id": uid,
                "token_idx": idx,
                "canonical_phone": iv.canonical_phone,
                "realized_phone": iv.realized_phone,
                "raw_label": iv.raw_label,
                "error_type": iv.error_type,
                "binary_label": lbl,
                "word": word_str,
                "word_position": pos,
                "duration_sec": dur,
                "xmin": iv.xmin,
                "xmax": iv.xmax,
                "adjacent_context": ctx_str,
                "matched_score": s_m,
                "mismatched_score": s_mm,
                "delta_score": s_m - s_mm,
            })

    total_tokens = len(all_tokens)
    print(f"  Successfully processed and enriched {total_tokens} tokens across 4 folds.")

    # 3. Overall Partition & Counts
    correct_tokens = [t for t in all_tokens if t["error_type"] == "correct"]
    sub_tokens = [t for t in all_tokens if t["error_type"] == "substitution"]
    del_tokens = [t for t in all_tokens if t["error_type"] == "deletion"]
    all_error_tokens = [t for t in all_tokens if t["binary_label"] == 1]
    
    assert len(all_tokens) == len(correct_tokens) + len(sub_tokens) + len(del_tokens), "Token count discrepancy"
    assert len(all_error_tokens) == len(sub_tokens) + len(del_tokens), "Error count discrepancy"

    print("\n[PARTITION] Authoritative 4-Fold Token Partition:")
    print(f"  Total Phone Tokens: {total_tokens:,}")
    print(f"  Correct Tokens:     {len(correct_tokens):,} ({len(correct_tokens)/total_tokens:.2%})")
    print(f"  Substitutions:      {len(sub_tokens):,} ({len(sub_tokens)/total_tokens:.2%} of tokens, {len(sub_tokens)/len(all_error_tokens):.2%} of errors)")
    print(f"  Deletions/Omissions:{len(del_tokens):,} ({len(del_tokens)/total_tokens:.2%} of tokens, {len(del_tokens)/len(all_error_tokens):.2%} of errors)")
    print(f"  Total Error Tokens: {len(all_error_tokens):,} ({len(all_error_tokens)/total_tokens:.2%})")

    # 4. Compute AP, ROC-AUC, and score summaries per error type
    y_all = np.array([t["binary_label"] for t in all_tokens])
    s_all_m = np.array([t["matched_score"] for t in all_tokens])
    s_all_mm = np.array([t["mismatched_score"] for t in all_tokens])
    
    # Substitution vs Correct
    sub_set = [t for t in all_tokens if t["error_type"] in ("correct", "substitution")]
    y_sub = np.array([1 if t["error_type"] == "substitution" else 0 for t in sub_set])
    s_sub_m = np.array([t["matched_score"] for t in sub_set])
    s_sub_mm = np.array([t["mismatched_score"] for t in sub_set])
    
    # Deletion vs Correct
    del_set = [t for t in all_tokens if t["error_type"] in ("correct", "deletion")]
    y_del = np.array([1 if t["error_type"] == "deletion" else 0 for t in del_set])
    s_del_m = np.array([t["matched_score"] for t in del_set])
    s_del_mm = np.array([t["mismatched_score"] for t in del_set])

    ap_all_m = average_precision_binary(y_all, s_all_m)
    ap_all_mm = average_precision_binary(y_all, s_all_mm)
    roc_all = roc_auc_binary(y_all, s_all_m)

    ap_sub_m = average_precision_binary(y_sub, s_sub_m)
    ap_sub_mm = average_precision_binary(y_sub, s_sub_mm)
    roc_sub = roc_auc_binary(y_sub, s_sub_m)

    ap_del_m = average_precision_binary(y_del, s_del_m)
    ap_del_mm = average_precision_binary(y_del, s_del_mm)
    roc_del = roc_auc_binary(y_del, s_del_m)

    print("\n[METRICS] Ranking Metrics by Error Type:")
    print(f"  Overall Errors (N={len(all_error_tokens)}): AP={ap_all_m:.6f}, Mismatched AP={ap_all_mm:.6f}, Gap={ap_all_m - ap_all_mm:+.6f}, ROC-AUC={roc_all:.6f}")
    print(f"  Substitutions  (N={len(sub_tokens)}): AP={ap_sub_m:.6f}, Mismatched AP={ap_sub_mm:.6f}, Gap={ap_sub_m - ap_sub_mm:+.6f}, ROC-AUC={roc_sub:.6f}")
    print(f"  Deletions      (N={len(del_tokens)}): AP={ap_del_m:.6f}, Mismatched AP={ap_del_mm:.6f}, Gap={ap_del_m - ap_del_mm:+.6f}, ROC-AUC={roc_del:.6f}")

    # 5. Diagnostic Fixed-FPR and Standard Operating Points
    correct_scores = np.array([t["matched_score"] for t in correct_tokens])
    tau_010 = find_threshold_at_fpr_ceiling(correct_scores, 0.10)
    tau_015 = find_threshold_at_fpr_ceiling(correct_scores, 0.15)
    tau_020 = find_threshold_at_fpr_ceiling(correct_scores, 0.20)

    operating_points_eval = [
        {"name": "Standard tau=0.50", "threshold": 0.50},
        {"name": "Standard tau=0.20", "threshold": 0.20},
        {"name": "FPR <= 20% Ceiling", "threshold": tau_020},
        {"name": "FPR <= 15% Ceiling", "threshold": tau_015},
        {"name": "FPR <= 10% Ceiling", "threshold": tau_010},
    ]

    operating_points_report = []
    for op in operating_points_eval:
        th = op["threshold"]
        m_all = calculate_operating_point(y_all, s_all_m, th)
        m_sub = calculate_operating_point(y_sub, s_sub_m, th)
        m_del = calculate_operating_point(y_del, s_del_m, th)
        
        entry = {
            "operating_point": op["name"],
            "threshold": round(float(th), 6),
            "fpr_on_negatives": round(float(m_all["fpr"]), 4),
            "all_errors": {
                "recall": round(float(m_all["recall"]), 4),
                "fnr": round(float(m_all["fnr"]), 4),
                "precision": round(float(m_all["precision"]), 4),
                "f1": round(float(m_all["f1"]), 4),
                "tp": m_all["tp"],
                "fn": m_all["fn"],
            },
            "substitutions": {
                "recall": round(float(m_sub["recall"]), 4),
                "fnr": round(float(m_sub["fnr"]), 4),
                "precision": round(float(m_sub["precision"]), 4),
                "f1": round(float(m_sub["f1"]), 4),
                "tp": m_sub["tp"],
                "fn": m_sub["fn"],
            },
            "deletions": {
                "recall": round(float(m_del["recall"]), 4),
                "fnr": round(float(m_del["fnr"]), 4),
                "precision": round(float(m_del["precision"]), 4),
                "f1": round(float(m_del["f1"]), 4),
                "tp": m_del["tp"],
                "fn": m_del["fn"],
            },
        }
        operating_points_report.append(entry)
        print(f"\n  [{op['name']} (tau={th:.4f})] Actual FPR: {m_all['fpr']:.2%}")
        print(f"    Substitutions: Recall={m_sub['recall']:.2%}, FNR={m_sub['fnr']:.2%}, F1={m_sub['f1']:.4f}")
        print(f"    Deletions:     Recall={m_del['recall']:.2%}, FNR={m_del['fnr']:.2%}, F1={m_del['f1']:.4f}")

    # 6. Disaggregated Breakdown by Speaker
    print("\n[BREAKDOWN] Disaggregating by Speaker (8 validation speakers)...")
    speaker_breakdown = {}
    for spk, demo in sorted(SPEAKER_DEMOGRAPHICS.items()):
        spk_tokens = [t for t in all_tokens if t["speaker"] == spk]
        spk_cor = [t for t in spk_tokens if t["error_type"] == "correct"]
        spk_sub = [t for t in spk_tokens if t["error_type"] == "substitution"]
        spk_del = [t for t in spk_tokens if t["error_type"] == "deletion"]
        
        y_spk_sub = np.array([1 if t["error_type"] == "substitution" else 0 for t in spk_cor + spk_sub])
        s_spk_sub = np.array([t["matched_score"] for t in spk_cor + spk_sub])
        ap_s_sub = average_precision_binary(y_spk_sub, s_spk_sub) if len(spk_sub) > 0 else float("nan")
        
        y_spk_del = np.array([1 if t["error_type"] == "deletion" else 0 for t in spk_cor + spk_del])
        s_spk_del = np.array([t["matched_score"] for t in spk_cor + spk_del])
        ap_s_del = average_precision_binary(y_spk_del, s_spk_del) if len(spk_del) > 0 else float("nan")

        # FNR at tau=0.50 and tau=0.20
        rec_sub_05 = np.mean([t["matched_score"] >= 0.50 for t in spk_sub]) if spk_sub else 0.0
        rec_del_05 = np.mean([t["matched_score"] >= 0.50 for t in spk_del]) if spk_del else 0.0
        rec_sub_02 = np.mean([t["matched_score"] >= 0.20 for t in spk_sub]) if spk_sub else 0.0
        rec_del_02 = np.mean([t["matched_score"] >= 0.20 for t in spk_del]) if spk_del else 0.0

        speaker_breakdown[spk] = {
            "speaker": spk,
            "fold": demo["fold"],
            "l1": demo["l1"],
            "gender": demo["gender"],
            "total_tokens": len(spk_tokens),
            "correct_tokens": len(spk_cor),
            "substitution_tokens": len(spk_sub),
            "deletion_tokens": len(spk_del),
            "omission_prevalence": round(len(spk_del) / len(spk_tokens), 6),
            "substitution_prevalence": round(len(spk_sub) / len(spk_tokens), 6),
            "substitution_ap": round(float(ap_s_sub), 6),
            "deletion_ap": round(float(ap_s_del), 6),
            "sub_fnr_at_050": round(float(1.0 - rec_sub_05), 4),
            "del_fnr_at_050": round(float(1.0 - rec_del_05), 4),
            "sub_fnr_at_020": round(float(1.0 - rec_sub_02), 4),
            "del_fnr_at_020": round(float(1.0 - rec_del_02), 4),
            "mean_correct_score": round(float(np.mean([t["matched_score"] for t in spk_cor])), 4),
            "mean_sub_score": round(float(np.mean([t["matched_score"] for t in spk_sub])), 4),
            "mean_del_score": round(float(np.mean([t["matched_score"] for t in spk_del])), 4),
        }

    # 7. Disaggregated Breakdown by Canonical Phoneme
    print("\n[BREAKDOWN] Disaggregating by Phoneme (39 classes)...")
    phone_breakdown = {}
    for p in STANDARD_39_ARPABET:
        p_tokens = [t for t in all_tokens if t["canonical_phone"] == p]
        if not p_tokens:
            continue
        p_cor = [t for t in p_tokens if t["error_type"] == "correct"]
        p_sub = [t for t in p_tokens if t["error_type"] == "substitution"]
        p_del = [t for t in p_tokens if t["error_type"] == "deletion"]
        
        rec_del_02 = np.mean([t["matched_score"] >= 0.20 for t in p_del]) if p_del else float("nan")
        rec_sub_02 = np.mean([t["matched_score"] >= 0.20 for t in p_sub]) if p_sub else float("nan")

        phone_breakdown[p] = {
            "phoneme": p,
            "is_vowel": p in VOWELS,
            "total_tokens": len(p_tokens),
            "correct_tokens": len(p_cor),
            "substitution_tokens": len(p_sub),
            "deletion_tokens": len(p_del),
            "omission_rate": round(len(p_del) / len(p_tokens), 6),
            "substitution_rate": round(len(p_sub) / len(p_tokens), 6),
            "del_recall_at_020": round(float(rec_del_02), 4) if not np.isnan(rec_del_02) else None,
            "del_fnr_at_020": round(float(1.0 - rec_del_02), 4) if not np.isnan(rec_del_02) else None,
            "mean_del_score": round(float(np.mean([t["matched_score"] for t in p_del])), 4) if p_del else None,
            "mean_sub_score": round(float(np.mean([t["matched_score"] for t in p_sub])), 4) if p_sub else None,
            "mean_cor_score": round(float(np.mean([t["matched_score"] for t in p_cor])), 4) if p_cor else None,
        }

    # 8. Disaggregated Breakdown by Word Position
    print("\n[BREAKDOWN] Disaggregating by Word Position...")
    position_breakdown = {}
    for pos in ["word_initial", "word_medial", "word_final", "single_phone_word"]:
        pos_tokens = [t for t in all_tokens if t["word_position"] == pos]
        pos_cor = [t for t in pos_tokens if t["error_type"] == "correct"]
        pos_sub = [t for t in pos_tokens if t["error_type"] == "substitution"]
        pos_del = [t for t in pos_tokens if t["error_type"] == "deletion"]
        
        rec_del_02 = np.mean([t["matched_score"] >= 0.20 for t in pos_del]) if pos_del else float("nan")
        rec_sub_02 = np.mean([t["matched_score"] >= 0.20 for t in pos_sub]) if pos_sub else float("nan")
        
        position_breakdown[pos] = {
            "word_position": pos,
            "total_tokens": len(pos_tokens),
            "correct_tokens": len(pos_cor),
            "substitution_tokens": len(pos_sub),
            "deletion_tokens": len(pos_del),
            "omission_rate": round(len(pos_del) / len(pos_tokens), 6),
            "substitution_rate": round(len(pos_sub) / len(pos_tokens), 6),
            "fraction_of_all_omissions": round(len(pos_del) / len(del_tokens), 6),
            "del_recall_at_020": round(float(rec_del_02), 4) if not np.isnan(rec_del_02) else None,
            "del_fnr_at_020": round(float(1.0 - rec_del_02), 4) if not np.isnan(rec_del_02) else None,
            "sub_recall_at_020": round(float(rec_sub_02), 4) if not np.isnan(rec_sub_02) else None,
            "sub_fnr_at_020": round(float(1.0 - rec_sub_02), 4) if not np.isnan(rec_sub_02) else None,
        }

    # 9. Disaggregated Breakdown by Duration Bins
    print("\n[BREAKDOWN] Disaggregating by Phoneme Duration...")
    duration_bins = [
        ("< 40ms", lambda dur: dur < 0.040),
        ("40-80ms", lambda dur: 0.040 <= dur < 0.080),
        ("80-120ms", lambda dur: 0.080 <= dur < 0.120),
        (">= 120ms", lambda dur: dur >= 0.120),
    ]
    duration_breakdown = {}
    for bin_name, bin_fn in duration_bins:
        bin_tokens = [t for t in all_tokens if bin_fn(t["duration_sec"])]
        bin_cor = [t for t in bin_tokens if t["error_type"] == "correct"]
        bin_sub = [t for t in bin_tokens if t["error_type"] == "substitution"]
        bin_del = [t for t in bin_tokens if t["error_type"] == "deletion"]
        
        rec_del_02 = np.mean([t["matched_score"] >= 0.20 for t in bin_del]) if bin_del else float("nan")

        duration_breakdown[bin_name] = {
            "duration_bin": bin_name,
            "total_tokens": len(bin_tokens),
            "correct_tokens": len(bin_cor),
            "substitution_tokens": len(bin_sub),
            "deletion_tokens": len(bin_del),
            "omission_rate": round(len(bin_del) / len(bin_tokens), 6),
            "substitution_rate": round(len(bin_sub) / len(bin_tokens), 6),
            "fraction_of_all_omissions": round(len(bin_del) / len(del_tokens), 6),
            "del_recall_at_020": round(float(rec_del_02), 4) if not np.isnan(rec_del_02) else None,
            "del_fnr_at_020": round(float(1.0 - rec_del_02), 4) if not np.isnan(rec_del_02) else None,
            "mean_del_score": round(float(np.mean([t["matched_score"] for t in bin_del])), 4) if bin_del else None,
            "mean_cor_score": round(float(np.mean([t["matched_score"] for t in bin_cor])), 4) if bin_cor else None,
        }

    # 10. Cross-Attention Temporal Localization Sample Audit
    cross_attn_audit = diagnose_cross_attention_sample(
        fold_utts_map=fold_utts_map,
        samples_per_speaker=args.attn_samples_per_speaker,
        device=args.device,
    )

    # 11. Extract Representative High-Confidence False Negatives
    print("\n[FALSE-NEGATIVES] Identifying representative high-confidence false negatives...")
    sorted_all_fn = sorted(all_error_tokens, key=lambda x: x["matched_score"])
    sorted_del_fn = sorted(del_tokens, key=lambda x: x["matched_score"])

    rep_fn_overall = sorted_all_fn[:25]
    rep_fn_deletions = sorted_del_fn[:25]

    # Combine unique representative false negatives
    rep_fn_combined = []
    seen = set()
    for item in rep_fn_overall + rep_fn_deletions:
        key = (item["speaker"], item["utterance_id"], item["token_idx"])
        if key not in seen:
            seen.add(key)
            rep_fn_combined.append(item)

    # 12. Load CTC metrics summary from FOUR_FOLD_VALIDATION_REPORT.json
    print(f"\n[CTC-AUDIT] Loading CTC performance statistics from {args.four_fold_report}...")
    with open(args.four_fold_report, "r") as f:
        ff_report = json.load(f)
    ctc_macro = {
        "ctc_per_macro": ff_report["primary_fixed_budget_3600_updates"]["macro_statistics"]["ctc_per"]["mean"],
        "ctc_sub_rate_macro": float(np.mean([
            ff_report["primary_fixed_budget_3600_updates"]["per_fold"][f"fold_{k}"]["ctc_sub_rate"]
            for k in [1, 2, 3, 4]
        ])),
        "ctc_del_rate_macro": float(np.mean([
            ff_report["primary_fixed_budget_3600_updates"]["per_fold"][f"fold_{k}"]["ctc_del_rate"]
            for k in [1, 2, 3, 4]
        ])),
        "ctc_ins_rate_macro": float(np.mean([
            ff_report["primary_fixed_budget_3600_updates"]["per_fold"][f"fold_{k}"]["ctc_ins_rate"]
            for k in [1, 2, 3, 4]
        ])),
        "ctc_blank_fraction_macro": float(np.mean([
            ff_report["primary_fixed_budget_3600_updates"]["per_fold"][f"fold_{k}"]["ctc_blank_fraction"]
            for k in [1, 2, 3, 4]
        ])),
    }
    print(f"  CTC Macro PER: {ctc_macro['ctc_per_macro']:.2%} "
          f"(Sub {ctc_macro['ctc_sub_rate_macro']:.2%}, Del {ctc_macro['ctc_del_rate_macro']:.2%}, Ins {ctc_macro['ctc_ins_rate_macro']:.2%}, Blank {ctc_macro['ctc_blank_fraction_macro']:.2%})")

    # 13. Write CSV Tables
    csv_speaker_path = os.path.join(args.outdir, "omission_error_breakdown_by_speaker.csv")
    with open(csv_speaker_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(speaker_breakdown.values())).keys()))
        writer.writeheader()
        writer.writerows(speaker_breakdown.values())
        
    csv_phone_path = os.path.join(args.outdir, "omission_error_breakdown_by_phone.csv")
    with open(csv_phone_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(phone_breakdown.values())).keys()))
        writer.writeheader()
        writer.writerows(phone_breakdown.values())

    csv_pos_path = os.path.join(args.outdir, "omission_error_breakdown_by_position.csv")
    with open(csv_pos_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(position_breakdown.values())).keys()))
        writer.writeheader()
        writer.writerows(position_breakdown.values())

    csv_dur_path = os.path.join(args.outdir, "omission_error_breakdown_by_duration.csv")
    with open(csv_dur_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(duration_breakdown.values())).keys()))
        writer.writeheader()
        writer.writerows(duration_breakdown.values())

    csv_fn_path = os.path.join(args.outdir, "representative_false_negatives.csv")
    with open(csv_fn_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "fold", "speaker", "l1", "utterance_id", "token_idx",
            "canonical_phone", "realized_phone", "raw_label", "error_type",
            "word", "word_position", "duration_sec", "xmin", "xmax",
            "adjacent_context", "matched_score", "mismatched_score", "delta_score",
        ])
        writer.writeheader()
        for item in rep_fn_combined:
            row = {k: item[k] for k in writer.fieldnames}
            row["duration_sec"] = round(row["duration_sec"], 4)
            row["xmin"] = round(row["xmin"], 4)
            row["xmax"] = round(row["xmax"], 4)
            row["matched_score"] = round(row["matched_score"], 6)
            row["mismatched_score"] = round(row["mismatched_score"], 6)
            row["delta_score"] = round(row["delta_score"], 6)
            writer.writerow(row)

    # 14. Compile Master Diagnostic Report
    audit_report = {
        "metadata": {
            "title": "LANGFENS RQ1-DIAG-01: Phoneme Omission & False-Negative Root Cause Diagnostic Report",
            "experiment": "RQ1-DIAG-01",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "evaluator_script": "audit_scripts/diagnose_omissions_and_false_negatives.py",
            "predictions_source": args.pred_file,
            "total_validation_utterances": len(pred_data["utterances"]),
            "total_phone_tokens": total_tokens,
            "speaker_census_count": len(SPEAKER_DEMOGRAPHICS),
            "folds_evaluated": [1, 2, 3, 4],
        },
        "token_census": {
            "total_tokens": total_tokens,
            "correct_tokens": len(correct_tokens),
            "correct_fraction": round(len(correct_tokens) / total_tokens, 6),
            "substitution_tokens": len(sub_tokens),
            "substitution_fraction_of_tokens": round(len(sub_tokens) / total_tokens, 6),
            "substitution_fraction_of_errors": round(len(sub_tokens) / len(all_error_tokens), 6),
            "deletion_tokens": len(del_tokens),
            "deletion_fraction_of_tokens": round(len(del_tokens) / total_tokens, 6),
            "deletion_fraction_of_errors": round(len(del_tokens) / len(all_error_tokens), 6),
            "total_errors": len(all_error_tokens),
            "error_prevalence": round(len(all_error_tokens) / total_tokens, 6),
        },
        "ranking_metrics_by_error_type": {
            "all_errors": {
                "count": len(all_error_tokens),
                "matched_ap": round(float(ap_all_m), 6),
                "mismatched_ap": round(float(ap_all_mm), 6),
                "acoustic_gap": round(float(ap_all_m - ap_all_mm), 6),
                "roc_auc": round(float(roc_all), 6),
                "score_mean": round(float(np.mean(s_all_m[y_all == 1])), 4),
                "score_median": round(float(np.median(s_all_m[y_all == 1])), 4),
            },
            "substitutions": {
                "count": len(sub_tokens),
                "matched_ap": round(float(ap_sub_m), 6),
                "mismatched_ap": round(float(ap_sub_mm), 6),
                "acoustic_gap": round(float(ap_sub_m - ap_sub_mm), 6),
                "roc_auc": round(float(roc_sub), 6),
                "score_mean": round(float(np.mean([t["matched_score"] for t in sub_tokens])), 4),
                "score_median": round(float(np.median([t["matched_score"] for t in sub_tokens])), 4),
            },
            "deletions": {
                "count": len(del_tokens),
                "matched_ap": round(float(ap_del_m), 6),
                "mismatched_ap": round(float(ap_del_mm), 6),
                "acoustic_gap": round(float(ap_del_m - ap_del_mm), 6),
                "roc_auc": round(float(roc_del), 6),
                "score_mean": round(float(np.mean([t["matched_score"] for t in del_tokens])), 4),
                "score_median": round(float(np.median([t["matched_score"] for t in del_tokens])), 4),
            },
            "correct_negatives": {
                "count": len(correct_tokens),
                "score_mean": round(float(np.mean(correct_scores)), 4),
                "score_median": round(float(np.median(correct_scores)), 4),
            },
        },
        "operating_points_comparison": operating_points_report,
        "ctc_asr_audit": {
            "macro_statistics": ctc_macro,
            "conceptual_distinction": (
                "CTC Levenshtein deletions reflect acoustic recognition failures where an acoustically "
                "realized phone was not decoded into text. In contrast, Gold Phoneme Omissions reflect "
                "actual non-native production deletions where the speaker omitted canonical phones from speech."
            ),
        },
        "cross_attention_temporal_localization": cross_attn_audit,
        "disaggregated_breakdowns": {
            "by_speaker": speaker_breakdown,
            "by_phoneme": phone_breakdown,
            "by_word_position": position_breakdown,
            "by_duration": duration_breakdown,
        },
        "representative_false_negatives_summary": {
            "top_10_overall_lowest_score": [
                {
                    "speaker": x["speaker"],
                    "fold": x["fold"],
                    "utterance_id": x["utterance_id"],
                    "token_idx": x["token_idx"],
                    "canonical_phone": x["canonical_phone"],
                    "error_type": x["error_type"],
                    "word": x["word"],
                    "duration_sec": round(x["duration_sec"], 4),
                    "matched_score": round(x["matched_score"], 6),
                }
                for x in rep_fn_overall[:10]
            ],
            "top_10_deletions_lowest_score": [
                {
                    "speaker": x["speaker"],
                    "fold": x["fold"],
                    "utterance_id": x["utterance_id"],
                    "token_idx": x["token_idx"],
                    "canonical_phone": x["canonical_phone"],
                    "error_type": x["error_type"],
                    "word": x["word"],
                    "duration_sec": round(x["duration_sec"], 4),
                    "matched_score": round(x["matched_score"], 6),
                }
                for x in rep_fn_deletions[:10]
            ],
        },
    }

    report_path = os.path.join(args.outdir, "OMISSION_AND_FALSE_NEGATIVE_REPORT.json")
    with open(report_path, "w") as f:
        json.dump(audit_report, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\n[DONE] Diagnostic run completed in {elapsed:.1f}s.")
    print(f"  Master Report: {report_path} (SHA-256: {compute_sha256(report_path)})")
    print(f"  Speaker Breakdown: {csv_speaker_path}")
    print(f"  Phone Breakdown: {csv_phone_path}")
    print(f"  Position Breakdown: {csv_pos_path}")
    print(f"  Duration Breakdown: {csv_dur_path}")
    print(f"  Representative FN: {csv_fn_path}")


if __name__ == "__main__":
    main()
