#!/usr/bin/env python3
"""
Offline Benchmark and Evaluation for P-SCORE (E0 Fold-1 VAL: NJS & RRBI).
Evaluates all 4 scorers (sf_s, sf_sd, greedy_dp, gop_sa).
Computes Omission AP, Substitution AP, Binary AP, Recall@FPR15, CTC PER,
and stratified breakdowns by speaker, word-final consonants, and confusable phones.
"""

import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

# Ensure local sources
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from langfens_prep.alignment import align_canonical_sequences
from langfens_prep.g2p import CMUDictG2P
from langfens_prep.phoneme_normalizer import (
    CTC_INDEX_TO_PHONE,
    PHONE_TO_CTC_INDEX,
    STANDARD_39_ARPABET,
)
from langfens_prep.textgrid_parser import parse_gold_textgrid_content
from langfens_prep.p_score.cache import CONTRACT_SHA256, SPLIT_SHA256, sha256

VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH",
    "IY", "OW", "OY", "UH", "UW"
}
CONSONANTS = set(STANDARD_39_ARPABET) - VOWELS

CONFUSABLE_GROUPS = {
    "dental_stops_fricatives": {"TH", "DH", "T", "D", "S", "Z"},
    "liquids_glides": {"L", "R", "W", "Y"},
    "labiodental_bilabial": {"B", "V", "P", "F"},
    "sibilants_affricates": {"S", "Z", "SH", "ZH", "CH", "JH"},
    "tense_lax_vowels": {"IY", "IH", "UW", "UH", "AE", "EH"},
}


def compute_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def average_precision_binary(labels: np.ndarray, scores: np.ndarray) -> float:
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


def roc_auc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    distinct_mask = np.r_[True, s[order[1:]] != s[order[:-1]]]
    distinct_indices = np.flatnonzero(distinct_mask)
    group_ends = np.r_[distinct_indices[1:], len(s)]
    for start, end in zip(distinct_indices, group_ends):
        if end - start > 1:
            mean_rank = (start + 1 + end) / 2.0
            ranks[order[start:end]] = mean_rank
    sum_ranks_pos = float(ranks[y].sum())
    u = sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u / (n_pos * n_neg))


def diagnostic_threshold(negative_scores: np.ndarray, ceiling: float = 0.15) -> float:
    s = np.asarray(negative_scores, dtype=np.float64)
    if len(s) == 0:
        return float("nan")
    candidates = np.r_[np.unique(s), np.nextafter(s.max(), np.inf)]
    return float(next(t for t in candidates if np.mean(s >= t) <= ceiling))


def compute_operating(y: np.ndarray, s: np.ndarray, tau: float) -> dict:
    preds = s >= tau
    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "threshold": tau,
        "recall": rec,
        "precision": prec,
        "fpr": fpr,
        "fnr": fnr,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }


def compute_levenshtein_edits(ref: list[int], hyp: list[int]) -> tuple[int, int, int, int]:
    n, m = len(ref), len(hyp)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    ops = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0=match, 1=sub, 2=del, 3=ins
    for i in range(1, n + 1):
        dp[i, 0] = i
        ops[i, 0] = 2
    for j in range(1, m + 1):
        dp[0, j] = j
        ops[0, j] = 3
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            sub_cost = dp[i - 1, j - 1] + cost
            del_cost = dp[i - 1, j] + 1
            ins_cost = dp[i, j - 1] + 1
            best = min(sub_cost, del_cost, ins_cost)
            dp[i, j] = best
            if best == sub_cost:
                ops[i, j] = 0 if cost == 0 else 1
            elif best == del_cost:
                ops[i, j] = 2
            else:
                ops[i, j] = 3
    # Backtrace
    i, j = n, m
    s_cnt, d_cnt, i_cnt = 0, 0, 0
    while i > 0 or j > 0:
        op = ops[i, j]
        if op == 0:
            i -= 1; j -= 1
        elif op == 1:
            s_cnt += 1; i -= 1; j -= 1
        elif op == 2:
            d_cnt += 1; i -= 1
        elif op == 3:
            i_cnt += 1; j -= 1
    return dp[n, m], s_cnt, d_cnt, i_cnt


def extract_word_positions(c_ivs, w_ivs) -> dict[int, str]:
    positions = {}
    for idx, iv in enumerate(c_ivs):
        p_center = (iv.xmin + iv.xmax) / 2.0
        matching_w = None
        for w in w_ivs:
            w_xmin = w["xmin"] if isinstance(w, dict) else w.xmin
            w_xmax = w["xmax"] if isinstance(w, dict) else w.xmax
            if w_xmin <= p_center <= w_xmax:
                matching_w = (w_xmin, w_xmax)
                break
        if matching_w is None:
            positions[idx] = "single_phone_word"
            continue
        word_phones = [j for j, civ in enumerate(c_ivs)
                       if matching_w[0] <= ((civ.xmin + civ.xmax) / 2.0) <= matching_w[1]]
        if len(word_phones) <= 1:
            positions[idx] = "single_phone_word"
        elif idx == word_phones[0]:
            positions[idx] = "word_initial"
        elif idx == word_phones[-1]:
            positions[idx] = "word_final"
        else:
            positions[idx] = "word_medial"
    return positions


def clean_for_json(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [clean_for_json(v) for v in obj]
    return obj


def main():
    print("[INIT] Starting P-SCORE offline benchmark evaluation...")
    runs_dir = Path("/home/khoa/Projects/KLTN/Dataset/p_score_runs/rq1_p_score_e0_fold1_val")
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Missing run output directory: {runs_dir}")

    # Checkpoint and split paths
    split_path = REPO_ROOT / "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json"
    zip_path = Path("/home/khoa/Projects/KLTN/Dataset/datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip")
    prompts_path = REPO_ROOT / "resources/L2-ARCTIC/prompts.json"
    cmudict_path = REPO_ROOT / "resources/CMUDict/cmudict.dict"

    split_data = json.loads(split_path.read_text())
    fold1 = next(f for f in split_data["folds"] if f["fold_index"] == 1)
    val_utts = fold1["val"]["utterances"]

    prompts = json.loads(prompts_path.read_text())
    g2p = CMUDictG2P(dict_path=cmudict_path, verify_checksum=True)

    # Open zip files for NJS and RRBI
    zf = zipfile.ZipFile(zip_path)
    spk_zfs = {
        "NJS": zipfile.ZipFile(io.BytesIO(zf.read("NJS.zip"))),
        "RRBI": zipfile.ZipFile(io.BytesIO(zf.read("RRBI.zip")))
    }

    # 1. Parse gold TextGrids for all 300 VAL utterances
    print("[ANNOTATIONS] Parsing gold TextGrids for 300 VAL utterances...")
    gold_data = {}
    for u in val_utts:
        spk, utt_name = u.split("/")
        tg_text = spk_zfs[spk].read(f"{spk}/annotation/{utt_name}.TextGrid").decode("utf-8")
        ann = parse_gold_textgrid_content(tg_text, f"{utt_name}.TextGrid")
        word_positions = extract_word_positions(ann.canonical_intervals, ann.words_intervals)

        # Realized reference phones for PER (chronological realized phones, excluding silence and deletion)
        ctc_ref_indices = [
            PHONE_TO_CTC_INDEX[iv.realized_phone]
            for iv in ann.intervals
            if iv.error_type not in ("silence", "deletion") and not iv.has_err_tag and iv.realized_phone in PHONE_TO_CTC_INDEX
        ]
        ref_phones = [CTC_INDEX_TO_PHONE[idx] for idx in ctc_ref_indices]

        gold_data[u] = {
            "speaker": spk,
            "utterance_id": utt_name,
            "canonical_intervals": ann.canonical_intervals,
            "word_positions": word_positions,
            "realized_phones": ref_phones,
            "realized_phone_ids": ctc_ref_indices,
        }

    # 2. Align G2P prompt phones to gold canonical intervals
    print("[ALIGNMENT] Performing deterministic Levenshtein alignment (Z-PROMPT-V1)...")
    aligned_corpus = {}
    total_gold_tokens = 0
    total_g2p_tokens = 0
    total_aligned_pairs = 0
    total_unaligned_gold = 0
    total_unaligned_g2p = 0

    for u in val_utts:
        spk, utt_name = u.split("/")
        prompt_text = prompts[utt_name]
        g2p_phones = g2p.phonemize(prompt_text)
        gold_c_phones = [iv.canonical_phone for iv in gold_data[u]["canonical_intervals"]]

        alignment = align_canonical_sequences(gold_c_phones, g2p_phones, tie_break="sub_first")
        aligned_corpus[u] = {
            "g2p_phones": g2p_phones,
            "gold_phones": gold_c_phones,
            "aligned_pairs": alignment["aligned_pairs"],
            "unaligned_gold": alignment["unaligned_gold"],
            "unaligned_g2p": alignment["unaligned_g2p"],
            "edits_summary": alignment["edits_summary"]
        }
        total_gold_tokens += len(gold_c_phones)
        total_g2p_tokens += len(g2p_phones)
        total_aligned_pairs += len(alignment["aligned_pairs"])
        total_unaligned_gold += len(alignment["unaligned_gold"])
        total_unaligned_g2p += len(alignment["unaligned_g2p"])

    print(f"  Total gold tokens: {total_gold_tokens}")
    print(f"  Total G2P tokens:  {total_g2p_tokens}")
    print(f"  Aligned pairs:     {total_aligned_pairs}")
    print(f"  Unaligned gold:    {total_unaligned_gold} (omissions from G2P)")
    print(f"  Unaligned G2P:     {total_unaligned_g2p} (insertions into G2P)")

    # 3. Load runtime predictions for each scorer and evaluate
    scorers = ["sf_s", "sf_sd", "greedy_dp", "gop_sa"]
    benchmark_results = {}
    all_scorer_rows = {}
    per_speaker_records = {}

    for scorer in scorers:
        print(f"\n[BENCHMARK] Evaluating scorer: {scorer}...")
        pred_path = runs_dir / scorer / "runtime_predictions.jsonl"
        utt_pred_path = runs_dir / scorer / "utterance_predictions.jsonl"

        # Load runtime predictions indexed by (speaker, utterance_id, prompt_index)
        prompt_preds = {}
        with open(pred_path) as f:
            for line in f:
                row = json.loads(line)
                key = (row["speaker_id"], row["utterance_id"], row["prompt_index"])
                prompt_preds[key] = row

        # Load utterance predictions for decoded phones (for PER)
        utt_preds = {}
        with open(utt_pred_path) as f:
            for line in f:
                row = json.loads(line)
                utt_preds[(row["speaker_id"], row["utterance_id"])] = row

        # Join with gold annotations
        offline_records = []
        for u in val_utts:
            spk, utt_name = u.split("/")
            c_ivs = gold_data[u]["canonical_intervals"]
            word_pos = gold_data[u]["word_positions"]
            al = aligned_corpus[u]

            for g_idx, p_idx, op_code in al["aligned_pairs"]:
                iv = c_ivs[g_idx]
                r_pred = prompt_preds.get((spk, utt_name, p_idx))
                if r_pred is None:
                    continue

                # Error types
                etype = iv.error_type  # "correct", "substitution", "deletion"
                bin_label = 1 if etype in ("substitution", "deletion") else 0
                pos_type = word_pos.get(g_idx, "unknown")
                c_phone = iv.canonical_phone

                # Determine confusable group
                conf_group = "other"
                for cg_name, cg_phones in CONFUSABLE_GROUPS.items():
                    if c_phone in cg_phones:
                        conf_group = cg_name
                        break

                rec = {
                    "scorer": scorer,
                    "speaker": spk,
                    "utterance_id": utt_name,
                    "gold_canonical_index": g_idx,
                    "prompt_index": p_idx,
                    "canonical_phone": c_phone,
                    "realized_phone": iv.realized_phone,
                    "error_type": etype,
                    "binary_label": bin_label,
                    "word_position": pos_type,
                    "is_word_final_consonant": (pos_type == "word_final" and c_phone in CONSONANTS),
                    "confusable_group": conf_group,
                    "status": r_pred["status"],
                    "s_error": r_pred["scores"]["s_error"],
                    "posterior_occupancy_normalized_error": (
                        r_pred["raw_scores"].get("posterior_occupancy_normalized_error")
                        if isinstance(r_pred.get("raw_scores"), dict) else None
                    ),
                    "finite": r_pred["finite"],
                }
                offline_records.append(rec)

        all_scorer_rows[scorer] = offline_records

        # Compute metrics on answered/ok records
        valid_recs = [r for r in offline_records if r["status"] == "ok" and r["s_error"] is not None]
        print(f"  Total offline aligned records: {len(offline_records)}, valid ok: {len(valid_recs)}")

        # 1. Binary AP
        y_bin = np.array([r["binary_label"] for r in valid_recs], dtype=np.int64)
        s_bin = np.array([r["s_error"] for r in valid_recs], dtype=np.float64)
        bin_ap = average_precision_binary(y_bin, s_bin) if len(np.unique(y_bin)) == 2 else float("nan")
        bin_auc = roc_auc_binary(y_bin, s_bin) if len(np.unique(y_bin)) == 2 else float("nan")

        # 2. Omission AP (deletion vs correct only)
        omiss_recs = [r for r in valid_recs if r["error_type"] in ("deletion", "correct")]
        y_omiss = np.array([1 if r["error_type"] == "deletion" else 0 for r in omiss_recs], dtype=np.int64)
        s_omiss = np.array([r["s_error"] for r in omiss_recs], dtype=np.float64)
        omiss_ap = average_precision_binary(y_omiss, s_omiss) if len(np.unique(y_omiss)) == 2 else float("nan")
        omiss_auc = roc_auc_binary(y_omiss, s_omiss) if len(np.unique(y_omiss)) == 2 else float("nan")

        # 3. Substitution AP (substitution vs correct only)
        sub_recs = [r for r in valid_recs if r["error_type"] in ("substitution", "correct")]
        y_sub = np.array([1 if r["error_type"] == "substitution" else 0 for r in sub_recs], dtype=np.int64)
        s_sub = np.array([r["s_error"] for r in sub_recs], dtype=np.float64)
        sub_ap = average_precision_binary(y_sub, s_sub) if len(np.unique(y_sub)) == 2 else float("nan")
        sub_auc = roc_auc_binary(y_sub, s_sub) if len(np.unique(y_sub)) == 2 else float("nan")

        # 4. Operating points at FPR ceilings 0.05, 0.10, 0.15 on omission
        correct_scores = np.array([r["s_error"] for r in valid_recs if r["error_type"] == "correct"], dtype=np.float64)
        tau_05 = diagnostic_threshold(correct_scores, 0.05) if len(correct_scores) else float("nan")
        tau_10 = diagnostic_threshold(correct_scores, 0.10) if len(correct_scores) else float("nan")
        tau_15 = diagnostic_threshold(correct_scores, 0.15) if len(correct_scores) else float("nan")

        op_05 = compute_operating(y_omiss, s_omiss, tau_05) if len(y_omiss) and math.isfinite(tau_05) else {}
        op_10 = compute_operating(y_omiss, s_omiss, tau_10) if len(y_omiss) and math.isfinite(tau_10) else {}
        op_15 = compute_operating(y_omiss, s_omiss, tau_15) if len(y_omiss) and math.isfinite(tau_15) else {}

        # 5. Greedy CTC PER (evaluated per utterance against realized phones)
        total_edits, total_subs, total_dels, total_ins, total_refs = 0, 0, 0, 0, 0
        for u in val_utts:
            spk, utt_name = u.split("/")
            ref_ids = gold_data[u]["realized_phone_ids"]
            u_p = utt_preds.get((spk, utt_name))
            dec_ids = u_p.get("decoded_phone_ids") if u_p else []
            if dec_ids is None:
                dec_ids = []
            edits, s_ed, d_ed, i_ed = compute_levenshtein_edits(ref_ids, dec_ids)
            total_edits += edits
            total_subs += s_ed
            total_dels += d_ed
            total_ins += i_ed
            total_refs += len(ref_ids)

        per = total_edits / total_refs if total_refs > 0 else float("nan")
        sub_rate = total_subs / total_refs if total_refs > 0 else float("nan")
        del_rate = total_dels / total_refs if total_refs > 0 else float("nan")
        ins_rate = total_ins / total_refs if total_refs > 0 else float("nan")

        # 6. Occupancy-normalized scores evaluation (auxiliary)
        occ_valid = [r for r in valid_recs if r["posterior_occupancy_normalized_error"] is not None]
        if occ_valid:
            y_occ_om = np.array([1 if r["error_type"] == "deletion" else 0 for r in occ_valid if r["error_type"] in ("deletion", "correct")])
            s_occ_om = np.array([r["posterior_occupancy_normalized_error"] for r in occ_valid if r["error_type"] in ("deletion", "correct")])
            occ_omiss_ap = average_precision_binary(y_occ_om, s_occ_om) if len(np.unique(y_occ_om)) == 2 else float("nan")
            y_occ_bin = np.array([r["binary_label"] for r in occ_valid])
            s_occ_bin = np.array([r["posterior_occupancy_normalized_error"] for r in occ_valid])
            occ_bin_ap = average_precision_binary(y_occ_bin, s_occ_bin) if len(np.unique(y_occ_bin)) == 2 else float("nan")
        else:
            occ_omiss_ap, occ_bin_ap = float("nan"), float("nan")

        benchmark_results[scorer] = {
            "scorer": scorer,
            "coverage": {
                "total_aligned_records": len(offline_records),
                "valid_ok_records": len(valid_recs),
                "abstained_records": len(offline_records) - len(valid_recs),
                "ok_fraction": len(valid_recs) / len(offline_records) if offline_records else 0.0,
            },
            "populations": {
                "binary": {"total": len(y_bin), "errors": int(y_bin.sum()), "correct": len(y_bin) - int(y_bin.sum())},
                "omission": {"total": len(y_omiss), "deletions": int(y_omiss.sum()), "correct": len(y_omiss) - int(y_omiss.sum())},
                "substitution": {"total": len(y_sub), "substitutions": int(y_sub.sum()), "correct": len(y_sub) - int(y_sub.sum())},
            },
            "primary_metrics": {
                "omission_ap": omiss_ap,
                "omission_roc_auc": omiss_auc,
                "substitution_ap": sub_ap,
                "substitution_roc_auc": sub_auc,
                "binary_ap": bin_ap,
                "binary_roc_auc": bin_auc,
            },
            "operating_points": {
                "tau_fpr15": op_15.get("threshold"),
                "omission_recall_at_fpr15": op_15.get("recall"),
                "omission_precision_at_fpr15": op_15.get("precision"),
                "actual_fpr15": op_15.get("fpr"),
                "omission_fnr_at_fpr15": op_15.get("fnr"),
                "f1_at_fpr15": op_15.get("f1"),
                "tau_fpr10": op_10.get("threshold"),
                "omission_recall_at_fpr10": op_10.get("recall"),
                "actual_fpr10": op_10.get("fpr"),
                "tau_fpr05": op_05.get("threshold"),
                "omission_recall_at_fpr05": op_05.get("recall"),
                "actual_fpr05": op_05.get("fpr"),
            },
            "greedy_ctc_per": {
                "ctc_per": per,
                "sub_rate": sub_rate,
                "del_rate": del_rate,
                "ins_rate": ins_rate,
                "total_edits": total_edits,
                "total_subs": total_subs,
                "total_dels": total_dels,
                "total_ins": total_ins,
                "total_ref_tokens": total_refs,
            },
            "occupancy_normalized_auxiliary": {
                "omission_ap": occ_omiss_ap,
                "binary_ap": occ_bin_ap,
            }
        }

        print(f"  Omission AP:     {omiss_ap:.6f}")
        print(f"  Substitution AP: {sub_ap:.6f}")
        print(f"  Binary AP:       {bin_ap:.6f}")
        print(f"  Recall @ FPR15:  {op_15.get('recall')}")
        print(f"  Precision@FPR15: {op_15.get('precision')}")
        print(f"  Achieved FPR15:  {op_15.get('fpr')}")
        print(f"  Greedy CTC PER:  {per:.6f} (refs: {total_refs})")

    # 4. Primary contrast: GOP-CTC-SF-SD minus GOP-CTC-SF-S
    delta_omiss_ap = benchmark_results["sf_sd"]["primary_metrics"]["omission_ap"] - benchmark_results["sf_s"]["primary_metrics"]["omission_ap"]
    delta_sub_ap = benchmark_results["sf_sd"]["primary_metrics"]["substitution_ap"] - benchmark_results["sf_s"]["primary_metrics"]["substitution_ap"]
    delta_bin_ap = benchmark_results["sf_sd"]["primary_metrics"]["binary_ap"] - benchmark_results["sf_s"]["primary_metrics"]["binary_ap"]
    delta_rec_fpr15 = (benchmark_results["sf_sd"]["operating_points"]["omission_recall_at_fpr15"] or 0) - (benchmark_results["sf_s"]["operating_points"]["omission_recall_at_fpr15"] or 0)

    print("\n=======================================================")
    print("PRIMARY CONTRAST: GOP-CTC-SF-SD minus GOP-CTC-SF-S")
    print(f"  Delta Omission AP:     {delta_omiss_ap:+.6f}")
    print(f"  Delta Substitution AP: {delta_sub_ap:+.6f}")
    print(f"  Delta Binary AP:       {delta_bin_ap:+.6f}")
    print(f"  Delta Recall @ FPR15:  {delta_rec_fpr15:+.6f}")
    print("=======================================================\n")

    # 5. Breakdowns by Speaker (NJS vs RRBI)
    speaker_breakdowns = []
    for scorer in scorers:
        for spk in ["NJS", "RRBI"]:
            spk_recs = [r for r in all_scorer_rows[scorer] if r["speaker"] == spk and r["status"] == "ok" and r["s_error"] is not None]
            y_b = np.array([r["binary_label"] for r in spk_recs], dtype=np.int64)
            s_b = np.array([r["s_error"] for r in spk_recs], dtype=np.float64)
            spk_bin_ap = average_precision_binary(y_b, s_b) if len(np.unique(y_b)) == 2 else float("nan")

            om_recs = [r for r in spk_recs if r["error_type"] in ("deletion", "correct")]
            y_o = np.array([1 if r["error_type"] == "deletion" else 0 for r in om_recs], dtype=np.int64)
            s_o = np.array([r["s_error"] for r in om_recs], dtype=np.float64)
            spk_om_ap = average_precision_binary(y_o, s_o) if len(np.unique(y_o)) == 2 else float("nan")

            sub_recs = [r for r in spk_recs if r["error_type"] in ("substitution", "correct")]
            y_s = np.array([1 if r["error_type"] == "substitution" else 0 for r in sub_recs], dtype=np.int64)
            s_s = np.array([r["s_error"] for r in sub_recs], dtype=np.float64)
            spk_sub_ap = average_precision_binary(y_s, s_s) if len(np.unique(y_s)) == 2 else float("nan")

            cor_scores = np.array([r["s_error"] for r in spk_recs if r["error_type"] == "correct"], dtype=np.float64)
            spk_tau15 = diagnostic_threshold(cor_scores, 0.15) if len(cor_scores) else float("nan")
            spk_op15 = compute_operating(y_o, s_o, spk_tau15) if len(y_o) and math.isfinite(spk_tau15) else {}

            speaker_breakdowns.append({
                "scorer": scorer,
                "speaker": spk,
                "total_tokens": len(spk_recs),
                "deletions": int(y_o.sum()) if len(y_o) else 0,
                "substitutions": int(y_s.sum()) if len(y_s) else 0,
                "correct": len(cor_scores),
                "omission_ap": spk_om_ap,
                "substitution_ap": spk_sub_ap,
                "binary_ap": spk_bin_ap,
                "omission_recall_at_fpr15": spk_op15.get("recall"),
                "achieved_fpr15": spk_op15.get("fpr"),
            })

    # 6. Breakdown by Word-Final Consonants
    wfc_breakdowns = []
    for scorer in scorers:
        for stratum, is_wfc in [("word_final_consonants", True), ("other_positions", False)]:
            st_recs = [r for r in all_scorer_rows[scorer] if r["is_word_final_consonant"] == is_wfc and r["status"] == "ok" and r["s_error"] is not None]
            y_b = np.array([r["binary_label"] for r in st_recs], dtype=np.int64)
            s_b = np.array([r["s_error"] for r in st_recs], dtype=np.float64)
            st_bin_ap = average_precision_binary(y_b, s_b) if len(np.unique(y_b)) == 2 else float("nan")

            om_recs = [r for r in st_recs if r["error_type"] in ("deletion", "correct")]
            y_o = np.array([1 if r["error_type"] == "deletion" else 0 for r in om_recs], dtype=np.int64)
            s_o = np.array([r["s_error"] for r in om_recs], dtype=np.float64)
            st_om_ap = average_precision_binary(y_o, s_o) if len(np.unique(y_o)) == 2 else float("nan")

            wfc_breakdowns.append({
                "scorer": scorer,
                "stratum": stratum,
                "total_tokens": len(st_recs),
                "deletions": int(y_o.sum()) if len(y_o) else 0,
                "errors": int(y_b.sum()) if len(y_b) else 0,
                "omission_ap": st_om_ap,
                "binary_ap": st_bin_ap,
            })

    # 7. Breakdown by Confusable Phone Groups
    conf_breakdowns = []
    for scorer in scorers:
        for cgroup in list(CONFUSABLE_GROUPS.keys()) + ["other"]:
            cg_recs = [r for r in all_scorer_rows[scorer] if r["confusable_group"] == cgroup and r["status"] == "ok" and r["s_error"] is not None]
            if not cg_recs:
                continue
            y_b = np.array([r["binary_label"] for r in cg_recs], dtype=np.int64)
            s_b = np.array([r["s_error"] for r in cg_recs], dtype=np.float64)
            cg_bin_ap = average_precision_binary(y_b, s_b) if len(np.unique(y_b)) == 2 else float("nan")

            om_recs = [r for r in cg_recs if r["error_type"] in ("deletion", "correct")]
            y_o = np.array([1 if r["error_type"] == "deletion" else 0 for r in om_recs], dtype=np.int64)
            s_o = np.array([r["s_error"] for r in om_recs], dtype=np.float64)
            cg_om_ap = average_precision_binary(y_o, s_o) if len(np.unique(y_o)) == 2 else float("nan")

            conf_breakdowns.append({
                "scorer": scorer,
                "confusable_group": cgroup,
                "total_tokens": len(cg_recs),
                "deletions": int(y_o.sum()) if len(y_o) else 0,
                "errors": int(y_b.sum()) if len(y_b) else 0,
                "omission_ap": cg_om_ap,
                "binary_ap": cg_bin_ap,
            })

    # 8. Output and artifact generation
    eval_dir = REPO_ROOT / "manifests/RQ1-P-SCORE/evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Save full BENCHMARK_REPORT.json
    report = {
        "metadata": {
            "experiment": "RQ1-P-SCORE",
            "model": "Baseline_R_E0",
            "checkpoint_sha256": "a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b",
            "contract_sha256": CONTRACT_SHA256,
            "split_sha256": SPLIT_SHA256,
            "cache_sha256": "a699fd9378c86cd348a0628a144ecacfef491ea8f90e0edd23b4e46932e3e829",
            "authorization_sha256": "5432c000111a8244949c241e3298c422b15e5062d5a6835e2db64383f8758b15",
            "evaluation_split": "Fold-1 VAL (NJS & RRBI)",
            "total_utterances": 300,
            "total_canonical_gold_tokens": 9956,
            "total_aligned_tokens": 9919,
        },
        "primary_contrast_sf_sd_minus_sf_s": {
            "delta_omission_ap": delta_omiss_ap,
            "delta_substitution_ap": delta_sub_ap,
            "delta_binary_ap": delta_bin_ap,
            "delta_omission_recall_at_fpr15": delta_rec_fpr15,
        },
        "scorers": benchmark_results,
        "speaker_breakdown": speaker_breakdowns,
        "word_final_consonant_breakdown": wfc_breakdowns,
        "confusable_phone_breakdown": conf_breakdowns,
    }

    report_path = eval_dir / "BENCHMARK_REPORT.json"
    with open(report_path, "w") as f:
        json.dump(clean_for_json(report), f, indent=2)
    print(f"[SAVE] Benchmark report written to: {report_path}")

    # Save metrics_summary.csv
    csv_path = eval_dir / "metrics_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "scorer", "total_records", "valid_ok_records", "omission_ap", "substitution_ap",
            "binary_ap", "omission_recall_at_fpr15", "omission_precision_at_fpr15",
            "achieved_fpr15", "ctc_per", "occ_norm_omission_ap", "occ_norm_binary_ap"
        ])
        w.writeheader()
        for s in scorers:
            res = benchmark_results[s]
            w.writerow({
                "scorer": s,
                "total_records": res["coverage"]["total_aligned_records"],
                "valid_ok_records": res["coverage"]["valid_ok_records"],
                "omission_ap": res["primary_metrics"]["omission_ap"],
                "substitution_ap": res["primary_metrics"]["substitution_ap"],
                "binary_ap": res["primary_metrics"]["binary_ap"],
                "omission_recall_at_fpr15": res["operating_points"]["omission_recall_at_fpr15"],
                "omission_precision_at_fpr15": res["operating_points"]["omission_precision_at_fpr15"],
                "achieved_fpr15": res["operating_points"]["actual_fpr15"],
                "ctc_per": res["greedy_ctc_per"]["ctc_per"],
                "occ_norm_omission_ap": res["occupancy_normalized_auxiliary"]["omission_ap"],
                "occ_norm_binary_ap": res["occupancy_normalized_auxiliary"]["binary_ap"],
            })
    print(f"[SAVE] Metrics summary written to: {csv_path}")

    # Save metrics_by_speaker.csv
    spk_csv_path = eval_dir / "metrics_by_speaker.csv"
    with open(spk_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(speaker_breakdowns[0].keys()))
        w.writeheader()
        w.writerows(speaker_breakdowns)
    print(f"[SAVE] Metrics by speaker written to: {spk_csv_path}")

    # Save breakdown_word_final_consonants.csv
    wfc_csv_path = eval_dir / "breakdown_word_final_consonants.csv"
    with open(wfc_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(wfc_breakdowns[0].keys()))
        w.writeheader()
        w.writerows(wfc_breakdowns)
    print(f"[SAVE] Word-final consonants breakdown written to: {wfc_csv_path}")

    # Save breakdown_confusable_phones.csv
    cg_csv_path = eval_dir / "breakdown_confusable_phones.csv"
    with open(cg_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(conf_breakdowns[0].keys()))
        w.writeheader()
        w.writerows(conf_breakdowns)
    print(f"[SAVE] Confusable phones breakdown written to: {cg_csv_path}")

    # Save compressed offline_token_evaluation.jsonl.gz for the primary contrast (sf_s and sf_sd)
    for scorer in scorers:
        out_jsonl_gz = eval_dir / f"offline_token_evaluation_{scorer}.jsonl.gz"
        with gzip.open(out_jsonl_gz, "wt", encoding="utf-8") as gf:
            for r in all_scorer_rows[scorer]:
                gf.write(json.dumps(r) + "\n")
        print(f"[SAVE] Offline token records written to: {out_jsonl_gz}")

    # Also copy run_manifest.json, SHA256SUMS from runs_dir
    (eval_dir / "run_manifest.json").write_text((runs_dir / "run_manifest.json").read_text())
    (eval_dir / "SHA256SUMS").write_text((runs_dir / "SHA256SUMS").read_text())
    print("[SUCCESS] All benchmark reports, breakdowns, and artifacts successfully generated!")


if __name__ == "__main__":
    main()
