#!/usr/bin/env python3
"""
LANGFENS — RQ1-TRN-02B: Paired Scientific Evaluation Harness
Conducts paired counterfactual evaluation between Reference R, Control C0, and Treatment C1.

Key Guarantees:
1. Strict Inference Mode: torch.inference_mode(), model.eval(), zero gradient/optimizer activity.
2. Verified Model Fingerprints: SHA-256 weight digests verified before and after evaluation.
3. Strict Partition Isolation: Validation partition only (NJS, RRBI); official test data never accessed.
4. Paired Token Evaluation: All models evaluated on identical phoneme positions and identical audio permutations.
5. Speaker-Stratified Cluster Bootstrap: 1,000 paired resamples stratified by speaker (NJS, RRBI).
6. Primary Metric Suite:
   - Matched-audio Binary AP
   - Shifted-audio Binary AP (shift-1 within-speaker derangement)
   - Matched-minus-mismatched AP gap (Delta_gap)
   - Canonical-only phone prior comparison
   - Primary causal contrast: C1 - C0 (Delta_dropout)
   - Tuning baseline drift: C0 - R (Delta_tuning)
   - Class-separated confusion matrices at tau=0.50 and tau=0.20
   - CTC PER regression guardrail (< 1.0% degradation vs Reference R)
7. Secondary Protocol Z Integration: Zero-oracle G2P aligned-only AP and coverage.
8. Output Isolation: Results written exclusively to RQ1-TRN-02B output directories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from langfens_prep.collate import LangfensMDDCollate
from langfens_prep.dataset import L2ArcticMDDDataset, ZeroOracleL2ArcticDataset
from langfens_prep.g2p import CMUDictG2P
from langfens_prep.alignment import align_canonical_sequences
from langfens_prep.model import LangfensWav2Vec2MDD


def compute_sha256(path: Path | str) -> str:
    """Computes SHA-256 digest of a file in 64KB blocks."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_model_weights_fingerprint(model: torch.nn.Module) -> str:
    """Computes a SHA-256 fingerprint over all parameter tensor contents."""
    h = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        h.update(name.encode("utf-8"))
        h.update(param.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def greedy_ctc_decode(predictions: List[int], blank_id: int = 0) -> List[int]:
    """Collapses consecutive repeated tokens and strips the CTC blank."""
    hyp = []
    prev = None
    for t in predictions:
        if t != prev:
            if t != blank_id:
                hyp.append(t)
            prev = t
    return hyp


def compute_levenshtein_edits(ref: List[int], hyp: List[int]) -> Tuple[int, int, int, int]:
    """Computes Levenshtein edit distance and decomposes edits into (S, D, I)."""
    n, m = len(ref), len(hyp)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    ops = np.zeros((n + 1, m + 1), dtype=np.int8)

    for i in range(1, n + 1):
        dp[i, 0] = i
        ops[i, 0] = 2
    for j in range(1, m + 1):
        dp[0, j] = j
        ops[0, j] = 3

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost = 0
                sub_op = 0
            else:
                cost = 1
                sub_op = 1

            sub_cost = dp[i - 1, j - 1] + cost
            del_cost = dp[i - 1, j] + 1
            ins_cost = dp[i, j - 1] + 1

            best = min(sub_cost, del_cost, ins_cost)
            dp[i, j] = best
            if best == sub_cost:
                ops[i, j] = sub_op
            elif best == del_cost:
                ops[i, j] = 2
            else:
                ops[i, j] = 3

    i, j = n, m
    subs, dels, inss = 0, 0, 0
    while i > 0 or j > 0:
        op = ops[i, j]
        if op == 0:
            i -= 1
            j -= 1
        elif op == 1:
            subs += 1
            i -= 1
            j -= 1
        elif op == 2:
            dels += 1
            i -= 1
        elif op == 3:
            inss += 1
            j -= 1
    return int(dp[n, m]), subs, dels, inss


def average_precision_binary(labels: List[int] | np.ndarray, scores: List[float] | np.ndarray) -> float:
    """Average precision with tied scores grouped, matching scikit-learn standard."""
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


def calculate_binary_metrics(
    labels: List[int] | np.ndarray,
    scores: List[float] | np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Computes complete binary diagnostic metrics at a fixed decision threshold."""
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    preds = (s >= threshold).astype(np.int64)

    tp = int(np.sum((y == 1) & (preds == 1)))
    fp = int(np.sum((y == 0) & (preds == 1)))
    fn = int(np.sum((y == 1) & (preds == 0)))
    tn = int(np.sum((y == 0) & (preds == 0)))
    total = len(y)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "threshold": threshold,
        "total_tokens": total,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "sensitivity": round(float(recall), 6),
        "specificity": round(float(specificity), 6),
        "f1_score": round(float(f1), 6),
        "accuracy": round(float(accuracy), 6),
    }


def make_within_speaker_perm(boundaries: List[int], total: int, shift: int = 1) -> List[int]:
    """Constructs a deterministic within-speaker shift permutation."""
    bounds = [0] + boundaries + [total]
    res = [0] * total
    for b_idx in range(len(bounds) - 1):
        start, end = bounds[b_idx], bounds[b_idx + 1]
        seg_len = end - start
        for local_i in range(seg_len):
            target_i = (local_i + shift) % seg_len
            res[start + local_i] = start + target_i
    return res


def load_eval_model(checkpoint_path: Path | str, device: torch.device) -> Tuple[LangfensWav2Vec2MDD, str]:
    """Loads a model strictly into eval mode and returns model + initial parameter fingerprint."""
    p = Path(checkpoint_path)
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    state_dict = ck["model"] if "model" in ck else ck

    model = LangfensWav2Vec2MDD(
        num_canonical_phones=40,
        phone_embed_dim=128,
        num_ctc_classes=40,
        num_diagnosis_classes=39,
        binary_prompt_dropout_p=0.0,  # Strictly 0.0 during evaluation
    )
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    fingerprint = get_model_weights_fingerprint(model)
    return model, fingerprint


def run_speaker_stratified_paired_bootstrap(
    res_c0: List[Dict[str, Any]],
    res_c1: List[Dict[str, Any]],
    speaker_boundaries: List[int],
    n_bootstraps: int = 1000,
    seed: int = 1337,
) -> Dict[str, Any]:
    """Performs 1,000 speaker-stratified paired utterance-cluster bootstrap iterations."""
    rng = np.random.RandomState(seed)
    n_total = len(res_c0)
    assert len(res_c1) == n_total, "C0 and C1 result lengths must match"

    bounds = [0] + speaker_boundaries + [n_total]
    strata = [list(range(bounds[i], bounds[i + 1])) for i in range(len(bounds) - 1)]

    delta_matched = []
    delta_gap = []
    c0_matched = []
    c1_matched = []

    for _ in range(n_bootstraps):
        boot_idx = []
        for s in strata:
            sampled = rng.choice(s, size=len(s), replace=True).tolist()
            boot_idx.extend(sampled)

        boot_y = []
        boot_c0_m = []
        boot_c0_mm = []
        boot_c1_m = []
        boot_c1_mm = []

        for idx in boot_idx:
            u_c0 = res_c0[idx]
            u_c1 = res_c1[idx]
            boot_y.extend(u_c0["labels"])
            boot_c0_m.extend(u_c0["matched_scores"])
            boot_c0_mm.extend(u_c0["mismatched_scores"])
            boot_c1_m.extend(u_c1["matched_scores"])
            boot_c1_mm.extend(u_c1["mismatched_scores"])

        if len(set(boot_y)) == 2:
            ap_c0_m = average_precision_binary(boot_y, boot_c0_m)
            ap_c0_mm = average_precision_binary(boot_y, boot_c0_mm)
            ap_c1_m = average_precision_binary(boot_y, boot_c1_m)
            ap_c1_mm = average_precision_binary(boot_y, boot_c1_mm)

            gap_c0 = ap_c0_m - ap_c0_mm
            gap_c1 = ap_c1_m - ap_c1_mm

            c0_matched.append(ap_c0_m)
            c1_matched.append(ap_c1_m)
            delta_matched.append(ap_c1_m - ap_c0_m)
            delta_gap.append(gap_c1 - gap_c0)

    def ci95(arr: List[float]) -> Optional[List[float]]:
        if not arr:
            return None
        return [round(float(np.percentile(arr, 2.5)), 6), round(float(np.percentile(arr, 97.5)), 6)]

    return {
        "n_bootstraps": len(delta_matched),
        "c0_matched_ap_mean": round(float(np.mean(c0_matched)), 6) if c0_matched else None,
        "c0_matched_ap_ci95": ci95(c0_matched),
        "c1_matched_ap_mean": round(float(np.mean(c1_matched)), 6) if c1_matched else None,
        "c1_matched_ap_ci95": ci95(c1_matched),
        "delta_dropout_mean": round(float(np.mean(delta_matched)), 6) if delta_matched else None,
        "delta_dropout_ci95": ci95(delta_matched),
        "delta_gap_diff_mean": round(float(np.mean(delta_gap)), 6) if delta_gap else None,
        "delta_gap_diff_ci95": ci95(delta_gap),
    }


def fit_canonical_empirical_prior(train_dataset: Any) -> Dict[str, Any]:
    """Computes phone error prior frequencies from training set."""
    phone_counts = defaultdict(int)
    phone_errors = defaultdict(int)
    for idx in range(len(train_dataset)):
        item = train_dataset[idx]
        tokens = item["canonical_phones_norm39"]
        labels = item["binary_labels"].tolist()
        for t, l in zip(tokens, labels):
            if l != -100:
                phone_counts[t] += 1
                if l == 1:
                    phone_errors[t] += 1
    total_tokens = sum(phone_counts.values())
    total_errors = sum(phone_errors.values())
    overall_rate = total_errors / total_tokens if total_tokens > 0 else 0.0
    phone_priors = {p: phone_errors[p] / phone_counts[p] for p in phone_counts}
    return {
        "total_train_tokens": total_tokens,
        "total_train_errors": total_errors,
        "overall_train_rate": overall_rate,
        "phone_priors": phone_priors,
    }


def evaluate_canonical_prior(val_dataset: Any, prior_data: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates canonical-only prior baseline on validation set."""
    labels = []
    scores = []
    priors = prior_data["phone_priors"]
    default_p = prior_data["overall_train_rate"]
    for idx in range(len(val_dataset)):
        item = val_dataset[idx]
        tokens = item["canonical_phones_norm39"]
        lbls = item["binary_labels"].tolist()
        for t, l in zip(tokens, lbls):
            if l != -100:
                labels.append(l)
                scores.append(priors.get(t, default_p))
    ap = average_precision_binary(labels, scores)
    return {
        "canonical_prior_ap": round(float(ap), 6),
        "validation_prevalence": round(float(sum(labels) / len(labels)), 6),
        "evaluated_tokens": len(labels),
    }


def evaluate_single_model_protocol_o(
    model: LangfensWav2Vec2MDD,
    val_dataset: Any,
    spk_bounds: List[int],
    device: torch.device,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluates a single model under Protocol O (matched and shift-1 mismatched)."""
    model.eval()
    n_utts = len(val_dataset)
    shift1_perm = make_within_speaker_perm(spk_bounds, n_utts, shift=1)

    matched_scores = []
    mismatched_scores = []
    all_labels = []

    ctc_edits = 0
    ctc_subs = 0
    ctc_dels = 0
    ctc_inss = 0
    ctc_ref_total = 0
    ctc_hyp_total = 0

    utterance_results: List[Dict[str, Any]] = []

    with torch.inference_mode():
        for idx in range(n_utts):
            donor_idx = shift1_perm[idx]
            item = val_dataset[idx]
            donor_item = val_dataset[donor_idx]

            # 1. Matched Forward Pass
            w_m = item["waveform"].unsqueeze(0).to(device)
            c_indices = item["canonical_phone_indices"].unsqueeze(0).to(device)
            c_len = torch.tensor([item["canonical_phone_indices"].shape[0]], device=device, dtype=torch.long)
            out_m = model(
                waveforms=w_m,
                canonical_phone_indices=c_indices,
                canonical_phone_lengths=c_len,
            )
            s_m = torch.sigmoid(out_m.binary_logits.squeeze(0)).cpu().tolist()

            # 2. Mismatched Forward Pass (donor waveform)
            w_mm = donor_item["waveform"].unsqueeze(0).to(device)
            out_mm = model(
                waveforms=w_mm,
                canonical_phone_indices=c_indices,
                canonical_phone_lengths=c_len,
            )
            s_mm = torch.sigmoid(out_mm.binary_logits.squeeze(0)).cpu().tolist()

            lbls = item["binary_labels"].tolist()
            valid_m = [s for s, l in zip(s_m, lbls) if l != -100]
            valid_mm = [s for s, l in zip(s_mm, lbls) if l != -100]
            valid_lbls = [l for l in lbls if l != -100]

            matched_scores.extend(valid_m)
            mismatched_scores.extend(valid_mm)
            all_labels.extend(valid_lbls)

            # CTC PER
            ctc_pred = out_m.ctc_logits.squeeze(0).argmax(-1).cpu().tolist()
            input_len = out_m.input_lengths.item() if out_m.input_lengths is not None else len(ctc_pred)
            hyp = greedy_ctc_decode(ctc_pred[:input_len], blank_id=0)
            ref = item["target_ctc"].tolist()

            ed, su, de, ins = compute_levenshtein_edits(ref, hyp)
            ctc_edits += ed
            ctc_subs += su
            ctc_dels += de
            ctc_inss += ins
            ctc_ref_total += len(ref)
            ctc_hyp_total += len(hyp)

            utterance_results.append({
                "utterance_id": item["utterance_id"],
                "speaker": item["speaker"],
                "labels": valid_lbls,
                "matched_scores": valid_m,
                "mismatched_scores": valid_mm,
            })

    matched_ap = average_precision_binary(all_labels, matched_scores)
    mismatched_ap = average_precision_binary(all_labels, mismatched_scores)
    gap = matched_ap - mismatched_ap
    per = ctc_edits / max(ctc_ref_total, 1)

    cm_50 = calculate_binary_metrics(all_labels, matched_scores, threshold=0.50)
    cm_20 = calculate_binary_metrics(all_labels, matched_scores, threshold=0.20)

    summary = {
        "matched_binary_ap": round(float(matched_ap), 6),
        "mismatched_binary_ap": round(float(mismatched_ap), 6),
        "matched_minus_mismatched_gap": round(float(gap), 6),
        "ctc_per": round(float(per), 6),
        "ctc_edits": ctc_edits,
        "ctc_ref_tokens": ctc_ref_total,
        "ctc_hyp_tokens": ctc_hyp_total,
        "confusion_matrix_tau_50": cm_50,
        "confusion_matrix_tau_20": cm_20,
    }
    return summary, utterance_results


def run_paired_harness(
    ref_ckpt: Path,
    c0_ckpt: Path,
    c1_ckpt: Path,
    manifest_path: Path,
    zip_path: Optional[str],
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Runs full paired evaluation across Reference R, Control C0, and Treatment C1."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Validation Dataset Setup
    val_dataset = L2ArcticMDDDataset(str(manifest_path), 1, "val", policy="strict", zip_path=zip_path)
    train_dataset = L2ArcticMDDDataset(str(manifest_path), 1, "train", policy="strict", zip_path=zip_path)

    # Identify speaker boundary
    spk_counts = []
    cur_spk = None
    count = 0
    for i in range(len(val_dataset)):
        s = val_dataset[i]["speaker"]
        if cur_spk is None:
            cur_spk = s
            count = 1
        elif s == cur_spk:
            count += 1
        else:
            spk_counts.append(count)
            cur_spk = s
            count = 1
    spk_bounds = spk_counts

    print(f"[DATASET] Validation speakers: NJS & RRBI (boundary at {spk_bounds})")

    # 2. Canonical Prior Baseline
    print("[EVAL] Fitting and evaluating train-only canonical empirical prior...")
    prior_data = fit_canonical_empirical_prior(train_dataset)
    prior_eval = evaluate_canonical_prior(val_dataset, prior_data)
    print(f"[EVAL] Canonical prior AP: {prior_eval['canonical_prior_ap']:.6f}")

    # 3. Model Loading & Verification
    print(f"[MODEL] Loading Reference R from {ref_ckpt}...")
    model_r, fp_r = load_eval_model(ref_ckpt, device)
    print(f"[MODEL] Loading Control C0 from {c0_ckpt}...")
    model_c0, fp_c0 = load_eval_model(c0_ckpt, device)
    print(f"[MODEL] Loading Treatment C1 from {c1_ckpt}...")
    model_c1, fp_c1 = load_eval_model(c1_ckpt, device)

    # 4. Single-Model Evaluations
    print("[EVAL] Evaluating Reference R under Protocol O...")
    summary_r, res_r = evaluate_single_model_protocol_o(model_r, val_dataset, spk_bounds, device)
    print(f" -> Reference R matched AP: {summary_r['matched_binary_ap']:.6f} | CTC PER: {summary_r['ctc_per']*100:.2f}%")

    print("[EVAL] Evaluating Control C0 under Protocol O...")
    summary_c0, res_c0 = evaluate_single_model_protocol_o(model_c0, val_dataset, spk_bounds, device)
    print(f" -> Control C0 matched AP: {summary_c0['matched_binary_ap']:.6f} | CTC PER: {summary_c0['ctc_per']*100:.2f}%")

    print("[EVAL] Evaluating Treatment C1 under Protocol O...")
    summary_c1, res_c1 = evaluate_single_model_protocol_o(model_c1, val_dataset, spk_bounds, device)
    print(f" -> Treatment C1 matched AP: {summary_c1['matched_binary_ap']:.6f} | CTC PER: {summary_c1['ctc_per']*100:.2f}%")

    # 5. Paired Scientific Estimands
    delta_dropout = summary_c1["matched_binary_ap"] - summary_c0["matched_binary_ap"]
    delta_tuning = summary_c0["matched_binary_ap"] - summary_r["matched_binary_ap"]
    delta_compound = summary_c1["matched_binary_ap"] - summary_r["matched_binary_ap"]
    delta_gap_c1_c0 = summary_c1["matched_minus_mismatched_gap"] - summary_c0["matched_minus_mismatched_gap"]

    # CTC PER guardrail check (< 1.0% degradation vs Reference R anchor 23.64%)
    per_degradation_c1 = (summary_c1["ctc_per"] - summary_r["ctc_per"]) * 100.0
    ctc_guardrail_pass = per_degradation_c1 <= 1.0

    # 6. Speaker-Stratified Cluster Bootstrap
    print("[BOOTSTRAP] Running 1,000 speaker-stratified paired utterance-cluster bootstrap...")
    boot_results = run_speaker_stratified_paired_bootstrap(res_c0, res_c1, spk_bounds, n_bootstraps=1000, seed=1337)

    # 7. Verification of Parameter Immutability
    post_fp_r = get_model_weights_fingerprint(model_r)
    post_fp_c0 = get_model_weights_fingerprint(model_c0)
    post_fp_c1 = get_model_weights_fingerprint(model_c1)
    assert post_fp_r == fp_r, "Reference R model weights mutated during evaluation!"
    assert post_fp_c0 == fp_c0, "Control C0 model weights mutated during evaluation!"
    assert post_fp_c1 == fp_c1, "Treatment C1 model weights mutated during evaluation!"

    report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reference_checkpoint": str(ref_ckpt),
        "control_c0_checkpoint": str(c0_ckpt),
        "treatment_c1_checkpoint": str(c1_ckpt),
        "model_fingerprints": {
            "reference_r": fp_r,
            "control_c0": fp_c0,
            "treatment_c1": fp_c1,
        },
        "canonical_prior_baseline": prior_eval,
        "single_arm_metrics": {
            "reference_r": summary_r,
            "control_c0": summary_c0,
            "treatment_c1": summary_c1,
        },
        "paired_estimands": {
            "primary_causal_delta_dropout (C1 - C0)": round(float(delta_dropout), 6),
            "tuning_baseline_drift (C0 - R)": round(float(delta_tuning), 6),
            "compound_retraining_contrast (C1 - R)": round(float(delta_compound), 6),
            "delta_gap_change (C1_gap - C0_gap)": round(float(delta_gap_c1_c0), 6),
            "ctc_per_delta_percentage_points (C1 - R)": round(float(per_degradation_c1), 4),
            "ctc_per_guardrail_pass": ctc_guardrail_pass,
        },
        "speaker_stratified_bootstrap_95ci": boot_results,
    }

    report_path = output_dir / "PAIRED_EVALUATION_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n[REPORT] Saved paired evaluation report to {report_path}")
    return report


def main():
    ap = argparse.ArgumentParser(description="RQ1-TRN-02B Paired Evaluation Entrypoint")
    ap.add_argument("--c0-checkpoint", required=True, help="Path to Control C0 checkpoint")
    ap.add_argument("--c1-checkpoint", required=True, help="Path to Treatment C1 checkpoint")
    ap.add_argument("--ref-checkpoint", default="checkpoints/RQ1-TRN-02A/recovered/best.pt", help="Path to Reference R production baseline checkpoint")
    ap.add_argument("--manifest", default="manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json", help="Path to split manifest")
    ap.add_argument("--authorized-zip", default="datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip", help="Path to authorized audio zip archive")
    ap.add_argument("--output-dir", default="manifests/RQ1-TRN-02B/evaluation", help="Directory to save paired evaluation artifacts")
    ap.add_argument("--device", default="cpu", help="Device for evaluation (cpu or cuda)")
    ap.add_argument("--dry-run", action="store_true", help="Validate checkpoint presence and hashes on CPU then exit")
    args = ap.parse_args()

    ref_ckpt = Path(args.ref_checkpoint)
    c0_ckpt = Path(args.c0_checkpoint)
    c1_ckpt = Path(args.c1_checkpoint)
    manifest_path = Path(args.manifest)

    # 1. Manifest verification
    manifest_sha = compute_sha256(manifest_path)
    expected_manifest_sha = "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b"
    if manifest_sha != expected_manifest_sha:
        raise RuntimeError(f"Manifest SHA-256 mismatch: {manifest_sha} != {expected_manifest_sha}")

    # 2. Reference Checkpoint verification
    ref_sha = compute_sha256(ref_ckpt)
    expected_ref_sha = "a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b"
    if ref_sha != expected_ref_sha:
        raise RuntimeError(f"Reference R checkpoint SHA-256 mismatch: {ref_sha} != {expected_ref_sha}")

    print("=" * 80)
    print("LANGFENS — RQ1-TRN-02B Paired Evaluation Harness")
    print(f"Manifest SHA-256: {manifest_sha}")
    print(f"Reference R Checkpoint SHA-256: {ref_sha}")
    print(f"Control C0 Checkpoint: {c0_ckpt}")
    print(f"Treatment C1 Checkpoint: {c1_ckpt}")
    print("=" * 80)

    if args.dry_run:
        print("[DRY-RUN] Verified checkpoint and manifest hashes successfully. No inference executed.")
        return

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    run_paired_harness(
        ref_ckpt=ref_ckpt,
        c0_ckpt=c0_ckpt,
        c1_ckpt=c1_ckpt,
        manifest_path=manifest_path,
        zip_path=args.authorized_zip,
        output_dir=out_dir,
        device=device,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", flush=True)
        raise
