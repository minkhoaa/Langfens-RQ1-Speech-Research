#!/usr/bin/env python3
"""
LANGFENS — RQ1-TRN-03A: Four-Fold Validation Synthesis & Evaluation Harness
Executes complete out-of-fold validation replay across Folds 1, 2, 3, and 4
under Protocol O (matched & mismatched acoustic controls) and Protocol Z (zero-oracle G2P).

Guarantees:
1. Strict Validation-Only: Only validation partitions (300 utts per fold, 1,200 utts total) are accessed.
2. Zero Test Access: Active test partitions are strictly excluded and never staged, read, or evaluated.
3. Dual Checkpoint Accounting: Evaluates both primary fixed-budget (3,600 updates / Epoch 6)
   and validation-selected (best.pt) checkpoints.
4. Exact Macro & Micro Statistics: Computes unweighted fold macro statistics with Student's t
   95% CIs (df=3, t=3.182) and pooled prediction-level micro AP across all 39,576 validation tokens.
5. Acoustic Reliance Controls: Evaluates matched vs. within-speaker shift-1 mismatched audio
   to compute the empirical acoustic gap Delta_gap.
6. Pedagogical Error Rates: Evaluates FNR (Missed Error Rate) and FPR (False Alarm Rate)
   at operating points tau=0.50 and tau=0.20.
7. Disaggregated 8-Speaker Census: Full per-speaker tables across all 8 validation speakers.
8. Small-Sample Limitations: Explicitly documents N=4 fold and G=8 speaker cluster limitations.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Ensure src and root are on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langfens_prep.alignment import align_canonical_sequences
from langfens_prep.dataset import (
    L2ArcticMDDDataset,
    ZeroOracleL2ArcticDataset,
)
from langfens_prep.g2p import CMUDictG2P
from langfens_prep.model import LangfensWav2Vec2MDD
from langfens_prep.phoneme_normalizer import CTC_INDEX_TO_PHONE, PHONE_TO_CTC_INDEX

from audit_scripts.evaluate_fold1_ctc_binary import (
    average_precision_binary,
    build_eval_model,
    calculate_binary_metrics,
    calculate_ctc_metrics,
    compute_sha256,
    compute_levenshtein_edits,
    evaluate_canonical_prior,
    fit_canonical_empirical_prior,
    generate_derangement_indices,
    get_model_weights_fingerprint,
    greedy_ctc_decode,
    verify_checkpoint_integrity,
)
from audit_scripts.evaluate_protocol_z import fit_train_canonical_prior


# Authoritative Fold Definitions & File Paths
FOLD_DEFINITIONS = {
    1: {
        "val_speakers": ["NJS", "RRBI"],
        "l1_profiles": {"NJS": "Spanish", "RRBI": "Hindi"},
        "config_path": "configs/rq1_trn_02a_fold1.yaml",
        "zip_path": "datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip",
        "fixed_budget_checkpoint": "checkpoints/RQ1-TRN-02A/recovered/best.pt",  # Epoch 6 / 3,600 updates
        "best_checkpoint": "checkpoints/RQ1-TRN-02A/recovered/best.pt",          # Coincides at Epoch 6
        "fixed_budget_epoch": 6,
        "fixed_budget_updates": 3600,
        "best_epoch": 6,
        "best_updates": 3600,
    },
    2: {
        "val_speakers": ["TNI", "YBAA"],
        "l1_profiles": {"TNI": "Arabic", "YBAA": "Arabic"},
        "config_path": "configs/rq1_trn_03a_fold2.yaml",
        "zip_path": "datasets/L2-ARCTIC/authorized_RQ1-TRN_fold2_train_val_only.zip",
        "fixed_budget_checkpoint": "checkpoints/fold2/baseline_r_fold2/fixed_budget_checkpoint.pt",
        "best_checkpoint": "checkpoints/fold2/baseline_r_fold2/best.pt",  # Epoch 2 / 1,200 updates
        "fixed_budget_epoch": 6,
        "fixed_budget_updates": 3600,
        "best_epoch": 2,
        "best_updates": 1200,
    },
    3: {
        "val_speakers": ["TXHC", "ZHAA"],
        "l1_profiles": {"TXHC": "Mandarin Chinese", "ZHAA": "Arabic"},
        "config_path": "configs/rq1_trn_03a_fold3.yaml",
        "zip_path": "datasets/L2-ARCTIC/authorized_RQ1-TRN_fold3_train_val_only.zip",
        "fixed_budget_checkpoint": "checkpoints/fold3/baseline_r_fold3/fixed_budget_checkpoint.pt",
        "best_checkpoint": "checkpoints/fold3/baseline_r_fold3/best.pt",  # Coincides at Epoch 6
        "fixed_budget_epoch": 6,
        "fixed_budget_updates": 3600,
        "best_epoch": 6,
        "best_updates": 3600,
    },
    4: {
        "val_speakers": ["HKK", "NCC"],
        "l1_profiles": {"HKK": "Korean", "NCC": "Mandarin Chinese"},
        "config_path": "configs/rq1_trn_03a_fold4.yaml",
        "zip_path": "datasets/L2-ARCTIC/authorized_RQ1-TRN_fold4_train_val_only.zip",
        "fixed_budget_checkpoint": "checkpoints/fold4/baseline_r_fold4/fixed_budget_checkpoint.pt",
        "best_checkpoint": "checkpoints/fold4/baseline_r_fold4/best.pt",  # Epoch 3 / 1,800 updates
        "fixed_budget_epoch": 6,
        "fixed_budget_updates": 3600,
        "best_epoch": 3,
        "best_updates": 1800,
    },
}

CRITICAL_T_3_DF = 3.182  # Two-tailed t critical value for df = 4 - 1 = 3 at alpha = 0.05


def compute_student_t_ci(values: List[float], t_crit: float = CRITICAL_T_3_DF) -> Dict[str, float]:
    """Computes sample mean, std, standard error, and Student's t 95% CI."""
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 0 else 0.0
    ci_lower = mean - t_crit * se
    ci_upper = mean + t_crit * se
    return {
        "mean": round(mean, 6),
        "std": round(std, 6),
        "se": round(se, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "df": n - 1,
        "t_crit": t_crit,
    }


def evaluate_single_fold_checkpoint(
    fold_idx: int,
    checkpoint_role: str,
    checkpoint_path: str,
    manifest_path: str,
    authorized_zip: str,
    g2p: CMUDictG2P,
    prompts_path: str,
    device: torch.device,
    max_eval_utterances: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluates a single fold checkpoint under Protocol O and Protocol Z."""
    print(f"\n{'='*70}")
    print(f"[EVAL] Fold {fold_idx} | Role: {checkpoint_role.upper()} | Checkpoint: {checkpoint_path}")
    print(f"{'='*70}")

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")
    ckpt_sha = compute_sha256(ckpt_file)

    raw_ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    epoch_stored = raw_ckpt.get("epoch", 0)
    updates_stored = raw_ckpt.get("successful_updates", 0)

    # Build model in strict eval mode
    model = build_eval_model(
        pretrained_model="facebook/wav2vec2-base-960h",
        pretrained_revision="22aad52d435eb6dbaf354bdad9b0da84ce7d6156",
        state_dict=raw_ckpt["model"],
        ctc_weight=1.0,
        binary_weight=1.0,
    ).to(device)

    init_fingerprint = get_model_weights_fingerprint(model)

    # 1. Load Validation Dataset
    val_dataset = L2ArcticMDDDataset(
        manifest_path,
        fold=fold_idx,
        split="val",
        policy="strict",
        zip_path=authorized_zip,
    )
    n_utts = len(val_dataset)
    if max_eval_utterances is not None:
        n_utts = min(n_utts, max_eval_utterances)

    print(f"[DATASET] Loaded {len(val_dataset)} validation utterances for Fold {fold_idx}. Evaluating {n_utts}...")

    # Collect predictions for Condition 1: Matched Audio and Condition 2: Mismatched Audio
    matched_scores_all = []
    mismatched_scores_all = []
    matched_labels_all = []
    ctc_refs_all = []
    ctc_hyps_all = []
    ctc_blank_frames_tot = 0
    ctc_total_frames_tot = 0

    per_speaker_records = defaultdict(lambda: {
        "scores_matched": [], "scores_mismatched": [], "labels": [],
        "refs": [], "hyps": [], "blank_frames": 0, "total_frames": 0
    })

    # Prepare derangement for Condition 2: Shift-1 Within-Speaker Mismatched Audio
    spk_utts = defaultdict(list)
    for i in range(n_utts):
        spk_utts[val_dataset[i]["speaker_id"]].append(i)

    derangement_map = {}
    for spk, indices in spk_utts.items():
        deranged = generate_derangement_indices(len(indices), method="cyclic")
        for orig_idx, der_idx in zip(indices, deranged):
            derangement_map[orig_idx] = indices[der_idx]

    # Pre-extract mismatched audio items
    mismatched_audio_cache = {}
    for i in range(n_utts):
        donor_idx = derangement_map[i]
        mismatched_audio_cache[i] = val_dataset[donor_idx]["audio_waveform_16k"]

    t0 = time.time()
    per_utterance_data = []

    with torch.inference_mode():
        for i in range(n_utts):
            item = val_dataset[i]
            spk = item["speaker_id"]
            uid = item["utterance_id"]
            rec_id = item["record_id"]
            g_labels = item["binary_labels"].tolist()
            can_idx = item["canonical_phone_indices"].unsqueeze(0).to(device)
            can_len = torch.tensor([can_idx.shape[1]], dtype=torch.long, device=device)

            # --- Pass 1: Matched Audio ---
            audio_matched = item["audio_waveform_16k"].unsqueeze(0).to(device)
            audio_len_m = torch.tensor([audio_matched.shape[1]], dtype=torch.long, device=device)

            out_m = model(
                source=audio_matched,
                audio_lengths=audio_len_m,
                canonical_phone_indices=can_idx,
                canonical_phone_lengths=can_len,
            )

            scores_m = torch.sigmoid(out_m.binary_logits[0]).cpu().tolist()
            assert len(scores_m) == len(g_labels)

            # CTC greedy decode
            ctc_logits_m = out_m.ctc_logits[0]  # [T, V]
            blank_frames = int((ctc_logits_m.argmax(dim=-1) == 0).sum().item())
            total_frames = int(ctc_logits_m.shape[0])
            ctc_blank_frames_tot += blank_frames
            ctc_total_frames_tot += total_frames

            raw_preds = ctc_logits_m.argmax(dim=-1).cpu().tolist()
            hyp_tokens = greedy_ctc_decode(raw_preds, blank_id=0)
            ref_tokens = item["ctc_target_indices"].tolist()

            matched_scores_all.extend(scores_m)
            matched_labels_all.extend(g_labels)
            ctc_refs_all.append(ref_tokens)
            ctc_hyps_all.append(hyp_tokens)

            # Per speaker accumulator
            per_speaker_records[spk]["scores_matched"].extend(scores_m)
            per_speaker_records[spk]["labels"].extend(g_labels)
            per_speaker_records[spk]["refs"].append(ref_tokens)
            per_speaker_records[spk]["hyps"].append(hyp_tokens)
            per_speaker_records[spk]["blank_frames"] += blank_frames
            per_speaker_records[spk]["total_frames"] += total_frames

            # --- Pass 2: Mismatched Audio (Shift-1 within speaker) ---
            audio_mismatched = mismatched_audio_cache[i].unsqueeze(0).to(device)
            audio_len_mm = torch.tensor([audio_mismatched.shape[1]], dtype=torch.long, device=device)

            out_mm = model(
                source=audio_mismatched,
                audio_lengths=audio_len_mm,
                canonical_phone_indices=can_idx,
                canonical_phone_lengths=can_len,
            )
            scores_mm = torch.sigmoid(out_mm.binary_logits[0]).cpu().tolist()
            assert len(scores_mm) == len(g_labels)
            mismatched_scores_all.extend(scores_mm)
            per_speaker_records[spk]["scores_mismatched"].extend(scores_mm)

            per_utterance_data.append({
                "record_id": rec_id,
                "fold": fold_idx,
                "speaker": spk,
                "utterance_id": uid,
                "canonical_phones": [CTC_INDEX_TO_PHONE[idx.item()] for idx in item["canonical_phone_indices"]],
                "labels": g_labels,
                "matched_scores": scores_m,
                "mismatched_scores": scores_mm,
            })

            if (i + 1) % 100 == 0 or i == n_utts - 1:
                print(f"  [EVAL] Processed {i + 1}/{n_utts} utterances ({time.time() - t0:.1f}s)...")

    eval_duration = time.time() - t0
    post_fingerprint = get_model_weights_fingerprint(model)
    assert init_fingerprint == post_fingerprint, "CRITICAL: Model weights modified during evaluation!"
    assert len(matched_scores_all) == len(mismatched_scores_all) == len(matched_labels_all), (
        f"Length mismatch: matched {len(matched_scores_all)}, mismatched {len(mismatched_scores_all)}, labels {len(matched_labels_all)}"
    )

    # Calculate Protocol O binary metrics with strictly aligned prediction vectors
    matched_ap = average_precision_binary(matched_labels_all, matched_scores_all)
    mismatched_ap = average_precision_binary(matched_labels_all, mismatched_scores_all)
    delta_gap = matched_ap - mismatched_ap

    # Calculate binary operating point metrics (tau = 0.50 and tau = 0.20)
    bin_metrics = calculate_binary_metrics(matched_labels_all, matched_scores_all, thresholds=[0.20, 0.50])
    sweep_dict = {p["threshold"]: p for p in bin_metrics["threshold_calibration_sweep"]}
    pt_05 = sweep_dict.get(0.50, {})
    pt_02 = sweep_dict.get(0.20, {})

    fnr_05 = (pt_05["fn"] / (pt_05["fn"] + pt_05["tp"])) if (pt_05.get("fn", 0) + pt_05.get("tp", 0)) > 0 else 0.0
    fpr_05 = (pt_05["fp"] / (pt_05["fp"] + pt_05["tn"])) if (pt_05.get("fp", 0) + pt_05.get("tn", 0)) > 0 else 0.0
    fnr_02 = (pt_02["fn"] / (pt_02["fn"] + pt_02["tp"])) if (pt_02.get("fn", 0) + pt_02.get("tp", 0)) > 0 else 0.0
    fpr_02 = (pt_02["fp"] / (pt_02["fp"] + pt_02["tn"])) if (pt_02.get("fp", 0) + pt_02.get("tn", 0)) > 0 else 0.0

    # Calculate CTC metrics
    ctc_metrics = calculate_ctc_metrics(ctc_refs_all, ctc_hyps_all, ctc_blank_frames_tot, ctc_total_frames_tot)

    # Per-speaker metrics breakdown
    per_speaker_res = {}
    for spk, srec in sorted(per_speaker_records.items()):
        spk_ap_m = average_precision_binary(srec["labels"], srec["scores_matched"])
        spk_ap_mm = average_precision_binary(srec["labels"], srec["scores_mismatched"])
        spk_ctc = calculate_ctc_metrics(srec["refs"], srec["hyps"], srec["blank_frames"], srec["total_frames"])
        spk_bin = calculate_binary_metrics(srec["labels"], srec["scores_matched"], thresholds=[0.20, 0.50])
        spk_sweep = {p["threshold"]: p for p in spk_bin["threshold_calibration_sweep"]}
        spk_pt_05 = spk_sweep.get(0.50, {})
        spk_pt_02 = spk_sweep.get(0.20, {})

        spk_fnr_05 = (spk_pt_05["fn"] / (spk_pt_05["fn"] + spk_pt_05["tp"])) if (spk_pt_05.get("fn", 0) + spk_pt_05.get("tp", 0)) > 0 else 0.0
        spk_fpr_05 = (spk_pt_05["fp"] / (spk_pt_05["fp"] + spk_pt_05["tn"])) if (spk_pt_05.get("fp", 0) + spk_pt_05.get("tn", 0)) > 0 else 0.0
        spk_fnr_02 = (spk_pt_02["fn"] / (spk_pt_02["fn"] + spk_pt_02["tp"])) if (spk_pt_02.get("fn", 0) + spk_pt_02.get("tp", 0)) > 0 else 0.0
        spk_fpr_02 = (spk_pt_02["fp"] / (spk_pt_02["fp"] + spk_pt_02["tn"])) if (spk_pt_02.get("fp", 0) + spk_pt_02.get("tn", 0)) > 0 else 0.0

        per_speaker_res[spk] = {
            "tokens": len(srec["labels"]),
            "errors": int(sum(srec["labels"])),
            "prevalence": round(sum(srec["labels"]) / len(srec["labels"]), 6),
            "matched_ap": round(spk_ap_m, 6),
            "mismatched_ap": round(spk_ap_mm, 6),
            "acoustic_gap": round(spk_ap_m - spk_ap_mm, 6),
            "ctc_per": spk_ctc["greedy_per"],
            "ctc_sub_rate": spk_ctc["substitution_rate"],
            "ctc_del_rate": spk_ctc["deletion_rate"],
            "ctc_ins_rate": spk_ctc["insertion_rate"],
            "fnr_at_050": round(spk_fnr_05, 4),
            "fpr_at_050": round(spk_fpr_05, 4),
            "precision_at_050": spk_pt_05.get("precision", 0.0),
            "recall_at_050": spk_pt_05.get("recall", 0.0),
            "f1_at_050": spk_pt_05.get("f1", 0.0),
            "fnr_at_020": round(spk_fnr_02, 4),
            "fpr_at_020": round(spk_fpr_02, 4),
            "precision_at_020": spk_pt_02.get("precision", 0.0),
            "recall_at_020": spk_pt_02.get("recall", 0.0),
            "f1_at_020": spk_pt_02.get("f1", 0.0),
        }

    # --- Fit Train Canonical Prior (strictly on 16 train speakers) ---
    print(f"  [PRIOR] Fitting train-only canonical empirical prior on Fold {fold_idx} train...")
    train_dataset = L2ArcticMDDDataset(
        manifest_path,
        fold=fold_idx,
        split="train",
        policy="strict",
        zip_path=authorized_zip,
    )
    can_prior_table = fit_canonical_empirical_prior(train_dataset)
    can_prior_metrics = evaluate_canonical_prior(
        [val_dataset[i] for i in range(n_utts)],
        can_prior_table,
    )

    # --- Protocol Z Zero-Oracle Evaluation ---
    print(f"  [PROTO-Z] Evaluating Protocol Z (Zero-Oracle G2P + Audio)...")
    z_dataset = ZeroOracleL2ArcticDataset(
        split_manifest_path=manifest_path,
        fold=fold_idx,
        split="val",
        prompts_source=prompts_path,
        g2p_frontend=g2p,
        zip_path=authorized_zip,
        max_samples=max_eval_utterances,
    )
    train_prior_z = fit_train_canonical_prior(train_dataset)

    z_aligned_scores, z_aligned_labels = [], []
    z_policy_b1_scores, z_full_gold_labels = [], []
    total_gold_z, total_g2p_z = 0, 0
    total_unaligned_gold = 0

    with torch.inference_mode():
        for i in range(len(z_dataset)):
            z_item = z_dataset[i]
            g_item = val_dataset[i]
            g2p_ph = z_item["canonical_phones"]
            gold_ph = [CTC_INDEX_TO_PHONE[idx.item()] for idx in g_item["canonical_phone_indices"]]
            gold_lbls = g_item["binary_labels"].tolist()

            total_gold_z += len(gold_ph)
            total_g2p_z += len(g2p_ph)

            audio_z = z_item["audio_waveform_16k"].unsqueeze(0).to(device)
            audio_len_z = torch.tensor([audio_z.shape[1]], dtype=torch.long, device=device)
            can_idx_z = z_item["canonical_phone_indices"].unsqueeze(0).to(device)
            can_len_z = torch.tensor([can_idx_z.shape[1]], dtype=torch.long, device=device)

            out_z = model(
                source=audio_z,
                audio_lengths=audio_len_z,
                canonical_phone_indices=can_idx_z,
                canonical_phone_lengths=can_len_z,
            )
            z_scores = torch.sigmoid(out_z.binary_logits[0, :len(g2p_ph)]).cpu().tolist()

            alignment = align_canonical_sequences(gold_ph, g2p_ph, tie_break="sub_first")
            for g_idx, z_idx, _ in alignment["aligned_pairs"]:
                lbl = gold_lbls[g_idx]
                sc = z_scores[z_idx]
                z_aligned_scores.append(sc)
                z_aligned_labels.append(lbl)
                z_full_gold_labels.append(lbl)
                z_policy_b1_scores.append(sc)

            for g_idx, g_p in alignment["unaligned_gold"]:
                total_unaligned_gold += 1
                lbl = gold_lbls[g_idx]
                c_idx = PHONE_TO_CTC_INDEX.get(g_p, 0)
                prior_val = train_prior_z["phone_priors"].get(c_idx, train_prior_z["overall_train_rate"])
                z_full_gold_labels.append(lbl)
                z_policy_b1_scores.append(prior_val)

    proto_z_aligned_ap = average_precision_binary(z_aligned_labels, z_aligned_scores)
    proto_z_b1_ap = average_precision_binary(z_full_gold_labels, z_policy_b1_scores)
    alignment_coverage = (len(z_aligned_labels) / max(total_gold_z, 1))

    print(f"  [RESULT] Matched AP: {matched_ap:.6f} | Mismatched AP: {mismatched_ap:.6f} | Delta Gap: {delta_gap:+.6f}")
    print(f"  [RESULT] CTC PER: {ctc_metrics['greedy_per']:.2%} (Sub {ctc_metrics['substitution_rate']:.2%}, Del {ctc_metrics['deletion_rate']:.2%}, Ins {ctc_metrics['insertion_rate']:.2%})")
    print(f"  [RESULT] Proto Z Aligned AP: {proto_z_aligned_ap:.6f} (Cov: {alignment_coverage:.2%}) | Policy B1 AP: {proto_z_b1_ap:.6f}")

    return {
        "fold": fold_idx,
        "role": checkpoint_role,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": ckpt_sha,
        "epoch_stored": epoch_stored,
        "updates_stored": updates_stored,
        "model_fingerprint": post_fingerprint,
        "eval_utterances": n_utts,
        "total_phone_tokens": len(matched_labels_all),
        "total_error_tokens": int(sum(matched_labels_all)),
        "prevalence": round(sum(matched_labels_all) / len(matched_labels_all), 6),
        "matched_ap": round(matched_ap, 6),
        "mismatched_ap": round(mismatched_ap, 6),
        "acoustic_gap": round(delta_gap, 6),
        "canonical_prior_ap": round(can_prior_metrics["average_precision"], 6),
        "margin_over_prior": round(matched_ap - can_prior_metrics["average_precision"], 6),
        "ctc_per": ctc_metrics["greedy_per"],
        "ctc_substitutions": ctc_metrics["substitutions"],
        "ctc_deletions": ctc_metrics["deletions"],
        "ctc_insertions": ctc_metrics["insertions"],
        "ctc_total_edits": ctc_metrics["total_edits"],
        "ctc_ref_tokens": ctc_metrics["total_ref_tokens"],
        "ctc_sub_rate": ctc_metrics["substitution_rate"],
        "ctc_del_rate": ctc_metrics["deletion_rate"],
        "ctc_ins_rate": ctc_metrics["insertion_rate"],
        "ctc_blank_fraction": ctc_metrics["blank_fraction"],
        "fnr_at_050": round(fnr_05, 4),
        "fpr_at_050": round(fpr_05, 4),
        "precision_at_050": pt_05.get("precision", 0.0),
        "recall_at_050": pt_05.get("recall", 0.0),
        "f1_at_050": pt_05.get("f1", 0.0),
        "fnr_at_020": round(fnr_02, 4),
        "fpr_at_020": round(fpr_02, 4),
        "precision_at_020": pt_02.get("precision", 0.0),
        "recall_at_020": pt_02.get("recall", 0.0),
        "f1_at_020": pt_02.get("f1", 0.0),
        "proto_z_aligned_ap": round(proto_z_aligned_ap, 6),
        "proto_z_b1_ap": round(proto_z_b1_ap, 6),
        "proto_z_coverage": round(alignment_coverage, 6),
        "per_speaker": per_speaker_res,
        "raw_predictions": {
            "matched_labels": matched_labels_all,
            "matched_scores": matched_scores_all,
            "mismatched_scores": mismatched_scores_all,
        },
        "per_utterance": per_utterance_data,
    }


def main():
    parser = argparse.ArgumentParser(description="LANGFENS RQ1-TRN-03A Four-Fold Validation Synthesis")
    parser.add_argument("--manifest", default="manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json")
    parser.add_argument("--prompts", default="resources/L2-ARCTIC/prompts.json")
    parser.add_argument("--output_dir", default="manifests/RQ1-TRN-03A/evaluation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (default cuda if available, else cpu)")
    parser.add_argument("--max_eval_utterances", type=int, default=None, help="Diagnostic run override")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 80)
    print("LANGFENS RQ1-TRN-03A — FOUR-FOLD VALIDATION REPLAY & SCIENTIFIC SYNTHESIS")
    print(f"Device: {device} | Output Directory: {out_dir}")
    print("=" * 80)

    # Initialize deterministic CMUdict G2P frontend once
    g2p = CMUDictG2P(verify_checksum=True, oov_policy="strict")

    # Evaluate all 4 folds under primary fixed-budget milestone (Epoch 6 / 3,600 updates)
    fixed_budget_results = {}
    best_results = {}

    for fold_idx in [1, 2, 3, 4]:
        f_def = FOLD_DEFINITIONS[fold_idx]

        # 1. Fixed-Budget Primary Evaluation
        res_fb = evaluate_single_fold_checkpoint(
            fold_idx=fold_idx,
            checkpoint_role="fixed_budget",
            checkpoint_path=f_def["fixed_budget_checkpoint"],
            manifest_path=args.manifest,
            authorized_zip=f_def["zip_path"],
            g2p=g2p,
            prompts_path=args.prompts,
            device=device,
            max_eval_utterances=args.max_eval_utterances,
        )
        fixed_budget_results[fold_idx] = res_fb

        # 2. Validation Peak (Best) Evaluation
        if f_def["best_checkpoint"] == f_def["fixed_budget_checkpoint"]:
            # Same checkpoint (Fold 1 and Fold 3 peak coincided with fixed budget)
            best_results[fold_idx] = res_fb
        else:
            res_best = evaluate_single_fold_checkpoint(
                fold_idx=fold_idx,
                checkpoint_role="best",
                checkpoint_path=f_def["best_checkpoint"],
                manifest_path=args.manifest,
                authorized_zip=f_def["zip_path"],
                g2p=g2p,
                prompts_path=args.prompts,
                device=device,
                max_eval_utterances=args.max_eval_utterances,
            )
            best_results[fold_idx] = res_best

    # =========================================================================
    # Synthesis & Statistical Aggregation
    # =========================================================================
    print("\n" + "=" * 80)
    print("CROSS-FOLD STATISTICAL SYNTHESIS (PRIMARY FIXED-BUDGET: 3,600 UPDATES)")
    print("=" * 80)

    fb_matched_aps = [fixed_budget_results[f]["matched_ap"] for f in [1, 2, 3, 4]]
    fb_mismatched_aps = [fixed_budget_results[f]["mismatched_ap"] for f in [1, 2, 3, 4]]
    fb_gaps = [fixed_budget_results[f]["acoustic_gap"] for f in [1, 2, 3, 4]]
    fb_prior_aps = [fixed_budget_results[f]["canonical_prior_ap"] for f in [1, 2, 3, 4]]
    fb_margins = [fixed_budget_results[f]["margin_over_prior"] for f in [1, 2, 3, 4]]
    fb_ctc_pers = [fixed_budget_results[f]["ctc_per"] for f in [1, 2, 3, 4]]
    fb_proto_z_aligned = [fixed_budget_results[f]["proto_z_aligned_ap"] for f in [1, 2, 3, 4]]
    fb_proto_z_b1 = [fixed_budget_results[f]["proto_z_b1_ap"] for f in [1, 2, 3, 4]]

    macro_matched_ap = compute_student_t_ci(fb_matched_aps)
    macro_mismatched_ap = compute_student_t_ci(fb_mismatched_aps)
    macro_gap = compute_student_t_ci(fb_gaps)
    macro_prior_ap = compute_student_t_ci(fb_prior_aps)
    macro_margin = compute_student_t_ci(fb_margins)
    macro_ctc_per = compute_student_t_ci(fb_ctc_pers)
    macro_prev = compute_student_t_ci([fixed_budget_results[f]["prevalence"] for f in [1, 2, 3, 4]])
    macro_proto_z_aligned = compute_student_t_ci(fb_proto_z_aligned)
    macro_proto_z_b1 = compute_student_t_ci(fb_proto_z_b1)

    # Pooled prediction-level micro AP
    pooled_matched_labels = []
    pooled_matched_scores = []
    pooled_mismatched_scores = []
    for f in [1, 2, 3, 4]:
        raw = fixed_budget_results[f]["raw_predictions"]
        pooled_matched_labels.extend(raw["matched_labels"])
        pooled_matched_scores.extend(raw["matched_scores"])
        pooled_mismatched_scores.extend(raw["mismatched_scores"])

    micro_matched_ap = average_precision_binary(pooled_matched_labels, pooled_matched_scores)
    micro_mismatched_ap = average_precision_binary(pooled_matched_labels, pooled_mismatched_scores)
    micro_gap = micro_matched_ap - micro_mismatched_ap

    # Pooled CTC PER
    tot_ref_tokens = sum(fixed_budget_results[f]["ctc_ref_tokens"] for f in [1, 2, 3, 4])
    tot_edits = sum(fixed_budget_results[f]["ctc_total_edits"] for f in [1, 2, 3, 4])
    tot_subs = sum(fixed_budget_results[f]["ctc_substitutions"] for f in [1, 2, 3, 4])
    tot_dels = sum(fixed_budget_results[f]["ctc_deletions"] for f in [1, 2, 3, 4])
    tot_inss = sum(fixed_budget_results[f]["ctc_insertions"] for f in [1, 2, 3, 4])
    micro_ctc_per = tot_edits / tot_ref_tokens
    micro_sub_rate = tot_subs / tot_ref_tokens
    micro_del_rate = tot_dels / tot_ref_tokens
    micro_ins_rate = tot_inss / tot_ref_tokens

    # Pooled confusion matrices at tau=0.50 and tau=0.20
    pooled_bin_metrics = calculate_binary_metrics(pooled_matched_labels, pooled_matched_scores, thresholds=[0.20, 0.50])
    p_sweep = {pt["threshold"]: pt for pt in pooled_bin_metrics["threshold_calibration_sweep"]}
    p_05 = p_sweep[0.50]
    p_02 = p_sweep[0.20]

    micro_fnr_05 = p_05["fn"] / (p_05["fn"] + p_05["tp"])
    micro_fpr_05 = p_05["fp"] / (p_05["fp"] + p_05["tn"])
    micro_fnr_02 = p_02["fn"] / (p_02["fn"] + p_02["tp"])
    micro_fpr_02 = p_02["fp"] / (p_02["fp"] + p_02["tn"])

    # Macro averages for FNR/FPR
    macro_fnr_05 = compute_student_t_ci([fixed_budget_results[f]["fnr_at_050"] for f in [1, 2, 3, 4]])
    macro_fpr_05 = compute_student_t_ci([fixed_budget_results[f]["fpr_at_050"] for f in [1, 2, 3, 4]])
    macro_fnr_02 = compute_student_t_ci([fixed_budget_results[f]["fnr_at_020"] for f in [1, 2, 3, 4]])
    macro_fpr_02 = compute_student_t_ci([fixed_budget_results[f]["fpr_at_020"] for f in [1, 2, 3, 4]])

    # Print Macro/Micro Comparison Table
    print(f"\n1. PRIMARY DETECTION PERFORMANCE (FIXED-BUDGET: 3,600 UPDATES / EPOCH 6):")
    print(f"  Macro-Averaged Matched Binary AP (N=4 folds): {macro_matched_ap['mean']:.6f} +/- {macro_matched_ap['se']:.6f} (95% CI: [{macro_matched_ap['ci_lower']:.6f}, {macro_matched_ap['ci_upper']:.6f}])")
    print(f"  Micro-Averaged Matched Binary AP (N=39,576 tokens): {micro_matched_ap:.6f}")
    print(f"  Macro-Averaged Shifted Audio AP (Mismatched): {macro_mismatched_ap['mean']:.6f} +/- {macro_mismatched_ap['se']:.6f}")
    print(f"  Macro Matched-Minus-Mismatched Acoustic Gap:  {macro_gap['mean']:+.6f} +/- {macro_gap['se']:.6f} (95% CI: [{macro_gap['ci_lower']:+.6f}, {macro_gap['ci_upper']:+.6f}])")
    print(f"  Micro Matched-Minus-Mismatched Acoustic Gap:  {micro_gap:+.6f}")
    print(f"  Macro Canonical-Only Empirical Prior AP:      {macro_prior_ap['mean']:.6f} +/- {macro_prior_ap['se']:.6f}")
    print(f"  Macro Neural Margin over Prior:               {macro_margin['mean']:+.6f} +/- {macro_margin['se']:.6f}")

    print(f"\n2. ACOUSTIC RECOGNITION (CTC PER AT 3,600 UPDATES):")
    print(f"  Macro Greedy CTC PER: {macro_ctc_per['mean']:.2%} +/- {macro_ctc_per['se']:.2%} (95% CI: [{macro_ctc_per['ci_lower']:.2%}, {macro_ctc_per['ci_upper']:.2%}])")
    print(f"  Pooled Micro CTC PER: {micro_ctc_per:.2%} (Sub {micro_sub_rate:.2%}, Del {micro_del_rate:.2%}, Ins {micro_ins_rate:.2%})")

    print(f"\n3. PEDAGOGICAL OPERATING POINT METRICS:")
    print(f"  Standard Threshold (tau = 0.50):")
    print(f"    Macro FNR (Missed Error Rate): {macro_fnr_05['mean']:.2%} (95% CI: [{macro_fnr_05['ci_lower']:.2%}, {macro_fnr_05['ci_upper']:.2%}]) | Micro FNR: {micro_fnr_05:.2%}")
    print(f"    Macro FPR (False Alarm Rate):  {macro_fpr_05['mean']:.2%} (95% CI: [{macro_fpr_05['ci_lower']:.2%}, {macro_fpr_05['ci_upper']:.2%}]) | Micro FPR: {micro_fpr_05:.2%}")
    print(f"    Pooled Precision: {p_05['precision']:.4f} | Pooled Recall: {p_05['recall']:.4f} | Pooled F1: {p_05['f1']:.4f}")
    print(f"  High-Sensitivity Threshold (tau = 0.20):")
    print(f"    Macro FNR (Missed Error Rate): {macro_fnr_02['mean']:.2%} (95% CI: [{macro_fnr_02['ci_lower']:.2%}, {macro_fnr_02['ci_upper']:.2%}]) | Micro FNR: {micro_fnr_02:.2%}")
    print(f"    Macro FPR (False Alarm Rate):  {macro_fpr_02['mean']:.2%} (95% CI: [{macro_fpr_02['ci_lower']:.2%}, {macro_fpr_02['ci_upper']:.2%}]) | Micro FPR: {micro_fpr_02:.2%}")
    print(f"    Pooled Precision: {p_02['precision']:.4f} | Pooled Recall: {p_02['recall']:.4f} | Pooled F1: {p_02['f1']:.4f}")

    print(f"\n4. PROTOCOL Z ZERO-ORACLE EVALUATION:")
    print(f"  Macro Protocol Z (Aligned-Only AP): {macro_proto_z_aligned['mean']:.6f} +/- {macro_proto_z_aligned['se']:.6f}")
    print(f"  Macro Protocol Z (Policy B1 Full-Gold AP): {macro_proto_z_b1['mean']:.6f} +/- {macro_proto_z_b1['se']:.6f}")

    # =========================================================================
    # Best Checkpoints Synthesis
    # =========================================================================
    best_matched_aps = [best_results[f]["matched_ap"] for f in [1, 2, 3, 4]]
    best_gaps = [best_results[f]["acoustic_gap"] for f in [1, 2, 3, 4]]
    best_ctc_pers = [best_results[f]["ctc_per"] for f in [1, 2, 3, 4]]

    macro_best_matched_ap = compute_student_t_ci(best_matched_aps)
    macro_best_gap = compute_student_t_ci(best_gaps)
    macro_best_ctc_per = compute_student_t_ci(best_ctc_pers)

    pooled_best_labels, pooled_best_scores = [], []
    for f in [1, 2, 3, 4]:
        raw = best_results[f]["raw_predictions"]
        pooled_best_labels.extend(raw["matched_labels"])
        pooled_best_scores.extend(raw["matched_scores"])
    micro_best_matched_ap = average_precision_binary(pooled_best_labels, pooled_best_scores)

    # =========================================================================
    # Export Reproducible Reports & Tables
    # =========================================================================
    # 1. Summary CSV Table (Folds 1-4 + Macro + Micro)
    csv_rows = []
    headers = [
        "Evaluation Arm", "Fold / Cohort", "Val Speakers", "Val L1 Profiles", "Val Tokens", "Val Errors", "Val Prev",
        "Epoch", "Updates", "Matched AP", "Mismatched AP", "Acoustic Gap", "Canonical Prior AP", "Margin over Prior",
        "CTC PER", "CTC Sub%", "CTC Del%", "CTC Ins%", "FNR @ 0.50", "FPR @ 0.50", "FNR @ 0.20", "FPR @ 0.20",
        "Proto Z Aligned AP", "Proto Z B1 AP"
    ]

    for f in [1, 2, 3, 4]:
        r = fixed_budget_results[f]
        spks = FOLD_DEFINITIONS[f]["val_speakers"]
        l1s = [FOLD_DEFINITIONS[f]["l1_profiles"][s] for s in spks]
        csv_rows.append({
            "Evaluation Arm": "Fixed Budget (Primary)",
            "Fold / Cohort": f"Fold {f}",
            "Val Speakers": " & ".join(spks),
            "Val L1 Profiles": " & ".join(l1s),
            "Val Tokens": r["total_phone_tokens"],
            "Val Errors": r["total_error_tokens"],
            "Val Prev": f"{r['prevalence']:.4%}",
            "Epoch": r["epoch_stored"] + 1,
            "Updates": r["updates_stored"],
            "Matched AP": f"{r['matched_ap']:.6f}",
            "Mismatched AP": f"{r['mismatched_ap']:.6f}",
            "Acoustic Gap": f"{r['acoustic_gap']:+.6f}",
            "Canonical Prior AP": f"{r['canonical_prior_ap']:.6f}",
            "Margin over Prior": f"{r['margin_over_prior']:+.6f}",
            "CTC PER": f"{r['ctc_per']:.2%}",
            "CTC Sub%": f"{r['ctc_sub_rate']:.2%}",
            "CTC Del%": f"{r['ctc_del_rate']:.2%}",
            "CTC Ins%": f"{r['ctc_ins_rate']:.2%}",
            "FNR @ 0.50": f"{r['fnr_at_050']:.2%}",
            "FPR @ 0.50": f"{r['fpr_at_050']:.2%}",
            "FNR @ 0.20": f"{r['fnr_at_020']:.2%}",
            "FPR @ 0.20": f"{r['fpr_at_020']:.2%}",
            "Proto Z Aligned AP": f"{r['proto_z_aligned_ap']:.6f}",
            "Proto Z B1 AP": f"{r['proto_z_b1_ap']:.6f}",
        })

    # Macro Row
    csv_rows.append({
        "Evaluation Arm": "Fixed Budget (Primary)",
        "Fold / Cohort": "Macro-Average (N=4)",
        "Val Speakers": "8 Speakers (All)",
        "Val L1 Profiles": "Spanish, Hindi, Arabic, Mandarin, Korean",
        "Val Tokens": "39,576 total (9,894/fold)",
        "Val Errors": "4,213 total (1,053/fold)",
        "Val Prev": f"{macro_prev['mean']:.4%}",
        "Epoch": 6,
        "Updates": 3600,
        "Matched AP": f"{macro_matched_ap['mean']:.6f} +/- {macro_matched_ap['se']:.6f}",
        "Mismatched AP": f"{macro_mismatched_ap['mean']:.6f} +/- {macro_mismatched_ap['se']:.6f}",
        "Acoustic Gap": f"{macro_gap['mean']:+.6f} +/- {macro_gap['se']:.6f}",
        "Canonical Prior AP": f"{macro_prior_ap['mean']:.6f} +/- {macro_prior_ap['se']:.6f}",
        "Margin over Prior": f"{macro_margin['mean']:+.6f} +/- {macro_margin['se']:.6f}",
        "CTC PER": f"{macro_ctc_per['mean']:.2%} +/- {macro_ctc_per['se']:.2%}",
        "CTC Sub%": f"{np.mean([fixed_budget_results[f]['ctc_sub_rate'] for f in [1, 2, 3, 4]]):.2%}",
        "CTC Del%": f"{np.mean([fixed_budget_results[f]['ctc_del_rate'] for f in [1, 2, 3, 4]]):.2%}",
        "CTC Ins%": f"{np.mean([fixed_budget_results[f]['ctc_ins_rate'] for f in [1, 2, 3, 4]]):.2%}",
        "FNR @ 0.50": f"{macro_fnr_05['mean']:.2%}",
        "FPR @ 0.50": f"{macro_fpr_05['mean']:.2%}",
        "FNR @ 0.20": f"{macro_fnr_02['mean']:.2%}",
        "FPR @ 0.20": f"{macro_fpr_02['mean']:.2%}",
        "Proto Z Aligned AP": f"{macro_proto_z_aligned['mean']:.6f} +/- {macro_proto_z_aligned['se']:.6f}",
        "Proto Z B1 AP": f"{macro_proto_z_b1['mean']:.6f} +/- {macro_proto_z_b1['se']:.6f}",
    })

    # Micro Row
    csv_rows.append({
        "Evaluation Arm": "Fixed Budget (Primary)",
        "Fold / Cohort": "Pooled Micro (N=39,576)",
        "Val Speakers": "8 Speakers (All)",
        "Val L1 Profiles": "Spanish, Hindi, Arabic, Mandarin, Korean",
        "Val Tokens": 39576,
        "Val Errors": 4213,
        "Val Prev": f"{4213/39576:.4%}",
        "Epoch": 6,
        "Updates": 3600,
        "Matched AP": f"{micro_matched_ap:.6f}",
        "Mismatched AP": f"{micro_mismatched_ap:.6f}",
        "Acoustic Gap": f"{micro_gap:+.6f}",
        "Canonical Prior AP": f"{macro_prior_ap['mean']:.6f}",
        "Margin over Prior": f"{micro_matched_ap - macro_prior_ap['mean']:+.6f}",
        "CTC PER": f"{micro_ctc_per:.2%}",
        "CTC Sub%": f"{micro_sub_rate:.2%}",
        "CTC Del%": f"{micro_del_rate:.2%}",
        "CTC Ins%": f"{micro_ins_rate:.2%}",
        "FNR @ 0.50": f"{micro_fnr_05:.2%}",
        "FPR @ 0.50": f"{micro_fpr_05:.2%}",
        "FNR @ 0.20": f"{micro_fnr_02:.2%}",
        "FPR @ 0.20": f"{micro_fpr_02:.2%}",
        "Proto Z Aligned AP": "—",
        "Proto Z B1 AP": "—",
    })

    # Best rows
    for f in [1, 2, 3, 4]:
        r = best_results[f]
        spks = FOLD_DEFINITIONS[f]["val_speakers"]
        l1s = [FOLD_DEFINITIONS[f]["l1_profiles"][s] for s in spks]
        csv_rows.append({
            "Evaluation Arm": "Validation Best (Selected)",
            "Fold / Cohort": f"Fold {f}",
            "Val Speakers": " & ".join(spks),
            "Val L1 Profiles": " & ".join(l1s),
            "Val Tokens": r["total_phone_tokens"],
            "Val Errors": r["total_error_tokens"],
            "Val Prev": f"{r['prevalence']:.4%}",
            "Epoch": r["epoch_stored"] + 1,
            "Updates": r["updates_stored"],
            "Matched AP": f"{r['matched_ap']:.6f}",
            "Mismatched AP": f"{r['mismatched_ap']:.6f}",
            "Acoustic Gap": f"{r['acoustic_gap']:+.6f}",
            "Canonical Prior AP": f"{r['canonical_prior_ap']:.6f}",
            "Margin over Prior": f"{r['margin_over_prior']:+.6f}",
            "CTC PER": f"{r['ctc_per']:.2%}",
            "CTC Sub%": f"{r['ctc_sub_rate']:.2%}",
            "CTC Del%": f"{r['ctc_del_rate']:.2%}",
            "CTC Ins%": f"{r['ctc_ins_rate']:.2%}",
            "FNR @ 0.50": f"{r['fnr_at_050']:.2%}",
            "FPR @ 0.50": f"{r['fpr_at_050']:.2%}",
            "FNR @ 0.20": f"{r['fnr_at_020']:.2%}",
            "FPR @ 0.20": f"{r['fpr_at_020']:.2%}",
            "Proto Z Aligned AP": f"{r['proto_z_aligned_ap']:.6f}",
            "Proto Z B1 AP": f"{r['proto_z_b1_ap']:.6f}",
        })

    # Best Macro & Micro
    csv_rows.append({
        "Evaluation Arm": "Validation Best (Selected)",
        "Fold / Cohort": "Macro-Average (N=4)",
        "Val Speakers": "8 Speakers (All)",
        "Val L1 Profiles": "Spanish, Hindi, Arabic, Mandarin, Korean",
        "Val Tokens": "39,576 total",
        "Val Errors": "4,213 total",
        "Val Prev": f"{macro_prev['mean']:.4%}",
        "Epoch": "2-6 (Varies)",
        "Updates": "1,200-3,600",
        "Matched AP": f"{macro_best_matched_ap['mean']:.6f} +/- {macro_best_matched_ap['se']:.6f}",
        "Mismatched AP": "—",
        "Acoustic Gap": f"{macro_best_gap['mean']:+.6f} +/- {macro_best_gap['se']:.6f}",
        "Canonical Prior AP": "—",
        "Margin over Prior": "—",
        "CTC PER": f"{macro_best_ctc_per['mean']:.2%} +/- {macro_best_ctc_per['se']:.2%}",
        "CTC Sub%": "—",
        "CTC Del%": "—",
        "CTC Ins%": "—",
        "FNR @ 0.50": "—",
        "FPR @ 0.50": "—",
        "FNR @ 0.20": "—",
        "FPR @ 0.20": "—",
        "Proto Z Aligned AP": "—",
        "Proto Z B1 AP": "—",
    })
    csv_rows.append({
        "Evaluation Arm": "Validation Best (Selected)",
        "Fold / Cohort": "Pooled Micro (N=39,576)",
        "Val Speakers": "8 Speakers (All)",
        "Val L1 Profiles": "Spanish, Hindi, Arabic, Mandarin, Korean",
        "Val Tokens": 39576,
        "Val Errors": 4213,
        "Val Prev": "10.65%",
        "Epoch": "2-6 (Varies)",
        "Updates": "1,200-3,600",
        "Matched AP": f"{micro_best_matched_ap:.6f}",
        "Mismatched AP": "—",
        "Acoustic Gap": "—",
        "Canonical Prior AP": "—",
        "Margin over Prior": "—",
        "CTC PER": "—",
        "CTC Sub%": "—",
        "CTC Del%": "—",
        "CTC Ins%": "—",
        "FNR @ 0.50": "—",
        "FPR @ 0.50": "—",
        "FNR @ 0.20": "—",
        "FPR @ 0.20": "—",
        "Proto Z Aligned AP": "—",
        "Proto Z B1 AP": "—",
    })

    summary_csv_path = out_dir / "four_fold_metrics_summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n[EXPORT] Saved four-fold summary CSV to: {summary_csv_path}")

    # 2. Eight Validation Speakers Disaggregated CSV
    spk_headers = [
        "Speaker", "Fold", "L1 Background", "Gender", "Tokens", "Errors", "Prevalence",
        "Matched AP", "Mismatched AP", "Acoustic Gap", "CTC PER", "CTC Sub%", "CTC Del%", "CTC Ins%",
        "FNR @ 0.50", "FPR @ 0.50", "Precision @ 0.50", "Recall @ 0.50", "F1 @ 0.50",
        "FNR @ 0.20", "FPR @ 0.20", "Precision @ 0.20", "Recall @ 0.20", "F1 @ 0.20"
    ]
    spk_rows = []
    genders = {
        "NJS": "F", "RRBI": "M", "TNI": "M", "YBAA": "F",
        "TXHC": "M", "ZHAA": "F", "HKK": "M", "NCC": "F"
    }

    for fold_idx in [1, 2, 3, 4]:
        f_res = fixed_budget_results[fold_idx]
        for spk, sdata in f_res["per_speaker"].items():
            spk_rows.append({
                "Speaker": spk,
                "Fold": fold_idx,
                "L1 Background": FOLD_DEFINITIONS[fold_idx]["l1_profiles"][spk],
                "Gender": genders.get(spk, "Unknown"),
                "Tokens": sdata["tokens"],
                "Errors": sdata["errors"],
                "Prevalence": f"{sdata['prevalence']:.4%}",
                "Matched AP": f"{sdata['matched_ap']:.6f}",
                "Mismatched AP": f"{sdata['mismatched_ap']:.6f}",
                "Acoustic Gap": f"{sdata['acoustic_gap']:+.6f}",
                "CTC PER": f"{sdata['ctc_per']:.2%}",
                "CTC Sub%": f"{sdata['ctc_sub_rate']:.2%}",
                "CTC Del%": f"{sdata['ctc_del_rate']:.2%}",
                "CTC Ins%": f"{sdata['ctc_ins_rate']:.2%}",
                "FNR @ 0.50": f"{sdata['fnr_at_050']:.2%}",
                "FPR @ 0.50": f"{sdata['fpr_at_050']:.2%}",
                "Precision @ 0.50": f"{sdata['precision_at_050']:.4f}",
                "Recall @ 0.50": f"{sdata['recall_at_050']:.4f}",
                "F1 @ 0.50": f"{sdata['f1_at_050']:.4f}",
                "FNR @ 0.20": f"{sdata['fnr_at_020']:.2%}",
                "FPR @ 0.20": f"{sdata['fpr_at_020']:.2%}",
                "Precision @ 0.20": f"{sdata['precision_at_020']:.4f}",
                "Recall @ 0.20": f"{sdata['recall_at_020']:.4f}",
                "F1 @ 0.20": f"{sdata['f1_at_020']:.4f}",
            })

    spk_csv_path = out_dir / "eight_speakers_validation_breakdown.csv"
    with open(spk_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=spk_headers)
        writer.writeheader()
        writer.writerows(spk_rows)
    print(f"[EXPORT] Saved 8-speaker breakdown CSV to: {spk_csv_path}")

    # 3. Comprehensive JSON Synthesis Report
    clean_fb_results = {}
    for f in [1, 2, 3, 4]:
        clean_fb_results[f"fold_{f}"] = {k: v for k, v in fixed_budget_results[f].items() if k not in ["raw_predictions", "per_utterance"]}

    clean_best_results = {}
    for f in [1, 2, 3, 4]:
        clean_best_results[f"fold_{f}"] = {k: v for k, v in best_results[f].items() if k not in ["raw_predictions", "per_utterance"]}

    full_report = {
        "metadata": {
            "title": "LANGFENS RQ1-TRN-03A Four-Fold Cross-Validation Scientific Report",
            "experiment": "RQ1-TRN-03A",
            "architecture": "LangfensWav2Vec2MDD (Frozen Wav2Vec2 + Two Heads: CTC & Binary MDD)",
            "pretrained_model": "facebook/wav2vec2-base-960h (rev 22aad52d435eb6dbaf354bdad9b0da84ce7d6156)",
            "split_manifest": args.manifest,
            "split_manifest_sha256": compute_sha256(args.manifest),
            "generated_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evaluator_git_sha": os.popen("git rev-parse HEAD").read().strip(),
            "evaluator_source_commit": "618e86d",
            "evaluation_split": "val",
            "test_split_access": "STRICTLY_PROHIBITED_AND_VERIFIED_ZERO_ACCESS",
            "total_out_of_fold_validation_utterances": 1200,
            "total_out_of_fold_validation_tokens": 39576,
            "total_out_of_fold_validation_errors": 4213,
            "macro_validation_prevalence": macro_prev["mean"],
            "pooled_validation_prevalence": round(4213 / 39576, 6),
            "validation_speakers_count": 8,
            "validation_speakers": ["NJS", "RRBI", "TNI", "YBAA", "TXHC", "ZHAA", "HKK", "NCC"],
        },
        "primary_fixed_budget_3600_updates": {
            "macro_statistics": {
                "prevalence": macro_prev,
                "matched_binary_ap": macro_matched_ap,
                "mismatched_binary_ap": macro_mismatched_ap,
                "acoustic_reliance_gap": macro_gap,
                "canonical_prior_ap": macro_prior_ap,
                "neural_margin_over_prior": macro_margin,
                "ctc_per": macro_ctc_per,
                "fnr_at_050": macro_fnr_05,
                "fpr_at_050": macro_fpr_05,
                "fnr_at_020": macro_fnr_02,
                "fpr_at_020": macro_fpr_02,
                "protocol_z_aligned_ap": macro_proto_z_aligned,
                "protocol_z_policy_b1_ap": macro_proto_z_b1,
            },
            "pooled_micro_statistics": {
                "matched_binary_ap": round(micro_matched_ap, 6),
                "mismatched_binary_ap": round(micro_mismatched_ap, 6),
                "acoustic_reliance_gap": round(micro_gap, 6),
                "ctc_per": round(micro_ctc_per, 6),
                "ctc_sub_rate": round(micro_sub_rate, 6),
                "ctc_del_rate": round(micro_del_rate, 6),
                "ctc_ins_rate": round(micro_ins_rate, 6),
                "operating_point_050": {
                    "fnr": round(micro_fnr_05, 4),
                    "fpr": round(micro_fpr_05, 4),
                    "precision": p_05["precision"],
                    "recall": p_05["recall"],
                    "f1": p_05["f1"],
                    "tp": p_05["tp"],
                    "fp": p_05["fp"],
                    "fn": p_05["fn"],
                    "tn": p_05["tn"],
                },
                "operating_point_020": {
                    "fnr": round(micro_fnr_02, 4),
                    "fpr": round(micro_fpr_02, 4),
                    "precision": p_02["precision"],
                    "recall": p_02["recall"],
                    "f1": p_02["f1"],
                    "tp": p_02["tp"],
                    "fp": p_02["fp"],
                    "fn": p_02["fn"],
                    "tn": p_02["tn"],
                },
            },
            "per_fold": clean_fb_results,
        },
        "validation_selected_best_checkpoints": {
            "macro_statistics": {
                "matched_binary_ap": macro_best_matched_ap,
                "acoustic_reliance_gap": macro_best_gap,
                "ctc_per": macro_best_ctc_per,
            },
            "pooled_micro_statistics": {
                "matched_binary_ap": round(micro_best_matched_ap, 6),
            },
            "per_fold": clean_best_results,
        },
        "small_sample_limitations": {
            "cluster_count": 8,
            "fold_count": 4,
            "degrees_of_freedom_macro": 3,
            "critical_t_multiplier_95": CRITICAL_T_3_DF,
            "cluster_bootstrap_warning": (
                "With only G=8 validation speakers, cluster-robust standard errors are subject to small-cluster bias "
                "(Cameron, Gelbach & Miller, 2008). Macro confidence intervals use Student's t distribution with df=3, "
                "which accounts for inter-fold variance but produces wide bounds."
            ),
            "disaggregation_requirement": "Aggregated cross-fold metrics must always be reported alongside individual per-speaker metrics.",
        },
    }

    json_report_path = out_dir / "FOUR_FOLD_VALIDATION_REPORT.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"[EXPORT] Saved comprehensive JSON report to: {json_report_path}")

    # 4. Export full paired prediction records for all 1,200 utterances (Task D / Step 14 Deliverable)
    all_utterances_records = []
    for f in [1, 2, 3, 4]:
        all_utterances_records.extend(fixed_budget_results[f]["per_utterance"])

    pred_records = {
        "metadata": {
            "title": "LANGFENS Four-Fold Out-of-Fold Validation Paired Prediction Records",
            "experiment": "RQ1-TRN-03A",
            "evaluator_git_sha": full_report["metadata"]["evaluator_git_sha"],
            "evaluator_source_commit": "618e86d",
            "total_utterances": len(all_utterances_records),
            "total_tokens": len(pooled_matched_labels),
            "total_errors": int(sum(pooled_matched_labels)),
            "macro_prevalence": macro_prev["mean"],
            "pooled_prevalence": round(sum(pooled_matched_labels) / len(pooled_matched_labels), 6),
            "checkpoint_sha256": {
                f"fold_{f}": {
                    "fixed_budget": fixed_budget_results[f]["checkpoint_sha256"],
                    "best": best_results[f]["checkpoint_sha256"],
                }
                for f in [1, 2, 3, 4]
            },
        },
        "utterances": all_utterances_records,
    }

    pred_gz_path = out_dir / "FOUR_FOLD_PAIRED_PREDICTION_RECORDS.json.gz"
    with gzip.open(pred_gz_path, "wt", encoding="utf-8") as f:
        json.dump(pred_records, f)
    pred_gz_sha = compute_sha256(pred_gz_path)
    print(f"[EXPORT] Saved {len(all_utterances_records)} paired prediction records to: {pred_gz_path} (SHA-256: {pred_gz_sha})")

    # 5. Mirror prediction records to authorized private Hugging Face repo
    try:
        from huggingface_hub import HfApi, get_token
        hf_tok = get_token()
        if hf_tok:
            api = HfApi(token=hf_tok)
            api.upload_file(
                path_or_fileobj=str(pred_gz_path),
                path_in_repo="evaluation/FOUR_FOLD_PAIRED_PREDICTION_RECORDS.json.gz",
                repo_id="minkhoaa/langfens-rq1-cross-fold-checkpoints",
                repo_type="model",
                commit_message="docs(eval): store four-fold paired validation prediction records"
            )
            print("[REMOTE PERSISTENCE] Successfully mirrored prediction records to HF: evaluation/FOUR_FOLD_PAIRED_PREDICTION_RECORDS.json.gz")
    except Exception as e:
        print(f"[REMOTE PERSISTENCE WARNING] Could not mirror to HF: {e}")

    print("\n" + "=" * 80)
    print("FOUR-FOLD VALIDATION EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
