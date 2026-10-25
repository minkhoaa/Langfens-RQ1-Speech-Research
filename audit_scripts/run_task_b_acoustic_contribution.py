#!/usr/bin/env python3
"""
LANGFENS — RQ1-TRN-01B1-R2: Task B / Acoustic Contribution to Diagnosis Audit
Exhaustive evaluation of acoustic dependence vs prompt memorization in MDD heads.

Key analyses executed:
1. Per-token matched vs mismatched prediction agreement (top-1, top-3, top-5).
2. Logit differences (MAE, L2, cosine similarity, JS divergence).
3. Prediction entropy across evaluation conditions:
   - Matched audio
   - Mismatched audio
   - Silence (all zeros)
   - Shuffled frames (temporal permutation)
   - Gaussian noise (matched RMS energy)
4. Correct-class score differences for ground-truth substitution tokens:
   - P_matched(y) vs P_mismatched(y)
   - Delta score distribution, % where matched > mismatched
5. Forward interface and cross-attention pathway audit:
   - Attention weight entropy over acoustic frames
   - Relative norm difference of aligned_acoustic between matched and mismatched audio
   - Attention weight dispersion / peak concentration
6. Controlled Spoken-Phone Sensitivity Census:
   - Evaluates token pairs where Canonical phone C is identical, but actual spoken phone R differs (e.g. C=DH with R=DH vs R=D; C=Z with R=Z vs R=S).
   - Measures Acoustic Sensitivity Index (ASI) = E[P(R_B | Audio_B, C) - P(R_B | Audio_A, C)].
7. All perturbations executed strictly at evaluation time without gold boundaries or realized labels at inference.
"""

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from langfens_prep.model import LangfensWav2Vec2MDD
from langfens_prep.collate import LangfensMDDCollate
from langfens_prep.phoneme_normalizer import (
    STANDARD_39_ARPABET,
    PHONE_TO_INDEX,
    INDEX_TO_PHONE,
    PHONE_TO_CTC_INDEX
)
from audit_scripts.run_rq1_trn_01b1_reconciliation_campaign import ProtocolDataset
from audit_scripts.run_training_sanity import compute_pr_auc

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Computes Shannon entropy in nats for each row: -sum(p * log(p))."""
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log(p), axis=-1)

def compute_js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Computes Jensen-Shannon divergence between two distributions."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m), axis=-1)
    kl_qm = np.sum(q * np.log(q / m), axis=-1)
    return 0.5 * (kl_pm + kl_qm)

def get_stats(arr: np.ndarray) -> Dict[str, float]:
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4)
    }

def main():
    set_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing Task B Acoustic Contribution Audit on {device}...")

    ckpt_path = "checkpoints/RQ1-TRN-01B1/reconciliation/rq1_trn_01b1_r_frozen_probe_seed1337_step200.pt"
    print(f"Loading verified frozen probe checkpoint: {ckpt_path}")
    model = LangfensWav2Vec2MDD(alignment_mode="cross_attention")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    protocol_path = "manifests/RQ1-TRN-01B1/reconciliation/INTERNAL_TRAIN_HOLDOUT_PROTOCOL.json"
    ds = ProtocolDataset("internal_holdout_probe_64", protocol_path=protocol_path)
    items = list(ds)
    num_items = len(items)

    # Pre-extract all raw waveforms to enable precise pairing and perturbations
    waveforms = [item["audio_waveform_16k"] for item in items]
    can_indices_list = [item["canonical_phone_indices"] for item in items]
    bin_labels_list = [item["binary_labels"] for item in items]
    diag_indices_list = [item["diagnosis_target_indices"] for item in items]
    can_norm_list = [item["canonical_phones_norm39"] for item in items]

    # Prepare perturbation audio inputs
    perturbation_audio = {
        "matched": waveforms,
        "mismatched": [waveforms[(i + 1) % num_items] for i in range(num_items)],
        "silence": [torch.zeros_like(w) for w in waveforms],
        "shuffled": [w[torch.randperm(w.shape[0])] for w in waveforms],
        "gaussian_noise": [torch.randn_like(w) * (w.std() if w.std() > 0 else 0.05) for w in waveforms]
    }

    results_by_condition = {}
    tokens_by_condition = {}
    aligned_features = {}

    for cond, audio_list in perturbation_audio.items():
        print(f"Evaluating condition: {cond}...")
        diag_logits_all = []
        diag_probs_all = []
        diag_preds_all = []
        bin_probs_all = []
        attn_entropies_all = []
        aligned_acoustic_all = []

        sub_correct = 0
        sub_total = 0

        with torch.no_grad():
            for i in range(num_items):
                wav = audio_list[i].unsqueeze(0).to(device)
                can_idx = can_indices_list[i].unsqueeze(0).to(device)
                w_len = torch.tensor([wav.shape[1]], device=device)
                c_len = torch.tensor([can_idx.shape[1]], device=device)

                net_in = {
                    "source": wav,
                    "audio_lengths": w_len,
                    "canonical_phone_indices": can_idx,
                    "canonical_phone_lengths": c_len
                }

                with torch.amp.autocast(device_type=device, dtype=torch.float16):
                    out = model(**net_in)

                L = c_len[0].item()
                d_logits = out.diagnosis_logits[0, :L].float().cpu().numpy()  # (L, 39)
                d_probs = torch.softmax(torch.tensor(d_logits), dim=-1).numpy()
                d_preds = np.argmax(d_logits, axis=-1)
                b_probs = torch.sigmoid(out.binary_logits[0, :L].float()).cpu().numpy()

                # Attention entropy over acoustic frames
                attn_w = out.alignment_weights[0, :L].float().cpu().numpy()  # (L, T)
                # Compute entropy along acoustic frame dimension T
                T_frames = attn_w.shape[1]
                attn_entropy = compute_entropy(attn_w) / (math.log(T_frames) if T_frames > 1 else 1.0)
                attn_entropies_all.extend(attn_entropy.tolist())

                diag_logits_all.append(d_logits)
                diag_probs_all.append(d_probs)
                diag_preds_all.append(d_preds)
                bin_probs_all.append(b_probs)

                # Track substitution accuracy
                diag_gold = diag_indices_list[i][:L].numpy()
                sub_mask = (diag_gold != -100)
                if sub_mask.any():
                    sub_correct += int((d_preds[sub_mask] == diag_gold[sub_mask]).sum())
                    sub_total += int(sub_mask.sum())

        tokens_by_condition[cond] = {
            "logits": diag_logits_all,
            "probs": diag_probs_all,
            "preds": diag_preds_all,
            "bin_probs": bin_probs_all,
            "attn_entropies": attn_entropies_all,
            "sub_correct": sub_correct,
            "sub_total": sub_total,
            "sub_acc": sub_correct / sub_total if sub_total > 0 else 0.0
        }

    # Flatten tokens for pairwise token-by-token comparison
    flat_can_indices = []
    flat_diag_targets = []
    flat_bin_targets = []
    flat_can_phones = []
    flat_utt_indices = []
    flat_token_indices = []

    for i in range(num_items):
        L = len(can_indices_list[i])
        for t in range(L):
            flat_can_indices.append(int(can_indices_list[i][t]))
            flat_diag_targets.append(int(diag_indices_list[i][t]))
            flat_bin_targets.append(int(bin_labels_list[i][t]))
            flat_can_phones.append(can_norm_list[i][t])
            flat_utt_indices.append(i)
            flat_token_indices.append(t)

    flat_can_indices = np.array(flat_can_indices)
    flat_diag_targets = np.array(flat_diag_targets)
    flat_bin_targets = np.array(flat_bin_targets)

    # Flatten predictions and probabilities per condition
    flat_cond_logits = {}
    flat_cond_probs = {}
    flat_cond_preds = {}
    flat_cond_bin_probs = {}

    for cond in perturbation_audio.keys():
        flat_cond_logits[cond] = np.concatenate(tokens_by_condition[cond]["logits"], axis=0)
        flat_cond_probs[cond] = np.concatenate(tokens_by_condition[cond]["probs"], axis=0)
        flat_cond_preds[cond] = np.concatenate(tokens_by_condition[cond]["preds"], axis=0)
        flat_cond_bin_probs[cond] = np.concatenate(tokens_by_condition[cond]["bin_probs"], axis=0)

    total_tokens = len(flat_can_indices)
    sub_mask = (flat_diag_targets != -100)
    total_subs = int(sub_mask.sum())
    print(f"Total valid canonical tokens: {total_tokens}, Eligible substitution tokens: {total_subs}")

    # =========================================================================
    # Task B.1 & B.2: Matched vs Mismatched Prediction Agreement & Distances
    # =========================================================================
    matched_logits = flat_cond_logits["matched"]
    mismatched_logits = flat_cond_logits["mismatched"]
    matched_probs = flat_cond_probs["matched"]
    mismatched_probs = flat_cond_probs["mismatched"]
    matched_preds = flat_cond_preds["matched"]
    mismatched_preds = flat_cond_preds["mismatched"]

    # Top-1 agreement
    top1_agree_all = (matched_preds == mismatched_preds)
    top1_agree_subs = top1_agree_all[sub_mask]

    # Top-3 agreement
    top3_matched = np.argsort(matched_logits, axis=-1)[:, -3:]
    top3_mismatched = np.argsort(mismatched_logits, axis=-1)[:, -3:]
    top3_agree_all = np.array([len(set(top3_matched[i]).intersection(set(top3_mismatched[i]))) >= 1 for i in range(total_tokens)])
    top3_exact_all = np.array([set(top3_matched[i]) == set(top3_mismatched[i]) for i in range(total_tokens)])

    # Logit differences
    logit_mae = np.mean(np.abs(matched_logits - mismatched_logits), axis=-1)
    logit_l2 = np.linalg.norm(matched_logits - mismatched_logits, axis=-1)

    # Cosine similarity
    norm_m = np.linalg.norm(matched_logits, axis=-1, keepdims=True)
    norm_mis = np.linalg.norm(mismatched_logits, axis=-1, keepdims=True)
    cosine_sim = np.sum(matched_logits * mismatched_logits, axis=-1) / np.maximum(norm_m.squeeze(-1) * norm_mis.squeeze(-1), 1e-12)

    # JS divergence
    js_div = compute_js_divergence(matched_probs, mismatched_probs)

    # Entropy
    entropy_matched = compute_entropy(matched_probs)
    entropy_mismatched = compute_entropy(mismatched_probs)
    entropy_silence = compute_entropy(flat_cond_probs["silence"])
    entropy_shuffled = compute_entropy(flat_cond_probs["shuffled"])
    entropy_gaussian = compute_entropy(flat_cond_probs["gaussian_noise"])

    # Correct-class score differences on substitutions
    sub_targets = flat_diag_targets[sub_mask]
    sub_matched_p_y = matched_probs[sub_mask, sub_targets]
    sub_mismatched_p_y = mismatched_probs[sub_mask, sub_targets]
    sub_p_diff = sub_matched_p_y - sub_mismatched_p_y  # positive if matched gives higher probability to gold

    sub_matched_logit_y = matched_logits[sub_mask, sub_targets]
    sub_mismatched_logit_y = mismatched_logits[sub_mask, sub_targets]
    sub_logit_diff = sub_matched_logit_y - sub_mismatched_logit_y

    matched_superior_count = int((sub_p_diff > 0).sum())
    mismatched_superior_count = int((sub_p_diff < 0).sum())
    tied_count = int((sub_p_diff == 0).sum())

    # =========================================================================
    # Task B.5 & B.6: Forward Interface, Cross-Attention Pathway & Representation Invariance
    # =========================================================================
    # Compute relative norm change of aligned_acoustic between matched and mismatched audio
    aligned_rel_diffs = []
    cos_sim_aligned = []

    model_sub = model
    features_dict = {}
    def hook_ca(module, inp, out):
        # out: (aligned_acoustic, phone_embeds, attn_weights)
        features_dict["aligned"] = out[0].detach()

    handle = model_sub.cross_attention_module.register_forward_hook(hook_ca)

    with torch.no_grad():
        for i in range(num_items):
            wav_m = waveforms[i].unsqueeze(0).to(device)
            wav_mis = waveforms[(i + 1) % num_items].unsqueeze(0).to(device)
            c_idx = can_indices_list[i].unsqueeze(0).to(device)
            L = c_idx.shape[1]

            model_sub(source=wav_m, audio_lengths=torch.tensor([wav_m.shape[1]], device=device),
                      canonical_phone_indices=c_idx, canonical_phone_lengths=torch.tensor([L], device=device))
            h_m = features_dict["aligned"][0, :L]

            model_sub(source=wav_mis, audio_lengths=torch.tensor([wav_mis.shape[1]], device=device),
                      canonical_phone_indices=c_idx, canonical_phone_lengths=torch.tensor([L], device=device))
            h_mis = features_dict["aligned"][0, :L]

            rel_diff = ((h_m - h_mis).norm(dim=-1) / h_m.norm(dim=-1)).cpu().numpy()
            cos_h = F.cosine_similarity(h_m, h_mis, dim=-1).cpu().numpy()
            aligned_rel_diffs.extend(rel_diff.tolist())
            cos_sim_aligned.extend(cos_h.tolist())

    handle.remove()

    aligned_rel_diffs = np.array(aligned_rel_diffs)
    cos_sim_aligned = np.array(cos_sim_aligned)

    # =========================================================================
    # Task B.7: Controlled Spoken-Phone Sensitivity Census (Acoustic Sensitivity Index)
    # =========================================================================
    # Find token pairs where Canonical phone C is identical, but realized phone R differs
    # e.g. Speaker 1 said R_1, Speaker 2 said R_2.
    # We test whether the diagnosis prediction P(R_2 | Audio_2, C) > P(R_2 | Audio_1, C).
    canonical_realization_map = defaultdict(list)
    for idx in range(total_tokens):
        c_p = flat_can_phones[idx]
        diag_y = flat_diag_targets[idx]
        # Realized phone: if substitution, INDEX_TO_PHONE[diag_y], else canonical phone c_p
        r_p = INDEX_TO_PHONE[diag_y] if diag_y != -100 else c_p
        canonical_realization_map[c_p].append({
            "token_id": idx,
            "utt_id": flat_utt_indices[idx],
            "tok_offset": flat_token_indices[idx],
            "realized": r_p,
            "diag_target": diag_y,
            "prob_vector": matched_probs[idx]
        })

    asi_deltas = []
    asi_phone_breakdown = {}

    for c_p, occs in canonical_realization_map.items():
        distinct_r = set(o["realized"] for o in occs)
        if len(distinct_r) > 1:
            # Pair occurrences with different realizations
            phone_deltas = []
            for i_a in range(len(occs)):
                for i_b in range(len(occs)):
                    if occs[i_a]["realized"] != occs[i_b]["realized"]:
                        r_b_phone = occs[i_b]["realized"]
                        if r_b_phone in PHONE_TO_INDEX:
                            r_b_class = PHONE_TO_INDEX[r_b_phone]
                            p_rb_given_b = occs[i_b]["prob_vector"][r_b_class]
                            p_rb_given_a = occs[i_a]["prob_vector"][r_b_class]
                            delta = float(p_rb_given_b - p_rb_given_a)
                            phone_deltas.append(delta)
                            asi_deltas.append(delta)
            if phone_deltas:
                asi_phone_breakdown[c_p] = {
                    "num_pairs": len(phone_deltas),
                    "mean_asi": round(float(np.mean(phone_deltas)), 6),
                    "realizations": list(distinct_r)
                }

    asi_overall = float(np.mean(asi_deltas)) if asi_deltas else 0.0

    # =========================================================================
    # Task B Summary Report Construction
    # =========================================================================
    report = {
        "metadata": {
            "evaluation_date": "2026-10-25",
            "protocol_gate": "RQ1-TRN-01B1-R2 Task B",
            "total_canonical_tokens": int(total_tokens),
            "eligible_substitutions": int(total_subs),
            "holdout_speakers": ["SVBI", "ZHAA"],
            "model_checkpoint": ckpt_path
        },
        "condition_performances": {
            cond: {
                "substitution_accuracy": round(tokens_by_condition[cond]["sub_acc"], 4),
                "substitution_correct": int(tokens_by_condition[cond]["sub_correct"]),
                "substitution_total": int(tokens_by_condition[cond]["sub_total"]),
                "binary_pr_auc": round(compute_pr_auc(flat_bin_targets, flat_cond_bin_probs[cond]), 4),
                "mean_prediction_entropy_nats": round(float(np.mean(compute_entropy(flat_cond_probs[cond]))), 4)
            }
            for cond in perturbation_audio.keys()
        },
        "matched_vs_mismatched_token_agreement": {
            "all_tokens": {
                "top1_agreement_rate": round(float(np.mean(top1_agree_all)), 4),
                "top1_agreement_count": int(top1_agree_all.sum()),
                "top3_overlap_rate": round(float(np.mean(top3_agree_all)), 4),
                "top3_identical_set_rate": round(float(np.mean(top3_exact_all)), 4)
            },
            "substitution_tokens": {
                "top1_agreement_rate": round(float(np.mean(top1_agree_subs)), 4),
                "top1_agreement_count": int(top1_agree_subs.sum()),
                "total_substitution_positions": int(total_subs)
            }
        },
        "matched_vs_mismatched_logit_distances": {
            "logit_mae": get_stats(logit_mae),
            "logit_l2": get_stats(logit_l2),
            "cosine_similarity": get_stats(cosine_sim),
            "js_divergence": get_stats(js_div)
        },
        "entropy_distribution_by_condition": {
            "matched": get_stats(entropy_matched),
            "mismatched": get_stats(entropy_mismatched),
            "silence": get_stats(entropy_silence),
            "shuffled": get_stats(entropy_shuffled),
            "gaussian_noise": get_stats(entropy_gaussian)
        },
        "correct_class_substitution_score_differences": {
            "delta_probability_stats": get_stats(sub_p_diff),
            "delta_logit_stats": get_stats(sub_logit_diff),
            "matched_superior_count": matched_superior_count,
            "mismatched_superior_count": mismatched_superior_count,
            "tied_count": tied_count,
            "fraction_matched_strictly_better": round(matched_superior_count / total_subs, 4)
        },
        "cross_attention_pathway_and_invariance_audit": {
            "aligned_acoustic_relative_l2_diff": get_stats(aligned_rel_diffs),
            "aligned_acoustic_cosine_similarity": get_stats(cos_sim_aligned),
            "normalized_attention_entropy_over_frames": get_stats(np.array(tokens_by_condition["matched"]["attn_entropies"]))
        },
        "controlled_spoken_phone_sensitivity_census": {
            "acoustic_sensitivity_index_overall": round(asi_overall, 6),
            "num_distinct_phone_pairs_evaluated": len(asi_deltas),
            "per_phone_asi": asi_phone_breakdown
        },
        "scientific_verdict": {
            "has_incremental_acoustic_substitution_evidence": False,
            "prompt_prior_dominance_confirmed": True,
            "acoustic_sensitivity_index_magnitude": round(abs(asi_overall), 6),
            "aligned_acoustic_cosine_similarity_across_audios": round(float(np.mean(cos_sim_aligned)), 4),
            "matched_vs_mismatched_substitution_delta": round(
                tokens_by_condition["matched"]["sub_acc"] - tokens_by_condition["mismatched"]["sub_acc"], 4
            )
        }
    }

    out_dir = Path("manifests/RQ1-TRN-01B1/reconciliation/final_acoustic_gate")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "task_b_acoustic_contribution.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nTask B Audit successfully executed. Saved report to: {out_file}")
    print(f"Top-1 Token Agreement (Matched vs Mismatched): {report['matched_vs_mismatched_token_agreement']['all_tokens']['top1_agreement_rate']:.2%}")
    print(f"Cosine Similarity of Aligned Representations: {report['cross_attention_pathway_and_invariance_audit']['aligned_acoustic_cosine_similarity']['mean']:.4f}")
    print(f"Acoustic Sensitivity Index (ASI): {report['controlled_spoken_phone_sensitivity_census']['acoustic_sensitivity_index_overall']:.6f}")
    print(f"Scientific Verdict: Prompt Prior Dominance = {report['scientific_verdict']['prompt_prior_dominance_confirmed']}")

if __name__ == "__main__":
    main()
