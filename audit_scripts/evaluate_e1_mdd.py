#!/usr/bin/env python3
"""
LANGFENS — RQ-MODEL-01 / E1: Unified Evaluation Harness.
Evaluates both Baseline R and E1 (Alignment-Aware MDD) models under standardized protocols.

Primary Registered Evaluation Metrics:
1. omission_ap: Average Precision specifically on deletion (omission) tokens vs. correct tokens.
2. omission_recall_fpr15: Diagnostic Recall on omission tokens at matched FPR <= 15% ceiling on correct tokens.
3. binary_ap: Overall Average Precision on all canonical phone error tokens vs. correct tokens.
4. ctc_per: Greedy CTC Phone Error Rate (acoustic fidelity guardrail).

Guarantees:
- Strict validation-only evaluation; zero access to sealed test set.
- Deployable evaluation without gold TextGrid timestamps as inputs.
- Dual evaluation under Protocol O (canonical prompt tokens) and Protocol Z (zero-oracle CMUdict G2P).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Ensure repo root and src are on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audit_scripts.evaluate_fold1_ctc_binary import (
    average_precision_binary,
    greedy_ctc_decode,
    compute_levenshtein_edits,
    build_eval_model,
)
from audit_scripts.diagnose_omissions_and_false_negatives import (
    roc_auc_binary,
    extract_word_positions,
)
from audit_scripts.diagnose_counterfactual_validity import (
    diagnostic_threshold,
    operating,
)
from langfens_prep.alignment import align_canonical_sequences
from langfens_prep.audio_loader import load_l2arctic_audio_from_zip, get_l2arctic_reader
from langfens_prep.dataset import load_arctic_prompts
from langfens_prep.g2p import CMUDictG2P
from langfens_prep.model import LangfensWav2Vec2MDD
from langfens_prep.phoneme_normalizer import PHONE_TO_CTC_INDEX, STANDARD_39_ARPABET
from langfens_prep.textgrid_parser import parse_gold_textgrid_content


DEFAULT_MANIFEST = Path("manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json")

FOLD_ZIPS = {
    1: "datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip",
    2: "datasets/L2-ARCTIC/authorized_RQ1-TRN_fold2_train_val_only.zip",
    3: "datasets/L2-ARCTIC/authorized_RQ1-TRN_fold3_train_val_only.zip",
    4: "datasets/L2-ARCTIC/authorized_RQ1-TRN_fold4_train_val_only.zip",
}

DEMOGRAPHICS = {
    "ABA":   {"gender": "M", "l1": "Arabic"},
    "SKA":   {"gender": "F", "l1": "Arabic"},
    "YBAA":  {"gender": "M", "l1": "Arabic"},
    "ZHAA":  {"gender": "F", "l1": "Arabic"},
    "BWC":   {"gender": "M", "l1": "Chinese"},
    "LXC":   {"gender": "F", "l1": "Chinese"},
    "NCC":   {"gender": "F", "l1": "Chinese"},
    "TXHC":  {"gender": "M", "l1": "Chinese"},
    "ASI":   {"gender": "M", "l1": "Hindi"},
    "RRBI":  {"gender": "M", "l1": "Hindi"},
    "SVBI":  {"gender": "F", "l1": "Hindi"},
    "TNI":   {"gender": "F", "l1": "Hindi"},
    "HJK":   {"gender": "F", "l1": "Korean"},
    "HKK":   {"gender": "M", "l1": "Korean"},
    "YDCK":  {"gender": "F", "l1": "Korean"},
    "YKWK":  {"gender": "M", "l1": "Korean"},
    "EBVS":  {"gender": "M", "l1": "Spanish"},
    "ERMS":  {"gender": "M", "l1": "Spanish"},
    "MBMPS": {"gender": "F", "l1": "Spanish"},
    "NJS":   {"gender": "F", "l1": "Spanish"},
    "HQTV":  {"gender": "M", "l1": "Vietnamese"},
    "PNV":   {"gender": "F", "l1": "Vietnamese"},
    "THV":   {"gender": "F", "l1": "Vietnamese"},
    "TLV":   {"gender": "M", "l1": "Vietnamese"},
}


def compute_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_primary_metrics(
    labels: List[int],
    scores: List[float],
    error_types: List[str],
    ref_ctc_all: List[List[int]],
    hyp_ctc_all: List[List[int]],
) -> Dict[str, Any]:
    """Computes the 4 registered primary metrics plus diagnostic breakdowns."""
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    et = np.asarray(error_types)

    # 1. Binary AP & ROC-AUC (all error tokens vs. correct)
    bin_ap = float(average_precision_binary(y, s))
    bin_auc = float(roc_auc_binary(y, s))

    # 2. Omission AP (deletions vs. correct only)
    omiss_mask = (et == "deletion") | (et == "correct")
    if np.any(et == "deletion") and np.any(et == "correct"):
        y_omiss = (et[omiss_mask] == "deletion").astype(np.int64)
        s_omiss = s[omiss_mask]
        omiss_ap = float(average_precision_binary(y_omiss, s_omiss))
        omiss_auc = float(roc_auc_binary(y_omiss, s_omiss))
    else:
        omiss_ap = float("nan")
        omiss_auc = float("nan")

    # 3. Recall @ FPR <= 15% (diagnostic operating point on omissions)
    correct_scores = s[et == "correct"]
    del_scores = s[et == "deletion"]
    if len(correct_scores) > 0 and len(del_scores) > 0:
        tau_15 = diagnostic_threshold(correct_scores, ceiling=0.15)
        op_15 = operating(y_omiss, s_omiss, tau_15)
        omiss_recall_fpr15 = float(op_15["tpr"])
        omiss_fnr_fpr15 = float(op_15["fnr"])
        actual_fpr15 = float(op_15["fpr"])
    else:
        tau_15 = float("nan")
        omiss_recall_fpr15 = float("nan")
        omiss_fnr_fpr15 = float("nan")
        actual_fpr15 = float("nan")

    # 4. Substitution AP (substitutions vs. correct only)
    sub_mask = (et == "substitution") | (et == "correct")
    if np.any(et == "substitution") and np.any(et == "correct"):
        y_sub = (et[sub_mask] == "substitution").astype(np.int64)
        s_sub = s[sub_mask]
        sub_ap = float(average_precision_binary(y_sub, s_sub))
        sub_auc = float(roc_auc_binary(y_sub, s_sub))
    else:
        sub_ap = float("nan")
        sub_auc = float("nan")

    # 5. Greedy CTC PER
    total_edits = 0
    total_subs = 0
    total_dels = 0
    total_ins = 0
    total_refs = 0
    for ref, hyp in zip(ref_ctc_all, hyp_ctc_all):
        t_edits, s_ed, d_ed, i_ed = compute_levenshtein_edits(ref, hyp)
        total_edits += t_edits
        total_subs += s_ed
        total_dels += d_ed
        total_ins += i_ed
        total_refs += len(ref)

    ctc_per = float(total_edits / total_refs) if total_refs > 0 else float("nan")

    return {
        "primary_metrics": {
            "omission_ap": round(omiss_ap, 6),
            "omission_recall_at_fpr15": round(omiss_recall_fpr15, 6),
            "binary_ap": round(bin_ap, 6),
            "ctc_per": round(ctc_per, 6),
        },
        "secondary_metrics": {
            "binary_roc_auc": round(bin_auc, 6),
            "omission_roc_auc": round(omiss_auc, 6),
            "omission_fnr_at_fpr15": round(omiss_fnr_fpr15, 6),
            "diagnostic_threshold_fpr15": round(tau_15, 6),
            "actual_correct_fpr15": round(actual_fpr15, 6),
            "substitution_ap": round(sub_ap, 6),
            "substitution_roc_auc": round(sub_auc, 6),
            "ctc_sub_rate": round(total_subs / total_refs, 6) if total_refs else None,
            "ctc_del_rate": round(total_dels / total_refs, 6) if total_refs else None,
            "ctc_ins_rate": round(total_ins / total_refs, 6) if total_refs else None,
            "total_ctc_edits": total_edits,
            "total_ctc_ref_tokens": total_refs,
        },
        "counts": {
            "total_tokens": len(y),
            "correct_tokens": int(np.sum(et == "correct")),
            "substitution_tokens": int(np.sum(et == "substitution")),
            "deletion_tokens": int(np.sum(et == "deletion")),
            "total_errors": int(np.sum(y == 1)),
        }
    }


def evaluate_checkpoint(
    ckpt_path: Path | str,
    fold_idx: int,
    device: torch.device,
    manifest_path: Path | str = DEFAULT_MANIFEST,
    max_utterances: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluates a checkpoint on the specified validation fold partition."""
    ckpt_path = Path(ckpt_path)
    ckpt_sha = compute_sha256(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    cfg = ckpt.get("metadata", {}).get("config", {})

    # Extract model architecture configuration
    pretrained_model = cfg.get("pretrained_model", "facebook/wav2vec2-base-960h")
    pretrained_revision = cfg.get("pretrained_revision", "22aad52d435eb6dbaf354bdad9b0da84ce7d6156")
    enable_skip_token = bool(cfg.get("enable_skip_token", False))
    monotonic_bias = bool(cfg.get("monotonic_bias", False))
    monotonic_bandwidth = float(cfg.get("monotonic_bandwidth", 3.0))

    # Detect if binary_head was saved with skip token dimension (897)
    if "binary_head.0.weight" in state_dict:
        in_feat = state_dict["binary_head.0.weight"].shape[1]
        if in_feat == 897:
            enable_skip_token = True

    model_kwargs = {
        "revision": pretrained_revision,
        "ctc_weight": 1.0,
        "binary_weight": 1.0,
        "enable_skip_token": enable_skip_token,
        "monotonic_bias": monotonic_bias,
        "monotonic_bandwidth": monotonic_bandwidth,
    }
    if "binary_pos_weight" in state_dict:
        model_kwargs["binary_pos_weight"] = float(state_dict["binary_pos_weight"].item())

    model = LangfensWav2Vec2MDD.from_pretrained(pretrained_model, **model_kwargs)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch! Missing: {missing}, Unexpected: {unexpected}")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    # Read split manifest and verify test set isolation
    with open(manifest_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    fold_obj = next(f for f in split_data["folds"] if f["fold_index"] == fold_idx)
    val_utts = fold_obj["val"]["utterances"]
    val_spks = set(fold_obj["val"]["speakers"])
    test_spks = set(fold_obj["test"]["speakers"])
    assert not (val_spks & test_spks), "CRITICAL: Validation and test speaker overlap!"

    if max_utterances is not None:
        val_utts = val_utts[:max_utterances]

    zip_path = FOLD_ZIPS[fold_idx]
    reader = get_l2arctic_reader(zip_path)
    prompts = load_arctic_prompts("resources/L2-ARCTIC/prompts.json")
    g2p = CMUDictG2P("resources/CMUDict/cmudict.dict", verify_checksum=True, oov_policy="strict")

    # Evaluation accumulators
    proto_o_labels = []
    proto_o_scores = []
    proto_o_error_types = []
    proto_o_phones = []
    proto_o_skip_weights = []
    ref_ctc_all = []
    hyp_ctc_all = []

    # Protocol Z accumulators
    proto_z_labels = []
    proto_z_scores = []

    # Attention localization accumulators
    in_window_masses = []
    uniform_masses = []
    peak_offsets = []

    token_records = []

    t0 = time.time()
    with torch.inference_mode():
        for u_idx, utt in enumerate(val_utts):
            spk, uid = utt.split("/")
            assert spk not in test_spks, f"Active test speaker {spk} accessed!"

            wav, sr, _ = load_l2arctic_audio_from_zip(zip_path, spk, uid)
            assert sr == 16000

            txt = reader.read_annotation_text(spk, uid)
            ann = parse_gold_textgrid_content(txt, f"{uid}.TextGrid")
            c_ivs = ann.canonical_intervals
            w_pos = extract_word_positions(c_ivs, ann.words_intervals)

            # Protocol O canonical tokens
            can_phones = [iv.canonical_phone for iv in c_ivs]
            can_indices = torch.tensor([[PHONE_TO_CTC_INDEX[p] for p in can_phones]], device=device)
            audio = wav.unsqueeze(0).to(device)

            out_o = model(
                waveforms=audio,
                audio_lengths=torch.tensor([wav.numel()], device=device),
                canonical_phone_indices=can_indices,
                canonical_phone_lengths=torch.tensor([can_indices.shape[1]], device=device)
            )

            b_logits = out_o.binary_logits[0, :len(can_phones)]
            b_scores = torch.sigmoid(b_logits).cpu().numpy().tolist()
            active_frames = int(out_o.input_lengths[0].item())

            # Decode CTC
            ctc_hyp_indices = greedy_ctc_decode(out_o.ctc_logits[0, :active_frames].argmax(dim=-1).cpu().tolist())
            ctc_ref_indices = [
                PHONE_TO_CTC_INDEX[iv.realized_phone]
                for iv in ann.intervals
                if iv.error_type not in ("silence", "deletion") and not iv.has_err_tag and iv.realized_phone in PHONE_TO_CTC_INDEX
            ]
            ref_ctc_all.append(ctc_ref_indices)
            hyp_ctc_all.append(ctc_hyp_indices)

            # Separate skip token from acoustic frames if enabled
            raw_attn = out_o.alignment_weights[0, :len(can_phones)].cpu().numpy()
            if enable_skip_token:
                # Index 0 is skip token; indices 1..1+active_frames are acoustic frames
                skip_weights_row = raw_attn[:, 0]
                acoustic_attn_w = raw_attn[:, 1:1 + active_frames]
            else:
                skip_weights_row = np.zeros(len(can_phones), dtype=np.float32)
                acoustic_attn_w = raw_attn[:, :active_frames]

            # Protocol Z
            z_phones = g2p.phonemize(prompts[uid])
            z_indices = torch.tensor([[PHONE_TO_CTC_INDEX[p] for p in z_phones]], device=device)
            out_z = model(
                waveforms=audio,
                audio_lengths=torch.tensor([wav.numel()], device=device),
                canonical_phone_indices=z_indices,
                canonical_phone_lengths=torch.tensor([z_indices.shape[1]], device=device)
            )
            z_scores = torch.sigmoid(out_z.binary_logits[0, :len(z_phones)]).cpu().numpy().tolist()
            z_alignment = align_canonical_sequences(can_phones, z_phones, tie_break="sub_first")
            for gold_i, z_i, op in z_alignment["aligned_pairs"]:
                proto_z_labels.append(c_ivs[gold_i].binary_label_strict)
                proto_z_scores.append(z_scores[z_i])

            # Accumulate Protocol O token metrics
            for idx, iv in enumerate(c_ivs):
                lbl = iv.binary_label_strict
                et = iv.error_type
                sc = b_scores[idx]
                sw = float(skip_weights_row[idx])

                proto_o_labels.append(lbl)
                proto_o_scores.append(sc)
                proto_o_error_types.append(et)
                proto_o_phones.append(iv.canonical_phone)
                proto_o_skip_weights.append(sw)

                # Attention window metrics on ACOUSTIC frames (excluding skip token)
                start_frame = min(active_frames - 1, max(0, int(iv.xmin / 0.020)))
                end_frame = min(active_frames, max(start_frame + 1, int(math.ceil(iv.xmax / 0.020))))
                ac_row = acoustic_attn_w[idx]
                ac_sum = float(ac_row.sum())
                abs_in_win = float(ac_row[start_frame:end_frame].sum())
                cond_in_win = float(abs_in_win / max(ac_sum, 1e-12))
                unif_mass = float((end_frame - start_frame) / max(active_frames, 1))
                peak_f = int(np.argmax(ac_row))
                peak_time = peak_f * 0.020
                center_time = (iv.xmin + iv.xmax) / 2.0
                offset = abs(peak_time - center_time)

                in_window_masses.append(abs_in_win)
                uniform_masses.append(unif_mass)
                peak_offsets.append(offset)

                token_records.append({
                    "fold": fold_idx,
                    "speaker": spk,
                    "l1": DEMOGRAPHICS[spk]["l1"],
                    "utterance_id": uid,
                    "token_idx": idx,
                    "phone": iv.canonical_phone,
                    "error_type": et,
                    "binary_label": lbl,
                    "binary_score": sc,
                    "skip_weight": sw,
                    "word_position": w_pos.get(idx, "unknown"),
                    "duration_sec": iv.xmax - iv.xmin,
                    "in_window_mass": abs_in_win,
                    "conditional_in_window_mass": cond_in_win,
                    "uniform_mass": unif_mass,
                    "peak_offset_sec": offset,
                })

    elapsed = time.time() - t0

    # Compute overall primary & secondary metrics
    o_metrics = compute_primary_metrics(
        proto_o_labels, proto_o_scores, proto_o_error_types, ref_ctc_all, hyp_ctc_all
    )

    # Compute Protocol Z metrics
    z_y = np.asarray(proto_z_labels, dtype=np.int64)
    z_s = np.asarray(proto_z_scores, dtype=np.float64)
    proto_z_ap = float(average_precision_binary(z_y, z_s)) if len(z_y) > 0 else float("nan")
    proto_z_auc = float(roc_auc_binary(z_y, z_s)) if len(z_y) > 0 else float("nan")

    cond_masses = [r["conditional_in_window_mass"] for r in token_records]
    del_skip_masses = [r["skip_weight"] for r in token_records if r["error_type"] == "deletion" and not math.isnan(r["skip_weight"])]
    cor_skip_masses = [r["skip_weight"] for r in token_records if r["error_type"] == "correct" and not math.isnan(r["skip_weight"])]
    all_skip_masses = [r["skip_weight"] for r in token_records if not math.isnan(r["skip_weight"])]

    attention_summary = {
        "mean_absolute_in_window_mass": round(float(np.mean(in_window_masses)), 6) if in_window_masses else None,
        "mean_conditional_in_window_mass": round(float(np.mean(cond_masses)), 6) if cond_masses else None,
        "mean_uniform_mass": round(float(np.mean(uniform_masses)), 6) if uniform_masses else None,
        "mean_peak_offset_seconds": round(float(np.mean(peak_offsets)), 6) if peak_offsets else None,
        "mean_skip_mass_all": round(float(np.mean(all_skip_masses)), 6) if all_skip_masses else None,
        "mean_skip_mass_deletions": round(float(np.mean(del_skip_masses)), 6) if del_skip_masses else None,
        "mean_skip_mass_correct": round(float(np.mean(cor_skip_masses)), 6) if cor_skip_masses else None,
    }

    # Disaggregated breakdowns by L1, speaker, position, phoneme
    by_l1 = {}
    for l1 in sorted({r["l1"] for r in token_records}):
        l1_recs = [r for r in token_records if r["l1"] == l1]
        l1_y = [r["binary_label"] for r in l1_recs]
        l1_s = [r["binary_score"] for r in l1_recs]
        l1_et = [r["error_type"] for r in l1_recs]
        l1_omiss = (np.asarray(l1_et) == "deletion") | (np.asarray(l1_et) == "correct")
        by_l1[l1] = {
            "tokens": len(l1_recs),
            "errors": int(sum(l1_y)),
            "deletions": int(sum(r["error_type"] == "deletion" for r in l1_recs)),
            "binary_ap": round(float(average_precision_binary(np.asarray(l1_y), np.asarray(l1_s))), 6),
            "omission_ap": round(float(average_precision_binary((np.asarray(l1_et)[l1_omiss] == "deletion").astype(int), np.asarray(l1_s)[l1_omiss])), 6) if sum(r["error_type"] == "deletion" for r in l1_recs) else None,
        }

    return {
        "provenance": {
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": ckpt_sha,
            "fold": fold_idx,
            "experiment": cfg.get("experiment", "unknown"),
            "model_arm": cfg.get("model_arm", "Baseline_R" if not enable_skip_token else "E1"),
            "enable_skip_token": enable_skip_token,
            "monotonic_bias": monotonic_bias,
            "evaluation_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "eval_duration_seconds": round(elapsed, 2),
            "utterances_evaluated": len(val_utts),
        },
        "primary_registered_metrics": o_metrics["primary_metrics"],
        "secondary_metrics": o_metrics["secondary_metrics"],
        "counts": o_metrics["counts"],
        "protocol_z": {
            "aligned_tokens": len(proto_z_labels),
            "coverage": round(len(proto_z_labels) / max(len(proto_o_labels), 1), 6),
            "aligned_ap": round(proto_z_ap, 6),
            "aligned_roc_auc": round(proto_z_auc, 6),
        },
        "attention_localization": attention_summary,
        "disaggregated_by_l1": by_l1,
        "token_records": token_records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to model checkpoint (.pt)")
    parser.add_argument("--fold", required=True, type=int, choices=(1, 2, 3, 4), help="Validation fold index")
    parser.add_argument("--device", default="cpu", help="Device to use ('cpu' or 'cuda')")
    parser.add_argument("--outdir", required=True, type=Path, help="New output directory for reports")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to split manifest")
    parser.add_argument("--max-utterances", type=int, default=None, help="Optional evaluation subset cap")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    if outdir.exists():
        parser.error(f"Output directory '{outdir}' already exists! Choose an independent new path.")
    outdir.mkdir(parents=True, exist_ok=False)

    device = torch.device(args.device)
    report = evaluate_checkpoint(
        ckpt_path=args.checkpoint,
        fold_idx=args.fold,
        device=device,
        manifest_path=args.manifest,
        max_utterances=args.max_utterances,
    )

    records = report.pop("token_records")

    report_path = outdir / "E1_EVALUATION_REPORT.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save summary CSV
    summary_path = outdir / "primary_metrics_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in report["primary_registered_metrics"].items():
            writer.writerow([k, v])
        for k, v in report["secondary_metrics"].items():
            writer.writerow([k, v])

    print("\n" + "=" * 70)
    print(f"EVALUATION COMPLETE: {report['provenance']['model_arm']} (Fold {args.fold})")
    print("=" * 70)
    for k, v in report["primary_registered_metrics"].items():
        print(f"  {k:26s}: {v}")
    print(f"  {'Binary ROC-AUC':26s}: {report['secondary_metrics']['binary_roc_auc']}")
    print(f"  {'Protocol Z Aligned AP':26s}: {report['protocol_z']['aligned_ap']}")
    print(f"  {'Mean Absolute In-Win Mass':26s}: {report['attention_localization']['mean_absolute_in_window_mass']}")
    print(f"  {'Mean Cond In-Win Mass':26s}: {report['attention_localization']['mean_conditional_in_window_mass']}")
    print(f"  {'Mean Deletion Skip Mass':26s}: {report['attention_localization']['mean_skip_mass_deletions']}")
    print(f"  {'Mean Peak Temporal Offset':26s}: {report['attention_localization']['mean_peak_offset_seconds']} s")
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
