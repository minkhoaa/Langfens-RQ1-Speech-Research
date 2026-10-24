#!/usr/bin/env python3
"""
LANGFENS RQ1-TRN-01B1-R: Comprehensive Controlled Generalization, CTC Recovery,
Acoustic Dependence & Multi-Seed Verification Campaign.

Integrates Tasks D, E, F, G:
  1. Protocol-driven execution using INTERNAL_TRAIN_HOLDOUT_PROTOCOL.json:
     - internal_train_probe_256 (14 train speakers, 256 utts)
     - internal_holdout_probe_64 (2 unseen holdout speakers: SVBI, ZHAA, 64 utts)
  2. Multi-seed stability: Seed 1337 and Seed 42
  3. CTC Length Recovery & Decoding Dynamics:
     - Decoded length ratio (L_hyp / L_ref)
     - Framewise blank probability distribution
     - Decomposition of edits: Substitutions, Deletions, Insertions, PER
     - Evaluation of decoding blank penalties beta in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
     - Prefix beam search decoding
  4. Acoustic Dependence Audit (Task E):
     - Matched audio + canonical prompt
     - Mismatched audio + canonical prompt
     - Silence / zero audio control
     - Shuffled acoustic frame control
     - Canonical-prior baseline
  5. Per-speaker and per-class accounting across all 39 ARPAbet classes.
  6. Checkpoint continuity and exact MiB resource accounting.
"""

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup

from langfens_prep.model import (
    LangfensWav2Vec2MDD,
    DEFAULT_PRETRAINED_CHECKPOINT,
    DEFAULT_PRETRAINED_REVISION
)
from langfens_prep.collate import LangfensMDDCollate
from langfens_prep.dataset import get_l2arctic_reader, load_l2arctic_audio_from_zip, parse_gold_textgrid_content
from langfens_prep.phoneme_normalizer import STANDARD_39_ARPABET, PHONE_TO_INDEX, PHONE_TO_CTC_INDEX, INDEX_TO_PHONE
from audit_scripts.run_training_sanity import compute_pr_auc


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def set_strict_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ProtocolDataset(Dataset):
    """Dataset driven strictly by the vetted internal protocol manifest."""
    def __init__(
        self,
        subset_name: str,
        protocol_path: str = "manifests/RQ1-TRN-01B1/reconciliation/INTERNAL_TRAIN_HOLDOUT_PROTOCOL.json",
        zip_path: str = "datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip",
        target_sr: int = 16000
    ):
        with open(protocol_path, "r", encoding="utf-8") as f:
            self.proto = json.load(f)

        assert subset_name in self.proto["subsets"], f"Unknown subset {subset_name}"
        self.subset_info = self.proto["subsets"][subset_name]
        self.entries = self.subset_info["selected_utterances"]
        self.zip_path = zip_path
        self.target_sr = target_sr
        self.reader = get_l2arctic_reader(zip_path)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        e = self.entries[idx]
        spk = e["speaker_id"]
        utt = e["utterance_id"]

        waveform, sr, _ = load_l2arctic_audio_from_zip(
            self.zip_path, spk, utt, self.target_sr
        )
        tg_text = self.reader.read_annotation_text(spk, utt)
        anno = parse_gold_textgrid_content(tg_text, f"{utt}.TextGrid")

        can_intervals = anno.canonical_intervals
        can_phones = [i.canonical_phone for i in can_intervals]
        can_idx = torch.tensor(
            [PHONE_TO_CTC_INDEX[p] for p in can_phones if p in PHONE_TO_CTC_INDEX],
            dtype=torch.long
        )

        realized = []
        for i in anno.intervals:
            if i.error_type not in ("silence", "deletion") and not i.has_err_tag:
                p = i.realized_phone
                if p and p in STANDARD_39_ARPABET:
                    realized.append(p)

        ctc_idx = torch.tensor(
            [PHONE_TO_CTC_INDEX[p] for p in realized],
            dtype=torch.long
        )
        bin_vals = [i.binary_label_strict for i in can_intervals]
        diag_idx = [
            PHONE_TO_INDEX[i.diagnosis_target] if (i.diagnosis_target is not None and i.diagnosis_target in PHONE_TO_INDEX)
            else -100
            for i in can_intervals
        ]
        aux_idx = [
            PHONE_TO_INDEX[i.realized_phone] if (i.error_type in ("correct", "substitution") and not i.has_err_tag and i.realized_phone in PHONE_TO_INDEX)
            else -100
            for i in can_intervals
        ]

        return {
            "record_id": f"l2arctic_{spk}_{utt}",
            "utterance_id": utt,
            "speaker_id": spk,
            "audio_waveform_16k": waveform,
            "canonical_phones_norm39": can_phones,
            "canonical_phone_indices": can_idx,
            "realized_phones_norm39": realized,
            "chronological_realized_phones": realized,
            "binary_labels": torch.tensor(bin_vals, dtype=torch.long),
            "ctc_target_indices": ctc_idx,
            "diagnosis_target_indices": torch.tensor(diag_idx, dtype=torch.long),
            "aux_phone_target_indices": torch.tensor(aux_idx, dtype=torch.long),
            "is_blinded": False,
            "policy": "strict",
            "duration_seconds": e["duration_seconds"]
        }


def compute_levenshtein_edits(ref: List[int], hyp: List[int]) -> Tuple[int, int, int, int]:
    n, m = len(ref), len(hyp)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost # substitution
            )

    i, j = n, m
    subs, dels, inss = 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            inss += 1
            j -= 1
        else:
            break

    return int(dp[n][m]), subs, dels, inss


def ctc_decode_with_penalty(
    log_probs: torch.Tensor,
    blank_id: int = 0,
    blank_penalty: float = 0.0
) -> List[int]:
    """Greedy CTC collapse with configurable blank logit penalty beta."""
    lp = log_probs.clone()
    if blank_penalty != 0.0:
        lp[:, blank_id] -= blank_penalty

    tokens = lp.argmax(dim=-1).tolist()
    collapsed = []
    prev = None
    for t in tokens:
        if t != prev:
            if t != blank_id:
                collapsed.append(t)
            prev = t
    return collapsed


def ctc_prefix_beam_search(
    log_probs: torch.Tensor,
    beam_width: int = 10,
    blank_id: int = 0
) -> List[int]:
    """Standard prefix beam search decoding for CTC (without language model or gold constraints)."""
    # log_probs: (T, V)
    T, V = log_probs.shape
    # Beams: mapping from prefix tuple to (p_blank, p_non_blank) in log space
    beams = {(): (0.0, -float('inf'))}

    for t in range(T):
        new_beams = defaultdict(lambda: (-float('inf'), -float('inf')))
        frame_lp = log_probs[t]

        for prefix, (pb, pnb) in beams.items():
            p_total = np.logaddexp(pb, pnb)

            # 1. Blank emission
            lp_blank = frame_lp[blank_id].item()
            new_pb, new_pnb = new_beams[prefix]
            new_pb = np.logaddexp(new_pb, p_total + lp_blank)
            new_beams[prefix] = (new_pb, new_pnb)

            # 2. Non-blank emissions
            for c in range(V):
                if c == blank_id:
                    continue
                lp_c = frame_lp[c].item()
                end_token = prefix[-1] if len(prefix) > 0 else None

                if c == end_token:
                    # Same token: can only extend if previous was blank
                    new_prefix = prefix + (c,)
                    nb_pb, nb_pnb = new_beams[new_prefix]
                    nb_pnb = np.logaddexp(nb_pnb, pb + lp_c)
                    new_beams[new_prefix] = (nb_pb, nb_pnb)

                    # Or collapse identical without extending
                    s_pb, s_pnb = new_beams[prefix]
                    s_pnb = np.logaddexp(s_pnb, pnb + lp_c)
                    new_beams[prefix] = (s_pb, s_pnb)
                else:
                    new_prefix = prefix + (c,)
                    nb_pb, nb_pnb = new_beams[new_prefix]
                    nb_pnb = np.logaddexp(nb_pnb, p_total + lp_c)
                    new_beams[new_prefix] = (nb_pb, nb_pnb)

        # Prune to beam_width
        sorted_prefixes = sorted(
            new_beams.keys(),
            key=lambda p: np.logaddexp(new_beams[p][0], new_beams[p][1]),
            reverse=True
        )[:beam_width]

        beams = {p: new_beams[p] for p in sorted_prefixes}

    best_prefix = max(beams.keys(), key=lambda p: np.logaddexp(beams[p][0], beams[p][1]))
    return list(best_prefix)


@torch.no_grad()
def evaluate_protocol_subset(
    model: torch.nn.Module,
    dataset: Dataset,
    device: str = "cuda",
    blank_penalties: List[float] = [0.0, 1.0, 2.0, 3.0, 4.0],
    run_beam_search: bool = True
) -> Dict[str, Any]:
    model.eval()
    collate_fn = LangfensMDDCollate(pad_token_id=-100, ctc_blank_id=0)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    tot_loss = 0.0
    tot_ctc = 0.0
    tot_bin = 0.0
    tot_diag = 0.0
    num_batches = 0

    all_bin_scores = []
    all_bin_targets = []
    all_bin_preds = []

    sub_diag_correct = 0
    sub_diag_total = 0
    sub_cm = np.zeros((39, 39), dtype=int)
    sub_gold_counts = np.zeros(39, dtype=int)
    sub_pred_counts = np.zeros(39, dtype=int)
    sub_corr_counts = np.zeros(39, dtype=int)

    # Per-speaker accounting
    spk_stats = defaultdict(lambda: {
        "utts": 0, "ref_tokens": 0, "hyp_tokens": 0, "edits": 0, "subs": 0, "dels": 0, "inss": 0,
        "sub_diag_corr": 0, "sub_diag_tot": 0, "bin_tp": 0, "bin_fp": 0, "bin_fn": 0, "bin_tn": 0
    })

    # CTC metrics tracking
    active_frames = 0
    blank_frames = 0
    blank_probs_list = []
    non_blank_max_probs = []

    ctc_edits = {b: 0 for b in blank_penalties}
    ctc_subs = {b: 0 for b in blank_penalties}
    ctc_dels = {b: 0 for b in blank_penalties}
    ctc_inss = {b: 0 for b in blank_penalties}
    ctc_hyps = {b: 0 for b in blank_penalties}
    ctc_ref_total = 0

    beam_edits = 0
    beam_subs = 0
    beam_dels = 0
    beam_inss = 0
    beam_hyps = 0

    for batch in loader:
        net_in = {k: v.to(device) for k, v in batch["net_input"].items()}
        tb = batch["target_binary"].to(device)
        tc = batch["target_ctc"].to(device)
        tl = batch["target_lengths"].to(device)
        td = batch["target_diagnosis"].to(device)

        with torch.amp.autocast(device_type=device, dtype=torch.float16):
            out = model(**net_in, target_binary=tb, target_ctc=tc, target_lengths=tl, target_diagnosis=td)

        tot_loss += out.loss.item()
        if out.loss_dict is not None:
            tot_ctc += out.loss_dict.get("ctc_loss", torch.tensor(0.0)).item()
            tot_bin += out.loss_dict.get("binary_loss", torch.tensor(0.0)).item()
            tot_diag += out.loss_dict.get("diagnosis_loss", torch.tensor(0.0)).item()
        num_batches += 1

        # 1. Binary MDD
        b_logits = out.binary_logits
        b_mask = (tb != -100)
        if b_mask.any():
            probs = torch.sigmoid(b_logits[b_mask]).float().cpu().numpy()
            tgts = tb[b_mask].cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_bin_scores.extend(probs.tolist())
            all_bin_targets.extend(tgts.tolist())
            all_bin_preds.extend(preds.tolist())

        # 2. Multi-class substitution diagnosis
        d_logits = out.diagnosis_logits
        d_mask = (td != -100)
        if d_mask.any():
            preds_39 = d_logits[d_mask].argmax(dim=-1).cpu().numpy()
            gold_39 = td[d_mask].cpu().numpy()
            sub_diag_correct += int((preds_39 == gold_39).sum())
            sub_diag_total += len(gold_39)
            for g, p in zip(gold_39, preds_39):
                sub_cm[g, p] += 1
                sub_gold_counts[g] += 1
                sub_pred_counts[p] += 1
                if g == p:
                    sub_corr_counts[g] += 1

        # Per-speaker breakdown within batch
        spk_ids = [rid.split("_")[1] for rid in batch["record_ids"]]
        for i_b, spk in enumerate(spk_ids):
            spk_stats[spk]["utts"] += 1
            # Sub diag for this speaker
            u_td = td[i_b]
            u_dmask = (u_td != -100)
            if u_dmask.any():
                u_preds = d_logits[i_b, u_dmask].argmax(dim=-1).cpu().numpy()
                u_gold = u_td[u_dmask].cpu().numpy()
                spk_stats[spk]["sub_diag_corr"] += int((u_preds == u_gold).sum())
                spk_stats[spk]["sub_diag_tot"] += len(u_gold)

            # Binary for this speaker
            u_tb = tb[i_b]
            u_bmask = (u_tb != -100)
            if u_bmask.any():
                u_probs = torch.sigmoid(b_logits[i_b, u_bmask]).float().cpu().numpy()
                u_tgts = u_tb[u_bmask].cpu().numpy()
                u_preds = (u_probs >= 0.5).astype(int)
                spk_stats[spk]["bin_tp"] += int(((u_preds == 1) & (u_tgts == 1)).sum())
                spk_stats[spk]["bin_fp"] += int(((u_preds == 1) & (u_tgts == 0)).sum())
                spk_stats[spk]["bin_fn"] += int(((u_preds == 0) & (u_tgts == 1)).sum())
                spk_stats[spk]["bin_tn"] += int(((u_preds == 0) & (u_tgts == 0)).sum())

        # 3. CTC Detailed Framing & Decoding
        c_logits = out.ctc_logits.float()
        c_lprobs = F.log_softmax(c_logits, dim=-1)
        c_probs = F.softmax(c_logits, dim=-1)

        for i_b, spk in enumerate(spk_ids):
            in_len = out.input_lengths[i_b].item()
            active_lp = c_lprobs[i_b, :in_len]
            active_p = c_probs[i_b, :in_len]

            # Framing stats
            b_probs = active_p[:, 0].cpu().numpy()
            blank_probs_list.extend(b_probs.tolist())
            nb_max = active_p[:, 1:].max(dim=-1).values.cpu().numpy()
            non_blank_max_probs.extend(nb_max.tolist())

            raw_argmax = active_lp.argmax(dim=-1).cpu().numpy()
            active_frames += in_len
            blank_frames += int(np.sum(raw_argmax == 0))

            ref_len = tl[i_b].item()
            ref_phones = tc[i_b, :ref_len].cpu().tolist()
            ctc_ref_total += ref_len
            spk_stats[spk]["ref_tokens"] += ref_len

            # Test different blank penalties
            for beta in blank_penalties:
                hyp_phones = ctc_decode_with_penalty(active_lp, blank_id=0, blank_penalty=beta)
                ctc_hyps[beta] += len(hyp_phones)
                edits, subs, dels, inss = compute_levenshtein_edits(ref_phones, hyp_phones)
                ctc_edits[beta] += edits
                ctc_subs[beta] += subs
                ctc_dels[beta] += dels
                ctc_inss[beta] += inss

                if beta == 0.0:
                    spk_stats[spk]["hyp_tokens"] += len(hyp_phones)
                    spk_stats[spk]["edits"] += edits
                    spk_stats[spk]["subs"] += subs
                    spk_stats[spk]["dels"] += dels
                    spk_stats[spk]["inss"] += inss

            # Prefix beam search
            if run_beam_search and i_b < 5:  # Bounded execution on representative subset
                b_hyp = ctc_prefix_beam_search(active_lp, beam_width=8, blank_id=0)
                beam_hyps += len(b_hyp)
                b_edits, b_subs, b_dels, b_inss = compute_levenshtein_edits(ref_phones, b_hyp)
                beam_edits += b_edits
                beam_subs += b_subs
                beam_dels += b_dels
                beam_inss += b_inss

    avg_loss = tot_loss / max(1, num_batches)
    avg_ctc = tot_ctc / max(1, num_batches)
    avg_bin = tot_bin / max(1, num_batches)
    avg_diag = tot_diag / max(1, num_batches)

    # Binary metrics
    bin_targets_np = np.array(all_bin_targets)
    bin_scores_np = np.array(all_bin_scores)
    bin_preds_np = np.array(all_bin_preds)

    tp_05 = int(np.sum((bin_preds_np == 1) & (bin_targets_np == 1)))
    fp_05 = int(np.sum((bin_preds_np == 1) & (bin_targets_np == 0)))
    fn_05 = int(np.sum((bin_preds_np == 0) & (bin_targets_np == 1)))
    tn_05 = int(np.sum((bin_preds_np == 0) & (bin_targets_np == 0)))

    prec_05 = float(tp_05 / (tp_05 + fp_05)) if (tp_05 + fp_05) > 0 else 0.0
    rec_05 = float(tp_05 / (tp_05 + fn_05)) if (tp_05 + fn_05) > 0 else 0.0
    f1_05 = float(2 * prec_05 * rec_05 / (prec_05 + rec_05)) if (prec_05 + rec_05) > 0 else 0.0
    acc_05 = float((tp_05 + tn_05) / len(bin_targets_np)) if len(bin_targets_np) > 0 else 0.0
    pr_auc = compute_pr_auc(bin_targets_np, bin_scores_np)

    # Dynamic baselines & assertions for diagnosis
    max_gold = int(np.max(sub_gold_counts)) if sub_diag_total > 0 else 0
    maj_idx = int(np.argmax(sub_gold_counts)) if sub_diag_total > 0 else 0
    maj_phone = STANDARD_39_ARPABET[maj_idx] if sub_diag_total > 0 else None
    maj_baseline = round(max_gold / sub_diag_total, 4) if sub_diag_total > 0 else 0.0
    chance_baseline = round(1.0 / len(STANDARD_39_ARPABET), 4)

    if sub_diag_total > 0:
        assert int(np.sum(sub_gold_counts)) == sub_diag_total
        assert int(np.sum(sub_cm)) == sub_diag_total

    classes_with_support = int(np.sum(sub_gold_counts > 0))
    classes_predicted = int(np.sum(sub_pred_counts > 0))
    classes_correct = int(np.sum(sub_corr_counts > 0))

    per_class_summary = {}
    for idx, phone in enumerate(STANDARD_39_ARPABET):
        gc = int(sub_gold_counts[idx])
        pc = int(sub_pred_counts[idx])
        cc = int(sub_corr_counts[idx])
        per_class_summary[phone] = {
            "gold_support": gc,
            "predicted_count": pc,
            "correct_count": cc,
            "recall": round(cc / gc, 4) if gc > 0 else None,
            "precision": round(cc / pc, 4) if pc > 0 else None
        }

    # Per-speaker processed metrics
    per_speaker_results = {}
    for spk, sdata in sorted(spk_stats.items()):
        s_per = sdata["edits"] / sdata["ref_tokens"] if sdata["ref_tokens"] > 0 else 1.0
        s_diag_acc = sdata["sub_diag_corr"] / sdata["sub_diag_tot"] if sdata["sub_diag_tot"] > 0 else 0.0
        s_tp, s_fp, s_fn = sdata["bin_tp"], sdata["bin_fp"], sdata["bin_fn"]
        s_p = s_tp / (s_tp + s_fp) if (s_tp + s_fp) > 0 else 0.0
        s_r = s_tp / (s_tp + s_fn) if (s_tp + s_fn) > 0 else 0.0
        s_f1 = (2 * s_p * s_r) / (s_p + s_r) if (s_p + s_r) > 0 else 0.0
        per_speaker_results[spk] = {
            "utterances": sdata["utts"],
            "ref_tokens": sdata["ref_tokens"],
            "hyp_tokens": sdata["hyp_tokens"],
            "length_ratio": round(sdata["hyp_tokens"] / max(1, sdata["ref_tokens"]), 4),
            "per": round(s_per, 4),
            "sub_diag_accuracy": round(s_diag_acc, 4),
            "sub_diag_correct": sdata["sub_diag_corr"],
            "sub_diag_total": sdata["sub_diag_tot"],
            "binary_f1_05": round(s_f1, 4),
            "binary_precision_05": round(s_p, 4),
            "binary_recall_05": round(s_r, 4)
        }

    # Framing distribution stats
    def pstats(arr):
        if len(arr) == 0:
            return {}
        a = np.array(arr)
        return {
            "min": round(float(np.min(a)), 4),
            "p25": round(float(np.percentile(a, 25)), 4),
            "median": round(float(np.median(a)), 4),
            "mean": round(float(np.mean(a)), 4),
            "p75": round(float(np.percentile(a, 75)), 4),
            "max": round(float(np.max(a)), 4)
        }

    # CTC penalties evaluation
    penalty_eval = {}
    for beta in blank_penalties:
        ref_tot = ctc_ref_total
        hyp_tot = ctc_hyps[beta]
        p_per = ctc_edits[beta] / ref_tot if ref_tot > 0 else 1.0
        penalty_eval[f"beta_{beta:.1f}"] = {
            "blank_penalty": beta,
            "hyp_tokens": hyp_tot,
            "ref_tokens": ref_tot,
            "length_ratio": round(hyp_tot / max(1, ref_tot), 4),
            "edits": ctc_edits[beta],
            "substitutions": ctc_subs[beta],
            "deletions": ctc_dels[beta],
            "insertions": ctc_inss[beta],
            "phone_error_rate": round(p_per, 4)
        }

    # Primary CTC metrics at beta=0.0 (greedy)
    greedy_hyp = ctc_hyps[0.0]
    greedy_per = ctc_edits[0.0] / ctc_ref_total if ctc_ref_total > 0 else 1.0

    return {
        "loss": round(avg_loss, 4),
        "ctc_loss": round(avg_ctc, 4),
        "binary_loss": round(avg_bin, 4),
        "diagnosis_loss": round(avg_diag, 4),
        "substitution_diagnosis": {
            "top1_accuracy": round(sub_diag_correct / sub_diag_total, 4) if sub_diag_total > 0 else 0.0,
            "correct_predictions": sub_diag_correct,
            "eligible_denominator": sub_diag_total,
            "uniform_chance_baseline": chance_baseline,
            "majority_class_baseline": maj_baseline,
            "majority_class_phone": maj_phone,
            "majority_class_count": max_gold,
            "classes_with_gold_support": classes_with_support,
            "classes_predicted": classes_predicted,
            "classes_with_correct_hits": classes_correct,
            "per_class_summary": per_class_summary,
            "confusion_matrix_39x39": sub_cm.tolist()
        },
        "binary_mdd": {
            "default_05": {
                "accuracy": round(acc_05, 4),
                "precision": round(prec_05, 4),
                "recall": round(rec_05, 4),
                "f1": round(f1_05, 4),
                "confusion_matrix": {"tp": tp_05, "fp": fp_05, "fn": fn_05, "tn": tn_05}
            },
            "train_only_pr_auc": round(pr_auc, 4)
        },
        "ctc_framing_and_length": {
            "active_frames": active_frames,
            "blank_frames": blank_frames,
            "blank_fraction": round(blank_frames / max(1, active_frames), 4),
            "ref_tokens": ctc_ref_total,
            "hyp_tokens_greedy": greedy_hyp,
            "length_ratio_greedy": round(greedy_hyp / max(1, ctc_ref_total), 4),
            "phone_error_rate_greedy": round(greedy_per, 4),
            "edits_greedy": ctc_edits[0.0],
            "subs_greedy": ctc_subs[0.0],
            "dels_greedy": ctc_dels[0.0],
            "inss_greedy": ctc_inss[0.0],
            "blank_probability_distribution": pstats(blank_probs_list),
            "non_blank_max_probability_distribution": pstats(non_blank_max_probs),
            "blank_penalty_sweep": penalty_eval
        },
        "per_speaker_breakdown": per_speaker_results
    }


@torch.no_grad()
def evaluate_acoustic_dependence_perturbations(
    model: torch.nn.Module,
    dataset: ProtocolDataset,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Task E: Controlled Evaluation-Only Perturbations
    Tests whether substitution diagnosis accuracy and binary MDD scores represent genuine acoustic
    dependence or memorization of canonical prompts.
    """
    model.eval()
    collate_fn = LangfensMDDCollate(pad_token_id=-100, ctc_blank_id=0)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    batches = list(loader)

    perturbation_modes = ["matched_baseline", "mismatched_audio", "silence_control", "shuffled_frames"]
    results = {}

    for mode in perturbation_modes:
        sub_correct = 0
        sub_total = 0
        bin_probs_all = []
        bin_targets_all = []
        ctc_hyp_lens = []

        for b_idx, batch in enumerate(batches):
            net_in = {k: v.to(device) for k, v in batch["net_input"].items()}
            tb = batch["target_binary"].to(device)
            tc = batch["target_ctc"].to(device)
            tl = batch["target_lengths"].to(device)
            td = batch["target_diagnosis"].to(device)

            if mode == "matched_baseline":
                pass  # Clean evaluation
            elif mode == "mismatched_audio":
                # Cycle audio waveform across batch elements (shift by +1)
                other_idx = (b_idx + 1) % len(batches)
                other_batch = batches[other_idx]
                net_in["source"] = other_batch["net_input"]["source"].to(device)
                if "padding_mask" in other_batch["net_input"] and other_batch["net_input"]["padding_mask"] is not None:
                    net_in["padding_mask"] = other_batch["net_input"]["padding_mask"].to(device)
                if "audio_lengths" in other_batch["net_input"] and other_batch["net_input"]["audio_lengths"] is not None:
                    net_in["audio_lengths"] = other_batch["net_input"]["audio_lengths"].to(device)
            elif mode == "silence_control":
                # Zero out audio waveform
                net_in["source"] = torch.zeros_like(net_in["source"])
            elif mode == "shuffled_frames":
                # Audio exists, but we shuffle the waveform chunks or zero the input lengths
                # Permute along time dimension
                perm = torch.randperm(net_in["source"].shape[1], device=device)
                net_in["source"] = net_in["source"][:, perm]

            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                out = model(**net_in, target_binary=tb, target_ctc=tc, target_lengths=tl, target_diagnosis=td)

            # Diagnosis accuracy
            d_logits = out.diagnosis_logits
            d_mask = (td != -100)
            if d_mask.any():
                preds = d_logits[d_mask].argmax(dim=-1).cpu().numpy()
                gold = td[d_mask].cpu().numpy()
                sub_correct += int((preds == gold).sum())
                sub_total += len(gold)

            # Binary scores
            b_logits = out.binary_logits
            b_mask = (tb != -100)
            if b_mask.any():
                probs = torch.sigmoid(b_logits[b_mask]).float().cpu().numpy()
                tgts = tb[b_mask].cpu().numpy()
                bin_probs_all.extend(probs.tolist())
                bin_targets_all.extend(tgts.tolist())

            # CTC hyp length
            c_logits = out.ctc_logits.float()
            c_lprobs = F.log_softmax(c_logits, dim=-1)
            for i in range(c_logits.shape[0]):
                in_len = out.input_lengths[i].item()
                active_lp = c_lprobs[i, :in_len]
                h = ctc_decode_with_penalty(active_lp, blank_id=0, blank_penalty=0.0)
                ctc_hyp_lens.append(len(h))

        acc = float(sub_correct / sub_total) if sub_total > 0 else 0.0
        bin_mean = float(np.mean(bin_probs_all)) if len(bin_probs_all) > 0 else 0.0
        bin_auc = compute_pr_auc(np.array(bin_targets_all), np.array(bin_probs_all))

        results[mode] = {
            "mode": mode,
            "substitution_diagnosis_accuracy": round(acc, 4),
            "substitution_correct": sub_correct,
            "substitution_total": sub_total,
            "binary_mean_error_score": round(bin_mean, 4),
            "binary_pr_auc": round(bin_auc, 4),
            "mean_hyp_length": round(float(np.mean(ctc_hyp_lens)), 2) if len(ctc_hyp_lens) > 0 else 0.0
        }

    # Calculate sensitivity drops relative to matched baseline
    base_acc = results["matched_baseline"]["substitution_diagnosis_accuracy"]
    base_auc = results["matched_baseline"]["binary_pr_auc"]

    sensitivity_report = {
        "matched_baseline_accuracy": base_acc,
        "mismatched_accuracy_drop": round(base_acc - results["mismatched_audio"]["substitution_diagnosis_accuracy"], 4),
        "silence_accuracy_drop": round(base_acc - results["silence_control"]["substitution_diagnosis_accuracy"], 4),
        "shuffled_accuracy_drop": round(base_acc - results["shuffled_frames"]["substitution_diagnosis_accuracy"], 4),
        "has_acoustic_dependence": (
            results["mismatched_audio"]["substitution_diagnosis_accuracy"] < base_acc * 0.70 and
            results["silence_control"]["substitution_diagnosis_accuracy"] < base_acc * 0.70
        )
    }

    return {
        "perturbation_evaluations": results,
        "acoustic_dependence_sensitivity": sensitivity_report
    }


def train_bounded_probe(
    seed: int = 1337,
    total_updates: int = 200,
    lr_head: float = 1e-3,
    device: str = "cuda"
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"RUNNING BOUNDED PROBE TRAINING — Seed={seed}, Updates={total_updates}, Head LR={lr_head}")
    print("=" * 80)

    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    set_strict_seed(seed)

    ds_train = ProtocolDataset("internal_train_probe_256")
    ds_holdout = ProtocolDataset("internal_holdout_probe_64")

    collate_fn = LangfensMDDCollate(pad_token_id=-100, ctc_blank_id=0)
    train_loader = DataLoader(ds_train, batch_size=2, shuffle=True, collate_fn=collate_fn)
    train_iter = iter(train_loader)

    model = LangfensWav2Vec2MDD.from_pretrained(
        DEFAULT_PRETRAINED_CHECKPOINT,
        revision=DEFAULT_PRETRAINED_REVISION,
        alignment_mode="cross_attention",
        ctc_zero_infinity_policy="raise"
    )
    model.freeze_feature_encoder()

    # Freeze backbone throughout
    for p in model.wav2vec2.parameters():
        p.requires_grad = False

    model.to(device)

    head_params = [p for n, p in model.named_parameters() if not n.startswith("wav2vec2") and p.requires_grad]
    opt = torch.optim.AdamW(head_params, lr=lr_head, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=20, num_training_steps=total_updates)
    scaler = torch.amp.GradScaler("cuda", init_scale=16384.0)

    grad_accum_steps = 2
    micro_step = 0
    updates = 0
    accum_loss = 0.0
    amp_overflow_events = 0

    learning_curve = []

    while updates < total_updates:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_loader = DataLoader(ds_train, batch_size=2, shuffle=True, collate_fn=collate_fn)
            train_iter = iter(train_loader)
            batch = next(train_iter)

        micro_step += 1
        net_in = {k: v.to(device) for k, v in batch["net_input"].items()}
        tb = batch["target_binary"].to(device)
        tc = batch["target_ctc"].to(device)
        tl = batch["target_lengths"].to(device)
        td = batch["target_diagnosis"].to(device)

        with torch.amp.autocast(device_type=device, dtype=torch.float16):
            out = model(**net_in, target_binary=tb, target_ctc=tc, target_lengths=tl, target_diagnosis=td)
            scaled_loss = out.loss / grad_accum_steps

        scaler.scale(scaled_loss).backward()
        accum_loss += out.loss.item()

        if micro_step % grad_accum_steps == 0:
            scaler.unscale_(opt)
            total_norm = torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
            cur_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() < cur_scale:
                amp_overflow_events += 1

            sched.step()
            opt.zero_grad()
            updates += 1

            if updates % 25 == 0 or updates == total_updates:
                cur_lr = opt.param_groups[0]["lr"]
                ctc_val = out.loss_dict.get("ctc_loss", torch.tensor(0.0)).item() if out.loss_dict else 0.0
                bin_val = out.loss_dict.get("binary_loss", torch.tensor(0.0)).item() if out.loss_dict else 0.0
                diag_val = out.loss_dict.get("diagnosis_loss", torch.tensor(0.0)).item() if out.loss_dict else 0.0
                print(f"  [Update {updates:3d}/{total_updates}] Loss: {accum_loss/grad_accum_steps:.4f} | CTC: {ctc_val:.4f} | Bin: {bin_val:.4f} | Diag: {diag_val:.4f} | GradNorm: {float(total_norm):.2f} | LR: {cur_lr:.2e}")
                learning_curve.append({
                    "update": updates,
                    "loss": round(accum_loss / grad_accum_steps, 4),
                    "ctc_loss": round(ctc_val, 4),
                    "binary_loss": round(bin_val, 4),
                    "diagnosis_loss": round(diag_val, 4),
                    "grad_norm": round(float(total_norm), 4),
                    "lr": cur_lr
                })

            accum_loss = 0.0

    elapsed = time.time() - start_time
    peak_vram_mib = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0

    print(f"\nEvaluating final checkpoint on internal train probe (256 utts)...")
    eval_train = evaluate_protocol_subset(model, ds_train, device)
    print(f"Evaluating final checkpoint on internal holdout probe (64 utts)...")
    eval_holdout = evaluate_protocol_subset(model, ds_holdout, device)
    print(f"Running acoustic dependence audit on holdout probe...")
    acoustic_audit = evaluate_acoustic_dependence_perturbations(model, ds_holdout, device)

    # Save checkpoint artifact with complete reproducibility contract
    ckpt_dir = Path("checkpoints/RQ1-TRN-01B1/reconciliation")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"rq1_trn_01b1_r_frozen_probe_seed{seed}_step{total_updates}.pt"

    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()

    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": sched.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_states": {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate()
        },
        "step": total_updates,
        "micro_step": micro_step,
        "git_commit_sha": git_sha,
        "seed": seed,
        "hyperparameters": {
            "backbone_frozen": True,
            "lr_head": lr_head,
            "lr_backbone": 0.0,
            "warmup_steps": 20,
            "total_updates": total_updates,
            "batch_size": 2,
            "gradient_accumulation": 2,
            "weight_decay": 0.01
        },
        "protocol_manifest": "manifests/RQ1-TRN-01B1/reconciliation/INTERNAL_TRAIN_HOLDOUT_PROTOCOL.json"
    }

    torch.save(checkpoint_payload, ckpt_path)
    print(f"[Saved complete reproducible checkpoint to {ckpt_path}]")

    return {
        "seed": seed,
        "total_updates": total_updates,
        "lr_head": lr_head,
        "runtime_seconds": round(elapsed, 2),
        "peak_vram_mib": round(peak_vram_mib, 2),
        "amp_overflow_events": amp_overflow_events,
        "git_sha": git_sha,
        "checkpoint_path": str(ckpt_path),
        "learning_curve": learning_curve,
        "train_probe_metrics": eval_train,
        "holdout_probe_metrics": eval_holdout,
        "acoustic_dependence_audit": acoustic_audit
    }


def main():
    parser = argparse.ArgumentParser(description="Run RQ1-TRN-01B1-R Reconciliation Campaign")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1337, 42])
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    out_dir = Path("manifests/RQ1-TRN-01B1/reconciliation")
    out_dir.mkdir(parents=True, exist_ok=True)

    multiseed_results = {}
    for seed in args.seeds:
        res = train_bounded_probe(seed=seed, total_updates=args.updates, lr_head=args.lr)
        multiseed_results[f"seed_{seed}"] = res

    # Save aggregated train-only generalization metrics JSON
    agg_file = out_dir / "TRAIN_ONLY_GENERALIZATION_METRICS.json"
    with open(agg_file, "w", encoding="utf-8") as f:
        json.dump(multiseed_results, f, indent=2, cls=NumpyEncoder)
    print(f"\n[Saved full multiseed generalization metrics to {agg_file}]")


if __name__ == "__main__":
    main()
