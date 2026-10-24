#!/usr/bin/env python3
"""
LANGFENS RQ1-TRN-01B1-R: Controlled 2x2 Factorial Ablation Campaign.

Isolates:
  Factor 1: Backbone Freezing Policy (Frozen vs Unfrozen)
  Factor 2: Multi-task Head Learning Rate (1e-4 vs 1e-3)

Cell Grid:
  - Cell (Unfrozen, Head LR 1e-4): Baseline historical (matched Variant A)
  - Cell (Unfrozen, Head LR 1e-3): Unfrozen backbone with elevated head LR
  - Cell (Frozen,   Head LR 1e-4): Frozen backbone with low head LR
  - Cell (Frozen,   Head LR 1e-3): Frozen backbone with elevated head LR (matched Variant D)

Preserves exact matching across all other hyperparameters:
  - Dataset: Fixed 32-utterance Fold 1 train-only subset (TrainSubsetDataset)
  - Batch size: 2, Gradient accumulation: 2 (effective batch size: 4)
  - Total updates: 100
  - Optimizer: AdamW, Weight Decay: 0.01, Betas: (0.9, 0.999), Eps: 1e-8
  - Scheduler: Cosine schedule with 10 warmup steps
  - AMP: torch.amp.autocast('cuda', dtype=torch.float16), GradScaler(init_scale=16384.0)
  - Blank bias: 0.0 (default)
  - Seed: 1337
"""

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from langfens_prep.model import (
    LangfensWav2Vec2MDD,
    DEFAULT_PRETRAINED_CHECKPOINT,
    DEFAULT_PRETRAINED_REVISION
)
from langfens_prep.collate import LangfensMDDCollate
from langfens_prep.phoneme_normalizer import STANDARD_39_ARPABET, PHONE_TO_INDEX
from audit_scripts.run_training_sanity import TrainSubsetDataset, compute_pr_auc, greedy_ctc_decode


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


def compute_levenshtein_edits(ref: List[int], hyp: List[int]):
    """Standard Levenshtein alignment returning (total_edits, subs, dels, inss)."""
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
    subs = 0
    dels = 0
    inss = 0
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

    return dp[n][m], subs, dels, inss


@torch.no_grad()
def evaluate_full_multitask(model: torch.nn.Module, dataset: TrainSubsetDataset, device: str = "cuda") -> Dict[str, Any]:
    model.eval()
    collate_fn = LangfensMDDCollate(pad_token_id=-100, ctc_blank_id=0)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    total_loss = 0.0
    total_ctc_loss = 0.0
    total_bin_loss = 0.0
    total_diag_loss = 0.0
    num_eval_batches = 0

    bin_preds_all = []
    bin_targets_all = []
    bin_scores_all = []

    sub_diag_correct = 0
    sub_diag_total = 0
    sub_class_gold_counts = np.zeros(len(STANDARD_39_ARPABET), dtype=int)
    sub_class_pred_counts = np.zeros(len(STANDARD_39_ARPABET), dtype=int)
    sub_class_correct_counts = np.zeros(len(STANDARD_39_ARPABET), dtype=int)
    sub_confusion_matrix = np.zeros((len(STANDARD_39_ARPABET), len(STANDARD_39_ARPABET)), dtype=int)

    aux_diag_correct = 0
    aux_diag_total = 0

    ctc_total_edits = 0
    ctc_total_ref_tokens = 0
    ctc_total_subs = 0
    ctc_total_dels = 0
    ctc_total_inss = 0
    ctc_total_active_frames = 0
    ctc_total_blank_frames = 0
    ctc_total_hyp_tokens = 0
    decoded_utterances = []

    for batch in loader:
        net_in = {k: v.to(device) for k, v in batch["net_input"].items()}
        target_bin = batch["target_binary"].to(device)
        target_ctc = batch["target_ctc"].to(device)
        target_lens = batch["target_lengths"].to(device)
        target_diag = batch["target_diagnosis"].to(device)

        with torch.amp.autocast(device_type=device, dtype=torch.float16):
            out = model(
                **net_in,
                target_binary=target_bin,
                target_ctc=target_ctc,
                target_lengths=target_lens,
                target_diagnosis=target_diag
            )

        total_loss += out.loss.item()
        if out.loss_dict is not None:
            total_ctc_loss += out.loss_dict.get("ctc_loss", torch.tensor(0.0)).item()
            total_bin_loss += out.loss_dict.get("binary_loss", torch.tensor(0.0)).item()
            total_diag_loss += out.loss_dict.get("diagnosis_loss", torch.tensor(0.0)).item()
        num_eval_batches += 1

        # 1. Binary MDD evaluation
        b_logits = out.binary_logits
        b_mask = (target_bin != -100)
        if b_mask.any():
            v_probs = torch.sigmoid(b_logits[b_mask]).float().cpu().numpy()
            v_targets = target_bin[b_mask].cpu().numpy()
            v_preds = (v_probs >= 0.5).astype(int)
            bin_scores_all.extend(v_probs.tolist())
            bin_targets_all.extend(v_targets.tolist())
            bin_preds_all.extend(v_preds.tolist())

        # 2. Multi-class substitution diagnosis
        d_logits = out.diagnosis_logits  # (B, T, 39)
        d_mask = (target_diag != -100)
        if d_mask.any():
            preds_39 = d_logits[d_mask].argmax(dim=-1).cpu().numpy()
            gold_39 = target_diag[d_mask].cpu().numpy()
            sub_diag_correct += int((preds_39 == gold_39).sum())
            sub_diag_total += len(gold_39)
            for g, p in zip(gold_39, preds_39):
                sub_confusion_matrix[g, p] += 1
                sub_class_gold_counts[g] += 1
                sub_class_pred_counts[p] += 1
                if g == p:
                    sub_class_correct_counts[g] += 1

        # 3. Auxiliary phone classification
        if "target_aux_phone" in batch and batch["target_aux_phone"] is not None:
            taux = batch["target_aux_phone"].to(device)
            aux_mask = (taux != -100)
            if aux_mask.any():
                preds_aux = d_logits[aux_mask].argmax(dim=-1).cpu().numpy()
                gold_aux = taux[aux_mask].cpu().numpy()
                aux_diag_correct += int((preds_aux == gold_aux).sum())
                aux_diag_total += len(gold_aux)

        # 4. CTC decoding and framewise metrics
        log_probs = F.log_softmax(out.ctc_logits, dim=-1)
        for b_idx in range(log_probs.shape[0]):
            in_len = out.input_lengths[b_idx].item()
            active_lp = log_probs[b_idx, :in_len]
            raw_tokens = active_lp.argmax(dim=-1).tolist()
            blanks = raw_tokens.count(0)
            ctc_total_active_frames += in_len
            ctc_total_blank_frames += blanks

            # Greedy collapse
            hyp_phones = []
            prev_token = 0
            for token in raw_tokens:
                if token != 0 and token != prev_token:
                    hyp_phones.append(int(token))
                prev_token = token

            ctc_total_hyp_tokens += len(hyp_phones)
            ref_len = target_lens[b_idx].item()
            ref_phones = target_ctc[b_idx, :ref_len].cpu().tolist()
            ctc_total_ref_tokens += ref_len

            edits, subs, dels, inss = compute_levenshtein_edits(ref_phones, hyp_phones)
            ctc_total_edits += edits
            ctc_total_subs += subs
            ctc_total_dels += dels
            ctc_total_inss += inss

            decoded_utterances.append({
                "hyp_len": len(hyp_phones),
                "ref_len": ref_len,
                "edits": edits,
                "subs": subs,
                "dels": dels,
                "inss": inss
            })

    avg_loss = total_loss / max(1, num_eval_batches)
    avg_ctc = total_ctc_loss / max(1, num_eval_batches)
    avg_bin = total_bin_loss / max(1, num_eval_batches)
    avg_diag = total_diag_loss / max(1, num_eval_batches)

    # Binary PR-AUC and F1
    bin_targets_np = np.array(bin_targets_all)
    bin_preds_np = np.array(bin_preds_all)
    bin_scores_np = np.array(bin_scores_all)

    tp_05 = int(np.sum((bin_preds_np == 1) & (bin_targets_np == 1)))
    fp_05 = int(np.sum((bin_preds_np == 1) & (bin_targets_np == 0)))
    fn_05 = int(np.sum((bin_preds_np == 0) & (bin_targets_np == 1)))
    tn_05 = int(np.sum((bin_preds_np == 0) & (bin_targets_np == 0)))

    prec_05 = float(tp_05 / (tp_05 + fp_05)) if (tp_05 + fp_05) > 0 else 0.0
    rec_05 = float(tp_05 / (tp_05 + fn_05)) if (tp_05 + fn_05) > 0 else 0.0
    f1_05 = float(2 * prec_05 * rec_05 / (prec_05 + rec_05)) if (prec_05 + rec_05) > 0 else 0.0
    acc_05 = float((tp_05 + tn_05) / len(bin_targets_np)) if len(bin_targets_np) > 0 else 0.0

    pr_auc = compute_pr_auc(bin_targets_np, bin_scores_np)

    cor_scores = bin_scores_np[bin_targets_np == 0]
    err_scores = bin_scores_np[bin_targets_np == 1]

    def calc_dist(arr):
        if len(arr) == 0:
            return {}
        return {
            "count": len(arr),
            "min": round(float(np.min(arr)), 4),
            "p25": round(float(np.percentile(arr, 25)), 4),
            "median": round(float(np.median(arr)), 4),
            "mean": round(float(np.mean(arr)), 4),
            "p75": round(float(np.percentile(arr, 75)), 4),
            "max": round(float(np.max(arr)), 4)
        }

    sub_diag_acc = float(sub_diag_correct / sub_diag_total) if sub_diag_total > 0 else 0.0
    aux_diag_acc = float(aux_diag_correct / aux_diag_total) if aux_diag_total > 0 else 0.0
    per = float(ctc_total_edits / ctc_total_ref_tokens) if ctc_total_ref_tokens > 0 else 1.0
    blank_frac = float(ctc_total_blank_frames / ctc_total_active_frames) if ctc_total_active_frames > 0 else 0.0
    non_empty_count = sum(1 for u in decoded_utterances if u["hyp_len"] > 0)

    # Dynamic baseline calculations & strict integrity assertions
    max_gold_count = int(np.max(sub_class_gold_counts)) if sub_diag_total > 0 else 0
    maj_class_idx = int(np.argmax(sub_class_gold_counts)) if sub_diag_total > 0 else 0
    majority_phone = STANDARD_39_ARPABET[maj_class_idx] if sub_diag_total > 0 else None
    dynamic_majority_baseline = round(max_gold_count / sub_diag_total, 4) if sub_diag_total > 0 else 0.0
    uniform_chance = round(1.0 / len(STANDARD_39_ARPABET), 4)

    if sub_diag_total > 0:
        assert int(np.sum(sub_class_gold_counts)) == sub_diag_total
        assert int(np.sum(sub_confusion_matrix)) == sub_diag_total
        assert int(np.sum(sub_confusion_matrix[maj_class_idx, :])) == max_gold_count

    # Per-class diagnosis coverage
    per_class_diagnosis = {}
    classes_with_support = 0
    classes_predicted = 0
    classes_correct = 0
    for idx, phone in enumerate(STANDARD_39_ARPABET):
        gold_c = int(sub_class_gold_counts[idx])
        pred_c = int(sub_class_pred_counts[idx])
        corr_c = int(sub_class_correct_counts[idx])
        if gold_c > 0:
            classes_with_support += 1
        if pred_c > 0:
            classes_predicted += 1
        if corr_c > 0:
            classes_correct += 1
        per_class_diagnosis[phone] = {
            "gold_support": gold_c,
            "predicted_count": pred_c,
            "correct_count": corr_c,
            "recall": round(corr_c / gold_c, 4) if gold_c > 0 else None,
            "precision": round(corr_c / pred_c, 4) if pred_c > 0 else None
        }

    model.train()
    return {
        "loss": round(avg_loss, 4),
        "ctc_loss": round(avg_ctc, 4),
        "binary_loss": round(avg_bin, 4),
        "diagnosis_loss": round(avg_diag, 4),
        "binary_mdd": {
            "default_05": {
                "accuracy": round(acc_05, 4),
                "precision": round(prec_05, 4),
                "recall": round(rec_05, 4),
                "f1": round(f1_05, 4),
                "confusion_matrix": {"tp": tp_05, "fp": fp_05, "fn": fn_05, "tn": tn_05}
            },
            "train_only_pr_auc": round(pr_auc, 4),
            "score_distribution": {
                "correct_tokens": calc_dist(cor_scores),
                "error_tokens": calc_dist(err_scores)
            }
        },
        "substitution_diagnosis": {
            "top1_accuracy": round(sub_diag_acc, 4),
            "correct_predictions": sub_diag_correct,
            "eligible_denominator": sub_diag_total,
            "uniform_chance_baseline": uniform_chance,
            "majority_class_baseline": dynamic_majority_baseline,
            "majority_class_phone": majority_phone,
            "majority_class_count": max_gold_count,
            "classes_with_gold_support": classes_with_support,
            "classes_predicted": classes_predicted,
            "classes_with_correct_hits": classes_correct,
            "per_class_summary": per_class_diagnosis,
            "confusion_matrix_39x39": sub_confusion_matrix.tolist()
        },
        "auxiliary_realized_phone_classification": {
            "top1_accuracy": round(aux_diag_acc, 4),
            "correct_predictions": aux_diag_correct,
            "evaluated_tokens_denominator": aux_diag_total
        },
        "ctc_metrics": {
            "phone_error_rate": round(per, 4),
            "total_edits": ctc_total_edits,
            "total_ref_tokens": ctc_total_ref_tokens,
            "active_frames": ctc_total_active_frames,
            "blank_frames": ctc_total_blank_frames,
            "blank_fraction": round(blank_frac, 4),
            "hyp_tokens_total": ctc_total_hyp_tokens,
            "mean_hyp_length": round(ctc_total_hyp_tokens / max(1, len(decoded_utterances)), 2),
            "substitutions": ctc_total_subs,
            "deletions": ctc_total_dels,
            "insertions": ctc_total_inss,
            "is_blank_collapsed": (per >= 0.9999 and blank_frac >= 0.9900 and ctc_total_hyp_tokens == 0),
            "non_empty_utterance_count": non_empty_count,
            "non_empty_utterance_fraction": round(non_empty_count / max(1, len(decoded_utterances)), 4)
        }
    }


def run_factorial_cell(
    cell_id: str,
    backbone_frozen: bool,
    lr_head: float,
    lr_backbone: float = 1e-4,
    total_updates: int = 100,
    seed: int = 1337,
    device: str = "cuda"
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"RUNNING 2x2 FACTORIAL CELL: {cell_id}")
    print(f"Backbone: {'FROZEN' if backbone_frozen else 'UNFROZEN'}, Head LR: {lr_head}, Backbone LR: {0.0 if backbone_frozen else lr_backbone}")
    print(f"Updates: {total_updates}, Seed: {seed}")
    print("=" * 80)

    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    set_strict_seed(seed)

    ds = TrainSubsetDataset()
    collate_fn = LangfensMDDCollate(pad_token_id=-100, ctc_blank_id=0)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
    batches = list(loader)
    num_batches = len(batches)

    # Initialize model
    model = LangfensWav2Vec2MDD.from_pretrained(
        DEFAULT_PRETRAINED_CHECKPOINT,
        revision=DEFAULT_PRETRAINED_REVISION,
        alignment_mode="cross_attention",
        ctc_zero_infinity_policy="raise"
    )
    model.freeze_feature_encoder()

    if backbone_frozen:
        for p in model.wav2vec2.parameters():
            p.requires_grad = False

    model.to(device)

    # Param groups
    head_params = [p for n, p in model.named_parameters() if not n.startswith("wav2vec2") and p.requires_grad]
    backbone_params = [p for n, p in model.wav2vec2.named_parameters() if p.requires_grad]

    param_groups = []
    if backbone_params and not backbone_frozen:
        param_groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": 0.01, "name": "backbone"})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr_head, "weight_decay": 0.01, "name": "heads"})

    opt = torch.optim.AdamW(param_groups)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=10, num_training_steps=total_updates)
    scaler = torch.amp.GradScaler("cuda", init_scale=16384.0)

    # Record Step 0 metrics
    eval_0 = evaluate_full_multitask(model, ds, device)
    print(f"  [Step  0] CTC Loss: {eval_0['ctc_loss']:.4f} | Blank%: {eval_0['ctc_metrics']['blank_fraction']*100:.2f}% | HypTokens: {eval_0['ctc_metrics']['hyp_tokens_total']} | PER: {eval_0['ctc_metrics']['phone_error_rate']:.4f} | SubDiag: {eval_0['substitution_diagnosis']['top1_accuracy']*100:.2f}% | BinF1: {eval_0['binary_mdd']['default_05']['f1']:.4f}")

    grad_accum_steps = 2
    micro_step = 0
    updates = 0
    accum_loss = 0.0
    batch_idx = 0
    amp_overflow_events = 0

    step_history = []
    checkpoint_snapshots = {}

    while updates < total_updates:
        batch = batches[batch_idx % num_batches]
        batch_idx += 1
        micro_step += 1

        net_in = {k: v.to(device) for k, v in batch["net_input"].items()}
        target_bin = batch["target_binary"].to(device)
        target_ctc = batch["target_ctc"].to(device)
        target_lens = batch["target_lengths"].to(device)
        target_diag = batch["target_diagnosis"].to(device)

        with torch.amp.autocast(device_type=device, dtype=torch.float16):
            out = model(
                **net_in,
                target_binary=target_bin,
                target_ctc=target_ctc,
                target_lengths=target_lens,
                target_diagnosis=target_diag
            )
            scaled_loss = out.loss / grad_accum_steps

        scaler.scale(scaled_loss).backward()
        accum_loss += out.loss.item()

        if micro_step % grad_accum_steps == 0:
            scaler.unscale_(opt)
            total_norm = torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=1.0
            )

            current_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            new_scale = scaler.get_scale()
            if new_scale < current_scale:
                amp_overflow_events += 1

            sched.step()
            opt.zero_grad()
            updates += 1

            if updates % 10 == 0 or updates == total_updates:
                head_lr = opt.param_groups[-1]["lr"]
                step_eval = evaluate_full_multitask(model, ds, device)
                ctc_item = out.loss_dict.get("ctc_loss", torch.tensor(0.0)).item() if out.loss_dict else 0.0
                print(f"  [Step {updates:2d}] Loss: {accum_loss/grad_accum_steps:.4f} | CTC: {ctc_item:.4f} | Blank%: {step_eval['ctc_metrics']['blank_fraction']*100:.2f}% | HypTokens: {step_eval['ctc_metrics']['hyp_tokens_total']} | PER: {step_eval['ctc_metrics']['phone_error_rate']:.4f} | SubDiag: {step_eval['substitution_diagnosis']['top1_accuracy']*100:.2f}% | GradNorm: {float(total_norm):.2f} | HeadLR: {head_lr:.2e}")
                step_history.append({
                    "update": updates,
                    "micro_step": micro_step,
                    "loss": round(accum_loss / grad_accum_steps, 4),
                    "grad_norm": round(float(total_norm), 4),
                    "head_lr": head_lr,
                    "metrics": step_eval
                })

            accum_loss = 0.0

    final_eval = evaluate_full_multitask(model, ds, device)
    elapsed = time.time() - start_time
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0
    peak_vram_mib = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0

    print(f"\nCOMPLETED {cell_id} in {elapsed:.2f}s | Peak VRAM: {peak_vram_mb:.2f} MiB")
    print(f"Final CTC Loss: {final_eval['ctc_loss']:.4f} | Blank%: {final_eval['ctc_metrics']['blank_fraction']*100:.2f}% | PER: {final_eval['ctc_metrics']['phone_error_rate']:.4f} | SubDiag: {final_eval['substitution_diagnosis']['top1_accuracy']*100:.2f}%")

    return {
        "cell_id": cell_id,
        "backbone_frozen": backbone_frozen,
        "lr_backbone": 0.0 if backbone_frozen else lr_backbone,
        "lr_head": lr_head,
        "total_updates": total_updates,
        "seed": seed,
        "runtime_seconds": round(elapsed, 2),
        "peak_vram_mib": round(peak_vram_mib, 2),
        "amp_overflow_events": amp_overflow_events,
        "optimizer": {
            "type": "AdamW",
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "param_groups": [
                {"name": g.get("name", f"group_{i}"), "lr": g["lr"], "params_count": sum(p.numel() for p in g["params"])}
                for i, g in enumerate(opt.param_groups)
            ]
        },
        "scheduler": {
            "type": "cosine_with_warmup",
            "warmup_steps": 10,
            "total_steps": total_updates
        },
        "initial_metrics": eval_0,
        "final_metrics": final_eval,
        "step_history": step_history
    }


def main():
    parser = argparse.ArgumentParser(description="Run 2x2 Factorial Ablation")
    parser.add_argument("--cell", choices=["all", "cell_unfrozen_lr1e4", "cell_unfrozen_lr1e3", "cell_frozen_lr1e4", "cell_frozen_lr1e3", "merge"], default="all")
    parser.add_argument("--force", action="store_true", help="Force rerun even if output json exists")
    args = parser.parse_args()

    out_dir = Path("manifests/RQ1-TRN-01B1/reconciliation")
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = {
        "cell_unfrozen_lr1e4": {
            "cell_id": "cell_unfrozen_lr1e4",
            "backbone_frozen": False,
            "lr_head": 1e-4,
            "lr_backbone": 1e-4
        },
        "cell_unfrozen_lr1e3": {
            "cell_id": "cell_unfrozen_lr1e3",
            "backbone_frozen": False,
            "lr_head": 1e-3,
            "lr_backbone": 1e-4
        },
        "cell_frozen_lr1e4": {
            "cell_id": "cell_frozen_lr1e4",
            "backbone_frozen": True,
            "lr_head": 1e-4,
            "lr_backbone": 0.0
        },
        "cell_frozen_lr1e3": {
            "cell_id": "cell_frozen_lr1e3",
            "backbone_frozen": True,
            "lr_head": 1e-3,
            "lr_backbone": 0.0
        }
    }

    results = {}
    if args.cell == "all":
        for cid, cfg in grid.items():
            single_path = out_dir / f"metrics_{cid}.json"
            if single_path.exists() and not args.force:
                print(f"[Skipping {cid} — already exists at {single_path}. Use --force to rerun.]")
                with open(single_path, "r", encoding="utf-8") as f:
                    results[cid] = json.load(f)
                continue
            res = run_factorial_cell(**cfg)
            results[cid] = res
            with open(single_path, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2, cls=NumpyEncoder)

        agg_path = out_dir / "FACTORIAL_2X2_METRICS.json"
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        print(f"\n[Saved full 2x2 factorial metrics to {agg_path}]")
    elif args.cell == "merge":
        merged = {}
        for cid in grid.keys():
            single_path = out_dir / f"metrics_{cid}.json"
            if single_path.exists():
                with open(single_path, "r", encoding="utf-8") as f:
                    merged[cid] = json.load(f)
            else:
                print(f"[Warning] Missing {single_path}")
        agg_path = out_dir / "FACTORIAL_2X2_METRICS.json"
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, cls=NumpyEncoder)
        print(f"\n[Merged {len(merged)} cells into {agg_path}]")
    else:
        cfg = grid[args.cell]
        res = run_factorial_cell(**cfg)
        single_path = out_dir / f"metrics_{args.cell}.json"
        with open(single_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, cls=NumpyEncoder)
        print(f"\n[Saved {args.cell} to {single_path}]")


if __name__ == "__main__":
    main()
