#!/usr/bin/env python3
"""
LANGFENS RQ1-TRN-01B1: Controlled CTC Stabilization Ablation Experiments.
Runs and evaluates 4 rigorous variants on the deterministic Fold 1 train-only subset:
  Variant A: Historical Control (Unfrozen backbone, blank_bias=0.0, AdamW lr=1e-4)
  Variant B: Negative Blank Bias Alone (Unfrozen backbone, blank_bias=-3.0, AdamW lr=1e-4)
  Variant C: Staged Optimization (Stage 1: Frozen backbone 50 updates, heads lr=1e-3; Stage 2: Unfrozen encoder 50 updates, backbone lr=2e-5, heads lr=2e-4)
  Variant D: Frozen Backbone Probe (Frozen backbone 100 updates, heads lr=1e-3)

Evaluates:
  1. CTC Loss, Framewise Blank Fraction, Hyp Tokens, Substitutions, Deletions, Insertions, PER
  2. Non-empty decoded utterance fraction
  3. Sample transcriptions (predicted vs ground truth realized phonemes)
  4. Binary MDD: Accuracy, Precision, Recall, F1 at default 0.5 threshold, PR-AUC, score distributions
  5. Substitution Diagnosis: Accuracy on eligible substitutions (N=124), comparison to majority (14.52%) and chance (2.56%), confusion matrix, per-class coverage
  6. GPU runtime, peak VRAM, AMP overflow events
"""

import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("."))

from langfens_prep.model import (
    LangfensWav2Vec2MDD,
    DEFAULT_PRETRAINED_CHECKPOINT,
    DEFAULT_PRETRAINED_REVISION
)
from langfens_prep.collate import LangfensMDDCollate
from langfens_prep.phoneme_normalizer import INDEX_TO_PHONE, STANDARD_39_ARPABET
from audit_scripts.run_training_sanity import (
    TrainSubsetDataset,
    detailed_levenshtein,
    greedy_ctc_decode,
    compute_pr_auc
)


def set_strict_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_full_multitask(
    model: LangfensWav2Vec2MDD,
    dataset: TrainSubsetDataset,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Evaluates CTC, Binary MDD, and Substitution Diagnosis with strict metric semantics.
    """
    model.eval()
    collate_fn = LangfensMDDCollate(pad_token_id=-100, ctc_blank_id=0)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    tot_loss = 0.0
    tot_ctc = 0.0
    tot_bin = 0.0
    tot_diag = 0.0
    num_batches = 0

    all_bin_preds = []
    all_bin_targets = []
    all_bin_probs = []

    # Substitution-only diagnosis (N=124)
    sub_diag_correct = 0
    sub_diag_total = 0
    sub_confusion_matrix = np.zeros((39, 39), dtype=int)
    sub_class_gold_counts = np.zeros(39, dtype=int)
    sub_class_pred_counts = np.zeros(39, dtype=int)
    sub_class_correct_counts = np.zeros(39, dtype=int)

    # Auxiliary realized-phone classification (N=1000)
    aux_diag_correct = 0
    aux_diag_total = 0

    # CTC detailed metrics
    ctc_total_edits = 0
    ctc_total_subs = 0
    ctc_total_dels = 0
    ctc_total_inss = 0
    ctc_total_ref_tokens = 0
    ctc_total_hyp_tokens = 0
    ctc_total_active_frames = 0
    ctc_total_blank_frames = 0
    decoded_utterances = []

    with torch.no_grad():
        for batch in loader:
            net_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["net_input"].items()}
            tb = batch["target_binary"].to(device)
            tc = batch["target_ctc"].to(device)
            tl = batch["target_lengths"].to(device)
            td = batch["target_diagnosis"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                out = model(**net_input, target_binary=tb, target_ctc=tc, target_lengths=tl, target_diagnosis=td)

            tot_loss += out.loss.item()
            tot_ctc += out.loss_dict.get("ctc_loss", torch.tensor(0.0)).item()
            tot_bin += out.loss_dict.get("binary_loss", torch.tensor(0.0)).item()
            tot_diag += out.loss_dict.get("diagnosis_loss", torch.tensor(0.0)).item()
            num_batches += 1

            # 1. Binary Metrics
            b_logits = out.binary_logits
            mask = (tb != -100)
            if mask.any():
                v_logits = b_logits[mask]
                v_targets = tb[mask].cpu().numpy()
                v_probs = torch.sigmoid(v_logits).float().cpu().numpy()
                v_preds = (v_probs >= 0.5).astype(int)

                all_bin_preds.extend(v_preds.tolist())
                all_bin_targets.extend(v_targets.tolist())
                all_bin_probs.extend(v_probs.tolist())

            # 2. Diagnosis Metrics: Strict Substitution-Only (td != -100)
            d_logits = out.diagnosis_logits  # (B, T, 39)
            d_mask = (td != -100)
            if d_mask.any():
                preds_39 = d_logits[d_mask].argmax(dim=-1).cpu().numpy()
                gold_39 = td[d_mask].cpu().numpy()
                sub_diag_correct += int((preds_39 == gold_39).sum())
                sub_diag_total += len(gold_39)
                for g, p in zip(gold_39, preds_39):
                    sub_confusion_matrix[g, p] += 1
                    sub_class_gold_counts[g] += 1
                    sub_class_pred_counts[p] += 1
                    if g == p:
                        sub_class_correct_counts[g] += 1

            # Auxiliary realized-phone
            if "target_aux_phone" in batch and batch["target_aux_phone"] is not None:
                taux = batch["target_aux_phone"].to(device)
                aux_mask = (taux != -100)
                if aux_mask.any():
                    preds_aux = d_logits[aux_mask].argmax(dim=-1).cpu().numpy()
                    gold_aux = taux[aux_mask].cpu().numpy()
                    aux_diag_correct += int((preds_aux == gold_aux).sum())
                    aux_diag_total += len(gold_aux)

            # 3. CTC Metrics: Strictly active acoustic frames
            log_probs = F.log_softmax(out.ctc_logits, dim=-1)
            for b_idx in range(log_probs.shape[0]):
                in_len = out.input_lengths[b_idx].item()
                active_lp = log_probs[b_idx, :in_len]
                raw_tokens = active_lp.argmax(dim=-1).tolist()
                blanks = raw_tokens.count(0)
                ctc_total_active_frames += in_len
                ctc_total_blank_frames += blanks

                hyp_tokens = greedy_ctc_decode(active_lp, blank_id=0)
                ctc_total_hyp_tokens += len(hyp_tokens)

                ref_len = tl[b_idx].item()
                ref_tokens = tc[b_idx, :ref_len].tolist()
                edits, s, d, ins = detailed_levenshtein(ref_tokens, hyp_tokens)
                ctc_total_edits += edits
                ctc_total_subs += s
                ctc_total_dels += d
                ctc_total_inss += ins
                ctc_total_ref_tokens += ref_len

                decoded_utterances.append({
                    "ref": [INDEX_TO_PHONE.get(idx - 1, str(idx)) for idx in ref_tokens],
                    "hyp": [INDEX_TO_PHONE.get(idx - 1, str(idx)) for idx in hyp_tokens],
                    "edits": edits,
                    "subs": s,
                    "dels": d,
                    "inss": ins,
                    "ref_len": ref_len,
                    "hyp_len": len(hyp_tokens)
                })

    avg_loss = tot_loss / max(1, num_batches)
    avg_ctc = tot_ctc / max(1, num_batches)
    avg_bin = tot_bin / max(1, num_batches)
    avg_diag = tot_diag / max(1, num_batches)

    # Binary summary
    y_true = np.array(all_bin_targets)
    y_pred_05 = np.array(all_bin_preds)
    y_scores = np.array(all_bin_probs)
    acc_05 = float((y_true == y_pred_05).mean()) if len(y_true) > 0 else 0.0

    tp_05 = int(((y_pred_05 == 1) & (y_true == 1)).sum())
    fp_05 = int(((y_pred_05 == 1) & (y_true == 0)).sum())
    fn_05 = int(((y_pred_05 == 0) & (y_true == 1)).sum())
    tn_05 = int(((y_pred_05 == 0) & (y_true == 0)).sum())

    prec_05 = float(tp_05 / (tp_05 + fp_05)) if (tp_05 + fp_05) > 0 else 0.0
    rec_05 = float(tp_05 / (tp_05 + fn_05)) if (tp_05 + fn_05) > 0 else 0.0
    f1_05 = float(2 * prec_05 * rec_05 / (prec_05 + rec_05)) if (prec_05 + rec_05) > 0 else 0.0
    pr_auc = compute_pr_auc(y_true, y_scores)

    cor_scores = y_scores[y_true == 0] if (y_true == 0).any() else np.array([])
    err_scores = y_scores[y_true == 1] if (y_true == 1).any() else np.array([])

    def calc_dist(arr: np.ndarray) -> Dict[str, Any]:
        if len(arr) == 0:
            return {}
        return {
            "count": int(len(arr)),
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
        assert int(np.sum(sub_class_gold_counts)) == sub_diag_total, (
            f"Gold class sum {int(np.sum(sub_class_gold_counts))} != sub_diag_total {sub_diag_total}"
        )
        assert int(np.sum(sub_confusion_matrix)) == sub_diag_total, (
            f"Confusion matrix sum {int(np.sum(sub_confusion_matrix))} != sub_diag_total {sub_diag_total}"
        )
        assert int(np.sum(sub_confusion_matrix[maj_class_idx, :])) == max_gold_count, (
            f"Majority row sum != max_gold_count ({max_gold_count})"
        )

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
            "mean_hyp_length": round(ctc_total_hyp_tokens / max(1, len(dataset)), 2),
            "substitutions": ctc_total_subs,
            "deletions": ctc_total_dels,
            "insertions": ctc_total_inss,
            "is_blank_collapsed": (ctc_total_blank_frames == ctc_total_active_frames),
            "non_empty_utterance_count": non_empty_count,
            "non_empty_utterance_fraction": round(non_empty_count / len(dataset), 4),
            "sample_utterances": decoded_utterances[:3]
        }
    }


def run_variant(
    variant_id: str,
    description: str,
    staged: bool = False,
    freeze_backbone_throughout: bool = False,
    initial_blank_bias: float = 0.0,
    lr_backbone: float = 1e-4,
    lr_head: float = 1e-4,
    stage1_updates: int = 50,
    total_updates: int = 100,
    stage2_lr_backbone: float = 2e-5,
    stage2_lr_head: float = 2e-4,
    seed: int = 1337,
    device: str = "cuda"
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"STARTING ABLATION: {variant_id} — {description}")
    print(f"Staged={staged}, FrozenThroughout={freeze_backbone_throughout}, BlankBias={initial_blank_bias}")
    print(f"Total Updates={total_updates}, Seed={seed}")
    print("=" * 80)

    start_time = time.time()
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

    if initial_blank_bias != 0.0:
        with torch.no_grad():
            model.ctc_head.bias.data[0] = initial_blank_bias

    if staged or freeze_backbone_throughout:
        for p in model.wav2vec2.parameters():
            p.requires_grad = False

    model.to(device)

    # Param groups for Stage 1
    head_params = [p for n, p in model.named_parameters() if not n.startswith("wav2vec2") and p.requires_grad]
    backbone_params = [p for n, p in model.wav2vec2.named_parameters() if p.requires_grad]

    param_groups = []
    if backbone_params and lr_backbone > 0:
        param_groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": 0.01})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr_head, "weight_decay": 0.01})

    opt = torch.optim.AdamW(param_groups)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=10, num_training_steps=stage1_updates if staged else total_updates)
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

    model.train()

    while updates < total_updates:
        # Check transition to Stage 2 if staged
        if staged and updates == stage1_updates and updates < total_updates:
            print(f"\n--- [Variant {variant_id}] Transitioning to Stage 2 at Update {updates} ---")
            # Unfreeze transformer encoder
            for p in model.wav2vec2.encoder.parameters():
                p.requires_grad = True

            st2_backbone_params = [p for p in model.wav2vec2.encoder.parameters() if p.requires_grad]
            st2_head_params = [p for n, p in model.named_parameters() if not n.startswith("wav2vec2") and p.requires_grad]

            opt = torch.optim.AdamW([
                {"params": st2_backbone_params, "lr": stage2_lr_backbone, "weight_decay": 0.01},
                {"params": st2_head_params, "lr": stage2_lr_head, "weight_decay": 0.01}
            ])
            sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=5, num_training_steps=total_updates - stage1_updates)

        cur_b_idx = batch_idx % num_batches
        batch = batches[cur_b_idx]
        batch_idx += 1
        micro_step += 1

        net_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["net_input"].items()}
        tb = batch["target_binary"].to(device)
        tc = batch["target_ctc"].to(device)
        tl = batch["target_lengths"].to(device)
        td = batch["target_diagnosis"].to(device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            out = model(**net_input, target_binary=tb, target_ctc=tc, target_lengths=tl, target_diagnosis=td)
            raw_loss = out.loss
            scaled_loss = raw_loss / grad_accum_steps

        scaler.scale(scaled_loss).backward()
        accum_loss += raw_loss.item()

        if micro_step % grad_accum_steps == 0:
            scaler.unscale_(opt)
            total_norm = torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0).item()
            old_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            new_scale = scaler.get_scale()

            if new_scale < old_scale:
                amp_overflow_events += 1
            else:
                updates += 1
                sched.step()
                step_loss = accum_loss / grad_accum_steps
                step_history.append({
                    "update": updates,
                    "micro_step": micro_step,
                    "loss": step_loss,
                    "ctc_loss": out.loss_dict.get("ctc_loss", torch.tensor(0.0)).item(),
                    "bin_loss": out.loss_dict.get("binary_loss", torch.tensor(0.0)).item(),
                    "diag_loss": out.loss_dict.get("diagnosis_loss", torch.tensor(0.0)).item(),
                    "grad_norm": total_norm,
                    "scale": new_scale,
                    "lr": sched.get_last_lr()[0]
                })

                if updates % 25 == 0 or updates == total_updates:
                    inter_eval = evaluate_full_multitask(model, ds, device)
                    model.train()
                    ctc_m = inter_eval["ctc_metrics"]
                    print(f"  [Step {updates:3d}] Loss: {step_loss:.4f} | CTC: {ctc_m['phone_error_rate']:.4f} (Subs:{ctc_m['substitutions']} Del:{ctc_m['deletions']} Ins:{ctc_m['insertions']}) | HypTokens: {ctc_m['hyp_tokens_total']} | Blank%: {ctc_m['blank_fraction']*100:.2f}% | SubDiag: {inter_eval['substitution_diagnosis']['top1_accuracy']*100:.2f}% | BinF1: {inter_eval['binary_mdd']['default_05']['f1']:.4f}")
                    if ctc_m["sample_utterances"][0]["hyp"]:
                        print(f"    Sample 0 Hyp: {ctc_m['sample_utterances'][0]['hyp'][:8]}")

            accum_loss = 0.0
            opt.zero_grad()

    elapsed = time.time() - start_time
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    # Final evaluation
    final_eval = evaluate_full_multitask(model, ds, device)
    print(f"\n[Variant {variant_id} COMPLETED in {elapsed:.2f}s | Peak VRAM: {peak_vram_mb:.1f} MB | AMP Overflows: {amp_overflow_events}]")
    print(f"  Final CTC Loss: {final_eval['ctc_loss']:.4f} | PER: {final_eval['ctc_metrics']['phone_error_rate']:.4f} | Blank%: {final_eval['ctc_metrics']['blank_fraction']*100:.2f}%")
    print(f"  Final Hyp Tokens: {final_eval['ctc_metrics']['hyp_tokens_total']} | Non-Empty Utterances: {final_eval['ctc_metrics']['non_empty_utterance_count']}/32 ({final_eval['ctc_metrics']['non_empty_utterance_fraction']*100:.1f}%)")
    print(f"  Final Sub Diag: {final_eval['substitution_diagnosis']['top1_accuracy']*100:.2f}% (Chance: 2.56%, Majority: 14.52%)")
    print(f"  Final Bin F1:   {final_eval['binary_mdd']['default_05']['f1']:.4f} | PR-AUC: {final_eval['binary_mdd']['train_only_pr_auc']:.4f}")

    # Checkpoint path (local only, not committed to git)
    ckpt_dir = Path("checkpoints/RQ1-TRN-01B1")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"pilot_{variant_id.lower()}_step_{total_updates}.pt"
    torch.save({
        "variant_id": variant_id,
        "description": description,
        "model_state_dict": model.state_dict(),
        "total_updates": total_updates,
        "seed": seed
    }, ckpt_path)

    # Clean up GPU
    del model, opt, sched, scaler
    torch.cuda.empty_cache()

    return {
        "variant_id": variant_id,
        "description": description,
        "staged": staged,
        "freeze_backbone_throughout": freeze_backbone_throughout,
        "initial_blank_bias": initial_blank_bias,
        "lr_backbone": lr_backbone,
        "lr_head": lr_head,
        "stage1_updates": stage1_updates,
        "total_updates": total_updates,
        "seed": seed,
        "runtime_seconds": round(elapsed, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "amp_overflow_events": amp_overflow_events,
        "initial_metrics": eval_0,
        "final_metrics": final_eval,
        "step_history": step_history
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run CTC stabilization ablations")
    parser.add_argument("--variant", choices=["all", "Variant_A", "Variant_B", "Variant_C", "Variant_D", "merge"], default="all")
    args = parser.parse_args()

    print("=" * 80)
    print(f"LANGFENS RQ1-TRN-01B1: CTC STABILIZATION ABLATION CAMPAIGN (Target: {args.variant})")
    print("=" * 80)

    out_dir = Path("manifests/RQ1-TRN-01B1")
    out_dir.mkdir(parents=True, exist_ok=True)

    variant_configs = {
        "Variant_A": {
            "variant_id": "Variant_A",
            "description": "Historical Training Configuration (Control Baseline): Unfrozen backbone, blank_bias=0.0, AdamW lr=1e-4",
            "staged": False,
            "freeze_backbone_throughout": False,
            "initial_blank_bias": 0.0,
            "lr_backbone": 1e-4,
            "lr_head": 1e-4,
            "total_updates": 100
        },
        "Variant_B": {
            "variant_id": "Variant_B",
            "description": "Negative Initial CTC Blank Bias (-3.0): Unfrozen backbone, blank_bias=-3.0, AdamW lr=1e-4",
            "staged": False,
            "freeze_backbone_throughout": False,
            "initial_blank_bias": -3.0,
            "lr_backbone": 1e-4,
            "lr_head": 1e-4,
            "total_updates": 100
        },
        "Variant_C": {
            "variant_id": "Variant_C",
            "description": "Staged Optimization: Stage 1 (Updates 1..50) Frozen backbone, heads lr=1e-3; Stage 2 (Updates 51..100) Unfrozen encoder, backbone lr=2e-5, heads lr=2e-4",
            "staged": True,
            "freeze_backbone_throughout": False,
            "initial_blank_bias": 0.0,
            "lr_backbone": 0.0,
            "lr_head": 1e-3,
            "stage1_updates": 50,
            "total_updates": 100,
            "stage2_lr_backbone": 2e-5,
            "stage2_lr_head": 2e-4
        },
        "Variant_D": {
            "variant_id": "Variant_D",
            "description": "Frozen Backbone Probe: Pretrained backbone frozen throughout 100 updates, heads lr=1e-3",
            "staged": False,
            "freeze_backbone_throughout": True,
            "initial_blank_bias": 0.0,
            "lr_backbone": 0.0,
            "lr_head": 1e-3,
            "total_updates": 100
        }
    }

    if args.variant in variant_configs:
        cfg = variant_configs[args.variant]
        res = run_variant(**cfg)
        single_file = out_dir / f"metrics_{args.variant}.json"
        with open(single_file, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"\n[Saved {args.variant} metrics to {single_file}]")

    elif args.variant == "all":
        results = {}
        for vid, cfg in variant_configs.items():
            res = run_variant(**cfg)
            results[vid] = res
            single_file = out_dir / f"metrics_{vid}.json"
            with open(single_file, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)

        out_file = out_dir / "CTC_PILOT_METRICS.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[All ablations saved to {out_file}]")

    if args.variant == "merge" or args.variant in variant_configs:
        # Check if all 4 exist and merge
        all_exist = all((out_dir / f"metrics_{vid}.json").exists() for vid in variant_configs)
        if all_exist:
            combined = {}
            for vid in variant_configs:
                with open(out_dir / f"metrics_{vid}.json", "r", encoding="utf-8") as f:
                    combined[vid] = json.load(f)
            out_file = out_dir / "CTC_PILOT_METRICS.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(combined, f, indent=2)
            print(f"\n[Merged all metrics into {out_file}]")


if __name__ == "__main__":
    main()
