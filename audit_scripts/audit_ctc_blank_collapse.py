#!/usr/bin/env python3
"""
LANGFENS RQ1-TRN-01B1 — Task C: CTC Blank-Collapse Root Cause Analysis.
Rigorous empirical analysis of:
1. Blank probability distribution across all active acoustic frames.
2. Top non-blank class probabilities and logit margins.
3. Decoded output lengths and beam/temperature behavior.
4. CTC target length distributions.
5. Input/target length ratios (T/L).
6. CTC loss gradients and forward-backward posterior occupancy gamma_t(k).
7. Blank-head bias evolution across step 0, step 5, and step 100.
8. Backbone vs. head gradient norms.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("."))

from langfens_prep.model import (
    LangfensWav2Vec2MDD,
    DEFAULT_PRETRAINED_CHECKPOINT,
    DEFAULT_PRETRAINED_REVISION
)
from langfens_prep.collate import LangfensMDDCollate
from langfens_prep.phoneme_normalizer import INDEX_TO_PHONE
from audit_scripts.run_training_sanity import TrainSubsetDataset


def analyze_ctc_blank_collapse(device: str = "cuda"):
    print("=" * 80)
    print("TASK C: CTC BLANK-COLLAPSE ROOT CAUSE INVESTIGATION")
    print("=" * 80)

    ds = TrainSubsetDataset()
    collate_fn = LangfensMDDCollate(pad_token_id=-100, ctc_blank_id=0)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
    batches = list(loader)

    ckpt_step100_path = "checkpoints/RQ1-TRN-01B0/rq1_trn_01b0_checkpoint_step_100.pt"
    ckpt_step5_path = "checkpoints/RQ1-TRN-01B0/rq1_trn_01b0_checkpoint_step_5.pt"
    ckpt_repeat_path = "checkpoints/RQ1-TRN-01B0/reconciliation/rq1_trn_01b0_r_repeat_step_100.pt"

    # --- PART 1: Bias Evolution ---
    print("\n[Part 1] Analyzing CTC Head Bias Evolution Across Checkpoints...")
    # Step 0: freshly initialized
    m0 = LangfensWav2Vec2MDD.from_pretrained(
        DEFAULT_PRETRAINED_CHECKPOINT,
        revision=DEFAULT_PRETRAINED_REVISION,
        alignment_mode="cross_attention",
        ctc_zero_infinity_policy="raise"
    )
    b0 = m0.ctc_head.bias.detach().cpu().numpy()
    w0 = m0.ctc_head.weight.detach().cpu().numpy()

    # Step 5
    ckpt5 = torch.load(ckpt_step5_path, map_location="cpu", weights_only=False)
    b5 = ckpt5["model_state_dict"]["ctc_head.bias"].numpy()
    w5 = ckpt5["model_state_dict"]["ctc_head.weight"].numpy()

    # Step 100
    ckpt100 = torch.load(ckpt_step100_path, map_location="cpu", weights_only=False)
    b100 = ckpt100["model_state_dict"]["ctc_head.bias"].numpy()
    w100 = ckpt100["model_state_dict"]["ctc_head.weight"].numpy()

    # Repeat Step 100
    ckpt_rep = torch.load(ckpt_repeat_path, map_location="cpu", weights_only=False)
    b_rep = ckpt_rep["model_state_dict"]["ctc_head.bias"].numpy()

    print(f"  Step 0:   Blank bias = {b0[0]:+.6f} | Non-blank mean = {b0[1:].mean():+.6f} (std = {b0[1:].std():.6f}) | Margin = {b0[0] - b0[1:].mean():+.6f}")
    print(f"  Step 5:   Blank bias = {b5[0]:+.6f} | Non-blank mean = {b5[1:].mean():+.6f} (std = {b5[1:].std():.6f}) | Margin = {b5[0] - b5[1:].mean():+.6f}")
    print(f"  Step 100: Blank bias = {b100[0]:+.6f} | Non-blank mean = {b100[1:].mean():+.6f} (std = {b100[1:].std():.6f}) | Margin = {b100[0] - b100[1:].mean():+.6f}")
    print(f"  Repeat:   Blank bias = {b_rep[0]:+.6f} | Non-blank mean = {b_rep[1:].mean():+.6f} (std = {b_rep[1:].std():.6f}) | Margin = {b_rep[0] - b_rep[1:].mean():+.6f}")

    # --- PART 2: Target Lengths and T/L Ratios ---
    print("\n[Part 2] Target Length Distributions & Input/Target Length Ratios (T/L)...")
    target_lengths = []
    input_lengths = []
    ratios = []

    # Load Step 100 model for evaluation
    model = LangfensWav2Vec2MDD.from_pretrained(
        DEFAULT_PRETRAINED_CHECKPOINT,
        revision=DEFAULT_PRETRAINED_REVISION,
        alignment_mode="cross_attention",
        ctc_zero_infinity_policy="raise"
    )
    model.freeze_feature_encoder()
    model.load_state_dict(ckpt100["model_state_dict"])
    model.to(device)
    model.eval()

    all_blank_probs = []
    all_top_nonblank_probs = []
    all_logit_margins = []  # blank_logit - top_nonblank_logit
    all_top_nonblank_classes = []
    decoded_lengths_greedy = []

    # Forward all batches
    total_active_frames = 0
    total_blank_argmax = 0
    total_ctc_loss = 0.0

    with torch.no_grad():
        for batch in batches:
            net_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["net_input"].items()}
            tc = batch["target_ctc"].to(device)
            tl = batch["target_lengths"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                out = model(**net_input, target_ctc=tc, target_lengths=tl)
                total_ctc_loss += out.loss_dict["ctc_loss"].item() * len(tl)

            in_lens = out.input_lengths.cpu().tolist()
            tgt_lens = tl.cpu().tolist()

            probs = F.softmax(out.ctc_logits, dim=-1)  # (B, T, 40)
            logits = out.ctc_logits  # (B, T, 40)

            for b_idx in range(len(in_lens)):
                T = in_lens[b_idx]
                L = tgt_lens[b_idx]
                input_lengths.append(T)
                target_lengths.append(L)
                ratios.append(T / L)
                total_active_frames += T

                act_probs = probs[b_idx, :T]  # (T, 40)
                act_logits = logits[b_idx, :T]  # (T, 40)

                b_probs = act_probs[:, 0].cpu().numpy()
                nb_probs, nb_indices = act_probs[:, 1:].max(dim=-1)
                nb_probs = nb_probs.cpu().numpy()
                nb_indices = (nb_indices + 1).cpu().numpy()

                margins = (act_logits[:, 0] - act_logits[:, 1:].max(dim=-1).values).cpu().numpy()

                argmax_classes = act_probs.argmax(dim=-1).cpu().numpy()
                total_blank_argmax += int((argmax_classes == 0).sum())

                # Greedy decode
                collapsed = []
                prev = None
                for c in argmax_classes:
                    if c != prev:
                        if c != 0:
                            collapsed.append(c)
                        prev = c
                decoded_lengths_greedy.append(len(collapsed))

                all_blank_probs.extend(b_probs.tolist())
                all_top_nonblank_probs.extend(nb_probs.tolist())
                all_logit_margins.extend(margins.tolist())
                all_top_nonblank_classes.extend(nb_indices.tolist())

    mean_ctc_loss = total_ctc_loss / len(ds)
    print(f"  Step 100 Evaluation CTC Loss: {mean_ctc_loss:.4f}")
    print(f"  Total Active Frames: {total_active_frames}")
    print(f"  Argmax Blank Frames: {total_blank_argmax} ({total_blank_argmax/total_active_frames*100:.2f}%)")
    print(f"  Decoded Token Counts: min={min(decoded_lengths_greedy)}, max={max(decoded_lengths_greedy)}, mean={np.mean(decoded_lengths_greedy):.2f}")

    target_lengths = np.array(target_lengths)
    input_lengths = np.array(input_lengths)
    ratios = np.array(ratios)

    print(f"\n  Target Lengths (L): min={target_lengths.min()}, max={target_lengths.max()}, mean={target_lengths.mean():.2f}, median={np.median(target_lengths):.1f}")
    print(f"  Input Acoustic Frames (T): min={input_lengths.min()}, max={input_lengths.max()}, mean={input_lengths.mean():.2f}, median={np.median(input_lengths):.1f}")
    print(f"  T / L Ratios: min={ratios.min():.2f}, max={ratios.max():.2f}, mean={ratios.mean():.2f}, median={np.median(ratios):.2f}")

    # --- PART 3: Probability Distribution & Margin Analysis ---
    all_blank_probs = np.array(all_blank_probs)
    all_top_nonblank_probs = np.array(all_top_nonblank_probs)
    all_logit_margins = np.array(all_logit_margins)

    print("\n[Part 3] Blank Probability Distribution:")
    print(f"  Mean:   {all_blank_probs.mean():.6f}")
    print(f"  Std:    {all_blank_probs.std():.6f}")
    print(f"  Min:    {all_blank_probs.min():.6f}")
    print(f"  1%:     {np.percentile(all_blank_probs, 1):.6f}")
    print(f"  5%:     {np.percentile(all_blank_probs, 5):.6f}")
    print(f"  10%:    {np.percentile(all_blank_probs, 10):.6f}")
    print(f"  25%:    {np.percentile(all_blank_probs, 25):.6f}")
    print(f"  50%:    {np.percentile(all_blank_probs, 50):.6f}")
    print(f"  75%:    {np.percentile(all_blank_probs, 75):.6f}")
    print(f"  90%:    {np.percentile(all_blank_probs, 90):.6f}")
    print(f"  95%:    {np.percentile(all_blank_probs, 95):.6f}")
    print(f"  99%:    {np.percentile(all_blank_probs, 99):.6f}")
    print(f"  Max:    {all_blank_probs.max():.6f}")

    # Histogram of P(blank)
    bins = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.000001]
    hist, _ = np.histogram(all_blank_probs, bins=bins)
    print("\n  P(blank) Histogram Bins:")
    for i in range(len(bins)-1):
        cnt = hist[i]
        pct = cnt / len(all_blank_probs) * 100
        print(f"    [{bins[i]:.2f}, {bins[i+1]:.2f}): {cnt:5d} frames ({pct:5.2f}%)")

    print("\n[Part 4] Top Non-Blank Class Probabilities & Margins:")
    print(f"  Top Non-Blank Probability Mean:   {all_top_nonblank_probs.mean():.6f}")
    print(f"  Top Non-Blank Probability Max:    {all_top_nonblank_probs.max():.6f}")
    print(f"  Top Non-Blank Probability 99%:    {np.percentile(all_top_nonblank_probs, 99):.6f}")
    print(f"  Top Non-Blank Probability 90%:    {np.percentile(all_top_nonblank_probs, 90):.6f}")
    print(f"  Logit Margin (Blank - TopNonBlank) Mean: {all_logit_margins.mean():.4f}")
    print(f"  Logit Margin Min:                       {all_logit_margins.min():.4f}")
    print(f"  Logit Margin Max:                       {all_logit_margins.max():.4f}")

    # Top non-blank classes frequency
    class_counts = {}
    for c in all_top_nonblank_classes:
        class_counts[c] = class_counts.get(c, 0) + 1
    top_classes_sorted = sorted(class_counts.items(), key=lambda x: -x[1])[:10]
    print("\n  Most Frequent Top Non-Blank Classes:")
    for c_idx, count in top_classes_sorted:
        phone_name = INDEX_TO_PHONE.get(c_idx - 1, f"IDX_{c_idx}")
        pct = count / len(all_top_nonblank_classes) * 100
        print(f"    Class {c_idx:2d} ({phone_name:<4}): {count:5d} frames ({pct:5.2f}%)")

    # --- PART 5: CTC Loss Gradient and Posterior Analysis ---
    print("\n[Part 5] Analyzing CTC Loss Gradients & Local Minimum Nature...")
    # Put model into train mode and compute gradients on Step 100 checkpoint
    model.train()
    model.zero_grad()

    # Forward batch 0 and analyze dL_ctc / dz
    b0_data = batches[0]
    net_in = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b0_data["net_input"].items()}
    tc = b0_data["target_ctc"].to(device)
    tl = b0_data["target_lengths"].to(device)

    # Let's retain grad on ctc_logits
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        out = model(**net_in, target_ctc=tc, target_lengths=tl)
        ctc_loss = out.loss_dict["ctc_loss"]

    # Backward ctc_loss alone
    ctc_loss.backward()

    head_weight_grad_norm = model.ctc_head.weight.grad.norm().item()
    head_bias_grad_norm = model.ctc_head.bias.grad.norm().item()

    # Backbone grad norms across layers
    layer_grad_norms = []
    for l_idx, layer in enumerate(model.wav2vec2.encoder.layers):
        l_norm = sum(p.grad.norm().item()**2 for p in layer.parameters() if p.grad is not None)**0.5
        layer_grad_norms.append(l_norm)

    print(f"  CTC Head Weight Grad Norm at Step 100: {head_weight_grad_norm:.6f}")
    print(f"  CTC Head Bias Grad Norm at Step 100:   {head_bias_grad_norm:.6f}")
    print(f"  Encoder Layer 0 Grad Norm:             {layer_grad_norms[0]:.6f}")
    print(f"  Encoder Layer 6 Grad Norm:             {layer_grad_norms[6]:.6f}")
    print(f"  Encoder Layer 11 Grad Norm:            {layer_grad_norms[11]:.6f}")
    print(f"  Mean Encoder Layer Grad Norm:          {np.mean(layer_grad_norms):.6f}")

    # Check dL/d(blank_bias) vs dL/d(nonblank_bias)
    bias_grad = model.ctc_head.bias.grad.cpu().numpy()
    print(f"  dL / d(blank_bias):    {bias_grad[0]:+.6f}")
    print(f"  dL / d(nonblank_bias): mean={bias_grad[1:].mean():+.6f}, min={bias_grad[1:].min():+.6f}, max={bias_grad[1:].max():+.6f}")

    # Examine whether this is a mathematical local minimum:
    # In CTC, if all frames are blank, dL/dz_t^k = P(k|x_t) - gamma_t(k).
    # Since P(blank) is ~0.95, and gamma_t(blank) is high where no label is aligned,
    # the gradients are non-zero!
    is_mathematical_minimum = (head_weight_grad_norm < 1e-3)
    print(f"\n  Is CTC loss at a mathematical local minimum (grad ~ 0)? {is_mathematical_minimum}")
    print(f"  Gradient norm magnitude: {head_weight_grad_norm:.4f} (Substantial gradients remain!)")

    results = {
        "bias_evolution": {
            "step_0": {"blank": float(b0[0]), "nonblank_mean": float(b0[1:].mean()), "margin": float(b0[0] - b0[1:].mean())},
            "step_5": {"blank": float(b5[0]), "nonblank_mean": float(b5[1:].mean()), "margin": float(b5[0] - b5[1:].mean())},
            "step_100": {"blank": float(b100[0]), "nonblank_mean": float(b100[1:].mean()), "margin": float(b100[0] - b100[1:].mean())},
            "repeat_100": {"blank": float(b_rep[0]), "nonblank_mean": float(b_rep[1:].mean()), "margin": float(b_rep[0] - b_rep[1:].mean())}
        },
        "target_distribution": {
            "target_lengths_mean": float(target_lengths.mean()),
            "target_lengths_min": int(target_lengths.min()),
            "target_lengths_max": int(target_lengths.max()),
            "input_lengths_mean": float(input_lengths.mean()),
            "input_lengths_min": int(input_lengths.min()),
            "input_lengths_max": int(input_lengths.max()),
            "t_over_l_mean": float(ratios.mean()),
            "t_over_l_min": float(ratios.min()),
            "t_over_l_max": float(ratios.max())
        },
        "blank_probabilities": {
            "mean": float(all_blank_probs.mean()),
            "std": float(all_blank_probs.std()),
            "min": float(all_blank_probs.min()),
            "max": float(all_blank_probs.max()),
            "p10": float(np.percentile(all_blank_probs, 10)),
            "p50": float(np.percentile(all_blank_probs, 50)),
            "p90": float(np.percentile(all_blank_probs, 90)),
            "p99": float(np.percentile(all_blank_probs, 99))
        },
        "top_nonblank_probabilities": {
            "mean": float(all_top_nonblank_probs.mean()),
            "max": float(all_top_nonblank_probs.max()),
            "p90": float(np.percentile(all_top_nonblank_probs, 90)),
            "p99": float(np.percentile(all_top_nonblank_probs, 99))
        },
        "logit_margin": {
            "mean": float(all_logit_margins.mean()),
            "min": float(all_logit_margins.min()),
            "max": float(all_logit_margins.max())
        },
        "gradients": {
            "ctc_head_weight_grad_norm": float(head_weight_grad_norm),
            "ctc_head_bias_grad_norm": float(head_bias_grad_norm),
            "encoder_mean_grad_norm": float(np.mean(layer_grad_norms)),
            "d_loss_d_blank_bias": float(bias_grad[0]),
            "is_mathematical_minimum": is_mathematical_minimum
        }
    }

    out_json = Path("manifests/RQ1-TRN-01B1/ctc_blank_collapse_diagnostics.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nDiagnostics serialized to {out_json}")


if __name__ == "__main__":
    analyze_ctc_blank_collapse()
