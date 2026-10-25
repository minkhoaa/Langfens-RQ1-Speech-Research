#!/usr/bin/env python3
"""
Quick exploratory pilot script to test CTC stabilization ideas on 32 utterances.
Tests:
- Historical (control): Unfrozen backbone, blank_bias=0, lr=1e-4
- Negative blank bias alone: blank_bias=-3.0, unfrozen backbone, lr=1e-4
- Frozen backbone: freeze wav2vec2 completely, train only heads, lr=5e-4
- Differential LR: backbone lr=1e-5, heads lr=5e-4, blank_bias=-3.0
"""

import copy
import os
import random
import sys
from pathlib import Path

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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_ctc(model, loader, device="cuda"):
    model.eval()
    total_frames = 0
    blank_frames = 0
    total_loss = 0.0
    decoded_seqs = []
    total_ref_tokens = 0
    total_hyp_tokens = 0

    with torch.no_grad():
        for batch in loader:
            net_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["net_input"].items()}
            tc = batch["target_ctc"].to(device)
            tl = batch["target_lengths"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                out = model(**net_input, target_ctc=tc, target_lengths=tl)
                total_loss += out.loss_dict["ctc_loss"].item() * len(tl)

            in_lens = out.input_lengths.cpu().tolist()
            logits = out.ctc_logits  # (B, T, 40)

            for b_idx in range(len(in_lens)):
                T = in_lens[b_idx]
                act_logits = logits[b_idx, :T]
                argmax_c = act_logits.argmax(dim=-1).cpu().tolist()
                total_frames += T
                blank_frames += sum(1 for c in argmax_c if c == 0)

                # Collapse
                collapsed = []
                prev = None
                for c in argmax_c:
                    if c != prev:
                        if c != 0:
                            collapsed.append(c)
                        prev = c
                decoded_seqs.append(collapsed)
                total_hyp_tokens += len(collapsed)
            total_ref_tokens += sum(tl.cpu().tolist())

    non_empty = sum(1 for s in decoded_seqs if len(s) > 0)
    return {
        "ctc_loss": total_loss / len(loader.dataset),
        "total_frames": total_frames,
        "blank_fraction": blank_frames / total_frames if total_frames > 0 else 0.0,
        "total_hyp_tokens": total_hyp_tokens,
        "non_empty_utts": non_empty,
        "total_utts": len(loader.dataset),
        "sample_decoded": [[INDEX_TO_PHONE.get(idx-1, str(idx)) for idx in s] for s in decoded_seqs[:3]]
    }


def run_test(variant_name, freeze_backbone=False, initial_blank_bias=0.0, lr_backbone=1e-4, lr_head=1e-4, steps=40):
    print(f"\n=======================================================")
    print(f"RUNNING PILOT: {variant_name}")
    print(f"freeze_backbone={freeze_backbone}, blank_bias={initial_blank_bias}, lr_backbone={lr_backbone}, lr_head={lr_head}, steps={steps}")
    print(f"=======================================================")

    set_seed(42)
    device = "cuda"

    ds = TrainSubsetDataset()
    collate_fn = LangfensMDDCollate(pad_token_id=-100, ctc_blank_id=0)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
    batches = list(loader)

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

    if freeze_backbone:
        for p in model.wav2vec2.parameters():
            p.requires_grad = False

    model.to(device)

    # Param groups
    backbone_params = [p for n, p in model.wav2vec2.named_parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("wav2vec2") and p.requires_grad]

    param_groups = []
    if backbone_params and lr_backbone > 0:
        param_groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": 0.01})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr_head, "weight_decay": 0.01})

    opt = torch.optim.AdamW(param_groups)
    scaler = torch.amp.GradScaler("cuda", init_scale=16384.0)

    # Initial eval
    eval_0 = evaluate_ctc(model, loader, device)
    print(f"  Step  0: CTC Loss = {eval_0['ctc_loss']:.4f} | Blank Fraction = {eval_0['blank_fraction']*100:.2f}% | Hyp Tokens = {eval_0['total_hyp_tokens']} | Non-Empty = {eval_0['non_empty_utts']}/{eval_0['total_utts']}")
    if eval_0["sample_decoded"][0]:
        print(f"    Sample 0: {eval_0['sample_decoded'][0][:10]}")

    model.train()
    grad_accum_steps = 2
    micro_step = 0
    updates = 0
    accum_loss = 0.0
    batch_idx = 0

    while updates < steps:
        batch = batches[batch_idx % len(batches)]
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
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            old_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            new_scale = scaler.get_scale()

            if new_scale >= old_scale:
                updates += 1
                if updates % 10 == 0:
                    eval_step = evaluate_ctc(model, loader, device)
                    model.train()
                    print(f"  Step {updates:2d}: CTC Loss = {eval_step['ctc_loss']:.4f} | Blank Fraction = {eval_step['blank_fraction']*100:.2f}% | Hyp Tokens = {eval_step['total_hyp_tokens']} | Non-Empty = {eval_step['non_empty_utts']}/{eval_step['total_utts']}")
                    if eval_step["sample_decoded"][0]:
                        print(f"    Sample 0: {eval_step['sample_decoded'][0][:10]}")

            accum_loss = 0.0
            opt.zero_grad()


if __name__ == "__main__":
    # Test 1: Historical Control
    run_test("Variant A (Control)", freeze_backbone=False, initial_blank_bias=0.0, lr_backbone=1e-4, lr_head=1e-4, steps=30)
    # Test 2: Negative Blank Bias Alone
    run_test("Variant B (Neg Blank Bias -3.0)", freeze_backbone=False, initial_blank_bias=-3.0, lr_backbone=1e-4, lr_head=1e-4, steps=30)
    # Test 3: Frozen Backbone Alone
    run_test("Variant C1 (Frozen Backbone, Head lr=5e-4)", freeze_backbone=True, initial_blank_bias=0.0, lr_backbone=0.0, lr_head=5e-4, steps=30)
    # Test 4: Frozen Backbone + Neg Blank Bias
    run_test("Variant C2 (Frozen Backbone + Neg Blank Bias -3.0)", freeze_backbone=True, initial_blank_bias=-3.0, lr_backbone=0.0, lr_head=5e-4, steps=30)
