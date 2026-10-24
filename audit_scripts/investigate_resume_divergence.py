#!/usr/bin/env python3
"""
Deep-dive diagnostic investigation of Checkpoint Resume Divergence (Task B).
Examines:
- Exact minibatch/utterance ID sequence and input tensor hashes
- Dataloader sampler position
- Gradient accumulation boundaries
- Python, NumPy, CPU, and CUDA RNG states
- Optimizer and scheduler states
- GradScaler behavior and scale history
- Model train/eval mode and Dropout behavior
- CUDA nondeterministic operations (ctc_loss backward, attention, etc.)
"""

import copy
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Any, List

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
from audit_scripts.run_training_sanity import TrainSubsetDataset


def tensor_hash(t: torch.Tensor) -> str:
    """Returns SHA256 of raw tensor bytes on CPU."""
    t_cpu = t.detach().cpu().contiguous()
    return hashlib.sha256(t_cpu.numpy().tobytes()).hexdigest()[:16]


def set_strict_determinism(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def detailed_resume_investigation(
    total_steps: int = 10,
    save_step: int = 5,
    seed: int = 1337,
    device: str = "cuda"
):
    print("=" * 80)
    print("RUNNING DETAILED RESUME DIVERGENCE AUDIT")
    print("=" * 80)

    ds = TrainSubsetDataset()
    collate_fn = LangfensMDDCollate(pad_token_id=-100, ctc_blank_id=0)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
    batches = list(loader)
    num_batches = len(batches)

    grad_accum_steps = 2

    # Track microstep logs
    run_a_micro = {}
    run_b_micro = {}

    # Track step logs
    run_a_steps = {}
    run_b_steps = {}

    # RUN A
    print("\n--- Executing Run A (Uninterrupted) ---")
    set_strict_determinism(seed)
    model_a = LangfensWav2Vec2MDD.from_pretrained(
        DEFAULT_PRETRAINED_CHECKPOINT,
        revision=DEFAULT_PRETRAINED_REVISION,
        alignment_mode="cross_attention",
        ctc_zero_infinity_policy="raise"
    )
    model_a.freeze_feature_encoder()
    model_a.to(device)
    model_a.train()

    opt_a = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_a.parameters()),
        lr=1e-4,
        weight_decay=0.01
    )
    scaler_a = torch.amp.GradScaler("cuda", init_scale=16384.0)
    sched_a = get_cosine_schedule_with_warmup(opt_a, num_warmup_steps=2, num_training_steps=total_steps)

    micro_a = 0
    updates_a = 0
    accum_loss_a = 0.0
    checkpoint_step_5 = None
    batch_idx_a = 0

    while updates_a < total_steps:
        cur_batch_idx = batch_idx_a % num_batches
        batch = batches[cur_batch_idx]
        batch_idx_a += 1
        micro_a += 1

        utt_ids = batch.get("metadata", {}).get("utterance_ids", [
            ds.entries[cur_batch_idx*2]["utterance_id"],
            ds.entries[min(cur_batch_idx*2+1, len(ds.entries)-1)]["utterance_id"]
        ])
        in_hash = tensor_hash(batch["net_input"]["source"])
        tgt_hash = tensor_hash(batch["target_ctc"])

        net_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["net_input"].items()}
        tb = batch["target_binary"].to(device)
        tc = batch["target_ctc"].to(device)
        tl = batch["target_lengths"].to(device)
        td = batch["target_diagnosis"].to(device)

        cuda_rng_before = torch.cuda.get_rng_state(device).numpy().tobytes()
        cuda_rng_hash = hashlib.sha256(cuda_rng_before).hexdigest()[:12]

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            out = model_a(**net_input, target_binary=tb, target_ctc=tc, target_lengths=tl, target_diagnosis=td)
            raw_loss = out.loss
            scaled_loss = raw_loss / grad_accum_steps

        scaler_a.scale(scaled_loss).backward()
        accum_loss_a += raw_loss.item()

        ctc_head_grad_norm = model_a.ctc_head.weight.grad.norm().item() if model_a.ctc_head.weight.grad is not None else 0.0

        run_a_micro[micro_a] = {
            "micro_step": micro_a,
            "cur_batch_idx": cur_batch_idx,
            "utt_ids": utt_ids,
            "input_hash": in_hash,
            "tgt_hash": tgt_hash,
            "cuda_rng_hash": cuda_rng_hash,
            "raw_loss": raw_loss.item(),
            "ctc_loss": out.loss_dict.get("ctc_loss", torch.tensor(0.0)).item(),
            "bin_loss": out.loss_dict.get("binary_loss", torch.tensor(0.0)).item(),
            "diag_loss": out.loss_dict.get("diagnosis_loss", torch.tensor(0.0)).item(),
            "ctc_head_grad_norm": ctc_head_grad_norm
        }

        if micro_a % grad_accum_steps == 0:
            scaler_a.unscale_(opt_a)
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model_a.parameters()), max_norm=1.0
            ).item()
            old_scale = scaler_a.get_scale()
            scaler_a.step(opt_a)
            scaler_a.update()
            new_scale = scaler_a.get_scale()

            if new_scale >= old_scale:
                updates_a += 1
                sched_a.step()
                step_loss = accum_loss_a / grad_accum_steps
                ctc_w_hash = tensor_hash(model_a.ctc_head.weight)
                run_a_steps[updates_a] = {
                    "step": updates_a,
                    "micro_step": micro_a,
                    "loss": step_loss,
                    "grad_norm": total_grad_norm,
                    "scale": new_scale,
                    "lr": sched_a.get_last_lr()[0],
                    "ctc_w_hash": ctc_w_hash,
                    "ctc_w_tensor": model_a.ctc_head.weight.clone().detach().cpu()
                }
                print(f"  [Run A] Step {updates_a:2d} (Micro {micro_a:2d}) | Loss: {step_loss:.8f} | Scale: {new_scale} | GradNorm: {total_grad_norm:.4f} | CTC-W-Hash: {ctc_w_hash}")

                if updates_a == save_step:
                    checkpoint_step_5 = {
                        "model_state_dict": copy.deepcopy(model_a.state_dict()),
                        "optimizer_state_dict": copy.deepcopy(opt_a.state_dict()),
                        "scaler_state_dict": copy.deepcopy(scaler_a.state_dict()),
                        "scheduler_state_dict": copy.deepcopy(sched_a.state_dict()),
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        "numpy_rng_state": np.random.get_state(),
                        "python_rng_state": random.getstate(),
                        "updates": updates_a,
                        "micro_step": micro_a,
                        "batch_idx": batch_idx_a
                    }

            accum_loss_a = 0.0
            opt_a.zero_grad()

    # Clean up Run A
    del model_a, opt_a, scaler_a, sched_a
    torch.cuda.empty_cache()

    # RUN B
    print("\n--- Executing Run B (Resuming from Step 5) ---")
    model_b = LangfensWav2Vec2MDD.from_pretrained(
        DEFAULT_PRETRAINED_CHECKPOINT,
        revision=DEFAULT_PRETRAINED_REVISION,
        alignment_mode="cross_attention",
        ctc_zero_infinity_policy="raise"
    )
    model_b.freeze_feature_encoder()
    model_b.load_state_dict(checkpoint_step_5["model_state_dict"])
    model_b.to(device)
    model_b.train()

    opt_b = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_b.parameters()),
        lr=1e-4,
        weight_decay=0.01
    )
    opt_b.load_state_dict(checkpoint_step_5["optimizer_state_dict"])

    scaler_b = torch.amp.GradScaler("cuda")
    scaler_b.load_state_dict(checkpoint_step_5["scaler_state_dict"])

    sched_b = get_cosine_schedule_with_warmup(opt_b, num_warmup_steps=2, num_training_steps=total_steps)
    sched_b.load_state_dict(checkpoint_step_5["scheduler_state_dict"])

    # Restore RNG states
    torch.set_rng_state(checkpoint_step_5["torch_rng_state"])
    if checkpoint_step_5["cuda_rng_state"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint_step_5["cuda_rng_state"])
    np.random.set_state(checkpoint_step_5["numpy_rng_state"])
    random.setstate(checkpoint_step_5["python_rng_state"])

    updates_b = checkpoint_step_5["updates"]
    micro_b = checkpoint_step_5["micro_step"]
    batch_idx_b = checkpoint_step_5["batch_idx"]
    accum_loss_b = 0.0

    while updates_b < total_steps:
        cur_batch_idx = batch_idx_b % num_batches
        batch = batches[cur_batch_idx]
        batch_idx_b += 1
        micro_b += 1

        utt_ids = batch.get("metadata", {}).get("utterance_ids", [
            ds.entries[cur_batch_idx*2]["utterance_id"],
            ds.entries[min(cur_batch_idx*2+1, len(ds.entries)-1)]["utterance_id"]
        ])
        in_hash = tensor_hash(batch["net_input"]["source"])
        tgt_hash = tensor_hash(batch["target_ctc"])

        net_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["net_input"].items()}
        tb = batch["target_binary"].to(device)
        tc = batch["target_ctc"].to(device)
        tl = batch["target_lengths"].to(device)
        td = batch["target_diagnosis"].to(device)

        cuda_rng_before = torch.cuda.get_rng_state(device).numpy().tobytes()
        cuda_rng_hash = hashlib.sha256(cuda_rng_before).hexdigest()[:12]

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            out = model_b(**net_input, target_binary=tb, target_ctc=tc, target_lengths=tl, target_diagnosis=td)
            raw_loss = out.loss
            scaled_loss = raw_loss / grad_accum_steps

        scaler_b.scale(scaled_loss).backward()
        accum_loss_b += raw_loss.item()

        ctc_head_grad_norm = model_b.ctc_head.weight.grad.norm().item() if model_b.ctc_head.weight.grad is not None else 0.0

        run_b_micro[micro_b] = {
            "micro_step": micro_b,
            "cur_batch_idx": cur_batch_idx,
            "utt_ids": utt_ids,
            "input_hash": in_hash,
            "tgt_hash": tgt_hash,
            "cuda_rng_hash": cuda_rng_hash,
            "raw_loss": raw_loss.item(),
            "ctc_loss": out.loss_dict.get("ctc_loss", torch.tensor(0.0)).item(),
            "bin_loss": out.loss_dict.get("binary_loss", torch.tensor(0.0)).item(),
            "diag_loss": out.loss_dict.get("diagnosis_loss", torch.tensor(0.0)).item(),
            "ctc_head_grad_norm": ctc_head_grad_norm
        }

        if micro_b % grad_accum_steps == 0:
            scaler_b.unscale_(opt_b)
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model_b.parameters()), max_norm=1.0
            ).item()
            old_scale = scaler_b.get_scale()
            scaler_b.step(opt_b)
            scaler_b.update()
            new_scale = scaler_b.get_scale()

            if new_scale >= old_scale:
                updates_b += 1
                sched_b.step()
                step_loss = accum_loss_b / grad_accum_steps
                ctc_w_hash = tensor_hash(model_b.ctc_head.weight)
                run_b_steps[updates_b] = {
                    "step": updates_b,
                    "micro_step": micro_b,
                    "loss": step_loss,
                    "grad_norm": total_grad_norm,
                    "scale": new_scale,
                    "lr": sched_b.get_last_lr()[0],
                    "ctc_w_hash": ctc_w_hash,
                    "ctc_w_tensor": model_b.ctc_head.weight.clone().detach().cpu()
                }
                print(f"  [Run B] Step {updates_b:2d} (Micro {micro_b:2d}) | Loss: {step_loss:.8f} | Scale: {new_scale} | GradNorm: {total_grad_norm:.4f} | CTC-W-Hash: {ctc_w_hash}")

            accum_loss_b = 0.0
            opt_b.zero_grad()

    # COMPARE MICROSTEPS
    print("\n" + "=" * 80)
    print("DETAILED MICROSTEP COMPARISON (Micro 11 to 20)")
    print("=" * 80)
    print(f"{'Micro':<6} {'Batch':<6} {'InHashMatch':<12} {'CUDARNGMatch':<13} {'LossA':<14} {'LossB':<14} {'LossDiff':<12}")
    for m in range(save_step * grad_accum_steps + 1, total_steps * grad_accum_steps + 1):
        ma = run_a_micro[m]
        mb = run_b_micro[m]
        in_match = (ma["input_hash"] == mb["input_hash"])
        rng_match = (ma["cuda_rng_hash"] == mb["cuda_rng_hash"])
        diff = abs(ma["raw_loss"] - mb["raw_loss"])
        print(f"{m:<6} {ma['cur_batch_idx']:<6} {str(in_match):<12} {str(rng_match):<13} {ma['raw_loss']:<14.8f} {mb['raw_loss']:<14.8f} {diff:<12.2e}")

    # COMPARE STEP UPDATES
    print("\n" + "=" * 80)
    print("STEP UPDATES COMPARISON (Steps 6 to 10)")
    print("=" * 80)
    print(f"{'Step':<6} {'Loss A':<14} {'Loss B':<14} {'Loss Diff':<12} {'GradNorm A':<12} {'GradNorm B':<12} {'Scale A/B':<10} {'CTC_W MaxDiff':<14}")
    for s in range(save_step + 1, total_steps + 1):
        sa = run_a_steps[s]
        sb = run_b_steps[s]
        diff = abs(sa["loss"] - sb["loss"])
        w_diff = (sa["ctc_w_tensor"] - sb["ctc_w_tensor"]).abs().max().item()
        print(f"{s:<6} {sa['loss']:<14.8f} {sb['loss']:<14.8f} {diff:<12.2e} {sa['grad_norm']:<12.6f} {sb['grad_norm']:<12.6f} {sa['scale']}/{sb['scale']:<6} {w_diff:<14.2e}")


if __name__ == "__main__":
    detailed_resume_investigation()
