"""LANGFENS RQ1-TRN-04 — Factorial Training Runner.

Executes controlled 2x2 factorial training across four matched arms:
- C0: Cross-Entropy loss, acoustic features disabled (zeros)
- T1: Class-Conditioned Focal loss, acoustic features disabled (zeros)
- T2: Cross-Entropy loss, acoustic features enabled (waveform features)
- T3: Class-Conditioned Focal loss, acoustic features enabled (waveform features)

Enforces:
- Training on FIT only (1,799 utterances, 12 speakers)
- Checkpoint selection on SELECT only (300 utterances, MBMPS/TLV)
- Calibration on CAL only (300 utterances, NCC/THV)
- Preserves VAL strictly unseen (300 utterances, NJS/RRBI)
- Sealed TEST speakers strictly prohibited (fail-closed)
- Pinned seed: 1337
- Batch size 2, gradient accumulation 2 (effective batch 4)
- Fixed budget: 3,600 successful updates
- Cosine schedule with 100 warmup updates over 15,000 horizon
- AdamW lr 0.001, weight decay 0.01, gradient clipping 1.0
- Dual checkpointing: fixed_budget_3600.pt and best.pt
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model

from langfens_prep.trn04.features import (
    extract_frame_acoustic_features,
    ACOUSTIC_FEATURE_DIM,
    FIT_LOG_ENERGY_MEAN,
    FIT_LOG_ENERGY_STD
)
from langfens_prep.trn04.model import (
    CompactFactorialOperationHead,
    FactorialHeadConfig,
    FactorialAudioAdapter,
    parameter_inventory
)
from langfens_prep.trn04.objectives import task_loss, realized_ctc_loss
from langfens_prep.p_sup.planning import learning_rate_multiplier
from langfens_prep.p_sup.dataset import (
    PSupL2ArcticDataset,
    collate_psup_batch,
    FIT_SPEAKERS,
    SELECT_SPEAKERS,
    CALIBRATION_SPEAKERS,
    VAL_SPEAKERS,
    SealedSpeakerContaminationError
)
from langfens_prep.p_sup.trainer import (
    seed_all,
    rng_state,
    restore_rng,
    compute_file_sha256,
    RemoteCheckpointMirror,
    average_precision_binary,
    greedy_ctc_decode,
    compute_levenshtein_edits,
    FrozenWav2Vec2AudioEncoder,
    DEFAULT_PRETRAINED_CHECKPOINT,
    DEFAULT_PRETRAINED_REVISION
)
from langfens_prep.p_sup.runtime import CONTRACT_SHA256, SEALED_SPEAKERS

ARM_CONFIGS = {
    "C0": {"objective": "operation_ce", "features_enabled": False, "description": "CE Loss, Acoustic Features Disabled"},
    "T1": {"objective": "class_focal", "features_enabled": False, "description": "Class Focal Loss, Acoustic Features Disabled"},
    "T2": {"objective": "operation_ce", "features_enabled": True, "description": "CE Loss, Acoustic Features Enabled"},
    "T3": {"objective": "class_focal", "features_enabled": True, "description": "Class Focal Loss, Acoustic Features Enabled"},
}


def evaluate_select_set(
    adapter: FactorialAudioAdapter,
    loader: DataLoader,
    device: torch.device,
    objective: str
) -> Dict[str, Any]:
    """Evaluates the SELECT partition for checkpoint selection and early stopping."""
    adapter.eval()
    all_binary_labels = []
    all_scores = []
    total_ctc_frames = 0
    total_blank_frames = 0
    total_ref_phones = 0
    total_edits = 0

    with torch.no_grad():
        for batch in loader:
            audio = batch["waveforms"].to(device)
            audio_lengths = batch["audio_lengths"].to(device)
            prompt_ids = batch["prompt_phone_ids"].to(device)
            prompt_lengths = batch["prompt_lengths"].to(device)

            out = adapter(
                audio=audio,
                audio_lengths=audio_lengths,
                prompt_phone_ids=prompt_ids,
                prompt_lengths=prompt_lengths
            )

            op_logits = out["operation_logits"]
            ctc_logits = out["ctc_logits"]
            mask = batch["operation_mask"].to(device)
            targets_binary = batch["binary"].to(device)

            # Error probability (sub + del vs correct)
            probs = op_logits.softmax(dim=-1)
            s_error = probs[..., 1] + probs[..., 2]

            valid_scores = s_error[mask].cpu().numpy().tolist()
            valid_targets = targets_binary[mask].cpu().numpy().tolist()
            all_scores.extend(valid_scores)
            all_binary_labels.extend(valid_targets)

            # CTC decoding
            fl = out["frame_lengths"]
            preds = ctc_logits.argmax(dim=-1)
            b_size = audio.shape[0]
            for b in range(b_size):
                f_len = fl[b].item()
                seq = preds[b, :f_len].cpu().tolist()
                total_ctc_frames += f_len
                total_blank_frames += sum(1 for p in seq if p == 0)

                hyp = greedy_ctc_decode(seq, blank_id=0)
                ref_len = batch["ctc_lengths"][b].item()
                ref = batch["ctc"][b, :ref_len].tolist()
                edits, _, _, _ = compute_levenshtein_edits(ref, hyp)
                total_edits += edits
                total_ref_phones += len(ref)

    bin_ap = average_precision_binary(all_binary_labels, all_scores)
    blank_frac = total_blank_frames / total_ctc_frames if total_ctc_frames > 0 else 1.0
    per = total_edits / total_ref_phones if total_ref_phones > 0 else 1.0

    return {
        "val_ap": bin_ap,
        "selection_ap": bin_ap,
        "blank_fraction": blank_frac,
        "per": per,
    }


def train_factorial_arm(
    arm: str,
    split_manifest_path: Path | str,
    zip_path: Path | str,
    output_dir: Path | str,
    seed: int = 1337,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 2,
    warmup_updates: int = 100,
    max_updates: int = 3600,
    lr: float = 0.001,
    weight_decay: float = 0.01,
    gradient_clip: float = 1.0,
    max_epochs: int = 25,
    device: Optional[torch.device] = None,
    remote_mirror: Optional[RemoteCheckpointMirror] = None,
    remote_mirror_prefix: Optional[str] = None,
    pretrained_backbone: str = DEFAULT_PRETRAINED_CHECKPOINT,
    pretrained_revision: str = DEFAULT_PRETRAINED_REVISION,
) -> Dict[str, Any]:
    """Executes matched training for one of the four factorial arms."""
    arm_key = arm.upper()
    if arm_key not in ARM_CONFIGS:
        raise ValueError(f"Unknown arm '{arm}'. Must be one of {list(ARM_CONFIGS.keys())}")

    cfg = ARM_CONFIGS[arm_key]
    objective = cfg["objective"]
    features_enabled = cfg["features_enabled"]

    out_p = Path(output_dir).resolve()
    out_p.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(f"[TRN-04] Starting Factorial Training Arm: {arm_key}")
    print(f"  Description:      {cfg['description']}")
    print(f"  Objective:        {objective}")
    print(f"  Acoustic Slot:    {'ENABLED (4D features)' if features_enabled else 'DISABLED (zeros)'}")
    print(f"  Device:           {device}")
    print(f"  Max Updates:      {max_updates}")
    print(f"  Seed:             {seed}")
    print(f"  Output Dir:       {out_p}")
    print("=" * 80)

    # 1. Pinned random seed across all libraries
    seed_all(seed)

    # 2. Frozen Wav2Vec2 backbone
    print(f"[MODEL] Loading frozen backbone {pretrained_backbone} (rev {pretrained_revision})...")
    wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_backbone, revision=pretrained_revision).to(device)
    encoder = FrozenWav2Vec2AudioEncoder(wav2vec2)

    # 3. Factorial Head initialization from pinned seed 1337
    torch.manual_seed(seed)
    head_config = FactorialHeadConfig()
    head = CompactFactorialOperationHead(head_config).to(device)

    # Verify parameter count
    inv = parameter_inventory(head)
    total_trainable = sum(v["numel"] for v in inv.values() if v["trainable"])
    print(f"[MODEL] Trainable head parameters: {total_trainable:,}")
    assert total_trainable == 267179, f"FAIL-CLOSED: Expected 267,179 parameters, got {total_trainable}"

    adapter = FactorialAudioAdapter(encoder, head, features_enabled=features_enabled).to(device)

    # 4. Datasets & Loaders
    print("[DATA] Loading FIT dataset...")
    fit_dataset = PSupL2ArcticDataset(
        split_manifest_path=split_manifest_path,
        zip_path=zip_path,
        split="fit"
    )
    assert len(fit_dataset) == 1799, f"FAIL-CLOSED: Expected 1,799 FIT utterances, got {len(fit_dataset)}"

    print("[DATA] Loading SELECT dataset...")
    select_dataset = PSupL2ArcticDataset(
        split_manifest_path=split_manifest_path,
        zip_path=zip_path,
        split="selection"
    )
    assert len(select_dataset) == 300, f"FAIL-CLOSED: Expected 300 SELECT utterances, got {len(select_dataset)}"

    train_loader = DataLoader(
        fit_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_psup_batch,
        num_workers=0
    )
    select_loader = DataLoader(
        select_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_psup_batch,
        num_workers=0
    )

    # 5. Optimizer, Scheduler, Scaler
    trainable_params = [p for p in head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate_multiplier)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 6. Training State
    successful_updates = 0
    skipped_amp = 0
    micro_step = 0
    best_ap = -1.0
    history = []
    patience_counter = 0
    fixed_budget_saved = False

    t_start = time.time()

    for epoch in range(1, max_epochs + 1):
        adapter.train()
        epoch_task_loss = 0.0
        epoch_ctc_loss = 0.0
        epoch_steps = 0

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            micro_step += 1

            audio = batch["waveforms"].to(device)
            audio_lengths = batch["audio_lengths"].to(device)
            prompt_ids = batch["prompt_phone_ids"].to(device)
            prompt_lengths = batch["prompt_lengths"].to(device)
            op_targets = batch["operation"].to(device)
            op_mask = batch["operation_mask"].to(device)
            ctc_targets = batch["ctc"].to(device)
            ctc_lengths = batch["ctc_lengths"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                out = adapter(
                    audio=audio,
                    audio_lengths=audio_lengths,
                    prompt_phone_ids=prompt_ids,
                    prompt_lengths=prompt_lengths
                )

                op_logits = out["operation_logits"]
                ctc_logits = out["ctc_logits"]
                frame_lengths = out["frame_lengths"]

                l_task = task_loss(
                    logits=op_logits,
                    operation=op_targets,
                    mask=op_mask,
                    prompt_phone_ids=prompt_ids,
                    objective=objective
                )
                l_ctc = realized_ctc_loss(
                    logits=ctc_logits,
                    frame_lengths=frame_lengths,
                    targets=ctc_targets,
                    target_lengths=ctc_lengths
                )
                total_loss = (l_task + l_ctc) / gradient_accumulation_steps

            if not torch.isfinite(total_loss):
                print(f"[WARN] Nonfinite loss at epoch {epoch}, step {micro_step}; skipping step.")
                continue

            scaler.scale(total_loss).backward()

            epoch_task_loss += l_task.item()
            epoch_ctc_loss += l_ctc.item()
            epoch_steps += 1

            if micro_step % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(trainable_params, max_norm=gradient_clip)

                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()

                if scale_after < scale_before:
                    skipped_amp += 1
                else:
                    successful_updates += 1
                    scheduler.step()

                optimizer.zero_grad()

                # Check fixed-budget checkpoint (exactly 3,600 updates)
                if successful_updates == max_updates and not fixed_budget_saved:
                    fixed_ckpt_path = out_p / "fixed_budget_3600.pt"
                    fixed_payload = {
                        "head": head.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict() if use_amp else None,
                        "epoch": epoch,
                        "micro_steps": micro_step,
                        "successful_updates": successful_updates,
                        "skipped_amp_attempts_total": skipped_amp,
                        "best_ap": best_ap,
                        "history": history,
                        "rng": rng_state(),
                        "metadata": {
                            "arm": arm_key,
                            "objective": objective,
                            "features_enabled": features_enabled,
                            "seed": seed,
                            "contract_sha256": CONTRACT_SHA256,
                            "is_fixed_budget_endpoint": True
                        }
                    }
                    torch.save(fixed_payload, fixed_ckpt_path)
                    fixed_budget_saved = True
                    print(f"[CHECKPOINT] Fixed-budget checkpoint saved: {fixed_ckpt_path}")
                    if remote_mirror:
                        rel = f"{remote_mirror_prefix}/fixed_budget_3600.pt" if remote_mirror_prefix else "fixed_budget_3600.pt"
                        remote_mirror.mirror_file(fixed_ckpt_path, rel)

                if successful_updates >= max_updates:
                    break

        avg_task = epoch_task_loss / max(epoch_steps, 1)
        avg_ctc = epoch_ctc_loss / max(epoch_steps, 1)

        # Validation on SELECT split
        val_res = evaluate_select_set(adapter, select_loader, device, objective)
        val_ap = val_res["val_ap"]
        blank_frac = val_res["blank_fraction"]
        per = val_res["per"]

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"[EPOCH {epoch:02d}] updates={successful_updates}/{max_updates} | "
            f"skipped_amp={skipped_amp} | task_loss={avg_task:.4f} | ctc_loss={avg_ctc:.4f} | "
            f"select_ap={val_ap:.4f} | blank_frac={blank_frac:.4f} | per={per:.4f} | lr={current_lr:.6f}"
        )

        epoch_record = {
            "epoch": epoch,
            "successful_updates": successful_updates,
            "skipped_amp": skipped_amp,
            "attempts": successful_updates + skipped_amp,
            "val_ap": val_ap,
            "selection_ap": val_ap,
            "blank_fraction": blank_frac,
            "per": per,
            "avg_task_loss": avg_task,
            "avg_ctc_loss": avg_ctc,
            "lr": current_lr
        }
        history.append(epoch_record)

        # Selection logic (best AP with healthy CTC)
        is_best = False
        if blank_frac < 0.99 and val_ap > best_ap:
            best_ap = val_ap
            patience_counter = 0
            is_best = True
            best_ckpt_path = out_p / "best.pt"
            best_payload = {
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if use_amp else None,
                "epoch": epoch,
                "micro_steps": micro_step,
                "successful_updates": successful_updates,
                "skipped_amp_attempts_total": skipped_amp,
                "best_ap": best_ap,
                "history": history,
                "rng": rng_state(),
                "metadata": {
                    "arm": arm_key,
                    "objective": objective,
                    "features_enabled": features_enabled,
                    "seed": seed,
                    "contract_sha256": CONTRACT_SHA256,
                    "is_best_endpoint": True
                }
            }
            torch.save(best_payload, best_ckpt_path)
            print(f"  -> NEW BEST SELECT AP: {best_ap:.6f} (saved best.pt)")
            if remote_mirror:
                rel = f"{remote_mirror_prefix}/best.pt" if remote_mirror_prefix else "best.pt"
                remote_mirror.mirror_file(best_ckpt_path, rel)
        else:
            patience_counter += 1

        # Always save last.pt
        last_ckpt_path = out_p / "last.pt"
        last_payload = {
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "epoch": epoch,
            "micro_steps": micro_step,
            "successful_updates": successful_updates,
            "skipped_amp_attempts_total": skipped_amp,
            "best_ap": best_ap,
            "history": history,
            "rng": rng_state(),
            "metadata": {
                "arm": arm_key,
                "objective": objective,
                "features_enabled": features_enabled,
                "seed": seed,
                "contract_sha256": CONTRACT_SHA256,
                "is_last_endpoint": True
            }
        }
        torch.save(last_payload, last_ckpt_path)

        if successful_updates >= max_updates:
            print(f"[BUDGET] Reached target {max_updates} updates. Concluding training.")
            break

    total_duration = time.time() - t_start

    # Write training summary
    summary = {
        "arm": arm_key,
        "objective": objective,
        "features_enabled": features_enabled,
        "description": cfg["description"],
        "seed": seed,
        "max_updates": max_updates,
        "successful_updates": successful_updates,
        "skipped_amp_attempts_total": skipped_amp,
        "total_epochs": epoch,
        "best_selection_ap": best_ap,
        "duration_seconds": total_duration,
        "history": history,
        "fixed_budget_checkpoint": str(out_p / "fixed_budget_3600.pt") if (out_p / "fixed_budget_3600.pt").exists() else None,
        "best_checkpoint": str(out_p / "best.pt") if (out_p / "best.pt").exists() else None,
        "last_checkpoint": str(out_p / "last.pt") if (out_p / "last.pt").exists() else None,
        "completed_at": datetime.now(timezone.utc).isoformat()
    }

    with open(out_p / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if remote_mirror:
        rel = f"{remote_mirror_prefix}/training_summary.json" if remote_mirror_prefix else "training_summary.json"
        remote_mirror.mirror_file(out_p / "training_summary.json", rel)

    print("=" * 80)
    print(f"[TRN-04] Arm {arm_key} Training Completed in {total_duration:.1f}s.")
    print(f"  Successful updates: {successful_updates}")
    print(f"  Skipped AMP:        {skipped_amp}")
    print(f"  Best SELECT AP:     {best_ap:.6f}")
    print("=" * 80)

    return summary
