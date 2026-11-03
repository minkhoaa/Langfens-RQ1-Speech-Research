"""End-to-end training pipeline for P-SUP Subtype Treatment and Matched Binary Control.

Supports:
a) Subtype treatment: CompactOperationHead with objective="operation_ce"
   (3-way CE: correct=0, substitution=1, deletion=2) + realized CTC loss.
b) Matched binary control: CompactOperationHead with objective="binary_collapsed"
   (collapsed binary logit: logsumexp(sub,del) - correct) + realized CTC loss.

Contracts:
- seed: 1337
- batch_size: 2, gradient_accumulation_steps: 2 (effective batch: 4)
- warmup_updates: 100, cosine schedule over 15,000 horizon
- max_successful_updates: 3600
- AdamW lr: 0.001, weight_decay: 0.01, gradient_clip: 1.0
- CUDA FP16 AMP (with float32 CPU fallback for testing)
- Dual checkpointing: fixed_budget_3600.pt and validation-selected best.pt
"""
from __future__ import annotations

import argparse
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

from langfens_prep.p_sup.model import (
    CompactOperationHead,
    HeadConfig,
    FrozenAudioAdapter,
    matched_initial_pair,
    parameter_inventory
)
from langfens_prep.p_sup.objectives import task_loss, realized_ctc_loss, collapsed_binary_logit
from langfens_prep.p_sup.planning import UpdateLedger, learning_rate_multiplier, accumulation_groups
from langfens_prep.p_sup.dataset import (
    PSupL2ArcticDataset,
    collate_psup_batch,
    PARTITION_SPEAKERS,
    FIT_SPEAKERS,
    SELECT_SPEAKERS,
    CALIBRATION_SPEAKERS,
    VAL_SPEAKERS,
    SealedSpeakerContaminationError
)
from langfens_prep.p_sup.runtime import CONTRACT_SHA256, SEALED_SPEAKERS

DEFAULT_PRETRAINED_CHECKPOINT = "facebook/wav2vec2-base-960h"
DEFAULT_PRETRAINED_REVISION = "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"


def seed_all(seed: int = 1337) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }


def restore_rng(s: Dict[str, Any]) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def compute_file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class FrozenWav2Vec2AudioEncoder(nn.Module):
    """Frozen Wav2Vec2 backbone encoder with padding-safe feature extraction."""
    def __init__(self, wav2vec2: Wav2Vec2Model):
        super().__init__()
        self.wav2vec2 = wav2vec2
        self.wav2vec2.eval()
        for p in self.wav2vec2.parameters():
            p.requires_grad_(False)

    def _get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        out = self.wav2vec2._get_feat_extract_output_lengths(input_lengths)
        if isinstance(out, int):
            out = torch.tensor([out], device=input_lengths.device, dtype=torch.long)
        return out.clamp(min=1)

    def forward(self, audio: torch.Tensor, audio_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = audio.shape[0]
        device = audio.device
        audio_lengths = audio_lengths.to(device=device, dtype=torch.long)

        # Fast vectorized path when all audio lengths in batch are identical
        if (audio_lengths == audio.shape[1]).all():
            fe = self.wav2vec2.feature_extractor(audio).transpose(1, 2)
            frame_output_lengths = self._get_feat_extract_output_lengths(audio_lengths)
            T_max = fe.shape[1]
            padded_features = fe
        else:
            # Per-utterance slice to prevent GroupNorm cross-contamination from padding
            feats = [
                self.wav2vec2.feature_extractor(audio[i:i+1, :audio_lengths[i]]).transpose(1, 2)
                for i in range(batch_size)
            ]
            frame_output_lengths = torch.tensor([f.shape[1] for f in feats], device=device, dtype=torch.long)
            T_max = frame_output_lengths.max().item()
            padded_features = torch.zeros(batch_size, T_max, feats[0].shape[-1], device=device, dtype=feats[0].dtype)
            for i, f in enumerate(feats):
                padded_features[i, :f.shape[1]] = f[0]

        frame_indices = torch.arange(T_max, device=device).unsqueeze(0)
        feat_attention_mask = frame_indices < frame_output_lengths.unsqueeze(1)

        hidden_states, _ = self.wav2vec2.feature_projection(padded_features)
        hidden_states = hidden_states.masked_fill(~feat_attention_mask.unsqueeze(-1), 0.0)

        encoder_outputs = self.wav2vec2.encoder(hidden_states, attention_mask=feat_attention_mask)
        hidden_states = encoder_outputs.last_hidden_state
        if getattr(self.wav2vec2, "adapter", None) is not None:
            hidden_states = self.wav2vec2.adapter(hidden_states)

        return hidden_states, frame_output_lengths


def average_precision_binary(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Average precision with tied scores grouped, matching sklearn convention."""
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError("AP labels must be a non-empty binary vector")
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    s = s[order]
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp = np.cumsum(y)[ends]
    fp = (ends + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def greedy_ctc_decode(frame_preds: Sequence[int], blank_id: int = 0) -> List[int]:
    hyp = []
    prev = None
    for t in frame_preds:
        if t != prev:
            if t != blank_id:
                hyp.append(int(t))
            prev = t
    return hyp


def compute_levenshtein_edits(ref: Sequence[int], hyp: Sequence[int]) -> Tuple[int, int, int, int]:
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    i, j = m, n
    subs, dels, inss = 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1; i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            inss += 1; j -= 1
        else:
            break
    return dp[m][n], subs, dels, inss


def save_checkpoint(
    path: Path,
    head: CompactOperationHead,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    successful_updates: int,
    micro_steps: int,
    best_ap: float,
    patience: int,
    skipped_amp: int,
    history: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    is_fixed_budget_endpoint: bool = False
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "head": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "successful_updates": successful_updates,
        "micro_steps": micro_steps,
        "best_ap": best_ap,
        "patience": patience,
        "skipped_amp_attempts_total": skipped_amp,
        "history": history,
        "rng": rng_state(),
        "metadata": metadata,
        "is_fixed_budget_endpoint": is_fixed_budget_endpoint
    }
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    sha256 = compute_file_sha256(path)
    return sha256


def evaluate_val_set(
    adapter: FrozenAudioAdapter,
    val_loader: DataLoader,
    device: torch.device,
    objective: str
) -> Dict[str, Any]:
    """Evaluate validation split and compute binary AP, blank fraction, PER."""
    adapter.eval()
    all_binary_labels = []
    all_scores = []
    total_ctc_frames = 0
    total_blank_frames = 0
    total_ref_phones = 0
    total_edits = 0

    with torch.no_grad():
        for batch in val_loader:
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

            op_logits = out["operation_logits"]  # (B, N, 3)
            ctc_logits = out["ctc_logits"]        # (B, T, 40)
            mask = batch["operation_mask"].to(device)  # (B, N)
            targets_binary = batch["binary"].to(device)

            # Compute error score
            if objective == "operation_ce":
                probs = op_logits.softmax(dim=-1)
                s_error = probs[..., 1] + probs[..., 2]
            else:
                s_error = torch.sigmoid(collapsed_binary_logit(op_logits))

            valid_scores = s_error[mask].cpu().numpy().tolist()
            valid_targets = targets_binary[mask].cpu().numpy().tolist()

            all_scores.extend(valid_scores)
            all_binary_labels.extend(valid_targets)

            # CTC Blank and PER accounting
            frame_lengths = out["frame_lengths"]
            preds = ctc_logits.argmax(dim=-1)
            for i, fl in enumerate(frame_lengths.tolist()):
                pred_seq = preds[i, :fl].cpu().tolist()
                total_ctc_frames += fl
                total_blank_frames += sum(1 for p in pred_seq if p == 0)

                hyp = greedy_ctc_decode(pred_seq, blank_id=0)
                ref_len = batch["ctc_lengths"][i].item()
                ref = batch["ctc"][i, :ref_len].tolist()
                edits, _, _, _ = compute_levenshtein_edits(ref, hyp)
                total_edits += edits
                total_ref_phones += len(ref)

    ap = average_precision_binary(all_binary_labels, all_scores)
    blank_fraction = total_blank_frames / total_ctc_frames if total_ctc_frames > 0 else 1.0
    per = total_edits / total_ref_phones if total_ref_phones > 0 else 1.0

    return {
        "val_ap": ap,
        "blank_fraction": blank_fraction,
        "per": per,
        "total_eval_tokens": len(all_binary_labels),
        "total_eval_frames": total_ctc_frames
    }


def train_p_sup(
    objective: str,
    split_manifest_path: str | Path,
    zip_path: str | Path,
    output_dir: str | Path,
    seed: int = 1337,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 2,
    warmup_updates: int = 100,
    cosine_horizon_updates: int = 15000,
    max_successful_updates: int = 3600,
    lr: float = 0.001,
    weight_decay: float = 0.01,
    gradient_clip: float = 1.0,
    ctc_weight: float = 1.0,
    task_weight: float = 1.0,
    pretrained_backbone: str = DEFAULT_PRETRAINED_CHECKPOINT,
    pretrained_revision: str = DEFAULT_PRETRAINED_REVISION,
    device: Optional[torch.device] = None,
    max_epochs: int = 25,
    early_stopping_patience: int = 5,
    train_partition: str = "fit",
    select_partition: str = "selection",
    cal_partition: str = "calibration",
    val_partition: str = "val",
    remote_mirror: Optional[Any] = None,
    remote_rel_prefix: Optional[str] = None
) -> Dict[str, Any]:
    """Execute official training for one P-SUP arm."""
    if objective not in ("operation_ce", "binary_collapsed"):
        raise ValueError(f"Unregistered P-SUP objective: {objective!r}")

    seed_all(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_cuda_amp = (device.type == "cuda")

    # 1. Datasets & Loaders
    # Training dataset: default FIT only (1,799 utterances across 12 speakers:
    # BWC, ERMS, HKK, HQTV, SKA, SVBI, TNI, TXHC, YBAA, YDCK, YKWK, ZHAA)
    train_dataset = PSupL2ArcticDataset(
        split_manifest_path=split_manifest_path,
        zip_path=zip_path,
        partition=train_partition
    )
    if train_partition == "fit":
        if len(train_dataset) != 1799:
            raise ValueError(
                f"FAIL-CLOSED: FIT training dataset must have exactly 1,799 utterances, got {len(train_dataset)}"
            )
        train_speakers = frozenset(spk for spk, _ in train_dataset.entries)
        if train_speakers != FIT_SPEAKERS:
            raise ValueError(
                f"FAIL-CLOSED: FIT training dataset speakers mismatch: {train_speakers} != {FIT_SPEAKERS}"
            )
        if train_speakers & SEALED_SPEAKERS:
            raise SealedSpeakerContaminationError(
                f"FAIL-CLOSED: Sealed test speaker found in FIT training dataset: {train_speakers & SEALED_SPEAKERS}"
            )
    else:
        train_speakers = frozenset(spk for spk, _ in train_dataset.entries)

    # Selection dataset: default SELECT (300 utterances across 2 speakers: MBMPS, TLV)
    # Used for validation loss, AP tracking, early stopping, and saving best.pt during training
    selection_dataset = PSupL2ArcticDataset(
        split_manifest_path=split_manifest_path,
        zip_path=zip_path,
        partition=select_partition
    )
    if select_partition in ("selection", "select"):
        if len(selection_dataset) != 300:
            raise ValueError(
                f"FAIL-CLOSED: SELECT dataset must have exactly 300 utterances, got {len(selection_dataset)}"
            )
        selection_speakers = frozenset(spk for spk, _ in selection_dataset.entries)
        if selection_speakers != SELECT_SPEAKERS:
            raise ValueError(
                f"FAIL-CLOSED: SELECT dataset speakers mismatch: {selection_speakers} != {SELECT_SPEAKERS}"
            )
        if selection_speakers & SEALED_SPEAKERS:
            raise SealedSpeakerContaminationError(
                f"FAIL-CLOSED: Sealed test speaker found in SELECT dataset: {selection_speakers & SEALED_SPEAKERS}"
            )
    else:
        selection_speakers = frozenset(spk for spk, _ in selection_dataset.entries)

    # Fold-1 VAL (300 utterances across NJS, RRBI) MUST be preserved strictly unseen during training & selection
    if train_speakers & VAL_SPEAKERS:
        raise ValueError(
            f"FAIL-CLOSED: Fold-1 VAL speakers leaked into training set: {train_speakers & VAL_SPEAKERS}"
        )
    if selection_speakers & VAL_SPEAKERS:
        raise ValueError(
            f"FAIL-CLOSED: Fold-1 VAL speakers leaked into selection set: {selection_speakers & VAL_SPEAKERS}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_psup_batch,
        num_workers=0,
        drop_last=False
    )
    selection_loader = DataLoader(
        selection_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_psup_batch,
        num_workers=0
    )
    val_loader = selection_loader  # Alias for validation during training

    # 2. Model & Backbone Initialization
    wav2vec2 = Wav2Vec2Model.from_pretrained(
        pretrained_backbone,
        revision=pretrained_revision
    ).to(device)
    encoder = FrozenWav2Vec2AudioEncoder(wav2vec2)

    head = CompactOperationHead(HeadConfig()).to(device)
    adapter = FrozenAudioAdapter(encoder, head).to(device)

    # 3. Optimizer & Scheduler
    trainable_params = [p for p in head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    def lr_lambda(step: int) -> float:
        return learning_rate_multiplier(step)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler(device="cuda", init_scale=4096, enabled=use_cuda_amp)
    ledger = UpdateLedger()

    # 4. Training Loop
    best_ap = -1.0
    patience = 0
    history = []
    micro_steps = 0
    fixed_budget_saved = False
    start_time = time.time()

    metadata = {
        "registration": "P-SUP-COMPACT-OP-V1",
        "objective": objective,
        "seed": seed,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "warmup_updates": warmup_updates,
        "cosine_horizon_updates": cosine_horizon_updates,
        "max_successful_updates": max_successful_updates,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "ctc_weight": ctc_weight,
        "task_weight": task_weight,
        "backbone": pretrained_backbone,
        "backbone_revision": pretrained_revision,
        "contract_sha256": CONTRACT_SHA256,
        "split_manifest_path": str(split_manifest_path),
        "split_manifest_sha256": compute_file_sha256(split_manifest_path),
        "zip_path": str(zip_path),
        "zip_sha256": compute_file_sha256(zip_path) if Path(zip_path).is_file() else None,
        "parameters": parameter_inventory(head),
        "fit_utterances": len(train_dataset),
        "selection_utterances": len(selection_dataset),
        "calibration_utterances": 300,
        "evaluation_utterances": 300,
        "fit_speakers": sorted(FIT_SPEAKERS),
        "selection_speakers": sorted(SELECT_SPEAKERS),
        "calibration_speakers": sorted(CALIBRATION_SPEAKERS),
        "evaluation_speakers": sorted(VAL_SPEAKERS)
    }

    print(f"[{objective.upper()}] Starting P-SUP training on {device} (AMP: {use_cuda_amp})...")
    print(f"[{objective.upper()}] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    for epoch in range(max_epochs):
        if ledger.fixed_budget_reached:
            break

        adapter.train()
        epoch_task_loss = 0.0
        epoch_ctc_loss = 0.0
        epoch_batches = len(train_loader)
        accum_groups = accumulation_groups(epoch_batches, gradient_accumulation_steps)

        optimizer.zero_grad()
        group_idx = 0
        batch_in_group = 0
        current_group_size = accum_groups[group_idx]

        for b_idx, batch in enumerate(train_loader):
            if ledger.fixed_budget_reached:
                break

            audio = batch["waveforms"].to(device)
            audio_lengths = batch["audio_lengths"].to(device)
            prompt_ids = batch["prompt_phone_ids"].to(device)
            prompt_lengths = batch["prompt_lengths"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda_amp):
                out = adapter(
                    audio=audio,
                    audio_lengths=audio_lengths,
                    prompt_phone_ids=prompt_ids,
                    prompt_lengths=prompt_lengths
                )

                loss_t = task_loss(
                    logits=out["operation_logits"],
                    operation=batch["operation"].to(device),
                    mask=batch["operation_mask"].to(device),
                    objective=objective
                )

                loss_c = realized_ctc_loss(
                    logits=out["ctc_logits"],
                    frame_lengths=out["frame_lengths"],
                    targets=batch["ctc"].to(device),
                    target_lengths=batch["ctc_lengths"].to(device)
                )

                total_loss = ctc_weight * loss_c + task_weight * loss_t

            loss_scaled = total_loss / current_group_size
            scaler.scale(loss_scaled).backward()
            micro_steps += 1
            batch_in_group += 1

            epoch_task_loss += loss_t.item()
            epoch_ctc_loss += loss_c.item()

            if batch_in_group == current_group_size:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=gradient_clip)

                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()

                overflow = (scale_after < scale_before)
                unskipped = ledger.record_attempt(overflow=overflow)
                if unskipped:
                    scheduler.step()

                optimizer.zero_grad()
                batch_in_group = 0
                group_idx += 1
                if group_idx < len(accum_groups):
                    current_group_size = accum_groups[group_idx]

                # Check exact fixed-budget endpoint
                if ledger.fixed_budget_reached and not fixed_budget_saved:
                    fb_path = output_dir / "fixed_budget_3600.pt"
                    save_checkpoint(
                        path=fb_path,
                        head=head,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        successful_updates=ledger.successful,
                        micro_steps=micro_steps,
                        best_ap=best_ap,
                        patience=patience,
                        skipped_amp=ledger.skipped,
                        history=history,
                        metadata=metadata,
                        is_fixed_budget_endpoint=True
                    )
                    fb_alias = output_dir / "fixed_budget_checkpoint.pt"
                    shutil.copyfile(str(fb_path), str(fb_alias))
                    fixed_budget_saved = True
                    print(f"[{objective.upper()}] Fixed-budget endpoint 3,600 reached and saved at epoch {epoch}!")
                    if remote_mirror is not None and remote_rel_prefix is not None:
                        try:
                            remote_mirror.mirror_file(fb_path, f"{remote_rel_prefix}/fixed_budget_3600.pt")
                            remote_mirror.mirror_file(fb_alias, f"{remote_rel_prefix}/fixed_budget_checkpoint.pt")
                        except Exception as e:
                            print(f"[{objective.upper()} MIRROR WARNING] Error mirroring fixed budget: {e}")

        # Epoch Validation on SELECT partition (MBMPS, TLV)
        val_metrics = evaluate_val_set(adapter, selection_loader, device, objective)
        val_ap = val_metrics["val_ap"]
        blank_frac = val_metrics["blank_fraction"]
        per = val_metrics["per"]

        epoch_record = {
            "epoch": epoch,
            "successful_updates": ledger.successful,
            "skipped_amp": ledger.skipped,
            "attempts": ledger.attempts,
            "val_ap": val_ap,
            "selection_ap": val_ap,
            "blank_fraction": blank_frac,
            "per": per,
            "avg_task_loss": epoch_task_loss / max(1, epoch_batches),
            "avg_ctc_loss": epoch_ctc_loss / max(1, epoch_batches),
            "lr": optimizer.param_groups[0]["lr"]
        }
        history.append(epoch_record)
        print(
            f"[{objective.upper()}] Epoch {epoch:02d} | Updates: {ledger.successful}/{max_successful_updates} "
            f"| Selection AP: {val_ap:.4f} | Blank: {blank_frac:.4f} | PER: {per:.4f}"
        )

        # Save last.pt each epoch
        last_path = output_dir / "last.pt"
        save_checkpoint(
            path=last_path,
            head=head,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            successful_updates=ledger.successful,
            micro_steps=micro_steps,
            best_ap=best_ap,
            patience=patience,
            skipped_amp=ledger.skipped,
            history=history,
            metadata=metadata
        )
        if remote_mirror is not None and remote_rel_prefix is not None:
            try:
                remote_mirror.mirror_file(last_path, f"{remote_rel_prefix}/last.pt")
            except Exception as e:
                print(f"[{objective.upper()} MIRROR WARNING] Error mirroring last.pt: {e}")

        # Check best.pt selection (strict binary AP on SELECT with blank_fraction < 0.99)
        if blank_frac < 0.99 and val_ap > best_ap:
            best_ap = val_ap
            patience = 0
            best_path = output_dir / "best.pt"
            save_checkpoint(
                path=best_path,
                head=head,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                successful_updates=ledger.successful,
                micro_steps=micro_steps,
                best_ap=best_ap,
                patience=patience,
                skipped_amp=ledger.skipped,
                history=history,
                metadata=metadata
            )
            print(f"[{objective.upper()}] -> New best selection AP: {best_ap:.4f} saved to best.pt")
            if remote_mirror is not None and remote_rel_prefix is not None:
                try:
                    remote_mirror.mirror_file(best_path, f"{remote_rel_prefix}/best.pt")
                except Exception as e:
                    print(f"[{objective.upper()} MIRROR WARNING] Error mirroring best.pt: {e}")
        else:
            patience += 1

    total_time = time.time() - start_time
    summary = {
        "objective": objective,
        "successful_updates": ledger.successful,
        "skipped_updates": ledger.skipped,
        "attempts": ledger.attempts,
        "best_ap": best_ap,
        "fixed_budget_saved": fixed_budget_saved,
        "total_time_seconds": total_time,
        "history": history
    }

    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    history_path = output_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    if remote_mirror is not None and remote_rel_prefix is not None:
        try:
            remote_mirror.mirror_file(summary_path, f"{remote_rel_prefix}/training_summary.json")
            remote_mirror.mirror_file(history_path, f"{remote_rel_prefix}/history.json")
        except Exception as e:
            print(f"[{objective.upper()} MIRROR WARNING] Error mirroring summary/history: {e}")

    return summary
