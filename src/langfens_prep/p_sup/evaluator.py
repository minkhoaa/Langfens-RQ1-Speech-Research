"""Strict evaluation of P-SUP checkpoints under Evaluation Contract V1.

Computes:
- Binary AP (substitution+deletion vs correct)
- Omission AP (legacy-comparable: deletion vs correct using s_error)
- Dedicated Omission AP (deletion vs correct using s_omit)
- Substitution AP (legacy-comparable: substitution vs correct using s_error)
- PER (greedy CTC decode vs chronological realized reference)
- Per-speaker breakdown (NJS, RRBI)
- Schema-compliant runtime and offline token outputs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model

from langfens_prep.p_sup.model import CompactOperationHead, HeadConfig, FrozenAudioAdapter
from langfens_prep.p_sup.objectives import collapsed_binary_logit
from langfens_prep.p_sup.dataset import PSupL2ArcticDataset, collate_psup_batch
from langfens_prep.p_sup.trainer import (
    FrozenWav2Vec2AudioEncoder,
    average_precision_binary,
    greedy_ctc_decode,
    compute_levenshtein_edits,
    compute_file_sha256,
    DEFAULT_PRETRAINED_CHECKPOINT,
    DEFAULT_PRETRAINED_REVISION
)
from langfens_prep.p_sup.runtime import (
    operation_runtime_rows,
    validate_complete_run,
    validate_run_manifest,
    CONTRACT_SHA256,
    BASE_SHA,
    SEALED_SPEAKERS
)


def evaluate_checkpoint_v1(
    checkpoint_path: Path | str,
    split_manifest_path: Path | str,
    zip_path: Path | str,
    output_dir: Path | str,
    device: Optional[torch.device] = None,
    pretrained_backbone: str = DEFAULT_PRETRAINED_CHECKPOINT,
    pretrained_revision: str = DEFAULT_PRETRAINED_REVISION
) -> Dict[str, Any]:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt_data = torch.load(ckpt_path, map_location="cpu")
    metadata = ckpt_data.get("metadata", {})
    objective = metadata.get("objective", "operation_ce")

    # Load frozen backbone and head
    wav2vec2 = Wav2Vec2Model.from_pretrained(
        pretrained_backbone,
        revision=pretrained_revision
    ).to(device)
    encoder = FrozenWav2Vec2AudioEncoder(wav2vec2)

    head = CompactOperationHead(HeadConfig()).to(device)
    head.load_state_dict(ckpt_data["head"])
    head.eval()

    adapter = FrozenAudioAdapter(encoder, head).to(device)
    adapter.eval()

    val_dataset = PSupL2ArcticDataset(
        split_manifest_path=split_manifest_path,
        zip_path=zip_path,
        split="val"
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_psup_batch,
        num_workers=0
    )

    all_rows = []
    expected_prompts = {}
    offline_eval_rows = []

    # Aggregators for AP and PER
    binary_gold = []
    error_scores = []

    omission_gold = []
    omission_error_scores = []
    dedicated_omission_scores = []

    substitution_gold = []
    substitution_error_scores = []

    speaker_metrics = {"NJS": {"binary_gold": [], "s_error": [], "edits": 0, "ref_len": 0},
                       "RRBI": {"binary_gold": [], "s_error": [], "edits": 0, "ref_len": 0}}

    total_ctc_frames = 0
    total_blank_frames = 0
    total_ref_phones = 0
    total_edits = 0

    run_id = f"eval_{ckpt_path.stem}"

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

            op_logits = out["operation_logits"]
            ctc_logits = out["ctc_logits"]
            frame_lengths = out["frame_lengths"]
            preds = ctc_logits.argmax(dim=-1)

            b_size = audio.shape[0]
            for b in range(b_size):
                spk = batch["speaker_ids"][b]
                uid = batch["utterance_ids"][b]
                p_len = batch["prompt_lengths"][b].item()
                prompts = batch["prompt_phone_ids"][b, :p_len].cpu().tolist()
                expected_prompts[(spk, uid)] = prompts

                utt_op_logits = op_logits[b, :p_len]
                rows = operation_runtime_rows(
                    logits=utt_op_logits,
                    prompt_phone_ids=prompts,
                    run_id=run_id,
                    condition="matched",
                    speaker_id=spk,
                    utterance_id=uid,
                    objective=objective
                )
                all_rows.extend(rows)

                # Offline tokens join
                op_targets = batch["operation"][b, :p_len].cpu().tolist()
                bin_targets = batch["binary"][b, :p_len].cpu().tolist()

                for idx in range(p_len):
                    s_err = rows[idx]["scores"]["s_error"]
                    s_om = rows[idx]["scores"]["s_omit"]
                    s_sub = rows[idx]["scores"]["s_sub"]
                    op_y = op_targets[idx]
                    bin_y = bin_targets[idx]

                    token_row = {
                        "speaker_id": spk,
                        "utterance_id": uid,
                        "prompt_index": idx,
                        "gold_canonical_index": idx,
                        "operation_target": op_y,
                        "binary_target": bin_y,
                        "s_error": s_err,
                        "s_omit": s_om,
                        "s_sub": s_sub
                    }
                    offline_eval_rows.append(token_row)

                    binary_gold.append(bin_y)
                    error_scores.append(s_err)
                    speaker_metrics[spk]["binary_gold"].append(bin_y)
                    speaker_metrics[spk]["s_error"].append(s_err)

                    # Omission population: deletion (2) vs correct (0), exclude substitution (1)
                    if op_y in (0, 2):
                        omission_gold.append(1 if op_y == 2 else 0)
                        omission_error_scores.append(s_err)
                        if s_om is not None:
                            dedicated_omission_scores.append(s_om)

                    # Substitution population: substitution (1) vs correct (0), exclude deletion (2)
                    if op_y in (0, 1):
                        substitution_gold.append(1 if op_y == 1 else 0)
                        substitution_error_scores.append(s_err)

                # CTC accounting
                fl = frame_lengths[b].item()
                pred_seq = preds[b, :fl].cpu().tolist()
                total_ctc_frames += fl
                total_blank_frames += sum(1 for p in pred_seq if p == 0)

                hyp = greedy_ctc_decode(pred_seq, blank_id=0)
                ref_len = batch["ctc_lengths"][b].item()
                ref = batch["ctc"][b, :ref_len].tolist()
                edits, _, _, _ = compute_levenshtein_edits(ref, hyp)
                total_edits += edits
                total_ref_phones += len(ref)

                speaker_metrics[spk]["edits"] += edits
                speaker_metrics[spk]["ref_len"] += len(ref)

    # Validate complete run
    run_validation = validate_complete_run(
        all_rows,
        run_id=run_id,
        condition="matched",
        expected_prompts=expected_prompts
    )

    # Compute metrics
    bin_ap = average_precision_binary(binary_gold, error_scores)
    om_ap_legacy = average_precision_binary(omission_gold, omission_error_scores)
    sub_ap = average_precision_binary(substitution_gold, substitution_error_scores)
    dedicated_om_ap = average_precision_binary(omission_gold, dedicated_omission_scores) if dedicated_omission_scores else None

    blank_fraction = total_blank_frames / total_ctc_frames if total_ctc_frames > 0 else 1.0
    per = total_edits / total_ref_phones if total_ref_phones > 0 else 1.0

    per_speaker_results = {}
    for spk, spk_data in speaker_metrics.items():
        spk_ap = average_precision_binary(spk_data["binary_gold"], spk_data["s_error"])
        spk_per = spk_data["edits"] / spk_data["ref_len"] if spk_data["ref_len"] > 0 else 1.0
        per_speaker_results[spk] = {
            "binary_ap": spk_ap,
            "per": spk_per,
            "tokens": len(spk_data["binary_gold"])
        }

    metrics_report = {
        "run_id": run_id,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": compute_file_sha256(ckpt_path),
        "objective": objective,
        "contract_sha256": CONTRACT_SHA256,
        "binary_ap": bin_ap,
        "omission_ap_legacy": om_ap_legacy,
        "omission_ap_dedicated": dedicated_om_ap,
        "substitution_ap": sub_ap,
        "per": per,
        "blank_fraction": blank_fraction,
        "speaker_breakdown": per_speaker_results,
        "run_validation": run_validation
    }

    # Write output files
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_report, f, indent=2)

    with open(output_dir / "runtime_predictions.jsonl", "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")

    with open(output_dir / "offline_token_evaluation.jsonl", "w") as f:
        for r in offline_eval_rows:
            f.write(json.dumps(r) + "\n")

    return metrics_report
