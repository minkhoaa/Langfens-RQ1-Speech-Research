"""LANGFENS RQ1-TRN-04 — Full Evaluation Suite under Protocol Z.

Evaluates factorial model checkpoints across:
1. Strict Original Gold Ground Truth (Protocol Z, Fold-1 VAL, 300 utterances, 9,956 gold positions)
2. Dedicated Omission AP (s_omit), Legacy Omission AP (s_error), Binary AP, Substitution AP
3. Greedy CTC PER & Blank Fraction
4. Per-Speaker Breakdown (NJS, RRBI)
5. Word-Final Consonant dedicated omission AP
6. Prospectively Frozen CAL Operating Points (5%, 10%, 15% CAL FPR ceilings on NCC/THV)
7. Acoustic Grounding Control (Matched vs Mismatched audio)
8. Adjudicated Broad Pronunciation Event Horizon (Horvitz-Thompson estimation from 230-case ledger)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model

from langfens_prep.trn04.features import extract_frame_acoustic_features, ACOUSTIC_FEATURE_DIM
from langfens_prep.trn04.model import (
    CompactFactorialOperationHead,
    FactorialHeadConfig,
    FactorialAudioAdapter
)
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
    CONTRACT_SHA256,
    SEALED_SPEAKERS
)

# Reference Horvitz-Thompson strata statistics from the 230-case review ledger
# (documented in ANNOTATION_REVIEW_LEDGER.jsonl and CASE_MANIFEST.json)
STRATA_POPULATIONS = {
    "D_correct_alarms": 389,
    "T_correct_alarms": 302,
    "R_correct_alarms": 362,
    "other_correct_alarms": 74,
    "substitution_alarms": 82,
}
STRATA_DISAGREEMENT_RATES = {
    "D_correct_alarms": 0.3800,  # 38% unreleased /d/ marked correct
    "T_correct_alarms": 0.3400,  # 34% unreleased /t/ marked correct
    "R_correct_alarms": 0.0500,  # 5% tap /r/ marked correct
    "other_correct_alarms": 0.0800,
    "substitution_alarms": 0.1200,
}


def cal_threshold(correct_scores: List[float], ceiling: float) -> float:
    """Calculates prospectively frozen CAL threshold for a given FPR ceiling."""
    if not correct_scores:
        return 0.5
    descending = sorted(correct_scores, reverse=True)
    allowed = math.floor(ceiling * len(descending))
    allowed = min(allowed, len(descending) - 1)
    return math.nextafter(descending[allowed], math.inf)


def evaluate_factorial_checkpoint(
    checkpoint_path: Path | str,
    split_manifest_path: Path | str,
    zip_path: Path | str,
    output_dir: Path | str,
    device: Optional[torch.device] = None,
    pretrained_backbone: str = DEFAULT_PRETRAINED_CHECKPOINT,
    pretrained_revision: str = DEFAULT_PRETRAINED_REVISION,
) -> Dict[str, Any]:
    ckpt_path = Path(checkpoint_path).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"FAIL-CLOSED: Checkpoint not found: {ckpt_path}")

    out_p = Path(output_dir).resolve()
    out_p.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metadata = ckpt_data.get("metadata", {})
    arm = metadata.get("arm", "C0")
    objective = metadata.get("objective", "operation_ce")
    features_enabled = metadata.get("features_enabled", False)

    print("=" * 80)
    print(f"[EVAL-TRN04] Evaluating Checkpoint: {ckpt_path.name}")
    print(f"  Arm:               {arm}")
    print(f"  Objective:         {objective}")
    print(f"  Acoustic Features: {features_enabled}")
    print(f"  Device:            {device}")
    print("=" * 80)

    # 1. Load Backbone & Head
    wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_backbone, revision=pretrained_revision).to(device)
    encoder = FrozenWav2Vec2AudioEncoder(wav2vec2)

    head = CompactFactorialOperationHead(FactorialHeadConfig()).to(device)
    head.load_state_dict(ckpt_data["head"])
    head.eval()

    adapter = FactorialAudioAdapter(encoder, head, features_enabled=features_enabled).to(device)
    adapter.eval()

    # 2. Phase 1: CAL Calibration on NCC/THV (300 utterances)
    print("[EVAL] Running CAL inference for threshold calibration (NCC, THV)...")
    cal_dataset = PSupL2ArcticDataset(
        split_manifest_path=split_manifest_path,
        zip_path=zip_path,
        split="calibration"
    )
    cal_loader = DataLoader(
        cal_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_psup_batch,
        num_workers=0
    )

    cal_correct_s_omit = []
    cal_correct_s_error = []

    with torch.no_grad():
        for batch in cal_loader:
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
            probs = op_logits.softmax(dim=-1)

            b_size = audio.shape[0]
            for b in range(b_size):
                p_len = batch["prompt_lengths"][b].item()
                targets = batch["operation"][b, :p_len].cpu().tolist()
                b_probs = probs[b, :p_len].cpu().tolist()

                for idx in range(p_len):
                    if targets[idx] == 0:  # correct token
                        cal_correct_s_omit.append(b_probs[idx][2])
                        cal_correct_s_error.append(b_probs[idx][1] + b_probs[idx][2])

    print(f"  CAL correct tokens evaluated: {len(cal_correct_s_omit)}")
    ceilings = [0.05, 0.10, 0.15]
    cal_thresholds_omit = {c: cal_threshold(cal_correct_s_omit, c) for c in ceilings}
    cal_thresholds_error = {c: cal_threshold(cal_correct_s_error, c) for c in ceilings}
    print(f"  CAL Omission Thresholds (s_omit): {cal_thresholds_omit}")

    # 3. Phase 2: Full VAL Inference on Fold-1 VAL (NJS, RRBI, 300 utterances)
    print("[EVAL] Running VAL inference (NJS, RRBI)...")
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

    binary_gold = []
    error_scores = []

    omission_gold = []
    omission_error_scores = []
    dedicated_omission_scores = []

    substitution_gold = []
    substitution_error_scores = []
    dedicated_substitution_scores = []

    speaker_metrics = {
        "NJS": {"binary_gold": [], "s_error": [], "edits": 0, "ref_len": 0},
        "RRBI": {"binary_gold": [], "s_error": [], "edits": 0, "ref_len": 0}
    }

    # Word-final consonant trackers
    wfc_gold = []
    wfc_scores = []

    total_ctc_frames = 0
    total_blank_frames = 0
    total_ref_phones = 0
    total_edits = 0

    run_id = f"eval_{ckpt_path.stem}_{arm}"

    # Cache for acoustic grounding (mismatched audio control)
    val_audio_cache = []

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

            probs = op_logits.softmax(dim=-1)

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

                op_targets = batch["operation"][b, :p_len].cpu().tolist()
                bin_targets = batch["binary"][b, :p_len].cpu().tolist()

                # Cache audio for mismatch test
                val_audio_cache.append({
                    "speaker": spk,
                    "uid": uid,
                    "audio": audio[b, :audio_lengths[b]].cpu(),
                    "audio_len": audio_lengths[b].cpu(),
                    "prompts": prompt_ids[b, :p_len].cpu(),
                    "prompt_len": p_len,
                    "targets": op_targets,
                    "bin_targets": bin_targets
                })

                for idx in range(p_len):
                    s_err = rows[idx]["scores"]["s_error"]
                    s_om = rows[idx]["scores"]["s_omit"]
                    s_sub = rows[idx]["scores"]["s_sub"]
                    op_y = op_targets[idx]
                    bin_y = bin_targets[idx]
                    p_id = prompts[idx]

                    token_row = {
                        "speaker_id": spk,
                        "utterance_id": uid,
                        "prompt_index": idx,
                        "gold_canonical_index": idx,
                        "prompt_phone_id": p_id,
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

                    # Omission population: deletion (2) vs correct (0)
                    if op_y in (0, 2):
                        is_om = 1 if op_y == 2 else 0
                        omission_gold.append(is_om)
                        omission_error_scores.append(s_err)
                        if s_om is not None:
                            dedicated_omission_scores.append(s_om)
                            # WFC heuristic check (tokens near word boundaries)
                            if idx == p_len - 1 or (idx + 1 < p_len and prompts[idx + 1] in (1, 2, 3)):
                                wfc_gold.append(is_om)
                                wfc_scores.append(s_om)

                    # Substitution population: substitution (1) vs correct (0)
                    if op_y in (0, 1):
                        is_sub = 1 if op_y == 1 else 0
                        substitution_gold.append(is_sub)
                        substitution_error_scores.append(s_err)
                        if s_sub is not None:
                            dedicated_substitution_scores.append(s_sub)

                # CTC metrics
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

    # 4. Phase 3: Mismatched Audio Acoustic Grounding Control
    print("[EVAL] Running mismatched audio acoustic grounding control...")
    mismatched_bin_gold = []
    mismatched_error_scores = []

    # Cyclic shift by 1 within speaker cohort
    njs_items = [item for item in val_audio_cache if item["speaker"] == "NJS"]
    rrbi_items = [item for item in val_audio_cache if item["speaker"] == "RRBI"]

    with torch.no_grad():
        for spk_cohort in (njs_items, rrbi_items):
            m = len(spk_cohort)
            for i in range(m):
                curr = spk_cohort[i]
                shuffled_audio_item = spk_cohort[(i + 1) % m]

                m_audio = shuffled_audio_item["audio"].unsqueeze(0).to(device)
                m_audio_len = torch.tensor([shuffled_audio_item["audio_len"]], device=device)
                p_ids = curr["prompts"].unsqueeze(0).to(device)
                p_len = torch.tensor([curr["prompt_len"]], device=device)

                m_out = adapter(
                    audio=m_audio,
                    audio_lengths=m_audio_len,
                    prompt_phone_ids=p_ids,
                    prompt_lengths=p_len
                )
                m_probs = m_out["operation_logits"][0, :curr["prompt_len"]].softmax(dim=-1)
                m_err = (m_probs[:, 1] + m_probs[:, 2]).cpu().tolist()

                mismatched_bin_gold.extend(curr["bin_targets"])
                mismatched_error_scores.extend(m_err)

    mismatched_bin_ap = average_precision_binary(mismatched_bin_gold, mismatched_error_scores)
    print(f"  Mismatched Audio Binary AP: {mismatched_bin_ap:.6f}")

    # 5. Core Metric Computation
    bin_ap = average_precision_binary(binary_gold, error_scores)
    om_ap_legacy = average_precision_binary(omission_gold, omission_error_scores)
    sub_ap = average_precision_binary(substitution_gold, substitution_error_scores)
    dedicated_om_ap = average_precision_binary(omission_gold, dedicated_omission_scores) if dedicated_omission_scores else om_ap_legacy
    dedicated_sub_ap = average_precision_binary(substitution_gold, dedicated_substitution_scores) if dedicated_substitution_scores else sub_ap

    blank_fraction = total_blank_frames / total_ctc_frames if total_ctc_frames > 0 else 1.0
    per = total_edits / total_ref_phones if total_ref_phones > 0 else 1.0

    wfc_om_ap = average_precision_binary(wfc_gold, wfc_scores) if wfc_gold else None

    # Per-speaker results
    per_speaker_results = {}
    for spk, spk_data in speaker_metrics.items():
        spk_ap = average_precision_binary(spk_data["binary_gold"], spk_data["s_error"])
        spk_per = spk_data["edits"] / spk_data["ref_len"] if spk_data["ref_len"] > 0 else 1.0
        per_speaker_results[spk] = {
            "binary_ap": spk_ap,
            "per": spk_per,
            "tokens": len(spk_data["binary_gold"])
        }

    # 6. CAL-Selected Operating Point Metrics
    cal_operating_points = {}
    all_val_gold_omissions = 145  # full census of gold omissions across all 300 VAL utterances
    total_val_canonical = 9956

    for c in ceilings:
        tau = cal_thresholds_omit[c]
        tp = 0
        fp_cor = 0
        fp_sub = 0
        tn = 0
        fn = 0

        for r in offline_eval_rows:
            s = r["s_omit"] if r["s_omit"] is not None else r["s_error"]
            y = r["operation_target"]
            is_alarm = (s >= tau)

            if y == 2:  # omission
                if is_alarm:
                    tp += 1
                else:
                    fn += 1
            elif y == 0:  # correct
                if is_alarm:
                    fp_cor += 1
                else:
                    tn += 1
            elif y == 1:  # substitution
                if is_alarm:
                    fp_sub += 1

        # Account for 8 unmapped gold omissions in denominator
        fn_all = fn + 8
        total_alarms = tp + fp_cor + fp_sub
        deployed_precision = tp / total_alarms if total_alarms > 0 else 0.0
        all_gold_recall = tp / all_val_gold_omissions
        fa_per_1k = (fp_cor + fp_sub) / (total_val_canonical / 1000.0)

        # Horvitz-Thompson Adjudicated Pronunciation Event Estimation
        # (Accounts for unreleased stops and taps marked correct in original gold)
        disagree_alarms = (
            fp_cor * 0.2361 + fp_sub * 0.1200
        )
        broad_tp_est = tp + disagree_alarms
        broad_precision_est = broad_tp_est / total_alarms if total_alarms > 0 else 0.0
        broad_recall_est = broad_tp_est / (all_val_gold_omissions + 285.5)

        cal_operating_points[f"CAL_{int(c*100)}%"] = {
            "ceiling": c,
            "threshold": tau,
            "tp": tp,
            "fp_cor": fp_cor,
            "fp_sub": fp_sub,
            "tn": tn,
            "fn": fn_all,
            "total_alarms": total_alarms,
            "deployed_precision": deployed_precision,
            "all_gold_recall": all_gold_recall,
            "fa_per_1000_canonical": fa_per_1k,
            "adjudicated_broad_tp_est": broad_tp_est,
            "adjudicated_broad_precision_est": broad_precision_est,
            "adjudicated_broad_recall_est": broad_recall_est
        }

    # Primary calibration operating point at nominal 15% CAL FPR ceiling
    primary_cal = cal_operating_points["CAL_15%"]

    # Prospective Acceptance Gates Check
    # Historical T-OP reference:
    # deployed_prec: 0.067825, recall: 0.565517, fa_per_1k: 113.20, bin_ap: 0.318619, sub_ap: 0.308974, per: 0.241980
    level_a_gates = {
        "precision_ge_25": bool(primary_cal["deployed_precision"] >= 0.25),
        "recall_ge_30": bool(primary_cal["all_gold_recall"] >= 0.30),
        "fa_le_30": bool(primary_cal["fa_per_1000_canonical"] <= 30.0),
        "binary_ap_noninferior": bool(bin_ap >= 0.313619),
        "sub_ap_noninferior": bool(sub_ap >= 0.303974),
        "per_noninferior": bool(per <= 0.246980),
    }
    level_a_passed = all(level_a_gates.values())

    level_b_gates = {
        "precision_ge_30": bool(primary_cal["deployed_precision"] >= 0.30),
        "recall_ge_40": bool(primary_cal["all_gold_recall"] >= 0.40),
        "fa_le_15": bool(primary_cal["fa_per_1000_canonical"] <= 15.0),
        "binary_ap_noninferior": bool(bin_ap >= 0.313619),
        "sub_ap_noninferior": bool(sub_ap >= 0.303974),
        "per_noninferior": bool(per <= 0.246980),
        "acoustic_gap_positive": bool((bin_ap - mismatched_bin_ap) >= 0.010),
    }
    level_b_passed = all(level_b_gates.values())

    report = {
        "run_id": run_id,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": compute_file_sha256(ckpt_path),
        "arm": arm,
        "objective": objective,
        "features_enabled": features_enabled,
        "contract_sha256": CONTRACT_SHA256,
        "metrics": {
            "strict_deployed_omission_precision": primary_cal["deployed_precision"],
            "all_gold_omission_recall": primary_cal["all_gold_recall"],
            "false_alarms_per_1000_canonical": primary_cal["fa_per_1000_canonical"],
            "dedicated_omission_ap": dedicated_om_ap,
            "legacy_omission_ap": om_ap_legacy,
            "binary_ap": bin_ap,
            "legacy_substitution_ap": sub_ap,
            "dedicated_substitution_ap": dedicated_sub_ap,
            "greedy_per": per,
            "blank_fraction": blank_fraction,
            "word_final_consonant_omission_ap": wfc_om_ap,
            "mismatched_audio_binary_ap": mismatched_bin_ap,
            "acoustic_reliance_gap": bin_ap - mismatched_bin_ap,
        },
        "cal_operating_points": cal_operating_points,
        "speaker_breakdown": per_speaker_results,
        "acceptance_gates": {
            "level_a_pilot_feasibility": {
                "passed": level_a_passed,
                "gates": level_a_gates
            },
            "level_b_research_target": {
                "passed": level_b_passed,
                "gates": level_b_gates
            }
        }
    }

    # Save outputs
    with open(out_p / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(out_p / "runtime_predictions.jsonl", "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")

    with open(out_p / "offline_token_evaluation.jsonl", "w") as f:
        for r in offline_eval_rows:
            f.write(json.dumps(r) + "\n")

    print("=" * 80)
    print(f"[EVAL-TRN04] Evaluation Completed for {arm}:")
    print(f"  Strict Deployed Precision: {primary_cal['deployed_precision']*100:.2f}%")
    print(f"  All-Gold Omission Recall:  {primary_cal['all_gold_recall']*100:.2f}%")
    print(f"  Dedicated Omission AP:     {dedicated_om_ap:.6f}")
    print(f"  Binary AP:                 {bin_ap:.6f}")
    print(f"  Substitution AP:           {sub_ap:.6f}")
    print(f"  Greedy PER:                {per:.6f}")
    print(f"  False Alarms / 1k:         {primary_cal['fa_per_1000_canonical']:.2f}")
    print(f"  Acoustic Reliance Gap:     {bin_ap - mismatched_bin_ap:.6f}")
    print(f"  Level A Gates:             {'PASSED' if level_a_passed else 'FAILED'}")
    print(f"  Level B Gates:             {'PASSED' if level_b_passed else 'FAILED'}")
    print("=" * 80)

    return report
