"""Evaluation-only CPU FP32 inference and Protocol Z offline joining for TRN-04.

Runtime entry points consume waveform and a written-prompt phone sequence only.
TextGrid parsing is deliberately isolated in ``join_gold_rows`` and must only
run after every CAL and VAL Z runtime file has been written and hashed.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import torch

from langfens_prep.trn04.features import ACOUSTIC_FEATURE_DIM, extract_frame_acoustic_features
from langfens_prep.trn04.model import CompactFactorialOperationHead, FactorialHeadConfig

ARMS = {"C0", "T1", "T2", "T3"}
ENDPOINTS = {"fixed_budget_3600", "best"}
OBJECTIVES = {"C0": "operation_ce", "T1": "class_focal", "T2": "operation_ce", "T3": "class_focal"}
FEATURES_ENABLED = {"C0": False, "T1": False, "T2": True, "T3": True}
SEALED_SPEAKERS = frozenset({"ABA", "ASI", "EBVS", "HJK", "LXC", "PNV"})


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_line(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json_line(row) + "\n")
    return sha256_file(path)


def make_head_forward(head: CompactFactorialOperationHead, *, arm: str,
                      encoder: torch.nn.Module) -> Callable[..., dict[str, torch.Tensor]]:
    """Build one immutable inference closure; all tensors stay CPU float32."""
    if arm not in ARMS:
        raise ValueError(f"unknown TRN-04 arm: {arm}")
    head.eval().requires_grad_(False)
    encoder.eval().requires_grad_(False)

    def forward(*, waveform: torch.Tensor, prompt_phone_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        if waveform.ndim != 1 or waveform.dtype != torch.float32 or prompt_phone_ids.ndim != 1:
            raise ValueError("expected one CPU float32 waveform and one prompt phone sequence")
        if waveform.device.type != "cpu" or prompt_phone_ids.device.type != "cpu":
            raise ValueError("TRN-04 runtime is CPU-only")
        if not waveform.numel() or not prompt_phone_ids.numel():
            raise ValueError("empty waveform or prompt is not scoreable")
        if not bool(torch.isfinite(waveform).all()):
            raise ValueError("nonfinite waveform")
        prompt_phone_ids = prompt_phone_ids.to(dtype=torch.long).unsqueeze(0)
        audio = waveform.unsqueeze(0)
        with torch.inference_mode():
            frame_lengths = torch.tensor([audio.shape[1]], dtype=torch.long)
            features, lengths = encoder(audio, frame_lengths)
            features = features.to(dtype=torch.float32, device="cpu")
            lengths = lengths.to(dtype=torch.long, device="cpu")
            if FEATURES_ENABLED[arm]:
                acoustic = extract_frame_acoustic_features(audio, lengths)
            else:
                acoustic = torch.zeros((1, features.shape[1], ACOUSTIC_FEATURE_DIM), dtype=torch.float32)
            result = head(features=features, acoustic_features=acoustic, frame_lengths=lengths,
                          prompt_phone_ids=prompt_phone_ids,
                          prompt_lengths=torch.tensor([prompt_phone_ids.shape[1]], dtype=torch.long))
        for key in ("operation_logits", "ctc_logits"):
            if not bool(torch.isfinite(result[key]).all()):
                raise ValueError(f"nonfinite {key}")
        return result
    return forward


def runtime_rows(*, logits: torch.Tensor, ctc_logits: torch.Tensor, prompt_phone_ids: list[int],
                 canonical_phones: list[str], speaker_id: str, utterance_id: str, arm: str, endpoint: str,
                 condition: str, audio_source_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Serialize raw FP32 outputs without gold or derived diagnosis fields."""
    if speaker_id in SEALED_SPEAKERS:
        raise ValueError("sealed TEST speaker rejected")
    if arm not in ARMS or endpoint not in ENDPOINTS:
        raise ValueError("unregistered arm/endpoint")
    if len(canonical_phones) != len(prompt_phone_ids) or logits.shape != (len(prompt_phone_ids), 3) or logits.dtype != torch.float32:
        raise ValueError("operation output shape/dtype mismatch")
    if not prompt_phone_ids or any(type(phone) is not int or not 1 <= phone <= 39 for phone in prompt_phone_ids):
        raise ValueError("invalid G2P prompt phone indices")
    logits = logits.detach().cpu()
    probs = logits.softmax(dim=-1)
    rows = []
    run_id = f"trn04r:{arm}:{endpoint}:Z_PROMPT_V1:{condition}"
    for index, phone in enumerate(prompt_phone_ids):
        p = probs[index]
        rows.append({
            "run_id": run_id, "protocol": "Z-PROMPT-V1", "condition": condition,
            "arm": arm, "endpoint": endpoint, "objective": OBJECTIVES[arm],
            "score_semantics": "three_class_operation_softmax; s_error=p_substitution+p_deletion; s_omit=p_deletion; s_sub=p_substitution; uncalibrated",
            "speaker_id": speaker_id,
            "utterance_id": utterance_id, "prompt_index": index,
            "prompt_phone_id": phone, "operation_target": None, "binary_target": None,
            "canonical_phone": canonical_phones[index], "gold_canonical_index": None,
            "scores": {"s_error": float(p[1] + p[2]), "s_omit": float(p[2]), "s_sub": float(p[1])},
            "probabilities": {"correct": float(p[0]), "substitution": float(p[1]), "deletion": float(p[2])},
            "logits": [float(x) for x in logits[index]],
            "audio_source_id": audio_source_id, "status": "ok",
        })
    ctc = ctc_logits.detach().cpu().float()
    ctc_ids = ctc.argmax(dim=-1).tolist()
    # Standard greedy CTC collapse: repeated frame labels collapse before blank removal.
    decoded: list[int] = []
    previous = None
    for token in ctc_ids:
        if token != previous and token != 0:
            decoded.append(int(token))
        previous = token
    recognition = {
        "run_id": run_id, "protocol": "Z-PROMPT-V1", "condition": condition,
        "arm": arm, "endpoint": endpoint, "speaker_id": speaker_id,
        "utterance_id": utterance_id, "audio_source_id": audio_source_id,
        "frame_argmax_ids": [int(x) for x in ctc_ids],
        "hypothesis_phone_ids": decoded, "frame_count": len(ctc_ids),
        "blank_frames": sum(x == 0 for x in ctc_ids),
        "ctc_logits_sha256": hashlib.sha256(ctc.numpy().tobytes()).hexdigest(),
    }
    return rows, recognition


def join_gold_rows(*, gold_positions: list[dict[str, Any]], prompt_phones: list[str],
                   runtime: list[dict[str, Any]], align: Callable[..., dict[str, Any]]) -> list[dict[str, Any]]:
    """Offline canonical-gold/prompt DP join; call only after runtime freeze."""
    if not runtime:
        raise ValueError("no runtime rows to join")
    aligned = align([p["canonical_phone"] for p in gold_positions], prompt_phones, tie_break="sub_first")
    prompt_for_gold = {gold: prompt for gold, prompt, _ in aligned["aligned_pairs"]}
    identity = {k: runtime[0][k] for k in ("run_id", "protocol", "condition", "arm", "endpoint", "objective", "score_semantics", "speaker_id", "utterance_id", "audio_source_id")}
    by_prompt = {row["prompt_index"]: row for row in runtime}
    result = []
    for gold_index, position in enumerate(gold_positions):
        prompt_index = prompt_for_gold.get(gold_index)
        source = by_prompt.get(prompt_index) if prompt_index is not None else None
        score_values = source["scores"] if source else {"s_error": None, "s_omit": None, "s_sub": None}
        result.append({
            **identity, "operation_target": position["operation_target"],
            "binary_target": position["binary_target"], "canonical_phone": position["canonical_phone"],
            "gold_canonical_index": gold_index, "prompt_index": prompt_index,
            **score_values, "scores": score_values,
            "probabilities": source["probabilities"] if source else None,
            "logits": source["logits"] if source else None,
            "mapping_status": "aligned" if source else "unmapped_gold_abstention",
            "word_final_consonant": position.get("word_final_consonant", False),
        })
    for prompt_index, _phone in aligned["unaligned_g2p"]:
        source = by_prompt[prompt_index]
        result.append({**identity, "operation_target": None, "binary_target": None,
                       "canonical_phone": source["canonical_phone"], "gold_canonical_index": None,
                       "prompt_index": prompt_index, **source["scores"], "scores": source["scores"],
                       "probabilities": source["probabilities"], "logits": source["logits"], "mapping_status": "prompt_extra",
                       "word_final_consonant": False})
    return result
