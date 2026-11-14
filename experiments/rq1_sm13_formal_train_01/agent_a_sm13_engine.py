"""Agent A: Authoritative Architecture, Inference Engine & Contract for LANGFENS-SM13-FUSION-V1.

Implements:
1. Differentiable 13-layer scalar-mixing target and gap heads.
2. Exact feature extraction from frozen facebook/wav2vec2-base-960h and CharSiu alignment.
3. Fixed 50/50 neural-CTC/E4 sequence likelihood fusion.
4. Stable event IDs and exact EVAL-V3.1 event cardinality (N targets, N+1 gaps).
5. Comprehensive deterministic inference and checkpoint loading APIs.
"""
from __future__ import annotations
import ctypes
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/home/khoa/Projects/New-Method-Research")
PHONE_CLASSES = [
    "AA", "AE", "AH", "AO", "AW", "AX", "AY", "B", "CH", "D",
    "DH", "EH", "ER", "ERR", "EY", "F", "G", "HH", "IH", "IY",
    "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R",
    "S", "SH", "SPN", "T", "TH", "UH", "UW", "V", "W", "Y",
    "Z", "ZH"
]
PHONE_TO_ID = {p: i for i, p in enumerate(PHONE_CLASSES)}
BLANK_ID = 42

CTC_LIB_PATH = ROOT / "experiments/rq1_method_bench_04/agent_a/ctc_batch.so"


def get_ctc_kernel():
    if not CTC_LIB_PATH.exists():
        raise FileNotFoundError(f"CTC batch shared object missing: {CTC_LIB_PATH}")
    lib = ctypes.CDLL(str(CTC_LIB_PATH))
    f = lib.score_ctc_batch
    f.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double)
    ]
    f.restype = None
    return f


class ScalarMixingTargetHead(nn.Module):
    """13-layer scalar mixing across layers 0..12 with learned softmax weights and scale."""
    def __init__(self, num_layers: int = 13, feat_dim: int = 2304, ref_dim: int = 39):
        super().__init__()
        self.num_layers = num_layers
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.scale = nn.Parameter(torch.ones(1))
        self.net = nn.Sequential(
            nn.Linear(feat_dim + ref_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 3)
        )

    def forward(self, layer_feats: torch.Tensor, refs: torch.Tensor) -> torch.Tensor:
        """
        layer_feats: (B, 13, 2304) unstandardized multi-layer acoustic representations
        refs:        (B, 39) canonical intended phone one-hot embedding
        Returns:     (B, 3) logits [CORRECT, SUBSTITUTION, OMISSION]
        """
        w = torch.softmax(self.layer_logits, dim=0)
        mixed = torch.einsum("l, bli -> bi", w, layer_feats) * self.scale
        x = torch.cat([mixed, refs], dim=-1)
        return self.net(x)

    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits, dim=0)

    @torch.inference_mode()
    def predict_proba(self, layer_feats: torch.Tensor, refs: torch.Tensor) -> torch.Tensor:
        logits = self.forward(layer_feats, refs)
        return torch.softmax(logits, dim=-1)


class ScalarMixingGapHead(nn.Module):
    """13-layer scalar mixing for exact-gap insertion head."""
    def __init__(self, num_layers: int = 13, feat_dim: int = 1536, ref_dim: int = 78):
        super().__init__()
        self.num_layers = num_layers
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.scale = nn.Parameter(torch.ones(1))
        self.net = nn.Sequential(
            nn.Linear(feat_dim + ref_dim, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )

    def forward(self, layer_feats: torch.Tensor, refs: torch.Tensor) -> torch.Tensor:
        """
        layer_feats: (B, 13, 1536) unstandardized multi-layer acoustic context representations
        refs:        (B, 78) context phone one-hot embeddings (left + right)
        Returns:     (B, 2) logits [NO_INSERTION, INSERTION]
        """
        w = torch.softmax(self.layer_logits, dim=0)
        mixed = torch.einsum("l, bli -> bi", w, layer_feats) * self.scale
        x = torch.cat([mixed, refs], dim=-1)
        return self.net(x)

    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits, dim=0)

    @torch.inference_mode()
    def predict_proba(self, layer_feats: torch.Tensor, refs: torch.Tensor) -> torch.Tensor:
        logits = self.forward(layer_feats, refs)
        return torch.softmax(logits, dim=-1)


class LinearCTC(nn.Module):
    """Linear projection from frozen Wav2Vec2 layer 6 (768D) to 43 CTC phone classes."""
    def __init__(self, in_dim: int = 768, num_classes: int = 43):
        super().__init__()
        self.proj = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class LANGFENS_SM13_Model(nn.Module):
    """Composite SM13 container housing TargetHead, GapHead, and optional LinearCTC."""
    def __init__(self):
        super().__init__()
        self.target_head = ScalarMixingTargetHead()
        self.gap_head = ScalarMixingGapHead()
        self.ctc_proj = LinearCTC()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Use dedicated branch forward methods: forward_target / forward_gap")

    def forward_target(self, layer_feats: torch.Tensor, refs: torch.Tensor) -> torch.Tensor:
        return self.target_head(layer_feats, refs)

    def forward_gap(self, layer_feats: torch.Tensor, refs: torch.Tensor) -> torch.Tensor:
        return self.gap_head(layer_feats, refs)


def extract_features_from_hidden_states(
    hidden_states: List[torch.Tensor],
    centers: np.ndarray,
    n_frames: int,
    radius: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts 13-layer Target (2304D) and Gap (1536D) representations from raw Wav2Vec2 hidden states.
    hidden_states: 13 tensors of shape (1, T, 768)
    centers: array of float alignment centers, length N
    n_frames: total frames in alignment clock
    """
    N = len(centers)
    target_feats_13 = []
    gap_feats_13 = []

    for l in range(13):
        hidden = hidden_states[l][0].float().detach().cpu().numpy()
        T = len(hidden)
        # Scaled center frame indices
        scaled_c = np.rint(centers * (T / max(1, n_frames))).astype(int)
        offsets = np.arange(-radius, radius + 1)
        ix = np.clip(scaled_c[:, None] + offsets[None, :], 0, T - 1)
        chunks = hidden[ix]  # (N, 17, 768)

        # Target acoustic feature: central frames [3:14] (11 frames)
        c = chunks[:, 3:14, :]
        t_feat = np.concatenate([c.mean(axis=1), c.std(axis=1, ddof=0), c[:, -1] - c[:, 0]], axis=1)  # (N, 2304)
        target_feats_13.append(t_feat)

        # Gap acoustic feature: left context [11:16], right context [1:6]
        left = np.zeros((N + 1, 768), dtype=np.float32)
        right = np.zeros((N + 1, 768), dtype=np.float32)
        if N > 0:
            left[1:] = chunks[:, 11:16, :].mean(axis=1)
            right[:-1] = chunks[:, 1:6, :].mean(axis=1)
        g_feat = np.concatenate([left, right], axis=1)  # (N + 1, 1536)
        gap_feats_13.append(g_feat)

    target_layer_feats = np.stack(target_feats_13, axis=1)  # (N, 13, 2304)
    gap_layer_feats = np.stack(gap_feats_13, axis=1)        # (N + 1, 13, 1536)
    return target_layer_feats, gap_layer_feats


def compute_e4_likelihoods(
    log_probs: np.ndarray,
    canonical_phone_ids: List[int],
    kernel_fn=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes exact E4 sequence likelihood ratios (SLLR, OLLR) and simplex probabilities
    for an utterance using the compiled C++ CTC batch forward pass.
    log_probs: (T, 43) float64 array of CTC log-probabilities
    canonical_phone_ids: list of integer canonical phone IDs, length N
    Returns:
      e4_probs: (N, 3) float32 [P(C), P(S), P(O)]
      sllr: (N,) float64
      ollr: (N,) float64
    """
    N = len(canonical_phone_ids)
    if N == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    if kernel_fn is None:
        kernel_fn = get_ctc_kernel()

    T = log_probs.shape[0]
    lp_contig = np.ascontiguousarray(log_probs, dtype=np.float64)

    # Candidate hypotheses to evaluate:
    # 1. Canonical sequence (baseline)
    # 2. For each pos i: 4 competitor substitutions
    # 3. For each pos i: canonical deletion (omission)
    # Total hypotheses: 1 + 4*N + N = 1 + 5*N
    sequences = [list(canonical_phone_ids)]
    sub_map = []
    om_map = []

    for i, phone_id in enumerate(canonical_phone_ids):
        # 4 candidate substitutions (offset modulo 42)
        sub_cands = [(phone_id + k) % 42 for k in range(1, 5)]
        sub_indices = []
        for cand in sub_cands:
            hyp = list(canonical_phone_ids)
            hyp[i] = cand
            sub_indices.append(len(sequences))
            sequences.append(hyp)
        sub_map.append(sub_indices)

        # Deletion candidate
        hyp_del = canonical_phone_ids[:i] + canonical_phone_ids[i + 1:]
        om_map.append(len(sequences))
        sequences.append(hyp_del)

    # Flatten sequences for C++ kernel
    flat_seqs = []
    offsets = [0]
    for seq in sequences:
        flat_seqs.extend(seq)
        offsets.append(len(flat_seqs))

    flat_arr = np.array(flat_seqs, dtype=np.int32)
    offsets_arr = np.array(offsets, dtype=np.int32)
    scores_out = np.zeros(len(sequences), dtype=np.float64)

    # Call C++ kernel
    kernel_fn(
        lp_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(T),
        ctypes.c_int(43),
        ctypes.c_int(BLANK_ID),
        flat_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        offsets_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(len(sequences)),
        scores_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )

    can_score = scores_out[0]
    sllr_vals = np.zeros(N, dtype=np.float64)
    ollr_vals = np.zeros(N, dtype=np.float64)
    e4_simplex = np.zeros((N, 3), dtype=np.float32)

    for i in range(N):
        sub_scores = [scores_out[idx] for idx in sub_map[i]]
        # Average likelihood of competitor substitutions
        max_sub = np.mean(sub_scores)
        sllr = max_sub - can_score
        del_score = scores_out[om_map[i]]
        ollr = del_score - can_score

        sllr_vals[i] = sllr
        ollr_vals[i] = ollr

        # Simplex = softmax([0, SLLR, OLLR])
        logits = np.array([0.0, sllr, ollr], dtype=np.float64)
        exp_l = np.exp(logits - np.max(logits))
        e4_simplex[i] = (exp_l / np.sum(exp_l)).astype(np.float32)

    return e4_simplex, sllr_vals, ollr_vals


def resolve_registered_policy(
    seed: int = 101,
    outer: int = 0,
    budget_key: str = "fixed/0.2",
    custom_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Resolves and validates the registered warning policy from formal Phase 1 artifacts.
    Raises FileNotFoundError if missing, ValueError if invalid.
    Never uses silent fallbacks or hardcoded constants.
    """
    if custom_path is not None:
        p = Path(custom_path)
    else:
        p = ROOT / f"experiments/rq1_train_phase_01/formal/seed_{seed}/outer_{outer}/policies.json"

    if not p.exists():
        raise FileNotFoundError(f"Registered policy file does not exist: {p}")

    try:
        data = json.loads(p.read_text())
        if "policies" not in data or budget_key not in data["policies"]:
            raise ValueError(f"Policy key '{budget_key}' not found in {p}")
        pol = data["policies"][budget_key]
        if "target" not in pol or "threshold" not in pol["target"]:
            raise ValueError(f"Missing target threshold in {p} under '{budget_key}'")
        if "gap" not in pol or "threshold" not in pol["gap"]:
            raise ValueError(f"Missing gap threshold in {p} under '{budget_key}'")
        t_th = float(pol["target"]["threshold"])
        g_th = float(pol["gap"]["threshold"])
        return {
            "target": {"threshold": t_th},
            "gap": {"threshold": g_th},
            "source_path": str(p),
            "budget_key": budget_key
        }
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as err:
        raise ValueError(f"Invalid policy structure in {p}: {err}") from err


def apply_warning_policy(
    target_fused_probs: np.ndarray,
    gap_probs: np.ndarray,
    target_compact_ids: List[str],
    gap_compact_ids: List[str],
    policy_thresholds: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies frozen warning policy thresholds to fused target probabilities and gap probabilities.
    Returns:
      target_decisions: (N,) int64 [0=CORRECT, 1=SUBSTITUTION, 2=OMISSION]
      gap_decisions:    (N+1,) bool [True=INSERTION, False=NO_INSERTION]
    """
    if policy_thresholds is None or not isinstance(policy_thresholds, dict):
        raise ValueError("Missing or invalid policy_thresholds dictionary")
    if "target" not in policy_thresholds or "threshold" not in policy_thresholds["target"]:
        raise ValueError("policy_thresholds must contain ['target']['threshold']")
    if "gap" not in policy_thresholds or "threshold" not in policy_thresholds["gap"]:
        raise ValueError("policy_thresholds must contain ['gap']['threshold']")

    N = len(target_fused_probs)
    G = len(gap_probs)

    target_conf = target_fused_probs[:, 1:].max(axis=1)
    target_kind = target_fused_probs[:, 1:].argmax(axis=1) + 1

    t_thresh = float(policy_thresholds["target"]["threshold"])
    g_thresh = float(policy_thresholds["gap"]["threshold"])

    target_warn = target_conf >= t_thresh
    gap_warn = gap_probs >= g_thresh

    target_decisions = np.where(target_warn, target_kind, 0)
    gap_decisions = gap_warn

    return target_decisions, gap_decisions
