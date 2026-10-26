"""
LANGFENS — Sequence Alignment and Representation Pooling
Provides:
1. Dynamic programming Levenshtein alignment between G2P canonical sequences and Gold canonical sequences.
2. Parameterized tie-breaking rules and alignment trace recording for Protocol Z verification.
3. Acoustic frame representation pooling for boundary-based analyses.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pool_frames_to_canonical_phones(
    frame_features: torch.Tensor,
    intervals_per_batch: List[List[Tuple[float, float]]],
    sample_rate: int = 16000,
    frame_stride_samples: int = 320,
    pooling_mode: str = "mean"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pools continuous acoustic frame representations (e.g. from Wav2Vec2 encoder)
    into token-level representations for canonical prompt phonemes.

    Guarantees:
    1. Every phoneme interval receives at least 1 acoustic frame (no empty slices).
    2. Boundaries are strictly clamped within [0, T_frames].
    3. Outputs zero-padded tensor (batch, max_phones, hidden_dim) and valid boolean mask (batch, max_phones).
    """
    assert pooling_mode in ("mean", "max"), f"Unsupported pooling mode: {pooling_mode}"
    batch_size, max_frames, hidden_dim = frame_features.shape
    device = frame_features.device
    dtype = frame_features.dtype

    num_phones_list = [len(intervals) for intervals in intervals_per_batch]
    max_phones = max(num_phones_list) if num_phones_list and max(num_phones_list) > 0 else 0

    if max_phones == 0:
        empty_pooled = torch.zeros(batch_size, 0, hidden_dim, device=device, dtype=dtype)
        empty_mask = torch.zeros(batch_size, 0, device=device, dtype=torch.bool)
        return empty_pooled, empty_mask

    pooled = torch.zeros(batch_size, max_phones, hidden_dim, device=device, dtype=dtype)
    mask = torch.zeros(batch_size, max_phones, device=device, dtype=torch.bool)

    seconds_per_frame = frame_stride_samples / sample_rate

    for b in range(batch_size):
        intervals = intervals_per_batch[b]
        for p_idx, (t_start, t_end) in enumerate(intervals):
            # Convert time in seconds to acoustic frame indices
            start_frame = int(t_start / seconds_per_frame)
            end_frame = int(t_end / seconds_per_frame)

            # Clamp to valid bounds
            start_frame = max(0, min(max_frames - 1, start_frame))
            end_frame = max(start_frame + 1, min(max_frames, end_frame))

            phone_frames = frame_features[b, start_frame:end_frame, :]

            if pooling_mode == "mean":
                pooled[b, p_idx, :] = phone_frames.mean(dim=0)
            elif pooling_mode == "max":
                pooled[b, p_idx, :] = phone_frames.max(dim=0).values

            mask[b, p_idx] = True

    return pooled, mask


def align_canonical_sequences(
    gold_phones: List[str],
    g2p_phones: List[str],
    tie_break: str = "sub_first"
) -> Dict[str, Any]:
    """
    Performs deterministic dynamic programming Levenshtein alignment between
    a gold canonical phone sequence (reference) and a G2P canonical sequence (hypothesis).

    Parameters:
        gold_phones: Gold reference phoneme sequence from TextGrid intervals (length N).
        g2p_phones: G2P predicted phoneme sequence from text prompt (length M).
        tie_break: Priority ordering for minimum cost ties ('sub_first', 'del_first', 'ins_first').

    Returns:
        Dict containing:
            - aligned_pairs: List of (gold_idx, g2p_idx, op_code) where op_code in (0=match, 1=substitution)
            - unaligned_gold: List of (gold_idx, gold_phone) (omitted by G2P / deletions)
            - unaligned_g2p: List of (g2p_idx, g2p_phone) (extra in G2P / insertions)
            - edits_summary: Dict with counts of (total_edits, substitutions, deletions, insertions)
            - alignment_trace: Step-by-step trace of aligned phone tokens and operations
    """
    valid_ties = ("sub_first", "del_first", "ins_first")
    if tie_break not in valid_ties:
        raise ValueError(f"Unknown tie_break policy: {tie_break}. Must be one of {valid_ties}")

    n = len(gold_phones)
    m = len(g2p_phones)

    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    # ops: 0=match, 1=sub, 2=del (gold unaligned), 3=ins (g2p unaligned)
    ops = np.zeros((n + 1, m + 1), dtype=np.int8)

    for i in range(1, n + 1):
        dp[i, 0] = i
        ops[i, 0] = 2  # Deletion from Gold
    for j in range(1, m + 1):
        dp[0, j] = j
        ops[0, j] = 3  # Insertion into G2P

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            is_match = (gold_phones[i - 1] == g2p_phones[j - 1])
            cost = 0 if is_match else 1
            sub_cost = dp[i - 1, j - 1] + cost
            del_cost = dp[i - 1, j] + 1
            ins_cost = dp[i, j - 1] + 1

            best = min(sub_cost, del_cost, ins_cost)
            dp[i, j] = best

            if tie_break == "sub_first":
                if best == sub_cost:
                    ops[i, j] = 0 if is_match else 1
                elif best == del_cost:
                    ops[i, j] = 2
                else:
                    ops[i, j] = 3
            elif tie_break == "del_first":
                if best == del_cost:
                    ops[i, j] = 2
                elif best == sub_cost:
                    ops[i, j] = 0 if is_match else 1
                else:
                    ops[i, j] = 3
            elif tie_break == "ins_first":
                if best == ins_cost:
                    ops[i, j] = 3
                elif best == sub_cost:
                    ops[i, j] = 0 if is_match else 1
                else:
                    ops[i, j] = 2

    # Backtrace
    i, j = n, m
    raw_steps = []
    while i > 0 or j > 0:
        op = ops[i, j]
        if op == 0:
            raw_steps.append((i - 1, j - 1, "match", gold_phones[i - 1], g2p_phones[j - 1]))
            i -= 1
            j -= 1
        elif op == 1:
            raw_steps.append((i - 1, j - 1, "substitution", gold_phones[i - 1], g2p_phones[j - 1]))
            i -= 1
            j -= 1
        elif op == 2:
            raw_steps.append((i - 1, None, "deletion", gold_phones[i - 1], None))
            i -= 1
        elif op == 3:
            raw_steps.append((None, j - 1, "insertion", None, g2p_phones[j - 1]))
            j -= 1

    raw_steps.reverse()

    aligned_pairs = []
    unaligned_gold = []
    unaligned_g2p = []
    trace = []
    n_subs, n_dels, n_inss = 0, 0, 0

    for step_idx, (g_idx, z_idx, op_name, g_phone, z_phone) in enumerate(raw_steps):
        step_dict = {
            "step": step_idx,
            "op": op_name,
            "gold_idx": g_idx,
            "g2p_idx": z_idx,
            "gold_phone": g_phone,
            "g2p_phone": z_phone
        }
        trace.append(step_dict)

        if op_name in ("match", "substitution"):
            op_code = 0 if op_name == "match" else 1
            aligned_pairs.append((g_idx, z_idx, op_code))
            if op_name == "substitution":
                n_subs += 1
        elif op_name == "deletion":
            unaligned_gold.append((g_idx, g_phone))
            n_dels += 1
        elif op_name == "insertion":
            unaligned_g2p.append((z_idx, z_phone))
            n_inss += 1

    return {
        "aligned_pairs": aligned_pairs,
        "unaligned_gold": unaligned_gold,
        "unaligned_g2p": unaligned_g2p,
        "edits_summary": {
            "total_edits": n_subs + n_dels + n_inss,
            "substitutions": n_subs,
            "deletions": n_dels,
            "insertions": n_inss
        },
        "alignment_trace": trace
    }
