from typing import List, Dict, Any, Optional, Tuple
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
