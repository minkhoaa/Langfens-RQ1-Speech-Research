"""LANGFENS RQ1-TRN-04 — Waveform-derived frame-level acoustic evidence extraction.

Extracts a compact, mathematically rigorous 4-dimensional frame-level acoustic feature vector:
1. Normalized Log Frame Energy: (log(mean(x^2) + eps) - mean) / std (FIT-derived statistics)
2. Relative RMS Energy: RMS_t / (max_tau RMS_tau + eps)
3. Delta Log Energy: E_t - E_{t-1} (with E_{-1} = E_0)
4. Zero-Crossing Rate: fraction of sign changes per frame

Alignment Invariant:
Window size = 400 samples (25ms at 16kHz)
Hop size = 320 samples (20ms at 16kHz)
Aligned strictly to Wav2Vec2 conv feature output rate (50 Hz).
Zero gold TextGrid boundaries or realization labels are accessed.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# Pinned empirical statistics computed strictly on the 1,799 FIT training utterances
FIT_LOG_ENERGY_MEAN: float = -6.047091
FIT_LOG_ENERGY_STD: float = 2.544367
ACOUSTIC_FEATURE_DIM: int = 4


def extract_frame_acoustic_features(
    audio: torch.Tensor,
    expected_frames: int | torch.Tensor,
    mean: float = FIT_LOG_ENERGY_MEAN,
    std: float = FIT_LOG_ENERGY_STD,
    window_size: int = 400,
    hop_size: int = 320,
    eps: float = 1e-7
) -> torch.Tensor:
    """Extracts 4D frame-level acoustic features aligned to Wav2Vec2 frames.

    Args:
        audio: (B, L) or (L,) float32 audio waveform at 16kHz.
        expected_frames: int or Tensor (B,) target frame count matching Wav2Vec2 output.
        mean: Log-energy mean computed strictly from FIT.
        std: Log-energy std computed strictly from FIT.
        window_size: Window size in samples (400 = 25ms @ 16kHz).
        hop_size: Stride in samples (320 = 20ms @ 16kHz).
        eps: Small constant for numerical stability.

    Returns:
        Tensor of shape (B, target_len, 4) with finite float32 features.
    """
    orig_ndim = audio.ndim
    if orig_ndim == 1:
        audio = audio.unsqueeze(0)
    b, l = audio.shape

    # Pad audio to center frames matching Wav2Vec2 receptive field
    pad_left = (window_size - hop_size) // 2
    pad_right = window_size
    padded = F.pad(audio, (pad_left, pad_right), mode="constant", value=0.0)

    # Unfold frames: shape (B, num_frames, window_size)
    frames = padded.unfold(dimension=-1, size=window_size, step=hop_size)

    # 1. Frame Log-Energy
    mean_sq = torch.mean(frames ** 2, dim=-1)
    log_energy = torch.log(mean_sq + eps)
    norm_log_energy = (log_energy - mean) / (std + eps)

    # 2. Relative RMS Energy
    rms = torch.sqrt(mean_sq + eps)
    max_rms = rms.max(dim=-1, keepdim=True)[0].clamp(min=eps)
    rel_rms = rms / max_rms

    # 3. Delta Log-Energy
    delta_log_energy = F.pad(log_energy[:, 1:] - log_energy[:, :-1], (1, 0), mode="constant", value=0.0)

    # 4. Zero-Crossing Rate
    signs = torch.sign(frames)
    zcr = torch.mean(torch.abs(signs[:, :, 1:] - signs[:, :, :-1]), dim=-1) / 2.0

    # Stack into (B, num_frames, 4)
    feat = torch.stack([norm_log_energy, rel_rms, delta_log_energy, zcr], dim=-1)

    # Align to target expected_frames
    if isinstance(expected_frames, int):
        target_len = expected_frames
    else:
        target_len = int(expected_frames.max().item())

    current_len = feat.shape[1]
    if current_len < target_len:
        feat = F.pad(feat, (0, 0, 0, target_len - current_len), mode="constant", value=0.0)
    elif current_len > target_len:
        feat = feat[:, :target_len, :]

    if orig_ndim == 1:
        feat = feat.squeeze(0)

    return feat
