"""LANGFENS RQ1-TRN-04 — Matched Factorial Architecture Specification.

All four experimental arms (C0, T1, T2, T3) share 100% identical capacity,
layer names, shapes, and initial tensor values (seed 1337).

Trainable Head Parameters: exactly 267,179 parameters.
Acoustic feature projection: nn.Linear(4, 128, bias=False).
When features are disabled (C0, T1), zero-valued features are supplied.
When features are enabled (T2, T3), waveform-derived features are supplied.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from torch import nn

from langfens_prep.trn04.features import extract_frame_acoustic_features, ACOUSTIC_FEATURE_DIM


@dataclass(frozen=True)
class FactorialHeadConfig:
    feature_dim: int = 768
    acoustic_dim: int = ACOUSTIC_FEATURE_DIM
    width: int = 128
    heads: int = 4
    max_prompt_length: int = 256
    dropout: float = 0.1


class CompactFactorialOperationHead(nn.Module):
    """Common 2x2 Factorial Operation Head with acoustic slot.

    Matched across all 4 experimental arms with identical parameter inventory (267,179 parameters).
    """
    def __init__(self, config: FactorialHeadConfig = FactorialHeadConfig()):
        super().__init__()
        self.config = config
        self.acoustic_projection = nn.Linear(config.feature_dim, config.width)
        # Shared architectural slot: bias=False ensures zero inputs give strictly zero contribution
        self.acoustic_feature_projection = nn.Linear(config.acoustic_dim, config.width, bias=False)
        self.phone_embedding = nn.Embedding(40, config.width, padding_idx=0)
        self.position_embedding = nn.Embedding(config.max_prompt_length, config.width)
        self.attention = nn.MultiheadAttention(config.width, config.heads, config.dropout, batch_first=True)
        self.acoustic_norm = nn.LayerNorm(config.width)
        self.operation_head = nn.Sequential(
            nn.Linear(2 * config.width, config.width),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.width, 3)
        )
        self.ctc_head = nn.Linear(config.feature_dim, 40)

    def forward(
        self,
        *,
        features: torch.Tensor,
        acoustic_features: torch.Tensor,
        frame_lengths: torch.Tensor,
        prompt_phone_ids: torch.Tensor,
        prompt_lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        b, t, d = features.shape
        n = prompt_phone_ids.shape[1]

        if d != self.config.feature_dim or prompt_phone_ids.shape[0] != b or not b or not t or not n:
            raise ValueError("invalid input shapes")
        if acoustic_features.shape != (b, t, self.config.acoustic_dim):
            raise ValueError(f"invalid acoustic_features shape: expected ({b}, {t}, {self.config.acoustic_dim}), got {acoustic_features.shape}")
        if n > self.config.max_prompt_length:
            raise ValueError("prompt exceeds registered position capacity")

        for lengths, limit in ((frame_lengths, t), (prompt_lengths, n)):
            if lengths.shape != (b,) or lengths.dtype != torch.long or not bool(((lengths > 0) & (lengths <= limit)).all()):
                raise ValueError("invalid nonempty lengths")

        if prompt_phone_ids.dtype != torch.long or not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(acoustic_features).all()):
            raise ValueError("invalid phone dtype/nonfinite features")

        prompt_mask = torch.arange(n, device=features.device)[None] < prompt_lengths[:, None]
        if not bool((((prompt_phone_ids >= 1) & (prompt_phone_ids <= 39)) | ~prompt_mask).all()):
            raise ValueError("unknown canonical phone")
        if bool((prompt_phone_ids[~prompt_mask] != 0).any()):
            raise ValueError("canonical padding must be zero")

        frame_mask = torch.arange(t, device=features.device)[None] >= frame_lengths[:, None]

        # Zero padded frames before projections: no padding leakage
        clean_features = features.masked_fill(frame_mask[..., None], 0.0)
        clean_acoustic = acoustic_features.masked_fill(frame_mask[..., None], 0.0)

        # Merge encoder features and acoustic slot features
        acoustic = self.acoustic_projection(clean_features) + self.acoustic_feature_projection(clean_acoustic)

        phone = self.phone_embedding(prompt_phone_ids)
        query = phone + self.position_embedding(torch.arange(n, device=features.device))[None]

        attended, attention = self.attention(query, acoustic, acoustic, key_padding_mask=frame_mask)
        joined = torch.cat((self.acoustic_norm(attended), phone), dim=-1)
        logits = self.operation_head(joined).masked_fill(~prompt_mask[..., None], 0.0)

        return {
            "operation_logits": logits,
            "ctc_logits": self.ctc_head(clean_features),
            "prompt_mask": prompt_mask,
            "frame_lengths": frame_lengths,
            "attention": attention.masked_fill(~prompt_mask[..., None], 0.0),
        }


class FactorialAudioAdapter(nn.Module):
    """Injected encoder returns (features, frame_lengths); accepts audio+prompt, no gold.

    Conditionally extracts 4D waveform acoustic features or supplies matched zero tensors.
    """
    def __init__(
        self,
        encoder: nn.Module,
        head: CompactFactorialOperationHead,
        features_enabled: bool = False
    ):
        super().__init__()
        self.encoder = encoder.requires_grad_(False).eval()
        self.head = head
        self.features_enabled = features_enabled

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(
        self,
        *,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        prompt_phone_ids: torch.Tensor,
        prompt_lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            features, lengths = self.encoder(audio, audio_lengths)

        b, t, _ = features.shape
        if self.features_enabled:
            # Runtime waveform-derived acoustic features (zero gold leakage)
            acoustic_features = extract_frame_acoustic_features(audio, lengths)
        else:
            # Feature-disabled slot: matched zeros
            acoustic_features = torch.zeros(b, t, self.head.config.acoustic_dim, device=features.device, dtype=features.dtype)

        return self.head(
            features=features,
            acoustic_features=acoustic_features,
            frame_lengths=lengths,
            prompt_phone_ids=prompt_phone_ids,
            prompt_lengths=prompt_lengths
        )


def matched_initial_quad(
    config: FactorialHeadConfig = FactorialHeadConfig(),
    seed: int = 1337
) -> Tuple[CompactFactorialOperationHead, CompactFactorialOperationHead, CompactFactorialOperationHead, CompactFactorialOperationHead]:
    """Generates four independently allocated, bit-identical model heads from pinned seed 1337.

    Returns: (c0, t1, t2, t3) with identical initial weights and parameter counts.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        c0 = CompactFactorialOperationHead(config)
        t1 = deepcopy(c0)
        t2 = deepcopy(c0)
        t3 = deepcopy(c0)
    return c0, t1, t2, t3


def parameter_inventory(model: nn.Module) -> Dict[str, Any]:
    """Returns parameter names, shapes, trainable status, and element counts."""
    return {
        name: {
            "shape": list(value.shape),
            "trainable": value.requires_grad,
            "numel": value.numel()
        }
        for name, value in model.named_parameters()
    }
