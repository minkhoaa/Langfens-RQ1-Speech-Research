"""Compact acoustic/reference routing shared exactly by treatment and binary control.

No pretrained download/load API exists here. A future authorized remote runner supplies
fresh pinned frozen-backbone features (or an independently constructed audio encoder).
"""
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class HeadConfig:
    feature_dim: int = 768
    width: int = 128
    heads: int = 4
    max_prompt_length: int = 256
    dropout: float = 0.1


class CompactOperationHead(nn.Module):
    def __init__(self, config: HeadConfig = HeadConfig()):
        super().__init__()
        self.config = config
        self.acoustic_projection = nn.Linear(config.feature_dim, config.width)
        self.phone_embedding = nn.Embedding(40, config.width, padding_idx=0)
        self.position_embedding = nn.Embedding(config.max_prompt_length, config.width)
        self.attention = nn.MultiheadAttention(config.width, config.heads, config.dropout, batch_first=True)
        self.acoustic_norm = nn.LayerNorm(config.width)
        self.operation_head = nn.Sequential(nn.Linear(2 * config.width, config.width), nn.GELU(),
                                            nn.Dropout(config.dropout), nn.Linear(config.width, 3))
        self.ctc_head = nn.Linear(config.feature_dim, 40)

    def forward(self, *, features, frame_lengths, prompt_phone_ids, prompt_lengths):
        b, t, d = features.shape
        n = prompt_phone_ids.shape[1]
        if d != self.config.feature_dim or prompt_phone_ids.shape[0] != b or not b or not t or not n:
            raise ValueError("invalid input shapes")
        if n > self.config.max_prompt_length:
            raise ValueError("prompt exceeds registered position capacity")
        for lengths, limit in ((frame_lengths, t), (prompt_lengths, n)):
            if lengths.shape != (b,) or lengths.dtype != torch.long or not bool(((lengths > 0) & (lengths <= limit)).all()):
                raise ValueError("invalid nonempty lengths")
        if prompt_phone_ids.dtype != torch.long or not bool(torch.isfinite(features).all()):
            raise ValueError("invalid phone dtype/nonfinite features")
        prompt_mask = torch.arange(n, device=features.device)[None] < prompt_lengths[:, None]
        if not bool((((prompt_phone_ids >= 1) & (prompt_phone_ids <= 39)) | ~prompt_mask).all()):
            raise ValueError("unknown canonical phone")
        if bool((prompt_phone_ids[~prompt_mask] != 0).any()):
            raise ValueError("canonical padding must be zero")
        frame_mask = torch.arange(t, device=features.device)[None] >= frame_lengths[:, None]
        # Zero padded frames before both heads: no padded-value leakage through projections.
        clean = features.masked_fill(frame_mask[..., None], 0)
        acoustic = self.acoustic_projection(clean)
        phone = self.phone_embedding(prompt_phone_ids)
        query = phone + self.position_embedding(torch.arange(n, device=features.device))[None]
        attended, attention = self.attention(query, acoustic, acoustic, key_padding_mask=frame_mask)
        joined = torch.cat((self.acoustic_norm(attended), phone), dim=-1)
        logits = self.operation_head(joined).masked_fill(~prompt_mask[..., None], 0)
        return {"operation_logits": logits, "ctc_logits": self.ctc_head(clean),
                "prompt_mask": prompt_mask, "frame_lengths": frame_lengths,
                "attention": attention.masked_fill(~prompt_mask[..., None], 0)}


class FrozenAudioAdapter(nn.Module):
    """Injected encoder returns (features, frame_lengths); accepts audio+prompt, no gold."""
    def __init__(self, encoder: nn.Module, head: CompactOperationHead):
        super().__init__()
        self.encoder = encoder.requires_grad_(False).eval()
        self.head = head

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, *, audio, audio_lengths, prompt_phone_ids, prompt_lengths):
        with torch.no_grad():
            features, lengths = self.encoder(audio, audio_lengths)
        return self.head(features=features, frame_lengths=lengths,
                         prompt_phone_ids=prompt_phone_ids, prompt_lengths=prompt_lengths)


def matched_initial_pair(config: HeadConfig = HeadConfig(), seed: int = 1337):
    """Identical tensors, disjoint storage, isolated seed scope; never a checkpoint warmstart."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        treatment = CompactOperationHead(config)
        control = deepcopy(treatment)
    return treatment, control


def parameter_inventory(model: nn.Module) -> dict:
    return {name: {"shape": list(value.shape), "trainable": value.requires_grad,
                   "numel": value.numel()} for name, value in model.named_parameters()}
