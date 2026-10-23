import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from langfens_prep.losses import (
    MaskedBCEWithLogitsLoss,
    MaskedCrossEntropyLoss,
    validate_ctc_contract
)
from langfens_prep.alignment import pool_frames_to_canonical_phones

@dataclass
class LangfensMDDOutput:
    ctc_logits: torch.Tensor
    binary_logits: Optional[torch.Tensor] = None
    diagnosis_logits: Optional[torch.Tensor] = None
    frame_features: Optional[torch.Tensor] = None
    input_lengths: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    loss_dict: Optional[Dict[str, torch.Tensor]] = None

class ConvFeatureExtractionLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias)
        self.layer_norm = nn.GroupNorm(num_groups=1, num_channels=out_channels, eps=1e-5)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T)
        x = self.conv(x)
        x = self.layer_norm(x)
        x = self.activation(x)
        return x

class Wav2Vec2FeatureEncoder(nn.Module):
    """
    Standard 7-layer Convolutional Feature Extractor matching Wav2Vec 2.0 specifications.
    Downsampling factor = 5 * 2 * 2 * 2 * 2 * 2 * 2 = 320x (16,000 Hz -> 50 Hz frame rate).
    """
    def __init__(self, conv_dim: int = 512):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            ConvFeatureExtractionLayer(1, conv_dim, kernel_size=10, stride=5, bias=False),
            ConvFeatureExtractionLayer(conv_dim, conv_dim, kernel_size=3, stride=2, bias=False),
            ConvFeatureExtractionLayer(conv_dim, conv_dim, kernel_size=3, stride=2, bias=False),
            ConvFeatureExtractionLayer(conv_dim, conv_dim, kernel_size=3, stride=2, bias=False),
            ConvFeatureExtractionLayer(conv_dim, conv_dim, kernel_size=3, stride=2, bias=False),
            ConvFeatureExtractionLayer(conv_dim, conv_dim, kernel_size=2, stride=2, bias=False),
            ConvFeatureExtractionLayer(conv_dim, conv_dim, kernel_size=2, stride=2, bias=False),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (batch_size, time)
        if x.dim() == 2:
            x = x.unsqueeze(1) # (B, 1, T)
        for layer in self.conv_layers:
            x = layer(x)
        return x.transpose(1, 2) # (B, T_frames, conv_dim)

class PositionalConvEmbedding(nn.Module):
    """Depthwise convolutional relative positional encoding."""
    def __init__(self, hidden_dim: int, kernel_size: int = 128, groups: int = 16):
        super().__init__()
        self.conv = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=groups
        )
        self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        residual = x
        out = self.conv(x.transpose(1, 2))
        out = out[:, :, :x.shape[1]]
        out = self.activation(out)
        return residual + out.transpose(1, 2)

class LangfensWav2Vec2MDD(nn.Module):
    """
    Multi-Task Neural Acoustic Architecture for Phoneme-level MDD.
    Combines:
    1. 7-layer Convolutional Feature Extractor (320x downsampling).
    2. Multi-layer Transformer Context Encoder with relative positional encoding.
    3. CTC Phoneme Recognition Head (40 classes: 1 blank + 39 standard ARPAbet phones).
    4. Canonical Token Binary Mispronunciation Detection Head (Masked BCE loss).
    5. Token-Level Substitution Diagnosis Head (39 ARPAbet classes, Masked CE loss).
    """
    def __init__(
        self,
        hidden_dim: int = 768,
        conv_dim: int = 512,
        num_transformer_layers: int = 4, # Configurable (12 for full base model, 2-4 for fast smoke tests)
        num_attention_heads: int = 8,
        intermediate_dim: int = 2048,
        num_canonical_phones: int = 40,
        phone_embed_dim: int = 128,
        num_ctc_classes: int = 40, # 0 = blank, 1..39 = standard ARPAbet phones
        num_diagnosis_classes: int = 39, # 0..38 = 39 ARPAbet classes
        dropout: float = 0.1,
        ctc_weight: float = 1.0,
        binary_weight: float = 1.0,
        diagnosis_weight: float = 0.5
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ctc_weight = ctc_weight
        self.binary_weight = binary_weight
        self.diagnosis_weight = diagnosis_weight
        self.gradient_checkpointing = False

        # 1. Convolutional feature extractor
        self.feature_extractor = Wav2Vec2FeatureEncoder(conv_dim=conv_dim)
        self.feature_projection = nn.Linear(conv_dim, hidden_dim)
        self.pos_conv_embed = PositionalConvEmbedding(hidden_dim=hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # 2. Transformer Context Network
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # 3. Canonical Phone Embedding (for conditioning binary detection head)
        self.canonical_phone_embedding = nn.Embedding(
            num_embeddings=num_canonical_phones,
            embedding_dim=phone_embed_dim,
            padding_idx=0
        )

        # 4. Multi-Task Heads
        # Head 1: Realized CTC Phone Recognition Head (Linear -> 40 classes)
        self.ctc_head = nn.Linear(hidden_dim, num_ctc_classes)

        # Head 2: Canonical Binary Detection Head (Acoustic Frame + Phone Embedding -> 1 logit)
        self.binary_head = nn.Sequential(
            nn.Linear(hidden_dim + phone_embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

        # Head 3: Substitution Diagnosis Head (Acoustic Frame -> 39 phone classes)
        self.diagnosis_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_diagnosis_classes)
        )

        # 5. Dedicated Loss Functions
        self.ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
        self.binary_loss_fn = MaskedBCEWithLogitsLoss(ignore_index=-100)
        self.diagnosis_loss_fn = MaskedCrossEntropyLoss(ignore_index=-100)

    def freeze_feature_encoder(self):
        """Freezes all parameters in the 7-layer convolutional waveform encoder."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def gradient_checkpointing_enable(self):
        """Enables gradient checkpointing for memory-constrained training."""
        self.gradient_checkpointing = True

    def _get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Computes downsampled acoustic frame length from raw audio sample length."""
        def _conv_out_length(input_length, filter_size, stride):
            return torch.div(input_length - filter_size, stride, rounding_mode="floor") + 1

        lengths = input_lengths
        lengths = _conv_out_length(lengths, 10, 5)
        for _ in range(4):
            lengths = _conv_out_length(lengths, 3, 2)
        lengths = _conv_out_length(lengths, 2, 2)
        lengths = _conv_out_length(lengths, 2, 2)
        return lengths.clamp(min=1)

    def forward(
        self,
        waveforms: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        canonical_phone_indices: Optional[torch.Tensor] = None,
        phone_intervals: Optional[List[List[Tuple[float, float]]]] = None,
        target_binary: Optional[torch.Tensor] = None,
        target_ctc: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        target_diagnosis: Optional[torch.Tensor] = None
    ) -> LangfensMDDOutput:
        batch_size = waveforms.shape[0]
        device = waveforms.device

        if audio_lengths is None:
            audio_lengths = torch.tensor([waveforms.shape[1]] * batch_size, device=device, dtype=torch.long)

        # 1. Convolutional feature extraction (downsampling 320x)
        conv_feats = self.feature_extractor(waveforms)
        hidden_states = self.feature_projection(conv_feats)
        hidden_states = self.pos_conv_embed(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # 2. Transformer Context Network
        if self.gradient_checkpointing and self.training:
            hidden_states = torch.utils.checkpoint.checkpoint(
                self.transformer_encoder, hidden_states, use_reentrant=False
            )
        else:
            hidden_states = self.transformer_encoder(hidden_states)

        frame_output_lengths = self._get_feat_extract_output_lengths(audio_lengths)

        # 3. Head 1: CTC Phoneme Recognition
        ctc_logits = self.ctc_head(hidden_states)

        # 4. Heads 2 & 3: Token-level Binary Detection & Diagnosis
        binary_logits = None
        diagnosis_logits = None

        if phone_intervals is not None and canonical_phone_indices is not None:
            # Pool acoustic frames to canonical token intervals
            pooled_acoustic, phone_mask = pool_frames_to_canonical_phones(
                frame_features=hidden_states,
                intervals_per_batch=phone_intervals,
                sample_rate=16000,
                frame_stride_samples=320,
                pooling_mode="mean"
            )

            # Retrieve canonical phone embeddings
            phone_embeds = self.canonical_phone_embedding(canonical_phone_indices)

            # Ensure matching token lengths
            min_len = min(pooled_acoustic.shape[1], phone_embeds.shape[1])
            pooled_acoustic = pooled_acoustic[:, :min_len, :]
            phone_embeds = phone_embeds[:, :min_len, :]

            # Binary Head: Concat pooled acoustic frame with canonical phone embedding
            binary_input = torch.cat([pooled_acoustic, phone_embeds], dim=-1)
            binary_logits = self.binary_head(binary_input).squeeze(-1) # (B, num_phones)

            # Diagnosis Head: Acoustic frame mapped to 39 phone classes
            diagnosis_logits = self.diagnosis_head(pooled_acoustic) # (B, num_phones, 39)

        # 5. Multi-Task Loss Calculation
        loss = None
        loss_dict = {}

        if target_ctc is not None and target_lengths is not None:
            # CTCLoss expects log_probs as (T, B, C)
            log_probs = F.log_softmax(ctc_logits.transpose(0, 1), dim=-1)
            ctc_loss = self.ctc_loss_fn(
                log_probs, target_ctc, frame_output_lengths, target_lengths
            )
            loss_dict["ctc_loss"] = ctc_loss

        if target_binary is not None and binary_logits is not None:
            # Truncate or match sequence dimensions if needed
            t_bin = target_binary[:, :binary_logits.shape[1]]
            b_loss = self.binary_loss_fn(binary_logits, t_bin)
            loss_dict["binary_loss"] = b_loss

        if target_diagnosis is not None and diagnosis_logits is not None:
            t_diag = target_diagnosis[:, :diagnosis_logits.shape[1]]
            d_loss = self.diagnosis_loss_fn(diagnosis_logits, t_diag)
            loss_dict["diagnosis_loss"] = d_loss

        if loss_dict:
            total_loss = torch.tensor(0.0, device=device)
            if "ctc_loss" in loss_dict:
                total_loss = total_loss + self.ctc_weight * loss_dict["ctc_loss"]
            if "binary_loss" in loss_dict:
                total_loss = total_loss + self.binary_weight * loss_dict["binary_loss"]
            if "diagnosis_loss" in loss_dict:
                total_loss = total_loss + self.diagnosis_weight * loss_dict["diagnosis_loss"]
            loss = total_loss

        return LangfensMDDOutput(
            ctc_logits=ctc_logits,
            binary_logits=binary_logits,
            diagnosis_logits=diagnosis_logits,
            frame_features=hidden_states,
            input_lengths=frame_output_lengths,
            loss=loss,
            loss_dict=loss_dict if loss_dict else None
        )
