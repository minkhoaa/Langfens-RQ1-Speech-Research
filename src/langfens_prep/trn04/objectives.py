"""LANGFENS RQ1-TRN-04 — Factorial Objectives Specification.

Supports:
1. "operation_ce": Standard 3-way Cross-Entropy (arms C0 and T2).
2. "class_focal": Class-Conditioned Focal Loss (arms T1 and T3).
   - Down-weights deletion penalty on high-frequency deletion phonemes {D, T, R}
     identified on the FIT cohort (56.69% of all FIT deletions).
   - Parameters: gamma=2.0, alpha_coronal_del=0.25, alpha_default=1.0.
   - Numerically stable under CUDA FP16 AMP (computed in float32).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from langfens_prep.p_sup.objectives import realized_ctc_loss, collapsed_binary_logit

OBJECTIVES = frozenset({"operation_ce", "class_focal"})

# Pinned coronal stops and rhotics in CTC phone index (1-based Arpabet)
# D: 9, R: 28, T: 31
CORONAL_RHOTIC_PHONE_IDS = (9, 28, 31)
DEFAULT_GAMMA: float = 2.0
DEFAULT_ALPHA_CORONAL_DEL: float = 0.25


def build_coronal_lookup(device: torch.device) -> torch.Tensor:
    """Builds a fast boolean lookup tensor for coronal/rhotic phonemes."""
    lookup = torch.zeros(41, dtype=torch.bool, device=device)
    lookup[list(CORONAL_RHOTIC_PHONE_IDS)] = True
    return lookup


def task_loss(
    logits: torch.Tensor,
    operation: torch.Tensor,
    mask: torch.Tensor,
    prompt_phone_ids: torch.Tensor,
    objective: str,
    gamma: float = DEFAULT_GAMMA,
    alpha_coronal_del: float = DEFAULT_ALPHA_CORONAL_DEL,
    eps: float = 1e-7
) -> torch.Tensor:
    """Computes the operation classification loss under the specified objective.

    Args:
        logits: (B, N, 3) raw model logits.
        operation: (B, N) target integer labels (0=correct, 1=substitution, 2=deletion).
        mask: (B, N) boolean prompt mask.
        prompt_phone_ids: (B, N) canonical prompt phone IDs (1..39).
        objective: "operation_ce" or "class_focal".
        gamma: Focal loss focusing parameter (default 2.0).
        alpha_coronal_del: Class conditioning weight for coronal deletions (default 0.25).
        eps: Small constant for numerical stability.

    Returns:
        Scalar loss tensor.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"Unregistered objective '{objective}'; expected one of {OBJECTIVES}")

    if logits.shape[:2] != operation.shape or mask.shape != operation.shape or mask.dtype != torch.bool:
        raise ValueError("Invalid loss shapes/mask")
    if not bool(mask.any()):
        raise ValueError("No operation supervision; not a successful learning batch")
    if not bool(((operation[mask] >= 0) & (operation[mask] <= 2)).all()):
        raise ValueError("Invalid operation labels (must be 0, 1, or 2)")

    # Cast to float32 for numerical stability in AMP
    z = logits[mask].float()
    y = operation[mask]
    p = prompt_phone_ids[mask]

    if not bool(torch.isfinite(z).all()):
        raise ValueError("Nonfinite logits detected in task_loss")

    if objective == "operation_ce":
        return F.cross_entropy(z, y, reduction="mean")

    elif objective == "class_focal":
        probs = F.softmax(z, dim=-1)
        p_t = probs.gather(1, y.unsqueeze(1)).squeeze(1)

        # Vectorized alpha calculation
        lookup = build_coronal_lookup(device=z.device)
        is_coronal_phone = lookup[p.clamp(0, 40)]
        is_coronal_del = (y == 2) & is_coronal_phone

        alpha = torch.ones_like(p_t)
        alpha = torch.where(is_coronal_del, torch.tensor(alpha_coronal_del, device=z.device, dtype=z.dtype), alpha)

        # Focal formulation: - alpha * (1 - p_t)^gamma * log(p_t + eps)
        focal_loss = - alpha * ((1.0 - p_t) ** gamma) * torch.log(p_t.clamp(min=eps))
        return focal_loss.mean()
