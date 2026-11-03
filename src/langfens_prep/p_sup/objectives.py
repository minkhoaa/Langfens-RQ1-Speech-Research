"""Forward-only objective algebra; callers alone control execution authorization."""
import torch
from torch.nn import functional as F

OBJECTIVES = {"operation_ce", "binary_collapsed"}


def collapsed_binary_logit(logits):
    return torch.logsumexp(logits[..., 1:], dim=-1) - logits[..., 0]


def task_loss(logits, operation, mask, objective):
    if objective not in OBJECTIVES:
        raise ValueError("unregistered objective")
    if logits.shape != (*operation.shape, 3) or mask.shape != operation.shape or mask.dtype != torch.bool:
        raise ValueError("invalid loss shapes/mask")
    if not bool(mask.any()):
        raise ValueError("no operation supervision; not a successful learning batch")
    if not bool(((operation[mask] >= 0) & (operation[mask] <= 2)).all()):
        raise ValueError("invalid operation labels")
    z, y = logits[mask].float(), operation[mask]
    if not bool(torch.isfinite(z).all()):
        raise ValueError("nonfinite logits")
    if objective == "operation_ce":
        return F.cross_entropy(z, y, reduction="mean")
    # All three output logits participate. Sub/del components are NOT identified subtypes.
    return F.binary_cross_entropy_with_logits(collapsed_binary_logit(z), (y != 0).float())


def realized_ctc_loss(logits, frame_lengths, targets, target_lengths):
    b, t, classes = logits.shape
    if classes != 40 or frame_lengths.shape != (b,) or target_lengths.shape != (b,) or targets.ndim != 2 or targets.shape[0] != b:
        raise ValueError("invalid CTC shapes")
    if frame_lengths.dtype != torch.long or target_lengths.dtype != torch.long or targets.dtype != torch.long:
        raise ValueError("CTC lengths/targets must be integer")
    if not bool(((frame_lengths > 0) & (frame_lengths <= t)).all()) or not bool(((target_lengths >= 0) & (target_lengths <= targets.shape[1])).all()):
        raise ValueError("invalid CTC lengths")
    for row, length, available in zip(targets, target_lengths.tolist(), frame_lengths.tolist()):
        sequence = row[:length]
        if not bool(((sequence >= 1) & (sequence <= 39)).all()):
            raise ValueError("blank/invalid phone in realized reference")
        minimum = length + int((sequence[1:] == sequence[:-1]).sum())
        if minimum > available:
            raise ValueError("infeasible repeated-phone CTC reference; refusing zero_infinity")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("nonfinite CTC logits")
    return F.ctc_loss(logits.float().log_softmax(-1).transpose(0, 1), targets,
                      frame_lengths, target_lengths, blank=0, reduction="mean", zero_infinity=False)
