"""Explicit TRAIN-only target transport; supervision is outside net_input."""
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence

from .inputs import PromptFeatures, collate_prompt_features
from .labels import OperationLabels


@dataclass(frozen=True)
class SupervisedFeatures:
    inputs: PromptFeatures
    labels: OperationLabels


def collate_supervised_features(items: list[SupervisedFeatures]) -> dict:
    if not items or any(not isinstance(x, SupervisedFeatures) for x in items):
        raise ValueError("empty or mixed runtime/supervised batch")
    for item in items:
        y = item.labels
        n = y.canonical_ids.numel()
        if not torch.equal(item.inputs.prompt_phone_ids, y.canonical_ids):
            raise ValueError("training canonical identities/length mismatch")
        if any(t.ndim != 1 or t.numel() != n or t.dtype != torch.long for t in
               (y.operation, y.binary, y.realization)):
            raise ValueError("misaligned targets")
        if not bool(((y.operation >= 0) & (y.operation <= 2)).all()):
            raise ValueError("invalid operation target")
        if not torch.equal(y.binary, (y.operation != 0).long()):
            raise ValueError("strict binary mismatch")
        if not bool(((y.realization == -100) | ((y.realization >= 0) & (y.realization < 39))).all()):
            raise ValueError("invalid realization target")
        if bool(((y.operation == 2) & (y.realization != -100)).any()):
            raise ValueError("deletion has no realization target")
        if y.ctc.ndim != 1 or y.ctc.dtype != torch.long or not bool(((y.ctc >= 1) & (y.ctc <= 39)).all()):
            raise ValueError("CTC targets contain blank/invalid phone")
    net = collate_prompt_features([x.inputs for x in items])
    targets = {name: pad_sequence([getattr(x.labels, field) for x in items], batch_first=True,
                                  padding_value=-100)
               for name, field in (("operation", "operation"), ("binary", "binary"),
                                   ("realization", "realization"))}
    targets["operation_mask"] = targets["operation"] != -100
    targets["binary_mask"] = targets["binary"] != -100
    targets["realization_mask"] = targets["realization"] != -100
    targets["ctc"] = pad_sequence([x.labels.ctc for x in items], batch_first=True, padding_value=0)
    targets["ctc_lengths"] = torch.tensor([x.labels.ctc.numel() for x in items], device=net["features"].device)
    return {"net_input": net, "targets": targets,
            "addition_gaps": [x.labels.addition_gaps for x in items],
            "unknown_realized_intervals": [x.labels.unknown_realized_intervals for x in items]}
