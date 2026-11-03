"""TRAIN/offline-only adapter for already parsed annotations, never runtime input."""
from dataclasses import dataclass

import torch

from langfens_prep.phoneme_normalizer import PHONE_TO_CTC_INDEX, PHONE_TO_INDEX
from langfens_prep.textgrid_parser import L2ArcticUtteranceAnnotation


class LabelContractError(ValueError):
    pass


@dataclass(frozen=True)
class OperationLabels:
    canonical_ids: torch.Tensor
    operation: torch.Tensor
    binary: torch.Tensor
    realization: torch.Tensor
    ctc: torch.Tensor
    addition_gaps: tuple[int, ...]
    unknown_realized_intervals: tuple[int, ...]


def build_operation_labels(annotation: L2ArcticUtteranceAnnotation) -> OperationLabels:
    """Preserve strict operation identity and chronological realized CTC, without timing."""
    canonical, operation, binary, realization, ctc = [], [], [], [], []
    gaps, unknown = [], []
    mapping = {"correct": 0, "substitution": 1, "deletion": 2}
    for interval_index, item in enumerate(annotation.intervals):
        kind = item.error_type
        if kind not in (*mapping, "addition", "silence"):
            raise LabelContractError(f"unsupported operation: {kind}")
        if item.is_canonical != (kind in mapping):
            raise LabelContractError("canonical/operation mismatch")
        if item.is_canonical:
            if item.token_index != len(canonical):
                raise LabelContractError("canonical indices must be contiguous, including repeats")
            if item.canonical_phone not in PHONE_TO_CTC_INDEX:
                raise LabelContractError("unknown canonical identity; refusing sequence shortening")
            target = mapping[kind]
            strict = int(target != 0)
            if item.binary_label_strict != strict:
                raise LabelContractError("strict binary/operation mismatch")
            canonical.append(PHONE_TO_CTC_INDEX[item.canonical_phone])
            operation.append(target)
            binary.append(strict)
            valid = kind != "deletion" and not item.has_err_tag and item.realized_phone in PHONE_TO_INDEX
            realization.append(PHONE_TO_INDEX[item.realized_phone] if valid else -100)
        elif item.token_index is not None:
            raise LabelContractError("noncanonical item has canonical index")
        if kind == "addition":
            gaps.append(len(canonical))  # gap before next canonical position; includes trailing gap
        if kind in ("correct", "substitution", "addition"):
            if not item.has_err_tag and item.realized_phone in PHONE_TO_CTC_INDEX:
                ctc.append(PHONE_TO_CTC_INDEX[item.realized_phone])
            else:
                unknown.append(interval_index)
    if not canonical:
        raise LabelContractError("no canonical supervision")
    return OperationLabels(*(torch.tensor(x, dtype=torch.long) for x in
                             (canonical, operation, binary, realization, ctc)),
                           tuple(gaps), tuple(unknown))
