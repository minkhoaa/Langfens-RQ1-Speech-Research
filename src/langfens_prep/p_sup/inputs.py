"""Gold-free feature/audio interfaces. No annotation reader, dataset or historical model imports."""
from dataclasses import dataclass
from typing import Mapping

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass(frozen=True)
class PromptFeatures:
    prompt_phone_ids: torch.Tensor
    features: torch.Tensor

    @classmethod
    def from_mapping(cls, value: Mapping) -> "PromptFeatures":
        if set(value) != {"prompt_phone_ids", "features"}:
            raise ValueError("runtime accepts only canonical prompt IDs and acoustic features")
        result = cls(**value)
        result.validate()
        return result

    def validate(self) -> None:
        p, f = self.prompt_phone_ids, self.features
        if p.ndim != 1 or not p.numel() or p.dtype != torch.long or not bool(((p >= 1) & (p <= 39)).all()):
            raise ValueError("nonempty canonical prompt must use integer CTC phone IDs 1..39")
        if f.ndim != 2 or not f.shape[0] or not f.shape[1] or not f.is_floating_point() or not bool(torch.isfinite(f).all()):
            raise ValueError("acoustic features must be finite nonempty frames x channels")
        if p.device != f.device:
            raise ValueError("prompt and features must share device")


def collate_prompt_features(items: list[PromptFeatures]) -> dict[str, torch.Tensor]:
    if not items:
        raise ValueError("empty batch")
    for item in items:
        if not isinstance(item, PromptFeatures):
            raise ValueError("runtime collator accepts only PromptFeatures")
        item.validate()
    device = items[0].features.device
    if len({(x.features.shape[1], x.features.device) for x in items}) != 1:
        raise ValueError("inconsistent feature dimension/device")
    phones = pad_sequence([x.prompt_phone_ids for x in items], batch_first=True)
    features = pad_sequence([x.features for x in items], batch_first=True)
    lengths = torch.tensor([x.features.shape[0] for x in items], device=device)
    return {"features": features, "frame_lengths": lengths, "prompt_phone_ids": phones,
            "prompt_lengths": torch.tensor([x.prompt_phone_ids.numel() for x in items], device=device)}
