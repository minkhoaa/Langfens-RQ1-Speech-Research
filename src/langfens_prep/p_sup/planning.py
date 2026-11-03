"""Pure metadata accounting only. No optimizer/scaler/scheduler construction or steps."""
from dataclasses import dataclass
import math

import torch


@dataclass
class UpdateLedger:
    attempts: int = 0
    successful: int = 0
    skipped: int = 0

    def record_attempt(self, *, overflow: bool):
        if type(overflow) is not bool or self.successful >= 3600:
            raise ValueError("invalid attempt/budget exhausted")
        self.attempts += 1
        self.skipped += int(overflow)
        self.successful += int(not overflow)
        return not overflow  # future remote scheduler advances only on True

    @property
    def fixed_budget_reached(self):
        return self.successful == 3600


def learning_rate_multiplier(successful_updates: int):
    if type(successful_updates) is not int or not 0 <= successful_updates <= 3600:
        raise ValueError("invalid successful-update index")
    if successful_updates < 100:
        return successful_updates / 100
    return 0.5 * (1 + math.cos(math.pi * (successful_updates - 100) / (15000 - 100)))


def accumulation_groups(microbatches: int, accumulation: int = 2):
    """Flush final partial group; divide each loss by its actual group size remotely."""
    if microbatches <= 0 or accumulation <= 0:
        raise ValueError("positive batch counts required")
    return tuple(min(accumulation, microbatches - start) for start in range(0, microbatches, accumulation))


def independent_rng_pair(seed=1337):
    return torch.Generator(device="cpu").manual_seed(seed), torch.Generator(device="cpu").manual_seed(seed)
