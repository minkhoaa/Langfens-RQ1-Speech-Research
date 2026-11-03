"""Frozen-emission, forward-only P-SCORE sidecar; no model or corpus loaders."""

from .ctc import ctc_forward, score_prompt, validate_emissions

__all__ = ["ctc_forward", "score_prompt", "validate_emissions"]
