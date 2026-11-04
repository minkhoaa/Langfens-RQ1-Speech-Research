"""LANGFENS RQ1-TRN-04 Package."""
from langfens_prep.trn04.features import extract_frame_acoustic_features, ACOUSTIC_FEATURE_DIM
from langfens_prep.trn04.model import (
    CompactFactorialOperationHead,
    FactorialHeadConfig,
    FactorialAudioAdapter,
    matched_initial_quad,
    parameter_inventory
)
from langfens_prep.trn04.objectives import task_loss
from langfens_prep.trn04.trainer import train_factorial_arm
from langfens_prep.trn04.evaluator import evaluate_factorial_checkpoint

__all__ = [
    "extract_frame_acoustic_features",
    "ACOUSTIC_FEATURE_DIM",
    "CompactFactorialOperationHead",
    "FactorialHeadConfig",
    "FactorialAudioAdapter",
    "matched_initial_quad",
    "parameter_inventory",
    "task_loss",
    "train_factorial_arm",
    "evaluate_factorial_checkpoint",
]
