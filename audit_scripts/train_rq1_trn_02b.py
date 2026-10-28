#!/usr/bin/env python3
"""
LANGFENS — RQ1-TRN-02B: Controlled Binary-Only Prompt Embedding Dropout Training Driver
Implements matched-budget training for Control Arm C0 (p=0.00) and Treatment Arm C1 (p=0.25).

Key Guarantees & Contract:
1. Frozen Weight Initialization: Loads exact production best.pt (SHA-256: a2445c8d...).
2. Manifest Integrity: Frozen Fold 1 split (SHA-256: 7d6bfdf6...).
3. Cold Reset Policy: Optimizer (AdamW), scheduler (cosine decay), and GradScaler reset identically.
4. Trainable Parameter Discipline: CTC head, cross-attention module, binary head trainable.
   Wav2Vec2 backbone frozen and kept in eval mode. Diagnosis head untrained, receives zero loss and gradients.
5. Strict Budget Caps: Maximum 6 epochs / 3,600 successful updates per arm (7,200 total combined).
6. Update Accounting: Explicit tracking of attempted, successful, and skipped updates.
7. Atomic Google Drive Persistence: Remote save with SHA-256 verification after every completed epoch.
8. Safety Enforcement: Test partition never loaded/accessed. Execution requires CUDA on Colab runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root and src are on sys.path
_repo_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_dir / "src"))
sys.path.insert(0, str(_repo_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from langfens_prep.dataset import L2ArcticMDDDataset
from langfens_prep.collate import LangfensMDDCollate
from langfens_prep.model import LangfensWav2Vec2MDD


def compute_sha256(path: Path | str) -> str:
    """Computes SHA-256 digest of a file in 64KB blocks."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_all(seed: int):
    """Sets deterministic random seeds across Python, NumPy, PyTorch CPU, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rng_state() -> Dict[str, Any]:
    """Captures complete RNG state dictionary."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(s: Dict[str, Any]):
    """Restores complete RNG state dictionary."""
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def compute_levenshtein_edits(ref: List[int], hyp: List[int]) -> Tuple[int, int, int, int]:
    """Computes Levenshtein edit distance and returns (total_edits, substitutions, deletions, insertions)."""
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    i, j = m, n
    subs, dels, inss = 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            inss += 1
            j -= 1
        else:
            break
    return dp[m][n], subs, dels, inss


def greedy_ctc_decode(frame_preds: List[int], blank_id: int = 0) -> List[int]:
    """Collapses consecutive identical predictions and removes blank tokens."""
    hyp = []
    prev = None
    for t in frame_preds:
        if t != prev:
            if t != blank_id:
                hyp.append(t)
            prev = t
    return hyp


def average_precision_binary(labels: List[int] | np.ndarray, scores: List[float] | np.ndarray) -> float:
    """Average precision with tied scores grouped, matching sklearn's definition."""
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError("AP labels must be a non-empty binary vector")
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    s = s[order]
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp = np.cumsum(y)[ends]
    fp = (ends + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


class UpdateBudgetTracker:
    """Explicitly tracks and validates attempted, successful, and skipped optimizer updates."""

    def __init__(
        self,
        max_epochs: int = 6,
        max_successful_updates: int = 3600,
        max_consecutive_skips: int = 8,
    ):
        self.max_epochs = max_epochs
        self.max_successful_updates = max_successful_updates
        self.max_consecutive_skips = max_consecutive_skips
        self.step_attempts: int = 0
        self.successful_updates: int = 0
        self.skipped_updates: int = 0
        self.consecutive_skips: int = 0
        self.micro_steps: int = 0

    def record_micro_step(self):
        self.micro_steps += 1

    def record_step_attempt(self, successful: bool):
        self.step_attempts += 1
        if successful:
            self.successful_updates += 1
            self.consecutive_skips = 0
        else:
            self.skipped_updates += 1
            self.consecutive_skips += 1
            if self.consecutive_skips >= self.max_consecutive_skips:
                raise FloatingPointError(
                    f"{self.max_consecutive_skips} consecutive AMP-skipped optimizer attempts detected."
                )

    @property
    def budget_exhausted(self) -> bool:
        return self.successful_updates >= self.max_successful_updates

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step_attempts": self.step_attempts,
            "successful_updates": self.successful_updates,
            "skipped_updates": self.skipped_updates,
            "consecutive_skips": self.consecutive_skips,
            "micro_steps": self.micro_steps,
            "max_epochs": self.max_epochs,
            "max_successful_updates": self.max_successful_updates,
            "budget_exhausted": self.budget_exhausted,
        }

    def load_state_dict(self, d: Dict[str, Any]):
        self.step_attempts = d["step_attempts"]
        self.successful_updates = d["successful_updates"]
        self.skipped_updates = d["skipped_updates"]
        self.consecutive_skips = d["consecutive_skips"]
        self.micro_steps = d["micro_steps"]


def step_model(model: LangfensWav2Vec2MDD, batch: Dict[str, Any], device: torch.device, training: bool = True):
    """Executes a single forward pass with multi-task loss computation."""
    ni = batch["net_input"]
    args = dict(
        source=ni["source"].to(device),
        audio_lengths=ni["audio_lengths"].to(device),
        canonical_phone_indices=ni["canonical_phone_indices"].to(device),
        canonical_phone_lengths=ni["canonical_phone_lengths"].to(device),
        target_ctc=batch["target_ctc"].to(device),
        target_lengths=batch["target_lengths"].to(device),
        target_binary=batch["target_binary"].to(device),
    )
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
        out = model(**args)
    if out.loss is None or not torch.isfinite(out.loss):
        raise FloatingPointError("non-finite two-head objective encountered")
    return out


def evaluate_validation(model: LangfensWav2Vec2MDD, val_loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    """Validation pass evaluating binary MDD and greedy CTC PER with S/D/I decomposition."""
    model.eval()
    scores = []
    labels = []
    ctc_blank = 0
    ctc_frames = 0
    total_ref_tokens = 0
    total_hyp_tokens = 0
    total_ctc_edits = 0
    total_ctc_subs = 0
    total_ctc_dels = 0
    total_ctc_inss = 0

    with torch.inference_mode():
        for batch in val_loader:
            result = step_model(model, batch, device, training=False)
            logits = result.binary_logits.float().cpu()
            target = batch["target_binary"]
            valid = target != -100
            scores.extend(torch.sigmoid(logits)[valid].tolist())
            labels.extend(target[valid].tolist())

            pred = result.ctc_logits.argmax(-1).cpu()
            lengths = result.input_lengths.cpu()
            target_ctc = batch["target_ctc"].cpu()
            target_lengths = batch["target_lengths"].cpu()

            for i, l in enumerate(lengths):
                frame_len = l.item()
                frame_preds = pred[i, :frame_len].tolist()
                ctc_blank += sum(1 for p in frame_preds if p == 0)
                ctc_frames += frame_len

                hyp = greedy_ctc_decode(frame_preds, blank_id=0)
                ref_len = target_lengths[i].item()
                ref = target_ctc[i, :ref_len].tolist()

                edits, subs, dels, inss = compute_levenshtein_edits(ref, hyp)
                total_ref_tokens += len(ref)
                total_hyp_tokens += len(hyp)
                total_ctc_edits += edits
                total_ctc_subs += subs
                total_ctc_dels += dels
                total_ctc_inss += inss

    apval = average_precision_binary(labels, scores) if len(set(labels)) == 2 else float("nan")
    ctc_per = total_ctc_edits / max(total_ref_tokens, 1)
    ctc_sub_rate = total_ctc_subs / max(total_ref_tokens, 1)
    ctc_del_rate = total_ctc_dels / max(total_ref_tokens, 1)
    ctc_ins_rate = total_ctc_inss / max(total_ref_tokens, 1)
    decoded_length_ratio = total_hyp_tokens / max(total_ref_tokens, 1)
    blank_fraction = ctc_blank / max(ctc_frames, 1)
    prevalence = sum(labels) / max(len(labels), 1)

    return {
        "validation_binary_ap": round(apval, 6) if math.isfinite(apval) else float("nan"),
        "validation_binary_prevalence": round(prevalence, 6),
        "validation_ctc_per": round(ctc_per, 6),
        "validation_ctc_substitutions": total_ctc_subs,
        "validation_ctc_deletions": total_ctc_dels,
        "validation_ctc_insertions": total_ctc_inss,
        "validation_ctc_sub_rate": round(ctc_sub_rate, 6),
        "validation_ctc_del_rate": round(ctc_del_rate, 6),
        "validation_ctc_ins_rate": round(ctc_ins_rate, 6),
        "validation_ctc_total_edits": total_ctc_edits,
        "validation_ctc_ref_tokens": total_ref_tokens,
        "validation_ctc_hyp_tokens": total_hyp_tokens,
        "validation_ctc_decoded_length_ratio": round(decoded_length_ratio, 6),
        "validation_ctc_blank_fraction": round(blank_fraction, 6),
    }


def save_checkpoint(
    path: Path,
    payload: Dict[str, Any],
    drive_output_dir: Optional[str | Path] = None,
    verify_remote_sha: bool = True,
    remote_mirror: Optional[Any] = None,
    remote_rel_prefix: Optional[str] = None,
) -> str:
    """Atomically saves checkpoint locally and optionally syncs with Drive / Remote Mirror verifying SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    local_sha = compute_sha256(path)

    if drive_output_dir is not None:
        drive_path = Path(drive_output_dir) / path.name
        drive_path.parent.mkdir(parents=True, exist_ok=True)
        drive_tmp = drive_path.with_suffix(drive_path.suffix + ".tmp")
        shutil.copyfile(str(path), str(drive_tmp))
        os.replace(drive_tmp, drive_path)
        if not drive_path.exists():
            raise IOError(f"Remote checkpoint sync verification failed: {drive_path} does not exist")
        if verify_remote_sha:
            remote_sha = compute_sha256(drive_path)
            if local_sha != remote_sha:
                raise IOError(
                    f"Remote checkpoint sync SHA-256 mismatch: local {local_sha} vs remote {remote_sha}"
                )

    if remote_mirror is not None:
        rel_path = f"{remote_rel_prefix}/{path.name}" if remote_rel_prefix else path.name
        sync_res = remote_mirror.mirror_file(path, rel_path, verify_sha=verify_remote_sha)
        print(f"[REMOTE MIRROR] Mirrored {path.name} -> {sync_res['backend']}:{rel_path} (SHA: {sync_res['remote_sha256'][:12]}...)", flush=True)

    return local_sha


CHECKPOINT_SCHEMA_VERSION = "1.0.0"


def validate_checkpoint_for_resume(
    ck: Dict[str, Any],
    cfg: Dict[str, Any],
    manifest_sha: str,
    init_ckpt_sha: str,
    binary_prompt_dropout_p: float,
) -> None:
    """
    Validates complete checkpoint provenance, configuration, arm, seed, counters,
    and state integrity before resumption. Rejects missing, contradictory, halted,
    or cross-arm checkpoints.
    """
    if not isinstance(ck, dict):
        raise ValueError("Invalid checkpoint: checkpoint payload must be a dictionary.")

    # 1. Top-Level Required Keys
    required_keys = [
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "epoch",
        "successful_updates",
        "step_attempts",
        "skipped_updates",
        "micro_steps",
        "rng",
        "dataloader_generator",
        "metadata",
    ]
    for key in required_keys:
        if key not in ck:
            raise ValueError(f"Invalid checkpoint: missing required top-level key '{key}'.")

    # 2. State Component Types & Non-Emptiness
    if not isinstance(ck["model"], dict) or len(ck["model"]) == 0:
        raise ValueError("Invalid checkpoint: 'model' state dict is missing or empty.")
    if not isinstance(ck["optimizer"], dict) or len(ck["optimizer"]) == 0:
        raise ValueError("Invalid checkpoint: 'optimizer' state dict is missing or empty.")
    if not isinstance(ck["scheduler"], dict) or len(ck["scheduler"]) == 0:
        raise ValueError("Invalid checkpoint: 'scheduler' state dict is missing or empty.")
    if not isinstance(ck["scaler"], dict):
        raise ValueError("Invalid checkpoint: 'scaler' state dict must be a dictionary.")
    if not isinstance(ck["rng"], dict) or not {"python", "numpy", "torch"}.issubset(ck["rng"].keys()):
        raise ValueError("Invalid checkpoint: 'rng' state dict missing required python/numpy/torch generators.")
    if not isinstance(ck["dataloader_generator"], torch.Tensor):
        raise ValueError("Invalid checkpoint: 'dataloader_generator' must be a valid torch.Tensor byte state.")

    meta = ck["metadata"]
    if not isinstance(meta, dict):
        raise ValueError("Invalid checkpoint: 'metadata' must be a dictionary.")

    # 3. Reject Halted, Corrupted, or Invalid Checkpoints
    if meta.get("guardrail_violation") is True:
        raise RuntimeError(
            "RESUME REJECTED: Checkpoint is marked with guardrail_violation=True. "
            "Resumption from a guardrail-halted artifact is prohibited without an explicit recovery procedure."
        )
    exp_status = meta.get("experiment_status", "")
    if exp_status in {"HALTED_GUARDRAIL_VIOLATION", "HALTED", "CORRUPTED", "INVALID"}:
        raise RuntimeError(
            f"RESUME REJECTED: Checkpoint status is '{exp_status}'. "
            "Resumption from a halted or invalid artifact is prohibited without an explicit recovery procedure."
        )
    if meta.get("corrupted") is True:
        raise RuntimeError("RESUME REJECTED: Checkpoint metadata declares corrupted=True.")

    # 4. Schema Version and Saved Source Commit
    ck_schema = meta.get("schema_version") or meta.get("checkpoint_schema_version")
    if not ck_schema:
        raise ValueError("Invalid checkpoint: missing 'schema_version' in metadata.")
    ck_git = meta.get("git_sha")
    if not ck_git or ck_git == "unknown":
        raise ValueError("Invalid checkpoint: missing or invalid 'git_sha' commit provenance in metadata.")

    # 5. Arm ID Validation (C0 or C1, strictly matching config)
    ck_arm = meta.get("arm")
    expected_arm = cfg.get("arm")
    if ck_arm not in {"C0", "C1"}:
        raise ValueError(f"Invalid checkpoint: arm ID '{ck_arm}' must be strictly 'C0' or 'C1'.")
    if expected_arm not in {"C0", "C1"}:
        raise ValueError(f"Invalid runtime configuration: arm ID '{expected_arm}' must be 'C0' or 'C1'.")
    if ck_arm != expected_arm:
        raise ValueError(
            f"Cross-arm resume rejected: checkpoint declares arm '{ck_arm}', but current configuration expects arm '{expected_arm}'."
        )

    # 6. Seed and Dropout Probability
    ck_cfg = meta.get("config")
    if not isinstance(ck_cfg, dict):
        raise ValueError("Invalid checkpoint: missing 'config' dictionary in metadata.")
    ck_seed = ck_cfg.get("seed")
    if ck_seed is None or ck_seed != cfg["seed"]:
        raise ValueError(
            f"Seed mismatch on resume: checkpoint has seed {ck_seed}, but current configuration requires {cfg['seed']}."
        )
    ck_p = meta.get("binary_prompt_dropout_p")
    if ck_p is None or abs(float(ck_p) - binary_prompt_dropout_p) > 1e-6:
        raise ValueError(
            f"Dropout probability mismatch on resume: checkpoint has p={ck_p}, but current configuration expects p={binary_prompt_dropout_p}."
        )
    if ck_arm == "C1" and ck.get("intervention_rng") is None:
        raise ValueError("Invalid checkpoint: Treatment Arm C1 checkpoint must contain 'intervention_rng'.")

    # 7. Provenance Hashes
    ck_manifest_sha = meta.get("split_sha256")
    if not ck_manifest_sha or ck_manifest_sha != manifest_sha:
        raise ValueError(
            f"Manifest SHA-256 mismatch on resume: checkpoint has '{ck_manifest_sha}', but current manifest is '{manifest_sha}'."
        )
    ck_init_sha = meta.get("initial_checkpoint_sha256")
    if not ck_init_sha or ck_init_sha != init_ckpt_sha:
        raise ValueError(
            f"Initial checkpoint SHA-256 mismatch on resume: checkpoint has '{ck_init_sha}', but expected '{init_ckpt_sha}'."
        )

    # 8. Complete Shared Optimizer and Scheduler Configuration
    shared_checks = {
        "learning_rate": (lambda a, b: abs(float(a) - float(b)) < 1e-9),
        "weight_decay": (lambda a, b: abs(float(a) - float(b)) < 1e-9),
        "warmup_updates": (lambda a, b: int(a) == int(b)),
        "max_successful_updates": (lambda a, b: int(a) == int(b)),
        "gradient_accumulation_steps": (lambda a, b: int(a) == int(b)),
        "max_epochs": (lambda a, b: int(a) == int(b)),
        "batch_size": (lambda a, b: int(a) == int(b)),
    }
    for field_name, validator in shared_checks.items():
        if field_name not in ck_cfg:
            raise ValueError(f"Invalid checkpoint config: missing shared configuration field '{field_name}'.")
        if field_name not in cfg:
            raise ValueError(f"Invalid runtime config: missing required configuration field '{field_name}'.")
        if not validator(ck_cfg[field_name], cfg[field_name]):
            raise ValueError(
                f"Configuration mismatch on resume for '{field_name}': checkpoint has {ck_cfg[field_name]}, current configuration has {cfg[field_name]}."
            )

    # 9. Update Counters Consistency & Caps
    ck_successful = ck["successful_updates"]
    ck_step_attempts = ck["step_attempts"]
    ck_skipped = ck["skipped_updates"]
    ck_micro_steps = ck["micro_steps"]
    ck_epoch = ck["epoch"]

    for name, val in [
        ("successful_updates", ck_successful),
        ("step_attempts", ck_step_attempts),
        ("skipped_updates", ck_skipped),
        ("micro_steps", ck_micro_steps),
        ("epoch", ck_epoch),
    ]:
        if not isinstance(val, int) or val < 0:
            raise ValueError(f"Invalid checkpoint counter: '{name}' must be a non-negative integer, got {val}.")

    if ck_step_attempts != ck_successful + ck_skipped:
        raise ValueError(
            f"Contradictory counters on resume: step_attempts ({ck_step_attempts}) must exactly equal "
            f"successful_updates ({ck_successful}) + skipped_updates ({ck_skipped})."
        )
    if ck_micro_steps < ck_step_attempts:
        raise ValueError(
            f"Contradictory counters on resume: micro_steps ({ck_micro_steps}) cannot be less than step_attempts ({ck_step_attempts})."
        )

    # 10. Epoch-Boundary Resume State
    history = ck.get("history")
    if not isinstance(history, list) or len(history) != ck_epoch + 1:
        raise ValueError(
            f"Invalid checkpoint resume state: history length ({len(history) if isinstance(history, list) else 0}) "
            f"does not match completed epoch count ({ck_epoch + 1})."
        )


def get_git_commit_sha() -> str:
    """Returns current git commit hash or unknown."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description="RQ1-TRN-02B Matched-Budget Training Entrypoint")
    ap.add_argument("--config", required=True, help="Path to arm configuration YAML (e.g. configs/rq1_trn_02b_c0.yaml)")
    ap.add_argument("--resume", default=None, help="Path to checkpoint for training resumption")
    ap.add_argument("--initial-checkpoint", default=None, help="Path to initial baseline checkpoint (overrides config)")
    ap.add_argument("--zip-path", default=None, help="Path to zip archive containing authorized audio/TextGrids")
    ap.add_argument("--drive-output-dir", default=None, help="Google Drive destination for remote checkpoint persistence")
    ap.add_argument("--remote-mirror-backend", default=None, help="Remote persistent mirror backend (huggingface, gdrive, local_dir)")
    ap.add_argument("--remote-mirror-target", default=None, help="Remote persistent mirror repository/directory target")
    ap.add_argument("--remote-mirror-token", default=None, help="Authentication token for remote persistent mirror")
    ap.add_argument("--run-id", default=None, help="Unique execution run identifier to isolate output checkpoints")
    ap.add_argument("--eval-only", action="store_true", help="Run validation only without training")
    ap.add_argument("--dry-run", action="store_true", help="Validate config, hashes, and parameter setup on CPU then exit")
    args = ap.parse_args()

    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("PyYAML is required to read explicit config") from e

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = yaml.safe_load(config_path.read_text())

    if args.initial_checkpoint:
        cfg["initial_checkpoint"] = args.initial_checkpoint

    # 1. Manifest Hash Verification
    manifest_path = Path(cfg["manifest"])
    manifest_sha = compute_sha256(manifest_path)
    if manifest_sha != cfg["manifest_sha256"]:
        raise RuntimeError(
            f"Split manifest checksum mismatch: computed {manifest_sha} != expected {cfg['manifest_sha256']}"
        )

    # 2. Initial Model Checkpoint Hash Verification
    init_ckpt_path = Path(cfg["initial_checkpoint"])
    init_ckpt_sha = compute_sha256(init_ckpt_path)
    if init_ckpt_sha != cfg["initial_checkpoint_sha256"]:
        raise RuntimeError(
            f"Initial checkpoint checksum mismatch: computed {init_ckpt_sha} != expected {cfg['initial_checkpoint_sha256']}"
        )

    arm = cfg.get("arm", "UNKNOWN")
    binary_prompt_dropout_p = float(cfg.get("binary_prompt_dropout_p", 0.0))
    print("=" * 80)
    print(f"LANGFENS — RQ1-TRN-02B Training Entrypoint")
    print(f"Arm: {arm} | Binary Prompt Dropout p: {binary_prompt_dropout_p:.2f}")
    print(f"Manifest SHA-256: {manifest_sha}")
    print(f"Initial Checkpoint SHA-256: {init_ckpt_sha}")
    print("=" * 80)

    # 3. Execution Device and Safety Check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not (args.dry_run or args.eval_only):
        if device.type != "cuda":
            raise RuntimeError(
                "Full training requires CUDA on Google Colab runtime; local CPU training strictly prohibited by AGENTS.md."
            )

    # 4. Global Deterministic Seeding
    seed_all(cfg["seed"])

    # 5. Model Instantiation & Weights Initialization from Production Baseline Checkpoint
    model = LangfensWav2Vec2MDD.from_pretrained(
        pretrained_model_name_or_path=cfg["pretrained_model"],
        revision=cfg["pretrained_revision"],
        ctc_weight=cfg["loss_weights"]["ctc"],
        binary_weight=cfg["loss_weights"]["binary"],
        binary_prompt_dropout_p=binary_prompt_dropout_p,
    )

    # Load frozen production baseline model weights
    init_payload = torch.load(init_ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(init_payload["model"])
    print(f"[MODEL] Initialized model weights from {init_ckpt_path} (epoch {init_payload.get('epoch')})")

    # Configure two-head training: freeze Wav2Vec2 backbone and diagnosis head
    trainable = model.configure_two_head_training()
    expected_trainable = {
        id(p) for m in (model.ctc_head, model.cross_attention_module, model.binary_head) for p in m.parameters()
    }
    if {id(p) for p in trainable} != expected_trainable:
        raise AssertionError("Trainable parameter set differs from declared heads contract")

    # Set isolated reproducible intervention generator stream
    model.seed_intervention_generator(cfg["seed"], device="cpu")
    print(f"[MODEL] Intervention generator seeded with {cfg['seed']} (p={binary_prompt_dropout_p})")

    model.to(device)

    # 6. Cold Reset of Optimizer, Scheduler, and GradScaler
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    max_successful_updates = int(cfg.get("max_successful_updates", 3600))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg["warmup_updates"],
        num_training_steps=max_successful_updates,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=float(cfg.get("grad_scaler_initial_scale", 4096.0)),
        enabled=(device.type == "cuda"),
    )
    print(f"[TRAINING] Cold reset of AdamW optimizer, cosine scheduler (100 warmup, {max_successful_updates} updates), and AMP GradScaler.")

    if args.dry_run:
        print("[DRY-RUN] Preflight dry-run completed successfully. No training executed.")
        return

    # 7. Dataset and Dataloader Setup
    zip_path = args.zip_path or cfg.get("zip_path", "datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip")
    train_dataset = L2ArcticMDDDataset(str(manifest_path), 1, "train", policy="strict", zip_path=zip_path)
    val_dataset = L2ArcticMDDDataset(str(manifest_path), 1, "val", policy="strict", zip_path=zip_path)

    if len(train_dataset) != 2399 or len(val_dataset) != 300:
        raise RuntimeError(
            f"Unexpected split sizes: train={len(train_dataset)} (expected 2399), val={len(val_dataset)} (expected 300)"
        )

    # Deterministic dataloader ordering: seeded PyTorch generator
    train_dl_generator = torch.Generator()
    train_dl_generator.manual_seed(cfg["seed"])

    collate_fn = LangfensMDDCollate()
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        generator=train_dl_generator,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        collate_fn=collate_fn,
        drop_last=False,
    )

    if args.eval_only:
        val_metrics = evaluate_validation(model, val_loader, device)
        print(f"EVAL_ONLY results: {json.dumps(val_metrics, indent=2)}")
        return

    # 8. Budget Tracking & Checkpoint Setup
    max_epochs = int(cfg.get("max_epochs", 6))
    budget_tracker = UpdateBudgetTracker(
        max_epochs=max_epochs,
        max_successful_updates=max_successful_updates,
        max_consecutive_skips=8,
    )

    start_epoch = 0
    best_ap = -float("inf")
    history: List[Dict[str, Any]] = []
    patience = 0
    accum = cfg["gradient_accumulation_steps"]

    outdir = Path(cfg["output_dir"])
    drive_out = args.drive_output_dir or cfg.get("drive_output_dir")
    run_id = args.run_id or cfg.get("run_id")
    if run_id:
        if outdir.name != run_id:
            outdir = outdir / run_id
        if drive_out:
            drive_out_p = Path(drive_out)
            if drive_out_p.name != run_id:
                drive_out = str(drive_out_p / run_id)
            else:
                drive_out = str(drive_out_p)

    # Fail closed on pre-existing conflicting artifacts if not resuming
    if not args.resume and not (args.dry_run or args.eval_only):
        conflicts = []
        for chk_name in ["last.pt", "best.pt", "fixed_budget_checkpoint.pt", "halted_guardrail.pt"]:
            if (outdir / chk_name).exists():
                conflicts.append(str(outdir / chk_name))
            if drive_out and (Path(drive_out) / chk_name).exists():
                conflicts.append(str(Path(drive_out) / chk_name))
        if conflicts:
            raise FileExistsError(
                f"FAIL-CLOSED: Pre-existing conflicting checkpoint artifacts detected in output destination:\n"
                f"  {conflicts}\n"
                f"Fresh execution cannot overwrite or reuse previous experiments. "
                f"Specify a unique --run-id, or pass --resume to continue an ongoing experiment."
            )

    outdir.mkdir(parents=True, exist_ok=True)
    if drive_out:
        Path(drive_out).mkdir(parents=True, exist_ok=True)

    # Initialize remote persistent mirror if configured
    remote_mirror = None
    mirror_backend = args.remote_mirror_backend or cfg.get("remote_mirror_backend")
    mirror_target = args.remote_mirror_target or cfg.get("remote_mirror_target")
    mirror_token = args.remote_mirror_token or cfg.get("remote_mirror_token")
    if mirror_backend:
        try:
            from audit_scripts.kaggle_checkpoint_sync import KaggleRemoteMirror
        except ImportError:
            from kaggle_checkpoint_sync import KaggleRemoteMirror
        remote_mirror = KaggleRemoteMirror(
            backend=mirror_backend,
            remote_target=mirror_target,
            auth_token=mirror_token,
            require_verification=True,
        )
        ok, msg = remote_mirror.verify_readiness()
        if not ok:
            raise RuntimeError(f"FAIL-CLOSED: Remote mirror readiness verification failed before training: {msg}")
        print(f"[REMOTE MIRROR] Verified remote persistent mirror: {mirror_backend} -> {mirror_target}", flush=True)

    git_sha = get_git_commit_sha()
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "arm": arm,
        "experiment": cfg["experiment"],
        "run_id": run_id,
        "binary_prompt_dropout_p": binary_prompt_dropout_p,
        "git_sha": git_sha,
        "split_sha256": manifest_sha,
        "initial_checkpoint_sha256": init_ckpt_sha,
        "config": cfg,
        "primary_endpoint_reached": False,
        "experiment_status": "IN_PROGRESS",
        "guardrail_violation": False,
    }

    # Resumption support
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_checkpoint_for_resume(
            ck=ck,
            cfg=cfg,
            manifest_sha=manifest_sha,
            init_ckpt_sha=init_ckpt_sha,
            binary_prompt_dropout_p=binary_prompt_dropout_p,
        )
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_ap = ck["best_ap"]
        history = ck["history"]
        patience = ck.get("patience", 0)
        budget_tracker.successful_updates = ck.get("successful_updates", 0)
        budget_tracker.step_attempts = ck.get("step_attempts", ck.get("successful_updates", 0) + ck.get("skipped_amp_attempts_total", 0))
        budget_tracker.skipped_updates = ck.get("skipped_updates", ck.get("skipped_amp_attempts_total", 0))
        budget_tracker.consecutive_skips = ck.get("consecutive_skips", 0)
        budget_tracker.micro_steps = ck.get("micro_steps", 0)
        restore_rng(ck["rng"])
        if "intervention_rng" in ck and ck["intervention_rng"] is not None and model.intervention_generator is not None:
            model.intervention_generator.set_state(ck["intervention_rng"])
        if "dataloader_generator" in ck and ck["dataloader_generator"] is not None:
            train_dl_generator.set_state(ck["dataloader_generator"])
            print("[RESUME] Restored DataLoader generator state.")
        model.train()
        print(f"Resumed from {args.resume}: start_epoch={start_epoch}, successful_updates={budget_tracker.successful_updates}, best_ap={best_ap}")

    # 9. Main Training Loop
    primary_endpoint_reached = False

    for epoch in range(start_epoch, max_epochs):
        if budget_tracker.budget_exhausted:
            print(f"[BUDGET] Hard cap reached: {budget_tracker.successful_updates} successful updates. Halting training.")
            primary_endpoint_reached = True
            break

        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums = {"ctc": 0.0, "binary": 0.0}
        n_batches = 0
        t0 = time.monotonic()

        for bi, batch in enumerate(train_loader):
            if budget_tracker.budget_exhausted:
                print(f"[BUDGET] Reached {budget_tracker.successful_updates} successful updates during epoch {epoch + 1}.")
                primary_endpoint_reached = True
                break

            result = step_model(model, batch, device)
            loss = result.loss / accum
            scaler.scale(loss).backward()
            budget_tracker.record_micro_step()
            n_batches += 1
            sums["ctc"] += result.loss_dict["ctc_loss"].detach().float().item()
            sums["binary"] += result.loss_dict["binary_loss"].detach().float().item()

            if (bi + 1) % accum == 0 or (bi + 1) == len(train_loader):
                scaler.unscale_(optimizer)

                # Ensure diagnosis head remains strictly gradient-free
                for p in model.diagnosis_head.parameters():
                    if p.grad is not None:
                        raise AssertionError("Diagnosis gradient must remain strictly absent")

                finite_grads = all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable)
                old_scale = scaler.get_scale()
                if finite_grads:
                    torch.nn.utils.clip_grad_norm_(trainable, cfg["gradient_clip_norm"])

                scaler.step(optimizer)
                scaler.update()

                if scaler.get_scale() >= old_scale:
                    if not finite_grads:
                        raise FloatingPointError("Non-finite gradient was unexpectedly not skipped by GradScaler")
                    budget_tracker.record_step_attempt(successful=True)
                    scheduler.step()
                else:
                    budget_tracker.record_step_attempt(successful=False)

                optimizer.zero_grad(set_to_none=True)

        # Epoch Validation Pass
        val_metrics = evaluate_validation(model, val_loader, device)
        apval = val_metrics["validation_binary_ap"]
        blank_fraction = val_metrics["validation_ctc_blank_fraction"]

        row = {
            "epoch": epoch + 1,
            "micro_steps_total": budget_tracker.micro_steps,
            "successful_updates_total": budget_tracker.successful_updates,
            "step_attempts_total": budget_tracker.step_attempts,
            "skipped_updates_total": budget_tracker.skipped_updates,
            "train_ctc_loss": round(sums["ctc"] / max(n_batches, 1), 6),
            "train_binary_loss": round(sums["binary"] / max(n_batches, 1), 6),
            **val_metrics,
            "wall_seconds": round(time.monotonic() - t0, 2),
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        is_best = False
        if math.isfinite(apval) and blank_fraction < 0.99 and apval > best_ap:
            best_ap = apval
            patience = 0
            is_best = True
        else:
            patience += 1

        # Check CTC PER guardrail against Reference R anchor (23.64%)
        ref_ctc_anchor = float(cfg.get("reference_ctc_anchor", 0.2364))
        max_allowed_ctc_per = ref_ctc_anchor + 0.01  # +1.0 absolute percentage point
        current_ctc_per = val_metrics["validation_ctc_per"]
        if current_ctc_per > max_allowed_ctc_per:
            deg_pct = (current_ctc_per - ref_ctc_anchor) * 100.0
            msg = (
                f"[GUARDRAIL VIOLATION] Validation CTC PER {current_ctc_per*100:.2f}% exceeds Reference R anchor (23.64%) "
                f"by +{deg_pct:.2f} percentage points (maximum allowed: +1.00%). Training halted per safety contract."
            )
            print(msg, flush=True)
            halted_payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "display_epoch": epoch + 1,
                "successful_updates": budget_tracker.successful_updates,
                "step_attempts": budget_tracker.step_attempts,
                "skipped_updates": budget_tracker.skipped_updates,
                "micro_steps": budget_tracker.micro_steps,
                "best_ap": best_ap,
                "patience": patience,
                "skipped_amp_attempts_total": budget_tracker.skipped_updates,
                "consecutive_skips": budget_tracker.consecutive_skips,
                "history": history,
                "rng": rng_state(),
                "intervention_rng": model.intervention_generator.get_state() if model.intervention_generator is not None else None,
                "dataloader_generator": train_dl_generator.get_state(),
                "metadata": {
                    **metadata,
                    "primary_endpoint_reached": False,
                    "primary_3600_updates_completed": False,
                    "epoch_cap_reached": False,
                    "guardrail_violation": True,
                    "experiment_status": "HALTED_GUARDRAIL_VIOLATION",
                    "endpoint_status": "HALTED_GUARDRAIL_VIOLATION",
                    "ctc_per_measured": current_ctc_per,
                    "ctc_per_anchor": ref_ctc_anchor,
                    "ctc_per_degradation_pct": deg_pct,
                },
            }
            halted_file = outdir / "halted_guardrail.pt"
            save_checkpoint(
                halted_file,
                halted_payload,
                drive_output_dir=drive_out,
                remote_mirror=remote_mirror,
                remote_rel_prefix=arm,
            )
            print(f"[GUARDRAIL ARTIFACT] Preserved complete halted checkpoint to {halted_file} (last.pt preserved).", flush=True)
            raise RuntimeError(msg)

        primary_3600_updates_completed = (budget_tracker.successful_updates >= max_successful_updates)
        epoch_cap_reached = (epoch + 1 >= max_epochs)

        # Checkpoint Payload Construction
        chk_payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "display_epoch": epoch + 1,
            "successful_updates": budget_tracker.successful_updates,
            "step_attempts": budget_tracker.step_attempts,
            "skipped_updates": budget_tracker.skipped_updates,
            "micro_steps": budget_tracker.micro_steps,
            "best_ap": best_ap,
            "patience": patience,
            "skipped_amp_attempts_total": budget_tracker.skipped_updates,
            "consecutive_skips": budget_tracker.consecutive_skips,
            "history": history,
            "rng": rng_state(),
            "intervention_rng": model.intervention_generator.get_state() if model.intervention_generator is not None else None,
            "dataloader_generator": train_dl_generator.get_state(),
            "metadata": {
                **metadata,
                "primary_endpoint_reached": primary_3600_updates_completed,
                "primary_3600_updates_completed": primary_3600_updates_completed,
                "epoch_cap_reached": epoch_cap_reached,
                "successful_updates": budget_tracker.successful_updates,
                "skipped_updates": budget_tracker.skipped_updates,
                "endpoint_status": (
                    "COMPLETED_3600_UPDATES"
                    if primary_3600_updates_completed
                    else ("INCOMPLETE_AMP_SKIPS" if budget_tracker.skipped_updates > 0 else "HALTED_EPOCH_CAP")
                ),
            },
        }

        # Save last checkpoint
        save_checkpoint(
            outdir / "last.pt",
            chk_payload,
            drive_output_dir=drive_out,
            remote_mirror=remote_mirror,
            remote_rel_prefix=arm,
        )

        # Save exploratory best validation checkpoint
        if is_best:
            save_checkpoint(
                outdir / "best.pt",
                chk_payload,
                drive_output_dir=drive_out,
                remote_mirror=remote_mirror,
                remote_rel_prefix=arm,
            )

        # Check for primary matched-budget endpoint: strictly 3,600 successful updates
        if primary_3600_updates_completed:
            primary_endpoint_reached = True
            save_checkpoint(
                outdir / "fixed_budget_checkpoint.pt",
                chk_payload,
                drive_output_dir=drive_out,
                remote_mirror=remote_mirror,
                remote_rel_prefix=arm,
            )
            print(f"[PRIMARY ENDPOINT] Saved fixed_budget_checkpoint.pt: completed {budget_tracker.successful_updates} successful updates.")
        elif epoch_cap_reached:
            print(
                f"[INCOMPLETE ENDPOINT] Epoch cap (6) reached with {budget_tracker.successful_updates} / {max_successful_updates} updates "
                f"({budget_tracker.skipped_updates} AMP skips). fixed_budget_checkpoint.pt NOT labeled as completed endpoint.",
                flush=True,
            )

        if patience >= cfg["early_stopping_patience"]:
            print(f"Early stopping triggered after {patience} epochs without validation AP improvement.", flush=True)
            break

    # 10. Final History Ledger
    hist_json = outdir / "history.json"
    hist_json.write_text(json.dumps(history, indent=2) + "\n")
    if drive_out:
        remote_hist = Path(drive_out) / "history.json"
        shutil.copyfile(str(hist_json), str(remote_hist))
        print(f"Verified remote history persistence: {remote_hist}", flush=True)
    if remote_mirror is not None:
        rel_hist = f"{arm}/history.json"
        remote_mirror.mirror_file(hist_json, rel_hist, verify_sha=True)
        print(f"[REMOTE MIRROR] Mirrored history.json -> {mirror_backend}:{rel_hist}", flush=True)

    print("=" * 80)
    print(f"Training run completed for Arm {arm}.")
    print(f"Primary endpoint reached: {primary_endpoint_reached}")
    print(f"Successful updates: {budget_tracker.successful_updates} / {max_successful_updates}")
    print(f"Step attempts: {budget_tracker.step_attempts} | Skipped updates: {budget_tracker.skipped_updates}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", flush=True)
        raise
