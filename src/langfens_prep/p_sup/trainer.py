"""End-to-end training pipeline for P-SUP Subtype Treatment and Matched Binary Control.

Supports:
a) Subtype treatment: CompactOperationHead with objective="operation_ce"
   (3-way CE: correct=0, substitution=1, deletion=2) + realized CTC loss.
b) Matched binary control: CompactOperationHead with objective="binary_collapsed"
   (collapsed binary logit: logsumexp(sub,del) - correct) + realized CTC loss.

Contracts:
- seed: 1337
- batch_size: 2, gradient_accumulation_steps: 2 (effective batch: 4)
- warmup_updates: 100, cosine schedule over 15,000 horizon
- max_successful_updates: 3600
- AdamW lr: 0.001, weight_decay: 0.01, gradient_clip: 1.0
- CUDA FP16 AMP (with float32 CPU fallback for testing)
- Dual checkpointing: fixed_budget_3600.pt and validation-selected best.pt
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model

from langfens_prep.p_sup.model import (
    CompactOperationHead,
    HeadConfig,
    FrozenAudioAdapter,
    matched_initial_pair,
    parameter_inventory
)
from langfens_prep.p_sup.objectives import task_loss, realized_ctc_loss, collapsed_binary_logit
from langfens_prep.p_sup.planning import UpdateLedger, learning_rate_multiplier, accumulation_groups
from langfens_prep.p_sup.dataset import (
    PSupL2ArcticDataset,
    collate_psup_batch,
    PARTITION_SPEAKERS,
    FIT_SPEAKERS,
    SELECT_SPEAKERS,
    CALIBRATION_SPEAKERS,
    VAL_SPEAKERS,
    SealedSpeakerContaminationError
)
from langfens_prep.p_sup.runtime import CONTRACT_SHA256, SEALED_SPEAKERS

DEFAULT_PRETRAINED_CHECKPOINT = "facebook/wav2vec2-base-960h"
DEFAULT_PRETRAINED_REVISION = "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"


def seed_all(seed: int = 1337) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }


def restore_rng(s: Dict[str, Any]) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def compute_file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class FrozenWav2Vec2AudioEncoder(nn.Module):
    """Frozen Wav2Vec2 backbone encoder with padding-safe feature extraction."""
    def __init__(self, wav2vec2: Wav2Vec2Model):
        super().__init__()
        self.wav2vec2 = wav2vec2
        self.wav2vec2.eval()
        for p in self.wav2vec2.parameters():
            p.requires_grad_(False)

    def _get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        out = self.wav2vec2._get_feat_extract_output_lengths(input_lengths)
        if isinstance(out, int):
            out = torch.tensor([out], device=input_lengths.device, dtype=torch.long)
        return out.clamp(min=1)

    def forward(self, audio: torch.Tensor, audio_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = audio.shape[0]
        device = audio.device
        audio_lengths = audio_lengths.to(device=device, dtype=torch.long)

        # Fast vectorized path when all audio lengths in batch are identical
        if (audio_lengths == audio.shape[1]).all():
            fe = self.wav2vec2.feature_extractor(audio).transpose(1, 2)
            frame_output_lengths = self._get_feat_extract_output_lengths(audio_lengths)
            T_max = fe.shape[1]
            padded_features = fe
        else:
            # Per-utterance slice to prevent GroupNorm cross-contamination from padding
            feats = [
                self.wav2vec2.feature_extractor(audio[i:i+1, :audio_lengths[i]]).transpose(1, 2)
                for i in range(batch_size)
            ]
            frame_output_lengths = torch.tensor([f.shape[1] for f in feats], device=device, dtype=torch.long)
            T_max = frame_output_lengths.max().item()
            padded_features = torch.zeros(batch_size, T_max, feats[0].shape[-1], device=device, dtype=feats[0].dtype)
            for i, f in enumerate(feats):
                padded_features[i, :f.shape[1]] = f[0]

        frame_indices = torch.arange(T_max, device=device).unsqueeze(0)
        feat_attention_mask = frame_indices < frame_output_lengths.unsqueeze(1)

        hidden_states, _ = self.wav2vec2.feature_projection(padded_features)
        hidden_states = hidden_states.masked_fill(~feat_attention_mask.unsqueeze(-1), 0.0)

        encoder_outputs = self.wav2vec2.encoder(hidden_states, attention_mask=feat_attention_mask)
        hidden_states = encoder_outputs.last_hidden_state
        if getattr(self.wav2vec2, "adapter", None) is not None:
            hidden_states = self.wav2vec2.adapter(hidden_states)

        return hidden_states, frame_output_lengths


def average_precision_binary(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Average precision with tied scores grouped, matching sklearn convention."""
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


def greedy_ctc_decode(frame_preds: Sequence[int], blank_id: int = 0) -> List[int]:
    hyp = []
    prev = None
    for t in frame_preds:
        if t != prev:
            if t != blank_id:
                hyp.append(int(t))
            prev = t
    return hyp


def compute_levenshtein_edits(ref: Sequence[int], hyp: Sequence[int]) -> Tuple[int, int, int, int]:
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
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1; i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            inss += 1; j -= 1
        else:
            break
    return dp[m][n], subs, dels, inss


def is_ephemeral_path(path: Path | str) -> bool:
    """Checks if path resides on ephemeral container filesystems (e.g. /kaggle/working, /kaggle, /content)."""
    p_str = str(Path(path).resolve())
    return (
        p_str.startswith("/kaggle")
        or "/kaggle/working" in p_str
        or p_str.startswith("/content")
        or "/content/" in p_str
    )


def verify_cuda_t4_environment(
    remote_execution: bool = False,
    device: Optional[torch.device | str] = None,
    require_t4: bool = True,
) -> Dict[str, Any]:
    """Fail-closed CUDA / Tesla T4 / FP16 AMP environment checks.

    In remote execution mode (or when remote execution is requested):
    - Asserts torch.cuda.is_available() is True
    - Asserts device type is CUDA
    - Verifies GPU device name contains 'Tesla T4' or 'T4'
    - Checks FP16 AMP support via compute capability and operational probe
    """
    if not remote_execution:
        cuda_avail = torch.cuda.is_available()
        return {
            "remote_execution": False,
            "cuda_available": cuda_avail,
            "device": str(device) if device is not None else ("cuda" if cuda_avail else "cpu"),
            "status": "LOCAL_OK",
        }

    if not torch.cuda.is_available():
        raise RuntimeError(
            "FAIL-CLOSED: CUDA is not available. Remote execution requires an authorized CUDA GPU (Nvidia Tesla T4). "
            "Local CPU fallback in remote mode is strictly prohibited."
        )

    dev = torch.device(device if device is not None else "cuda")
    if dev.type != "cuda":
        raise RuntimeError(
            f"FAIL-CLOSED: Expected CUDA device in remote execution mode, got device type '{dev.type}'."
        )

    device_idx = dev.index if dev.index is not None else 0
    if device_idx >= torch.cuda.device_count():
        raise RuntimeError(
            f"FAIL-CLOSED: CUDA device index {device_idx} exceeds available device count {torch.cuda.device_count()}."
        )

    gpu_name = torch.cuda.get_device_name(device_idx)
    if require_t4:
        if "T4" not in gpu_name and "Tesla T4" not in gpu_name:
            raise RuntimeError(
                f"FAIL-CLOSED: Remote GPU accelerator '{gpu_name}' does not match required Nvidia Tesla T4."
            )

    # Check FP16 AMP support
    try:
        major, minor = torch.cuda.get_device_capability(device_idx)
        if major < 5 or (major == 5 and minor < 3):
            raise RuntimeError(
                f"FAIL-CLOSED: Compute capability ({major}, {minor}) does not support FP16 AMP."
            )

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            probe_tensor = torch.ones((2, 2), device=f"cuda:{device_idx}", dtype=torch.float16)
            probe_out = probe_tensor @ probe_tensor
            if probe_out.dtype != torch.float16:
                raise RuntimeError("Autocast produced non-float16 output in operational probe.")
        _ = torch.amp.GradScaler(device="cuda", enabled=True)
    except Exception as e:
        raise RuntimeError(
            f"FAIL-CLOSED: FP16 AMP is unsupported or failed probe on device cuda:{device_idx}: {e}"
        ) from e

    return {
        "remote_execution": True,
        "cuda_available": True,
        "device": f"cuda:{device_idx}",
        "gpu_name": gpu_name,
        "compute_capability": (major, minor),
        "fp16_amp_supported": True,
        "status": "VERIFIED_TESLA_T4",
    }


class RemoteCheckpointMirror:
    """Manages atomic remote checkpoint mirroring with cryptographic SHA-256 readback.

    Supports:
    - 'huggingface': Hugging Face Hub model repository via HfApi and hf_hub_download.
    - 'local_dir' / 'directory': Persistent non-ephemeral directory with cryptographic readback.
    """
    SUPPORTED_BACKENDS = ("huggingface", "local_dir", "directory")

    def __init__(
        self,
        backend: str = "local_dir",
        remote_target: Optional[str | Path] = None,
        auth_token: Optional[str] = None,
        require_verification: bool = True,
        allow_local_test: bool = False,
    ):
        backend_lower = backend.lower()
        if backend_lower not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported remote mirror backend '{backend}'. Must be one of: {self.SUPPORTED_BACKENDS}"
            )
        self.backend = backend_lower
        self.remote_target = (
            str(remote_target)
            if remote_target
            else os.environ.get("REMOTE_MIRROR_TARGET") or os.environ.get("KAGGLE_REMOTE_MIRROR_TARGET")
        )
        self.auth_token = (
            auth_token
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
            or os.environ.get("KAGGLE_REMOTE_MIRROR_TOKEN")
        )
        self.require_verification = require_verification
        self.allow_local_test = allow_local_test
        self._verified = False

    def verify_readiness(self) -> Tuple[bool, str]:
        """Probes whether the remote persistent target is configured, authorized, and writable."""
        if not self.remote_target:
            return (
                False,
                "FAIL-CLOSED: No remote persistent mirror target configured. "
                "Ephemeral /kaggle/working cannot be used as sole storage.",
            )

        if self.backend in ("local_dir", "directory"):
            p = Path(self.remote_target).resolve()
            if not self.allow_local_test and is_ephemeral_path(p):
                return (
                    False,
                    f"FAIL-CLOSED: Ephemeral /kaggle path '{p}' cannot be used as durable remote mirror target.",
                )
            try:
                p.mkdir(parents=True, exist_ok=True)
                probe = p / f".probe_readiness_{int(time.time() * 1000)}.tmp"
                probe.write_text("probe")
                if not probe.is_file() or probe.read_text() != "probe":
                    return False, f"FAIL-CLOSED: Local mirror directory '{p}' readback probe failed."
                probe.unlink()
                self._verified = True
                return True, f"Local mirror directory '{p}' verified."
            except Exception as e:
                return False, f"FAIL-CLOSED: Failed to access local mirror directory '{p}': {e}"

        elif self.backend == "huggingface":
            token = self.auth_token
            if not token:
                return (
                    False,
                    "FAIL-CLOSED: Hugging Face remote mirror requires valid HF_TOKEN.",
                )
            try:
                from huggingface_hub import HfApi, hf_hub_download

                api = HfApi(token=token)
                repo_id = self.remote_target
                probe_name = f".probe_mirror_{int(time.time() * 1000)}.tmp"
                probe_content = f"LANGFENS_PROBE_{time.time()}".encode("utf-8")
                probe_sha = hashlib.sha256(probe_content).hexdigest()

                with tempfile.NamedTemporaryFile(delete=False) as tf:
                    tf.write(probe_content)
                    tmp_local = tf.name

                try:
                    api.upload_file(
                        path_or_fileobj=tmp_local,
                        path_in_repo=probe_name,
                        repo_id=repo_id,
                        repo_type="model",
                    )
                    downloaded = hf_hub_download(
                        repo_id=repo_id,
                        filename=probe_name,
                        repo_type="model",
                        token=token,
                        force_download=True,
                    )
                    readback_sha = compute_file_sha256(downloaded)
                    if readback_sha != probe_sha:
                        return (
                            False,
                            f"FAIL-CLOSED: Hugging Face probe readback SHA mismatch: sent={probe_sha} != read={readback_sha}",
                        )
                finally:
                    if Path(tmp_local).exists():
                        Path(tmp_local).unlink()
                    try:
                        api.delete_file(path_in_repo=probe_name, repo_id=repo_id, repo_type="model")
                    except Exception:
                        pass

                self._verified = True
                return True, f"Hugging Face repository '{repo_id}' verified with authenticated write-readback probe."
            except Exception as e:
                return (
                    False,
                    f"FAIL-CLOSED: Hugging Face remote mirror probe failed for repo '{self.remote_target}': {e}",
                )

        return False, f"Unknown backend: {self.backend}"

    def mirror_file(
        self,
        local_path: Path | str,
        remote_rel_path: str,
        verify_sha: bool = True,
    ) -> Dict[str, Any]:
        """Atomically mirrors a local file to persistent remote storage with SHA-256 verification."""
        local_p = Path(local_path).resolve()
        if not local_p.is_file():
            raise FileNotFoundError(f"Local file to mirror does not exist: {local_p}")

        if not self._verified and self.require_verification:
            ok, msg = self.verify_readiness()
            if not ok:
                raise RuntimeError(msg)

        local_sha = compute_file_sha256(local_p)
        file_size = local_p.stat().st_size
        t0 = time.time()

        if self.backend in ("local_dir", "directory"):
            remote_dest = Path(self.remote_target) / remote_rel_path
            remote_dest.parent.mkdir(parents=True, exist_ok=True)
            tmp_remote = remote_dest.with_suffix(remote_dest.suffix + ".mirror_tmp")
            shutil.copyfile(str(local_p), str(tmp_remote))
            os.replace(tmp_remote, remote_dest)

            if not remote_dest.is_file():
                raise IOError(f"FAIL-CLOSED: Remote mirrored file does not exist: {remote_dest}")

            if verify_sha:
                remote_sha = compute_file_sha256(remote_dest)
                if remote_sha != local_sha:
                    raise IOError(
                        f"FAIL-CLOSED: Remote mirror SHA-256 mismatch for '{remote_rel_path}': "
                        f"local={local_sha} != remote={remote_sha}"
                    )
            else:
                remote_sha = local_sha

            return {
                "status": "SYNCED",
                "backend": self.backend,
                "local_path": str(local_p),
                "remote_path": str(remote_dest),
                "local_sha256": local_sha,
                "remote_sha256": remote_sha,
                "bytes": file_size,
                "duration_seconds": time.time() - t0,
            }

        elif self.backend == "huggingface":
            token = self.auth_token
            if not token:
                raise RuntimeError("FAIL-CLOSED: Hugging Face token missing during mirror_file.")

            from huggingface_hub import HfApi, hf_hub_download

            api = HfApi(token=token)
            repo_id = self.remote_target

            api.upload_file(
                path_or_fileobj=str(local_p),
                path_in_repo=remote_rel_path,
                repo_id=repo_id,
                repo_type="model",
            )

            if verify_sha:
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=remote_rel_path,
                    repo_type="model",
                    token=token,
                    force_download=True,
                )
                remote_sha = compute_file_sha256(downloaded)
                if remote_sha != local_sha:
                    raise IOError(
                        f"FAIL-CLOSED: Hugging Face remote mirror SHA-256 readback mismatch for '{remote_rel_path}': "
                        f"local={local_sha} != remote={remote_sha}"
                    )
            else:
                remote_sha = local_sha

            return {
                "status": "SYNCED",
                "backend": self.backend,
                "local_path": str(local_p),
                "remote_path": f"{repo_id}/{remote_rel_path}",
                "local_sha256": local_sha,
                "remote_sha256": remote_sha,
                "bytes": file_size,
                "duration_seconds": time.time() - t0,
            }

        raise NotImplementedError(f"FAIL-CLOSED: Backend '{self.backend}' not supported.")

    def download_file(
        self,
        remote_rel_path: str,
        local_dest_path: Path | str,
        verify_sha: bool = True,
    ) -> Path:
        """Downloads a checkpoint from remote persistent storage and optionally verifies SHA-256."""
        dest_p = Path(local_dest_path).resolve()
        dest_p.parent.mkdir(parents=True, exist_ok=True)

        if self.backend in ("local_dir", "directory"):
            src_p = Path(self.remote_target) / remote_rel_path
            if not src_p.is_file():
                raise FileNotFoundError(f"File not found in local persistent mirror: {src_p}")
            shutil.copyfile(str(src_p), str(dest_p))
            if verify_sha:
                src_sha = compute_file_sha256(src_p)
                dest_sha = compute_file_sha256(dest_p)
                if src_sha != dest_sha:
                    raise IOError(
                        f"FAIL-CLOSED: Download SHA-256 mismatch for '{remote_rel_path}': "
                        f"remote={src_sha} != local={dest_sha}"
                    )
            return dest_p

        elif self.backend == "huggingface":
            token = self.auth_token
            if not token:
                raise RuntimeError("FAIL-CLOSED: Hugging Face token missing for download.")
            from huggingface_hub import hf_hub_download

            repo_id = str(self.remote_target)
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=remote_rel_path,
                token=token,
                local_dir=str(dest_p.parent),
                repo_type="model",
                force_download=True,
            )
            downloaded_p = Path(downloaded)
            if downloaded_p.resolve() != dest_p:
                shutil.copyfile(str(downloaded_p), str(dest_p))
            return dest_p

        raise ValueError(f"Unsupported backend for download: {self.backend}")

    def file_exists(self, remote_rel_path: str) -> bool:
        """Checks if a file exists in the remote persistent store."""
        if self.backend in ("local_dir", "directory"):
            return (Path(self.remote_target) / remote_rel_path).is_file()
        elif self.backend == "huggingface":
            token = self.auth_token
            if not token:
                return False
            try:
                from huggingface_hub import HfApi

                api = HfApi(token=token)
                files = api.list_repo_files(repo_id=str(self.remote_target), repo_type="model")
                return remote_rel_path in files
            except Exception:
                return False
        return False


REQUIRED_CHECKPOINT_KEYS = frozenset({
    "head",
    "optimizer",
    "scheduler",
    "scaler",
    "epoch",
    "micro_steps",
    "successful_updates",
    "skipped_amp_attempts_total",
    "best_ap",
    "history",
    "rng",
    "metadata",
})


def validate_checkpoint_payload(payload: Any) -> None:
    """Validates full checkpoint payload structure and invariants."""
    if not isinstance(payload, dict):
        raise ValueError("Invalid checkpoint: payload must be a dictionary.")

    for k in REQUIRED_CHECKPOINT_KEYS:
        if k not in payload:
            if k == "head" and "model" in payload:
                continue
            raise ValueError(f"Invalid checkpoint: missing required key '{k}'.")

    rng = payload["rng"]
    if not isinstance(rng, dict):
        raise ValueError("Invalid checkpoint: 'rng' must be a dictionary.")
    for rng_key in ("python", "numpy", "torch"):
        if rng_key not in rng:
            raise ValueError(f"Invalid checkpoint: 'rng' missing '{rng_key}' state.")

    succ = payload["successful_updates"]
    if not isinstance(succ, int) or succ < 0 or succ > 3600:
        raise ValueError(f"Invalid checkpoint: successful_updates must be in [0, 3600], got {succ}.")

    skipped = payload["skipped_amp_attempts_total"]
    if not isinstance(skipped, int) or skipped < 0:
        raise ValueError(f"Invalid checkpoint: skipped_amp_attempts_total must be non-negative integer, got {skipped}.")

    epoch = payload["epoch"]
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"Invalid checkpoint: epoch must be non-negative integer, got {epoch}.")


def save_checkpoint(
    path: Path,
    head: CompactOperationHead,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    successful_updates: int,
    micro_steps: int,
    best_ap: float,
    patience: int,
    skipped_amp: int,
    history: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    is_fixed_budget_endpoint: bool = False,
    remote_mirror: Optional[Any] = None,
    remote_rel_prefix: Optional[str] = None,
    verify_remote_sha: bool = True,
    require_remote_mirror: bool = False,
) -> str:
    path = Path(path).resolve()
    if (require_remote_mirror or is_ephemeral_path(path)) and remote_mirror is None:
        raise RuntimeError(
            f"FAIL-CLOSED: Checkpoint path '{path}' is on ephemeral storage (/kaggle/working) "
            "without a verified remote persistent mirror. Ephemeral storage must never be treated as durable."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "head": head.state_dict(),
        "model": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "successful_updates": successful_updates,
        "micro_steps": micro_steps,
        "best_ap": best_ap,
        "patience": patience,
        "skipped_amp_attempts_total": skipped_amp,
        "history": history,
        "rng": rng_state(),
        "metadata": metadata,
        "is_fixed_budget_endpoint": is_fixed_budget_endpoint,
    }
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    sha256 = compute_file_sha256(path)

    # Remote persistent mirror synchronization with cryptographic readback
    if remote_mirror is not None:
        rel_dest = f"{remote_rel_prefix}/{path.name}" if remote_rel_prefix else path.name
        sync_result = remote_mirror.mirror_file(path, rel_dest, verify_sha=verify_remote_sha)
        if not isinstance(sync_result, dict) or sync_result.get("status") != "SYNCED":
            raise IOError(f"FAIL-CLOSED: Remote mirror synchronization unverified for '{path.name}'. Result: {sync_result}")
        if verify_remote_sha and sync_result.get("remote_sha256") != sha256:
            raise IOError(
                f"FAIL-CLOSED: Remote mirror SHA-256 mismatch for '{path.name}': "
                f"local={sha256} != remote={sync_result.get('remote_sha256')}"
            )
        print(
            f"[REMOTE MIRROR] Mirrored {path.name} -> {sync_result.get('backend', 'remote')}:{rel_dest} "
            f"(SHA: {sync_result.get('remote_sha256', sha256)[:12]}...)",
            flush=True,
        )

    return sha256


def resume_from_checkpoint(
    path: Path | str,
    head: Optional[CompactOperationHead] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    device: Optional[torch.device | str] = None,
    remote_mirror: Optional[Any] = None,
    remote_rel_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Restores full training state from checkpoint: head, optimizer, scheduler, scaler, RNG, counters.

    If checkpoint does not exist locally and remote_mirror is provided, recovers it from remote_mirror.
    """
    ckpt_p = Path(path).resolve()
    if not ckpt_p.is_file():
        if remote_mirror is not None:
            rel = remote_rel_path or ckpt_p.name
            print(f"[RESUME] Checkpoint '{ckpt_p.name}' missing locally; recovering from remote mirror '{rel}'...")
            remote_mirror.download_file(rel, ckpt_p, verify_sha=True)
        else:
            raise FileNotFoundError(f"FAIL-CLOSED: Checkpoint not found at '{ckpt_p}'.")

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(ckpt_p, map_location=target_device, weights_only=False)
    validate_checkpoint_payload(payload)

    # 1. Restore Head Weights
    head_state = payload.get("head") or payload.get("model")
    if head is None:
        head = CompactOperationHead(HeadConfig()).to(target_device)
    head.load_state_dict(head_state)

    # 2. Setup Scheduler first (so its constructor does not overwrite restored optimizer lr)
    if optimizer is None:
        trainable = [p for p in head.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=0.001, weight_decay=0.01)

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate_multiplier)
    scheduler.load_state_dict(payload["scheduler"])

    # 3. Restore Optimizer State (loaded after scheduler to preserve exact param_groups)
    optimizer.load_state_dict(payload["optimizer"])

    # 4. Restore Scaler State
    if scaler is not None and payload.get("scaler") is not None and scaler.is_enabled():
        scaler.load_state_dict(payload["scaler"])

    # 5. Restore RNG State (restored at the end so model parameter initialization doesn't consume RNG stream)
    restore_rng(payload["rng"])

    return {
        "head": head,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
        "epoch": payload["epoch"],
        "micro_steps": payload["micro_steps"],
        "successful_updates": payload["successful_updates"],
        "skipped_amp_attempts_total": payload["skipped_amp_attempts_total"],
        "best_ap": payload["best_ap"],
        "patience": payload.get("patience", 0),
        "history": payload["history"],
        "metadata": payload["metadata"],
        "is_fixed_budget_endpoint": payload.get("is_fixed_budget_endpoint", False),
        "payload": payload,
    }


def evaluate_val_set(
    adapter: FrozenAudioAdapter,
    val_loader: DataLoader,
    device: torch.device,
    objective: str
) -> Dict[str, Any]:
    """Evaluate validation split and compute binary AP, blank fraction, PER."""
    adapter.eval()
    all_binary_labels = []
    all_scores = []
    total_ctc_frames = 0
    total_blank_frames = 0
    total_ref_phones = 0
    total_edits = 0

    with torch.no_grad():
        for batch in val_loader:
            audio = batch["waveforms"].to(device)
            audio_lengths = batch["audio_lengths"].to(device)
            prompt_ids = batch["prompt_phone_ids"].to(device)
            prompt_lengths = batch["prompt_lengths"].to(device)

            out = adapter(
                audio=audio,
                audio_lengths=audio_lengths,
                prompt_phone_ids=prompt_ids,
                prompt_lengths=prompt_lengths
            )

            op_logits = out["operation_logits"]  # (B, N, 3)
            ctc_logits = out["ctc_logits"]        # (B, T, 40)
            mask = batch["operation_mask"].to(device)  # (B, N)
            targets_binary = batch["binary"].to(device)

            # Compute error score
            if objective == "operation_ce":
                probs = op_logits.softmax(dim=-1)
                s_error = probs[..., 1] + probs[..., 2]
            else:
                s_error = torch.sigmoid(collapsed_binary_logit(op_logits))

            valid_scores = s_error[mask].cpu().numpy().tolist()
            valid_targets = targets_binary[mask].cpu().numpy().tolist()

            all_scores.extend(valid_scores)
            all_binary_labels.extend(valid_targets)

            # CTC Blank and PER accounting
            frame_lengths = out["frame_lengths"]
            preds = ctc_logits.argmax(dim=-1)
            for i, fl in enumerate(frame_lengths.tolist()):
                pred_seq = preds[i, :fl].cpu().tolist()
                total_ctc_frames += fl
                total_blank_frames += sum(1 for p in pred_seq if p == 0)

                hyp = greedy_ctc_decode(pred_seq, blank_id=0)
                ref_len = batch["ctc_lengths"][i].item()
                ref = batch["ctc"][i, :ref_len].tolist()
                edits, _, _, _ = compute_levenshtein_edits(ref, hyp)
                total_edits += edits
                total_ref_phones += len(ref)

    ap = average_precision_binary(all_binary_labels, all_scores)
    blank_fraction = total_blank_frames / total_ctc_frames if total_ctc_frames > 0 else 1.0
    per = total_edits / total_ref_phones if total_ref_phones > 0 else 1.0

    return {
        "val_ap": ap,
        "blank_fraction": blank_fraction,
        "per": per,
        "total_eval_tokens": len(all_binary_labels),
        "total_eval_frames": total_ctc_frames
    }


def train_p_sup(
    objective: str,
    split_manifest_path: str | Path,
    zip_path: str | Path,
    output_dir: str | Path,
    seed: int = 1337,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 2,
    warmup_updates: int = 100,
    cosine_horizon_updates: int = 15000,
    max_successful_updates: int = 3600,
    lr: float = 0.001,
    weight_decay: float = 0.01,
    gradient_clip: float = 1.0,
    ctc_weight: float = 1.0,
    task_weight: float = 1.0,
    pretrained_backbone: str = DEFAULT_PRETRAINED_CHECKPOINT,
    pretrained_revision: str = DEFAULT_PRETRAINED_REVISION,
    device: Optional[torch.device] = None,
    max_epochs: int = 25,
    early_stopping_patience: int = 5,
    train_partition: str = "fit",
    select_partition: str = "selection",
    cal_partition: str = "calibration",
    val_partition: str = "val",
    remote_mirror: Optional[Any] = None,
    remote_rel_prefix: Optional[str] = None,
    remote_execution: bool = False,
    resume_checkpoint_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Execute official training for one P-SUP arm."""
    if objective not in ("operation_ce", "binary_collapsed"):
        raise ValueError(f"Unregistered P-SUP objective: {objective!r}")

    seed_all(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_cuda_amp = (device.type == "cuda")

    # 0. Fail-closed hardware & persistence verification
    _ = verify_cuda_t4_environment(
        remote_execution=remote_execution,
        device=device,
        require_t4=True,
    )
    if remote_execution or is_ephemeral_path(output_dir):
        if remote_mirror is None:
            raise RuntimeError(
                f"FAIL-CLOSED: Remote execution or ephemeral output directory '{output_dir}' requires "
                "a verified remote persistent mirror. Ephemeral /kaggle/working cannot be treated as durable."
            )
        ok, msg = remote_mirror.verify_readiness()
        if not ok:
            raise RuntimeError(f"FAIL-CLOSED: Remote persistent mirror verification failed: {msg}")

    # 1. Datasets & Loaders
    # Training dataset: default FIT only (1,799 utterances across 12 speakers:
    # BWC, ERMS, HKK, HQTV, SKA, SVBI, TNI, TXHC, YBAA, YDCK, YKWK, ZHAA)
    train_dataset = PSupL2ArcticDataset(
        split_manifest_path=split_manifest_path,
        zip_path=zip_path,
        partition=train_partition
    )
    if train_partition == "fit":
        if len(train_dataset) != 1799:
            raise ValueError(
                f"FAIL-CLOSED: FIT training dataset must have exactly 1,799 utterances, got {len(train_dataset)}"
            )
        train_speakers = frozenset(spk for spk, _ in train_dataset.entries)
        if train_speakers != FIT_SPEAKERS:
            raise ValueError(
                f"FAIL-CLOSED: FIT training dataset speakers mismatch: {train_speakers} != {FIT_SPEAKERS}"
            )
        if train_speakers & SEALED_SPEAKERS:
            raise SealedSpeakerContaminationError(
                f"FAIL-CLOSED: Sealed test speaker found in FIT training dataset: {train_speakers & SEALED_SPEAKERS}"
            )
    else:
        train_speakers = frozenset(spk for spk, _ in train_dataset.entries)

    # Selection dataset: default SELECT (300 utterances across 2 speakers: MBMPS, TLV)
    # Used for validation loss, AP tracking, early stopping, and saving best.pt during training
    selection_dataset = PSupL2ArcticDataset(
        split_manifest_path=split_manifest_path,
        zip_path=zip_path,
        partition=select_partition
    )
    if select_partition in ("selection", "select"):
        if len(selection_dataset) != 300:
            raise ValueError(
                f"FAIL-CLOSED: SELECT dataset must have exactly 300 utterances, got {len(selection_dataset)}"
            )
        selection_speakers = frozenset(spk for spk, _ in selection_dataset.entries)
        if selection_speakers != SELECT_SPEAKERS:
            raise ValueError(
                f"FAIL-CLOSED: SELECT dataset speakers mismatch: {selection_speakers} != {SELECT_SPEAKERS}"
            )
        if selection_speakers & SEALED_SPEAKERS:
            raise SealedSpeakerContaminationError(
                f"FAIL-CLOSED: Sealed test speaker found in SELECT dataset: {selection_speakers & SEALED_SPEAKERS}"
            )
    else:
        selection_speakers = frozenset(spk for spk, _ in selection_dataset.entries)

    # Fold-1 VAL (300 utterances across NJS, RRBI) MUST be preserved strictly unseen during training & selection
    if train_speakers & VAL_SPEAKERS:
        raise ValueError(
            f"FAIL-CLOSED: Fold-1 VAL speakers leaked into training set: {train_speakers & VAL_SPEAKERS}"
        )
    if selection_speakers & VAL_SPEAKERS:
        raise ValueError(
            f"FAIL-CLOSED: Fold-1 VAL speakers leaked into selection set: {selection_speakers & VAL_SPEAKERS}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_psup_batch,
        num_workers=0,
        drop_last=False
    )
    selection_loader = DataLoader(
        selection_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_psup_batch,
        num_workers=0
    )
    val_loader = selection_loader  # Alias for validation during training

    # 2. Model & Backbone Initialization
    wav2vec2 = Wav2Vec2Model.from_pretrained(
        pretrained_backbone,
        revision=pretrained_revision
    ).to(device)
    encoder = FrozenWav2Vec2AudioEncoder(wav2vec2)

    head = CompactOperationHead(HeadConfig()).to(device)
    adapter = FrozenAudioAdapter(encoder, head).to(device)

    # 3. Optimizer & Scheduler
    trainable_params = [p for p in head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    def lr_lambda(step: int) -> float:
        return learning_rate_multiplier(step)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler(device="cuda", init_scale=4096, enabled=use_cuda_amp)
    ledger = UpdateLedger()

    # 4. Checkpoint Resumption (if requested)
    best_ap = -1.0
    patience = 0
    history = []
    micro_steps = 0
    fixed_budget_saved = False
    start_epoch = 0

    if resume_checkpoint_path is not None:
        rel_resume = f"{remote_rel_prefix}/{Path(resume_checkpoint_path).name}" if remote_rel_prefix else Path(resume_checkpoint_path).name
        resumed = resume_from_checkpoint(
            path=resume_checkpoint_path,
            head=head,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            remote_mirror=remote_mirror,
            remote_rel_path=rel_resume,
        )
        start_epoch = resumed["epoch"] + 1
        micro_steps = resumed["micro_steps"]
        best_ap = resumed["best_ap"]
        patience = resumed["patience"]
        history = list(resumed["history"])
        ledger.successful = resumed["successful_updates"]
        ledger.skipped = resumed["skipped_amp_attempts_total"]
        ledger.attempts = ledger.successful + ledger.skipped
        fixed_budget_saved = resumed.get("is_fixed_budget_endpoint", False) or (ledger.successful >= max_successful_updates)
        print(f"[{objective.upper()}] Resumed from checkpoint at epoch {resumed['epoch']} (updates: {ledger.successful}/{max_successful_updates})")

    start_time = time.time()

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True
        ).strip()
    except Exception:
        git_sha = None

    metadata = {
        "registration": "P-SUP-COMPACT-OP-V1",
        "objective": objective,
        "git_sha": git_sha,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "warmup_updates": warmup_updates,
        "cosine_horizon_updates": cosine_horizon_updates,
        "max_successful_updates": max_successful_updates,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "ctc_weight": ctc_weight,
        "task_weight": task_weight,
        "backbone": pretrained_backbone,
        "backbone_revision": pretrained_revision,
        "contract_sha256": CONTRACT_SHA256,
        "split_manifest_path": str(split_manifest_path),
        "split_manifest_sha256": compute_file_sha256(split_manifest_path),
        "zip_path": str(zip_path),
        "zip_sha256": compute_file_sha256(zip_path) if Path(zip_path).is_file() else None,
        "parameters": parameter_inventory(head),
        "fit_utterances": len(train_dataset),
        "selection_utterances": len(selection_dataset),
        "calibration_utterances": 300,
        "evaluation_utterances": 300,
        "fit_speakers": sorted(FIT_SPEAKERS),
        "selection_speakers": sorted(SELECT_SPEAKERS),
        "calibration_speakers": sorted(CALIBRATION_SPEAKERS),
        "evaluation_speakers": sorted(VAL_SPEAKERS)
    }

    print(f"[{objective.upper()}] Starting P-SUP training on {device} (AMP: {use_cuda_amp})...")
    print(f"[{objective.upper()}] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    for epoch in range(start_epoch, max_epochs):
        if ledger.fixed_budget_reached:
            break

        adapter.train()
        epoch_task_loss = 0.0
        epoch_ctc_loss = 0.0
        epoch_batches = len(train_loader)
        accum_groups = accumulation_groups(epoch_batches, gradient_accumulation_steps)

        optimizer.zero_grad()
        group_idx = 0
        batch_in_group = 0
        current_group_size = accum_groups[group_idx]

        for b_idx, batch in enumerate(train_loader):
            if ledger.fixed_budget_reached:
                break

            audio = batch["waveforms"].to(device)
            audio_lengths = batch["audio_lengths"].to(device)
            prompt_ids = batch["prompt_phone_ids"].to(device)
            prompt_lengths = batch["prompt_lengths"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda_amp):
                out = adapter(
                    audio=audio,
                    audio_lengths=audio_lengths,
                    prompt_phone_ids=prompt_ids,
                    prompt_lengths=prompt_lengths
                )

                loss_t = task_loss(
                    logits=out["operation_logits"],
                    operation=batch["operation"].to(device),
                    mask=batch["operation_mask"].to(device),
                    objective=objective
                )

                loss_c = realized_ctc_loss(
                    logits=out["ctc_logits"],
                    frame_lengths=out["frame_lengths"],
                    targets=batch["ctc"].to(device),
                    target_lengths=batch["ctc_lengths"].to(device)
                )

                total_loss = ctc_weight * loss_c + task_weight * loss_t

            loss_scaled = total_loss / current_group_size
            scaler.scale(loss_scaled).backward()
            micro_steps += 1
            batch_in_group += 1

            epoch_task_loss += loss_t.item()
            epoch_ctc_loss += loss_c.item()

            if batch_in_group == current_group_size:
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=gradient_clip)

                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()

                overflow = (scale_after < scale_before) or (not math.isfinite(norm.item()))
                unskipped = ledger.record_attempt(overflow=overflow)
                if unskipped:
                    scheduler.step()

                optimizer.zero_grad()
                batch_in_group = 0
                group_idx += 1
                if group_idx < len(accum_groups):
                    current_group_size = accum_groups[group_idx]

                # Check exact fixed-budget endpoint
                if ledger.fixed_budget_reached and not fixed_budget_saved:
                    fb_path = output_dir / "fixed_budget_3600.pt"
                    save_checkpoint(
                        path=fb_path,
                        head=head,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        successful_updates=ledger.successful,
                        micro_steps=micro_steps,
                        best_ap=best_ap,
                        patience=patience,
                        skipped_amp=ledger.skipped,
                        history=history,
                        metadata=metadata,
                        is_fixed_budget_endpoint=True,
                        remote_mirror=remote_mirror,
                        remote_rel_prefix=remote_rel_prefix,
                        verify_remote_sha=True,
                        require_remote_mirror=remote_execution,
                    )
                    fb_alias = output_dir / "fixed_budget_checkpoint.pt"
                    shutil.copyfile(str(fb_path), str(fb_alias))
                    fixed_budget_saved = True
                    print(f"[{objective.upper()}] Fixed-budget endpoint 3,600 reached and saved at epoch {epoch}!")
                    if remote_mirror is not None and remote_rel_prefix is not None:
                        try:
                            remote_mirror.mirror_file(fb_path, f"{remote_rel_prefix}/fixed_budget_3600.pt")
                            remote_mirror.mirror_file(fb_alias, f"{remote_rel_prefix}/fixed_budget_checkpoint.pt")
                        except Exception as e:
                            print(f"[{objective.upper()} MIRROR WARNING] Error mirroring fixed budget: {e}")

        # Epoch Validation on SELECT partition (MBMPS, TLV)
        val_metrics = evaluate_val_set(adapter, selection_loader, device, objective)
        val_ap = val_metrics["val_ap"]
        blank_frac = val_metrics["blank_fraction"]
        per = val_metrics["per"]

        epoch_record = {
            "epoch": epoch,
            "successful_updates": ledger.successful,
            "skipped_amp": ledger.skipped,
            "attempts": ledger.attempts,
            "val_ap": val_ap,
            "selection_ap": val_ap,
            "blank_fraction": blank_frac,
            "per": per,
            "avg_task_loss": epoch_task_loss / max(1, epoch_batches),
            "avg_ctc_loss": epoch_ctc_loss / max(1, epoch_batches),
            "lr": optimizer.param_groups[0]["lr"]
        }
        history.append(epoch_record)
        print(
            f"[{objective.upper()}] Epoch {epoch:02d} | Updates: {ledger.successful}/{max_successful_updates} "
            f"| Selection AP: {val_ap:.4f} | Blank: {blank_frac:.4f} | PER: {per:.4f}"
        )

        # Save last.pt each epoch
        last_path = output_dir / "last.pt"
        save_checkpoint(
            path=last_path,
            head=head,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            successful_updates=ledger.successful,
            micro_steps=micro_steps,
            best_ap=best_ap,
            patience=patience,
            skipped_amp=ledger.skipped,
            history=history,
            metadata=metadata,
            remote_mirror=remote_mirror,
            remote_rel_prefix=remote_rel_prefix,
            verify_remote_sha=True,
            require_remote_mirror=remote_execution,
        )
        if remote_mirror is not None and remote_rel_prefix is not None:
            try:
                remote_mirror.mirror_file(last_path, f"{remote_rel_prefix}/last.pt")
            except Exception as e:
                print(f"[{objective.upper()} MIRROR WARNING] Error mirroring last.pt: {e}")

        # Check best.pt selection (strict binary AP on SELECT with blank_fraction < 0.99)
        if blank_frac < 0.99 and val_ap > best_ap:
            best_ap = val_ap
            patience = 0
            best_path = output_dir / "best.pt"
            save_checkpoint(
                path=best_path,
                head=head,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                successful_updates=ledger.successful,
                micro_steps=micro_steps,
                best_ap=best_ap,
                patience=patience,
                skipped_amp=ledger.skipped,
                history=history,
                metadata=metadata,
                remote_mirror=remote_mirror,
                remote_rel_prefix=remote_rel_prefix,
                verify_remote_sha=True,
                require_remote_mirror=remote_execution,
            )
            print(f"[{objective.upper()}] -> New best selection AP: {best_ap:.4f} saved to best.pt")
            if remote_mirror is not None and remote_rel_prefix is not None:
                try:
                    remote_mirror.mirror_file(best_path, f"{remote_rel_prefix}/best.pt")
                except Exception as e:
                    print(f"[{objective.upper()} MIRROR WARNING] Error mirroring best.pt: {e}")
        else:
            patience += 1

        if patience >= early_stopping_patience:
            print(
                f"[{objective.upper()}] Early stopping triggered after {patience} epochs "
                "without AP improvement on SELECT partition."
            )
            break

        if ledger.fixed_budget_reached:
            print(
                f"[{objective.upper()}] Reached registered fixed-budget ceiling of "
                f"{max_successful_updates} successful updates. Terminating training."
            )
            break

    total_time = time.time() - start_time
    summary = {
        "objective": objective,
        "successful_updates": ledger.successful,
        "skipped_updates": ledger.skipped,
        "attempts": ledger.attempts,
        "best_ap": best_ap,
        "fixed_budget_saved": fixed_budget_saved,
        "total_time_seconds": total_time,
        "history": history
    }

    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    history_path = output_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    if remote_mirror is not None:
        rel_summary = f"{remote_rel_prefix}/training_summary.json" if remote_rel_prefix else "training_summary.json"
        rel_history = f"{remote_rel_prefix}/history.json" if remote_rel_prefix else "history.json"
        remote_mirror.mirror_file(summary_path, rel_summary, verify_sha=True)
        remote_mirror.mirror_file(history_path, rel_history, verify_sha=True)

    return summary
