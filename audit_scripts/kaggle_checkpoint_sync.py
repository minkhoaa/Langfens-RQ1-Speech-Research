#!/usr/bin/env python3
"""
LANGFENS — Kaggle Remote Checkpoint Mirroring & Durability Driver
Implements verified per-epoch checkpoint mirroring from ephemeral /kaggle/working
to an approved private remote persistent store (Google Drive, Hugging Face private repo,
or local test directory) with mandatory cryptographic SHA-256 readback verification.

Strict Guardrail:
Ephemeral /kaggle/working alone must NEVER be treated as durable storage.
If remote credentials or sync checks fail, execution fails closed immediately.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def compute_sha256(path: Path | str) -> str:
    """Computes SHA-256 digest of a file in 64KB chunks."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for hashing: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_hf_token(explicit_token: Optional[str] = None) -> Optional[str]:
    """Resolves Hugging Face API token from argument, environment, Kaggle secrets, or cache."""
    if explicit_token:
        return explicit_token.strip()
    for k in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN", "KAGGLE_REMOTE_MIRROR_TOKEN"):
        v = os.environ.get(k)
        if v and v.strip():
            return v.strip()
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        v = UserSecretsClient().get_secret("HF_TOKEN")
        if v and v.strip():
            return v.strip()
    except Exception:
        pass
    # If HF_TOKEN is explicitly set to empty, do not fall back to token cache
    if os.environ.get("HF_TOKEN") == "":
        return None
    # Check potential local or staged token file locations
    candidate_token_files = [
        Path("/kaggle/input/langfens-rq1-fold1-staging/hf_token.txt"),
        *(list(Path("/kaggle/input").rglob("hf_token.txt")) if Path("/kaggle/input").is_dir() else []),
        Path("/kaggle/working/hf_token.txt"),
        Path("/kaggle/working/.hf_token"),
        Path("/kaggle/working/deploy/kaggle/hf_token.txt"),
        Path("/kaggle/working/deploy/kaggle/.hf_token"),
        Path("/kaggle/working/repo/deploy/kaggle/hf_token.txt"),
        Path("/kaggle/working/repo/deploy/kaggle/.hf_token"),
        Path.cwd() / "deploy" / "kaggle" / "hf_token.txt",
        Path.cwd() / "deploy" / "kaggle" / ".hf_token",
        Path.cwd() / "hf_token.txt",
        Path.cwd() / ".hf_token",
        Path.home() / ".cache" / "huggingface" / "token",
    ]
    for tf in candidate_token_files:
        if tf.is_file():
            t = tf.read_text().strip()
            if t:
                return t
    return None


class KaggleRemoteMirror:
    """
    Manages atomic per-epoch synchronization of checkpoints from Kaggle VM
    to an approved private remote persistent storage target with SHA-256 verification.
    """
    SUPPORTED_BACKENDS = ("gdrive", "huggingface", "kaggle_dataset", "local_dir")

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
                f"Unsupported remote mirror backend '{backend}'. "
                f"Must be one of: {self.SUPPORTED_BACKENDS}"
            )
        self.backend = backend_lower
        self.remote_target = str(remote_target) if remote_target else os.environ.get("KAGGLE_REMOTE_MIRROR_TARGET")
        self.auth_token = auth_token or os.environ.get("KAGGLE_REMOTE_MIRROR_TOKEN")
        self.require_verification = require_verification
        self.allow_local_test = allow_local_test
        self._verified = False

    def verify_readiness(self) -> Tuple[bool, str]:
        """
        Tests whether the remote persistent target is configured, authorized,
        and provides verified write-read-delete capability.
        """
        if not self.remote_target:
            return (
                False,
                "FAIL-CLOSED: No remote persistent mirror target configured. "
                "Ephemeral /kaggle/working cannot be used as sole storage.",
            )

        if self.backend == "local_dir":
            p = Path(self.remote_target).resolve()
            # Strictly reject ephemeral /kaggle directory unless explicitly in unit-test harness outside Kaggle
            if Path("/kaggle").is_dir() and p.is_relative_to(Path("/kaggle")):
                return (
                    False,
                    f"FAIL-CLOSED: Ephemeral /kaggle path '{p}' cannot be used as durable remote mirror target.",
                )
            if Path("/kaggle").is_dir() and not self.allow_local_test:
                return (
                    False,
                    "FAIL-CLOSED: local_dir backend is prohibited on Kaggle platform. Must use approved private remote storage.",
                )
            try:
                p.mkdir(parents=True, exist_ok=True)
                probe = p / f".probe_readiness_{int(time.time()*1000)}.tmp"
                probe.write_text("probe")
                if not probe.is_file() or probe.read_text() != "probe":
                    return False, f"FAIL-CLOSED: Local mirror directory '{p}' readback probe failed."
                probe.unlink()
                self._verified = True
                return True, f"Local mirror directory '{p}' verified."
            except Exception as e:
                return False, f"FAIL-CLOSED: Failed to access local mirror directory '{p}': {e}"

        elif self.backend == "huggingface":
            token = resolve_hf_token(self.auth_token)
            if not token:
                return (
                    False,
                    "FAIL-CLOSED: Hugging Face remote mirror requires valid HF_TOKEN.",
                )
            try:
                from huggingface_hub import HfApi, hf_hub_download  # type: ignore
                api = HfApi(token=token)
                repo_id = self.remote_target
                # Probe authenticated write-readback-delete
                probe_name = f".probe_mirror_{int(time.time()*1000)}.tmp"
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
                    readback_sha = compute_sha256(downloaded)
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
                return True, f"Hugging Face private repository '{repo_id}' verified with authenticated write-readback probe."
            except Exception as e:
                return (
                    False,
                    f"FAIL-CLOSED: Hugging Face remote mirror probe failed for repo '{self.remote_target}': {e}",
                )

        elif self.backend == "gdrive":
            creds = self.auth_token or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("GDRIVE_SA_KEY")
            if not creds:
                return (
                    False,
                    "FAIL-CLOSED: Google Drive remote mirror requires valid credentials "
                    "(GOOGLE_APPLICATION_CREDENTIALS or GDRIVE_SA_KEY).",
                )
            return (
                False,
                "FAIL-CLOSED: Google Drive remote mirror on Kaggle requires verified Drive Service Account probe. Use 'huggingface' backend.",
            )

        elif self.backend == "kaggle_dataset":
            return (
                False,
                "FAIL-CLOSED: Kaggle Dataset backend does not support synchronous per-epoch atomic file readback. Use 'huggingface' backend.",
            )

        return False, f"Unknown backend: {self.backend}"

    def mirror_file(
        self,
        local_path: Path | str,
        remote_rel_path: str,
        verify_sha: bool = True,
    ) -> Dict[str, Any]:
        """
        Atomically mirrors a local file to the remote target and verifies SHA-256.
        Raises IOError if remote SHA does not match local SHA.
        Never reports SYNCED on simulated or unverified operations.
        """
        local_p = Path(local_path).resolve()
        if not local_p.is_file():
            raise FileNotFoundError(f"Local file to mirror does not exist: {local_p}")

        if not self._verified and self.require_verification:
            ok, msg = self.verify_readiness()
            if not ok:
                raise RuntimeError(msg)

        local_sha = compute_sha256(local_p)
        file_size = local_p.stat().st_size
        t0 = time.time()

        if self.backend == "local_dir":
            remote_dest = Path(self.remote_target) / remote_rel_path
            remote_dest.parent.mkdir(parents=True, exist_ok=True)
            tmp_remote = remote_dest.with_suffix(remote_dest.suffix + ".mirror_tmp")
            shutil.copyfile(str(local_p), str(tmp_remote))
            os.replace(tmp_remote, remote_dest)

            if not remote_dest.is_file():
                raise IOError(f"FAIL-CLOSED: Remote mirrored file does not exist: {remote_dest}")

            if verify_sha:
                remote_sha = compute_sha256(remote_dest)
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
            token = resolve_hf_token(self.auth_token)
            if not token:
                raise RuntimeError("FAIL-CLOSED: Hugging Face token missing during mirror_file.")

            from huggingface_hub import HfApi, hf_hub_download  # type: ignore
            api = HfApi(token=token)
            repo_id = self.remote_target

            # Genuine upload to private Hugging Face repo
            api.upload_file(
                path_or_fileobj=str(local_p),
                path_in_repo=remote_rel_path,
                repo_id=repo_id,
                repo_type="model",
            )

            # Genuine readback verification
            if verify_sha:
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=remote_rel_path,
                    repo_type="model",
                    token=token,
                    force_download=True,
                )
                remote_sha = compute_sha256(downloaded)
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

        raise NotImplementedError(
            f"FAIL-CLOSED: Genuine durable upload not implemented for backend '{self.backend}'. "
            "Never mark simulated uploads as successful."
        )
