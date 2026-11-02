"""One fail-closed evaluator for all frozen RQ1-BENCH-01 precision profiles.

No model targets or timestamps enter forward. Protocol O uses gold phone identities;
Protocol Z uses written prompts, with gold alignment performed only after forward.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gzip
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
import time
import zipfile

REPOSITORY = Path(__file__).resolve().parents[2]
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
PROFILES = ("LOCAL_FP32", "KAGGLE_FP32", "KAGGLE_FP16")
PRETRAINED_MODEL = "facebook/wav2vec2-base-960h"
PRETRAINED_REVISION = "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
SEMANTICS = {
    "protocol_o": "gold canonical phone identities, no gold times in forward",
    "protocol_z": "CMUDictG2P strict; offline alignment diag_del_ins (= sub_first)",
    "preprocessing": "project audio_loader; mono 16000 Hz; normalize_peak=False; remove_dc=False",
    "attention": "original averaged attention; active lengths; floor(start/.02), ceil(end/.02), min 1 frame; peak=frame*.02",
    "accumulation": "NumPy float64; logits/attention cast float32 before postprocessing",
    "latency": "synchronized O+Z forwards only; one untimed O+Z warmup per fold",
    "seed": 1337, "threads": 1, "batch_size": 1,
}
SOURCE_FILES = tuple(sorted({
    str(p.relative_to(REPOSITORY))
    for package in ("langfens_bench", "langfens_prep")
    for p in (REPOSITORY / "src" / package).glob("*.py")
} | {
    "configs/bench_dependency_contract.json",
    "audit_scripts/evaluate_fold1_ctc_binary.py",
    "audit_scripts/diagnose_omissions_and_false_negatives.py",
}))


class BenchmarkBlocked(RuntimeError):
    """An unmet prerequisite; never triggers a device/network fallback."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_json(path: Path, value):
    path.write_bytes(json_bytes(value))


def write_predictions(path: Path, value):
    # filename and mtime must not leak output directory/time into prediction bytes.
    with path.open("xb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as zipped:
            zipped.write(json_bytes(value))


def repository_path(relative: str, root: Path = REPOSITORY) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise BenchmarkBlocked(f"Path escapes root: {relative}")
    return path


def artifact_path(root: Path, entry: dict) -> Path:
    path = repository_path(entry["path"], root)
    # Kaggle auto-extracts ZIP uploads. A .zip.bin transport preserves the exact
    # registered bytes; the contract's logical path and SHA remain unchanged.
    if not path.is_file() and path.suffix == ".zip":
        transport = repository_path(entry["path"] + ".bin", root)
        if transport.is_file():
            return transport
    return path


def verify_hash(root: Path, entry: dict) -> Path:
    path = artifact_path(root, entry)
    if sha256(path) != entry["sha256"]:
        raise BenchmarkBlocked(f"SHA-256 mismatch: {path}")
    return path


def validate_ids(contract: dict, split: dict) -> dict:
    folds = {f["fold_index"]: f for f in split["folds"]}
    if len(contract["utterances"]) != 80 or len(contract["folds"]) != 4:
        raise BenchmarkBlocked("Frozen benchmark requires four folds and 80 IDs")
    if {f["fold"] for f in contract["folds"]} != {1, 2, 3, 4}:
        raise BenchmarkBlocked("Duplicate/missing benchmark folds")
    seen = set()
    counts = {}
    for u in contract["utterances"]:
        f = folds[u["fold"]]
        speaker, uid = u["speaker"], u["utterance_id"]
        key = f"{speaker}/{uid}"
        prediction_id = f"{u['fold']}/{key}"
        if u["prediction_id"] != prediction_id or prediction_id in seen:
            raise BenchmarkBlocked(f"Duplicate/invalid prediction ID: {prediction_id}")
        if (key not in f["val"]["utterances"] or speaker not in f["val"]["speakers"]
                or speaker in f["test"]["speakers"]):
            raise BenchmarkBlocked(f"Non-validation/active TEST ID refused: {prediction_id}")
        seen.add(prediction_id)
        counts[(u["fold"], speaker)] = counts.get((u["fold"], speaker), 0) + 1
    if len(counts) != 8 or set(counts.values()) != {10}:
        raise BenchmarkBlocked("Expected ten IDs per validation speaker, eight speakers")
    return folds


def audit_package(path: Path, fold: dict) -> dict:
    """Inspect central directories only; never open a WAV/TextGrid member.

    Nested zip containers are read for inventory, not extracted. Every contained
    file must be an exact train/validation manifest member in its own fold.
    """
    authorized = set(fold["train"]["speakers"]) | set(fold["val"]["speakers"])
    if authorized & set(fold["test"]["speakers"]):
        raise BenchmarkBlocked("Split speaker overlap")
    allowed = set(fold["train"]["utterances"]) | set(fold["val"]["utterances"])
    count = 0
    with zipfile.ZipFile(path) as master:
        names = master.namelist()
        if len(names) != len(set(names)) or set(names) != {f"{s}.zip" for s in authorized}:
            raise BenchmarkBlocked(f"Unexpected/duplicate/active TEST outer archive member: {path}")
        for speaker in sorted(authorized):
            expected = {f"{speaker}/{kind}/{uid}.{ext}"
                        for key in allowed for s, uid in [key.split("/")]
                        if s == speaker
                        for kind, ext in (("wav", "wav"), ("annotation", "TextGrid"))}
            with zipfile.ZipFile(io.BytesIO(master.read(f"{speaker}.zip"))) as inner:
                members = inner.namelist()
                if len(members) != len(set(members)) or set(members) != expected:
                    raise BenchmarkBlocked(f"Unauthorized/missing/duplicate inner member: fold {fold['fold_index']} {speaker}")
                count += len(members)
    return {"authorized_speakers": sorted(authorized), "members": count,
            "active_test_excluded": True, "payload_members_read": 0}


def configure_torch(profile: str):
    import numpy as np
    import torch
    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if profile == "LOCAL_FP32":
        return torch.device("cpu"), {"device": "cpu", "processor": platform.processor()}
    if not (Path("/kaggle/input").is_dir() and Path("/kaggle/working").is_dir()
            and os.environ.get("KAGGLE_KERNEL_RUN_TYPE")):
        raise BenchmarkBlocked("Kaggle runtime identity not verified")
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise BenchmarkBlocked("Verified CUDA Tesla T4 cuda:0 required; no fallback")
    torch.cuda.manual_seed_all(1337)
    return torch.device("cuda:0"), {
        "device": "cuda:0", "name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    }


def check_offline_cache() -> dict:
    from huggingface_hub import try_to_load_from_cache
    def cached(name):
        value = try_to_load_from_cache(PRETRAINED_MODEL, name, revision=PRETRAINED_REVISION)
        return value if isinstance(value, str) and Path(value).is_file() else None
    config = cached("config.json")
    weights = cached("model.safetensors") or cached("pytorch_model.bin")
    if not config or not weights:
        raise BenchmarkBlocked(f"Missing offline pretrained cache: {PRETRAINED_MODEL}@{PRETRAINED_REVISION}")
    return {"model": PRETRAINED_MODEL, "revision": PRETRAINED_REVISION,
            "snapshot_path": str(Path(config).parent),
            "config_sha256": sha256(Path(config)), "weights_sha256": sha256(Path(weights)),
            "offline_only": True}


def checkpoint_config(checkpoint: dict, fold: int) -> dict:
    cfg = checkpoint["metadata"]["config"]
    if (cfg["fold"] != fold or cfg["pretrained_model"] != PRETRAINED_MODEL
            or cfg["pretrained_revision"] != PRETRAINED_REVISION
            or cfg["backbone_frozen"] is not True
            or cfg["loss_weights"] != {"ctc": 1.0, "binary": 1.0}
            or cfg.get("alignment_mode", "cross_attention") != "cross_attention"
            or cfg.get("diagnosis_mode", "standard") != "standard"
            or checkpoint["successful_updates"] != 3600):
        raise BenchmarkBlocked(f"Checkpoint/config contract violation: fold {fold}")
    import torch
    if any(t.is_floating_point() and t.dtype != torch.float32 for t in checkpoint["model"].values()):
        raise BenchmarkBlocked(f"Stored checkpoint weights not FP32: fold {fold}")
    return cfg


def scoped_contract(contract: dict, selected_fold: int | None = None) -> dict:
    """Select execution inputs without changing any frozen contract pins."""
    if selected_fold is None:
        return contract
    if selected_fold not in (1, 2, 3, 4):
        raise BenchmarkBlocked("Fold scope must be one of 1, 2, 3, 4")
    folds = [f for f in contract["folds"] if f["fold"] == selected_fold]
    utterances = [u for u in contract["utterances"] if u["fold"] == selected_fold]
    if len(folds) != 1 or not utterances:
        raise BenchmarkBlocked(f"Missing/duplicate fold scope: {selected_fold}")
    return {**contract, "folds": folds, "utterances": utterances}


def provenance(contract_path: Path, contract: dict, profile: str, selected_fold: int | None = None) -> dict:
    scope = scoped_contract(contract, selected_fold)
    sources = {name: sha256(REPOSITORY / name) for name in SOURCE_FILES}
    try:
        commit = subprocess.check_output(["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "schema_version": 1, "benchmark_id": contract["benchmark_id"],
        "contract_sha256": sha256(contract_path), "profile": profile,
        "evaluator_source_sha256": sources,
        "evaluator_semantics_sha256": hashlib.sha256(json_bytes({"sources": sources, "semantics": SEMANTICS})).hexdigest(),
        "source_commit": commit, "semantics": SEMANTICS,
        # These maps preserve all frozen pins, not a claim of unselected artifact access.
        "checkpoint_sha256": {str(f["fold"]): f["checkpoint"]["sha256"] for f in contract["folds"]},
        "package_sha256": {str(f["fold"]): f["package"]["sha256"] for f in contract["folds"]},
        "scope": "fold" if selected_fold is not None else "full", "fold": selected_fold,
        "scope_fold": selected_fold,
        "ordered_prediction_ids": [u["prediction_id"] for u in scope["utterances"]],
        "seed": 1337, "threads": 1, "batch_size": 1,
        "precision": {"stored_weights": "float32", "forward": "cuda_autocast_float16" if profile == "KAGGLE_FP16" else "float32",
                      "postprocessing": "float32", "metrics": "numpy.float64", "tf32": False},
        "preprocessing": SEMANTICS["preprocessing"], "status": "PREFLIGHT",
    }


def audit_remote_inputs(contract: dict, artifact_root: Path, selected_fold: int,
                        inventory_path=None, inventory_sha256=None) -> dict:
    """Prove mount isolation by inventory only; unknown mounted files fail closed.

    Only the two hash-pinned selected-fold artifacts may be mounted as inputs.
    Code/resources/offline cache must be provisioned outside the input mount;
    absent a separate approved inventory, extra inputs cannot be certified safe.
    """
    input_root = KAGGLE_INPUT.resolve()
    if inventory_path is not None:
        from .staging import audit_approved_inputs
        return audit_approved_inputs(contract, artifact_root, selected_fold,
                                     inventory_path, inventory_sha256, input_root)
    if not artifact_root.resolve().is_relative_to(input_root):
        raise BenchmarkBlocked("Remote artifact root must be under /kaggle/input")
    selected = scoped_contract(contract, selected_fold)["folds"][0]
    expected = {artifact_path(artifact_root, selected[k]) for k in ("checkpoint", "package")}
    for fold in contract["folds"]:
        if fold["fold"] != selected_fold:
            for kind in ("checkpoint", "package"):
                path = artifact_root / fold[kind]["path"]
                if path.exists() or path.is_symlink():
                    raise BenchmarkBlocked(f"Other fold artifact mounted: {path}")
    if not input_root.is_dir():
        raise BenchmarkBlocked("Missing /kaggle/input; mounted isolation unverified")
    seen, count = set(), 0
    def fail_walk(error):
        raise error
    for directory, dirs, files in os.walk(input_root, followlinks=False, onerror=fail_walk):
        for name in dirs + files:
            path = Path(directory) / name
            count += 1
            if count > 10000 or path.is_symlink():
                raise BenchmarkBlocked("Mounted isolation unverified: excessive inventory or symlink")
        for name in files:
            path = Path(directory) / name
            if not path.is_file() or path.resolve() not in expected:
                raise BenchmarkBlocked(f"Mounted isolation unverified: unauthorized input {path}")
            seen.add(path.resolve())
    if seen != expected:
        raise BenchmarkBlocked("Mounted isolation unverified: selected artifact missing")
    return {"status": "VERIFIED", "selected_fold": selected_fold, "files": len(seen), "payload_reads": 0}


def preflight(contract: dict, artifact_root: Path, profile: str, selected_fold: int | None = None,
              input_inventory=None, input_inventory_sha256=None) -> tuple[dict, dict]:
    checks, blockers, paths = {}, [], {}
    if profile != "LOCAL_FP32" and selected_fold is None:
        return {"status": "BLOCKED", "blockers": ["Remote profiles require --fold; cross-fold staging is refused"], "checks": checks}, paths
    if profile != "LOCAL_FP32":
        try:
            checks["mounted_input_isolation"] = audit_remote_inputs(
                contract, artifact_root, selected_fold, input_inventory, input_inventory_sha256)
        except (BenchmarkBlocked, OSError, ValueError, KeyError) as exc:
            return {"status": "BLOCKED", "blockers": [str(exc)], "checks": checks}, paths
    try:
        scope = scoped_contract(contract, selected_fold)
        checks["selected_folds"] = [f["fold"] for f in scope["folds"]]
        if any(contract.get(k) != v for k, v in {
            "schema_version": 1, "seed": 1337, "threads": 1, "batch_size": 1,
            "warmup_utterances_per_fold": 1, "repeats": 1, "thresholds": [.2, .5],
        }.items()):
            raise BenchmarkBlocked("Unsupported frozen execution contract")
        split_path = verify_hash(REPOSITORY, contract["split"])
        folds = validate_ids(contract, json.loads(split_path.read_text()))
        checks["split_ids"] = "VERIFIED"
        for name, entry in contract["resources"].items():
            paths[name] = verify_hash(REPOSITORY, entry)
        # Verify every in-scope artifact before any inventory/payload read.
        # Unselected fold packages/checkpoints are never resolved, hashed or opened.
        for fold in scope["folds"]:
            paths[fold["fold"]] = {kind: verify_hash(artifact_root, fold[kind])
                                   for kind in ("checkpoint", "package")}
        checks["all_hashes"] = "VERIFIED_IN_SCOPE"
        checks["verified_artifacts"] = {
            kind + "_sha256": {str(f["fold"]): f[kind]["sha256"] for f in scope["folds"]}
            for kind in ("checkpoint", "package")}
        checks["archive_inventories"] = {
            str(f["fold"]): audit_package(paths[f["fold"]]["package"], folds[f["fold"]])
            for f in scope["folds"]}
    except (BenchmarkBlocked, OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        blockers.append(f"artifacts: {type(exc).__name__}: {exc}")
    try:
        from .dependencies import validate_dependency_contract
        checks["dependency_contract"] = validate_dependency_contract()
        if checks["dependency_contract"]["status"] != "VERIFIED":
            raise BenchmarkBlocked("; ".join(checks["dependency_contract"]["blockers"]))
        # Import actual runtime dependencies, not just their stale lockfile entries.
        import numpy
        import torch
        import torchaudio
        import soundfile
        import transformers
        import scipy
        from langfens_prep.g2p import CMUDictG2P
        from langfens_prep.dataset import load_arctic_prompts
        device, checks["hardware"] = configure_torch(profile)
        checks["offline_cache"] = check_offline_cache()
        paths["device"] = device
        paths["offline_model"] = checks["offline_cache"]["snapshot_path"]
        checks["torch_build"] = {"cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                                  "configuration": torch.__config__.show()}
        if not blockers:
            g2p = CMUDictG2P(paths["cmudict"], oov_policy="strict")
            prompts = load_arctic_prompts(paths["prompts"])
            # Refuse absent prompts/OOV before opening any selected payload.
            paths["z_phones"] = {u["prediction_id"]: g2p.phonemize(prompts[u["utterance_id"]])
                                 for u in scope["utterances"]}
            if not all(paths["z_phones"].values()):
                raise BenchmarkBlocked("Empty Protocol Z prompt phones")
            checks["checkpoint_configs"] = {}
            for f in scope["folds"]:
                ckpt = torch.load(paths[f["fold"]]["checkpoint"], map_location="cpu", weights_only=False)
                cfg = checkpoint_config(ckpt, f["fold"])
                checks["checkpoint_configs"][str(f["fold"])] = {
                    "config": cfg, "epoch": ckpt["epoch"], "successful_updates": ckpt["successful_updates"],
                    "skipped_updates": ckpt.get("skipped_amp_attempts_total"),
                }
                del ckpt
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
        blockers.append(f"dependencies/hardware/cache/config: {type(exc).__name__}: {exc}")
    return {"status": "BLOCKED" if blockers else "READY", "blockers": blockers, "checks": checks}, paths


def attention_metrics(attention, intervals, active_frames: int) -> dict:
    """Offline localization only; float32 conversion precedes every reduction."""
    import numpy as np
    a = np.asarray(attention, dtype=np.float32)[:len(intervals), :active_frames].astype(np.float64)
    if a.shape != (len(intervals), active_frames) or active_frames < 1 or not np.isfinite(a).all():
        raise BenchmarkBlocked("Invalid attention shape/values")
    rows = []
    for weights, interval in zip(a, intervals):
        start, end = interval.xmin, interval.xmax
        lo = min(active_frames - 1, max(0, int(start / .02)))
        hi = min(active_frames, max(lo + 1, math.ceil(end / .02)))
        total = weights.sum()
        if total <= 0 or (weights < 0).any():
            raise BenchmarkBlocked("Invalid attention probability mass")
        p = weights / total
        positive = p[p > 0]
        entropy = -np.sum(positive * np.log(positive)) / np.log(active_frames) if active_frames > 1 else 0.0
        rows.append({"in_window_mass": float(weights[lo:hi].sum()),
                     "uniform_in_window_mass": (hi - lo) / active_frames,
                     "normalized_entropy": float(entropy),
                     "absolute_peak_offset_seconds": abs(float(weights.argmax()) * .02 - (start + end) / 2)})
    return {"active_frames": active_frames, "tokens": rows,
            **{key: float(np.mean([r[key] for r in rows], dtype=np.float64)) if rows else None
               for key in ("in_window_mass", "uniform_in_window_mass", "normalized_entropy", "absolute_peak_offset_seconds")}}


def finite_or_none(value):
    return float(value) if value is not None and math.isfinite(value) else None


def binary_metrics(labels, scores) -> dict:
    from audit_scripts.evaluate_fold1_ctc_binary import average_precision_binary
    from audit_scripts.diagnose_omissions_and_false_negatives import roc_auc_binary
    return {"n": len(labels), "positives": int(sum(labels)),
            "ap": finite_or_none(average_precision_binary(labels, scores)) if labels else None,
            "roc_auc": finite_or_none(roc_auc_binary(labels, scores)) if labels else None}


def aggregate_metrics(records: list[dict], thresholds=(.2, .5)) -> dict:
    import numpy as np
    from audit_scripts.evaluate_fold1_ctc_binary import compute_levenshtein_edits
    labels = [v for r in records for v in r["labels"]]
    scores = [v for r in records for v in r["binary_scores"]]
    edits = sum(compute_levenshtein_edits(r["ctc_reference_ids"], r["ctc_ids"])[0] for r in records)
    ref_count = sum(len(r["ctc_reference_ids"]) for r in records)
    omissions = [s for r in records for e, s in zip(r["error_types"], r["binary_scores"]) if e == "deletion"]
    z_labels, z_scores = [], []
    for r in records:
        for gold, z, _ in r["protocol_z"]["aligned_pairs"]:
            z_labels.append(r["labels"][gold])
            z_scores.append(r["protocol_z"]["binary_scores"][z])
    z = binary_metrics(z_labels, z_scores)
    attn = [t for r in records for t in r["attention_metrics"]["tokens"]]
    latencies = np.asarray([r["latency_seconds"] for r in records], dtype=np.float64)
    o = binary_metrics(labels, scores)
    quality = {
        "ap": o["ap"], "roc_auc": o["roc_auc"],
        "per": edits / ref_count if ref_count else None,
        "attention_mass": float(np.mean([t["in_window_mass"] for t in attn], dtype=np.float64)) if attn else None,
        "attention_entropy": float(np.mean([t["normalized_entropy"] for t in attn], dtype=np.float64)) if attn else None,
        "attention_peak_offset_seconds": float(np.mean([t["absolute_peak_offset_seconds"] for t in attn], dtype=np.float64)) if attn else None,
        "protocol_z": {"ap": z["ap"], "roc_auc": z["roc_auc"],
                       "coverage": z["n"] / len(labels) if labels else None},
    }
    for threshold in thresholds:
        omission_scores = np.asarray(omissions, dtype=np.float64)
        quality[f"omission_recall_{threshold}"] = float(np.mean(omission_scores >= threshold)) if omissions else None
        quality[f"omission_fnr_{threshold}"] = float(np.mean(omission_scores < threshold)) if omissions else None
    return {
        "quality": quality,
        "performance": {"latency_p50_seconds": float(np.quantile(latencies, .5)) if records else None,
                        "latency_p95_seconds": float(np.quantile(latencies, .95)) if records else None,
                        "throughput_utterances_per_second": len(records) / float(latencies.sum()) if records and latencies.sum() > 0 else None},
        "counts": {"utterances": len(records), "tokens": len(labels), "error_tokens": o["positives"],
                   "omission_tokens": len(omissions), "ctc_edits": edits, "ctc_reference_tokens": ref_count,
                   "protocol_z_aligned_tokens": z["n"],
                   "protocol_z_unaligned_gold": sum(r["protocol_z"]["unaligned_gold_count"] for r in records),
                   "protocol_z_unaligned_g2p": sum(r["protocol_z"]["unaligned_g2p_count"] for r in records)},
        "attention_uniform_mass": float(np.mean([t["uniform_in_window_mass"] for t in attn], dtype=np.float64)) if attn else None,
    }


def compute_quality(records: list[dict], thresholds=(.2, .5)) -> dict:
    """Recompute merge quality through the same scorer used by evaluation."""
    return aggregate_metrics(records, thresholds=thresholds)["quality"]


def forward_pair(model, waveform, o_phones, z_phones, profile: str, device):
    import torch
    from langfens_prep.phoneme_normalizer import PHONE_TO_CTC_INDEX
    waveforms = waveform.unsqueeze(0).to(device=device, dtype=torch.float32)
    def inputs(phones):
        return {"waveforms": waveforms,
                "audio_lengths": torch.tensor([waveform.numel()], device=device),
                "canonical_phone_indices": torch.tensor([[PHONE_TO_CTC_INDEX[p] for p in phones]], device=device),
                "canonical_phone_lengths": torch.tensor([len(phones)], device=device)}
    # Tensor preparation and offline work are excluded from forward timing.
    o_inputs, z_inputs = inputs(o_phones), inputs(z_phones)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    context = torch.autocast(device_type="cuda", dtype=torch.float16) if profile == "KAGGLE_FP16" else nullcontext()
    with torch.inference_mode(), context:
        o, z = model(**o_inputs), model(**z_inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return o, z, time.perf_counter() - start


def evaluate(contract: dict, paths: dict, profile: str, selected_fold: int | None = None) -> tuple[list[dict], dict]:
    if profile != "LOCAL_FP32" and selected_fold is None:
        raise BenchmarkBlocked("Remote profiles require --fold; cross-fold staging is refused")
    contract = scoped_contract(contract, selected_fold)
    if profile != "LOCAL_FP32":
        for fold in contract["folds"]:
            for kind in ("checkpoint", "package"):
                if not Path(paths[fold["fold"]][kind]).resolve().is_relative_to(KAGGLE_INPUT.resolve()):
                    raise BenchmarkBlocked("Remote selected-fold artifacts must be under /kaggle/input")
    import torch
    from audit_scripts.evaluate_fold1_ctc_binary import build_eval_model, greedy_ctc_decode
    from langfens_prep.alignment import align_canonical_sequences
    from langfens_prep.audio_loader import load_l2arctic_audio_from_zip, get_l2arctic_reader
    from langfens_prep.phoneme_normalizer import PHONE_TO_CTC_INDEX
    from langfens_prep.textgrid_parser import parse_gold_textgrid_content
    records, peak_allocated, peak_reserved = [], 0, 0
    device = paths["device"]
    for f in contract["folds"]:
        fold = f["fold"]
        checkpoint = torch.load(paths[fold]["checkpoint"], map_location="cpu", weights_only=False)
        cfg = checkpoint_config(checkpoint, fold)
        # An explicit verified snapshot path prevents downloads even when a caller
        # imported Hugging Face before this package set its offline environment.
        model = build_eval_model(paths["offline_model"], cfg["pretrained_revision"], checkpoint["model"],
                                 ctc_weight=cfg["loss_weights"]["ctc"], binary_weight=cfg["loss_weights"]["binary"]).to(device)
        del checkpoint
        if model.training or any(p.requires_grad or p.dtype != torch.float32 for p in model.parameters()):
            raise BenchmarkBlocked("Model must be frozen eval with FP32 stored weights")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        package = str(paths[fold]["package"])
        reader = get_l2arctic_reader(package)
        try:
            for index, u in enumerate(x for x in contract["utterances"] if x["fold"] == fold):
                # Full whitelist/hash/inventory preflight completed before this first payload read.
                ann = parse_gold_textgrid_content(reader.read_annotation_text(u["speaker"], u["utterance_id"]), u["utterance_id"])
                intervals = ann.canonical_intervals
                phones = [iv.canonical_phone for iv in intervals]
                labels = [iv.binary_label_strict for iv in intervals]
                if not phones or any(y not in (0, 1) for y in labels):
                    raise BenchmarkBlocked(f"Invalid active gold tokens: {u['prediction_id']}")
                waveform, sr, _ = load_l2arctic_audio_from_zip(package, u["speaker"], u["utterance_id"])
                if sr != 16000 or not torch.isfinite(waveform).all():
                    raise BenchmarkBlocked("Invalid normalized input audio")
                z_phones = paths["z_phones"][u["prediction_id"]]
                if index == 0:
                    warm_o, warm_z, _ = forward_pair(model, waveform, phones, z_phones, profile, device)
                    del warm_o, warm_z
                output, z_output, elapsed = forward_pair(model, waveform, phones, z_phones, profile, device)
                active_frames = int(output.input_lengths[0])
                logits = output.binary_logits[0, :len(phones)].float()
                z_logits = z_output.binary_logits[0, :len(z_phones)].float()
                attention = output.alignment_weights[0, :len(phones), :active_frames].float()
                ctc = output.ctc_logits[0, :active_frames].float()
                if not all(torch.isfinite(t).all() for t in (logits, z_logits, attention, ctc)):
                    raise BenchmarkBlocked(f"Nonfinite model output: {u['prediction_id']}")
                alignment = align_canonical_sequences(phones, z_phones, tie_break="sub_first")
                ref = [PHONE_TO_CTC_INDEX[iv.realized_phone] for iv in ann.intervals
                       if iv.error_type not in ("silence", "deletion") and not iv.has_err_tag
                       and iv.realized_phone in PHONE_TO_CTC_INDEX]
                records.append({**u, "checkpoint_sha256": f["checkpoint"]["sha256"],
                                "labels": labels, "error_types": [iv.error_type for iv in intervals],
                                "canonical_phones": phones, "binary_logits": logits.cpu().tolist(),
                                "binary_scores": logits.sigmoid().cpu().tolist(),
                                "ctc_ids": greedy_ctc_decode(ctc.argmax(-1).cpu().tolist()),
                                "ctc_reference_ids": ref,
                                "attention_metrics": attention_metrics(attention.cpu().numpy(), intervals, active_frames),
                                "protocol_z": {"canonical_phones": z_phones, "binary_logits": z_logits.cpu().tolist(),
                                               "binary_scores": z_logits.sigmoid().cpu().tolist(),
                                               "aligned_pairs": alignment["aligned_pairs"],
                                               "unaligned_gold_count": len(alignment["unaligned_gold"]),
                                               "unaligned_g2p_count": len(alignment["unaligned_g2p"]),
                                               "tie_break": "sub_first", "tie_break_semantics": "diag_del_ins"},
                                "latency_seconds": elapsed})
                del output, z_output, logits, z_logits, attention, ctc
        finally:
            reader.close()
        if device.type == "cuda":
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
        del model
    # Preserve frozen contract order, not incidental fold grouping order.
    by_id = {r["prediction_id"]: r for r in records}
    records = [by_id[u["prediction_id"]] for u in contract["utterances"]]
    return records, {"peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                     "peak_vram_allocated_bytes": peak_allocated if device.type == "cuda" else None,
                     "peak_vram_reserved_bytes": peak_reserved if device.type == "cuda" else None}


def environment_snapshot(metadata: dict) -> dict:
    versions = {}
    for name in ("torch", "torchaudio", "numpy", "scipy", "soundfile", "transformers", "huggingface-hub", "psutil", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"metadata": metadata, "python": sys.version, "platform": platform.platform(),
            "installed_versions": versions,
            "deterministic_environment": {k: os.environ.get(k) for k in ("MKL_CBWR", "CUBLAS_WORKSPACE_CONFIG", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")},
            "artifacts": {}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "evaluate"))
    parser.add_argument("--profile", required=True, choices=PROFILES)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4), help="Evaluate one fold only; required for remote profiles")
    parser.add_argument("--authorize-remote-inference", action="store_true")
    parser.add_argument("--input-inventory", type=Path)
    parser.add_argument("--input-inventory-sha256")
    args = parser.parse_args(argv)
    if args.profile != "LOCAL_FP32" and not args.outdir.resolve().is_relative_to(KAGGLE_WORKING.resolve()):
        parser.error("Remote output must be under /kaggle/working")
    if args.outdir.exists() or args.outdir.is_symlink():
        parser.error("New output path required; refusing overwrite")
    contract_path = args.contract if args.contract.is_absolute() else REPOSITORY / args.contract
    contract = json.loads(contract_path.read_text())
    metadata = provenance(contract_path, contract, args.profile, args.fold)
    environment = environment_snapshot(metadata)
    args.outdir.mkdir(parents=True, exist_ok=False)
    try:
        if args.command == "evaluate" and args.profile != "LOCAL_FP32" and not args.authorize_remote_inference:
            raise BenchmarkBlocked("Remote evaluation requires --authorize-remote-inference")
        if args.profile != "LOCAL_FP32" and args.fold is None:
            raise BenchmarkBlocked("Remote profiles require --fold; cross-fold staging is refused")
        inventory_options = ({"input_inventory": args.input_inventory,
                              "input_inventory_sha256": args.input_inventory_sha256}
                             if args.input_inventory is not None else {})
        metadata["input_inventory_sha256"] = args.input_inventory_sha256
        result, paths = preflight(contract, args.artifact_root.resolve(), args.profile, args.fold,
                                  **inventory_options)
        environment["preflight"] = result
        if result["status"] != "READY":
            raise BenchmarkBlocked("; ".join(result["blockers"]))
        metadata["status"] = "PREFLIGHT_READY"
        if args.command == "evaluate":
            records, memory = evaluate(contract, paths, args.profile, args.fold)
            if [r["prediction_id"] for r in records] != metadata["ordered_prediction_ids"]:
                raise BenchmarkBlocked("Incomplete/misordered evaluation")
            metadata["status"] = "COMPLETED"
            metrics = aggregate_metrics(records)
            metrics["performance"].update(memory)
            metrics["by_fold"] = {str(f): aggregate_metrics([r for r in records if r["fold"] == f])
                                  for f in sorted({r["fold"] for r in records})}
            write_json(args.outdir / "metrics.json", {"metadata": metadata, **metrics})
            write_predictions(args.outdir / "predictions.json.gz", {"metadata": metadata, "utterances": records})
            environment["artifacts"] = {name: sha256(args.outdir / name) for name in ("metrics.json", "predictions.json.gz")}
        write_json(args.outdir / "environment.json", environment)
        print(json.dumps({"status": metadata["status"], "outdir": str(args.outdir)}))
        return 0
    except Exception as exc:
        metadata["status"] = "BLOCKED" if isinstance(exc, BenchmarkBlocked) else "FAILED"
        environment["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(args.outdir / "environment.json", environment)
        print(f"{metadata['status']}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
