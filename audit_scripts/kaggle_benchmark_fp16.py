#!/usr/bin/env python3
"""
LANGFENS — Disposable 50-Update Kaggle T4 CUDA FP16 AMP Real-Data Benchmark Driver
Measures authentic update latency, GPU peak memory, validation throughput,
and runtime telemetry on Nvidia Tesla T4 GPU before any official C0/C1 launch.

Scientific & Operational Invariants:
1. Disposable & Isolated: NEVER counts toward official Control Arm C0.
   Outputs strictly to isolated benchmark directory.
2. Identical Precision Regime: CUDA FP16 AMP (torch.autocast, float16).
3. Matched Architecture & Weights: Initialized from Baseline R best.pt (SHA-256 a2445c8d...),
   frozen Wav2Vec2 backbone, untrained zero-gradient diagnosis head, CTC+binary heads trainable.
4. Real Fold 1 Data: Batch size 2, accumulation 2, seed 1337, AdamW lr 1e-3, cosine warmup.
5. Telemetry: Tracks p50/p90/p95 latency, allocated/reserved VRAM, validation overhead,
   and derives full-arm (3,600 update) runtime projection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from langfens_prep.dataset import L2ArcticMDDDataset
from langfens_prep.collate import LangfensMDDCollate
from langfens_prep.model import LangfensWav2Vec2MDD

try:
    from audit_scripts.kaggle_stage_and_verify import (
        EXPECTED_BASELINE_SHA256,
        EXPECTED_MANIFEST_SHA256,
        EXPECTED_ZIP_SHA256,
        compute_sha256,
        verify_kaggle_gpu_device,
        verify_split_manifest,
        verify_baseline_weights,
        verify_and_stage_fold1_data,
    )
except ImportError:
    from kaggle_stage_and_verify import (
        EXPECTED_BASELINE_SHA256,
        EXPECTED_MANIFEST_SHA256,
        EXPECTED_ZIP_SHA256,
        compute_sha256,
        verify_kaggle_gpu_device,
        verify_split_manifest,
        verify_baseline_weights,
        verify_and_stage_fold1_data,
    )


def seed_all(seed: int):
    """Sets deterministic random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_kaggle_fp16_benchmark(
    config_path: str | Path,
    kaggle_input_dir: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    max_updates: int = 50,
    allow_mock_gpu: bool = False,
) -> Dict[str, Any]:
    """
    Executes disposable 50-update real-data FP16 benchmark on Kaggle Tesla T4 GPU.
    Returns comprehensive telemetry dictionary.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    work_out = Path(output_dir or cfg.get("output_dir", "/kaggle/working/benchmark")).resolve()
    work_out.mkdir(parents=True, exist_ok=True)

    # 1. Device Verification
    gpu_info = verify_kaggle_gpu_device(allow_mock=allow_mock_gpu)
    device = torch.device("cuda" if torch.cuda.is_available() and not allow_mock_gpu else "cpu")
    if device.type != "cuda" and not allow_mock_gpu:
        raise RuntimeError("FAIL-CLOSED: Benchmark requires CUDA Tesla T4 GPU.")

    # 2. Immutable Asset Verification
    manifest_path = Path(cfg.get("manifest", "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json"))
    manifest_sha = verify_split_manifest(manifest_path)

    init_ckpt = Path(cfg.get("initial_checkpoint", "checkpoints/RQ1-TRN-02A/recovered/best.pt"))
    if not init_ckpt.is_file() and kaggle_input_dir:
        cand = Path(kaggle_input_dir) / "best.pt"
        if cand.is_file():
            init_ckpt = cand
    baseline_sha = verify_baseline_weights(init_ckpt)

    # 3. Data Staging
    zip_source = Path(cfg.get("zip_path", "datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip"))
    if not zip_source.exists() and kaggle_input_dir:
        zip_source = Path(kaggle_input_dir)
    staged_zip = verify_and_stage_fold1_data(zip_source, target_staging_dir=work_out / "staged")

    # 4. Deterministic Seeding & Model Instantiation
    seed_all(cfg["seed"])

    model = LangfensWav2Vec2MDD.from_pretrained(
        pretrained_model_name_or_path=cfg["pretrained_model"],
        revision=cfg["pretrained_revision"],
        ctc_weight=cfg["loss_weights"]["ctc"],
        binary_weight=cfg["loss_weights"]["binary"],
        binary_prompt_dropout_p=0.00,  # Benchmark uses p=0.0
    )

    init_payload = torch.load(init_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(init_payload["model"])
    trainable = model.configure_two_head_training()
    model.seed_intervention_generator(cfg["seed"], device="cpu")
    model.to(device)

    # 5. Optimizer, Scheduler, and AMP Scaler
    accum = int(cfg.get("gradient_accumulation_steps", 2))
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.get("warmup_updates", 10),
        num_training_steps=max_updates,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=float(cfg.get("grad_scaler_initial_scale", 4096.0)),
        enabled=(device.type == "cuda"),
    )

    # 6. Dataloader Setup
    train_dataset = L2ArcticMDDDataset(str(manifest_path), 1, "train", policy="strict", zip_path=str(staged_zip))
    val_dataset = L2ArcticMDDDataset(str(manifest_path), 1, "val", policy="strict", zip_path=str(staged_zip))

    train_dl_generator = torch.Generator()
    train_dl_generator.manual_seed(cfg["seed"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=LangfensMDDCollate(),
        generator=train_dl_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=LangfensMDDCollate(),
    )

    # 7. Benchmark Loop Execution
    model.train()
    optimizer.zero_grad(set_to_none=True)

    update_latencies: List[float] = []
    vram_samples: List[Dict[str, float]] = []
    successful_updates = 0
    attempted_updates = 0
    skipped_updates = 0
    micro_steps = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    update_t0 = time.perf_counter()

    for bi, batch in enumerate(train_loader):
        if successful_updates >= max_updates:
            break

        args = dict(
            audio_waveforms=batch["audio"].to(device),
            audio_lengths=batch["audio_lengths"].to(device),
            canonical_tokens=batch["canonical_tokens"].to(device),
            canonical_lengths=batch["canonical_lengths"].to(device),
            target_ctc=batch["target_ctc"].to(device),
            target_lengths=batch["target_lengths"].to(device),
            target_binary=batch["target_binary"].to(device),
        )

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            out = model(**args)

        loss = out.loss / accum
        scaler.scale(loss).backward()
        micro_steps += 1

        if (bi + 1) % accum == 0:
            attempted_updates += 1
            scaler.unscale_(optimizer)

            # Gradient check: diagnosis head must have zero gradients
            for p in model.diagnosis_head.parameters():
                if p.grad is not None:
                    raise AssertionError("Diagnosis gradient must remain strictly absent.")

            finite_grads = all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable)
            old_scale = scaler.get_scale()
            if finite_grads:
                torch.nn.utils.clip_grad_norm_(trainable, cfg["gradient_clip_norm"])

            scaler.step(optimizer)
            scaler.update()

            update_duration = time.perf_counter() - update_t0

            if scaler.get_scale() >= old_scale:
                successful_updates += 1
                scheduler.step()
                update_latencies.append(update_duration)
            else:
                skipped_updates += 1

            if torch.cuda.is_available():
                vram_samples.append({
                    "allocated_mb": round(torch.cuda.memory_allocated(device) / (1024 * 1024), 2),
                    "reserved_mb": round(torch.cuda.memory_reserved(device) / (1024 * 1024), 2),
                })

            optimizer.zero_grad(set_to_none=True)
            update_t0 = time.perf_counter()

    # 8. Single Validation Pass Throughput Measurement
    model.eval()
    val_t0 = time.perf_counter()
    val_count = 0
    with torch.inference_mode():
        for batch in val_loader:
            args = dict(
                audio_waveforms=batch["audio"].to(device),
                audio_lengths=batch["audio_lengths"].to(device),
                canonical_tokens=batch["canonical_tokens"].to(device),
                canonical_lengths=batch["canonical_lengths"].to(device),
                target_ctc=batch["target_ctc"].to(device),
                target_lengths=batch["target_lengths"].to(device),
                target_binary=batch["target_binary"].to(device),
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                _ = model(**args)
            val_count += len(batch["audio"])

    val_duration = time.perf_counter() - val_t0
    val_throughput = val_count / max(val_duration, 1e-6)

    # 9. Compute Telemetry Statistics
    lat_arr = np.array(update_latencies) if update_latencies else np.array([0.0])
    mean_lat = float(np.mean(lat_arr))
    p50_lat = float(np.percentile(lat_arr, 50))
    p90_lat = float(np.percentile(lat_arr, 90))
    p95_lat = float(np.percentile(lat_arr, 95))
    min_lat = float(np.min(lat_arr))
    max_lat = float(np.max(lat_arr))

    peak_allocated_mb = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0
    peak_reserved_mb = round(torch.cuda.max_memory_reserved(device) / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0

    # Extrapolate full 3,600-update arm runtime (including 6 validation passes)
    projected_arm_seconds = (mean_lat * 3600) + (val_duration * 6)
    projected_arm_hours = projected_arm_seconds / 3600.0

    telemetry: Dict[str, Any] = {
        "benchmark_schema_version": "1.0.0",
        "experiment": "RQ1-TRN-02B-BENCHMARK-KAGGLE-T4",
        "arm": "BENCHMARK",
        "disposable_benchmark": True,
        "counts_toward_c0": False,
        "platform": "KAGGLE_T4_CUDA_FP16_AMP",
        "gpu": gpu_info,
        "successful_updates": successful_updates,
        "attempted_updates": attempted_updates,
        "skipped_updates": skipped_updates,
        "micro_steps": micro_steps,
        "latency_telemetry_seconds": {
            "mean": round(mean_lat, 4),
            "p50": round(p50_lat, 4),
            "p90": round(p90_lat, 4),
            "p95": round(p95_lat, 4),
            "min": round(min_lat, 4),
            "max": round(max_lat, 4),
        },
        "vram_telemetry_mb": {
            "peak_allocated_mb": peak_allocated_mb,
            "peak_reserved_mb": peak_reserved_mb,
        },
        "validation_telemetry": {
            "duration_seconds": round(val_duration, 2),
            "samples_evaluated": val_count,
            "throughput_samples_per_sec": round(val_throughput, 2),
        },
        "full_arm_runtime_projection": {
            "projected_seconds_per_arm": round(projected_arm_seconds, 1),
            "projected_hours_per_arm": round(projected_arm_hours, 3),
            "projected_combined_c0_c1_hours": round(projected_arm_hours * 2, 3),
            "budget_cap_updates": 3600,
            "epochs": 6,
        },
        "immutable_assets": {
            "manifest_sha256": manifest_sha,
            "baseline_sha256": baseline_sha,
            "staged_zip_sha256": compute_sha256(staged_zip),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Save telemetry JSON
    telemetry_file = work_out / "benchmark_kaggle_t4_fp16_telemetry.json"
    telemetry_file.write_text(json.dumps(telemetry, indent=2) + "\n")
    print(f"[BENCHMARK] Telemetry written to {telemetry_file}")

    return telemetry


def main():
    parser = argparse.ArgumentParser(description="LANGFENS — Disposable Kaggle T4 FP16 Benchmark")
    parser.add_argument("--config", type=str, default="configs/rq1_trn_02b_benchmark_kaggle.yaml", help="Path to benchmark config")
    parser.add_argument("--kaggle-input-dir", type=str, default="/kaggle/input/langfens-fold1-train-val-baseline", help="Kaggle input dir")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/benchmark", help="Benchmark output dir")
    parser.add_argument("--max-updates", type=int, default=50, help="Number of successful updates (fixed 50)")
    parser.add_argument("--allow-mock-gpu", action="store_true", help="Allow mock GPU for testing only")
    args = parser.parse_args()

    telemetry = run_kaggle_fp16_benchmark(
        config_path=args.config,
        kaggle_input_dir=args.kaggle_input_dir,
        output_dir=args.output_dir,
        max_updates=args.max_updates,
        allow_mock_gpu=args.allow_mock_gpu,
    )
    print(json.dumps(telemetry, indent=2))


if __name__ == "__main__":
    main()
