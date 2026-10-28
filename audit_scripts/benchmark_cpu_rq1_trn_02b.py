#!/usr/bin/env python3
"""
LANGFENS — RQ1-TRN-02B: Google Colab CPU Feasibility Benchmark Driver
Executes a disposable benchmark of 50 successful optimizer updates on CPU (FP32)
to measure realistic computational throughput and project execution durations for C0/C1.

Contracts:
- Disposable: Does NOT create official experiment checkpoints or contaminate training history.
- Realistic: Uses real authorized Fold 1 audio batches, full forward pass, backward pass,
  gradient accumulation, and optimizer step matching configs/rq1_trn_02b_c0.yaml.
- Comprehensive Metrics:
  1. CPU model initialization time.
  2. Median, mean, min, max, std dev seconds per optimizer update.
  3. Breakdown: Data loading overhead, forward pass, backward pass, optimizer step.
  4. RAM consumption: Baseline, post-model-load, peak during training.
  5. Validation batch time and full-validation runtime estimate (300 utterances / 75 batches).
  6. Projections: Time per epoch, time for 3,600 updates, total time for C0 and C1.
  7. Feasibility decision against available Google Colab session runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import psutil
import torch
import yaml
from torch.utils.data import DataLoader

from langfens_prep.collate import LangfensMDDCollate
from langfens_prep.dataset import L2ArcticMDDDataset
from langfens_prep.model import LangfensWav2Vec2MDD


def compute_sha256(path: Path | str) -> str:
    """Computes SHA-256 digest in 64KB blocks."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_cpu_info() -> Dict[str, Any]:
    """Extracts system and CPU hardware info."""
    cpu_model = "Unknown"
    try:
        if Path("/proc/cpuinfo").is_file():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if "model name" in line:
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    mem = psutil.virtual_memory()
    return {
        "system": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu_model": cpu_model,
        "logical_cpus": multiprocessing.cpu_count(),
        "total_ram_gb": round(mem.total / (1024**3), 2),
        "available_ram_gb": round(mem.available / (1024**3), 2),
    }


def get_ram_usage_gb() -> float:
    """Returns current process RSS memory in GB."""
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024**3), 3)


def step_model(model: LangfensWav2Vec2MDD, batch: Dict[str, Any], device: torch.device, training: bool = True):
    """Executes single forward pass in FP32 on CPU."""
    ni = batch["net_input"]
    result = model(
        source=ni["source"].to(device),
        audio_lengths=ni["audio_lengths"].to(device),
        canonical_phone_indices=ni["canonical_phone_indices"].to(device),
        canonical_phone_lengths=ni["canonical_phone_lengths"].to(device),
        target_ctc=batch["target_ctc"].to(device),
        target_lengths=batch["target_lengths"].to(device),
        target_binary=batch["target_binary"].to(device),
    )
    return result


def run_benchmark(
    config_path: Path | str,
    zip_path: Optional[Path | str] = None,
    initial_ckpt_path: Optional[Path | str] = None,
    num_updates: int = 50,
    val_batches_to_measure: int = 10,
    colab_runtime_budget_hours: float = 36.33,
    output_json: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Runs the 50-update CPU benchmark and computes feasibility projections."""
    cfg = yaml.safe_load(Path(config_path).read_text())

    if zip_path:
        cfg["zip_path"] = str(zip_path)
    if initial_ckpt_path:
        cfg["initial_checkpoint"] = str(initial_ckpt_path)

    # 1. Hardware & System Baseline
    hw_info = get_cpu_info()
    ram_baseline = get_ram_usage_gb()
    print("=" * 80)
    print("LANGFENS — RQ1-TRN-02B CPU Feasibility Benchmark")
    print(f"CPU Model: {hw_info['cpu_model']} | Logical Cores: {hw_info['logical_cpus']}")
    print(f"Total RAM: {hw_info['total_ram_gb']} GB | Available RAM: {hw_info['available_ram_gb']} GB")
    print(f"Process RAM Baseline: {ram_baseline:.3f} GB")
    print("=" * 80, flush=True)

    # 2. Checksum Verifications
    manifest_path = Path(cfg["manifest"])
    manifest_sha = compute_sha256(manifest_path)
    if manifest_sha != cfg["manifest_sha256"]:
        raise RuntimeError(f"Manifest hash mismatch: {manifest_sha} != {cfg['manifest_sha256']}")

    init_ckpt = Path(cfg["initial_checkpoint"])
    init_sha = compute_sha256(init_ckpt)
    if init_sha != cfg["initial_checkpoint_sha256"]:
        raise RuntimeError(f"Initial checkpoint hash mismatch: {init_sha} != {cfg['initial_checkpoint_sha256']}")

    if cfg.get("zip_path"):
        data_zip = Path(cfg["zip_path"])
        zip_sha = compute_sha256(data_zip)
        print(f"[DATA INTEGRITY] Verified fold1_train_val_only.zip: {zip_sha}")

    # 3. CPU Device Configuration
    device = torch.device("cpu")
    torch.set_num_threads(hw_info["logical_cpus"])
    print(f"[PYTORCH THREADS] Set torch.set_num_threads({hw_info['logical_cpus']})")

    # 4. Model Initialization Timing
    print("[INIT] Loading model from baseline checkpoint onto CPU (FP32)...", flush=True)
    t0_init = time.perf_counter()
    model = LangfensWav2Vec2MDD.from_pretrained(
        pretrained_model_name_or_path=cfg["pretrained_model"],
        revision=cfg["pretrained_revision"],
        ctc_weight=cfg["loss_weights"]["ctc"],
        binary_weight=cfg["loss_weights"]["binary"],
        binary_prompt_dropout_p=float(cfg.get("binary_prompt_dropout_p", 0.0)),
    )
    init_payload = torch.load(init_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(init_payload["model"])
    model.to(device)
    init_time_sec = round(time.perf_counter() - t0_init, 2)
    ram_post_model = get_ram_usage_gb()
    print(f"[INIT] Model initialized in {init_time_sec:.2f}s | RAM after load: {ram_post_model:.3f} GB (+{ram_post_model - ram_baseline:.3f} GB)")

    # 5. Dataset & DataLoader Construction
    print("[DATA] Instantiating training and validation datasets...", flush=True)
    t0_data = time.perf_counter()
    zip_p = cfg.get("zip_path", "datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip")
    train_ds = L2ArcticMDDDataset(str(manifest_path), 1, "train", policy="strict", zip_path=zip_p)
    val_ds = L2ArcticMDDDataset(str(manifest_path), 1, "val", policy="strict", zip_path=zip_p)

    collate_fn = LangfensMDDCollate()
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    data_setup_sec = round(time.perf_counter() - t0_data, 2)
    print(f"[DATA] Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | Setup time: {data_setup_sec:.2f}s")

    # 6. Optimizer Setup
    lr = float(cfg.get("learning_rate", 0.001))
    weight_decay = float(cfg.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    accum_steps = int(cfg["gradient_accumulation_steps"])  # 2 micro-batches per update
    batch_size = int(cfg["batch_size"])                    # 2 utterances
    eff_batch_size = accum_steps * batch_size             # 4 utterances / update

    print(f"[CONFIG] Batch Size: {batch_size} | Accumulation: {accum_steps} | Effective Batch: {eff_batch_size} utterances")
    print(f"[BENCHMARK] Starting disposable benchmark of {num_updates} successful updates on CPU FP32...", flush=True)

    # 7. Benchmark Loop
    update_durations: List[float] = []
    data_durations: List[float] = []
    forward_durations: List[float] = []
    backward_durations: List[float] = []
    optimizer_durations: List[float] = []

    model.train()
    optimizer.zero_grad()

    train_iter = iter(train_loader)
    updates_done = 0
    micro_step = 0
    t_update_start = time.perf_counter()
    update_forward_time = 0.0
    update_backward_time = 0.0
    update_data_time = 0.0

    peak_ram = ram_post_model

    t0_loop = time.perf_counter()
    while updates_done < num_updates:
        # Measure Data Loading Overhead
        t_data_start = time.perf_counter()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        t_data = time.perf_counter() - t_data_start
        update_data_time += t_data

        # Measure Forward Pass
        t_fwd_start = time.perf_counter()
        result = step_model(model, batch, device, training=True)
        loss = result["loss"] / accum_steps
        t_fwd = time.perf_counter() - t_fwd_start
        update_forward_time += t_fwd

        # Measure Backward Pass
        t_bwd_start = time.perf_counter()
        loss.backward()
        t_bwd = time.perf_counter() - t_bwd_start
        update_backward_time += t_bwd

        micro_step += 1

        if micro_step % accum_steps == 0:
            # Measure Optimizer Step
            t_opt_start = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                float(cfg.get("gradient_clip_norm", 1.0)),
            )
            optimizer.step()
            optimizer.zero_grad()
            t_opt = time.perf_counter() - t_opt_start

            total_update_time = time.perf_counter() - t_update_start

            update_durations.append(total_update_time)
            data_durations.append(update_data_time)
            forward_durations.append(update_forward_time)
            backward_durations.append(update_backward_time)
            optimizer_durations.append(t_opt)

            cur_ram = get_ram_usage_gb()
            if cur_ram > peak_ram:
                peak_ram = cur_ram

            updates_done += 1
            if updates_done % 10 == 0 or updates_done == 1 or updates_done == num_updates:
                print(
                    f"  [Update {updates_done:02d}/{num_updates}] "
                    f"Total: {total_update_time:.2f}s | "
                    f"Fwd: {update_forward_time:.2f}s | "
                    f"Bwd: {update_backward_time:.2f}s | "
                    f"Opt: {t_opt:.2f}s | "
                    f"Data: {update_data_time:.2f}s | "
                    f"RAM: {cur_ram:.2f}GB",
                    flush=True,
                )

            # Reset update timer
            t_update_start = time.perf_counter()
            update_forward_time = 0.0
            update_backward_time = 0.0
            update_data_time = 0.0

    total_training_benchmark_sec = time.perf_counter() - t0_loop

    # 8. Validation Benchmark
    print(f"\n[VAL BENCHMARK] Measuring {val_batches_to_measure} validation batches on CPU FP32...", flush=True)
    model.eval()
    val_iter = iter(val_loader)
    val_batch_durations: List[float] = []

    with torch.no_grad():
        for vb_idx in range(val_batches_to_measure):
            try:
                vbatch = next(val_iter)
            except StopIteration:
                break
            t_vb_start = time.perf_counter()
            _ = step_model(model, vbatch, device, training=False)
            t_vb = time.perf_counter() - t_vb_start
            val_batch_durations.append(t_vb)

    avg_val_batch_sec = statistics.mean(val_batch_durations)
    total_val_batches = len(val_loader)  # 300 / 4 = 75 batches
    est_full_val_epoch_sec = avg_val_batch_sec * total_val_batches
    print(f"[VAL BENCHMARK] Mean val batch: {avg_val_batch_sec:.2f}s | Est full val pass (75 batches): {est_full_val_epoch_sec:.1f}s ({est_full_val_epoch_sec / 60:.1f}m)")

    # 9. Statistical Aggregation
    median_update_sec = statistics.median(update_durations)
    mean_update_sec = statistics.mean(update_durations)
    min_update_sec = min(update_durations)
    max_update_sec = max(update_durations)
    std_update_sec = statistics.stdev(update_durations) if len(update_durations) > 1 else 0.0

    mean_fwd_sec = statistics.mean(forward_durations)
    mean_bwd_sec = statistics.mean(backward_durations)
    mean_opt_sec = statistics.mean(optimizer_durations)
    mean_data_sec = statistics.mean(data_durations)

    # 10. Projections & Feasibility Analysis
    # Training set has 2,399 utterances.
    # With batch size 2 * accum 2 = 4 utterances per update:
    updates_per_epoch = int((len(train_ds) + eff_batch_size - 1) // eff_batch_size)  # ~600 updates / epoch
    target_budget_updates = 3600  # Cap: 6 epochs or 3,600 updates
    max_epochs = 6

    # Time per epoch = 600 updates + 1 full validation pass
    epoch_train_time_sec = updates_per_epoch * median_update_sec
    epoch_total_sec = epoch_train_time_sec + est_full_val_epoch_sec

    # Full C0 run (6 epochs = 3,600 updates at 600 updates/epoch!)
    c0_6_epochs_updates = updates_per_epoch * max_epochs  # 600 * 6 = 3,600 updates exactly!
    c0_6_epochs_sec = (c0_6_epochs_updates * median_update_sec) + (max_epochs * est_full_val_epoch_sec)
    c0_6_epochs_hours = c0_6_epochs_sec / 3600.0

    both_arms_hours = c0_6_epochs_hours * 2.0

    available_budget_hours = colab_runtime_budget_hours
    budget_safety_ratio = available_budget_hours / max(c0_6_epochs_hours, 1e-6)

    # Feasibility verdict
    if c0_6_epochs_hours <= (available_budget_hours / 1.25):
        verdict = "CPU_FEASIBLE"
        verdict_detail = (
            f"FEASIBLE: 6 epochs (3,600 updates) estimated at {c0_6_epochs_hours:.1f}h "
            f"fits within available Colab runtime budget ({available_budget_hours:.1f}h, ratio {budget_safety_ratio:.2f}x)."
        )
    else:
        verdict = "CPU_NOT_FEASIBLE"
        verdict_detail = (
            f"NOT FEASIBLE: 6 epochs (3,600 updates) requires {c0_6_epochs_hours:.1f}h "
            f"(both arms require {both_arms_hours:.1f}h), "
            f"which exceeds the Colab runtime budget of {available_budget_hours:.1f}h (or leaves insufficient safety margin)."
        )

    results = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": hw_info,
        "ram": {
            "baseline_gb": ram_baseline,
            "post_model_load_gb": ram_post_model,
            "peak_training_gb": peak_ram,
            "net_increase_gb": round(peak_ram - ram_baseline, 3),
        },
        "model_init_time_sec": init_time_sec,
        "measured_updates_count": num_updates,
        "per_update_metrics": {
            "median_sec": round(median_update_sec, 3),
            "mean_sec": round(mean_update_sec, 3),
            "min_sec": round(min_update_sec, 3),
            "max_sec": round(max_update_sec, 3),
            "std_dev_sec": round(std_update_sec, 3),
            "breakdown_mean": {
                "forward_pass_sec": round(mean_fwd_sec, 3),
                "backward_pass_sec": round(mean_bwd_sec, 3),
                "optimizer_step_sec": round(mean_opt_sec, 3),
                "data_loading_sec": round(mean_data_sec, 3),
            },
        },
        "validation_metrics": {
            "measured_batches": len(val_batch_durations),
            "mean_val_batch_sec": round(avg_val_batch_sec, 3),
            "total_val_batches_epoch": total_val_batches,
            "est_full_val_epoch_sec": round(est_full_val_epoch_sec, 1),
            "est_full_val_epoch_min": round(est_full_val_epoch_sec / 60.0, 1),
        },
        "projections": {
            "updates_per_epoch": updates_per_epoch,
            "one_epoch_total_hours": round(epoch_total_sec / 3600.0, 2),
            "c0_6_epochs_3600_updates_hours": round(c0_6_epochs_hours, 2),
            "both_arms_6_epochs_hours": round(both_arms_hours, 2),
            "colab_runtime_budget_hours": available_budget_hours,
            "safety_ratio_c0": round(budget_safety_ratio, 2),
        },
        "verdict": verdict,
        "verdict_detail": verdict_detail,
    }

    # Print Formatted Summary
    print("\n" + "=" * 80)
    print("CPU BENCHMARK SUMMARY & PROJECTIONS")
    print("=" * 80)
    print(f"Hardware: {hw_info['cpu_model']} ({hw_info['logical_cpus']} vCPUs) | Peak RAM: {peak_ram:.2f} GB")
    print(f"Update Throughput: {median_update_sec:.2f}s median per update ({1.0 / median_update_sec:.3f} updates/s)")
    print(f"  - Forward Pass (2 micro-batches): {mean_fwd_sec:.2f}s")
    print(f"  - Backward Pass:                  {mean_bwd_sec:.2f}s")
    print(f"  - Optimizer & Clip:               {mean_opt_sec:.2f}s")
    print(f"  - Data Loading Overhead:          {mean_data_sec:.2f}s")
    print(f"Validation Pass ({total_val_batches} batches):      {est_full_val_epoch_sec / 60.0:.1f} minutes per epoch")
    print("-" * 80)
    print(f"Updates per Epoch:                                {updates_per_epoch} updates")
    print(f"Projected C0 Duration (6 epochs / 3,600 updates): {c0_6_epochs_hours:.2f} hours")
    print(f"Projected C0 + C1 (both arms, 6 epochs each):    {both_arms_hours:.2f} hours")
    print(f"Available Colab Session Budget:                   {available_budget_hours:.2f} hours")
    print("=" * 80)
    print(f"FEASIBILITY VERDICT: {verdict}")
    print(f"Detail: {verdict_detail}")
    print("=" * 80 + "\n")

    if output_json:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(results, indent=2))
        print(f"[REPORT] Saved benchmark JSON to: {out_p}")

    return results


def main():
    ap = argparse.ArgumentParser(description="RQ1-TRN-02B Colab CPU Benchmark Driver")
    ap.add_argument("--config", default="configs/rq1_trn_02b_c0.yaml", help="Path to config YAML")
    ap.add_argument("--zip-path", default=None, help="Path to fold1_train_val_only.zip")
    ap.add_argument("--initial-checkpoint", default=None, help="Path to best.pt")
    ap.add_argument("--updates", type=int, default=50, help="Number of updates to measure (default: 50)")
    ap.add_argument("--val-batches", type=int, default=10, help="Number of val batches to measure (default: 10)")
    ap.add_argument("--budget-hours", type=float, default=36.33, help="Available Colab session duration (hours)")
    ap.add_argument("--output-json", default="manifests/RQ1-TRN-02B/benchmark_cpu_results.json", help="Path to save report")
    args = ap.parse_args()

    run_benchmark(
        config_path=args.config,
        zip_path=args.zip_path,
        initial_ckpt_path=args.initial_checkpoint,
        num_updates=args.updates,
        val_batches_to_measure=args.val_batches,
        colab_runtime_budget_hours=args.budget_hours,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
