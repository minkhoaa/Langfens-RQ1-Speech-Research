"""
audit_scripts/gpu_feasibility_bench.py
Independent GPU Feasibility Benchmarking for Full Pretrained Wav2Vec 2.0 MDD Architecture
Conforms to RQ1-TRN-01A-R Task E.
"""

import os
import gc
import json
import time
import torch
import torch.nn as nn
from langfens_prep.model import LangfensWav2Vec2MDD

def reset_cuda_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def run_feasibility_benchmark():
    if not torch.cuda.is_available():
        print("CUDA is not available on this system.")
        return

    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
    total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
    total_vram_mb = total_vram_bytes / (1024 ** 2)

    print(f"================================================================================")
    print(f"GPU Feasibility Benchmark: {device_name} ({total_vram_mb:.1f} MiB Physical VRAM)")
    print(f"================================================================================")

    # 1. Inspect Pretrained Model Parameters
    print("\n[1] Initializing Full Pretrained LangfensWav2Vec2MDD (facebook/wav2vec2-base-960h)...")
    t0 = time.time()
    model = LangfensWav2Vec2MDD(pretrained_model_name_or_path="facebook/wav2vec2-base-960h")
    load_time = time.time() - t0
    
    backbone_params = sum(p.numel() for p in model.wav2vec2.parameters())
    feature_extractor_params = sum(p.numel() for p in model.wav2vec2.feature_extractor.parameters())
    encoder_params = sum(p.numel() for p in model.wav2vec2.encoder.parameters())
    heads_params = (
        sum(p.numel() for p in model.ctc_head.parameters()) +
        sum(p.numel() for p in model.binary_head.parameters()) +
        sum(p.numel() for p in model.diagnosis_head.parameters()) +
        sum(p.numel() for p in model.cross_attention_module.parameters())
    )
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Backbone parameters:          {backbone_params:,} ({backbone_params/1e6:.2f}M)")
    print(f"  - Conv Feature Extractor:   {feature_extractor_params:,} ({feature_extractor_params/1e6:.2f}M)")
    print(f"  - 12-Layer Transformer:     {encoder_params:,} ({encoder_params/1e6:.2f}M)")
    print(f"Custom MDD Heads parameters:  {heads_params:,} ({heads_params/1e6:.2f}M)")
    print(f"Total Model parameters:       {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Model instantiation time:     {load_time:.2f}s")

    # 2. Transfer to GPU and measure baseline static memory
    reset_cuda_memory()
    model = model.to(device)
    static_allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
    static_reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
    print(f"Static Model VRAM Footprint:  {static_allocated_mb:.1f} MiB allocated ({static_reserved_mb:.1f} MiB reserved)")

    results = []

    # Test matrix:
    # Durations: 1.0s (16k), 2.0s (32k), 3.0s (48k), 5.0s (80k)
    # Batch sizes: 1, 2, 4
    # AMP fp16 vs FP32
    # Gradient checkpointing: True vs False
    # Feature extractor frozen: True vs False

    matrix = [
        # (batch_size, duration_sec, amp, grad_checkpoint, freeze_feat, label)
        (1, 1.0, True, True, True, "B=1, 1.0s, AMP fp16, GradCheckpoint, FrozenFeat"),
        (1, 2.0, True, True, True, "B=1, 2.0s, AMP fp16, GradCheckpoint, FrozenFeat"),
        (1, 3.0, True, True, True, "B=1, 3.0s, AMP fp16, GradCheckpoint, FrozenFeat"),
        (1, 5.0, True, True, True, "B=1, 5.0s, AMP fp16, GradCheckpoint, FrozenFeat"),
        (2, 2.0, True, True, True, "B=2, 2.0s, AMP fp16, GradCheckpoint, FrozenFeat"),
        (2, 3.0, True, True, True, "B=2, 3.0s, AMP fp16, GradCheckpoint, FrozenFeat"),
        (4, 2.0, True, True, True, "B=4, 2.0s, AMP fp16, GradCheckpoint, FrozenFeat"),
        (4, 3.0, True, True, True, "B=4, 3.0s, AMP fp16, GradCheckpoint, FrozenFeat"),
        # Comparisons:
        (1, 3.0, False, True, True, "B=1, 3.0s, FP32, GradCheckpoint, FrozenFeat"),
        (1, 3.0, True, False, True, "B=1, 3.0s, AMP fp16, NO GradCheckpoint, FrozenFeat"),
        (1, 3.0, True, True, False, "B=1, 3.0s, AMP fp16, GradCheckpoint, UnfrozenFeat"),
    ]

    print(f"\n[2] Executing Bounded GPU Training Step Matrix (Forward + Loss + Backward + Step)...")
    print(f"{'Config Description':<48} | {'Status':<6} | {'Peak Alloc':<10} | {'Peak Rsvd':<10} | {'Step Time':<9}")
    print("-" * 92)

    for bsz, dur, amp_enabled, grad_ckpt, freeze_feat, label in matrix:
        reset_cuda_memory()
        
        # Configure model state
        if freeze_feat:
            model.freeze_feature_encoder()
        else:
            for p in model.wav2vec2.feature_extractor.parameters():
                p.requires_grad = True

        if grad_ckpt:
            model.gradient_checkpointing_enable()
        else:
            model.wav2vec2.encoder.gradient_checkpointing = False

        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
        scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

        # Create synthetic input
        sample_len = int(dur * 16000)
        num_phones = max(2, int(dur * 8)) # ~8 phones/sec
        waveforms = torch.randn(bsz, sample_len, device=device)
        audio_lengths = torch.full((bsz,), sample_len, dtype=torch.long, device=device)
        canonical_indices = torch.randint(1, 40, (bsz, num_phones), device=device)
        target_binary = torch.randint(0, 2, (bsz, num_phones), device=device)
        target_ctc = torch.randint(1, 40, (bsz, num_phones), device=device)
        target_lengths = torch.full((bsz,), num_phones, dtype=torch.long, device=device)
        target_diagnosis = torch.randint(0, 39, (bsz, num_phones), device=device)

        status = "PASS"
        step_time_ms = 0.0
        peak_alloc_mb = 0.0
        peak_rsvd_mb = 0.0

        try:
            torch.cuda.reset_peak_memory_stats()
            t_start = time.time()
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=amp_enabled):
                out = model(
                    waveforms=waveforms,
                    audio_lengths=audio_lengths,
                    canonical_phone_indices=canonical_indices,
                    target_binary=target_binary,
                    target_ctc=target_ctc,
                    target_lengths=target_lengths,
                    target_diagnosis=target_diagnosis
                )
                loss = out.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()

            step_time_ms = (time.time() - t_start) * 1000
            peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            peak_rsvd_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

        except torch.cuda.OutOfMemoryError as e:
            status = "OOM"
            peak_alloc_mb = total_vram_mb
            peak_rsvd_mb = total_vram_mb
            torch.cuda.empty_cache()
        except Exception as e:
            status = f"ERR: {type(e).__name__}"

        results.append({
            "label": label,
            "batch_size": bsz,
            "duration_sec": dur,
            "amp": amp_enabled,
            "grad_checkpoint": grad_ckpt,
            "freeze_feat": freeze_feat,
            "status": status,
            "peak_alloc_mb": round(peak_alloc_mb, 1),
            "peak_rsvd_mb": round(peak_rsvd_mb, 1),
            "step_time_ms": round(step_time_ms, 1)
        })

        alloc_str = f"{peak_alloc_mb:.1f} MiB" if status == "PASS" else "N/A"
        rsvd_str = f"{peak_rsvd_mb:.1f} MiB" if status == "PASS" else "N/A"
        time_str = f"{step_time_ms:.1f} ms" if status == "PASS" else "N/A"
        print(f"{label:<48} | {status:<6} | {alloc_str:<10} | {rsvd_str:<10} | {time_str:<9}")

    # Output json summary
    os.makedirs("manifests/RQ1-TRN-01A/reconciliation", exist_ok=True)
    summary_path = "manifests/RQ1-TRN-01A/reconciliation/gpu_feasibility_measurements.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "device_name": device_name,
            "total_physical_vram_mb": round(total_vram_mb, 1),
            "static_allocated_mb": round(static_allocated_mb, 1),
            "static_reserved_mb": round(static_reserved_mb, 1),
            "backbone_parameters": backbone_params,
            "total_parameters": total_params,
            "matrix_results": results
        }, f, indent=2)
    print(f"\nSaved raw GPU feasibility measurements to {summary_path}")

if __name__ == "__main__":
    run_feasibility_benchmark()
