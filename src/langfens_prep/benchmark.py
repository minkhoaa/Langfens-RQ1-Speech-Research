import io
import json
import os
import platform
import time
from typing import Dict, Any, List

import numpy as np
import psutil
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader

from langfens_prep.audio_loader import (
    load_l2arctic_audio_from_zip,
    get_l2arctic_reader,
    AudioStagingManager
)
from langfens_prep.dataset import L2ArcticMDDDataset
from langfens_prep.collate import LangfensMDDCollate

def get_system_environment_info() -> Dict[str, Any]:
    mem = psutil.virtual_memory()
    return {
        "os_platform": platform.platform(),
        "cpu_model": platform.processor() or "x86_64",
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "ram_total_gb": round(mem.total / (1024**3), 2),
        "ram_available_gb": round(mem.available / (1024**3), 2),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchaudio_version": torchaudio.__version__
    }

def benchmark_wav_access(zip_path: str, speaker: str = "TLV", num_utts: int = 50) -> Dict[str, Any]:
    """Measures direct archive extraction latency vs. local staging latency."""
    reader = get_l2arctic_reader(zip_path)
    szf = reader.get_speaker_zf(speaker)
    wav_names = [n for n in szf.namelist() if n.startswith(f"{speaker}/wav/") and n.endswith(".wav")][:num_utts]
    
    # Warm-up
    for _ in range(5):
        _ = szf.read(wav_names[0])
        
    direct_latencies_ms = []
    for name in wav_names:
        t0 = time.perf_counter()
        _ = szf.read(name)
        direct_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        
    # Stage files locally
    staging = AudioStagingManager()
    staged_map = staging.stage_l2arctic_utterances(
        zip_path, {speaker: [os.path.basename(n).replace(".wav", "") for n in wav_names]}
    )
    
    # Warm-up staged
    first_path = next(iter(staged_map.values()))
    for _ in range(5):
        with open(first_path, "rb") as f:
            _ = f.read()
            
    staged_latencies_ms = []
    for path in staged_map.values():
        t0 = time.perf_counter()
        with open(path, "rb") as f:
            _ = f.read()
        staged_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        
    staging.cleanup()
    
    return {
        "num_samples": len(wav_names),
        "direct_archive_median_ms": round(float(np.median(direct_latencies_ms)), 4),
        "direct_archive_p95_ms": round(float(np.percentile(direct_latencies_ms, 95)), 4),
        "staged_nvme_median_ms": round(float(np.median(staged_latencies_ms)), 4),
        "staged_nvme_p95_ms": round(float(np.percentile(staged_latencies_ms, 95)), 4),
        "speedup_ratio": round(float(np.median(direct_latencies_ms) / max(np.median(staged_latencies_ms), 1e-6)), 2)
    }

def benchmark_resampling(zip_path: str, speaker: str = "TLV", num_samples: int = 50) -> Dict[str, Any]:
    """Measures 44.1 kHz -> 16 kHz bandlimited resampling latency on CPU."""
    reader = get_l2arctic_reader(zip_path)
    szf = reader.get_speaker_zf(speaker)
    wav_names = [n for n in szf.namelist() if n.startswith(f"{speaker}/wav/") and n.endswith(".wav")][:num_samples]
    
    waveforms = []
    for name in wav_names:
        data, sr = sf.read(io.BytesIO(szf.read(name)), dtype="float32")
        waveforms.append(torch.from_numpy(data))
        
    # Warm-up
    for _ in range(5):
        _ = torchaudio.functional.resample(waveforms[0], orig_freq=44100, new_freq=16000)
        
    resample_latencies_ms = []
    for wf in waveforms:
        t0 = time.perf_counter()
        _ = torchaudio.functional.resample(wf, orig_freq=44100, new_freq=16000)
        resample_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        
    return {
        "num_samples": len(waveforms),
        "median_latency_ms": round(float(np.median(resample_latencies_ms)), 4),
        "p95_latency_ms": round(float(np.percentile(resample_latencies_ms, 95)), 4),
        "mean_latency_ms": round(float(np.mean(resample_latencies_ms)), 4)
    }

def benchmark_collation(batch_sizes: List[int] = [4, 8, 16], num_repeats: int = 30) -> Dict[str, Any]:
    """Measures variable-length batch padding and collation latency."""
    collate_fn = LangfensMDDCollate()
    results = {}
    
    for bs in batch_sizes:
        synthetic_batch = []
        for i in range(bs):
            seq_len = 16000 * np.random.randint(1, 5) # 1 to 5 seconds
            phone_len = np.random.randint(5, 25)
            synthetic_batch.append({
                "record_id": f"rec_{i}",
                "audio_waveform_16k": torch.randn(seq_len, dtype=torch.float32),
                "binary_labels": torch.randint(0, 2, (phone_len,), dtype=torch.long),
                "ctc_target_indices": torch.randint(1, 40, (phone_len,), dtype=torch.long)
            })
            
        # Warm-up
        for _ in range(5):
            _ = collate_fn(synthetic_batch)
            
        latencies_ms = []
        for _ in range(num_repeats):
            t0 = time.perf_counter()
            _ = collate_fn(synthetic_batch)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            
        results[f"batch_size_{bs}"] = {
            "median_latency_ms": round(float(np.median(latencies_ms)), 4),
            "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 4)
        }
    return results

def benchmark_dataloader_throughput(
    manifest_path: str = "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json",
    worker_counts: List[int] = [0, 2, 4],
    samples_per_run: int = 100,
    batch_size: int = 8
) -> Dict[str, Any]:
    """Measures real DataLoader throughput across different worker configurations."""
    results = {}
    collate_fn = LangfensMDDCollate()
    process = psutil.Process()
    
    for workers in worker_counts:
        dataset = L2ArcticMDDDataset(
            split_manifest_path=manifest_path,
            fold=1,
            split="train",
            max_samples=samples_per_run
        )
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            collate_fn=collate_fn
        )
        
        # Warm-up: 1 batch
        try:
            it = iter(loader)
            _ = next(it)
        except Exception:
            pass
            
        t0 = time.perf_counter()
        mem_before_mb = process.memory_info().rss / (1024**2)
        total_items = 0
        
        for batch in loader:
            total_items += batch["net_input"]["source"].shape[0]
            
        elapsed_sec = time.perf_counter() - t0
        mem_after_mb = process.memory_info().rss / (1024**2)
        
        throughput_utts_sec = total_items / elapsed_sec if elapsed_sec > 0 else 0.0
        median_time_per_utt_ms = (elapsed_sec / total_items) * 1000.0 if total_items > 0 else 0.0
        
        results[f"workers_{workers}"] = {
            "num_workers": workers,
            "total_utterances": total_items,
            "elapsed_seconds": round(elapsed_sec, 3),
            "throughput_utts_per_sec": round(throughput_utts_sec, 2),
            "median_time_per_utt_ms": round(median_time_per_utt_ms, 2),
            "memory_rss_mb": round(mem_after_mb, 2)
        }
        
    return results

def run_cpu_benchmarks() -> Dict[str, Any]:
    print("=== LANGFENS RQ1-PREP-01: CPU Micro-Benchmarking ===")
    zip_path = "datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip"
    
    env_info = get_system_environment_info()
    print(f"-> Host CPU: {env_info['cpu_model']} ({env_info['cpu_logical_cores']} vCPUs), RAM: {env_info['ram_available_gb']} GB available")
    
    print("-> 1. Measuring WAV access latency (Direct archive vs Local staging)...")
    wav_bench = benchmark_wav_access(zip_path, speaker="TLV", num_utts=50)
    print(f"   Direct Archive Median: {wav_bench['direct_archive_median_ms']} ms, Staged NVMe Median: {wav_bench['staged_nvme_median_ms']} ms")
    
    print("-> 2. Measuring 44.1 kHz -> 16 kHz sinc resampling latency...")
    resample_bench = benchmark_resampling(zip_path, speaker="TLV", num_samples=50)
    print(f"   Resampling Median: {resample_bench['median_latency_ms']} ms, p95: {resample_bench['p95_latency_ms']} ms")
    
    print("-> 3. Measuring variable-length batch collation latency...")
    collate_bench = benchmark_collation()
    print(f"   Collation (Batch Size 8) Median: {collate_bench['batch_size_8']['median_latency_ms']} ms")
    
    print("-> 4. Measuring DataLoader throughput across workers [0, 2, 4]...")
    loader_bench = benchmark_dataloader_throughput(worker_counts=[0, 2, 4], samples_per_run=100, batch_size=8)
    for k, v in loader_bench.items():
        print(f"   {k}: {v['throughput_utts_per_sec']} utts/sec ({v['median_time_per_utt_ms']} ms/utt)")
        
    return {
        "environment": env_info,
        "wav_access_benchmark": wav_bench,
        "resampling_benchmark": resample_bench,
        "collation_benchmark": collate_bench,
        "dataloader_throughput": loader_bench
    }

if __name__ == "__main__":
    results = run_cpu_benchmarks()
    with open("manifests/RQ1-PREP-01/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Benchmark complete. Results saved to manifests/RQ1-PREP-01/benchmark_results.json")
