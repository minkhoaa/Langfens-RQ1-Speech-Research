#!/usr/bin/env python3
"""Extract TRAIN-only SM13 layer-6 windows and adapter prefix activations."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2Model

ROOT = Path("/home/khoa/Projects/New-Method-Research")
OUT = ROOT / "experiments/sm13_three_directions_20261113"
ADAPTER_OUT = OUT / "encoder_adapter"
SHARED_OUT = OUT / "shared"
MANIFEST = ROOT / "data/derived/l2arctic/manifests/train.jsonl"
ALIGN = ROOT / "experiments/rq1_encoder_bench_01/features/alignment_centers.npz"
MODEL_ID = "facebook/wav2vec2-base-960h"

sys.path.insert(0, str(ROOT / "experiments/rq1_encoder_bench_01/common"))
from dataset_util import DatasetManager  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pooled_windows(hidden: np.ndarray, centers: np.ndarray, nframes: int):
    hlen = len(hidden)
    scaled = np.rint(centers * (hlen / max(1, nframes))).astype(np.int64)
    ix = np.clip(scaled[:, None] + np.arange(-8, 9)[None, :], 0, hlen - 1)
    chunks = hidden[ix]
    left = np.zeros((len(centers) + 1, 5, 768), dtype=np.float16)
    right = np.zeros_like(left)
    if len(centers):
        left[1:] = chunks[:, 11:16].astype(np.float16)
        right[:-1] = chunks[:, 1:6].astype(np.float16)
    return chunks.astype(np.float16), np.concatenate([left, right], axis=1)


def main():
    torch.set_num_threads(2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    ADAPTER_OUT.mkdir(parents=True, exist_ok=True)
    SHARED_OUT.mkdir(parents=True, exist_ok=True)
    dm = DatasetManager()
    recs = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    assert [r["utterance_id"] for r in recs] == dm.order
    align = np.load(ALIGN)
    nt, ng = len(dm.yt), len(dm.yg)
    target_path = SHARED_OUT / "target_w2v2_960h_l6_windows.npy"
    gap_path = SHARED_OUT / "gap_w2v2_960h_l6_windows.npy"
    prefix_path = ADAPTER_OUT / "prefix_l3_frames.f16"
    offsets_path = ADAPTER_OUT / "prefix_l3_offsets.npy"
    target_map = np.lib.format.open_memmap(target_path, mode="w+", dtype=np.float16,
                                           shape=(nt, 17, 768))
    gap_map = np.lib.format.open_memmap(gap_path, mode="w+", dtype=np.float16,
                                        shape=(ng, 10, 768))
    offsets = np.zeros((len(recs), 2), dtype=np.int64)
    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = Wav2Vec2Model.from_pretrained(MODEL_ID).to(device).eval()
    cursor_t = cursor_g = cursor_frames = 0
    t0 = time.perf_counter()
    with prefix_path.open("wb") as prefix_f, torch.inference_mode():
        for i, rec in enumerate(recs):
            uid = rec["utterance_id"]
            samples, sr = sf.read(rec["audio_path"], dtype="float32", always_2d=True)
            if sr != 16000:
                raise ValueError(f"{uid}: expected 16kHz, got {sr}")
            waveform = fe(samples.mean(axis=1), sampling_rate=sr,
                          return_tensors="pt")["input_values"].to(device)
            outputs = model(waveform, output_hidden_states=True)
            centers = np.asarray(align[f"centers__{uid}"])
            nframes = int(align[f"nframes__{uid}"])
            nphones = len(centers)
            # hidden_states[4] is the output of zero-based transformer block 3,
            # exactly the input required by zero-based block 4 below.
            state3 = outputs.hidden_states[4][0].float().cpu().numpy()
            state6 = outputs.hidden_states[6][0].float().cpu().numpy()
            if state3.shape != state6.shape:
                raise AssertionError(f"hidden-state length mismatch at {uid}")
            offsets[i] = (cursor_frames, cursor_frames + len(state3))
            prefix_f.write(state3.astype(np.float16, copy=False).tobytes(order="C"))
            cursor_frames += len(state3)
            target_win, gap_win = pooled_windows(state6, centers, nframes)
            target_map[cursor_t:cursor_t+nphones] = target_win
            gap_map[cursor_g:cursor_g+nphones+1] = gap_win
            cursor_t += nphones
            cursor_g += nphones + 1
            if (i + 1) % 100 == 0 or i + 1 == len(recs):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                print(f"utterances={i+1}/{len(recs)} targets={cursor_t}/{nt} "
                      f"gaps={cursor_g}/{ng} frames={cursor_frames} "
                      f"elapsed_s={time.perf_counter()-t0:.1f}", flush=True)
    target_map.flush(); gap_map.flush()
    np.save(offsets_path, offsets)
    assert cursor_t == nt and cursor_g == ng
    del model, target_map, gap_map
    if device.type == "cuda":
        torch.cuda.empty_cache()
    receipt = {
        "status": "COMPLETE", "device": str(device), "model_id": MODEL_ID,
        "feature_extractor": "AutoFeatureExtractor.from_pretrained(model_id), input 16kHz, default Wav2Vec2 normalization enabled",
        "manifest": str(MANIFEST), "manifest_sha256": sha256(MANIFEST),
        "alignment": str(ALIGN), "alignment_sha256": sha256(ALIGN),
        "ordered_utterances": len(recs), "target_windows_shape": [nt, 17, 768],
        "gap_windows_shape": [ng, 10, 768], "prefix_state": "hidden_states[4] (after zero-based transformer block 3; input to block 4)",
        "prefix_dtype": "float16", "prefix_total_frames": int(cursor_frames),
        "target_window_rule": "center scaled to hidden-state frames; offsets -8..+8 inclusive; edge clipped",
        "gap_window_rule": "previous center offsets 11:16 then next center offsets 1:6; missing boundary side zeros",
        "files": {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                  for p in (target_path, gap_path, prefix_path, offsets_path)},
        "elapsed_seconds": time.perf_counter() - t0,
    }
    (ADAPTER_OUT / "raw_cache_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
