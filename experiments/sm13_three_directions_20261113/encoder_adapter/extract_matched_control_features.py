#!/usr/bin/env python3
"""Materialize frozen layer 5–9 features from the adapter's exact fp16 prefix."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2Model

ROOT = Path("/home/khoa/Projects/New-Method-Research")
HERE = ROOT / "experiments/sm13_three_directions_20261113/encoder_adapter"
OUT = HERE / "matched_control_features"
PREFIX = HERE / "prefix_l3_frames.f16"
OFFSETS = HERE / "prefix_l3_offsets.npy"
ALIGN = ROOT / "experiments/rq1_encoder_bench_01/features/alignment_centers.npz"
MODEL_ID = "facebook/wav2vec2-base-960h"
sys.path.insert(0, str(ROOT / "experiments/sm13_three_directions_20261113/encoder_adapter"))
from run_encoder_adapter import DatasetManager, get_adapted_layers  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    torch.set_num_threads(2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    dm = DatasetManager()
    manifest = ROOT / "data/derived/l2arctic/manifests/train.jsonl"
    recs = [json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    if [r["utterance_id"] for r in recs] != dm.order:
        raise AssertionError("manifest order differs from DatasetManager")
    align = np.load(ALIGN)
    offsets = np.load(OFFSETS)
    prefix = np.memmap(PREFIX, mode="r", dtype=np.float16).reshape(-1, 768)
    target_path = OUT / "target_l5_9_prefix_f16.npy"
    gap_path = OUT / "gap_l5_9_prefix_f16.npy"
    target = np.lib.format.open_memmap(target_path, mode="w+", dtype=np.float32,
                                       shape=(len(dm.yt), 5, 2304))
    gap = np.lib.format.open_memmap(gap_path, mode="w+", dtype=np.float32,
                                    shape=(len(dm.yg), 5, 1536))
    model = Wav2Vec2Model.from_pretrained(MODEL_ID).to(device).eval()
    t0 = time.perf_counter(); cursor_t = cursor_g = 0
    for i, rec in enumerate(recs):
        uid = rec["utterance_id"]
        a, b = offsets[i]
        h0 = torch.as_tensor(np.asarray(prefix[a:b], dtype=np.float32).copy(), device=device)
        centers = np.asarray(align[f"centers__{uid}"]); nframes = int(align[f"nframes__{uid}"])
        with torch.inference_mode():
            t_feat, g_feat = get_adapted_layers(model, h0, centers, nframes, with_grad=False)
        n_t = len(centers); n_g = n_t + 1
        target[cursor_t:cursor_t+n_t] = t_feat.cpu().numpy()
        gap[cursor_g:cursor_g+n_g] = g_feat.cpu().numpy()
        cursor_t += n_t; cursor_g += n_g
        if (i + 1) % 300 == 0 or i + 1 == len(recs):
            if device.type == "cuda": torch.cuda.synchronize(device)
            print(f"utterances={i+1}/{len(recs)} targets={cursor_t}/{len(dm.yt)} "
                  f"gaps={cursor_g}/{len(dm.yg)} elapsed_s={time.perf_counter()-t0:.1f}",
                  flush=True)
    target.flush(); gap.flush()
    if cursor_t != len(dm.yt) or cursor_g != len(dm.yg):
        raise AssertionError(f"feature coverage mismatch t={cursor_t} g={cursor_g}")
    del model, target, gap
    if device.type == "cuda": torch.cuda.empty_cache()
    receipt = {
        "status": "COMPLETE", "model_id": MODEL_ID, "device": str(device),
        "manifest": str(manifest), "alignment": str(ALIGN),
        "input_prefix": str(PREFIX), "prefix_dtype": "float16",
        "blocks": [4,5,6,7,8], "pooled_hidden_state_indices": [5,6,7,8,9],
        "target_shape": [len(dm.yt),5,2304], "gap_shape": [len(dm.yg),5,1536],
        "files": {str(p.name): {"bytes":p.stat().st_size,"sha256":sha256(p)}
                  for p in (target_path,gap_path)},
        "elapsed_seconds": time.perf_counter()-t0,
        "purpose": "Exact frozen control feature parity with LoRA candidate fp16 cached-prefix path; no labels used during extraction",
    }
    (OUT / "receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2),flush=True)


if __name__ == "__main__": main()
