#!/usr/bin/env python3
"""Reconstruct CharSiu Viterbi spans for a fixed TRAIN-OOF sample.

Uses cached CTC logits and the exact topology/tie ordering of
rq1_encoder_bench_01/features/precompute_alignment.py. No model inference.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer

import run_audit


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CACHE = ROOT / "experiments/evaluation_v3/closure/waveform_full/utterances"
TOK = Path.home() / ".cache/huggingface/hub/models--charsiu--tokenizer_en_cmu/snapshots/10507401aedf5e0aba164128535b49225ff95260"
ALIGN = ROOT / "experiments/rq1_encoder_bench_01/features/alignment_centers.npz"


def reconstruct(logits: np.ndarray, phones: list[str], vocab: dict[str, int]) -> tuple[list[np.ndarray], np.ndarray]:
    sil = vocab["[SIL]"]
    ids = [vocab.get(p, sil) for p in phones]
    lp = torch.log_softmax(torch.as_tensor(logits, dtype=torch.float64), -1).numpy()
    n, t = len(ids), lp.shape[0]
    labels = np.empty(2 * n, dtype=np.int64)
    sep = np.zeros(2 * n, bool)
    for i, p in enumerate(ids):
        labels[2 * i] = p
        labels[2 * i + 1] = sil
        sep[2 * i + 1] = True
    dp = np.full((t, 2 * n), -np.inf)
    back = np.full((t, 2 * n), -1, np.int32)
    dp[0, 0] = lp[0, labels[0]]
    if n:
        dp[0, 1] = lp[0, labels[1]]
    for f in range(1, t):
        for s in range(2 * n):
            opts = [(dp[f - 1, s], s)]
            if s:
                opts.append((dp[f - 1, s - 1], s - 1))
            if s > 1 and sep[s - 1]:
                opts.append((dp[f - 1, s - 2], s - 2))
            val, prev = max(opts, key=lambda x: x[0])
            dp[f, s] = val + lp[f, labels[s]]
            back[f, s] = prev
    s = int(np.argmax(dp[-1]))
    path = np.empty(t, np.int32)
    for f in range(t - 1, -1, -1):
        path[f] = s
        s = int(back[f, s]) if f else s
    spans = [np.flatnonzero(path == 2 * i) for i in range(n)]
    centers = np.asarray([float(ix.mean()) if ix.size else (i + 0.5) * t / max(1, n) - 0.5
                          for i, ix in enumerate(spans)], dtype=np.float32)
    return spans, centers


def sample_events(d: dict) -> list[int]:
    rng = np.random.default_rng(101)
    selected = []
    seen_utterances = set()
    for cat in ("Vowel", "Stop"):
        pool = np.flatnonzero((d["manner"] == cat) & (d["y"] == 1))
        shuffled = rng.permutation(pool).tolist()
        for i in run_audit.choose_diverse(shuffled, d, len(shuffled)):
            uid = d["rows"][i]["utterance"]
            if uid in seen_utterances:
                continue
            selected.append(i)
            seen_utterances.add(uid)
            if sum(d["manner"][j] == cat for j in selected) == 30:
                break
    assert len(selected) == 60
    return selected


def summary(rows: list[dict], cat: str) -> dict:
    x = [r for r in rows if r["manner"] == cat]
    start = np.asarray([r["abs_start_error_ms"] for r in x])
    end = np.asarray([r["abs_end_error_ms"] for r in x])
    return {"n": len(x), "median_abs_start_error_ms": float(np.median(start)),
            "median_abs_end_error_ms": float(np.median(end)),
            "p90_abs_start_error_ms": float(np.percentile(start, 90)),
            "p90_abs_end_error_ms": float(np.percentile(end, 90)),
            "either_boundary_over_50ms": int(np.sum((start > 50) | (end > 50))),
            "either_boundary_over_100ms": int(np.sum((start > 100) | (end > 100))),
            "empty_predicted_span": int(sum(r["empty_predicted_span"] for r in x))}


def main() -> None:
    d = run_audit.load_data()
    selected = sample_events(d)
    by_utt = defaultdict(list)
    for i, r in enumerate(d["rows"]):
        by_utt[r["utterance"]].append(i)
    tok = AutoTokenizer.from_pretrained(str(TOK), local_files_only=True)
    vocab = tok.get_vocab()
    output = []
    with np.load(ALIGN) as centers_cache:
        for i in selected:
            r = d["rows"][i]
            uid = r["utterance"]
            with np.load(CACHE / f"{uid}.npz") as z:
                phones = z["canonical_phones"].astype(str).tolist()
                logits = z["ctc_logits"]
            ordered = by_utt[uid]
            assert phones == [d["rows"][j]["phone"] for j in ordered]
            spans, centers = reconstruct(logits, phones, vocab)
            assert len(centers) == len(centers_cache[f"centers__{uid}"])
            assert np.allclose(centers, centers_cache[f"centers__{uid}"], atol=1e-4)
            ix = spans[r["slot"]]
            seconds_per_frame = sf.info(r["audio"]).duration / len(logits)
            if ix.size:
                pred_start = float(ix.min() * seconds_per_frame)
                pred_end = float((ix.max() + 1) * seconds_per_frame)
                abs_start = abs(pred_start - r["xmin"]) * 1000
                abs_end = abs(pred_end - r["xmax"]) * 1000
            else:
                pred_start = pred_end = None
                abs_start = abs_end = None
            output.append({"event_key": r["key"], "manner": d["manner"][i],
                           "speaker": r["speaker"], "target_phone": r["phone"],
                           "realized_phone": r["realized"], "neural_s_score": float(d["s"][i]),
                           "gold_start_s": r["xmin"], "gold_end_s": r["xmax"],
                           "pred_start_s": pred_start, "pred_end_s": pred_end,
                           "abs_start_error_ms": abs_start, "abs_end_error_ms": abs_end,
                           "empty_predicted_span": not bool(ix.size)})
    assert all(not r["empty_predicted_span"] for r in output)
    with (OUT / "BOUNDARY_PROBE_EVENTS.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    report = {"scope": "60 sampled TRAIN-OOF gold substitutions, 30 vowel + 30 stop, distinct utterances; seed 101; no test access",
              "source": "cached CTC logits; exact CharSiu Viterbi topology from precompute_alignment.py; center parity checked per utterance",
              "results": {cat: summary(output, cat) for cat in ("Vowel", "Stop")},
              "limits": "Sample estimates only. Frame-clock mapping uses utterance duration / CTC frames. Manual boundaries are not an independent second annotation."}
    (OUT / "BOUNDARY_PROBE.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
