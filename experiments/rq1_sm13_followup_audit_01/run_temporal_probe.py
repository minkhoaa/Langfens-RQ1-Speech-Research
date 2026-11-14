#!/usr/bin/env python3
"""Fixed TRAIN-only static-vs-temporal feature ablation for vowel/stop errors."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from itertools import groupby
from pathlib import Path

import numpy as np
import soundfile as sf
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

import run_audit as audit


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FEATURE = HERE / "TEMPORAL_7X32MS_FEATURES.npy"
GROUPS = (("ABA", "ASI", "BWC"), ("EBVS", "HJK", "HKK"),
          ("HQTV", "LXC", "NJS"), ("PNV", "SKA", "SVBI"))
OFFSETS_S = np.array((-.12, -.08, -.04, 0, .04, .08, .12), dtype=np.float64)
SR = 16000
NFFT = 512
N_BANDS = 16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mel_bank() -> np.ndarray:
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    mels = 2595 * np.log10(1 + freqs / 700)
    edges = np.linspace(mels[0], mels[-1], N_BANDS + 2)
    bank = []
    for left, center, right in zip(edges[:-2], edges[1:-1], edges[2:]):
        band = np.maximum(0, np.minimum((mels - left) / (center - left),
                                        (right - mels) / (right - center)))
        bank.append(band / max(band.sum(), 1e-12))
    return np.asarray(bank, dtype=np.float32)


def extract(d: dict) -> np.ndarray:
    rows = d["rows"]
    feats = np.empty((len(rows), 222), dtype=np.float32)
    bank = mel_bank()
    hann = np.hanning(NFFT).astype(np.float32)
    local = np.arange(NFFT, dtype=np.int64) - NFFT // 2
    offset = 0
    t0 = time.perf_counter()
    for n, (_, group) in enumerate(groupby(rows, key=lambda r: r["utterance"]), 1):
        events = list(group)
        audio, sr = sf.read(events[0]["audio"], dtype="float32")
        if sr != SR or audio.ndim != 1 or len(audio) < NFFT:
            raise ValueError(f"bad source WAV: {events[0]['audio']}")
        center_s = np.asarray([e["aligned_center_s"] for e in events], dtype=np.float64)
        center_ix = np.rint((center_s[:, None] + OFFSETS_S[None, :]) * SR).astype(np.int64)
        ix = np.clip(center_ix[:, :, None] + local[None, None, :], 0, len(audio) - 1)
        windows = audio[ix]
        power = np.abs(np.fft.rfft(windows * hann, axis=-1)) ** 2
        logmel = np.log1p(power.reshape(-1, NFFT // 2 + 1) @ bank.T).reshape(len(events), 7, N_BANDS)
        delta = np.diff(logmel, axis=1)
        log_rms = np.log(np.maximum(np.mean(windows * windows, axis=-1), 1e-8))
        zcr = np.mean(windows[:, :, 1:] * windows[:, :, :-1] < 0, axis=-1)
        feat = np.concatenate((logmel.reshape(len(events), -1), delta.reshape(len(events), -1),
                               log_rms, zcr), axis=1).astype(np.float32)
        assert feat.shape == (len(events), 222)
        feats[offset:offset + len(events)] = feat
        offset += len(events)
        if n % 300 == 0:
            print(f"extracted {n}/1800 utterances in {time.perf_counter()-t0:.1f}s", flush=True)
    if offset != 59562 or not np.isfinite(feats).all():
        raise ValueError(f"temporal feature coverage invalid: {offset}")
    np.save(FEATURE, feats)
    print(f"saved {FEATURE}, sha256={sha256(FEATURE)}", flush=True)
    return feats


def model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=50, l2_regularization=1.0,
        early_stopping=False, random_state=seed,
    )


def metric(y: np.ndarray, s: np.ndarray) -> dict:
    return {"n": len(y), "support": int(y.sum()),
            "ap": float(average_precision_score(y, s)),
            "auroc": float(roc_auc_score(y, s))}


def train(d: dict, temporal: np.ndarray) -> None:
    t0 = time.perf_counter()
    with np.load(ROOT / "data/derived/l2arctic/features/train_features.npz") as f:
        static = f["target_X"].astype(np.float32)
        if not np.array_equal(f["target_Y"], d["y"]):
            raise ValueError("static feature labels do not match TRAIN manifest")
        meta = f["target_meta"]
        if not all(uid == row["utterance"] and int(slot) == row["slot"]
                   for (uid, slot), row in zip(meta, d["rows"])):
            raise ValueError("static feature order does not match TRAIN manifest")
    if static.shape != (59562, 51) or temporal.shape != (59562, 222):
        raise ValueError("feature dimensions mismatch preregistration")
    matrix = {"static": static, "temporal": np.concatenate((static, temporal), axis=1)}
    speaker = d["speaker"]
    manner = d["manner"]
    y = d["y"] == 1
    scores = {name: d["s"].copy() for name in matrix}
    fold_records = []
    for fold, held_speakers in enumerate(GROUPS):
        held_group = np.isin(speaker, held_speakers)
        for target_manner in ("Vowel", "Stop"):
            fit = np.flatnonzero(~held_group & (manner == target_manner))
            held = np.flatnonzero(held_group & (manner == target_manner))
            positive = int(y[fit].sum())
            negative = len(fit) - positive
            if positive == 0 or negative == 0:
                raise ValueError(f"empty class: fold={fold}, manner={target_manner}")
            weight = np.where(y[fit], np.sqrt(negative / positive), 1.0).astype(np.float32)
            for name, X in matrix.items():
                start = time.perf_counter()
                clf = model(101 + fold)
                with threadpool_limits(limits=4):
                    clf.fit(X[fit], y[fit].astype(int), sample_weight=weight)
                    s = clf.predict_proba(X[held])[:, 1]
                scores[name][held] = s
                row = {"fold": fold, "held_speakers": list(held_speakers),
                       "manner": target_manner, "features": name,
                       "fit_n": len(fit), "fit_positive": positive,
                       "held_n": len(held), "held_positive": int(y[held].sum()),
                       "held_ap": float(average_precision_score(y[held], s)),
                       "seconds": time.perf_counter() - start}
                fold_records.append(row)
                print(f"fold={fold} {target_manner} {name} AP={row['held_ap']:.4f} "
                      f"time={row['seconds']:.1f}s", flush=True)
    if not all(np.isfinite(s).all() for s in scores.values()):
        raise ValueError("OOF scores nonfinite")
    np.savez_compressed(HERE / "TEMPORAL_PROBE_OOF.npz",
                        target_keys=np.asarray([r["key"] for r in d["rows"]]),
                        gold_target=d["y"], baseline=d["s"],
                        static=scores["static"], temporal=scores["temporal"])
    result = {"protocol": "TEMPORAL_PROBE_PREREGISTRATION.md", "exploratory": True,
              "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                              "sklearn": sklearn.__version__, "os": platform.platform(),
                              "cpu": platform.processor()},
              "feature_shape": list(temporal.shape), "feature_sha256": sha256(FEATURE),
              "static_shape": list(static.shape), "fold_records": fold_records,
              "seconds": time.perf_counter() - t0, "metrics": {}, "frontier": {}}
    all_speakers = set(speaker)
    for target_manner in ("Vowel", "Stop"):
        mask = manner == target_manner
        result["metrics"][target_manner] = {
            "sm13": metric(y[mask], d["s"][mask]),
            "static": metric(y[mask], scores["static"][mask]),
            "temporal": metric(y[mask], scores["temporal"][mask]),
        }
        result["frontier"][target_manner] = {}
        for name, s in (("sm13", d["s"]), *scores.items()):
            probe = dict(d)
            probe["s"] = s
            result["frontier"][target_manner][name] = {
                "fwr_only": audit.select(probe, all_speakers, target_manner, None),
                "fwr_p50": audit.select(probe, all_speakers, target_manner, 0.5),
            }
    (HERE / "TEMPORAL_PROBE_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"metrics": result["metrics"], "frontier": result["frontier"]}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    if args.extract_only and args.train_only:
        parser.error("choose one mode")
    d = audit.load_data()
    if args.train_only:
        temporal = np.load(FEATURE)
    else:
        temporal = extract(d)
    if not args.extract_only:
        train(d, temporal)


if __name__ == "__main__":
    main()
