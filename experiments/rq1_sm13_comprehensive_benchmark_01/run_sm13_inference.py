#!/usr/bin/env python3
"""
experiments/rq1_sm13_comprehensive_benchmark_01/run_sm13_inference.py

Phase 2 — Full-Corpus Inference Engine
Executes end-to-end SM13 Seed 101 inference across:
1. L2-ARCTIC DEV_REPLICATION (300 utterances, speakers TXHC & YKWK).
2. SpeechOcean762 (5,000 utterances across 250 speakers).

Implements:
- Checkpointed, restartable execution.
- SHA-256 validation of frozen assets.
- Immutable event-level scores saved to NPZ.
- Latency, RTF, throughput, and GPU memory profiling.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import logging
import math
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, Wav2Vec2ForCTC

ROOT = Path("/home/khoa/Projects/New-Method-Research")
sys.path.insert(0, str(ROOT))

from experiments.rq1_encoder_bench_01.features.precompute_alignment import align_centers

OUT_DIR = ROOT / "experiments/rq1_sm13_comprehensive_benchmark_01"
CKPT_DIR = OUT_DIR / "checkpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Frozen Asset Paths & Expected Checksums
CKPT_TARGET = ROOT / "experiments/rq1_sm13_formal_train_01/checkpoints/target_sm13_seed101_fulltrain.pt"
CKPT_GAP = ROOT / "experiments/rq1_sm13_formal_train_01/checkpoints/gap_sm13_seed101_fulltrain.pt"
CKPT_CTC = ROOT / "experiments/rq1_sm13_formal_train_01/checkpoints/ctc_registered_seed101.pt"
POLICY_BASELINE = ROOT / "experiments/rq1_sm13_dev_calibration_01/FROZEN_CALIBRATED_POLICY.json"
POLICY_ASYMMETRIC = ROOT / "FROZEN_ASYMMETRIC_POLICY.json"
CTC_LIB_PATH = ROOT / "experiments/rq1_method_bench_04/agent_a/ctc_batch.so"

HUB = Path.home() / ".cache/huggingface/hub"
CHARSIU_MODEL_PATH = HUB / "models--charsiu--en_w2v2_fc_10ms/snapshots/e9bf8dd314313fc57f6e4d0b5425bde4bbeac80f"
CHARSIU_TOK_PATH = HUB / "models--charsiu--tokenizer_en_cmu/snapshots/10507401aedf5e0aba164128535b49225ff95260"

EXPECTED_HASHES = {
    "target_checkpoint": (CKPT_TARGET, "551c5f6079109e8decd49d6fe92ed16ec394a41177b1b87af9e64c95618f6ba6"),
    "gap_checkpoint": (CKPT_GAP, "34de5bbfc2003176730bfdacbf23b1cfb54ffb861f42d8b0181295ed912b4f98"),
    "ctc_checkpoint": (CKPT_CTC, "63d3d6147db237a38ec673b5142ba4f4f29ac43991b6124aead745b2a05b015a"),
    "baseline_policy": (POLICY_BASELINE, "dc7da3bcbc28918fcf39bdb1beba4ab990219d4467e56cdabbaebb0ac35e88e1"),
    "asymmetric_policy": (POLICY_ASYMMETRIC, "e74ea31804f2d00d731aefffbfef2d613e6112d7965c4cbf60807b0bcee45023"),
}

PHONE_CLASSES = [
    "AA", "AE", "AH", "AO", "AW", "AX", "AY", "B", "CH", "D",
    "DH", "EH", "ER", "ERR", "EY", "F", "G", "HH", "IH", "IY",
    "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R",
    "S", "SH", "SPN", "T", "TH", "UH", "UW", "V", "W", "Y",
    "Z", "ZH"
]
PHONE_TO_ID = {p: i for i, p in enumerate(PHONE_CLASSES)}
BLANK_ID = 42
TIE_PREFIX = "LANGFENS-MB07-warning-tie-v1:"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def tie_hash(event_id: str) -> int:
    return int.from_bytes(hashlib.sha256((TIE_PREFIX + str(event_id)).encode()).digest()[:8], "big")


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("SM13Inference")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s | %(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def get_ctc_kernel():
    lib = ctypes.CDLL(str(CTC_LIB_PATH))
    f = lib.score_ctc_batch
    f.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double)
    ]
    f.restype = None
    return f


class ScalarMixingTargetHead(nn.Module):
    def __init__(self, num_layers: int = 13, feat_dim: int = 2304, ref_dim: int = 39):
        super().__init__()
        self.num_layers = num_layers
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.scale = nn.Parameter(torch.ones(1))
        self.net = nn.Sequential(
            nn.Linear(feat_dim + ref_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 3)
        )

    def forward(self, layer_feats: torch.Tensor, refs: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.layer_logits, dim=0)
        mixed = torch.einsum("l, bli -> bi", w, layer_feats) * self.scale
        x = torch.cat([mixed, refs], dim=-1)
        return self.net(x)

    @torch.inference_mode()
    def predict_proba(self, layer_feats: torch.Tensor, refs: torch.Tensor) -> torch.Tensor:
        logits = self.forward(layer_feats, refs)
        return torch.softmax(logits, dim=-1)


class ScalarMixingGapHead(nn.Module):
    def __init__(self, num_layers: int = 13, feat_dim: int = 1536, ref_dim: int = 78):
        super().__init__()
        self.num_layers = num_layers
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.scale = nn.Parameter(torch.ones(1))
        self.net = nn.Sequential(
            nn.Linear(feat_dim + ref_dim, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )

    def forward(self, layer_feats: torch.Tensor, refs: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.layer_logits, dim=0)
        mixed = torch.einsum("l, bli -> bi", w, layer_feats) * self.scale
        x = torch.cat([mixed, refs], dim=-1)
        return self.net(x)

    @torch.inference_mode()
    def predict_proba(self, layer_feats: torch.Tensor, refs: torch.Tensor) -> torch.Tensor:
        logits = self.forward(layer_feats, refs)
        return torch.softmax(logits, dim=-1)


class LinearCTC(nn.Module):
    def __init__(self, in_dim: int = 768, num_classes: int = 43):
        super().__init__()
        self.proj = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def compute_e4_likelihoods(
    log_probs: np.ndarray,
    canonical_phone_ids: List[int],
    kernel_fn
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = len(canonical_phone_ids)
    if N == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    T = log_probs.shape[0]
    lp_contig = np.ascontiguousarray(log_probs, dtype=np.float64)

    sequences = [list(canonical_phone_ids)]
    sub_map = []
    om_map = []

    for i, phone_id in enumerate(canonical_phone_ids):
        sub_cands = [(phone_id + k) % 42 for k in range(1, 5)]
        sub_indices = []
        for cand in sub_cands:
            hyp = list(canonical_phone_ids)
            hyp[i] = cand
            sub_indices.append(len(sequences))
            sequences.append(hyp)
        sub_map.append(sub_indices)

        hyp_del = canonical_phone_ids[:i] + canonical_phone_ids[i + 1:]
        om_map.append(len(sequences))
        sequences.append(hyp_del)

    flat_seqs = []
    offsets = [0]
    for seq in sequences:
        flat_seqs.extend(seq)
        offsets.append(len(flat_seqs))

    flat_arr = np.array(flat_seqs, dtype=np.int32)
    offsets_arr = np.array(offsets, dtype=np.int32)
    scores_out = np.zeros(len(sequences), dtype=np.float64)

    kernel_fn(
        lp_contig.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(T),
        ctypes.c_int(43),
        ctypes.c_int(BLANK_ID),
        flat_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        offsets_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(len(sequences)),
        scores_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )

    can_score = scores_out[0]
    sllr_vals = np.zeros(N, dtype=np.float64)
    ollr_vals = np.zeros(N, dtype=np.float64)
    e4_simplex = np.zeros((N, 3), dtype=np.float32)

    for i in range(N):
        sub_scores = [scores_out[idx] for idx in sub_map[i]]
        max_sub = np.mean(sub_scores)
        sllr = max_sub - can_score
        del_score = scores_out[om_map[i]]
        ollr = del_score - can_score

        sllr_vals[i] = sllr
        ollr_vals[i] = ollr

        logits = np.array([0.0, sllr, ollr], dtype=np.float64)
        exp_l = np.exp(logits - np.max(logits))
        e4_simplex[i] = (exp_l / np.sum(exp_l)).astype(np.float32)

    return e4_simplex, sllr_vals, ollr_vals


def extract_features_from_hidden_states(
    hidden_states: List[torch.Tensor],
    centers: np.ndarray,
    n_frames: int,
    radius: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    N = len(centers)
    target_feats_13 = []
    gap_feats_13 = []

    for l in range(13):
        hidden = hidden_states[l][0].float().detach().cpu().numpy()
        T = len(hidden)
        scaled_c = np.rint(centers * (T / max(1, n_frames))).astype(int)
        offsets = np.arange(-radius, radius + 1)
        ix = np.clip(scaled_c[:, None] + offsets[None, :], 0, T - 1)
        chunks = hidden[ix]

        c = chunks[:, 3:14, :]
        t_feat = np.concatenate([c.mean(axis=1), c.std(axis=1, ddof=0), c[:, -1] - c[:, 0]], axis=1)
        target_feats_13.append(t_feat)

        left = np.zeros((N + 1, 768), dtype=np.float32)
        right = np.zeros((N + 1, 768), dtype=np.float32)
        if N > 0:
            left[1:] = chunks[:, 11:16, :].mean(axis=1)
            right[:-1] = chunks[:, 1:6, :].mean(axis=1)
        g_feat = np.concatenate([left, right], axis=1)
        gap_feats_13.append(g_feat)

    target_layer_feats = np.stack(target_feats_13, axis=1)
    gap_layer_feats = np.stack(gap_feats_13, axis=1)
    return target_layer_feats, gap_layer_feats


def apply_class_threshold(event_ids: np.ndarray, scores: np.ndarray, threshold: float, cutoff: int) -> np.ndarray:
    out = np.zeros(len(scores), dtype=bool)
    for i, (eid, s) in enumerate(zip(event_ids, scores)):
        if s > threshold:
            out[i] = True
        elif s < threshold:
            out[i] = False
        else:
            out[i] = (tie_hash(eid) >= cutoff)
    return out


def resolve_so_conflict(warn_s: np.ndarray, warn_o: np.ndarray, p_s: np.ndarray, p_o: np.ndarray,
                        th_s: float, th_o: float, event_ids: np.ndarray) -> np.ndarray:
    n = len(warn_s)
    dec = np.zeros(n, dtype=np.int64)
    only_s = warn_s & (~warn_o)
    only_o = warn_o & (~warn_s)
    conflict = warn_s & warn_o

    dec[only_s] = 1  # SUBSTITUTION
    dec[only_o] = 2  # OMISSION

    if np.any(conflict):
        for idx in np.flatnonzero(conflict):
            m_s = p_s[idx] - th_s
            m_o = p_o[idx] - th_o
            if m_s > m_o:
                dec[idx] = 1
            elif m_o > m_s:
                dec[idx] = 2
            else:
                dec[idx] = 1 if (tie_hash(event_ids[idx]) % 2 == 1) else 2
    return dec


def run_dev_replication_inference(
    charsiu_model, charsiu_tok, w2v2_model, model_t, model_g, ctc, ctc_kernel,
    p2i_39, tau_T_base, tau_G_base, tau_S, cut_S, tau_O, cut_O, tau_I, cut_I,
    device, logger
):
    save_path = OUT_DIR / "predictions_sm13_seed101_dev_replication.npz"
    if save_path.exists():
        logger.info(f"dev_replication predictions already exist at {save_path}, skipping inference.")
        return

    logger.info("Executing SM13 Seed 101 inference on L2-ARCTIC DEV_REPLICATION (300 utterances)...")
    manifest_path = ROOT / "data/derived/l2arctic/manifests/dev_replication.jsonl"
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    is_cuda = (device.type == "cuda")
    labels_map = ("CORRECT", "SUBSTITUTION", "OMISSION")

    all_target_keys, all_gap_keys = [], []
    all_target_speakers, all_gap_speakers = [], []
    all_gold_target, all_gold_gap = [], []
    all_neural_t_probs, all_fixed_t_probs = [], []
    all_gap_probs, all_e4_probs = [], []
    all_sllr, all_ollr = [], []
    per_utt_latencies = []
    durations = []

    t_start = time.perf_counter()
    for idx, r in enumerate(tqdm(records, desc="DEV_REPLICATION")):
        t0 = time.perf_counter()
        uid = r["utterance_id"]
        spk = r["speaker_id"]
        audio_full_path = ROOT / r["audio_path"]
        canonical_phones = [tp["target_phone"] for tp in r["target_phones"]]
        N = len(canonical_phones)

        samples, sr = sf.read(str(audio_full_path), dtype="float32", always_2d=True)
        samples = samples.mean(axis=1)
        durations.append(len(samples) / sr)
        wav_t = torch.as_tensor(samples, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.inference_mode():
            char_logits = charsiu_model(wav_t).logits[0]
            if is_cuda: torch.cuda.synchronize()
            char_logits_np = char_logits.cpu().numpy()

        centers, nframes = align_centers(char_logits_np, canonical_phones, charsiu_tok)

        with torch.inference_mode():
            out = w2v2_model(wav_t, output_hidden_states=True)
            if is_cuda: torch.cuda.synchronize()

        t_feat, g_feat = extract_features_from_hidden_states(out.hidden_states, centers, nframes)

        ref_t_np = np.zeros((N, 39), dtype=np.float32)
        for i, p in enumerate(canonical_phones):
            if p in p2i_39: ref_t_np[i, p2i_39[p]] = 1.0

        ref_g_np = np.zeros((N + 1, 78), dtype=np.float32)
        for i in range(N + 1):
            left = canonical_phones[i - 1] if i > 0 else "[SIL]"
            right = canonical_phones[i] if i < N else "[SIL]"
            if left in p2i_39: ref_g_np[i, p2i_39[left]] = 1.0
            if right in p2i_39: ref_g_np[i, 39 + p2i_39[right]] = 1.0

        bx_t = torch.as_tensor(t_feat, dtype=torch.float32, device=device)
        br_t = torch.as_tensor(ref_t_np, dtype=torch.float32, device=device)
        with torch.inference_mode():
            neural_t_np = model_t.predict_proba(bx_t, br_t).cpu().numpy()

        bx_g = torch.as_tensor(g_feat, dtype=torch.float32, device=device)
        br_g = torch.as_tensor(ref_g_np, dtype=torch.float32, device=device)
        with torch.inference_mode():
            gap_p_np = model_g.predict_proba(bx_g, br_g)[:, 1].cpu().numpy()

        with torch.inference_mode():
            ctc_logits = ctc(out.hidden_states[6][0])
            ctc_logp_np = F.log_softmax(ctc_logits, dim=-1).cpu().numpy()

        can_ctc_ids = [PHONE_TO_ID.get(p, 0) for p in canonical_phones]
        e4_simplex, sllr_vals, ollr_vals = compute_e4_likelihoods(ctc_logp_np, can_ctc_ids, ctc_kernel)

        fixed_t_np = (0.5 * neural_t_np + 0.5 * e4_simplex).astype(np.float32)

        utt_time_ms = (time.perf_counter() - t0) * 1000
        per_utt_latencies.append(utt_time_ms)

        all_neural_t_probs.append(neural_t_np)
        all_fixed_t_probs.append(fixed_t_np)
        all_gap_probs.append(gap_p_np)
        all_e4_probs.append(e4_simplex)
        all_sllr.append(sllr_vals)
        all_ollr.append(ollr_vals)

        for i, tp in enumerate(r["target_phones"]):
            all_target_keys.append(f"L2_ARCTIC|{spk}|{uid}|target|{i}")
            all_target_speakers.append(spk)
            all_gold_target.append(labels_map.index(tp["label"]))

        for j, gp in enumerate(r["gaps"]):
            all_gap_keys.append(f"L2_ARCTIC|{spk}|{uid}|gap|{j}")
            all_gap_speakers.append(spk)
            all_gold_gap.append(int(bool(gp["has_insertion"])))

    total_time_sec = time.perf_counter() - t_start
    total_audio_sec = sum(durations)
    rtf = total_time_sec / total_audio_sec

    tkeys = np.array(all_target_keys)
    gkeys = np.array(all_gap_keys)
    p_neu = np.concatenate(all_neural_t_probs, axis=0)
    p_fix = np.concatenate(all_fixed_t_probs, axis=0)
    pg = np.concatenate(all_gap_probs, axis=0)

    # Decisions Fixed 50/50
    conf_base = p_fix[:, 1:].max(axis=1)
    kind_base = p_fix[:, 1:].argmax(axis=1) + 1
    warn_t_base = conf_base >= tau_T_base
    dec_t_base = np.where(warn_t_base, kind_base, 0)
    dec_g_base = pg >= tau_G_base

    # Decisions Asymmetric
    warn_s_asym = apply_class_threshold(tkeys, p_neu[:, 1], tau_S, cut_S)
    warn_o_asym = apply_class_threshold(tkeys, p_fix[:, 2], tau_O, cut_O)
    warn_i_asym = apply_class_threshold(gkeys, pg, tau_I, cut_I)
    dec_t_asym = resolve_so_conflict(warn_s_asym, warn_o_asym, p_neu[:, 1], p_fix[:, 2], tau_S, tau_O, tkeys)
    dec_g_asym = warn_i_asym

    np.savez_compressed(
        save_path,
        target_keys=tkeys,
        gap_keys=gkeys,
        target_speakers=np.array(all_target_speakers),
        gap_speakers=np.array(all_gap_speakers),
        gold_target=np.array(all_gold_target, dtype=np.int64),
        gold_gap=np.array(all_gold_gap, dtype=np.int64),
        probs_target_neural=p_neu,
        probs_target_fixed=p_fix,
        probs_gap=pg,
        e4_probs=np.concatenate(all_e4_probs, axis=0),
        sllr=np.concatenate(all_sllr, axis=0),
        ollr=np.concatenate(all_ollr, axis=0),
        decisions_target_baseline=dec_t_base,
        decisions_gap_baseline=dec_g_base,
        decisions_target_asymmetric=dec_t_asym,
        decisions_gap_asymmetric=dec_g_asym,
        total_time_sec=total_time_sec,
        mean_latency_ms=float(np.mean(per_utt_latencies)),
        p50_latency_ms=float(np.percentile(per_utt_latencies, 50)),
        p95_latency_ms=float(np.percentile(per_utt_latencies, 95)),
        rtf=float(rtf),
        audio_duration_sec=float(total_audio_sec)
    )
    logger.info(f"Saved DEV_REPLICATION predictions to {save_path} (Time: {total_time_sec:.2f}s, RTF: {rtf:.4f})")


def run_speechocean_inference(
    charsiu_model, charsiu_tok, w2v2_model, model_t, model_g, ctc, ctc_kernel,
    p2i_39, tau_T_base, tau_G_base, tau_S, cut_S, tau_O, cut_O, tau_I, cut_I,
    device, logger
):
    save_path = OUT_DIR / "predictions_sm13_seed101_speechocean.npz"
    ckpt_file = CKPT_DIR / "speechocean_checkpoint.pkl"

    if save_path.exists():
        logger.info(f"SpeechOcean predictions already exist at {save_path}, skipping inference.")
        return

    logger.info("Executing SM13 Seed 101 inference on SpeechOcean762 (5,000 utterances)...")

    so_base = ROOT / "data/original_sources/speechocean762/speechocean762"
    with open(so_base / "resource/scores.json", "r", encoding="utf-8") as f:
        scores = json.load(f)
    with open(so_base / "resource/scores-detail.json", "r", encoding="utf-8") as f:
        scores_detail = json.load(f)

    # Build speaker/split/age/gender lookup
    meta_lookup = {}
    for split in ["train", "test"]:
        with open(so_base / split / "utt2spk") as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 2:
                    meta_lookup[p[0]] = {"spk": p[1], "split": split.upper()}

        with open(so_base / split / "spk2gender") as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 2:
                    spk, g = p[0], p[1].upper()
                    for uid, m in meta_lookup.items():
                        if m["spk"] == spk: m["gender"] = g

        with open(so_base / split / "spk2age") as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 2:
                    spk, a = p[0], int(p[1])
                    for uid, m in meta_lookup.items():
                        if m["spk"] == spk:
                            m["age"] = a
                            m["age_group"] = "CHILD" if a < 18 else "ADULT"

    def clean_phone(p: str) -> str:
        return re.sub(r"[0-9]+$", "", p.strip().upper())

    # Resume from checkpoint if present
    processed_uids = set()
    ckpt_data = {
        "target_keys": [], "gap_keys": [],
        "target_speakers": [], "gap_speakers": [],
        "target_splits": [], "gap_splits": [],
        "target_ages": [], "target_genders": [],
        "human_phone_scores": [], "gold_gap": [],
        "neural_t_probs": [], "fixed_t_probs": [],
        "gap_probs": [], "e4_probs": [], "sllr": [], "ollr": [],
        "latencies": [], "durations": []
    }

    if ckpt_file.exists():
        logger.info(f"Loading checkpoint from {ckpt_file}...")
        with open(ckpt_file, "rb") as f:
            ckpt_data = pickle.load(f)
        processed_uids = set(k.split("|")[2] for k in ckpt_data["target_keys"])
        logger.info(f"Resuming: {len(processed_uids)} utterances already completed.")

    uids = list(scores.keys())
    is_cuda = (device.type == "cuda")
    t_start = time.perf_counter()

    for idx, uid in enumerate(tqdm(uids, desc="SpeechOcean762")):
        if uid in processed_uids:
            continue

        meta = meta_lookup.get(uid, {"spk": "UNK", "split": "UNK", "gender": "UNK", "age": -1, "age_group": "UNK"})
        spk = meta["spk"]
        split = meta["split"]
        gender = meta.get("gender", "UNK")
        age = meta.get("age", -1)

        wav_path = so_base / "WAVE" / f"SPEAKER{spk}" / f"{uid}.WAV"
        if not wav_path.exists():
            logger.warning(f"Audio missing for {uid}: {wav_path}")
            continue

        obj = scores[uid]
        detail_obj = scores_detail.get(uid, {})

        # Parse canonical phones and human accuracy scores
        canonical_phones = []
        phone_scores = []
        for w in obj["words"]:
            phs = w.get("phones", [])
            pas = w.get("phones-accuracy", [])
            for p, pa in zip(phs, pas):
                cp = clean_phone(p)
                if cp:
                    canonical_phones.append(cp)
                    phone_scores.append(float(pa))

        N = len(canonical_phones)
        if N == 0:
            continue

        # Parse gold insertions from scores_detail
        # An insertion [P] in word w is placed in the corresponding gap
        gap_insertions = [0] * (N + 1)
        cur_p_idx = 0
        for w in detail_obj.get("words", []):
            ref_phs = [clean_phone(p) for p in w.get("ref-phones", "").split() if clean_phone(p)]
            nw = len(ref_phs)
            for ann in w.get("phones", []):
                # Count insertions inside word
                tokens = ann.split()
                # If any token has [phone], find its position relative to ref_phs
                t_p_offset = 0
                for tok in tokens:
                    if "[" in tok:
                        g_pos = cur_p_idx + t_p_offset
                        if g_pos <= N:
                            gap_insertions[g_pos] = 1
                    else:
                        t_p_offset += 1
            cur_p_idx += nw

        t0 = time.perf_counter()
        samples, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        samples = samples.mean(axis=1)
        dur = len(samples) / sr
        ckpt_data["durations"].append(dur)
        wav_t = torch.as_tensor(samples, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.inference_mode():
            char_logits = charsiu_model(wav_t).logits[0]
            if is_cuda: torch.cuda.synchronize()
            char_logits_np = char_logits.cpu().numpy()

        centers, nframes = align_centers(char_logits_np, canonical_phones, charsiu_tok)

        with torch.inference_mode():
            out = w2v2_model(wav_t, output_hidden_states=True)
            if is_cuda: torch.cuda.synchronize()

        t_feat, g_feat = extract_features_from_hidden_states(out.hidden_states, centers, nframes)

        ref_t_np = np.zeros((N, 39), dtype=np.float32)
        for i, p in enumerate(canonical_phones):
            if p in p2i_39: ref_t_np[i, p2i_39[p]] = 1.0

        ref_g_np = np.zeros((N + 1, 78), dtype=np.float32)
        for i in range(N + 1):
            left = canonical_phones[i - 1] if i > 0 else "[SIL]"
            right = canonical_phones[i] if i < N else "[SIL]"
            if left in p2i_39: ref_g_np[i, p2i_39[left]] = 1.0
            if right in p2i_39: ref_g_np[i, 39 + p2i_39[right]] = 1.0

        bx_t = torch.as_tensor(t_feat, dtype=torch.float32, device=device)
        br_t = torch.as_tensor(ref_t_np, dtype=torch.float32, device=device)
        with torch.inference_mode():
            neural_t_np = model_t.predict_proba(bx_t, br_t).cpu().numpy()

        bx_g = torch.as_tensor(g_feat, dtype=torch.float32, device=device)
        br_g = torch.as_tensor(ref_g_np, dtype=torch.float32, device=device)
        with torch.inference_mode():
            gap_p_np = model_g.predict_proba(bx_g, br_g)[:, 1].cpu().numpy()

        with torch.inference_mode():
            ctc_logits = ctc(out.hidden_states[6][0])
            ctc_logp_np = F.log_softmax(ctc_logits, dim=-1).cpu().numpy()

        can_ctc_ids = [PHONE_TO_ID.get(p, 0) for p in canonical_phones]
        e4_simplex, sllr_vals, ollr_vals = compute_e4_likelihoods(ctc_logp_np, can_ctc_ids, ctc_kernel)

        fixed_t_np = (0.5 * neural_t_np + 0.5 * e4_simplex).astype(np.float32)

        utt_time_ms = (time.perf_counter() - t0) * 1000
        ckpt_data["latencies"].append(utt_time_ms)

        ckpt_data["neural_t_probs"].append(neural_t_np)
        ckpt_data["fixed_t_probs"].append(fixed_t_np)
        ckpt_data["gap_probs"].append(gap_p_np)
        ckpt_data["e4_probs"].append(e4_simplex)
        ckpt_data["sllr"].append(sllr_vals)
        ckpt_data["ollr"].append(ollr_vals)

        for i in range(N):
            ckpt_data["target_keys"].append(f"SPEECHOCEAN|{spk}|{uid}|target|{i}")
            ckpt_data["target_speakers"].append(spk)
            ckpt_data["target_splits"].append(split)
            ckpt_data["target_ages"].append(age)
            ckpt_data["target_genders"].append(gender)
            ckpt_data["human_phone_scores"].append(phone_scores[i])

        for j in range(N + 1):
            ckpt_data["gap_keys"].append(f"SPEECHOCEAN|{spk}|{uid}|gap|{j}")
            ckpt_data["gap_speakers"].append(spk)
            ckpt_data["gap_splits"].append(split)
            ckpt_data["gold_gap"].append(gap_insertions[j])

        processed_uids.add(uid)

        # Checkpoint every 500 utterances
        if (len(processed_uids) % 500 == 0) or (len(processed_uids) == len(uids)):
            with open(ckpt_file, "wb") as f:
                pickle.dump(ckpt_data, f)
            logger.info(f"Checkpointed {len(processed_uids)} / {len(uids)} utterances to {ckpt_file}")
            gc.collect()
            if is_cuda: torch.cuda.empty_cache()

    total_time_sec = time.perf_counter() - t_start
    total_audio_sec = sum(ckpt_data["durations"])
    rtf = total_time_sec / total_audio_sec if total_audio_sec > 0 else 0.0

    tkeys = np.array(ckpt_data["target_keys"])
    gkeys = np.array(ckpt_data["gap_keys"])
    p_neu = np.concatenate(ckpt_data["neural_t_probs"], axis=0)
    p_fix = np.concatenate(ckpt_data["fixed_t_probs"], axis=0)
    pg = np.concatenate(ckpt_data["gap_probs"], axis=0)

    # Decisions Fixed 50/50
    conf_base = p_fix[:, 1:].max(axis=1)
    kind_base = p_fix[:, 1:].argmax(axis=1) + 1
    warn_t_base = conf_base >= tau_T_base
    dec_t_base = np.where(warn_t_base, kind_base, 0)
    dec_g_base = pg >= tau_G_base

    # Decisions Asymmetric
    warn_s_asym = apply_class_threshold(tkeys, p_neu[:, 1], tau_S, cut_S)
    warn_o_asym = apply_class_threshold(tkeys, p_fix[:, 2], tau_O, cut_O)
    warn_i_asym = apply_class_threshold(gkeys, pg, tau_I, cut_I)
    dec_t_asym = resolve_so_conflict(warn_s_asym, warn_o_asym, p_neu[:, 1], p_fix[:, 2], tau_S, tau_O, tkeys)
    dec_g_asym = warn_i_asym

    np.savez_compressed(
        save_path,
        target_keys=tkeys,
        gap_keys=gkeys,
        target_speakers=np.array(ckpt_data["target_speakers"]),
        gap_speakers=np.array(ckpt_data["gap_speakers"]),
        target_splits=np.array(ckpt_data["target_splits"]),
        gap_splits=np.array(ckpt_data["gap_splits"]),
        target_ages=np.array(ckpt_data["target_ages"], dtype=np.int32),
        target_genders=np.array(ckpt_data["target_genders"]),
        human_phone_scores=np.array(ckpt_data["human_phone_scores"], dtype=np.float32),
        gold_gap=np.array(ckpt_data["gold_gap"], dtype=np.int64),
        probs_target_neural=p_neu,
        probs_target_fixed=p_fix,
        probs_gap=pg,
        e4_probs=np.concatenate(ckpt_data["e4_probs"], axis=0),
        sllr=np.concatenate(ckpt_data["sllr"], axis=0),
        ollr=np.concatenate(ckpt_data["ollr"], axis=0),
        decisions_target_baseline=dec_t_base,
        decisions_gap_baseline=dec_g_base,
        decisions_target_asymmetric=dec_t_asym,
        decisions_gap_asymmetric=dec_g_asym,
        total_time_sec=total_time_sec,
        mean_latency_ms=float(np.mean(ckpt_data["latencies"])),
        p50_latency_ms=float(np.percentile(ckpt_data["latencies"], 50)),
        p95_latency_ms=float(np.percentile(ckpt_data["latencies"], 95)),
        rtf=float(rtf),
        audio_duration_sec=float(total_audio_sec)
    )
    logger.info(f"Saved SpeechOcean762 predictions to {save_path} (Time: {total_time_sec:.2f}s, RTF: {rtf:.4f})")
    # Clean up checkpoint file upon full completion
    if ckpt_file.exists():
        ckpt_file.unlink()


def main():
    logger = setup_logger(OUT_DIR / "inference_execution.log")
    logger.info("================================================================================")
    logger.info("LANGFENS SM13 Comprehensive Benchmark — Corpus-Wide Inference Engine")
    logger.info("================================================================================")

    # 1. Cryptographic validation of frozen assets
    logger.info("Validating frozen asset hashes...")
    for name, (path, exp_hash) in EXPECTED_HASHES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing asset: {path}")
        real_hash = sha256_file(path)
        if real_hash != exp_hash:
            raise ValueError(f"Checksum mismatch for {name}: expected {exp_hash}, got {real_hash}")
        logger.info(f"  [OK] {name}: {real_hash[:16]}... matches")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Inference Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load thresholds
    pol_base = json.loads(POLICY_BASELINE.read_text())
    tau_T_base = float(pol_base["target"]["threshold"])
    tau_G_base = float(pol_base["gap"]["threshold"])

    pol_asym = json.loads(POLICY_ASYMMETRIC.read_text())
    th_asym = pol_asym["operating_parameters"]["thresholds"]
    tau_S = float(th_asym["tau_substitution"])
    cut_S = int(th_asym["tie_cutoff_substitution"])
    tau_O = float(th_asym["tau_omission"])
    cut_O = int(th_asym["tie_cutoff_omission"])
    tau_I = float(th_asym["tau_insertion"])
    cut_I = int(th_asym["tie_cutoff_insertion"])

    # Load neural models
    logger.info("Loading backbones and frozen heads...")
    charsiu_tok = AutoTokenizer.from_pretrained(CHARSIU_TOK_PATH)
    charsiu_model = Wav2Vec2ForCTC.from_pretrained(CHARSIU_MODEL_PATH).eval().to(device)
    w2v2_model = AutoModel.from_pretrained("facebook/wav2vec2-base-960h").eval().to(device)

    model_t = ScalarMixingTargetHead().eval().to(device)
    model_t.load_state_dict(torch.load(CKPT_TARGET, map_location=device, weights_only=True))

    model_g = ScalarMixingGapHead().eval().to(device)
    model_g.load_state_dict(torch.load(CKPT_GAP, map_location=device, weights_only=True))

    ctc = LinearCTC().eval().to(device)
    sd_ctc = torch.load(CKPT_CTC, map_location=device, weights_only=False)
    ctc.load_state_dict(sd_ctc["state_dict"])
    ctc_kernel = get_ctc_kernel()

    phones_39 = sorted([p for p in charsiu_tok.get_vocab() if p not in {"[SIL]", "[UNK]", "[PAD]", "<s>", "</s>"}])
    p2i_39 = {p: i for i, p in enumerate(phones_39)}

    peak_before = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    logger.info(f"Models initialized. Initial GPU VRAM: {peak_before:.1f} MB")

    # Run DEV_REPLICATION
    run_dev_replication_inference(
        charsiu_model, charsiu_tok, w2v2_model, model_t, model_g, ctc, ctc_kernel,
        p2i_39, tau_T_base, tau_G_base, tau_S, cut_S, tau_O, cut_O, tau_I, cut_I,
        device, logger
    )

    # Run SpeechOcean762
    run_speechocean_inference(
        charsiu_model, charsiu_tok, w2v2_model, model_t, model_g, ctc, ctc_kernel,
        p2i_39, tau_T_base, tau_G_base, tau_S, cut_S, tau_O, cut_O, tau_I, cut_I,
        device, logger
    )

    peak_total = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    logger.info(f"All inference complete! Peak total GPU VRAM: {peak_total:.1f} MB")


if __name__ == "__main__":
    main()
