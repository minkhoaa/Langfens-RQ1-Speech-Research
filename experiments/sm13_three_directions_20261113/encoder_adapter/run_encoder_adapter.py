#!/usr/bin/env python3
"""Matched frozen middle-mixture vs attention-LoRA SM13 event experiment."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
import transformers
from transformers import Wav2Vec2Model

ROOT = Path("/home/khoa/Projects/New-Method-Research")
EXP = ROOT / "experiments/sm13_three_directions_20261113"
HERE = EXP / "encoder_adapter"
SHARED = EXP / "shared"
FEATS = ROOT / "experiments/rq1_encoder_bench_01/features"
MANIFEST = ROOT / "data/derived/l2arctic/manifests/train.jsonl"
ALIGN = FEATS / "alignment_centers.npz"
PREFIX = HERE / "prefix_l3_frames.f16"
OFFSETS = HERE / "prefix_l3_offsets.npy"
MODEL_ID = "facebook/wav2vec2-base-960h"
LAYERS = (5, 6, 7, 8, 9)  # HF hidden_states outputs after transformer blocks 4..8
BLOCKS = (4, 5, 6, 7, 8)  # zero-based Wav2Vec2 transformer blocks receiving LoRA

sys.path.insert(0, str(ROOT / "experiments/rq1_encoder_bench_01/common"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP / "shared"))
from dataset_util import DatasetManager  # noqa: E402
from training_engine import Standardizer, TargetHead, GapHead, gap_weights, seed_everything  # noqa: E402
_protocol_path = SHARED / "protocol.py"
_protocol_spec = importlib.util.spec_from_file_location("sm13_shared_protocol", _protocol_path)
if _protocol_spec is None or _protocol_spec.loader is None:
    raise ImportError(f"cannot load shared protocol from {_protocol_path}")
_shared_protocol = importlib.util.module_from_spec(_protocol_spec)
sys.modules[_protocol_spec.name] = _shared_protocol
_protocol_spec.loader.exec_module(_shared_protocol)
if Path(_shared_protocol.__file__).resolve() != _protocol_path.resolve():
    raise AssertionError(f"wrong protocol imported: {_shared_protocol.__file__}")
SplitProtocol = _shared_protocol.SplitProtocol
evaluate_fold = _shared_protocol.evaluate_fold
save_fold = _shared_protocol.save_fold


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features,
                                                device=base.weight.device, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank,
                                                device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        if not self.enabled:
            return base
        return base + torch.nn.functional.linear(
            torch.nn.functional.linear(x, self.lora_A), self.lora_B
        ) * self.scale


def patch_lora(model: Wav2Vec2Model):
    for p in model.parameters():
        p.requires_grad_(False)
    for block_idx in BLOCKS:
        attn = model.encoder.layers[block_idx].attention
        attn.q_proj = LoRALinear(attn.q_proj)
        attn.v_proj = LoRALinear(attn.v_proj)
    return [p for n, p in model.named_parameters() if "lora_" in n]


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pool_layer(hidden: torch.Tensor, centers: np.ndarray, nframes: int):
    """Differentiable pooling identical to the frozen per-layer acoustic cache."""
    n = hidden.shape[0]
    center_ix = np.rint(centers * (n / max(1, nframes))).astype(np.int64)
    offsets = np.arange(-8, 9)
    ix = np.clip(center_ix[:, None] + offsets[None, :], 0, n - 1)
    ids = torch.as_tensor(ix, dtype=torch.long, device=hidden.device)
    chunks = hidden[ids]
    c = chunks[:, 3:14, :]
    target = torch.cat((c.mean(1), c.std(1, correction=0), c[:, 10] - c[:, 0]), -1)
    left = torch.zeros((len(centers) + 1, 768), dtype=hidden.dtype, device=hidden.device)
    right = torch.zeros_like(left)
    if len(centers):
        left[1:] = chunks[:, 11:16].mean(1)
        right[:-1] = chunks[:, 1:6].mean(1)
    return target, torch.cat((left, right), dim=-1)


def get_adapted_layers(model, prefix_hidden: torch.Tensor, centers: np.ndarray,
                       nframes: int, with_grad: bool):
    """Run suffix blocks 4–11 from cached frozen block-3 outputs."""
    h = prefix_hidden.unsqueeze(0)
    t_layers, g_layers = [], []
    cm = torch.enable_grad() if with_grad else torch.inference_mode()
    with cm:
        for i in range(4, 12):
            h = model.encoder.layers[i](h, attention_mask=None)
            if i in BLOCKS:
                t, g = pool_layer(h[0], centers, nframes)
                t_layers.append(t); g_layers.append(g)
    return torch.stack(t_layers, dim=1), torch.stack(g_layers, dim=1)


def load_cached_layers(dm: DatasetManager, domain: str):
    name = "target" if domain == "t" else "gap"
    return [np.load(FEATS / f"{name}_acoustic_w2v2_960h_l{l}.npy", mmap_mode="r")
            for l in LAYERS]


def load_matched_prefix_layers(domain: str):
    name = "target" if domain == "t" else "gap"
    path = HERE / "matched_control_features" / f"{name}_l5_9_prefix_f16.npy"
    if not path.exists():
        raise FileNotFoundError(f"matched fp16-prefix control cache missing: {path}")
    stack = np.load(path, mmap_mode="r")
    return [stack[:, i, :] for i in range(5)]


def mix_cached(layers, ix: np.ndarray, logits: torch.Tensor, device: torch.device):
    w = torch.softmax(logits, dim=0)
    batch = np.stack([np.asarray(a[ix], dtype=np.float32) for a in layers], axis=1)
    return torch.as_tensor(batch, dtype=torch.float32, device=device), w


def fit_scaler(layers, refs, indices):
    f = np.asarray(layers[0][indices], dtype=np.float32) / len(layers)
    for layer in layers[1:]:
        f += np.asarray(layer[indices], dtype=np.float32) / len(layers)
    x = np.concatenate([f, np.asarray(refs[indices], dtype=np.float32)], axis=1)
    mean = x.mean(axis=0); scale = x.std(axis=0, ddof=0)
    if f.shape[1] == 1536:
        scale = np.maximum(scale, 1e-4)
    else:
        scale = scale + 1e-5
    return Standardizer(mean, scale)


def norm_batch(feats, refs, scaler, w, device):
    mixed = torch.einsum("nld,l->nd", feats, w)
    full = torch.cat((mixed, refs), dim=-1)
    mean = torch.as_tensor(scaler.mean, dtype=torch.float32, device=device)
    scale = torch.as_tensor(scaler.scale, dtype=torch.float32, device=device)
    return (full - mean) / scale


def build_fold_data(dm, split, seed, device):
    t_layers, g_layers = load_cached_layers(dm, "t"), load_cached_layers(dm, "g")
    tscaler = fit_scaler(t_layers, dm.target_refs, np.flatnonzero(split.t_fit))
    gscaler = fit_scaler(g_layers, dm.gap_refs, np.flatnonzero(split.g_fit))
    y_t = torch.as_tensor(dm.yt, dtype=torch.long, device=device)
    y_g = torch.as_tensor(dm.yg, dtype=torch.long, device=device)
    t_w = torch.tensor([1.0, 2.5, 4.0], device=device)
    g_w = gap_weights(dm.yg[split.g_fit]).to(device)
    return t_layers, g_layers, tscaler, gscaler, y_t, y_g, t_w, g_w


def train_static_warmstart(dm, split, data, seed, device, epochs=2):
    t_layers, g_layers, tscaler, gscaler, yt, yg, tw, gw = data
    set_seed(seed)
    th, gh = TargetHead().to(device), GapHead().to(device)
    t_mix, g_mix = nn.Parameter(torch.zeros(5, device=device)), nn.Parameter(torch.zeros(5, device=device))
    opt_t = torch.optim.AdamW([*th.parameters(), t_mix], lr=1e-3, weight_decay=0.01)
    opt_g = torch.optim.AdamW([*gh.parameters(), g_mix], lr=8e-4, weight_decay=0.0001)
    fit_t, fit_g = np.flatnonzero(split.t_fit), np.flatnonzero(split.g_fit)
    xt = torch.as_tensor(dm.target_refs, dtype=torch.float32, device=device)
    xg = torch.as_tensor(dm.gap_refs, dtype=torch.float32, device=device)
    for ep in range(epochs):
        for arr, model, mix, opt, refs, y, weight, bs in (
            (fit_t, th, t_mix, opt_t, xt, yt, tw, 512),
            (fit_g, gh, g_mix, opt_g, xg, yg, gw, 1024),
        ):
            perm = np.random.permutation(arr)
            model.train()
            for start in range(0, len(perm), bs):
                ix = perm[start:start+bs]
                ix_t = torch.as_tensor(ix, dtype=torch.long, device=device)
                fl, w = mix_cached(t_layers if model is th else g_layers, ix, mix, device)
                bx = norm_batch(fl, refs[ix_t], tscaler if model is th else gscaler, w, device)
                loss = nn.functional.cross_entropy(model(bx), y[ix_t], weight=weight)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return {
        "target_head": copy.deepcopy(th.state_dict()), "gap_head": copy.deepcopy(gh.state_dict()),
        "target_mix": t_mix.detach().cpu(), "gap_mix": g_mix.detach().cpu(),
    }


def cached_uid_features(dm, uid, t_layers, g_layers, t_mix, g_mix, device):
    ts, gs = dm.target_slices[uid], dm.gap_slices[uid]
    wt, wg = torch.softmax(t_mix, 0), torch.softmax(g_mix, 0)
    tl = torch.as_tensor(np.stack([np.asarray(x[ts], dtype=np.float32) for x in t_layers], 1), device=device)
    gl = torch.as_tensor(np.stack([np.asarray(x[gs], dtype=np.float32) for x in g_layers], 1), device=device)
    return torch.einsum("nld,l->nd", tl, wt), torch.einsum("nld,l->nd", gl, wg)


def prefix_uid(index, offset_map, prefix_map, device):
    a, b = offset_map[index]
    arr = np.asarray(prefix_map[a:b], dtype=np.float32).copy()
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def optimizer_step_arm(arm, dm, split, data, init, seed, fold, device, epochs=2):
    t_layers, g_layers, tscaler, gscaler, yt, yg, tw, gw = data
    th, gh = TargetHead().to(device), GapHead().to(device)
    th.load_state_dict(init["target_head"]); gh.load_state_dict(init["gap_head"])
    t_mix = nn.Parameter(init["target_mix"].to(device).clone())
    g_mix = nn.Parameter(init["gap_mix"].to(device).clone())
    model = None; lora_params = []
    if arm == "lora":
        model = Wav2Vec2Model.from_pretrained(MODEL_ID).to(device).eval()
        lora_params = patch_lora(model)
        for module in model.modules():
            if isinstance(module, LoRALinear): module.enabled = True
    param_groups = [
        {"params": list(th.parameters()) + [t_mix], "lr": 1e-3, "weight_decay": 0.01},
    ]
    if arm == "lora":
        param_groups.append({"params": lora_params, "lr": 1e-5, "weight_decay": 0.01})
    optimizer = torch.optim.AdamW(param_groups)
    # Reset after model/LoRA construction so both arms use identical shuffles and
    # head-dropout streams for the same fold and seed.
    set_seed(seed)
    recs = [json.loads(s) for s in MANIFEST.read_text().splitlines() if s.strip()]
    rec_by_uid = {r["utterance_id"]: r for r in recs}
    index_by_uid = {u: i for i, u in enumerate(dm.order)}
    offsets = np.load(OFFSETS)
    prefix_map = np.memmap(PREFIX, mode="r", dtype=np.float16).reshape(-1, 768)
    align = np.load(ALIGN)
    refs_t = torch.as_tensor(dm.target_refs, dtype=torch.float32, device=device)
    refs_g = torch.as_tensor(dm.gap_refs, dtype=torch.float32, device=device)
    mean_t = torch.as_tensor(tscaler.mean, device=device); scale_t = torch.as_tensor(tscaler.scale, device=device)
    mean_g = torch.as_tensor(gscaler.mean, device=device); scale_g = torch.as_tensor(gscaler.scale, device=device)
    uids = [u for i, u in enumerate(dm.order) if split.utt_fit[i]]
    grad_report = {"lora_A_grad_norm": 0.0, "lora_B_grad_norm": 0.0, "mix_grad_norm": 0.0}
    trainable_params = (sum(p.numel() for p in th.parameters()) + 5 + sum(p.numel() for p in lora_params))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    for ep in range(epochs):
        perm = np.random.permutation(uids)
        sum_loss = 0.0; n_updates = 0
        for step, uid in enumerate(perm):
            t_slice, g_slice = dm.target_slices[uid], dm.gap_slices[uid]
            centers = np.asarray(align[f"centers__{uid}"])
            nframes = int(align[f"nframes__{uid}"])
            if arm == "lora":
                h0 = prefix_uid(index_by_uid[uid], offsets, prefix_map, device)
                tl, gl = get_adapted_layers(model, h0, centers, nframes, with_grad=True)
                tx = torch.einsum("nld,l->nd", tl, torch.softmax(t_mix, 0))
                xt = torch.cat((tx, refs_t[t_slice]), -1)
                xt = (xt - mean_t) / scale_t
            else:
                tx, gx = cached_uid_features(dm, uid, t_layers, g_layers, t_mix, g_mix, device)
                xt = torch.cat((tx, refs_t[t_slice]), -1)
                xt = (xt - mean_t) / scale_t
            lt = nn.functional.cross_entropy(th(xt), yt[t_slice], weight=tw)
            loss = lt / 8.0  # gradient accumulation over eight utterances
            loss.backward()
            sum_loss += float(loss.detach()) * 16.0
            if ep == 0 and (step == 0 or step == 8) and arm == "lora":
                for p in lora_params:
                    if p.grad is not None:
                        nm = "lora_A_grad_norm" if any("lora_A" in n and q is p for n, q in model.named_parameters()) else "lora_B_grad_norm"
                        grad_report[nm] += float(p.grad.detach().norm())
                if t_mix.grad is not None:
                    grad_report["mix_grad_norm"] = float(t_mix.grad.detach().norm())
            if (step + 1) % 8 == 0 or (step + 1) == len(perm):
                torch.nn.utils.clip_grad_norm_(list(th.parameters()) + [t_mix] + lora_params, 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); n_updates += 1
            if (step + 1) % 300 == 0 or step + 1 == len(perm):
                print(f"fold={fold} arm={arm} epoch={ep+1}/{epochs} step={step+1}/{len(perm)} "
                      f"mean_loss={sum_loss/(step+1):.5f}", flush=True)
        print(f"fold={fold} arm={arm} epoch={ep+1} optimizer_updates={n_updates}", flush=True)
    train_sec = time.perf_counter() - t0
    if arm == "lora":
        if not (grad_report["lora_A_grad_norm"] > 0 and grad_report["lora_B_grad_norm"] > 0
                and grad_report["mix_grad_norm"] > 0):
            raise AssertionError(f"missing/nonzero gradient check failed: {grad_report}")
    artifact = {"target_head": th.state_dict(), "gap_head": gh.state_dict(),
                "target_mix": t_mix.detach().cpu(), "gap_mix": g_mix.detach().cpu(),
                "scaler_t_mean": tscaler.mean, "scaler_t_scale": tscaler.scale,
                "scaler_g_mean": gscaler.mean, "scaler_g_scale": gscaler.scale}
    if arm == "lora":
        artifact["lora"] = {n: p.detach().cpu() for n, p in model.state_dict().items()
                            if "lora_" in n}
    return artifact, {"training_seconds": train_sec, "gradient_check": grad_report,
                      "fit_utterances": len(uids), "epochs": epochs,
                      "trainable_parameter_count": int(trainable_params),
                      "peak_cuda_memory_allocated_bytes": (int(torch.cuda.max_memory_allocated(device))
                                                           if device.type == "cuda" else None),
                      "optimizer_updates_per_epoch": int((len(uids) + 7) // 8),
                      "objective": "target_only; shared all13 gap probabilities used during evaluation"}


def load_arm(artifact, device, arm):
    th, gh = TargetHead().to(device), GapHead().to(device)
    th.load_state_dict(artifact["target_head"]); gh.load_state_dict(artifact["gap_head"])
    tm = nn.Parameter(artifact["target_mix"].to(device)); gm = nn.Parameter(artifact["gap_mix"].to(device))
    tscaler = Standardizer(artifact["scaler_t_mean"], artifact["scaler_t_scale"])
    gscaler = Standardizer(artifact["scaler_g_mean"], artifact["scaler_g_scale"])
    model = None
    if arm == "lora":
        model = Wav2Vec2Model.from_pretrained(MODEL_ID).to(device).eval()
        patch_lora(model)
        model.load_state_dict(artifact["lora"], strict=False)
    return model, th.eval(), gh.eval(), tm, gm, tscaler, gscaler


@torch.inference_mode()
def predict_uid(arm, uid, uid_ix, dm, model, th, gh, tm, gm, tscaler, gscaler,
                data, offsets, prefix_map, align, device):
    t_layers, g_layers = data[0], data[1]
    ts, gs = dm.target_slices[uid], dm.gap_slices[uid]
    if arm == "lora":
        h0 = prefix_uid(uid_ix, offsets, prefix_map, device)
        centers = np.asarray(align[f"centers__{uid}"]); nframes = int(align[f"nframes__{uid}"])
        tl, gl = get_adapted_layers(model, h0, centers, nframes, with_grad=False)
        tx = torch.einsum("nld,l->nd", tl, torch.softmax(tm, 0))
        gx = torch.einsum("nld,l->nd", gl, torch.softmax(gm, 0))
    else:
        tx, gx = cached_uid_features(dm, uid, t_layers, g_layers, tm, gm, device)
    xt = torch.cat((tx, torch.as_tensor(dm.target_refs[ts], dtype=torch.float32, device=device)), -1)
    xg = torch.cat((gx, torch.as_tensor(dm.gap_refs[gs], dtype=torch.float32, device=device)), -1)
    xt = (xt - torch.as_tensor(tscaler.mean, device=device)) / torch.as_tensor(tscaler.scale, device=device)
    xg = (xg - torch.as_tensor(gscaler.mean, device=device)) / torch.as_tensor(gscaler.scale, device=device)
    return torch.softmax(th(xt), -1).cpu().numpy(), torch.softmax(gh(xg), -1)[:, 1].cpu().numpy()


def preflight(dm, device):
    print("Running disabled-adapter equivalence and gradient preflight", flush=True)
    model = Wav2Vec2Model.from_pretrained(MODEL_ID).to(device).eval()
    lora = patch_lora(model)
    manifest = [json.loads(s) for s in MANIFEST.read_text().splitlines() if s.strip()]
    rec = manifest[0]; uid = rec["utterance_id"]
    samples, sr = sf.read(rec["audio_path"], dtype="float32", always_2d=True)
    from transformers import AutoFeatureExtractor
    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    wav = fe(samples.mean(1), sampling_rate=sr, return_tensors="pt")["input_values"].to(device)
    dm = DatasetManager()
    uid_ix = dm.order.index(uid)
    offsets = np.load(OFFSETS)
    prefix_map = np.memmap(PREFIX, mode="r", dtype=np.float16).reshape(-1, 768)
    h0 = prefix_uid(uid_ix, offsets, prefix_map, device)
    with torch.inference_mode():
        full = model(wav, output_hidden_states=True).hidden_states
        prefix_error = float((h0 - full[4][0].float()).abs().max().item())
        for module in model.modules():
            if isinstance(module, LoRALinear): module.enabled = False
        h = h0.unsqueeze(0); deltas = []
        for i in range(4, 12):
            h = model.encoder.layers[i](h, attention_mask=None)
            if i in BLOCKS:
                deltas.append(float((h[0] - full[i+1][0]).abs().max().item()))
    # Convert the frozen inference cache to an ordinary tensor before the
    # gradient-path check; autograd cannot save inference-mode tensors.
    max_abs = max(deltas)
    if prefix_error > 0.02 or max_abs > 0.02:
        raise AssertionError(f"cached prefix/suffix mismatch prefix={prefix_error} suffix={max_abs}")
    # Check that the pooling for every mixed state reproduces the repository's
    # frozen per-layer acoustic caches (including exact alignment/frame mapping).
    ts, gs = dm.target_slices[uid], dm.gap_slices[uid]
    align = np.load(ALIGN)
    centers = np.asarray(align[f"centers__{uid}"]); nframes = int(align[f"nframes__{uid}"])
    pool_errors = []
    for li in LAYERS:
        p_t, p_g = pool_layer(full[li][0].float(), centers, nframes)
        c_t = np.load(FEATS / f"target_acoustic_w2v2_960h_l{li}.npy", mmap_mode="r")[ts]
        c_g = np.load(FEATS / f"gap_acoustic_w2v2_960h_l{li}.npy", mmap_mode="r")[gs]
        pool_errors.extend((float(np.max(np.abs(p_t.cpu().numpy() - c_t))),
                            float(np.max(np.abs(p_g.cpu().numpy() - c_g)))))
    max_pool_error = max(pool_errors)
    if max_pool_error > 1e-3:
        raise AssertionError(f"frozen pooled feature cache mismatch max_abs={max_pool_error}")
    for module in model.modules():
        if isinstance(module, LoRALinear): module.enabled = True
    model.zero_grad(set_to_none=True)
    h = h0.unsqueeze(0)
    for i in range(4, 9): h = model.encoder.layers[i](h, attention_mask=None)
    loss = h.square().mean(); loss.backward()
    norms = [float(p.grad.norm()) for p in lora if p.grad is not None]
    if len(norms) != len(lora) or max(norms, default=0) <= 0:
        raise AssertionError(f"LoRA preflight gradients invalid: {norms}")
    report = {"cached_prefix_max_abs_error_vs_full": prefix_error,
              "max_hidden_abs_error_vs_full": max_abs, "per_mixed_state_max_abs": deltas,
              "max_pooled_feature_error_vs_frozen_cache": max_pool_error,
              "preflight_lora_grad_norms": norms, "all_lora_trainable": len(lora),
              "target_layers": list(LAYERS), "adapted_blocks_zero_based": list(BLOCKS),
              "prefix_cache": str(PREFIX), "prefix_dtype": "float16"}
    (HERE / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def evaluate_arm(arm, artifact, dm, split, data, fold, seed, device, out_dir):
    model, th, gh, tm, gm, tscaler, gscaler = load_arm(artifact, device, arm)
    offsets = np.load(OFFSETS)
    prefix_map = np.memmap(PREFIX, mode="r", dtype=np.float16).reshape(-1, 768)
    align = np.load(ALIGN)
    uid_ix = {u:i for i,u in enumerate(dm.order)}
    all_uids = [u for i,u in enumerate(dm.order) if split.utt_cal[i] or split.utt_held[i]]
    cal_t, held_t = [], []
    for uid in all_uids:
        tp, _ = predict_uid(arm, uid, uid_ix[uid], dm, model, th, gh, tm, gm,
                             tscaler, gscaler, data, offsets, prefix_map, align, device)
        if split.utt_cal[uid_ix[uid]]:
            cal_t.append(tp)
        else:
            held_t.append(tp)
    base_path = SHARED / "baseline" / f"sm13_scalar_mix_seed{seed}_fold{fold}_scores.npz"
    if not base_path.exists():
        raise FileNotFoundError(f"canonical shared gap baseline required before evaluation: {base_path}")
    with np.load(base_path) as base:
        cal_gap = np.asarray(base["cal_gap_probs"])
        held_gap = np.asarray(base["held_gap_probs"])
    result = evaluate_fold(dm, fold, np.concatenate(cal_t), cal_gap,
                           np.concatenate(held_t), held_gap,
                           f"encoder_adapter_{arm}", seed)
    result["gap_probability_source"] = str(base_path)
    result["gap_model"] = "shared canonical all13-layer SM13 gap model; identical for both paired arms"
    save_fold(result, out_dir)
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return {k:v for k,v in result.items() if k not in ("target_predictions","gap_predictions")}


def run_pair(dm, fold, seed, device, epochs):
    split = SplitProtocol(dm).split(fold)
    data = build_fold_data(dm, split, seed, device)
    init = train_static_warmstart(dm, split, data, seed + 50_000 + fold * 1000, device, epochs=2)
    out_dir = HERE / "folds"
    out_dir.mkdir(exist_ok=True)
    torch.save(init, out_dir / f"warmstart_seed{seed}_fold{fold}.pt")
    arm_results = {}
    for arm in ("control", "lora"):
        ckpt = out_dir / f"{arm}_seed{seed}_fold{fold}.pt"
        if ckpt.exists():
            print(f"Skipping existing checkpoint {ckpt}", flush=True)
            artifact = torch.load(ckpt, map_location="cpu", weights_only=False)
            train_receipt = json.loads(ckpt.with_suffix(".json").read_text())
        else:
            artifact, train_receipt = optimizer_step_arm(
                arm, dm, split, data, init, seed + fold * 100,
                fold, device, epochs=epochs)
            torch.save(artifact, ckpt)
            ckpt.with_suffix(".json").write_text(json.dumps(train_receipt, indent=2) + "\n")
        arm_results[arm] = evaluate_arm(arm, artifact, dm, split, data, fold, seed, device, out_dir)
        arm_results[arm]["training"] = train_receipt
        del artifact
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
    pair = {"fold": fold, "seed": seed, "control": arm_results["control"], "lora": arm_results["lora"]}
    (out_dir / f"pair_seed{seed}_fold{fold}.json").write_text(json.dumps(pair, indent=2) + "\n")
    return pair


def run_matched_control(dm, fold, seed, device, epochs):
    """Rerun only the frozen arm on exact fp16-prefix features; reuse candidate."""
    split = SplitProtocol(dm).split(fold)
    data = build_fold_data(dm, split, seed, device)  # retains fit-only scaler from prereg warmstart
    init_path = HERE / "folds" / f"warmstart_seed{seed}_fold{fold}.pt"
    if not init_path.exists():
        raise FileNotFoundError(f"fit-only warmstart missing: {init_path}")
    init = torch.load(init_path, map_location="cpu", weights_only=False)
    matched_t, matched_g = load_matched_prefix_layers("t"), load_matched_prefix_layers("g")
    rng = np.random.default_rng(seed + fold * 100)
    t_probe = rng.choice(np.flatnonzero(split.t_fit), size=min(256, int(split.t_fit.sum())), replace=False)
    g_probe = rng.choice(np.flatnonzero(split.g_fit), size=min(256, int(split.g_fit.sum())), replace=False)
    t_diff = np.concatenate([np.abs(np.asarray(matched_t[i][t_probe]) -
                                    np.asarray(data[0][i][t_probe])).ravel() for i in range(5)])
    g_diff = np.concatenate([np.abs(np.asarray(matched_g[i][g_probe]) -
                                    np.asarray(data[1][i][g_probe])).ravel() for i in range(5)])
    parity = {"sampled_fit_target_events": len(t_probe), "sampled_fit_gap_events": len(g_probe),
              "target_feature_max_abs_change": float(t_diff.max()),
              "target_feature_mean_abs_change": float(t_diff.mean()),
              "gap_feature_max_abs_change": float(g_diff.max()),
              "gap_feature_mean_abs_change": float(g_diff.mean())}
    data = (matched_t, matched_g, *data[2:])
    artifact, train_receipt = optimizer_step_arm(
        "control_prefix", dm, split, data, init, seed + fold * 100, fold, device, epochs=epochs)
    out_dir = HERE / "folds"; out_dir.mkdir(exist_ok=True)
    ckpt = out_dir / f"control_prefix_seed{seed}_fold{fold}.pt"
    torch.save(artifact, ckpt)
    ckpt.with_suffix(".json").write_text(json.dumps(train_receipt, indent=2) + "\n")
    matched = evaluate_arm("control_prefix", artifact, dm, split, data, fold, seed, device, out_dir)
    matched["training"] = train_receipt
    old_pair_path = out_dir / f"pair_seed{seed}_fold{fold}.json"
    if not old_pair_path.exists():
        raise FileNotFoundError(f"candidate pair summary missing: {old_pair_path}")
    old_pair = json.loads(old_pair_path.read_text())
    result = {
        "fold": fold, "seed": seed,
        "control_prefix": matched,
        "lora": old_pair["lora"],
        "control_float32_cache_superseded": old_pair["control"],
        "control_parity": "fp16 prefix cache → frozen blocks4–8 pooled features; same warmstart/scaler/schedule",
        "input_precision_change_summary": parity,
    }
    (out_dir / f"matched_pair_seed{seed}_fold{fold}.json").write_text(
        json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", type=int, default=[0,1,2,3])
    parser.add_argument("--seeds", nargs="+", type=int, default=[101,202])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--matched-control-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda": torch.cuda.set_device(device)
    if not (PREFIX.exists() and OFFSETS.exists()):
        raise FileNotFoundError("raw prefix cache missing; run extract_shared_raw_cache.py first")
    dm = DatasetManager()
    snapshots = Path.home() / ".cache/huggingface/hub/models--facebook--wav2vec2-base-960h/snapshots"
    model_revisions = sorted(p.name for p in snapshots.iterdir()) if snapshots.exists() else []
    metadata = {
        "model_id": MODEL_ID, "cached_model_snapshot_ids": model_revisions,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "cuda_runtime": torch.version.cuda, "python_version": platform.python_version(),
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cpu_threads": torch.get_num_threads(), "folds": args.folds, "seeds": args.seeds,
        "warmstart_epochs": 2, "paired_training_epochs": args.epochs,
        "shared_protocol_path": str(Path(_shared_protocol.__file__).resolve()),
        "canonical_gap_source": "shared/baseline/sm13_scalar_mix_seed{seed}_fold{fold}_scores.npz",
    }
    (HERE / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if not args.skip_preflight:
        preflight(dm, device)
    if args.preflight_only:
        return
    if args.matched_control_only:
        results = [run_matched_control(dm, fold, seed, device, args.epochs)
                   for fold in args.folds for seed in args.seeds]
        (HERE / "folds" / "matched_control_summary.json").write_text(
            json.dumps(results, indent=2) + "\n")
        return
    results = []
    for fold in args.folds:
        for seed in args.seeds:
            print(f"START fold={fold} seed={seed} device={device}", flush=True)
            results.append(run_pair(dm, fold, seed, device, args.epochs))
    (HERE / "folds" / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
