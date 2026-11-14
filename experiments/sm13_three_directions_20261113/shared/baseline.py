"""Fold-excluded 13-layer SM13 neural baseline over frozen pooled features.

This is a fit-only refit for this exploratory 6/3/3 protocol. It intentionally
does not load or reuse full-TRAIN SM13 checkpoints.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

ROOT = Path("/home/khoa/Projects/New-Method-Research")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/sm13_three_directions_20261113/shared"))
sys.path.insert(0, str(ROOT / "experiments/rq1_sm13_formal_train_01"))
from protocol import SplitProtocol  # noqa: E402
from agent_a_sm13_engine import ScalarMixingGapHead, ScalarMixingTargetHead  # noqa: E402

FEATURES = ROOT / "experiments/rq1_encoder_bench_01/features"
OUT = ROOT / "experiments/sm13_three_directions_20261113/shared/baseline"
LAYERS = 13
TRAINING = {
    "target": {"epochs": 12, "batch_size": 512, "learning_rate": 1e-3, "weight_decay": 1e-2},
    "gap": {"epochs": 5, "batch_size": 1024, "learning_rate": 8e-4, "weight_decay": 1e-4},
}


def _seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _open_layers(domain: str):
    arrays=[]
    for layer in range(LAYERS):
        p=FEATURES/f"{domain}_acoustic_w2v2_960h_l{layer}.npy"
        if not p.exists(): raise FileNotFoundError(f"missing frozen pooled feature cache: {p}")
        arrays.append(np.load(p,mmap_mode="r"))
    return arrays


def _batch_features(maps, refs, indices):
    ac = np.stack([np.asarray(m[indices],dtype=np.float32) for m in maps],axis=1) if isinstance(maps,list) else np.asarray(maps[indices],dtype=np.float32)
    return torch.from_numpy(ac), torch.from_numpy(np.asarray(refs[indices], dtype=np.float32))


def _train(domain: str, dm, fit_mask, seed: int, device: torch.device, maps_override=None):
    _seed(seed)
    cfg=TRAINING[domain]
    maps = maps_override if maps_override is not None else _open_layers("target" if domain == "target" else "gap")
    refs = dm.target_refs if domain == "target" else dm.gap_refs
    y = dm.yt if domain == "target" else dm.yg
    model = (ScalarMixingTargetHead() if domain == "target" else ScalarMixingGapHead()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    if domain == "target":
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0,2.5,4.0],device=device))
    else:
        counts=np.bincount(y[fit_mask],minlength=2).astype(np.float32)
        w=np.sqrt(float(fit_mask.sum())/(2*np.maximum(counts,1))); w=w/w.mean()
        criterion=nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32,device=device))
    fit_ix = np.flatnonzero(fit_mask)
    fit_ac=(np.stack([np.asarray(m[fit_ix],dtype=np.float32) for m in maps],axis=1)
            if isinstance(maps,list) else np.asarray(maps[fit_ix],dtype=np.float32))
    fit_refs=np.asarray(refs[fit_ix],dtype=np.float32)
    fit_y=y[fit_ix]
    history=[]
    for epoch in range(cfg["epochs"]):
        order=np.random.permutation(len(fit_ix)); model.train(); total=0.; n=0
        for start in range(0,len(order),cfg["batch_size"]):
            ix=order[start:start+cfg["batch_size"]]
            ac=torch.from_numpy(fit_ac[ix]); rf=torch.from_numpy(fit_refs[ix])
            logits=(model(ac.to(device),rf.to(device)))
            target=torch.as_tensor(fit_y[ix],dtype=torch.long,device=device)
            loss=criterion(logits,target)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += float(loss.detach())*len(ix); n+=len(ix)
        history.append(total/max(1,n))
    del fit_ac,fit_refs,fit_y
    return model, maps, refs, {"loss_by_epoch":history,"trainable_params":sum(p.numel() for p in model.parameters()),**cfg}


@torch.inference_mode()
def _predict(model, maps, refs, indices, device):
    model.eval(); chunks=[]
    for start in range(0,len(indices),512):
        ix=indices[start:start+512]
        ac,rf=_batch_features(maps,refs,ix)
        chunks.append(torch.softmax(model(ac.to(device),rf.to(device)),dim=-1).cpu().numpy())
    return np.concatenate(chunks,axis=0)


def run_fold_baseline(dm, fold: int, seed: int, device: str = "cuda", out_dir: str | Path = OUT) -> dict[str,Any]:
    """Train fit-only SM13 heads and return probabilities in cal/held mask order."""
    split=SplitProtocol(dm).split(fold); dev=torch.device(device if device!="cuda" or torch.cuda.is_available() else "cpu")
    start=time.perf_counter(); out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    model_t,maps_t,refs_t,stats_t=_train("target",dm,split.t_fit,seed,dev)
    model_g,maps_g,refs_g,stats_g=_train("gap",dm,split.g_fit,seed+1000,dev)
    cal_t_ix=np.flatnonzero(split.t_cal); held_t_ix=np.flatnonzero(split.t_held)
    cal_g_ix=np.flatnonzero(split.g_cal); held_g_ix=np.flatnonzero(split.g_held)
    result={
        "cal_target_probs":_predict(model_t,maps_t,refs_t,cal_t_ix,dev),
        "held_target_probs":_predict(model_t,maps_t,refs_t,held_t_ix,dev),
        "cal_gap_probs":_predict(model_g,maps_g,refs_g,cal_g_ix,dev)[:,1],
        "held_gap_probs":_predict(model_g,maps_g,refs_g,held_g_ix,dev)[:,1],
        "fit_speakers":list(split.fit_speakers),"calibration_speakers":list(split.calibration_speakers),
        "held_speakers":list(split.held_speakers),"seed":seed,"fold":fold,
        "target_train":stats_t,"gap_train":stats_g,"optimizer":"AdamW",
        "elapsed_seconds":time.perf_counter()-start,
    }
    stem=f"sm13_scalar_mix_seed{seed}_fold{fold}"
    torch.save({"target":model_t.state_dict(),"gap":model_g.state_dict()},out/f"{stem}.pt")
    np.savez_compressed(out/f"{stem}_scores.npz",cal_target_probs=result["cal_target_probs"],
        held_target_probs=result["held_target_probs"],cal_gap_probs=result["cal_gap_probs"],
        held_gap_probs=result["held_gap_probs"])
    receipt={k:v for k,v in result.items() if not k.endswith("_probs")}
    (out/f"{stem}.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
    del maps_t,maps_g,model_t,model_g; gc.collect()
    if dev.type=="cuda": torch.cuda.empty_cache()
    return result


def run_fold_l6_target(dm, fold: int, seed: int, device: str = "cuda",
                       out_dir: str | Path = OUT) -> dict[str, Any]:
    """Matched pooled-L6 target MLP; repeats L6 in scalar-mix slots to keep head fixed."""
    split=SplitProtocol(dm).split(fold)
    dev=torch.device(device if device!="cuda" or torch.cuda.is_available() else "cpu")
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    one=np.load(FEATURES/"target_acoustic_w2v2_960h_l6.npy",mmap_mode="r")
    maps=np.broadcast_to(one[:,None,:],(len(one),LAYERS,one.shape[1]))
    model,maps,refs,stats=_train("target",dm,split.t_fit,seed,dev,maps_override=maps)
    cp=np.flatnonzero(split.t_cal); hp=np.flatnonzero(split.t_held)
    scores={"cal_target_probs":_predict(model,maps,refs,cp,dev),
            "held_target_probs":_predict(model,maps,refs,hp,dev),
            "fit_speakers":list(split.fit_speakers),"calibration_speakers":list(split.calibration_speakers),
            "held_speakers":list(split.held_speakers),"seed":seed,"fold":fold,
            "target_train":stats,"optimizer":"AdamW"}
    stem=f"l6_pooled_target_seed{seed}_fold{fold}"
    torch.save(model.state_dict(),out/f"{stem}.pt")
    np.savez_compressed(out/f"{stem}_scores.npz",**{k:v for k,v in scores.items() if k.endswith("_probs")})
    (out/f"{stem}.json").write_text(json.dumps({k:v for k,v in scores.items() if not k.endswith("_probs")},indent=2)+"\n")
    del one,maps,model; gc.collect()
    if dev.type=="cuda": torch.cuda.empty_cache()
    return scores

