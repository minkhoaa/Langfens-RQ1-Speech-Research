"""Run temporal CNN against shared fit-only SM13 scalar-mix baseline."""
from __future__ import annotations

import fcntl
import argparse
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path("/home/khoa/Projects/New-Method-Research")
BASE = ROOT / "experiments/sm13_three_directions_20261113"
sys.path.insert(0, str(BASE / "shared"))
sys.path.insert(0, str(BASE / "temporal"))
import baseline as baseline_module  # noqa: E402
from baseline import run_fold_baseline, run_fold_l6_target  # noqa: E402
from model import TemporalEventHead  # noqa: E402
from protocol import SEEDS, SplitProtocol, evaluate_fold, save_fold  # noqa: E402

SHARED = BASE / "shared"
OUT = BASE / "temporal"
LOCK = BASE / "gpu.lock"
TRAINING = {
    "target": {"epochs": 12, "batch_size": 512, "lr": 1e-3, "weight_decay": 1e-2},
    "gap": {"epochs": 5, "batch_size": 1024, "lr": 8e-4, "weight_decay": 1e-4},
}
INFER_BATCH = 512


def seed_all(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def class_weights(y, mask, domain, device):
    if domain == "target":
        return torch.tensor([1.0,2.5,4.0],dtype=torch.float32,device=device)
    counts=np.bincount(y[mask],minlength=2).astype(np.float32)
    w=np.sqrt(float(mask.sum())/(2*np.maximum(counts,1))); w/=w.mean()
    return torch.tensor(w,dtype=torch.float32,device=device)


def train_temporal(domain, dm, split, seed, device, target_windows, gap_windows):
    is_target=domain=="target"
    cfg=TRAINING[domain]
    frames=target_windows if is_target else gap_windows
    ref=dm.target_refs if is_target else dm.gap_refs
    y=dm.yt if is_target else dm.yg
    fit_mask=split.t_fit if is_target else split.g_fit
    indices=np.flatnonzero(fit_mask)
    ref=np.asarray(ref,dtype=np.float32)
    seed_all(seed)
    model=TemporalEventHead(ref.shape[1],3 if is_target else 2).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
    criterion=nn.CrossEntropyLoss(weight=class_weights(y,fit_mask,domain,device))
    losses=[]; started=time.perf_counter()
    for epoch in range(cfg["epochs"]):
        model.train(); order=np.random.permutation(indices); total=0.; count=0
        for start in range(0,len(order),cfg["batch_size"]):
            ix=order[start:start+cfg["batch_size"]]
            bx=torch.as_tensor(np.asarray(frames[ix],dtype=np.float32),device=device)
            br=torch.as_tensor(ref[ix],device=device)
            by=torch.as_tensor(y[ix],dtype=torch.long,device=device)
            logits=model(bx,br); loss=criterion(logits,by)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total+=float(loss.detach())*len(ix); count+=len(ix)
        losses.append(total/max(1,count))
    return model,{"loss_by_epoch":losses,"train_seconds":time.perf_counter()-started,
                  "trainable_params":sum(p.numel() for p in model.parameters()),**cfg}


@torch.inference_mode()
def predict_temporal(model, domain, dm, split, device, target_windows, gap_windows):
    is_target=domain=="target"
    frames=target_windows if is_target else gap_windows
    ref=dm.target_refs if is_target else dm.gap_refs
    cal_mask=split.t_cal if is_target else split.g_cal
    held_mask=split.t_held if is_target else split.g_held
    ref=np.asarray(ref,dtype=np.float32); model.eval()
    def run(ix):
        parts=[]
        for start in range(0,len(ix),INFER_BATCH):
            sub=ix[start:start+INFER_BATCH]
            bx=torch.as_tensor(np.asarray(frames[sub],dtype=np.float32),device=device)
            br=torch.as_tensor(ref[sub],device=device)
            parts.append(torch.softmax(model(bx,br),-1).cpu().numpy())
        return np.concatenate(parts,axis=0)
    cal=run(np.flatnonzero(cal_mask)); held=run(np.flatnonzero(held_mask))
    return (cal if is_target else cal[:,1]), (held if is_target else held[:,1])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--baseline-only",action="store_true",help="fit/save shared SM13 and L6 pooled controls, then release GPU")
    parser.add_argument("--smoke",action="store_true",help="one-fold one-epoch train/evaluate/save path in temporal/smoke")
    parser.add_argument("--seed",type=int,choices=SEEDS,help="run one registered seed (for per-job GPU scheduling)")
    parser.add_argument("--fold",type=int,choices=range(4),help="run one outer fold (for per-job GPU scheduling)")
    args=parser.parse_args()
    run_out=OUT/"smoke" if args.smoke else OUT
    run_out.mkdir(parents=True,exist_ok=True)
    run_seeds=(101,) if args.smoke else ((args.seed,) if args.seed is not None else SEEDS)
    run_folds=(0,) if args.smoke else ((args.fold,) if args.fold is not None else range(4))
    if args.smoke:
        for cfg in TRAINING.values(): cfg.update(epochs=1,batch_size=512)
        for cfg in baseline_module.TRAINING.values(): cfg.update(epochs=1,batch_size=512)
    receipt=BASE/"encoder_adapter/raw_cache_receipt.json"
    target_path=SHARED/"target_w2v2_960h_l6_windows.npy"
    gap_path=SHARED/"gap_w2v2_960h_l6_windows.npy"
    if not receipt.exists() or not target_path.exists() or not gap_path.exists():
        raise FileNotFoundError("encoder_adapter must finish TRAIN raw L6 window cache before temporal training")
    dm=SplitProtocol().dm
    target=np.load(target_path,mmap_mode="r"); gap=np.load(gap_path,mmap_mode="r")
    if target.shape!=(len(dm.yt),17,768) or gap.shape!=(len(dm.yg),10,768):
        raise ValueError(f"unexpected temporal cache shapes {target.shape} {gap.shape}")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_info={"python":sys.version,"torch":torch.__version__,"device":str(device),
       "platform":platform.platform(),"cpu":platform.processor(),"cpu_count":os.cpu_count(),
       "cuda_name":torch.cuda.get_device_name(0) if device.type=="cuda" else None,
       "cuda_memory_bytes":torch.cuda.get_device_properties(0).total_memory if device.type=="cuda" else None}
    all_summaries=[]
    previous=[]
    result_path=run_out/"temporal_results.json"
    if not args.baseline_only and result_path.exists():
        old=json.loads(result_path.read_text())
        if old.get("status")=="completed": previous=old.get("runs",[])
    baseline_out=run_out/"shared_baseline"
    with LOCK.open("a+") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        (run_out/"hardware.json").write_text(json.dumps(device_info,indent=2)+"\n")
        for seed in run_seeds:
            for fold in run_folds:
                if not args.baseline_only and not args.smoke and args.seed is None and args.fold is None:
                    tstem=f"temporal_cnn_target_shared_sm13_gap_seed{seed}_fold{fold}"
                    fstem=f"temporal_cnn_target_gap_l6_seed{seed}_fold{fold}"
                    complete=(OUT/"target_only_predictions"/f"{tstem}.npz").exists() and \
                        (OUT/"target_only_predictions"/f"{tstem}.json").exists() and \
                        (OUT/"full_temporal_predictions"/f"{fstem}.npz").exists() and \
                        (OUT/"full_temporal_predictions"/f"{fstem}.json").exists() and \
                        (OUT/f"temporal_cnn_l6_seed{seed}_fold{fold}.pt").exists()
                    if complete:
                        print(f"[temporal] skip completed seed={seed} fold={fold}",flush=True)
                        continue
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
                print(f"[temporal] seed={seed} fold={fold} device={device}",flush=True)
                split=SplitProtocol(dm).split(fold)
                baseline_scores=baseline_out/f"sm13_scalar_mix_seed{seed}_fold{fold}_scores.npz"
                if baseline_scores.exists():
                    z=np.load(baseline_scores)
                    baseline={k:z[k] for k in ("cal_target_probs","held_target_probs","cal_gap_probs","held_gap_probs")}
                else:
                    baseline=run_fold_baseline(dm,fold,seed,str(device),out_dir=baseline_out)
                bres=evaluate_fold(dm,fold,baseline["cal_target_probs"],baseline["cal_gap_probs"],
                    baseline["held_target_probs"],baseline["held_gap_probs"],"sm13_scalar_mix",seed)
                save_fold(bres,run_out/"baseline_predictions")
                baseline_summary={"metrics":bres["metrics"],"classes":bres["classes"]}
                l6_scores=baseline_out/f"l6_pooled_target_seed{seed}_fold{fold}_scores.npz"
                if l6_scores.exists():
                    z=np.load(l6_scores); l6={k:z[k] for k in ("cal_target_probs","held_target_probs")}
                else:
                    l6=run_fold_l6_target(dm,fold,seed,str(device),out_dir=baseline_out)
                l6res=evaluate_fold(dm,fold,l6["cal_target_probs"],baseline["cal_gap_probs"],
                    l6["held_target_probs"],baseline["held_gap_probs"],"l6_pooled_target_shared_sm13_gap",seed)
                save_fold(l6res,run_out/"l6_pooled_predictions")
                if args.baseline_only:
                    all_summaries.append({"seed":seed,"fold":fold,"baseline":baseline_summary,
                        "l6_pooled_target_shared_sm13_gap":{"metrics":l6res["metrics"],"classes":l6res["classes"]}})
                    print(json.dumps({"seed":seed,"fold":fold,"baseline_macro3_ap":bres["metrics"]["macro3_error_ap"],
                        "l6_pool_macro3_ap":l6res["metrics"]["macro3_error_ap"],"baseline_only":True}),flush=True)
                    del baseline,bres,l6,l6res; gc.collect()
                    if device.type=="cuda": torch.cuda.empty_cache()
                    fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
                    continue
                seed0=seed
                tm,ts=train_temporal("target",dm,split,seed0,device,target,gap)
                gm,gs=train_temporal("gap",dm,split,seed0+1000,device,target,gap)
                tc,th=predict_temporal(tm,"target",dm,split,device,target,gap)
                tgcal,tgheld=predict_temporal(gm,"gap",dm,split,device,target,gap)
                primary=evaluate_fold(dm,fold,tc,baseline["cal_gap_probs"],th,baseline["held_gap_probs"],
                    "temporal_cnn_target_shared_sm13_gap",seed)
                primary["target_train_stats"]=ts
                primary["config"]={**TRAINING["target"],"optimizer":"AdamW","target_window_frames":17,
                    "frame_dim":768,"architecture":"2-layer Conv1d with learned attention pooling and temporal mean",
                    "gap_branch":"shared fold-specific 13-layer SM13 gap baseline"}
                save_fold(primary,run_out/"target_only_predictions")
                full=evaluate_fold(dm,fold,tc,tgcal,th,tgheld,"temporal_cnn_target_gap_l6",seed)
                full["target_train_stats"]=ts; full["gap_train_stats"]=gs
                full["config"]={"target":TRAINING["target"],"gap":TRAINING["gap"],"optimizer":"AdamW","target_window_frames":17,
                    "gap_window_frames":10,"frame_dim":768,"architecture":"2-layer Conv1d with learned attention pooling and temporal mean"}
                save_fold(full,run_out/"full_temporal_predictions")
                torch.save({"target":tm.state_dict(),"gap":gm.state_dict()},
                    run_out/f"temporal_cnn_l6_seed{seed}_fold{fold}.pt")
                summary={"seed":seed,"fold":fold,"baseline":baseline_summary,
                         "l6_pooled_target_shared_sm13_gap":{"metrics":l6res["metrics"],"classes":l6res["classes"]},
                         "temporal_target_shared_sm13_gap":{"metrics":primary["metrics"],"classes":primary["classes"]},
                         "full_temporal_target_gap":{"metrics":full["metrics"],"classes":full["classes"]},
                         "temporal_train_target":ts,"temporal_train_gap":gs}
                all_summaries.append(summary)
                print(json.dumps({"seed":seed,"fold":fold,
                    "baseline_macro3_ap":bres["metrics"]["macro3_error_ap"],
                    "l6_pool_macro3_ap":l6res["metrics"]["macro3_error_ap"],
                    "temporal_target_macro3_ap":primary["metrics"]["macro3_error_ap"],
                    "temporal_full_macro3_ap":full["metrics"]["macro3_error_ap"]}),flush=True)
                del tm,gm,baseline,bres,l6,l6res,primary,full; gc.collect()
                if device.type=="cuda": torch.cuda.empty_cache()
                fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
    if args.baseline_only or args.smoke:
        merged=all_summaries
    else:
        merged_by={(r["seed"],r["fold"]):r for r in previous}
        merged_by.update({(r["seed"],r["fold"]):r for r in all_summaries})
        merged=[merged_by[k] for k in sorted(merged_by)]
    (run_out/"temporal_results.json").write_text(json.dumps({"status":"baseline_completed" if args.baseline_only else ("smoke_completed" if args.smoke else "completed"),"device":device_info,
      "seeds":list(SEEDS),"training_configs":TRAINING,"runs":merged},indent=2,allow_nan=False)+"\n")


if __name__=="__main__": main()

