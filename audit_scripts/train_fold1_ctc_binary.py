#!/usr/bin/env python3
"""Reproducible Fold 1 CTC + strict binary MDD training (official test never loaded)."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from langfens_prep.dataset import L2ArcticMDDDataset
from langfens_prep.collate import LangfensMDDCollate
from langfens_prep.model import LangfensWav2Vec2MDD


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def restore_rng(s):
    random.setstate(s["python"]); np.random.set_state(s["numpy"]); torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(s["cuda"])


def average_precision_binary(labels, scores):
    """Average precision with tied scores grouped, matching sklearn's definition."""
    y=np.asarray(labels,dtype=np.int64); s=np.asarray(scores,dtype=np.float64)
    if y.size==0 or not np.isin(y,[0,1]).all(): raise ValueError("AP labels must be a non-empty binary vector")
    positives=int(y.sum())
    if positives==0: return float("nan")
    order=np.argsort(-s,kind="mergesort"); y=y[order]; s=s[order]
    ends=np.r_[np.flatnonzero(np.diff(s)!=0),len(s)-1]
    tp=np.cumsum(y)[ends]; fp=(ends+1)-tp
    precision=tp/(tp+fp); recall=tp/positives
    return float(np.sum(np.diff(np.r_[0.0,recall])*precision))


def ctc_batch_targets(batch):
    # Existing model contract accepts padded target matrix and per-item lengths.
    return batch["target_ctc"]


def step_model(model, batch, device, scaler, training=True):
    ni=batch["net_input"]
    args=dict(source=ni["source"].to(device), audio_lengths=ni["audio_lengths"].to(device),
              canonical_phone_indices=ni["canonical_phone_indices"].to(device),
              canonical_phone_lengths=ni["canonical_phone_lengths"].to(device),
              target_ctc=batch["target_ctc"].to(device), target_lengths=batch["target_lengths"].to(device),
              target_binary=batch["target_binary"].to(device))
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type=="cuda")):
        out=model(**args)  # Diagnosis labels are deliberately never supplied.
    if out.loss is None or not torch.isfinite(out.loss): raise FloatingPointError("non-finite two-head objective")
    return out


def checkpoint(path, model, optimizer, scheduler, scaler, epoch, updates, micro, best, history, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
                "scaler":scaler.state_dict(),"epoch":epoch,"successful_updates":updates,"micro_steps":micro,
                "best_ap":best,"history":history,"rng":rng_state(),"metadata":metadata},tmp)
    os.replace(tmp,path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/rq1_trn_02a_fold1.yaml")
    ap.add_argument("--resume",default=None)
    ap.add_argument("--max-train-batches",type=int,default=None,help="Bounded diagnostic run only; not a full experiment")
    args=ap.parse_args()
    try: import yaml
    except ImportError as e: raise RuntimeError("PyYAML is required to read explicit config") from e
    cfg=yaml.safe_load(Path(args.config).read_text())
    manifest=Path(cfg["manifest"]); digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    if digest != cfg["manifest_sha256"]: raise RuntimeError(f"split manifest checksum mismatch: {digest}")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type!="cuda": raise RuntimeError("Full training requires CUDA; refusing silent CPU execution")
    seed_all(cfg["seed"])
    train=L2ArcticMDDDataset(str(manifest),1,"train",policy="strict",zip_path=cfg["zip_path"])
    val=L2ArcticMDDDataset(str(manifest),1,"val",policy="strict",zip_path=cfg["zip_path"])
    if len(train)!=2399 or len(val)!=300: raise RuntimeError(f"Unexpected split sizes train={len(train)} val={len(val)}")
    collate=LangfensMDDCollate()
    train_loader=DataLoader(train,batch_size=cfg["batch_size"],shuffle=True,num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],collate_fn=collate,drop_last=False)
    val_loader=DataLoader(val,batch_size=cfg["batch_size"],shuffle=False,num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],collate_fn=collate,drop_last=False)
    model=LangfensWav2Vec2MDD.from_pretrained(cfg["pretrained_model"],revision=cfg["pretrained_revision"],
         ctc_weight=cfg["loss_weights"]["ctc"],binary_weight=cfg["loss_weights"]["binary"])
    trainable=model.configure_two_head_training()
    expected={id(p) for m in (model.ctc_head,model.cross_attention_module,model.binary_head) for p in m.parameters()}
    if {id(p) for p in trainable}!=expected: raise AssertionError("trainable parameter set differs from declared heads")
    model.to(device); model.train()
    optimizer=torch.optim.AdamW(trainable,lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    accum=cfg["gradient_accumulation_steps"]
    total_updates=math.ceil(len(train_loader)/accum)*cfg["max_epochs"]
    scheduler=get_cosine_schedule_with_warmup(optimizer,cfg["warmup_updates"],total_updates)
    # Real Fold 1 AMP probes showed the generic 65536 scale overflows fp16 gradients;
    # 4096 is the tested stable starting scale and remains adaptive.
    scaler=torch.amp.GradScaler("cuda",init_scale=4096.0,enabled=True)
    start_epoch=0; updates=0; micro=0; best=-float("inf"); history=[]
    meta={"git_sha":os.popen("git rev-parse HEAD").read().strip(),"split_sha256":digest,"config":cfg}
    if args.resume:
        ck=torch.load(args.resume,map_location="cpu",weights_only=False); model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"]); scheduler.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start_epoch=ck["epoch"]+1; updates=ck["successful_updates"]; micro=ck["micro_steps"]
        best=ck["best_ap"]; history=ck["history"]; restore_rng(ck["rng"]); model.train()
    patience=0; skipped_amp=0; consecutive_skips=0; outdir=Path(cfg["output_dir"]); outdir.mkdir(parents=True,exist_ok=True)
    for epoch in range(start_epoch,cfg["max_epochs"]):
        model.train(); optimizer.zero_grad(set_to_none=True); sums={"ctc":0.,"binary":0.}; n=0; t0=time.monotonic()
        for bi,batch in enumerate(train_loader):
            if args.max_train_batches is not None and bi>=args.max_train_batches: break
            result=step_model(model,batch,device,scaler)
            loss=result.loss/accum; scaler.scale(loss).backward(); micro+=1; n+=1
            sums["ctc"]+=result.loss_dict["ctc_loss"].detach().float().item()
            sums["binary"]+=result.loss_dict["binary_loss"].detach().float().item()
            if (bi+1)%accum==0 or bi+1==len(train_loader):
                scaler.unscale_(optimizer)
                for p in model.diagnosis_head.parameters():
                    if p.grad is not None: raise AssertionError("diagnosis gradient must remain absent")
                finite_grads=all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable)
                old_scale=scaler.get_scale()
                if finite_grads:
                    torch.nn.utils.clip_grad_norm_(trainable,cfg["gradient_clip_norm"])
                scaler.step(optimizer); scaler.update()
                if scaler.get_scale()>=old_scale:
                    if not finite_grads: raise FloatingPointError("non-finite gradient was not skipped by GradScaler")
                    updates+=1; scheduler.step(); consecutive_skips=0
                else:
                    skipped_amp+=1; consecutive_skips+=1
                    if consecutive_skips>=8: raise FloatingPointError("8 consecutive AMP-skipped optimizer attempts")
                optimizer.zero_grad(set_to_none=True)
            if args.max_train_batches is not None and bi+1>=args.max_train_batches: break
        if args.max_train_batches is not None: break
        model.eval(); scores=[]; labels=[]; ctc_blank=0; ctc_frames=0
        with torch.no_grad():
            for batch in val_loader:
                result=step_model(model,batch,device,scaler,False)
                logits=result.binary_logits.float().cpu(); target=batch["target_binary"]
                valid=target!=-100; scores.extend(torch.sigmoid(logits)[valid].tolist()); labels.extend(target[valid].tolist())
                pred=result.ctc_logits.argmax(-1); lengths=result.input_lengths
                for i,l in enumerate(lengths): ctc_blank+=(pred[i,:l]==0).sum().item(); ctc_frames+=l.item()
        apval=average_precision_binary(labels,scores) if len(set(labels))==2 else float("nan")
        row={"epoch":epoch+1,"micro_steps_total":micro,"successful_updates_total":updates,
             "train_ctc_loss":sums["ctc"]/max(n,1),"train_binary_loss":sums["binary"]/max(n,1),
             "validation_binary_ap":apval,"validation_ctc_blank_fraction":ctc_blank/max(ctc_frames,1),
             "skipped_amp_attempts_total":skipped_amp,"wall_seconds":time.monotonic()-t0}
        history.append(row); print(json.dumps(row),flush=True)
        checkpoint(outdir/"last.pt",model,optimizer,scheduler,scaler,epoch,updates,micro,best,history,meta)
        # Predeclared selection: highest validation AP among epochs with blank fraction < .99.
        if math.isfinite(apval) and row["validation_ctc_blank_fraction"]<0.99 and apval>best:
            best=apval; patience=0; checkpoint(outdir/"best.pt",model,optimizer,scheduler,scaler,epoch,updates,micro,best,history,meta)
        else: patience+=1
        if patience>=cfg["early_stopping_patience"]: break
    (outdir/"history.json").write_text(json.dumps(history,indent=2)+"\n")

if __name__=="__main__":
    try: main()
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}",flush=True); raise
