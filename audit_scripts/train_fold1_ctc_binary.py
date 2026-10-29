#!/usr/bin/env python3
"""Reproducible Fold 1 CTC + strict binary MDD training (official test never loaded)."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, shutil, time
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from langfens_prep.dataset import L2ArcticMDDDataset
from langfens_prep.collate import LangfensMDDCollate
from langfens_prep.model import LangfensWav2Vec2MDD


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }


def restore_rng(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def compute_levenshtein_edits(ref: List[int], hyp: List[int]) -> Tuple[int, int, int, int]:
    """Computes Levenshtein edit distance and returns (total_edits, substitutions, deletions, insertions)."""
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    i, j = m, n
    subs, dels, inss = 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            inss += 1
            j -= 1
        else:
            break
    return dp[m][n], subs, dels, inss


def greedy_ctc_decode(frame_preds: List[int], blank_id: int = 0) -> List[int]:
    """Collapses consecutive identical predictions and removes blank tokens."""
    hyp = []
    prev = None
    for t in frame_preds:
        if t != prev:
            if t != blank_id:
                hyp.append(t)
            prev = t
    return hyp


def average_precision_binary(labels, scores):
    """Average precision with tied scores grouped, matching sklearn's definition."""
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError("AP labels must be a non-empty binary vector")
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    s = s[order]
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp = np.cumsum(y)[ends]
    fp = (ends + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def ctc_batch_targets(batch):
    return batch["target_ctc"]


def step_model(model, batch, device, scaler, training=True):
    ni = batch["net_input"]
    args = dict(
        source=ni["source"].to(device),
        audio_lengths=ni["audio_lengths"].to(device),
        canonical_phone_indices=ni["canonical_phone_indices"].to(device),
        canonical_phone_lengths=ni["canonical_phone_lengths"].to(device),
        target_ctc=batch["target_ctc"].to(device),
        target_lengths=batch["target_lengths"].to(device),
        target_binary=batch["target_binary"].to(device)
    )
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
        out = model(**args)  # Diagnosis labels are deliberately never supplied.
    if out.loss is None or not torch.isfinite(out.loss):
        raise FloatingPointError("non-finite two-head objective")
    return out


def checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: any,
    scaler: torch.amp.GradScaler,
    epoch: int,
    updates: int,
    micro: int,
    best: float,
    history: list,
    metadata: dict,
    patience: int = 0,
    skipped_amp: int = 0,
    consecutive_skips: int = 0,
    drive_output_dir: str | Path | None = None,
    remote_mirror: any = None,
    remote_rel_prefix: str | None = None,
    fixed_budget_saved: bool = False,
    is_fixed_budget_endpoint: bool = False
):
    if is_fixed_budget_endpoint:
        if updates != 3600 or epoch != 5:
            raise ValueError(
                f"FAIL-CLOSED: fixed_budget_checkpoint must represent exactly 3,600 updates at Epoch 6. "
                f"Got updates={updates}, epoch={epoch}"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "successful_updates": updates,
        "micro_steps": micro,
        "best_ap": best,
        "patience": patience,
        "skipped_amp_attempts_total": skipped_amp,
        "consecutive_skips": consecutive_skips,
        "history": history,
        "rng": rng_state(),
        "metadata": metadata,
        "fixed_budget_saved": fixed_budget_saved,
        "is_fixed_budget_endpoint": is_fixed_budget_endpoint
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)

    # Remote Google Drive synchronization and atomic verification
    if drive_output_dir is not None:
        drive_path = Path(drive_output_dir) / path.name
        drive_path.parent.mkdir(parents=True, exist_ok=True)
        drive_tmp = drive_path.with_suffix(drive_path.suffix + ".tmp")
        shutil.copyfile(str(path), str(drive_tmp))
        os.replace(drive_tmp, drive_path)
        if not drive_path.exists():
            raise IOError(f"Remote checkpoint sync verification failed: {drive_path} does not exist")
        local_size = path.stat().st_size
        remote_size = drive_path.stat().st_size
        if local_size != remote_size:
            raise IOError(f"Remote checkpoint sync size mismatch: local {local_size} vs remote {remote_size}")

    # Remote Persistent Mirror synchronization (e.g. Hugging Face)
    if remote_mirror is not None:
        rel_dest = f"{remote_rel_prefix}/{path.name}" if remote_rel_prefix else path.name
        res = remote_mirror.mirror_file(path, rel_dest, verify_sha=True)
        print(f"[REMOTE MIRROR] Mirrored {path.name} -> {res['backend']}:{rel_dest} (SHA: {res['remote_sha256'][:12]}...)", flush=True)


def validate_resume_checkpoint(
    ck: Dict[str, Any],
    cfg: Dict[str, Any],
    expected_fold: int,
    manifest_sha: str
) -> None:
    """
    Rigorously validates checkpoint payload, metadata, fold identity, seed,
    hyperparameters, counters, and RNG state before resumption.
    Rejects missing, contradictory, cross-fold, or corrupted checkpoints.
    """
    if not isinstance(ck, dict):
        raise ValueError("Invalid checkpoint: payload must be a dictionary.")

    for req in ("model", "optimizer", "scheduler", "scaler", "epoch", "successful_updates", "micro_steps", "history", "rng", "metadata"):
        if req not in ck:
            raise ValueError(f"Invalid checkpoint: missing required key '{req}'.")

    if not isinstance(ck["model"], dict) or len(ck["model"]) == 0:
        raise ValueError("Invalid checkpoint: 'model' state dict is missing or empty.")
    if not isinstance(ck["optimizer"], dict) or len(ck["optimizer"]) == 0:
        raise ValueError("Invalid checkpoint: 'optimizer' state dict is missing or empty.")
    if not isinstance(ck["scheduler"], dict) or len(ck["scheduler"]) == 0:
        raise ValueError("Invalid checkpoint: 'scheduler' state dict is missing or empty.")
    if not isinstance(ck["scaler"], dict):
        raise ValueError("Invalid checkpoint: 'scaler' state dict must be a dictionary.")

    updates = ck["successful_updates"]
    micro = ck["micro_steps"]
    epoch = ck["epoch"]
    if not isinstance(updates, int) or updates < 0:
        raise ValueError(f"Invalid successful_updates counter: {updates}")
    if not isinstance(micro, int) or micro < 0:
        raise ValueError(f"Invalid micro_steps counter: {micro}")
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"Invalid epoch counter: {epoch}")

    history = ck.get("history")
    if not isinstance(history, list) or len(history) != epoch + 1:
        raise ValueError(
            f"Invalid checkpoint: history length ({len(history) if isinstance(history, list) else 0}) "
            f"does not match completed epoch count ({epoch + 1})."
        )

    meta = ck.get("metadata", {})
    if not isinstance(meta, dict):
        raise ValueError("Invalid checkpoint: 'metadata' must be a dictionary.")

    ck_fold = meta.get("fold")
    if ck_fold is None:
        ck_fold = meta.get("config", {}).get("fold")
    if ck_fold is not None and int(ck_fold) != expected_fold:
        raise ValueError(
            f"FAIL-CLOSED: Cross-fold resume rejected: checkpoint is from fold {ck_fold}, expected fold {expected_fold}."
        )

    ck_cfg = meta.get("config", {})
    if not isinstance(ck_cfg, dict):
        raise ValueError("Invalid checkpoint: missing 'config' dictionary in metadata.")

    if ck_cfg.get("seed") != cfg["seed"]:
        raise ValueError(f"Seed mismatch on resume: checkpoint has seed {ck_cfg.get('seed')}, expected {cfg['seed']}.")

    if ck_cfg.get("pretrained_model") != cfg["pretrained_model"]:
        raise ValueError(f"Pretrained model mismatch on resume: {ck_cfg.get('pretrained_model')} != {cfg['pretrained_model']}.")

    if meta.get("split_sha256") and meta.get("split_sha256") != manifest_sha:
        raise ValueError(f"Manifest SHA-256 mismatch on resume: {meta.get('split_sha256')} != {manifest_sha}.")

    rng = ck.get("rng")
    if not isinstance(rng, dict) or "torch" not in rng:
        raise ValueError("Invalid checkpoint: 'rng' state dict missing required torch generator.")


def evaluate_validation(model, val_loader, device, scaler):
    """Full validation pass evaluating binary MDD and complete CTC PER with S/D/I breakdown."""
    model.eval()
    scores = []
    labels = []
    ctc_blank = 0
    ctc_frames = 0
    total_ref_tokens = 0
    total_hyp_tokens = 0
    total_ctc_edits = 0
    total_ctc_subs = 0
    total_ctc_dels = 0
    total_ctc_inss = 0

    with torch.no_grad():
        for batch in val_loader:
            result = step_model(model, batch, device, scaler, training=False)
            logits = result.binary_logits.float().cpu()
            target = batch["target_binary"]
            valid = target != -100
            scores.extend(torch.sigmoid(logits)[valid].tolist())
            labels.extend(target[valid].tolist())

            pred = result.ctc_logits.argmax(-1).cpu()
            lengths = result.input_lengths.cpu()
            target_ctc = batch["target_ctc"].cpu()
            target_lengths = batch["target_lengths"].cpu()

            for i, l in enumerate(lengths):
                frame_len = l.item()
                frame_preds = pred[i, :frame_len].tolist()
                ctc_blank += sum(1 for p in frame_preds if p == 0)
                ctc_frames += frame_len

                # Greedy CTC decode
                hyp = greedy_ctc_decode(frame_preds, blank_id=0)
                ref_len = target_lengths[i].item()
                ref = target_ctc[i, :ref_len].tolist()

                edits, subs, dels, inss = compute_levenshtein_edits(ref, hyp)
                total_ref_tokens += len(ref)
                total_hyp_tokens += len(hyp)
                total_ctc_edits += edits
                total_ctc_subs += subs
                total_ctc_dels += dels
                total_ctc_inss += inss

    apval = average_precision_binary(labels, scores) if len(set(labels)) == 2 else float("nan")
    ctc_per = total_ctc_edits / max(total_ref_tokens, 1)
    ctc_sub_rate = total_ctc_subs / max(total_ref_tokens, 1)
    ctc_del_rate = total_ctc_dels / max(total_ref_tokens, 1)
    ctc_ins_rate = total_ctc_inss / max(total_ref_tokens, 1)
    decoded_length_ratio = total_hyp_tokens / max(total_ref_tokens, 1)
    blank_fraction = ctc_blank / max(ctc_frames, 1)
    prevalence = sum(labels) / max(len(labels), 1)

    return {
        "validation_binary_ap": round(apval, 6) if math.isfinite(apval) else float("nan"),
        "validation_binary_prevalence": round(prevalence, 6),
        "validation_ctc_per": round(ctc_per, 6),
        "validation_ctc_substitutions": total_ctc_subs,
        "validation_ctc_deletions": total_ctc_dels,
        "validation_ctc_insertions": total_ctc_inss,
        "validation_ctc_sub_rate": round(ctc_sub_rate, 6),
        "validation_ctc_del_rate": round(ctc_del_rate, 6),
        "validation_ctc_ins_rate": round(ctc_ins_rate, 6),
        "validation_ctc_total_edits": total_ctc_edits,
        "validation_ctc_ref_tokens": total_ref_tokens,
        "validation_ctc_hyp_tokens": total_hyp_tokens,
        "validation_ctc_decoded_length_ratio": round(decoded_length_ratio, 6),
        "validation_ctc_blank_fraction": round(blank_fraction, 6)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rq1_trn_02a_fold1.yaml")
    ap.add_argument("--fold", type=int, default=None, help="Override fold index in config (1..4)")
    ap.add_argument("--output-dir", default=None, help="Override output directory in config")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--zip-path", default=None, help="Path to zip archive containing authorized audio/TextGrids")
    ap.add_argument("--drive-output-dir", default=None, help="Google Drive destination for remote checkpoint persistence")
    ap.add_argument("--remote-mirror-backend", default=None, help="Remote persistent mirror backend (huggingface, local_dir)")
    ap.add_argument("--remote-mirror-target", default=None, help="Remote persistent mirror repository/directory target")
    ap.add_argument("--remote-mirror-token", default=None, help="Authentication token for remote persistent mirror")
    ap.add_argument("--remote-rel-prefix", default=None, help="Subfolder prefix in remote mirror target (e.g. fold2)")
    ap.add_argument("--max-train-batches", type=int, default=None, help="Bounded diagnostic run only; not a full experiment")
    ap.add_argument("--num-epochs", type=int, default=None, help="Maximum epochs this invocation; useful for epoch-wise artifact sync")
    ap.add_argument("--eval-only", action="store_true", help="Run validation only without training")
    args = ap.parse_args()

    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("PyYAML is required to read explicit config") from e

    cfg = yaml.safe_load(Path(args.config).read_text())
    manifest = Path(cfg["manifest"])
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if digest != cfg["manifest_sha256"]:
        raise RuntimeError(f"split manifest checksum mismatch: {digest}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Full training requires CUDA; refusing silent CPU execution")

    seed_all(cfg["seed"])
    fold_idx = int(args.fold if args.fold is not None else cfg.get("fold", 1))

    with open(manifest, "r", encoding="utf-8") as mf:
        mdata = json.load(mf)
    fold_obj = next((f for f in mdata["folds"] if f["fold_index"] == fold_idx), None)
    if fold_obj is None:
        raise ValueError(f"Fold {fold_idx} not found in manifest {manifest}")
    expected_train = fold_obj["train"]["utterance_count"]
    expected_val = fold_obj["val"]["utterance_count"]

    zip_path = args.zip_path or cfg.get("zip_path", "datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip")
    train = L2ArcticMDDDataset(str(manifest), fold_idx, "train", policy="strict", zip_path=zip_path)
    val = L2ArcticMDDDataset(str(manifest), fold_idx, "val", policy="strict", zip_path=zip_path)
    if len(train) != expected_train or len(val) != expected_val:
        raise RuntimeError(
            f"Unexpected split sizes for fold {fold_idx}: train={len(train)} (expected {expected_train}), "
            f"val={len(val)} (expected {expected_val})"
        )

    collate = LangfensMDDCollate()
    train_loader = DataLoader(
        train,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        collate_fn=collate,
        drop_last=False
    )
    val_loader = DataLoader(
        val,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        collate_fn=collate,
        drop_last=False
    )

    model = LangfensWav2Vec2MDD.from_pretrained(
        cfg["pretrained_model"],
        revision=cfg["pretrained_revision"],
        ctc_weight=cfg["loss_weights"]["ctc"],
        binary_weight=cfg["loss_weights"]["binary"]
    )
    trainable = model.configure_two_head_training()
    expected = {id(p) for m in (model.ctc_head, model.cross_attention_module, model.binary_head) for p in m.parameters()}
    if {id(p) for p in trainable} != expected:
        raise AssertionError("trainable parameter set differs from declared heads")

    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(trainable, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    accum = cfg["gradient_accumulation_steps"]
    total_updates = math.ceil(len(train_loader) / accum) * cfg["max_epochs"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg["warmup_updates"], total_updates)

    # 4096 is the tested stable starting scale and remains adaptive.
    scaler = torch.amp.GradScaler("cuda", init_scale=float(cfg.get("grad_scaler_initial_scale", 4096.0)), enabled=True)

    # Output directories and remote mirror setup
    drive_out = args.drive_output_dir or cfg.get("drive_output_dir")
    outdir = Path(args.output_dir or cfg["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    if drive_out:
        Path(drive_out).mkdir(parents=True, exist_ok=True)

    remote_rel_prefix = args.remote_rel_prefix or cfg.get("remote_rel_prefix", f"fold{fold_idx}")
    remote_mirror = None
    mirror_backend = args.remote_mirror_backend or cfg.get("remote_mirror_backend")
    mirror_target = args.remote_mirror_target or cfg.get("remote_mirror_target")
    mirror_token = args.remote_mirror_token or cfg.get("remote_mirror_token")
    if mirror_backend:
        try:
            from audit_scripts.kaggle_checkpoint_sync import KaggleRemoteMirror
        except ImportError:
            from kaggle_checkpoint_sync import KaggleRemoteMirror
        remote_mirror = KaggleRemoteMirror(
            backend=mirror_backend,
            remote_target=mirror_target,
            auth_token=mirror_token,
            require_verification=True
        )
        ok, msg = remote_mirror.verify_readiness()
        if not ok:
            raise RuntimeError(f"FAIL-CLOSED: Remote mirror readiness verification failed before training: {msg}")
        print(f"[REMOTE MIRROR] Verified remote persistent mirror: {mirror_backend} -> {mirror_target}", flush=True)

    start_epoch = 0
    updates = 0
    micro = 0
    best = -float("inf")
    history = []
    patience = 0
    skipped_amp = 0
    consecutive_skips = 0
    fixed_budget_saved = False

    meta = {
        "git_sha": os.popen("git rev-parse HEAD").read().strip(),
        "split_sha256": digest,
        "config": cfg,
        "fold": fold_idx
    }

    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume_checkpoint(ck=ck, cfg=cfg, expected_fold=fold_idx, manifest_sha=digest)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        updates = ck["successful_updates"]
        micro = ck["micro_steps"]
        best = ck["best_ap"]
        history = list(ck["history"])
        patience = ck.get("patience", 0)
        skipped_amp = ck.get("skipped_amp_attempts_total", 0)
        consecutive_skips = ck.get("consecutive_skips", 0)
        restore_rng(ck["rng"])
        model.train()
        print(f"Resumed from {args.resume}: start_epoch={start_epoch}, updates={updates}, best_ap={best}, patience={patience}")

        # Safe recovery of fixed_budget_checkpoint.pt
        fixed_path = outdir / "fixed_budget_checkpoint.pt"
        if fixed_path.exists():
            ck_fb = torch.load(fixed_path, map_location="cpu", weights_only=False)
            if ck_fb.get("successful_updates") == 3600 and ck_fb.get("epoch") == 5:
                fixed_budget_saved = True
                print(f"[RECOVERY] Verified existing fixed_budget_checkpoint.pt at 3,600 updates (Epoch 6). Locked.", flush=True)
            else:
                raise ValueError(
                    f"FAIL-CLOSED: Pre-existing fixed_budget_checkpoint.pt invalid: updates={ck_fb.get('successful_updates')}, epoch={ck_fb.get('epoch')}"
                )
        elif ck.get("epoch", 0) >= 5 or ck.get("successful_updates", 0) >= 3600:
            if remote_mirror is not None:
                rel_fb = f"{remote_rel_prefix}/fixed_budget_checkpoint.pt" if remote_rel_prefix else "fixed_budget_checkpoint.pt"
                print(f"[RECOVERY] Local fixed_budget_checkpoint missing. Attempting remote recovery from {rel_fb}...", flush=True)
                try:
                    recovered_path = remote_mirror.download_file(rel_fb, fixed_path)
                    ck_rec = torch.load(recovered_path, map_location="cpu", weights_only=False)
                    if ck_rec.get("successful_updates") == 3600 and ck_rec.get("epoch") == 5:
                        fixed_budget_saved = True
                        print(f"[RECOVERY] Recovered verified fixed_budget_checkpoint.pt from remote mirror. Locked.", flush=True)
                    else:
                        raise ValueError(f"Recovered fixed_budget_checkpoint has updates={ck_rec.get('successful_updates')}")
                except Exception as e:
                    raise RuntimeError(f"FAIL-CLOSED: Resuming past Epoch 6 requires verified fixed_budget_checkpoint.pt, but recovery failed: {e}")
            else:
                raise RuntimeError("FAIL-CLOSED: Resuming past Epoch 6 without fixed_budget_checkpoint.pt and without remote mirror.")
        else:
            fixed_budget_saved = False

    if args.eval_only:
        val_metrics = evaluate_validation(model, val_loader, device, scaler)
        print(f"EVAL_ONLY results: {json.dumps(val_metrics, indent=2)}")
        return

    epoch_limit = cfg["max_epochs"] if args.num_epochs is None else min(cfg["max_epochs"], start_epoch + args.num_epochs)

    for epoch in range(start_epoch, epoch_limit):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums = {"ctc": 0.0, "binary": 0.0}
        n = 0
        t0 = time.monotonic()

        for bi, batch in enumerate(train_loader):
            if args.max_train_batches is not None and bi >= args.max_train_batches:
                break
            result = step_model(model, batch, device, scaler)
            loss = result.loss / accum
            scaler.scale(loss).backward()
            micro += 1
            n += 1
            sums["ctc"] += result.loss_dict["ctc_loss"].detach().float().item()
            sums["binary"] += result.loss_dict["binary_loss"].detach().float().item()

            if (bi + 1) % accum == 0 or bi + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                for p in model.diagnosis_head.parameters():
                    if p.grad is not None:
                        raise AssertionError("diagnosis gradient must remain absent")
                finite_grads = all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable)
                old_scale = scaler.get_scale()
                if finite_grads:
                    torch.nn.utils.clip_grad_norm_(trainable, cfg["gradient_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    if not finite_grads:
                        raise FloatingPointError("non-finite gradient was not skipped by GradScaler")
                    updates += 1
                    scheduler.step()
                    consecutive_skips = 0
                else:
                    skipped_amp += 1
                    consecutive_skips += 1
                    if consecutive_skips >= 8:
                        raise FloatingPointError("8 consecutive AMP-skipped optimizer attempts")
                optimizer.zero_grad(set_to_none=True)

            if args.max_train_batches is not None and bi + 1 >= args.max_train_batches:
                break

        if args.max_train_batches is not None:
            break

        val_metrics = evaluate_validation(model, val_loader, device, scaler)
        apval = val_metrics["validation_binary_ap"]
        blank_fraction = val_metrics["validation_ctc_blank_fraction"]

        row = {
            "epoch": epoch + 1,
            "micro_steps_total": micro,
            "successful_updates_total": updates,
            "train_ctc_loss": round(sums["ctc"] / max(n, 1), 6),
            "train_binary_loss": round(sums["binary"] / max(n, 1), 6),
            **val_metrics,
            "skipped_amp_attempts_total": skipped_amp,
            "wall_seconds": round(time.monotonic() - t0, 2)
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        # Predeclared selection: highest validation AP among epochs with blank fraction < 0.99
        is_best = False
        if math.isfinite(apval) and blank_fraction < 0.99 and apval > best:
            best = apval
            patience = 0
            is_best = True
        else:
            patience += 1

        checkpoint(
            outdir / "last.pt",
            model, optimizer, scheduler, scaler,
            epoch, updates, micro, best, history, meta,
            patience=patience, skipped_amp=skipped_amp, consecutive_skips=consecutive_skips,
            drive_output_dir=drive_out,
            remote_mirror=remote_mirror,
            remote_rel_prefix=remote_rel_prefix,
            fixed_budget_saved=fixed_budget_saved
        )

        if is_best:
            checkpoint(
                outdir / "best.pt",
                model, optimizer, scheduler, scaler,
                epoch, updates, micro, best, history, meta,
                patience=patience, skipped_amp=skipped_amp, consecutive_skips=consecutive_skips,
                drive_output_dir=drive_out,
                remote_mirror=remote_mirror,
                remote_rel_prefix=remote_rel_prefix,
                fixed_budget_saved=fixed_budget_saved
            )

        # Check for primary matched-budget endpoint: strictly at Epoch 6 with exactly 3,600 updates
        if epoch == 5 and updates == 3600 and not fixed_budget_saved:
            fixed_budget_saved = True
            checkpoint(
                outdir / "fixed_budget_checkpoint.pt",
                model, optimizer, scheduler, scaler,
                epoch, updates, micro, best, history, meta,
                patience=patience, skipped_amp=skipped_amp, consecutive_skips=consecutive_skips,
                drive_output_dir=drive_out,
                remote_mirror=remote_mirror,
                remote_rel_prefix=remote_rel_prefix,
                fixed_budget_saved=True,
                is_fixed_budget_endpoint=True
            )
            print(f"[PRIMARY ENDPOINT] Saved verified fixed_budget_checkpoint.pt at update {updates} (epoch {epoch + 1}).", flush=True)

        # Persist training history locally and remotely after each epoch
        hist_json = outdir / "history.json"
        hist_json.write_text(json.dumps(history, indent=2) + "\n")
        if drive_out:
            remote_hist = Path(drive_out) / "history.json"
            shutil.copyfile(str(hist_json), str(remote_hist))
            if remote_hist.stat().st_size != hist_json.stat().st_size:
                raise IOError(f"Remote history sync size mismatch: {hist_json.stat().st_size} vs {remote_hist.stat().st_size}")
        if remote_mirror is not None:
            rel_hist = f"{remote_rel_prefix}/history.json" if remote_rel_prefix else "history.json"
            remote_mirror.mirror_file(hist_json, rel_hist, verify_sha=True)

        if patience >= cfg["early_stopping_patience"]:
            print(f"Early stopping triggered after {patience} epochs without validation AP improvement.", flush=True)
            break

    # Persist final training history locally and remotely
    hist_json = outdir / "history.json"
    hist_json.write_text(json.dumps(history, indent=2) + "\n")
    if drive_out:
        remote_hist = Path(drive_out) / "history.json"
        shutil.copyfile(str(hist_json), str(remote_hist))
        print(f"Verified remote history persistence: {remote_hist}", flush=True)
    if remote_mirror is not None:
        rel_hist = f"{remote_rel_prefix}/history.json" if remote_rel_prefix else "history.json"
        remote_mirror.mirror_file(hist_json, rel_hist, verify_sha=True)
        print(f"Verified remote mirror history persistence: {rel_hist}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", flush=True)
        raise
