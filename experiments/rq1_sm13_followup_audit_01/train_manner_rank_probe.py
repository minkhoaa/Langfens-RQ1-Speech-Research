#!/usr/bin/env python3
"""One-seed, four-outer-fold SM13 target-head ranking pilot on TRAIN only.

This is an exploratory, predeclared single-candidate screen. It does not select
a deployable threshold or access DEV/TEST data. The candidate adds a within-
manner hard-negative ranking term to the exact SM13 target-head CE training.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/rq1_encoder_bench_01/common"))
from dataset_util import DatasetManager  # noqa: E402
from training_engine import seed_everything  # noqa: E402
from experiments.rq1_sm13_formal_train_01.agent_a_sm13_engine import ScalarMixingTargetHead  # noqa: E402
import experiments.rq1_sm13_followup_audit_01.run_audit as audit  # noqa: E402


SEED = 101
LAMBDA = 0.1
HARD_NEG_K = 32
EPOCHS = 12
BATCH_SIZE = 512
CE_WEIGHTS = (1.0, 2.5, 4.0)
FEAT_DIR = ROOT / "experiments/rq1_encoder_bench_01/features"
BASE_PRED = ROOT / "experiments/rq1_method_select_02r/agent_a/predictions/predictions_sm13_seed101.npz"


def read_batch(layers: list[np.ndarray], index: np.ndarray) -> np.ndarray:
    return np.stack([layer[index] for layer in layers], axis=1)


def ranking_loss(logits: torch.Tensor, labels: torch.Tensor, manner: torch.Tensor) -> torch.Tensor:
    margin = logits[:, 1] - logits[:, 0]
    terms = []
    for code in (1, 2):  # vowel and stop, respectively
        pos = margin[(manner == code) & (labels == 1)]
        neg = margin[(manner == code) & (labels == 0)]
        if pos.numel() and neg.numel():
            hard = torch.topk(neg, k=min(HARD_NEG_K, neg.numel())).values
            terms.append(F.softplus(hard.unsqueeze(0) - pos.unsqueeze(1)).mean())
    return torch.stack(terms).mean() if terms else margin.new_zeros(())


def metric(y: np.ndarray, score: np.ndarray) -> dict:
    return {
        "support": int(y.sum()), "n": len(y),
        "ap": float(average_precision_score(y, score)),
        "auroc": float(roc_auc_score(y, score)),
    }


def main() -> None:
    t0 = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the matched SM13 pilot")
    device = torch.device("cuda")
    dm = DatasetManager()
    layers = [np.load(FEAT_DIR / f"target_acoustic_w2v2_960h_l{l}.npy", mmap_mode="r")
              for l in range(13)]
    if any(layer.shape != (len(dm.yt), 2304) for layer in layers):
        raise ValueError("SM13 acoustic cache shape mismatch")
    with np.load(BASE_PRED) as baseline_file:
        if not np.array_equal(baseline_file["target_keys"], dm.target_keys):
            raise ValueError("baseline prediction/manifest order mismatch")
        baseline_score = baseline_file["probs_target_neural"][:, 1].copy()
    d = audit.load_data()
    if not np.array_equal(d["s"], baseline_score):
        raise ValueError("audit/baseline score mismatch")
    manner = np.where(d["manner"] == "Vowel", 1, np.where(d["manner"] == "Stop", 2, 0)).astype(np.int64)
    new_score = np.full(len(dm.yt), np.nan, dtype=np.float32)
    ckpt_dir = HERE / "manner_rank_probe_checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    fold_records = []

    for fold in range(4):
        fold_start = time.perf_counter()
        outer = dm.outer_masks(fold)
        fit = np.flatnonzero(outer["t_fit"])
        held = np.flatnonzero(outer["t_held"])
        seed_everything(SEED + 100 + fold)  # exact outer seed schedule of SM13 reproduction
        model = ScalarMixingTargetHead().to(device)
        if sum(p.numel() for p in model.parameters()) != 633361:
            raise ValueError("wrong SM13 target-head architecture")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        weights = torch.tensor(CE_WEIGHTS, dtype=torch.float32, device=device)
        epoch_losses = []

        for epoch in range(EPOCHS):
            model.train()
            perm = np.random.permutation(len(fit))
            total = 0.0
            batches = 0
            for i in range(0, len(fit), BATCH_SIZE):
                ix = fit[perm[i:i + BATCH_SIZE]]
                x = torch.from_numpy(read_batch(layers, ix)).to(device)
                refs = torch.from_numpy(dm.target_refs[ix]).float().to(device)
                y = torch.from_numpy(dm.yt[ix]).to(device)
                m = torch.from_numpy(manner[ix]).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x, refs)
                loss = F.cross_entropy(logits, y, weight=weights) + LAMBDA * ranking_loss(logits, y, m)
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
                batches += 1
            epoch_losses.append(total / batches)
            print(f"fold={fold} epoch={epoch+1:02d} loss={epoch_losses[-1]:.5f}", flush=True)

        model.eval()
        with torch.inference_mode():
            for i in range(0, len(held), 1024):
                ix = held[i:i + 1024]
                x = torch.from_numpy(read_batch(layers, ix)).to(device)
                refs = torch.from_numpy(dm.target_refs[ix]).float().to(device)
                new_score[ix] = model.predict_proba(x, refs)[:, 1].cpu().numpy()
        torch.save(model.state_dict(), ckpt_dir / f"target_seed{SEED}_fold{fold}.pt")
        fold_records.append({
            "fold": fold, "held_speakers": outer["held_speakers"],
            "train_rows": len(fit), "held_rows": len(held),
            "epoch_losses": epoch_losses, "seconds": time.perf_counter() - fold_start,
        })
        print(f"fold={fold} done, seconds={fold_records[-1]['seconds']:.1f}", flush=True)
        del model, optimizer
        torch.cuda.empty_cache()

    if not np.isfinite(new_score).all():
        raise ValueError("OOF coverage incomplete")
    np.savez_compressed(HERE / "MANNER_RANK_PROBE_OOF.npz", target_keys=dm.target_keys,
                        gold_target=dm.yt, score=new_score)
    results = {"protocol": "SM13 target head, 4 outer speaker folds, seed 101; same CE/optimizer/epochs/architecture as baseline plus fixed within-manner top-32 pairwise softplus, lambda=0.1. No model or threshold selection on held speakers.",
               "exploratory": True, "seed": SEED, "lambda": LAMBDA, "hard_neg_k": HARD_NEG_K,
               "folds": fold_records, "seconds": time.perf_counter() - t0,
               "metrics": {}, "frontier": {}}
    for name in ("Vowel", "Stop", "Other"):
        mask = d["manner"] == name
        y = (dm.yt[mask] == 1).astype(int)
        results["metrics"][name] = {"baseline": metric(y, baseline_score[mask]),
                                     "candidate": metric(y, new_score[mask])}
    candidate = dict(d)
    candidate["s"] = new_score
    all_speakers = set(dm.target_speaker)
    for name in ("Vowel", "Stop"):
        results["frontier"][name] = {
            "baseline_fwr_only": audit.select(d, all_speakers, name, None),
            "candidate_fwr_only": audit.select(candidate, all_speakers, name, None),
            "baseline_fwr_p50": audit.select(d, all_speakers, name, 0.5),
            "candidate_fwr_p50": audit.select(candidate, all_speakers, name, 0.5),
        }
    (HERE / "MANNER_RANK_PROBE_RESULTS.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"metrics": results["metrics"], "frontier": results["frontier"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
