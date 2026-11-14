#!/usr/bin/env python3
"""Fixed one-seed SM13 vowel/stop specialist ablation on TRAIN outer folds.

Each specialist uses the original SM13 head, weighted CE, optimizer, epoch
count, and frozen acoustic features. Only the training population is restricted
to one target manner. This tests competition in the global target-head fit.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/rq1_encoder_bench_01/common"))
from dataset_util import DatasetManager  # noqa: E402
from training_engine import seed_everything  # noqa: E402
from experiments.rq1_sm13_formal_train_01.agent_a_sm13_engine import ScalarMixingTargetHead  # noqa: E402
import experiments.rq1_sm13_followup_audit_01.run_audit as audit  # noqa: E402


FEAT_DIR = ROOT / "experiments/rq1_encoder_bench_01/features"
SEED = 101


def read_batch(layers: list[np.ndarray], ix: np.ndarray) -> np.ndarray:
    return np.stack([layer[ix] for layer in layers], axis=1)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    t0 = time.perf_counter()
    device = torch.device("cuda")
    dm = DatasetManager()
    data = audit.load_data()
    with np.load(ROOT / "experiments/rq1_method_select_02r/agent_a/predictions/predictions_sm13_seed101.npz") as p:
        if not np.array_equal(p["target_keys"], dm.target_keys):
            raise ValueError("baseline event order mismatch")
        baseline = p["probs_target_neural"][:, 1].copy()
    if not np.array_equal(baseline, data["s"]):
        raise ValueError("audit baseline mismatch")
    layers = [np.load(FEAT_DIR / f"target_acoustic_w2v2_960h_l{l}.npy", mmap_mode="r")
              for l in range(13)]
    if any(x.shape != (len(dm.yt), 2304) for x in layers):
        raise ValueError("feature shape mismatch")
    new_scores = {m: np.full(len(dm.yt), np.nan, dtype=np.float32) for m in ("Vowel", "Stop")}
    ckpt = HERE / "manner_specialist_checkpoints"
    ckpt.mkdir(exist_ok=True)
    fits = []
    weights = torch.tensor((1.0, 2.5, 4.0), dtype=torch.float32, device=device)

    for manner in ("Vowel", "Stop"):
        for fold in range(4):
            start = time.perf_counter()
            outer = dm.outer_masks(fold)
            fit = np.flatnonzero(outer["t_fit"] & (data["manner"] == manner))
            held = np.flatnonzero(outer["t_held"] & (data["manner"] == manner))
            seed_everything(SEED + 100 + fold)
            model = ScalarMixingTargetHead().to(device)
            if sum(p.numel() for p in model.parameters()) != 633361:
                raise ValueError("wrong SM13 architecture")
            opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            losses = []
            for epoch in range(12):
                model.train()
                perm = np.random.permutation(len(fit))
                total = 0.0
                batches = 0
                for i in range(0, len(fit), 512):
                    ix = fit[perm[i:i + 512]]
                    x = torch.from_numpy(read_batch(layers, ix)).to(device)
                    ref = torch.from_numpy(dm.target_refs[ix]).float().to(device)
                    y = torch.from_numpy(dm.yt[ix]).to(device)
                    opt.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(x, ref), y, weight=weights)
                    loss.backward()
                    opt.step()
                    total += float(loss.detach())
                    batches += 1
                losses.append(total / batches)
                print(f"{manner} fold={fold} epoch={epoch+1:02d} loss={losses[-1]:.5f}", flush=True)
            model.eval()
            with torch.inference_mode():
                for i in range(0, len(held), 1024):
                    ix = held[i:i + 1024]
                    x = torch.from_numpy(read_batch(layers, ix)).to(device)
                    ref = torch.from_numpy(dm.target_refs[ix]).float().to(device)
                    new_scores[manner][ix] = model.predict_proba(x, ref)[:, 1].cpu().numpy()
            torch.save(model.state_dict(), ckpt / f"{manner.lower()}_seed{SEED}_fold{fold}.pt")
            fits.append({"manner": manner, "fold": fold, "held_speakers": outer["held_speakers"],
                         "fit_rows": len(fit), "held_rows": len(held), "gold_fit_s": int((dm.yt[fit] == 1).sum()),
                         "gold_held_s": int((dm.yt[held] == 1).sum()), "epoch_losses": losses,
                         "seconds": time.perf_counter() - start})
            print(f"{manner} fold={fold} done {fits[-1]['seconds']:.1f}s", flush=True)
            del model, opt
            torch.cuda.empty_cache()

    combined = baseline.copy()
    for manner in ("Vowel", "Stop"):
        mask = data["manner"] == manner
        if not np.isfinite(new_scores[manner][mask]).all():
            raise ValueError(f"OOF coverage incomplete for {manner}")
        combined[mask] = new_scores[manner][mask]
    np.savez_compressed(HERE / "MANNER_SPECIALIST_OOF.npz", target_keys=dm.target_keys,
                        gold_target=dm.yt, score=combined)
    result = {"protocol": "One seed 101, 4 original outer speaker folds. Original 633361-parameter SM13 head and training settings, but one target manner per head; no DEV/TEST access.",
              "exploratory": True, "fits": fits, "seconds": time.perf_counter() - t0,
              "metrics": {}, "frontier": {}}
    all_speakers = set(dm.target_speaker)
    candidate = dict(data)
    candidate["s"] = combined
    for manner in ("Vowel", "Stop", "Other"):
        mask = data["manner"] == manner
        y = dm.yt[mask] == 1
        result["metrics"][manner] = {"support": int(y.sum()),
                                      "baseline_ap": float(average_precision_score(y, baseline[mask])),
                                      "specialist_ap": float(average_precision_score(y, combined[mask]))}
        if manner != "Other":
            result["frontier"][manner] = {
                "baseline_fwr_only": audit.select(data, all_speakers, manner, None),
                "specialist_fwr_only": audit.select(candidate, all_speakers, manner, None),
                "baseline_fwr_p50": audit.select(data, all_speakers, manner, 0.5),
                "specialist_fwr_p50": audit.select(candidate, all_speakers, manner, 0.5),
            }
    (HERE / "MANNER_SPECIALIST_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"metrics": result["metrics"], "frontier": result["frontier"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
