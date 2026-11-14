"""Fit the fold-local teacher, select soft pseudo-targets, and evaluate expansion."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/khoa/Projects/New-Method-Research")
HERE = ROOT / "experiments/sm13_three_directions_20261113/data_expansion"
SHARED = ROOT / "experiments/sm13_three_directions_20261113/shared"
sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(ROOT / "experiments/rq1_sm13_formal_train_01"))

import baseline  # noqa: E402
from protocol import SplitProtocol, evaluate_fold, save_fold  # noqa: E402
from agent_a_sm13_engine import ScalarMixingTargetHead  # noqa: E402
from dataset_util import DatasetManager  # noqa: E402

CONFIDENCE_MIN = 0.90
LABELLED_SHARE = 0.75
PSEUDO_SHARE = 0.25
CLASS_WEIGHTS = (1.0, 2.5, 4.0)
TARGET_TRAINING = baseline.TRAINING["target"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_baseline(dm, fold: int, seed: int, device: str):
    base_dir = SHARED / "baseline"
    stem = f"sm13_scalar_mix_seed{seed}_fold{fold}"
    ckpt = base_dir / f"{stem}.pt"
    scores_path = base_dir / f"{stem}_scores.npz"
    receipt_path = base_dir / f"{stem}.json"
    if not (ckpt.exists() and scores_path.exists() and receipt_path.exists()):
        return baseline.run_fold_baseline(dm, fold, seed, device=device)
    split = SplitProtocol(dm).split(fold)
    scores = np.load(scores_path)
    receipt = json.loads(receipt_path.read_text())
    return {
        "cal_target_probs": scores["cal_target_probs"],
        "held_target_probs": scores["held_target_probs"],
        "cal_gap_probs": scores["cal_gap_probs"],
        "held_gap_probs": scores["held_gap_probs"],
        "fit_speakers": list(split.fit_speakers),
        "calibration_speakers": list(split.calibration_speakers),
        "held_speakers": list(split.held_speakers),
        "target_train": receipt.get("target_train", {}),
        "gap_train": receipt.get("gap_train", {}),
        "elapsed_seconds": receipt.get("elapsed_seconds", None),
    }


@torch.inference_mode()
def teacher_probs(teacher, pseudo_x: np.ndarray, pseudo_refs: np.ndarray, device: torch.device):
    teacher.eval()
    results = []
    for start in range(0, len(pseudo_x), 512):
        x = torch.as_tensor(pseudo_x[start:start + 512], dtype=torch.float32, device=device)
        r = torch.as_tensor(pseudo_refs[start:start + 512], dtype=torch.float32, device=device)
        results.append(torch.softmax(teacher(x, r), dim=1).cpu().numpy())
    return np.concatenate(results, axis=0)


def train_candidate(dm, fold: int, seed: int, pseudo_x: np.ndarray, pseudo_refs: np.ndarray,
                    pseudo_targets: np.ndarray, y_fit: np.ndarray, fit_indices: np.ndarray,
                    device: torch.device, smoke: bool = False):
    set_seed(seed)
    maps = baseline._open_layers("target")
    refs = dm.target_refs
    model = ScalarMixingTargetHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TARGET_TRAINING["learning_rate"],
                                  weight_decay=TARGET_TRAINING["weight_decay"])
    class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)
    batch_size = TARGET_TRAINING["batch_size"]
    labelled_per_batch = int(batch_size * LABELLED_SHARE)
    pseudo_per_batch = batch_size - labelled_per_batch
    epochs = 1 if smoke else TARGET_TRAINING["epochs"]
    full_steps = math.ceil(len(fit_indices) / batch_size)
    steps_per_epoch = min(2, full_steps) if smoke else full_steps
    rng = np.random.default_rng(seed + 98031 + fold * 17)
    loss_history, labelled_exposure, pseudo_exposure = [], 0, 0

    for epoch in range(epochs):
        labeled_order = rng.permutation(fit_indices)
        pseudo_order = rng.permutation(len(pseudo_x))
        labeled_cursor = pseudo_cursor = 0
        model.train()
        total, n_total = 0.0, 0
        for _step in range(steps_per_epoch):
            if labeled_cursor + labelled_per_batch <= len(labeled_order):
                lix = labeled_order[labeled_cursor:labeled_cursor + labelled_per_batch]
                labeled_cursor += labelled_per_batch
            else:
                remaining = labeled_order[labeled_cursor:]
                need = labelled_per_batch - len(remaining)
                lix = np.concatenate((remaining, rng.choice(fit_indices, size=need, replace=True)))
                labeled_cursor = len(labeled_order)
            if pseudo_cursor + pseudo_per_batch <= len(pseudo_order):
                pix = pseudo_order[pseudo_cursor:pseudo_cursor + pseudo_per_batch]
                pseudo_cursor += pseudo_per_batch
            else:
                remaining = pseudo_order[pseudo_cursor:]
                need = pseudo_per_batch - len(remaining)
                pix = np.concatenate((remaining, rng.choice(len(pseudo_x), size=need, replace=True)))
                pseudo_cursor = len(pseudo_order)
            lac, lrf = baseline._batch_features(maps, refs, lix)
            pac = torch.as_tensor(pseudo_x[pix], dtype=torch.float32)
            prf = torch.as_tensor(pseudo_refs[pix], dtype=torch.float32)
            features = torch.cat((lac, pac), dim=0).to(device)
            phone_refs = torch.cat((lrf, prf), dim=0).to(device)
            gold = torch.as_tensor(y_fit[lix], dtype=torch.long, device=device)
            soft = torch.as_tensor(pseudo_targets[pix], dtype=torch.float32, device=device)
            logp = F.log_softmax(model(features, phone_refs), dim=1)
            one_hot = F.one_hot(gold, num_classes=3).float()
            targets = torch.cat((one_hot, soft), dim=0)
            weighted_targets = targets * class_weights.unsqueeze(0)
            per_row = -(weighted_targets * logp).sum(dim=1)
            loss = per_row.sum() / weighted_targets.sum().clamp_min(1e-12)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(features)
            n_total += len(features)
            labelled_exposure += len(lix)
            pseudo_exposure += len(pix)
        loss_history.append(total / max(1, n_total))
        print(f"fold={fold} seed={seed} epoch={epoch+1}/{epochs} loss={loss_history[-1]:.6f}", flush=True)
    return model, maps, refs, {
        "loss_by_epoch": loss_history,
        "trainable_params": sum(p.numel() for p in model.parameters()),
        "optimizer_updates": epochs * steps_per_epoch,
        "labeled_fit_exposures": labelled_exposure,
        "pseudo_exposures": pseudo_exposure,
        "labeled_share_per_batch": LABELLED_SHARE,
        "pseudo_share_per_batch": PSEUDO_SHARE,
        "pseudo_threshold": CONFIDENCE_MIN,
        "epochs": epochs,
        "smoke_only": smoke,
        "batch_size": batch_size,
        "learning_rate": TARGET_TRAINING["learning_rate"],
        "weight_decay": TARGET_TRAINING["weight_decay"],
        "class_weights": list(CLASS_WEIGHTS),
    }


@torch.inference_mode()
def predict_indices(model, maps, refs, indices: np.ndarray, device: torch.device):
    model.eval()
    rows = []
    batch_size = TARGET_TRAINING["batch_size"]
    for start in range(0, len(indices), batch_size):
        ix = indices[start:start + batch_size]
        ac, rf = baseline._batch_features(maps, refs, ix)
        rows.append(torch.softmax(model(ac.to(device), rf.to(device)), dim=1).cpu().numpy())
    return np.concatenate(rows, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(4))
    parser.add_argument("--seed", type=int, required=True, choices=(101, 202))
    parser.add_argument("--smoke", action="store_true", help="one-epoch, two-update integration smoke; never a benchmark result")
    args = parser.parse_args()
    run_started = time.perf_counter()
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = DatasetManager()
    split = SplitProtocol(dm).split(args.fold)
    receipt_path = HERE / "features" / f"fold{args.fold}_extraction_receipt.json"
    if not receipt_path.exists():
        raise FileNotFoundError(f"extract fit-speaker data first: {receipt_path}")
    extraction_meta = json.loads(receipt_path.read_text())
    if extraction_meta["fit_speakers"] != list(split.fit_speakers):
        raise AssertionError("extracted speaker fit set differs from protocol")
    feature_parts, reference_parts = [], []
    for speaker in split.fit_speakers:
        spk_cache = HERE / "features" / f"speaker_{speaker}_unlabeled_features.npz"
        spk_receipt = HERE / "features" / f"speaker_{speaker}_extraction_receipt.json"
        if not spk_cache.exists() or not spk_receipt.exists():
            raise FileNotFoundError(f"missing speaker feature cache: {spk_cache}")
        with np.load(spk_cache) as z:
            feature_parts.append(z["target_features"])
            reference_parts.append(z["target_refs"])
    pseudo_x = np.concatenate(feature_parts, axis=0)
    pseudo_refs_all = np.concatenate(reference_parts, axis=0)
    event_rows = [json.loads(line) for line in (HERE / "features" / f"fold{args.fold}_event_manifest.jsonl").read_text().splitlines() if line.strip()]
    if len(event_rows) != len(pseudo_x) or len(pseudo_refs_all) != len(pseudo_x):
        raise AssertionError("feature rows and event manifest are misaligned")

    output = HERE / ("smoke" if args.smoke else "results")
    output.mkdir(parents=True, exist_ok=True)
    base = ensure_baseline(dm, args.fold, args.seed, str(device))
    baseline_eval = evaluate_fold(
        dm, args.fold, base["cal_target_probs"], base["cal_gap_probs"],
        base["held_target_probs"], base["held_gap_probs"], "data_expansion_labelled_control", args.seed)
    baseline_eval_dir = output / "control_results" if args.smoke else HERE / "control_results"
    baseline_eval_dir.mkdir(parents=True, exist_ok=True)
    save_fold(baseline_eval, baseline_eval_dir)
    stem = f"sm13_scalar_mix_seed{args.seed}_fold{args.fold}"
    base_ckpt = torch.load(SHARED / "baseline" / f"{stem}.pt", map_location="cpu", weights_only=True)
    teacher = ScalarMixingTargetHead().to(device)
    teacher.load_state_dict(base_ckpt["target"])
    probs = teacher_probs(teacher, pseudo_x, pseudo_refs_all, device)
    confidence = probs.max(axis=1)
    selected = confidence >= CONFIDENCE_MIN
    all_class_counts = {name: int(np.sum(probs.argmax(axis=1) == i))
                        for i, name in enumerate(("CORRECT", "SUBSTITUTION", "OMISSION"))}
    if selected.sum() == 0:
        (output / f"threshold_feasibility_seed{args.seed}_fold{args.fold}.json").write_text(json.dumps({
            "status": "INFEASIBLE_AT_PREREGISTERED_CONFIDENCE_THRESHOLD",
            "confidence_threshold": CONFIDENCE_MIN,
            "teacher_probability_rows": int(len(confidence)),
            "teacher_confidence_max": float(confidence.max()) if len(confidence) else None,
            "teacher_confidence_quantiles": np.quantile(confidence, [0, .25, .5, .75, 1]).tolist() if len(confidence) else [],
            "teacher_predicted_class_counts_before_filter": all_class_counts,
            "seed": args.seed, "fold": args.fold,
            "fit_speakers": list(split.fit_speakers),
        }, indent=2, allow_nan=False) + "\n")
        raise RuntimeError("no pseudo-targets passed preregistered 0.90 confidence threshold")
    pseudo_x = pseudo_x[selected]
    pseudo_refs = pseudo_refs_all[selected]
    pseudo_targets = probs[selected]
    selected_rows = [dict(row, teacher_confidence=float(confidence[i]),
                          teacher_prediction=("CORRECT", "SUBSTITUTION", "OMISSION")[int(probs[i].argmax())],
                          teacher_probabilities=[float(x) for x in probs[i]],
                          pseudo_selected=bool(selected[i]))
                     for i, row in enumerate(event_rows) if selected[i]]
    pseudo_counts = {name: sum(r["teacher_prediction"] == name for r in selected_rows)
                     for name in ("CORRECT", "SUBSTITUTION", "OMISSION")}
    candidate, maps, refs, train_stats = train_candidate(
        dm, args.fold, args.seed, pseudo_x, pseudo_refs, pseudo_targets,
        dm.yt, np.flatnonzero(split.t_fit), device, smoke=args.smoke)

    cal_t = predict_indices(candidate, maps, refs, np.flatnonzero(split.t_cal), device)
    held_t = predict_indices(candidate, maps, refs, np.flatnonzero(split.t_held), device)
    tag = "data_expansion_smoke" if args.smoke else "data_expansion_target"
    result = evaluate_fold(dm, args.fold, cal_t, base["cal_gap_probs"], held_t,
                           base["held_gap_probs"], tag, args.seed)
    result["target_train"] = train_stats
    result["pseudo_data"] = {
        "source": extraction_meta["source_repo"],
        "source_revision": extraction_meta["source_revision"],
        "fit_speakers": list(split.fit_speakers),
        "calibration_speakers_excluded": list(split.calibration_speakers),
        "held_speakers_excluded": list(split.held_speakers),
        "selected_utterances": 50 * len(split.fit_speakers),
        "feature_rows_before_confidence_filter": int(len(confidence)),
        "pseudo_target_rows_retained": int(selected.sum()),
        "teacher_confidence_min": CONFIDENCE_MIN,
        "teacher_confidence_quantiles": np.quantile(confidence[selected], [0, .25, .5, .75, 1]).tolist(),
        "teacher_predicted_class_counts": pseudo_counts,
        "probabilities_are_fold_fit_teacher_pseudo_targets": True,
        "canonical_prompts_not_treated_as_actual_pronunciation_truth": True,
        "extraction_receipt": str(receipt_path.relative_to(ROOT)),
    }
    save_fold(result, output)
    torch.save({"target": candidate.state_dict()}, output / f"{stem}_{'smoke_' if args.smoke else ''}target.pt")
    np.savez_compressed(output / f"{stem}_pseudo_targets.npz",
                        event_id=np.asarray([f"L2_ARCTIC_UNLABELED|{r['speaker_id']}|{r['utterance_id']}|target|{r['position']}" for r in selected_rows]),
                        teacher_probabilities=probs[selected].astype(np.float32),
                        teacher_confidence=confidence[selected].astype(np.float32),
                        teacher_class=np.asarray([r["teacher_prediction"] for r in selected_rows]))
    (output / f"{stem}_pseudo_manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected_rows))
    (output / f"{stem}_training_receipt.json").write_text(json.dumps({
        "tag": "data_expansion_target", "seed": args.seed, "fold": args.fold,
        "fit_speakers": list(split.fit_speakers), "calibration_speakers": list(split.calibration_speakers),
        "held_speakers": list(split.held_speakers), "target_train": train_stats,
        "pseudo_data": result["pseudo_data"], "metrics": result["metrics"],
        "classes": result["classes"], "unchanged_insertion_reference": "shared fold baseline",
        "device": str(device), "elapsed_seconds": time.perf_counter() - run_started,
        "smoke_only": args.smoke,
    }, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"fold": args.fold, "seed": args.seed, "pseudo_rows": int(selected.sum()),
                      "predicted_class_counts": pseudo_counts, "metrics": result["metrics"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
