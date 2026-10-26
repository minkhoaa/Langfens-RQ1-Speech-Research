#!/usr/bin/env python3
"""
LANGFENS — RQ1-TRN-02A-EVAL-R2: Dedicated CPU-Compatible Evaluation Pipeline
Performs independent Fold 1 validation inference and acoustic controls on CPU without CUDA.

Key Guarantees:
1. Strict inference-only mode: model.eval() and torch.inference_mode().
2. Zero gradient or optimizer execution; verified parameter immutability.
3. Strict enforcement: split is 'val' only; official test data is never accessed.
4. Pinned pretrained revision and strict state_dict loading.
5. Paired matched vs. mismatched acoustic controls and train-only canonical prior baseline.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langfens_prep.collate import LangfensMDDCollate
from langfens_prep.dataset import L2ArcticMDDDataset
from langfens_prep.model import LangfensWav2Vec2MDD


def compute_sha256(path: Path | str) -> str:
    """Computes SHA-256 digest of a file in 64KB blocks."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_model_weights_fingerprint(model: torch.nn.Module) -> str:
    """Computes a SHA-256 fingerprint over all parameter tensor contents."""
    h = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        h.update(name.encode("utf-8"))
        h.update(param.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def greedy_ctc_decode(predictions: List[int], blank_id: int = 0) -> List[int]:
    """Collapses consecutive repeated tokens and strips the CTC blank."""
    hyp = []
    prev = None
    for t in predictions:
        if t != prev:
            if t != blank_id:
                hyp.append(t)
            prev = t
    return hyp


def compute_levenshtein_edits(ref: List[int], hyp: List[int]) -> Tuple[int, int, int, int]:
    """Computes total Levenshtein distance and decomposes edits into (S, D, I)."""
    n, m = len(ref), len(hyp)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    ops = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0=match, 1=sub, 2=del, 3=ins

    for i in range(1, n + 1):
        dp[i, 0] = i
        ops[i, 0] = 2  # Del
    for j in range(1, m + 1):
        dp[0, j] = j
        ops[0, j] = 3  # Ins

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost = 0
                sub_op = 0
            else:
                cost = 1
                sub_op = 1

            sub_cost = dp[i - 1, j - 1] + cost
            del_cost = dp[i - 1, j] + 1
            ins_cost = dp[i, j - 1] + 1

            best = min(sub_cost, del_cost, ins_cost)
            dp[i, j] = best

            if best == sub_cost:
                ops[i, j] = sub_op
            elif best == del_cost:
                ops[i, j] = 2
            else:
                ops[i, j] = 3

    # Backtrace
    i, j = n, m
    s, d, ins = 0, 0, 0
    while i > 0 or j > 0:
        op = ops[i, j]
        if op == 0:
            i -= 1
            j -= 1
        elif op == 1:
            s += 1
            i -= 1
            j -= 1
        elif op == 2:
            d += 1
            i -= 1
        elif op == 3:
            ins += 1
            j -= 1

    total = s + d + ins
    return total, s, d, ins


def average_precision_binary(labels: List[int] | np.ndarray, scores: List[float] | np.ndarray) -> float:
    """Average precision with tied scores grouped, matching scikit-learn's exact definition."""
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


def generate_derangement_indices(
    n: int,
    method: str = "cyclic",
    speaker_boundaries: Optional[List[int]] = None
) -> List[int]:
    """Generates a deterministic derangement of 0..n-1 with zero self-matches."""
    if n <= 1:
        raise ValueError(f"Derangement requires n >= 2, got {n}")

    if method == "cyclic":
        # Cyclic shift by 1: pi(i) = (i + 1) % n
        return [(i + 1) % n for i in range(n)]

    elif method == "within_speaker_cyclic":
        if not speaker_boundaries:
            return [(i + 1) % n for i in range(n)]
        # Filter internal boundaries strictly 0 < b < n
        valid_bounds = sorted([b for b in speaker_boundaries if 0 < b < n])
        if not valid_bounds:
            return [(i + 1) % n for i in range(n)]
        bounds = [0] + valid_bounds + [n]
        result = [0] * n
        for b_idx in range(len(bounds) - 1):
            start = bounds[b_idx]
            end = bounds[b_idx + 1]
            seg_len = end - start
            if seg_len <= 1:
                raise ValueError(f"Speaker segment length {seg_len} cannot be deranged without self-match")
            for local_i in range(seg_len):
                result[start + local_i] = start + ((local_i + 1) % seg_len)
        return result
    else:
        raise ValueError(f"Unknown derangement method: {method}")


def calculate_binary_metrics(
    labels: List[int],
    scores: List[float],
    thresholds: Optional[List[float]] = None
) -> Dict[str, Any]:
    """Calculates comprehensive binary classification metrics over threshold sweep."""
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    total_tokens = len(y)
    total_errors = int(y.sum())
    prevalence = total_errors / max(total_tokens, 1)

    ap = average_precision_binary(y, s) if len(set(y)) == 2 else float("nan")

    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]

    sweep = []
    best_f1 = -1.0
    best_th = 0.5
    best_op = {}

    for th in thresholds:
        preds = (s >= th).astype(np.int64)
        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        tn = int(((preds == 0) & (y == 0)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        point = {
            "threshold": round(float(th), 2),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }
        sweep.append(point)

        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            best_op = point

    # Operating point at 0.50
    op_05 = next((p for p in sweep if math.isclose(p["threshold"], 0.50, abs_tol=1e-3)), sweep[0])

    return {
        "average_precision": round(float(ap), 6) if math.isfinite(ap) else float("nan"),
        "prevalence": round(float(prevalence), 6),
        "total_phone_tokens": total_tokens,
        "total_error_tokens": total_errors,
        "operating_point_standard_05": op_05,
        "operating_point_optimal_f1": best_op,
        "threshold_calibration_sweep": sweep
    }


def calculate_ctc_metrics(
    references: List[List[int]],
    hypotheses: List[List[int]],
    blank_frames: int,
    total_frames: int
) -> Dict[str, Any]:
    """Calculates CTC greedy recognition metrics."""
    total_ref = sum(len(r) for r in references)
    total_hyp = sum(len(h) for h in hypotheses)
    total_subs = 0
    total_dels = 0
    total_inss = 0
    total_edits = 0

    for ref, hyp in zip(references, hypotheses):
        edits, s, d, ins = compute_levenshtein_edits(ref, hyp)
        total_edits += edits
        total_subs += s
        total_dels += d
        total_inss += ins

    per = total_edits / max(total_ref, 1)
    sub_rate = total_subs / max(total_ref, 1)
    del_rate = total_dels / max(total_ref, 1)
    ins_rate = total_inss / max(total_ref, 1)
    len_ratio = total_hyp / max(total_ref, 1)
    blank_frac = blank_frames / max(total_frames, 1)

    return {
        "greedy_per": round(float(per), 6),
        "substitutions": total_subs,
        "deletions": total_dels,
        "insertions": total_inss,
        "total_edits": total_edits,
        "total_ref_tokens": total_ref,
        "total_hyp_tokens": total_hyp,
        "substitution_rate": round(float(sub_rate), 6),
        "deletion_rate": round(float(del_rate), 6),
        "insertion_rate": round(float(ins_rate), 6),
        "decoded_length_ratio": round(float(len_ratio), 6),
        "blank_fraction": round(float(blank_frac), 6)
    }


def verify_checkpoint_integrity(checkpoint_path: Path | str, expected_sha: Optional[str] = None) -> Dict[str, Any]:
    """Verifies that the checkpoint exists, checks SHA256 if requested, and validates dictionary keys."""
    p = Path(checkpoint_path)
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    sha = compute_sha256(p)
    if expected_sha and sha != expected_sha:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {sha} != {expected_sha}")

    # Safe loading of local PyTorch checkpoint dictionary
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint must deserialize to a dictionary, got {type(ckpt)}")

    if "model" not in ckpt:
        raise KeyError("Checkpoint missing required 'model' state dict")

    return {
        "path": str(p),
        "sha256": sha,
        "epoch": ckpt.get("epoch"),
        "successful_updates": ckpt.get("successful_updates"),
        "best_ap": ckpt.get("best_ap"),
        "metadata": ckpt.get("metadata", {}),
        "model_keys_count": len(ckpt["model"])
    }


def build_eval_model(
    pretrained_model: str,
    pretrained_revision: str,
    state_dict: Dict[str, torch.Tensor],
    ctc_weight: float = 1.0,
    binary_weight: float = 1.0
) -> LangfensWav2Vec2MDD:
    """Instantiates the exact Langfens model, loads production weights strictly, and sets to eval mode."""
    model = LangfensWav2Vec2MDD.from_pretrained(
        pretrained_model,
        revision=pretrained_revision,
        ctc_weight=ctc_weight,
        binary_weight=binary_weight
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch! Missing: {missing}, Unexpected: {unexpected}")

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def fit_canonical_empirical_prior(train_dataset: Any) -> Dict[str, Any]:
    """Fits prompt-only empirical prior P(error | canonical_phone) on Fold 1 train items."""
    phone_errors = defaultdict(int)
    phone_totals = defaultdict(int)

    for item in train_dataset:
        can_indices = item["canonical_phone_indices"].tolist()
        labels = item["binary_labels"].tolist()
        for c, y in zip(can_indices, labels):
            if y != -100:
                phone_totals[c] += 1
                if y == 1:
                    phone_errors[c] += 1

    total_tokens = sum(phone_totals.values())
    total_errors = sum(phone_errors.values())
    overall_rate = total_errors / max(total_tokens, 1)

    phone_priors = {}
    for c, tot in phone_totals.items():
        phone_priors[int(c)] = phone_errors[c] / tot

    return {
        "total_train_tokens": total_tokens,
        "total_train_errors": total_errors,
        "overall_train_rate": overall_rate,
        "phone_priors": phone_priors
    }


def evaluate_canonical_prior(val_dataset: Any, prior_data: Dict[str, Any]) -> Dict[str, Any]:
    """Applies canonical empirical prior to validation items and calculates metrics."""
    labels = []
    scores = []
    priors_map = prior_data["phone_priors"]
    default_rate = prior_data["overall_train_rate"]

    for item in val_dataset:
        can_indices = item["canonical_phone_indices"].tolist()
        lbls = item["binary_labels"].tolist()
        for c, y in zip(can_indices, lbls):
            if y != -100:
                labels.append(int(y))
                scores.append(float(priors_map.get(int(c), default_rate)))

    metrics = calculate_binary_metrics(labels, scores)
    metrics["scores"] = scores
    metrics["labels"] = labels
    metrics["total_tokens"] = len(labels)
    return metrics


def run_utterance_clustered_bootstrap(
    utterance_results: List[Dict[str, Any]],
    n_bootstraps: int = 1000,
    seed: int = 1337
) -> Dict[str, Any]:
    """Performs cluster-bootstrap at the utterance level to obtain 95% CIs for matched vs mismatched AP."""
    rng = np.random.RandomState(seed)
    n_utts = len(utterance_results)
    if n_utts == 0:
        return {}

    matched_aps = []
    mismatched_aps = []
    delta_aps = []

    for _ in range(n_bootstraps):
        boot_idx = rng.choice(n_utts, size=n_utts, replace=True)
        boot_y = []
        boot_matched_s = []
        boot_mismatched_s = []

        for idx in boot_idx:
            res = utterance_results[idx]
            boot_y.extend(res["labels"])
            boot_matched_s.extend(res["matched_scores"])
            boot_mismatched_s.extend(res["mismatched_scores"])

        if len(set(boot_y)) == 2:
            m_ap = average_precision_binary(boot_y, boot_matched_s)
            mm_ap = average_precision_binary(boot_y, boot_mismatched_s)
            matched_aps.append(m_ap)
            mismatched_aps.append(mm_ap)
            delta_aps.append(m_ap - mm_ap)

    def ci95(arr):
        if not arr:
            return None
        return [round(float(np.percentile(arr, 2.5)), 6), round(float(np.percentile(arr, 97.5)), 6)]

    return {
        "n_bootstraps": len(matched_aps),
        "matched_ap_mean": round(float(np.mean(matched_aps)), 6) if matched_aps else None,
        "matched_ap_ci95": ci95(matched_aps),
        "mismatched_ap_mean": round(float(np.mean(mismatched_aps)), 6) if mismatched_aps else None,
        "mismatched_ap_ci95": ci95(mismatched_aps),
        "delta_ap_mean": round(float(np.mean(delta_aps)), 6) if delta_aps else None,
        "delta_ap_ci95": ci95(delta_aps)
    }


def main():
    parser = argparse.ArgumentParser(description="RQ1-TRN-02A CPU-Only Evaluation & Replay")
    parser.add_argument("--config", default="configs/rq1_trn_02a_fold1.yaml", help="Path to experiment config")
    parser.add_argument("--checkpoint", default="checkpoints/RQ1-TRN-02A/recovered/best.pt", help="Path to checkpoint")
    parser.add_argument("--zip-path", default=None, help="Path to authorized train/val audio zip")
    parser.add_argument("--device", default="cpu", help="Device (cpu strictly enforced/default)")
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size (default 1)")
    parser.add_argument("--output-dir", default="manifests/RQ1-TRN-02A/evaluation_reconciliation", help="Output directory")
    parser.add_argument("--max-eval-utterances", type=int, default=None, help="Diagnostic/smoke run size")
    parser.add_argument("--run-controls", action="store_true", default=True, help="Run matched/mismatched and baseline controls")
    parser.add_argument("--split", default="val", help="Evaluation split (must be 'val')")
    args = parser.parse_args()

    # Safety: strict prohibition against accessing official test split
    if args.split != "val":
        raise ValueError(f"Safety violation: evaluation split must be 'val', requested '{args.split}'")

    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("PyYAML required to read config") from e

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text())

    manifest_path = Path(cfg["manifest"])
    manifest_sha = compute_sha256(manifest_path)
    if manifest_sha != cfg["manifest_sha256"]:
        raise ValueError(f"Manifest checksum mismatch: {manifest_sha} != {cfg['manifest_sha256']}")

    zip_path = args.zip_path or cfg.get("zip_path", "datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip")
    if not Path(zip_path).is_file():
        raise FileNotFoundError(f"Authorized dataset zip not found: {zip_path}")
    zip_sha = compute_sha256(zip_path)

    # Checkpoint verification
    ckpt_info = verify_checkpoint_integrity(args.checkpoint)
    print(f"[CHECKPOINT] Loaded: {args.checkpoint} (SHA-256: {ckpt_info['sha256']})")
    print(f"[CHECKPOINT] Epoch: {ckpt_info['epoch']} (0-based, corresponds to epoch {ckpt_info['epoch'] + 1})")
    print(f"[CHECKPOINT] Successful updates: {ckpt_info['successful_updates']}, Best AP: {ckpt_info['best_ap']}")

    device = torch.device(args.device)
    print(f"[DEVICE] Executing on: {device}")

    # Build model and verify initial fingerprint
    raw_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_eval_model(
        pretrained_model=cfg["pretrained_model"],
        pretrained_revision=cfg["pretrained_revision"],
        state_dict=raw_ckpt["model"],
        ctc_weight=cfg["loss_weights"]["ctc"],
        binary_weight=cfg["loss_weights"]["binary"]
    ).to(device)

    initial_fingerprint = get_model_weights_fingerprint(model)
    print(f"[INTEGRITY] Pre-evaluation model fingerprint: {initial_fingerprint}")

    # Load validation dataset
    val_dataset = L2ArcticMDDDataset(str(manifest_path), fold=1, split="val", policy="strict", zip_path=zip_path)
    if len(val_dataset) != 300:
        raise ValueError(f"Expected 300 validation utterances, found {len(val_dataset)}")

    eval_count = min(len(val_dataset), args.max_eval_utterances) if args.max_eval_utterances else len(val_dataset)
    print(f"[DATASET] Evaluating {eval_count} utterances (Fold 1 validation: NJS & RRBI)...")

    # Evaluate validation utterances (Condition 1: Matched Audio)
    matched_scores = []
    matched_labels = []
    ctc_refs = []
    ctc_hyps = []
    ctc_blank_frames = 0
    ctc_total_frames = 0

    per_speaker_matched = defaultdict(lambda: {"scores": [], "labels": [], "refs": [], "hyps": [], "blank": 0, "frames": 0, "utts": 0})
    utterance_results = []

    process = psutil.Process(os.getpid())
    t0_eval = time.monotonic()
    peak_ram_bytes = process.memory_info().rss

    # Evaluate utterance by utterance (or configurable batch size)
    # Using batch_size=1 gives exact isolation and matches unpadded single-utterance inference
    collate_fn = LangfensMDDCollate()
    loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    with torch.inference_mode():
        processed_utts = 0
        for batch in loader:
            if args.max_eval_utterances and processed_utts >= args.max_eval_utterances:
                break

            ni = batch["net_input"]
            args_fwd = dict(
                source=ni["source"].to(device),
                audio_lengths=ni["audio_lengths"].to(device),
                canonical_phone_indices=ni["canonical_phone_indices"].to(device),
                canonical_phone_lengths=ni["canonical_phone_lengths"].to(device)
            )

            out = model(**args_fwd)

            # Check finiteness
            if not torch.isfinite(out.binary_logits).all():
                raise FloatingPointError("Non-finite binary logits encountered during evaluation")
            if not torch.isfinite(out.ctc_logits).all():
                raise FloatingPointError("Non-finite CTC logits encountered during evaluation")

            batch_scores = torch.sigmoid(out.binary_logits.float()).cpu()
            target_binary = batch["target_binary"].cpu()
            pred_ctc = out.ctc_logits.argmax(-1).cpu()
            input_lengths = out.input_lengths.cpu()
            target_ctc = batch["target_ctc"].cpu()
            target_lengths = batch["target_lengths"].cpu()

            bs = batch_scores.shape[0]
            for b in range(bs):
                if args.max_eval_utterances and processed_utts >= args.max_eval_utterances:
                    break

                rec_id = batch["record_ids"][b]
                speaker = rec_id.split("_")[1]

                valid_mask = target_binary[b] != -100
                u_scores = batch_scores[b][valid_mask].tolist()
                u_labels = target_binary[b][valid_mask].tolist()

                matched_scores.extend(u_scores)
                matched_labels.extend(u_labels)

                # Speaker tracking
                per_speaker_matched[speaker]["scores"].extend(u_scores)
                per_speaker_matched[speaker]["labels"].extend(u_labels)
                per_speaker_matched[speaker]["utts"] += 1

                # CTC tracking
                flen = input_lengths[b].item()
                fpreds = pred_ctc[b, :flen].tolist()
                u_blank = sum(1 for p in fpreds if p == 0)
                ctc_blank_frames += u_blank
                ctc_total_frames += flen
                per_speaker_matched[speaker]["blank"] += u_blank
                per_speaker_matched[speaker]["frames"] += flen

                hyp = greedy_ctc_decode(fpreds, blank_id=0)
                rlen = target_lengths[b].item()
                ref = target_ctc[b, :rlen].tolist()
                ctc_refs.append(ref)
                ctc_hyps.append(hyp)
                per_speaker_matched[speaker]["refs"].append(ref)
                per_speaker_matched[speaker]["hyps"].append(hyp)

                utterance_results.append({
                    "record_id": rec_id,
                    "speaker": speaker,
                    "labels": u_labels,
                    "matched_scores": u_scores,
                    "matched_binary_logits": out.binary_logits[b][valid_mask].cpu().tolist(),
                    "ctc_ref": ref,
                    "ctc_hyp": hyp,
                    "audio_length_samples": int(ni["audio_lengths"][b].item()),
                    "canonical_length": len(u_labels)
                })

                current_ram = process.memory_info().rss
                if current_ram > peak_ram_bytes:
                    peak_ram_bytes = current_ram

                processed_utts += 1

    total_eval_time = time.monotonic() - t0_eval
    per_utt_time = total_eval_time / max(processed_utts, 1)
    peak_ram_mb = peak_ram_bytes / (1024 * 1024)
    print(f"[EVALUATION] Completed {processed_utts} utterances in {total_eval_time:.2f}s ({per_utt_time:.3f}s/utt, Peak RAM: {peak_ram_mb:.1f} MB)")

    # Assert parameter immutability
    post_fingerprint = get_model_weights_fingerprint(model)
    if initial_fingerprint != post_fingerprint:
        raise AssertionError("Model parameters were modified during evaluation!")
    print(f"[INTEGRITY] Post-evaluation model fingerprint verified: {post_fingerprint} (UNCHANGED)")

    # Compute matched metrics
    matched_binary_metrics = calculate_binary_metrics(matched_labels, matched_scores)
    matched_ctc_metrics = calculate_ctc_metrics(ctc_refs, ctc_hyps, ctc_blank_frames, ctc_total_frames)

    # Per-speaker matched breakdown
    per_speaker_metrics = {}
    for spk, data in per_speaker_matched.items():
        spk_bin = calculate_binary_metrics(data["labels"], data["scores"])
        spk_ctc = calculate_ctc_metrics(data["refs"], data["hyps"], data["blank"], data["frames"])
        per_speaker_metrics[spk] = {
            "utterances": data["utts"],
            "phone_tokens": spk_bin["total_phone_tokens"],
            "error_tokens": spk_bin["total_error_tokens"],
            "prevalence": spk_bin["prevalence"],
            "binary_ap": spk_bin["average_precision"],
            "f1_at_05": spk_bin["operating_point_standard_05"]["f1"],
            "precision_at_05": spk_bin["operating_point_standard_05"]["precision"],
            "recall_at_05": spk_bin["operating_point_standard_05"]["recall"],
            "ctc_per": spk_ctc["greedy_per"],
            "ctc_sub_rate": spk_ctc["substitution_rate"],
            "ctc_del_rate": spk_ctc["deletion_rate"],
            "ctc_ins_rate": spk_ctc["insertion_rate"],
            "ctc_decoded_length_ratio": spk_ctc["decoded_length_ratio"],
            "ctc_blank_fraction": spk_ctc["blank_fraction"]
        }

    # Acoustic Controls
    controls_results = {}
    if args.run_controls:
        print("[CONTROLS] Running acoustic contribution controls...")

        # Condition 2: Mismatched Audio
        # Create deterministic derangement within speakers (or across set)
        n_eval = len(utterance_results)
        # Check speaker boundary
        spk_counts = [sum(1 for u in utterance_results if u["speaker"] == "NJS")]
        derange_indices = generate_derangement_indices(
            n_eval,
            method="within_speaker_cyclic" if len(spk_counts) == 1 and spk_counts[0] < n_eval else "cyclic",
            speaker_boundaries=spk_counts if spk_counts[0] < n_eval else None
        )

        mismatched_scores = []
        mismatched_logits = []
        per_speaker_mismatched = defaultdict(lambda: {"scores": [], "labels": []})

        with torch.inference_mode():
            for i, d_idx in enumerate(derange_indices):
                orig_item = val_dataset[i]
                permuted_audio_item = val_dataset[d_idx]

                # Prompt and labels come strictly from orig_item
                can_indices = orig_item["canonical_phone_indices"].unsqueeze(0).to(device)
                can_len = torch.tensor([len(orig_item["canonical_phone_indices"])], device=device)

                # Audio waveform comes strictly from permuted_audio_item
                mismatch_audio = permuted_audio_item["audio_waveform_16k"].unsqueeze(0).to(device)
                mismatch_len = torch.tensor([len(permuted_audio_item["audio_waveform_16k"])], device=device)

                out_mm = model(
                    source=mismatch_audio,
                    audio_lengths=mismatch_len,
                    canonical_phone_indices=can_indices,
                    canonical_phone_lengths=can_len
                )

                lbls = orig_item["binary_labels"]
                valid = lbls != -100
                mm_sc = torch.sigmoid(out_mm.binary_logits[0]).cpu()[valid].tolist()
                mm_lg = out_mm.binary_logits[0].cpu()[valid].tolist()

                mismatched_scores.extend(mm_sc)
                mismatched_logits.extend(mm_lg)
                utterance_results[i]["mismatched_scores"] = mm_sc
                utterance_results[i]["mismatched_binary_logits"] = mm_lg

                spk = utterance_results[i]["speaker"]
                per_speaker_mismatched[spk]["scores"].extend(mm_sc)
                per_speaker_mismatched[spk]["labels"].extend(utterance_results[i]["labels"])

        mismatched_binary_metrics = calculate_binary_metrics(matched_labels, mismatched_scores)

        # Condition 3: Canonical Empirical Prior Baseline (learned strictly from Fold 1 Train)
        print("[CONTROLS] Fitting Canonical-Only Empirical Prior strictly on Fold 1 Train (2,399 utterances)...")
        train_dataset = L2ArcticMDDDataset(str(manifest_path), fold=1, split="train", policy="strict", zip_path=zip_path)
        canonical_prior_data = fit_canonical_empirical_prior(train_dataset)
        canonical_prior_metrics = evaluate_canonical_prior(
            [val_dataset[i] for i in range(processed_utts)],
            canonical_prior_data
        )

        # Condition 4: Constant Prevalence Baseline
        prev_val = matched_binary_metrics["prevalence"]
        prev_scores = [prev_val] * len(matched_labels)
        prevalence_metrics = {
            "prevalence": prev_val,
            "average_precision": prev_val
        }

        # Paired comparisons across matched vs mismatched
        m_s = np.asarray(matched_scores, dtype=np.float64)
        mm_s = np.asarray(mismatched_scores, dtype=np.float64)
        score_diff = m_s - mm_s

        # Prediction agreement at tau=0.50 and tau=optimal (0.20)
        agree_05 = float(( (m_s >= 0.5) == (mm_s >= 0.5) ).mean())
        agree_opt = float(( (m_s >= 0.2) == (mm_s >= 0.2) ).mean())

        # Sensitivity: fraction of tokens where |matched - mismatched| > threshold
        abs_diff = np.abs(score_diff)
        frac_gt_001 = float((abs_diff > 0.01).mean())
        frac_gt_005 = float((abs_diff > 0.05).mean())
        frac_gt_010 = float((abs_diff > 0.10).mean())

        # Utterance-clustered bootstrap analysis
        print("[CONTROLS] Running 1,000 utterance-clustered bootstrap resamples...")
        boot_results = run_utterance_clustered_bootstrap(utterance_results, n_bootstraps=1000, seed=cfg.get("seed", 1337))

        # Multiple within-speaker derangement comparison
        print("[CONTROLS] Evaluating multiple within-speaker permutations for sensitivity analysis...")
        def make_within_speaker_perm(mode: str, boundaries: List[int], total: int) -> List[int]:
            bounds = [0] + boundaries + [total]
            res = [0] * total
            for b_idx in range(len(bounds) - 1):
                start, end = bounds[b_idx], bounds[b_idx + 1]
                seg_len = end - start
                for local_i in range(seg_len):
                    if mode == "shift_1":
                        target_i = (local_i + 1) % seg_len
                    elif mode == "shift_2":
                        target_i = (local_i + 2) % seg_len
                    elif mode == "shift_75":
                        target_i = (local_i + seg_len // 2) % seg_len
                    elif mode == "reverse":
                        target_i = (seg_len - 1) - local_i
                    else:
                        raise ValueError(f"Unknown mode {mode}")
                    res[start + local_i] = start + target_i
            return res

        permutation_sensitivity = {}
        with torch.inference_mode():
            for p_mode in ["shift_1", "shift_2", "shift_75", "reverse"]:
                p_indices = make_within_speaker_perm(p_mode, spk_counts if spk_counts[0] < n_eval else [], n_eval)
                p_scores_all = []
                p_scores_njs = []
                p_scores_rrbi = []

                for idx, p_donor in enumerate(p_indices):
                    orig_item = val_dataset[idx]
                    donor_item = val_dataset[p_donor]
                    can_idx = orig_item["canonical_phone_indices"].unsqueeze(0).to(device)
                    can_l = torch.tensor([len(orig_item["canonical_phone_indices"])], device=device)
                    audio_w = donor_item["audio_waveform_16k"].unsqueeze(0).to(device)
                    audio_l = torch.tensor([len(donor_item["audio_waveform_16k"])], device=device)

                    out_p = model(source=audio_w, audio_lengths=audio_l, canonical_phone_indices=can_idx, canonical_phone_lengths=can_l)
                    v_mask = orig_item["binary_labels"] != -100
                    sc_p = torch.sigmoid(out_p.binary_logits[0]).cpu()[v_mask].tolist()
                    p_scores_all.extend(sc_p)

                    if orig_item["speaker_id"] == "NJS":
                        p_scores_njs.extend(sc_p)
                    else:
                        p_scores_rrbi.extend(sc_p)

                ap_p_all = average_precision_binary(matched_labels, p_scores_all)
                ap_p_njs = average_precision_binary(per_speaker_matched["NJS"]["labels"], p_scores_njs)
                ap_p_rrbi = average_precision_binary(per_speaker_matched["RRBI"]["labels"], p_scores_rrbi)
                permutation_sensitivity[p_mode] = {
                    "overall_ap": round(float(ap_p_all), 6),
                    "njs_ap": round(float(ap_p_njs), 6),
                    "rrbi_ap": round(float(ap_p_rrbi), 6),
                    "delta_ap_over_matched": round(float(matched_binary_metrics["average_precision"] - ap_p_all), 6)
                }

        # Fine-grained threshold sweep (0.01 step) for fair baseline comparison
        print("[CONTROLS] Computing unified threshold sweep (step 0.01) for neural vs canonical prior...")
        def fine_sweep(y_true, s_pred):
            y_arr = np.array(y_true)
            s_arr = np.array(s_pred)
            best_f1, best_th, best_p, best_r = -1.0, None, None, None
            th_results = {}
            for th_f in np.arange(0.01, 1.0, 0.01):
                th_f = round(float(th_f), 2)
                pred = (s_arr >= th_f).astype(int)
                tp = int(((pred == 1) & (y_arr == 1)).sum())
                fp = int(((pred == 1) & (y_arr == 0)).sum())
                fn = int(((pred == 0) & (y_arr == 1)).sum())
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                th_results[th_f] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}
                if f1 > best_f1:
                    best_f1 = f1
                    best_th = th_f
                    best_p = p
                    best_r = r
            return {
                "best_f1": round(best_f1, 4),
                "best_threshold": best_th,
                "precision_at_best": round(best_p, 4),
                "recall_at_best": round(best_r, 4),
                "threshold_points": th_results
            }

        neural_fine_sweep = fine_sweep(matched_labels, matched_scores)
        prior_fine_sweep = fine_sweep(matched_labels, canonical_prior_metrics["scores"])

        # Unified fixed thresholds comparison
        fixed_thresholds_comparison = {}
        for fixed_t in [0.10, 0.15, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40, 0.50]:
            fixed_t = round(float(fixed_t), 2)
            fixed_thresholds_comparison[str(fixed_t)] = {
                "neural_model": neural_fine_sweep["threshold_points"].get(fixed_t),
                "canonical_prior": prior_fine_sweep["threshold_points"].get(fixed_t)
            }

        controls_results = {
            "conditions": {
                "condition_1_matched_audio": {
                    "description": "True paired acoustic waveform with true canonical prompt",
                    "binary_ap": matched_binary_metrics["average_precision"],
                    "optimal_f1_step_05": matched_binary_metrics["operating_point_optimal_f1"]["f1"],
                    "optimal_threshold_step_05": matched_binary_metrics["operating_point_optimal_f1"]["threshold"],
                    "optimal_f1_step_01": neural_fine_sweep["best_f1"],
                    "optimal_threshold_step_01": neural_fine_sweep["best_threshold"],
                    "f1_at_05": matched_binary_metrics["operating_point_standard_05"]["f1"]
                },
                "condition_2_mismatched_audio": {
                    "description": "Deterministic deranged within-speaker acoustic waveform (shift_1) with original prompt and labels",
                    "binary_ap": mismatched_binary_metrics["average_precision"],
                    "optimal_f1_step_05": mismatched_binary_metrics["operating_point_optimal_f1"]["f1"],
                    "optimal_threshold_step_05": mismatched_binary_metrics["operating_point_optimal_f1"]["threshold"],
                    "f1_at_05": mismatched_binary_metrics["operating_point_standard_05"]["f1"]
                },
                "condition_3_canonical_empirical_prior": {
                    "description": "Prompt-only Bayesian lookup P_train(error | canonical_phone) computed from 2,399 Fold 1 train utterances",
                    "binary_ap": canonical_prior_metrics["average_precision"],
                    "optimal_f1_step_05": canonical_prior_metrics["operating_point_optimal_f1"]["f1"],
                    "optimal_threshold_step_05": canonical_prior_metrics["operating_point_optimal_f1"]["threshold"],
                    "optimal_f1_step_01": prior_fine_sweep["best_f1"],
                    "optimal_threshold_step_01": prior_fine_sweep["best_threshold"],
                    "f1_at_05": canonical_prior_metrics["operating_point_standard_05"]["f1"]
                },
                "condition_4_constant_prevalence": {
                    "description": "Constant trivial baseline predicting validation base rate",
                    "binary_ap": prevalence_metrics["average_precision"],
                    "prevalence": prevalence_metrics["prevalence"]
                }
            },
            "permutation_sensitivity": permutation_sensitivity,
            "fine_grained_sweeps": {
                "neural_model": {
                    "optimal_f1": neural_fine_sweep["best_f1"],
                    "optimal_threshold": neural_fine_sweep["best_threshold"],
                    "precision": neural_fine_sweep["precision_at_best"],
                    "recall": neural_fine_sweep["recall_at_best"]
                },
                "canonical_prior": {
                    "optimal_f1": prior_fine_sweep["best_f1"],
                    "optimal_threshold": prior_fine_sweep["best_threshold"],
                    "precision": prior_fine_sweep["precision_at_best"],
                    "recall": prior_fine_sweep["recall_at_best"]
                }
            },
            "fixed_thresholds_comparison": fixed_thresholds_comparison,
            "paired_matched_vs_mismatched": {
                "delta_ap (matched - mismatched)": round(float(matched_binary_metrics["average_precision"] - mismatched_binary_metrics["average_precision"]), 6),
                "mean_score_diff": round(float(np.mean(score_diff)), 6),
                "std_score_diff": round(float(np.std(score_diff)), 6),
                "median_score_diff": round(float(np.median(score_diff)), 6),
                "iqr_score_diff": round(float(np.percentile(score_diff, 75) - np.percentile(score_diff, 25)), 6),
                "mean_absolute_difference": round(float(np.mean(abs_diff)), 6),
                "prediction_agreement_at_05": round(agree_05, 4),
                "prediction_agreement_at_02": round(agree_opt, 4),
                "sensitivity_fractions": {
                    "abs_diff_gt_0.01": round(frac_gt_001, 4),
                    "abs_diff_gt_0.05": round(frac_gt_005, 4),
                    "abs_diff_gt_0.10": round(frac_gt_010, 4)
                }
            },
            "per_speaker_controls": {
                spk: {
                    "matched_ap": per_speaker_metrics[spk]["binary_ap"],
                    "mismatched_ap": calculate_binary_metrics(per_speaker_mismatched[spk]["labels"], per_speaker_mismatched[spk]["scores"])["average_precision"]
                }
                for spk in per_speaker_matched
            },
            "utterance_clustered_bootstrap": boot_results
        }

    # Final report structure
    report = {
        "metadata": {
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_sha": os.popen("git rev-parse HEAD").read().strip(),
            "checkpoint_path": args.checkpoint,
            "checkpoint_sha256": ckpt_info["sha256"],
            "checkpoint_epoch_stored": ckpt_info["epoch"],
            "checkpoint_epoch_display": ckpt_info["epoch"] + 1,
            "checkpoint_successful_updates": ckpt_info["successful_updates"],
            "checkpoint_best_ap": ckpt_info["best_ap"],
            "training_execution_sha": ckpt_info.get("metadata", {}).get("git_sha", "caa56783113d40eac438e5f458cdb313b8932218"),
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "authorized_zip_path": zip_path,
            "authorized_zip_sha256": zip_sha,
            "device": str(device),
            "eval_batch_size": args.batch_size,
            "evaluated_utterances": processed_utts,
            "model_weights_fingerprint": post_fingerprint,
            "runtime_performance": {
                "total_wall_seconds": round(total_eval_time, 2),
                "seconds_per_utterance": round(per_utt_time, 4),
                "peak_process_ram_mb": round(peak_ram_mb, 2)
            }
        },
        "binary_mdd": matched_binary_metrics,
        "ctc_recognition": matched_ctc_metrics,
        "per_speaker_validation_breakdown": per_speaker_metrics,
        "acoustic_contribution_controls": controls_results
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / ("CPU_SMOKE_TEST_REPORT.json" if args.max_eval_utterances and args.max_eval_utterances <= 4 else "INDEPENDENT_VALIDATION_REPLAY.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[REPORT] Successfully saved evaluation report to: {out_file}")

    # Export paired prediction records for independent statistical verification (Task D)
    if args.max_eval_utterances is None or args.max_eval_utterances >= 300:
        corrections_dir = out_dir / "corrections"
        corrections_dir.mkdir(parents=True, exist_ok=True)
        pred_export = {
            "metadata": {
                "description": "Per-utterance matched and mismatched prediction records for Fold 1 validation",
                "split": "val",
                "fold": 1,
                "total_utterances": len(utterance_results),
                "total_phone_tokens": len(matched_labels),
                "total_error_tokens": sum(matched_labels),
                "evaluator_source_sha": os.popen("git rev-parse HEAD").read().strip(),
                "checkpoint_sha256": ckpt_info["sha256"],
                "schema": {
                    "record_id": "L2-ARCTIC utterance record identifier (e.g. l2arctic_NJS_arctic_a0003)",
                    "speaker": "Speaker code (NJS or RRBI)",
                    "utterance_id": "ARCTIC utterance ID",
                    "mismatched_audio_record_id": "Utterance record ID of the donor audio under shift_1 derangement",
                    "canonical_phone_indices": "Canonical ARPAbet phoneme indices fed to model cross-attention",
                    "binary_reference_labels": "Ground-truth binary MDD reference labels (0=correct, 1=error)",
                    "matched_binary_scores": "Predicted error probabilities from matched acoustic inference",
                    "mismatched_binary_scores": "Predicted error probabilities from mismatched audio inference (shift_1)",
                    "matched_ctc_reference": "Ground-truth CTC reference phone token indices",
                    "matched_ctc_hypothesis": "Greedy decoded CTC hypothesis phone token indices"
                }
            },
            "utterances": [
                {
                    "record_id": u["record_id"],
                    "speaker": u["speaker"],
                    "utterance_id": u["record_id"].split("_")[-1],
                    "mismatched_audio_record_id": val_dataset[derange_indices[idx]]["record_id"],
                    "canonical_phone_indices": val_dataset[idx]["canonical_phone_indices"].tolist(),
                    "binary_reference_labels": u["labels"],
                    "matched_binary_scores": [float(s) for s in u["matched_scores"]],
                    "mismatched_binary_scores": [float(s) for s in u["mismatched_scores"]],
                    "matched_ctc_reference": u["ctc_ref"],
                    "matched_ctc_hypothesis": u["ctc_hyp"]
                }
                for idx, u in enumerate(utterance_results)
            ]
        }
        pred_out_file = corrections_dir / "PAIRED_PREDICTION_RECORDS.json"
        with open(pred_out_file, "w", encoding="utf-8") as f:
            json.dump(pred_export, f, indent=2)
        print(f"[RECORDS] Successfully exported paired prediction records to: {pred_out_file}")

    return report


if __name__ == "__main__":
    main()
