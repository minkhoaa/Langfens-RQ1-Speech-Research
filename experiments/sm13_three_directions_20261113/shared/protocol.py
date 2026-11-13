"""Shared held-out-speaker protocol for the SM13 three-direction study.

For outer fold f, GROUPS[f] is the held set, GROUPS[(f+1)%4] is calibration,
and the remaining six speakers are fit-only. The ordinary research corpus here
is the L2-ARCTIC TRAIN manifest only. No dev/test/sealed corpus is loaded.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/home/khoa/Projects/New-Method-Research")
sys.path.insert(0, str(ROOT / "experiments/rq1_encoder_bench_01/common"))
sys.path.insert(0, str(ROOT / "experiments/rq1_train_phase_01"))
sys.path.insert(0, str(ROOT))

from dataset_util import DatasetManager, GROUPS  # noqa: E402
from research.evaluation_v3.evaluator import UnifiedEvaluatorV3  # noqa: E402
from preflight.agent_b.joint_policy import freeze_threshold, validate_exclusion  # noqa: E402

SEEDS = (101, 202)
BUDGET = 0.2


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    fit_speakers: tuple[str, ...]
    calibration_speakers: tuple[str, ...]
    held_speakers: tuple[str, ...]
    t_fit: np.ndarray
    t_cal: np.ndarray
    t_held: np.ndarray
    g_fit: np.ndarray
    g_cal: np.ndarray
    g_held: np.ndarray
    utt_fit: np.ndarray
    utt_cal: np.ndarray
    utt_held: np.ndarray


class SplitProtocol:
    def __init__(self, dm: DatasetManager | None = None):
        self.dm = dm or DatasetManager()
        all_speakers = set(self.dm.speakers.values())
        assert len(all_speakers) == 12
        assert set().union(*(set(g) for g in GROUPS)) == all_speakers

    def split(self, fold: int) -> FoldSplit:
        if fold not in range(4):
            raise ValueError("fold must be 0..3")
        held = tuple(GROUPS[fold])
        cal = tuple(GROUPS[(fold + 1) % 4])
        fit = tuple(sorted(set(self.dm.speakers.values()) - set(held) - set(cal)))
        if len(fit) != 6 or len(cal) != 3 or len(held) != 3:
            raise AssertionError("split must be 6 fit / 3 calibration / 3 held speakers")
        validate_exclusion(fit, (*cal, *held))
        ts = self.dm.target_speaker
        gs = self.dm.gap_speaker
        us = np.asarray([self.dm.speakers[u] for u in self.dm.order])
        t_fit, t_cal, t_held = (np.isin(ts, s) for s in (fit, cal, held))
        g_fit, g_cal, g_held = (np.isin(gs, s) for s in (fit, cal, held))
        utt_fit, utt_cal, utt_held = (np.isin(us, s) for s in (fit, cal, held))
        assert not np.any(t_fit & (np.isin(ts, (*cal, *held))))
        assert not np.any(g_fit & (np.isin(gs, (*cal, *held))))
        assert np.all(t_fit | t_cal | t_held) and np.all(g_fit | g_cal | g_held)
        return FoldSplit(fold, fit, cal, held, t_fit, t_cal, t_held,
                         g_fit, g_cal, g_held, utt_fit, utt_cal, utt_held)


def _gold_rows(dm: DatasetManager, tmask: np.ndarray, gmask: np.ndarray):
    labels = ("CORRECT", "SUBSTITUTION", "OMISSION")
    gt, gg = [], []
    for key, y in zip(dm.target_keys[tmask], dm.yt[tmask]):
        ds, sp, uid, _, pos = str(key).split("|")
        gt.append({"dataset_id": ds, "speaker_id": sp, "utterance_id": uid,
                   "position": int(pos), "gold_label": labels[int(y)]})
    for key, y in zip(dm.gap_keys[gmask], dm.yg[gmask]):
        ds, sp, uid, _, pos = str(key).split("|")
        gg.append({"dataset_id": ds, "speaker_id": sp, "utterance_id": uid,
                   "gap_position": int(pos), "gold_has_insertion": bool(y)})
    return gt, gg


def _prediction_rows(dm: DatasetManager, tmask: np.ndarray, gmask: np.ndarray,
                     tp: np.ndarray, gp: np.ndarray, td: np.ndarray, gd: np.ndarray):
    labels = ("CORRECT", "SUBSTITUTION", "OMISSION")
    pt, pg = [], []
    for key, p, decision, gold in zip(dm.target_keys[tmask], tp, td, dm.yt[tmask]):
        ds, sp, uid, _, pos = str(key).split("|")
        pt.append({"dataset_id": ds, "speaker_id": sp, "utterance_id": uid,
                   "position": int(pos), "gold_label": labels[int(gold)],
                   "pred_label": labels[int(decision)],
                   "prob_correct": float(p[0]), "prob_substitution": float(p[1]),
                   "prob_omission": float(p[2])})
    for key, p, decision, gold in zip(dm.gap_keys[gmask], gp, gd, dm.yg[gmask]):
        ds, sp, uid, _, pos = str(key).split("|")
        pg.append({"dataset_id": ds, "speaker_id": sp, "utterance_id": uid,
                   "gap_position": int(pos), "gold_has_insertion": bool(gold),
                   "pred_has_insertion": bool(decision),
                   "prob_insertion": float(p)})
    return pt, pg


def evaluate_fold(dm: DatasetManager, fold: int, cal_target_probs: np.ndarray,
                  cal_gap_probs: np.ndarray, held_target_probs: np.ndarray,
                  held_gap_probs: np.ndarray, tag: str, seed: int) -> dict[str, Any]:
    """Freeze B=.2 warning cutoffs on calibration speakers; score held speakers once.

    Probabilities must align exactly to boolean mask order in FoldSplit. Target
    score is the monotonic [0,1] transform of max(P(S),P(O))-P(C); its warning
    type is argmax(P(S),P(O)).
    The same deterministic event-key hash resolves score ties at calibration and
    test time. Held labels are passed only to EVAL-V3.1 after decisions are frozen.
    """
    split = SplitProtocol(dm).split(fold)
    for name, arr, expected in (
        ("cal target", cal_target_probs, int(split.t_cal.sum())),
        ("held target", held_target_probs, int(split.t_held.sum())),
        ("cal gap", cal_gap_probs, int(split.g_cal.sum())),
        ("held gap", held_gap_probs, int(split.g_held.sum())),
    ):
        if len(arr) != expected:
            raise ValueError(f"{name} rows={len(arr)} expected={expected}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} probabilities contain nonfinite values")
    if cal_target_probs.shape[1:] != (3,) or held_target_probs.shape[1:] != (3,):
        raise ValueError("target probabilities must have shape [N,3]")
    for x in (cal_target_probs, held_target_probs):
        if np.any(x < 0) or np.any(x > 1) or not np.allclose(x.sum(1), 1, atol=1e-3):
            raise ValueError("target probabilities must form a valid simplex")
    for x in (cal_gap_probs, held_gap_probs):
        if np.any(x < 0) or np.any(x > 1):
            raise ValueError("gap probabilities must be in [0,1]")

    cal_tkeys = dm.target_keys[split.t_cal].astype(str).tolist()
    cal_gkeys = dm.gap_keys[split.g_cal].astype(str).tolist()
    held_tkeys = dm.target_keys[split.t_held].astype(str).tolist()
    held_gkeys = dm.gap_keys[split.g_held].astype(str).tolist()
    cal_tgold = dm.yt[split.t_cal]
    cal_ggold = dm.yg[split.g_cal]
    t_scores = (np.max(cal_target_probs[:, 1:], axis=1) - cal_target_probs[:, 0] + 1.0) / 2.0
    t_types = np.argmax(cal_target_probs[:, 1:], axis=1) + 1
    t_false = (t_types != cal_tgold)
    tc = freeze_threshold(cal_tkeys, t_scores, t_false, 0.8 * BUDGET,
                          int(split.utt_cal.sum()))
    gc = freeze_threshold(cal_gkeys, cal_gap_probs, cal_ggold == 0, 0.2 * BUDGET,
                          int(split.utt_cal.sum()))
    held_t_scores = (np.max(held_target_probs[:, 1:], axis=1) - held_target_probs[:, 0] + 1.0) / 2.0
    held_t_types = np.argmax(held_target_probs[:, 1:], axis=1) + 1
    td_warn = np.asarray(tc.apply(held_tkeys, held_t_scores), dtype=bool)
    td = np.where(td_warn, held_t_types, 0)
    gd = np.asarray(gc.apply(held_gkeys, held_gap_probs), dtype=bool)

    gt, gg = _gold_rows(dm, split.t_held, split.g_held)
    pt, pg = _prediction_rows(dm, split.t_held, split.g_held,
                              held_target_probs, held_gap_probs, td, gd)
    eval_result = UnifiedEvaluatorV3().evaluate(pt, pg, gt, gg)
    # Oracle fixed-precision points are diagnostic only: they use held labels.
    oracle = {}
    held_gold = dm.yt[split.t_held]
    for cls, cls_name in ((1, "substitution"), (2, "omission")):
        scores = held_target_probs[:, cls]
        y = (held_gold == cls).astype(int)
        order = np.argsort(-scores, kind="mergesort")
        cum_tp = np.cumsum(y[order])
        precision = cum_tp / np.arange(1, len(order) + 1)
        recall = cum_tp / max(1, int(y.sum()))
        for target_precision in (0.75, 0.80):
            valid = np.flatnonzero(precision >= target_precision)
            n_take = int(valid[-1] + 1) if valid.size else 0
            oracle[f"held_oracle_{cls_name}_recall_at_precision_{target_precision:.2f}"] = (
                float(recall[n_take - 1]) if n_take else 0.0,
                float(precision[n_take - 1]) if n_take else None,
                n_take,
            )
    return {
        "tag": tag, "seed": int(seed), "fold": int(fold),
        "fit_speakers": list(split.fit_speakers),
        "calibration_speakers": list(split.calibration_speakers),
        "held_speakers": list(split.held_speakers),
        "n_fit_targets": int(split.t_fit.sum()), "n_cal_targets": int(split.t_cal.sum()),
        "n_held_targets": int(split.t_held.sum()), "n_fit_gaps": int(split.g_fit.sum()),
        "n_cal_gaps": int(split.g_cal.sum()), "n_held_gaps": int(split.g_held.sum()),
        "n_cal_utterances": int(split.utt_cal.sum()),
        "target_threshold": tc.threshold, "target_tie_cutoff": tc.tie_cutoff,
        "target_cal_selected": tc.selected_count, "target_cal_false_warnings": tc.false_warnings,
        "target_cal_false_budget": tc.false_budget,
        "gap_threshold": gc.threshold, "gap_tie_cutoff": gc.tie_cutoff,
        "gap_cal_selected": gc.selected_count, "gap_cal_false_warnings": gc.false_warnings,
        "gap_cal_false_budget": gc.false_budget,
        "oracle_fixed_precision_diagnostics_only": oracle,
        "metrics": eval_result["summary"], "classes": eval_result["classes"],
        "target_predictions": pt, "gap_predictions": pg,
    }


def save_fold(result: dict[str, Any], out_dir: str | Path) -> None:
    """Persist keyed exact held predictions and metadata as JSON + compact NPZ."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    stem = f"{result['tag']}_seed{result['seed']}_fold{result['fold']}"
    pt, pg = result["target_predictions"], result["gap_predictions"]
    np.savez_compressed(
        out / f"{stem}.npz",
        target_event_id=np.asarray([f"{r['dataset_id']}|{r['speaker_id']}|{r['utterance_id']}|target|{r['position']}" for r in pt]),
        target_gold=np.asarray([r["gold_label"] for r in pt]),
        target_pred=np.asarray([r["pred_label"] for r in pt]),
        target_probabilities=np.asarray([[r["prob_correct"],r["prob_substitution"],r["prob_omission"]] for r in pt], dtype=np.float32),
        gap_event_id=np.asarray([f"{r['dataset_id']}|{r['speaker_id']}|{r['utterance_id']}|gap|{r['gap_position']}" for r in pg]),
        gap_gold=np.asarray([r["gold_has_insertion"] for r in pg], dtype=bool),
        gap_pred=np.asarray([r["pred_has_insertion"] for r in pg], dtype=bool),
        gap_probability=np.asarray([r["prob_insertion"] for r in pg], dtype=np.float32),
    )
    meta = {k: v for k,v in result.items() if k not in ("target_predictions","gap_predictions")}
    (out / f"{stem}.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")

