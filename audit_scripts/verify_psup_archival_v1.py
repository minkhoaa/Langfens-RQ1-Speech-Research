"""Independently recompute archival P-SUP scores; never imply fresh inference.

Historical evaluator.py uses annotation-derived canonical phones even though its
runtime serializer hardcodes Z-PROMPT-V1. This audit preserves those source files
and explicitly identifies the actual oracle protocol in a separate receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def average_precision(labels, scores):
    if not labels or not sum(labels):
        return None
    ranked = sorted(zip(scores, labels), reverse=True)
    positives = sum(labels)
    tp = fp = 0
    ap = 0.0
    index = 0
    while index < len(ranked):
        end = index
        group_tp = 0
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            group_tp += ranked[end][1]
            end += 1
        tp += group_tp
        fp += end - index - group_tp
        ap += group_tp / positives * tp / (tp + fp)
        index = end
    return ap


def operating_points(labels, scores):
    """VAL-derived diagnostic curves, NOT CAL-derived deployment thresholds."""
    n_pos, n_neg = sum(labels), len(labels) - sum(labels)
    points = []
    ranked = sorted(zip(scores, labels), reverse=True)
    tp = fp = 0
    # A finite reject-all score boundary, preserving score>=threshold semantics.
    points.append((math.nextafter(max(scores), math.inf), tp, fp))
    index = 0
    while index < len(ranked):
        end = index
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            tp += ranked[end][1]
            fp += 1 - ranked[end][1]
            end += 1
        points.append((ranked[index][0], tp, fp))
        index = end
    result = {}
    for ceiling in (.05, .10, .15):
        eligible = [p for p in points if n_neg and p[2] / n_neg <= ceiling]
        threshold, tp, fp = eligible[-1]
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / n_pos if n_pos else None
        result[str(ceiling)] = dict(threshold=threshold, TP=tp, FP=fp,
            TN=n_neg-fp, FN=n_pos-tp, achieved_fpr=fp/n_neg,
            precision=precision, recall=recall,
            f1=2*tp/(2*tp+fp+n_pos-tp) if 2*tp+fp+n_pos-tp else None,
            threshold_status="VAL-derived retrospective diagnostic; not calibrated")
    return result


def audit(directory, allowed_ids=None):
    runtime_path = directory / "runtime_predictions.jsonl"
    offline_path = directory / "offline_token_evaluation.jsonl"
    runtime = [json.loads(line) for line in runtime_path.read_text().splitlines()]
    offline = [json.loads(line) for line in offline_path.read_text().splitlines()]
    key = lambda row: (row["speaker_id"], row["utterance_id"], row["prompt_index"])
    runtime_map = {key(row): row for row in runtime}
    offline_map = {key(row): row for row in offline}
    if len(runtime_map) != len(runtime) or len(offline_map) != len(offline):
        raise ValueError("duplicate runtime/offline token keys")
    if set(runtime_map) != set(offline_map):
        raise ValueError("runtime/offline keys do not match")
    if set(row["speaker_id"] for row in runtime) != {"NJS", "RRBI"}:
        raise ValueError("unexpected speaker population")
    actual_ids = list(dict.fromkeys(f'{row["speaker_id"]}/{row["utterance_id"]}' for row in runtime))
    if allowed_ids is not None and actual_ids != allowed_ids:
        raise ValueError("utterance cohort/order differs from frozen Fold-1 VAL")
    for row in offline:
        if row["operation_target"] not in (0, 1, 2):
            raise ValueError("invalid operation")
        if row["binary_target"] != int(row["operation_target"] != 0):
            raise ValueError("binary/operation mismatch")
        for name in ("s_error", "s_omit", "s_sub"):
            score = row[name]
            if score != runtime_map[key(row)]["scores"][name]:
                raise ValueError("runtime/offline score mismatch")
            if score is not None and not math.isfinite(score):
                raise ValueError("nonfinite score")
    result = dict(evidence_status="archival-only", metric_recomputation="reproduced from archival full-precision predictions; no fresh inference",
        actual_protocol="O-ARCHIVE (annotation-derived canonical positions)",
        serialized_protocols=sorted({row["protocol"] for row in runtime}),
        protocol_mismatch="Historical serializer hardcoded Z-PROMPT-V1; evaluator dataset builds prompt from gold canonical labels, so NOT genuine prompt-only inference.",
        token_count=len(offline), label_counts=dict(Counter(str(row["operation_target"]) for row in offline)),
        speaker_counts=dict(Counter(row["speaker_id"] for row in offline)),
        utterance_count=len({key(row)[:2] for row in offline}),
        ordered_utterance_id_sha256=hashlib.sha256(('\n'.join(actual_ids)+'\n').encode()).hexdigest(),
        artifact_hashes={p.name: sha256(p) for p in (runtime_path, offline_path, directory/"metrics.json")}, metrics={})
    for metric, operations, positive, score_name in (
        ("binary", (0,1,2), None, "s_error"),
        ("omission_legacy", (0,2), 2, "s_error"),
        ("substitution_legacy", (0,1), 1, "s_error"),
        ("omission_dedicated", (0,2), 2, "s_omit"),
        ("substitution_dedicated", (0,1), 1, "s_sub")):
        rows = [row for row in offline if row["operation_target"] in operations]
        if any(row[score_name] is None for row in rows):
            result["metrics"][metric] = dict(evidence_status="unavailable", reason="binary_collapsed subtype probabilities unidentified", ap=None)
            continue
        labels = [row["binary_target"] if positive is None else int(row["operation_target"] == positive) for row in rows]
        scores = [row[score_name] for row in rows]
        result["metrics"][metric] = dict(evidence_status="archival-only", score=score_name,
            n=len(labels), positives=sum(labels), negatives=len(labels)-sum(labels),
            ap=average_precision(labels, scores), retrospective_fpr_points=operating_points(labels, scores))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split-manifest", type=Path, default=Path(__file__).resolve().parents[1]/"manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json")
    args = parser.parse_args()
    manifest_hash = sha256(args.split_manifest)
    if manifest_hash != '7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b':
        raise ValueError('Split manifest hash mismatch')
    manifest = json.loads(args.split_manifest.read_text())
    allowed_ids = next(f for f in manifest['folds'] if f['fold_index'] == 1)['val']['utterances']
    names = ("t_op_fixed_budget_3600", "t_op_best", "b_control_fixed_budget_3600", "b_control_best")
    result = {"source_sha256": sha256(__file__), "split_manifest_sha256": manifest_hash,
        "configurations": {name: audit(args.archive_root/name, allowed_ids) for name in names}}
    if args.output.exists():
        raise FileExistsError("Never overwrite historical verification receipts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name, data in result["configurations"].items():
        print(name, data["token_count"], {key: value["ap"] for key, value in data["metrics"].items()})


if __name__ == "__main__":
    main()
