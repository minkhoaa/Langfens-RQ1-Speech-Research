"""Independent model-free metric accounting for frozen RQ1-TRN-04R records."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

OBJECTIVE_SCORE_SUPPORT = {
    "operation_ce": True,
    "class_focal": True,
    "binary_collapsed": False,
}


def objective_supports_subtypes(objective: str) -> bool:
    try:
        return OBJECTIVE_SCORE_SUPPORT[objective]
    except KeyError as exc:
        raise ValueError(f"Unknown score objective: {objective!r}") from exc


SCORE_FAMILIES = {
    "binary": ("s_error", None, {0, 1, 2}),
    "omission_legacy": ("s_error", 2, {0, 2}),
    "omission_dedicated": ("s_omit", 2, {0, 2}),
    "substitution_legacy": ("s_error", 1, {0, 1}),
    "substitution_dedicated": ("s_sub", 1, {0, 1}),
}


def _finite(value: Any, field: str, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be finite numeric" + (" or null" if nullable else ""))
    return float(value)


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Non-interpolated AP; tied scores enter as a single threshold group."""
    if len(labels) != len(scores):
        raise ValueError("labels/scores length mismatch")
    if not labels:
        return None
    pairs = []
    for label, score in zip(labels, scores):
        if label not in (0, 1):
            raise ValueError("AP labels must be binary")
        pairs.append((_finite(score, "AP score"), label))
    positives = sum(y for _, y in pairs)
    if positives == 0:
        return None
    pairs.sort(key=lambda pair: pair[0], reverse=True)
    tp = fp = 0
    area = 0.0
    previous_recall = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        group_pos = sum(y for _, y in pairs[index:end])
        tp += group_pos
        fp += end - index - group_pos
        recall = tp / positives
        area += (recall - previous_recall) * tp / (tp + fp)
        previous_recall = recall
        index = end
    return area


def _validate_gold(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for source in rows:
        row = dict(source)
        index = row.get("gold_canonical_index")
        if index is None:
            continue
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(f"Invalid gold_canonical_index: {index!r}")
        key = (row.get("speaker_id"), row.get("utterance_id"), index)
        if key in seen:
            raise ValueError(f"Duplicate gold key: {key}")
        seen.add(key)
        op = row.get("operation_target")
        if op not in (0, 1, 2):
            raise ValueError(f"Unknown operation_target at {key}: {op!r}")
        if row.get("binary_target") != int(op != 0):
            raise ValueError(f"binary_target disagrees with operation_target at {key}")
        for field, _, _ in SCORE_FAMILIES.values():
            if field in row:
                _finite(row[field], field, nullable=True)
        result.append(row)
    return result


def calibration_threshold(correct_scores: Sequence[float], fpr_ceiling: float) -> dict[str, Any]:
    """Choose inclusive score>=tau threshold using CAL correct-only records."""
    if not math.isfinite(fpr_ceiling) or not 0 <= fpr_ceiling < 1:
        raise ValueError("FPR ceiling must lie in [0,1)")
    values = sorted((_finite(v, "CAL correct score") for v in correct_scores), reverse=True)
    if not values:
        return {"threshold": None, "status": "unavailable", "reason": "no finite CAL correct scores"}
    permitted = math.floor(fpr_ceiling * len(values))
    threshold = math.nextafter(values[permitted], math.inf)
    return {"threshold": threshold, "status": "available", "correct_n": len(values),
            "nominal_fpr_ceiling": fpr_ceiling,
            "cal_false_alarms": sum(value >= threshold for value in values),
            "cal_achieved_fpr": sum(value >= threshold for value in values) / len(values),
            "rule": "lowest empirical threshold satisfying ceiling, score>=threshold; ties not split"}


def _confusion(gold: Sequence[Mapping[str, Any]], score_name: str, positive_op: int | None,
               threshold: float) -> dict[str, Any]:
    # Deployment accounting always traverses the complete gold census: a substitution
    # alarm on an omission detector is an FP_sub, although substitutions are excluded
    # from that detector's AP ranking population.
    eligible = list(gold)
    is_positive = lambda r: bool(r["binary_target"]) if positive_op is None else r["operation_target"] == positive_op
    alarmed = lambda r: r.get(score_name) is not None and r[score_name] >= threshold
    tp = sum(is_positive(r) and alarmed(r) for r in eligible)
    fp_correct = sum(r["operation_target"] == 0 and alarmed(r) for r in eligible)
    fp_sub = sum(r["operation_target"] == 1 and alarmed(r) for r in eligible)
    positives = sum(is_positive(r) for r in eligible)
    correct = sum(r["operation_target"] == 0 for r in eligible)
    abstained_error = sum(is_positive(r) and r.get(score_name) is None for r in eligible)
    abstained_correct = sum(r["operation_target"] == 0 and r.get(score_name) is None for r in eligible)
    fn = positives - tp
    fp = fp_correct + fp_sub
    return {"threshold": threshold, "TP": tp, "FP": fp, "FP_correct": fp_correct,
            "FP_sub": fp_sub, "FN": fn, "TN_correct": correct - fp_correct,
            "false_alarms_per_1000_full_gold": 1000 * fp / len(gold) if gold else None,
            "correct_false_alarms_per_1000_full_gold": 1000 * fp_correct / len(gold) if gold else None,
            "positive_gold": positives, "correct_gold": correct,
            "recall_aligned": tp / (positives - abstained_error) if positives > abstained_error else None,
            "recall_all_gold": tp / positives if positives else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "correct_fpr_answered": fp_correct / (correct - abstained_correct) if correct > abstained_correct else None,
            "correct_fpr_all_gold_lower": fp_correct / correct if correct else None,
            "correct_fpr_all_gold_upper": (fp_correct + abstained_correct) / correct if correct else None,
            "abstained_true_errors": abstained_error, "abstained_correct": abstained_correct,
            "abstentions_are_no_alarm_and_errors_count_as_missed": True}


def _one_family(gold: list[dict[str, Any]], cal: list[dict[str, Any]], name: str,
                subtype_supported: bool) -> dict[str, Any]:
    score_name, positive_op, operations = SCORE_FAMILIES[name]
    if not subtype_supported and name.endswith("dedicated"):
        return {"status": "unavailable", "ap": None,
                "reason": "binary_collapsed objective does not identify subtype scores"}
    eligible = [r for r in gold if r["operation_target"] in operations]
    scored = [r for r in eligible if r.get(score_name) is not None]
    labels = [int(r["binary_target"] if positive_op is None else r["operation_target"] == positive_op) for r in scored]
    values = [r[score_name] for r in scored]
    positives = sum(labels)
    ap = average_precision(labels, values) if positives else None
    cal_scores = [r[score_name] for r in cal if r["operation_target"] == 0 and r.get(score_name) is not None]
    points = {}
    for ceiling in (.05, .10, .15):
        cut = calibration_threshold(cal_scores, ceiling)
        points[str(ceiling)] = ({**cut, "status": "unavailable"} if cut["threshold"] is None else
                                {**cut, "status": "available",
                                 "counts": _confusion(gold, score_name, positive_op, cut["threshold"])})
    return {"status": "computed", "score": score_name,
            "score_semantics": ("T-OP dedicated subtype score; not interchangeable with s_error" if score_name in ("s_omit", "s_sub") else "error-ranking score"),
            "ap": ap, "ap_population": "finite-score aligned gold rows for this endpoint only",
            "n_population": len(eligible), "n_scored": len(scored), "n_abstained": len(eligible)-len(scored),
            "positives_scored": positives, "negatives_scored": len(scored)-positives,
            "prevalence_scored": positives / len(scored) if scored else None,
            "cal_correct_count": len(cal_scores), "cal_operating_points": points}


def _edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> tuple[int, int, int, int]:
    # Deterministic edit tie priority: substitution, deletion, insertion.
    n, m = len(reference), len(hypothesis)
    dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1): dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1): dp[0][j] = (j, 0, 0, j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = int(reference[i-1] != hypothesis[j-1])
            a, s, d, ins = dp[i-1][j-1]; sub = (a+cost, s+cost, d, ins)
            a, s, d, ins = dp[i-1][j]; delete = (a+1, s, d+1, ins)
            a, s, d, ins = dp[i][j-1]; insert = (a+1, s, d, ins+1)
            dp[i][j] = min((sub, delete, insert), key=lambda value: value[0])
    return dp[n][m]


def recognition_summary(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    total = [0, 0, 0, 0]
    references = 0
    by_speaker: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in records:
        if "ref_phones" in row and "hyp_phones" in row:
            counts = _edit_distance(row["ref_phones"], row["hyp_phones"])
            if row.get("edit_counts") is not None and list(counts) != list(row["edit_counts"]):
                raise ValueError("Saved CTC edit counts disagree with saved reference/hypothesis")
            ref_n = len(row["ref_phones"])
        elif "edit_counts" in row and ("reference_phone_count" in row or "n_reference" in row):
            counts = tuple(row["edit_counts"])
            ref_n = row.get("reference_phone_count", row.get("n_reference"))
            if len(counts) != 4 or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in counts):
                raise ValueError("Invalid saved CTC edit_counts")
        else:
            raise ValueError("Recognition rows need ref_phones/hyp_phones or edit_counts/reference_phone_count")
        total = [a+b for a, b in zip(total, counts)]
        references += ref_n
        by_speaker[row.get("speaker_id", "unknown")][0] += sum(counts)
        by_speaker[row.get("speaker_id", "unknown")][1] += ref_n
    return {"status": "computed", "edits": total[0], "substitutions": total[1],
            "deletions": total[2], "insertions": total[3], "reference_phones": references,
            "PER": total[0] / references if references else None,
            "by_speaker": {s: {"edits": v[0], "reference_phones": v[1], "PER": v[0]/v[1] if v[1] else None} for s,v in by_speaker.items()},
            "semantics": "saved greedy CTC recognition errors; not learner human omission labels"}


def summarize_cell(runtime: Iterable[Mapping[str, Any]], tokens: Iterable[Mapping[str, Any]],
                   cal_tokens: Iterable[Mapping[str, Any]], recognition: Iterable[Mapping[str, Any]],
                   *, objective: str, cell_id: str,
                   cal_runtime: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Audit and compute one immutable endpoint/protocol/condition cell."""
    runtime_rows = [dict(r) for r in runtime]
    cal_runtime_rows = [dict(r) for r in cal_runtime] if cal_runtime is not None else None
    token_rows = [dict(r) for r in tokens]
    cal_rows = [dict(r) for r in cal_tokens]
    subtype_supported = objective_supports_subtypes(objective)
    gold = _validate_gold(token_rows)
    cal = _validate_gold(cal_rows)
    for partition, records in (("VAL", token_rows), ("CAL", cal_rows), ("runtime VAL", runtime_rows),
                               ("runtime CAL", cal_runtime_rows or [])):
        for row in records:
            scores = row.get("scores", row)
            error = scores.get("s_error")
            omission, substitution = scores.get("s_omit"), scores.get("s_sub")
            if subtype_supported and error is not None and (omission is None or substitution is None):
                raise ValueError(f"{partition}: {objective} requires three-way s_omit and s_sub with s_error")
            if not subtype_supported and (omission is not None or substitution is not None):
                raise ValueError(f"{partition}: binary_collapsed must leave dedicated subtype scores null")
    extra_rows = [r for r in token_rows if r.get("gold_canonical_index") is None]
    by_key = {}
    for row in runtime_rows:
        key = (row.get("speaker_id"), row.get("utterance_id"), row.get("prompt_index"))
        if key in by_key:
            raise ValueError(f"Duplicate runtime prompt key {key}")
        by_key[key] = row
        scores = row.get("scores", row)
        for name in ("s_error", "s_omit", "s_sub"):
            if name in scores: _finite(scores[name], name, nullable=True)
    def validate_join(offline_rows, runtime_index, partition):
        expected = set()
        for row in offline_rows:
            idx = row.get("prompt_index")
            if idx is None: continue
            key = (row.get("speaker_id"), row.get("utterance_id"), idx)
            expected.add(key)
            if key not in runtime_index:
                raise ValueError(f"{partition} offline row has no runtime record: {key}")
            scores = runtime_index[key].get("scores", runtime_index[key])
            for name in ("s_error", "s_omit", "s_sub"):
                if name in row and row[name] != scores.get(name):
                    raise ValueError(f"{partition} runtime/offline {name} mismatch at {key}")
        actual = set(runtime_index)
        if actual != expected:
            raise ValueError(f"{partition} runtime/offline prompt-key coverage differs: runtime_only={len(actual-expected)}, offline_only={len(expected-actual)}")
    validate_join(token_rows, by_key, "VAL")
    if cal_runtime_rows is not None:
        cal_index = {}
        for row in cal_runtime_rows:
            key = (row.get("speaker_id"), row.get("utterance_id"), row.get("prompt_index"))
            if key in cal_index:
                raise ValueError(f"Duplicate CAL runtime prompt key {key}")
            cal_index[key] = dict(row)
        validate_join(cal_rows, cal_index, "CAL")
    speaker = {}
    phones = {}
    for group_name, group_key in (("speaker", "speaker_id"), ("phone", "canonical_phone")):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in gold: groups[str(row.get(group_key, "unknown"))].append(row)
        target = speaker if group_name == "speaker" else phones
        for value, subset in groups.items():
            target[value] = {name: _one_family(subset, cal, name, subtype_supported)
                             for name in SCORE_FAMILIES}
    return {"cell_id": cell_id, "objective": objective, "gold_rows": len(gold), "actual_label_counts": dict(Counter(str(r["operation_target"]) for r in gold)),
            "coverage": {"gold_positions": len(gold), "answered_s_error": sum(r.get("s_error") is not None for r in gold),
                         "abstained_s_error": sum(r.get("s_error") is None for r in gold),
                         "prompt_extras": len(extra_rows), "runtime_prompts": len(runtime_rows)},
            "metrics": {name: _one_family(gold, cal, name, subtype_supported) for name in SCORE_FAMILIES},
            "by_speaker": speaker, "by_phone": phones,
            "extra_prompt_alarms": {name: {"scored": sum(r.get(name) is not None for r in extra_rows),
                                                   "alarms_at_thresholds": {
                                                       str(ceiling): sum(r.get(name) is not None and r[name] >= cut["threshold"]
                                                                         for r in extra_rows)
                                                       for ceiling in (.05, .10, .15)
                                                       for cut in [calibration_threshold([c[name] for c in cal if c["operation_target"] == 0 and c.get(name) is not None], ceiling)]
                                                       if cut["threshold"] is not None}
                                            } for name in ("s_error", "s_omit", "s_sub")},
            "recognition": recognition_summary(recognition)}


def factorial_contrasts(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Contrasts on a named scalar endpoint; accepts no implicit endpoint selection."""
    result = {}
    for endpoint in ("binary_ap", "omission_dedicated_ap", "omission_recall_all_gold", "correct_fpr_all_gold"):
        vals = {key: cells[key].get(endpoint) for key in ("C0", "T1", "T2", "T3")}
        if any(v is None or not math.isfinite(v) for v in vals.values()):
            result[endpoint] = {"status": "unavailable", "reason": "missing/nonfinite registered cell endpoint", "values": vals}
            continue
        result[endpoint] = {"status": "computed", "values": vals,
                            "matched_contrasts": {"T1_minus_C0": vals["T1"]-vals["C0"],
                                                  "T2_minus_C0": vals["T2"]-vals["C0"],
                                                  "T3_minus_C0": vals["T3"]-vals["C0"]},
                            "interaction_T3_minus_T2_minus_T1_plus_C0": vals["T3"]-vals["T2"]-vals["T1"]+vals["C0"]}
    return result
