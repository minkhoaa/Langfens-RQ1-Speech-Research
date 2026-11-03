"""Strict runtime-only exchange and complete prompt accounting (rq1-runtime-v1)."""

from collections import Counter
import math
from typing import Any

from .ctc import GRAPH_REVISION, score_prompt, validate_emissions
from .controls import greedy_decode, levenshtein, self_aligned_gop

SCHEMA = "rq1-runtime-v1"
PROTOCOL = "Z-PROMPT-V1"
SCORERS = ("sf_s", "sf_sd", "greedy_dp", "gop_sa")
FIELDS = {
    "schema_version", "run_id", "protocol", "condition", "speaker_id", "utterance_id",
    "prompt_index", "prompt_phone_id", "word_index", "scores", "raw_scores",
    "probabilities", "score_orientation", "score_semantics", "alignment", "uncertainty",
    "confidence", "status", "abstention_reason", "finite", "threshold", "calibration_reference",
}
SEALED_SPEAKERS = frozenset(("ABA", "ASI", "EBVS", "HJK", "LXC", "PNV"))
SEMANTICS = {
    "sf_s": "log P_CTC(S union) - log P_CTC(prompt); raw uncalibrated error ranking; subtype unavailable",
    "sf_sd": "log P_CTC(SD union) - log P_CTC(prompt); raw uncalibrated error ranking; subtype unavailable",
    "greedy_dp": "hard prompt-to-greedy recognition mismatch, not a human error/omission probability; subtype unavailable",
    "gop_sa": "negative mean log canonical posterior at canonical Viterbi phone-state activations; not SF; subtype unavailable",
}


def finite_tree(value: Any) -> bool:
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(x) for x in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(x) for x in value)
    return value is None or type(value) in (str, int, bool)


def reject_gold_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lower = key.lower()
            if (any(x in lower for x in ("gold", "perceived", "textgrid", "timestamp", "timing"))
                    or lower in {"labels", "operation", "operation_label", "error_type", "realized_sequence"}):
                raise ValueError("forbidden runtime gold/annotation field")
            reject_gold_fields(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            reject_gold_fields(child)


def runtime_key(row: dict) -> tuple:
    return tuple(row[x] for x in ("run_id", "protocol", "condition", "speaker_id",
                                  "utterance_id", "prompt_index"))


def validate_runtime_row(row: dict) -> None:
    if set(row) != FIELDS:
        raise ValueError("runtime fields differ from rq1-runtime-v1")
    reject_gold_fields(row)
    if not finite_tree(row):
        raise ValueError("runtime JSON must be finite")
    if row["schema_version"] != SCHEMA or row["protocol"] != PROTOCOL:
        raise ValueError("runtime version/protocol mismatch")
    for field in ("run_id", "condition", "speaker_id", "utterance_id", "score_semantics"):
        if type(row[field]) is not str or not row[field]:
            raise ValueError("runtime identity/semantics required")
    if row["speaker_id"] in SEALED_SPEAKERS:
        raise ValueError("sealed TEST speaker prohibited")
    if type(row["prompt_index"]) is not int or row["prompt_index"] < 0:
        raise ValueError("invalid prompt_index")
    if type(row["prompt_phone_id"]) is not int or not 1 <= row["prompt_phone_id"] <= 39:
        raise ValueError("invalid prompt_phone_id")
    word = row["word_index"]
    if word is not None and (type(word) is not int or word < 0):
        raise ValueError("invalid word_index")
    if row["score_orientation"] != "higher_is_error":
        raise ValueError("wrong score orientation")
    if row["status"] not in {"ok", "abstain", "failure"} or type(row["finite"]) is not bool:
        raise ValueError("invalid runtime status")
    if set(row["scores"]) != {"s_error", "s_omit", "s_sub"}:
        raise ValueError("score keys required")
    for value in row["scores"].values():
        if value is not None and type(value) not in (int, float):
            raise ValueError("scores must be numeric or null")
    if not isinstance(row["raw_scores"], dict):
        raise ValueError("raw_scores object required")
    for name in ("probabilities", "alignment", "uncertainty", "confidence"):
        if row[name] is not None and not isinstance(row[name], dict):
            raise ValueError("invalid nullable runtime object")
    if row["threshold"] is not None and type(row["threshold"]) not in (int, float):
        raise ValueError("invalid threshold")
    if row["calibration_reference"] is not None and type(row["calibration_reference"]) is not str:
        raise ValueError("invalid calibration reference")
    if row["status"] == "ok":
        if row["scores"]["s_error"] is None or not row["finite"] or row["abstention_reason"] is not None:
            raise ValueError("ok rows require a finite error score and no abstention")
    elif (any(v is not None for v in row["scores"].values()) or row["finite"]
          or type(row["abstention_reason"]) is not str or not row["abstention_reason"]):
        raise ValueError("abstention/failure requires null scores, finite=false and reason")


def validate_bundle(rows: list[dict], expected: dict[tuple, tuple[int, int | None]]) -> dict:
    """Exact keys AND phone/word identity; duplicates cannot hide missing positions."""
    seen = set()
    for row in rows:
        validate_runtime_row(row)
        key = runtime_key(row)
        if key in seen:
            raise ValueError("duplicate runtime key")
        if key not in expected or expected[key] != (row["prompt_phone_id"], row["word_index"]):
            raise ValueError("unexpected runtime key/phone/word mapping")
        seen.add(key)
    if seen != set(expected):
        raise ValueError("incomplete prompt/failure accounting")
    return dict(Counter(row["status"] for row in rows))


def _number(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def _base(record: dict, run_id: str, scorer: str, index: int) -> dict:
    return {
        "schema_version": SCHEMA, "run_id": run_id, "protocol": PROTOCOL,
        "condition": record["condition"], "speaker_id": record["speaker_id"],
        "utterance_id": record["utterance_id"], "prompt_index": index,
        "prompt_phone_id": record["prompt_phone_ids"][index],
        "word_index": record["word_indices"][index],
        "scores": {"s_error": None, "s_omit": None, "s_sub": None},
        "raw_scores": {"scorer_id": scorer, "graph_revision": GRAPH_REVISION},
        "probabilities": None, "score_orientation": "higher_is_error",
        "score_semantics": SEMANTICS[scorer], "alignment": None, "uncertainty": None,
        "confidence": None, "status": "abstain", "abstention_reason": None,
        "finite": False, "threshold": None, "calibration_reference": None,
    }


def score_record(record: dict, run_id: str, scorer: str) -> tuple[list[dict], dict]:
    """Validated envelope in; every prompt position out, including input failures."""
    if scorer not in SCORERS:
        raise ValueError("unregistered scorer")
    rows = [_base(record, run_id, scorer, i) for i in range(len(record["prompt_phone_ids"]))]
    utterance = {"run_id": run_id, "protocol": PROTOCOL,
                 "condition": record["condition"], "speaker_id": record["speaker_id"],
                 "utterance_id": record["utterance_id"], "scorer_id": scorer,
                 "prompt_count": len(rows), "valid_frames": record["valid_frames"],
                 "decoded_phone_ids": None, "status": "ok", "reason": None}
    try:
        if record["recording_status"] != "ok":
            raise ValueError("upstream_recording_failure:" + record["failure_reason"])
        emissions = validate_emissions(record["log_emissions"], record["valid_frames"])
        utterance["decoded_phone_ids"] = list(greedy_decode(emissions))
        if not emissions:
            utterance.update(status="abstain", reason="no_valid_frames")
        if not rows:
            utterance.update(status="abstain", reason="empty_prompt")
        if scorer in {"sf_s", "sf_sd"}:
            scores = score_prompt(emissions, record["prompt_phone_ids"],
                                  "S" if scorer == "sf_s" else "SD")
            for row, score in zip(rows, scores):
                row["raw_scores"].update(
                    canonical_log_likelihood=_number(score.canonical_log_mass),
                    relaxed_log_likelihood=_number(score.relaxed_log_mass),
                    zero_canonical_mass=score.canonical_log_mass == -math.inf,
                    zero_relaxed_mass=score.relaxed_log_mass == -math.inf,
                    raw_gop=score.raw_gop,
                    posterior_phone_state_occupancy=score.posterior_target_occupancy,
                    occupancy_floor=1.0,
                    posterior_occupancy_normalized_error=score.occupancy_normalized_error,
                    normalization="raw", occupancy_definition="exact_posterior_nonblank_prompt_phone_state_visits")
                row["scores"]["s_error"] = score.s_error
                row.update(status=score.status, abstention_reason=score.reason,
                           finite=score.status == "ok")
                row["uncertainty"] = {"calibrated": False, "context_fixed": True,
                                      "physical_localization_supported": False}
        elif scorer == "greedy_dp":
            alignment = levenshtein(record["prompt_phone_ids"], utterance["decoded_phone_ids"])
            utterance["recognition_distance"] = alignment.distance
            utterance["recognition_insertions"] = sum(s.recognition_edit == "insertion" for s in alignment.steps)
            for step in alignment.steps:
                if step.prompt_index is None:
                    continue
                row = rows[step.prompt_index]
                indicator = float(step.recognition_edit != "match")
                row["raw_scores"].update(hard_recognition_mismatch=indicator,
                                         tie_policy="diagonal_deletion_insertion")
                row["alignment"] = {"decoded_index": step.decoded_index,
                                    "recognition_edit": step.recognition_edit}
                row["uncertainty"] = {"multiple_optimal_edit_paths": alignment.ambiguous,
                                      "calibrated": False}
                reason = "no_valid_frames" if not emissions else (
                    "ambiguous_recognition_edit_path" if alignment.ambiguous else None)
                row.update(status="abstain" if reason else "ok", abstention_reason=reason,
                           finite=reason is None)
                row["scores"]["s_error"] = None if reason else indicator
        else:
            alignment = self_aligned_gop(emissions, record["prompt_phone_ids"])
            for i, row in enumerate(rows):
                if alignment is None or not emissions:
                    row["abstention_reason"] = "canonical_alignment_infeasible_or_no_frames"
                    continue
                row["scores"]["s_error"] = alignment.error_scores[i]
                row.update(status="ok", finite=True)
                row["raw_scores"].update(activation_count=len(alignment.phone_frames[i]),
                                         canonical_viterbi_log_mass=alignment.log_path_mass)
                row["alignment"] = {"model_activation_frames": list(alignment.phone_frames[i]),
                                    "physical_phone_segment": False}
                row["uncertainty"] = {"tied_best_path": alignment.tied_best_path,
                                      "calibrated": False}
        if any(row["status"] != "ok" for row in rows) and utterance["status"] == "ok":
            utterance.update(status="abstain", reason="one_or_more_prompt_scores_abstained")
    except (ValueError, ArithmeticError) as exc:
        utterance.update(status="failure", reason=str(exc))
        for row in rows:
            row.update(status="failure", abstention_reason=str(exc), finite=False)
            row["scores"] = {"s_error": None, "s_omit": None, "s_sub": None}
    for row in rows:
        validate_runtime_row(row)
    return rows, utterance
