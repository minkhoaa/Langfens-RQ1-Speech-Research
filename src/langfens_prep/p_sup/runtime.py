"""rq1-runtime-v1 serialization; offline labels have no place in this API."""
import json
import math
from collections.abc import Mapping

import torch

SCHEMA_VERSION = "rq1-runtime-v1"
PROTOCOL = "Z-PROMPT-V1"
SEALED_SPEAKERS = frozenset({"ABA", "ASI", "EBVS", "HJK", "LXC", "PNV"})
FIELDS = frozenset({"schema_version", "run_id", "protocol", "condition", "speaker_id", "utterance_id",
                    "prompt_index", "prompt_phone_id", "word_index", "scores", "raw_scores", "probabilities",
                    "score_orientation", "score_semantics", "alignment", "uncertainty", "confidence", "status",
                    "abstention_reason", "finite", "threshold", "calibration_reference"})
MANIFEST_FIELDS = frozenset({"schema_version", "contract_version", "contract_sha256", "arm", "scorer",
    "score_semantics", "git_base", "git_tip", "git_dirty", "source_hashes", "config_sha256", "package_sha256",
    "label_map_sha256", "prompt_sha256", "g2p_sha256", "weight_sha256", "allowed_id_sha256",
    "fit_speakers", "selection_speakers", "calibration_speakers", "evaluation_speakers", "seeds_rng",
    "hardware", "precision", "determinism", "dependencies", "endpoint", "successful_updates", "skipped_updates",
    "authorization_id", "attempted_settings", "calibration_identity", "timestamps", "output_hashes"})
CONTRACT_SHA256 = "4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2"
BASE_SHA = "0d268ff8c444e16fd76aca2f4ad6ece966908edc"


def finite_json(value):
    """Reject NaN/Inf recursively, even in nested optional objects."""
    return json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _gold_free(value):
    if isinstance(value, Mapping):
        for key, child in value.items():
            lower = key.lower()
            if any(term in lower for term in ("gold", "textgrid", "perceived", "realized", "timestamp", "operation_label", "target", "xmin", "xmax", "error_type", "binary_label")):
                raise ValueError("gold/annotation fields forbidden in runtime")
            _gold_free(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _gold_free(child)


def validate_runtime_row(row):
    if set(row) != FIELDS:
        raise ValueError("unknown/missing runtime field")
    _gold_free(row)
    finite_json(row)
    if row["schema_version"] != SCHEMA_VERSION or row["protocol"] != PROTOCOL or row["score_orientation"] != "higher_is_error":
        raise ValueError("wrong schema/protocol/orientation")
    for field in ("run_id", "condition", "speaker_id", "utterance_id", "score_semantics"):
        if not isinstance(row[field], str) or not row[field]:
            raise ValueError(f"missing identity/semantics: {field}")
    if row["speaker_id"] in SEALED_SPEAKERS:
        raise ValueError("active TEST speaker forbidden")
    if type(row["prompt_index"]) is not int or row["prompt_index"] < 0 or type(row["prompt_phone_id"]) is not int or not 1 <= row["prompt_phone_id"] <= 39:
        raise ValueError("invalid prompt position/phone")
    if row["word_index"] is not None and (type(row["word_index"]) is not int or row["word_index"] < 0):
        raise ValueError("invalid prompt word index")
    if set(row["scores"]) != {"s_error", "s_omit", "s_sub"} or not isinstance(row["raw_scores"], dict):
        raise ValueError("invalid scores")
    for value in row["scores"].values():
        if value is not None and not _number(value):
            raise ValueError("invalid score")
    for field in ("probabilities", "alignment", "uncertainty", "confidence"):
        if row[field] is not None and not isinstance(row[field], dict):
            raise ValueError(f"invalid nullable object: {field}")
    if row["probabilities"] is not None:
        probabilities = row["probabilities"]
        if not probabilities or any(not _number(p) or not 0 <= p <= 1 for p in probabilities.values()):
            raise ValueError("probabilities require finite numeric values in [0,1]")
        if set(probabilities) == {"correct", "substitution", "deletion"}:
            if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-6):
                raise ValueError("operation probabilities must sum to one")
            expected = {"s_error": probabilities["substitution"] + probabilities["deletion"],
                        "s_sub": probabilities["substitution"], "s_omit": probabilities["deletion"]}
            if any(not _number(row["scores"][k]) or not math.isclose(row["scores"][k], v, abs_tol=1e-6)
                   for k, v in expected.items()):
                raise ValueError("operation score/probability mismatch")
    if row["status"] not in {"ok", "abstain", "failure"} or type(row["finite"]) is not bool:
        raise ValueError("invalid status/finite")
    if row["status"] == "ok":
        if row["scores"]["s_error"] is None or not row["finite"] or row["abstention_reason"] is not None:
            raise ValueError("ok row requires finite error score and no abstention")
    elif not isinstance(row["abstention_reason"], str) or not row["abstention_reason"] or any(x is not None for x in row["scores"].values()) or row["probabilities"] is not None:
        raise ValueError("failed/abstained rows require reason and null scores/probabilities")
    if row["threshold"] is not None and (not _number(row["threshold"]) or not row["calibration_reference"]):
        raise ValueError("threshold requires finite value and calibration reference")
    if row["calibration_reference"] is not None and (not isinstance(row["calibration_reference"], str) or not row["calibration_reference"]):
        raise ValueError("invalid calibration reference")
    return row


def operation_runtime_rows(*, logits, prompt_phone_ids, run_id, condition, speaker_id, utterance_id,
                           objective, failure_reason=None):
    """One output per prompt position, including nonfinite/missing-output failures."""
    if objective not in {"operation_ce", "binary_collapsed", "class_focal"}:
        raise ValueError("unregistered objective")
    if not prompt_phone_ids or any(type(p) is not int or not 1 <= p <= 39 for p in prompt_phone_ids):
        raise ValueError("invalid canonical prompt; use utterance failure ledger for empty/OOV prompt")
    if logits is not None and logits.shape != (len(prompt_phone_ids), 3):
        failure_reason = failure_reason or "output_shape_mismatch"
        logits = None
    rows = []
    for index, phone in enumerate(prompt_phone_ids):
        z = logits[index].detach().float().cpu() if logits is not None else None
        reason = failure_reason or ("missing_output" if z is None else None)
        if z is not None and not bool(torch.isfinite(z).all()):
            reason = reason or "nonfinite_logits"
        scores = {"s_error": None, "s_omit": None, "s_sub": None}
        raw, probabilities = {}, None
        if not reason:
            p = z.softmax(-1).tolist()
            scores["s_error"] = p[1] + p[2]
            raw = {"operation_logits": z.tolist()}
            if objective == "operation_ce":
                scores.update(s_omit=p[2], s_sub=p[1])
                probabilities = {"correct": p[0], "substitution": p[1], "deletion": p[2]}
        row = dict(schema_version=SCHEMA_VERSION, run_id=run_id, protocol=PROTOCOL, condition=condition,
                   speaker_id=speaker_id, utterance_id=utterance_id, prompt_index=index,
                   prompt_phone_id=phone, word_index=None, scores=scores, raw_scores=raw,
                   probabilities=probabilities, score_orientation="higher_is_error",
                   score_semantics=("uncalibrated three-way softmax; no calibrated confidence" if objective == "operation_ce"
                                    else "binary collapse of same three logits; subtype unidentified/null; not calibrated"),
                   alignment=None, uncertainty=None, confidence=None, status="failure" if reason else "ok",
                   abstention_reason=reason, finite=not bool(reason), threshold=None, calibration_reference=None)
        rows.append(validate_runtime_row(row))
    return rows


def validate_complete_run(rows, *, run_id, condition, expected_prompts, utterance_failures=()):
    """expected_prompts maps authorized (speaker, utterance) to ordered phone IDs.

    Empty/OOV prompts require an utterance-level failure, not a fabricated token.
    """
    seen, failed = set(), set()
    expected = {}
    for (speaker, utterance), phones in expected_prompts.items():
        if speaker in SEALED_SPEAKERS:
            raise ValueError("TEST membership forbidden")
        for index, phone in enumerate(phones):
            if type(phone) is not int or not 1 <= phone <= 39:
                raise ValueError("invalid expected prompt")
            expected[(speaker, utterance, index)] = phone
    for failure in utterance_failures:
        if set(failure) != {"speaker_id", "utterance_id", "reason"} or not isinstance(failure["reason"], str) or not failure["reason"]:
            raise ValueError("invalid utterance failure")
        key = failure["speaker_id"], failure["utterance_id"]
        if key not in expected_prompts or expected_prompts[key] or key in failed:
            raise ValueError("utterance failure must uniquely account for an empty prompt")
        failed.add(key)
    for row in rows:
        validate_runtime_row(row)
        if row["run_id"] != run_id or row["condition"] != condition:
            raise ValueError("cross-run/condition row")
        key = row["speaker_id"], row["utterance_id"], row["prompt_index"]
        if key in seen or expected.get(key) != row["prompt_phone_id"]:
            raise ValueError("duplicate/extra/mismatched prompt position")
        seen.add(key)
    if seen != set(expected) or failed != {key for key, phones in expected_prompts.items() if not phones}:
        raise ValueError("incomplete prompt/utterance failure accounting")
    return {"utterances": len(expected_prompts), "prompt_positions": len(expected),
            "ok": sum(row["status"] == "ok" for row in rows),
            "abstain": sum(row["status"] == "abstain" for row in rows),
            "failure": sum(row["status"] == "failure" for row in rows), "utterance_failures": len(failed)}


def validate_run_manifest(manifest):
    if set(manifest) != MANIFEST_FIELDS:
        raise ValueError("missing/unknown V1 run provenance field")
    finite_json(manifest)
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["contract_version"] != "V1" or manifest["contract_sha256"] != CONTRACT_SHA256 or manifest["git_base"] != BASE_SHA or manifest["arm"] != "P-SUP":
        raise ValueError("wrong contract/base/arm")
    hash_fields = ("config_sha256", "package_sha256", "label_map_sha256", "prompt_sha256", "g2p_sha256", "weight_sha256", "allowed_id_sha256")
    def digest(value, size=64):
        return isinstance(value, str) and len(value) == size and all(c in "0123456789abcdef" for c in value)
    for field in hash_fields:
        if not digest(manifest[field]):
            raise ValueError(f"unverified hash: {field}")
    if not digest(manifest["git_tip"], 40) or type(manifest["git_dirty"]) is not bool:
        raise ValueError("invalid source receipt")
    for field in ("source_hashes", "output_hashes"):
        if not isinstance(manifest[field], dict) or not manifest[field] or not all(digest(v) for v in manifest[field].values()):
            raise ValueError(f"unverified hash inventory: {field}")
    partitions = []
    for field in ("fit_speakers", "selection_speakers", "calibration_speakers", "evaluation_speakers"):
        speakers = manifest[field]
        if not isinstance(speakers, list) or not speakers or any(not isinstance(s, str) or not s for s in speakers) or len(set(speakers)) != len(speakers) or set(speakers) & SEALED_SPEAKERS:
            raise ValueError("invalid/sealed speaker partition")
        if any(set(speakers) & p for p in partitions):
            raise ValueError("non-disjoint provenance partitions")
        partitions.append(set(speakers))
    for field in ("scorer", "score_semantics", "seeds_rng", "hardware", "precision", "determinism", "dependencies", "endpoint", "authorization_id", "attempted_settings", "timestamps"):
        if not manifest[field]:
            raise ValueError(f"missing provenance: {field}")
    for field in ("successful_updates", "skipped_updates"):
        if type(manifest[field]) is not int or manifest[field] < 0:
            raise ValueError("invalid update counts")
    if manifest["successful_updates"] > 3600:
        raise ValueError("update budget exceeded")
    if manifest["endpoint"] == "fixed_budget_3600" and manifest["successful_updates"] != 3600:
        raise ValueError("false fixed-budget endpoint")
    return manifest
