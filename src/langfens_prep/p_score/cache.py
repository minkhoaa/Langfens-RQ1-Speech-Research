"""Hashed external cache boundary, no model/dataset/weight loading shortcuts.

Synthetic mode accepts synthetic namespaces only. Future frozen-cache mode
requires a separately authorized, SHA-pinned registration and exact allowlist.
The registration records approval; it is not itself evidence approval exists.
"""

import hashlib
import json
from pathlib import Path
import re

from .ctc import GRAPH_REVISION, validate_phones
from .runtime import PROTOCOL, SCHEMA, SEALED_SPEAKERS, finite_tree, reject_gold_fields

CONTRACT_SHA256 = "4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2"
SPLIT_SHA256 = "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b"
E0_SHA256 = "a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b"
PHONE_MAP = ("<BLANK> AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG "
             "OW OY P R S SH T TH UH UW V W Y Z ZH").split()
PROVENANCE_FIELDS = {
    "kind", "contract_sha256", "phone_map", "model_sha256", "emission_source_commit",
    "preprocessing_sha256", "prompt_frontend_sha256", "source_manifest_sha256",
    "precision", "runtime_inputs", "source_authorization_id",
}
RECORD_FIELDS = {
    "speaker_id", "utterance_id", "source_speaker_id", "source_utterance_id", "condition",
    "prompt_phone_ids", "word_indices", "log_emissions", "valid_frames",
    "recording_status", "failure_reason",
}
CONFIG = {
    "schema_version": "p-score-config-v1", "runtime_schema": SCHEMA,
    "protocol": PROTOCOL, "graph_revision": GRAPH_REVISION,
    "normalization": "raw", "occupancy_floor": 1,
    "scorers": ["sf_s", "sf_sd", "greedy_dp", "gop_sa"],
    "calibration": None, "sdi_supported": False,
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def parse_json(data: bytes):
    def reject_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    value = json.loads(data, object_pairs_hook=_unique_object, parse_constant=reject_constant)
    if not finite_tree(value):
        raise ValueError("nonfinite JSON number or unsupported value")
    return value


def json_bytes(value) -> bytes:
    if not finite_tree(value):
        raise ValueError("nonfinite JSON output")
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _hash(value, length: int = 64) -> bool:
    return type(value) is str and re.fullmatch("[0-9a-f]{" + str(length) + "}", value) is not None


def _identity(speaker: str, utterance: str) -> str:
    if (type(speaker) is not str or type(utterance) is not str
            or not re.fullmatch(r"[A-Za-z0-9_-]+", speaker)
            or not re.fullmatch(r"[A-Za-z0-9_-]+", utterance)):
        raise ValueError("invalid speaker/utterance identity")
    if speaker in SEALED_SPEAKERS:
        raise ValueError("sealed TEST speaker prohibited")
    return speaker + "/" + utterance


def record_key(record: dict) -> tuple[str, str, str]:
    return record["condition"], record["speaker_id"], record["utterance_id"]


def validate_cache(cache: dict, mode: str) -> dict:
    reject_gold_fields(cache)
    if set(cache) != {"schema_version", "provenance", "records"} or cache["schema_version"] != "p-score-cache-v1":
        raise ValueError("cache schema mismatch")
    provenance = cache["provenance"]
    if set(provenance) != PROVENANCE_FIELDS:
        raise ValueError("cache provenance fields mismatch")
    if provenance["contract_sha256"] != CONTRACT_SHA256 or provenance["phone_map"] != PHONE_MAP:
        raise ValueError("contract or phone/blank mapping mismatch")
    if provenance["precision"] != "float32-log-probabilities":
        raise ValueError("cache must declare FP32 log probabilities")
    if provenance["runtime_inputs"] != "waveform_and_written_prompt_only":
        raise ValueError("cache source is not prompt-only")
    for field in ("preprocessing_sha256", "prompt_frontend_sha256", "source_manifest_sha256"):
        if not _hash(provenance[field]):
            raise ValueError("missing source/cache provenance hash")
    if not _hash(provenance["emission_source_commit"], 40):
        raise ValueError("missing emission-source commit")
    if type(provenance["source_authorization_id"]) is not str or not provenance["source_authorization_id"]:
        raise ValueError("source authorization identity required")
    if mode == "synthetic":
        if provenance["kind"] != "synthetic" or provenance["model_sha256"] is not None:
            raise ValueError("synthetic mode cannot accept a real model/cache")
    elif mode == "frozen-cache":
        if provenance["kind"] != "frozen-e0" or provenance["model_sha256"] != E0_SHA256:
            raise ValueError("only registered frozen E0 identity accepted")
    else:
        raise ValueError("unsupported input mode")
    if type(cache["records"]) is not list or not cache["records"]:
        raise ValueError("nonempty explicit record inventory required")
    seen = set()
    for record in cache["records"]:
        if set(record) != RECORD_FIELDS:
            raise ValueError("cache record fields mismatch")
        identity = _identity(record["speaker_id"], record["utterance_id"])
        source = _identity(record["source_speaker_id"], record["source_utterance_id"])
        if mode == "synthetic" and not all(x.startswith("SYNTH_") for x in
                                          (record["speaker_id"], record["source_speaker_id"])):
            raise ValueError("synthetic namespace required")
        if record["condition"] not in {"matched", "within_speaker_cyclic"}:
            raise ValueError("unregistered acoustic condition")
        if record["condition"] == "matched" and identity != source:
            raise ValueError("matched source identity differs")
        if record["condition"] == "within_speaker_cyclic" and (
                identity == source or record["source_speaker_id"] != record["speaker_id"]):
            raise ValueError("mismatch must be within-speaker without fixed points")
        if type(record["prompt_phone_ids"]) is not list:
            raise ValueError("prompt list required")
        validate_phones(record["prompt_phone_ids"], 40)
        words = record["word_indices"]
        if (type(words) is not list or len(words) != len(record["prompt_phone_ids"])
                or any(w is not None and (type(w) is not int or w < 0) for w in words)):
            raise ValueError("invalid prompt word mapping")
        if type(record["valid_frames"]) is not int or record["valid_frames"] < 0:
            raise ValueError("invalid cache valid_frames")
        if type(record["log_emissions"]) is not list:
            raise ValueError("emission array required")
        # JSON null is the only encoding of log zero (-inf); no JSON NaN/Infinity.
        for row in record["log_emissions"]:
            if type(row) is not list or any(x is not None and type(x) not in (int, float) for x in row):
                raise ValueError("invalid log-emission values")
        if record["recording_status"] not in {"ok", "failure"}:
            raise ValueError("invalid upstream recording status")
        reason = record["failure_reason"]
        if record["recording_status"] == "ok":
            if reason is not None:
                raise ValueError("ok record has failure reason")
        elif (type(reason) is not str or not reason or record["log_emissions"]
              or record["valid_frames"] != 0):
            raise ValueError("failure must have reason and no emissions")
        key = record_key(record)
        if key in seen:
            raise ValueError("duplicate cache record identity")
        seen.add(key)
    return cache


def authorize_cache(cache: dict, cache_hash: str, authorization: dict,
                    split_path: Path) -> None:
    """Future gate: exact cache/provenance and complete allowlisted population.

    Reads only frozen split metadata, never TEST contents, audio, or labels.
    Operator registration must be reviewed independently, not self-certified.
    """
    fields = {"schema_version", "authorization_id", "scope", "contract_sha256", "cache_sha256",
              "provenance", "allowed_ids", "conditions", "fit_speakers", "selection_speakers",
              "calibration_speakers", "evaluation_speakers", "parity_disposition"}
    if set(authorization) != fields or authorization["schema_version"] != "p-score-authorization-v1":
        raise ValueError("authorization fields mismatch")
    if (authorization["scope"] != "frozen_emission_scoring_only"
            or not authorization["authorization_id"]
            or authorization["contract_sha256"] != CONTRACT_SHA256
            or authorization["cache_sha256"] != cache_hash
            or authorization["provenance"] != cache["provenance"]
            or not authorization["parity_disposition"]):
        raise ValueError("authorization/cache/provenance mismatch")
    data = split_path.read_bytes()
    if sha256(data) != SPLIT_SHA256:
        raise ValueError("frozen split hash mismatch")
    split = parse_json(data)
    fold = next(x for x in split["folds"] if x["fold_index"] == 1)
    permitted = set(fold["train"]["utterances"]) | set(fold["val"]["utterances"])
    allowed = authorization["allowed_ids"]
    if (type(allowed) is not list or not allowed or len(set(allowed)) != len(allowed)
            or not set(allowed) <= permitted):
        raise ValueError("authorization includes non-train/val or duplicate IDs")
    conditions = authorization["conditions"]
    if (type(conditions) is not list or not conditions or len(set(conditions)) != len(conditions)
            or not set(conditions) <= {"matched", "within_speaker_cyclic"}):
        raise ValueError("invalid authorized conditions")
    expected = {(condition, *identity.split("/")) for condition in conditions for identity in allowed}
    if {record_key(r) for r in cache["records"]} != expected:
        raise ValueError("incomplete or extra authorized utterance accounting")
    ordered_by_speaker = {}
    for identity in allowed:
        speaker, _ = identity.split("/")
        if speaker in SEALED_SPEAKERS:
            raise ValueError("sealed TEST prohibited")
        ordered_by_speaker.setdefault(speaker, []).append(identity)
    actual_speakers = set(ordered_by_speaker)
    if set(authorization["evaluation_speakers"]) != actual_speakers:
        raise ValueError("evaluation speaker declaration mismatch")
    for field in ("fit_speakers", "selection_speakers", "calibration_speakers", "evaluation_speakers"):
        speakers = authorization[field]
        if (type(speakers) is not list or len(set(speakers)) != len(speakers)
                or set(speakers) & SEALED_SPEAKERS):
            raise ValueError("invalid role-speaker declaration")
    for record in cache["records"]:
        source = _identity(record["source_speaker_id"], record["source_utterance_id"])
        if source not in allowed:
            raise ValueError("source not allowlisted")
        if record["condition"] == "within_speaker_cyclic":
            group = ordered_by_speaker[record["speaker_id"]]
            own = _identity(record["speaker_id"], record["utterance_id"])
            if len(group) < 2 or source != group[(group.index(own) + 1) % len(group)]:
                raise ValueError("mismatch is not ordered within-speaker cyclic shift-1")


def decoded_record(record: dict) -> dict:
    result = dict(record)
    result["log_emissions"] = [[-float("inf") if x is None else x for x in row]
                               for row in record["log_emissions"]]
    return result
