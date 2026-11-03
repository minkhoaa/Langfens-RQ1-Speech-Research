"""Dependency-free validation of this directory's runtime schema, not a scorer.

Only the JSON Schema keywords used by PREDICTION_SCHEMA_V1.json are implemented.
No model, corpus, arm source, network, or training imports. See accounting document.
"""
import json
import math
import re
from pathlib import Path

SCHEMA = json.loads(Path(__file__).with_name("PREDICTION_SCHEMA_V1.json").read_text())
KEY_FIELDS = ("run_id", "protocol", "condition", "speaker_id", "utterance_id", "prompt_index")
SEALED_SPEAKERS = frozenset(("ABA", "ASI", "EBVS", "HJK", "LXC", "PNV"))


def _reject(message):
    raise ValueError(message)


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            _reject("duplicate JSON name: " + key)
        out[key] = value
    return out


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        _reject("nonfinite JSON number")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                _reject("non-string object name")
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)


def _metadata_origin(value):
    if isinstance(value, str) and re.search(r"gold|textgrid|oracle|perceived|realized", value, re.I):
        _reject("forbidden annotation provenance in runtime metadata")
    if isinstance(value, dict):
        for item in value.values():
            _metadata_origin(item)
    elif isinstance(value, list):
        for item in value:
            _metadata_origin(item)


def _schema(value, rule):
    if "$ref" in rule:
        return _schema(value, SCHEMA["$defs"][rule["$ref"].split("/")[-1]])
    if "anyOf" in rule:
        for option in rule["anyOf"]:
            try:
                _schema(value, option)
                return
            except ValueError:
                pass
        _reject("no matching schema alternative")
    if "const" in rule and value != rule["const"]:
        _reject("wrong schema constant")
    if "enum" in rule and value not in rule["enum"]:
        _reject("invalid enum")
    types = {
        "null": lambda x: x is None,
        "boolean": lambda x: type(x) is bool,
        "integer": lambda x: type(x) is int,
        "number": lambda x: type(x) in (int, float),
        "string": lambda x: isinstance(x, str),
        "object": lambda x: isinstance(x, dict),
        "array": lambda x: isinstance(x, list),
    }
    if "type" in rule and not types[rule["type"]](value):
        _reject("wrong schema type: " + rule["type"])
    for bound, compare in (("minimum", lambda a, b: a < b), ("maximum", lambda a, b: a > b)):
        if bound in rule and compare(value, rule[bound]):
            _reject("schema numeric bound")
    if "minLength" in rule and len(value) < rule["minLength"]:
        _reject("empty required string")
    if "pattern" in rule and re.search(rule["pattern"], value) is None:
        _reject("forbidden or off-schema metadata name")
    if isinstance(value, dict):
        if set(rule.get("required", ())) - value.keys():
            _reject("missing required fields")
        properties = rule.get("properties", {})
        extra = rule.get("additionalProperties", True)
        for key, item in value.items():
            if "propertyNames" in rule:
                _schema(key, rule["propertyNames"])
            if key in properties:
                _schema(item, properties[key])
            elif extra is False:
                _reject("off-schema field: " + key)
            elif isinstance(extra, dict):
                _schema(item, extra)
    if isinstance(value, list) and "items" in rule:
        for item in value:
            _schema(item, rule["items"])


def validate_record(record):
    """Validate a parsed row; return it. Parse text with loads_record first."""
    _finite(record)
    _schema(record, SCHEMA)
    for field in ("raw_scores", "probabilities", "alignment", "uncertainty", "confidence"):
        _metadata_origin(record[field])
    if record["speaker_id"] in SEALED_SPEAKERS:
        _reject("sealed Fold-1 TEST speaker")
    if record["status"] == "ok":
        if not record["finite"] or record["scores"]["s_error"] is None:
            _reject("ok requires finite error score")
        if record["abstention_reason"] is not None:
            _reject("ok cannot have abstention reason")
    else:
        if record["abstention_reason"] is None:
            _reject("failure/abstention requires reason")
        if any(value is not None for value in record["scores"].values()):
            _reject("non-ok rows cannot issue decision scores")
    if (record["threshold"] is None) != (record["calibration_reference"] is None):
        _reject("threshold and calibration identity must be paired")
    return record


def loads_record(text):
    """Reject duplicate names at every depth and non-JSON NaN/Infinity literals."""
    value = json.loads(text, object_pairs_hook=_pairs,
                       parse_constant=lambda token: _reject("invalid JSON constant: " + token))
    return validate_record(value)


def dumps_record(record):
    """Validate and serialize one row as strict finite JSON (never NaN/Infinity)."""
    validate_record(record)
    return json.dumps(record, allow_nan=False, separators=(",", ":"))


def validate_bundle(records, expected):
    """Require exact position accounting against an independently frozen ledger.

    expected maps six-part KEY_FIELDS tuples to (prompt_phone_id, word_index).
    It must come from allowed prompt metadata, never from rows being validated.
    This function does not open or derive any real-data ledger.
    """
    seen = set()
    for record in records:
        validate_record(record)
        key = tuple(record[field] for field in KEY_FIELDS)
        if key in seen:
            _reject("duplicate runtime position")
        seen.add(key)
        if key not in expected:
            _reject("unexpected runtime position")
        if (record["prompt_phone_id"], record["word_index"]) != expected[key]:
            _reject("prompt identity mismatch")
    if seen != set(expected):
        _reject("incomplete runtime position accounting")
    return {"expected_positions": len(expected), "recorded_positions": len(seen)}
