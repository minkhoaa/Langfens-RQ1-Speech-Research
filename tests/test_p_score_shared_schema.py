"""Bridge P-SCORE's real emitted rows to the frozen shared strict validator.

Only synthetic log-emission arrays are used. The validator is loaded directly
from the synchronized shared readiness directory; it is not copied or weakened.
"""

import importlib.util
import math
from pathlib import Path

import pytest

from langfens_prep.p_score.runtime import reject_gold_fields, score_record

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SHARED_VALIDATOR_PATH = (
    REPOSITORY_ROOT / "manifests/RQ1-WORKSPACE-PREP-01/implementation-readiness/"
    "strict_runtime_validator.py"
).resolve()
assert SHARED_VALIDATOR_PATH.is_relative_to(REPOSITORY_ROOT.resolve())
assert SHARED_VALIDATOR_PATH.is_file()
spec = importlib.util.spec_from_file_location("rq1_shared_strict_runtime_validator", SHARED_VALIDATOR_PATH)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)

SCORERS = ("sf_s", "sf_sd", "greedy_dp", "gop_sa")


def test_p_score_shared_validator_is_loaded_from_this_checkout():
    expected = (REPOSITORY_ROOT / "manifests/RQ1-WORKSPACE-PREP-01/"
                "implementation-readiness/strict_runtime_validator.py").resolve()
    assert SHARED_VALIDATOR_PATH == expected
    assert Path(shared.__file__).resolve() == expected
    assert expected.is_relative_to(REPOSITORY_ROOT.resolve())


def _emissions(symbols):
    rows = []
    for symbol in symbols:
        row = [0.1 / 39] * 40
        row[symbol] = 0.9
        rows.append([math.log(value) for value in row])
    return rows


def _record(kind="normal"):
    record = {
        "speaker_id": "SYNTH_A", "utterance_id": "repeat_1",
        "source_speaker_id": "SYNTH_A", "source_utterance_id": "repeat_1",
        "condition": "matched", "prompt_phone_ids": [1, 1, 2],
        "word_indices": [0, 0, 0], "log_emissions": _emissions([1, 0, 1, 2]),
        "valid_frames": 4, "recording_status": "ok", "failure_reason": None,
    }
    if kind == "no_frames":
        record.update(log_emissions=[], valid_frames=0)
    elif kind == "recording_failure":
        record.update(log_emissions=[], valid_frames=0, recording_status="failure",
                      failure_reason="synthetic_recording_failure")
    elif kind == "invalid_emissions":
        record["log_emissions"] = [[math.nan] * 40]
        record["valid_frames"] = 1
    return record


def _expected(rows, record):
    result = {}
    for row in rows:
        key = tuple(row[field] for field in shared.KEY_FIELDS)
        index = row["prompt_index"]
        result[key] = (record["prompt_phone_ids"][index], record["word_indices"][index])
    return result


@pytest.mark.parametrize("scorer", SCORERS)
def test_p_score_emitted_rows_pass_shared_schema_for_repeated_prompt(scorer):
    record = _record()
    rows, _ = score_record(record, "schema_bridge_normal_" + scorer, scorer)
    assert len(rows) == len(record["prompt_phone_ids"])
    assert shared.validate_bundle(rows, _expected(rows, record))["recorded_positions"] == 3
    for row in rows:
        assert shared.loads_record(shared.dumps_record(row)) == row
    assert [row["prompt_index"] for row in rows] == [0, 1, 2]
    if scorer in {"sf_s", "sf_sd"}:
        assert "posterior_phone_state_occupancy" in rows[0]["raw_scores"]
        assert "posterior_target_occupancy" not in rows[0]["raw_scores"]


@pytest.mark.parametrize("kind", ("no_frames", "recording_failure", "invalid_emissions"))
@pytest.mark.parametrize("scorer", SCORERS)
def test_p_score_shared_schema_accounts_abstention_and_failure_positions(kind, scorer):
    record = _record(kind)
    rows, utterance = score_record(record, "schema_bridge_" + kind + "_" + scorer, scorer)
    expected = _expected(rows, record)
    assert len(rows) == len(record["prompt_phone_ids"])
    assert shared.validate_bundle(rows, expected)["recorded_positions"] == 3
    for row in rows:
        assert shared.loads_record(shared.dumps_record(row)) == row
        assert row["status"] in {"abstain", "failure"}
        assert row["scores"] == {"s_error": None, "s_omit": None, "s_sub": None}
    assert utterance["status"] in {"abstain", "failure"}


def test_p_score_nested_gold_fields_remain_rejected_by_both_guards():
    with pytest.raises(ValueError, match="forbidden"):
        reject_gold_fields({"raw_scores": {"nested": [{"gold_labels": []}]}})
    record = _record()
    rows, _ = score_record(record, "schema_bridge_nested_gold", "sf_sd")
    rows[0]["raw_scores"]["nested"] = [{"gold_labels": []}]
    with pytest.raises(ValueError):
        shared.validate_record(rows[0])
