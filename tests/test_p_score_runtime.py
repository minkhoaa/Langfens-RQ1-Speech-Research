"""Synthetic cache/runtime boundary tests; never load audio, weights, or corpus."""

import json
import math
from pathlib import Path

import pytest

from langfens_prep.p_score.cache import (
    CONFIG, PROVENANCE_FIELDS, RECORD_FIELDS, decoded_record, json_bytes,
    parse_json, validate_cache,
)
from langfens_prep.p_score.cli import run
from langfens_prep.p_score.runtime import reject_gold_fields, score_record, validate_bundle


def _record(*, failure=False):
    row = [math.log(1 / 40)] * 40
    return {
        "speaker_id": "SYNTH_A", "utterance_id": "utt1",
        "source_speaker_id": "SYNTH_A", "source_utterance_id": "utt1",
        "condition": "matched", "prompt_phone_ids": [1, 1],
        "word_indices": [0, 0], "log_emissions": [] if failure else [row, row, row],
        "valid_frames": 0 if failure else 3,
        "recording_status": "failure" if failure else "ok",
        "failure_reason": "synthetic_input_failure" if failure else None,
    }


def _cache():
    provenance = {
        "kind": "synthetic", "contract_sha256": "4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2",
        "phone_map": ["<BLANK>", "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"],
        "model_sha256": None, "emission_source_commit": "a" * 40,
        "preprocessing_sha256": "b" * 64, "prompt_frontend_sha256": "c" * 64,
        "source_manifest_sha256": "d" * 64, "precision": "float32-log-probabilities",
        "runtime_inputs": "waveform_and_written_prompt_only", "source_authorization_id": "synthetic-fixture",
    }
    assert set(provenance) == PROVENANCE_FIELDS
    return {"schema_version": "p-score-cache-v1", "provenance": provenance,
            "records": [_record()]}


def test_p_score_runtime_accounts_all_repeated_positions_and_scorers():
    record = _record()
    for scorer in ("sf_s", "sf_sd", "greedy_dp", "gop_sa"):
        rows, utterance = score_record(record, "synthetic_run_" + scorer, scorer)
        expected = {(row["run_id"], row["protocol"], row["condition"], row["speaker_id"], row["utterance_id"], i): (phone, record["word_indices"][i])
                    for i, (row, phone) in enumerate(zip(rows, record["prompt_phone_ids"]))}
        counts = validate_bundle(rows, expected)
        assert sum(counts.values()) == 2
        assert set(counts) <= {"ok", "abstain", "failure"}
        assert utterance["status"] in {"ok", "abstain"}
        assert rows[0]["prompt_index"] != rows[1]["prompt_index"]
        assert all(row["scores"]["s_omit"] is None and row["scores"]["s_sub"] is None for row in rows)


def test_p_score_runtime_upstream_failure_keeps_each_prompt_position():
    rows, utterance = score_record(_record(failure=True), "synthetic_run", "sf_sd")
    assert [r["status"] for r in rows] == ["failure", "failure"]
    assert all(r["scores"] == {"s_error": None, "s_omit": None, "s_sub": None} for r in rows)
    assert utterance["status"] == "failure"
    expected = {(rows[i]["run_id"], rows[i]["protocol"], rows[i]["condition"], rows[i]["speaker_id"], rows[i]["utterance_id"], i): (rows[i]["prompt_phone_id"], rows[i]["word_index"]) for i in range(2)}
    with pytest.raises(ValueError, match="incomplete"):
        validate_bundle(rows[:1], expected)


def test_p_score_runtime_rejects_nested_annotations_and_timing():
    for forbidden in ({"details": {"perceived_phone": 1}}, {"aux": [{"timing": [0, 1]}]},
                      {"operation_label": "deletion"}, {"TextGrid": "annotation"}):
        with pytest.raises(ValueError, match="forbidden"):
            reject_gold_fields(forbidden)


def test_p_score_cache_parser_rejects_duplicate_keys_and_nonfinite_constants():
    with pytest.raises(ValueError, match="duplicate"):
        parse_json(b'{"a":1,"a":2}')
    with pytest.raises(ValueError, match="nonfinite"):
        parse_json(b'{"a":NaN}')


def test_p_score_cache_schema_synthetic_only_and_exact_fields():
    cache = _cache()
    assert validate_cache(cache, "synthetic") is cache
    with pytest.raises(ValueError, match="schema"):
        validate_cache({**cache, "extra": True}, "synthetic")
    realish = {**cache, "provenance": {**cache["provenance"], "kind": "frozen-e0"}}
    with pytest.raises(ValueError, match="synthetic mode"):
        validate_cache(realish, "synthetic")
    malformed = {**cache, "records": [{**cache["records"][0], "gold_labels": []}]}
    with pytest.raises(ValueError, match="forbidden"):
        validate_cache(malformed, "synthetic")
    assert set(cache["records"][0]) == RECORD_FIELDS


def test_p_score_synthetic_cli_writes_hashed_manifest_and_refuses_overwrite(tmp_path):
    cache_path, config_path, output = tmp_path / "fixture.json", Path("configs/rq1_p_score_raw.json"), tmp_path / "run"
    cache_path.write_bytes(json_bytes(_cache()))
    manifest = run(cache_path, output, config_path, "synthetic_run", repo=Path.cwd())
    assert manifest["training"] == "NOT_STARTED"
    assert manifest["weights_loaded"] is False and manifest["new_model_artifact"] is None
    assert manifest["normalization"] == "raw"
    assert manifest["occupancy_normalization"].endswith("not_paper_Eq13")
    assert set(manifest["coverage"]) == set(CONFIG["scorers"])
    sums = (output / "SHA256SUMS").read_text().splitlines()
    assert len(sums) == 4 * 3 + 1
    parsed = json.loads((output / "run_manifest.json").read_text())
    assert parsed["source_sha256"] and parsed["git_dirty"]
    with pytest.raises(ValueError, match="exists"):
        run(cache_path, output, config_path, "another_run", repo=Path.cwd())
