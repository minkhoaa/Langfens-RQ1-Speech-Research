"""Synthetic JSON-only tests. No audio, labels, weights, torch, network or fitting."""
import copy
import json
import unittest

from strict_runtime_validator import KEY_FIELDS, dumps_record, loads_record, validate_bundle, validate_record


def row(index=0):
    return {
        "schema_version": "rq1-runtime-v1", "run_id": "synthetic-only",
        "protocol": "Z-PROMPT-V1", "condition": "fixture",
        "speaker_id": "SYNTH", "utterance_id": "fixture",
        "prompt_index": index, "prompt_phone_id": 1, "word_index": None,
        "scores": {"s_error": 0.25, "s_omit": None, "s_sub": None},
        "raw_scores": {"raw_gop": -0.25}, "probabilities": None,
        "score_orientation": "higher_is_error",
        "score_semantics": "Synthetic negative GOP; unsupported subtypes null; not probability",
        "alignment": None, "uncertainty": None, "confidence": None,
        "status": "ok", "abstention_reason": None, "finite": True,
        "threshold": None, "calibration_reference": None,
    }


def key(record):
    return tuple(record[field] for field in KEY_FIELDS)


class SharedRuntimeTests(unittest.TestCase):
    def test_valid_roundtrip(self):
        encoded = dumps_record(row())
        self.assertEqual(loads_record(encoded), row())
        self.assertNotIn('": ', encoded)
        self.assertNotIn(", ", encoded)

    def test_strict_serializer_rejects_nonfinite(self):
        r = row()
        r["raw_scores"]["raw_gop"] = float("nan")
        with self.assertRaises(ValueError):
            dumps_record(r)

    def test_repeated_phone_positions_are_distinct(self):
        records = [row(0), row(1)]
        ledger = {key(r): (1, None) for r in records}
        self.assertEqual(validate_bundle(records, ledger)["recorded_positions"], 2)

    def test_duplicate_position_rejected(self):
        r = row()
        with self.assertRaises(ValueError):
            validate_bundle([r, copy.deepcopy(r)], {key(r): (1, None)})

    def test_missing_position_rejected(self):
        with self.assertRaises(ValueError):
            validate_bundle([row()], {key(row()): (1, None), key(row(1)): (1, None)})

    def test_unexpected_position_rejected(self):
        with self.assertRaises(ValueError):
            validate_bundle([row()], {})

    def test_prompt_identity_rejected(self):
        with self.assertRaises(ValueError):
            validate_bundle([row()], {key(row()): (2, None)})

    def test_top_level_gold_and_extra_rejected(self):
        for name in ("gold_phone", "perceived_phones", "textgrid", "operation_labels", "extra"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_record(dict(row(), **{name: 1}))

    def test_nested_gold_rejected(self):
        for name in ("gold_index", "GOLD", "textgrid_time", "perceived_ids", "realized_phone",
                     "operation_id", "binary_labels", "xmin", "reference_phone", "target"):
            r = row()
            r["raw_scores"] = {"nested": [{name: 1}]}
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_record(r)

    def test_explicit_annotation_provenance_rejected(self):
        for origin in ("TextGrid", "gold", "oracle", "perceived_sequence", "realized_sequence"):
            r = row()
            r["alignment"] = {"method": origin, "start": 0.0}
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                validate_record(r)

    def test_duplicate_json_names_rejected(self):
        for text in ('{"run_id":"x","run_id":"y"}',
                     '{"raw_scores":{"x":1,"x":2}}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                loads_record(text)

    def test_nonfinite_literals_and_overflow_rejected(self):
        for token in ("NaN", "Infinity", "-Infinity", "1e999"):
            text = json.dumps(row()).replace('"raw_gop": -0.25', '"raw_gop": ' + token)
            with self.subTest(token=token), self.assertRaises(ValueError):
                loads_record(text)

    def test_programmatic_nested_nonfinite_rejected(self):
        r = row()
        r["uncertainty"] = {"values": [float("inf")]}
        with self.assertRaises(ValueError):
            validate_record(r)

    def test_invalid_phone_and_indices_rejected(self):
        for field, value in (("prompt_phone_id", 0), ("prompt_phone_id", 40),
                             ("prompt_index", -1), ("prompt_index", True),
                             ("word_index", -1), ("prompt_phone_id", 1.0)):
            r = dict(row(), **{field: value})
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_record(r)

    def test_schema_protocol_orientation_rejected(self):
        for field, value in (("schema_version", "v0"), ("protocol", "O-ARCHIVE"),
                             ("score_orientation", "higher_is_correct")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_record(dict(row(), **{field: value}))

    def test_sealed_speaker_rejected(self):
        with self.assertRaises(ValueError):
            validate_record(dict(row(), speaker_id="ABA"))

    def test_failure_accounts_for_expected_position(self):
        r = row()
        r.update(status="failure", abstention_reason="synthetic impossible graph", finite=False)
        r["scores"] = dict.fromkeys(r["scores"], None)
        self.assertEqual(validate_bundle([r], {key(r): (1, None)})["recorded_positions"], 1)

    def test_status_consistency_rejected(self):
        variants = [dict(row(), finite=False), dict(row(), abstention_reason="bad"),
                    dict(row(), status="failure"), dict(row(), status="abstain", abstention_reason="bad")]
        r = row()
        r["scores"]["s_error"] = None
        variants.append(r)
        for i, record in enumerate(variants):
            with self.subTest(case=i), self.assertRaises(ValueError):
                validate_record(record)

    def test_threshold_calibration_pair(self):
        validate_record(dict(row(), threshold=0.5, calibration_reference="synthetic-cal-only"))
        with self.assertRaises(ValueError):
            validate_record(dict(row(), threshold=0.5))

    def test_probability_range_and_bool_rejected(self):
        for value in (-0.1, 1.1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_record(dict(row(), probabilities={"p_error": value}))

    def test_required_field_and_score_extra_rejected(self):
        r = row()
        del r["confidence"]
        with self.assertRaises(ValueError):
            validate_record(r)
        r = row()
        r["scores"]["invented"] = 0.5
        with self.assertRaises(ValueError):
            validate_record(r)

    def test_unknown_subtypes_remain_null(self):
        r = validate_record(row())
        self.assertIsNone(r["scores"]["s_omit"])
        self.assertIsNone(r["scores"]["s_sub"])
        self.assertIsNone(r["probabilities"])


if __name__ == "__main__":
    unittest.main()
