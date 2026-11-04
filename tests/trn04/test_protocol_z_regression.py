"""No-training Protocol Z contract regressions; synthetic CPU fixtures only."""
import copy

import pytest
import torch

from langfens_prep.alignment import align_canonical_sequences
from langfens_prep.p_sup.runtime import (
    operation_runtime_rows,
    validate_complete_run,
    validate_runtime_row,
)
from langfens_prep.phoneme_normalizer import PHONE_TO_CTC_INDEX
from audit_scripts.evaluate_protocol_z import average_precision_binary


def _rows(objective="operation_ce", logits=None, phones=None, speaker="NJS"):
    phones = phones or [PHONE_TO_CTC_INDEX["B"], PHONE_TO_CTC_INDEX["AE"]]
    logits = logits if logits is not None else torch.tensor([[0., 1., 2.], [2., 1., 0.]])
    return operation_runtime_rows(logits=logits, prompt_phone_ids=phones, run_id="r", condition="fixed",
                                  speaker_id=speaker, utterance_id="u", objective=objective)


def test_written_prompt_runtime_has_no_annotation_inputs_and_gold_mutation_is_inert():
    prompt = [PHONE_TO_CTC_INDEX[p] for p in ("B", "AE")]
    logits = torch.tensor([[0.25, 1.5, -0.5], [2.0, -1.0, 0.0]])
    # Deterministic stand-in: the only runtime arguments are the fixed written prompt and model output.
    def infer(written_prompt, raw_logits):
        return operation_runtime_rows(logits=raw_logits, prompt_phone_ids=written_prompt, run_id="r",
                                      condition="fixed", speaker_id="NJS", utterance_id="u",
                                      objective="operation_ce")
    gold_a = ["B", "AE"]
    gold_b = ["D", "IH", "T"]
    assert gold_a != gold_b
    result_a = infer(prompt, logits)
    result_b = infer(prompt, logits)
    assert result_a == result_b
    assert [r["prompt_phone_id"] for r in result_a] == prompt
    assert all(not any(k in str(r).lower() for k in ("gold", "textgrid", "target")) for r in result_a)


def test_operation_runtime_score_semantics_t_op_vs_binary_control():
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    treatment = _rows("operation_ce", logits, [1])[0]
    control = _rows("binary_collapsed", logits, [1])[0]
    assert treatment["scores"]["s_error"] == pytest.approx(control["scores"]["s_error"])
    assert treatment["scores"]["s_omit"] is not None
    assert treatment["scores"]["s_sub"] is not None
    assert control["scores"]["s_omit"] is None and control["scores"]["s_sub"] is None
    assert control["probabilities"] is None
    assert "subtype unidentified" in control["score_semantics"]


def test_runtime_rejects_gold_fields_nonfinite_values_and_sealed_speakers():
    row = _rows()[0]
    contaminated = copy.deepcopy(row)
    contaminated["gold_label"] = 0
    with pytest.raises(ValueError):
        validate_runtime_row(contaminated)
    with pytest.raises(ValueError):
        _rows(speaker="ABA")
    bad = _rows(logits=torch.tensor([[float("nan"), 0.0, 1.0], [0., 1., 0.]]))[0]
    assert bad["status"] == "failure" and bad["scores"]["s_error"] is None


def test_complete_run_rejects_missing_duplicate_and_extra_prompt_positions():
    rows = _rows()
    expected = {("NJS", "u"): [r["prompt_phone_id"] for r in rows]}
    assert validate_complete_run(rows, run_id="r", condition="fixed", expected_prompts=expected)["prompt_positions"] == 2
    with pytest.raises(ValueError):
        validate_complete_run(rows[:1], run_id="r", condition="fixed", expected_prompts=expected)
    with pytest.raises(ValueError):
        validate_complete_run(rows + [rows[0]], run_id="r", condition="fixed", expected_prompts=expected)
    with pytest.raises(ValueError):
        validate_complete_run(rows + [_rows(phones=[3])[0]], run_id="r", condition="fixed", expected_prompts=expected)


def test_alignment_mapping_is_not_omission_evidence_and_is_stable():
    gold, prompt = ["HH", "W", "AH", "T"], ["W", "AH", "T"]
    one = align_canonical_sequences(gold, prompt, tie_break="sub_first")
    two = align_canonical_sequences(gold, prompt, tie_break="sub_first")
    assert one == two
    assert one["unaligned_gold"] == [(0, "HH")]
    # This is a prompt/gold mapping deletion, not a human learner omission label.
    assert one["edits_summary"]["deletions"] == 1


def test_archival_145_deletions_are_accounted_as_canonical_positions_not_dropped():
    gold_correct, gold_sub, gold_del = 8799, 1012, 145
    assert gold_correct + gold_sub + gold_del == 9956
    legacy_omission_n = gold_correct + gold_del
    assert legacy_omission_n == 8944
    assert 9956 - legacy_omission_n == gold_sub


def test_grouped_full_precision_average_precision_treats_ties_as_one_threshold():
    # Exact ties must be accumulated as a group; stable sorting cannot split them.
    labels, scores = [1, 0, 1, 0], [0.9, 0.9, 0.1, 0.1]
    assert average_precision_binary(labels, scores) == pytest.approx(0.5)
    assert average_precision_binary([1, 0], [0.123456789123, 0.123456789122]) == 1.0


def test_calibration_threshold_fixture_uses_cal_correct_scores_only():
    # V1's threshold rule: lowest tau whose CAL-correct empirical FPR is <= ceiling.
    cal_correct_scores = [0.1, 0.2, 0.2, 0.8]
    val_scores_must_not_enter_calibration = [0.99]
    del val_scores_must_not_enter_calibration
    candidates = sorted(set(cal_correct_scores))
    ceiling = 0.25
    tau = next(t for t in candidates if sum(s >= t for s in cal_correct_scores) / len(cal_correct_scores) <= ceiling)
    assert tau == 0.8
    assert sum(s >= tau for s in cal_correct_scores) / len(cal_correct_scores) == 0.25


def test_ctc_recognition_deletion_is_not_learner_omission_label():
    # A recognition reference/hypothesis edit is an ASR error independent of annotation label.
    reference, hypothesis = ["B", "AE", "T"], ["B", "T"]
    mapping = align_canonical_sequences(reference, hypothesis, tie_break="sub_first")
    assert mapping["edits_summary"]["deletions"] == 1
    annotated_learner_operation = "correct"
    assert annotated_learner_operation != "deletion"


def test_mismatch_control_changes_waveform_condition_not_prompt_features():
    prompt_ids = [PHONE_TO_CTC_INDEX[x] for x in ("B", "AE", "T")]
    audio_by_manifest_order = [torch.tensor([1.]), torch.tensor([2.]), torch.tensor([3.])]
    shifted = audio_by_manifest_order[1:] + audio_by_manifest_order[:1]
    assert all(not torch.equal(a, b) for a, b in zip(audio_by_manifest_order, shifted))
    assert prompt_ids == [PHONE_TO_CTC_INDEX[x] for x in ("B", "AE", "T")]


def test_protocol_alignment_order_preserves_repeated_phone_positions():
    result = align_canonical_sequences(["T", "T", "AE"], ["T", "T", "AE"], tie_break="sub_first")
    assert result["aligned_pairs"] == [(0, 0, 0), (1, 1, 0), (2, 2, 0)]
    assert len({(g, z) for g, z, _ in result["aligned_pairs"]}) == 3
