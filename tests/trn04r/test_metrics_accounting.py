import pytest

from langfens_prep.trn04r.metrics import (average_precision, summarize_cell, factorial_contrasts,
                                            objective_supports_subtypes, _confusion, recognition_summary,
                                            validate_speaker_census, VAL_SPEAKERS, CAL_SPEAKERS)
from audit_scripts.recompute_trn04r_metrics import (primary_scalar_values, evaluate_registered_gates,
                                                   validate_recognition_pair)


def token(i, op, score, *, phone="AA", prompt=None, dedicated=None):
    return {"speaker_id": "S", "utterance_id": "u", "gold_canonical_index": i,
            "prompt_index": i if prompt is None else prompt, "canonical_phone": phone,
            "operation_target": op, "binary_target": int(op != 0), "s_error": score,
            "s_omit": dedicated if dedicated is not None else score,
            "s_sub": dedicated if dedicated is not None else score}


def test_grouped_tie_ap_is_not_row_order_dependent():
    assert average_precision([1, 0], [0.5, 0.5]) == 0.5
    assert average_precision([0, 1], [0.5, 0.5]) == 0.5
    assert average_precision([1, 0, 1], [0.9, 0.8, 0.8]) == pytest.approx(0.8333333333333333)


def test_synthetic_full_gold_alarm_accounting_keeps_substitution_fp_distinct():
    rows = [token(0, 0, .1), token(1, 2, .9), token(2, 1, .8),
            token(3, 0, None),
            {"speaker_id": "S", "utterance_id": "u", "gold_canonical_index": None,
             "prompt_index": 4, "operation_target": None, "binary_target": None,
             "canonical_phone": None, "s_error": .99, "s_omit": .99, "s_sub": .99}]
    cal = [token(i, 0, score) for i, score in enumerate([.1, .2, .3, .4])]
    runtime = [{"speaker_id": "S", "utterance_id": "u", "prompt_index": i,
                "s_error": row["s_error"], "s_omit": row["s_omit"], "s_sub": row["s_sub"]}
               for i, row in enumerate(rows[:4])]
    for row in runtime:
        row["s_omit"] = row["s_sub"] = row["s_error"]
    runtime.append({"speaker_id": "S", "utterance_id": "u", "prompt_index": 4,
                    "s_error": .99, "s_omit": .99, "s_sub": .99})
    cal_runtime = [{"speaker_id": "S", "utterance_id": "u", "prompt_index": i,
                    "s_error": row["s_error"], "s_omit": row["s_omit"], "s_sub": row["s_sub"]}
                   for i, row in enumerate(cal)]
    result = summarize_cell(runtime, rows, cal, [], objective="operation_ce", cell_id="synthetic", cal_runtime=cal_runtime)
    alarm = result["metrics"]["omission_legacy"]["cal_operating_points"]["0.1"]["counts"]
    assert alarm["TP"] == 1
    assert alarm["FP_correct"] == 0
    assert alarm["FP_sub"] == 1
    assert alarm["FN"] == 0
    assert alarm["recall_all_gold"] == 1
    assert alarm["false_alarms_per_1000_full_gold"] == pytest.approx(250)
    assert result["coverage"] == {"gold_positions": 4, "answered_s_error": 3,
                                   "abstained_s_error": 1, "prompt_extras": 1,
                                   "runtime_prompts": 5}
    assert result["extra_prompt_alarms"]["s_error"]["scored"] == 1


def test_binary_collapsed_control_does_not_claim_dedicated_subtype_score():
    rows = [token(0, 0, .2, dedicated=None), token(1, 2, .8, dedicated=None)]
    for row in rows:
        row["s_omit"] = row["s_sub"] = None
    cal = [token(i, 0, .1 + i*.01, dedicated=None) for i in range(3)]
    for row in cal:
        row["s_omit"] = row["s_sub"] = None
    runtime = [{"speaker_id": "S", "utterance_id": "u", "prompt_index": i,
                "s_error": row["s_error"], "s_omit": None, "s_sub": None} for i, row in enumerate(rows)]
    cal_runtime = [{"speaker_id": "S", "utterance_id": "u", "prompt_index": i,
                    "s_error": row["s_error"], "s_omit": None, "s_sub": None}
                   for i, row in enumerate(cal)]
    out = summarize_cell(runtime, rows, cal, [], objective="binary_collapsed", cell_id="B-Control", cal_runtime=cal_runtime)
    assert out["metrics"]["omission_dedicated"]["status"] == "unavailable"
    assert out["metrics"]["substitution_dedicated"]["ap"] is None
    assert out["recognition"]["reference_phones"] == 0
    rows[1]["s_omit"] = .7
    with pytest.raises(ValueError, match="binary_collapsed must leave"):
        summarize_cell(runtime, rows, cal, [], objective="binary_collapsed", cell_id="B-Control",
                       cal_runtime=cal_runtime)


def test_class_focal_has_dedicated_three_way_scores_while_binary_collapsed_does_not():
    rows = [token(0, 0, .1, dedicated=.02), token(1, 2, .9, dedicated=.8)]
    cal = [token(i, 0, .1 + i*.01, dedicated=.02 + i*.01) for i in range(3)]
    runtime = [{"speaker_id": "S", "utterance_id": "u", "prompt_index": i,
                "s_error": row["s_error"], "s_omit": row["s_omit"], "s_sub": row["s_sub"]}
               for i, row in enumerate(rows)]
    cal_runtime = [{"speaker_id": "S", "utterance_id": "u", "prompt_index": i,
                    "s_error": row["s_error"], "s_omit": row["s_omit"], "s_sub": row["s_sub"]}
                   for i, row in enumerate(cal)]
    assert objective_supports_subtypes("class_focal") is True
    assert objective_supports_subtypes("operation_ce") is True
    assert objective_supports_subtypes("binary_collapsed") is False
    result = summarize_cell(runtime, rows, cal, [], objective="class_focal", cell_id="T1",
                            cal_runtime=cal_runtime)
    assert result["metrics"]["omission_dedicated"]["ap"] == 1.0
    rows[1]["s_omit"] = None
    with pytest.raises(ValueError, match="requires all three scores"):
        summarize_cell(runtime, rows, cal, [], objective="class_focal", cell_id="T1",
                       cal_runtime=cal_runtime)


def test_nonfinite_and_inconsistent_labels_fail_explicitly():
    bad = [token(0, 0, float("nan"))]
    with pytest.raises(ValueError, match="finite"):
        summarize_cell([], bad, [], [], objective="operation_ce", cell_id="bad")
    bad_label = [token(0, 2, .5)]
    bad_label[0]["binary_target"] = 0
    with pytest.raises(ValueError, match="disagrees"):
        summarize_cell([], bad_label, [], [], objective="operation_ce", cell_id="bad-label")


def test_confusions_do_not_count_positive_subtypes_again_as_false_positives():
    rows = [token(0, 0, .7), token(1, 1, .8), token(2, 2, .9)]
    binary = _confusion(rows, "s_error", None, .75)
    assert (binary["TP"], binary["FP"], binary["FN"], binary["TN"]) == (2, 0, 0, 1)
    assert binary["FP_sub"] == binary["FP_omission"] == binary["FP_other_subtype"] == 0
    omission = _confusion(rows, "s_error", 2, .75)
    assert (omission["TP"], omission["FP"], omission["FP_correct"], omission["FP_sub"]) == (1, 1, 0, 1)
    assert omission["FP_other_subtype"] == 1 and omission["FN"] == 0
    substitution = _confusion(rows, "s_error", 1, .75)
    assert (substitution["TP"], substitution["FP"], substitution["FP_correct"],
            substitution["FP_omission"]) == (1, 1, 0, 1)
    assert substitution["FP_other_subtype"] == 1 and substitution["FN"] == 0


def test_recognition_per_by_speaker_uses_edit_count_not_subedit_sum():
    summary = recognition_summary([
        {"speaker_id": "NJS", "ref_phones": ["AA", "B"], "hyp_phones": ["AA", "B"]},
        {"speaker_id": "RRBI", "ref_phones": ["AA", "B"], "hyp_phones": ["D"]},
    ])
    assert summary["edits"] == 2
    assert summary["by_speaker"]["RRBI"]["edits"] == 2
    assert summary["by_speaker"]["RRBI"]["PER"] == 1.0


def test_speaker_censuses_distinguish_cal_val_and_reject_sealed():
    validate_speaker_census([{"speaker_id": "NJS"}, {"speaker_id": "RRBI"}], VAL_SPEAKERS, "VAL")
    validate_speaker_census([{"speaker_id": "NCC"}, {"speaker_id": "THV"}], CAL_SPEAKERS, "CAL")
    with pytest.raises(ValueError, match="expected speakers"):
        validate_speaker_census([{"speaker_id": "NJS"}, {"speaker_id": "RRBI"}], CAL_SPEAKERS, "CAL")
    with pytest.raises(ValueError, match="sealed TEST"):
        validate_speaker_census([{"speaker_id": "ABA"}], VAL_SPEAKERS, "VAL")


def test_primary_scalar_uses_dedicated_omission_at_cal_15():
    result = {"metrics": {
        "binary": {"ap": .4},
        "omission_legacy": {"ap": .2, "cal_operating_points": {"0.1": {"counts": {"precision": .1}}}},
        "omission_dedicated": {"ap": .3, "cal_operating_points": {
            "0.1": {"counts": {"precision": .11, "recall_all_gold": .12, "false_alarms_per_1000_full_gold": 11}},
            "0.15": {"counts": {"precision": .45, "recall_all_gold": .55, "false_alarms_per_1000_full_gold": 15}}}},
        "substitution_legacy": {"ap": .25}, "substitution_dedicated": {"ap": .26}},
        "recognition": {"PER": .24},
        "mismatched_acoustic_control": {"metrics": {"binary": {"ap": .35}}}}
    values = primary_scalar_values(result)
    assert values["omission_precision_cal15"] == .45
    assert values["omission_recall_all_gold_cal15"] == .55
    assert values["omission_fa_per_1000_cal15"] == 15
    assert values["acoustic_gap_binary_ap"] == pytest.approx(.05)


def test_recognition_runtime_and_offline_rows_preserve_hypotheses_separately():
    runtime = [{"speaker_id": "NJS", "utterance_id": "u1", "hyp_phones": ["AA", "B"]}]
    offline = [{"speaker_id": "NJS", "utterance_id": "u1", "ref_phones": ["AA"],
                "hyp_phones": ["AA", "B"], "edit_counts": [1, 0, 0, 1]}]
    validate_recognition_pair(runtime, offline)
    runtime[0]["hyp_phones"] = ["D"]
    with pytest.raises(ValueError, match="hypothesis changed"):
        validate_recognition_pair(runtime, offline)


def test_registered_gates_use_frozen_levels_and_fixed_budget_only():
    scalar = {cell: {"omission_precision_cal15": .30,
                     "omission_recall_all_gold_cal15": .40,
                     "omission_fa_per_1000_cal15": 15,
                     "binary_ap": .313619,
                     "substitution_legacy_ap": .303974,
                     "PER": .246980,
                     "acoustic_gap_binary_ap": .010}
              for cell in ("C0", "T1", "T2", "T3")}
    gates = evaluate_registered_gates(scalar, "fixed_budget_3600")
    assert gates["T1"]["Level_A"]["status"] == "pass"
    assert gates["T1"]["Level_B"]["status"] == "pass"
    assert evaluate_registered_gates(scalar, "best")["T1"]["Level_A"]["status"] == "not_applicable"
    scalar["T3"]["PER"] = None
    assert evaluate_registered_gates(scalar, "fixed_budget_3600")["T3"]["Level_A"]["status"] == "unavailable"


def test_factorial_contrasts_use_registered_directions():
    data = {"C0": {"binary_ap": .1, "omission_dedicated_ap": .2, "omission_recall_all_gold": .3, "correct_fpr_all_gold": .05},
            "T1": {"binary_ap": .2, "omission_dedicated_ap": .3, "omission_recall_all_gold": .4, "correct_fpr_all_gold": .06},
            "T2": {"binary_ap": .4, "omission_dedicated_ap": .5, "omission_recall_all_gold": .6, "correct_fpr_all_gold": .07},
            "T3": {"binary_ap": .8, "omission_dedicated_ap": .9, "omission_recall_all_gold": .8, "correct_fpr_all_gold": .08}}
    result = factorial_contrasts(data)["binary_ap"]
    assert result["matched_contrasts"] == {"T1_minus_C0": pytest.approx(.1),
                                           "T2_minus_C0": pytest.approx(.3),
                                           "T3_minus_C0": pytest.approx(.7)}
    assert result["interaction_T3_minus_T2_minus_T1_plus_C0"] == pytest.approx(.3)
