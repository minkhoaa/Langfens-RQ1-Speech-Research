import pytest

from langfens_prep.trn04r.metrics import average_precision, summarize_cell, factorial_contrasts


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
    result = summarize_cell(runtime, rows, cal, [], control=False, cell_id="synthetic", cal_runtime=cal_runtime)
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


def test_binary_control_does_not_claim_dedicated_subtype_score():
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
    out = summarize_cell(runtime, rows, cal, [], control=True, cell_id="B-Control", cal_runtime=cal_runtime)
    assert out["metrics"]["omission_dedicated"]["status"] == "unavailable"
    assert out["metrics"]["substitution_dedicated"]["ap"] is None
    assert out["recognition"]["reference_phones"] == 0


def test_nonfinite_and_inconsistent_labels_fail_explicitly():
    bad = [token(0, 0, float("nan"))]
    with pytest.raises(ValueError, match="finite"):
        summarize_cell([], bad, [], [], control=False, cell_id="bad")
    bad_label = [token(0, 2, .5)]
    bad_label[0]["binary_target"] = 0
    with pytest.raises(ValueError, match="disagrees"):
        summarize_cell([], bad_label, [], [], control=False, cell_id="bad-label")


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
