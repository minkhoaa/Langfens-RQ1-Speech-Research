"""Integration contracts for the new A runtime and B model-free metrics API.

The APIs are intentionally imported inside tests so this lane can keep the
integration cases pending until the coordinator overlays both production files.
"""
import pytest
import torch

from langfens_prep.alignment import align_canonical_sequences


def _api():
    runtime = pytest.importorskip("langfens_prep.trn04.runtime")
    metrics = pytest.importorskip("langfens_prep.trn04r.metrics")
    model = pytest.importorskip("langfens_prep.trn04.model")
    return runtime, metrics, model


class _TinyFrozenEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, waveform, lengths):
        self.seen = waveform.detach().clone()
        pooled = waveform.unfold(1, 4, 4).mean(-1)
        features = torch.stack([pooled + i * 0.01 for i in range(8)], dim=-1)
        return features, torch.full_like(lengths, features.shape[1])


def _make_forward(runtime, model, arm="T1"):
    config = model.FactorialHeadConfig(feature_dim=8, acoustic_dim=4, width=8, heads=2,
                                       max_prompt_length=16, dropout=0.0)
    head = model.CompactFactorialOperationHead(config)
    encoder = _TinyFrozenEncoder()
    return runtime.make_head_forward(head, arm=arm, encoder=encoder), head, encoder


def test_real_head_runtime_uses_only_prompt_and_waveform_gold_variant_inert():
    runtime, _, model = _api()
    forward, head, _encoder = _make_forward(runtime, model)
    waveform = torch.linspace(-0.5, 0.5, 1600, dtype=torch.float32)
    from langfens_prep.phoneme_normalizer import PHONE_TO_CTC_INDEX
    phones = ["AA", "AE", "AH"]
    prompt_ids = torch.tensor([PHONE_TO_CTC_INDEX[p] for p in phones], dtype=torch.long)
    output_before = forward(waveform=waveform, prompt_phone_ids=prompt_ids)
    raw_before = output_before["operation_logits"].squeeze(0).clone()
    ctc_before = output_before["ctc_logits"].squeeze(0)
    runtime_before, _ = runtime.runtime_rows(
        logits=raw_before, ctc_logits=ctc_before, prompt_phone_ids=prompt_ids.tolist(),
        canonical_phones=phones, speaker_id="NJS", utterance_id="synthetic",
        arm="T1", endpoint="best", condition="matched", audio_source_id="source-a")

    # Gold only enters the separate offline join, after raw runtime output exists.
    gold_a = [{"canonical_phone": p, "operation_target": 0, "binary_target": 0} for p in phones]
    gold_b = [{"canonical_phone": "B", "operation_target": 2, "binary_target": 1},
              {"canonical_phone": "AE", "operation_target": 0, "binary_target": 0},
              {"canonical_phone": "AH", "operation_target": 0, "binary_target": 0}]
    joined_a = runtime.join_gold_rows(gold_positions=gold_a, prompt_phones=phones,
                                       runtime=runtime_before, align=align_canonical_sequences)
    joined_b = runtime.join_gold_rows(gold_positions=gold_b, prompt_phones=phones,
                                       runtime=runtime_before, align=align_canonical_sequences)
    output_after = forward(waveform=waveform, prompt_phone_ids=prompt_ids)
    assert torch.equal(raw_before, output_after["operation_logits"].squeeze(0))
    assert len(runtime_before) == len(phones)
    assert joined_a[0]["operation_target"] != joined_b[0]["operation_target"]
    assert all(not any(term in key.lower() for term in ("gold", "target", "textgrid", "operation_target", "binary_target"))
               for row in runtime_before for key in row)
    assert all(p.requires_grad is False for p in head.parameters())


def test_runtime_rows_exclude_even_null_annotation_field_names_and_encode_objectives():
    runtime, _, _ = _api()
    from langfens_prep.phoneme_normalizer import PHONE_TO_CTC_INDEX
    logits = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)
    ctc = torch.zeros((2, 40), dtype=torch.float32)
    for arm in ("C0", "T1", "T2", "T3"):
        rows, _ = runtime.runtime_rows(logits=logits, ctc_logits=ctc, prompt_phone_ids=[PHONE_TO_CTC_INDEX["AA"]],
                                       canonical_phones=["AA"], speaker_id="NJS", utterance_id="u",
                                       arm=arm, endpoint="best", condition="matched", audio_source_id="a")
        forbidden = ("gold", "target", "textgrid", "operation_target", "binary_target")
        assert not any(any(term in key.lower() for term in forbidden) for row in rows for key in row)
        assert rows[0]["scores"]["s_omit"] is not None
        assert rows[0]["scores"]["s_sub"] is not None
        assert rows[0]["objective"] == ("class_focal" if arm in {"T1", "T3"} else "operation_ce")
        assert [row["prompt_index"] for row in rows] == [0]
    with pytest.raises(ValueError, match="sealed TEST"):
        runtime.runtime_rows(logits=logits, ctc_logits=ctc, prompt_phone_ids=[PHONE_TO_CTC_INDEX["AA"]],
                             canonical_phones=["AA"], speaker_id="ABA", utterance_id="u",
                             arm="T1", endpoint="best", condition="matched", audio_source_id="a")


def test_join_retains_unmapped_gold_and_prompt_extras():
    runtime, _, _ = _api()
    from langfens_prep.phoneme_normalizer import PHONE_TO_CTC_INDEX
    phones = ["B", "AE"]
    rows, _ = runtime.runtime_rows(logits=torch.tensor([[1., 0., 0.], [0., 0., 1.]]),
                                   ctc_logits=torch.zeros((2, 40)), prompt_phone_ids=[PHONE_TO_CTC_INDEX[p] for p in phones],
                                   canonical_phones=["B", "AE"], speaker_id="NJS", utterance_id="join",
                                   arm="T1", endpoint="best", condition="matched", audio_source_id="a")
    gold_with_unmapped = [{"canonical_phone": "HH", "operation_target": 2, "binary_target": 1},
                          {"canonical_phone": "B", "operation_target": 0, "binary_target": 0},
                          {"canonical_phone": "AE", "operation_target": 0, "binary_target": 0}]
    unmapped = runtime.join_gold_rows(gold_positions=gold_with_unmapped, prompt_phones=["B", "AE"], runtime=rows,
                                      align=align_canonical_sequences)
    assert sum(r["mapping_status"] == "unmapped_gold_abstention" for r in unmapped) == 1
    assert len([r for r in unmapped if r.get("gold_canonical_index") is not None]) == len(gold_with_unmapped)
    gold_with_extra = [{"canonical_phone": "B", "operation_target": 0, "binary_target": 0}]
    extra = runtime.join_gold_rows(gold_positions=gold_with_extra, prompt_phones=["B", "AE"], runtime=rows,
                                   align=align_canonical_sequences)
    assert sum(r["mapping_status"] == "prompt_extra" for r in extra) == 1


def test_metrics_production_ap_and_dynamic_false_negative_accounting():
    runtime, metrics, _ = _api()
    assert metrics.objective_supports_subtypes("class_focal") is True
    assert metrics.objective_supports_subtypes("binary_collapsed") is False
    assert metrics.average_precision([1, 0, 1, 0], [0.9, 0.9, 0.1, 0.1]) == pytest.approx(0.5)
    # Two VAL gold positions: one omission is deliberately unanswered; one correct is answered.
    tokens = [
        {"speaker_id": "NJS", "utterance_id": "v", "gold_canonical_index": 0,
         "operation_target": 2, "binary_target": 1, "prompt_index": None,
         "s_error": None, "s_omit": None, "s_sub": None, "canonical_phone": "D"},
        {"speaker_id": "NJS", "utterance_id": "v", "gold_canonical_index": 1,
         "operation_target": 0, "binary_target": 0, "prompt_index": 0,
         "s_error": .1, "s_omit": .1, "s_sub": .1, "canonical_phone": "AE"},
    ]
    # Include a runtime record for every prompt index in offline rows.
    val_runtime = [{"speaker_id": "NJS", "utterance_id": "v", "prompt_index": 0,
                    "scores": {"s_error": .1, "s_omit": .1, "s_sub": .1}}]
    cal_tokens = [{"speaker_id": "NCC", "utterance_id": "c", "gold_canonical_index": 0,
                   "operation_target": 0, "binary_target": 0, "prompt_index": 0,
                   "s_error": .2, "s_omit": .2, "s_sub": .2, "canonical_phone": "AE"}]
    cal_runtime = [{"speaker_id": "NCC", "utterance_id": "c", "prompt_index": 0,
                    "scores": {"s_error": .2, "s_omit": .2, "s_sub": .2}}]
    summary = metrics.summarize_cell(val_runtime, tokens, cal_tokens, [], objective="class_focal",
                                     cell_id="tiny", cal_runtime=cal_runtime)
    omission = summary["metrics"]["omission_dedicated"]["cal_operating_points"]["0.05"]["counts"]
    assert omission["FN"] == 1
    assert summary["coverage"]["gold_positions"] == 2


def test_focal_dedicated_omission_uses_s_omit_for_cal_and_deployment_without_fallback():
    _, metrics, _ = _api()
    val_tokens, val_runtime = [], []
    for i in range(21):
        operation = 2 if i == 0 else 0
        speaker, utterance = "NJS", f"v{i}"
        score = {"s_error": .01 if i == 0 else .99,
                 "s_omit": .8 if i == 0 else .2,
                 "s_sub": .01}
        val_tokens.append({"speaker_id": speaker, "utterance_id": utterance,
                           "gold_canonical_index": 0, "operation_target": operation,
                           "binary_target": int(operation != 0), "prompt_index": 0,
                           "canonical_phone": "D", **score})
        val_runtime.append({"speaker_id": speaker, "utterance_id": utterance,
                            "prompt_index": 0, "scores": score})
    cal_tokens, cal_runtime = [], []
    for i in range(20):
        speaker, utterance = "NCC", f"c{i}"
        score = {"s_error": .99, "s_omit": .8 if i == 0 else .1, "s_sub": .01}
        cal_tokens.append({"speaker_id": speaker, "utterance_id": utterance,
                           "gold_canonical_index": 0, "operation_target": 0,
                           "binary_target": 0, "prompt_index": 0,
                           "canonical_phone": "D", **score})
        cal_runtime.append({"speaker_id": speaker, "utterance_id": utterance,
                            "prompt_index": 0, "scores": score})
    result = metrics.summarize_cell(val_runtime, val_tokens, cal_tokens, [],
                                    objective="class_focal", cell_id="focal-score-family",
                                    cal_runtime=cal_runtime)
    dedicated = result["metrics"]["omission_dedicated"]
    point = dedicated["cal_operating_points"]["0.05"]
    assert dedicated["score"] == "s_omit"
    assert .1 < point["threshold"] < .8
    assert point["counts"]["TP"] == 1
    assert point["counts"]["FN"] == 0
    assert point["counts"]["FP_correct"] == 20
    # If no dedicated score exists, focal metrics must reject rather than fall back to s_error.
    broken = [dict(row, s_omit=None) for row in val_tokens]
    broken_cal = [dict(row, s_omit=None) for row in cal_tokens]
    broken_runtime = [dict(row, scores={**row["scores"], "s_omit": None}) for row in val_runtime]
    broken_cal_runtime = [dict(row, scores={**row["scores"], "s_omit": None}) for row in cal_runtime]
    with pytest.raises(ValueError):
        metrics.summarize_cell(broken_runtime, broken, broken_cal, [], objective="class_focal",
                               cell_id="must-not-fallback", cal_runtime=broken_cal_runtime)


def test_production_cal_threshold_uses_cal_correct_scores_and_ceiling():
    _, metrics, _ = _api()
    cut = metrics.calibration_threshold([0.1, 0.2, 0.3, 0.4], 0.25)
    assert cut["status"] == "available"
    assert cut["cal_achieved_fpr"] <= 0.25
    assert cut["threshold"] > 0.3  # floor(0.25 * 4) permits one false alarm; ties are not split


def test_waveform_mismatch_reaches_encoder_and_feature_extractor_with_prompt_fixed():
    runtime, _, model = _api()
    normal_forward, _head1, normal_encoder = _make_forward(runtime, model, arm="T2")
    mismatch_forward, _head2, mismatch_encoder = _make_forward(runtime, model, arm="T2")
    waveform = torch.linspace(-0.5, 0.5, 1600, dtype=torch.float32)
    shifted_waveform = waveform.roll(17)
    prompt = torch.tensor([1, 2, 3], dtype=torch.long)
    normal = normal_forward(waveform=waveform, prompt_phone_ids=prompt)
    mismatch = mismatch_forward(waveform=shifted_waveform, prompt_phone_ids=prompt)
    assert torch.equal(normal_encoder.seen[0], waveform)
    assert torch.equal(mismatch_encoder.seen[0], shifted_waveform)
    assert not torch.equal(normal_encoder.seen, mismatch_encoder.seen)
    assert torch.equal(prompt, torch.tensor([1, 2, 3]))
    # The actual 4-D helper is waveform-derived and changes on this counterfactual.
    from langfens_prep.trn04.features import extract_frame_acoustic_features
    f1 = extract_frame_acoustic_features(waveform.unsqueeze(0), 5)
    f2 = extract_frame_acoustic_features(shifted_waveform.unsqueeze(0), 5)
    assert f1.shape == f2.shape == (1, 5, 4)
    assert not torch.equal(f1, f2)


def test_runtime_freeze_hash_precedes_synthetic_cal_and_val_gold_join(tmp_path):
    runtime, _, _ = _api()
    from langfens_prep.phoneme_normalizer import PHONE_TO_CTC_INDEX
    frozen = {}
    for partition, speaker in (("CAL", "NCC"), ("VAL", "NJS")):
        rows, _ = runtime.runtime_rows(logits=torch.tensor([[0., 1., 2.]]),
                                       ctc_logits=torch.zeros((2, 40)),
                                       prompt_phone_ids=[PHONE_TO_CTC_INDEX["AA"]], canonical_phones=["AA"],
                                       speaker_id=speaker, utterance_id="frozen", arm="T1", endpoint="best",
                                       condition="matched", audio_source_id=f"{partition}-audio")
        path = tmp_path / f"{partition}.jsonl"
        digest = runtime.write_jsonl(path, rows)
        assert path.is_file() and digest
        frozen[partition] = (rows, digest)
    # Gold fixtures are intentionally created/joined only after both runtime stores are hashed.
    gold = [{"canonical_phone": "AA", "operation_target": 2, "binary_target": 1}]
    for partition in ("CAL", "VAL"):
        rows, digest = frozen[partition]
        assert digest == runtime.sha256_file(tmp_path / f"{partition}.jsonl")
        joined = runtime.join_gold_rows(gold_positions=gold, prompt_phones=["AA"], runtime=rows,
                                        align=align_canonical_sequences)
        assert len(joined) == 1


def test_partition_metadata_is_speaker_disjoint_and_test_remains_denylisted():
    # Contract-frozen speaker groups, tested without opening manifests/data payloads.
    groups = [set("BWC ERMS HKK HQTV SKA SVBI TNI TXHC YBAA YDCK YKWK ZHAA".split()),
              {"MBMPS", "TLV"}, {"NCC", "THV"}, {"NJS", "RRBI"}]
    SEALED_SPEAKERS = {"ABA", "ASI", "EBVS", "HJK", "LXC", "PNV"}
    assert all(not (a & b) for i, a in enumerate(groups) for b in groups[i + 1:])
    assert all(not (g & set(SEALED_SPEAKERS)) for g in groups)
    assert set(SEALED_SPEAKERS) == {"ABA", "ASI", "EBVS", "HJK", "LXC", "PNV"}
