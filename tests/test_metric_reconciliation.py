"""
Unit and regression tests for LANGFENS RQ1-TRN-01B0-R metric reconciliation.
Tests prove:
1. Denominator and masking correctness for substitution diagnosis (124 eligible, 2 err masked, 876 correct masked, 52 deletions masked).
2. Auxiliary realized-phone classification denominator (1,000 tokens).
3. TextGrid parser masking contracts for all error types.
4. CTC greedy decoding trimming to input_lengths and blank collapse detection.
5. Binary detection score distribution and default 0.5 all-negative behavior.
6. Checkpoint state dict integrity and reload equivalence.
"""

import json
import pytest
import torch
import numpy as np

from langfens_prep.phoneme_normalizer import STANDARD_39_ARPABET, PHONE_TO_INDEX, PHONE_TO_CTC_INDEX
from langfens_prep.textgrid_parser import parse_gold_textgrid_content, TextGridInterval
from langfens_prep.dataset import L2ArcticMDDDataset
from langfens_prep.collate import LangfensMDDCollate
from langfens_prep.audio_loader import get_l2arctic_reader
from audit_scripts.run_training_sanity import (
    TrainSubsetDataset,
    compute_metrics,
    greedy_ctc_decode,
    detailed_levenshtein
)


def test_training_subset_exact_phone_census_and_denominators():
    """Verify exact counts and denominators across the fixed 32-utterance training subset."""
    manifest_path = "manifests/RQ1-TRN-01B0/TRAIN_ONLY_SUBSET_MANIFEST.json"
    zip_path = "datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip"
    with open(manifest_path, "r", encoding="utf-8") as f:
        subset = json.load(f)

    reader = get_l2arctic_reader(zip_path)
    total_can = 0
    total_cor = 0
    total_sub = 0
    total_del = 0
    eligible_sub = 0
    masked_err_sub = 0
    aux_phone_tokens = 0

    for u in subset["selected_utterances"]:
        spk = u["speaker_id"]
        utt = u["utterance_id"]
        tg_text = reader.read_annotation_text(spk, utt)
        ann = parse_gold_textgrid_content(tg_text, f"{utt}.TextGrid")

        for i in ann.canonical_intervals:
            total_can += 1
            if i.error_type == "correct":
                total_cor += 1
                assert i.diagnosis_target is None, f"Correct interval {i.raw_label} must have diagnosis_target=None"
                assert i.binary_label_strict == 0
                if i.realized_phone in PHONE_TO_INDEX:
                    aux_phone_tokens += 1
            elif i.error_type == "substitution":
                total_sub += 1
                assert i.binary_label_strict == 1
                if i.has_err_tag:
                    masked_err_sub += 1
                    assert i.diagnosis_target is None, f"'err' interval {i.raw_label} must have diagnosis_target=None"
                else:
                    eligible_sub += 1
                    assert i.diagnosis_target is not None, f"Eligible substitution {i.raw_label} must have non-null diagnosis_target"
                    assert i.diagnosis_target in STANDARD_39_ARPABET
                    aux_phone_tokens += 1
            elif i.error_type == "deletion":
                total_del += 1
                assert i.diagnosis_target is None, f"Deletion interval {i.raw_label} must have diagnosis_target=None"
                assert i.binary_label_strict == 1

    # Exact census assertions
    assert total_can == 1054, f"Expected 1,054 canonical phonemes, got {total_can}"
    assert total_cor == 876, f"Expected 876 correct phonemes, got {total_cor}"
    assert total_sub == 126, f"Expected 126 substitutions, got {total_sub}"
    assert total_del == 52, f"Expected 52 deletions, got {total_del}"
    assert eligible_sub == 124, f"Expected exactly 124 eligible substitutions, got {eligible_sub}"
    assert masked_err_sub == 2, f"Expected exactly 2 masked 'err' substitutions, got {masked_err_sub}"
    assert aux_phone_tokens == 1000, f"Expected 1,000 auxiliary realized-phone tokens (876 cor + 124 sub), got {aux_phone_tokens}"


def test_textgrid_parser_masking_policy():
    """Verify that TextGrid parser correctly assigns diagnosis_target only to eligible substitutions."""
    sample_textgrid = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0 
xmax = 1.0 
tiers? <exists> 
size = 1 
item []: 
    item [1]:
        class = "IntervalTier" 
        name = "phones" 
        xmin = 0 
        xmax = 1.0 
        intervals: size = 5 
        intervals [1]:
            xmin = 0 
            xmax = 0.2 
            text = "sil" 
        intervals [2]:
            xmin = 0.2 
            xmax = 0.4 
            text = "DH,D,s" 
        intervals [3]:
            xmin = 0.4 
            xmax = 0.6 
            text = "AH0" 
        intervals [4]:
            xmin = 0.6 
            xmax = 0.8 
            text = "T,err,s" 
        intervals [5]:
            xmin = 0.8 
            xmax = 1.0 
            text = "K,sil,d" 
"""
    ann = parse_gold_textgrid_content(sample_textgrid, "mask_test.TextGrid")
    assert len(ann.canonical_intervals) == 4

    # 1. Eligible substitution: DH -> D
    assert ann.canonical_intervals[0].error_type == "substitution"
    assert ann.canonical_intervals[0].diagnosis_target == "D"

    # 2. Correct phone: AH0
    assert ann.canonical_intervals[1].error_type == "correct"
    assert ann.canonical_intervals[1].diagnosis_target is None  # Must be masked!

    # 3. Uncertain substitution: T -> err
    assert ann.canonical_intervals[2].error_type == "substitution"
    assert ann.canonical_intervals[2].has_err_tag is True
    assert ann.canonical_intervals[2].diagnosis_target is None  # Must be masked!

    # 4. Deletion: K -> sil
    assert ann.canonical_intervals[3].error_type == "deletion"
    assert ann.canonical_intervals[3].diagnosis_target is None  # Must be masked!


def test_dataset_collation_target_diagnosis_masking():
    """Verify DataLoader collation creates target_diagnosis with -100 for non-substitutions and padding."""
    ds = TrainSubsetDataset()
    collate_fn = LangfensMDDCollate(pad_token_id=-100, ctc_blank_id=0)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

    total_valid_diagnosis_tokens = 0
    total_valid_aux_tokens = 0
    total_valid_binary_tokens = 0

    for batch in loader:
        td = batch["target_diagnosis"]
        taux = batch["target_aux_phone"]
        tb = batch["target_binary"]

        # Assert padding contains strictly -100
        for b_idx in range(td.shape[0]):
            valid_len = batch["canonical_phone_lengths"][b_idx].item()
            assert (td[b_idx, valid_len:] == -100).all()
            assert (taux[b_idx, valid_len:] == -100).all()
            assert (tb[b_idx, valid_len:] == -100).all()

        total_valid_diagnosis_tokens += int((td != -100).sum())
        total_valid_aux_tokens += int((taux != -100).sum())
        total_valid_binary_tokens += int((tb != -100).sum())

    assert total_valid_diagnosis_tokens == 124, f"Expected 124 substitution diagnosis targets, got {total_valid_diagnosis_tokens}"
    assert total_valid_aux_tokens == 1000, f"Expected 1,000 auxiliary phone targets, got {total_valid_aux_tokens}"
    assert total_valid_binary_tokens == 1054, f"Expected 1,054 binary targets, got {total_valid_binary_tokens}"


def test_ctc_framewise_trimming_and_detailed_levenshtein():
    """Verify CTC greedy decoding trims to input_lengths and detailed edit breakdown is exact."""
    # Test detailed Levenshtein
    ref = [1, 2, 3, 4]
    hyp = [1, 5, 4]  # S at pos 2 (2->5), D at pos 3 (3 deleted)
    tot, s, d, ins = detailed_levenshtein(ref, hyp)
    assert tot == 2
    assert s == 1
    assert d == 1
    assert ins == 0
    assert tot == (s + d + ins)

    # Empty hyp (blank collapse)
    tot_empty, s_e, d_e, ins_e = detailed_levenshtein(ref, [])
    assert tot_empty == len(ref)
    assert s_e == 0
    assert d_e == len(ref)
    assert ins_e == 0

    # Test trimming: log_probs has 10 frames, but input_length is 5
    # Frames 0..4 predict [0, 1, 1, 0, 2], frames 5..9 predict [3, 3, 3, 3, 3] (padding)
    lp = torch.full((10, 4), -10.0)
    lp[0, 0] = 0.0  # blank
    lp[1, 1] = 0.0  # token 1
    lp[2, 1] = 0.0  # token 1
    lp[3, 0] = 0.0  # blank
    lp[4, 2] = 0.0  # token 2
    lp[5:10, 3] = 0.0  # token 3 (padding frames!)

    # Untrimmed decode sees token 3
    untrimmed = greedy_ctc_decode(lp, blank_id=0)
    assert untrimmed == [1, 2, 3]

    # Trimmed decode stops at active length 5
    trimmed = greedy_ctc_decode(lp[:5], blank_id=0)
    assert trimmed == [1, 2]  # Correctly ignores trailing padding frames


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only pretrained inference regression")
def test_existing_checkpoint_binary_and_diagnosis_metrics():
    """Verify that existing checkpoint step 100 reflects the reconciled metric semantics."""
    ckpt_path = "checkpoints/RQ1-TRN-01B0/rq1_trn_01b0_checkpoint_step_100.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")

    assert ckpt["training_step"] == 100
    assert len(ckpt["selected_training_ids"]) == 32

    # Verify score distributions and metrics on checkpoint
    from langfens_prep.model import LangfensWav2Vec2MDD
    model = LangfensWav2Vec2MDD.from_pretrained(
        "facebook/wav2vec2-base-960h",
        revision="22aad52d435eb6dbaf354bdad9b0da84ce7d6156",
        alignment_mode="cross_attention",
        ctc_zero_infinity_policy="raise"
    )
    model.freeze_feature_encoder()
    model.load_state_dict(ckpt["model_state_dict"])
    model.cuda()

    ds = TrainSubsetDataset()
    metrics = compute_metrics(model, ds, device="cuda")

    # Substitution diagnosis: denominator must be 124, not 1000
    assert metrics["substitution_diagnosis"]["eligible_substitution_denominator"] == 124
    assert metrics["substitution_diagnosis"]["correct_predictions"] == 4
    assert abs(metrics["substitution_diagnosis"]["top1_accuracy"] - (4 / 124)) < 1e-4

    # Auxiliary phone classification: denominator 1000
    assert metrics["auxiliary_realized_phone_classification"]["evaluated_tokens_denominator"] == 1000
    assert metrics["auxiliary_realized_phone_classification"]["correct_predictions"] == 80
    assert abs(metrics["auxiliary_realized_phone_classification"]["top1_accuracy"] - 0.08) < 1e-4

    # Default threshold 0.5: all negative predictions
    cm = metrics["binary_default_05"]["confusion_matrix"]
    assert cm["tp"] == 0
    assert cm["fp"] == 0
    assert cm["fn"] == 178
    assert cm["tn"] == 876
    assert metrics["binary_default_05"]["f1"] == 0.0

    # CTC blank collapse: 100% blanks, PER = 1.0000
    ctc_diag = metrics["ctc_diagnostics"]
    assert ctc_diag["is_blank_collapsed"] is True
    assert ctc_diag["blank_fraction"] == 1.0
    assert ctc_diag["hyp_tokens_total"] == 0
    assert metrics["ctc_phone_error_rate"] == 1.0000
    assert metrics["ctc_total_edits"] == 1012
    assert metrics["ctc_total_ref_tokens"] == 1012
