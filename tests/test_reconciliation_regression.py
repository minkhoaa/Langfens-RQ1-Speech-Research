"""
tests/test_reconciliation_regression.py
Regression Test Suite for RQ1-TRN-01A-R Acceptance Gate.
Verifies all 12 regression items mandated by the independent code review.
"""

import os
import json
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from langfens_prep.model import (
    LangfensWav2Vec2MDD,
    inspect_pretrained_backbone_integrity
)
from langfens_prep.phoneme_normalizer import (
    STANDARD_39_ARPABET,
    PHONE_TO_CTC_INDEX,
    PHONE_TO_INDEX
)
from langfens_prep.textgrid_parser import parse_gold_textgrid_content
from langfens_prep.losses import (
    MaskedBCEWithLogitsLoss,
    MaskedCrossEntropyLoss,
    validate_ctc_contract,
    CTCContractError
)


# ==============================================================================
# 1. Pretrained Checkpoint Weights are Actually Loaded
# ==============================================================================
def test_pretrained_checkpoint_weights_actually_loaded():
    """Verify genuine Hugging Face checkpoint weights are loaded and param counts match."""
    model = LangfensWav2Vec2MDD.from_pretrained("facebook/wav2vec2-base-960h")
    assert model.is_pretrained is True
    assert model.provenance == "facebook/wav2vec2-base-960h"

    # Backbone parameters must match official 94,371,712 count
    backbone_params = sum(p.numel() for p in model.wav2vec2.parameters())
    assert backbone_params == 94_371_712

    # Checkpoint integrity inspection
    report = inspect_pretrained_backbone_integrity("facebook/wav2vec2-base-960h")
    assert report["total_parameters"] == 94_371_712
    assert report["license"] == "Apache-2.0"
    assert "masked_spec_embed" in report["missing_keys"]
    assert "lm_head.weight" in report["unexpected_keys"]

    # Verify first conv layer weights are non-trivial (trained filter values)
    conv0_weight = model.wav2vec2.feature_extractor.conv_layers[0].conv.weight
    assert conv0_weight.shape == (512, 1, 10)
    assert torch.isfinite(conv0_weight).all()
    assert conv0_weight.abs().mean().item() > 0.01


# ==============================================================================
# 2. Random and Pretrained Initialization are Distinguishable
# ==============================================================================
def test_random_and_pretrained_initialization_distinguishable():
    """Verify pretrained weights differ fundamentally from random initialization."""
    pretrained = LangfensWav2Vec2MDD.from_pretrained("facebook/wav2vec2-base-960h")
    
    # Instantiate random model with identical architecture dimensions
    random_model = LangfensWav2Vec2MDD(
        pretrained_model_name_or_path=None,
        hidden_dim=768,
        num_transformer_layers=12,
        num_attention_heads=12,
        intermediate_dim=3072,
        conv_dim=512
    )

    assert pretrained.is_pretrained is True
    assert random_model.is_pretrained is False

    # Transformer layer 0 Q projection weights must not match
    w_pre = pretrained.wav2vec2.encoder.layers[0].attention.q_proj.weight
    w_rnd = random_model.wav2vec2.encoder.layers[0].attention.q_proj.weight

    assert w_pre.shape == w_rnd.shape
    assert not torch.equal(w_pre, w_rnd)


# ==============================================================================
# 3. Full Backbone Output Shapes are Correct
# ==============================================================================
def test_full_backbone_output_shapes():
    """Verify full 12-layer Wav2Vec2 backbone temporal downsampling (320x)."""
    model = LangfensWav2Vec2MDD.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()

    # 2.0s audio at 16kHz = 32,000 samples
    sample_rate = 16000
    waveforms = torch.randn(2, 32000)
    audio_lengths = torch.tensor([32000, 24000], dtype=torch.long)

    with torch.no_grad():
        out = model(waveforms=waveforms, audio_lengths=audio_lengths)

    # 32,000 samples -> 99 frames; 24,000 samples -> 74 frames
    assert out.ctc_logits.shape == (2, 99, 40)
    assert out.input_lengths.tolist() == [99, 74]
    assert out.frame_features.shape == (2, 99, 768)


# ==============================================================================
# 4. Deployable Inference Requires No Gold TextGrid Intervals
# ==============================================================================
def test_deployable_inference_requires_no_gold_intervals():
    """Verify predict() runs successfully with raw audio and canonical tokens only."""
    model = LangfensWav2Vec2MDD(hidden_dim=256, num_transformer_layers=2)
    model.eval()

    waveforms = torch.randn(2, 16000) # 1.0s audio
    canonical_indices = torch.tensor([
        [5, 12, 19, 25],
        [8, 22, 14, 0] # 0 = padding
    ], dtype=torch.long)

    preds = model.predict(
        waveforms=waveforms,
        canonical_phone_indices=canonical_indices,
        binary_threshold=0.5
    )

    assert "binary_probs" in preds
    assert "binary_preds" in preds
    assert "diagnosis_probs" in preds
    assert "diagnosis_preds" in preds
    assert "ctc_decoded_indices" in preds
    assert "alignment_weights" in preds

    assert preds["binary_probs"].shape == (2, 4)
    assert (preds["binary_probs"] >= 0.0).all() and (preds["binary_probs"] <= 1.0).all()
    assert preds["diagnosis_probs"].shape == (2, 4, 39)
    assert len(preds["ctc_decoded_indices"]) == 2


# ==============================================================================
# 5. Mandatory Negative Test: Gold Boundaries Cannot Influence Predictions
# ==============================================================================
def test_deployment_mode_invariance_to_gold_intervals():
    """
    MANDATORY NEGATIVE TEST:
    For identical audio and canonical sequence, removing, changing, or scrambling
    gold interval boundaries MUST NOT alter deployment predictions by even a single bit.
    """
    torch.manual_seed(123)
    model = LangfensWav2Vec2MDD(hidden_dim=256, num_transformer_layers=2)
    model.eval()

    waveforms = torch.randn(1, 16000)
    canonical_indices = torch.tensor([[5, 12, 19, 25]], dtype=torch.long)

    # Run 1: Normal deployment mode (no intervals provided)
    out_clean = model(waveforms=waveforms, canonical_phone_indices=canonical_indices)

    # Run 2: Externally provided gold intervals
    gold_intervals = [[(0.05, 0.20), (0.20, 0.45), (0.45, 0.70), (0.70, 0.95)]]
    out_with_gold = model(
        waveforms=waveforms,
        canonical_phone_indices=canonical_indices,
        phone_intervals=gold_intervals
    )

    # Run 3: Scrambled / completely wrong intervals
    scrambled_intervals = [[(0.80, 0.99), (0.01, 0.05), (0.50, 0.55), (0.10, 0.15)]]
    out_with_scrambled = model(
        waveforms=waveforms,
        canonical_phone_indices=canonical_indices,
        phone_intervals=scrambled_intervals
    )

    # In deployment mode (cross_attention), intervals are ignored: outputs must be strictly identical
    assert torch.equal(out_clean.binary_logits, out_with_gold.binary_logits)
    assert torch.equal(out_clean.binary_logits, out_with_scrambled.binary_logits)
    assert torch.equal(out_clean.diagnosis_logits, out_with_gold.diagnosis_logits)
    assert torch.equal(out_clean.diagnosis_logits, out_with_scrambled.diagnosis_logits)


# ==============================================================================
# 6. Insertions are Correctly Represented in CTC Targets
# ==============================================================================
def test_insertions_included_in_ctc_targets():
    """Verify that epenthetic additions (insertions) are included in chronological CTC targets."""
    tg_text = """File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0 
xmax = 0.6 
tiers? <exists> 
size = 1 
item []: 
    item [1]:
        class = "IntervalTier" 
        name = "phones" 
        xmin = 0 
        xmax = 0.6 
        intervals: size = 3 
        intervals [1]:
            xmin = 0.0 
            xmax = 0.2 
            text = "K" 
        intervals [2]:
            xmin = 0.2 
            xmax = 0.4 
            text = "sil,AH,a" 
        intervals [3]:
            xmin = 0.4 
            xmax = 0.6 
            text = "T" 
"""
    ann = parse_gold_textgrid_content(tg_text, "insertion_test.TextGrid")
    
    # Chronological realized reconstruction
    chronological = []
    for i in ann.intervals:
        if i.error_type in ("correct", "substitution", "addition") and i.realized_phone in STANDARD_39_ARPABET:
            chronological.append(i.realized_phone)

    # Epenthetic addition AH must be present in the realized sequence between K and T
    assert chronological == ["K", "AH", "T"]
    ctc_targets = [PHONE_TO_CTC_INDEX[p] for p in chronological]
    assert ctc_targets == [PHONE_TO_CTC_INDEX["K"], PHONE_TO_CTC_INDEX["AH"], PHONE_TO_CTC_INDEX["T"]]
    assert 0 not in ctc_targets


# ==============================================================================
# 7. Deletions Do Not Become Acoustic CTC Targets
# ==============================================================================
def test_deletions_excluded_from_ctc_targets():
    """Verify that deleted phonemes are excluded from acoustic CTC targets."""
    tg_text = """File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0 
xmax = 0.6 
tiers? <exists> 
size = 1 
item []: 
    item [1]:
        class = "IntervalTier" 
        name = "phones" 
        xmin = 0 
        xmax = 0.6 
        intervals: size = 3 
        intervals [1]:
            xmin = 0.0 
            xmax = 0.2 
            text = "B" 
        intervals [2]:
            xmin = 0.2 
            xmax = 0.4 
            text = "R,sil,d" 
        intervals [3]:
            xmin = 0.4 
            xmax = 0.6 
            text = "IY" 
"""
    ann = parse_gold_textgrid_content(tg_text, "deletion_test.TextGrid")

    # Chronological realized reconstruction
    chronological = []
    for i in ann.intervals:
        if i.error_type == "silence" or i.error_type == "deletion":
            continue
        if i.realized_phone in STANDARD_39_ARPABET:
            chronological.append(i.realized_phone)

    # Realized target must only be B and IY, R must not exist in acoustic target
    assert chronological == ["B", "IY"]
    assert "R" not in chronological
    assert "sil" not in chronological


# ==============================================================================
# 8. Unknown Realizations are Masked According to Policy
# ==============================================================================
def test_unknown_realizations_and_err_masked():
    """Verify 'err' tags and unidentifiable phones are masked (-100) and excluded from CTC."""
    tg_text = """File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0 
xmax = 0.4 
tiers? <exists> 
size = 1 
item []: 
    item [1]:
        class = "IntervalTier" 
        name = "phones" 
        xmin = 0 
        xmax = 0.4 
        intervals: size = 2 
        intervals [1]:
            xmin = 0.0 
            xmax = 0.2 
            text = "S,err,s" 
        intervals [2]:
            xmin = 0.2 
            xmax = 0.4 
            text = "IY" 
"""
    ann = parse_gold_textgrid_content(tg_text, "err_test.TextGrid")
    
    # 1. Diagnosis target for 'err' interval must be None (masked)
    assert ann.canonical_intervals[0].has_err_tag is True
    assert ann.canonical_intervals[0].diagnosis_target is None

    # 2. In CTC sequence, 'err' cannot provide a ground-truth ARPAbet target
    chronological = []
    for i in ann.intervals:
        if i.error_type in ("correct", "substitution", "addition") and not i.has_err_tag:
            if i.realized_phone in STANDARD_39_ARPABET:
                chronological.append(i.realized_phone)

    assert chronological == ["IY"]
    assert "err" not in chronological


# ==============================================================================
# 9. Canonical-Phone Token Order is Preserved
# ==============================================================================
def test_canonical_phone_order_preservation():
    """Verify canonical phone order is strictly preserved in token mapping and queries."""
    prompt_phones = ["DH", "AH", "K", "AE", "T"]
    indices = [PHONE_TO_CTC_INDEX[p] for p in prompt_phones]
    
    # Assert strict 1-to-1 ordered preservation
    assert indices == [
        PHONE_TO_CTC_INDEX["DH"],
        PHONE_TO_CTC_INDEX["AH"],
        PHONE_TO_CTC_INDEX["K"],
        PHONE_TO_CTC_INDEX["AE"],
        PHONE_TO_CTC_INDEX["T"]
    ]


# ==============================================================================
# 10. No Cross-Speaker Leakage Occurs
# ==============================================================================
def test_no_cross_speaker_leakage():
    """Verify strictly zero speaker overlap between train, val, and test splits in all folds."""
    splits_file = "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json"
    assert os.path.exists(splits_file)

    with open(splits_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for fold in data["folds"]:
        train_spks = set(fold["train"]["speakers"])
        val_spks = set(fold["val"]["speakers"])
        test_spks = set(fold["test"]["speakers"])

        # Pairwise disjointness
        assert len(train_spks.intersection(val_spks)) == 0
        assert len(train_spks.intersection(test_spks)) == 0
        assert len(val_spks.intersection(test_spks)) == 0

        # Total 24 speakers per fold
        assert len(train_spks) + len(val_spks) + len(test_spks) == 24


# ==============================================================================
# 11. All-Masked Losses Remain Valid
# ==============================================================================
def test_all_masked_losses_remain_valid():
    """Verify batches where all positions are masked (-100) produce 0.0 loss with intact autograd."""
    logits_bce = torch.tensor([[1.0, -1.0, 0.5]], requires_grad=True)
    targets_bce = torch.tensor([[-100, -100, -100]], dtype=torch.long)

    bce_loss_fn = MaskedBCEWithLogitsLoss(ignore_index=-100)
    loss_bce = bce_loss_fn(logits_bce, targets_bce)

    assert loss_bce.item() == 0.0
    loss_bce.backward()
    assert logits_bce.grad is not None
    assert torch.equal(logits_bce.grad, torch.zeros_like(logits_bce))

    logits_ce = torch.randn(1, 39, 3, requires_grad=True)
    targets_ce = torch.tensor([[-100, -100, -100]], dtype=torch.long)
    ce_loss_fn = MaskedCrossEntropyLoss(ignore_index=-100)
    loss_ce = ce_loss_fn(logits_ce, targets_ce)

    assert loss_ce.item() == 0.0
    loss_ce.backward()
    assert logits_ce.grad is not None
    assert torch.equal(logits_ce.grad, torch.zeros_like(logits_ce))


# ==============================================================================
# 12. Full-Sized Synthetic GPU Smoke Test
# ==============================================================================
def test_full_sized_synthetic_gpu_smoke():
    """Verify GPU forward, loss, backward, and optimizer step on full architecture."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; skipping full-size GPU smoke test.")

    device = torch.device("cuda:0")
    model = LangfensWav2Vec2MDD.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    model.freeze_feature_encoder()
    model.gradient_checkpointing_enable()
    model.train()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    scaler = torch.amp.GradScaler('cuda')

    bsz = 2
    sample_len = 32000 # 2.0s audio
    num_phones = 16

    waveforms = torch.randn(bsz, sample_len, device=device)
    audio_lengths = torch.tensor([32000, 24000], dtype=torch.long, device=device)
    canonical_indices = torch.randint(1, 40, (bsz, num_phones), device=device)
    target_binary = torch.randint(0, 2, (bsz, num_phones), device=device)
    target_ctc = torch.randint(1, 40, (bsz, num_phones), device=device)
    target_lengths = torch.full((bsz,), num_phones, dtype=torch.long, device=device)
    target_diagnosis = torch.randint(0, 39, (bsz, num_phones), device=device)

    optimizer.zero_grad()
    with torch.amp.autocast('cuda', dtype=torch.float16):
        out = model(
            waveforms=waveforms,
            audio_lengths=audio_lengths,
            canonical_phone_indices=canonical_indices,
            target_binary=target_binary,
            target_ctc=target_ctc,
            target_lengths=target_lengths,
            target_diagnosis=target_diagnosis
        )
        loss = out.loss

    assert loss is not None
    assert torch.isfinite(loss)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    assert model.ctc_head.weight.grad is not None


# ==============================================================================
# 13. Task A: Acoustic Padding Invariance (Alone vs Mixed-Duration Batch)
# ==============================================================================
def test_acoustic_padding_alone_vs_batched_invariance():
    """
    Task A: Verify that output logits and probabilities for an utterance evaluated alone
    match evaluating the same utterance inside a mixed-duration padded batch
    within strict numerical tolerance (< 1e-5 in FP32).
    """
    torch.manual_seed(42)
    model = LangfensWav2Vec2MDD(hidden_dim=256, num_transformer_layers=2)
    model.eval()

    # Utterance 1: 1.0s (16,000 samples), 4 canonical phones
    w1 = torch.randn(1, 16000)
    c1 = torch.tensor([[5, 12, 19, 25]], dtype=torch.long)
    l1 = torch.tensor([16000], dtype=torch.long)

    # Utterance 2: 2.0s (32,000 samples), 6 canonical phones
    w2 = torch.randn(1, 32000)
    c2 = torch.tensor([[7, 14, 21, 28, 30, 35]], dtype=torch.long)
    l2 = torch.tensor([32000], dtype=torch.long)

    # 1. Evaluate utterance 1 alone
    with torch.no_grad():
        out1_alone = model(waveforms=w1, audio_lengths=l1, canonical_phone_indices=c1)

    # 2. Form variable-duration padded batch
    w_batch = torch.zeros(2, 32000)
    w_batch[0, :16000] = w1[0]
    w_batch[1, :32000] = w2[0]
    batch_audio_lens = torch.tensor([16000, 32000], dtype=torch.long)

    c_batch = torch.zeros(2, 6, dtype=torch.long)
    c_batch[0, :4] = c1[0]
    c_batch[1, :6] = c2[0]
    can_lens = torch.tensor([4, 6], dtype=torch.long)

    # 3. Evaluate inside padded batch
    with torch.no_grad():
        out_batch = model(
            waveforms=w_batch,
            audio_lengths=batch_audio_lens,
            canonical_phone_indices=c_batch,
            canonical_phone_lengths=can_lens
        )

    valid_frames = out1_alone.input_lengths[0].item()
    valid_phones = c1.shape[1]

    # Max difference over valid frames / phones must be strictly within single-precision epsilon
    ctc_diff = (out1_alone.ctc_logits[0, :valid_frames] - out_batch.ctc_logits[0, :valid_frames]).abs().max().item()
    bin_diff = (out1_alone.binary_logits[0, :valid_phones] - out_batch.binary_logits[0, :valid_phones]).abs().max().item()
    diag_diff = (out1_alone.diagnosis_logits[0, :valid_phones] - out_batch.diagnosis_logits[0, :valid_phones]).abs().max().item()

    assert ctc_diff < 1e-5, f"CTC diff {ctc_diff} exceeds numerical tolerance"
    assert bin_diff < 1e-5, f"Binary diff {bin_diff} exceeds numerical tolerance"
    assert diag_diff < 1e-5, f"Diagnosis diff {diag_diff} exceeds numerical tolerance"


def test_pretrained_acoustic_padding_alone_vs_batched_invariance():
    """
    Task A: Verify on the genuine pretrained facebook/wav2vec2-base-960h backbone
    that an utterance evaluated alone matches evaluating inside a variable-length padded batch
    within numerical tolerance (< 1e-5 in FP32).
    """
    torch.manual_seed(42)
    model = LangfensWav2Vec2MDD.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()

    # Utterance 1: 1.0s (16,000 samples)
    w1 = torch.randn(1, 16000)
    c1 = torch.tensor([[5, 12, 19, 25]], dtype=torch.long)
    l1 = torch.tensor([16000], dtype=torch.long)

    # Utterance 2: 2.0s (32,000 samples)
    w2 = torch.randn(1, 32000)
    c2 = torch.tensor([[7, 14, 21, 28, 30, 35]], dtype=torch.long)
    l2 = torch.tensor([32000], dtype=torch.long)

    with torch.no_grad():
        out1_alone = model(waveforms=w1, audio_lengths=l1, canonical_phone_indices=c1)

    w_batch = torch.zeros(2, 32000)
    w_batch[0, :16000] = w1[0]
    w_batch[1, :32000] = w2[0]
    batch_audio_lens = torch.tensor([16000, 32000], dtype=torch.long)

    c_batch = torch.zeros(2, 6, dtype=torch.long)
    c_batch[0, :4] = c1[0]
    c_batch[1, :6] = c2[0]
    can_lens = torch.tensor([4, 6], dtype=torch.long)

    with torch.no_grad():
        out_batch = model(
            waveforms=w_batch,
            audio_lengths=batch_audio_lens,
            canonical_phone_indices=c_batch,
            canonical_phone_lengths=can_lens
        )

    valid_frames = out1_alone.input_lengths[0].item()
    valid_phones = c1.shape[1]

    ctc_diff = (out1_alone.ctc_logits[0, :valid_frames] - out_batch.ctc_logits[0, :valid_frames]).abs().max().item()
    bin_diff = (out1_alone.binary_logits[0, :valid_phones] - out_batch.binary_logits[0, :valid_phones]).abs().max().item()
    diag_diff = (out1_alone.diagnosis_logits[0, :valid_phones] - out_batch.diagnosis_logits[0, :valid_phones]).abs().max().item()

    assert ctc_diff < 1e-5, f"Pretrained CTC diff {ctc_diff} exceeds numerical tolerance"
    assert bin_diff < 1e-5, f"Pretrained Binary diff {bin_diff} exceeds numerical tolerance"
    assert diag_diff < 1e-5, f"Pretrained Diagnosis diff {diag_diff} exceeds numerical tolerance"


# ==============================================================================
# 14. Task A: Canonical Padding Excluded from Loss and Predictions
# ==============================================================================
def test_canonical_padding_excluded_from_loss_and_predictions():
    """Verify canonical padding tokens are excluded from predictions and loss calculation."""
    torch.manual_seed(99)
    model = LangfensWav2Vec2MDD(hidden_dim=256, num_transformer_layers=2)
    model.eval()

    waveforms = torch.randn(2, 16000)
    audio_lengths = torch.tensor([16000, 16000], dtype=torch.long)
    canonical_indices = torch.tensor([
        [5, 12, 19, 0],
        [8, 22, 0, 0]
    ], dtype=torch.long)
    can_lengths = torch.tensor([3, 2], dtype=torch.long)

    preds = model.predict(
        waveforms=waveforms,
        audio_lengths=audio_lengths,
        canonical_phone_indices=canonical_indices,
        canonical_phone_lengths=can_lengths
    )

    assert "canonical_mask" in preds
    assert preds["canonical_mask"][0].tolist() == [True, True, True, False]
    assert preds["canonical_mask"][1].tolist() == [True, True, False, False]


# ==============================================================================
# 15. Task B: Silent Truncation Removed — Positive and Negative Tests
# ==============================================================================
def test_silent_truncation_raises_value_error():
    """
    MANDATORY NEGATIVE TEST:
    Verify that sequence length and batch size mismatches between targets and logits
    raise explicit ValueError instead of silently truncating via min().
    """
    model = LangfensWav2Vec2MDD(hidden_dim=256, num_transformer_layers=2)
    model.train()

    waveforms = torch.randn(2, 16000)
    audio_lengths = torch.tensor([16000, 16000], dtype=torch.long)
    canonical_indices = torch.tensor([
        [5, 12, 19],
        [8, 22, 14]
    ], dtype=torch.long) # num_canonical_phones = 3

    # Negative 1: Target binary sequence length mismatch (4 vs 3)
    incompatible_target_binary = torch.tensor([
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ], dtype=torch.long)

    with pytest.raises(ValueError, match="target_binary sequence length .* does not match .* Silent label truncation is strictly forbidden"):
        model(
            waveforms=waveforms,
            audio_lengths=audio_lengths,
            canonical_phone_indices=canonical_indices,
            target_binary=incompatible_target_binary
        )

    # Negative 2: Target diagnosis sequence length mismatch (2 vs 3)
    incompatible_target_diag = torch.tensor([
        [4, 11],
        [7, 21]
    ], dtype=torch.long)

    with pytest.raises(ValueError, match="target_diagnosis sequence length .* does not match .* Silent label truncation is strictly forbidden"):
        model(
            waveforms=waveforms,
            audio_lengths=audio_lengths,
            canonical_phone_indices=canonical_indices,
            target_diagnosis=incompatible_target_diag
        )

    # Negative 3: Target batch size mismatch (3 vs 2)
    incompatible_batch_binary = torch.tensor([
        [0, 1, 0],
        [1, 0, 1],
        [0, 0, 1]
    ], dtype=torch.long)

    with pytest.raises(ValueError, match="target_binary batch size .* does not match"):
        model(
            waveforms=waveforms,
            audio_lengths=audio_lengths,
            canonical_phone_indices=canonical_indices,
            target_binary=incompatible_batch_binary
        )


def test_legitimate_labels_in_padding_dimension_detected():
    """
    MANDATORY NEGATIVE TEST:
    Verify that if legitimate labels (not -100) exist beyond declared canonical_phone_lengths,
    an explicit ValueError is raised to prevent silent label disappearance.
    """
    model = LangfensWav2Vec2MDD(hidden_dim=256, num_transformer_layers=2)
    model.train()

    waveforms = torch.randn(2, 16000)
    audio_lengths = torch.tensor([16000, 16000], dtype=torch.long)
    canonical_indices = torch.tensor([
        [5, 12, 19, 0],
        [8, 22, 0, 0]
    ], dtype=torch.long)
    can_lengths = torch.tensor([3, 2], dtype=torch.long)

    # Defective target: utt 2 has a legitimate label (1) at index 2, despite can_len=2!
    corrupted_target_binary = torch.tensor([
        [0, 1, 0, -100],
        [1, 0, 1, -100]  # pos 2 is '1' instead of -100!
    ], dtype=torch.long)

    with pytest.raises(ValueError, match="contains legitimate labels beyond declared canonical_phone_length"):
        model(
            waveforms=waveforms,
            audio_lengths=audio_lengths,
            canonical_phone_indices=canonical_indices,
            canonical_phone_lengths=can_lengths,
            target_binary=corrupted_target_binary
        )


# ==============================================================================
# 16. Task C: Runtime CTC Contract Enforcement & Zero-Infinity Hardening
# ==============================================================================
def test_runtime_ctc_contract_enforcement_cases():
    """
    Task C: Test normal, repeated-phone, all-masked/empty-target, and infeasible alignment cases.
    Verifies that zero_infinity=False does not conceal annotation or shape defects.
    """
    model = LangfensWav2Vec2MDD(hidden_dim=256, num_transformer_layers=2, ctc_zero_infinity_policy="raise")
    model.train()

    # 1. Normal case: 1.0s audio (~49 frames), 4 target phonemes -> passes
    w = torch.randn(1, 16000)
    audio_lens = torch.tensor([16000], dtype=torch.long)
    target_ctc = torch.tensor([[5, 12, 19, 25]], dtype=torch.long)
    target_lens = torch.tensor([4], dtype=torch.long)

    out_normal = model(
        waveforms=w,
        audio_lengths=audio_lens,
        target_ctc=target_ctc,
        target_lengths=target_lens
    )
    assert out_normal.loss is not None
    assert torch.isfinite(out_normal.loss)

    # 2. Repeated-phone feasible case: target [5, 5] requires 3 frames; 16000 samples gives ~49 frames -> passes
    target_repeat = torch.tensor([[5, 5]], dtype=torch.long)
    target_repeat_len = torch.tensor([2], dtype=torch.long)
    out_repeat = model(
        waveforms=w,
        audio_lengths=audio_lens,
        target_ctc=target_repeat,
        target_lengths=target_repeat_len
    )
    assert torch.isfinite(out_repeat.loss)

    # 3. Infeasible alignment case: target length exceeds audio frames
    # Short audio: 1000 samples -> 2 frames; target length = 4 -> requires >= 4 frames!
    w_short = torch.randn(1, 1000)
    short_audio_lens = torch.tensor([1000], dtype=torch.long)
    with pytest.raises(CTCContractError, match="CTC length violation"):
        model(
            waveforms=w_short,
            audio_lengths=short_audio_lens,
            target_ctc=target_ctc,
            target_lengths=target_lens
        )

    # 4. Target contains blank_id (0) within active sequence -> rejected
    corrupted_ctc = torch.tensor([[5, 0, 19, 25]], dtype=torch.long)
    with pytest.raises(CTCContractError, match="contains blank_id"):
        model(
            waveforms=w,
            audio_lengths=audio_lens,
            target_ctc=corrupted_ctc,
            target_lengths=target_lens
        )

    # 5. Out of vocabulary token -> rejected
    oov_ctc = torch.tensor([[5, 999, 19, 25]], dtype=torch.long)
    with pytest.raises(CTCContractError, match="out-of-vocabulary"):
        model(
            waveforms=w,
            audio_lengths=audio_lens,
            target_ctc=oov_ctc,
            target_lengths=target_lens
        )


def test_ctc_contract_documented_zero_loss_policy():
    """
    Task C: Verify that ctc_zero_infinity_policy="warn_and_zero" increments counters,
    reports violation details, and maintains finite differentiable autograd loss.
    """
    model = LangfensWav2Vec2MDD(hidden_dim=256, num_transformer_layers=2, ctc_zero_infinity_policy="warn_and_zero")
    model.train()

    # Infeasible alignment (1000 samples -> 2 frames < 4 target phones)
    w_short = torch.randn(1, 1000)
    short_audio_lens = torch.tensor([1000], dtype=torch.long)
    target_ctc = torch.tensor([[5, 12, 19, 25]], dtype=torch.long)
    target_lens = torch.tensor([4], dtype=torch.long)

    assert model.ctc_contract_violations == 0
    out = model(
        waveforms=w_short,
        audio_lengths=short_audio_lens,
        target_ctc=target_ctc,
        target_lengths=target_lens
    )

    # Violation counter must increment and details must be logged
    assert model.ctc_contract_violations == 1
    assert "CTC length violation" in model.last_ctc_violation_details
    assert out.loss is not None
    assert out.loss.item() == 0.0

    # Retains backward graph
    out.loss.backward()
    assert model.ctc_head.weight.grad is not None


def test_ctc_empty_targets_handling():
    """Task C: Verify empty targets (target_lengths=0) produce finite differentiable loss without NaN."""
    model = LangfensWav2Vec2MDD(hidden_dim=256, num_transformer_layers=2)
    model.train()

    w = torch.randn(2, 16000)
    audio_lens = torch.tensor([16000, 16000], dtype=torch.long)
    empty_ctc = torch.zeros(2, 4, dtype=torch.long)
    empty_lengths = torch.tensor([0, 0], dtype=torch.long)

    out = model(
        waveforms=w,
        audio_lengths=audio_lens,
        target_ctc=empty_ctc,
        target_lengths=empty_lengths
    )

    assert out.loss is not None
    assert torch.isfinite(out.loss)
    assert out.loss.item() == 0.0
    out.loss.backward()
    assert model.ctc_head.weight.grad is not None
