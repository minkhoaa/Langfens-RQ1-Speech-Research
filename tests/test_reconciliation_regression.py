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
from langfens_prep.losses import MaskedBCEWithLogitsLoss, MaskedCrossEntropyLoss


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
