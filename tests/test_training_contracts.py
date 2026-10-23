import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from langfens_prep.phoneme_normalizer import (
    STANDARD_39_ARPABET,
    PHONE_TO_CTC_INDEX,
    PHONE_TO_INDEX,
    normalize_phoneme
)
from langfens_prep.losses import (
    MaskedBCEWithLogitsLoss,
    MaskedCrossEntropyLoss,
    validate_ctc_contract
)
from langfens_prep.collate import LangfensMDDCollate
from langfens_prep.textgrid_parser import parse_gold_textgrid_content

def test_binary_mdd_targets_distinguish_valid_from_padding():
    """Verify that binary labels are strictly 0 or 1, and padding is strictly distinct (-100)."""
    valid_labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    padded = torch.tensor([0, 1, 0, 1, -100, -100], dtype=torch.long)
    
    assert set(valid_labels.tolist()).issubset({0, 1})
    pad_mask = (padded == -100)
    assert pad_mask.sum().item() == 2
    assert set(padded[~pad_mask].tolist()).issubset({0, 1})
    assert -100 not in {0, 1}

def test_bce_with_logits_loss_masked_vs_unmasked_catastrophe():
    """
    CRITICAL STATIC CONTRACT TEST:
    Demonstrates that passing padded -100 into standard nn.BCEWithLogitsLoss
    causes massive mathematical loss distortion, whereas MaskedBCEWithLogitsLoss
    evaluates strictly over valid binary positions.
    """
    logits = torch.tensor([[2.0, -1.0, 0.5, 0.0]], requires_grad=True)
    # Valid targets: 1, 0; Padded targets: -100, -100
    targets_with_pad = torch.tensor([[1, 0, -100, -100]], dtype=torch.long)
    
    # Standard BCEWithLogitsLoss treats -100 as real binary target
    unmasked_loss = F.binary_cross_entropy_with_logits(
        logits, targets_with_pad.float(), reduction="mean"
    )
    # Notice: with -100 target, loss at pos 2 is: -100 * log(sig(0.5)) + 101 * log(1-sig(0.5))
    # which is approx 62.4 + 101 * (-0.974) -> huge negative/corrupted loss
    assert unmasked_loss.item() != pytest.approx(0.22, abs=0.5)
    
    # Masked loss computes loss strictly over valid positions (indices 0 and 1)
    masked_loss_module = MaskedBCEWithLogitsLoss(ignore_index=-100)
    masked_loss = masked_loss_module(logits, targets_with_pad)
    
    expected_valid_loss = F.binary_cross_entropy_with_logits(
        logits[:, :2], torch.tensor([[1.0, 0.0]]), reduction="mean"
    )
    assert pytest.approx(masked_loss.item(), 1e-5) == expected_valid_loss.item()

def test_masked_bce_gradient_flow():
    """Verify gradients for padded positions in MaskedBCEWithLogitsLoss are strictly zero."""
    logits = torch.tensor([[1.5, -2.0, 3.0]], requires_grad=True)
    targets = torch.tensor([[1, 0, -100]], dtype=torch.long)
    
    loss_fn = MaskedBCEWithLogitsLoss(ignore_index=-100)
    loss = loss_fn(logits, targets)
    loss.backward()
    
    assert logits.grad is not None
    # Gradient on padded position (index 2) must be exactly 0.0
    assert logits.grad[0, 2].item() == 0.0
    # Gradients on active positions (indices 0 and 1) must be non-zero
    assert logits.grad[0, 0].item() != 0.0
    assert logits.grad[0, 1].item() != 0.0

def test_masked_crossentropy_loss_masks_deletions_and_err():
    """
    Verify MaskedCrossEntropyLoss ignores:
    1. Padded tokens (-100)
    2. Deletions (-100)
    3. Unaligned or 'err' tags (-100)
    """
    num_classes = 39
    seq_len = 4
    batch_size = 1
    
    # logits shape: (batch_size, num_classes, seq_len)
    logits = torch.randn(batch_size, num_classes, seq_len, requires_grad=True)
    
    # Target: pos 0 is class 5 (valid), pos 1 is -100 ('err'), pos 2 is -100 (deletion), pos 3 is -100 (pad)
    targets = torch.tensor([[5, -100, -100, -100]], dtype=torch.long)
    
    loss_fn = MaskedCrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(logits, targets)
    loss.backward()
    
    # Expected loss is simply cross entropy on pos 0
    expected_loss = F.cross_entropy(logits[:, :, :1], torch.tensor([[5]]))
    assert pytest.approx(loss.item(), 1e-5) == expected_loss.item()
    
    # Gradients on positions 1, 2, 3 must be exactly 0.0
    assert (logits.grad[:, :, 1:] == 0.0).all()
    # Gradient on pos 0 must be non-zero
    assert (logits.grad[:, :, 0] != 0.0).any()

def test_ctc_blank_id_never_in_target_sequences():
    """Verify that blank ID 0 never enters valid target sequences."""
    # 1. Check all standard 39 phones map to >= 1
    for phone in STANDARD_39_ARPABET:
        ctc_idx = PHONE_TO_CTC_INDEX[phone]
        assert ctc_idx > 0, f"Phone {phone} mapped to non-positive CTC index {ctc_idx}"
        assert ctc_idx <= 39
    
    assert 0 not in PHONE_TO_CTC_INDEX.values()

def test_ctc_contract_validation():
    """Verify validate_ctc_contract enforces mathematical preconditions."""
    N, T, C = 2, 10, 40 # 40 classes (0 = blank, 1..39 phones)
    log_probs = F.log_softmax(torch.randn(N, T, C), dim=-1)
    
    # Valid case: T_in = 8, 10; L_tgt = 4, 6 (input >= target)
    input_lengths = torch.tensor([8, 10], dtype=torch.long)
    target_lengths = torch.tensor([4, 6], dtype=torch.long)
    targets = torch.tensor([
        [1, 5, 12, 39, 0, 0],
        [2, 4, 10, 15, 20, 25]
    ], dtype=torch.long)
    
    valid, msg = validate_ctc_contract(log_probs, targets, input_lengths, target_lengths, blank_id=0)
    assert valid is True
    assert msg is None
    
    # Failure 1: Input length < target length (impossible alignment)
    invalid_input_lengths = torch.tensor([3, 10], dtype=torch.long) # 3 < 4!
    valid, msg = validate_ctc_contract(log_probs, targets, invalid_input_lengths, target_lengths, blank_id=0)
    assert valid is False
    assert "CTC length violation" in msg
    
    # Failure 2: Target contains blank ID (0) within active sequence
    invalid_targets = torch.tensor([
        [1, 0, 12, 39, 0, 0], # pos 1 is blank 0!
        [2, 4, 10, 15, 20, 25]
    ], dtype=torch.long)
    valid, msg = validate_ctc_contract(log_probs, invalid_targets, input_lengths, target_lengths, blank_id=0)
    assert valid is False
    assert "contains blank_id (0) within active sequence" in msg
    
    # Failure 3: NaN in log_probs
    nan_log_probs = log_probs.clone()
    nan_log_probs[0, 0, 0] = float("nan")
    valid, msg = validate_ctc_contract(nan_log_probs, targets, input_lengths, target_lengths, blank_id=0)
    assert valid is False
    assert "NaN or Inf" in msg

def test_realized_sequence_semantics():
    """Verify addition, deletion, and 'err' uncertainty semantics in TextGrid parsing."""
    textgrid_snippet = """File type = "ooTextFile"
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
            xmin = 0.0 
            xmax = 0.2 
            text = "sil" 
        intervals [2]:
            xmin = 0.2 
            xmax = 0.4 
            text = "K,G,s" 
        intervals [3]:
            xmin = 0.4 
            xmax = 0.6 
            text = "AE,sil,d" 
        intervals [4]:
            xmin = 0.6 
            xmax = 0.8 
            text = "T,err,s" 
        intervals [5]:
            xmin = 0.8 
            xmax = 1.0 
            text = "sil,AH,a" 
"""
    ann = parse_gold_textgrid_content(textgrid_snippet, "test_semantics.TextGrid")
    
    # Invariant: silence and addition are non-canonical
    canonical = ann.canonical_intervals
    assert len(canonical) == 3 # K->G (sub), AE->sil (del), T->err (sub with err)
    
    # Sub 1: K -> G
    assert canonical[0].error_type == "substitution"
    assert canonical[0].canonical_phone == "K"
    assert canonical[0].realized_phone == "G"
    assert canonical[0].diagnosis_target == "G"
    assert canonical[0].binary_label_strict == 1
    
    # Del 2: AE -> sil
    assert canonical[1].error_type == "deletion"
    assert canonical[1].canonical_phone == "AE"
    assert canonical[1].realized_phone == "sil"
    assert canonical[1].diagnosis_target is None # Deletions masked from diagnosis!
    assert canonical[1].binary_label_strict == 1
    
    # Sub 3: T -> err
    assert canonical[2].error_type == "substitution"
    assert canonical[2].has_err_tag is True
    assert canonical[2].diagnosis_target is None # 'err' tags masked from 39-class diagnosis!
    assert canonical[2].binary_label_strict == 1
    
    # Addition: sil -> AH
    addition = [i for i in ann.intervals if i.error_type == "addition"][0]
    assert addition.is_canonical is False
    assert addition.realized_phone == "AH"

def test_collate_with_masks_and_diagnosis():
    """Verify LangfensMDDCollate emits target_binary_mask and target_diagnosis."""
    item1 = {
        "record_id": "utt1",
        "audio_waveform_16k": torch.randn(1600),
        "binary_labels": torch.tensor([0, 1, 0], dtype=torch.long),
        "ctc_target_indices": torch.tensor([5, 12], dtype=torch.long),
        "diagnosis_target_indices": torch.tensor([4, 11, -100], dtype=torch.long),
        "is_blinded": False
    }
    item2 = {
        "record_id": "utt2",
        "audio_waveform_16k": torch.randn(800),
        "binary_labels": torch.tensor([1], dtype=torch.long),
        "ctc_target_indices": torch.tensor([7], dtype=torch.long),
        "diagnosis_target_indices": torch.tensor([6], dtype=torch.long),
        "is_blinded": False
    }
    
    collator = LangfensMDDCollate()
    batch = collator([item1, item2])
    
    assert "target_binary" in batch
    assert "target_binary_mask" in batch
    assert "target_diagnosis" in batch
    
    assert batch["target_binary"].shape == (2, 3)
    assert batch["target_binary_mask"].shape == (2, 3)
    assert batch["target_diagnosis"].shape == (2, 3)
    
    # Utt 1: 3 valid binary labels
    assert batch["target_binary_mask"][0].tolist() == [True, True, True]
    # Utt 2: 1 valid label, 2 padded (-100)
    assert batch["target_binary_mask"][1].tolist() == [True, False, False]
    assert batch["target_binary"][1, 1].item() == -100
    assert batch["target_binary"][1, 2].item() == -100
    
    # Diagnosis: pos 2 of utt 1 is -100 (masked deletion/err); pos 1 and 2 of utt 2 are -100 (padded)
    assert batch["target_diagnosis"][0, 2].item() == -100
    assert batch["target_diagnosis"][1, 1].item() == -100

def test_masked_bce_all_masked_gradient_connectivity():
    """Verify that when an entire batch is masked (-100), loss is 0.0 and logits.grad is populated with 0.0."""
    logits = torch.tensor([[1.0, 2.0, -1.0]], requires_grad=True)
    all_masked_targets = torch.tensor([[-100, -100, -100]], dtype=torch.long)

    loss_fn = MaskedBCEWithLogitsLoss(ignore_index=-100)
    loss = loss_fn(logits, all_masked_targets)

    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits))

def test_masked_ce_all_masked_avoids_nan():
    """Verify that when an entire batch is masked (-100), MaskedCrossEntropyLoss returns 0.0 instead of NaN."""
    logits = torch.randn(2, 39, requires_grad=True)
    all_masked_targets = torch.tensor([-100, -100], dtype=torch.long)

    loss_fn = MaskedCrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(logits, all_masked_targets)

    assert not torch.isnan(loss)
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits))

def test_ctc_contract_adjacent_repeated_tokens():
    """Verify validate_ctc_contract catches impossible alignments with adjacent repeated tokens."""
    N, T, C = 1, 2, 40
    log_probs = F.log_softmax(torch.randn(N, T, C), dim=-1)

    # Target sequence with adjacent repeated tokens: [5, 5] (L = 2)
    # Since adjacent tokens are identical, CTC requires intervening blank, so min frames = 3!
    targets = torch.tensor([[5, 5]], dtype=torch.long)
    input_lengths = torch.tensor([2], dtype=torch.long) # T = 2 < 3!
    target_lengths = torch.tensor([2], dtype=torch.long)

    valid, msg = validate_ctc_contract(log_probs, targets, input_lengths, target_lengths, blank_id=0)
    assert valid is False
    assert "adjacent identical tokens" in msg

    # With T = 3, it must become valid
    input_lengths_3 = torch.tensor([3], dtype=torch.long)
    log_probs_3 = F.log_softmax(torch.randn(1, 3, C), dim=-1)
    valid_3, msg_3 = validate_ctc_contract(log_probs_3, targets, input_lengths_3, target_lengths, blank_id=0)
    assert valid_3 is True
    assert msg_3 is None

def test_frame_to_phone_pooling_guarantees():
    """Verify frame-to-phone pooling guarantees non-empty slices and proper clamping."""
    from langfens_prep.alignment import pool_frames_to_canonical_phones

    batch_size = 1
    T_frames = 10
    hidden_dim = 16
    frame_features = torch.randn(batch_size, T_frames, hidden_dim)

    # Interval with duration < 1 frame stride (0.001s < 0.02s)
    intervals = [[(0.010, 0.011), (0.05, 0.15)]]

    pooled, mask = pool_frames_to_canonical_phones(
        frame_features=frame_features,
        intervals_per_batch=intervals,
        sample_rate=16000,
        frame_stride_samples=320,
        pooling_mode="mean"
    )

    assert pooled.shape == (1, 2, hidden_dim)
    assert mask.shape == (1, 2)
    assert mask.all()
    assert torch.isfinite(pooled).all()

