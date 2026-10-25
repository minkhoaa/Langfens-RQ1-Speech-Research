import os
import tempfile
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from langfens_prep.model import LangfensWav2Vec2MDD, LangfensMDDOutput

@pytest.fixture
def synthetic_mdd_batch():
    """Generates deterministic synthetic speech batch for unit & smoke testing."""
    torch.manual_seed(42)
    batch_size = 2
    sample_rate = 16000
    # 0.5s and 0.4s waveforms
    audio_len1 = int(0.5 * sample_rate) # 8000 samples -> ~25 frames
    audio_len2 = int(0.4 * sample_rate) # 6400 samples -> ~20 frames
    max_samples = 8000

    waveforms = torch.randn(batch_size, max_samples)
    audio_lengths = torch.tensor([audio_len1, audio_len2], dtype=torch.long)

    # Intervals: 3 phonemes in utt 1, 2 phonemes in utt 2
    phone_intervals = [
        [(0.05, 0.15), (0.15, 0.30), (0.30, 0.45)],
        [(0.05, 0.20), (0.20, 0.38)]
    ]
    # Canonical phone indices: (B, 3)
    canonical_indices = torch.tensor([
        [5, 12, 19],
        [8, 22, 0] # 0 = padding
    ], dtype=torch.long)

    # Target binary: (B, 3)
    target_binary = torch.tensor([
        [0, 1, 0],
        [1, 0, -100]
    ], dtype=torch.long)

    # Target CTC: (B, 3)
    target_ctc = torch.tensor([
        [5, 12, 19],
        [8, 22, 0]
    ], dtype=torch.long)
    target_lengths = torch.tensor([3, 2], dtype=torch.long)

    # Target diagnosis: (B, 3)
    target_diagnosis = torch.tensor([
        [4, 11, 18],
        [7, 21, -100]
    ], dtype=torch.long)

    return {
        "waveforms": waveforms,
        "audio_lengths": audio_lengths,
        "phone_intervals": phone_intervals,
        "canonical_phone_indices": canonical_indices,
        "target_binary": target_binary,
        "target_ctc": target_ctc,
        "target_lengths": target_lengths,
        "target_diagnosis": target_diagnosis
    }

def test_model_forward_ctc_only(synthetic_mdd_batch):
    """Verify forward pass with CTC head only."""
    model = LangfensWav2Vec2MDD(
        hidden_dim=256,
        conv_dim=128,
        num_transformer_layers=2,
        num_attention_heads=4,
        intermediate_dim=512
    )
    model.eval()

    with torch.no_grad():
        out = model(
            waveforms=synthetic_mdd_batch["waveforms"],
            audio_lengths=synthetic_mdd_batch["audio_lengths"]
        )

    assert isinstance(out, LangfensMDDOutput)
    assert out.ctc_logits is not None
    assert out.ctc_logits.shape[0] == 2
    assert out.ctc_logits.shape[2] == 40 # 40 CTC classes
    assert out.input_lengths is not None
    assert (out.input_lengths > 0).all()

def test_model_multitask_forward_and_loss(synthetic_mdd_batch):
    """Verify multi-task forward pass, all heads, and loss computation."""
    model = LangfensWav2Vec2MDD(
        hidden_dim=256,
        conv_dim=128,
        num_transformer_layers=2,
        num_attention_heads=4,
        intermediate_dim=512
    )
    model.train()

    out = model(
        waveforms=synthetic_mdd_batch["waveforms"],
        audio_lengths=synthetic_mdd_batch["audio_lengths"],
        canonical_phone_indices=synthetic_mdd_batch["canonical_phone_indices"],
        phone_intervals=synthetic_mdd_batch["phone_intervals"],
        target_binary=synthetic_mdd_batch["target_binary"],
        target_ctc=synthetic_mdd_batch["target_ctc"],
        target_lengths=synthetic_mdd_batch["target_lengths"],
        target_diagnosis=synthetic_mdd_batch["target_diagnosis"]
    )

    assert out.ctc_logits.shape[0] == 2
    assert out.binary_logits.shape == (2, 3)
    assert out.diagnosis_logits.shape == (2, 3, 39)
    assert out.loss is not None
    assert torch.isfinite(out.loss)
    assert "ctc_loss" in out.loss_dict
    assert "binary_loss" in out.loss_dict
    assert "diagnosis_loss" in out.loss_dict

    # Verify backward pass
    out.loss.backward()
    # Check gradients exist on heads
    assert model.ctc_head.weight.grad is not None
    assert model.binary_head[0].weight.grad is not None
    assert model.diagnosis_head[0].weight.grad is not None

def test_freeze_feature_encoder():
    """Verify feature extractor freezing functionality."""
    model = LangfensWav2Vec2MDD(hidden_dim=128, conv_dim=64, num_transformer_layers=1)
    model.freeze_feature_encoder()

    for param in model.feature_extractor.parameters():
        assert param.requires_grad is False

    # Transformer and heads must remain trainable
    for param in model.transformer_encoder.parameters():
        assert param.requires_grad is True
    assert model.ctc_head.weight.requires_grad is True

def test_checkpoint_serialization(synthetic_mdd_batch):
    """Verify model state saving and loading consistency."""
    model1 = LangfensWav2Vec2MDD(hidden_dim=128, conv_dim=64, num_transformer_layers=1)
    
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        torch.save(model1.state_dict(), tmp_path)
        
        model2 = LangfensWav2Vec2MDD(hidden_dim=128, conv_dim=64, num_transformer_layers=1)
        model2.load_state_dict(torch.load(tmp_path, weights_only=True))

        # Check weights are strictly identical
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_optimizer_step_adamw(synthetic_mdd_batch):
    """Verify AdamW optimizer step updates parameters cleanly without NaN/Inf."""
    model = LangfensWav2Vec2MDD(hidden_dim=128, conv_dim=64, num_transformer_layers=1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    initial_ctc_weight = model.ctc_head.weight.clone()

    out = model(
        waveforms=synthetic_mdd_batch["waveforms"],
        audio_lengths=synthetic_mdd_batch["audio_lengths"],
        canonical_phone_indices=synthetic_mdd_batch["canonical_phone_indices"],
        phone_intervals=synthetic_mdd_batch["phone_intervals"],
        target_binary=synthetic_mdd_batch["target_binary"],
        target_ctc=synthetic_mdd_batch["target_ctc"],
        target_lengths=synthetic_mdd_batch["target_lengths"],
        target_diagnosis=synthetic_mdd_batch["target_diagnosis"]
    )

    out.loss.backward()
    optimizer.step()

    # Parameter must have updated
    assert not torch.equal(model.ctc_head.weight, initial_ctc_weight)
    assert torch.isfinite(model.ctc_head.weight).all()

def test_gpu_forward_backward_if_available(synthetic_mdd_batch):
    """Smoke test on GPU if CUDA is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available in current environment; skipping GPU smoke test.")

    model = LangfensWav2Vec2MDD(
        hidden_dim=256,
        conv_dim=128,
        num_transformer_layers=2,
        num_attention_heads=4,
        intermediate_dim=512
    ).cuda()

    w = synthetic_mdd_batch["waveforms"].cuda()
    al = synthetic_mdd_batch["audio_lengths"].cuda()
    ci = synthetic_mdd_batch["canonical_phone_indices"].cuda()
    pi = synthetic_mdd_batch["phone_intervals"]
    tb = synthetic_mdd_batch["target_binary"].cuda()
    tc = synthetic_mdd_batch["target_ctc"].cuda()
    tl = synthetic_mdd_batch["target_lengths"].cuda()
    td = synthetic_mdd_batch["target_diagnosis"].cuda()

    # Test under AMP fp16
    with torch.amp.autocast('cuda', dtype=torch.float16):
        out = model(
            waveforms=w,
            audio_lengths=al,
            canonical_phone_indices=ci,
            phone_intervals=pi,
            target_binary=tb,
            target_ctc=tc,
            target_lengths=tl,
            target_diagnosis=td
        )

    assert out.loss is not None
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert model.ctc_head.weight.grad is not None


def test_mixed_duration_batch_via_padding_mask():
    """Verify forward and predict handle padding_mask boolean tensor correctly."""
    model = LangfensWav2Vec2MDD(hidden_dim=128, conv_dim=64, num_transformer_layers=1)
    model.eval()

    # 2 waveforms padded to 24,000 samples (sample 1: 16,000, sample 2: 24,000)
    w = torch.randn(2, 24000)
    # padding_mask: True where padded
    mask = torch.tensor([
        [False] * 16000 + [True] * 8000,
        [False] * 24000
    ], dtype=torch.bool)

    can_idx = torch.tensor([[5, 12, 19, 0], [7, 14, 21, 28]], dtype=torch.long)
    can_lens = torch.tensor([3, 4], dtype=torch.long)

    with torch.no_grad():
        out = model(waveforms=w, padding_mask=mask, canonical_phone_indices=can_idx, canonical_phone_lengths=can_lens)
        preds = model.predict(waveforms=w, canonical_phone_indices=can_idx, canonical_phone_lengths=can_lens)

    assert out.input_lengths is not None
    assert out.input_lengths[0] < out.input_lengths[1]
    assert "canonical_mask" in preds
    assert preds["canonical_mask"][0, 3].item() is False
    assert preds["canonical_mask"][1, 3].item() is True
