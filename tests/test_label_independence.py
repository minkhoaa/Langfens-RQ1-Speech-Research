import pytest
import torch
from langfens_prep.model import LangfensWav2Vec2MDD


def test_label_independence_on_dummy_inputs():
    """Verifies that target_ctc and target_binary have ZERO influence on model logits."""
    model = LangfensWav2Vec2MDD(
        hidden_dim=32, conv_dim=16, num_transformer_layers=1,
        num_attention_heads=4, intermediate_dim=64, dropout=0.1
    )
    model.eval()

    batch_size = 2
    raw_audio_len = 16000
    num_phones = 5

    source = torch.randn(batch_size, raw_audio_len)
    audio_lengths = torch.tensor([raw_audio_len, raw_audio_len // 2])
    can_indices = torch.randint(1, 40, (batch_size, num_phones))
    can_lengths = torch.tensor([num_phones, num_phones - 1])

    target_ctc = torch.randint(1, 40, (batch_size, num_phones))
    target_lengths = torch.tensor([num_phones, num_phones - 1])
    target_binary = torch.randint(0, 2, (batch_size, num_phones))
    # Strict contract: padding positions beyond declared length must be -100
    for b in range(batch_size):
        target_binary[b, can_lengths[b]:] = -100
        target_ctc[b, target_lengths[b]:] = 0

    with torch.inference_mode():
        out_no_targets = model(
            source=source,
            audio_lengths=audio_lengths,
            canonical_phone_indices=can_indices,
            canonical_phone_lengths=can_lengths
        )

        out_with_targets = model(
            source=source,
            audio_lengths=audio_lengths,
            canonical_phone_indices=can_indices,
            canonical_phone_lengths=can_lengths,
            target_ctc=target_ctc,
            target_lengths=target_lengths,
            target_binary=target_binary
        )

    # 1. Prediction-only pass produces no loss
    assert out_no_targets.loss is None
    assert out_with_targets.loss is not None
    assert torch.isfinite(out_with_targets.loss)

    # 2. CTC logits must be exactly identical (diff == 0.0)
    assert torch.equal(out_no_targets.ctc_logits, out_with_targets.ctc_logits)
    assert (out_no_targets.ctc_logits - out_with_targets.ctc_logits).abs().max().item() == 0.0

    # 3. Binary logits must be exactly identical (diff == 0.0)
    assert torch.equal(out_no_targets.binary_logits, out_with_targets.binary_logits)
    assert (out_no_targets.binary_logits - out_with_targets.binary_logits).abs().max().item() == 0.0

    # 4. Input lengths must be identical
    assert torch.equal(out_no_targets.input_lengths, out_with_targets.input_lengths)


def test_label_independence_on_real_checkpoint_weights():
    """Verifies label independence with production model weights if checkpoint is present."""
    from pathlib import Path
    ckpt_path = Path("checkpoints/RQ1-TRN-02A/recovered/best.pt")
    if not ckpt_path.is_file():
        pytest.skip("Production checkpoint not found; skipping weights test")

    model = LangfensWav2Vec2MDD.from_pretrained(
        "facebook/wav2vec2-base-960h",
        revision="22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    source = torch.randn(1, 16000)
    audio_lengths = torch.tensor([16000])
    can_indices = torch.tensor([[1, 2, 3, 4]])
    can_lengths = torch.tensor([4])
    target_ctc = torch.tensor([[1, 2, 3, 4]])
    target_lengths = torch.tensor([4])
    target_binary = torch.tensor([[0, 1, 0, 1]])

    with torch.inference_mode():
        out_no_targets = model(
            source=source,
            audio_lengths=audio_lengths,
            canonical_phone_indices=can_indices,
            canonical_phone_lengths=can_lengths
        )
        out_with_targets = model(
            source=source,
            audio_lengths=audio_lengths,
            canonical_phone_indices=can_indices,
            canonical_phone_lengths=can_lengths,
            target_ctc=target_ctc,
            target_lengths=target_lengths,
            target_binary=target_binary
        )

    assert torch.equal(out_no_targets.ctc_logits, out_with_targets.ctc_logits)
    assert torch.equal(out_no_targets.binary_logits, out_with_targets.binary_logits)
