import pytest
import torch
import numpy as np

def test_load_audio_from_l2arctic_nested_archive():
    from langfens_prep.audio_loader import load_l2arctic_audio_from_zip
    
    zip_path = "datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip"
    speaker = "TLV"
    utt_id = "arctic_a0001"
    
    waveform, sr, metrics = load_l2arctic_audio_from_zip(
        zip_path=zip_path,
        speaker=speaker,
        utterance_id=utt_id,
        target_sr=16000
    )
    
    assert isinstance(waveform, torch.Tensor)
    assert waveform.dtype == torch.float32
    assert sr == 16000
    assert waveform.ndim == 1 # 1D mono waveform
    assert waveform.shape[0] > 0
    assert torch.isfinite(waveform).all()
    assert -1.0 <= waveform.min().item() and waveform.max().item() <= 1.0
    assert metrics.raw_sr == 44100
    assert metrics.target_sr == 16000
    assert metrics.raw_channels == 1
    assert metrics.duration_seconds > 0.0

def test_load_audio_from_speechocean_archive():
    from langfens_prep.audio_loader import load_speechocean_audio_from_tar
    
    tar_path = "datasets/SpeechOcean762/original/speechocean762.tar.gz"
    member_path = "speechocean762/WAVE/SPEAKER0001/000010011.WAV"
    
    waveform, sr, metrics = load_speechocean_audio_from_tar(
        tar_path=tar_path,
        member_path=member_path,
        target_sr=16000
    )
    
    assert isinstance(waveform, torch.Tensor)
    assert waveform.dtype == torch.float32
    assert sr == 16000
    assert waveform.shape[0] > 0
    assert torch.isfinite(waveform).all()
    assert metrics.raw_sr == 16000
    assert metrics.duration_seconds > 0.0

def test_configurable_normalization_and_dc():
    from langfens_prep.audio_loader import process_waveform
    
    raw = torch.tensor([0.2, 0.4, -0.6, 0.1], dtype=torch.float32)
    # Without peak norm / dc removal
    out1, m1 = process_waveform(raw, normalize_peak=False, remove_dc=False)
    assert torch.equal(out1, raw)
    
    # With peak norm
    out2, m2 = process_waveform(raw, normalize_peak=True, remove_dc=False)
    assert pytest.approx(out2.abs().max().item(), 1e-4) == 1.0
    
    # With DC removal
    biased = raw + 0.5
    out3, m3 = process_waveform(biased, normalize_peak=False, remove_dc=True)
    assert pytest.approx(out3.mean().item(), abs=1e-6) == 0.0
