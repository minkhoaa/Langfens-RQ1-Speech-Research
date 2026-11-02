"""FP16 shares the reviewed bootstrap/evaluator; precision tests use CPU fakes only."""
from contextlib import nullcontext
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch

from langfens_bench import runtime

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'deploy/bench_kaggle_fp16/bench_kaggle_fp16.py'


def test_bootstraps_identical_except_fixed_precision_profile():
    fp32 = (ROOT / 'deploy/bench_kaggle_fp32/kernel.py').read_text()
    assert SCRIPT.read_text() == fp32.replace('PROFILE = "KAGGLE_FP32"', 'PROFILE = "KAGGLE_FP16"')
    assert runpy.run_path(str(SCRIPT))['PROFILE'] == 'KAGGLE_FP16'


def test_private_metadata_and_interfaces_match():
    fp32 = json.loads((ROOT / 'deploy/bench_kaggle_fp32/kernel-metadata.json').read_text())
    fp16 = json.loads((SCRIPT.parent / 'kernel-metadata.json').read_text())
    assert fp16['id'] == 'minkhoaa/langfens-rq1-bench-fp16-private'
    assert fp16['code_file'] == SCRIPT.name
    for key in ('machine_shape', 'enable_gpu', 'is_private', 'enable_internet',
                'dataset_sources', 'kernel_sources', 'model_sources', 'competition_sources'):
        assert fp16[key] == fp32[key]
    assert fp16['is_private'] is True and fp16['enable_internet'] is False


@pytest.mark.parametrize('argv', [[], ['--fold', 'all'], ['--fold', '0'],
                                  ['--fold', '1', '--source-sha256', '0' * 64, 'evaluate']])
def test_missing_fold_source_pin_or_authorization_refused_before_staging(argv):
    bootstrap = runpy.run_path(str(SCRIPT))['bootstrap']
    with pytest.raises(SystemExit) as exc:
        bootstrap(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize('profile', ['LOCAL_FP32', 'KAGGLE_FP32', 'KAGGLE_FP16'])
def test_only_fp16_uses_explicit_cuda_fp16_autocast_during_inference(monkeypatch, profile):
    contexts, calls = [], []
    def autocast(**kwargs):
        contexts.append(kwargs)
        return nullcontext()
    def model(**kwargs):
        assert torch.is_inference_mode_enabled()
        assert kwargs['waveforms'].dtype == torch.float32
        calls.append(kwargs)
        return SimpleNamespace(binary_logits=torch.tensor([[1.0]]))
    monkeypatch.setattr(torch, 'autocast', autocast)
    # CPU fake tests control flow only; this is NOT a CUDA benchmark.
    runtime.forward_pair(model, torch.zeros(400), ['AA'], ['AA'], profile, torch.device('cpu'))
    assert len(calls) == 2
    assert contexts == ([{'device_type': 'cuda', 'dtype': torch.float16}] if profile == 'KAGGLE_FP16' else [])


def test_stored_weights_never_half_converted_and_fp32_postprocessing_present():
    source = (ROOT / 'src/langfens_bench/runtime.py').read_text()
    assert '.half()' not in source
    assert 'output.binary_logits[0, :len(phones)].float()' in source
    assert 'z_output.binary_logits[0, :len(z_phones)].float()' in source
    assert 'output.alignment_weights[0, :len(phones), :active_frames].float()' in source
    assert 'torch.cuda.synchronize(device)' in source
    assert 'torch.cuda.max_memory_allocated(device)' in source
    assert 'torch.cuda.max_memory_reserved(device)' in source
