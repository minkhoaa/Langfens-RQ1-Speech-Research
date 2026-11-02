"""Synthetic bundle tests only: no platform results, datasets or model inference."""
import copy
import gzip
import json
from pathlib import Path

import pytest

from langfens_bench.compare import QUALITY, PERFORMANCE, compare, digest, main


def dump(path, obj):
    path.write_text(json.dumps(obj))


@pytest.fixture
def bundles(tmp_path):
    contract = {'seed': 1337, 'folds': [{'fold': 1, 'checkpoint': {'sha256': 'a' * 64}}],
                'utterances': [{'prediction_id': '1/S/u', 'fold': 1, 'speaker': 'S', 'utterance_id': 'u'}]}
    contract_path = tmp_path / 'contract.json'
    dump(contract_path, contract)
    metadata = {'status': 'COMPLETED', 'contract_sha256': digest(contract_path), 'seed': 1337,
                'checkpoint_sha256': {'1': 'a' * 64}, 'profile': 'LOCAL_FP32',
                'evaluator_semantics_sha256': 'b' * 64}
    record = {**contract['utterances'][0], 'checkpoint_sha256': 'a' * 64,
              'labels': [0, 1], 'error_types': ['correct', 'deletion'], 'canonical_phones': ['AA', 'T'],
              'binary_scores': [0.1, 0.6], 'binary_logits': [-2.2, 0.4], 'ctc_reference_ids': [1], 'ctc_ids': [1],
              'attention_metrics': {'in_window_mass': 0.1, 'normalized_entropy': 0.5, 'absolute_peak_offset_seconds': 0.2},
              'protocol_z': {'canonical_phones': ['AA', 'T'], 'binary_scores': [0.1, 0.6], 'binary_logits': [-2.2, 0.4],
                             'aligned_pairs': [[0, 0, 0], [1, 1, 0]], 'unaligned_gold_count': 0,
                             'unaligned_g2p_count': 0, 'tie_break': 'sub_first'}}
    quality = {k: 0.5 for k in QUALITY}
    quality['protocol_z'] = {'ap': 0.5, 'roc_auc': 0.5, 'coverage': 1.0}
    performance = {k: 1.0 for k in PERFORMANCE}
    dirs = [tmp_path / 'local', tmp_path / 'gpu32', tmp_path / 'gpu16']
    for directory, profile in zip(dirs, ('LOCAL_FP32', 'KAGGLE_FP32', 'KAGGLE_FP16')):
        directory.mkdir()
        meta = {**metadata, 'profile': profile}
        if profile == 'LOCAL_FP32':
            perf = {**performance, 'peak_vram_allocated_bytes': None, 'peak_vram_reserved_bytes': None}
        else:
            perf = performance
        dump(directory / 'metrics.json', {'metadata': meta, 'quality': quality, 'performance': perf,
                                          'by_fold': {'1': {'quality': quality}}})
        with gzip.GzipFile(filename=str(directory / 'predictions.json.gz'), mode='wb', mtime=0) as f:
            f.write(json.dumps({'metadata': meta, 'utterances': [record]}).encode())
        dump(directory / 'environment.json', {'metadata': meta, 'artifacts': {
            f: digest(directory / f) for f in ('metrics.json', 'predictions.json.gz')}})
    return dirs, contract_path


def mutate(directory, file, change):
    path = directory / file
    if file.endswith('.gz'):
        with gzip.open(path, 'rt') as f:
            obj = json.load(f)
        change(obj)
        with gzip.GzipFile(filename=str(path), mode='wb', mtime=0) as f:
            f.write(json.dumps(obj).encode())
    else:
        obj = json.loads(path.read_text())
        change(obj)
        dump(path, obj)
    if file != 'environment.json':
        envpath = directory / 'environment.json'
        env = json.loads(envpath.read_text())
        env['artifacts'][file] = digest(path)
        dump(envpath, env)


@pytest.mark.parametrize('pair', [(0, 1), (1, 2)])
def test_synthetic_identical_bundles_pass(bundles, pair):
    dirs, contract = bundles
    result = compare(dirs[pair[0]], dirs[pair[1]], contract_path=contract)
    assert result['status'] == 'PASS'  # synthetic comparison, not platform parity
    assert result['ctc_exact_utterances'] == 1


def test_pointwise_drift_not_waived_by_equal_aggregate_metrics(bundles):
    dirs, contract = bundles
    mutate(dirs[1], 'predictions.json.gz', lambda p: p['utterances'][0]['binary_scores'].__setitem__(0, 0.101))
    result = compare(*dirs[:2], contract_path=contract)
    assert result['status'] == 'FAIL'
    assert '1/S/u/binary_scores' in result['failed_checks']


def test_fp16_has_explicit_different_registered_tolerance(bundles):
    dirs, contract = bundles
    mutate(dirs[2], 'predictions.json.gz', lambda p: p['utterances'][0]['binary_scores'].__setitem__(0, 0.101))
    assert compare(*dirs[1:], contract_path=contract)['status'] == 'PASS'


@pytest.mark.parametrize('field', ['prediction_id', 'checkpoint_sha256', 'speaker'])
def test_identity_mismatch_refused(bundles, field):
    dirs, contract = bundles
    mutate(dirs[1], 'predictions.json.gz', lambda p: p['utterances'][0].__setitem__(field, 'wrong'))
    with pytest.raises(ValueError):
        compare(*dirs[:2], contract_path=contract)


def test_semantic_mismatch_refused(bundles):
    dirs, contract = bundles
    for name in ('environment.json', 'metrics.json', 'predictions.json.gz'):
        mutate(dirs[1], name, lambda p: p['metadata'].__setitem__('evaluator_semantics_sha256', 'c' * 64))
    with pytest.raises(ValueError, match='semantics'):
        compare(*dirs[:2], contract_path=contract)


def test_hash_tamper_refused(bundles):
    dirs, contract = bundles
    with (dirs[1] / 'metrics.json').open('a') as f:
        f.write(' ')
    with pytest.raises(ValueError, match='hash'):
        compare(*dirs[:2], contract_path=contract)


def test_null_quality_is_incomplete_not_pass(bundles):
    dirs, contract = bundles
    mutate(dirs[1], 'metrics.json', lambda p: p['quality'].__setitem__('ap', None))
    assert compare(*dirs[:2], contract_path=contract)['status'] == 'INCOMPLETE'


def test_missing_metric_refused(bundles):
    dirs, contract = bundles
    mutate(dirs[1], 'metrics.json', lambda p: p['quality'].pop('per'))
    with pytest.raises(KeyError):
        compare(*dirs[:2], contract_path=contract)


def test_nonfinite_refused(bundles):
    dirs, contract = bundles
    mutate(dirs[1], 'metrics.json', lambda p: p['quality'].__setitem__('ap', float('nan')))
    with pytest.raises(ValueError, match='Nonfinite'):
        compare(*dirs[:2], contract_path=contract)


def test_cli_blocked_records_and_existing_outdir_refused(bundles, tmp_path):
    dirs, contract = bundles
    out = tmp_path / 'blocked'
    assert main(['--reference', str(dirs[0]), '--candidate', str(tmp_path / 'absent'),
                 '--contract', str(contract), '--outdir', str(out)]) == 1
    assert json.loads((out / 'comparison.json').read_text())['status'] == 'BLOCKED'
    with pytest.raises(SystemExit):
        main(['--reference', str(dirs[0]), '--candidate', str(dirs[1]), '--outdir', str(out)])


def test_reusing_output_as_both_platforms_refused(bundles):
    dirs, contract = bundles
    with pytest.raises(ValueError, match='Independent'):
        compare(dirs[0], dirs[0], contract_path=contract)


def test_threshold_override_is_explicit_and_hashed(bundles, tmp_path):
    dirs, contract = bundles
    path = Path(__file__).resolve().parents[1] / 'configs/bench_parity_thresholds.json'
    limits = json.loads(path.read_text())
    mutate(dirs[1], 'metrics.json', lambda p: p['quality'].__setitem__('ap', 0.5015))
    assert compare(*dirs[:2], contract_path=contract)['status'] == 'FAIL'
    limits['LOCAL_FP32_vs_KAGGLE_FP32']['quality_atol'] = 0.002
    custom = tmp_path / 'custom.json'
    dump(custom, limits)
    result = compare(*dirs[:2], custom, contract)
    assert result['status'] == 'PASS'
    assert result['thresholds_sha256'] == digest(custom)
