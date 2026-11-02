"""Metadata-only fold merge; never stages/opens audio or model files."""
import gzip
import json

import pytest

from langfens_bench.compare import digest, _bundle
from langfens_bench.merge import merge


def dump(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def fold_runs(tmp_path, monkeypatch):
    from langfens_bench import runtime
    monkeypatch.setattr(runtime, 'compute_quality', lambda records, thresholds: {'n': len(records)})
    contract = {'seed': 1337, 'thresholds': [0.2, 0.5],
                'folds': [{'fold': k, 'checkpoint': {'sha256': str(k) * 64}} for k in range(1, 5)],
                'utterances': [{'fold': k, 'speaker': f'S{k}', 'utterance_id': 'u', 'prediction_id': f'{k}/S{k}/u'} for k in range(1, 5)]}
    cp = tmp_path / 'contract.json'
    dump(cp, contract)
    dirs = []
    for u in contract['utterances']:
        k = u['fold']
        directory = tmp_path / str(k)
        directory.mkdir()
        dirs.append(directory)
        meta = {'scope_fold': k, 'status': 'COMPLETED', 'contract_sha256': digest(cp), 'seed': 1337,
                'checkpoint_sha256': {str(f['fold']): f['checkpoint']['sha256'] for f in contract['folds']},
                'profile': 'KAGGLE_FP32', 'evaluator_semantics_sha256': 'a' * 64}
        record = {**u, 'checkpoint_sha256': str(k) * 64, 'latency_seconds': float(k),
                  'labels': [1], 'binary_scores': [0.4], 'binary_logits': [-0.4],
                  'error_types': ['deletion'], 'canonical_phones': ['T'], 'ctc_reference_ids': []}
        dump(directory / 'metrics.json', {'metadata': meta, 'performance': {
            'peak_ram_bytes': k * 100, 'peak_vram_allocated_bytes': k * 50, 'peak_vram_reserved_bytes': k * 60}})
        with gzip.open(directory / 'predictions.json.gz', 'wt') as f:
            json.dump({'metadata': meta, 'utterances': [record]}, f)
        dump(directory / 'environment.json', {'metadata': meta, 'artifacts': {
            n: digest(directory / n) for n in ('metrics.json', 'predictions.json.gz')}})
    return dirs, cp


def test_fold_merge_reorders_records_and_computes_latency_quantiles(fold_runs, tmp_path):
    dirs, cp = fold_runs
    out = tmp_path / 'merged'
    merge(list(reversed(dirs)), out, cp)
    m = json.loads((out / 'metrics.json').read_text())
    assert m['quality'] == {'n': 4}
    assert m['performance']['latency_p50_seconds'] == 2.5
    assert m['performance']['throughput_utterances_per_second'] == 0.4
    assert m['performance']['peak_ram_bytes'] == 400
    assert m['metadata']['scope_fold'] is None
    _bundle(out, json.loads(cp.read_text()), digest(cp))


def test_fold_merge_rejects_duplicate_incomplete_and_overwrite(fold_runs, tmp_path):
    dirs, cp = fold_runs
    for bad in (dirs[:3], [dirs[0]] * 4):
        with pytest.raises(ValueError, match='distinct'):
            merge(bad, tmp_path / 'bad', cp)
    out = tmp_path / 'merged'
    merge(dirs, out, cp)
    with pytest.raises(ValueError, match='new output'):
        merge(dirs, out, cp)
