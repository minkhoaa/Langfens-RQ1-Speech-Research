"""Fail-closed comparison of completed inference runs; never launches evaluation."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
QUALITY = (
    'ap', 'roc_auc', 'per', 'omission_recall_0.2', 'omission_fnr_0.2',
    'omission_recall_0.5', 'omission_fnr_0.5', 'attention_mass',
    'attention_entropy', 'attention_peak_offset_seconds',
)
PERFORMANCE = (
    'latency_p50_seconds', 'latency_p95_seconds', 'throughput_utterances_per_second',
    'peak_ram_bytes', 'peak_vram_allocated_bytes', 'peak_vram_reserved_bytes',
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_json(path, compressed=False):
    opener = gzip.open if compressed else open
    with opener(path, 'rt', encoding='utf-8') as f:
        return json.load(f, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(f'Nonfinite JSON: {s}')))


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Expected finite numeric value, got {value!r}')
    return float(value)


def _bundle(directory, contract, contract_sha, scope_fold=None):
    directory = Path(directory)
    env = load_json(directory / 'environment.json')
    metrics = load_json(directory / 'metrics.json')
    pred = load_json(directory / 'predictions.json.gz', compressed=True)
    meta = pred['metadata']
    expected_weights = {str(f['fold']): f['checkpoint']['sha256'] for f in contract['folds']}
    if meta['status'] != 'COMPLETED' or meta['contract_sha256'] != contract_sha:
        raise ValueError('Incomplete run or mismatched contract')
    if meta['seed'] != contract['seed'] or meta['checkpoint_sha256'] != expected_weights:
        raise ValueError('Seed/checkpoint identity mismatch')
    semantic = meta['evaluator_semantics_sha256']
    if not isinstance(semantic, str) or len(semantic) != 64 or any(c not in '0123456789abcdef' for c in semantic):
        raise ValueError('Invalid evaluator semantic digest')
    for doc in (env, metrics):
        other = doc['metadata']
        for key in ('status', 'contract_sha256', 'seed', 'checkpoint_sha256', 'profile', 'evaluator_semantics_sha256'):
            if other[key] != meta[key]:
                raise ValueError(f'Inconsistent bundle metadata: {key}')
    for name in ('metrics.json', 'predictions.json.gz'):
        if env['artifacts'][name] != digest(directory / name):
            raise ValueError(f'Output hash mismatch: {name}')
    rows = pred['utterances']
    ids = [r['prediction_id'] for r in rows]
    expected_rows = [r for r in contract['utterances'] if scope_fold is None or r['fold'] == scope_fold]
    expected = [r['prediction_id'] for r in expected_rows]
    if ids != expected or len(ids) != len(set(ids)):
        raise ValueError('Prediction IDs/order/coverage mismatch')
    for row, frozen in zip(rows, expected_rows):
        for key in ('fold', 'speaker', 'utterance_id'):
            if row[key] != frozen[key]:
                raise ValueError(f'Record identity mismatch: {key}')
        if row['checkpoint_sha256'] != expected_weights[str(row['fold'])]:
            raise ValueError('Record checkpoint mismatch')
        n = len(row['labels'])
        for key in ('binary_scores', 'binary_logits', 'error_types', 'canonical_phones'):
            if len(row[key]) != n:
                raise ValueError(f'Active-token length mismatch: {key}')
        if not all(x in (0, 1) for x in row['labels']):
            raise ValueError('Invalid binary labels')
        if any(finite(s) < 0 or finite(s) > 1 for s in row['binary_scores']):
            raise ValueError('Invalid probability')
    return env, metrics, pred


def compare(reference, candidate, thresholds_path=None, contract_path=None):
    """Validate immutable bundle identity, then apply registered pointwise + metric gates."""
    if Path(reference).resolve() == Path(candidate).resolve():
        raise ValueError('Independent output paths required')
    contract_path = Path(contract_path or REPO / 'manifests/RQ1-BENCH-01/contract.json')
    thresholds_path = Path(thresholds_path or REPO / 'configs/bench_parity_thresholds.json')
    contract, contract_sha = load_json(contract_path), digest(contract_path)
    thresholds = load_json(thresholds_path)
    re, rm, rp = _bundle(reference, contract, contract_sha)
    ce, cm, cp = _bundle(candidate, contract, contract_sha)
    rmeta, cmeta = rp['metadata'], cp['metadata']
    if rmeta['evaluator_semantics_sha256'] != cmeta['evaluator_semantics_sha256']:
        raise ValueError('Evaluator semantics mismatch')
    contrast = rmeta['profile'] + '_vs_' + cmeta['profile']
    if contrast not in ('LOCAL_FP32_vs_KAGGLE_FP32', 'KAGGLE_FP32_vs_KAGGLE_FP16'):
        raise ValueError(f'Unsupported platform contrast: {contrast}')
    limits = thresholds[contrast]
    for value in limits.values():
        if finite(value) < 0:
            raise ValueError('Thresholds must be nonnegative')
    checks, undefined = {}, []

    def gate(name, a, b, limit):
        if a is None or b is None:
            undefined.append(name)
            return
        delta = abs(finite(b) - finite(a))
        checks[name] = {'reference': a, 'candidate': b, 'absolute_delta': delta,
                        'atol': limit, 'pass': delta <= limit}

    def vector_gate(name, a, b, limit):
        if len(a) != len(b) or not a:
            raise ValueError(f'Missing or unequal vector: {name}')
        diffs = [abs(finite(x) - finite(y)) for x, y in zip(a, b)]
        checks[name] = {'max_absolute_delta': max(diffs), 'mean_absolute_delta': sum(diffs) / len(diffs),
                        'n': len(diffs), 'atol': limit, 'pass': max(diffs) <= limit}

    ctc_equal = 0
    for r, c in zip(rp['utterances'], cp['utterances']):
        uid = r['prediction_id']
        for field in ('labels', 'error_types', 'canonical_phones', 'ctc_reference_ids'):
            if r[field] != c[field]:
                raise ValueError(f'Offline reference semantics differ: {uid}/{field}')
        vector_gate(uid + '/binary_scores', r['binary_scores'], c['binary_scores'], limits['probability_atol'])
        vector_gate(uid + '/binary_logits', r['binary_logits'], c['binary_logits'], limits['logit_atol'])
        ctc_equal += r['ctc_ids'] == c['ctc_ids']
        for field in ('in_window_mass', 'normalized_entropy', 'absolute_peak_offset_seconds'):
            limit = limits['attention_peak_offset_seconds_atol'] if field == 'absolute_peak_offset_seconds' else limits['attention_atol']
            gate(uid + '/attention/' + field, r['attention_metrics'][field], c['attention_metrics'][field], limit)
            if 'tokens' in r['attention_metrics'] or 'tokens' in c['attention_metrics']:
                vector_gate(uid + '/attention/tokens/' + field,
                            [t[field] for t in r['attention_metrics']['tokens']],
                            [t[field] for t in c['attention_metrics']['tokens']], limit)
        rz, cz = r['protocol_z'], c['protocol_z']
        for field in ('canonical_phones', 'aligned_pairs', 'unaligned_gold_count', 'unaligned_g2p_count', 'tie_break'):
            if rz[field] != cz[field]:
                raise ValueError(f'Protocol Z alignment semantics differ: {uid}/{field}')
        vector_gate(uid + '/protocol_z/scores', rz['binary_scores'], cz['binary_scores'], limits['probability_atol'])
        vector_gate(uid + '/protocol_z/logits', rz['binary_logits'], cz['binary_logits'], limits['logit_atol'])
    for key in QUALITY:
        limit = (limits['attention_peak_offset_seconds_atol'] if key == 'attention_peak_offset_seconds'
                 else limits['attention_atol'] if key.startswith('attention_') else limits['quality_atol'])
        gate('quality/' + key, rm['quality'][key], cm['quality'][key], limit)
    for key in ('ap', 'roc_auc', 'coverage'):
        gate('quality/protocol_z/' + key, rm['quality']['protocol_z'][key], cm['quality']['protocol_z'][key], limits['quality_atol'])
    for fold in (str(f['fold']) for f in contract['folds']):
        rq, cq = rm['by_fold'][fold]['quality'], cm['by_fold'][fold]['quality']
        for key in QUALITY:
            limit = (limits['attention_peak_offset_seconds_atol'] if key == 'attention_peak_offset_seconds'
                     else limits['attention_atol'] if key.startswith('attention_') else limits['quality_atol'])
            gate('fold/' + fold + '/' + key, rq[key], cq[key], limit)
        for key in ('ap', 'roc_auc', 'coverage'):
            gate('fold/' + fold + '/protocol_z/' + key, rq['protocol_z'][key], cq['protocol_z'][key], limits['quality_atol'])
    performance = {}
    for key in PERFORMANCE:
        a, b = rm['performance'][key], cm['performance'][key]
        for value in (a, b):
            if value is not None and finite(value) < 0:
                raise ValueError('Negative performance measurement')
        # CPU VRAM is explicitly null, not an invented GPU observation.
        if key in PERFORMANCE[:4] and (a is None or b is None):
            raise ValueError(f'Missing performance measurement: {key}')
        performance[key] = {'reference': a, 'candidate': b,
                            'candidate_over_reference': b / a if a and b is not None else None}
    failed = [name for name, check in checks.items() if not check['pass']]
    return {'schema_version': 1, 'status': 'FAIL' if failed else 'INCOMPLETE' if undefined else 'PASS',
            'contrast': contrast, 'contract_sha256': contract_sha, 'thresholds_sha256': digest(thresholds_path),
            'thresholds': limits, 'evaluator_semantics_sha256': rmeta['evaluator_semantics_sha256'],
            'reference': str(Path(reference).resolve()), 'candidate': str(Path(candidate).resolve()),
            'prediction_ids': [r['prediction_id'] for r in rp['utterances']],
            'ctc_exact_utterances': ctc_equal, 'utterances': len(rp['utterances']),
            'checks': checks, 'failed_checks': failed, 'undefined_checks': undefined,
            'performance': performance,
            'limitations': ['Comparison is only as valid as the provenance of executed input bundles.',
                            'No threshold relaxation or historical archive-parity waiver.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--thresholds', default=str(REPO / 'configs/bench_parity_thresholds.json'))
    parser.add_argument('--contract', default=str(REPO / 'manifests/RQ1-BENCH-01/contract.json'))
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args(argv)
    out = Path(args.outdir)
    if out.exists():
        parser.error('Output path exists; use an independent new directory')
    try:
        report = compare(args.reference, args.candidate, args.thresholds, args.contract)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report = {'status': 'BLOCKED', 'error': str(exc)}
    out.mkdir(parents=True, exist_ok=False)
    (out / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(report['status'])
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
