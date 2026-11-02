"""Merge four isolated fold runs without ever co-staging cross-fold audio."""
import argparse
import copy
import json
from pathlib import Path

from .compare import REPO, PERFORMANCE, _bundle, digest, finite, load_json


def merge(inputs, output, contract_path=None):
    from .runtime import compute_quality, write_json, write_predictions
    import numpy as np
    contract_path = Path(contract_path or REPO / 'manifests/RQ1-BENCH-01/contract.json')
    contract = load_json(contract_path)
    expected_folds = {f['fold'] for f in contract['folds']}
    if len(inputs) != len(expected_folds) or len({Path(p).resolve() for p in inputs}) != len(inputs):
        raise ValueError('Exactly four distinct isolated-fold bundles required')
    out = Path(output)
    if out.exists():
        raise ValueError('Independent new output directory required')
    bundles, seen = [], set()
    for directory in inputs:
        metadata = load_json(Path(directory) / 'environment.json')['metadata']
        fold = metadata['scope_fold']
        if fold not in expected_folds or fold in seen:
            raise ValueError('Duplicate, missing, or invalid fold scope')
        env, metrics, pred = _bundle(directory, contract, digest(contract_path), scope_fold=fold)
        bundles.append((fold, env, metrics, pred, directory))
        seen.add(fold)
    bundles.sort(key=lambda b: b[0])
    meta = copy.deepcopy(bundles[0][3]['metadata'])
    for _, env, metrics, pred, _ in bundles:
        for key in ('contract_sha256', 'evaluator_semantics_sha256', 'checkpoint_sha256', 'profile', 'seed'):
            if pred['metadata'][key] != meta[key]:
                raise ValueError(f'Incompatible fold-run provenance: {key}')
    lookup = {r['prediction_id']: r for _, _, _, pred, _ in bundles for r in pred['utterances']}
    records = [lookup[r['prediction_id']] for r in contract['utterances']]
    latencies = [finite(r['latency_seconds']) for r in records]
    if not latencies or min(latencies) <= 0:
        raise ValueError('Positive per-utterance forward latency required')
    meta['scope_fold'] = None
    meta['scope'] = 'full'
    meta['fold'] = None
    meta['ordered_prediction_ids'] = [r['prediction_id'] for r in records]
    meta['status'] = 'COMPLETED'
    meta['merged_from_isolated_fold_runs'] = True
    performance = {'latency_p50_seconds': float(np.quantile(latencies, .5)),
                   'latency_p95_seconds': float(np.quantile(latencies, .95)),
                   'throughput_utterances_per_second': len(latencies) / sum(latencies)}
    for key in PERFORMANCE[3:]:
        values = [m['performance'][key] for _, _, m, _, _ in bundles]
        performance[key] = max(finite(v) for v in values if v is not None) if any(v is not None for v in values) else None
    metrics = {'metadata': meta, 'quality': compute_quality(records, contract['thresholds']),
               'performance': performance,
               'by_fold': {str(f): {'quality': compute_quality([r for r in records if r['fold'] == f], contract['thresholds'])}
                           for f in sorted(expected_folds)},
               'telemetry_note': 'Latency from synchronized O+Z forwards; peaks are max of four separate fold processes, not concurrent residency.'}
    env = {'metadata': meta, 'fold_environments': {str(f): e for f, e, _, _, _ in bundles},
           'input_bundles': {str(f): {'directory': str(Path(d).resolve()), 'environment_sha256': digest(Path(d) / 'environment.json')}
                             for f, _, _, _, d in bundles},
           'artifacts': {}}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'metrics.json', metrics)
    write_predictions(out / 'predictions.json.gz', {'metadata': meta, 'utterances': records})
    env['artifacts'] = {name: digest(out / name) for name in ('metrics.json', 'predictions.json.gz')}
    write_json(out / 'environment.json', env)
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs=4, required=True)
    parser.add_argument('--contract', default=str(REPO / 'manifests/RQ1-BENCH-01/contract.json'))
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args(argv)
    try:
        merge(args.inputs, args.outdir, args.contract)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f'BLOCKED: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
