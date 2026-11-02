"""Explicit SHA-pinned mounted-input inventory; no extraction or remote operations."""
import argparse
import hashlib
import json
import os
from pathlib import Path

REV = '22aad52d435eb6dbaf354bdad9b0da84ce7d6156'
CACHE = f'langfens-bench-cache/hub/models--facebook--wav2vec2-base-960h/snapshots/{REV}/'
SOURCE = 'langfens-bench-source/langfens-bench-source.bin'
AUXILIARY = {SOURCE, CACHE + 'config.json', CACHE + 'model.safetensors'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def audit_approved_inputs(contract, artifact_root, selected_fold, inventory_path, inventory_sha256, input_root):
    root, inventory = Path(input_root).resolve(), Path(inventory_path)
    if not inventory_sha256 or inventory.is_symlink() or sha(inventory) != inventory_sha256:
        raise ValueError('Approved input inventory SHA-256 mismatch or missing external approval digest')
    spec = json.loads(inventory.read_text())
    raw = json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()
    if (spec.get('schema_version') != 1 or spec['fold'] != selected_fold or spec['contract_identity_sha256'] != hashlib.sha256(raw).hexdigest()
            or set(spec['auxiliary']) != AUXILIARY):
        raise ValueError('Input inventory scope/contract/auxiliary allowlist mismatch')
    selected = next(f for f in contract['folds'] if f['fold'] == selected_fold)
    from .runtime import artifact_path
    expected = {artifact_path(Path(artifact_root), selected[k]): selected[k]['sha256']
                for k in ('checkpoint', 'package')}
    expected.update({root / name: digest for name, digest in spec['auxiliary'].items()})
    # Inventory itself is authorized by its EXTERNAL digest, avoiding self-hash cycles.
    inventory_resolved = inventory.resolve()
    if inventory_resolved.is_relative_to(root):
        expected[inventory_resolved] = inventory_sha256
    if any(not p.is_relative_to(root) for p in expected):
        raise ValueError('Approved mounted input escapes immutable root')
    seen, count = set(), 0
    def fail(error):
        raise error
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=fail):
        for name in dirs + files:
            p = Path(directory) / name
            count += 1
            if count > 10000 or p.is_symlink():
                raise ValueError('Unknown symlink or excessive mounted input inventory')
        for name in files:
            p = (Path(directory) / name).resolve()
            if p not in expected or not p.is_file():
                raise ValueError(f'Unauthorized mounted input: {p}')
            seen.add(p)
    if seen != set(expected):
        raise ValueError('Missing approved mounted input')
    # Inventory is checked in full before any extra-file bytes are read. Artifact
    # hashes follow in normal preflight; only known source/cache hashes read here.
    for name, expected_sha in spec['auxiliary'].items():
        if sha(root / name) != expected_sha:
            raise ValueError(f'Approved source/cache SHA-256 mismatch: {name}')
    return {'status': 'VERIFIED', 'selected_fold': selected_fold, 'files': len(seen),
            'inventory_sha256': inventory_sha256, 'payload_reads': 0,
            'note': 'Only selected fold artifacts plus externally pinned source ZIP/cache/inventory'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold', type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument('--source-archive', type=Path, required=True)
    parser.add_argument('--cache-snapshot', type=Path, required=True)
    parser.add_argument('--contract', type=Path, default=Path(__file__).resolve().parents[2] / 'manifests/RQ1-BENCH-01/contract.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    contract = json.loads(args.contract.read_text())
    raw = json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()
    spec = {'schema_version': 1, 'fold': args.fold,
            'contract_identity_sha256': hashlib.sha256(raw).hexdigest(),
            'auxiliary': {SOURCE: sha(args.source_archive),
                          CACHE + 'config.json': sha(args.cache_snapshot / 'config.json'),
                          CACHE + 'model.safetensors': sha(args.cache_snapshot / 'model.safetensors')}}
    with args.output.open('x') as f:
        json.dump(spec, f, indent=2)
        f.write('\n')
    print(json.dumps({'path': str(args.output.resolve()), 'sha256': sha(args.output),
                      'source_sha256': spec['auxiliary'][SOURCE]}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
