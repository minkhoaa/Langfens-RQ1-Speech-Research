"""Synthetic mount inventories: source/cache allowed only by an external hash pin."""
import hashlib
import json
from pathlib import Path

import pytest

from langfens_bench.staging import AUXILIARY, audit_approved_inputs, sha


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / 'input'
    artifacts = root / 'fold1'
    spec = {'fold': 1, 'checkpoint': {'path': 'checkpoints/fold1.pt'}, 'package': {'path': 'datasets/fold1.zip'}}
    for key in ('checkpoint', 'package'):
        path = artifacts / spec[key]['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'synthetic')
        spec[key]['sha256'] = sha(path)
    contract = {'folds': [spec]}
    aux = {}
    for name in AUXILIARY:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'synthetic auxiliary')
        aux[name] = sha(path)
    manifest = {'schema_version': 1, 'fold': 1,
                'contract_identity_sha256': hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                'auxiliary': aux}
    inventory = root / 'approved-inventory.json'
    inventory.write_text(json.dumps(manifest))
    return contract, artifacts, inventory, root


def test_only_selected_fold_source_cache_and_inventory_accepted(inputs):
    contract, artifacts, inventory, root = inputs
    report = audit_approved_inputs(contract, artifacts, 1, inventory, sha(inventory), root)
    assert report['status'] == 'VERIFIED' and report['files'] == 6
    assert report['payload_reads'] == 0


def test_binary_zip_transport_preserves_contract_hash_and_scope(inputs):
    from langfens_bench.runtime import verify_hash
    contract, artifacts, inventory, root = inputs
    entry = contract['folds'][0]['package']
    original = artifacts / entry['path']
    transport = original.with_name(original.name + '.bin')
    original.rename(transport)
    assert verify_hash(artifacts, entry) == transport
    assert audit_approved_inputs(contract, artifacts, 1, inventory, sha(inventory), root)['status'] == 'VERIFIED'


@pytest.mark.parametrize('kind', ['extra', 'symlink', 'cache_tamper', 'wrong_approval', 'wrong_fold', 'wrong_contract'])
def test_inventory_fails_closed_on_unknown_inputs_or_drift(inputs, kind):
    contract, artifacts, inventory, root = inputs
    approval = sha(inventory)
    fold = 1
    if kind == 'extra':
        (root / 'sealed-test.wav').write_bytes(b'never read')
    elif kind == 'symlink':
        (root / 'alias').symlink_to(inventory)
    elif kind == 'cache_tamper':
        (root / sorted(AUXILIARY)[0]).write_bytes(b'tampered')
    elif kind == 'wrong_approval':
        approval = '0' * 64
    elif kind == 'wrong_fold':
        fold = 2
    elif kind == 'wrong_contract':
        contract = {**contract, 'seed': 999}
    with pytest.raises(ValueError):
        audit_approved_inputs(contract, artifacts, fold, inventory, approval, root)
