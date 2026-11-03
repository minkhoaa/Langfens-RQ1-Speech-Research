"""Synthetic JSON/tensor/metadata fixtures only, no services or artifact loading."""
from copy import deepcopy
from pathlib import Path
import importlib.util
import json

import pytest
import torch

from langfens_prep.p_sup.runtime import (BASE_SHA, CONTRACT_SHA256, MANIFEST_FIELDS,
    operation_runtime_rows, validate_complete_run, validate_run_manifest, validate_runtime_row)


def rows(objective='operation_ce', logits=None):
    return operation_runtime_rows(logits=torch.zeros(2, 3) if logits is None else logits,
                                  prompt_phone_ids=[1, 1], run_id='synthetic', condition='matched',
                                  speaker_id='SYNTHETIC', utterance_id='fixture', objective=objective)


def test_treatment_control_full_precision_null_subtype_and_repeated_keys():
    treatment = rows()
    assert [r['prompt_index'] for r in treatment] == [0, 1]
    assert treatment[0]['scores']['s_error'] == pytest.approx(2 / 3)
    assert treatment[0]['scores']['s_omit'] == pytest.approx(1 / 3)
    assert treatment[0]['confidence'] is None and treatment[0]['threshold'] is None
    control = rows('binary_collapsed')
    assert control[0]['scores']['s_error'] == treatment[0]['scores']['s_error']
    assert control[0]['scores']['s_sub'] is None and control[0]['scores']['s_omit'] is None
    assert control[0]['probabilities'] is None and 'unidentified' in control[0]['score_semantics']
    result = validate_complete_run(treatment, run_id='synthetic', condition='matched',
                                  expected_prompts={('SYNTHETIC', 'fixture'): [1, 1]})
    assert result == {'utterances': 1, 'prompt_positions': 2, 'ok': 2, 'abstain': 0, 'failure': 0, 'utterance_failures': 0}


def test_nonfinite_missing_output_and_empty_prompt_accounted():
    output = rows(logits=torch.tensor([[float('nan'), 0., 0.], [0., float('inf'), 0.]]))
    assert all(r['status'] == 'failure' and not r['finite'] for r in output)
    json.dumps(output, allow_nan=False)
    missing = operation_runtime_rows(logits=None, prompt_phone_ids=[1], run_id='synthetic', condition='matched',
                                    speaker_id='SYNTHETIC', utterance_id='fixture', objective='operation_ce')
    assert missing[0]['abstention_reason'] == 'missing_output'
    expected = {('SYNTHETIC', 'fixture'): [1, 1], ('SYNTHETIC', 'empty'): []}
    with pytest.raises(ValueError, match='incomplete'):
        validate_complete_run(output, run_id='synthetic', condition='matched', expected_prompts=expected)
    result = validate_complete_run(output, run_id='synthetic', condition='matched', expected_prompts=expected,
                                  utterance_failures=[{'speaker_id': 'SYNTHETIC', 'utterance_id': 'empty', 'reason': 'OOV'}])
    assert result['failure'] == 2 and result['utterance_failures'] == 1


@pytest.mark.parametrize('mode', ['duplicate', 'missing', 'extra', 'phone', 'condition'])
def test_accounting_rejects_incomplete_or_duplicate(mode):
    output = rows()
    if mode == 'duplicate': output.append(output[0])
    if mode == 'missing': output.pop()
    if mode == 'extra': output[1]['prompt_index'] = 2
    if mode == 'phone': output[1]['prompt_phone_id'] = 2
    if mode == 'condition': output[1]['condition'] = 'mismatched'
    with pytest.raises(ValueError):
        validate_complete_run(output, run_id='synthetic', condition='matched',
                              expected_prompts={('SYNTHETIC', 'fixture'): [1, 1]})


@pytest.mark.parametrize('key', ['gold_canonical_index', 'perceived_sequence', 'TextGrid', 'operation_labels', 'gold_times', 'xmin', 'xmax', 'binary_label'])
def test_runtime_rejects_gold_top_level_and_nested(key):
    row = rows()[0]; row[key] = 0
    with pytest.raises(ValueError): validate_runtime_row(row)
    row = rows()[0]; row['alignment'] = {key: []}
    with pytest.raises(ValueError): validate_runtime_row(row)


@pytest.mark.parametrize('field,value', [('speaker_id', 'ABA'), ('prompt_phone_id', 0), ('prompt_index', True),
    ('threshold', 0.5), ('status', 'success'), ('word_index', -1), ('finite', False),
    ('probabilities', {'correct': 0.5, 'substitution': 0.5, 'deletion': 0.5}),
    ('probabilities', {'correct': 1.0, 'substitution': 0.0, 'deletion': 0.0}),
    ('probabilities', {'error': 'high'}),
    ('confidence', {'value': float('inf')}), ('scores', {'s_error': float('nan'), 's_sub': None, 's_omit': None})])
def test_invalid_runtime_values(field, value):
    row = rows()[0]; row[field] = value
    with pytest.raises((ValueError, TypeError)): validate_runtime_row(row)


def synthetic_manifest():
    # Fake digest fixtures are only validator tests, never evidence of a model/artifact.
    m = {key: 'synthetic' for key in MANIFEST_FIELDS}
    m.update(schema_version='rq1-runtime-v1', contract_version='V1', contract_sha256=CONTRACT_SHA256,
             git_base=BASE_SHA, git_tip='a' * 40, git_dirty=False, arm='P-SUP',
             source_hashes={'synthetic.py': 'a' * 64}, output_hashes={'synthetic.jsonl': 'b' * 64},
             fit_speakers=['FIT'], selection_speakers=['SELECT'], calibration_speakers=['CAL'], evaluation_speakers=['VAL'],
             successful_updates=0, skipped_updates=0, endpoint='synthetic_only', calibration_identity=None)
    for field in ('config_sha256', 'package_sha256', 'label_map_sha256', 'prompt_sha256', 'g2p_sha256', 'weight_sha256', 'allowed_id_sha256'):
        m[field] = 'c' * 64
    return m


def test_manifest_complete_fields_and_no_fabricated_fixed_budget():
    m = synthetic_manifest(); assert validate_run_manifest(m) is m
    for field in MANIFEST_FIELDS:
        broken = deepcopy(m); del broken[field]
        with pytest.raises(ValueError): validate_run_manifest(broken)
    m['endpoint'] = 'fixed_budget_3600'; m['successful_updates'] = 3598
    with pytest.raises(ValueError, match='false fixed'): validate_run_manifest(m)
    m['successful_updates'] = 3600; validate_run_manifest(m)
    m['successful_updates'] = 3601
    with pytest.raises(ValueError, match='budget'): validate_run_manifest(m)


@pytest.mark.parametrize('field,value', [('weight_sha256', None), ('git_dirty', 'false'), ('fit_speakers', ['ABA']),
    ('evaluation_speakers', ['FIT']), ('source_hashes', {}), ('precision', ''), ('successful_updates', -1)])
def test_manifest_invalid_provenance(field, value):
    m = synthetic_manifest(); m[field] = value
    with pytest.raises(ValueError): validate_run_manifest(m)


def test_offline_preflight_hold_not_training_readiness():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('p_sup_preflight', root / 'audit_scripts/p_sup_preflight.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    result = module.preflight(root)
    assert result['status'] == 'PREPARATION_AUDITED_EXPERIMENT_HOLD'
    assert result['attempts'] == result['successful_updates'] == result['skipped_updates'] == 0
    assert result['new_model_artifact'] is None and not result['training_entrypoint_available']
    metadata = json.loads((root / 'deploy/kaggle_p_sup_preflight/kernel-metadata.json').read_text())
    assert metadata['is_private'] is True and metadata['machine_shape'] == 'NvidiaTeslaT4'


@pytest.mark.parametrize('field,value', [('execute_training', True), ('warmstart', 'archived-E1'),
    ('fit_speakers', ['NJS']), ('max_successful_updates', 3598), ('seed', 42),
    ('precision', 'cpu_fp32'), ('head', {'width': 256}), ('task_weight', 2.0)])
def test_preflight_rejects_registration_drift(monkeypatch, field, value):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('p_sup_preflight_negative', root / 'audit_scripts/p_sup_preflight.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    original_read = Path.read_text
    config_path = root / 'configs/rq1_p_sup_v1.json'
    config = json.loads(original_read(config_path)); config[field] = value
    def read_text(path, *args, **kwargs):
        return json.dumps(config) if path == config_path else original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read_text)
    with pytest.raises(ValueError):
        module.preflight(root)


@pytest.mark.parametrize('mutation', ['internet', 'public', 'source', 'extra_file', 'remote_import'])
def test_preflight_package_fails_closed(monkeypatch, tmp_path, mutation):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('p_sup_preflight_package_negative', root / 'audit_scripts/p_sup_preflight.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    package = root / 'deploy/kaggle_p_sup_preflight'
    metadata_path = package / 'kernel-metadata.json'
    original_metadata = metadata_path.read_text()
    original_code = (package / 'preflight.py').read_text()
    try:
        metadata = json.loads(original_metadata)
        if mutation == 'internet': metadata['enable_internet'] = True
        elif mutation == 'public': metadata['is_private'] = False
        elif mutation == 'source': metadata['dataset_sources'] = ['unapproved/data']
        elif mutation == 'remote_import': (package / 'preflight.py').write_text(original_code + '\\nimport kaggle\\n')
        elif mutation == 'extra_file': (package / 'unexpected.bin').write_bytes(b'fixture')
        metadata_path.write_text(json.dumps(metadata))
        with pytest.raises(ValueError): module.preflight(root)
    finally:
        metadata_path.write_text(original_metadata)
        (package / 'preflight.py').write_text(original_code)
        (package / 'unexpected.bin').unlink(missing_ok=True)


def test_bad_output_shape_still_accounts_for_every_valid_prompt():
    output = rows(logits=torch.zeros(1, 3))
    assert len(output) == 2
    assert all(r['status'] == 'failure' and r['abstention_reason'] == 'output_shape_mismatch' for r in output)
    result = validate_complete_run(output, run_id='synthetic', condition='matched',
                                  expected_prompts={('SYNTHETIC', 'fixture'): [1, 1]})
    assert result['failure'] == 2
