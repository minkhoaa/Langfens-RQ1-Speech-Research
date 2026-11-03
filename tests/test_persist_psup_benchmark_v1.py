"""Persistence helper dry tests; no network calls or authentication."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'persist_v1', Path(__file__).parents[1] / 'audit_scripts/persist_psup_benchmark_v1.py')
persist = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(persist)


def test_raw_token_predictions_are_allowed(tmp_path):
    raw = tmp_path/'offline_token_evaluation.jsonl'
    raw.write_text('{}\n')
    files = persist.inventory(tmp_path)
    assert files == [{'path': raw.name, 'sha256': persist.sha256(raw), 'bytes': 3}]


def test_compressed_token_predictions_are_allowed(tmp_path):
    (tmp_path/'offline_token_evaluation.jsonl.gz').write_bytes(b'compressed')
    assert len(persist.inventory(tmp_path)) == 1


def test_arbitrary_gzip_is_rejected(tmp_path):
    (tmp_path/'archive.tar.gz').write_bytes(b'archive')
    with pytest.raises(ValueError, match='Non-benchmark'):
        persist.inventory(tmp_path)


def test_weight_upload_is_rejected(tmp_path):
    (tmp_path/'best.pt').write_bytes(b'weights')
    with pytest.raises(ValueError, match='Non-benchmark'):
        persist.inventory(tmp_path)


def test_credentials_upload_is_rejected(tmp_path):
    (tmp_path/'credential.json').write_text('{}')
    with pytest.raises(ValueError, match='Non-benchmark'):
        persist.inventory(tmp_path)


def test_symlink_is_rejected(tmp_path):
    (tmp_path/'target.json').write_text('{}')
    (tmp_path/'link.json').symlink_to(tmp_path/'target.json')
    with pytest.raises(ValueError, match='Symlink'):
        persist.inventory(tmp_path)


def test_empty_directory_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='Empty'):
        persist.inventory(tmp_path)
