"""Packaging allowlist must never include inherited data/history or credentials."""
import hashlib
import zipfile

import pytest

from langfens_bench.kaggle import build_source_package

REQUIRED = (
    'LICENSE', 'resources/CMUDict/LICENSE',
    'src/langfens_bench/__init__.py', 'src/langfens_bench/__main__.py',
    'src/langfens_bench/runtime.py', 'src/langfens_bench/kaggle.py',
    'src/langfens_prep/model.py',
    'audit_scripts/evaluate_fold1_ctc_binary.py', 'audit_scripts/diagnose_omissions_and_false_negatives.py',
    'manifests/RQ1-BENCH-01/contract.json', 'manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json',
    'configs/bench_parity_thresholds.json', 'configs/bench_dependency_contract.json',
    'resources/CMUDict/cmudict.dict',
    'resources/L2-ARCTIC/prompts.json',
)


def source_tree(tmp_path):
    root = tmp_path / 'repo'
    for name in REQUIRED:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}' if name.endswith('.json') else '# synthetic\n')
    for name in ('.env', 'checkpoints/best.pt', 'datasets/data.wav',
                 'manifests/RQ1-DIAG-02B/evaluation-final/tokens.csv.gz', 'src/langfens_bench/__pycache__/test.pyc',
                 'audit_scripts/train_fold1_ctc_binary.py'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('MUST NOT BE PACKAGED')
    return root


def test_allowlisted_source_zip_reproducible_and_no_data(tmp_path):
    root = source_tree(tmp_path)
    a, b = tmp_path / 'a.zip', tmp_path / 'b.zip'
    first, second = build_source_package(root, a), build_source_package(root, b)
    assert first == second == hashlib.sha256(a.read_bytes()).hexdigest()
    assert a.read_bytes() == b.read_bytes()
    with zipfile.ZipFile(a) as archive:
        assert 'src/langfens_bench/runtime.py' in archive.namelist()
        assert all('MUST NOT BE PACKAGED' not in archive.read(n).decode() for n in archive.namelist())
        assert 'configs/bench_parity_thresholds.json' in archive.namelist()
    with pytest.raises((ValueError, FileExistsError)):
        build_source_package(root, a)


def test_symlink_source_refused(tmp_path):
    root = source_tree(tmp_path)
    path = root / 'src/langfens_bench/runtime.py'
    path.unlink()
    path.symlink_to(root / '.env')
    with pytest.raises(ValueError, match='unsafe|symlink'):
        build_source_package(root, tmp_path / 'bad.zip')
