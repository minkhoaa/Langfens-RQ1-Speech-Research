"""Version identity is necessary, never sufficient for remote readiness."""
import json
from importlib import metadata

from langfens_bench.dependencies import validate_dependency_contract


def test_measured_local_stack_matches_registered_versions():
    result = validate_dependency_contract()
    assert result['status'] == 'VERIFIED', result['blockers']
    assert result['cuda_wheel_availability'].startswith('UNVERIFIED')


def test_version_drift_and_missing_package_fail_closed(tmp_path, monkeypatch):
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps({'python': '0.0.0', 'versions': {'a': '1.0', 'b': '2.0'}}))
    def version(name):
        if name == 'a':
            return '1.1+cu130'
        raise metadata.PackageNotFoundError(name)
    monkeypatch.setattr(metadata, 'version', version)
    result = validate_dependency_contract(path)
    assert result['status'] == 'BLOCKED'
    assert len(result['blockers']) == 3


def test_cpu_cuda_build_suffix_recorded_but_not_version_drift(tmp_path, monkeypatch):
    import platform
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps({'python': platform.python_version(), 'versions': {'torch': '1.0'}}))
    monkeypatch.setattr(metadata, 'version', lambda _: '1.0+cpu')
    assert validate_dependency_contract(path)['status'] == 'VERIFIED'
    monkeypatch.setattr(metadata, 'version', lambda _: '1.0+cu130')
    assert validate_dependency_contract(path)['status'] == 'VERIFIED'
