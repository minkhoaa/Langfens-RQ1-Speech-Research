"""Measured dependency identity gate, separate from imports/hardware verification."""
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform

DEFAULT_CONTRACT = Path(__file__).resolve().parents[2] / 'configs/bench_dependency_contract.json'


def validate_dependency_contract(path=DEFAULT_CONTRACT):
    path = Path(path)
    raw = path.read_bytes()
    spec = json.loads(raw)
    actual_python = platform.python_version()
    versions, blockers = {}, []
    if actual_python != spec['python']:
        blockers.append(f"Python {actual_python} != registered {spec['python']}")
    for package, expected in spec['versions'].items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            actual = None
        versions[package] = actual
        if actual is None or actual.split('+', 1)[0] != expected:
            blockers.append(f'{package} {actual!r} != registered {expected}')
    return {'status': 'BLOCKED' if blockers else 'VERIFIED',
            'contract_sha256': hashlib.sha256(raw).hexdigest(), 'python': actual_python,
            'versions': versions, 'blockers': blockers,
            'cuda_wheel_availability': 'UNVERIFIED; must pass actual remote import/hardware preflight'}
