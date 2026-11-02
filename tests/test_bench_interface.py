"""Public command routing must not execute hardware, staging or model logic."""
import importlib

import pytest

from langfens_bench.__main__ import main


@pytest.mark.parametrize('command,module', [('preflight', 'runtime'), ('evaluate', 'runtime'),
                                           ('merge', 'merge'), ('compare', 'compare')])
def test_unified_dispatch(monkeypatch, command, module):
    target = importlib.import_module('langfens_bench.' + module)
    calls = []
    monkeypatch.setattr(target, 'main', lambda argv: calls.append(argv) or 0)
    assert main([command, '--help']) == 0
    assert calls == [([command, '--help'] if module == 'runtime' else ['--help'])]


def test_unified_help_lists_all_commands(capsys):
    assert main(['--help']) == 0
    assert '{preflight,evaluate,merge,compare}' in capsys.readouterr().out
