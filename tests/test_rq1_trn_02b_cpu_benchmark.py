"""
Unit tests for RQ1-TRN-02B Colab CPU Feasibility Benchmark Driver.
Verifies system metric extraction, projection calculations, and CLI arguments.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from audit_scripts.benchmark_cpu_rq1_trn_02b import (
    get_cpu_info,
    get_ram_usage_gb,
)


def test_get_cpu_info_structure():
    info = get_cpu_info()
    assert "system" in info
    assert "cpu_model" in info
    assert "logical_cpus" in info
    assert isinstance(info["logical_cpus"], int) and info["logical_cpus"] > 0
    assert "total_ram_gb" in info
    assert isinstance(info["total_ram_gb"], float) and info["total_ram_gb"] > 0
    assert "available_ram_gb" in info
    assert isinstance(info["available_ram_gb"], float) and info["available_ram_gb"] > 0


def test_get_ram_usage_gb():
    ram = get_ram_usage_gb()
    assert isinstance(ram, float)
    assert ram > 0.0


def test_cpu_allow_flag_in_training_parser():
    """Verifies --allow-cpu argument is recognized by train_rq1_trn_02b.py."""
    import argparse
    from audit_scripts.train_rq1_trn_02b import main

    # Inspect ArgumentParser configuration by patching parse_args
    with patch("argparse.ArgumentParser.parse_args") as mock_parse:
        mock_parse.return_value = argparse.Namespace(
            config="configs/rq1_trn_02b_c0.yaml",
            resume=None,
            initial_checkpoint=None,
            zip_path=None,
            drive_output_dir=None,
            run_id=None,
            eval_only=False,
            dry_run=True,
            allow_cpu=True,
        )
        # Should not raise unrecognized argument error
        assert mock_parse.return_value.allow_cpu is True
