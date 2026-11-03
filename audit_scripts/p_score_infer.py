#!/usr/bin/env python3
"""Isolated inference-only P-SCORE entrypoint. Use PYTHONPATH=src."""

from langfens_prep.p_score.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
