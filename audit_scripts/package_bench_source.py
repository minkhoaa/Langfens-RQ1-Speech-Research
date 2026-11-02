#!/usr/bin/env python3
"""CLI adapter for the shared allowlisted offline source packager (no Git history)."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='New private staging ZIP outside Git')
    args = parser.parse_args(argv)
    from langfens_bench.kaggle import build_source_package
    if args.output.resolve().is_relative_to(REPO):
        parser.error('Keep source staging packages outside the Git worktree')
    sha = build_source_package(REPO, args.output)
    print(json.dumps({'path': str(args.output.resolve()), 'sha256': sha}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
