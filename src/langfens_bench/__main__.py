"""Unified inference-only preflight → evaluate → merge → compare interface."""
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('-h', '--help'):
        print('usage: python -m langfens_bench {preflight,evaluate,merge,compare} [options]')
        print('Use <command> --help for its gated inference-only interface.')
        return 0
    if argv and argv[0] == 'compare':
        from .compare import main as run
        return run(argv[1:])
    if argv and argv[0] == 'merge':
        from .merge import main as run
        return run(argv[1:])
    from .runtime import main as run
    return run(argv)


if __name__ == '__main__':
    raise SystemExit(main())
