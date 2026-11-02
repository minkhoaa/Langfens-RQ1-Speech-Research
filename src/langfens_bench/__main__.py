"""Shared CLI for local and explicitly authorized Kaggle inference."""
from .runtime import main

if __name__ == "__main__":
    raise SystemExit(main())
