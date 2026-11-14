# Docs-only publication validation —2026-11-14

User narrowed publication to documentation for speed. No new source/config/tests, JSON evidence payloads, datasets, weights, predictions, venvs or credentials are added by this commit. Existing tracked source/history is retained.

- New entrypoint: `ONBOARDING.md`.
- SM13 chain: `docs/SM13_DEPENDENCIES.md` and `environments/research-runtime-20261114.txt`.
- Experiment/report index: `docs/EXPERIMENT_CATALOG.md`.
- Local historical receipts/artifacts not already tracked remain local; their paths are declarations, not newly published verification.
- Owner runtime after cleanup:94main/105effective package versions matched inventory; CPU imports and85targetedSM13tests passed. This is not a whole-repository or clean-teammate-install PASS claim.
- No training, remote launch, threshold retuning or full-paper reproduction is performed/certified by this documentation commit.

Publication checks: explicit docs-only staging, <=1MB/newfile, known secret-pattern scan, and new onboarding/dependency/catalog local-link check. Actual checks are recorded before commit; no claim that every historical report link resolves without private artifacts.
