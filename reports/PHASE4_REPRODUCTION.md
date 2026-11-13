# LANGFENS Phase 4 — reproduction and evidence register

## Environment and source state

- Repository: `/home/khoa/Projects/New-Method-Research`
- Starting Git HEAD: `aa4e6ea2147180a4a820d77a71911f333fcb6721`
- Starting tree was substantially untracked/dirty. Do not reset/clean it; no existing work was intentionally overwritten.
- GTX 1650 4 GiB. Initial `nvidia-smi`: no listed process, 2 MiB VRAM, 0% utilization. This is a point-in-time observation, not a resource lease. No Phase 4 neural GPU job or inference was run.
- EVAL-V3 release directory: `experiments/evaluation_v3/closure/release_final/`; source release records EVAL-V3.1, DEV_SEARCH development-exposed, annotation-derived intended reference independence unverified.

## Reproduce executed work

1. Frozen composition arithmetic audit (CPU; no labels, training, fitting or forward pass):

```bash
PYTHONPATH=. python experiments/phase4/composition/recompute_official_compositions.py
```

Verifies target/gap event metadata and weighted component arrays against frozen official EXP-29/36/43 outputs. Expected: PASS; all target/gap max absolute score deltas 0. Audit includes recipe and score-file hashes: `experiments/phase4/composition/official_recomposition_audit.json`.

2. Unified-student input/mask tests:

```bash
python -m unittest discover -s experiments/phase4/unified_student -p 'test_*.py'
```

Expected: 4 tests pass. These validate interface/masks only; no acoustic model was trained or inferred.

3. Phase 4 report outputs are divided by workstream in `reports/PHASE4_[A-D]_*.md`. The acoustic report is a registered protocol/provenance audit, not an executed intervention study. Confidence and composition results derive from canonical closure tables and frozen score artifacts.

## Key provenance identifiers

- EVAL-V3 split manifest hash in canonical release: `a4ab6c7df7bd65b5d213d306c2fdfde6c316cbf7768a3eb63d056775351f99d2`.
- EXP-24 frozen checkpoint: `78e906d0f5369c23196dab3b8a7816b1f061c380dbfb7ddadd31149e88321b59`; DEV_SEARCH score file: `bf157001d92e225c24c9da3e329dc70648d8bf6b944961333444ec946299a53a`.
- EXP-34 frozen checkpoint: `1c6b167c87ced83a6873d79f96205d119b4bba4ff0dcbf9e597d7e53826a8ac6`; DEV_SEARCH score file: `76cfad9475547b289a4d9ca3ff39dab3ecbc985eb90b179db891a467fbca20c9`.
- EXP-36 frozen composition recipe: `1b1a5a644e51e399def0e41b7d87e885fa5420ee9e9da434bf3c38524a282f01`; DEV_SEARCH score file: `ad06fde5813406089fccb3a7ff92aa07c95a8e7bd07371940888c54204b4b1ae`.
- EXP-29 DEV_SEARCH composition score: `601d877deeb9dca72d489263970954a83a2d0b4efad5b72ebccf9abf979014ab`.
- EXP-43 DEV_SEARCH composition score: `0d9651b9a7c5288997a727039e041e7465de572420f2c015824423fa8f620eff`.

Hashes identify the cited files; they do not establish data validity, reference independence or test-set generalization.

## Not reproduced / not run

- EXP-24+34 historical SEARCH composition, its exact coefficients and candidate-selection trail.
- Historical cumulative/leave-one-out ablations and significance claims.
- New audio/reference-only, lexical neutralization, prosodic intervention, mismatch sensitivity, CTC omission experiments, full PR curves and confidence calibration fit/evaluation.
- Phase 4 D0/D1/D2 neural models, matched seeds 101/202/303, runtime benchmark, D3-D6 gated components.
- End-to-end comparison against hidden/unseen speaker population. No protected data was accessed.

SEARCH score comparisons are descriptive exploratory results only. Any confirmation requires a new authorized, untouched population and independently verified intended references.
