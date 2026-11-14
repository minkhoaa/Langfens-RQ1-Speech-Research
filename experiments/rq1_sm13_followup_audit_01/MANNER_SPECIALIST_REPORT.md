# SM13 vowel/stop specialist ablation

**Date:** 2026-11-13  
**Decision:** Reject this loss/data-routing-only candidate. It does not improve the useful low-warning operating region.  
**Scope:** TRAIN speaker OOF, seed 101, four original outer folds. No DEV or TEST access; thresholds below are retrospective and not deployable.

## Question and fixed design

Does the original global C/S/O target-head training lose vowel/stop discrimination because other manners dominate its gradients? We trained one vowel head and one stop head per outer fold. Each is the original **633,361-parameter SM13 `ScalarMixingTargetHead`** with the same 13-layer cached features, phone reference, 12 epochs, AdamW 0.001 / 0.01, batch 512, and `[1, 2.5, 4]` weighted CE. The sole design change is restriction of training rows to that target manner. The same seed schedule initializes each head. Held-speaker rows are never used for fitting. At evaluation, the vowel head scores vowel targets, the stop head scores stop targets, and the archived global model scores all other manners. The frozen O/I score streams remain fixed.

This is one exploratory ablation of the original head, not a tuned architecture search. It used the cooperative GPU lease and finished in **108.1 seconds**. Eight new checkpoints and exact-order OOF scores were saved separately; no frozen production artifact was overwritten.

## Results

| Manner | Gold substitutions | Global baseline AP | Specialist AP | Outer-fold AP directions |
|---|---:|---:|---:|---|
| Vowel | 2,280 | 0.2348 | 0.2335 | Improved in 2/4 folds |
| Stop | 862 | 0.1968 | 0.1845 | **Worse in 4/4 folds** |
| Other, unchanged routing | 4,239 | 0.5246 | 0.5246 | Identical by construction |

At a retrospectively selected single-manner threshold with joint FWR ≤0.20, the global baseline reaches **81 vowel TP / 143 FP** (3.55% recall, 36.16% precision) and **78 stop TP / 143 FP** (9.05% recall, 35.29% precision). The specialist routing reaches only **64 vowel TP / 141 FP** (2.81% recall, 31.22% precision) and **57 stop TP / 143 FP** (6.61% recall, 28.50% precision). There is no nonempty operating point with ≥50% precision for either specialist manner in this search. Vowel and stop thresholds are separate counterfactuals, each assessed with the other alert streams held at their corresponding candidate scores.

The result weakens the simple explanation that global class competition alone causes the failure. It does **not** prove that the frozen encoder lacks the relevant acoustic contrast: specialist training uses fewer examples, one seed, the original pooling, and the same labels. The next discriminating test should change the evidence available to the scorer, such as explicit vowel trajectories or stop release/closure context, and compare at matched warning risk with speaker-held-out validation. Human listening remains necessary to decide whether all `*` deviations should trigger corrective feedback.

## Reproduction

```bash
.venv/bin/python coordination/phase3_gpu_job.py --job SM13-MANNER-SPECIALISTS-S101 -- .venv/bin/python experiments/rq1_sm13_followup_audit_01/train_manner_specialists.py
```

Machine-readable results: `MANNER_SPECIALIST_RESULTS.json`; scores: `MANNER_SPECIALIST_OOF.npz`; checkpoints: `manner_specialist_checkpoints/`.
