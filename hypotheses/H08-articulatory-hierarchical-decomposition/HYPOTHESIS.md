# Hypothesis H08: Hierarchical Multi-Task Articulatory Feature Network (HM-ADN)

## 1. Proposition
Phonetic pronunciation errors in L2 speech are not arbitrary token misclassifications; they follow structured articulatory trajectories governed by Place, Manner, and Voicing breakdowns (e.g. /θ/ -> [s] preserves manner and voicing while shifting place; /d/ -> [t] preserves place and manner while losing voicing).

Single-task classifiers suffer from representation collapse, struggling to separate subtle phonetic shifts from normal accent allophones. By training a shared neural sequence encoder with auxiliary multi-task articulatory supervision:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diag}} + 0.3 \mathcal{L}_{\text{manner}} + 0.3 \mathcal{L}_{\text{place}} + 0.2 \mathcal{L}_{\text{voicing}} + 0.5 \mathcal{L}_{\text{gap}}$$
the shared representation space is regularized into phonologically coherent manifolds.

This inductive bias will:
1. Substantially elevate `SUBSTITUTION` F1 beyond 0.40 on `DEV_SEARCH`,
2. Improve `OMISSION` precision by anchoring absence of manner gestures,
3. Maintain or improve `CORRECT` and `INSERTION` metrics,
4. Drive Weighted Utility beyond **0.380** on `DEV_SEARCH` and beyond **0.430** on `DEV_REPLICATION`.

## 2. Experimental Plan
- Method ID: `M13-articulatory-multitask-scorer`
- Run ID: `EXP-13-ARTICULATORY-DECOMPOSITION`
- Train on `TRAIN` for 30 epochs with multi-task articulatory targets.
- Calibrate on `DEV_CAL`.
- Primary locked evaluation on `DEV_SEARCH`.
