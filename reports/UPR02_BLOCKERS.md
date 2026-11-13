# UPR-02 BLOCKERS & RISK REGISTER
**Campaign:** UPR-02  
**Updated:** 2026-11-06T01:25:00Z  

## Current Blockers
- **None.** All authorized data splits (`TRAIN`: 1,800, `DEV_SEARCH`: 599, `DEV_CAL`: 300) are fully verified and present locally. GPU is idle and unblocked.

## Identified Risks & Mitigations
1. **Risk:** Overfitting to dev set by unconstrained hyperparameter sweeps.
   - **Mitigation:** DEV_SEARCH is reserved for architectural exploration and diagnostic ablation; DEV_CAL is strictly reserved for calibration and threshold freezing. Final candidate evaluation runs under frozen rules.
2. **Risk:** Concurrency violation on GPU.
   - **Mitigation:** All GPU tasks strictly acquire `/home/khoa/Projects/Pronunciation-RD-Coordination/locks/gpu.lock` via non-blocking or managed timeout flock.
3. **Risk:** Incomparable evaluation protocols.
   - **Mitigation:** Use unified evaluator `evaluator/unified_evaluator.py` with strict event ID binding across all experiments.
