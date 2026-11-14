# Agent B — focal-loss gamma 2.0 screening

## Protocol

Distinct namespace: `experiments/agent_b_focal_gamma2p0/`; no gamma 1.5 outputs were overwritten. This was the matched gold-only GRU-64 pipeline, seed 101, 12 epochs / 684 updates, batch 32, AdamW (lr 0.001, weight decay 0.0001), class-weighted focal objective on both target and gap heads with gamma 2.0. Acoustic normalization statistics and square-root inverse class weights were derived from TRAIN only. The checkpoint was selected by minimum unweighted target-plus-gap cross-entropy on DEV_CAL (epoch 11); the predeclared CAL_MAX_ERROR_MACRO_F1 thresholds were fitted on DEV_CAL only. EVAL-V3 DEV_SEARCH metrics below are exploratory development results, not a blind evaluation or deployment result. CUDA was required and used: NVIDIA GeForce GTX 1650.

Command:

```bash
.venv/bin/python coordination/phase3_gpu_job.py --job agent-b-focal-gamma2p0-seed101 -- .venv/bin/python experiments/agent_b_focal_gamma2p0/train_focal_gamma2p0.py > experiments/agent_b_focal_gamma2p0/train.log 2>&1
```

## DEV_SEARCH EVAL-V3 results (exploratory only)

AP is ranking-based; F1 uses the policy fitted on DEV_CAL. Complete event coverage: 19,755 targets and 20,354 gaps, 599 utterances, no missing events.

| Domain / class | AP | F1 |
|---|---:|---:|
| Correct (target; secondary) | 0.974581 | 0.859526 |
| Substitution | 0.353573 | 0.360105 |
| Omission | 0.159107 | 0.260645 |
| Insertion (exact gap) | 0.191841 | 0.270627 |
| **Macro3 error (S/O/I)** | **0.234840 AP** | **0.297126 F1** |
| **Macro4 mixed-domain F1** | — | **0.437726** |

Macro4 mixes target and gap domains and is not a four-way softmax metric. The DEV_CAL policy thresholds were substitution 0.403808, omission 0.520964, insertion 0.512724. On DEV_SEARCH the joint false-warning rate was 6.933 per utterance, far above the 0.20 reference; this policy is not safe/deployable. The model's FWR/per-class and full evaluator details are in `results.json`.

## Run record

- Runtime: 46.377 seconds; peak allocated VRAM 44.094 MiB.
- Best checkpoint epoch 11; 684 optimizer updates.
- Checkpoint: `experiments/agent_b_focal_gamma2p0/focal_gamma2p0_seed101.pt` (320,342 bytes).
- SHA-256: `0c69ec9bf82c249de1696d1f866002478bf8f05090283a2fb88b9845e7069b72`.
- Artifacts: `train_focal_gamma2p0.py`, `train.log`, `results.json`, checkpoint, `focal_gamma2p0_dev_cal.npz`, and `focal_gamma2p0_dev_search.npz`.

## Interpretation and next step

Against the prior same-seed gamma 1.5 screening, gamma 2.0 is worse on macro3 error AP (0.234840 vs 0.239906), macro3 error F1 (0.297126 vs 0.301233), and macro4 F1 (0.437726 vs 0.441777). Relative to gamma 1.5, substitution AP/F1 fell (0.353573/0.360105 vs 0.358542/0.367920); omission AP/F1 was nearly unchanged/slightly lower (0.159107/0.260645 vs 0.160967/0.264591); insertion AP/F1 fell (0.191841/0.270627 vs 0.200208/0.271186). Correct F1 also decreased (0.859526 vs 0.863409). This single-seed screen does not support increasing gamma from 1.5 to 2.0; it is not evidence of a robust causal effect. The next step is to stop gamma escalation and, if the objective remains of interest, run the preplanned matched-seed gamma 0/1/2 comparison with identical safeguards and compare seed-level uncertainty. Treat DEV_SEARCH as development-exposed (including these exploratory results); do not use sealed data for tuning.
