# H-006 Cycle 2: articulatory auxiliary objective screen

**Status:** executed controlled single-seed screen; **SUPPORTED_EXPLORATORY for a mixed score change only**, not a robust method gain. No integration recommendation from this one seed.

## Hypothesis and experiment

Tested whether auxiliary place/manner/voicing-relevant supervision can shape the shared student representation without giving the auxiliary branch a reference-phone shortcut. Intended-phone articulatory vectors were generated using the local `PHONOLOGICAL_FEATURE_TABLE`. The auxiliary branch received the same acoustic/context sequence encoder with its 39-D reference-phone identity dimensions set to zero; the primary S/O and N+1 gap heads retained the standard 51-D target and 6-D gap input contract. Auxiliary target was a 16-dimensional binary phonological vector, optimized with masked BCE at weight 0.1.

The primary treatment was compared against an architecture-matched control with the same auxiliary head and branch, seed, data, class weights, optimizer, training budget and checkpoint rule, but auxiliary loss weight zero. Both models have 87,957 parameters (12% larger than the standard Gold-Only GRU at 78,661 parameters). This controls parameter count for the causal treatment comparison. It is a single seed, not a reproducibility result.

The 16D table is coarse and has known feature collisions; this screen does not isolate separate categorical place, manner and voicing heads. The pre-existing table includes voicing (dimension 3), manner cues (dimensions 0–5), and place cues (dimensions 6–8), as well as other properties. Therefore this tests combined articulatory multi-label supervision, not the individual H006-B/C/D component claims.

## Provenance and CUDA receipts

- Runtime: `/home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python`, PyTorch 2.14.1+cu130, CUDA 13.0; NVIDIA GeForce GTX 1650.
- Code: `code/train_articulatory_aux.py`; current SHA-256 `871ff38dab20ab4dd5aabdfe4436651beaa456c7f063e9d0d2c26adc2e84126b`. The treatment was run before the later addition of a configurable CLI weight; at execution it used the hard-coded 0.1 auxiliary coefficient. The matched control was run from the same code after adding `--aux-weight 0`.
- TRAIN feature SHA-256: `e85cc2844c5b5b785e9f3b8b9b424ec60c5979829a1511a6fd804d1761ec9fd1`.
- Both arms used seed 101, 12 epochs, 684 updates, standard TRAIN-derived scaling/class weights, lowest unweighted DEV_CAL gold-loss checkpoint.
- Treatment job ID `RC2-H006-ARTIC-AUX-CUDA130-S101-R2`: wrapper return code 0; 55.87 s, peak allocated 51.39 MiB; checkpoint SHA-256 `a30a734ccaba7fb91c8df855a35927efa96936f032dbd52e2ac007e3b91c8e79`.
- Matched control job ID `RC2-H006-ARTIC-ARCH-CONTROL-CUDA130-S101`: wrapper return code 0; 66.32 s, peak allocated 51.39 MiB; checkpoint SHA-256 `b3432da6381451e10a64588d43f239d38ab41dd857142f0c14b8a8b95056aa52`.
- Both jobs went through `coordination/phase3_gpu_job.py`; CUDA leases were released. The ledger records active python processes and GPU utilization. GPU concurrency overlapped briefly with the separately leased H-002 extraction; sampled memory stayed below 1.2 GiB, no VRAM pressure was observed.
- H006 initial attempt `RC2-H006-ARTIC-AUX-CUDA130-S101` failed before training, return code 1, because importing the teacher source module required missing sklearn. The retry fixed this by using the canonical 39-phone list literal already in the source. No neural update or output came from the failed attempt.
- Prediction/evaluation artifacts are under this namespace; both EVAL-V3 outputs preserve 19,755 targets, 20,354 exact gaps and 599 utterances.

## EVAL-V3 results

DEV_SEARCH is development-exposed; metrics below are descriptive. AP values do not depend on the decision thresholds.

| DEV_SEARCH AP | Matched aux-weight-0 control | Articulatory weight 0.1 | Delta |
|---|---:|---:|---:|
| S | 0.364602 | 0.364057 | -0.000545 |
| O | 0.159433 | 0.160935 | +0.001502 |
| exact-gap I | 0.186715 | 0.192187 | +0.005472 |
| Macro-3 Error AP | 0.236917 | 0.239060 | +0.002143 |

At the evaluator’s 0.20 policy budget, Macro-3 Error F1 was `0.113712` with auxiliary supervision and `0.104897` in the matched control. Class precision/recall/F1 for the treatment were:

- S: `1.0000 / 0.000566 / 0.001131`
- O: `0.2222 / 0.01914 / 0.03524`
- exact-gap I: `0.3765 / 0.2560 / 0.3048`

At policy budgets 0.05, 0.10 and 0.50, treatment vs control Macro-3 Error F1 was `0.025063 vs 0.025063`, `0.101587 vs 0.034250`, and `0.175405 vs 0.163591`. These are CAL-selected policies applied to SEARCH; behavior varies with operating point. They do not establish a general calibration or unseen-speaker gain. Search speakers are only four.

Gold-Only 78,661-parameter same-cycle control APs were `0.364602 / 0.159433 / 0.186715`, Macro-3 AP `0.236917`. The 0-weight architecture control matches those APs exactly, confirming that the capacity-matched branch itself did not shift raw ranking in this seed. Treatment changes ranking slightly in the positive direction for O and I and slightly negative for S. There is no multi-seed evidence.

## Decision

The single-seed combined auxiliary objective yielded a small mixed raw ranking shift and higher Macro-3 Error F1 at some EVAL-V3 operating budgets. Given tiny AP deltas, single-seed SEARCH exposure, four SEARCH speakers, coarse feature table, and no separate place/manner/voicing ablations, classify as **SUPPORTED_EXPLORATORY only**. It is not enough to select a unified architecture or integrate into Phase 4. A matched multi-seed, TRAIN-speaker-grouped screen with separate auxiliaries and the same 0-weight architecture control is needed.

## Artifacts and hashes

- Treatment checkpoint: `checkpoints/RC2-H006-ARTIC-AUX-CUDA130_seed101.pt`, SHA-256 `a30a734ccaba7fb91c8df855a35927efa96936f032dbd52e2ac007e3b91c8e79`.
- Control checkpoint: `checkpoints/RC2-H006-ARTIC-ARCH-CONTROL-CUDA130_seed101.pt`, SHA-256 `b3432da6381451e10a64588d43f239d38ab41dd857142f0c14b8a8b95056aa52`.
- Treatment SEARCH predictions SHA-256 `138d7a63073842034602b5631ea786a3c666127ee1272ad7ee27f0fa522ca72c`; EVAL-V3 JSON SHA-256 `dcfd98b8bbc1d93ebb18fe2f51ced4c565294e6c7c2cc6f9fb1518ff1e8c68c0`.
- Control SEARCH predictions SHA-256 `9b6e36adf250a46cf47b44f3ee5df9de48648ae2e7dbb48e58047b14ed586d5c`; EVAL-V3 JSON SHA-256 `faeff2f4d3a45fc0ac3f0a3771ea31cc8545a1b90f405099515ccadb799fb10c`.

## Reproduction

```bash
.venv/bin/python coordination/phase3_gpu_job.py --job RC2-H006-ARTIC-AUX-CUDA130-S101-R2 -- /home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python -u research/long_running_method_discovery/cycle_02/h006_articulatory/code/train_articulatory_aux.py --arm RC2-H006-ARTIC-AUX-CUDA130 --epochs 12 --seed 101 --aux-weight 0.1
.venv/bin/python coordination/phase3_gpu_job.py --job RC2-H006-ARTIC-ARCH-CONTROL-CUDA130-S101 -- /home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python -u research/long_running_method_discovery/cycle_02/h006_articulatory/code/train_articulatory_aux.py --arm RC2-H006-ARTIC-ARCH-CONTROL-CUDA130 --epochs 12 --seed 101 --aux-weight 0
```
