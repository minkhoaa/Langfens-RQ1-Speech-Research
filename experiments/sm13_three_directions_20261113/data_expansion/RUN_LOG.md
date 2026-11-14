# Run log

## Environment

- Python 3.11.17; PyTorch 2.6.0+cu124; Transformers 5.22.1; torchaudio 2.6.0+cu124; NumPy 2.4.6; SciPy 1.17.1.
- NVIDIA GeForce GTX 1650, 4 GiB VRAM, driver 620.70.06; 12th Gen Intel Core i5-12450H, 12 logical CPUs.
- All CUDA training and audio extraction jobs used the shared exclusive flock `experiments/sm13_three_directions_20261113/gpu.lock`; CPU BLAS/OpenMP thread caps were 2.

## Preprocessing parity gate

The first two candidate fits and initial extra-audio caches were rejected because they used direct unnormalized Wav2Vec2 input and a SciPy resampler. Their checkpoints, predictions, receipts, and feature arrays are retained under `REJECTED_PREPROCESSING_MISMATCH/`, excluded from every result below. The corrected path now matches `tools/parse_l2arctic.py` audio conversion and `experiments/rq1_encoder_bench_01/features/extract_all_features.py` Wav2Vec2 normalization. On annotated TRAIN probe `ABA_arctic_a0003`, raw archive conversion matched the local waveform exactly; all 13 pooled layers matched the historical cache at numerical precision (`INPUT_PARITY_AUDIT.json`). The Wav2Vec2-specific normalization is kept separate from direct CharSiU input.

## Corrected full runs

The exact command form is:

```bash
flock -n experiments/sm13_three_directions_20261113/gpu.lock \
  -c 'OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python experiments/sm13_three_directions_20261113/data_expansion/extract_pseudo_features.py --fold FOLD'

flock -n experiments/sm13_three_directions_20261113/gpu.lock \
  -c 'OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python experiments/sm13_three_directions_20261113/data_expansion/train_eval.py --fold FOLD --seed SEED'
```

Each actual stdout/stderr stream is retained in the corresponding file under `logs/`. Full 12-epoch runs only are included in the final benchmark; no smoke result is included as a model result.

| Run | Status | Log |
|---|---|---|
| corrected extraction fold 0 | complete; 84.0 s | `logs/extract_fold0.log` |
| corrected candidate fold 0, seed 101 | complete; 12 epochs, 708 updates | `logs/fit_fold0_seed101.log` |
| corrected candidate fold 0, seed 202 | complete; 12 epochs, 708 updates | `logs/fit_fold0_seed202.log` |
| corrected extraction fold 1 | complete; 44.8 s | `logs/extract_fold1.log` |
| corrected extraction fold 2 | complete; 44.7 s | `logs/extract_fold2.log` |
| corrected extraction fold 3 | complete; cache-only CPU receipt; 4.1 s | `logs/extract_fold3_cpu.log` |
| corrected candidate fold 1, seed 101 | complete; 12 epochs, 708 updates; 48.8 s | `logs/fit_fold1_seed101.log` |
| corrected candidate fold 1, seed 202 | complete; 12 epochs, 708 updates; 44.4 s | `logs/fit_fold1_seed202.log` |
| corrected candidate fold 2, seed 101 | complete; 12 epochs, 696 updates; 43.4 s | `logs/fit_fold2_seed101.log` |
| corrected candidate fold 2, seed 202 | complete; 12 epochs, 696 updates; 44.5 s | `logs/fit_fold2_seed202.log` |
| corrected candidate fold 3, seed 101 | complete; 12 epochs, 708 updates; 44.2 s | `logs/fit_fold3_seed101.log` |
| corrected candidate fold 3, seed 202 | complete; 12 epochs, 708 updates; 44.4 s | `logs/fit_fold3_seed202.log` |

The six remaining candidate jobs were launched by `run_remaining_fits.sh`; the queue log records each start and release. All eight folds/seeds have retained fold-keyed EVAL-V3.1 predictions, candidate checkpoint, fold-fit teacher pseudo probabilities/manifests, confidence and class-count receipts. `summarize_results.py` pooled the four held folds separately within each seed/system and wrote `FOLD_SEED_RESULTS.csv`, `POOLED_OOF_RESULTS.json`, and `BENCHMARK_REPORT.md`.
