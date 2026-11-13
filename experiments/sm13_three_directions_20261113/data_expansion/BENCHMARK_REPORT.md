# Fit-speaker pseudo-label data expansion

**Question.** Does adding a fixed 50 previously unannotated prompted utterances per fit speaker with high-confidence, fold-fit-teacher soft pseudo-targets improve exact target-event prediction at a matched optimizer-update budget?

**Paired runs available:** 8/8 fold-seed pairs. Primary results pool held predictions across all four folds separately for each seed and system, then run EVAL-V3.1 once. Fold-level results are supplementary; seed means are descriptive, not confidence intervals.

## Pooled out-of-fold results

Each seed row pools all 1,800 distinct held-out utterances, exact N target events, and exact N+1 gaps. Per-fold calibration-frozen decisions are preserved; continuous scores are pooled only after held predictions are fixed. Candidate insertion probabilities are unchanged shared baseline scores.

| Seed | System | Utterances | Targets | Gaps | Macro-3 AP | Macro-3 F1 | FWR | Sub AP | Omi AP | Ins AP |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | control | 1800 | 59562 | 61362 | 0.2469 | 0.1249 | 0.2189 | 0.3827 | 0.2525 | 0.1054 |
| 101 | candidate | 1800 | 59562 | 61362 | 0.2463 | 0.1239 | 0.2128 | 0.3782 | 0.2551 | 0.1054 |
| 202 | control | 1800 | 59562 | 61362 | 0.2507 | 0.1299 | 0.2350 | 0.3744 | 0.2647 | 0.1130 |
| 202 | candidate | 1800 | 59562 | 61362 | 0.2481 | 0.1406 | 0.2556 | 0.3703 | 0.2609 | 0.1130 |

Across the two seeds, averaging the per-seed pooled values gives control→candidate Macro-3 AP 0.2488→0.2472, Macro-3 F1 0.1274→0.1323, and realized FWR 0.2269→0.2342.
Candidate Macro-3 AP is lower for both seeds. Although thresholds were frozen using calibration-only B=0.2, candidate realized FWR exceeds 0.20 for both seeds (0.2128 and 0.2556); this experiment does not pass that operating target. The metric changes are mixed and do not establish a consistent gain from this self-training expansion.

### Pooled class P/R/F1/AP

| Seed | System | Class | Precision | Recall | F1 | AP |
|---:|---|---|---:|---:|---:|---:|
| 101 | control | Correct | 0.8525 | 0.9943 | 0.9180 | 0.9458 |
| 101 | control | Substitution | 0.6715 | 0.0504 | 0.0938 | 0.3827 |
| 101 | control | Omission | 0.5252 | 0.0800 | 0.1388 | 0.2525 |
| 101 | control | Insertion | 0.3798 | 0.0875 | 0.1422 | 0.1054 |
| 101 | candidate | Correct | 0.8528 | 0.9944 | 0.9182 | 0.9447 |
| 101 | candidate | Substitution | 0.6792 | 0.0565 | 0.1043 | 0.3782 |
| 101 | candidate | Omission | 0.5489 | 0.0707 | 0.1252 | 0.2551 |
| 101 | candidate | Insertion | 0.3798 | 0.0875 | 0.1422 | 0.1054 |
| 202 | control | Correct | 0.8534 | 0.9942 | 0.9184 | 0.9444 |
| 202 | control | Substitution | 0.6707 | 0.0599 | 0.1100 | 0.3744 |
| 202 | control | Omission | 0.5187 | 0.0685 | 0.1210 | 0.2647 |
| 202 | control | Insertion | 0.3836 | 0.1000 | 0.1586 | 0.1130 |
| 202 | candidate | Correct | 0.8541 | 0.9933 | 0.9185 | 0.9439 |
| 202 | candidate | Substitution | 0.6657 | 0.0648 | 0.1180 | 0.3703 |
| 202 | candidate | Omission | 0.5406 | 0.0838 | 0.1452 | 0.2609 |
| 202 | candidate | Insertion | 0.3836 | 0.1000 | 0.1586 | 0.1130 |

### Fold-seed variation

These per-held-group rows are supplementary and show variation between folds.

| Fold | Seed | Held speakers | Control AP | Candidate AP | Δ AP | Control FWR | Candidate FWR | Δ FWR | Pseudo rows | Teacher-predicted pseudo C/S/O |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 101 | ABA;ASI;BWC | 0.2971 | 0.2881 | -0.0090 | 0.2311 | 0.2444 | +0.0133 | 4859 | 4612/134/113 |
| 0 | 202 | ABA;ASI;BWC | 0.2908 | 0.2976 | +0.0068 | 0.2667 | 0.2778 | +0.0111 | 4381 | 4189/144/48 |
| 1 | 101 | EBVS;HJK;HKK | 0.1896 | 0.1869 | -0.0027 | 0.2311 | 0.2111 | -0.0200 | 4807 | 4650/109/48 |
| 1 | 202 | EBVS;HJK;HKK | 0.1876 | 0.1858 | -0.0018 | 0.2267 | 0.2733 | +0.0467 | 4999 | 4856/124/19 |
| 2 | 101 | HQTV;LXC;NJS | 0.2908 | 0.2986 | +0.0078 | 0.2556 | 0.2089 | -0.0467 | 4510 | 4338/132/40 |
| 2 | 202 | HQTV;LXC;NJS | 0.3026 | 0.3081 | +0.0055 | 0.2622 | 0.3022 | +0.0400 | 5336 | 5163/122/51 |
| 3 | 101 | PNV;SKA;SVBI | 0.2271 | 0.2307 | +0.0036 | 0.1578 | 0.1867 | +0.0289 | 5150 | 4895/144/111 |
| 3 | 202 | PNV;SKA;SVBI | 0.2224 | 0.2231 | +0.0006 | 0.1844 | 0.1689 | -0.0156 | 5725 | 5571/47/107 |

## Protocol and provenance

- Outer folds use `GROUPS[f]` as held, `GROUPS[(f+1)%4]` as calibration, and the remaining six TRAIN speakers as fit. No held/calibration speaker audio enters pseudo-labeling or fit adaptation.
- Exactly 50 unique extra utterances per fit speaker were selected before teacher prediction by SHA-256 ordering of nonmanifest utterance IDs. Extra acoustic features are cached once per speaker and reused only on folds where that speaker is in the fit set.
- A fold-fit SM13 scalar-mixing target teacher generates three-class probabilities. Only phone rows with maximum teacher probability ≥0.90 are used, as soft pseudo-targets. The prompt supplies intended phones for alignment/reference vectors and is not an actual-pronunciation correctness label.
- Candidate training uses the same target head, class weights, 12 epochs, batch size 512, AdamW learning rate 0.001, weight decay 0.01, and optimizer-update count as the labelled-only target baseline. Each batch has 75% labelled fit rows and 25% pseudo rows, so pseudo examples replace labelled exposures; this is reported in `FOLD_SEED_RESULTS.csv`.
- The insertion head and its continuous probabilities are unchanged from the shared fold baseline. EVAL-V3.1 scores exact N target events and N+1 gaps, with thresholds frozen on calibration speakers under B=0.2. Held labels only enter EVAL-V3.1. Oracle fixed-precision metrics are diagnostics only.
- Source: L2-ARCTIC v5.0 TRAIN speaker archives from the pinned public mirror revision in `SOURCE_MANIFEST.json`, CC BY-NC 4.0; source licence, README, checksums, per-utterance hashes, extraction receipts, pseudo-target files, and keyed held predictions are retained in this directory.
- Per-run pseudo confidence min/Q25/median/Q75/max and teacher-predicted C/S/O counts are in `FOLD_SEED_RESULTS.csv`; full row-level teacher probability vectors and confidence values are in each pseudo manifest and retained-pseudo NPZ.
- Update counts observed: 696, 708.

## Limits

This tests only the within-TRAIN-speaker pseudo-label self-training subcase of data expansion: it adds 600 extra clips across the same 12 speakers, not new speaker diversity or an external corpus, and no human error annotation. The 0.90 confidence filter and pseudo classes do not establish pseudo-label correctness because the extra utterances have no adjudicated realized-phone labels. Any benefit or harm is limited to the selected 50 utterances per fit speaker, this frozen teacher, and this training schedule. The eight fold-seed runs overlap in speakers and are not independent samples.

## Reproduce

```bash
for fold in 0 1 2 3; do
  flock -n experiments/sm13_three_directions_20261113/gpu.lock env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python experiments/sm13_three_directions_20261113/data_expansion/extract_pseudo_features.py --fold "$fold"
done
for fold in 0 1 2 3; do
  for seed in 101 202; do
    flock -n experiments/sm13_three_directions_20261113/gpu.lock env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python experiments/sm13_three_directions_20261113/data_expansion/train_eval.py --fold "$fold" --seed "$seed"
  done
done
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python experiments/sm13_three_directions_20261113/data_expansion/summarize_results.py
```

The actual run log is stored in `RUN_LOG.md`; per-run commands and hardware are recorded there.
