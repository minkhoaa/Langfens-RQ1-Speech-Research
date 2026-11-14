# CROTTC-IF full-FIT learning curve

**Status:** bounded two-seed learning curve completed on the frozen M0 fold-0 FIT population. This is the compact local adaptation described below, not a reproduction of the paper.

## Protocol and artifacts

- Data: 628 exact-sequence-eligible FIT utterances; 450 CAL; 450 speaker-disjoint HELD. The other 272 FIT candidates were excluded by the frozen exact-sequence eligibility rule. All 386 CAL utterances with eligible realized sequences contribute to PER; strict SOI diagnostics use all 450 CAL utterances.
- Official TEST access was disabled in `scaling.json`. The runner reads only FIT sequence/annotation inputs and CAL requests, exact-sequence supervision, and annotations. HELD IDs are read from the frozen role list and HELD features are used for prediction only. HELD request/annotation labels are never opened, and HELD scores do not affect any decision.
- Frozen manifest SHA-256: `d788ed545d465dab809d50ba2a06950a472445e44214ebe530b3d01752a44072`.
- Frozen 768-dimensional H6 feature files were SHA-256 checked and preloaded once for both seeds. Training used CPU with two threads, 16 fixed epochs per seed, and seed values 101 and 202.
- The total run took **542.829 seconds** against the registered 1,800-second cap. The recorded budget receipt is `experiments/three_method_fulldata_20261113/crottc_if/budget_receipt.json`.
- Each seed has 16 atomic, reload-verified checkpoints containing model, optimizer, scheduler, scaler, RNG state, and metadata. Each epoch also has CAL and HELD decoded phone sequences. Results are under `experiments/three_method_fulldata_20261113/crottc_if/seed101/` and `seed202/`.

The runner invocation was:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m research.three_method_scaling.crottc_curve \
  --manifest experiments/three_method_fulldata_20261113/scaling.json \
  --cal-supervision experiments/three_method_fulldata_20261113/cal_supervision.jsonl \
  --cal-annotations experiments/three_method_fulldata_20261113/cal_annotations.jsonl \
  --output experiments/three_method_fulldata_20261113/crottc_if
```

The runner reuses the pilot's compact CROTTC-IF adaptation: full-sequence monotone OTTC, two-view symmetric-KL consistency regularization, an acoustic-attention autoregressive phone decoder, a FIT-only canonical/error teacher with phone-distribution distillation into student logits, and a FIT-supervised output-length head. The acoustic features are frozen Wav2Vec2-base hidden-state-6 features; this is not end-to-end WavLM-Large, the paper's full subphonetic-state topology, its Transformer/LM decoder, or its joint AM/LM search.

## CAL learning curve

Values are the mean across two seeds; brackets show the two per-seed values' range. PER is exact realized-phone sequence edit rate over the 386 eligible CAL utterances. Macro3 SOI F1 and FWR use deterministic hard edit alignment against the canonical sequence and all CAL event annotations. The hard path supplies no continuous event scores, so AP is not estimated.

| Epoch | FIT loss | CAL realized PER | CAL strict macro3 SOI F1 | CAL joint FWR |
|---:|---:|---:|---:|---:|
| 1 | 7.4208 | 0.8938 [0.8793, 0.9083] | 0.0773 [0.0764, 0.0781] | 25.3311 [25.2733, 25.3889] |
| 2 | 6.4037 | 0.8519 [0.8337, 0.8701] | 0.0804 [0.0792, 0.0816] | 24.9278 [24.7956, 25.0600] |
| 3 | 5.6939 | 0.8444 [0.8378, 0.8509] | 0.0855 [0.0853, 0.0857] | 24.0578 [24.0156, 24.1000] |
| 4 | 5.1951 | 0.8277 [0.8187, 0.8366] | 0.0892 [0.0879, 0.0904] | 23.2800 [23.1533, 23.4067] |
| 5 | 4.7899 | 0.7794 [0.7742, 0.7846] | 0.0918 [0.0905, 0.0932] | 22.1211 [22.1000, 22.1422] |
| 6 | 4.4597 | 0.7677 [0.7631, 0.7723] | 0.0910 [0.0890, 0.0930] | 21.2689 [21.2533, 21.2844] |
| 7 | 4.1366 | 0.7252 [0.7158, 0.7346] | 0.0955 [0.0938, 0.0972] | 20.4289 [20.4022, 20.4556] |
| 8 | 3.8659 | 0.6983 [0.6952, 0.7015] | 0.0991 [0.0975, 0.1007] | 19.6089 [19.5933, 19.6244] |
| 9 | 3.6266 | 0.6679 [0.6668, 0.6691] | 0.1017 [0.1009, 0.1025] | 18.8711 [18.7711, 18.9711] |
| 10 | 3.4135 | 0.6574 [0.6498, 0.6649] | 0.1036 [0.1028, 0.1044] | 18.4322 [18.4044, 18.4600] |
| 11 | 3.2441 | 0.6492 [0.6315, 0.6669] | 0.1095 [0.1082, 0.1108] | 18.1411 [18.1400, 18.1422] |
| 12 | 3.1195 | 0.6470 [0.6388, 0.6552] | 0.1079 [0.1063, 0.1095] | 17.7756 [17.6956, 17.8556] |
| 13 | 3.0239 | 0.6412 [0.6319, 0.6506] | 0.1104 [0.1054, 0.1154] | 17.7289 [17.6533, 17.8044] |
| 14 | 2.9701 | 0.6342 [0.6275, 0.6409] | 0.1071 [0.1070, 0.1071] | 17.6122 [17.5800, 17.6444] |
| 15 | 2.9262 | 0.6287 [0.6238, 0.6336] | 0.1075 [0.1013, 0.1137] | 17.5622 [17.5089, 17.6156] |
| 16 | 2.9062 | 0.6285 [0.6199, 0.6371] | 0.1094 [0.1052, 0.1136] | 17.5611 [17.5356, 17.5867] |

The total FIT loss and CAL PER decreased across the curve, while typed SOI F1 improved only slightly and remained low. Strict FWR stayed far above one warning per utterance, reflecting the hard sequence alignment's weak typed decisions. Phone recognition learning therefore did not translate into useful typed error localization in this compact adaptation.

## Loss components and evidence

Every epoch's mean total, OTTC, consistency, autoregressive LM, length, teacher, and distillation losses are stored in each seed's `history.json`. The end-to-end paths completed without nonfinite loss; all 32 epoch checkpoints were saved and reloaded during the run. The required output record counts are 900 (450 CAL + 450 HELD) per epoch per seed. The artifacts expose the full curves for the root-owned matched-split comparison with baseline and SM13. Those comparisons are descriptive: the arms have different architectures, objectives, optimizers, and feature interfaces, so they are not compute-matched causal method rankings.

The four synthetic scaling-runner tests passed in 1.60 seconds using the repository virtual environment and two CPU threads.

This curve does not establish HELD performance. Root scoring must use the registered per-epoch predictions and held labels outside this trainer; HELD must not be used to select an epoch or alter the method.
