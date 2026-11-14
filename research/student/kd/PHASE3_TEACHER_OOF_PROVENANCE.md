# Phase 3 teacher OOF provenance — SUPERSEDED

The initial run's I-F1/I-Precision source mapping was incorrect. Its score artifacts and fold checkpoints are preserved under `experiments/student_training/kd/superseded_wrong_i_teacher_sources/` and MUST NOT be used as correctly sourced I teacher supervision. Corrected versioned artifacts and source mappings are recorded in `PHASE3_TEACHER_OOF_PROVENANCE_V2.md`.


This artifact records three-speaker-grouped-fold OOF replicas trained on TRAIN only. The run produced 12 fold checkpoints, with eight epochs per model/fold. Device: NVIDIA GeForce GTX 1650 CUDA 12.4; elapsed 36.6s; peak allocated VRAM 22.2 MiB.

S fitting excludes gold omission events, while all held-out target events receive scores. OOF normalization is fit on fold-training speakers only. Frozen full-TRAIN checkpoints produce DEV_CAL/DEV_SEARCH scores. Calibration is isotonic and DEV_CAL-only; DEV_SEARCH is transformed but never used to fit it.

{
  "device": "NVIDIA GeForce GTX 1650",
  "cuda": "12.4",
  "seed": 101,
  "folds": [
    {
      "fold": 0,
      "heldout_speakers": [
        "BWC",
        "HJK",
        "HQTV",
        "LXC"
      ],
      "train_speakers": [
        "ABA",
        "ASI",
        "EBVS",
        "HKK",
        "NJS",
        "PNV",
        "SKA",
        "SVBI"
      ]
    },
    {
      "fold": 1,
      "heldout_speakers": [
        "ABA",
        "PNV",
        "SKA",
        "SVBI"
      ],
      "train_speakers": [
        "ASI",
        "BWC",
        "EBVS",
        "HJK",
        "HKK",
        "HQTV",
        "LXC",
        "NJS"
      ]
    },
    {
      "fold": 2,
      "heldout_speakers": [
        "ASI",
        "EBVS",
        "HKK",
        "NJS"
      ],
      "train_speakers": [
        "ABA",
        "BWC",
        "HJK",
        "HQTV",
        "LXC",
        "PNV",
        "SKA",
        "SVBI"
      ]
    }
  ],
  "training_epochs_per_fold": 8,
  "normalization": "fit only on each fold training speakers",
  "S_fit_excludes_omission": true,
  "S_scored_on_all_heldout_targets": true,
  "event_id": "l2arctic:speaker:utterance:target|gap:index",
  "hashes": {
    "data/derived/l2arctic/manifests/train.jsonl": "1f4da8518d9b184fb737b93c7887ecd7c3e73d6e83260e54651074831f61bcff",
    "data/derived/l2arctic/features/train_features.npz": "e85cc2844c5b5b785e9f3b8b9b424ec60c5979829a1511a6fd804d1761ec9fd1",
    "models/teachers/TEACHER_S_CHAMPION_checkpoint.pt": "47e804011f6fa744bf09dbd13896cfcd9ef874534d9eaa9f403e37265b000510",
    "models/teachers/TEACHER_O_CHAMPION_checkpoint.pt": "80618baaefe59eff0da3db0a9a64941fb2bdb1a10146c60ee93b286f9634db76",
    "models/teachers/TEACHER_I_CHAMPION_checkpoint.pt": "301a3fc7e8f627a058c165be53cd024a9aaa3e60b0f598e36170322e1d0f31bd",
    "models/teachers/TEACHER_I_EXP02_checkpoint.pt": "9207a8d15012bbd5a6f150c4187981402e9a6e2dedcd43391907aa07899625a1",
    "research/student/kd/run_teacher_oof.py": "0b26b65cc15969d129b22a0fc5154864a1ea84954d15e71e783348dfd897d75c"
  },
  "elapsed_seconds": 36.618505239486694,
  "peak_allocated_vram_mb": 22.23046875,
  "calibration": "isotonic fitted on DEV_CAL only",
  "calibration_metrics": {
    "dev_cal_s": {
      "raw_brier": 0.13579094409942627,
      "cal_brier": 0.11480854451656342,
      "raw_auprc": 0.47650758830050277,
      "cal_auprc": 0.47482391905071614,
      "raw_ece15": 0.10053172983903519,
      "cal_ece15": 9.70734313515098e-09
    },
    "dev_search_s": {
      "raw_brier": 0.10570457577705383,
      "cal_brier": 0.07211378216743469,
      "raw_auprc": 0.3833314226675234,
      "cal_auprc": 0.3607627237571072,
      "raw_ece15": 0.14927791311933372,
      "cal_ece15": 0.06242943941656772
    },
    "dev_cal_o": {
      "raw_brier": 0.07438666373491287,
      "cal_brier": 0.02998422645032406,
      "raw_auprc": 0.4969212399589645,
      "cal_auprc": 0.49255266911604617,
      "raw_ece15": 0.17705325757446463,
      "cal_ece15": 1.0164519307758436e-09
    },
    "dev_search_o": {
      "raw_brier": 0.06065257638692856,
      "cal_brier": 0.011760840192437172,
      "raw_auprc": 0.1415016259981061,
      "cal_auprc": 0.1353049384652385,
      "raw_ece15": 0.18205652303392147,
      "cal_ece15": 0.015966217958122506
    },
    "dev_cal_i_f1": {
      "raw_brier": 0.1284330189228058,
      "cal_brier": 0.00968219805508852,
      "raw_auprc": 0.021434786198691595,
      "cal_auprc": 0.022137790274841654,
      "raw_ece15": 0.30537969820256505,
      "cal_ece15": 3.955622224181708e-10
    },
    "dev_search_i_f1": {
      "raw_brier": 0.10217662155628204,
      "cal_brier": 0.006093326956033707,
      "raw_auprc": 0.027022043179354933,
      "cal_auprc": 0.013946277113537988,
      "raw_ece15": 0.2770243252776626,
      "cal_ece15": 0.003122927614280798
    },
    "dev_cal_i_prec": {
      "raw_brier": 0.029880987480282784,
      "cal_brier": 0.009442540816962719,
      "raw_auprc": 0.054569113532895015,
      "cal_auprc": 0.05792681891002932,
      "raw_ece15": 0.11488964012371528,
      "cal_ece15": 3.572035816840903e-10
    },
    "dev_search_i_prec": {
      "raw_brier": 0.022831998765468597,
      "cal_brier": 0.006068041082471609,
      "raw_auprc": 0.029388580012471903,
      "cal_auprc": 0.02235806999979099,
      "raw_ece15": 0.10423701780111441,
      "cal_ece15": 0.0022978452557059683
    }
  },
  "limitations": [
    "fold heads are economical champion-architecture replicas, 8 epochs vs original full-training champion budgets; not equivalent reproductions",
    "S binary score is substitution event probability, O is omission event probability; they are not a shared simplex",
    "no frame-level acoustic representations; only approved frozen cached slot/gap feature arrays"
  ]
}
