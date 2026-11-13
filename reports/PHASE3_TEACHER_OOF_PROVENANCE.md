# Phase 3 teacher OOF provenance — corrected I sources

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
    "models/teachers/TEACHER_I_EXP01_checkpoint.pt": "98da8bb5ff756df026e7ec17e17b4c5c60b8149c6cdba8c269005831238c6220",
    "models/teachers/TEACHER_I_CHAMPION_checkpoint.pt": "301a3fc7e8f627a058c165be53cd024a9aaa3e60b0f598e36170322e1d0f31bd",
    "research/student/kd/run_teacher_oof.py": "215f2fc0d852ddcbebe74fc5c86984f3b7694045cb815376b8432db13a41655c"
  },
  "elapsed_seconds": 21.852437496185303,
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
      "raw_brier": 0.01464506983757019,
      "cal_brier": 0.009069209918379784,
      "raw_auprc": 0.11457946542709094,
      "cal_auprc": 0.10876824228040347,
      "raw_ece15": 0.015024920754189762,
      "cal_ece15": 4.584761106835911e-07
    },
    "dev_search_i_f1": {
      "raw_brier": 0.009761849418282509,
      "cal_brier": 0.006078771781176329,
      "raw_auprc": 0.03476013316123945,
      "cal_auprc": 0.026286891324664694,
      "raw_ece15": 0.011150901272344428,
      "cal_ece15": 0.0017544652394005252
    },
    "dev_cal_i_prec": {
      "raw_brier": 0.12768004834651947,
      "cal_brier": 0.009341135621070862,
      "raw_auprc": 0.06406433789466938,
      "cal_auprc": 0.06219582158069243,
      "raw_ece15": 0.30536520959386043,
      "cal_ece15": 7.633909685339514e-10
    },
    "dev_search_i_prec": {
      "raw_brier": 0.10221526771783829,
      "cal_brier": 0.006074974313378334,
      "raw_auprc": 0.027563282200573393,
      "cal_auprc": 0.016168436607194407,
      "raw_ece15": 0.2774554391946049,
      "cal_ece15": 0.0027909403440085963
    }
  },
  "limitations": [
    "fold heads are economical champion-architecture replicas, 8 epochs vs original full-training champion budgets; not equivalent reproductions",
    "S binary score is substitution event probability, O is omission event probability; they are not a shared simplex",
    "no frame-level acoustic representations; only approved frozen cached slot/gap feature arrays"
  ],
  "teacher_source_mapping": {
    "if1": "TeacherI_Network / TEACHER_I_EXP01; extract_gap_context_features (23D); TRAIN normalization fit on fold-training speakers, std<1e-4 set to 1, normalized features clipped [-10,10], fit samples gap duration>=2 frames",
    "iprec": "TeacherI_ChampionNet / TEACHER_I_CHAMPION; build_epenthesis_features (25D); TRAIN normalization fit on fold-training speakers, std<1e-4 set to 1, normalized features clipped [-10,10], fit samples gap duration>=2 frames",
    "dev_frozen_checkpoints": {
      "if1": "models/teachers/TEACHER_I_EXP01_checkpoint.pt",
      "iprec": "models/teachers/TEACHER_I_CHAMPION_checkpoint.pt"
    }
  },
  "calibration_application": "Isotonic calibration fitted on DEV_CAL; mapping also applied to OOF TRAIN as an explicitly shifted/extrapolated training score. It is not TRAIN-label fitted.",
  "score_artifacts_sha256": {
    "teacher_scores_v2_train.npz": "d8c4376fdf7e413fbebf668cd8ef51670088bcedecbaad93ee156378b5fe14e2",
    "teacher_scores_v2_dev_cal.npz": "ce816030d92e8490a3896185e19e36fdbceabc14a0c32bc0e9a133864cd1ce93",
    "teacher_scores_v2_dev_search.npz": "634960f5aeb55c35096edcf6e7a28f9903042d49f5c3137c840fd9fc2737bf96"
  }
}
