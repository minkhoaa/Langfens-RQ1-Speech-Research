# Invalid sensitivity run preserved; do not evaluate

`RC2-H002-ASR-FINETUNED-EXTRACT-S101-R4` extracted full split features and `RC2-H002-ASR-FINETUNED-ATTN-TRAIN-S101` trained an 8-epoch head, but the extraction fed raw waveforms to a checkpoint whose `preprocessor_config.json` sets `do_normalize=true`. The run is out-of-contract and its checkpoint/predictions must not be used as evidence. The head job also stopped before prediction publication when an independent gap metadata comparison exposed dtype differences (text vs object); after inspection the ID values match when string-normalized.

The checkpoint and all completed raw-input caches are retained here. Failed runner-wrapper attempts R2/R3 are separately preserved in `../attempt2` and `../attempt3`. The corrected experiment uses `AutoFeatureExtractor` from the pinned local model and a new experiment ID `RC2-H002-ASR-FINETUNED-ATTN-NORM-S101`.
