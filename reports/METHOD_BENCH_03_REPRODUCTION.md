# METHOD-BENCH-03 reproduction and evidence map

## Protocol and environment

Repository: `/home/khoa/Projects/New-Method-Research`; checks recorded 2026-11-08. The current `nvidia-smi` check could not communicate with the NVIDIA driver; historical zero-byte slot files were not treated as active leases. CTC feature extraction/training, edit DP and grouped OOF jobs reported here ran on CPU. TRAIN is fitting data; grouped TRAIN OOF is the primary transfer diagnostic; DEV_CAL is for declared policy selection; DEV_SEARCH is exposed exploratory development. No protected holdout was accessed.

The authoritative TRAIN manifest is `data/derived/l2arctic/manifests/train.jsonl`. It supplies intended and realized target phones, event labels, exact gap IDs and insertion phones. CTC supervision uses only the derived TRAIN realized sequence; gold timing/labels do not enter inference. The detailed data count and pretrained checkpoint audit are in `METHOD_BENCH_03_DATA_METHOD_FEASIBILITY.md` and Agent A receipts.

## Agent A — genuine CTC

### Log-mel BiGRU

ID `RQ1-MB03-A-CTC-LOGMEL-GRU-S101`; seed 101; 8 epochs. Authoritative evaluation correction: `experiments/rq1_method_bench_03/agent_a/RQ1-MB03-A-CTC-EVAL-CORRECTED.json`; receipt/checkpoint hash in `RQ1-MB03-A-CTC-RECEIPT.json`. Source and commands:

```sh
.venv/bin/python experiments/rq1_method_bench_03/agent_a/train_ctc.py --seed 101 --epochs 8
.venv/bin/python experiments/rq1_method_bench_03/agent_a/eval_ctc_checkpoint.py
```

### Frozen Wav2Vec2 CTC heads

IDs `RQ1-MB03-A-CTC-W2V2H6-LIN-S101`, `...-S202`, `...-S303`; frozen `facebook/wav2vec2-base-960h` hidden-state layer 6 plus trainable 43-way linear CTC head. Separate checkpoints and receipts are stored in `experiments/rq1_method_bench_03/agent_a/`. Seeded reproduction commands:

```sh
.venv/bin/python experiments/rq1_method_bench_03/agent_a/train_w2v2_ctc.py --seed 101
.venv/bin/python experiments/rq1_method_bench_03/agent_a/train_w2v2_ctc.py --seed 202
.venv/bin/python experiments/rq1_method_bench_03/agent_a/train_w2v2_ctc.py --seed 303
```

The continuous hidden-state cache was extracted once from audio; seeds 202/303 reuse it. Output archives record posterior/log-posterior values, blank ID, LANGFENS class order, offsets, utterance/speaker IDs and 20-ms clocks. `W2V2_CTC_SEED_PER.csv` is the compact recognition summary. The separate eSpeak/IPA checkpoint retrieval, license metadata, file hash and CPU probe are receipted, but no validated IPA-to-LANGFENS score is reported.

## Agent B — probabilistic edit inference

Source: `experiments/rq1_method_bench_03/agent_b/probabilistic_edit.py`; paired E1/E2/E3 evaluator: `evaluate_ctc_edit.py`; CAL policy application and seed/split outputs are under `agent_b/ctc_edit_run/`. Run synthetic exhaustive fixtures with:

```sh
.venv/bin/python experiments/rq1_method_bench_03/agent_b/synthetic_fixtures.py
```

Empirical outputs are saved for log-mel and W2V2 seeds 101/202/303, each with separate input namespaces, predictions, EVAL JSONs and receipts. The final W2V2 runs use beam 10; an earlier log-mel beam-3 run is retained as an initial diagnostic. Commands in the source report can regenerate files and overwrite ledgers; they were not rerun during final report reconciliation. Candidate-normalized sequence weights have limited beam coverage and must not be described as full CTC/edit posteriors.

## Agent C — speaker transfer and alternative method

Grouped OOF source/receipt: `experiments/rq1_method_bench_03/agent_c/train_speaker_adversary.py` and `speaker_grouped_oof_receipt.json`. Random OOF diagnostic source/receipt: `train_utterance_random_oof.py` and `utterance_random_oof_receipt.json`. EVAL-V3.1 join/predictions and fold outputs are saved alongside. Commands:

```sh
.venv/bin/python -u experiments/rq1_method_bench_03/agent_c/train_speaker_adversary.py
OMP_NUM_THREADS=4 .venv/bin/python -u experiments/rq1_method_bench_03/agent_c/train_utterance_random_oof.py
```

Receipts include fold membership, configurations, input/code/prediction hashes, event coverage, device and runtime. Random and grouped folds answer different questions; only grouped splits inform speaker transfer.

## Historical baselines and evidence limits

METHOD-BENCH-02/UNIFIED-02 and earlier campaign artifacts were retained; historical experiments were not rerun. See the final leaderboard CSV for category separation between trained checkpoints and score compositions. Detailed factual results and limitations live in the linked Agent A/B/C reports. DEV_SEARCH is exposed and does not support a blind-test claim. No independent external method reproduction, full CTC lattice posterior, E4 likelihood-ratio run, matched grouped-OOF CTC/event scores, or sealed evaluation is claimed.
