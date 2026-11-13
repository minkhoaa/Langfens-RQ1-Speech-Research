# METHOD-BENCH-03 Agent A — genuine phoneme CTC

## Result

A genuine utterance-level CTC phone recognizer was trained with a valid `torch.nn.CTCLoss` objective on TRAIN annotation-derived **realized** phone sequences. This was not a frame classifier or an inference-time use of gold errors. For each TRAIN utterance, non-omitted target `realized_phone` labels and all annotated gap-insertion phones were ordered by annotated interval; those intervals were used only to form the training sequence target. At inference, only audio is used. Canonical `target_phone` sequences were scored separately as an alternate reference, never substituted for realized references.

The authoritative TRAIN manifest contains 59,562 targets and 61,362 exact gaps: 50,356 correct, 7,381 substitution, 1,825 omission targets; 560 positive insertion gaps with 620 annotated inserted phones. Realized target phone is present for 57,737 targets and absent for the 1,825 omissions. Thus the annotations provide realized-phone supervision, not merely canonical phones. One-off annotation token variants were normalized explicitly in both references and training targets: `ER)→ER`, `D_→D`, `Z_→Z`, `Y_→Y`, and `AH)→AH`.

## Model, split, and results

Seed 101, CPU, one run. Input: utterance-CMVN 40-bin log-mel, 10-ms STFT frames, average pool ×2 to 20-ms; `Linear(40,96) + GELU + LayerNorm + bidirectional GRU(80 per direction) + 43-way projection` (42 phones plus blank ID 42), 96,491 parameters. AdamW (`lr=8e-4`, `weight_decay=1e-4`), gradient clip 5, batch 16, 8 epochs. Training loss is CTC log-probability with `blank=42`, `zero_infinity=True`; no invalid-topology batches were skipped. Training fit used 10 TRAIN speakers, with HJK/HKK held out from fitting (300 utterances) and evaluated as speaker-held-out transfer. Epoch CTC loss decreased 5.1211 → 2.0744. CPU training wall time 157.4 s. Checkpoint SHA-256: `1594cd7866be64dc6fd47ef6f213dfc0c0c3d920371ba2ccbe777dfd25687bd5`.

PER is shown separately against realized and canonical references. S/D/I are counts from greedy CTC-collapse edit alignment; beam column is the un-LM'd width-10 CTC prefix-beam result. Development values are descriptive only; DEV_SEARCH is already exposed and was not used to select the checkpoint.

| Split | Utterances | Reference | Greedy PER (S/D/I) | Beam-10 PER (S/D/I) |
|---|---:|---|---:|---:|
| TRAIN fit, 10 speakers | 1,500 | realized | .7181 (4,733 / 30,132 / 16) | .6240 (7,022 / 23,251 / 37) |
|  |  | canonical | .7442 (5,718 / 31,275 / 10) | .6589 (8,327 / 24,398 / 35) |
| TRAIN held-out speakers HJK/HKK | 300 | realized | .7899 (953 / 6,774 / 1) | .7207 (1,529 / 5,520 / 2) |
|  |  | canonical | .8021 (1,061 / 6,830 / 1) | .7367 (1,668 / 5,577 / 3) |
| DEV_CAL (ERMS, THV) | 300 | realized | .7577 (868 / 6,474 / 1) | .6664 (1,310 / 5,143 / 5) |
|  |  | canonical | .7866 (1,073 / 6,826 / 1) | .7040 (1,576 / 5,492 / 2) |
| DEV_SEARCH (NCC, RRBI, YBAA, YDCK) | 599 | realized | .7613 (2,071 / 12,905 / 5) | .6802 (3,105 / 10,272 / 7) |
|  |  | canonical | .7753 (2,330 / 12,982 / 5) | .6978 (3,424 / 10,352 / 10) |

Greedy blank argmax fractions: TRAIN fit .9159, held-out TRAIN .9268, DEV_CAL .9321, DEV_SEARCH .9264. These high values accompany severe under-decoding: deletions dominate and insertion counts are near zero. On DEV_SEARCH, greedy realized-reference phone error rates are especially high for `OW`, `UH`, `AW`, `AA`, `AO`, `UW`, `Y`, and `SH` (each >99% errors among reference counts >=100); common substitutions include D→T (78), Z→S (45), AH→T (31), and IH→T (29). This is a weak baseline, not evidence of useful transfer. Per-speaker realized greedy PER on held-out TRAIN: HJK .788, HKK .792; DEV_SEARCH: NCC .743, RRBI .787, YBAA .756, YDCK .758. Detailed speaker metrics and per-utterance mappings are in the receipts/decodes.

## Frozen Wav2Vec2 CTC-head comparison

The second, separate experiment uses the continuous hidden-state stream from frozen `facebook/wav2vec2-base-960h`, hidden layer 6 (768 dimensions, 20-ms clock), with only a trainable `Linear(768,43)` CTC projection. It does not use CharSiU frame-classifier logits. It consumes full utterance audio and is not the cached H-002 17-frame forced-center event crop. The existing extraction receipt verifies the ASR-finetuned encoder provenance and that realized phones, error labels, and gold timing were absent from feature generation. Only labels from the 10 TRAIN fit speakers train the head; HJK/HKK are held out for primary speaker-grouped development. Seeds 101, 202, and 303 use identical eight-epoch optimizer, batching, and split settings; the latter two reuse the exact cached continuous feature arrays and do not rerun encoder extraction.

All 2,699 authorized utterances (2.73 audio hours) were extracted on CPU in the measured campaign run; extraction plus training/evaluation took 769.9 seconds. A standalone CPU probe measured 0.087–0.093 real-time factor and 1.16 GiB peak RSS. The head used the same realized-sequence targets, seed 101, 8 epochs, batch 16, AdamW (`lr=1e-3`, `weight_decay=1e-4`), CTC blank ID 42, and no invalid-topology batches. CTC loss decreased 10.086→1.629.

| Split | Utterances | Reference | Greedy PER (S/D/I) | Beam-10 PER (S/D/I) |
|---|---:|---|---:|---:|
| TRAIN fit, 10 speakers | 1,500 | realized | .3897 (4,399 / 14,091 / 441) | .3193 (6,684 / 7,688 / 1,138) |
|  |  | canonical | .4285 (5,657 / 15,224 / 425) | .3642 (8,278 / 8,764 / 1,065) |
| TRAIN held-out HJK/HKK | 300 | realized | .4130 (849 / 3,127 / 64) | .3276 (1,256 / 1,801 / 148) |
|  |  | canonical | .4277 (965 / 3,181 / 62) | .3444 (1,392 / 1,853 / 144) |
| DEV_CAL (ERMS, THV) | 300 | realized | .4137 (1,019 / 2,895 / 95) | .3472 (1,562 / 1,540 / 263) |
|  |  | canonical | .4587 (1,271 / 3,244 / 92) | .3981 (1,849 / 1,889 / 260) |
| DEV_SEARCH (NCC, RRBI, YBAA, YDCK) | 599 | realized | .3931 (1,697 / 5,884 / 154) | .3208 (2,652 / 3,190 / 470) |
|  |  | canonical | .4071 (1,893 / 5,978 / 171) | .3368 (2,897 / 3,277 / 480) |

Greedy blank argmax fractions for seed 101 are .6994 (TRAIN fit), .6874 (held-out TRAIN), .7215 (DEV_CAL), and .7043 (DEV_SEARCH), much lower than the log-mel model’s .916–.932. Seed-101 held-out TRAIN greedy PER is .399 for HJK and .427 for HKK (4,923 and 4,860 reference phones). DEV_CAL is .433 for ERMS (5,000 phones) and .394 for THV (4,691). DEV_SEARCH is .440 NCC (4,923), .368 RRBI (4,927), .369 YBAA (4,933), and .396 YDCK (4,895). This is better than the scratch log-mel model, but recognition error remains high.

### Seed replication

The fixed three-seed run repeats the checkpoint without using DEV_SEARCH to tune epochs, weights, or decoding. Beam-10 realized-reference PER is similar across seeds; detailed greedy/beam counts and supports are in `W2V2_CTC_SEED_PER.csv` and the separate seed receipts.

| Seed | TRAIN fit greedy/beam | Held-out TRAIN greedy/beam | DEV_CAL greedy/beam | DEV_SEARCH greedy/beam |
|---:|---:|---:|---:|---:|
| 101 | .3897 / .3193 | .4130 / .3276 | .4137 / .3472 | .3931 / .3208 |
| 202 | .3881 / .3149 | .4114 / .3229 | .4087 / .3355 | .3884 / .3112 |
| 303 | .4105 / .3302 | .4415 / .3405 | .4308 / .3547 | .4150 / .3328 |
| Mean (sample SD) | .3961 (.0125) / .3215 (.0079) | .4220 (.0169) / .3303 (.0091) | .4177 (.0116) / .3458 (.0097) | .3988 (.0142) / .3216 (.0108) |

Thus the Wav2Vec2 recognition improvement over log-mel repeats across three seeds on the same speaker-held-out groups and both development splits. Each seed has its own `RQ1-MB03-A-CTC-W2V2H6-LIN-S{101,202,303}.pt` checkpoint and matching `_RECEIPT.json`; the CAL/SEARCH posterior and decode files are namespaced `w2v2_s{seed}_...`. This is a recognition result only; Agent B reports the separate matched MDD event AP comparisons by seed.

## Acoustic-free canonical sequence control

For an acoustic-free reference, the intended canonical target-phone sequence is treated as the hypothesis and compared with the manifest’s realized sequence (non-omitted realized target phones plus gap insertions). This is a diagnostic built from authoritative labels, not a deployable recognizer or an inference-time feature. It is a sequence-PER control, not an MDD AP comparison.

| Split | Reference phones | Substitutions | Deletions | Insertions | Canonical-sequence PER |
|---|---:|---:|---:|---:|---:|
| TRAIN | 58,357 | 6,612 | 552 | 1,757 | .1529 |
| DEV_CAL | 9,691 | 1,716 | 90 | 442 | .2320 |
| DEV_SEARCH | 19,678 | 1,658 | 129 | 206 | .1013 |

Neither acoustic recognizer approaches this control: frozen-Wav2Vec2 beam PER is .347 DEV_CAL/.321 DEV_SEARCH, and log-mel beam PER is .666/.680. The campaign therefore did not show that acoustic recognition improves phone transcription over the canonical sequence. These PER values must not be compared numerically with historical phone-prior MDD AP.

## Acoustic outputs for Agent B

Log-mel files `experiments/rq1_method_bench_03/agent_a/{train_fit,train_speaker_holdout,dev_cal,dev_search}_posteriors.npz` contain concatenated float16 `posteriors` and `log_posteriors`, the phone-class order plus `<blank>`, blank ID 42, 20-ms frame clock, per-utterance `[flat_start, frame_count]` offsets, and `[utterance_id,speaker_id,frame_count,realized_ref_len,canonical_ref_len]` metadata. Decodes preserve both references and greedy/beam hypotheses. Wav2Vec2 files use the same schema under the `w2v2_` prefix, including `w2v2_dev_cal_posteriors.npz` and `w2v2_dev_search_posteriors.npz`. Prefix beam retains up to ten hypotheses with top-12 token pruning per frame; its stored prefix scores are search scores, not exact full-sequence rescoring. These are frame posteriors, not fabricated event-aligned probabilities.

## Pretrained checkpoint feasibility (audit only)

The supervisor downloaded the authorized `facebook/wav2vec2-lv-60-espeak-cv-ft` weights to `/tmp` (revision `fdf22f0210322b799533bc499dd3e489a9452bdc`, 1,263,535,127 bytes, SHA-256 `3173bde9e9ce490fa0f989e413c42f25bc1820c020adc1e6b9b87025b3cfcc5e`). Its [model card](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft) reports Apache-2.0 and multilingual phonetic output in IPA and cites [Simple and Effective Zero-shot Cross-lingual Phoneme Recognition](https://arxiv.org/abs/2109.11680). Offline inference loaded the 392-logit model and decoded one TRAIN and one DEV_CAL utterance at CPU RTF .293/.270, with 1.94 GiB peak RSS.

Tokenizer IDs were verified: `<pad>` is CTC blank ID 0, `<s>`=1, `</s>`=2, `<unk>`=3. Common English correspondences include `iː→IY`, `ɪ→IH`, `ɛ→EH`, `æ→AE`, `θ→TH`, `ð→DH`, `ʃ→SH`, `ʒ→ZH`, `tʃ→CH`, `dʒ→JH`, `ŋ→NG`, and `aɪ→AY`. However, the 392-entry multilingual vocabulary includes composite tokens and context/dialect ambiguity (`ə` vs `ʌ`, rhotic vowels, tokens such as `əl`, and LANGFENS classes `SPN`/`ERR` without a direct IPA equivalent). A validated IPA-to-LANGFENS tokenizer segmentation and converter was not established, so this is a verified external IPA-output probe, not a LANGFENS PER/MDD result. Hashes and probe data are in `FACEBOOK_ESPEAK_CHECKPOINT_RECEIPT.json` and `FACEBOOK_ESPEAK_ONE_UTTERANCE_PROBE.json`.

The GTX 1650 is unavailable in the current session: `nvidia-smi` cannot communicate with the NVIDIA driver. Both trainable CTC heads and hidden-state extraction ran on CPU.

## Reproduction and limitations

```sh
.venv/bin/python experiments/rq1_method_bench_03/agent_a/train_ctc.py --seed 101 --epochs 8
.venv/bin/python experiments/rq1_method_bench_03/agent_a/eval_ctc_checkpoint.py
.venv/bin/python experiments/rq1_method_bench_03/agent_a/train_w2v2_ctc.py --seed 101
.venv/bin/python experiments/rq1_method_bench_03/agent_a/train_w2v2_ctc.py --seed 202
.venv/bin/python experiments/rq1_method_bench_03/agent_a/train_w2v2_ctc.py --seed 303
.venv/bin/python experiments/rq1_method_bench_03/agent_a/probe_facebook_espeak.py
```

The evaluation-only correction receipt supersedes the log-mel run’s interim edit-count bug; it did not retrain or select that model. Use `RQ1-MB03-A-CTC-RECEIPT.json` and the three seed-specific W2V2 receipts for authoritative metrics. Scripts, checkpoints, features, posteriors, decodes, external model receipts, and canonical-sequence control are in `experiments/rq1_method_bench_03/agent_a/`. No protected data were accessed. Limitations: one log-mel seed and one frozen-encoder architecture family (with three projection-head seeds), exposed development evaluation, no external LM, and no validated external IPA-to-LANGFENS converter. The per-phone log-mel diagnostic is descriptive, not an inference mapping or tuning procedure.
