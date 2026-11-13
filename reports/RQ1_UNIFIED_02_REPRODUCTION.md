# RQ1-UNIFIED-02 — Tái lập thí nghiệm

## Môi trường / split

- GTX 1650 4GB, CUDA khả dụng; PyTorch `.venv` 2.6.0+cu124. `python` hệ thống không có sklearn nên dừng preflight; các neural runs và evaluation sau đó chạy bằng `.venv/bin/python`, không CPU fallback huấn luyện.
- DATA partitions từ manifest EVAL-V3.1: TRAIN 61,362 gaps/560 insertion positives; DEV_CAL 10,343/102; DEV_SEARCH 20,354/125. DEV_SEARCH exposed, exploratory only; không dùng làm training/selection data. Threshold chọn từ DEV_CAL, trừ raw AP không phụ thuộc ngưỡng.
- EVAL helper: `research/long_running_method_discovery/cycle_02/h002_ssl/normalized_v1/evaluate_h002_evaluator_v3.py`, SHA-256 `877d2be390c01666853ddf459e1da2273b899a4338c2948cb1cdd853da211811`; evaluator source `research/evaluation_v3/evaluator.py`, SHA-256 `76e41f906e5d953ee55cee8156379644bdd96bf96efab69e991b87d7f857d4a2`.

## Baseline model and feature lineage

- U2 seed101 original: checkpoint `experiments/rq1_unified_01/agent_a/RQ1-U2-FLAT-TWOHEAD-S101.pt`, SHA-256 `1978633b08034904d566273cb29f3f8305c604ed86e3be1145e1bc70d6375c92`; SEARCH prediction NPZ SHA-256 `75497724fc5d362dbb60d0ed9bda35bdb86f9f380eb8b6445b87e0e922c47622`.
- U2 source `experiments/rq1_unified_01/agent_a/train_twohead.py`, SHA-256 `12ec7bf6a288acfa3178458145d0cbdf905120cc500831fa27192031a5f52736`.
- Train `train_gap_context.npy` SHA-256 `1e4f472768dc2da90ab1aa01c171608a869ca48b0f87b5adabd9d03600be1395` (shape 61,362×1,614); labels/features file `data/derived/l2arctic/features/train_features.npz`, SHA `e85cc2844c5b5b785e9f3b8b9b424ec60c5979829a1511a6fd804d1761ec9fd1`. (Numpy feature file contains full target/gap rows; labels are separately hashed.)
- Frozen branch substitution: `experiments/rq1_unified_02/agent_a/frozen_score_substitutions.py`, SHA `3a48129c26af7a94bfc9f81766a9b1dfca5c2b63e5b83026818f033b46820d40`; output JSON SHA `3ae66b3d01c3f5d514dde46b0d2fa7d0884ae57301d76c2c25e160311b346657`. Script asserts exact target/gap metadata and labels for U2, H-002, GoldOnly; EXP43 gap keys are checked against canonical manifest.

## New controlled CUDA screens

### G1 reduced-width bottleneck

```bash
.venv/bin/python experiments/rq1_unified_02/supervisor/train_gap_bottleneck.py
.venv/bin/python experiments/rq1_unified_02/supervisor/evaluate_gap_bottleneck.py
.venv/bin/python experiments/rq1_unified_02/supervisor/score_gap_arms.py
```

`RQ1-U02-G1-BOTTLENECK128-S101`: same 1,614D input, 128→64 head, seed101, weighted CE, 8 epochs. Train time 7.860s, peak 27.899 MiB, 215,106 parameters. Checkpoint SHA `138a8b71a5c40d45caaac983520a5fcf811ff79d88d3dbd1055edf1c0520b6d3`; SEARCH predictions SHA `1b701f601d5f0d24e35a2b6eb78c8307d62e40a22811c83818fec418b078eb56`. Script SHA `1b4de53556a727ae888feb2998fd27b0cfb1e13434a22fdaa218008f17a05d88`; evaluator replay script SHA `b1a2e7c85c61511ecb70d4a5b09764dcc14d92325e4d7c5ed7a8b86ddb8c9c29`; branch CAL20 scorer SHA `fe37e2956faa9f90cd9cb563094954abd73bb92805b19e2d9aafc4972a11dce2`.

Recorded values: TRAIN AP-I in-sample .978514/BCE .009531; CAL AP-I .136409/BCE .065084; SEARCH AP-I .125929; AP-S/O unchanged .371360/.185318; Macro3 AP .227536. EVAL-V3.1 CAL20 Macro3 F1 .140180, Macro4 .341776, binary MD AP .363315, FWR .198664. `GAP_ARM_EVALS.json` holds complete classes, confusion matrices, thresholds and hashes.

### Balanced-minibatch screen

```bash
.venv/bin/python experiments/rq1_unified_02/agent_b/balanced_gap.py
.venv/bin/python experiments/rq1_unified_02/supervisor/score_gap_arms.py
```

`RQ1-U2-GAP-BALANCED-S101`: 64 positive + 64 negative sampled each batch; 8 epochs, 480 steps/epoch, 128 batch size; seed101; original gap-head shape. Script SHA `3f440ac2033c434196a7459785588c9256475cb64883c1eca693414d09fb5858`; receipt SHA `572084a5dfe04b99250d0e028f64a0fb141d57a81654d30e6f44823509a1df75`; checkpoint SHA `6a59b02d2b73f0c110afdddd9bb1d886d3b11834d91c7a0fb73094280340024b`; SEARCH prediction SHA `89b0fa1a02eeecee5642547bacd7019091009567e40dfd13ff7cd26b30ce5186`. Time 6.841s; peak 36.811 MiB; 430,018 params. TRAIN AP-I .992556/BCE .002783; CAL AP-I .094039/BCE .118980; SEARCH AP-I .125692; Macro3 AP .227457. CAL20 Macro3 F1 .149231/Macro4 .348564/FWR .190317. Full output in `experiments/rq1_unified_02/supervisor/GAP_ARM_EVALS.json`.

### G2/G3 feature screens

G2 script `experiments/rq1_unified_02/agent_a/train_g2_gap6.py` SHA `443ad968d34afe3eea21e535370d84ff6ce0de6d71d92af0d1e441c1939b0857`; receipt SHA `02c7168c433d7e732c35f97e484028f4af65cf5dd6a3c9f5d0dee742d334304a`. Existing 6D `gap_X` only, 6→128→64→2 MLP, 8 epochs, 9,282 params; CAL/SEARCH AP-I .036339/.025938, 7.204s, 18.169 MiB. Checkpoint/prediction hashes are in its receipt and leaderboard.

G3 script `experiments/rq1_unified_02/agent_a/train_g3_concat.py` SHA `81ed534ad379bf47f74d627cdbe126fd90e4b67356f68fc377f527e07934692b`; receipt SHA `3e00447688d0a7902534a37cbc45584d88be855638e8855c4d3c1bbbb0ba70af`. G3 concatenates same 6D `gap_X` to U2's original 1614D edge/phone input; 1620→256→64→2, 8 epochs, 431,554 params; TRAIN/CAL/SEARCH AP-I .976238/.174269/.134162, SEARCH Macro3=.230280, 10.341s/30.063 MiB. Checkpoint SHA `7cdbf7e2e90df94a7e5adaa7d9d37dbefce37365b7dc15284cb249d36fe8dda6`; SEARCH predictions SHA `f9ecd377ba937ebb31e832e2d1186bfae019b870d0f936c89d8e08bc07dbe60d`. G3 was repeated at seeds 202 and 303 with matching U2 target scores. Paired AP-I CAL deltas vs U2: +.03616/+ .00945/+.01770; SEARCH: +.01617/−.00183/−.03332. Mean SEARCH AP-I=.132672 vs U2=.138999; mean Macro3 AP=.225785 vs .227894. The CAL effect did not transfer to SEARCH; no repeatable generalization improvement. Seed202 checkpoint SHA `9d026a098d52fd2bcc8ed0fe5b93c190f45ab902c918e0ae228afecd4af1a9fc`, SEARCH predictions `1a01a3d250e1b65da96063738e9d8dc945aa034b344f5fcde1073232c818b532`; seed303 checkpoint `d04512f27b760ba57a9918c341e62a5976c6e70402484d1bca46f8be7c0a4f3a`, SEARCH predictions `c938dab6282d6b170e90fbd6253e3b78d48f2c3f0f183beb33255d26741837c6`. Do not select on SEARCH alone.

### G6 paired crop-edge temporal screen

`experiments/rq1_unified_02/agent_a/train_g6_temporal.py` SHA `ecb7de03504c1b42132f032caaa58865cb7f4ea37da32c0cc9eca9ab55d9373f`; run under CUDA lease via `.venv/bin/python` (the initial system-Python preflight failed only because sklearn was missing). `RQ1-U02-G6-PAIRED-CROP-EDGE-TEMPORAL-S101`: 8 epochs, 480 steps/epoch, 18×768 paired crop-relative sequence + Conv1d, 168,258 parameters, 32.109s, 83.962 MiB peak. CAL AP-I .062795; SEARCH AP-I .108997, Macro3 AP .221892; CAL20 Macro3 F1 .110666, Macro4 .319640, FWR .165275. Checkpoint SHA `14eedb7173edb253cdf7e3470e10f1ae7f7c50992b87f30ddfecb70e9a3de011`; SEARCH prediction SHA `52eb6f852e2ec4c07ef91b6ffdb5090aa65f9a689eede154c241233983a772d2`; receipt SHA `c5ac7f0186066ff7a95171090ef4f2ce4edb27ec8d50dc646d924805b5532f31`. Exact N+1 IDs and U2 target scores were kept fixed. Caveat: G6 uses crop-relative sequences and Conv1d but omits the two 39D intended-phone vectors, so it is a negative screen with a reference-input confound. G6-REF control is completed below.

`G6-REF`: `experiments/rq1_unified_02/agent_a/train_g6_ref.py` and receipt `RQ1-U02-G6-REF-PAIRED-CROP-EDGE-TEMPORAL-S101.json`; same 18×768 paired crop-sequence + Conv1d with both adjacent 39D phone vectors restored. 8 epochs, 173,250 params, 35.797s, 84.130 MiB. CAL AP-I .100776; SEARCH AP-I .099673, Macro3 AP .218784; CAL20 Macro3 F1 .119996, Macro4 .326638, FWR .186978. Checkpoint SHA `2815a4b03c5bb67b67c3ef773c541f59c35f77a1ba5b2a3e0d8e34759139d6bf`; SEARCH predictions SHA `951c39177fa285053a71f9891a51874de8032387f99bb8fdcb266cd29e091be9`. This control shows phone-vector removal accounts for some CAL loss but not the SEARCH underperformance. One seed; crop-relative, not continuous interval.

### G6-PREDICTED-BOUNDARY continuous W2V test

`experiments/rq1_unified_02/agent_a/train_g6_predicted_boundary.py` extracts one contiguous 17×768 normalized W2V hidden_states[6] window at each exact N+1 gap, using archived CharSiU CTC logits aligned to intended phones only, mapped by predicted CTC-to-W2V frame-clock ratio; positions outside the utterance are zero-padded. It appends the two intended-phone vectors and trains the same Conv1d family (173,250 parameters, 8 epochs, batch128, square-root inverse-frequency weighted CE), keeping U2-S101 target predictions fixed. No gold timing, realized phones or labels enter boundary features. Independent full-wave target-crop parity was <2e-3 and exact event order checks passed. Evaluation from saved predictions with EVAL-V3.1: insertion AP TRAIN/CAL/SEARCH=.832383/.102291/.080151; SEARCH Macro3 AP=.212276, CAL20 Macro3 F1=.130327, Macro4=.334386, FWR=.193656, exact-gap TP/FP/FN/TN=13/27/112/20202. Relative to U2, insertion AP and Macro3 AP decline; this specific continuous-window intervention is negative, not proof that all possible boundary hypotheses fail. CUDA lease lasted 193.920s (includes extraction/training/inference); telemetry max GPU memory-used field=1,136 MiB. Source SHA `ec815a432e52fe7b799ba1f7ac78d4e4966920612265c1d722056b54e1803668`; CPU-only receipt finalizer SHA `8198795ccc5303c9f8a0c0e1cd12a461e8f4c8d3069dca4e0452910ccf3f2d18`; corrected receipt SHA `b451a7e752311eaf09862706863e5073ad7cd41ed9f6b80cf90b2fce55392871`; checkpoint SHA `f9dc8c58ddae34df3bf7e81d6b921c6a9f1815ce4fb4c7c93291dbda3f273582`; CAL/SEARCH prediction SHA `33c0a3306dbd5b07094c3a899a9a862f2c832edf7534e870e7b00bf7d7178ad1` / `1dcb6d4bee1946b9eaf82217cdf7eb25558d1c91fd496c04e52a64b217add76b`. The first frame-mapping and one script preflight attempts failed before training; preserve logs. The successful CUDA run trained and wrote checkpoint/predictions, but its original process exited nonzero after training at JSON serialization (NumPy arrays). Receipt and EVAL-V3.1 policy outputs were reconstructed from persisted predictions without retraining or further CUDA. The receipts disclose the failed posthoc serialization and reference validation lineage limitation.

## Existing controlled U3 and multi-seed controls

Use retained U2/U3 and GAPW3 files under `experiments/rq1_unified_01/agent_a/`; no completed models were restarted. U2 and GAPW3 seed101/202/303 receipts and EVAL-V3.1 JSONs are present there. Historical paired AP means: U2 .368327/.176356/.138999/.227894 (S/O/I/Macro3), GAPW3 .368327/.176360/.139019/.227901; paired insertion AP delta +.000020. U3 seed101 Macro3=.220784 vs U2=.224890.

## PR curves and Precision@K exports

```bash
.venv/bin/python experiments/rq1_unified_02/supervisor/export_ranking_artifacts.py
```

The script validates exact target/gap IDs against U2-S101 on the common DEV_SEARCH population, then exports per-class stepwise PR thresholds/precision/recall/AP and support-sized Precision@K/Recall@K for 17 available trained arms (including G6-PREDICTED-BOUNDARY). Script SHA `c4669b2f0c91ddb3a154105bbcd624b201b37b559da4ee4bc508ab9c16e979ec`; PR gzip SHA `cd8c128fe1473138879d9949080c3527088f92e42ef4abb6e0f1f82013d22ca1`; P@K CSV SHA `1fc656f5e13821f99e9c20d3ab0007b75bfe9edf385489fbb19361ec4e5f510a`. DEV_SEARCH is exposed; these exports are descriptive only.

## Historical hard-negative reference (not restarted)

Existing `experiments/hard_negative_mining/run.py` SHA `bcd1cc0f20ca262253cc2da5f7e55bf1ff6b88e6e28a0f344979dee46134e5bd`; result JSON SHA `f599a80c987ce8de2b0f7f220e0b94b18c7e663a47a4ad350ef4574d10c6a3ef`; checkpoint SHA `278ac8a04dda59816d988fa485297b3ea20f51467e4e05fef2e346401cdbb549`. Prior seed101 CUDA experiment trained GRU64 12 epochs, batch32, lambda=.1 top-128 gold-negative logits from TRAIN minibatches. EVAL-V3.1 SEARCH AP S/O/I/M3=.368680/.159243/.185065/.237663; CAL_MAX F1 Macro3=.292242, Macro4=.435748, FWR=6.716. Relative to its GoldOnly reference, M3 AP gain only +.000746, AP-I −.001650 and M3 F1 lower (.29224 vs .30014). This is a different 51D/6D GRU input and is historical, not a matched U2 gap-head trial; it was recovered, not rerun.

### G9 alignment-confidence auxiliary screen

`extract_g9_ctc_confidence.py` SHA `a88bf66aed236c13b08f25411903a33071d058188a227d457dad18eb34afd5ce`; preflight receipt SHA `82973566934d99315e8f9cb0eb1361f3d5d67aa16dc6727ef467ada7d0cbc476`. The extractor validated all 2,699 archived utterances, canonical intended-phone identities and N+1 event order; it used only `ctc_logits` and intended phones, no gap/target labels or gold timing. G9 runner `train_g9_ctc_concat.py` SHA `0d6d12caa1becb29d097e1bf2ee99bc1964b9bb2530a5c38a5f39b3d25b57345`; G9 receipt SHA `1dcd999550c54df3e8fbb68873aaaa0a9784812eea879f855b521c65a6be006b`. G9 appends six H003 local CTC posterior/entropy summaries to the 1620D G3 input (433,090 params), seed101, 8 epochs. TRAIN/CAL/SEARCH AP-I=.996735/.159784/.155957; SEARCH Macro3=.237545; CAL20 Macro3 F1=.118343; time 10.272s, peak VRAM 30.099 MiB. Checkpoint/prediction hashes are in receipt/leaderboard. It mixes CharSiU CTC features with W2V and is not same-encoder-only; SEARCH is exposed. Matched seed202 replication also completed: CAL/SEARCH AP-I=.134220/.132255, against U2-S202 .128568/.137553 and G3-S202 .138015/.135722. Thus G9-S101 uplift did not replicate on SEARCH; seed202 receipt SHA `5da7520eece8a681343c06d7ab340b05ce7d635c6c8d1b2601c53443209b4e83`, checkpoint SHA `7cdd13f59bb76e58cf80c7142b9385de64c443b7130621e1b561d9f23671f3f1`, SEARCH prediction SHA `160e739ee21dbce57d14c1b9dd5f79cd0959f397bf7d350bbbde227ccdbe832d`. CAL gains over U2 were +.02167/+ .00565, but G9 did not beat G3 on CAL or SEARCH across these seeds; no robust auxiliary gain.

### G8 broad adjacent-phone context

`experiments/rq1_unified_02/agent_a/train_g8_broad.py` SHA `742ee1b8c892f669f9a8d7a72f0fb5fc4ad6879568d5cc2cf430cb9a80e9cf84`; receipt SHA `eab5e3c6add15411919bbc69ec6da695559395e54ee822964b5da43f8ee11a9b`. G8 uses all 17 W2V layer-6 frames from each adjacent target crop (34×768) plus 78D phone reference, Conv1d, seed101. Exact N+1 IDs, U2 target scores fixed. CAL/SEARCH AP-I=.079218/.116505, SEARCH Macro3=.224394; CAL20 M3 F1=.146075, M4=.346197, FWR=.188648. Parameters 173,250; training 54.808s; peak VRAM 140.161 MiB. Checkpoint SHA `73fdc4e0f6d57ee3d3889c4beaefbb74103e055b8c03f63a87d4b91ed5c3a0ef`; SEARCH predictions SHA `cc4088567e0b292f47a3ebd94fed88c27e3614284a35f4a2d614658ede7a9afd`. This broader context did not improve raw AP over G0 and is not a continuous boundary sequence.

## Matched U2/U3 multi-seed CUDA follow-ups

Fresh same-code/same-cache 8-epoch pairs were completed for seeds202 and303 using `experiments/rq1_unified_02/agent_b/train_twohead_local.py` (byte-identical source trainer hash `12ec7bf6a288acfa3178458145d0cbdf905120cc500831fa27192031a5f52736`). Each model took one exclusive lease at a time on GTX 1650; all EVAL-V3.1 outputs are under `experiments/rq1_unified_02/agent_b/`.

| Arm | Seed | SEARCH AP-S/O/I/M3 | CAL20 M3 F1 | Training s / peak MiB | Checkpoint SHA |
|---|---:|---|---:|---|---|
| U2 flat | 202 | .362558/.170159/.137553/.223423 | .119090 | 26.663 / 41.327 | `59072056d761d9b178e0eb4c8956adcd3606ce2d3d9df516af459217e06108de` |
| U3 hierarchy | 202 | .358654/.172383/.143724/.224920 | .074159 | 25.468 / 41.326 | `3bc4d43108db07719d873f6812c6dcb27822109ab0b4c8a5fa2b0803a9b83e2b` |
| U2 flat | 303 | .371064/.173591/.161451/.235369 | .097422 | 23.504 / 41.327 | `43f1c3bd93771fc669df80fafad057b4001813538cb465fb005a8dcc86f82756` |
| U3 hierarchy | 303 | .364913/.176963/.115738/.219205 | .071525 | 30.332 / 41.326 | `a2d679ea1139780dbee239e85566788032928f00073803b791a0e17dc13ea379` |

Seed202 prediction hashes are in `RQ1-U2-FLAT-TWOHEAD-S202_receipt.json` / `RQ1-U3-HIERARCHICAL-TWOHEAD-S202_receipt.json`; seed303 in analogous receipts. Paired U3−U2 Macro3 AP deltas at seeds101/202/303 are −.004106/+.001497/−.016164; CAL20 Macro3 F1 is lower in all three. This is not evidence of a hierarchy benefit.

## G3 inference acoustic perturbation (Q10)

CPU-only script `experiments/rq1_unified_02/agent_c/g3_gap_inference_perturbations.py` SHA `02de8c9a7472eae62e95696daa3923a61585b245d57c4c024a3697508ae28f4a`; result JSON SHA `3418faf57a1f9e77a157b7ffc59d3927ac3fbe0e859d4208e4c7084f1987f02b`. It loads G3-S101 checkpoint/features, holds target scores/IDs/labels fixed, confirms baseline prediction parity within 7.8e-7, and computes baseline / zero W2V / zero gap_X / shuffled W2V / reference-only inference perturbations. SEARCH AP-I .134162 baseline, .046932/.083611/.011221/.019925 under the four perturbations. These are OOD inference changes, not retrained modality controls.

## CPU audit reproductions

Agent A's geometry/ranking audit: `experiments/rq1_unified_02/agent_a/u2_cpu_diagnostics.py` SHA `cb1517317071ea3394fbb808ed5249f6b20ec71ef50a60fbcea3ef23271eecdf`; output JSON SHA `18b2aeee280b0d00fe25e77d9741158e87ae25cdacdbf05b11e474a3718e17fe`. Reproduces U2 TRAIN/CAL/SEARCH AP/AUC `.97698/.99986`, `.13811/.81530`, `.11799/.78343`; class score distributions; feature covariance effective rank ≈236/768 acoustic; SEARCH exact gap-position counts (124/125 positives internal, 0 leading, 1 trailing). It supports overfit/poor transfer but does not isolate cause.

```bash
.venv/bin/python experiments/rq1_unified_02/agent_a/frozen_score_substitutions.py
.venv/bin/python experiments/rq1_unified_02/agent_c/analyze_matched_target_acoustic.py
python experiments/rq1_unified_02/agent_b/diagnostics.py
```

Agent C script SHA `5f5bbaf6c398dddb9efda0c68950c57cbcd84d5b392645f516884d1e93432b3b`; result JSON SHA `e094e154101fb98481035a3a38e33c35ffdd2551d7a66b10ed6c7d4b333c25c4`. It reports exact populations, acoustic controls, class confusions and provenance. Agent B diagnostics state zero GPU training; balanced arm was subsequently executed by supervisor as recorded above.

## Reproduction limitations

All new neural screens are seed101 only. DEV_SEARCH is development-exposed and must not be treated as sealed generalization. No temporal gap-context experiment, multi-seed G1/balanced replication, independent-vs-joint matched training, or integrated candidate is present. No protected holdout was accessed. Completion is therefore partial rather than a claim that all requested hypotheses were causally resolved.
