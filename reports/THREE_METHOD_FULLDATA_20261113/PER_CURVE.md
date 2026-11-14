# PER-MDD full-data phone-pool curve

## Method and data

This is a training-free pool-size curve for the PER-MDD retrieval method in [arXiv:2511.20107v1, §§2.1–2.2](https://arxiv.org/html/2511.20107v1). For each FIT utterance, the pool includes a HuBERT frame nearest the midpoint of each realized target-phone span and insertion span. At inference, every frame in the query is compared by cosine similarity to the pool; top 10 neighbors with similarity at least 0.7 vote by plurality, and a frame with no passing neighbor is blank. Adjacent duplicate frame labels are collapsed, then blanks are removed.

The frozen manifest is `experiments/three_method_fulldata_20261113/scaling.json` (SHA-256 `d788ed545d465dab809d50ba2a06950a472445e44214ebe530b3d01752a44072`). It defines four nested FIT subsets of 24, 96, 256, and 628 utterances. The first three subsets are balanced across the six FIT speakers. The 628 point includes all exact-sequence-eligible FIT utterances and preserves their natural speaker frequencies (HQTV 113, LXC 137, NJS 134, PNV 125, SKA 68, SVBI 51). The corresponding pools contain 723, 2,966, 7,946, and 19,806 phone-frame vectors.

All four points use the same 450 CAL and 450 HELD utterances (163,760 query frames). FIT, CAL, and HELD speakers are disjoint. Request records carry reference sequences, but the loader drops those fields; CAL/HELD annotations are never opened by the runner. The evaluator scored predictions after inference. The official test population was not accessed.

## Results

| FIT pool utterances | Pool frames | CAL macro 3-class F1 | HELD macro 3-class F1 | HELD FWR | HELD realized PER | Blank query frames | Mean collapsed phones / utterance |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 723 | 0.0916 | 0.0945 | 23.22 | 1.858 | 18.33% | 79.80 |
| 96 | 2,966 | 0.0873 | 0.0896 | 21.00 | 1.640 | 7.01% | 77.62 |
| 256 | 7,946 | 0.0800 | 0.0685 | 19.16 | 1.478 | 3.19% | 73.85 |
| 628 | 19,806 | 0.0939 | 0.0646 | 17.58 | 1.310 | 1.34% | 69.90 |

The HELD F1 curve declines from 0.0945 at 24 FIT utterances to 0.0646 at 628. FWR and realized PER also decline, so these metrics do not move together. Substitution recall falls from 0.366 to 0.110 over those endpoints; omission recall remains below 0.006, while insertion recall rises from 0.642 to 0.769. This describes the trade-off: fewer false warnings and lower realized phone error rate, but lower macro detection F1 and much weaker substitution recall.

The frame counts suggest a possible reason: as the pool grows, blank predictions become rare, from 18.33% to 1.34%, while the mean collapsed sequence shortens from 79.8 to 69.9 phones. More query frames receive a phone label, but consecutive labels collapse into longer runs. This is consistent with smoother retrieval outputs and lower substitution sensitivity; the curve alone does not establish the cause. Threshold and top-k stayed fixed across all points, and HELD was not used for tuning.

## Backbone fidelity and limits

The run uses the paper's named ASR-finetuned checkpoint, `facebook/hubert-large-ls960-ft`, revision `ece5fabbf034c1073acae96d5401b25be96709d8`, with weights SHA-256 `9cf43abec3f0410ad6854afa4d376c69ccb364b48ddddfd25c4c5aa16398eab0`. The CTC head is discarded; the frozen frame encoder's `last_hidden_state` supplies embeddings. The paper describes frame embeddings as `f(x_t)` but does not identify a hidden layer, so exact layer-level equivalence cannot be confirmed. The paper's reported pool uses 500 training recordings; our largest point uses 628 exact-eligible FIT utterances from this frozen fold. It is a full-data curve for this pilot population, not an exact reproduction of the paper's experiment.

The initial extraction encoded 1,528 waveforms once and cached 1.12 GB of fp32 frame features in 200.56 seconds, within the 1,800-second budget. The original receipt was overwritten by a cache-hit verification run; [TIMING_PROVENANCE.json](/home/khoa/Projects/New-Method-Research/experiments/three_method_fulldata_20261113/per_mdd/TIMING_PROVENANCE.json) records that the initial duration is supported by the captured tool transcript for execution session 10559. The current receipt's 21.29 seconds is only the cache-hit verification/regeneration. The longest selected waveform was 9.054 seconds and yielded 452 × 1,024 frame embeddings; its preprofile peak GPU allocation was 1.44 GB on the 4 GB device. No optimizer updates were performed.

The independent provenance audit recomputed every pool vector from FIT annotations and cached HuBERT frames, including the midpoint frame index, and found exact agreement for all four pools. It also confirmed identical CAL/HELD ID order and counts at every pool point, with evaluator-facing predictions containing only utterance ID, role, and recognized phones.

## Artifacts

- `experiments/three_method_fulldata_20261113/per_mdd/recognized_phones_pool{24,96,256,628}.jsonl`: evaluator-facing predictions.
- `per_mdd/pool_provenance_{24,96,256,628}.jsonl`: FIT-only phone span and selected frame provenance.
- `per_mdd/receipt.json`: model, manifest and audio hashes, budgets, pool composition, frame blank rates, and extraction/cache timings.
- `experiments/three_method_fulldata_20261113/per_curve_metrics.json`: per-point CAL and HELD metrics from the frozen evaluator.

