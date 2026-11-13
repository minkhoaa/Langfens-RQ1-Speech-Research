# Agent A — unified acoustic detector

## Scope and source path

U1 reuses the normalized H-002 frozen Wav2Vec2 Base 960h `hidden_states[6]` cache, its 17-frame reference-conditioned target attention head, existing checkpoint, prediction bundles, and EVAL-V3.1 results. The encoder is ASR-finetuned, not a pretraining-only SSL model. U1 keeps the Gold-Only exact-gap scores and is therefore a score bundle, not a single two-head neural detector.

U2 trains the H-002 attention target head together with a new neural insertion head on the exact `N+1` gaps. The target stream predicts Correct/Substitution/Omission with a 3-way softmax. The independent gap stream predicts no insertion/insertion with a 2-way softmax. Each gap input concatenates two 768D local acoustic edge pools and two 39D intended-phone vectors. Internal gaps pool the trailing/leading five frames from adjacent predicted-center crops; utterance-edge gaps use an all-zero missing-neighbor sentinel. Gap loss labels come from the exact source gap metadata. Target/gap order, counts, and gap keys are checked against the canonical manifests and feature artifacts.

The H-002 acoustic centers come from a local frozen CharSiU frame classifier plus the intended phone sequence. The classifier output is a frame classifier, despite the inherited cache field name `ctc_logits`; no CTC sequence-likelihood claim is made. Wav2Vec2 waveform normalization follows its local checkpoint feature extractor. No gold realized phone, annotation boundary, or error label enters inference features.

## CUDA runs

All runs used the GTX 1650 4 GB and the shared `coordination/phase3_gpu_job.py` lease wrapper. The first U2 launch reached gap feature preparation but failed before model training because a newly written feature width assumed 39D phone inputs incorrectly (`1590` instead of `2×768 + 2×39 = 1614`). Only that incomplete generated cache was removed; the script and cache dimensions were corrected. No result from the failed preflight was treated as training evidence.

| Arm | Target/gap setup | Parameters | Training time | Peak allocated VRAM | Checkpoint SHA-256 |
|---|---|---:|---:|---:|---|
| U2 flat, seed 101 | Flat 3-class target CE; original sqrt-frequency gap weights (positive/negative ratio 10.42) | 653,893 | 24.88 s | 41.33 MiB | `1978633b08034904d566273cb29f3f8305c604ed86e3be1145e1bc70d6375c92` |
| U3 hierarchical, seed 101 | Omission BCE on all targets + conditional S/C BCE on non-omissions; same gap head | 653,764 | 24.62 s | 41.33 MiB | `c3bd772121c854cfb8c0118422a5b2a658ab5e6931a9df18f7b4b955ecd15538` |
| U2 gap-weight 3, seed 101 | U2 flat target branch unchanged; positive/negative gap loss-weight ratio set to 3 | 653,893 | 23.56 s | 41.33 MiB | `d408cae05f85c43c3b21bffe0082d405e5219fa234bd2210f84e12ba9c963206` |

The U2 and U3 CUDA training commands returned nonzero only after checkpoint, prediction, and training receipt writes, when the bundled evaluator import failed because `.venv-cuda` does not include scikit-learn. Each evaluation was then replayed successfully from the project `.venv`, which has scikit-learn. EVAL-V3.1 outputs below were produced by the established H-002 adapter; threshold policies use DEV_CAL only, with DEV_SEARCH remaining exposed and exploratory.

## EVAL-V3.1 results

DEV_SEARCH has 19,755 target events and 20,354 exact gaps across 599 utterances. Each model produced complete target and gap coverage with no missing events or simplex violations. AP is threshold-independent; all reported F1 values use the adapter's independently selected DEV_CAL threshold policy. The unrestricted row is diagnostic and has no warning-budget constraint.

| Arm | AP S | AP O | AP I | Macro3 error AP | Macro3 error F1, CAL 10% | Macro3 error F1, CAL 20% | Macro3 F1, unrestricted | Macro4 mixed F1, unrestricted | Joint FWR, unrestricted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gold-Only seed 101 reference | .36460 | .15943 | .18672 | .23692 | .03425 | — | .30014 | .44578 | 3.691* |
| H-002 target + Gold-Only gap U1 | .37781 | .16729 | .18672 | .24394 | .05464 | — | .30950 | .45920 | 4.397 |
| U2 flat | .37136 | .18532 | .11799 | .22489 | .05905 | .12542 | .28035 | .43674 | 4.654 |
| U3 hierarchical | .36241 | .17612 | .12382 | .22078 | .06117 | .07234 | .26545 | .42697 | 4.295 |
| U2 flat, gap positive/negative weight 3 | .37136 | .18532 | .12735 | .22801 | .05411 | .11834 | .29556 | .44815 | 4.446 |

`*` Gold-Only unrestricted FWR is reported in the inherited H-002 registry; rows are historical source results. The warning-budget policies are not cross-arm calibrated to one shared threshold: each is selected on DEV_CAL.

U2 improved substitution/omission AP slightly over the Gold-Only target head, but its new gap head reduced insertion AP by .06872 and lowered Macro3 AP and unrestricted Macro3 F1. U3 further lowered S/O AP relative to U2, and its modest insertion AP gain did not compensate; no U4 omission architecture was run. The ratio-3 gap-loss ablation raised insertion AP by .00936 over U2 and Macro3 AP by .00312, with unchanged S/O AP. Its unrestricted Macro3 F1 increased by .01521, but CAL-selected 10% and 20% warning-budget Macro3 F1 fell slightly. This is a small one-seed diagnostic, not evidence of a general weight-ratio optimum or a unified-model win.

### Paired seed 202/303 confirmation screen

The seed-101 gap-weight result was followed by the requested two-seed paired screen. Each pair used the same seed, TRAIN population, cached contexts, eight epochs, target branch, and optimizer. Only the gap positive/negative class-weight ratio changed. U2 used the original sqrt-frequency ratio; GAPW3 used ratio 3. Every run had 653,893 parameters and 41.33 MiB peak allocated VRAM. Training times were 22.67 s (U2-202), 22.81 s (GAPW3-202), 23.02 s (U2-303), and 22.92 s (GAPW3-303).

| Seed | Arm | AP S | AP O | AP I | Macro3 error AP | CAL20 Macro3 error F1 |
|---:|---|---:|---:|---:|---:|---:|
| 202 | U2 | .36256 | .17016 | .13755 | .22342 | .11909 |
| 202 | GAPW3 | .36256 | .17016 | .16774 | .23349 | .14448 |
| 202 | GAPW3−U2 | .00000 | .00000 | +.03019 | +.01006 | +.02539 |
| 303 | U2 | .37106 | .17359 | .16145 | .23537 | .09742 |
| 303 | GAPW3 | .37106 | .17359 | .12197 | .22221 | .09742 |
| 303 | GAPW3−U2 | .00000 | .00000 | −.03948 | −.01316 | .00000 |
| Mean, seeds 202/303 | U2 | .36681 | .17188 | .14950 | .22940 | .10826 |
| Mean, seeds 202/303 | GAPW3 | .36681 | .17188 | .14485 | .22785 | .12095 |
| Mean paired delta | GAPW3−U2 | .00000 | .00000 | −.00465 | −.00155 | +.01270 |

The insertion AP and Macro3 AP gains from seed 101 did not replicate: seed 202 improved, while seed 303 regressed enough that the two-seed means favored the original gap loss. CAL20 F1 improved on average but only through seed 202; seed 303 tied. This small screen does not support changing the default gap loss or claiming a robust benefit. No further architecture or weight screens were run.

Official receipts and EVAL-V3.1 outputs are retained for `RQ1-U2-FLAT-TWOHEAD-S202`, `RQ1-U2-FLAT-TWOHEAD-GAPW3-S202`, `RQ1-U2-FLAT-TWOHEAD-S303`, and `RQ1-U2-FLAT-TWOHEAD-GAPW3-S303` in the Agent A output directory. The CUDA wrapper logs record each serialized lease. Training commands returned nonzero after artifact writes because scikit-learn is absent from `.venv-cuda`; all four evaluation replays succeeded from the project `.venv`.

The official JSON files preserve full per-class counts, policies, and confusion matrices:

- `experiments/rq1_unified_01/agent_a/RQ1-U2-FLAT-TWOHEAD-S101_EVAL-V3.1.json`
- `experiments/rq1_unified_01/agent_a/RQ1-U3-HIERARCHICAL-TWOHEAD-S101_EVAL-V3.1.json`
- `experiments/rq1_unified_01/agent_a/RQ1-U2-FLAT-TWOHEAD-GAPW3-S101_EVAL-V3.1.json`

Training receipts, checkpoints, and CAL/SEARCH prediction bundles share those arm names in `experiments/rq1_unified_01/agent_a/`. The failed first preflight is logged in `RQ1-U2-FLAT-TWOHEAD-S101.log`; successful training wrapper logs are retained with `-R2`, `RQ1-U3-...`, and `RQ1-U2-...-GAPW3-...` names.

## WAV + intended-phone inference interface

`experiments/rq1_unified_01/agent_a/infer_twohead.py` accepts an input WAV, JSON array of intended phone symbols, and a checkpoint filename. It runs the local CharSiU frame classifier on the audio to obtain forced centers, extracts normalized Wav2Vec2 hidden states, forms target and exact `N+1` gap contexts, and emits per-target and per-gap probabilities. It needs the local model snapshots and CUDA; it does not depend on precomputed `ctc_logits` artifacts or training labels.

Example:

```bash
/home/khoa/Projects/KLTN/Dataset/.venv-cuda/bin/python \
  experiments/rq1_unified_01/agent_a/infer_twohead.py \
  --audio path/to/audio.wav \
  --phones-json path/to/intended_phones.json \
  --checkpoint RQ1-U2-FLAT-TWOHEAD-S101.pt \
  --output predictions.json
```

The JSON output contains `N` target rows and `N+1` gap rows, an input/checkpoint receipt, and a separate `.receipt.json` with the prediction file hash. For `N=0`, the interface emits zero target rows and one gap; it pools all Wav2Vec2 frames into the left acoustic field and zeros the absent neighbor and both phone vectors. That whole-utterance case was not represented in training, so its gap score is an extrapolation; the `N=0` path is implemented but has not been runtime-verified. A local U2 inference run on `YBAA_arctic_a0003.wav` completed with 39 target rows and 40 gap rows, 1.61 s measured inference time, and 438.76 MiB peak allocated VRAM. The WAV SHA-256 is `45a0246c2aeee9f9348f5b8690f17dfc7f627ce860a772b74331ef23781f9813`; the checkpoint SHA-256 is `1978633b08034904d566273cb29f3f8305c604ed86e3be1145e1bc70d6375c92`. The JSON phone list came from the utterance prompt via local CMUdict (SHA-256 `c0eb0949af713c5d2dbaea50cb6d10eb33634c14d04ad3ff1a3ed881c956fae0`); no gold labels or boundaries were supplied. The output SHA-256 is `550ff1752907b7a8ec56b959d86397581515247bfeb6c6b4b6984ed8a43dc80a`; the sidecar receipt SHA-256 is `e0dc56499643553da342ceeb18605b3c442186be95bd3858ba350854b7f9416d`.

The inference wrapper is a functional local research interface, not an independent-reference guarantee. Prediction validity still depends on callers supplying a genuinely intended phone sequence; annotation-derived references remain a documented risk in the inherited H-002 training/evaluation provenance. DEV_SEARCH results are development-exposed, single-seed exploratory findings, not sealed-test generalization.
