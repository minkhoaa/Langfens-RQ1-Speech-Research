# Phoneme CTC recognition — RQ1-METHOD-BENCH-03

## Methods and supervision

Two genuine utterance-level CTC recognizers were trained against TRAIN realized-phone sequences: (1) 40-bin waveform log-mel, utterance CMVN, BiGRU and 43-way phone/blank projection, seed 101; (2) frozen continuous `facebook/wav2vec2-base-960h` hidden layer 6 (768D, 20 ms) plus trainable linear 43-way CTC projection, seeds 101/202/303. Both use actual CTC loss, valid sequence lengths, explicit blank ID 42, greedy/beam decoding and continuous utterance clocks. No frame-classifier logits or annotation labels enter inference.

TRAIN realized sequence targets use non-omitted `realized_phone` labels plus insertion phones ordered by annotation intervals. TRAIN contains 50,356 correct, 7,381 substitution, 1,825 omission target labels and 560 insertion-positive gaps with 620 inserted phones. Canonical intended sequences are scored separately. Target and gap labels/timing are used for supervision/evaluation only. The data audit, full architecture, receipts, per-speaker results and feature provenance are in [Agent A report](METHOD_BENCH_03_A_PHONEME_CTC.md).

## Recognition results

| Recognizer | TRAIN fit realized PER, greedy / beam-10 | Held-speaker TRAIN realized PER, greedy / beam-10 | DEV_CAL realized PER, greedy / beam-10 | DEV_SEARCH realized PER, greedy / beam-10 |
|---|---:|---:|---:|---:|
| Log-mel BiGRU, seed 101 | .7181 / .6240 | .7899 / .7207 | .7577 / .6664 | .7613 / .6802 |
| Frozen W2V2 layer 6 + linear CTC, mean seeds 101/202/303 | .3961 / .3215 | .4220 / .3303 | .4177 / .3458 | .3988 / .3216 |

W2V2 beam-10 per-seed PER:

| Seed | TRAIN fit | HJK/HKK held out | DEV_CAL | DEV_SEARCH |
|---:|---:|---:|---:|---:|
| 101 | .3193 | .3276 | .3472 | .3208 |
| 202 | .3149 | .3229 | .3355 | .3112 |
| 303 | .3302 | .3405 | .3547 | .3328 |
| Mean (sample SD) | .3215 (.0079) | .3303 (.0091) | .3458 (.0097) | .3216 (.0108) |

Seed-specific W2V2 posteriors and event scores are preserved separately. The W2V2 recognition gain over log-mel repeats across all three seeds. Beam decoding improves PER, but W2V2 remains substantially worse than the annotation-derived canonical sequence diagnostic: realized-reference canonical-sequence PER is .1529 TRAIN, .2320 CAL and .1013 SEARCH. That diagnostic uses gold annotations and is not a deployable inference baseline.

## Blank behavior and checkpoint audit

Log-mel greedy blank argmax fraction is .916–.932 across splits and outputs are deletion-heavy. W2V2 seed 101 blank argmax fraction is .687–.722; its PER is materially better but still high. The published `facebook/wav2vec2-lv-60-espeak-cv-ft` model was recovered at revision `fdf22f0210322b799533bc499dd3e489a9452bdc` (1,263,535,127 bytes, SHA-256 `3173bde9e9ce490fa0f989e413c42f25bc1820c020adc1e6b9b87025b3cfcc5e`, Apache-2.0 metadata). Its 392-token IPA/eSpeak output loaded and ran on CPU, but composite/contextual token mapping to the 42 LANGFENS phones is unvalidated; no LANGFENS PER is claimed.

The current NVIDIA driver was unavailable to `nvidia-smi`; all feature extraction, training and evaluation reported here ran on CPU. The pretrained IPA probe used roughly 2 GiB peak RSS. The three W2V2 projection-head seeds reuse the same cached continuous hidden-state arrays; they are independent training seeds, not independent encoder runs.

## Reproduction and limits

Scripts, checkpoints, input/posterior arrays, tokenizer inventory and hashes are under `experiments/rq1_method_bench_03/agent_a/`. Commands and environment map: [reproduction report](METHOD_BENCH_03_REPRODUCTION.md). W2V2 three-seed MDD impact is evaluated separately in [probabilistic edit localization](METHOD_BENCH_03_PROBABILISTIC_EDIT_LOCALIZATION.md). DEV_SEARCH was exposed and is exploratory. PER improvement does not establish MDD improvement; downstream AP is weak relative to historical models and specialist score compositions.
