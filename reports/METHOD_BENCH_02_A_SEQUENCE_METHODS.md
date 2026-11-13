# Method Bench 02 — Agent A: Sequence and Edit-Path Methods

## Scope and findings

This pass audited the existing H04/H05 artifacts and ran one constrained topology ablation on the frozen H-001 edit-path experiment. No neural model was trained and no CTC likelihood was inferred from frame-classifier scores.

H04 (`EXP-08-PHONOTACTIC-CONTEXT`) has a trained BiLSTM checkpoint and cached predictions, but those prediction files contain 500 target and 500 gap rows, while the current DEV_SEARCH population is 19,755 targets and 20,354 exact gaps across 599 utterances. The cached H04 predictions therefore cannot support a full-population EVAL-V3.1 replay on the current ledger. The old H04 review reports insertion AP 0.2016, omission AP 0.1725, macro AP 0.4991, and macro F1 0.5126; those are historical results from its older evaluation path and are not treated as directly comparable current EVAL-V3.1 evidence here.

Separately, the full verified waveform cache makes a fresh H04-style substitution screen feasible despite the 500-row historical H04 JSON. A frozen CPU screen was run over all 19,755 DEV_SEARCH targets from the cached CharSiU frame-classifier logits. It uses a forced canonical-phone alignment, a framewise hard mismatch score, an all-phone competitor margin, a TRAIN-only set of common realized-phone confusions, and a TRAIN-only per-phone substitution prior. All candidate sets/priors were derived from TRAIN; DEV_SEARCH was used only for descriptive EVAL-V3.1 metrics.

H05's proposed SpeechOcean762 transfer run does not have local `data/original_sources/speechocean762` files or generated predictions. A local Hugging Face snapshot named `facebook/wav2vec2-lv-60-espeak-cv-ft` is a genuine phone-oriented CTC model by its local metadata: `Wav2Vec2ForCTC`, a 392-token vocabulary, `Wav2Vec2PhonemeCTCTokenizer`, eSpeak `en-us`, and IPA phone tokens. The local snapshot does **not** include model weights—only `config.json`, preprocessor/tokenizer configuration, `vocab.json`, and the special-token map—so it is not locally runnable. It is a plausible route for H05 once weights and SpeechOcean762 audio are available, subject to mapping L2-ARCTIC's phone inventory to the model's IPA tokens. No network download was attempted.

As a local exact-gap feasibility screen, an additional frame-decoder adaptation greedily collapses adjacent argmax frame labels, removes SIL spans, and edit-aligns decoded phone spans to the intended sequence. This is separate from SpeechOcean762 transfer. It uses no realized-phone or gold-timing information to build predictions. A surplus phone is assigned to a gap only if every minimum-edit alignment places it at that same gap; otherwise affected gaps abstain.

The current H-001 alignment artifacts are from `charsiu/en_w2v2_fc_10ms`, configured as `Wav2Vec2ForFrameClassification`, not a verified CTC-trained phone model. Its frame scores can support constrained max-sum sequence-path comparisons, but must not be called CTC likelihoods. The existing H-001 A/B run compares forced-path evidence against a phone-deletion path retaining adjacent SIL states. This pass adds C: the same per-phone deletion path with every run of adjacent SIL states collapsed to one SIL state. It retains the same Gold-Only substitution and exact-gap insertion heads and fits its omission mapping on TRAIN only.

## Controlled ablation results

The run verified 5,398 local cache files, reproduced the existing H-001 source population exactly, and evaluated all 599 DEV_SEARCH utterances with 19,755 target events and 20,354 exact gaps under EVAL-V3.1. The gold counts are 1,768 substitutions, 209 omissions, and 125 insertions. DEV_SEARCH remains development-exposed; these results are exploratory and no method or parameter was selected from its labels.

| DEV_SEARCH arm | Omission AP | Substitution AP | Insertion AP | Macro-3 error AP | Macro-3 error F1 | Macro-4 F1 | Binary MD F1 | False warnings | Joint FWR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A: forced Viterbi | 0.03677 | 0.36460 | 0.18672 | 0.19603 | 0.21586 | 0.39320 | 0.39899 | 1,735 | 2.8965 |
| B: optional phone, retain adjacent SIL | 0.08793 | 0.36460 | 0.18672 | 0.21308 | 0.21565 | 0.39313 | 0.39890 | 1,720 | 2.8715 |
| C: optional phone, collapse adjacent SIL | 0.08439 | 0.36460 | 0.18672 | 0.21190 | 0.21944 | 0.39611 | 0.40083 | 1,702 | 2.8414 |

C versus B lowers omission AP by 0.00354 (about 4.0% relative) and macro-3 error AP by 0.00118, while raising macro-3 error F1 by 0.00379 and reducing false warnings by 18. B and C each improve omission AP over forced Viterbi A by 0.05117 and 0.04762 respectively. C does not dominate B; collapsing the duplicate SIL states is not supported as an AP improvement. Omission F1 is 0.00939 for C, versus 0 for both A and B, under the shared TRAIN-fitted calibrator/argmax policy. Substitution and insertion scores are unchanged by design. EVAL-V3.1's binary misdiagnosis score semantics are the validated target-simplex sum `p_substitution + p_omission`; the omission ranking score is the raw path evidence, not that probability.

### H04-style frozen GOP score screen

S-class AP uses the full target population under EVAL-V3.1. Raw S scores are compared while retaining the unchanged Gold-Only probability simplex and decisions. The evaluator's binary misdiagnosis score remains `p_substitution + p_omission`.

| Score source | S AP | S F1 under unchanged Gold-Only argmax | Description |
|---|---:|---:|---|
| Gold-Only S probability | 0.36460 | 0.41388 | Reference head from the frozen Gold-Only checkpoint |
| Hard frame mismatch | 0.14508 | 0.41388 | Fraction of forced-aligned frames whose top label differs from the intended phone |
| Canonical vs all phones | 0.18817 | 0.41388 | Best mean non-SIL, noncanonical log posterior minus canonical mean log posterior |
| TRAIN-confusion alternatives | 0.19487 | 0.41388 | Same margin using up to five common TRAIN annotated realized-phone substitutions per intended phone; candidate coverage was 19,755/19,755 |
| TRAIN per-phone prior | 0.29923 | 0.41388 | Smoothed TRAIN substitution prevalence by intended phone, with no acoustic input |

The best frame-score S ranking (0.19487 AP) is well below the Gold-Only S head (0.36460); the prior-only control (0.29923) also outperforms both acoustic GOP-style rankings. This is negative evidence for these simple frame-aggregated alternatives on this population. The forced-path candidate comparison is target-aligned, not an unrestricted phone-sequence decoder. These are frame-classifier GOP-style discriminative scores, not CTC likelihoods. Details and input hashes are in [the H04 receipt](/home/khoa/Projects/New-Method-Research/experiments/rq1_method_bench_02/agent_a/frame_gop_run_v2/EVALUATION_RECEIPT.json) and [the implementation](/home/khoa/Projects/New-Method-Research/experiments/rq1_method_bench_02/agent_a/frame_gop_substitution.py).

### Frame-decoded extra-phone exact-gap screen

This uses the same 599 DEV_SEARCH utterances, 19,755 targets, and all 20,354 exact gaps. It finds 568 surplus decoded phone spans whose gap assignment is invariant across optimal edit paths. The deterministic ambiguity audit finds 1,015 ambiguous gap events (4.99% of the gap population) across 359 utterances with multiple equal-cost optimal paths. Ambiguous gaps abstain; no gold label is used to break ties.

| Metric | Frame-decoded extra-phone score | Gold-Only gap head reference |
|---|---:|---:|
| Exact-gap insertion AP, full population | 0.02825 | 0.18672 |
| Insertion F1 under decoded gap decisions / reference head | 0.08483 | 0.23370 |
| Insertion precision | 0.05328 | 0.32441 |
| Insertion recall | 0.20800 | 0.54107 |

On the unambiguous-gap subset, insertion AP is 0.03448; on the ambiguous-gap subset it is 0.02660. Full-population EVAL-V3.1 AP uses the raw count of unambiguous extra-phone spans assigned to each gap. For calibration, the unchanged Gold-Only gap probability remains present, so its Brier/ECE are not evidence that the frame-decoder count is calibrated. The screen is negative: this framewise decoding adaptation is substantially below the existing gap head's AP and should not be treated as an insertion model. It is neither CTC decoding nor a CTC likelihood.

## Reproduction

Command run:

```sh
.venv/bin/python experiments/rq1_method_bench_02/agent_a/single_gap_ablation.py
```

The system `python` lacked `numba`; the repo `.venv` Python 3.11 environment was used. The script writes all generated artifacts under `experiments/rq1_method_bench_02/agent_a/single_gap_run/`. The run receipt is [EVALUATION_RECEIPT.json](/home/khoa/Projects/New-Method-Research/experiments/rq1_method_bench_02/agent_a/single_gap_run/EVALUATION_RECEIPT.json); the implementation is [single_gap_ablation.py](/home/khoa/Projects/New-Method-Research/experiments/rq1_method_bench_02/agent_a/single_gap_ablation.py).

H04 screen command:

```sh
.venv/bin/python experiments/rq1_method_bench_02/agent_a/frame_gop_substitution.py
.venv/bin/python experiments/rq1_method_bench_02/agent_a/frame_decode_insertions.py
```

Its outputs are isolated under `experiments/rq1_method_bench_02/agent_a/frame_gop_run_v2/`.
Insertion decode outputs are under `experiments/rq1_method_bench_02/agent_a/frame_decode_run_v5/`.

Frame-decoded insertion screen command:

```sh
.venv/bin/python experiments/rq1_method_bench_02/agent_a/frame_decode_insertions.py
```

The final ambiguity-audited output is [EVALUATION_RECEIPT.json](/home/khoa/Projects/New-Method-Research/experiments/rq1_method_bench_02/agent_a/frame_decode_run_v5/EVALUATION_RECEIPT.json); the phone-to-gap ledger is in that folder's `decoded_insertions.csv`.

## Limits

- C uses a frame classifier's per-frame phone scores as discriminative path potentials. It is neither a CTC forward score nor a generative sequence likelihood.
- The canonical intended phone sequence comes from the annotation-derived manifest; independent reference lineage is not established. No gold timing, realized-phone labels, or error labels were used to construct path scores.
- The C topology collapses adjacent SIL runs but does not resolve whether the deleted phone's surrounding silence should map to a particular exact gap event. Insertion scoring remains the unchanged learned gap head.
- Frame-decoded extras are assigned by sequence edit distance only. Repeated phones can yield multiple equally good alignments; the final frame-decoder run abstains for all gaps touched by ambiguous paths, and its unambiguous insertion AP remains low. Mapping multiple extra spans to one gap is represented as a count score, not multiple independent events.
- The one-dimensional omission calibrator was fit on TRAIN and is in-sample there. DEV_SEARCH was not used to fit it or choose the topology.
- H05 remains blocked on local weights, local SpeechOcean762 data, and a verified cross-inventory phone mapping; this was an artifact audit only.
