# METHOD-BENCH-05 Agent C — Direct-event acoustic models (non-CTC)

## Scope and status

This independent workstream asked whether direct acoustic error-event classification can supply useful evidence without first decoding a CTC phone sequence. The experiment was run on the four authoritative speaker-held-out TRAIN folds under EVAL-V3.1. The selected model scores target substitutions (S) and omissions (O); it has no exact-gap insertion output. It does not settle insertion localization.

**Verdict:** a small direct C/S/O MLP transfers useful target-event ranking above prevalence. On the same grouped TRAIN target population it exceeds Agent B seed101 canonical E4(mean) AP-S/O by `.06330/.01474`, but this is not a one-factor architecture comparison; it remains slightly below the earlier grouped-OOF supervised neural target head. E4(max) is reported only as an aggregation sensitivity check. Ablations show both audio and reference-phone identity matter: audio-only has stronger omission AP than reference-only; combining them yields the strongest scores in this experiment. The reference-only substitution AP is substantial, so the full model's S ranking cannot be attributed to acoustic evidence alone. It has no insertion output.

## External discovery and method shortlist

Searches were performed 2026-11-08 with queries covering direct MDD from self-supervised speech representations, weakly supervised pronunciation-error detection, jointly learned acoustic/phoneme alignment, and allophonic modeling. Primary paper pages/PDFs were opened where accessible. This is a focused discovery, not a systematic or exhaustive review.

| Work | Published mechanism / reported evidence | Relevance and boundary for LANGFENS |
|---|---|---|
| Vidal, Riera & Ferrer, SLaTE 2023, [Mispronunciation detection using self-supervised speech representations](https://www.isca-archive.org/slate_2023/vidal23_slate.pdf) | Directly compares downstream phone recognition with a downstream model trained for phone-level mispronunciation detection on non-native annotated data, across L2-Arctic and EpaDB; authors report the direct target-task route performs best overall. | Closest support for testing a direct supervised target-event head against recognition-first scores. Their metrics/populations differ; no result is transferred to LANGFENS. Our MLP is an adaptation, not a reproduction. |
| Korzekwa et al., Interspeech 2021, [Weakly-supervised word-level pronunciation error detection in non-native English speech](https://www.isca-archive.org/interspeech_2021/korzekwa21b_interspeech.pdf) | Canonical-text-conditioned word-error model trained jointly with an attention phoneme recognizer; the paper reports improved word-level AUC on two L2 corpora. It motivates auxiliary recognition as regularization when direct error labels are limited. | Independent route toward direct error modeling with temporal attention and weaker supervision. We did not implement its word-level objective or auxiliary recognizer. |
| Yu et al., ICASSP 2022, [Phoneme Mispronunciation Detection by Jointly Learning To Align](https://ieeexplore.ieee.org/document/9746727/) | Indexed publisher abstract describes acoustic and canonical-phone encoders with attention fusion and joint alignment/MDD training; reports a public L2-ARCTIC evaluation. | Joint learned alignment is a plausible alternative to fixed forced timing, but is nearer to Agent A's temporal alignment question. Full publisher text was unavailable in this pass; implementation details beyond the abstract are not asserted or used. |
| Choi et al., NAACL 2025, [Leveraging Allophony in Self-Supervised Speech Models for Atypical Pronunciation Assessment](https://aclanthology.org/2025.naacl-long.132/) | MixGoP fits multi-subcluster Gaussian mixture models to frozen SSL phone features to model within-phone variation; the paper reports best performance on four of five dysarthric/non-native datasets. | A strong alternative to single phone prototypes and a future candidate for phone-conditioned normality models. It is a phoneme-distribution method, not the direct event classifier implemented here; no claim that our model reproduces it. |
| Shahin & Ahmed, Interspeech 2024, [Phonological-Level Mispronunciation Detection and Diagnosis](https://www.isca-archive.org/interspeech_2024/shahin24_interspeech.pdf) | Models pronunciation through articulatory/phonological attributes rather than attempting to enumerate all categorical phone confusions. | Suggests an interpretable acoustic attribute route. It was not implemented here and is not represented by the current S/O metrics. |

The selected experiment is most closely motivated by Vidal et al.'s direct error-supervision comparison. The model here receives TRAIN EVAL-V3.1 target labels during fitting, unlike weakly supervised WEAKLY-S; it also does not learn the paper's joint alignment, MixGoP distributions, or phonological attribute targets. These distinctions matter when comparing published claims with our measured result.

## Implemented experiments

### Primary direct model

**ID:** `RQ1-MB05-C-DIRECT-MLP-OOF-S101`  
**Source:** `experiments/rq1_method_bench_05/agent_c/direct_event_mlp.py`  
**Receipt:** `experiments/rq1_method_bench_05/agent_c/direct_event_mlp_receipt.json`

For each target, the input is a frozen H-002 layer-6 embedding crop (11 frames, offsets 3:14), summarized by temporal mean, standard deviation, and end-minus-start delta (2,304 acoustic values), plus the intended-phone one-hot (39 values). H-002 uses ASR-finetuned `facebook/wav2vec2-base-960h` hidden states, not a pretraining-only SSL checkpoint. The 17-frame cached crop was centered by a local frozen CharSiU frame-classifier plus intended-phone monotonic Viterbi alignment; the extraction receipt records no use of gold timing, realized phones, or labels in features. The model pools the inner 11 frames into three statistics and a two-hidden-layer MLP predicts C/S/O. The four authoritative folds fit feature standardization and model weights on nine TRAIN speakers and score the three held-out speakers. No held-out labels enter fitting. Inputs are WAV-derived cached representations and canonical reference phones; the inference features contain neither gold realized phones nor gold annotation boundaries. The frozen feature extractor itself was not retrained in this experiment.

Training was CPU-only, four folds × 12 epochs, batch 512, AdamW learning rate 0.001, hidden widths 256/128, weighted cross-entropy weights `[1, 2.5, 4]`, seeds 101+fold. Runtime was 35.42 seconds. Peak memory was not measured. No CUDA lease/checkpoint was used. The method creates no gap events, so I, Macro-3 AP, Macro-4 F1, and insertion PR are unsupported and are intentionally not invented.

| Held-out TRAIN speakers | S support | O support | AP-S | AP-O |
|---|---:|---:|---:|---:|
| ABA / ASI / BWC | 1,859 | 357 | .41236 | .24459 |
| EBVS / HJK / HKK | 1,524 | 301 | .31944 | .17574 |
| HQTV / LXC / NJS | 2,079 | 773 | .39824 | .37264 |
| PNV / SKA / SVBI | 1,919 | 394 | .40154 | .27812 |
| **Concatenated TRAIN OOF** | **7,381** | **1,825** | **.38302** | **.27410** |

TRAIN prevalence / prior-only AP is .12392 for S and .03064 for O. Pooled macro-2 AP is .32856. It is not a Macro-3 result.

### Acoustic/reference ablation controls

**ID:** `RQ1-MB05-C-DIRECT-MLP-ACOUSTIC-REFERENCE-ABLATION-S101`  
**Source/receipt:** `experiments/rq1_method_bench_05/agent_c/direct_event_mlp_controls.py`, `direct_event_mlp_controls_receipt.json`

These controls use the same fold definition, architecture, optimization budget, loss weights, and seed schedule, retraining each fold head from scratch. Acoustic-only retains the 2,304 temporal summary values and removes the reference one-hot. Reference-only retains the 39-d intended-phone one-hot and removes all acoustic features.

| TRAIN OOF input | AP-S | AP-O | Support S/O |
|---|---:|---:|---:|
| Full acoustic + reference (primary) | .38302 | .27410 | 7,381 / 1,825 |
| Acoustic only | .29915 | .18003 | 7,381 / 1,825 |
| Reference phone only | .30966 | .08945 | 7,381 / 1,825 |

This control indicates that acoustic features improve O ranking relative to reference-only and that the combination performs best. Reference-only S AP (.30966) shows that phone-conditioned label base rates contribute materially to substitution ranking. These comparisons do not prove causal or fully speaker-invariant acoustic information: reference class priors can themselves transfer across speakers, and all scores are on the same TRAIN OOF folds.

### PR curves and operating diagnostics

`experiments/rq1_method_bench_05/agent_c/evaluate_direct_event.py` evaluates the saved OOF predictions and writes `direct_event_mlp_pr_curves.csv` (all unique score-threshold PR coordinates for S and O) and `direct_event_mlp_diagnostics.json`.

| Class | AP | P@top 1% / R@top 1% | P@top 5% / R@top 5% | P@top 10% / R@top 10% | P / R / F1 at 0.5 |
|---|---:|---:|---:|---:|---:|
| S | .38302 | .6812 / .0550 | .5190 / .2095 | .4358 / .3517 | .4298 / .3627 / .3934 |
| O | .27410 | .5017 / .1638 | .2679 / .4373 | .1796 / .5863 | .3978 / .2729 / .3237 |

For a TRAIN-only operating diagnostic, a threshold maximizing F1 on the other three folds' OOF scores was selected separately for each held fold, then applied to that fold. Pooled cross-fit S F1=.41945 (precision=.35306, recall=.51660, false-warning rate on correct targets=.13184); O F1=.34033 (precision=.31596, recall=.36877, false-warning rate=.02409). This is a cross-fit TRAIN policy estimate, not a threshold chosen on DEV_CAL or a reported test-set policy. At a fixed threshold .5 the three-way argmax accuracy is .83139, macro-F1=.54218, and C-vs-error F1=.90515; these threshold metrics remain distinct from AP.

## Matched neural context and interpretation

### Comparison with the Agent B canonical E4 grouped result

Agent B's seed101 canonical E4(mean) pooled four-fold TRAIN OOF scores are AP-S `.31972`, AP-O `.25936`, and exact-gap AP-I `.03159` on the same 59,562 target events and four held-speaker groups. Against this matched-population reference, the direct MLP's target AP `.38302/.27410` is higher by `.06330` AP-S and `.01474` AP-O. This is a matched-population comparison, not a one-factor architecture comparison: E4(mean) is a CTC sequence-likelihood event scorer using fold-valid CTC posteriors and its competitor construction, while the direct MLP is a supervised C/S/O model on frozen H-002 acoustic crops and intended-phone features. Their training supervision, feature extraction, scoring mechanisms, and candidate design differ.

As aggregation sensitivity, E4(max) scores AP-S `.32935`, AP-O `.25936`, AP-I `.03709`; it raises insertion AP relative to E4(mean) but does not change O AP and remains below the direct MLP on target S/O. Since the direct head has no exact-gap output, no direct-vs-E4 insertion or Macro-3 comparison is possible.

METHOD-BENCH-03's earlier grouped OOF supervised neural target head reported AP-S/O `.423209/.276706`. It remains higher than this direct MLP for both target classes, by `.04019` AP-S and `.00260` AP-O. The MB04 correct-phone cosine margin (`.211316/.165356`) is lower than this direct MLP for both S and O. These grouped TRAIN OOF comparisons provide context, not a controlled one-factor architecture ablation. No cross-population claim is made here.

The direct head does not require a decoded realized-phone sequence and therefore provides an independent target-event acoustic score path. The acoustic-only control remains above S/O prevalence, but the phone-only substitution baseline is strong. We therefore find evidence for acoustic contribution, especially for O, not evidence that direct modeling beats the existing supervised neural frontier. No DEV_SEARCH data or thresholds were used. No protected holdout was accessed.

## Reproduction and artifacts

From repository root, replay the primary run and the distinct input controls:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u experiments/rq1_method_bench_05/agent_c/direct_event_mlp.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u experiments/rq1_method_bench_05/agent_c/direct_event_mlp_controls.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python experiments/rq1_method_bench_05/agent_c/evaluate_direct_event.py
```

OOF probabilities: `direct_event_mlp_train_oof.npy`; ablations: `direct_event_mlp_acoustic_only_train_oof.npy`, `direct_event_mlp_reference_only_train_oof.npy`. PR coordinates and detailed fold/threshold metrics are in the evaluator outputs noted above. Inputs and event order are recorded in the receipt and are the cached H-002 `train_frame_context.npy`, `train_reference.npy`, `train_labels.npy`, and `train_event_ids.jsonl` under `research/long_running_method_discovery/cycle_02/h002_ssl/normalized_v1/`.

## Next justified experiment

The next direct-event experiment should preserve the four-fold grouped design, use frame-level acoustic features with learned temporal pooling or a small recurrent/attention encoder, and compare against the same C/S/O MLP and acoustic-only controls. It should train on TRAIN only, use EVAL-V3.1 events with intended reference input, and report per-fold plus pooled OOF AP. Insertion should remain a separate exact-gap task unless the model has an explicit inference-time gap representation and baseline. A MixGoP-style multi-cluster acoustic model is another scientifically distinct candidate; its phoneme-conditioned normality assumption must be tested against direct supervised heads rather than assumed superior.
