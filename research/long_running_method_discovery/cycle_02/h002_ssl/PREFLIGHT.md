# H-002 frame-level SSL features — preflight and experiment contract

**Status: BLOCKED for neural experiment / prepared from CPU-side evidence.** No frame extraction, training, or inference ran in this workstream.

## Current control and feature assets

The Phase 3 control is a shared biGRU-64 student with target and gap heads (78,661 parameters). Its cached inputs are not frame states: `target_X` has shape `(59562, 51)` and `gap_X` has shape `(61362, 6)` for TRAIN. DEV_CAL shapes are `(10043,51)` and `(10343,6)`; DEV_SEARCH shapes are `(19755,51)` and `(20354,6)`. The target cache concatenates 39 reference-phone one-hot fields and 12 CTC/acoustic summaries. The gap cache contains six summary fields. Existing Phase 3 report says these are insufficient to recover contextual frame trajectories.

Feature cache SHA-256:

| Artifact | SHA-256 |
|---|---|
| `data/derived/l2arctic/features/train_features.npz` | `e85cc2844c5b5b785e9f3b8b9b424ec60c5979829a1511a6fd804d1761ec9fd1` |
| `data/derived/l2arctic/features/dev_cal_features.npz` | `29af1e5781c8c36dc854232739beeb072516d88e45fd8bac9a11ce4508c710be` |
| `data/derived/l2arctic/features/dev_search_features.npz` | `8d6a208e6778ae3448aa4ef5b550cd56f3db2af6ef3511a5f0bd3417128e27e3` |

`tools/extract_acoustic_features.py` transiently computes logits/log-probabilities/probabilities from `charsiu/en_w2v2_fc_10ms`, but stores only pooled summaries. That model is an ASR/phone classifier and not an untouched self-supervised encoder. A local Hugging Face snapshot `facebook/wav2vec2-base-960h` is present with a `Wav2Vec2ForCTC` config and ASR tokenizer/head; it is also ASR-finetuned. Its config has 12 transformer layers, hidden size 768. These can form a separately labeled fine-tuned-encoder sensitivity arm, but cannot honestly stand in as the primary pure-SSL checkpoint. The inspected cache did not show HuBERT or a separate unfine-tuned wav2vec2 pretraining checkpoint. This is a bounded inventory search, not proof that no alternate storage path exists.

## Falsifiable matched test

Primary claim: contextual frame representations improve error ranking over summary features when model capacity and TRAIN budget are controlled.

1. **A control:** reproduce matched Gold-Only biGRU from cached 51D target and 6D gap sequences, same speaker-grouped TRAIN fold, seeds and update budget.
2. **B single intervention:** frozen layerwise frame states from an authorized locally available *pretraining-only* SSL checkpoint; first test a single frozen intermediate layer selected by TRAIN-only grouped validation. Use the same lightweight target/gap heads and optimizer budget. No large Conformer.
3. **C pooling ablation:** compare fixed local Gaussian pooling and reference-conditioned local attention against a simple temporal pooling baseline. Any such comparison after B is a secondary registered ablation; do not bundle multiple changes into the A→B causal comparison.
4. **Reference/acoustic controls:** reference-only features; acoustic-only pooled frames; full audio+reference; and a label-free acoustic perturbation sensitivity check on paired audio. Gold target boundaries must never drive pooling or attention. Use the frozen CTC-derived alignment only as a model-generated input if the exact feature contract already supports it; otherwise let reference-conditioned attention predict a monotonic local window without gold spans.
5. **Evaluation:** preserve event keys and full populations; score all 19,755 target slots and 20,354 exact gaps on DEV_SEARCH only after selection is frozen on TRAIN/DEV_CAL. Use `UnifiedEvaluatorV3`; publish per-class AP, Macro-3 Error AP, precision/recall/F1, Macro-3 Error F1, Macro-4 F1, reliability, per-speaker summaries, acoustic sensitivity, parameters, runtime, and peak allocated VRAM. DEV_SEARCH is exposed and exploratory, not an independent confirmation set.

Falsify the hypothesis if the matched frame model gives no stable S/O AP improvement or its gain disappears under the audio-dependence controls / is explained by the reference-only control. Report I changes and speaker heterogeneity even if S/O improve. Do not choose layer, pooling, threshold, or architecture by DEV_SEARCH.

## Literature evidence

- Pasad, Chou & Livescu, “Layer-wise Analysis of a Self-supervised Speech Representation Model” (2021), analyzes intermediate wav2vec 2.0 representations and reports that phone information is salient around Base layers 6–7, while contextual and word information varies by layer. This supports a TRAIN-selected layer probe, not assuming the last layer is best. [Primary paper](https://arxiv.org/abs/2107.04734).
- English, Kelleher & Carson-Berndsen, “Domain-Informed Probing of wav2vec 2.0 Embeddings for Phonetic Features” (SIGMORPHON 2022) reports contextual transformer representations improve phonetic feature capture on its probing setup. It does not establish mispronunciation detection gains. [ACL Anthology paper](https://aclanthology.org/2022.sigmorphon-1.9/).
- Baevski et al., “wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations” (NeurIPS 2020) establishes the masked latent contrastive pretraining framework. Fine-tuning status must be tracked separately when reusing an ASR checkpoint. [NeurIPS paper](https://papers.nips.cc/paper_files/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html).
- Dieck et al., “Wav2vec behind the Scenes: How end2end Models learn Phonetics” (Interspeech 2022) reports phonetic information in hidden activations, while contextual layers also carry language-related patterns. This motivates explicit lexical/reference-only controls. [ISCA Archive paper](https://www.isca-archive.org/interspeech_2022/dieck22_interspeech.html).

## Blockers and exact status

- `nvidia-smi` fails to communicate with the NVIDIA driver. No feature extraction, forward pass, training, model selection, metrics, CUDA receipt, or checkpoint was produced.
- The local candidate Wav2Vec2 checkpoints are ASR/phone-task finetuned, not pure SSL weights; no HuBERT checkpoint was found in the examined HF cache.
- No frame-level hidden-state cache was found. Existing NPZs contain only fixed-size summaries.
- R2 was launched in this session with the smaller `gpt-6-luna` model through the native collaboration mechanism; this remains CPU-side preparation, not an executed experiment.
- Dataset reference phones are annotation-derived; independence remains unverified.

**Decision:** unresolved; no support/contradiction claim. The next minimum step is to identify/authorize local pretraining-only weights and a live CUDA lease, then cache frozen features once and run the matched control/intervention on speaker-grouped TRAIN folds.
