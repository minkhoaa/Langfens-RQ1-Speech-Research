# METHOD-BENCH-07 Agent C — phonological-attribute method and grouped TRAIN experiment

## Result

Executed `RQ1-MB07-C-PHONOLOGICAL-ATTRIBUTE-DEVIATION-S707`, a four-fold speaker-held-out TRAIN adaptation of the phonological-feature detection/difference method of Shahin & Ahmed (Interspeech 2024). Pooled AP is **S .23462 / O .12643** (supports 7,381 / 1,825; priors .12392 / .03064). This exceeds prevalence but is far below the existing direct MLP reference (.38302/.27410; MB05 C) and the E4 mean reference (.31972/.25936; MB05 C). This method produces only a target-event error score; no insertion score or unified S/O/I metric is defined. Negative result, not a paper reproduction.

## Primary source and distinction

Shahin, Mostafa & Beena Ahmed. “Phonological-Level Mispronunciation Detection and Diagnosis.” *Interspeech 2024*, pp. 307–311, DOI: [10.21437/Interspeech.2024-2217](https://doi.org/10.21437/Interspeech.2024-2217); [primary ISCA PDF](https://www.isca-archive.org/interspeech_2024/shahin24_interspeech.pdf). The paper argues that phone labels miss uncategorical deviations, and predicts binary phonological attributes from wav2vec2 representations; it compares recognized and canonical attribute sequences to diagnose feature insertions/deletions/replacements. Sec. 2.1, pp. 307–308: a pretrained SSL representation sequence feeds a linear output layer with one dimension per attribute; the paper models 35 attributes. The paper's described L2-ARCTIC setup uses 18 training speakers and six held-out speakers, and native LibriSpeech/TIMIT to support attribute learning (Sec. 2.2). The abstract reports lower FAR/FRR/diagnostic error than its phoneme baseline; those are not LANGFENS metrics.

**Code/replication status:** this campaign did not record or use a paper-linked public code repository or pretrained weights; public-code availability is therefore *unverified*, not asserted absent. The method was adapted from the primary paper's described objective and local available attribute table. This adaptation learns 16 binary ARPAbet articulatory attributes from correct TRAIN target crops only, then computes the mean absolute difference between predicted attribute probabilities and the canonical phone's fixed attributes. It is a phone-independent attribute detector followed by a feature-space mismatch score, unlike E4 CTC posterior scoring, the direct event-class MLP, temporal logistic regression, or phone-conditioned GMM. Fidelity is **partial/adaptation**: it uses the paper's attribute prediction + canonical-vs-recognized feature difference mechanism, but not the paper's 35-feature inventory, frame-sequence alignment, wav2vec2 representations trained/replayed in its recipe, auxiliary native corpora, or feature insertion/deletion localization. LANGFENS's local feature table has 16 dimensions, with some classes collapsed (e.g. phone pairs share feature vectors), so diagnosis is correspondingly coarse.

## Experiment and provenance

Code: `experiments/rq1_method_bench_07/agent_c/phonological_attributes.py`; receipt and scores: `phonological_attribute_receipt.json`, `phonological_attribute_train_oof.npz`; supervisor-generated, independently manifest-joined event-level file: `phonological_attribute_train_oof_keyed.npz` (SHA-256 `2a01732a6977b3970372396fdc7855025150a98250ab0fa913fb42fea8524383`). All 59,562 `(utterance_id,slot_idx)` keys were unique and joined to the authoritative TRAIN manifest; labels, speakers and class supports matched exactly. Inputs: cached H-002 `train_frame_context.npy`, `train_labels.npy`, `train_reference.npy`, `train_event_ids.jsonl` under `research/long_running_method_discovery/cycle_02/h002_ssl/normalized_v1/`. The acoustic detector input is the mean of frames `[3:14]` from the frozen 768-d H-002 event crop (ASR-finetuned wav2vec2-base-960h hidden representations); no reference is fed into the network. Reference-phone one-hot selects the expected 16-feature vector only at scoring. Per fold, standardization and a 256/128 MLP are fitted on correct examples from the nine training speakers, 10 epochs, BCE-with-logits, AdamW lr .001, batch up to 512; prediction on the other three speakers is soft attribute mismatch. Fold seeds 707+fold. This is an adaptation using categorical correct/error labels only to select correctly pronounced examples for attribute-target fitting; no substitution/omission labels train the attribute detector. No DEV_SEARCH fitting/tuning or protected data. CPU; runtime 26.03 s; peak memory not measured; no checkpoint.

| Fold held speakers | n | AP-S | AP-O |
|---|---:|---:|---:|
| ABA / ASI / BWC | 14,816 | .24979 | .09215 |
| EBVS / HJK / HKK | 14,848 | .19646 | .08440 |
| HQTV / LXC / NJS | 14,989 | .24148 | .22210 |
| PNV / SKA / SVBI | 14,909 | .25833 | .10289 |
| **Pooled OOF** | **59,562** | **.23462** | **.12643** |

Comparison context is not a matched one-factor ablation: the MB05 direct MLP trains on C/S/O labels, while this detector trains attributes only on correct targets. Against direct MLP this loses .14840 AP-S and .14767 AP-O. Against E4(mean) it loses .08510/.13293. Fold 2 has higher O AP than other folds, so the pooled outcome is speaker-sensitive. No thresholds were tuned and no F1 is claimed.

Reproduce from repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u experiments/rq1_method_bench_07/agent_c/phonological_attributes.py
```

## Limitations / next falsifiable test

The paper's feature inventory and sequence-level alignment are not replicated; the local handcrafted 16-feature table is imperfect and conflates multiple phones. A stronger faithful test should train frame-level attribute heads on native correct speech plus TRAIN correct targets, align attribute sequences against canonical attributes without gold realized timing, and compare held-speaker AP/diagnosis to same-fold E4 and direct models. Current lower scores do not falsify feature-based diagnosis in general. Inference depends on canonical phone identity and available fixed feature mapping; this is not calibrated, does not localize feature-level edits, and cannot assess insertions.
