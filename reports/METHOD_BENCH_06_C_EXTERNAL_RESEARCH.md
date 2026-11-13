# METHOD-BENCH-06 Agent C — external method and grouped benchmark

## Verdict

**Experiment executed; the MixGoP-inspired adaptation did not outperform the matched direct neural target head.** It did produce an interpretable, non-CTC acoustic likelihood baseline under the four established speaker folds. This is a measured negative result, not a reproduction of the paper's benchmark.

## External discovery

Search date: 2026-11-08 (latest date covered by this search; not a claim of exhaustive coverage through the end of November). I searched primary ACL Anthology and ISCA Archive records for phoneme-level pronunciation assessment and frozen speech-representation methods, rather than extending the established E4/U2/CTC recognition agenda. I prioritized inference objectives that could be adapted to available TRAIN target crops and labels. This is focused discovery, not systematic review.

| Work | Mechanism and evidence in source | Relevance / distinction |
|---|---|---|
| Choi et al., NAACL 2025, [Leveraging Allophony in Self-Supervised Speech Models for Atypical Pronunciation Assessment](https://aclanthology.org/2025.naacl-long.132/) | MixGoP replaces phoneme-classifier posterior scoring with a phone-conditional Gaussian-mixture likelihood: multiple components represent within-phone/allophonic variation, without softmax normalization across phones. The paper evaluates atypical pronunciation on five datasets and reports strongest results on four; its described setup uses 32 mixture components and subsamples 512 features per phoneme. | Selected method: intended-phone-conditioned acoustic normality can be scored without CTC decoding. We adapt only density scoring using cached LANGFENS event crops, not the source representation, sampling/fitting recipe, or task. |
| Vidal, Riera & Ferrer, SLaTE 2023, [Mispronunciation detection using self-supervised speech representations](https://www.isca-archive.org/slate_2023/vidal23_slate.pdf) | Compares a downstream phone-recognition pipeline with a direct phone-level mispronunciation detector built on self-supervised representations, evaluated on L2-Arctic and EpaDB. | Distinct non-CTC direct-supervision route; not implemented here. The neural reference below is prior LANGFENS work, not a reproduction of Vidal et al. |
| Korzekwa et al., Interspeech 2021, [Weakly-supervised word-level pronunciation error detection in non-native English speech](https://www.isca-archive.org/interspeech_2021/korzekwa21b_interspeech.pdf) | Canonical-text-conditioned word error detection jointly with an attention phoneme recognizer; paper reports word-level AUC on two L2 corpora. | Candidate for limited-label temporal modeling, but its word-level objective and recognition joint-training are outside this small adaptation. |

The selected paper's ACL Anthology primary record and linked PDF were consulted. Our inference unit is a canonical target-phone event: the reference phone selects its fitted GMM and negative log likelihood is the error score. The paper's atypical-pronunciation task is not equivalent to LANGFENS categorical S/O prediction. This adaptation tests only whether its phone-conditioned density-ranking idea is feasible/useful here; it does not reproduce the source task or claim equivalence. We omit the paper's feature extractor, frame treatment/pooling, mixture count, sampling protocol, and datasets. H-002 is a frozen ASR-finetuned wav2vec2 representation per MB05 provenance, not the pretraining-only representation used in the paper.

## Executed experiment

**ID:** `RQ1-MB06-C-MIXGOP-ADAPT-GROUPED-S101`  
**Source:** `experiments/rq1_method_bench_06/agent_c/mixgop_adaptation.py`  
**Receipt and OOF scores:** `experiments/rq1_method_bench_06/agent_c/mixgop_receipt.json`, `mixgop_train_oof_scores.npz`

For target embedding vector \(z\) and intended phone \(p\), each fold fits a PCA transform on outer-training features, then a diagonal GMM only on outer-training *correct* events with reference phone \(p\). The error score is negative log density:

\[
s(z,p)=-\log\sum_{m=1}^{K}\pi_{pm}\,\mathcal N(z;\mu_{pm},\operatorname{diag}(\sigma^2_{pm})).
\]

Higher scores indicate lower likelihood under the typical/correct phone model. (K=1) is a unimodal control; (K=4) tests whether subclusters help. The code uses the mean of H-002 frames `[3:14]` per target, fold-fitted 24-dimensional whitened PCA, and one phone model per observed class. It does not use CTC, realized-phone annotations, error labels for model fit, gold timing, DEV_CAL, or DEV_SEARCH. All 37 represented phone classes had GMMs in each fold.

### Grouped TRAIN OOF results

| Model | Fold 0 S/O AP | Fold 1 S/O AP | Fold 2 S/O AP | Fold 3 S/O AP | Pooled AP-S | Pooled AP-O |
|---|---:|---:|---:|---:|---:|---:|
| GMM K=1 | .1746 / .0755 | .1765 / .1033 | .2131 / .1956 | .2091 / .0898 | .19170 | .11509 |
| GMM K=4 | .1761 / .0870 | .1741 / .1042 | .2165 / .2181 | .2112 / .0936 | .19307 | .12649 |
| MB05 direct neural C/S/O head | — | — | — | — | .423209 | .276706 |

Supports are the shared TRAIN totals: 7,381 substitutions and 1,825 omissions among 59,562 targets. Fold memberships were ABA/ASI/BWC; EBVS/HJK/HKK; HQTV/LXC/NJS; PNV/SKA/SVBI. The neural comparator is the historical MB03 four-fold grouped TRAIN OOF target-head reference (also documented in `reports/METHOD_BENCH_04_FINAL_SCIENTIFIC_REPORT.md`), not a newly trained matched-run baseline; it reported S/O/I AP `.423209/.276706/.162125`. Its S/O results are relevant target-task context, not a controlled one-factor comparison: representation, optimization and fitting differ. It additionally has insertion output while this density model does not. GMM K=4 modestly exceeds K=1 in pooled AP by .00136 S and .01140 O, though fold behavior is mixed. Both are below the direct target-head reference.

This method scores targets only. It has no insertion output and does not support AP-I, Macro-3 AP, Macro-3 F1, or Macro-4 F1. No operating threshold or calibration was fit. No independent seed confirmation was run because the first screen was substantially below its matched neural reference.

## Execution and reproduction

**Status:** VERIFIED local execution.  
**Device:** CPU (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`); no CUDA job or checkpoint.  
**Runtime:** 4.34 seconds in the verified rerun recorded in `mixgop_receipt.json`; peak memory was not measured.  
**Inputs:** `research/long_running_method_discovery/cycle_02/h002_ssl/normalized_v1/{train_frame_context.npy,train_reference.npy,train_labels.npy,train_event_ids.jsonl}`.  
**Output identity:** seed schedule `101 + fold`; exact grouped predictions saved in compressed NPZ with labels, references and speaker IDs.

From repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u experiments/rq1_method_bench_06/agent_c/mixgop_adaptation.py
```

## Interpretation and next direction

The multi-component acoustic density has measurable ranking signal above prevalence, but it is not competitive with supervised direct C/S/O prediction on these target events. K=4's modest omission gain over K=1 is concentrated in fold 2, so it is not sufficient evidence that allophonic subclusters transfer consistently to unseen speakers. The next useful test would need a representation/segmentation change justified by the task or more faithful original MixGoP features and controls; merely increasing the number of Gaussians is not justified by this screen. No method is proposed as a unified S/O/I model from this result.
