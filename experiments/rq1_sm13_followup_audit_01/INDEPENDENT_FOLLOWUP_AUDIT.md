# SM13 follow-up audit: vowel/stop risk, labels, and alignment

**Date:** 2026-11-13  
**Population:** TRAIN speaker-grouped OOF, seed 101 only; 12 speakers, 1,800 utterances, 59,562 target slots, 61,362 gap slots.  
**Data boundary:** This audit did not open or score `TEST_SEALED`. All policy searches below are diagnostic, not deployable policy selection.  
**Inputs:** Frozen Neural-S, Fixed-O and gap-I scores. Other substitution alerts, O and I remain frozen except in the explicitly optimistic single-stream upper-bound calculation.

## Verdict

The existing SM13 score does not support the preregistered illustrative gate of **at least 15% recall and at least 50% precision** for either vowels or stops at **joint FWR ≤ 0.20**, using one threshold per manner. This remains true even when every other alert stream is switched off and the entire false-warning allowance is assigned to one manner. The result is an empirical property of these frozen scores on exposed TRAIN OOF, not a Bayes-optimal bound or a guarantee against a new scoring model.

The older diagnostic report's claim of **582 same-phone substitutions** is inconsistent with both its own `DIAGNOSTIC_CONFUSIONS.json` and the TRAIN manifest. The correct normalized count is **838/7,381**, including **262/862 stop substitutions**. A subsequent raw TextGrid audit established that **836/838** have a perceived-phone `*` deviation marker that the parser strips; the other two differ in a trailing digit. None has literally identical raw target and perceived labels. These are not demonstrated annotation mistakes. A sensitivity calculation does not show that removing them would fix stop ranking.

Alignment centers and a sampled reconstruction of exact CharSiu Viterbi spans show mostly small positional error, with a minority of substantial errors. Alignment is unlikely to be the dominant explanation for 0% operating recall, though it cannot be ruled out for individual cases.

## 1. Frozen-score operating frontier

The frozen policy has 216 false warnings on TRAIN OOF, or 0.1200 per utterance. Its vowel and stop substitution alert counts are both zero. For each manner independently, we evaluated every attainable Neural-S threshold in `[0,1]`, retained frozen O/I and other-manner S decisions, and used the registered score-margin S/O arbitration. We optimized true positives subject to at most 360 total false warnings (1,800 × 0.20), with and without a 50% manner precision requirement.

| Policy on TRAIN OOF | Vowel TP/2,280 | Vowel FP | Vowel precision | Joint FWR | Stop TP/862 | Stop FP | Stop precision | Joint FWR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen | 0 (0%) | 0 | undefined | 0.1200 | 0 (0%) | 0 | undefined | 0.1200 |
| Best manner threshold, risk only | 81 (3.55%) | 143 | 36.16% | 0.1994 | 78 (9.05%) | 143 | 35.29% | 0.1994 |
| Best manner threshold, risk + precision ≥50% | 9 (0.39%) | 9 | 50.00% | 0.1250 | 22 (2.55%) | 19 | 53.66% | 0.1306 |

The vowel and stop rows are **separate counterfactuals**. They cannot both spend the same remaining false-warning budget simultaneously.

### Optimistic single-stream upper bound

We then switched **all other warnings off** and allocated the full allowance of 360 false warnings to one manner's Neural-S score. This deliberately sacrifices every other class and is more favorable to the target manner than any normal joint policy.

| Single manner only | TP/support | Recall | FP | Precision | FWR |
|---|---:|---:|---:|---:|---:|
| Vowel, risk only | 188/2,280 | 8.25% | 360 | 34.31% | 0.2000 |
| Vowel, precision ≥50% | 9/2,280 | 0.39% | 9 | 50.00% | 0.0050 |
| Stop, risk only | 140/862 | 16.24% | 355 | 28.28% | 0.1972 |
| Stop, precision ≥50% | 22/862 | 2.55% | 19 | 53.66% | 0.0106 |

Thus a threshold change alone cannot attain 15% recall and 50% precision on these frozen scores, even with the entire error budget. The values are **retrospective upper bounds for a single global threshold per manner on this empirical population**. Phone-specific or context-specific scoring is a different hypothesis.

### Speaker-held-out policy transfer check

A separate four-fold `GroupKFold` over the 12 TRAIN speakers selected a manner threshold on nine speakers' OOF scores and applied it unchanged to three held-out speakers. The underlying model scores were already speaker OOF; this additional split holds speakers out from **threshold choice**. All other alert streams stayed frozen.

| Train selection rule | Held-out vowel TP/FP | Held-out vowel precision/recall/FWR | Held-out stop TP/FP | Held-out stop precision/recall/FWR |
|---|---:|---|---:|---|
| Max recall, train FWR ≤0.20 | 94/151 | 38.37% / 4.12% / **0.2039** | 79/157 | 33.47% / 9.16% / **0.2072** |
| Max recall, train FWR ≤0.20 and train manner precision ≥50% | 4/10 | 28.57% / 0.18% / 0.1256 | 19/26 | 42.22% / 2.20% / 0.1344 |

The risk-only thresholds overshot the pooled 0.20 FWR cap after transfer. The precision-constrained thresholds also failed to preserve 50% precision on held-out speakers. With 12 speakers and four folds these are descriptive transfer checks, not narrow confidence intervals.

## 2. Label and score audit

`SUBSTITUTION` events with equal **normalized** `target_phone` and `realized_phone` occur **838** times in the TRAIN manifest, or **11.35% of all 7,381 substitution labels**. The already archived `DIAGNOSTIC_CONFUSIONS.json` independently records the same 838. By manner: **262 stop**, **24 vowel**, **552 other**. Common normalized pairs are `R→R` 321, `T→T` 149, `D→D` 79 and `ER→ER` 26. The older report instead says 582 overall, `R→R` 344 and `ER→ER` 148; those prose figures should not be reused.

The original diagnostic `compute_manner_metrics` treats every gold `SUBSTITUTION` as positive. Its AP and recall tables therefore **include** the equal-phone rows; they were not excluded from the published manner metrics.

| Sensitivity calculation on TRAIN OOF | Vowel | Stop |
|---|---:|---:|
| Equal-phone / gold substitution support | 24/2,280 (1.05%) | 262/862 (30.39%) |
| Neural-S AP with current labels | 0.2348 | 0.1968 |
| Neural-S AP after excluding only equal-phone substitutions | 0.2344 | 0.1543 |
| AUROC with current labels → excluding equal-phone rows | 0.7280 → 0.7291 | 0.7044 → 0.6908 |
| Median score, equal-phone vs different-phone positive | 0.222 vs 0.271 | 0.190 vs 0.147 |

Exclusion changes prevalence and the evaluated population, so the AP values are not a direct causal effect estimate. It does show that the simple claim “equal normalized-phone labels explain low stop AP” is unsupported: their median model score is *higher* than that of distinct-phone stop substitutions, and removing them lowers absolute AP. We do **not** relabel these events as correct.

The parser references `data/original_sources/l2arctic/l2arctic_release_v5.0.zip`, which is not present at that local path after the documented dataset migration to `kltn_gdrive:KLTN-Dataset/datasets/L2-ARCTIC/original/`. The derived manifest drops raw TextGrid notation and comments. We subsequently recovered the raw labels via byte-range reads of the v5 speaker ZIP mirror and matched all TRAIN TextGrid intervals to the manifest. The parser's `clean_phone` removes `*` and trailing digits. The corpus [official annotation guide](https://psi.engr.tamu.edu/l2-arctic-corpus-docs/) explicitly defines `*` as a perceived phone deviating from standard American pronunciation.

### Full raw TextGrid reconciliation

`extract_review_annotations.py` checked all **1,800 TRAIN TextGrids and 59,562 target intervals** against the manifest's slot order, timestamps, labels, and normalized phone identities. Every comparison passed. Of the 838 same normalized-phone substitutions, **836** have a perceived-phone `*` marker; the other two raw labels are `IY1,IY0,s` and `V,V8,s`. **Zero** have literally identical raw target and perceived spellings. This establishes what the labels encode, but it does not independently verify that each audible deviation or feedback judgment is correct. The `*` marker is not an input to the frozen target head at inference, so stripping it from the recorded realized-phone field is not itself a demonstrated cause of low recall.

Artifacts: `RAW_SAME_PHONE_LABELS.csv` (all 838 rows), `RAW_TEXTGRID_REVIEW.csv` (136 review-queue rows), and `RAW_TEXTGRID_REVIEW_SUMMARY.json`. The source ZIP mirror is [L2-ARCTIC v5 on Hugging Face](https://huggingface.co/datasets/chikingsley/l2-arctic-release-v5.0/tree/main); only TRAIN speaker annotation ranges were fetched. The annotation filenames, intervals, labels, and normalized phones matched the local TRAIN manifest. The original archive remains in the user's documented Drive remote.

## 3. Alignment audit

The preceding report compared **durations**. Here we also compared the cached SM13 CharSiu **alignment center** with the midpoint of the manual interval on every TRAIN OOF vowel/stop substitution. The center cache was generated by the same `align_centers` routine used to prepare SM13 OOF acoustic features.

| Gold substitutions | Vowel (n=2,280) | Stop (n=862) |
|---|---:|---:|
| Median absolute center displacement | 6.33 ms | 7.23 ms |
| 90th percentile center displacement | 29.35 ms | 40.79 ms |
| Center displaced >50 ms | 5.31% | 7.66% |
| Center displaced >100 ms | 1.89% | 2.20% |
| Center outside manual interval | 5.79% | 9.16% |

For a direct start/end check, `boundary_probe.py` reconstructed the exact CharSiu Viterbi path from existing cached CTC logits for a fixed random, speaker-diverse sample of **30 vowel + 30 stop gold substitutions** from distinct TRAIN utterances. Every reconstructed path matched the saved center cache. Frame positions were mapped to seconds using utterance duration divided by CTC frame count.

| Sampled substitutions | Vowel (n=30) | Stop (n=30) |
|---|---:|---:|
| Median absolute start error | 8.08 ms | 11.53 ms |
| Median absolute end error | 9.34 ms | 7.35 ms |
| At least one boundary >50 ms | 1/30 | 4/30 |
| At least one boundary >100 ms | 0/30 | 1/30 |

These measurements support mostly accurate timing while revealing a nonzero problematic tail. They do not demonstrate that the aligner is causally irrelevant to individual misses. The manual intervals are the corpus's existing annotation rather than an independently re-annotated boundary standard.

## Review packet and next decision

`AUDIO_REVIEW_QUEUE.csv` lists **136** TRAIN events with valid local audio paths, crop times, target/realized phones, Neural-S scores, prompt text and blank fields for two independent reviewers and adjudication. It includes 40 same normalized-phone cases (8 vowel, 24 stop, 8 other), plus high-scoring clean phones and low/high-scoring substitutions in both target manners. Their raw TextGrid labels are now in `RAW_TEXTGRID_REVIEW.csv`; 39/40 have `*` and one has a stress-digit difference. Reviewers should judge the audible contrast and whether a correction would be pedagogically warranted, then compare with the corpus label. The CSV is a review queue, **not a completed human annotation**.

For practical review, `make_review_clips.py` created **136 short local WAV clips** with 250 ms context on each side, a randomized `AUDIO_REVIEW_BLIND.csv` with target phone but without gold labels or model scores, a keyed `AUDIO_REVIEW_WITH_CLIPS.csv`, and `AUDIO_REVIEW.html` for playback. The page hides the original label behind a disclosure control. [AUDIO_REVIEW_PROTOCOL.md](AUDIO_REVIEW_PROTOCOL.md) defines first-pass codes and adjudication. Two independent reviewers should fill the blind sheet before adjudication. These contain TRAIN audio only. No reviewer fields were filled automatically.

The next model experiment should target **new evidence at the top of the score ranking**, such as local transition and pair-specific acoustic cues, and be evaluated with speaker-held-out threshold choice at matched realized FWR. The frozen-score result makes another global or manner-only threshold sweep low priority. Any new model needs a genuinely new prospective speaker cohort for final confirmation; the historical TEST and DEV_REPLICATION cohorts have already been exposed.

**Subsequent pilot:** A controlled seed-101 SM13 target-head loss-only ranking probe has now been run. It modestly improved stop AP but worsened vowel AP and failed the useful-recall gate. See [MANNER_RANK_PROBE_REPORT.md](MANNER_RANK_PROBE_REPORT.md). The frozen production model and policy remain unchanged.

**Specialist ablation:** Manner-specific heads using the original SM13 architecture and weighted CE, trained only on their own manner, also failed. Stop AP fell from 0.1968 to 0.1845 in all four outer folds, and both manner-specific low-warning frontiers worsened. See [MANNER_SPECIALIST_REPORT.md](MANNER_SPECIALIST_REPORT.md). These one-seed results weaken a simple global-gradient-competition explanation, without isolating the encoder as the cause.

## Reproduction and provenance

```bash
.venv/bin/python experiments/rq1_sm13_followup_audit_01/run_audit.py
.venv/bin/python experiments/rq1_sm13_followup_audit_01/boundary_probe.py
.venv/bin/python experiments/rq1_sm13_followup_audit_01/extract_review_annotations.py
.venv/bin/python experiments/rq1_sm13_followup_audit_01/make_review_clips.py
```

- `AUDIT_RESULTS.json`: exact threshold points, four held-out folds, label sensitivity and center statistics.
- `AUDIO_REVIEW_QUEUE.csv`: 136 candidate audio events for independent listening.
- `BOUNDARY_PROBE.json` and `BOUNDARY_PROBE_EVENTS.csv`: sampled start/end reconstruction and event details.
- OOF prediction SHA-256: `2c27addc691502b7758cb265b3e8f03648986d3e378d89a03ae87a5eb5cca094`.
- TRAIN manifest SHA-256: `1f4da8518d9b184fb737b93c7887ecd7c3e73d6e83260e54651074831f61bcff`.
- Center cache SHA-256: `275d707949c4fef785611ef37e2f18906b912974eb7dbd841c06baed85c069c7`.

Neither script trains a model, runs neural inference, edits the frozen policy, or opens `TEST_SEALED`.
