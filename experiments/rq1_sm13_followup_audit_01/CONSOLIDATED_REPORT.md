# SM13 vowel/stop diagnostic and improvement pilots — consolidated report

**Run date:** 2026-11-13  
**Scope:** TRAIN only, 12 speakers, 1,800 utterances, 59,562 target-phone slots and 61,362 gap slots. All reported validation folds are speaker-grouped. No `TEST_SEALED` data were opened for this work. Historical DEV/TEST cohorts used in earlier experiments are considered exposed and are not fresh confirmation sets.

## Executive result

The frozen SM13 policy is conservative: on TRAIN OOF it produced zero true-positive substitution warnings for vowel and stop groups, with 216 false warnings overall (FWR 0.120). Threshold-only changes cannot meet the illustrative target of at least 15% recall and 50% precision at FWR ≤0.20 on these frozen scores. Three training/scoring experiments were subsequently tried: pairwise ranking loss, manner-specific neural heads, and static/temporal acoustic specialists. None provides a validated fix. The most promising simple static specialist achieves pooled 5.53% vowel recall and 7.66% stop recall after nested threshold transfer, with two of four folds exceeding FWR 0.20. The five-pair phone-conditioned pilot achieves 0.79% and 0.35% recall over all vowel and stop substitution support, respectively, and under 50% precision.

**Decision:** keep the frozen checkpoint and operating policy, and keep the 51D candidate as an unpromoted research control. First complete a leakage-controlled, matched-joint-FWR comparison between frozen SM13 and 51D. Only if 51D fails to provide a meaningful recall gain in that comparison should work move from threshold engineering to diagnosing model discrimination. Do not choose a new architecture or objective until a controlled experiment identifies the likely cause.

## 1. Was earlier work mostly missing analysis?

No broad conclusion of that kind is supported. The historical SM13 method-selection experiment reported a 9.2–10.6% relative Macro-3 F1 gain at matched realized FWR over its Phase-01 comparator. That was a real aggregate result. Its selection gate did not require minimum precision and recall separately for vowels and stops, so it could not detect that these groups had zero recall at the frozen policy.

The later diagnostic report also contained errors that needed correction: it reported 582 rather than 838 same-normalized-phone substitutions, misinterpreted normalized labels as annotation anomalies, and misstated the SM13 head size. These mistakes affected causal interpretation and next-step selection; they do not invalidate every earlier benchmark. The core process gap was a missing subgroup operating gate and insufficient threshold-transfer validation for the rare, difficult groups.

## 2. Frozen-score limits and policy transfer

The frozen score was evaluated at every attainable manner threshold with O/I and other-manner alerts held fixed and the registered S/O score-margin arbitration applied. At FWR ≤0.20, each manner was allowed to spend the remaining budget separately; these are counterfactuals and cannot be combined into one operating policy.

| TRAIN OOF policy | Vowel TP / 2,280 | Vowel FP | Precision | Stop TP / 862 | Stop FP | Precision | Joint FWR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen policy | 0 (0%) | 0 | — | 0 (0%) | 0 | — | 0.1200 |
| Best retrospective manner threshold, risk only | 81 (3.55%) | 143 | 36.16% | 78 (9.05%) | 143 | 35.29% | 0.1994* |
| Best retrospective manner threshold, precision ≥50% | 9 (0.39%) | 9 | 50.00% | 22 (2.55%) | 19 | 53.66% | 0.1250* / 0.1306* |

`*` Separate manner-specific operating points. The vowel and stop points cannot share the same remaining FP allocation. Even in an optimistic single-stream bound that turns all other alerts off and gives one manner all 360 false warnings, the frozen score reaches only 8.25% vowel recall at 34.31% precision; stop reaches 16.24% recall at 28.28% precision. With precision constrained to ≥50%, the best recalls are 0.39% and 2.55%. A single manner threshold cannot meet both target metrics on the frozen scores.

An additional four-fold policy-transfer check chose thresholds on nine TRAIN speakers and applied them to three held speakers. Risk-only selection transferred at 4.12% vowel recall / 38.37% precision / FWR 0.2039 and 9.16% stop recall / 33.47% precision / FWR 0.2072. Adding a training precision constraint did not preserve it on held speakers: vowel was 0.18% recall / 28.57% precision / FWR 0.1256; stop was 2.20% / 42.22% / 0.1344. These are descriptive checks on a small speaker set, not tight confidence bounds.

## 3. Label and alignment audit

There are 838/7,381 (11.35%) substitution rows with equal **normalized** target and realized phones: 262 stop, 24 vowel, and 552 other. Common pairs include R→R (321), T→T (149), D→D (79), ER→ER (26). The raw TextGrid audit matched all 1,800 TRAIN files and all 59,562 target intervals to the manifest. Of the 838 cases, 836 have a perceived-phone `*` marker removed by phone normalization; two differ by a trailing digit. No raw pair is literally identical. The corpus guide says `*` marks a perceived-phone deviation from standard American English. This does not prove each case warrants corrective feedback, so these rows were not relabeled.

Removing equal-normalized-phone substitution rows as a sensitivity analysis changes vowel AP from 0.2348 to 0.2344 and stop AP from 0.1968 to 0.1543; the comparison changes prevalence and population and is not causal. It does not support the claim that these rows explain the weak stop ranking.

On gold substitution events, CharSiu predicted center displacement from manual-interval midpoint had median 6.33 ms (vowel) and 7.23 ms (stop); 90th percentiles 29.35/40.79 ms; >50 ms in 5.31%/7.66%; center outside interval in 5.79%/9.16%. A separate exact-path reconstruction on 30 vowel and 30 stop substitutions matched the saved center cache. Alignment is unlikely to explain the population-wide failure, although it can affect individual clips.

## 4. Where frozen-score misses concentrate

The following per-phone points are retrospective diagnostics with other warning streams fixed. The vowel and stop budgets are evaluated separately.

| Target | Gold substitutions | TP | FP | Missed | Most common missed pair |
|---|---:|---:|---:|---:|---|
| IH | 467 | 8 | 29 | 459 | IH→IY, 367 |
| AH | 463 | 18 | 19 | 445 | AH→AO, 148 |
| OW | 263 | 26 | 44 | 237 | OW→AO, 210 |
| AE | 223 | 9 | 11 | 214 | AE→AA, 122 |
| D | 263 | 9 | 24 | 254 | D→T, 104 |
| T | 269 | 18 | 40 | 251 | normalized T→T, 136; raw labels often carry `*` |
| P | 177 | 44 | 49 | 133 | P→B, 95 |

IH, AH and OW comprise 1,193/2,280 vowel substitutions (52.3%); D and T comprise 532/862 stop substitutions (61.7%). This motivated a five-pair pilot. It did not target the full confusion space.

## 5. Experiments run

### 5.1 Within-manner pairwise ranking loss

One seed, four outer speaker folds, original SM13 target head and training settings plus top-32 within-manner pairwise softplus (λ=0.1). Stop AP rose 0.1968→0.2190, while vowel AP fell 0.2348→0.2275. At a retrospective FWR≤0.20 manner point, stop was 38 TP/38 FP (4.41% recall, 50% precision); vowel was 168 TP/144 FP (7.37%, 53.85%). Under nested held-speaker transfer, stop was 26/26 (3.02%, 50.0%, FWR 0.1428); vowel risk-only was 73/138 (3.20%, 34.60%, FWR 0.2044). This does not meet the joint gate.

### 5.2 Manner-specific SM13 neural heads

Original 633,361-parameter head and weighted CE, trained separately on only vowel or stop events. AP: vowel 0.2348→0.2335; stop 0.1968→0.1845, worse in all four folds. At retrospective FWR≤0.20, global baseline produced vowel 81/143 (3.55%, 36.16%) and stop 78/143 (9.05%, 35.29%); specialist route produced vowel 64/141 (2.81%, 31.22%) and stop 57/143 (6.61%, 28.50%). No specialist manner point reached 50% precision. Simple global-gradient competition is therefore insufficient as a root-cause explanation; the result does not identify which representation component is limiting.

### 5.3 Explicit seven-window temporal features

Seven 32 ms Hann windows at offsets −120, −80, −40, 0, +40, +80, +120 ms around the **predicted** CharSiu center, with 16 mel bands, adjacent deltas, log RMS and ZCR (222 new dimensions), concatenated to the static 51D features. Same HGB classifier, weights, seed and outer speaker groups. No manual boundaries or DEV/TEST inputs.

| Manner | SM13 AP | Static 51D AP | Static best retrospective recall (P≥50%, FWR≤0.20) | Temporal 273D AP | Temporal recall at same constraints |
|---|---:|---:|---:|---:|---:|
| Vowel | 0.2348 | 0.2808 | 168/2,280 = 7.37% (168 TP, 144 FP) | 0.2946 | 145/2,280 = 6.36% (145 TP, 142 FP) |
| Stop | 0.1968 | 0.2364 | 118/862 = 13.69% (118 TP, 116 FP) | 0.2367 | 82/862 = 9.51% (82 TP, 81 FP) |

These are separate same-data threshold searches and optimistic screening bounds; one manner at a time gets the remaining warning budget. The temporal features raised vowel AP but worsened the low-risk frontier for both manners. Stop this exact recipe. The static control looked more promising and therefore received a nested transfer check.

### 5.4 Static 51D specialist, nested threshold transfer

For each outer fold, per-manner specialists trained on nine speakers produced scores on three held speakers. Six-speaker inner fits generated nine-speaker inner OOF scores to select vowel/stop thresholds jointly; thresholds transferred unchanged. This candidate was selected after viewing exploratory results, so the check is development-exposed, not independent confirmation.

| Fold | Held speakers | Vowel TP/FP | Vowel recall/precision | Stop TP/FP | Stop recall/precision | Joint FWR |
|---|---|---:|---:|---:|---:|---:|
| 0 | ABA, ASI, BWC | 23/14 | 4.76%/62.16% | 11/4 | 3.97%/73.33% | 0.176 |
| 1 | EBVS, HJK, HKK | 31/29 | 5.50%/51.67% | 6/13 | 5.22%/31.58% | **0.227** |
| 2 | HQTV, LXC, NJS | 38/22 | 5.37%/63.33% | 24/11 | 14.63%/68.57% | 0.156 |
| 3 | PNV, SKA, SVBI | 34/25 | 6.46%/57.63% | 25/9 | 8.17%/73.53% | **0.204** |
| **Pooled** | **12 speakers** | **126/90** | **5.53%/58.33%** | **66/37** | **7.66%/64.08%** | **0.191** |

The frozen policy on these utterances has 216 total FP (FWR 0.120). Two folds exceed the 0.20 FWR gate; fold 1 stop precision is below 50%. Other frozen streams used 0.136, 0.133, 0.082 and 0.129 FP/utterance in folds 0–3, so pooled FWR hides per-speaker variation. Candidate fails 15% recall in both manners and is not promoted.

### 5.5 Five phone-contrast specialists

Fixed pairs: IH→IY, AH→AO, OW→AO, D→T, T→D. A separate binary target-phone substitution model used the static 51D features. Inner-speaker thresholds were selected with ≥50% inner precision per emitting pair and a joint allowance of 0.08 added FP/utterance, then applied unchanged to outer speakers. The pair identities were not input features. Frozen O/I and the existing S/O arbitration were retained. This is development-only and targets just these pairs.

#### Outer-fold metrics, denominators are the full manner substitution support

| Fold | Held speakers | Vowel TP/FP | Vowel recall/precision | Stop TP/FP | Stop recall/precision | Total FP / 450 | FWR |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | ABA, ASI, BWC | 1/1 | 0.21%/50.00% | 0/1 | 0.00%/0.00% | 62 | 0.138 |
| 1 | EBVS, HJK, HKK | 8/15 | 1.42%/34.78% | 0/0 | 0.00%/— | 65 | 0.144 |
| 2 | HQTV, LXC, NJS | 0/1 | 0.00%/0.00% | 1/0 | 0.61%/100.00% | 30 | 0.067 |
| 3 | PNV, SKA, SVBI | 9/6 | 1.71%/60.00% | 2/3 | 0.65%/40.00% | 50 | 0.111 |
| **Pooled** | **12 speakers** | **18/23** | **0.79%/43.90%** | **3/4** | **0.35%/42.86%** | **207 / 1,800** | **0.115** |

#### Pair-level outer results

| Directed pair | Fold 0 TP/FP (support) | Fold 1 | Fold 2 | Fold 3 | Pooled TP/FP / support | Pooled recall / precision |
|---|---:|---:|---:|---:|---:|---:|
| IH→IY | 0/0 (98) | 0/0 (141) | 0/0 (174) | 0/0 (54) | 0/0 / 467 | 0.00% / — |
| AH→AO | 0/0 (95) | 1/6 (106) | 0/1 (166) | 1/0 (96) | 2/7 / 463 | 0.43% / 22.22% |
| OW→AO | 1/1 (47) | 7/9 (61) | 0/0 (73) | 8/6 (82) | 16/16 / 263 | 6.08% / 50.00% |
| D→T | 0/0 (61) | 0/0 (42) | 1/0 (64) | 2/3 (96) | 3/3 / 263 | 1.14% / 50.00% |
| T→D | 0/1 (112) | 0/0 (15) | 0/0 (40) | 0/0 (102) | 0/1 / 269 | 0.00% / 0.00% |

The pair specialist therefore does not meet the precision or recall gate. Most detections came from OW→AO; IH→IY and T→D produced no true positives. Pooled warning rate below 0.20 reflects both missed detections and the fact that the pair specialist replaces the frozen S score for the five target phones; it must not be read as a standalone performance improvement.

## 6. Independent listening remains outstanding

A blind 136-clip packet is prepared with audio page, blind sheet, keyed evidence sheet and protocol. Two independent reviewers should hear each clip and code 0 (acceptable realization/no corrective feedback), 1 (clear actionable deviation) or 2 (uncertain/audio insufficient), before revealing corpus labels or comparing answers. No human votes have been entered. This step determines whether the desired educational feedback target agrees with corpus substitution and `*` deviation labels; no model can answer that question from these data alone.

## 7. Conclusions and next work

Evidence supports these conclusions:

1. The frozen SM13 score and threshold calibration do not provide useful vowel/stop substitution warnings at the requested precision and warning rate.
2. Label normalization and alignment have been audited; neither currently explains the broad performance gap. The raw `*` labels encode meaningful pronunciation deviations, but their actionability needs listener review.
3. Loss-only ranking, simple manner specialists, the tested temporal features, nested static specialists and five phone-pair specialists all failed the joint operating gate. AP or lower FWR alone is insufficient.
4. Weak phone-specific generalization and threshold transfer across speakers are material observed problems. Exact causal representation limits remain unproven.

### Agreed next experiment: matched-FWR with leakage control

1. Compare frozen SM13 and the static 51D scorer using identical outer speaker groups and the same O/I alert streams.
2. For every outer fold, refit each candidate only on the nine outer-training speakers. Within those nine, generate inner out-of-fold scores by fitting on six speakers and scoring the other three. The outer three speakers must not affect model fitting, feature or score calibration, threshold choice, or the inner score distribution.
3. Choose each candidate's vowel and stop thresholds jointly on inner scores under the same total warning allowance (joint FWR ≤0.20, maximum 90 false warnings over 450 outer-fold utterances). Apply those thresholds unchanged to outer predictions. Report realized FWR; compare policies only at matched FWR, and do not tune on outer labels.
4. Primary outcomes: pooled and per-fold vowel/stop recall, precision, TP/FP, joint FWR, paired per-speaker recall differences, and uncertainty intervals. State ahead of time what recall difference counts as meaningful; show the full empirical distribution because only 12 speakers are available. Include the frozen policy at its original operating threshold as an operational reference, but do not mistake FWR 0.12 for a matched-budget comparator.
5. If 51D does not retain a meaningful advantage over SM13 at matched FWR, then investigate discrimination causes through controlled probes/ablations (e.g. feature information, phone-conditioned separability, speaker/domain effects) before selecting a new architecture or loss. If it does retain an advantage, replicate across seeds and a genuinely new speaker cohort before promotion.

The prior nested 51D run is useful pilot evidence but does not satisfy this next gate: it compared a 0.191-FWR candidate to the 0.120-FWR frozen operating point, used a development-exposed candidate, and had two outer folds above 0.20. It is not the matched-FWR result requested here. Independent listening remains valuable for defining which corpus deviations deserve feedback; no reviewer votes have yet been collected.

## Reproduction files

All artifacts are in this directory. Main machine result files: `AUDIT_RESULTS.json`, `MANNER_RANK_PROBE_RESULTS.json`, `MANNER_SPECIALIST_RESULTS.json`, `TEMPORAL_PROBE_RESULTS.json`, `NESTED_STATIC_RESULTS.json`, and `PHONE_CONTRAST_RESULTS.json`. Runners: `run_audit.py`, `train_manner_rank_probe.py` / `evaluate_manner_rank_probe.py`, `train_manner_specialists.py` / `evaluate_manner_specialists.py`, `run_temporal_probe.py`, `validate_static_specialist_nested.py`, and `run_phone_contrast_pilot.py`. Listening files: `AUDIO_REVIEW.html`, `AUDIO_REVIEW_BLIND.csv`, `AUDIO_REVIEW_WITH_CLIPS.csv`, `AUDIO_REVIEW_PROTOCOL.md`.

This consolidated report supersedes the separate narrative reports in this directory as the main reading entry point. Protocols, row-level CSV/NPZ artifacts, logs and machine JSON remain the detailed reproducibility record.
