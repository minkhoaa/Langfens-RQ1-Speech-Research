# RQ1-DIAG-02B — Counterfactual validity and CTC discrimination

## Disposition

**PARTIAL — NO-GO for authorizing alignment-aware training from the present causal evidence.**

The DIAG-02 *causal proof* is invalidated, not the broader alignment hypothesis. Projection-correct interventions and matched CTC comparisons provide exploratory evidence, not proof that temporal attention causes natural omission failures. No training is launched or proposed for automatic execution.

Canonical evidence: [`evaluation-final/REPORT.json`](evaluation-final/REPORT.json), SHA-256 `38aa543a46d5c2da856e0d73f1887a12285576a205c66fc41c0faa194f8647b6`. Numerical corrections: [`reconciliation-final/CORRECTIONS.json`](reconciliation-final/CORRECTIONS.json) and [`corrections.csv`](reconciliation-final/corrections.csv). Historical reports remain unchanged apart from appended supersession notices.

## 1. Scope, provenance and acceptance

- Base: `feat/rq1-diag-02-temporal-alignment-validation @ dd2dee9a2063284d2a0f9fbf1c18315f3a7cd774`.
- Dedicated branch: `feat/rq1-diag-02b-counterfactual-validity`.
- Local CPU inference only; parameters have `requires_grad=False`, `eval()` and `torch.inference_mode()`. **Zero backward calls, optimizer updates, training or GPU execution.**
- The frozen manifest and prediction archive hashes match their registered values. All 1,199 prediction IDs are checked against their own fold's validation whitelist **before** any annotation read. No sealed TEST audio/annotation member is accessed; TEST metadata is used solely for exclusion checks. Authorized fold-package hashes also match the registry.
- Census/archived metrics: all 39,576 validation tokens. New inference: the same first ten omission-containing utterances per speaker as DIAG-02: 80 utterances, 2,951 tokens (2,566 correct, 278 substitutions, 107 omissions), eight speakers. This is omission-enriched, not a representative prevalence sample.
- Waveform interventions: two preselected matched pairs per speaker, 32 tokens (16 omissions, 16 controls), 64 masked full-encoder forwards. Not a full-corpus waveform experiment.
- Primary run: Python 3.14.8, PyTorch 2.14.1+cpu, one CPU thread, `MKL_CBWR=COMPATIBLE`, deterministic algorithms enabled, seed 1337; complete runtime versions/configuration in JSON.

| Acceptance requirement | Evidence | Disposition |
|---|---|---|
| Baseline inference parity | Head reconstruction and all-open attention mask vs `model.forward()`: maximum logit error **0** over 80 utterances; repeated forwards exact | VERIFIED on bounded sample |
| Original checkpoint/configuration and archived-score parity | Four SHA-256/config checks pass; **9/80** archived-score comparisons exceed unchanged 0.000200 tolerance | **NOT fully satisfied** |
| Matched CTC discrimination | 66 exact speaker/phone/word-position pairs, duration caliper 20 ms, no replacement; 10-ms sensitivity check | EXECUTED |
| Original MHA semantics | Original module performs learned per-head Q/K/V, softmax, concatenation and output projection; independent projected-head regression | VERIFIED |
| Waveform vs feature interventions | Separately labeled, separate denominators and common-token comparison; duration-matched random controls | EXECUTED, waveform sample small |
| Numerical reconciliation | Boundary records, historical macro recalculation, stale-value and SHA ledgers | EXECUTED |
| Sensitivity vs causal mechanism | Explicitly separated below | SATISFIED in interpretation |

**Overall scientific acceptance is not PASS.** The 1,199-utterance full-forward replay and deployment calibration were not executed.

### Checkpoint identity

Each endpoint has exactly 3,600 successful training updates; Fold 1 uses its original epoch-6 `best.pt`, Folds 2–4 their fixed-budget epoch-6 endpoints. Full checkpoint metadata, original skipped-update counters and frozen-backbone configuration are in the primary report. Strict state-dict loading succeeds. No `best.pt` substitution for other folds.

| Fold | Verified SHA-256 |
|---|---|
| 1 | `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b` |
| 2 | `ab4c589445feac42636296cd36d7cf2ae7eba7360445f0084454c87fa2fef508` |
| 3 | `bbee9dd392806fc81c88e41018edc7adb270d71b5047930f5ce78a69c538a441` |
| 4 | `c502271feba097781a6b0bdfe413085271d3a03647353f272dccd29bbb1499f6` |

### Parity and numerical drift

The first attempt stopped on archived-score error 0.000313178; [`FAILED_ATTEMPT.md`](FAILED_ATTEMPT.md) records the interruption. Subsequent runs explicitly used `--continue-on-archive-drift`, **without changing tolerance**. The canonical one-thread run has maximum archive probability error **0.001304954**, on HKK/arctic_a0105. Its nine failed utterances are listed in the JSON and `parity.csv`.

Sample Binary AP: archived **0.292173109**, canonical CPU **0.292111817** (difference −0.000061292). This small ranking change does not waive pointwise parity. The earlier four-thread continuation has maximum drift 0.002100974 and is retained as preliminary evidence; do not mix its estimates with the canonical run.

A fresh process under the canonical one-thread environment reproduces all **saved baseline logits exactly** (maximum error 0). Scalar-versus-vector sigmoid evaluation differs by at most **5.96e−8**; its verifier labels probability bit differences `NUMERICAL_DRIFT`, not a failure of logit determinism. See `reconciliation-final/fresh_process_forward.csv`. Cross-thread numerical equivalence and the exclusive cause of the historical CUDA-to-CPU drift remain unresolved. Backend/conditioning effects are plausible, not proven.

## 2. Archived metric reproduction

These are calculations from the verified saved prediction archive, **not** claims of a new full forward replay:

| Contrast | N | Positives | AP | ROC-AUC | Matched-minus-mismatched AP |
|---|---:|---:|---:|---:|---:|
| All errors vs correct | 39,576 | 4,213 | 0.272580 | 0.729022 | +0.010619 |
| Substitution vs correct | 38,947 | 3,584 | 0.264270 | 0.734851 | +0.010715 |
| Omission vs correct | 35,992 | 629 | 0.030506 | 0.695813 | +0.001677 |

Omission-vs-correct prevalence is **1.7476%**, not the all-token omission prevalence 1.5893%. No new uncertainty claim about the census omission acoustic gap is inferred from its small point estimate.

## 3. CTC discrimination with matched controls

### Design

For each sampled omission, greedily select the nearest-duration correctly realized token with **exact speaker, canonical phone and word position**, within 20 ms, without replacement. Ties use token ID; matching never uses predictions. Unmapped word positions are excluded. **66/107 omissions matched; 41 are explicitly unmatched.** The 132-token case-control sample has 50% omission prevalence; its AP cannot be compared directly with census AP 0.030506.

Mean matched annotation durations: omission **51.58 ms**, control **54.41 ms**. The caliper does not ensure exact duration equality. A tighter 10-ms caliper retains 58 pairs; a negative-duration baseline detects residual duration effects (AUC 0.551 at 20 ms, 0.514 at 10 ms).

Scores are omission-oriented: mean blank posterior; one minus mean/peak canonical posterior; blank-minus-canonical and max-noncanonical-minus-canonical margins; one minus canonical argmax emission fraction. “Noncanonical” includes blank. Canonical frame emission counts and neighboring-span canonical peaks are retained in raw records. These are interval-conditioned frame summaries, not an alignment-free deployable omission decoder.

### Results

95% intervals below are **500 paired speaker-cluster percentile bootstrap** resamples. Whole speakers, retaining both members of every pair, are resampled. They are exploratory at **G=8**, not population guarantees or multiplicity-corrected confirmatory tests.

| Score | AP | ROC-AUC [95% CI] | Diagnostic TPR / FNR at FPR ≤15% |
|---|---:|---|---|
| Mean blank | 0.644 | 0.653 [0.586, 0.746] | 0.242 / 0.758 |
| Missing canonical mean | 0.586 | 0.635 [0.530, 0.732] | 0.242 / 0.758 |
| Missing canonical peak | 0.587 | 0.637 [0.546, 0.724] | 0.227 / 0.773 |
| Blank−canonical margin | 0.634 | 0.648 [0.564, 0.748] | 0.242 / 0.758 |
| Noncanonical−canonical margin | 0.632 | 0.654 [0.553, 0.763] | 0.227 / 0.773 |
| Missing canonical emission fraction | 0.582 | 0.634 [0.515, 0.755] | 0.000 / 1.000 |
| Original Binary MDD probability | 0.484 | 0.482 [0.444, 0.511] | 0.106 / 0.894 |

Actual FPR is 0.1364 for continuous CTC scores and Binary MDD; the tied emission-fraction score requires a threshold above its maximum to respect the ceiling, yielding FPR/TPR zero. All thresholds are selected on these same controls; their TPRs and fixed-threshold bootstrap intervals are **diagnostic, optimistic selection summaries**, not deployment calibration or independent performance estimates.

Matched mean P(blank): omissions **0.8244**, controls **0.7322**. Mean canonical posterior: **0.0917 vs 0.1797**; peak canonical posterior: **0.2859 vs 0.4690**. Correct phones often have high blank probabilities too. At the tighter 10-ms caliper, blank AUC is **0.641 [0.564, 0.755]** and missing-peak AUC **0.640 [0.519, 0.742]**. Per-speaker scores, AP/AUC/TPR intervals and negative-duration control are in JSON.

**Inference:** modest matched frame-level discrimination is observed. Neither >84% blank on omissions nor an AUC above chance establishes reliable omission detection, sufficient CTC guidance, or information being causally “bypassed” by Binary MDD.

## 4. Counterfactual validity and attention interventions

### Historical defect

The original “forced attention” code substitutes a **uniform row in head-averaged weights** and multiplies it by **raw hidden states**. It bypasses learned value and output projections, head splitting/concatenation and learned Q/K softmax. It is not the original MHA under a temporal mask. Replaying it recovers the reported all-token drift about **+0.25515**, but this is explicitly labeled **invalid historical replay**, not causal evidence of MLP OOD breakdown.

### Valid intervention

Use the existing `nn.MultiheadAttention` with boolean additive exclusion semantics (`True` = blocked), original learned projections and **individual head** softmax. Apply the original acoustic padding mask and actual frame lengths. No fully masked query is allowed. Gold windows enter only the offline intervention mask; baseline `model.forward()` receives no TextGrid timestamps. Padded-token handling is preserved and tested; only active tokens enter records and metrics.

Three same-width restrictions: annotated local window; adjacent neighboring window (clipped without shrinking); random nonoverlapping same-width window when feasible. One random intervention placement per token is paired with local; these differ from the 100-placement random attention-mass baseline.

| Restriction | All-token mean Δprob | Omission-vs-correct AP | Paired AP change [95% CI] | Omission recall at τ=.20 | Correct-token FPR at τ=.20 |
|---|---:|---:|---|---:|---:|
| Baseline | — | 0.06899 | — | 0.3364 | 0.1633 |
| Local | +0.02362 | 0.07262 | +0.00363 [−0.00448, +0.00990] | 0.5514 | 0.2342 |
| Neighbor | +0.02436 | 0.06556 | −0.00343 [−0.01567, +0.00232] | 0.4673 | 0.2311 |
| Random | +0.02514 | 0.06494 | −0.00405 [−0.01518, +0.00156] | 0.4579 | 0.2288 |

Local-minus-random omission probability effect is **+0.01451 [0.00873, 0.01943]**; correct-token effect is **−0.00226 [−0.00418, −0.00030]**. This is exploratory evidence of location-specific sensitivity under the artificial restrictions. The all-error local-minus-random AP contrast is **+0.01022 [−0.00305, +0.02595]**. A higher recall at one uncalibrated threshold accompanies higher FPR; it is not a demonstrated operating-point improvement or proof of a natural-omission mechanism. The corrected drift is approximately tenfold smaller than the invalid historical drift.

## 5. Acoustic interventions: waveform is not feature erasure

**Feature erasure:** zero post-encoder contextual features within the frame window, rerun the original attention/head; the encoder is NOT rerun. Local waveform information may remain in contextual features elsewhere.

**Waveform erasure:** zero exactly the annotated sample interval in the original waveform, preserve waveform length, rerun the **full encoder and model.forward()**. Controls zero an equally long randomly placed nonoverlapping sample interval. Zeroing can introduce unnatural boundary discontinuities; it is not a natural phoneme omission or a time-removal/splicing experiment.

| Mode | Tokens | Mean absolute Δprob | Mean Δlogit | Omission-vs-correct ΔAP [95% CI] | Δ omission recall at τ=.20 |
|---|---:|---:|---:|---|---:|
| Feature local | 2,951 | 0.001113 | +0.002837 | +0.000535 [−0.000291, +0.002441] | +0.01869 |
| Feature random | 2,951 | 0.000935 | +0.002466 | +0.000221 [−0.000155, +0.000933] | 0 |
| Waveform local | 32 | 0.007617 | +0.001497 | −0.036257 [−0.077823, +0.001524] | 0 |
| Waveform random | 32 | 0.012423 | +0.016333 | −0.045746 [−0.102258, +0.005776] | +0.0625 |

Feature AP rows exclude substitutions (2,673 tokens); mean changes use all 2,951. Waveform AP is case-control AP on 16/16 tokens; baseline **0.55021**, local **0.51396**, random **0.50447**. Do not compare those APs to full-sample APs as if the populations were the same.

On the **same 32 waveform-audit tokens**, feature mean |Δprob| is **0.001651 local / 0.001979 random**, versus waveform **0.007617 / 0.012423**. Feature and waveform sensitivity are therefore observably different probes. Waveform paired local-minus-random AP is **+0.00949 [−0.02281, +0.04091]**. Neither probe establishes a robust beneficial omission-detection intervention.

Speaker-aware paired summaries (including signed/absolute changes, AP, recall, FPR, τ=.20/.50, and bootstrap intervals) are available for every mode. Feature and waveform zeroing are sensitivity evidence only; neither identifies the causal effect of a naturally omitted phoneme.

## 6. Individual-head attention

Means across all sampled tokens:

| Head | Mean maximum weight | Normalized entropy | Absolute peak offset, s | Mean unique peak frames/utterance |
|---|---:|---:|---:|---:|
| 0 | 0.99172 | 0.00366 | 1.2953 | 6.10 |
| 1 | 0.99298 | 0.00315 | 1.3402 | 6.93 |
| 2 | 0.99272 | 0.00328 | 1.3489 | 6.43 |
| 3 | 0.99160 | 0.00395 | 1.3706 | 7.65 |

The concentration/anchoring observation is reproducible, but is not literal saturation of every token or proof of irrelevance. Correct/substitution/omission per-head masses, entropies and offsets are retained. Mean token-normalized uniform enrichment for omissions is **~1.3966×**, not 0.00×; per-head omission enrichment spans roughly **0.90–1.99×**. Small absolute mass alone cannot be interpreted without duration matching.

Uniform baseline is **K/T** at each token's active frame length. The duration-matched 100-placement random baseline includes the last legal placement. New aggregation uses **ratio of summed masses**, never unstable per-token division by almost-zero random mass, and reports paired observed-minus-random mass with speaker summaries. Historical random enrichment values around 10^15 are numerical artifacts, not scientific effects. Small differences in head statistics across thread configurations are retained, not rounded into claims of bit identity.

## 7. Numerical reconciliation and historical errata

The machine-readable correction ledger enumerates locations, historical claims, verified replacements and dispositions. Key findings:

1. **293 vs 284 under 40 ms:** raw binary subtraction gives 293; rounding durations to four decimals or using integer 0.1-ms annotation ticks gives **284**. Nine nominally 40-ms intervals become slightly less than .04 in floating point; their full IDs/endpoints are saved. Under 80 ms the analogous counts are **545 raw vs 542 rounded**. The historical 86.65% uses the raw convention; rounded count is 86.1685%. No labels were changed.
2. **0.001845 vs 0.001720:** recomputing from the eight stored speaker values gives **0.00172025**, rounded **0.001720**, t CI **[0.000678, 0.002763]**. The claimed 0.001845 / [0.000886, 0.002804] is unsupported. Pooled omission mean **0.001714** is a distinct estimand. Canonical one-thread legacy replay is approximately 0.001721 speaker macro; this small numerical replay difference does not validate the stale 0.001845.
3. **0.00× vs 1.40×:** historical pooled mean token-normalized omission enrichment is **1.3966×**; only NCC's speaker value rounds to zero. The global “complete collapse / no localization” claim overstates the evidence.
4. **Other stale DIAG-02 values:** recommendation says P(blank)=82.8% rather than measured **84.47%**; says masking <.001 while measured omission/all-token means are .001714/.001113; CTC macro CI prose [0.7885,0.8961] differs from stored **[0.788281,0.896394]**.
5. **Random baseline prose:** correct/substitution/omission/all masses stated as .0275/.0289/.0171/.0273; stored JSON is **.028590/.030389/.017846/.028370**. Historical correct/all random enrichment means are **4.85e15/4.22e15**, unstable ratios, not meaningful enrichment.
6. **Head offsets:** prose 1.385/1.422/1.378/1.382 s conflicts with historical JSON **1.2953/1.3402/1.3485/1.3698 s**. Per-head and head-averaged peak offsets must not be substituted. Head maximum-weight *means* >.991 do not mean every token: historical minima are **.4392/.5078/.3386/.5020**. “6–7” omits H3's **7.66** mean unique frames.
7. **Span identity:** old code tested rounded duration agreement, not endpoint identity. An independent endpoint audit now confirms **500/629 exact endpoint matches**, preserving that observation. It does not establish a physically observed omitted-sound interval or causal acoustic absence.
8. **Artifact provenance:** **7/8** full hashes in the DIAG-02 checkpoint ledger do not match base files (script, five CSVs, test); the master JSON hash does match. Those eight files are independently verified byte-identical to `dd2dee9`. The stale ledger is inherited, not caused by DIAG-02B. Actual digests are saved; no tampering explanation is asserted.
9. **Chronology:** old checkpoints say 01:30/01:40 +07, but report metadata says **2026-11-02 14:17:16 +07**. No earlier execution time is newly certified.
10. **Unsupported conclusions:** CTC blank as “>84% omission confidence”; “verified primary cause”; required joint retraining; MLP OOD proof; word-final/coarticulatory/accent-transfer causal explanations; validation omission prevalence labeled training prevalence; untested statistical claims about the omission gap. These are superseded by evidence-ranked hypotheses, not silently repaired. DIAG-01's comparison of interval attention mass to point baseline 1/T is also inappropriate; use K/T.

The frozen original DIAG-02 tests check saved claims and JSON values; their passing does not validate an intervention's semantics or confer causal proof. New tests exercise actual MHA projections, padding, baseline parity, matching, controls and recomputation from raw evidence.

## 8. Evidence-ranked hypotheses and decision

| Rank | Hypothesis | Evidence level | What remains unproven |
|---|---|---|---|
| 1 | Binary MDD ranking relies strongly on token/prior/context correlates and weakly distinguishes matched omissions | Strong descriptive census deficit; matched Binary AUC ~.482; modest local-feature sensitivity | A specific prior-domination mechanism and its fraction of natural errors |
| 2 | Temporal routing may limit useful local evidence | Concentrated distant attention; valid local-vs-random omission sensitivity; modest matched CTC discrimination | Routing as the causal bottleneck; alignment-training benefit; acoustic sufficiency of CTC |
| 3 | Short annotation spans and boundary uncertainty limit frame discrimination | Median ~40 ms; frame quantization; 41 unmatched omissions; duration sensitivity audit | Physical duration of absent sounds; generalization to other annotations/populations |
| 4 | Threshold choice contributes to reported FNR | Recall changes at fixed thresholds, often with FPR inflation | A safe deployable threshold, pedagogical readiness, or a solution to ranking deficits |
| 5 | Historical +.26 drift demonstrates MLP OOD failure requiring retraining | **Rejected as established evidence:** counterfactual bypassed projections | Whether any genuine OOD mechanism exists remains untested |

**NO-GO** for treating DIAG-02 as causal authorization for alignment-aware training, and **NO-GO** for deploying gold-window inference clipping. The modest corrected local-attention signal keeps the alignment hypothesis open; it does not establish a robust AP gain. The historical causal proof and claimed CTC readiness are withdrawn. This is not proof that alignment-aware training cannot work.

Before any new authorization: resolve archived-score parity/provenance, review the small-cluster limitations and matching exclusions, and independently specify a validation-only causal/robustness test or training proposal. Any future threshold selection needs separate deployment calibration data/protocol. No extrapolation to TEST, unmeasured speakers, accents or corpora is justified.

## 9. Reproduction and evidence inventory

Use **new output directories**; scripts refuse overwrites:

```bash
PYTHONPATH=.:src HF_HUB_OFFLINE=1 MKL_CBWR=COMPATIBLE OMP_NUM_THREADS=1 \
  .venv/bin/python audit_scripts/diagnose_counterfactual_validity.py \
  --outdir /tmp/diag02b-new --threads 1 --continue-on-archive-drift

PYTHONPATH=.:src HF_HUB_OFFLINE=1 MKL_CBWR=COMPATIBLE OMP_NUM_THREADS=1 \
  .venv/bin/python audit_scripts/reconcile_diag02b_evidence.py \
  --run /tmp/diag02b-new --verify-forward --outdir /tmp/diag02b-reconciliation-new

PYTHONPATH=.:src HF_HUB_OFFLINE=1 .venv/bin/pytest -q tests/test_rq1_diag_*.py
```

Without `--continue-on-archive-drift`, archive parity failure stops inference; the explicit continuation flag never converts failure into PASS. Primary inference executed in **489.6 s**. Preliminary four-thread continuation executed in **102.2 s**; its outputs are historical, not the canonical report.

- `evaluation-final/{tokens,heads,interventions,legacy_replay}.csv.gz`: losslessly compressed raw records; no audio, model weights or credentials.
- `evaluation-final/{matched_pairs,parity,duration_boundary_disagreements,speaker_interventions,aggregate_ctc,aggregate_interventions}.csv`.
- `evaluation-final/{REPORT,RUN_STATE,ARTIFACT_SHA256}.json`: metrics, authorization state, hashes.
- `reconciliation-final/{CORRECTIONS,RECONCILIATION,ARTIFACT_SHA256}.json`, CSV correction/hash/forward ledgers.
- `evaluation/`: preserved first interrupted attempt, **no completed inference report**.
- `evaluation-continuation/`, `reconciliation/`: preserved preliminary runs; superseded by `*-final/`.
- `FAILED_ATTEMPT.md`, `PROCESS_CHECKPOINT.md`: failure, restart and research handoff.

**Executed regression suite: 31 passed.** This is the three DIAG test modules, not the full repository suite. Full-suite tests were not run because unrelated tests can invoke backward/optimizer behavior outside this task's constraints.

**STOP: scientific audit and recommendation only. No training.**
