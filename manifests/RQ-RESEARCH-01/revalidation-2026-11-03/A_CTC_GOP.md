# Lane A — CTC/GOP and segmentation-free deletion scoring

**Independent research report; read-only.** I inspected the outer checkout (`b25a7595360294798d8d06b37592dd65dd4c9c33`), not the dirty nested `repo/`. No repository or product files were changed. No model weights, audio, labels, or sealed TEST members were read; no training, inference, tests, or GPU work were run. **GO below means research/pilot consideration only, not authorization to execute.**

Evidence labels: **VERIFIED_LOCAL** = inspected repository/archive record; **VERIFIED_PRIMARY** = inspected original paper; **VERIFIED_IMPLEMENTATION** = inspected code or implementation location; **INFERENCE/PROPOSAL** = analysis/design; **UNVERIFIED** = not independently established.

## 1. Thesis scope gate

**VERIFIED_LOCAL — authoritative thesis:** `/home/khoa/Projects/KLTN/DC KLTN0127 23520772&23520757_TranThiHongYen (1).md`, read in full. It is one long Markdown line, so conventional line numbers do not distinguish passages. The relevant headings and exact Vietnamese excerpts are:

- **§2.2(b), “Phân tích và hỗ trợ sửa lỗi phát âm”:** “tập trung phát hiện một số nhóm lỗi mục tiêu ở cấp độ âm vị”; “xem xét mức độ tin cậy”; feedback includes “giải thích nguyên nhân và hướng dẫn người học luyện tập lại.”
- **§2.3, RQ1:** “Phương pháp phân tích âm học ở cấp độ âm vị đạt hiệu quả như thế nào trong việc phát hiện các nhóm lỗi phát âm mục tiêu trên dữ liệu tiếng Anh có nhãn tham chiếu?” The expected contribution combines acoustic analysis with learning feedback.
- **§3, scope/limitations:** prioritize “bài luyện có văn bản tham chiếu” and labeled audio; do not aim to detect all errors in free conversation. **§4(c)** explicitly identifies substitutions, final-consonant deletion (“bỏ phụ âm cuối”), and confusable pairs as targets.
- **§4(c), method:** forced alignment and GOP are proposed research options, but GOP “không được xem là bằng chứng độc lập để kết luận chính xác loại lỗi phát âm”; specific error categories require a method checked against labeled reference data. Low-confidence results should be hedged or prompt another attempt.
- **§2.3 RQ2 / §4(e):** Multi-Agent is for automated exercise generation, not RQ1 phonetic diagnosis. Grammar GEC references in §4(f) address grammar, not validated acoustic MDD.

**Gate:** phoneme-level GOP, CTC, alignment-free acoustic scoring, omission/substitution distinctions, confidence, and cautious feedback fit RQ1. A recognizer family is not inherently out of scope when used as narrowly controlled acoustic evidence. Existing baseline/training contracts remain frozen; novel objectives require later explicit authorization and separate configurations.

## 2. Local empirical evidence and provenance

**VERIFIED_LOCAL — archive records, not fresh execution.** I read the required `RQ-MODEL-01` report, process checkpoint, benchmark summary, and both E0/E1 `E1_EVALUATION_REPORT.json` files. The archived Kaggle benchmark records inference-only execution on 300 Fold-1 validation utterances (9,956 canonical phones; 145 deletions), with 0 training updates. Provenance records identify E0 Baseline R (`a2445c…a948b`) and E1 skip-token checkpoint (`3b6867…79d24`), each at the fixed 3,600-update endpoint. These are archival claims; I did not independently verify remote artifacts or rerun evaluation.

In those archived records, E1 **did not improve** the deletion target: omission AP 0.032252 vs E0 0.035020; omission recall at the stated 15%-FPR operating point 0.365517 vs 0.400000; binary AP 0.309293 vs 0.317021. CTC PER improved slightly (0.232365 vs 0.236818), but E1 deletion skip mass was 0.000000. Treat this as one Fold-1 comparison, not a general result about CTC or all deletion methods.

**VERIFIED_LOCAL — cautionary correction:** `manifests/RQ1-DIAG-02B/SCIENTIFIC_REPORT.md` supersedes DIAG-02’s causal claims. It reports partial archive parity failure (9/80 archived-score comparisons beyond tolerance), small eight-speaker diagnostic samples, and only exploratory matched CTC separation. It explicitly withdraws the claim that high CTC blank on deletion frames proves omission readiness or causal bypass. Blank is a CTC symbol, not an omission label: correct speech also has blank-dominant frames. Do not turn greedy-decoder edits, blank argmax, or label-conditioned frame summaries into gold deletion diagnoses.

**VERIFIED_IMPLEMENTATION:** `src/langfens_prep/alignment.py:73+` supplies deterministic reference-sequence edit alignment and configurable tie-breaking; `src/langfens_prep/g2p.py:1–26` documents pinned CMUdict G2P; `src/langfens_prep/model.py:51–180` defines canonical-phone cross-attention, optional skip token, and monotonic bias. `pool_frames_to_canonical_phones` (`alignment.py:16–70`) clamps every interval to at least one frame, so it cannot represent a true zero-duration phone; using TextGrid spans at inference would be oracle leakage, not deployment evidence. Existing RQ1 config set includes `configs/rq1_trn_02a_fold1.yaml` and `configs/rq_model_01_e1_fold1.yaml`; do not alter them for a pilot.

**Separate omission proposal/source:** the requested independently located “latest omission research proposal” was **not found** in the inspected likely manifests/docs. The `RQ-MODEL-01` report is an implementation/baseline report, not a substitute proposal. `RQ1-DIAG-02B` is a correction report, not that proposal. I therefore cannot attribute an additional proposal or its claims.

**Product seam (read-only):** I read the AI-service and speaking-service `AGENTS.md` files. Speaking service’s `Features/RabbitMq/SpeakingGradingConsumer.cs` handles uploaded speech/transcription and correlated grading; AI service owns grading/explanations (`app/main.py` registers `speaking_grade`). This is a plausible future integration seam, not evidence that phoneme timestamps, confidence, or omission-specific feedback are already supported. Service guidance warns against fabricated success on model failure. No product code was changed.

## 3. Method comparison

| Method | Substitution and deletion potential | Supervision actually needed | Acoustic evidence, confidence, feedback | Integration and cost (unmeasured costs identified) |
|---|---|---|---|---|
| **Classical GOP / forced alignment** | Standard GOP is useful for canonical-phone mismatch and substitution screening. Forced alignment assigns every reference phone a path; a missing phone can be squeezed into a short path or absorbed by neighbors. It does **not** natively certify omission. | Acoustic phone models plus canonical phone transcript; aligner. Labeled MDD data needed to calibrate error labels/thresholds. | Phone likelihood/posterior or likelihood ratio over an aligned span; confidence requires held-out calibration. Explain “weak acoustic match,” not “definite deletion.” | Honest thesis comparator; additional aligner/acoustic model and phone posterior interfaces. Potentially CPU feasible with an existing aligner; setup, model compatibility, and latency not measured. |
| **CTC-GOP / posterior marginalization** | Competing-phone posterior at a canonical phone position can flag substitutions; blank mass can indicate uncertainty/absence, but not uniquely deletion. Shared frames, blank, and context confound it. | Frame posteriors from a CTC phone recognizer; canonical reference. Labeled validation needed for score calibration and error-type evaluation. | More principled than interpreting a single decoded path: compare normalized path mass for canonical phone versus alternatives. It offers evidence, not an absence label. | Likely reuses existing CTC head; modest extra inference calculation, no new backbone. Exact published “CTC-GOP” implementation for LANGFENS was not verified; marginalization details must be specified before piloting. |
| **Sequence likelihood ratio / leave-one-phone-out** | A phone-specific deletion contrast is possible: compare acoustic sequence likelihood under canonical `…p…` versus a sequence that omits `p`; useful for deletion and mismatch, but can confound insertion/substitution and neighboring-phone acoustics. | A normalized sequence model/scorer and canonical pronunciation; labeled validation for thresholds. | Use log-likelihood ratio per effective duration / local context, not raw sequence likelihood. Calibration and uncertainty are essential. It can support “evidence favors omission” only if alternatives and length are controlled. | Likely expensive relative to greedy CTC: requires multiple rescoring hypotheses or lattice/forward passes. Exact latency, GPU cost, and normalization behavior unmeasured. |
| **Free-phone greedy CTC + reference edit alignment** | Produces interpretable substitutions/deletions/insertions relative to canonical phone string; weakest ambiguity control for repeated phones and near-tied edit paths. A decoder “deletion” is a mismatch hypothesis, **not** proof of gold omission. | CTC recognizer and canonical phones; no phone boundary labels at inference. Labeled speakers for error analysis. | Expose posterior margins and alignment ambiguity; avoid confident learner advice when repeated phones permit multiple edit paths. | Cheapest candidate: one existing CTC forward, greedy decode, DP. Existing local DP exists. CPU scoring overhead is likely small, but unmeasured. |
| **Calibrated late fusion** | Potentially combines CTC-GOP, normalized likelihood contrasts, greedy edit, and baseline binary score; could cover substitutions and deletions better than one cue, but may learn speaker/context shortcuts. | All constituent signals plus labeled train data; calibration/threshold selection on validation only. | Calibrate per-phone or hierarchical confidence; retain separate error-type probabilities and abstain. Feedback must reflect uncertainty and acoustic evidence. | No acoustic backbone change; scoring/calibration and output schema changes. Cheap inference once scores exist; data leakage and small validation-speaker calibration are key risks. |

**Mathematical safeguards (INFERENCE):** CTC sequence probability is \(P(y\mid x)=\sum_{\pi:B(\pi)=y}\prod_t p_t(\pi_t\mid x)\). For candidate phone \(p_i\), compare e.g. \(\log P(y\mid x)-\log P(y_{\setminus i}\mid x)\), but raw difference is length/context biased and includes paths where neighboring phones explain the acoustics. Normalize against a matched alternative set and duration/context baseline; do not infer posterior odds without calibrated models and explicit priors. A forced canonical path is constrained to include every target phone; Viterbi alignment can allocate one or more frames to it, including repeated adjacent labels requiring blank separation. It cannot produce a zero-frame skip unless the model explicitly includes a skip transition or scores a competing shorter sequence. Conversely, a zero-frame interval in a frame-level scheme is an operational convention, not evidence that the phone was physically absent. Short phone spans (many under 80 ms in historical diagnostics) are particularly sensitive to frame stride and boundary uncertainty.

## 4. Primary-paper and implementation evidence

| Primary source (direct inspection) | Supported claim and limits | Implementation / status |
|---|---|---|
| **Xu et al., “Explore wav2vec 2.0 for Mispronunciation Detection,” Interspeech 2021.** [Paper](https://www.isca-archive.org/interspeech_2021/xu21k_interspeech.pdf), [archive page](https://www.isca-archive.org/interspeech_2021/xu21k_interspeech.html). **VERIFIED_PRIMARY**, paper abstract, method, Table 2. | Uses wav2vec 2.0 representations with phone-aligned segments and classification; reports F1 0.610 on L2-ARCTIC (Table 2 compares its variants and GOP-related baselines). Crucially, authors say pooling over alignment interval requires HMM-DNN forced alignment or manual annotation and removing this process is future work. This supports feasible acoustic scoring, **not segmentation-free deletion detection**. | Paper points to Kaldi GOP examples; no author implementation/checkpoint/license verified. LANGFENS has an acoustic CTC model and DP, but not the paper’s GOP implementation. |
| **Korzekwa et al., “Weakly-Supervised Word-Level Pronunciation Error Detection in Non-Native English Speech,” Interspeech 2021.** [Paper](https://www.isca-archive.org/interspeech_2021/korzekwa21b_interspeech.pdf), [archive page](https://www.isca-archive.org/interspeech_2021/korzekwa21b_interspeech.html). **VERIFIED_PRIMARY**, abstract, method, Table 2. | Explicitly names replacement, addition, deletion as error types, but supervises **word-level** error and does not provide phone-specific deletion labels. Multi-task word error detection plus phone recognition; results depend on corpora/splits in paper. This is not deletion-specific phone evidence and cannot directly validate LANGFENS phone-deletion scoring. | No code/checkpoint/license verified from inspected paper/archive. |
| **Cao et al., “An Analysis of Goodness of Pronunciation for Child Speech,” Interspeech 2023.** [Paper](https://www.isca-archive.org/interspeech_2023/cao23_interspeech.pdf). **VERIFIED_PRIMARY**, §2 GOP definitions, §3, Table 2. | Compares GOP definitions and reports dataset-dependent score distributions; uses Kaldi-trained ASR models. Shows GOP distribution varies with speech population and scoring method, warning against transporting thresholds. Not a deletion-method paper. | Paper directly links [GOPs repository](https://github.com/frank613/GOPs). Repository page was reachable, but I did not verify a runnable checkpoint, license terms, or deletion-specific code. |
| **Wang et al., “Goodness of Pronunciation (GOP) Scoring for Pronunciation Training,” 2007** (foundational source cited by Cao et al.). [DOI search](https://doi.org/10.1016/j.specom.2006.10.001). **UNVERIFIED_PRIMARY** here: original full paper was not retrieved; no detailed claims attributed to it beyond the secondary citation in inspected Cao paper. | Classical likelihood-based comparator is thesis-compatible; specifics should be checked against primary text before final citation. | Kaldi ASR/GOP recipe is an implementation family, not verified as interchangeable with this exact paper. |
| **Cao et al., “Segmentation-Free Goodness of Pronunciation,” IEEE/ACM TASLP 34 (2026), 796–807.** [DOI](https://doi.org/10.1109/TASLPRO.2026.3654852). **UNVERIFIED_PRIMARY** for technical claims: Crossref verifies title/authors/year/pages and CC BY 4.0 metadata, but the IEEE PDF request returned HTTP 418. | Highly relevant 2026 lead; I cannot verify its method, deletion results, or claims without paper text. Do not cite as evidence that segmentation-free deletion scoring works. | Code/checkpoint availability and license for implementation not verified. |

**Implementation evidence:** LANGFENS `alignment.py` DP aligns G2P/canonical phone sequences with explicit deterministic tie-breaks; this is reference-to-reference alignment, not acoustic CTC decoding. `model.py` exposes framewise CTC logits and existing cross-attention. **No verified local CTC-GOP, sequence LLR deletion scorer, or posterior calibration implementation was located.** The requested omission proposal was not independently found.

## 5. Shortlist and minimal falsifiable pilots

**Ranked for research/pilot consideration only** (rubric dimensions each 1–5; higher is better; values are judgments, not measured wins):

| Candidate | Academic evidence | Omission potential | LANGFENS fit | Feasibility/compute | Feedback suitability | Overall uncertainty |
|---|---:|---:|---:|---:|---:|---|
| CTC posterior marginalization + matched acoustic controls | 4 | 3 | 5 | 4 | 3 | High: blank is non-identifying; exact score must be operationalized |
| Greedy CTC/reference DP with posterior margins | 4 | 2 | 5 | 5 | 2 | Moderate-high: inexpensive baseline, weak decoder-error specificity |
| Normalized phone deletion LLR (with insertion/substitution alternatives) | 3 | 4 | 3 | 2 | 3 | Very high: normalization/model details absent, repeated phones complicate contrasts |
| Classical forced-aligner/GOP comparator | 5 | 2 | 3 | 3 | 3 | Moderate: established comparator, but forced alignment risks hiding true skips |
| Calibrated late fusion of frozen scores | 3 | 4 | 4 | 4 | 4 | High: overfit/leakage risk, especially with few validation speakers |

**Minimal validation-only design (proposal):**

1. Freeze the existing Baseline R and E0 scoring contract; do not retrain or change model objectives. On common Fold-1 train/validation-only data, compute candidate scores as *offline analysis only*. No TEST access. Separate canonical target sequence from realized/gold labels: reference text/G2P is inference input; TextGrid phone boundaries and omission labels are evaluation targets only.
2. Establish deletion ground truth from labeled validation targets; do not label CTC decoder deletions as truth. Report deletion AP, recall at fixed correct-token FPR, substitution AP, full binary AP, and speaker-level results. Include phone duration, word position/final coda, repeated-phone contexts, and phone identity strata. With only Fold-1 validation speakers, confidence intervals and calibration are exploratory.
3. For CTC scores, compare blank posterior / canonical posterior margin against matched correct phones (same speaker, canonical phone, word position, and duration caliper), plus the existing binary score and simple duration-only baseline. Pre-register matching and tie handling; disclose unmatched cases. A matched case-control AP is not population AP.
4. For a deletion LLR, score full canonical sequence and each leave-one-phone-out candidate plus plausible substitution/insertion alternatives. Normalize for effective sequence length and candidate count; compare to phone-identity and duration-matched nulls. Test repeated-phone examples separately because the CTC collapse map and edit alignment can shift which duplicate receives the match.
5. A falsifiable positive result requires better deletion ranking/recall at a fixed false-positive budget than frozen baseline and matched controls, without meaningful regression in substitution/full-task metrics. Calibration must be fit only on train or held-out calibration speakers and reported separately from validation ranking; do not pick threshold and claim unbiased performance on the same validation labels. No positive result means stop rather than broaden to cross-fold execution.

## 6. Risks, complementarity, confidence, and gaps

- **Complementary methods:** Forced alignment/GOP is the honest thesis comparator; CTC marginalization and free decoding are distinct evidence sources; calibrated late fusion is only warranted if each source adds held-out signal. The archived E1 result argues against another unconstrained learned skip token without changed evidence/supervision.
- **Score validity:** raw sequence likelihood favors shorter hypotheses unless normalized; phone duration, candidate count, priors, and context affect scores. CTC blank posterior is expected between repeated labels and within many correctly realized phones. A greedy edit is one path through ambiguous sequence space.
- **Identifiability:** A reference-only acoustic model cannot prove “absence” from blank alone: silence, low energy, coarticulation, boundary shift, and model uncertainty can yield similar evidence. Short-phone intervals and annotation boundaries compound ambiguity. Distinguish *low support for canonical phone* from *true deletion*.
- **Calibration/feedback:** Validation-selected score thresholds are not independent deployment estimates. Per-phone calibration risks sparse data; speaker leakage and tuning on validation can inflate performance. Use abstention/“try again” language where evidence is weak; never generate categorical pedagogical feedback from an uncalibrated GOP/CTC score.
- **Evidence gaps:** No independently located omission-specific proposal; no verified code/checkpoint/license for a CTC-GOP or deletion-LLR implementation; 2026 segmentation-free paper could not be inspected; no fresh local evaluation. Classical GOP foundational paper was not retrieved directly. Do not report GPU timings, benchmark improvements, or deletion-specific gains beyond archived local evidence.
- **Confidence:** High on thesis scope, local E0/E1 archive values, DP/G2P/model source paths, and distinction between CTC blank/decoder edits and gold omission. Moderate on relative feasibility judgments. Low on expected deletion gains, current 2026 method details, and generalization beyond the available validation speakers.

**Disposition: GO for research/pilot design only; no execution authorization.** Do not alter baseline contracts or consume TEST.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Concise report gives a research-only disposition and documents residual risks, evidence limits, source gaps, and the archived nature of empirical findings."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "Read-only source inspection and public-paper retrieval into /tmp",
      "result": "passed",
      "summary": "Inspected required thesis, repository docs/source/archive JSON and service guidance; retrieved and read available primary PDFs. No tests or model execution performed."
    }
  ],
  "validationOutput": [
    "No repository/product files changed; no training, inference, audio/label loading, or TEST access."
  ],
  "residualRisks": [
    "2026 segmentation-free paper full text and implementation were inaccessible; technical claims remain unverified.",
    "No independent omission proposal was located; no fresh evaluation was run.",
    "Deletion-specific calibration and generalization remain unproven."
  ],
  "noStagedFiles": true,
  "diffSummary": "No changes.",
  "reviewFindings": [
    "no blockers"
  ],
  "manualNotes": "Output is returned as the requested report for harness persistence; the archive-only E0/E1 results are not a fresh execution or authorization to run a pilot."
}
```