# LANGFENS RQ1 — EVALUATION_CONTRACT_V1

Status: **FROZEN shared documentary contract; execution NO-GO.** Applies equally to P-SCORE and P-SUP. This new contract supplements, never replaces, historical O/Z evaluators/contracts. Amend through a separately reviewed V2 in both lanes, not independent edits to V1. Freeze precedes worktree branching. SHA-256 is recorded in `CONTRACT_SHA256` and process checkpoints (not self-referentially here).

## 1. Authorized IDs, units and sealing

Authoritative IDs: `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`, SHA-256 `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`. Select exactly `fold_index == 1`; use its `train.utterances` and `val.utterances` in original order. IDs are `SPEAKER/UTTERANCE`; never glob archives or infer membership from filenames. Ordered-list digest serialization is UTF-8 `"\n".join(ids) + "\n"`.

| Partition | Speakers | Utterances | Ordered-ID SHA-256 |
|---|---|---:|---|
| TRAIN | BWC ERMS HKK HQTV MBMPS NCC SKA SVBI THV TLV TNI TXHC YBAA YDCK YKWK ZHAA | 2,399 | `2ebe54f2fc77e2a3ee9890ce3dc21f725897ebbe85f812d3c017c3b495f14b27` |
| VAL | NJS RRBI | 300 | `6d34642b15bbf80cbd5bd102b5574ab36f5adac74be6d6d0ae7218b002eff065` |
| TEST — SEALED | ABA ASI EBVS HJK LXC PNV | 900 | No test-ID export; manifest metadata only |

No TEST waveform, TextGrid, label, prediction, staging, extraction, calibration, retrieval or evaluation access. No cross-fold artifact importing information about active Fold-1 TEST. Manifest partition/speaker metadata is permitted for denylisting. Setup reads no corpus audio/labels. Future access is allowlist-first, train/val-only package with cryptographic verification. No new data/weights/secrets in Git.

## 2. Canonical positions, strict labels and masks

Gold evaluation key: `(fold=1, speaker_id, utterance_id, gold_canonical_index)`; index is zero-based parser chronological canonical position, not phone identity. Correct/substitution/deletion each consume one position; silence/addition do not. Repeated identical phones remain distinct positions. Prompt runtime keys use `prompt_index` in a separate namespace; gold keys are attached offline only.

Phone order: AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH. CTC: blank/pad 0, phones 1–39. Diagnosis/realization: phones 0–38, ignore -100. Blank is neither a learner deletion nor a 40th phone. AX→AH, stress stripping, asterisk/typo handling follow pinned `phoneme_normalizer.py`, preserving raw metadata offline.

Strict binary target: correct=0; substitution OR deletion=1. All `s` annotations remain substitutions, including starred same-base variants. Do not derive KEEP from normalized-phone equality. Unknown realized `err` retains its operation/binary target but has no realization target. Deletion placeholder `sil` is not an observed acoustic segment or CTC target. Correct/substitution/deletion operation IDs for a future extension are 0/1/2; padding/unsupported operation=-100 with explicit boolean mask. Malformed/unknown canonical identities fail validation rather than silently shortening sequences.

CTC reference is chronological realized valid phones: include valid additions/correct/substitutions, exclude silence/deletions/unknown realized phones. Optional replacement diagnosis valid only on substitutions with known realized identity; auxiliary realized-phone labels valid on correct/substitution only. Masks distinguish padding, unknown identity and absent realization. Never mask deletion out of operation/binary loss. Insertions require separate gap/sequence outputs; no arbitrary insertion class is smuggled into canonical-token AP. No label/collator/model changes are authorized by this design.

## 3. Protocol O versus Protocol Z

**O-ARCHIVE:** annotation-derived canonical identities are oracle information. Historical O may use them as phone queries without timestamps. This is not gold-free deployment. Preserve archival evaluator code/metrics/denominators unchanged; no O real-data replay authorized now. New runtime accepts no TextGrid-derived identities, timings, word boundaries, realized transcript, operation tags or diagnosis.

**Z-PROMPT-V1:** waveform + written prompt only; pinned prompt table/dictionary/G2P, strict OOV policy, first dictionary pronunciation as current baseline, 16 kHz audio preprocessing. No hidden TextGrid calls in dataset/collator/runtime. Gold-label access occurs only in a separate offline evaluation join after predictions and hashes are frozen. Use deterministic canonical-gold versus prompt DP `sub_first` (match, substitution, deletion, insertion priority) from pinned `alignment.py`; record mapping edits and unmatched indices. This mapping is not acoustic omission evidence. Alternate pronunciation/tie policies are separately named sensitivity strata, not silent primary changes.

Compare candidate and contemporaneous prompt-only E0 replay on identical IDs, mappings, emissions where applicable, precision and coverage. Historical O, historical Z and new Z results remain separate tables; .316062, .317012 and .317021 are not interchangeable baseline numbers. A prompt mismatch cannot be declared a learner deletion. Golden timestamp-derived features are forbidden at inference even if available during training/offline evaluation.

## 4. Immutable E0/E1 references

Metadata source: `manifests/RQ-MODEL-01/evaluation/E1_FOLD1_BENCHMARK_SUMMARY.json`; archival evaluator source `baa4989243d5ff46b8b3bf5a3c280cdf2857382b`. E0 original training source `caa56783113d40eac438e5f458cdb313b8932218`.

- E0 `best.pt`: epoch 6 / 3,600 updates; weight SHA-256 `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`.
- E1 `fixed_budget_checkpoint.pt`: 3,600 updates; SHA-256 `3b686730cc0a1101a1bbc96b59dc7bd38a9fc3cc7da7effe26b2d9498f079d24`. E1 was trained; zero rounded skip mass is not a per-token mathematical-zero proof.
- Frozen backbone `facebook/wav2vec2-base-960h@22aad52d435eb6dbaf354bdad9b0da84ce7d6156`, eval. E0/E1 binary input widths 896/897 and skip/config semantics differ. No strict interchangeability or warm-start assumed.
- Archived O: E0/E1 omission AP .035020/.032252; binary AP .317021/.309293; PER .236818/.232365; omission recall at diagnostic FPR≤.15 .400000/.365517. Historical Z aligned N=9,919. These are cited archives, NOT newly reproduced results.

Remote artifact availability/weight hashes are not reverified in setup. Full-precision E1 predictions are missing from inspected archive; rounded summary cannot reconstruct AP/paired uncertainty. DIAG-02B has 9/80 archive parity failures: PARTIAL/causal-training NO-GO retained. E1 rollout remains NO-GO. No historical tolerance relaxation, evaluator patch or baseline substitution. Before experimental interpretation, resolve parity or prospectively scope a new evaluator stratum and reproduce its own control; cannot claim repaired historical parity.

## 5. Scores, populations and metrics

Historical Fold-1 gold population: 9,956 canonical tokens = 8,799 correct + 1,012 substitutions + 145 deletions. These are archival counts, to be checked only in separately authorized evaluation.

| Endpoint | Positive / negative | N | Default score |
|---|---|---:|---|
| Binary AP | substitution+deletion / correct | 9,956 | `s_error` |
| Omission AP (legacy-comparable) | deletion / correct; exclude substitutions | 8,944 | `s_error` |
| Substitution AP (legacy-comparable) | substitution / correct; exclude deletions | 9,811 | `s_error` |
| Dedicated omission/substitution AP | same respective populations | as above | `s_omit` / `s_sub`, separately named |

E0/E1 `s_error=sigmoid(binary_logit)`. Three-way treatment: `s_error=1-p_correct`, `s_omit=p_deletion`, `s_sub=p_substitution`. GOP/log-ratio/DP gaps must specify sign and normalization; larger error score means more suspected error. Uncalibrated raw scores are not probabilities. Ranking AP uses grouped ties and existing non-interpolated convention, sum of recall increments times precision at each distinct descending score. Report score family and prevalence. No AP from thresholded predictions when a continuous score exists. No candidate-specific label denominator or deletion-vs-all AP mislabeled as deletion-vs-correct.

For every registered operating score: TP/FP/TN/FN, recall=TP/(TP+FN), precision=TP/(TP+FP), F1, FNR, achieved correct-token FPR=FP/(FP+TN), threshold and tie policy. Precision on the omission subset differs from full deployed feedback precision: report both. Undefined metrics are null plus reason/count, never a favorable zero. Report omission recall/precision at FPR ceilings .05/.10/.15 and achieved FPR, not only nominal ceiling. Primary operating thresholds are fixed using CALIBRATION correct scores, predict error for `score>=tau`; choose lowest available tau satisfying empirical FPR ceiling, including a reject-all endpoint. VAL-derived ceilings are separately marked retrospective diagnostic curves, not calibrated guarantees.

PER: greedy collapse consecutive frame repeats then remove blank, against chronological realized reference; `(S+D+I)/N_realized`, save integer edit/reference counts. Historical reference N=9,881. CTC recognition deletions are not human omissions. Alternate decoding PER is auxiliary. Report micro and individual NJS/RRBI metrics, confusion counts, phone/word-final/confusable strata with sparse-cell flags; no unsupported Vietnamese-generalization claim.

## 6. Coverage, failures and prediction exchange

Every allowed utterance and every prompt position gets an output or failure record; every gold canonical position gets an offline join row, including unmatched/abstained cases. Report utterance/prompt/gold mapping coverage, counts by label/speaker, invalid/nonfinite scores and recording failures. Aligned-only AP is explicitly conditional. Primary V1 policy for unmapped/invalid raw scores is **abstention, no invented probability**: no validation-prevalence imputation, candidate-specific dropping, max fusion or .90 deletion heuristic. Report all missed errors/false reassurance and risk-versus-coverage; at operating points count abstained true errors as undetected, report answered-only and all-gold denominators plus best/worst unresolved-correct FPR bounds. No full-population AP superiority claim with incomplete score coverage. A TRAIN-derived calibrated B1/prior fallback requires a separately frozen extension before use, not retrospective rewriting.

Required exchange schema (design, no predictions generated):

- `run_manifest.json`: schema/contract version+hash, arm/scorer/score semantics, Git base+tip+dirty status, source/config/package/label-map/prompt/G2P/weight hashes, allowed ID hash, fit/selection/calibration/evaluation speaker IDs, seeds/RNG, hardware/precision/determinism/dependencies, endpoint/update/skips, authorization ID, attempted settings, calibration identity, timestamps and output hashes.
- `runtime_predictions.jsonl`: unique `(run_id, protocol, condition, speaker_id, utterance_id, prompt_index)`; prompt phone/word index; continuous raw scores and named normalized scores, logits/probabilities if supported; score orientation; model-derived alignment/uncertainty or null; `status`, abstention/failure reason, finite-value validity, threshold/calibration reference. No gold fields. Runtime prompt text may be stored only under approved private-data policy, not public Git.
- `offline_token_evaluation.jsonl`: join key above plus `gold_canonical_index` (nullable for extra prompt positions), strict binary/operation/phone/word-position labels, evaluation masks, mapping operation/status, unmapped indices, linked runtime hash; annotation timings only optional explicitly offline diagnostics, never runtime inputs. One gold-key row per condition, preserve repeated positions. Stable ordering by manifest utterance then canonical index; prompt-only extras are separate typed rows.
- `utterance_predictions.jsonl`: decoded phone IDs, emission-cache hash/shape/valid frames, reference edit counts offline only, failure and resource records.
- `metrics.json` plus `metrics_by_speaker.csv`: score/population/protocol/condition/coverage/checkpoint stratum, integer counts and full precision aggregates, threshold provenance and uncertainty. Save machine values without six-decimal truncation; round display only. JSON nonfinite values prohibited; null with explicit reason. Private durable prediction store; Git contains schemas/manifests/hashes only.

Any future schema validator must reject duplicate keys, TEST membership, wrong phone maps, inconsistent masks/counts, gold runtime fields, nonfinite scores, unauthorized model/calibration identity and incomplete failure accounting. Recompute metrics from full precision records; file existence alone is not scientific PASS.

## 7. Speaker-disjoint fitting, selection, calibration and uncertainty

Frozen reduced-TRAIN stratum selected without labels: sort TRAIN speakers by SHA-256 of UTF-8 `rq1-contract-v1:1337:SPEAKER`; first two SELECT, next two CAL, rest FIT. Original utterance order retained within each set.

- FIT (1,799 utterances): BWC ERMS HKK HQTV SKA SVBI TNI TXHC YBAA YDCK YKWK ZHAA.
- SELECT (300): MBMPS TLV. Used for matched checkpoint/objective-selection opportunities only, no gradient updates.
- CAL (300): NCC THV. Calibration and fixed operating thresholds only; no parameter/architecture selection, no acoustic training.
- VAL (300): NJS RRBI. Reused exploratory screen, no tuning/calibration in this new stratum. TEST remains sealed.

P-SUP treatment and contemporaneous binary control share these partitions, initialization, routing, capacity accounting and successful-update budget. A candidate trained on FIT cannot refit on SELECT/CAL and still claim untouched calibration. Verify future subtype support before fitting; if inadequate, STOP and prospectively version/review partition changes, never inspect VAL to pick them. All optimizer/model fitting, including learned calibration/prior models, needs separate remote authorization; no local fitting.

Frozen E0 was trained on all original TRAIN, including SELECT/CAL. Thus P-SCORE calibration on CAL is **not speaker-disjoint from E0 acoustic training**; disclose this irreducible historical limitation. Its VAL remains unseen to acoustic fitting but has been repeatedly reused. Raw frozen-weight ranking is the primary P-SCORE screen; no learned fusion/calibration currently approved. Fully disjoint E0 calibration would require a separately authorized new model or new independent speakers; no claim V1 solves this.

Report both speakers and paired deltas; G=2 cannot establish population significance. Optional paired speaker bootstrap (fixed seed 1337, 1,000 resamples) is exploratory only; no token-independent significance or seed inflation of speaker count. Catalog all attempted arms, hyperparameters, failures and selection opportunities. No universal P-SCORE-before-P-SUP dependency. Confirmatory testing requires separately authorized untouched speakers; TEST cannot be unsealed by this contract.

## 8. Acoustic grounding, feedback and practical acceptance

Registered controls: same waveform+prompt; deterministic within-speaker cyclic shift-1 audio mismatch in manifest order (no fixed points, mapping/hash saved); canonical phone/position-only prior; relevant blank/duration/greedy-DP controls; repeated phones, variants, short/coarticulated realized phones and silence/noise fixtures. Text priors use FIT only for newly fitted models. Waveform/feature masking is a separately authorized sensitivity probe, not a natural deletion or causal proof. Preserve real attention projections, no DIAG-02 forced-attention bypass. Blank/skip mass and recognition gaps alone do not establish acoustic absence/localization.

RQ1 includes **detection AND feedback**: later blinded expert review stratifies true/false detections, misses, wrong subtype, substitutions/omissions and abstentions; rates accuracy, clarity and error-fit with reviewer agreement. Evidence and sourced practice guidance are separate. Neutral hint/retry on uncertain cases; no precise missing-phone segment or physical articulator-cause assertion, no 'no errors' guarantee from abstention. Record guidance/RAG/TTS source/version/license. SpeechOcean762 transfer/ordinal assessment is a separately approved RQ1 endpoint, not the same event taxonomy; RQ2 remains Multi-Agent Automated Test Generation. Integration/history/access/error recovery and actual p50/p95 latency/memory/cost remain future requirements.

## 9. GO/NO-GO and authorization boundary

GO now: documentation, isolated worktrees and lightweight Git/filesystem/document checks only. NO-GO now: scoring/model/training implementation, unit/model execution, real-data inference, model fitting/calibration, backward/optimizer, GPU, remote staging/deployment and TEST access. Both arms are independent design-ready lanes, not selected architectures.

Before future execution: obtain missing exact omission proposal/version or explicit operator acceptance of the source limitation; approve one primary contrast and practical omission-AP/recall margins, binary/substitution noninferiority, acceptable feedback false alarms/coverage; fix scorer graph/objective/initialization and schema; clear relevant licenses; implement and verify synthetic CPU-only no-training tests under separate authorization; pin source/inputs/checkpoints and handle parity qualification; verify private durable storage and remote quota where relevant. No numerical scientific-success margins are invented here. Pending items mean **execution NO-GO**, not incomplete workspace creation. Freeze additional prospective settings before seeing results and stop for operator authorization.
