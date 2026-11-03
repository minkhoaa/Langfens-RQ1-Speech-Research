# Shared runtime exchange v1 — implementation-only addendum

Binding artifact: `PREDICTION_SCHEMA_V1.json`; top-level shape follows the current operator request. No V1 evaluation-contract bytes change. This is an output record, **not** an inference-input envelope. Independent arm adapters must not import each other's source. QA may copy these shared files byte-identically after reviewing SHA256SUMS; no peer checkout reads are needed.

## Shape and semantics

All schema properties are required, nullable where specified. `schema_version="rq1-runtime-v1"`, `protocol="Z-PROMPT-V1"`; unique key `(run_id, protocol, condition, speaker_id, utterance_id, prompt_index)`. Prompt index is zero-based position, not phone identity; repeated identical phones never overwrite each other. `prompt_phone_id` is CTC 1..39, never blank 0 or diagnosis 0..38. `word_index` is prompt-derived zero-based index or null, never annotation word timing.

- `scores={s_error,s_omit,s_sub}`: finite number or null; **larger means more suspected error**. Unsupported subtypes stay null with explicit `score_semantics`, never zero/fabricated probabilities. Raw negative GOP is a rank/likelihood contrast, not a calibrated probability. Name graph, sign, raw versus occupancy normalization and available subtype semantics.
- `raw_scores`: object for algorithm-specific outputs (e.g. `raw_gop`, `canonical_log_likelihood`, `relaxed_log_likelihood`, `occupancy`, `operation_logits`). Old arm plans' extra top-level fields move here or into the run manifest; do not extend the agreed top-level shape independently.
- `probabilities`: null or named numbers/null in [0,1]. Describe joint versus conditional meanings in `score_semantics` and run manifest. A three-way treatment must separately validate sum-to-one and `s_error=1-p_correct`; binary-only capacity-control subtype logits are unidentifiable, not diagnosis. Generic schema does not impose a three-way model on GOP/binary outputs.
- `alignment`, `uncertainty`, `confidence`: null or finite JSON objects. Alignment is model/prompt-derived only; any frame bounds refer to model activations, not a physical absent-phone interval. Declare method, units, ambiguity and meaning. Confidence is not automatically calibrated; unavailable targets stay null. Recording, mapping, detection and subtype confidence differ.
- `status`: `ok`, `abstain` or `failure`. `ok` requires finite numeric `s_error`, `finite=true`, null reason. Non-ok requires nonempty `abstention_reason` and all decision scores null. Finite partial diagnostics may remain in `raw_scores`. `finite=false` flags an invalid source result whose nonfinite values have been removed/nullified, **never** permits NaN/Infinity in JSON. Abstention with valid-but-uncertain diagnostics can use `finite=true`.
- `threshold` and `calibration_reference` are both null or both present. No threshold is inferred from VAL. The paired reference identifies a frozen external registration; this validator does not authenticate that registration.

Metadata names at every depth are lowercase ASCII snake-case and reject gold/TextGrid/perceived/realized/label/target/timestamp/reference-phone/operation-ID-or-tag/interval-boundary names. Numerical operation **logits** are allowed; operation **labels** are not. Keep method/config/weight identifiers in the run manifest where possible. The validator additionally rejects metadata string values explicitly naming gold/TextGrid/oracle/perceived/realized provenance. Integer transport fields require actual JSON integer values (the helper intentionally rejects float `1.0` and booleans). These defensive checks are not proof of data origin: hiding gold values in an innocuous key is still forbidden. No schema can establish acoustic grounding or provenance by itself.

## Required validation order

1. Parse text with duplicate-object-name rejection **at every depth**. Reject NaN, Infinity, -Infinity, overflow such as `1e999`, and nonfinite values recursively in programmatic objects. Serialize with `allow_nan=False`.
2. Apply the JSON schema (strict additionalProperties at top level and scores), then semantic status/threshold/phone-map rules. `strict_runtime_validator.py` implements only the schema subset used here, dependency-free; it is not advertised as a general JSON Schema engine. External Draft 2020-12 validators are also possible, but step 1 and bundle checks remain mandatory.
3. Validate against a separately frozen expected prompt ledger: exact allowed keys, phone IDs, word indices and conditions. Reject duplicate/missing/extra rows, sealed TEST speakers and unauthorized IDs **before** opening any future data. `validate_bundle(records, expected)` checks row set/identity equality, not the authority or authenticity of the supplied ledger. Expected keys must not be derived from the records being checked.
4. Authenticate run/calibration/config/source/package identities and allowlist against V1. Arm validators additionally check graph/objective-specific invariants, probabilities/masks/counts and independent state inventories. No passed schema test substitutes for these checks.

The shared validator's sealed-speaker guard is Fold-1-only metadata. Tests use synthetic IDs (plus a denied speaker string), no corpus records. No real prediction file is generated here.

## Complete accounting and run manifest

Every allowed utterance appears once per condition in a separate utterance ledger, even if prompt construction fails. Every known prompt position has exactly one runtime row, including explicit failure/abstention rows. An utterance with no recoverable prompt must have a typed utterance failure with stage/reason and expected prompt count null; **do not fabricate prompt_index=-1** or discard it. Empty legitimate prompts have declared zero positions. The row validator alone cannot verify that utterance ledger; future bundle validators must reconcile both. Gold positions are joined **offline only after runtime hashes freeze**, including unmapped gold rows and prompt-only extras. No gold fields enter runtime or inference inputs.

`run_manifest.json` must record all V1 §6 provenance, with explicit null+reason for inapplicable/unavailable fields (never invented hashes):

| Group | Required contents |
|---|---|
| Identity | schema version/hash; contract version/path/hash; run ID, arm/scorer, graph/objective/control/normalization/score semantics; authorization ID/scope |
| Source | frozen common base, actual Git tip/branch/dirty status and diff hash; source/config/package/label-map/prompt/dictionary/G2P hashes; dependency versions |
| Artifact | pretrained revision, actual checkpoint hash/endpoint and verification mode; cache hash/shape/valid lengths; runtime/offline/output hashes and timestamps |
| Population | allowed ordered-ID hash; FIT/SELECT/CAL/evaluation speaker lists and exact role/overlap disclosure; expected conditions, utterance count, per-utterance prompt counts and ledger hash |
| Reproducibility | seeds/RNG identity; hardware/precision/determinism; attempted settings/search/selection opportunities; initial/copy/freeze inventories and independent states where applicable |
| Exposure | training status, successful/skipped/attempted updates, epoch/data exposure, fixed versus best endpoint; no new model artifact if none |
| Calibration | calibration registration/identity, fit speaker/role, score/threshold/tie policy; E0's acoustic-training CAL overlap; reused VAL disclosure |
| Accounting | expected/observed utterances/positions; ok/abstain/failure/nonfinite counts and reasons; missing/extras/duplicates; all output/file checksums |

For this tranche: training NOT_STARTED, successful updates 0, skipped 0, attempts 0, no new model artifact. Model/metric/performance validation is NOT_RUN. Metrics belong to a later separately authorized offline evaluator; no current scientific success or winner follows from serialization checks.
