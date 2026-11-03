# P-SUP — natural label transport, collator and output design

No source/model/training code is modified. This is a proposed extension, not an implemented schema or training authorization. Shared V1 indexing/strict masks/populations are binding.

## Source seams and proposed fields

`textgrid_parser.py` already retains `error_type`, canonical `token_index`, strict binary label, same-base/asterisk and unknown-realization flags. `dataset.py` currently produces binary, chronological realized CTC, substitution diagnosis and auxiliary realized-phone targets. `collate.py` does not transport operation targets. Its blinded branch has a documented statically observed `padded_aux_targets` initialization defect; do not silently fix it here. Current dataset `blind_labels=True` still reads TextGrid/canonical identities and therefore is NOT the new prompt-only inference interface.

Future extension should be additive and separately reviewed, with legacy behavior unchanged:

| Field | Per-item shape/type | Batch/padding | Semantics |
|---|---|---|---|
| `operation_targets` | length L int64 | B×Lmax, -100 | correct0/substitution1/deletion2 from natural annotation, no synthetic DP labels |
| `operation_mask` | length L bool | false pad | valid canonical operation, includes deletion and same-base substitution |
| `existence_targets` | optional length L | -100 pad + separate mask | if chosen hierarchy: deletion0 vs correct/substitution1; unknown realized identity does not erase known operation |
| `realization_targets` | optional length L 0–38 | -100 | only known correct/substitution identities; no deletion/err target |
| `canonical_lengths` / stable token keys | L and identity | explicit length + metadata | canonical order never filtered without label/key update |

Only approved primary parameterization emits/consumes necessary fields. Unsupported operation/canonical identity fails with explicit typed reason; do not turn a dropped label into correct. Silence/addition lacks canonical index and cannot consume operation slot. Addition remains in valid realized CTC sequence; arbitrary insertion supervision would need a distinct gap/sequence extension. `err` substitution retains substitution and binary error, masks replacement. Strict same-base starred substitution remains substitution even when phones normalize equal. Deletion zero duration remains deletion; never discard it because an interval is empty or pool one gold frame to pretend it was realized.

CTC target remains realized chronology including known additions; blank0 vs diagnosis phone0 distinction tested. Preserve legacy binary target and mask for parity/control audit. All operation/label tensors are TRAIN supervision or offline evaluation only, outside `net_input`. Runtime `net_input` contains waveform/length/mask and prompt-derived canonical tokens/length only; no annotation IDs/times/word positions/realized phones/operation masks masquerading as inputs. New prompt-only loader must not call annotation readers at all. Split/ID checks run before opening audio or labels.

Collator design must reject mixed blinded/labeled batches or explicitly support every branch consistently, initialize ALL optional outputs, align target lengths/keys, use separate audio/canonical/operation/CTC masks and assert valid labels. No broadcast of the first sample's blinded flag across incompatible items. Padded/unknown/all-masked losses return deliberate finite no-supervision status without dividing by zero; all-masked operation data is a gate failure, not a successful learning batch.

## Output exchange (future private artifacts)

Use shared `run_manifest.json`, `runtime_predictions.jsonl`, `offline_token_evaluation.jsonl`, `utterance_predictions.jsonl`, `metrics.json`, `metrics_by_speaker.csv`, `failures.jsonl`, SHA256SUMS outside Git. No prediction records are generated in setup.

Runtime token row extends shared schema with `head_parameterization`, `objective_revision`, `capacity_control_id`, `checkpoint_endpoint`, optional `operation_logits[3]` and `operation_probabilities[3]` or documented hierarchical equivalents, full precision `s_error`, optional `s_omit`/`s_sub`, calibration ID, uncertainty/status. Probabilities must sum to one for a three-way treatment; subtype probabilities from binary-only capacity control are unidentifiable and must not be called calibrated diagnosis. No runtime gold fields. Reference keys use prompt indices; mapping to gold occurs in offline rows only after runtime SHA is frozen.

Offline row includes operation/binary/realization validity masks and strict gold labels, not a phone-equality heuristic. Per-gold-key failure/mapping rows and prompt extras retained; all labels/populations V1. Both shared error-score and subtype-specific omission/substitution AP named. Metrics save TP/FP/TN/FN, achieved FPR and threshold source, PER edit/reference counts, all-gold coverage/abstention and per-speaker strata; no omitted hard cases. Run manifest records tensor initialization/freeze inventory, allowed IDs, source/config/loss/label-map hashes, independent state seeds and checkpoint successful/skipped updates. Checkpoint metadata ties output to exact objective and control, not only a model shape.

## Pending implementation/CPU tests — NOT executed

1. After objective authorization, add isolated label-adapter fixtures and explicit new collator version or opt-in extension; historical evaluator/default semantics immutable. Fix blinded-output defect only in separately reviewed scoped change with regression fixtures.
2. Synthetic TextGrid snippets only: correct, substitution, same-base-star, deletion with zero/nonzero duration, unknown err, silence, insertion, repeat phones and invalid tags; assert exact operation/binary/CTC/masks/key alignment. No corpus files.
3. Unequal lengths, empty/all-masked/invalid canonical labels; pad -100, blank0 and phone0 not confused. Blinded/labeled/mixed batches; optional fields initialized; no label/timestamp in runtime inputs. Spy/mock annotation reader must raise if runtime touches it.
4. Objective algebra tested on plain numeric fixtures: class mapping, `s_error`, BCE-vs-three-logit aggregate consistency and hierarchical normalization if selected; no tensor autograd/backward/optimizer/model download/forward. Gradient-learning correctness would require separately authorized remote tests, not this CPU plan.
5. Metadata-only loader/schedule/checkpoint fixtures: no TEST opening, exact successful-update endpoint, skip/scheduler accounting, independent state identities, wrong arm/fold/objective/contract resume rejection, restored RNG/scaler and immutable endpoint. Mock mirror interruption/corruption/readback failure fail closed.
6. Output schema finite values/key uniqueness/coverage, strict labels and subtype-vs-generic-score endpoints; grouped ties/precision/achieved-FPR/PER numeric fixtures. Compare untouched source/evaluator hashes before/after any later implementation.

Passing such future synthetic tests establishes schema/control correctness only, not learnability, acoustic grounding or diagnosis readiness. Actual subtype support, numerical behavior under remote AMP and all scientific metrics remain unmeasured.
