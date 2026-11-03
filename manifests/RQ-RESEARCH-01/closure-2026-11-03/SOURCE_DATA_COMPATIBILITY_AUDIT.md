# Source/data/checkpoint compatibility audit — correction addendum

Full independent evidence: [Agent B](independent/B_SOURCE_DATA_COMPATIBILITY.md), including thirteen pinned-byte comparisons, original E0 training-source inspection, exact code ranges and missing-proposal search. This summary does not implement fixes or load weights.

## 1. Verified source identity

Outer `research-phase@b25a7595360294798d8d06b37592dd65dd4c9c33`; nested `repo@baa4989243d5ff46b8b3bf5a3c280cdf2857382b`. Thirteen critical outer/nested working file byte contents match committed baa4989: model/dataset/collate/parser/normalizer/alignment/G2P/losses, E1 evaluator, main trainer, fold evaluator and E0/E1 configs. Coordinator independently checked ten of these. E0 originally trained at `caa56783113d40eac438e5f458cdb313b8932218`; B inspected its target/head construction as Git objects, not by assuming all later code was its original source.

Pre-existing nested dirt consists of fourteen Python executable-mode changes `100755→100644` and two deleted PDF symlinks, not algorithm-byte differences in those scripts. All preserved. No Git index staged, commit, push or source repair in this process.

## 2. Actual contracts and touchpoints

| Seam | Executed semantics | Compatibility / future authorized touchpoint |
|---|---|---|
| `phoneme_normalizer.py:5–23,53–123` | CTC/canonical IDs 0 blank/pad, 1–39 phones; diagnosis IDs 0–38 (0=AA). Stress/asterisk collapsed in base identity | Adapters must explicitly map phone order/blank; do not load stock word-ASR or author phone-head weights as equivalent |
| `textgrid_parser.py:159–258` | Strict substitutions include same-base starred variants; deletions retain canonical index/error1, realized string `sil`, no diagnosis identity; additions lack canonical index | `sil` placeholder is not an acoustic segment or CTC deletion target. KEEP-by-phone-equality loses strict errors. `err` supports error label but no valid replacement target |
| `dataset.py:97–153` | Chronological realized CTC excludes deletion/silence/unknowns, retains valid additions; binary includes substitutions+deletions; diagnosis masks deletion/unknown/correct | A subtype/edit arm needs explicit label transport and masks; changing CTC to canonical targets is a changed objective, potentially encouraging absent-phone hallucination |
| `collate.py:25–87` | Pads binary/CTC/diagnosis/auxiliary targets; no subtype tensor | Future new targets need separately reviewed schema. Static defect: blinded branch fails to initialize `padded_aux_targets` before return; **not fixed or executed** |
| `model.py:137–207` | Skip vector + soft diagonal bias; zero skip-position bias | Not a hard monotone/edit-state posterior; direct skip supervision changes loss contract |
| `model.py:350–403,434–456` | 40-way CTC; binary; 39-way diagnosis. Frozen backbone and diagnosis in official two-head training | Diagnosis tensor existence is not trained diagnosis capability. New subtype/realization loss/head needs explicit distinct-arm approval |
| `model.py:575–607` | Explicit oracle branch can pool annotation windows; deployable cross-attention ignores supplied intervals | New runtime must be prompt/G2P-only. Training labels/offline scoring do not license inference gold |
| `alignment.py:16–223` | Minimum-one-frame interval pooling; deterministic canonical-gold/G2P reconciliation | Not existing acoustic CTC Viterbi alignment or omission decoder. DP gaps in this utility can be G2P disagreement rather than speech errors |
| `g2p.py:24–26,95–112,199–207` | Pinned dictionary; one primary pronunciation; strict OOV handling | Variants, normalization and repeated phones can mimic errors. New pronunciation policy must be versioned, not silently alter old metrics |
| `evaluate_e1_mdd.py:108–206,301–358` | Sigmoid error score; deletion-vs-correct AP; separate realized-phone PER; O oracle canonical identities; Z prompt-only inference with offline gold joining | Preserve evaluator. New scorers get separately registered score/coverage outputs and full denominators; no gold-aware model calls |
| `evaluate_e1_mdd.py:229–267` | Strict checkpoint load; 897-wide binary input can detect skip structure; monotonic bias from metadata | Config-only computations may differ despite matching tensor shapes. Strict load alone does not establish identical semantics |

Old `RQ1-STD-01/MDD_LABEL_SCHEMA.md:60–71` examples include deletion SIL/blank/unknown labels inconsistent with actual realized-target filtering/masking. This addendum records the discrepancy; historical schema is not rewritten.

## 3. Dataset support and limits

Existing aggregate census, not raw reinspection: Fold-1 TRAIN **79,426** canonical tokens = 67,585 correct + 9,377 substitutions + 2,464 deletions; VAL **9,956** = 8,799 + 1,012 + 145. Sixteen fit speakers/two VAL speakers; no Vietnamese validation speaker. Some train phone identities have zero deletion support (AW/OW/OY/ZH). Counts justify attempting compact supervision, not claims of sufficiency, acoustic identifiability or all-phone/L1 coverage. Overlapping fold training records are not independent data.

Zero-duration deletion intervals, if present, remain binary errors and absent CTC targets; the parser does not require positive duration. Exact zero-duration counts were not available from the inspected aggregate bins; **not invented**. Minimum-one-frame pooling/diagnostics is not a physical duration measurement.

L2-ARCTIC v5.0 is registered CC BY-NC 4.0; SpeechOcean762 CC BY 4.0. Keep assets private/noncommercial under current project policy; these dataset declarations do not clear external code/model/teaching-source licenses. SpeechOcean score parser does not deliver an equivalent correct/substitution/deletion event taxonomy. Extra transcripts used by another paper are not automatically available in this repository.

## 4. Genuine versus conditional compatibility

- **Exact-weight sidecars:** existing free-phone decoding/DP, canonical/self-aligned scoring controls and segmentation-free S/SD/SDI graph scoring can reuse frozen CTC emissions. They require new validated inference/scoring code, compatible labels, repeat/terminal/occupancy handling and authorization; “training-free” is not “zero compute.” Cao code emits raw GOP and occupancy separately; verify numerical normalization versus score normalization. Code-reuse license remains unresolved.
- **Natural subtype/existence/edit objectives:** labels largely exist, but subtype transport, strict same-base/unknown/replacement/insertion rules and new loss/output mappings do not. A global 41-logit head can include REPLACE-to-self, but its strict-error semantics must be explicit; excluding the canonical replacement leaves only 38 valid replacements per slot. Arbitrary insertions require gap events or a realized decoder.
- **E0/E1 checkpoints:** E0 binary input width896; E1 width897 plus skip parameter. They are not strict interchangeable states. New heads require listed partial initialization/reinitialized tensors and fresh optimizer/scheduler/scaler/RNG, not full resume. Preserve original-pretrained independent-fold rules.
- **Joint/attribute/retrieval methods:** plausible adapter paths but new objectives/indexes/ontologies/model-specific licenses. Retrieval bank must be active-fold TRAIN-only; deterministic traits derived from a guessed phone add no independent acoustic evidence. Official phonological implementation provenance remains unresolved.
- **No existing architecture is newly certified compatible by an execution test.** Structural source compatibility is not load parity, calibration, acoustic grounding, resource feasibility or pedagogical readiness. Existing E1 Folds2–4 configs do not authorize rollout.

## 5. Missing latest omission proposal

B enumerated documentary files in outer `docs/` (7), `manifests/` (261), nested equivalents (7/240), and `/home/khoa/Projects/KLTN` top-level files, using omission/deletion/proposal and relevant Vietnamese/GOP terms; no raw data/archive traversal. **No distinct latest proposal was located.** Nearest sources actually read: DIAG-01 checkpoint §11 (three experimental ideas), RQ-MODEL-01 report §§4–6 (implemented E1 proposal), and both research checkpoints. None is asserted to be the intended source.

The source gap is accounted for, not resolved. Operator must provide/identify exact path/version before claiming proposal conformance. The original thesis is located; the earlier session's separately requested RQ1 source also remains unidentified, but is not silently added as a new prerequisite to this request. No missing content is reconstructed by conjecture.
