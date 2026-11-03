# Agent B — actual source, data semantics and candidate compatibility audit

**Existing process:** RQ-RESEARCH-01 research-only closure; not a new phase.  
**Disposition:** source/data audit delivered with explicit limitations; **no scientific PASS or implementation authorization**.  
**Execution:** documentation/source reads, Git-object comparisons, small JSON aggregations, primary-document downloads to memory only. No model loading, inference, training, backward/optimizer operation, GPU/remote execution/deploy, or raw dataset/archive traversal. Only this isolated report was written; repository/index/checkpoints/history were not edited.

## 1. Decision-relevant result

1. The actual model has **40 CTC outputs: blank ID 0 and 39 normalized phones IDs 1–39**. CTC learns chronological **realized/perceived** phones, not the canonical prompt; annotated deletions and silence are removed, valid additions retained. A CTC blank is neither a deletion label nor proof of silence.
2. The binary task retains omitted canonical phones with error label 1. The existing 39-class diagnosis head masks deletions and is frozen/untrained in the two-head baseline and E1 training path. Its output cannot substantiate pronunciation diagnosis.
3. E1 is a trained, different structural arm: one learned skip vector, soft diagonal attention bias, and a 897-input binary head versus E0's 896. Its **archived Fold-1 comparison is negative** for omission AP, recall at the registered FPR ceiling, and binary AP. That supports retaining E0, not a universal rejection of alignment or deletion-aware modeling.
4. The lowest-disruption candidate seam is **read-only scoring of frozen CTC emissions plus canonical G2P phones**. Segmentation-free GOP-S/SD/SDI needs new graph/scoring code, not new CTC target labels. Free-phone decoding + DP is a useful recognizer-error-sensitive control, not verified omission localization. Forced canonical alignment cannot represent an omitted phone as absent unless its graph explicitly permits deletion.
5. Subtype/DELETE supervision and edit-aware tagging require **new task contracts**, masking rules, checkpoint schemas and approval. A 41-class KEEP/DELETE/REPLACE representation is not lossless for the current strict annotations: same-base accented substitutions can be errors even when canonical and realized normalized phones match.
6. **No distinct latest omission-detection proposal was found in the searched documentary scopes.** Closest records are identified below, not passed off as the missing document. No proposed architecture is selected or implemented here.

## 2. Authority and provenance: three source identities, not one

### Original thesis authority

Directly read `/home/khoa/Projects/KLTN/DC KLTN0127 23520772&23520757_TranThiHongYen (1).md`, especially physical **line 14**, sections **2.2(b), 2.3/RQ1, 3(b)/Giới hạn nghiên cứu, 4(c)/Speaking, and 4(g)/evaluation**. Its body is one long table line; section locators are more meaningful than fabricated line ranges. A read-only Python print expanded the original line in memory.

The thesis distinguishes conversational STT from acoustic pronunciation assessment, prioritizes exercises with reference text and labeled phoneme data, considers forced alignment/GOP/pretrained acoustic classifiers, and does **not** treat GOP alone as proof of error type. It explicitly calls for calibrated/verified classification and limiting conclusions/repeating the exercise at low confidence. RQ1 includes accurate, clear, appropriate corrective feedback; RQ2 concerns Multi-Agent content generation. The thesis proposes methods for investigation rather than mandates that any alignment implementation must win. A segmentation-free alternative is a reasoned method refinement, but its superiority is a transfer hypothesis until tested.

### Verified Git state

- Outer `/home/khoa/Projects/KLTN/Dataset`: branch `research-phase`, HEAD **`b25a7595360294798d8d06b37592dd65dd4c9c33`**.
- `git ls-tree HEAD repo`: gitlink **`baa4989243d5ff46b8b3bf5a3c280cdf2857382b`**.
- Nested `repo` HEAD independently resolves to that full SHA.
- Outer pre-existing status at this audit: modified `docs/process-checkpoints/INDEX.md`, dirty nested `repo`, untracked `manifests/RQ-RESEARCH-01/revalidation-2026-11-03/`. These are not this worker's edits.
- Nested dirty state: **14 Python script mode changes `100755 → 100644`**, and two deleted **PDF symlinks** (`references/papers/baevski2020_wav2vec2.pdf`, `zhang2021_speechocean762.pdf`). `git diff --stat` and `--summary` establish these facts; do not inaccurately describe the listed scripts as modified algorithm bytes. Do not repair this pre-existing state during research.

Byte comparison against `git -C repo show baa4989:<path>` found **outer file bytes = nested pinned object bytes = nested working bytes** for these 13 touchpoints:

- `src/langfens_prep/{model,dataset,collate,textgrid_parser,phoneme_normalizer,alignment,g2p,losses}.py`;
- `audit_scripts/{evaluate_e1_mdd,train_fold1_ctc_binary,evaluate_fold1_ctc_binary}.py`;
- `configs/{rq_model_01_e1_fold1,rq1_trn_02a_fold1}.yaml`.

Thus line citations below to those outer files also identify the **pinned baa4989 implementation bytes**, not an assumption that a dirty checkout was executed. Original E0 training was earlier: Git object **`caa56783113d40eac438e5f458cdb313b8932218`** exists and its dataset/trainer/model were separately inspected with `git show`. Its trainer lines 111–123 supplies CTC+binary only; lines 312–315 configure the two-head model. Its dataset lines 106–131 has the same realized-phone/deletion filtering, and model lines 308–323 freezes backbone/diagnosis. This confirms historical target/head semantics without asserting the entire current model equals its original training revision.

### Archived execution, not a new replay

The archived E1 training log `/tmp/e1_train_output/langfens-rq-model-01-e1-fold1-train.log:71–74,132–135` records clean pinned checkout of **baa4989** and E1 execution. Lines **1826–1827** record epoch 6, 7,200 microsteps, **3,600 successful updates, zero skipped AMP attempts**. Lines **2404–2405**, corroborated by `/tmp/e1_train_output/checkpoints/fold1/e1_fold1_alignment_aware/training_endpoint_report.json:1–12`, identify the fixed-budget SHA as **`3b686730cc0a1101a1bbc96b59dc7bd38a9fc3cc7da7effe26b2d9498f079d24`**. This matches archived E1 evaluation provenance. E1 training occurred historically even though the later benchmark was inference-only. No weights were loaded or rehashed here; `/tmp` remains ephemeral evidence, not durable model storage.

## 3. Evidence table

| Finding | Directly inspected underlying evidence | Evidence level and limitation |
|---|---|---|
| Exact phone IDs and normalization | `src/langfens_prep/phoneme_normalizer.py:5–23,53–123` | Pinned source; no audio/model execution |
| Deletion annotation is a canonical error, not a CTC blank target | `textgrid_parser.py:133–156,213–257`; `dataset.py:94–148` | Parser and target-construction code; physical silence/absence not measured |
| Two-head training freezes diagnosis/backbone | `model.py:434–456`; `train_fold1_ctc_binary.py:111–133,536–566` | Source and archived execution records, not fresh gradient verification |
| E1 soft prior/skip feature | `model.py:96–103,137–206,350–356,659–664`; E1 YAML `:1–43` | Actual formulation; no direct skip/monotonic loss |
| Historical evaluation unit | `evaluate_e1_mdd.py:111–206,291–387` | Omission-vs-correct AP excludes substitutions; realized-phone PER is a separate denominator |
| E0/E1 empirical comparison | `manifests/RQ-MODEL-01/evaluation/RQ_MODEL_01_E{0,1}_FOLD1_EVAL/E1_EVALUATION_REPORT.json:1–76`; summary `:1–51` | Archived same-harness results; not independently rerun |
| Train/val support | `TRAIN_COVERAGE_CENSUS.json`, `fold_n/{train,val}/spk_stats` (line starts below) | Reaggregated saved counts only; no new annotation access |
| Gold-free runtime path | `dataset.py:281–393`; `g2p.py:24–26,62–112,199–227`; evaluator `:344–359` | Runtime inputs separable from offline gold scoring; G2P mismatches remain |
| DIAG-02B supersession | `manifests/RQ1-DIAG-02B/evaluation-final/REPORT.json:1–36,884–1022`; scientific report §§1–3,7 | Archived partial result; no causal training GO |
| Primary GOP graph semantics | Cao et al., **https://arxiv.org/pdf/2507.16838v3**, §III-B/C/D, Fig.3, Eqs.5–13 | Primary PDF accessed and cached copy byte-verified; no Langfens reproduction |
| Author implementation seam | **https://github.com/frank613/CTC-based-GOP/tree/ffc4823330efe2c836691931a456ea9e3360871e**, `taslpro26/gop_sf_sd_norm.py:104–135,155–282` | Source read only; no license-named file in complete tree; reuse permission unresolved |

Recomputed SHA-256 values:

- Frozen split: **`7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`**.
- E0/E1 benchmark summary: **`006611ccf0796938d4114bf62a1d17979c0af9ece362415dc1600670e6b086a3`**.
- Train-coverage census: **`24cfa9cc48382f73078bd6b3498aa57f30f109ef92419ddeb3a11d2d482eeb53`**.
- Downloaded arXiv v3 PDF: **`6f4c5396dd20218c5a94bc8fa317ba56c1b31bfaa9bbde71670504f6acbdc58d`**; equals existing `/tmp/langfens_sfgop.pdf`. The extracted text `/tmp/langfens_sfgop.txt` was read; primary sections/equations, not PDF text line numbers, are citations.
- Pinned author `taslpro26/gop_sf_sd_norm.py`: **`2fb86f3120874f155e4b32d69ff4b0427c5d5a35ba0aaa6c11081d9cff670f66`**, downloaded to memory and equals the inspected cached source. README likewise byte-matched. GitHub tree API returned the full pinned tree (`truncated=false`), no license-named entries. The README links IEEE document **https://ieeexplore.ieee.org/document/11355844**, but that IEEE page itself was **not accessed** here.

These hashes verify bytes, not causal claims, licensing rights, current remote durability or model correctness.

## 4. Actual target, annotation and inference contracts

### 4.1 Label identities and target construction

`STANDARD_39_ARPABET` fixes alphabetical phone order. CTC and canonical embeddings use blank/pad **0**, phones **1–39**; diagnosis uses phones **0–38**. They are different spaces: a diagnosis ID of 0 means **AA**, not CTC blank. Binary 0 means correct, binary 1 means strict annotated error; `-100` masks labels. `AX` folds to `AH`, stress and `*` do not survive in the 39-phone token identity, while normalization metadata retains them (`phoneme_normalizer.py:5–23,77–123`). No explicit `<DELETE>`, `<SIL>` or `<UNK_PHONE>` is a CTC class.

`textgrid_parser.py:159–211` treats every `s` tag as strict binary error, including same-base accented variants. Diagnosis is valid perceived phone only when known and not `err`. `:213–235` retains a deletion's canonical index/phone and binary label 1, assigns realized string `sil`, and sets diagnosis target `None`. `:237–258` excludes additions from canonical tokens but keeps their perceived phone as an acoustic event.

`dataset.py:94–148` constructs:

- **Canonical input** from TextGrid canonical intervals for the training/Protocol-O route, including deleted phones.
- **CTC target** from chronological realized phones across all intervals: skip silence, skip deletions, skip `err`, retain valid correct/substitution/addition phones.
- **Diagnosis target** only for eligible substitutions, otherwise `-100`; auxiliary perceived-phone target includes eligible correct/substitution tokens, excludes deletion.

This separation is correct for realized-phone transcription. It does not mean CTC has no information relevant to omissions: training on a shortened realized sequence can provide acoustic sequence evidence. It means there is **no canonical-position deletion class or explicit deletion-localization objective in this CTC contract**. Conversely, replacing the realized target by canonical phones would encourage the model to emit an expected-but-unspoken phone and would change what PER means; it is not a neutral repair.

**Documentation correction needed, not applied:** `manifests/RQ1-STD-01/MDD_LABEL_SCHEMA.md:60–71` still illustrates deletion CTC as `<SIL>/Blank`, uncertain CTC as `<UNK_PHONE>`, and deletion diagnosis as `<SIL>`. Actual executed source excludes/masks these. Future reporting should describe executed semantics and append an erratum rather than silently reinterpreting old metrics.

### 4.2 Zero-duration and transition intervals

The parser preserves `xmin,xmax` on a deletion without requiring positive duration (`textgrid_parser.py:213–235`). Binary and CTC targets are independent of interval width, so a zero-duration deletion, if present, remains a binary error and is still omitted from CTC targets.

The optional oracle pooling utility **forces at least one frame** even for a zero-width/subframe interval (`alignment.py:23–25,50–65`). E1 attention diagnostics similarly clamp to at least one frame (`evaluate_e1_mdd.py:374–385`), using floor start/ceil end at nominal 20-ms stride. That is a computational convention, not evidence that an absent phone acoustically occupies a frame. Do not interpret the forced width as a measured phone duration or deployable omission localization.

**Uncertainty:** the inspected train/val aggregate census only supplies duration bins, including `<=20ms`; it cannot establish the exact count of zero/negative durations. No raw labels were revisited to fill that gap. Annotation transition spans, zero waveform amplitude, stop closures, CTC blanks and omissions must remain distinct. DIAG-02B's independent endpoint observation concerns annotation geometry, not an observed segment of an absent sound.

### 4.3 Alignment and losses

Canonical embeddings plus position embeddings query continuous frame keys/values (`model.py:126–135,191–207`). E1 prepends a learnable vector, not an acoustic silence sample (`:137–156`). Its bias is exactly **`-bandwidth * ((i+0.5)/N_b - (t+0.5)/T_b)^2`**, with per-utterance valid lengths; skip bias is zero (`:157–187`). This is a soft positional preference, **not** a hard monotonic edit path and not a deletion-supervised skip probability.

`CTCLoss(blank=0,zero_infinity=False)` is used (`model.py:387`). CTC validation checks finite emissions, lengths including adjacent repeats, nonblank active targets and vocabulary range (`losses.py:206–330`). Binary BCE explicitly masks `-100` and validates 0/1 targets (`losses.py:6–70`). Loss assembly sums supplied losses with weights (`model.py:787–794`); `train_fold1_ctc_binary.py:111–133` supplies only CTC and binary targets. No alignment, skip-supervision, duration, subtype or edit loss is present in E0/E1.

`configure_two_head_training()` freezes all backbone and diagnosis parameters and keeps the backbone in eval even during head training (`model.py:434–456`). Diagnosis logits are nevertheless computed (`:664–678`). **Output existence is not trained competence**. “Unused diagnosis head” means unused in the objective and not validated for diagnosis, not absent from state dictionaries.

### 4.4 Runtime versus offline scoring

The deployable model receives audio and canonical prompt IDs; the cross-attention branch ignores `phone_intervals`. An explicit `alignment_mode='oracle'` branch can pool gold windows and is non-deployable (`model.py:575–607`). E1 config says cross-attention. No gold timestamps enter the evaluator's model calls (`evaluate_e1_mdd.py:308–314,347–352`).

Protocol O still gets canonical phones from gold TextGrid **token identities** (`:301–308`), so it is not fully oracle-free. Protocol Z derives input phones from written prompt + pinned CMUdict (`:344–352`). Gold canonical labels and boundaries are read in the evaluator for **offline alignment, scoring and attention diagnostics**, not passed to the Z model. The whole evaluation script is therefore not a zero-TextGrid program even though its Z inference inputs are gold-free.

`align_canonical_sequences()` (`alignment.py:76–223`) is deterministic reference-to-reference G2P/gold alignment, with substitution-first ties by default. Its returned “deletion” is an unaligned **gold canonical versus G2P** token, not an acoustic omission. It is not CTC Viterbi alignment. Protocol-Z aligned-only AP covers **9,919/9,956** tokens here, not all canonical tokens; lost coverage requires explicit reporting, not hidden denominator changes.

CMUdict checksum is pinned in `g2p.py:24–26`; dictionary loading excludes alternative pronunciations (`:95–112`) and strict OOV raises (`:199–207`). Stress collapse, one-primary-pronunciation choice, AX folding and OOV handling are consequential compatibility limitations for phonological/recognition methods.

**Static integration defect:** `collate.py:37–43` leaves `padded_aux_targets` undefined in the blinded/None-label branch, yet returns it at `:87`. This would obstruct a generic blinded/ZeroOracle dataset + existing collator route. The current E1 Z evaluator builds tensors directly and bypasses that collator, so this does **not** invalidate the archived Z result. No runtime reproduction or fix was attempted.

## 5. What the saved dataset and E0/E1 evidence actually support

### 5.1 Authorized train/val aggregates

I summed `spk_stats` from the existing `TRAIN_COVERAGE_CENSUS.json`, without opening labels/audio. Key starts: Fold-1 train/val speaker statistics **:274/:620**; Fold-2 **:922/:1265**; Fold-3 **:1568/:1920**; Fold-4 **:2225/:2570**.

| Fold | Train canonical / substitution / deletion | Validation canonical / substitution / deletion | Train / val speakers |
|---|---:|---:|---:|
| 1 | 79,426 / 9,377 / 2,464 | 9,956 / 1,012 / 145 | 16 / 2 |
| 2 | 79,480 / 9,635 / 2,324 | 9,873 / 822 / 99 | 16 / 2 |
| 3 | 79,472 / 9,653 / 2,522 | 9,881 / 896 / 227 | 16 / 2 |
| 4 | 79,499 / 10,045 / 2,321 | 9,866 / 854 / 158 | 16 / 2 |

Frozen metadata confirms train/val speaker disjointness and eight distinct validation speakers. None of the four Vietnamese speaker IDs HQTV/PNV/THV/TLV is in validation. A speaker can legitimately rotate between train/test across different folds; no cross-fold label/audio pooling is authorized by this audit.

These counts establish available labeled support, **not statistical sufficiency, adequate subtype coverage, annotation reliability or guaranteed generalization**. Fold-1 train has four phone identities with zero deletions (`AW,OW,OY,ZH` in saved `phone_stats`), so “39 phones covered” does not mean deletion coverage for all phones. Overlapping folds are not independent new datasets. The report's categorical “fully sufficient/no manual labeling” claims are stronger than counts alone justify. Broad new labeling is not demonstrated necessary either; targeted adjudication and deployment-domain coverage would need separate approval.

Dataset license provenance, as **registered not re-fetched from official sites here**, is L2-ARCTIC release v5.0 / **CC BY-NC 4.0**, source **https://psi.engr.tamu.edu/l2-arctic-corpus/**; SpeechOcean762 / **CC BY 4.0**, source **https://openslr.org/101/** (`manifests/RQ1-SPLIT-01/SPLIT_PROVENANCE.json:60–84`). Maintain private/noncommercial constraints; these metadata entries do not settle every downstream checkpoint or deployment license. SpeechOcean's dataset class returns multidimensional scores and words (`dataset.py:176–259`), not the same L2-ARCTIC strict deletion/substitution labels. It cannot be silently pooled as equivalent omission supervision.

### 5.2 Archived matched-harness comparison

| Metric, Fold-1 validation | E0 | E1 | E1−E0 |
|---|---:|---:|---:|
| Omission AP | 0.035020 | 0.032252 | −0.002768 |
| Omission recall at correct-token FPR ceiling 15% | 0.400000 | 0.365517 | −0.034483 |
| All-error binary AP | 0.317021 | 0.309293 | −0.007728 |
| Realized-phone greedy CTC PER | 0.236818 | 0.232365 | −0.004453 |

Underlying JSON paths are in the evidence table. Both contain 300 utterances, 9,956 canonical tokens, 8,799 correct/1,012 substitutions/145 deletions, and **9,881 realized CTC reference tokens**. Omission AP is computed against correct tokens only (`evaluate_e1_mdd.py:129–139`), so its chance-prevalence reference is **145/(8,799+145)**, not 145/9,956. Recall thresholds are selected on these validation correct tokens (`:141–156`), not an independent deployment calibration set.

The E1 skip mass on deletions is reported **0.000000 after six-decimal rounding**, versus correct mean **0.000031**. Source averages then rounds (`:421–438`). Say “no useful separation at reported precision,” not proof that every omission receives mathematically exact zero or that a particular causal mechanism has been identified.

Do not overwrite historical E0 AP **0.316062** (`RQ1-TRN-02A/EVALUATION_REPORT.json:1–9`) with **0.317021** from this later harness. The latter is the appropriate comparator for the E1 replay; differing backends/precision/path provenance are not automatically equivalent, and no new parity experiment was authorized.

**Concrete documentation error:** RQ-MODEL-01 checkpoint §3.3 reverses NJS/RRBI language labels. Actual evaluator `DEMOGRAPHICS` at `evaluate_e1_mdd.py:83–94` says **NJS Spanish, RRBI Hindi**, consistent with each JSON's `disaggregated_by_l1` counts (5,011/4,945). Preserve the numerical arrays and append correction to attribution, not relabel the samples.

DIAG-02B's underlying JSON remains `PARTIAL_ARCHIVE_PARITY_FAILED` / `NO_GO_ALIGNMENT_TRAINING_CAUSAL_GATE` (`REPORT.json:1–3`): max archived probability drift **0.001304954** vs tolerance **0.0002** at `:884–887`; the matched blank AUC **0.652663** at `:1021` is modest and interval-conditioned, not deployable omission accuracy. Historical forced-attention/blank-readiness claims cannot be used as renewed training authorization. No source-only audit resolves that parity limitation.

## 6. Candidate compatibility: interfaces and unproven assumptions

All rows are **design audit / transfer hypotheses**, not new empirical performance findings. No timing, GPU-memory or latency estimate is offered.

| Candidate | Reusable seam | Required new work/contract | Unsupported assumption or failure mode |
|---|---|---|---|
| **GOP-SF-S** | Frozen `ctc_logits`, active frame lengths, G2P canonical phones | New marginal graph scorer; wildcard nonblank labels; repeated-label tests; score orientation/calibration | S graph permits one alternative phone, **not deletion**; low score does not identify subtype |
| **GOP-SF-SD** | Same 40-class emission inventory; no gold timing required | Optional target-phone graph (`.?`), correct blank/repeated-neighbor path accounting, normalized/scaled or log-space numerics | Deletion hypothesis in denominator is not a calibrated deletion posterior; canonical surrounding context can be wrong |
| **GOP-SF-SDI** | Same emissions and canonical contexts | Arbitrary target-region sequence graph (`.*`); explicit insertion handling and diagnostic attribution | More flexible denominator can absorb neighboring errors; insertions are not canonical-token labels in current benchmark |
| **GOP self-alignment / forced canonical alignment** | Existing CTC outputs; canonical IDs | A new CTC alignment algorithm (none in current `alignment.py`), failure diagnostics, deployable confidence | Mandatory canonical phone occupancy can fabricate a segment for a deleted phone; CTC activation spans are not phonetic boundaries |
| **Free-phone greedy/beam + DP** | Existing greedy collapse and Levenshtein edit counting | Canonical-versus-predicted alignment with token correspondence, uncertainty, score extraction; new tie policies versioned separately | Recognizer deletion on a correctly spoken phone and true omission are confounded; equal-cost/repeated-phone alignment is ambiguous |
| **Correct/substitution/deletion subtype head** | Canonical annotation error type and pooled acoustic/query features | New multiclass targets/collator/loss/head and probability mapping `P(error)=P(sub)+P(del)` | Dedicated loss may learn phone/position/L1 priors, not local acoustic absence; imbalance treatment alone need not improve ranking |
| **KEEP/DELETE/REPLACE edit-aware tagging** | Canonical token index, known perceived phone and deletion flags | New edit inventory, uncertainty/distortion policy, binary/subtype projections and optionally insertion events | Same-base strict errors collide with KEEP or REPLACE-to-self; `err` has no valid replacement; direct speech evidence required |
| **Retrieval / exemplar hybrids** | Frozen acoustic representations, phone IDs, optional model-derived segments | Train-only per-fold index, privacy/license/provenance policy, calibrated distances/rejection | No emitted segment exists to retrieve for an omission; nearest neighbor can return false reassurance or identity/accent priors |
| **Phonological attributes / confusable-phone priors** | Normalized phone identities and optional learned acoustic features | Audited attribute ontology, phone-map validation, auxiliary target/evaluation design | Deterministic attributes from predicted phones add no independent acoustic evidence; they cannot create deletion grounding |

### 6.1 GOP primary evidence and implementation caveats

Cao et al. **arXiv:2507.16838v3**, §III-B/C, Eqs.5–12 and Fig.3 defines a canonical sequence numerator and relaxed target-region denominator, with alignment uncertainty marginalized. The S/SD/SDI distinction is **one phoneme / optional phoneme / arbitrary phoneme sequence**, respectively. The score is a difference of negative-log CTC losses, **`L(relaxed)-L(canonical)`**, a goodness score; low goodness must be oriented appropriately for error-oriented AP. It is not the same as “average blank probability in an expected window.” §III-D/Eq.13 normalizes by model activation occupancy with a floor of 1, not a gold or physically measured phone duration. Correct-phone CTC blank occupancy is often high; conventional frame-averaged GOP is not automatically compatible with peaky CTC emissions.

The author implementation at pinned commit **ffc4823330efe2c836691931a456ea9e3360871e** is inspectable, but **not drop-in Langfens code**:

- It imports/loads `Wav2Vec2ForCTC` and processor/tokenizer and uses a local CSV/CTM dataset pipeline (`gop_sf_sd_norm.py:1–19,36–98,256–316`), whereas Langfens is a custom multihead state dictionary with its own phone order.
- Its useful seam is the probability-matrix DP routine, not loading an unrelated author model or replacing the frozen backbone.
- `ctc_loss_denom` has explicit skip/empty/repeat-path and adjacent-next-label disambiguation logic (`:171–265`); these cannot be approximated by blank counts without changing the method.
- The file writes **raw GOP and occupancy as separate columns** (`:275–282`). The suffix `norm` also relates to normalized forward variables/numerical stability; do not assume the written scalar has already been divided by occupancy. Equations and downstream feature code must be reconciled before adaptation.
- The complete pinned repository tree has no license-named file; **publication availability is not code-reuse permission**. A clean-room implementation from the published mathematics, with appropriate citation and license review, is distinct from copying code. No code copied here.

The primary paper has results on its own datasets/models/tasks. I do not import its AUC/correlation values as Langfens omission AP or promise a gain. SD is structurally attractive for the existing canonical-token omission problem; S is an ablation, SDI an optional insertion-robustness question, not automatically a better variant.

### 6.2 Free-phone recognition and forced alignment are different experiments

`evaluate_fold1_ctc_binary.py:60–70` greedily collapses repetitions then strips blank; `:73–132` counts edit distance. The historical PER reference is realized speech. A proposed omission detector must instead compare **known canonical prompt** to **predicted realized phones**, while preserving the historical realized-phone PER guardrail separately. That creates a new error detector, not a reinterpretation of existing CTC deletion counts.

Equal-cost DP choices, merged repeated phones, unsupported/uncertain realizations and G2P variant disagreement need explicit ambiguity/abstention handling. Hard edit indicators produce highly tied scores and can have an unhelpful fixed-FPR operating point; a continuous uncertainty score requires a specified derivation/calibration. Neither canonical-G2P alignment nor a best CTC path independently localizes an acoustic omission.

Ordinary forced alignment of a fully canonical sequence requires the canonical labels to occur on a valid path. A short assigned span may be pressure from that constraint, not a real omission detector. Alignment can still serve as an investigational baseline or locate realized phones; it should not be categorically rejected based on an unmeasured runtime claim, nor endorsed as proven omission localization. A skip-enabled/marginal graph is a materially different contract.

### 6.3 Subtype/edit supervision and feedback

Annotations permit a compact subtype objective, but existing returned samples do not expose subtype tensors. New target construction is needed. Keeping the frozen backbone and original CTC loss is feasible structurally; adding a third supervised head/loss changes the **two-head training contract**, requiring explicit approval and separate arm identity/checkpoint metadata.

A proposed 41-class inventory (`KEEP`, `DELETE`, 39 `REPLACE_phone` tags) cannot simply equate “no phone identity edit” with correct under the strict benchmark. `CPL,CPL*,s` is strict error with identical normalized phone; `CPL,err,s` is error without a known replacement. Options include explicit distortion/unknown tags, an independent binary path, or documented masking for replacement accuracy while retaining those tokens in binary evaluation. None is chosen here. Additions need a between-token event representation, not forced DELETE/replacement labels. **DELETE supervision alone is not evidence that the model attends to absence acoustically.**

GECToR itself is a text grammatical-editing precedent, not a validated speech-MDD result. No GECToR primary paper or speech implementation was independently accessed in this lane; no performance claim is made. A classifier over speech+prompt features is a new supervised speech adaptation, not validated merely by naming it “GECToR-Speech.”

Retrieval/phonological explanations can follow validated, uncertainty-aware acoustic evidence. They must not escalate a phone-prior guess into a confident claim about a learner's tongue position or the physiological cause of an error. The thesis's abstention/re-recording provision is compatible with low-confidence or conflicting detectors and independent feedback evaluation.

## 7. Checkpoint structural compatibility and approval boundaries

Source-backed compatibility, **not a fresh checkpoint-load test**:

- E0 default structure preserves a 40-output linear CTC head, 39-output diagnosis head and **896-input** binary head; no skip parameter is registered in state dictionaries when disabled (`model.py:96–101,347–365`). The existing strict-load regression exists (`tests/test_model_e1_alignment.py:176–193`) but was **not run**.
- E1 adds `cross_attention_module.skip_token` and changes first binary layer width to **897**. E0↔E1 is **not** strict interchangeable state loading. A scorer operating outside the model can retain E0 weights byte-for-byte.
- `evaluate_e1_mdd.py:229–267` reconstructs skip/bias from checkpoint config, additionally detects width 897, then uses `strict=True`. The monotonic bias is a configuration-controlled computation, not a tensor guaranteed by state-dict shape. A missing/stale `monotonic_bias` metadata value could yield a structurally strict load with different inference semantics. Preserve full config and source identity; shape checks alone are insufficient.
- Replacing binary/diagnosis heads by subtype/edit heads invalidates strict loading for changed tensors. “Load backbone/CTC, initialize new head” is **partial initialization**, not full-checkpoint compatibility or resume. Optimizer/scheduler/scaler/RNG cannot be blindly reused after a trainable-parameter/schema change.
- Trainer checkpoints atomically store model, optimizer, scheduler, scaler, RNG, config, counters and provenance (`train_fold1_ctc_binary.py:137–183`). Their fixed-budget marker requires exactly 3,600 successful updates at epoch index 5 (`:155–160`). `max_epochs:25` is the scheduler horizon; E1's execution cap is 10 and early stopping can terminate earlier. Config presence alone does not prove any fold was trained or authorized.
- E1 Fold-2–4 YAMLs/tests are implementation artifacts, **not rollout authorization**. Existing disposition remains no rollout. Any new head/loss, alignment supervision, calibration fitting, new inference pilot, remote staging, TEST evaluation or deployment requires separate operator approval. Cross-fold initialization must respect the original-pretrained, no-Fold-1-warm-start rule.

## 8. Missing proposal search: bounded, explicit limitation

Searched file inventory and documentary text in:

1. Outer `docs/`: **7** `.md/.txt/.rst/.yaml/.yml` files.
2. Outer `manifests/`: **261** such files, including both existing RQ-RESEARCH-01 sessions.
3. Nested `repo/docs/`: **7** such files.
4. Nested `repo/manifests/`: **240** such files.
5. `/home/khoa/Projects/KLTN` **top-level files only**: the original thesis Markdown and `mount_colab_drive.py`; no additional top-level text proposal was listed. No descent into datasets or unrelated projects.

Filename checks included `omission`, `deletion`, `proposal`; content checks included combinations of latest/omission/deletion/proposal, Vietnamese đề xuất/bỏ/âm, and GOP-SF/SD/SDI. The researcher did not search raw datasets or archives. References to a missing proposal in prior research reports were discovery leads only, not independent evidence that a proposal exists.

**Nearest located documents, directly read:**

- `manifests/RQ1-DIAG-01/checkpoints/01-phoneme-omission-and-false-negative-root-cause.md`, **§11, lines 264–289**: CTC-guided alignment, position/duration priors, 3-class subtype experiment proposals. Numeric gates are proposed targets, not observed successes; their causal rationale must be read with DIAG-02B corrections.
- `manifests/RQ-MODEL-01/RQ_MODEL_01_REPORT.md`, **§§4–6**: sufficiency interpretation and implemented E1 skip/prior design; superseded on readiness/outcome by the benchmark artifacts. Its causal/“completely sufficient” prose is not independent empirical proof.
- RQ-RESEARCH-01 original and revalidation process checkpoints: existing synthesis/continuation state, not a separately authored omission proposal.

**Stop limitation:** cannot certify conformance to an unidentified latest proposal. Coordinator should ask the operator to identify/provide it before claiming that requested source was reviewed. This does not prevent the source/data contract audit above; it prevents presenting the nearest historical document as a substitute.

## 9. Actionable corrections, uncertainty and safe restart

### Corrections to carry forward without historical rewrites

1. Append documentation errata for actual realized-phone CTC/masked diagnosis semantics, the NJS/RRBI attribution error, and “E1 benchmark inference-only” versus “E1 never trained.”
2. Describe rounded skip means accurately; do not claim exact per-token zeros or a proved mechanism from rounded aggregates.
3. Preserve historical baseline AP and denominators; label the later same-harness E0 comparator separately. Keep omission-vs-correct, all-token omission prevalence, canonical DP edits and realized PER distinct.
4. Register subtype/edit label policies for same-base accented errors, `err`, insertions and zero-duration annotations before implementation. Preserve strict binary evaluation unless an explicitly separate sensitivity contract is approved.
5. Treat a CTC scorer as a sidecar candidate; specify graph semantics, confidence orientation, repeated-label/adjacent-phone behavior, numerical scaling and denominator coverage. Do not alter weights merely to make an adapter load.
6. Record the static blinded-collator defect and potential config-only inference mismatch for a future authorized change; neither is fixed here.

### Residual risks

- No fresh weights inspection, model/checkpoint load, regression run, scientific reproduction, calibrated operating point or deployment benchmark was performed. Existing test files are inspected specifications, not current test PASS evidence.
- E1's single-fold/two-speaker negative result is not a statistically decisive comparison of all omission methods; no new confidence interval is inferred.
- Aggregate support does not resolve subtype ambiguity, zero-duration counts, phonological validity, out-of-domain accents or Vietnamese deployment generalization.
- DIAG-02B remains partial; source inspection does not establish its archive-drift cause.
- Author GOP code reuse license is unresolved; licenses for other candidate code/checkpoints were not independently checked. L2-ARCTIC and SpeechOcean license statements above are registered provenance, not newly fetched official license text.
- No direct retrieval/phonology/GECToR primary-study result is attested by this lane. Compatibility analysis must not be cited as empirical endorsement.
- Missing latest proposal remains a source-coverage limitation.

### Actual reads/operations and restart state

Read `AGENTS.md`, the global index (both portions), RQ-RESEARCH-01 original/revalidation checkpoints, RQ-MODEL-01 report/checkpoint, DIAG-01 §11 and adjacent source-audit discussion, DIAG-02B checkpoint/scientific report and selected underlying JSON fields; read the source/config/evaluator files and normalization/alignment/data contracts cited above. Selected existing audit summaries were inspected for zero-duration metadata; none supplied the missing count. Small JSON scripts reaggregated only saved train/val counts and frozen split metadata. Primary GOP PDF, README and DP source were retrieved/read as described. A bounded `/tmp` filename search encountered permission-denied system-service directories; no protected-directory traversal/retry occurred. Initial guessed source names `phonemes.py`/`textgrid.py` did not exist; actual files `phoneme_normalizer.py`/`textgrid_parser.py` were then read.

**Current model state in this process:** not started; **0 successful optimizer updates, 0 skipped updates, no new model artifact**. Last-valid model identities are unchanged archived E0/E1 endpoints; no durability revalidation occurred. No files staged, committed or pushed. Coordinator owns the required process checkpoint/index update; this worker writes none in the repository.

Safe restart: verify outer HEAD and nested pinned SHA above, preserve all pre-existing dirt, recheck the three lightweight artifact hashes, read coordinator closure and DIAG-02B disposition, obtain the missing proposal identity and explicit authorization before any next action beyond documentary research. Do not execute commands printed in historical launch/readiness documents. **Stop at research closure.**

```acceptance-report
{
  "criteriaSatisfied": [
    {"id": "criterion-1", "status": "satisfied", "evidence": "Self-contained source/data compatibility audit with pinned-source byte checks, line citations, archived empirical results, candidate contract boundaries and explicit missing-source/residual-risk statements."}
  ],
  "changedFiles": ["/home/khoa/.pi/agent/sessions/--home-khoa-Projects-KLTN-Dataset--/subagent-artifacts/outputs/723092bc-f51f-4a2a-8cd9-bf63b7c93d0b/reports/B_SOURCE_DATA_COMPATIBILITY.md"],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {"command": "git rev-parse/branch/status/ls-tree; git -C repo rev-parse/status/diff --summary", "result": "passed", "summary": "Verified outer b25a759 and nested baa4989; nested script dirt consists of mode changes plus two deleted PDF symlinks."},
    {"command": "Read-only Python byte comparison against git show baa4989 for 13 source/config paths", "result": "passed", "summary": "All 13 outer and nested working file byte contents match pinned objects."},
    {"command": "Read-only SHA-256 and JSON train/val/split-metadata aggregation", "result": "passed", "summary": "Frozen split, benchmark summary and census hashes computed; saved train/val counts reconciled."},
    {"command": "Primary arXiv v3 and pinned GitHub source/tree HTTP reads to memory", "result": "passed", "summary": "PDF and two source documents matched inspected cached copies; complete pinned tree contains no license-named file."},
    {"command": "Bounded /tmp filename discovery", "result": "failed", "summary": "Useful archived documentation paths found; unrelated protected system-service directories returned permission denied; no retry."},
    {"command": "Unit/model/inference/training tests or experiments", "result": "not-run", "summary": "Not authorized for this source-only research lane."}
  ],
  "validationOutput": ["13/13 critical source/config byte comparisons equal pinned baa4989", "No raw dataset/archive or sealed TEST audio/labels read", "Only isolated report output written; repository unchanged by this worker"],
  "residualRisks": ["Distinct latest omission proposal not located in enumerated scopes", "No model replay or weight rehash; archived evidence only", "Subtype/edit policies and code-reuse licensing unresolved", "DIAG-02B archive parity remains partial"],
  "noStagedFiles": true,
  "diffSummary": "One isolated Markdown report; no repository edits.",
  "reviewFindings": ["Static issue: src/langfens_prep/collate.py:37-43,87 leaves padded_aux_targets undefined for blinded labels", "Documentation issue: RQ-MODEL-01 checkpoint reverses NJS/RRBI language labels", "Contract issue: legacy label-schema examples do not reflect executed deletion/err CTC and diagnosis masking"],
  "manualNotes": "Research-only closure contribution. No scientific PASS, new experiment, implementation, training authorization or independent latest-proposal review is claimed. Coordinator owns process checkpoint/index update."
}
```
