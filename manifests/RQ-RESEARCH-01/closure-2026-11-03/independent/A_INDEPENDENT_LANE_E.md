# Independent Lane E reconstruction — acoustic realization and compact joint/edit-aware MDD

**Disposition:** research-only contribution to closure of existing **RQ-RESEARCH-01**; no architecture selected, no pilot executed, no implementation or training authorization implied.

## 1. Scope, independence and actual execution

Source order used: **original thesis proposal → source/archived experiment artifacts → directly accessed primary papers → explicitly marked transfer hypotheses**. Repository references below are relative to `/home/khoa/Projects/KLTN/Dataset` unless absolute.

- Verified outer HEAD `b25a7595360294798d8d06b37592dd65dd4c9c33`, branch `research-phase`; nested `repo` HEAD `baa4989243d5ff46b8b3bf5a3c280cdf2857382b`.
- Read both `AGENTS.md` files, the global checkpoint index, RQ-RESEARCH-01 original/revalidation checkpoints, RQ-MODEL-01 checkpoint and DIAG-02B checkpoint **as handoff/state records, not independent scientific evidence**.
- **Independence qualification:** required checkpoint reads incidentally exposed the historical ranking table in the original research checkpoint and the revalidation checkpoint's coordinator disposition. I did not open any prior Lane E report, coordinator supplement, integrated decision report, or another child's report. Those rankings and interpretations are not inputs to the comparisons below. Thus this is a separately derived analysis with disclosed incidental context exposure, not a claim of perfectly blinded reconstruction.
- Directly read the original thesis `/home/khoa/Projects/KLTN/DC KLTN0127 23520772&23520757_TranThiHongYen (1).md`, including its bibliography. Because most content occupies one exceptionally long Markdown line, also extracted the RQ1, Speaking methodology and research-limitations sections **in memory** to inspect the complete relevant text without truncation.
- Inspected source files, documentation, train/validation aggregate census, archived metric JSON, and primary-paper text. Network access used `requests`; PDF text extraction used `pdftotext` through stdin/stdout. No paper download was deliberately persisted in the repository. Harness-generated command-output logs in `/tmp` were read where tool truncation hid earlier output.
- **No** dataset archive traversal, WAV/TextGrid sample access, TEST audio/labels, model loading, inference, fitting/calibration, backward/optimizer, GPU, remote deployment, commit or push. Current research training state: **0 successful updates; 0 skipped updates; no new model artifact**.
- The only deliberate write is this harness-bound report. Both Git indexes were empty at final check. Nested dirty source/deletions were preserved. A new outer `manifests/RQ-RESEARCH-01/closure-2026-11-03/` directory appeared during this read-only work; it was not opened or written by this agent and is not evidence here.

**Main result:** the data/source contract already supports more acoustic-realization supervision than a binary-only interpretation suggests, but it does **not** already support a trained DELETE-aware diagnosis head, complete insertion supervision at canonical slots, or validated omission localization. Compact supervised tagging, free-phone recognition plus alignment, skip-aware probabilistic scoring, and reference-aware joint learning remain distinct hypotheses. Their evidence should not be collapsed into “GECToR-Speech is validated.”

## 2. Thesis authority: what actually needs to be demonstrated

The original proposal, **§2.3 RQ1**, asks about phoneme-level acoustic detection of selected pronunciation errors and the accuracy, clarity and appropriateness of resulting instructional feedback. **§3, Speaking/limitations** prioritizes reference-text exercises and explicitly does not promise complete error detection in unrestricted conversation. **§4(c), Xây dựng hệ thống luyện Speaking**, separates conversational STT from pronunciation analysis; considers alignment plus GOP or pretrained acoustic classifiers; identifies substitutions, final-consonant omissions and confusable pairs as candidate targets conditional on labels; and requires low-confidence restraint/re-recording. GOP alone is expressly not independent proof of an exact error type.

Consequences:

1. The thesis permits comparing compact acoustic classifiers with alignment/scoring methods. It does not mandate an autoregressive decoder, a GECToR derivative, or an exhaustive edit taxonomy.
2. Its proposed forced-alignment step is a method to investigate, not evidence that canonical-only forced alignment correctly localizes an absent phone. Any methodological departure should be explained as handling an absent realization, not as ignoring the proposal.
3. GECToR appears in **§4(f), Writing/Grammar**, and bibliography **[10]**. That establishes relevance to structured text editing, **not speech-MDD validation**.
4. RQ1 feedback quality needs its own evaluation; predicted articulatory attributes or a phone substitution cannot establish a physiological cause of the learner's error. RQ2 remains Multi-Agent content generation, not an acoustic-model lane.

## 3. Evidence and provenance ledger

“Accessed” means source text or artifact was actually inspected in this run. Archived metrics were **not re-executed**.

| ID | Underlying source actually accessed | Supports | Does not establish |
|---|---|---|---|
| T | Original thesis absolute path above; §§2.3, 3, 4(c), 4(f), 4(g), bibliography [10] | Targeted reference-aware acoustic MDD, restrained feedback; GECToR is a Grammar reference | A speech-tagging architecture or its expected performance |
| R1 | `repo/src/langfens_prep/textgrid_parser.py:149–262`; `dataset.py:98–149` | Substitution/omission/addition metadata; realized-phone reconstruction; masking policies | Completeness/quality of raw annotations, independently checked here |
| R2 | `repo/src/langfens_prep/model.py:52–207, 346–414, 434–456, 658–794`; `repo/audit_scripts/train_fold1_ctc_binary.py:536–569` | Current cross-attention, CTC, binary/diagnosis heads and two-head freeze contract | A trained third head, deletion probability from skip attention, causal grounding |
| R3 | `repo/src/langfens_prep/collate.py:27–95`; `losses.py:162–206` | Existing labels/tensors and ignored-target behavior | A supplied subtype tensor or canonical-gap insertion targets |
| R4 | `manifests/RQ-MODEL-01/TRAIN_COVERAGE_CENSUS.json`, `fold_1`…`fold_4` → `train`/`val` → `phone_stats` | Aggregate train/validation support, including sparse/zero omission cells | Pairwise realization coverage, insertions, confidence or new annotation audit |
| R5 | `manifests/RQ-MODEL-01/evaluation/E1_FOLD1_BENCHMARK_SUMMARY.json:1–57`; corresponding E1 `E1_EVALUATION_REPORT.json:1–77` | Archived E0/E1 identities, omission metrics and rounded skip behavior | New replay, statistical significance, precise unrounded skip values, independent weight verification |
| R6 | `repo/audit_scripts/evaluate_e1_mdd.py:108–206` | Historical omission AP = deletion versus correct only; substitution negatives excluded | All-token omission classification or insertion scoring |
| R7 | `manifests/RQ1-DIAG-02B/evaluation-final/REPORT.json:1–43`; checkpoint for state only | Artifact explicitly declares parity failure/NO-GO, validation population | Causal training readiness or numerical replay in this run |
| D | Archived original corpus documentation `repo/references/dataset-documentation/L2-ARCTIC/l2-arctic-corpus-docs.html:346–352`, canonical URL `https://psi.engr.tamu.edu/l2-arctic-corpus-docs/`; corpus page `l2-arctic-corpus.html:529`; `LICENSE.txt:1–3` | Original annotation conventions; archived release declaration **CC BY-NC 4.0** | Fresh live-site verification, commercial deployment clearance |
| P1 | Omelianchuk et al., **GECToR**, BEA 2020, §§3–6, pp.164–166; `https://aclanthology.org/2020.bea-1.16.pdf`, DOI `10.18653/v1/2020.bea-1.16` | Token edits, softmax tagging, iterative correction on **written GEC** | Acoustic alignment, speech diagnosis or speech speedup. ACL landing-page footer verifies paper materials CC BY 4.0; code/model license not audited |
| P2 | Zheng et al., **CoCA-MDD**, Interspeech 2022, §§2.1–2.4, 3, Table 1; `https://www.isca-archive.org/interspeech_2022/zheng22e_interspeech.pdf`, DOI `10.21437/Interspeech.2022-11155` | **Speech-specific** coupled reference/acoustic attention, CTC + phone prediction + binary classification and fusion | Transfer to frozen-base Langfens; direct supervised omission localization; author code/license not audited |
| P3 | Wu et al., **Transformer Based End-to-End Mispronunciation Detection and Diagnosis**, Interspeech 2021, §§2–3, Tables 1–2; `https://www.isca-archive.org/interspeech_2021/wu21h_interspeech.pdf`, DOI `10.21437/Interspeech.2021-1467` | Speech-specific free-phone CTC/encoder-decoder alternatives, evaluated on CU-CHLOE | Same corpus, same freeze regime, equal updates or omission-only advantage here; implementation/license unverified |
| P4 | Cao et al., **Segmentation-free Goodness of Pronunciation**, `https://arxiv.org/abs/2507.16838v3`, PDF `https://arxiv.org/pdf/2507.16838v3`; revised **5 Feb 2026**; §§III–V, Eqs.5–16, Tables I–II; metadata DOI `10.1109/TASLPRO.2026.3654852` | Speech-specific CTC alignment marginalization with S/SD/SDI alternatives; compact downstream models | Langfens replication; separate omission AP; validated insertion-event localization. arXiv page verifies **CC BY 4.0**; final publisher version and linked implementation not inspected |
| P5 | Ristad & Yianilos, **Learning String Edit Distance**, `https://arxiv.org/pdf/cmp-lg/9610005`, retrieved PDF identifies **v3, 2 Nov 1997**, §§1–2.4 | Stochastic substitution/deletion/insertion transduction; sum versus best path; DP and parameter tying | End-to-end acoustic MDD or learned omission grounding. License/author code not verified |
| P6 | Chen et al., **An Alignment Method Leveraging Articulatory Features for MDD in L2 English**, Interspeech 2022, §§2–3; `https://www.isca-archive.org/interspeech_2022/chen22l_interspeech.pdf`, DOI `10.21437/Interspeech.2022-10309` | Speech-specific reference fusion with phonological constraints; explicitly discusses false acceptance from misleading reference text and skip/unknown behavior | Reproducibility here, calibrated omission posteriors or compatibility with unchanged baseline weights; code/license unverified |
| P7 | Shahin & Ahmed, **Phonological-Level Mispronunciation Detection and Diagnosis**, Interspeech 2024, §§2–3, Table 2; `https://www.isca-archive.org/interspeech_2024/shahin24_interspeech.pdf`, DOI `10.21437/Interspeech.2024-2217` | Speech-specific attribute streams with separable CTC/shared blank and attribute-based analysis | Frozen-base equivalence; standalone omission evidence; measured tongue position. Code/license unverified |
| P8 | Ryu et al., **A Joint Model for Pronunciation Assessment and MDD with Multi-task Learning**, Interspeech 2023, §2, §3.1, §3.3, Tables 1–3; `https://www.isca-archive.org/interspeech_2023/ryu23_interspeech.pdf`, DOI `10.21437/Interspeech.2023-337` | Speech-specific CTC realized-phone recognition + sentence scoring; discusses extra realized transcripts in SpeechOcean762 | A reference-conditioned edit tagger; same label schema as local parser; an omission-specific gain. Code/license not audited |
| P9 | Chu et al., **Recognize Mispronunciations…One Edit Distance Finite State Automaton**, Interspeech 2020, §2.2, Fig.2, §3; `https://www.isca-archive.org/interspeech_2020/chu20_interspeech.pdf`, DOI `10.21437/Interspeech.2020-3109` | Speech-specific constrained S/D/I decoder and deletion as a bypass arc | An MDD detection benchmark: main reported experiment is non-native ASR improvement; no direct Langfens omission validation |
| P10 | Zhang et al., **End-to-End Mispronunciation Detection with Simulated Error Distance**, Interspeech 2022, §2, §3; `https://www.isca-archive.org/interspeech_2022/zhang22p_interspeech.pdf`, DOI `10.21437/Interspeech.2022-870` | Distinguishes text corruption from acoustic-unit perturbation; joint error/phone objectives | Natural omission realism, compact frozen-head equivalence or a deletion-specific guarantee |

Primary papers were read for method/results sections, not represented as fully audited implementations. ISCA PDF copyright notices do not by themselves verify a reusable software license. No literature performance/latency number is imported as a forecast for this repository.

### Verified byte identities

Recomputed in this run:

- Frozen split JSON SHA-256: `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`.
- E0/E1 summary JSON SHA-256: `006611ccf0796938d4114bf62a1d17979c0af9ece362415dc1600670e6b086a3`.
- Inspected nested `model.py`: `af2a1271c3e4a120f1505aa576bb2d75dc3ed90495b3b7ba675096ebb332f592`.
- Inspected nested `dataset.py`: `b23058e75777816d429b1097eabd637576b1d4b37f6cbf963e4e13a910daaa6c`.
- Retrieved GECToR PDF: `156298dbe8b4d829c6679cddeb811b1af0b30ea6d7f59bf40df3298f586b5934`.
- Retrieved CoCA-MDD PDF: `5a6329350d55d57389ee349ae70460af9c1aebaf9b80047ece9bd5a6570dab82`.
- Retrieved segmentation-free GOP v3 PDF: `6f4c5396dd20218c5a94bc8fa317ba56c1b31bfaa9bbde71670504f6acbdc58d`.

These identify bytes accessed, not experimental reproduction or model-weight integrity.

## 4. What supervision exists, and what remains missing

### 4.1 Realizations are available, with consequential exclusions

The original corpus conventions distinguish `CPL,PPL,s`, `sil,PPL,a`, and `CPL,sil,d`; uncertain realizations can be `err` and deviations marked `*` [D]. The documentation describes deletion annotation via an expected silent segment. **That annotation convention is not an inference rule equating silence with deletion**: a pause, closure, low energy, and CTC blank are not interchangeable error labels.

The actual parser/source [R1–R3] implements:

| Target | Available now | Missing/caveat |
|---|---|---|
| Canonical correctness | Strict and lenient binary labels on canonical tokens | No deletion-versus-substitution distinction in the binary tensor |
| Canonical subtype | `error_type` retains correct/substitution/deletion | Dataset/collate do not supply a dedicated subtype training tensor |
| Actual/realized phone sequence | Chronological valid 39-phone correct productions, substitutions **and additions**, used as CTC targets | Omissions, silence, `err` and nonstandard realizations excluded. This is a normalized partial categorical description, not complete phonetic truth |
| Canonical-token realization | Auxiliary 39-phone targets for correct and identifiable substitutions | Deletions/uncertain realizations masked; no DELETE category |
| Substitution diagnosis | Existing 39-way diagnosis target for eligible substitutions | Correct/deletion/uncertain tokens masked; official two-head trainer freezes this head |
| Insertions | Addition intervals and realized identities enter CTC sequence | No canonical index, binary target or gap-level insertion tensor; actual per-fold addition support not established by the phone census |
| Timing | Annotated boundaries exist in corpus | Not read as raw labels here; not permissible inference inputs; deletion intervals do not prove acoustic realization timing |

**Important strict-label trap:** some substitutions preserve the normalized base phone. A 39-phone realization classifier can predict the same base phone as the reference while the strict binary label is still an error (`textgrid_parser.py:166–190`). Therefore `P(error)=1−P(realized_phone=canonical_phone)` cannot exactly reproduce strict historical semantics. A separate subtype/error channel or explicit same-base deviation treatment is necessary; silently relabeling those tokens as KEEP would change the task.

For `err`, subtype supervision can still say “substitution/error” when the actual phone identity is unavailable. Mask the realization loss; do not invent a 39-phone target. Excluding uncertain phones from a whole-utterance CTC target also means that target is not an exhaustive transcript of everything acoustically produced; any new method must account for that limitation rather than declaring all unmodeled content to be omission or blank.

### 4.2 Available aggregate support is uneven

Independent bounded summation of the archived `phone_stats` (train/validation only) gives:

| Fold | Train correct / substitution / deletion | Train total | Validation correct / substitution / deletion |
|---|---|---:|---|
| 1 | 67,585 / 9,377 / 2,464 | 79,426 | 8,799 / 1,012 / 145 |
| 2 | 67,521 / 9,635 / 2,324 | 79,480 | 8,952 / 822 / 99 |
| 3 | 67,297 / 9,653 / 2,522 | 79,472 | 8,758 / 896 / 227 |
| 4 | 67,133 / 10,045 / 2,321 | 79,499 | 8,854 / 854 / 158 |

Source: [R4], specific JSON keys in the ledger. These are existing audit-artifact counts, not a new raw-data census. Folds rotate speakers and overlapping examples: **do not sum these training rows as independent observations**. Fold 1 has zero training omissions for OW/AW/OY/ZH; other folds also have zero cells. A global subtype head shares data; a full canonical-phone × realization confusion table is much sparser and its pairwise coverage was not established. Counts alone do not prove acoustic separability.

## 5. Mathematical distinctions that determine the design space

Let audio be `x`, canonical phones `c₁…cN`, normalized realized sequence `r`, and frames `h₁…hT` from the frozen acoustic encoder. The formulas below are explanatory candidate formulations, **not implemented repository objectives**.

### A. Supervised subtype and acoustic realization

A compact subtype head can model

`qᵢ = P(zᵢ ∈ {correct, substitution, deletion} | x,c)`

with masked multiclass CE. A second head models `P(vᵢ | x,c)` over identifiable actual phones, masking deleted/uncertain targets. Alternatively a joint realization alphabet adds **DELETE**, but strict same-base errors still need a separate error/subtype channel.

A proposed multi-task objective could be

`L = λctc Lctc(r|x) + λbin LBCE(y|x,c) + λsub LCE(z|x,c) + λreal LCE(v|x,c)`.

The existing official objective is the CTC+binary subset, not this extension. Head count, targets, gradient routing, masking and weighting are distinct interventions. Deletion supervision teaches an error class; it does **not** force attention to relevant neighboring transitions or prevent canonical-phone/position priors from solving the task. If pooling remains diffuse, a new DELETE label can merely name the old shortcut.

### B. Free-phone recognition plus deterministic edit alignment

The acoustic branch models

`Pctc(r|x) = Σ[π:B(π)=r] ∏t pt(πt|x)`.

Decode a free-phone hypothesis, then align it to `c` using DP. The classic edit recurrence compares substitution/match, deletion `(cᵢ,ε)`, and insertion `(ε,rⱼ)`, with polynomial `O(NM)` alignment for a fixed hypothesis of length `M` [P5]. This exposes a realized-sequence diagnostic independently of canonical-query classification and is speech-supported by [P3].

But decoder errors are not learner errors. A deleted phone in canonical-to-hypothesis DP can be an acoustic recognizer miss; a plausible substituted phone can be a recognition error; repeated phones admit ambiguous edit locations. A hard DP path is discrete, not a calibrated omission probability. Changing tie-breaks changes localization even if total edit distance is unchanged. Offline alignment to gold labels for scoring must be kept separate from inference alignment to the reference prompt.

### C. Probabilistic edit/alignment and explicit skips

For fixed `c,r`, replace the minimum-cost edit path by a sum over valid edit paths. A stochastic transducer uses factors for `(a,b)`, `(a,ε)`, `(ε,b)` and a valid normalization/termination scheme [P5, §2]. Forward–backward yields posterior edit occupancy instead of committing to one alignment. A phone string edit model **alone has no acoustic grounding**; acoustics enter only through recognition/posteriors or acoustic-conditioned factors.

One conceptual acoustic combination is

`P(e|x,c) = Σr Pctc(r|x) Pedit(e|c,r)`,

provided the second term is a properly normalized conditional distribution over alignments. Summing only a decoder N-best set is an approximation with missing mass; using arbitrary edit weights without a partition function does not create probabilities.

An acoustic/reference lattice can instead include consuming-frame match/substitution/insertion transitions and a deletion transition that **advances the canonical index without consuming an acoustic realization**. Its normalized path distribution must be defined explicitly. CTC blank consumes a time step and disappears during collapse; it does not advance a canonical omission state. Avoid epsilon cycles, duplicate path counting, repeat-phone mistakes and all-delete/all-insert degeneracy. Acoustic posteriors, edit priors and duration restrictions can trade off non-identifiably: more deletion mass can reflect a stronger prior, not better acoustic evidence.

### D. Segmentation-free acoustic scoring is a compact, separate probabilistic alternative

Cao et al. [P4, §III, Eq.12] compare the canonical CTC sequence with a target-position wildcard graph:

`GOP-SF(i) = LCTC(Lalternative,x) − LCTC(c,x)`.

The alternative can permit substitution only (**S**), substitution or an empty realization (**SD**), or arbitrary replacement phone sequences (**SDI**). It marginalizes alignments rather than trusting a gold interval or one Viterbi path. The primary distinguishes inferred **model activation** alignment from physical phone segmentation (§III-A), which directly matters for omission claims.

This score contrasts correct realization with a family of alternatives; **it is not itself a deletion-only posterior**. SDI's ability to contain insertion paths is not an insertion-event evaluation. The paper's §V-A explicitly says canonical-token evaluation does not fully measure insertion potential; Table I and its discussion do not establish “more permissive is always better.” Its main experiments use CMU Kids/SpeechOcean762 and a different acoustic training setup, not the frozen Langfens checkpoint.

Practical issues: log-domain/scaled arithmetic, repeated labels and graph boundary handling, zero/near-zero activation counts, uncertain targets, other errors in supposedly fixed left/right context, and calibration. The paper's `O(T(N+|V|))` graph calculation is **per target** (§III-E); scoring every target naively multiplies work by `N`. No wall-clock or memory estimate is verified here.

### E. Reference-aware joint learning

CoCA-MDD is direct speech evidence, unlike GECToR: a shared matching map couples frame-to-reference and reference-to-frame attention; training combines realized-phone CTC, phone-prediction CE and binary-state BCE [P2, Eqs.1–10]. Its paper describes pretraining an acoustic model and adding reference/phone/state modules, not simply swapping a classifier in this repository. Its phone/state targets arise from aligned reference and pronounced sequences; its CTC stream still uses sequence alignment for errors. Its fusion Table 1 also illustrates a missed phone that the classifier does not rescue.

Thus an auxiliary realization objective has speech-specific precedent, but the paper is not proof that a compact frozen-encoder adaptation solves omission. Coupling reference into recognition may help alignment **or erase the very acoustic deviation MDD needs to expose**. Chen et al. [P6] independently identify misleading canonical information as a false-acceptance risk and use phonological constraints; they do not establish a universal fix.

Ryu et al. [P8] is another meaning of “joint”: sentence-level assessment plus CTC phone recognition, not reference-conditioned edit tagging. Its §3.1 uses extra realized-phone transcription information in SpeechOcean762; one cannot infer that the local score parser supplies that same training target. Different meanings of “joint” must not be treated as interchangeable evidence.

## 6. Compact edit-aware tagging: what transfers from GECToR, and what does not

GECToR [P1] demonstrates a **text** representation choice: KEEP/DELETE/APPEND/REPLACE transformations with a pretrained text encoder, softmax prediction and possibly repeated passes. It assumes observed source tokens to edit. A learner's actual phones are **not observed** from the prompt; they must be inferred acoustically.

Two speech adaptations are fundamentally different:

1. **Canonical → realized description:** tag expected phones with KEEP/SUB/DELETE, plus gap insertions. DELETE means an expected realization is absent. Input positions exist for omissions, but the realization must be acoustically inferred.
2. **Recognized realized → corrected canonical:** edit a free-phone hypothesis toward the reference. Restoring an omitted target phone may be an INSERT/APPEND, whereas deleting an extra recognized phone addresses epenthesis. This is correction of a recognizer hypothesis, not independent evidence the learner made that error.

Naming both “GECToR-Speech” hides opposite edit directions. For the first, a single canonical token tag cannot describe unrestricted insertion runs; explicit inter-token/boundary slots, multi-label factorization, or a bounded/iterative edit mechanism is needed. Multiple adjacent omissions, repeated phones, and edits at utterance boundaries need an unambiguous definition. The second risks canonicalizing away true errors and feeding its own rewritten hypotheses back into diagnosis.

What transfers plausibly: compact structured labels, parameter sharing, transparent error-type interfaces and bounded edit outputs. What **does not** transfer without a new speech experiment: detection quality, omission localization, acoustic conditioning, GEC throughput, or instructional validity. A parallel subtype+realization head with no text-rewriting loop may test the useful idea more cleanly than replicating GECToR's iterative correction machinery. This is a pilot-design observation, not an architecture selection.

## 7. Comparative assessment without a winner

Complexity labels are engineering judgments from the inspected seams; they are **not measured compute/latency claims**.

| Family | Acoustic grounding mechanism | Supervision fit | Implementation/compatibility burden | Characteristic falsifiable failure |
|---|---|---|---|---|
| Supervised subtype + realization | Predicted pooling/context feeding subtype and actual-phone heads | Subtype derivable; actual phone partially available; DELETE is new target; insertion gaps missing | Low-to-moderate module work but material loader/loss/trainer/contract changes; old diagnosis head is not an omission head | Improves only phone/position prior; acoustically mismatched inputs perform similarly; normalized same-phone errors lost |
| Free-phone CTC + hard DP | Independent audio-to-phone sequence then reference comparison | Existing realized CTC sequence is closest fit, subject to masked unknowns | Low decoder/alignment work relative to new model; can preserve weights for an authorized inference pilot | Recognizer deletions mislabeled as learner omissions; ambiguous repeated-phone location; hard decisions poorly calibrated |
| Soft edit transducer over phone hypotheses | Audio enters through phone posterior/hypotheses; marginalization accounts for edit ambiguity | Gold paired phone strings useful; uncertain strings need masks; support sparse for pair-specific costs | Moderate DP/normalization/calibration work; no baseline tensor changes required for frozen postprocessing | Edit priors dominate; N-best loses relevant sequence; posterior changes under arbitrary costs rather than acoustic evidence |
| Explicit acoustic skip/edit lattice | Competing evidence paths with non-consuming canonical deletion transitions | Can use sequence and/or subtype labels; timing not required at inference | Higher validation burden: custom graph, repeats, normalization, gradients if trained | All-delete/insert paths, epsilon bugs, neighbor borrowing; well-behaved likelihood but wrong errors |
| Reference-aware joint model | Shared acoustic/reference mapping with recognition and classification objectives | CTC and identifiable realization targets available; new subtype/gap contracts optional | Moderate-to-high architecture/checkpoint change; CoCA is not drop-in | Prompt copying erases errors; extra objective improves PER but not omission; classifier and recognizer share errors |
| Segmentation-free CTC score + small calibrator | Relative evidence of canonical versus SD/SDI graph over all alignments | Can reuse existing CTC output; calibration needs authorized train/validation labels | Moderate scorer correctness work; checkpoint weights may remain unchanged | Other reference errors dominate score; wildcard rewards irrelevant alternatives; not deletion-specific |
| Phonological attributes | Acoustic prediction of articulatory-category streams or auxiliary attributes | Mapping from known realized phones supplies weak categorical targets; `err` remains unknown | Moderate new target/CTC-stream work; P7 fine-tunes frontend, unlike current freeze contract | Good attribute averages conceal omission errors; independent streams disagree temporally; labels mistaken for measured articulation |
| Simple acoustic compatibility classifier / score fusion | Small classifier on predicted-segment/CTC evidence, or late fusion with binary score | Existing binary/subtype metadata; no extra foundation training needed conceptually | Low-to-moderate; calibration/fusion still requires separate authorization | Recalibration rather than improved ranking; apparent gain only on reused validation or inherited prompt prior |

A canonical-only constrained aligner belongs in a **control**, not an omission oracle: it requires every expected token to occupy some model path unless explicitly changed. A decoded phone timestamp also localizes a model emission, not necessarily the physical absent phone. At most an omission can be associated with an expected position and an uncertain neighborhood between realized neighbors; claiming an acoustically observed missing-phone interval would be misleading.

## 8. What the archived experiment actually rules out

The archived Fold 1 summary [R5] reports omission AP **0.035020 → 0.032252** for E0→E1 and omission recall at the historical FPR ceiling **0.400000 → 0.365517**. The E1 detailed JSON records mean skip mass on deletions as **0.0**, rounded to six places by `evaluate_e1_mdd.py:431`; this is not proof of an exact mathematical zero in every token. CTC PER improved while omission metrics worsened. These are **archived empirical results**, not fresh confirmation or uncertainty-tested causal effects.

The inspected source shows E1's static learnable skip vector competing with acoustic keys/values and a diagonal positional bias, with no explicit skip-supervision term in the official CTC+binary loss [R2]. This supports a narrow observation: **that tested skip-token formulation did not deliver the registered omission benefit**. It does not refute all deletion-state lattices, supervised subtype objectives, probabilistic scoring, or reference-aware joint methods. Conversely, it does not justify asserting direct skip supervision would work.

Metric semantics must stay unchanged: historical omission AP excludes substitutions from its negatives [R6]. A new omission detector should additionally report deletion-versus-all-other-canonical-tokens as a **new, separately named analysis**, not silently compare it to the old AP. Insertions remain outside the canonical-token denominator. The DIAG-02B artifact explicitly remains `PARTIAL_ARCHIVE_PARITY_FAILED` / `NO_GO_ALIGNMENT_TRAINING_CAUSAL_GATE` [R7]; neither a new paper nor this report repairs that archive parity issue.

## 9. Independent parallel pilot ideas and falsification controls

**These are designs only.** Any model inference, calibration fit, implementation or training needs a separate operator decision. All training remains authorized-remote-GPU-only. The arms below are parallel diagnostic contrasts, not a ranking or a sequence of authorized phases.

### Shared comparison contract

- Keep original split, active TEST sealing, backbone revision/freeze, text/G2P provenance and historical evaluation code/denominators fixed. Do not use gold TextGrid boundaries or actual phones as inference inputs.
- For new trained arms, preregister the permitted objective/head changes, same starting initialization regime, independent optimizer/scheduler/scaler/RNG, successful-update accounting and fixed-budget versus validation-selected endpoints. Do not disguise a resumed baseline as independent original-initialization training; changing the training contract requires approval.
- Include unchanged binary and unchanged free-phone/DP controls. Use the historical binary score for historical omission metrics; label any omission-specific channel separately.
- Report omission and substitution ranking, correct-phone FPR, thresholded recall, realization confusion/coverage, uncertain-target masking, and denominator/abstention coverage. Choose thresholds on permitted calibration/validation only; validation-selected operating points remain diagnostic rather than deployment claims.
- Break down by held-out speaker, phone, reference position and relevant omission contexts. Current validation reuse and small speaker counts limit uncertainty claims. Do not treat repeated folds' training instances as independent samples.
- Require gains beyond text/phone/position-only and **within-speaker mismatched-audio** controls. Counterfactual waveform/feature interventions are optional separately authorized diagnostics, not proofs about natural omissions; heed DIAG-02B validity/parity problems.
- Track prediction coverage and off-prompt/uncertain cases, not just scores on successfully aligned pairs. Independently evaluate feedback accuracy/clarity/appropriateness if pedagogical claims are made.

| Pilot | Isolated question | Essential controls | Evidence that would falsify the intended benefit |
|---|---|---|---|
| S: subtype versus realized-phone supervision | Does explicit deletion subtype add acoustic evidence beyond the binary head, and does identifiable realization training help? | Binary-only; subtype-only; realization-only; combined, with same encoder/pooling and controlled extra parameters; masks for `err`; strict same-base-error audit | Only intercept/threshold shift; no ranking/held-out-speaker benefit; improvement survives removing/replacing audio; realization confusion unchanged |
| F: fixed-checkpoint free-phone + DP | Is independent recognized realization informative for omissions under the current checkpoint? | Greedy unchanged decoder; stable DP tie-break; separately inspect alternate optimal paths; no reference-constrained recognition | Canonical omissions mostly recognizer errors on correct phones; localization unstable across tied paths; no useful complementary cases beyond binary |
| P: probabilistic edit versus hard DP | Does alignment uncertainty help rather than just changing edit priors? | Same phone hypotheses/posteriors; tied versus phone-specific edit factors; train-only priors; path-normalization and truncation-coverage audit | Gains disappear with equally calibrated hard-DP features; posterior dominated by priors; poor repeated-phone/adjacent-error behavior or missing-mass sensitivity |
| G: segmentation-free S/SD/SDI scoring | Does permitting absent realization help beyond substitution-only alternatives using exactly the same acoustic outputs? | S versus SD versus SDI; canonical-score-only; duration/phone prior; compare emission self-alignment separately from physical segmentation | SD does not discriminate natural omissions from correct matched contexts; improvements only synthetic substitutions; permissive wildcard worsens correct-phone FPR/coverage |
| J: compact reference-aware joint supervision | Does adding a realization constraint improve detection instead of inducing prompt copying? | Same heads/parameters with auxiliary loss off; independent audio-only CTC stream; corrupted-reference and mismatched-audio diagnostics; matched update budget | PER improves without omission/ranking benefit; outputs remain canonical under conflicting audio; errors correlated across supposedly independent branches |
| C: small acoustic/fusion alternative | Is expensive architectural coupling necessary for the useful signal? | Same available features; simple calibrator/fusion versus new head; source-isolated calibration; text-only and duration-only baselines | New joint model adds no out-of-speaker ranking/coverage benefit beyond simple fusion; apparent fusion benefit vanishes outside repeatedly reused validation |

Synthetic prompt editing is a control for text/audio incompatibility, **not automatically a natural-omission generator**. Inserting an expected phone into the prompt while leaving audio unchanged resembles one omission-label construction, but can expose lexical/G2P artifacts. Deleting a phone from the prompt instead resembles an insertion relative to that prompt. Acoustic zeroing/splicing or discrete-unit replacement changes the signal in other ways and may introduce artifacts; [P10] supports investigating acoustic versus text perturbations, not declaring them equivalent to true deletions.

Before any pilot, mathematical scorer tests should cover adjacent repeated phones, multiple/edge omissions, multiple insertion runs, empty alternatives, unknown realizations, impossible CTC lengths, padding invariance, and toy-path probability enumeration. These tests are proposed, **not executed** here.

## 10. Checkpoint compatibility and actionable documentary corrections

1. **Preserve baseline endpoints.** `configure_two_head_training()` freezes the entire Wav2Vec2 backbone **and diagnosis head**, training CTC, cross-attention and binary modules (`model.py:434–449`). An existing 39-way diagnosis tensor in a state dict is not evidence of trained diagnosis or a DELETE class.
2. **Separate postprocessing from architecture changes.** Free-phone DP, properly specified probabilistic scoring and score-only fusion can conceptually reuse exact baseline weights. They still change the evaluated system and require separate decoding/calibration provenance. Added subtype/DELETE/gap heads or new reference coupling change parameter shapes and optimizer state. No strict full-state resume into those models should be claimed.
3. **Do not silently transplant 39→40 outputs.** Any allowed partial initialization must list copied/reinitialized tensors and start a new optimizer/scheduler/scaler/RNG history for the approved experiment. Keep original baseline `.pt` metadata and histories untouched. A new run ID is not permission to violate original-pretrained independent-fold requirements.
4. **Expose label provenance in future documentation.** State which targets are raw subtype, normalized actual phone, strict binary, lenient binary, DP-derived edit, or pseudo-label. Keep uncertainty masks separate from correct labels. Specify edit direction and insertion-slot convention before naming a tagger.
5. **Correct the conceptual conflations.** DELETE ≠ CTC blank ≠ silence; auxiliary CE ≠ grounded diagnosis; attention normalization ≠ monotonic/skip alignment; canonical forced alignment ≠ omission localization; a better phone recognizer ≠ a better omission detector; text GEC evidence ≠ speech-MDD validation.
6. **Keep evidence levels visible.** Published speech results support plausibility only. Archived E1 results constrain the tested formulation only. None of the candidate gains, memory use, throughput, training stability or product latency has been established in this run. An unsupported numerical feasibility ranking would manufacture precision.
7. **Use modest feedback language.** “The expected final /T/ may be missing; please repeat” can be evaluated as a tentative output. “You failed to close your tongue at time t” requires evidence not supplied by these labels or models. Attribute predictions are model categories, not direct articulator measurements.

## 11. Limitations and safe restart state

- No raw annotation/sample audit; train/validation counts are aggregate artifacts. Insertion-slot support, confusion-pair coverage, annotation reliability, phone-to-G2P disagreement and acoustic identifiability remain unmeasured.
- No model checkpoint opened or rehashed; archived metric JSON does not independently prove training/update history or reproduce results. The last valid model artifacts remain those documented by the existing campaign; no replacement is created here.
- No candidate code or author implementation audited. Primary formulas are verified in papers; exact implementation details, licenses and resource requirements remain unverified. A source named “RefTextLAS” was not established in this research and is not used to support claims.
- No separately supplied latest omission proposal or additional RQ1 source was available as an authoritative input; the supplied original thesis is the authority actually read. Do not infer that an unlocated proposal has been reviewed.
- Network/documentation tooling had bounded failures: `git submodule status` failed because no `.gitmodules` mapping exists for `repo`; nested Git commands established its HEAD directly. `bs4` was absent, so HTML extraction used the standard library without installing anything. One Crossref query returned non-JSON and was not treated as evidence; primary ISCA/ACL/arXiv sources were then accessed directly. An initially guessed evaluator filename did not exist; `evaluate_e1_mdd.py` was located and read instead.
- Required state-record reads caused the incidental ranking exposure disclosed in §1. This does not justify describing the report as a blinded independent replication.

**Restart:** verify both HEADs, source byte identities, unchanged frozen split, dirty nested checkout and current authorization; read the coordinator's resulting closure checkpoint/index. Retain E1 NO-GO and DIAG-02B partial-parity limitations. Next decision is whether any separately preregistered comparison is worth authorizing, not which architecture this report has chosen. Coordinator owns repository checkpoint/index updates. Stop after this documentary handoff; no implicit pilot, staging, inference or training follows.

## Acceptance report

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Self-contained comparative research report with original-thesis scope, source/artifact evidence table, mathematical distinctions, supervision audit, falsifiable parallel pilots, checkpoint compatibility and explicit limitations. No architecture chosen or experiment claimed."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "git rev-parse HEAD; git branch --show-current; git status --short; git -C repo rev-parse HEAD; git -C repo status --short",
      "result": "passed",
      "summary": "Verified both supplied HEADs and preserved pre-existing nested modifications."
    },
    {
      "command": "git submodule status",
      "result": "failed",
      "summary": "No .gitmodules mapping for repo; direct nested Git inspection used instead."
    },
    {
      "command": "Read-only Python SHA-256 and train/validation phone_stats aggregation",
      "result": "passed",
      "summary": "Split and E0/E1 summary digests match recorded identities; bounded aggregate counts reported without raw data access."
    },
    {
      "command": "requests primary-document retrieval and pdftotext stdin/stdout extraction",
      "result": "passed",
      "summary": "Direct ACL, ISCA and revision-pinned arXiv method/results evidence inspected; no author implementation executed."
    },
    {
      "command": "git diff --cached --quiet; git -C repo diff --cached --quiet",
      "result": "passed",
      "summary": "Both indexes empty at final check."
    },
    {
      "command": "Model/unit tests, inference, calibration, training, remote execution",
      "result": "not-run",
      "summary": "Outside research-only authority; zero successful or skipped optimizer updates."
    }
  ],
  "validationOutput": [
    "Only documentation/source inspection, JSON aggregation, hashes and primary-document processing were performed.",
    "No scientific PASS, model replay, candidate result or deployment-readiness claim."
  ],
  "residualRisks": [
    "Incidental historical-ranking exposure through mandatory checkpoint reads limits blinding, explicitly disclosed.",
    "No raw-label/weight audit, candidate execution, insertion-gap census or author-code/license verification.",
    "Historical archive-parity failure and E1 omission NO-GO remain unresolved.",
    "Proposed acoustic grounding and transfer benefits remain falsifiable hypotheses."
  ],
  "noStagedFiles": true,
  "diffSummary": "No repository edits; only the authorized isolated harness report was written.",
  "reviewFindings": [
    "No implementation blockers adjudicated: research-only report, not a code review or authorization gate.",
    "Scientific correction: strict same-base substitutions cannot be recovered by realized-phone mismatch alone."
  ],
  "manualNotes": "Source inspection and document retrieval are not experiments. The coordinator owns the required process checkpoint/index closure; this report makes no repository changes."
}
```
