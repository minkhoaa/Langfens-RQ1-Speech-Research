# LANGFENS RQ1-METHOD-BENCH-01 — Method taxonomy

**Prepared:** 2026-11-08  
**Scope:** Methods that could provide phone-level substitution (S), omission (O), or exact N+1-gap insertion (I) evidence from learner audio and the intended phone sequence. This is a scoping review and experiment-selection document, not a claim of comparative LANGFENS performance.

## Review question and screening rule

Which distinct computational mechanisms from pronunciation assessment, speech recognition, anomaly detection, speech representation learning, and structured prediction can contribute valid event-level S/O/I scores under the LANGFENS inference contract? Included work proposes or evaluates a mechanism plausibly transferable to phone-level pronunciation analysis. Adjacent clinical, child-speech, or language-specific work is retained only when it contributes a transferable mechanism; its published scores are never pooled with LANGFENS scores.

The working database contains 22 unique works or source records, including foundational and recent work (2018–2026). Records distinguish original papers, preprints, proceedings records, and secondary challenge reports. A source record is not automatically a fully read paper: unchecked full texts, code, checkpoints, and licenses remain marked as such in the database and comprehensive report. De-duplication is by DOI/arXiv ID/title; the Phonological-Level Wav2Vec2 preprint and its journal/proceedings versions are treated as one method family, not independent evidence.

## Families and their transfer logic

| Family | Representative mechanisms | New evidence signal | Natural event support | Main caveat |
|---|---|---|---|---|
| Canonical-phone confidence and alignment | GOP/posterior margins; forced alignment; phone-level binary error classifiers | Acoustic likelihood of intended phone versus alternatives within a localized segment | S; O/I generally weak or unavailable without a sequence/gap model | Alignment errors can masquerade as pronunciation errors; canonical-phone priors may dominate |
| Alignment-free sequence recognition and edit diagnosis | CTC, hybrid CTC-attention, RNN-T, constrained edit-distance phone decoding | A decoded realized phone sequence and sequence posterior | S/O from edit alignment; I from extra realized phones, but exact gap is a mapping problem | Recognition errors and edit ties; hypothesis-to-reference mapping must preserve acoustic timing and topology |
| Similarity-aware phone recognition | Soft phone targets weighted by phonetic/articulatory similarity; verbatim recognition | Realized-phone distribution plus graded confusability | Potentially all S/O/I after edit mapping | Synthetic accent data and sequence WPER/AER do not validate event AP; gap mapping remains unproven |
| Phonological/articulatory attributes | Manner/place/voicing/phonological-feature heads; attribute mismatch | Feature-level acoustic deviations rather than categorical phone identity | S naturally; O/I need occupancy/duration or alignment features | Feature inventory may be language-specific; attributes can improve diagnosis without improving detection |
| Normal-pronunciation anomaly detection | Per-phone one-class SVM; correct-only prototypes; contrastive positive/pseudo-negative bags | Distance or energy from a learned normal manifold | S/O localized to target windows; I needs explicit gap representation | Speaker/syllable identity can dominate distance; correct-only data do not guarantee learner-error ranking |
| Discrete SSL units and surprisal | Quantized speech tokens, native token language model, text-to-unit DTW | Acoustic token rarity and mismatch against text-conditioned expected units | Local S/O candidates; I may be represented as unmatched token runs at a gap | Published utterance score/PCC is not event-level AP; quantizer and unit-to-phone mapping can blur boundaries |
| Confusion graphs and constrained alternatives | L1-specific directed phone graphs; restricted GOP alternatives | Prior over plausible substitution paths conditioned on target/L1 | S; possible O/I only with sequence/gap extension | Graph priors can increase apparent sensitivity without supporting acoustic evidence; leakage-safe construction required |
| Cross-modal attention and audio-language models | Whisper/audio-text cross-attention; joint speech-text encoders; audio-text prompting | Text-conditioned acoustic localization or generated realized-phone sequence | Phone S/O after alignment; I only with exact-gap extraction/adaptation | Large-model compute; attention is not necessarily explanation; word/utterance outputs are not event scores |
| Perceptual and population-norm models | Listener/annotator aggregation; native norm versus target-population norm | Normative acceptability or perceptual consensus under uncertainty | Could reframe S; does not intrinsically identify O/I | Requires suitable listener labels and norm definition; differs from acoustic truth labels |

## Cross-family comparison against LANGFENS weaknesses

| Observed problem to investigate | Mechanisms with a plausible path | Falsifiable evidence needed |
|---|---|---|
| Weak omission discrimination | Realized-phone sequence/edit diagnosis; alignment-free CTC; duration/occupancy heads; discrete-unit DTW | O AP rises on matched targets; deleted phones are supported by acoustic sequence evidence and not only duration priors |
| Rare insertion positives | Explicit N+1 gap classifier; unmatched recognized-phone spans; token-run surprisal at reference gaps | Exact-gap I AP/recall improves; leading/internal/trailing gap coverage is complete and does not use gold timing at inference |
| Inaccurate or restrictive alignment | Alignment-free CTC/RNN-T; hybrid CTC-attention; cross-attention localization; uncertainty-aware alignment | Matched gains persist when alignment confidence is stratified; event localization is audited against gold only after scoring |
| Inadequate acoustic representation | SSL continuous features; discrete SSL units; phonological attributes; pitch/prosody additions | Gain survives matched model/data controls and speaker-held-out analysis; class-specific acoustic rationale is visible |
| Correct-versus-substituted confusion | Soft phone similarity; phonological attributes; calibrated alternatives; posterior uncertainty | S AP improves without merely re-ranking phone confusability or accepting common substitutions |
| Dependence on intended-phone priors | Verbatim recognition, audio-only anomaly models, token surprisal | Compare audio-only and target-conditioned variants; measure score changes on matched audio with alternate references |
| Limited speaker generalization | Correct-only prototypes, multi-speaker normalization, contrastive invariance, robust SSL embeddings | Held-out-speaker performance and per-speaker uncertainty improve, not just pooled AP |
| Weak discrimination-to-F1 conversion | DEV_CAL-only calibration, class-wise thresholding, prevalence-aware decision policy | Fixed policy chosen on DEV_CAL converts ranking to F1 on SEARCH without using SEARCH labels; AP and F1 are reported separately |

These are mechanisms to test, not established root causes. The project’s historical error analyses and current EVAL-V3 controls are context; they do not prove that any one mechanism is the limiting factor.

## Compatibility rule

Every adapted method must receive genuine audio and the supplied intended phone sequence. It may use TRAIN labels for learned model fitting and DEV_CAL for documented calibration/selection; DEV_SEARCH is descriptive only. S and O scores are attached to canonical target events. I scores are attached to every exact gap in the N+1 topology, including both utterance edges. A model whose native output lacks one or more event types is a specialist and is evaluated only on the supported event population. No missing class score is imputed.

## Selected diversity set

The initial benchmark set is: (1) realized-phone recognition with similarity-aware targets, (2) normal-only/contrastive anomaly scoring, (3) discrete-unit surprisal with text-guided alignment, (4) constrained confusion-graph GOP, and (5) alignment-free CTC sequence scoring. This spans sequence recognition, normality modeling, self-supervised discrete representation, phonological priors, and alignment-free scoring. The first four have distinct hypotheses; CTC is a high-value control family and may be a frozen-logit CPU screen if training/reproduction is infeasible. Existing SylPh-inspired prototype output is retained as a negative/limited feasibility result, not promoted as faithful reproduction. OpenPA and joint speech-text encoders remain gated until model, resource, and exact-event feasibility are verified.

## Primary source links

See `METHOD_BENCH_01_PAPER_DATABASE.csv` for a de-duplicated source inventory, status, source URL and fit notes. High-information primary sources include the CTC phone-level assessment framework ([Cao et al., Interspeech 2024](https://www.isca-archive.org/interspeech_2024/cao24b_interspeech.pdf)), phonological-level MDD ([Shahin & Ahmed, Interspeech 2024](https://www.isca-archive.org/interspeech_2024/shahin24_interspeech.pdf)), similarity-aware phone recognition ([Zhou et al., Interspeech 2025](https://doi.org/10.21437/Interspeech.2025-2417)), constrained phonological GOP ([Parikh et al., Interspeech 2025](https://doi.org/10.21437/Interspeech.2025-829)), SylPh ([Jin et al., Interspeech 2025](https://doi.org/10.21437/Interspeech.2025-1285)), discrete-unit surprisal ([Sara & Chowdhury, arXiv:2606.19910](https://arxiv.org/abs/2606.19910)), and alignment-free OpenPA ([Chao et al., arXiv:2610.13826](https://arxiv.org/abs/2610.13826)).
