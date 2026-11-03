# Agent F — Cross-session correction ledger for RQ-RESEARCH-01

## Disposition and scope

**Research-only closure review; not a new phase, architecture selection, or execution authorization.** Preserve both historical report sets. The original synthesis substantially overstated evidence, and 10/13 original central weighted totals are incorrect. The newer synthesis corrects the principal scientific claims and all its 13 central totals are arithmetically correct, but raw revalidation reports still contain 11 arithmetic errors, source conflation, and precision/calibration inconsistencies. Neither report set establishes a Langfens candidate improvement.

The missing **new independent E report** is real and verified from saved runtime records. It is **not** the historical Lane E: that historical report exists and its distinct completed session is verified. Six historical sessions do not establish six independent scientific validations: historical B read D, and historical E read C and D before writing. New A/B/C/D/F have distinct Luna sessions, but independence means separate analyses—not independent data, replications, or unanimous scientific attestation.

Only this isolated harness-bound report was written. No repository, index, checkpoint, source, configuration, dataset, or historical report was edited. No model loading, inference, training, optimizer/backward, GPU/remote execution, deployment, raw-data traversal, or TEST audio/labels occurred. The coordinator owns the process checkpoint.

### Citation conventions

- **O/** = `manifests/RQ-RESEARCH-01/` (original historical campaign).
- **N/** = `manifests/RQ-RESEARCH-01/revalidation-2026-11-03/`.
- **T** = `/home/khoa/Projects/KLTN/DC KLTN0127 23520772&23520757_TranThiHongYen (1).md`. Body is physical line 14; bibliography is line 15. Cite its actual sections, not invented line ranges.
- **H** = `/home/khoa/.pi/agent/sessions/--home-khoa-Projects-KLTN-Dataset--/`.
- **R-old** = `H/2026-11-03T03-59-04-886Z_01a19f8f-82b6-7221-bcbf-1fa2d66fc44e/`.
- **R-new** = `H/2026-11-03T04-33-56-358Z_01a19faf-6c86-7221-bcbf-1fa59bf65e84/`.
- Session locators below are actual JSONL physical lines, not reconstructed conversation numbering.

Evidence classes: **FACT** = inspected documentary/source/archived fact; **PRIMARY** = finding in inspected primary literature, not locally reproduced; **ESTIMATE** = explicit unmeasured cost or judgment; **HYPOTHESIS** = proposed transfer/causal claim; **UNSUPPORTED** = claim exceeds accessible evidence. Confidence concerns the correction, not candidate efficacy.

## 1. Actual inputs, reads, and verification

Read `AGENTS.md`, the global `docs/process-checkpoints/INDEX.md` in two ranges, original and revalidation process checkpoints, both integrated reports, all N lane reports/addendum/supplement and source/thesis matrices. Inspected targeted substantive passages, comparison/pilot/epistemic sections, and claim-search results across **all six original lane reports**; this is a targeted cross-session audit, not a claim to have exhaustively re-reviewed every original sentence. Directly inspected T's scope, RQ1/RQ2, methodology, feedback, limitations, and bibliography.

Underlying evidence additionally read:

- `manifests/RQ1-DIAG-02B/SCIENTIFIC_REPORT.md`, §§1–8 and evidence inventory;
- `manifests/RQ1-DIAG-01/evaluation/OMISSION_AND_FALSE_NEGATIVE_REPORT.json`, aggregate/operating-point passages;
- `manifests/RQ1-TRN-03A/checkpoints/19-gate-c-scientific-closure.md`, §§1–6;
- `manifests/RQ-MODEL-01/evaluation/E1_FOLD1_BENCHMARK_SUMMARY.json`;
- archived E1 training log at `/tmp/e1_train_output/langfens-rq-model-01-e1-fold1-train.log:1826–1827,2404–2405` and its `checkpoints/fold1/e1_fold1_alignment_aware/training_endpoint_report.json`;
- `src/langfens_prep/alignment.py:1–110`, `dataset.py:97–158`, `model.py:434–451`;
- frozen train-coverage aggregate search results, **not raw annotations**;
- both saved workflow status receipts and retained child sessions (structured tool/provenance extraction), plus historical E metadata;
- primary-paper text passages from Tu v1, GOPT v1, Peng, and Cao segmentation-free GOP v3, plus relevant cached author-code comments. These were existing local paper artifacts/caches; **no publisher or GitHub retrieval was newly performed**.

Recomputed original and newer central rubric rows and raw newer lane totals; required 13 actual central rows rather than accepting a zero-row parser result. Rechecked all ten N fixed-document checksums. Recomputed frozen split, summary, thesis and S1 PDF hashes. Read-only inventory of filenames under `docs/` and `manifests/` did not identify a separate latest omission proposal or separate RQ1 source. This bounded search is not proof such a source does not exist elsewhere.

### Evidence and source identity table

| Source actually checked | Identity / exact locator | Admitted evidence and limitation |
|---|---|---|
| Original thesis | T §§2.2(b),2.3,3,4(c),(f),(g),5; SHA-256 `98fa922584b750510d4b0c183c59a84455da9e64ab231b59bdb58862c56133d3` | Highest scope authority. GECToR belongs to Grammar §4(f); no universal <300 ms SLA or speech tagging superiority. |
| Frozen split metadata | `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`; SHA `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b` | Digest verification only; no audio/labels read. |
| E0/E1 aggregate archive | `manifests/RQ-MODEL-01/evaluation/E1_FOLD1_BENCHMARK_SUMMARY.json:1–62`; SHA `006611ccf0796938d4114bf62a1d17979c0af9ece362415dc1600670e6b086a3` | 300 validation utterances, 9,956 canonical tokens, 145 omissions; archived metrics and checkpoint identities, not a rerun. |
| E1 training metadata | E1 log lines above; endpoint JSON fields `successful_updates`, `fixed_budget_sha256` | Trained model: 3,600 successful updates, zero AMP skips in epoch-6 log; evaluated digest `3b686730cc0a1101a1bbc96b59dc7bd38a9fc3cc7da7effe26b2d9498f079d24`. No weight-byte rehash or remote durability verification. |
| DIAG-02B | `SCIENTIFIC_REPORT.md` §§1–3,7–8 | Causal conclusions withdrawn; 9/80 parity failures; 66 matched pairs/G=8; archive association, not deployment readiness. |
| Tu retrieval primary | `references/papers/tu2025_retrieval_mdd.pdf`; cached identical extraction `/tmp/tu2025.txt`; arXiv **2511.20107v1**, Table 1 and §3.4; https://arxiv.org/abs/2511.20107v1 | F1 69.60%, PER 104.08%, FRR 4.43%, FAR 32.44%; insertions explain metric divergence. PDF SHA `2eb40576a3c2f4bf59f1471c8bb3c240711d0f13d9af31c19e0454532fac65d1`. Published DOI `10.1109/ICASSP55912.2026.11462649` is recorded by N matrix, not publisher-verified in this audit. No implementation license independently verified. |
| GOPT primary | `references/papers/gong2022_gopt.pdf`; identical extraction `/tmp/gopt.txt`; arXiv **2205.03432v1**, Table 1 and §4.1; https://arxiv.org/abs/2205.03432v1 | Completeness PCC .155/.011 and authors' imbalance hypothesis; not a deletion-specific causal experiment. PDF SHA `44c5f27ea0977b40396e6fc538a1636074b1484c6b574b968997789b0ceb851c`. |
| Peng primary identity | `references/papers/peng2021_wav2vec2_mdd.pdf`; identical extraction `/tmp/peng2021.txt` header | **A study on fine-tuning wav2vec2.0 Model for the task of Mispronunciation Detection and Diagnosis**, Peng et al., Interspeech 2021. PDF SHA `8c56fc608b97146b401aa86ddd5390b405e9809ed31ca3f4ac73d267f4aacce9`. Distinct from Xu's paper in N/S2. No new omission-only result verified here. |
| Cao segmentation-free primary | `/tmp/langfens_sfgop.pdf`, extraction `/tmp/langfens_sfgop.txt:23,446–487,522–527,562–572`; https://arxiv.org/abs/2507.16838v3 | Header explicitly **v3, 5 Feb 2026**. §§III-C–E, Table I: optional target graph, normalized occupancy, mixed-error AUC .914 SD/.891 S/.859 SDI. PDF SHA `6f4c5396dd20218c5a94bc8fa317ba56c1b31bfaa9bbde71670504f6acbdc58d`. Not proof of journal/preprint byte identity or Langfens omission benefit. |
| Author scoring-code snapshot | `/tmp/langfens-sfgop-taslpro26_gop_sf_sd_norm.py:153–177,193,240–255`; N/S1 records https://github.com/frank613/CTC-based-GOP at `ffc4823330efe2c836691931a456ea9e3360871e` | Cached source comments explicitly restrict next-phone alternatives for disambiguation. Commit provenance/license status are documented by N coordinator, **not independently refetched** here; code not executed. No verified code-reuse license in this audit. |

`pdftotext <local-paper> -` exactly matched the existing Tu/GOPT/Peng text caches byte-for-byte. A PDF font-weight warning occurred during documentation extraction; no model/library execution was involved. Paper/extraction identity does not validate a paper's scientific claims.

## 2. Independent-report provenance: what receipts actually establish

The authoritative full receipts inspected are:

- `/tmp/pi-subagents-uid-1000/async-subagent-runs/06c9aabf-1eb7-490d-91aa-6b18f579aad3/status.json`;
- `/tmp/pi-subagents-uid-1000/async-subagent-runs/986566c1-3f78-45f4-8063-55d0545809d8/status.json`.

The saved N `research-revalidation-workflow.json` under `H/subagent-artifacts/outputs/986566c1-3f78-45f4-8063-55d0545809d8/` is a **truncated text receipt despite its .json suffix**, not a complete machine-parseable receipt. Use the full status plus retained sessions, not that short display alone.

| Lane | Historical run / actual model | New run / actual model | Provenance finding |
|---|---|---|---|
| A | `cf468ac9-3140-41f3-a2d8-82a79b7ce0be`, Gemini 3.8 Flash | `2e5af637-cece-4c93-869e-a34878f036ea`, Luna | Both distinct completed sessions. New session final text at lines 52 and 62 contains full report then correction; addendum is not another independent lane. |
| B | `d1673445-f010-4641-a7ae-7e84d6c515e1`, Gemini | `d227a505-7a92-4fbc-b6a9-afac37fbc275`, Luna | Historical B reads historical D at its `R-old/<run>/run-0/session.jsonl:123–124`. Not blind independent assessment. |
| C | `10ff2e6e-ff35-4a6d-85d8-46e3ad52fdf8`, Gemini | `a57536e3-dfbd-4e16-afa9-3103c11d113a`, Luna | New C reads historical C at session line 51; this is disclosed historical context, not independent replication. |
| D | `a472e314-4afd-4e31-ae29-f85b7426aa4b`, Gemini | `e7e17b60-161d-47dd-9f52-b1f0a654547c`, Luna | Historical D has no primary PDF/network retrieval calls in retained tool stream. New D contains initial and revised finals (lines 21,59); recovery `1a4b7453-0f00-48e9-96f7-70d7a3869c99` reuses its session, not a sixth/new independent analysis. |
| E | `6ad4ba4b-4696-420e-b5ce-29185be6a762`, Gemini | `8c706527-13e8-482b-af2f-75f1692104a2`, Luna | **Historical E completed and exists.** Reads C and D at original session lines 166–169. **New E failed**: its retained session has only two supervisor calls, lines 7,12; no source reads or final report. Recovery `99e7276e-b08a-4e30-8800-923bc29ad404` is not a completed replacement. |
| F | `bd7817cb-c697-4543-8777-8cff71222649`, Gemini | `0523a322-ffb6-4ecc-a044-11577c2f5d04`, Luna | Distinct completed sessions; new F reads historical F at session line 55. Neither old nor new report is another experiment. |

Actual model identifiers in session model-change records: historical `antigravity/gemini-3.8-flash`; newer `openai-codex/gpt-6-luna`. New launch/fleet receipts designate fresh contexts; distinct sessions and source calls are corroborated. This does not establish statistical independence, blind review, or absence of shared prompted assumptions.

Historical E metadata at `H/subagent-artifacts/6ad4ba4b-4696-420e-b5ce-29185be6a762_researcher_meta.json` explicitly uses **attested** acceptance, empty runtime checks/verify runs. Its child report says computational feasibility was verified, but recorded validation is only document writing/hash. **Runtime completion/attestation is not scientific verification.** Historical D's missing primary calls similarly preclude accepting all its literature as directly inspected.

The original seven reports really total **288,062 bytes**, as O checkpoint says; all stated component byte sizes match current documents. E SHA `439bcf7221159893fee0a321ddfcdc8b72a1733da887d05ae7ae3de3c0eac117` matches its runtime metadata. These validate document existence/identity—not the phrases “peer-reviewed analysis,” “all outputs verified,” or collective scientific certification.

## 3. Correction ledger

Each row identifies a historical/raw claim, the authority that controls the correction, and what remains unresolved. **No historical source should be overwritten to enact these corrections.**

| ID / class | Claim and exact claim source | Authoritative evidence and correction | Confidence / unresolved question |
|---|---|---|---|
| F01 — UNSUPPORTED → HYPOTHESIS | GECToR-Speech is #1, 93.20, “exact thesis solution,” “Why It Wins”: O `INTEGRATED_DECISION_REPORT.md` §§3–5,7.1; O E §§2.1,3.3,10. | T §4(f), ref [10] is **grammatical** error correction (https://aclanthology.org/2020.bea-1.16/), not acoustic MDD. A categorical OMIT target is a plausible new speech objective, not a validated adaptation or guaranteed winner. N integrated §§1,3 correctly demotes it to reserve (73). | **High.** Which speech-specific primary result validates the proposed exact architecture? None established here. |
| F02 — UNSUPPORTED → HYPOTHESIS | DELETE supervision “forces” acoustic absence learning; “root cause proven”: O integrated §§1.1,5.1,7.1; O E §§3.3,10; O C §4. | DIAG-02B §§7–8 explicitly withdraws causal proof. Direct subtype gradients are real mathematical properties, but models can learn canonical phone/position/L1 priors while ignoring audio. New supervision does not ensure alignment or acoustic grounding. | **High.** Requires separately authorized matched controls/audio-dependence validation, not another report. |
| F03 — HYPOTHESIS / inconsistent gates | O integrated §6.1 D-1 requires blank mean ≥1.5× correct; original D §8 D-1 instead ≥2×; “zero compute” and AP .12/.15 treated as expected signals. | DIAG-02B §3 reports .8244/.7322 = **1.125922×**. With that control mean, 1.5× needs 1.0983 and 2× needs 1.4644, impossible for a posterior mean ≤1. These are incompatible with the cited control distribution, not scientific laws for a future different span extractor. | **High.** What predicted-span population and justified effect target would replace these arbitrary gates? No execution authorized. |
| F04 — UNSUPPORTED | Canonical Viterbi gives “exact” acoustic boundaries, all omissions shrink to one 20-ms frame with blank .95–.99: O integrated §§1.2,3,5.1; O D §2.1(3),§6.3,§8; O F §§4.3,11. | `alignment.py:16–70` merely pools provided intervals and guarantees ≥1 frame; `:73–110` is gold-canonical/G2P DP. A fixed-target CTC path must contain every required label. A constrained alignment is a model-conditional assignment, not physical absence localization or guaranteed duration. Cao v3 §II-B/C discusses segmentation ambiguity/CTC activation mismatch. | **High.** A real omission hypothesis needs alternatives/skip paths; their calibrated relation to gold omissions remains unmeasured. |
| F05 — FACT versus HYPOTHESIS | TextGrid short durations and matching transition spans are portrayed as acoustic absence indicators: O A §4.1; O F §4.3; O D §8. | DIAG-02B §7: 500/629 endpoint matches are annotation facts, not physical omitted-phone segments. <80 ms counts depend on float convention (545 raw vs 542 rounded). Duration shortcuts may reproduce annotation policy, not detect non-realization. | **High.** Can a predicted, no-gold-time feature distinguish short correct phones and omitted targets? Unknown. |
| F06 — PRIMARY overextended | Retrieval F1 69.60/PER104.08 scientifically invalidates retrieval; no query implies universal omission impossibility: O F:22,§4.1,§11; O integrated §§1.2,7.1. | Tu v1 Table 1/§3.4 explicitly describes insertion-sensitive PER versus canonical-phone MDD metrics. PER=(S+D+I)/N can exceed 100%. Missing a phone-specific frame does not prevent a **sequence-level** missing-target hypothesis. No omission-only AP/recall is shown. Reject universal impossibility/fraud language; retain concern about metric coverage and lack of directly comparable evidence. | **High.** Does a train-only retrieval+sequence comparator add useful omission evidence under the frozen protocol? Unmeasured. |
| F07 — scope UNSUPPORTED | Whisper/AED/RNN-T are automatic thesis violations; unverified latency, OOM and divergence are categorical exclusions: O integrated §§1.2,3,7.1; O B §§2.3,8,12; O E §§6–7. | T §4(c) explicitly proposes Whisper/equivalent for conversational STT and separately assesses phonemes. T §3 excludes foundation-language-model training from scratch and comprehensive free-conversation diagnosis—not every narrow pretrained acoustic recognizer. Full replacement can be deferred on necessity/evidence/engineering grounds, not blanket family prohibition. | **High** on scope; **low** on unmeasured cost. What exact initialized bounded model is being compared? |
| F08 — FACT | “Inference-only” means E1 untrained: ambiguity addressed by N `A_PROVENANCE_CORRECTION.md`; N integrated §1. | E1 log:1826–1827 has 3,600 updates, zero skips; endpoint digest matches summary. Only the later E0/E1 benchmark had zero updates. E1's negative result is not an untrained null. Original O reports generally already discuss E1 training; do not misattribute this confusion to every original lane. | **High** for archive metadata; remote weight-byte identity/durability not reverified. |
| F09 — provenance mismatch | Best training AP .3156 vs later replay .309293 suggests unexplained drift: N A correction. | E1 endpoint `best_val_ap` .3156 refers to different selected weights (`best_sha256` starts `312081...`). Epoch-6 fixed-budget history is **.309018** at the evaluated `3b6867...` endpoint. Compare .309018 with .309293 first; PER .233377 vs .232365 also differs. N evidence-matrix correction already notes this. | **High.** Exact evaluator/backend/target differences remain unresolved; don't declare parity or causal degradation from mixed endpoints. |
| F10 — precision overstatement | Skip mass is “exactly zero” / complete activation collapse: O integrated and E; N B/D/F prose. | Summary JSON reports 0.0 and correct .000031 at serialized precision. N integrated §1 appropriately says rounded .000000, no demonstrated separation. Reported zero is not proof every underlying token/head mass is mathematical zero. | **High.** Would full-precision token scores alter the descriptive claim? Not inspected; no re-evaluation. |
| F11 — FACT / denominator error | Omission FNR **59.90% at 10% FPR**: O integrated §1.1; O C §1.1; O E §9. | DIAG-01 aggregate JSON:183–208 has pooled **deletion FNR .8251** (519/629), recall .1749 at .10 FPR. Step19 §4's **59.90%** is **Fold-1 all-error binary FNR**, not pooled omission FNR. Preserve population and error-type denominators. | **High.** Historical summaries require append-only correction; do not silently change evaluator semantics. |
| F12 — FACT / denominator error | Omission AP .030506 is compared with random prevalence .0159: O integrated §1.1; O A/C §1. | DIAG-02B §2: omission-vs-correct AP uses **629/(35,363+629)=1.7476%**, not 629/39,576=1.5893%. The latter is all-token omission prevalence. This distinction applies to AP baselines/enrichment. | **High.** Use each score's actual eligibility set. |
| F13 — HYPOTHESIS mislabeled fact | Small acoustic gap +.001677 “proves” acoustically ungrounded omissions, “statistically indistinguishable from zero”: O integrated §1.1; O C §1. | DIAG-02B §2 explicitly declines a new uncertainty claim from this small census point estimate. Its matched binary AUC .482 is descriptive on a selected sample; neither establishes information absence or a universal architectural impossibility. | **High.** No justified hypothesis test of the stated causal mechanism established. |
| F14 — PRIMARY versus causal speculation | GOPT completeness PCC .155/.011 demonstrates forced-alignment structural deletion failure: O A §2.2; O integrated §§1.2,2/P06. | GOPT v1 Table 1/§4.1 genuinely reports those poor completeness correlations; authors hypothesize highly imbalanced completeness scores (`/tmp/gopt.txt:555–560`). No deletion-specific causal study isolates alignment. Forced alignment need not use gold timestamps at inference; O C literature table's contrary implication is wrong. | **High.** Which controlled deletion analysis proves the alternative mechanism? Not supplied. |
| F15 — unsupported source/metric transfer | Phonological feature scores imply validated articulatory diagnosis and omission rescue: O integrated §1.2; O F §§4.2,11. | N/S10–S11 explicitly separate related 2024 and 2023/2025 works, feature DER/accuracy and aggregate MDD. These are not deletion AP, calibrated articulatory error confidence, or independent replications. Mapping a target to a feature description is not evidence the learner produced the asserted trait. | **High** on metric/claim separation; **medium** on literature detail (exact papers not independently re-reviewed here). Which validated per-error feedback result supports deployment? |
| F16 — FACT / statistical error | Focal pilot's `p=.056` and mechanistic gradient-starvation causal explanation: O C §3.1. | Step19 §§2,7 distinguishes **bootstrap positive frequency .056** from paired permutation **p=.479**; explicitly replaces speculative gradient/overfit mechanisms. Generic focal/weighted BCE screening failure is not proof all omission-specific objectives fail. Weighted arm is **3,598**, not a 3,600-update fixed-budget endpoint. | **High.** No new mechanism experiment or cross-fold probability can be inferred. |
| F17 — HYPOTHESIS versus guarantee | Fusion with hard DP probability .90 instantly eliminates omission FNR while retaining substitution AP: O integrated §5.1 A.2; O A §9.2. | Greedy decoder edits are hypotheses, not truth; .90 is an arbitrary uncalibrated value. `max` fusion can promote false positives, change AP on correct/substitution sets, and double-count correlated CTC/binary evidence. N integrated §§3–5 properly makes complementary value conditional. | **High.** Held-out benefit and calibration absent. |
| F18 — candidate identity inconsistency | Integrated original rejection of **A.3/F.3 = external Kaldi/MFA**: O integrated §§3–4. | O A §9.3 actually defines **A.3 = CTC trellis with explicit deletion arc**; O F §9.3 defines **F3 = constrained Viterbi caliper probe**. These are not identical to an external MFA family. Scientific scores/rejections cannot be assigned to different methods under reused IDs. | **High.** Freeze candidate IDs and definitions before any later comparison. |
| F19 — design inconsistency | Original E2-A expected AP ≥.12/recall≥.60 but its metric gates are AP≥.10/recall≥.55/binary≥.32; integrated E-1 uses .15/.55/.317: O E §§8.1; O integrated §6.2; O checkpoint §4. | These are differing **proposed targets**, not observed results or a single consistent pre-registered acceptance contract. Likewise D-1 .15 versus integrated .12 and 2× versus1.5×. N integrated §5 correctly refuses silent adoption. | **High.** Which protocol was prospectively frozen, when, and by whom? No independent preregistration receipt established here. |
| F20 — ESTIMATE / unit error | Hardware “profiling”: E table gives seconds/step and hours/3,600; guarantees convergence/CPU latency: O E §§6.1–6.2; propagated by O integrated §3. | Arithmetic only: .041×3600=147.6 seconds (2.46 minutes), .044×3600=158.4 seconds (2.64 minutes), .285×3600=1026 seconds (17.1 minutes), .340×3600=1224 seconds (20.4 minutes)—**not corresponding hours**. This is a ~60× unit inconsistency, not corrected measured runtimes. No profiling receipt validates exact 3.6 GB/14.2ms/42.5ms or convergence. | **High.** Must measure approved actual workload/hardware; do not use arithmetic outputs as portable estimates. |
| F21 — ESTIMATE / contradictory timing | Original A table says <30 min/3,600; integrated says ~2.5h and D CPU ~3min; E says 2.6h: O A §3; O integrated §§4,6; O E §6. | No common measured benchmark supports these cost comparisons. E1 archived job reaches endpoint around 629 seconds in its own environment; this is historical runtime evidence, not new architecture timing or a budget promise. “Zero training” is not “zero compute,” and logistic regression is fitting. Current rules prohibit executing it locally under a training guise. | **High.** Distinguish preprocessing, encoder/scorer, optimizer, I/O and job overhead; remote authorization needed for learning. |
| F22 — source-version FACT | N A says 2026 segmentation-free source inaccessible; N integrated now ranks it first. | Not a contradiction after **additional coordinator access**: N checkpoint final disposition and source matrix record arXiv v3 and code snapshot. Direct cached v3 passages inspected here confirm optional phone/empty SD and Table I mixed-error AUC. Do not retroactively call this independent lane-A verification or deletion-only evidence. | **High.** Exact journal parity/code license/compatible phone checkpoint unresolved. |
| F23 — source identity FACT | Original broad “Peng wav2vec2” citation versus N/S2 Xu “Explore wav2vec 2.0…”; new A's foundational “Wang 2007 GOP” DOI. | Peng and Xu are **distinct primary papers**; new Xu evidence requiring alignment cannot silently replace original Peng recognition evidence. T ref[3] is Witt & Young **2000**, DOI `10.1016/S0167-6393(99)00044-8`. N matrix rightly excludes questionable uninspected Wang 2007 attribution. | **High.** Maintain paper-specific title/author/version/table locators; foundational Witt full text not retrieved in this audit. |
| F24 — source conflation | N revised D primary item 2 names Raffel 2017 but describes MoChA/chunks and speech translation. | N matrix S17 corrects distinct sources: Raffel et al. **Online and Linear-Time Attention…**, ICML2017, https://arxiv.org/abs/1704.00784; Chiu & Raffel **Monotonic Chunkwise Attention**, ICLR2018, https://arxiv.org/abs/1712.05382. Neither alone supplies an omitted-reference state or measured Langfens latency. | **High** on identity; exact original implementation/license not reverified. |
| F25 — implementation FACT | Existing third diagnosis head supports substitution feedback; “same two heads” implies unchanged contract: O A §3; N C opening/§5. | `model.py:434–449` freezes diagnosis in two-head training. `dataset.py:105–118,137–153` omits deleted phones from realized CTC and masks realized/diagnosis labels. Untrained architecture output is not validated diagnosis. Adding skip loss changes two-loss objective even if output head count stays two. | **High.** Separate approved objective/config/schema needed; no edits authorized. |
| F26 — label semantics FACT | O E:193–196 calls vocabulary exactly41 but excludes current target from 39 replacements; DELETE terminology borrowed as pedagogical action. | With 39 phones and target excluded, available replacement identities per slot are **38**, plus KEEP/OMIT=40 valid actions; a global41-logit head can exist but requires masking/semantics specified. Canonical→realized OMIT corresponds pedagogically to **restoring/inserting** expected sound, not deleting it. | **High.** Define fixed global vocabulary, per-slot masking, distortion/invalid labels and action direction before implementation. |
| F27 — provenance overstatement | O checkpoint §1: six independent exhaustive audits, peer-reviewed analysis; integrated §8 collective attestation. | Saved status confirms six completed sessions, not six independent validations; B/E cross-reads and D's lack of primary retrieval are direct counterevidence to a blind/exhaustive reading claim. Runtime metadata says attested, with no verify runs. N five plus coordinator supplement is correctly labeled partial. | **High.** Scientific primary coverage must be audited claim-by-claim, not inferred from report count or bytes. |
| F28 — uncertainty / missing input | Formal review complete despite missing latest omission proposal/separate RQ1 source. | N checkpoint and every relevant new lane disclose missing input; bounded filename inventory here did not locate it. T is located and authoritative; old reports are not substitutes for an unidentified newer proposal. | **High** on current limitation, not global absence. Supply exact path/version or explicitly identify intended source before claiming full-input closure. |
| F29 — historical denominator ambiguity | O A §1.1 and epistemic table use 53.57% with different Vietnamese/omission denominators; integrated uses “errors among Vietnamese.” | These are different quantities: share of deletions attributable to a population versus deletion fraction within that population's errors. No Vietnamese validation speaker is included in cited eight-speaker validation analyses. Training census demographics cannot establish Vietnamese validation effectiveness. | **High** on denominator distinction; **unresolved** exact original 53.57 denominator, not independently reconstructed here. Do not reuse without underlying aggregate row/formula. |
| F30 — no observed candidate result | “GO/PASS”, “exact”, “guarantees” or invented phase RQ-MODEL-02: O B §12; O integrated §§3–7. | T describes investigation and does not presume superiority; AGENTS requires authorization. Scope fit is not performance, deployment readiness, training authorization, or causal scientific PASS. Current assignment is closure of RQ-RESEARCH-01 only. | **High.** Candidate experiments remain not run and not authorized. |

## 4. Ranking arithmetic and consistency audit

### Original integrated rubric: ten incorrect totals

Weights are .25/.25/.20/.15/.15. Recalculation uses exactly the components printed at O `INTEGRATED_DECISION_REPORT.md:97–109`; no rescoring judgments were substituted.

| Original row | Printed | Recomputed | Printed minus recomputed |
|---|---:|---:|---:|
| E.1 GECToR-Speech | 93.20 | **93.30** | −.10 |
| D.1 canonical Viterbi | 91.55 | **91.85** | −.30 |
| A.2 dual path | 90.40 | **90.80** | −.40 |
| D.2 supervised alignment | 87.20 | **87.45** | −.25 |
| C.1 three-class | 86.90 | **87.30** | −.40 |
| F.2 attributes | 83.60 | **83.40** | +.20 |
| A.1 greedy CTC | 83.70 | **84.30** | −.60 |
| F.1 retrieval | 48.75 | 48.75 | 0 |
| D.3 MoChA | 59.75 | 59.75 | 0 |
| E.2 AED | 57.00 | **56.50** | +.50 |
| E.3 RNN-T | 51.75 | **51.00** | +.75 |
| A.3/F.3 relabeled external alignment | 53.75 | 53.75 | 0 |
| B.1 Whisper | 47.25 | **46.25** | +1.00 |

Even on printed totals, A.1 **83.70 > F.2 83.60**, contrary to #7 versus #6. Recomputed A.1 84.30 likewise exceeds F.2 83.40. Arithmetic repair alone leaves the leading E/D/A ordering unchanged; it **does not** rehabilitate unsupported evidence scores (e.g. academic92 for an unvalidated speech adaptation).

### New central ranking: all 13 rows correct

N integrated `:68–80` explicitly uses component scale1–5 and formula `5E+5M+4L+3C+3F`. Recomputed totals, in table order: **84,82,78,74,73,68,66,65,62,60,60,58,50**. This confirms the checkpoint's **central-table** arithmetic claim. It does not verify empirical ranking or robust superiority. ±1 criterion contributes3–5 points; leading differences are small relative to subjective judgment uncertainty, as N itself explains.

Original and new scores must not be compared as measured changes or treated as the same frozen candidate set: definitions, evidence and numerical scale changed. New segmentation-free scorer is not original D-1 canonical-duration pooling. New compact operation classification overlaps three-class/existence directions; it is not an independently validated “family winner.”

### Raw newer lanes: 11 remaining errors out of 16 printed weighted rows

| Report/line | Candidate | Printed /5 | Correct /5 |
|---|---|---:|---:|
| N B:57 | current CTC analysis | 3.90 | **3.95** |
| N B:60 | alternate SSL recognizer | 2.95 | **2.90** |
| N C:80 | contrastive | 2.45 | **2.40** |
| N C:81 | synthetic deletion | 2.25 | **2.20** |
| N D:48 | edit lattice | 3.55 | **3.50** |
| N D:50 | E1 skip graft | 3.10 | **3.05** |
| N D:51 | fixed-target CTC | 3.40 | **3.35** |
| N F:60 | train-only retrieval | 3.25 | **3.35** |
| N F:61 | attributes | 3.35 | **3.40** |
| N F:62 | reference audio/TTS | 2.80 | **2.75** |
| N F:63 | sequence hybrid | 4.00 | **3.95** |

Other B/C/D rows match. A reports components without a printed weighted total, so is excluded from this 16-row denominator. D's heading “corrected arithmetic” and statement that prior figures were wrong do **not** make its revised numbers correct. The N central table does not inherit these numbers, appropriately; preserve raw lanes with this qualification rather than rewriting them.

## 5. What is established, and what is not

**Established documentary/archived facts:** thesis scope; two distinct historical/new session groups; historical six completed versus new five completed/one failed; original document byte total; E1 trained provenance matched by recorded checkpoint digest; E1 aggregate validation degradation in omission/binary AP despite improved PER; DIAG-02B's partial parity and withdrawn causal claims; arithmetic errors and corrected values above.

**Primary literature findings:** Tu's retrieval metrics and insertion limitation; GOPT's low completeness correlations and authors' imbalance interpretation; Cao v3's optional-target graph formulation and mixed-real-error AUC. These are published/preprint results under their own protocols, not independent Langfens results. The code comment restricts some alternatives; code availability alone is not correctness, licensing, or reproduction.

**Estimates/judgments:** rubric scores, compatibility/engineering rankings, presumed small extra head cost, all unmeasured candidate times/memory and deployment latency. Original numerical timing claims are not admitted as measured evidence.

**Hypotheses/speculation:** explicit OMIT/skip loss improves natural omission detection; optional graphs transfer to current phone emissions; attention routing is the dominant causal bottleneck; fusion adds complementary signal; attributes improve actual feedback quality. New synthesis appropriately conditions these, but does not resolve them.

Deletion is **not** CTC blank, silence, or an empty forced-alignment window. Canonical alignment is **not** omission localization. DELETE supervision is **not** acoustic grounding. GECToR is **not** validated speech MDD. Gold TextGrid timestamps remain unavailable as inference inputs. These distinctions hold regardless of historical rank or model family.

## 6. Actionable append-only corrections and safe restart

1. Coordinator should append/link this correction ledger to closure documentation; **retain all historical sources unchanged**. Explicitly supersede original winner/causal/readiness assertions, while preserving original metrics and evaluator identities.
2. Keep original E separate from missing new E. Say “six historical completed sessions; five fresh-context revalidation analyses plus non-independent coordinator E supplement.” Do not imply six experiments, independent replications, or unanimous verified findings.
3. Record original central arithmetic repair, raw-new arithmetic errors, candidate-ID drift and differing proposed pilot gates. Do not relabel subjective rankings as statistical or preregistered results.
4. Cite E1 training log/endpoint separately from inference-only benchmark; use matched endpoint comparisons and rounded-zero language. Preserve pooled omission versus Fold-1 full-binary denominators.
5. Stop at research. Do not run D-1/P1/P2/P3, logistic fitting, model inference, training, staging or deployment. Missing latest proposal/separate source means full-input review remains **limited**, not silently complete.
6. Before any later authorized continuation: verify outer HEAD `b25a7595360294798d8d06b37592dd65dd4c9c33` on `research-phase`, nested HEAD `baa4989243d5ff46b8b3bf5a3c280cdf2857382b`, frozen split/summary/report hashes and dirty-source boundaries; obtain exact missing source/version and explicit operator decision. This role neither repairs the E runtime nor launches replacement children.

At initial inspection outer status already had modified index, dirty nested `repo`, and untracked N reports. Nested status contained14 modified audit scripts and2 deleted reference PDFs; no repair or read of deleted files was attempted. Later outer status additionally showed `manifests/RQ-RESEARCH-01/closure-2026-11-03/` from concurrent coordinator work; it was not this agent's output and was not audited. No staged paths were returned by `git diff --cached --name-only`; tracked `git diff --check` returned clean. Do not confuse concurrent documentation changes with a clean overall working tree.

### Residual risks / explicit limitations

- Separate latest omission proposal and separate RQ1 source remain unidentified; stop with that limitation rather than inventing their content.
- Literature inspection here was targeted primary-passage verification and source/provenance comparison, not a new exhaustive systematic review. Some N literature/implementation/license claims remain coordinator/lane-attributed, not independently fetched or fully revalidated.
- Retained session receipts verify provenance and actual tool calls, not scientific correctness or all historical policy compliance. Partial reads/shared sources do not establish blind independence.
- No checkpoint weights, model outputs, remote stores, code correctness, inference cost, statistical power, feedback quality or deployment readiness were independently tested here. Fold-1 has two validation speakers; DIAG-02B's eight-speaker archive still has parity limitations.
- Byte/hash checks attest artifacts, not scientific PASS. No candidate experiment was performed.

```acceptance-report
{
  "criteriaSatisfied": [
    {"id":"criterion-1","status":"satisfied","evidence":"Delivered a self-contained cross-session correction ledger with exact report/source/session locators, arithmetic recomputation, provenance distinctions, uncertainty, actionable corrections and residual risks."}
  ],
  "changedFiles": ["/home/khoa/.pi/agent/sessions/--home-khoa-Projects-KLTN-Dataset--/subagent-artifacts/outputs/723092bc-f51f-4a2a-8cd9-bf63b7c93d0b/reports/F_CROSS_SESSION_CORRECTION_LEDGER.md"],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {"command":"Read-only git HEAD/status checks, git diff --cached --name-only, git diff --check","result":"passed","summary":"Expected HEADs verified; no staged paths; tracked diff check clean; pre-existing/concurrent dirty documentation and nested source preserved."},
    {"command":"Python documentation-only rubric recomputation","result":"passed","summary":"13 original and13 new central rows actually checked; found10 old central errors,11 raw-new lane errors; new central13/13 correct."},
    {"command":"sha256sum -c manifests/RQ-RESEARCH-01/revalidation-2026-11-03/SHA256SUMS (from its directory)","result":"passed","summary":"All10 fixed documents matched; this is document integrity only."},
    {"command":"Read-only retained workflow/session parsing and document/PDF hashes; pdftotext comparison","result":"passed","summary":"Verified six historical completed sessions versus five new/one failed; established historical cross-reads; Tu/GOPT/Peng cached text matched local primary PDFs."},
    {"command":"Unit/regression/model tests or candidate experiments","result":"not-run","summary":"Outside this research-only role; no model loading/inference/training/TEST access."}
  ],
  "validationOutput": ["Original report sizes sum288062 bytes.","Original central weighted totals:10/13 incorrect; new central:13/13 correct; raw B/C/D/F:11/16 incorrect.","New E retained session contains only two supervisor calls and no source reads/final research report."],
  "residualRisks": ["Missing latest omission proposal/separate RQ1 source.","No new experiments, weight-byte verification, remote durability audit or deployment validation.","Primary inspection targeted; unrefetched code/license/version claims explicitly attributed; historical/new reports are not independent replications."],
  "noStagedFiles": true,
  "diffSummary": "Only isolated harness report written; no repository/history modifications.",
  "reviewFindings": ["blocker: original winner/causal/readiness and scope assertions overstate evidence.","blocker: full-input/six-new-independent-report closure remains unsupported.","documentation: append-only corrections required for arithmetic, denominator and source-identity inconsistencies."],
  "manualNotes": "Coordinator owns checkpoint/index updates. Current safe state is STOP after research; preserve historical E and distinguish it from missing revalidation E. No replacement child or experiment launched."
}
```
