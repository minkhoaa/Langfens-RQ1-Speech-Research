# LANGFENS Phase 4 Agent B — Acoustic dependence audit

## Scope and evidence status

This is a source/provenance audit and preregistered analysis plan, not a new model evaluation. No checkpoint was run and no outcome is asserted here. The historical component metrics cited below are diagnostic context, not official canonical scores. The official comparison anchor is EVAL-V3's frozen `NATIVE_DIRECT_HEAD` EXP-24 checkpoint inference receipt; this establishes a reproducible inference artifact, not parity with the historical run or causal attribution.

Work is restricted to this report and `experiments/phase4/acoustic_dependence/`. No SEARCH optimization, training, or GPU inference was performed. Existing GPU records include Phase-3 leases; old telemetry is not a current availability signal. No new GPU lease was acquired. Do not use the unlocked/stale-looking Phase-3 lockfiles as evidence of availability; any future CUDA work must use the approved lease mechanism and check it immediately before execution.

## Lineage and provenance

| Component | Source role described by repository | Inputs / provenance | Current evidence and caveat |
|---|---|---|---|
| EXP-24 `M24-ctc-attention-joint-decoder` | Acoustic/target+gap neural network; monotonic local Gaussian attention; 55 target and 6 gap input dimensions in canonical replay | TRAIN-derived feature normalizer, frozen checkpoint, adapter, train features; `DEV_CAL` and `DEV_SEARCH` inputs and outputs in checkpoint replay | Receipt says `CHECKPOINT_INFERENCE_REPLAY`, new frozen inference, not historical metric parity; CUDA GTX 1650; no labels/realized phones as model inputs. This is the strongest provenance receipt inspected. |
| EXP-34 `M34-phonotactic-ngram-prober` | Phonotactic/lexical-prior-conditioned diagnostic model | Feature archives and CMUdict; run spec claims target dim 63, gap dim 11 | Historical `RUN_SPEC.yaml` records local reproduced metrics, but a matched-prior ablation with verified source-feature ancestry is not established here. Dictionary coverage/OOV and stress/variant policy can confound acoustic interpretation. |
| EXP-28 `M28-prosodic-energy-model` | Waveform prosodic-energy features; target/gap scoring | Feature archives plus 16 kHz derived waveforms; run spec claims target dim 60, gap dim 9 | Historical run spec records local reproduced metrics. Need verify exact extraction alignment and source hashes before treating it as a current reproducible result. |

EXP-24 replay receipt hashes: receipt `10a5990584d74d29a88eaf6a87c075aa011d8bc5f51d798f5df6e6b184d485cb` (published in `TMCI_REPRODUCTION.md`); checkpoint `78e906d0f5369c23196dab3b8a7816b1f061c380dbfb7ddadd31149e88321b59`; replay `DEV_SEARCH.npz` `bf157001d92e225c24c9da3e329dc70648d8bf6b944961333444ec946299a53a`. Receipt declares adapter `1bb4b9c1c8a84985acc2e9a8ad9668c603fa505f9f1b954c5f5e2ae74c125041`, normalizer `92ef38023de16dfb29c2057f265ce2a16fe5ae8efccfa5e8dcb51fc4fb656df1`, and source `35d5a2ac0b54a57ef1ace5f11558105556c6f8a9f6b3d4b19a3359a7073e3a48`. Manifest hashes are recorded in `TMCI_REPRODUCTION.md`. Hashes bind files, not data collection validity, semantic correctness, or independence.

Historical component APs in TMCI component-attribution tables (EXP-24 Macro-3 0.276819; EXP-34 0.263731; EXP-28 0.263494) are DEV_SEARCH values from a development-exposed set and must not be represented as independent final estimates. The official evaluator/canonical release, not historical prose or ad-hoc rescoring, governs any future score report. No canonical score for any proposed intervention below exists yet.

## Question and inference boundary

Test whether EXP-24's ranking depends materially on audio-derived evidence versus reference/lexical structure, whether prosodic energy contributes under a controlled intervention, and where CTC's topology omits or confounds events. A good model score or alignment against reference phones is not causal evidence that acoustic information drove the decision. Gold boundaries/realized-phone labels may only be used as **diagnostics** after frozen inference; they must never determine deployed boundaries, features, or thresholds. Clearly label any gold-aligned analysis as oracle/gold diagnostic, separate from inference-valid results.

Reference-only and mismatched-reference perturbations are falsification/sensitivity controls, not valid task outcomes. Do not reuse their predicted labels, thresholds, or scores as task predictions. Keep original reference and audio paired for all principal estimates.

## Required experiment matrix (before any inference)

1. **Reference-only baseline:** no waveform-derived features; reference sequence, phone identity/context, and permitted metadata only. Establish lexical/reference prior capacity. This is explicitly not an acoustic model.
2. **Full audio baseline:** frozen checkpoint and exact official evaluator inputs. Preserve normalizer, output semantics, event IDs, split, checkpoint, and evaluator version.
3. **Lexical prior neutralization:** hold audio/checkpoint fixed; neutralize EXP-34-derived transition/cluster prior channels using a prespecified neutral value/mask, without replacing phones, changing event set, or refitting on SEARCH. Add prior-only branch as complementary control. Predeclare whether this is feature ablation or intervention and record any out-of-distribution consequence.
4. **Prosody intervention:** hold all else fixed; mask/neutralize only EXP-28 energy/prosody channels, including waveform-derived energy at target and gap sites. Compare with full audio and prosody-only control. Do not alter waveform timing or reference labels. Verify extraction using source audio hashes and deterministic feature hashes.
5. **Acoustic versus lexical:** paired within-utterance/speaker comparisons for full, reference-only, acoustic-only (audio-derived channels; lexical channels removed), and lexical-only conditions. Define exact feature groups from actual source implementation before evaluation; run-time label names are not proof of modality.
6. **Mismatched/perturbed sensitivity:** substitute a different utterance's reference with controlled phone-count/context matching, or permute permitted lexical features within speaker. Report only score/ranking change and failure/sensitivity diagnostics; never compute or publish these as valid S/O/I performance. Keep labels attached to original true event identity only for paired diagnostic analysis and avoid claiming perturbed outcomes are valid predictions.
7. **CTC omissions/topology:** document blank/repeat-collapse and forced-alignment transition graph; enumerate which intended phones/gaps have no represented path or arbitrary intended-phone deletion edge. Analyze deletion evidence and aligner failures separately. Never equate CTC blank probability with omission or insertion ground truth.

No claim of causal dependence is warranted unless a controlled, paired ablation/intervention changes the frozen inference output as predicted while preserving event population and evaluator protocol. If interventions induce unsupported feature distributions, report them as sensitivity only.

## Population, gaps, and diagnostics

For every utterance with N intended phones preserve N target slots and **exactly N+1 insertion-gap slots**, indexed 0..N (leading, inter-phone, trailing). Require event-key equality, unique IDs, and count assertions before metric computation; missing leading/trailing gaps invalidate the run. Do not derive gap population from detected events or labels. Report target S/O/C and gap I as separate populations using the canonical evaluator's definitions.

Primary estimates must be inference-valid and based on official evaluation inputs. For every variant, include target/gap counts and hash-verified score files, then official evaluator outputs. Stratify by speaker, L1, and phone/phonological family descriptively; paired uncertainty resamples speakers (only four DEV_SEARCH speakers means uncertainty is intrinsically coarse). Gold alignment, realized phones, and hand-labeled acoustic diagnosis may explain errors only in a separately marked diagnostic table, never alter predictions. Avoid token-level significance as a substitute for speaker-level inference.

## Reporting requirements and stop conditions

- Archive exact source/code/config/checkpoint/normalizer/feature/waveform/manifest hashes, evaluator version, output hashes, device receipt/lease, and command line for every completed run. Recompute hashes; never copy a hash from narrative alone.
- Report official canonical AP/PR outputs and protocol-compliant calibration/threshold results separately from diagnostics. Do not optimize thresholds or feature choices on DEV_SEARCH. If choosing a feature mask or threshold, select on DEV_CAL and freeze before any held-out evaluation.
- Do not call historical values canonical without confirming identical evaluator, lineage, and source outputs. The inspected EXP-24 receipt specifically disclaims historical metric parity; EXP-28/34 run specs are not sufficient by themselves for equivalent official receipts.
- Stop if event keys/counts differ, gold-derived fields enter inference, source audio mapping is ambiguous, lexical perturbations leak across splits, or no authorized GPU lease is available for required CUDA inference.

## Unresolved / limitations

This audit did not execute the matrix. Exact feature extraction and model input ancestry for EXP-28/34, associated hashes/receipts, matched component output files, official evaluator reruns, and gold-diagnostic provenance remain to be verified. Source historical tables establish hypotheses, not intervention effects. Current GPU availability was not established: inspected lease ledger is historical; no live GPU/lease probe was made. There are no new scores or tests from this report.
