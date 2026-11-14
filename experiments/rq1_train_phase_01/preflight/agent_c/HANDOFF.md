# Agent C independent scientific/data audit — TRAIN-PHASE-01

**Role/result:** independent read-only audit of the frozen MB08 evidence and proposed TRAIN-PHASE-01 protocol. No training, tuning, outer-label analysis, protected/TEST_SEALED access, DEV_SEARCH access, CUDA, or repository-outside-lane writes performed. Supervisor remains authoritative for verdict and reports. **Disposition: CONDITIONAL_GO remains consistent with the evidence; do not treat this audit as launch authorization.**

## Evidence examined and status

- Read the acceptance freeze, phase0 receipt, MB08 final report, readiness gate, TRAIN-PHASE-01 specification, machine config, independent input/unknown audits, independent keyed MB03 metrics and receipt, feature extraction receipt/producer, manifest and parser source.
- Agent A and B preflight lane directories were empty at inspection. No preflight runner/policy implementation was available for review. This cannot be represented as a tested new runner or joint policy.
- Existing Agent C MB08 evidence is present and source-hash indexed in the supplied receipt. The standalone new preflight output here is this handoff only; I did not create/overwrite any legacy evidence.

### Audio, intended phones, prompt lineage, unknown handling

**VERIFIED (source lineage to current feature inputs):** TRAIN manifest rows bind `audio_path` (16-kHz WAV), `prompt_text`, `target_phones`, and IDs. The H-002 feature producer reads audio from `audio_path`, canonical intended-phone strings and reference vectors from the cached frame entry; the extraction receipt hashes its script, model, frame-cache index, WAV manifest and feature outputs. Its documented and executable acoustic alignment uses CharSiU frame logits plus `canonical_phones`, and explicitly excludes gold timing, realized-phone labels and error labels. `tools/parse_l2arctic.py` reads transcript strings from PROMPTS, but builds `target_phones` from the TextGrid phones tier, including the canonical side of annotated correct/substitution/deletion labels. Consequently the available implementation does **not** establish a deterministic PROMPTS-text → intended-phone dictionary lineage for all utterances; manifest coexistence of prompt text and phones is not proof that phones came from text. Do not claim text-only input lineage from intent or manifest fields alone.

**Decision on supplied intended phones: SAFE/IN-SCOPE, WITH CONDITIONS.** The frozen spec explicitly authorizes supplied intended phones, and model alignment/inference uses intended—not realized—phones. There is no finding that this operator-provided input itself is leakage. Preserve its provenance distinction from prompt-only inference. Prompt-only/text-derived inference remains **INCONCLUSIVE** pending tracing the cache creator and phone source to fixed dictionary/text rather than TextGrid annotation and verification of all 59,562 mappings. The four unsupported reference phones—`ABA_arctic_a0025::21 ER)`, `BWC_arctic_a0434::28 D_`, `SVBI_arctic_a0006::7 SPN`, `HJK_arctic_a0104::4 Z_`—are confirmed as zero39 vectors. Keep all events and zero vectors with explicit UNKNOWN state; never substitute phone0/AA. These semantics are frozen in the phase config; the new runner still needs tests proving them end to end.

### IDs, fold topology, alignment/training inference, baseline reproduction

**VERIFIED from existing audit artifacts:** TRAIN population 1,800 utterances / 12 speakers / 59,562 targets and 61,362 gaps (560 I); four official speaker folds and their event supports are consistent across spec and independent receipt. The independently rebuilt MB03 adapter reconstructs producer order via manifest traversal within each named fold, then keys rows by utterance/slot (not fold counts alone); receipt records unique IDs and exact N/N+1 topology. Alignment crop extraction uses predicted acoustic alignment conditioned on intended phones, with 17-frame H6 crop and no gold timing at feature generation; gold timing/realized phones serve dataset labels/eligible training or evaluation, not held-out input alignment. The phase spec appropriately requires separate target/gap modules and fit-partition-only normalizers/competitors, speaker exclusion in inner/outer fits, fixed terminal epochs, and persisted IDs/RNG/checkpoints. These are requirements, **not validation of a phase-01 runner**; no runner exists in inspected preflight lane.

**MEASURED/VERIFIED existing baseline:** independent stable-tie recomputation in `experiments/rq1_method_bench_08/agent_c/independent_metrics.json` reproduces keyed MB03 PLAIN AP-S `.4232088151` (support 7,381), AP-O `.2767063392` (1,825), AP-I `.1621248970` (560), Macro3 Error AP `.2873466837`, and P@560 `.2428571`. MB03 producer arrays lacked embedded IDs; reconstruction is from producer source ordering plus authoritative manifest, so this is source/prediction reproduction, **not checkpoint-level replay**. Historical direct H-002 neural checkpoints are also absent; do not call their evidence checkpoint replay. MB07 fixed neural+E4 target AP is `.40274182/.33078349` seed101 and `.40207387/.33399786` seed202, independently recomputed from preserved arrays. Seed replication uses the same TRAIN speakers, not an independent cohort.

Existing evidence also preserves explicit qualifications: historical MB07 `realized_target_FWR` is total warnings/utterance, not false warnings/utterance; the MB07 phone prior's last-42 slice includes three acoustic dimensions and misselects an acoustic category for 14/59,562 events; ordinary meta crossfit does not exclude meta-held speakers from every base fit generating its meta-training features. These do not invalidate threshold-free AP for the unaffected direct/fixed scorer but prohibit treating affected legacy policy/prior results as primary-valid.

## Spec validation / preflight blockers

The immutable design is coherent with MB08: independent H-002 target MLP (2343→256→128→3) plus continuous-frame CTC (768→43, blank42) and canonical-mean E4 with fixed 0.5/0.5 S/O fusion; direct H-002 fallback; independent MB03 exact-gap head (1614→128→64→2); frozen acoustic extractor and no shared trainable trunk. Losses, fit-only scalers/weights, optimizer settings, seed schedules, official speaker folds, and fixed target12/gap5/CTC8 terminal epochs are specified. Joint warning budgets `.2/.5/1/2` with `.8/.2` target/gap allocation, wrong-type S↔O as FP, deterministic event tie order, and inner-OOF-only thresholds are specified. No formal training or architecture ablation is tested; shared-trunk/staged alternatives must remain NOT_TESTED, not implied by modular results. No full-run wall/RSS profile or new-runner restart equivalence is established. The fixed 45-min/seed CPU budget and 2-thread configuration are a plan, not measured full-budget feasibility.

**Blocking before formal training:** (1) implement and freeze the phase runner/config/source hashes; (2) test finite forward/backward, parameter update, RNG/optimizer restart equivalence, CPU/thread constraints, event N/N+1 coverage, duplicates/missing IDs, unknown-zero preservation, and inference label exclusion; (3) create eligible target *and gap* inner OOF streams with fit/exclusion/checkpoint provenance and freeze joint false-warning thresholds before outer application; (4) record a bounded eligible-input timing/maxRSS pilot; (5) keep prompt-only claims gated. Missing complete-system operating metrics remain null/NOT_ESTIMABLE, not zero. No need to demand prompt-only inference when supplied intended phones are explicitly in scope.

## Risk register (audit summary)

| ID | Risk | Status / severity | Required treatment |
|---|---|---|---|
| C1 | Prompt text → supplied intended-phone lineage unproven; manifest phones are derived from TextGrid phone tier | INCONCLUSIVE; claim-scope blocker only | Supplied intended phones explicitly allowed; do not call prompt-only. Trace cache creator/text-dictionary path only if prompt-only claim needed. |
| C2 | Unknown tokens lose categorical reference representation if silently argmaxed | Four known zero39 events; material input-contract risk | Explicit UNKNOWN plus zero vector and full event coverage; test, no AA/phone0 fallback. |
| C3 | New runner absent; no restart/identity/inference-parity validation | NOT_TESTED; launch blocker | Implement bounded runner and tests; freeze source/config before outcomes. |
| C4 | No eligible inner joint target+gap policy artifacts available in A/B preflight lanes | INCONCLUSIVE/NOT_ESTIMABLE; full-system policy blocker | Generate inner OOF both heads, freeze cutoffs before outer application; report missing metrics null. |
| C5 | MB03/legacy direct checkpoint replay not possible from retained artifacts | VERIFIED limitation | Describe as keyed source/prediction reproduction, not checkpoint replay. |
| C6 | Full controlled-run CPU wall/RSS not measured | NOT_TESTED; feasibility blocker | Small eligible CPU timing/RSS pilot before outer outcomes; no extrapolated claim as measurement. |
| C7 | Legacy FWR label, prior-slice bug and meta crossfit limitation | VERIFIED historical qualifications | Exclude from primary policy; preserve historic results and labels; use frozen fixed fusion and corrected future controls. |
| C8 | TRAIN repeated exploration; seed repeats share speakers | VERIFIED design limitation | No external/generalization or safety claim; protected holdout remains prohibited. |

## Recommended supervisor action

Maintain CONDITIONAL_GO only. Have A/B artifacts returned for review when complete; then independently review the frozen runner/policy source and receipts before any formal run. Required checks are bounded implementation and joint-policy integration, not renewed model search. Do not alter any legacy MB08 receipt/report based on this lane.

### Acceptance report
```acceptance-report
{
  "criteriaSatisfied": [
    {"id": "criterion-1", "status": "satisfied", "evidence": "Read-only audit scoped to frozen MB08 specification, existing TRAIN evidence and this lane; no training, protected data, tuning or outside-lane edits."},
    {"id": "criterion-2", "status": "satisfied", "evidence": "This handoff records lineage decision, unknown-phone handling, MB03 independent metric reproduction, policy/runner status, evidence labels and bounded blockers with source artifact paths."}
  ],
  "changedFiles": ["experiments/rq1_train_phase_01/preflight/agent_c/HANDOFF.md"],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {"command": "Read frozen acceptance/specification and existing audit receipts/source files (read-only)", "result": "passed", "summary": "Reviewed authoritative plan, config, MB03 independent audit and H-002 lineage sources; A/B preflight lanes were empty."},
    {"command": "git diff --cached --name-only", "result": "passed", "summary": "No staged files reported."}
  ],
  "validationOutput": ["Existing independent audit receipt reports MB03 AP-I 0.1621248969942254 and P@560 0.24285714285714285; keyed MB03 adapter SHA256 8a5692e43f64b08d60252b9f6a8c63d8be7f8c5e6d6dcfa4d0e26ef7e65bb42b."],
  "residualRisks": ["Prompt-only phone lineage inconclusive", "Phase-01 runner and restart/inference tests absent", "Eligible joint inner target-gap policy not available", "Full-run CPU wall/RSS not measured"],
  "noStagedFiles": true,
  "diffSummary": "Added one lane-local audit handoff; no legacy/source artifacts changed.",
  "reviewFindings": ["no blockers to the audit handoff; formal training remains blocked on the bounded preflight conditions above"],
  "manualNotes": "Supplied intended phones are safe/in-scope under the explicit specification; this does not certify text-only lineage. Supervisor owns the final verdict."
}
```
