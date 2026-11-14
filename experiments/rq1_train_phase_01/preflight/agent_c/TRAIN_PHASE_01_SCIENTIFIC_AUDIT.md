# TRAIN-PHASE-01 scientific/data audit (Agent C; candidate, supervisor authoritative)

**Audit scope:** read-only review of the frozen MB08 protocol, TRAIN lineage, A/B preflight and completed bounded B continuation; bounded independent TRAIN-only hashing, event/source checks, threshold recomputation and frozen-policy score application. No formal phase01 run, protected/TEST_SEALED/DEV_SEARCH access, CUDA, or changes outside Agent C. Evidence labels are MEASURED / VERIFIED / INCONCLUSIVE / NOT_TESTED; all absent/invalid metrics stay null/NOT_ESTIMABLE.

## Bottom line and correction to prior handoffs

The two prior Agent C handoffs made **factually incorrect source-absence claims**: MB03's exact-gap PLAIN producer and eligible seed101 target inner scores are preserved and source-linked. Correct evidence is in MB03 `agent_c/train_speaker_adversary.py` (functions `get_arrays`, `Detector`, `fit_predict`, `main`), MB07 `agent_a/nested_direct_seed101_inner_oof.npz`, `ctc_nested_e4_seed101_inner.npz`, its fit receipt and `evaluate_nested_e4_fusion.py`, and the independent target verification under the supervisor lane. These old handoffs remain unchanged as audit history; this report supersedes those specific assertions. This is **source/score reproduction**, not checkpoint replay for historical target heads.

B subsequently executed the explicitly bounded seed101 gap work using the MB03 PLAIN producer semantics: 12 six-speaker inner fits, then 4 nine-speaker outer refits only after policy cutoffs froze. It saved keyed inner/outer streams, 16 fit checkpoints, fit/held IDs, fit normalizers, hashes and receipts. Agent C independently re-derived all 96 target/gap cutoff objects for the 48 fold×method×budget combinations, and independently applied those frozen thresholds to the TRAIN outer streams. This is a **MEASURED bounded mixed assembly** (reused MB07 target predictions + new MB03-style gap heads), not formal controlled TRAIN-PHASE-01 training or an independent cohort.

**Current gate evidence:** the B policy portion is materially complete as a bounded policy study. A's current gap-input helper passes an independent exact N+1 boundary check, but A's current synthetic pytest suite still has two failing assertions/fixtures and its runner hash no longer matches its original source-freeze receipt. End-to-end audio/feature/CTC/E4 integration, full restart exactness and eligible TRAIN smoke remain unvalidated. Do not treat the entire phase01 runner or full 45-minute seed budget as passed.

## Data lineage, intended phones and unknowns

### TRAIN WAV / manifest / feature chain — VERIFIED, bounded full-TRAIN integrity check

`independent_train_audit.py` verified all 1,800 manifest rows are TRAIN, all 12 official speakers are present, every manifest `audio_path` resolves within `data/derived/l2arctic/wavs_16k`, and every WAV decodes as nonempty 16-kHz audio. SHA-256, bytes and audio metadata for every TRAIN WAV are preserved in `TRAIN_WAV_SHA256.csv` (CSV SHA-256 `4a70ad56cd6fc78452192f4b923d0c613d24c94d4f5da3baee6685fb8db1aef6`). The authoritative TRAIN manifest SHA is `1f4da8518d9b184fb737b93c7887ecd7c3e73d6e83260e54651074831f61bcff`.

The H-002 extraction receipt binds the TRAIN WAV manifest, H6 feature array, label/reference arrays and event metadata. Its producer reads audio from each TRAIN manifest `audio_path`, applies the frozen Wav2Vec2 feature extractor and extracts H6 acoustic state. It conditions local monotonic phone alignment on cached CharSiU frame logits and `canonical_phones`; the receipt declares no gold timing, realized phones or labels in feature generation. The cached `(utterance_id,slot_idx)` order exactly matches all 59,562 manifest target rows and `(N,39)` reference vectors.

### Text versus supplied intended phones — supplied phones SAFE/IN-SCOPE; text-only INCONCLUSIVE

The TRAIN manifest co-locates `prompt_text` and `target_phones`, but that does not establish text-to-phone derivation. `tools/parse_l2arctic.py` reads transcript text from PROMPTS and parses target canonical phones from the TextGrid phone tier (canonical side of annotated correct/substitution/deletion entries). The H-002 extraction path consumes cached `canonical_phones`; no complete raw prompt-text-to-dictionary-to-phone lineage was found. Thus **no text-only claim** is supported.

The frozen protocol explicitly authorizes supplied intended phones, so the intended-phone-conditioned input is **SAFE/IN-SCOPE** and need not be prohibited. It is not a leakage finding. Keep raw audio + supplied intended phones as the declared input contract; distinguish it from prompt-only inference. The primary target 39D reference for each of the four unsupported spellings remains an explicit all-zero row and its event slot is retained:

| Event | Raw supplied/canonical intended token | Reference vector |
|---|---|---|
| `ABA_arctic_a0025::21` | `ER)` | 39D all-zero, VERIFIED |
| `BWC_arctic_a0434::28` | `D_` | 39D all-zero, VERIFIED |
| `SVBI_arctic_a0006::7` | `SPN` | 39D all-zero, VERIFIED |
| `HJK_arctic_a0104::4` | `Z_` | 39D all-zero, VERIFIED |

The older CTC/E4 inventory has **explicit, distinct** normalization: `ER)`→`ER`, `D_`→`D`, `Z_`→`Z`; `SPN` is an actual 42-phone inventory class. That mapping supplies E4's intended sequence but does not mutate the primary target reference vector. E4 class lookup fails closed for an unsupported unnormalized phone rather than silently assigning AA/phone0. H-002 acoustic forced alignment separately maps OOV phone spellings to `[SIL]` through `vocab.get(p,sil)`; this is an explicit alignment-only rule and does not alter the reference vector or event identity. Preserve/test these separate contracts; do not describe this as silent AA fallback.

## Identity, split exclusions, source and metric audit

TRAIN has 1,800 utterances, 12 speakers, 59,562 targets (C 50,356/S 7,381/O 1,825), 61,362 exact gaps (I 560), and official folds F0 ABA/ASI/BWC; F1 EBVS/HJK/HKK; F2 HQTV/LXC/NJS; F3 PNV/SKA/SVBI. All manifest target/gap event IDs and N/N+1 cardinalities were revalidated. All WAV/key/cache checks and hashes are in `independent_train_audit.json`.

### MB03 PLAIN baseline — independently MEASURED from archived keyed predictions

Recomputed stable-tie AP from `experiments/rq1_method_bench_08/agent_c/mb03_plain_keyed.npz`:

| Metric | Independent result | Support |
|---|---:|---:|
| AP-S | .4232088151 | 7,381 |
| AP-O | .2767063392 | 1,825 |
| AP-I exact gap | .1621248970 | 560 |
| Macro3 Error AP | .2873466837 | — |
| P@560 insertion | .2428571 | 560 selected |

IDs were reconstructed from MB03's documented manifest traversal within named folds and are now explicitly keyed. Producer arrays historically lacked IDs; do not call this checkpoint replay. Historical decisions were not frozen, so MB03 F1/FWR are not inferred from these AP numbers.

### Eligible archived target inner OOF — VERIFIED, score/source reconstruction

Agent C independently joined seed101 direct-neural inner OOF indices/rows to the complete CTC/E4 nested payload. For each of four outer folds, the 9-speaker outer training pool is completely covered by three local inner groups (0/1/2); each 6-speaker base fit excludes its three inner-held speakers and the three outer-held speakers. The 12 CTC checkpoints' SHA hashes, fit speakers, held speakers and fit-local E4 competitors match their receipts. Each held inner utterance has N S scores and N O scores plus N+1 gap diagnostics; target E4 simplex is recomputed as softmax([0,SLLR,OLLR]). Across four outer pools there are 178,686 verified target-event score joins, with no duplicate/missing event in any pool. Supervisor's `inner_target_verification.json` independently records `VERIFIED_ELIGIBLE_KEYED_INNER_TARGETS`, same fold exclusions and source hashes. Historical direct-target weights/checkpoints are not saved; provenance is archived score/source reconstruction.

### MB03 PLAIN producer correction and B bounded gap fits — VERIFIED / MEASURED

The complete producer `experiments/rq1_method_bench_03/agent_c/train_speaker_adversary.py` defines `get_arrays()` with H6 target and exact N+1 adjacent edge context; `Detector` with task and dormant speaker branch; `fit_predict()` PLAIN path; fit-only scaler, class counts/weights, AdamW, minibatch order and 5-epoch loop. Calling PLAIN uses task CE only; the speaker branch is instantiated but its GRL/speaker loss is inactive. B's `run_gap_refits.py` follows those PLAIN semantics on the specified new inner/outer fit partitions and persists its normalizer/checkpoint/fit/held IDs. Refs used in gap context are the supplied intended 39D vectors; edge acoustic/phone contexts are zero as specified. This corrects B's initial claim that the producer recipe was absent. Historical checkpoint replay is not asserted.

B fit schedule: inner seed `101+10*outer+local_inner`, 12 fits; outer seed `201+outer`, 4 fits. All are CPU, two torch threads, within the frozen 20-minute B cap: inner12 wall `21.7413s`; outer4 wall `11.8279s`. Exact per-fit seconds, class counts, fit/held speakers, checkpoint and normalizer hashes are in `agent_b/results/{inner_receipt,outer_receipt}.json`; keyed data in `inner_predictions.npz` and `outer_predictions.npz`. There are 184,086 inner rows across the four outer pools (61,362 unique gap IDs repeated once in each three-fold outer-pool set) and 61,362 unique outer gap predictions. The boundary composition and independent manifest labels match.

## Policy audit and independent outer results (MEASURED, bounded, not formal phase01)

B froze sources, fit IDs, inner gap scores and eligible target score hashes before cutoff selection, then froze policies before outer refits/application. Agent C independently rederived 96 cutoffs: 4 outer folds × 3 target methods (direct, E4-only, fixed50 neural+E4) × 4 budgets × 2 heads (target/gap), equal to 48 policy pairs. Exact threshold, tie cutoff, prefix count, cumulative FP, floor budget and `fit_ids` all match B's `thresholds_frozen.json`. Target false is `predicted S/O type != gold S/O`, so wrong S↔O types are FP. Gap false is predicted insertion on a gold-negative gap. The deterministic tie key is descending first eight SHA-256 bytes of `LANGFENS-MB07-warning-tie-v1:` + event ID. `independent_policy_audit.json` is the full independently recalculated receipt; `independent_joint_metrics.csv` is the pooled summary. The raw B inner/outer prediction NPZs and all checkpoints are preserved.

Independent application of those frozen cutoffs to the four untouched-by-cutoff outer TRAIN folds gives the following pooled results. AP is threshold-free; operating metrics apply the pre-frozen per-fold thresholds. No threshold/method/budget was selected from outer outcomes.

| Target scorer | B | AP-S | AP-O | AP-I new gap | Macro3 Error AP | Macro3 Error F1 | Joint FWR | Total warnings | False warnings | Wrong-type S/O FP events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct | .2 | .384080 | .265655 | .174642 | .274792 | .110711 | .143333 | 624 | 258 | 9 |
| Direct | .5 | .384080 | .265655 | .174642 | .274792 | .169229 | .378333 | 1,469 | 681 | 36 |
| Direct | 1 | .384080 | .265655 | .174642 | .274792 | .222239 | .780000 | 2,779 | 1,404 | 83 |
| Direct | 2 | .384080 | .265655 | .174642 | .274792 | .283400 | 1.743889 | 5,590 | 3,139 | 178 |
| E4-only | .2 | .272225 | .229009 | .174642 | .225292 | .140457 | .280000 | 887 | 504 | 99 |
| E4-only | .5 | .272225 | .229009 | .174642 | .225292 | .186450 | .604444 | 1,737 | 1,088 | 211 |
| E4-only | 1 | .272225 | .229009 | .174642 | .225292 | .206695 | 1.090556 | 2,895 | 1,963 | 415 |
| E4-only | 2 | .272225 | .229009 | .174642 | .225292 | .204288 | 2.075556 | 5,008 | 3,736 | 810 |
| Fixed 50/50 neural+E4 | .2 | .402742 | .330783 | .174642 | .302723 | .161780 | .202222 | 920 | 364 | 29 |
| Fixed 50/50 neural+E4 | .5 | .402742 | .330783 | .174642 | .302723 | .227474 | .498333 | 1,980 | 897 | 71 |
| Fixed 50/50 neural+E4 | 1 | .402742 | .330783 | .174642 | .302723 | .274982 | .975556 | 3,500 | 1,756 | 123 |
| Fixed 50/50 neural+E4 | 2 | .402742 | .330783 | .174642 | .302723 | .329746 | 1.870556 | 6,153 | 3,367 | 275 |

At fixed 50/50, B=.2 per-fold joint FWR is .25778/.22889/.14889/.17333 (F0-F3), rather than a guarantee of ≤.2. The study is one repeatedly explored 12-speaker TRAIN cohort; no deployment/safety or independent-cohort claim. These are mixed seed101 assemblies: archived target scores plus new five-epoch exact-gap head fits. They are not a full formal phase01 trained end-to-end model.

## Frozen phase01 design reviewed

Candidate authoritative plan remains `reports/METHOD_BENCH_08_TRAIN_PHASE_01_SPEC.md` and machine config hash `6206c19d…cb1` (phase0 receipt). The specified primary is modular: separate target H-002 `2343→256→128→3`, continuous-frame CTC `768→43` blank42 + canonical-mean E4, fixed .5/.5 neural/E4 S/O fusion, direct H-002 fallback, and independent exact-gap MB03 `1614→128→64→2`; no shared trainable trunk. Target weighted CE `[1,2.5,4]`, AdamW `.001/.01`, batch512, 12 fixed terminal epochs; CTC CTCLoss zero-infinity blank42, AdamW `.001/.0001`, batch16 utterances, clip5, 8 epochs; gap fit-only square-root class weights normalized mean1, AdamW `.0008/.0001`, batch1024, 5 epochs. Fit-only target/gap scalers and competitor generation remain mandatory.

Official folds are preserved. Formal seed plan is 101/202/303; inner seed `base+10*outer+local_inner` (local indices 0,1,2), outer refit `base+100+outer`, initialize all RNG before model construction. B's bounded seed101 gap work follows those new inner/outer schedules; existing target and CTC archived seed101 inner receipts use local 0/1/2. B's outer gap seed schedule matches `201+outer`. Do not change architecture/loss/budget based on this bounded policy result. Shared-trunk and staged alternatives remain NOT_TESTED.

## Readiness blockers versus nonblockers

**Blocking the complete A runner/formal launch at this audit:** current `agent_a/runner.py` no longer matches its original `SOURCE_FREEZE.json` runner hash; no amended A freeze was present. Rerun of current `test_runner.py` had **2 failed, 4 passed**. The current `gap_input` is now correct N+1: Agent C's independent synthetic first/interior/last edge check passed. The two failing tests appear to be invalid fixtures: first sends N=12 labels to the N+1=13 gap output; second expects the trailing gap's previous-phone 39D left reference to be zero. A must correct tests and freeze the exact current runner/test/config hash bundle before further outcomes. A's handoff still does not claim full end-to-end audio/feature/E4 integration. Exact producer E4 parity, audio+supplied-intended-phone inference export, all-real event coverage/unknown handling, and label-free inference path remain NOT_TESTED. Only direct target next-update restart was previously tested; gap/CTC integrated interrupted-vs-uninterrupted optimizer/RNG/sampler/cursor equivalence remains NOT_TESTED. No eligible TRAIN runner smoke is verified.

**No longer a blocker for bounded joint policy:** exact MB03 source and target inner provenance exist; B completed bounded gap refits. Historical target checkpoints remain absent but score/source fit provenance is verified. Do not repeat old source-absence statement.

**Nonblocker under explicit input scope:** prompt-only lineage remains INCONCLUSIVE. Supplied intended phones are permitted. The explicit E4 aliases and SIL-only alignment fallback must remain distinct and visible.

**Still NOT_TESTED:** full phase01 per-seed ≤45-minute timing/maxRSS with all branches; extrapolating B's 33.57-second gap components and historical 614.95-second CTC inner receipt is not a measured complete seed. No full 3-seed/fold controlled training, independent-cohort confirmation, or deployment evidence.

## Handoff files and provenance

- `independent_train_audit.py` — executable, read-only TRAIN lineage + score/policy audit; requires `INDEPENDENT_AUDIT_FREEZE.json` and checks all source hashes before execution.
- `INDEPENDENT_AUDIT_FREEZE.json` — script/config/input/fit/policy provenance frozen before the successful independent metric run.
- `independent_train_audit.json`, `independent_policy_audit.json`, `independent_joint_metrics.csv`, `TRAIN_WAV_SHA256.csv`, `INDEPENDENT_AUDIT_RECEIPT.json`, `independent_audit_run.log` — machine receipts, summary metrics, WAV integrity and exact bounded run output.
- `risk_register.csv`, `TRAIN_PHASE_01_FROZEN_PROTOCOL.md` — concise risk and candidate protocol; supervisor authority unchanged.
- `pre_freeze_exploratory/` retains preliminary pre-freeze audit outputs without overwriting them.
- B results remain in `preflight/agent_b/results/`; A/B code and reports were read-only from this lane.

Supervisor independently reported verification of 136 MB08 artifact hashes / 283 frozen sources / 41 EVAL contract tests; those are supervisor receipt results, not claimed as rerun by Agent C. This lane independently verified its listed source hashes, WAV/manifest join, B fit/policy receipts and MB03/AP metrics. Old `HANDOFF.md` and `FINAL_HANDOFF.0eb6c855.md` remain unchanged; this report corrects their source-absence assertion.
