# TRAIN-PHASE-01 scientific and reproducibility audit

**Supervisor-integrated finding: GO for infrastructure preflight.** Primary data/score lineage and bounded joint-policy evidence survive independent verification. The subsequently completed controlled driver, archival/selection lifecycle and whole-seed stopping controls now pass bounded tests and independent review. Formal training remains NOT_RUN; this is not deployment or full-run performance approval. The earlier conditional audit is preserved under `preflight/orchestration_v2/prior_reports/`.

Evidence root: `experiments/rq1_train_phase_01/preflight/`. Agent C's independent audit/protocol/risk files are preserved in `agent_c/`; supervisor receipts and authoritative risk register are in `supervisor/`. Intermediate C findings about failing A tests/absent freeze were snapshots during concurrent correction and are superseded by final A freeze and supervisor24 passing tests. C's final `FINAL_HANDOFF.0eb6c855.0e1f2305.061fc1f1.md` explicitly reconciles those findings, confirms its own successful A7/integration8 reruns, and supported that earlier conditional disposition; the later orchestration closure and fresh review supersede its then-unimplemented-driver status. Its reference to24 tests is to the final strict-boundary suite (`final_runner_policy_tests.log`), not the earlier21-test log. A provider timeout before that handoff was preserved and recovered by exactly one same-model native retry; no fits or tests were repeated on the retry.

## 1. WAV, written text and supplied intended phones

**VERIFIED:** C's TRAIN-only audit checked all1,800 manifest audio paths, nonempty16-kHz WAV headers and per-WAV SHA256. `agent_c/TRAIN_WAV_SHA256.csv` is the complete inventory; its hash is `4a70ad56cd6fc78452192f4b923d0c613d24c94d4f5da3baee6685fb8db1aef6`. Full target/gap topology, H002 event/reference order and four zero39 references match the authoritative TRAIN manifest (SHA256 `1f4da8518d9b184fb737b93c7887ecd7c3e73d6e83260e54651074831f61bcff`).

The H002 extraction receipt binds the frozen encoder snapshot, waveform normalization, predicted CharSiU alignment, H6 crop/reference/event outputs and TRAIN WAV manifest. Source inspection found no gold timing, realized phones or error labels in held-out acoustic feature generation. Current preflight reuses those audited caches; it does **not** re-extract the encoder or establish fresh waveform-level parity by execution.

**Important lineage limit:** `tools/parse_l2arctic.py` reads written PROMPTS text but obtains canonical target phones from the TextGrid phone tier. Co-location of text and phones is not a deterministic dictionary/text→phone certificate. Therefore audio+**supplied intended phones** is the authorized scope; audio+prompt-text-only lineage remains **INCONCLUSIVE**. Canonical supplied-phone conditioning is not being presented as an independently text-derived frontend. This is nonblocking only under the explicitly declared supplied-phone contract.

## 2. Unknowns and training/inference separation

| Event | Raw token | Neural/gap reference | CTC/E4 intended-token rule |
|---|---|---|---|
|ABA_arctic_a0025::21|ER)|zero39|explicit alias ER|
|BWC_arctic_a0434::28|D_|zero39|explicit alias D|
|SVBI_arctic_a0006::7|SPN|zero39|inventory class32|
|HJK_arctic_a0104::4|Z_|zero39|explicit alias Z|

All four zero vectors and event slots were independently verified in the full cache and exercised in complete-utterance strict inference tests. CTC's documented aliases Y_→Y/AH)→AH are also retained; unregistered tokens fail. H002's alignment-only OOV→SIL mapping is separate and disclosed. There is no silent AA/phone0 replacement of unknown references.

Gold timing/realized phone sequences may supervise eligible CTC fitting; labels may train heads, fit inner policies and evaluate outputs. They are not inference tensor features. The accepted typed strict cache inference facade only exposes acoustic contexts, supplied phones/reference vectors and identity metadata; metadata speaker IDs are used for keys, never network inputs. It generates per-utterance N/N+1 contexts/keys itself and rejects gold dictionaries and malformed/nonfinite inputs.

A's original permissive low-level multi-utterance helper is not accepted alone as an export boundary. `supervisor/validated_inference.py` plus three executed strict tests closes that boundary for the declared cached scope. No raw-text frontend or deployment claim follows.

## 3. Speaker grouping, event identity and baseline reproduction

Official four held groups are ABA/ASI/BWC; EBVS/HJK/HKK; HQTV/LXC/NJS; PNV/SKA/SVBI. Authoritative TRAIN has59,562 targets (C50,356/S7,381/O1,825),61,362 gaps (I560). Each outer fit uses nine other speakers; each local inner fit uses six and excludes both three-speaker validation and outer groups. All fit-derived transforms/weights/competitors must follow that exclusion.

**VERIFIED independently by supervisor and C:** seed101 archived neural/CTC inner joins cover178,686 target rows across four outer pools exactly once per pool;12 CTC checkpoint hashes and fit/held metadata match. Supervisor also rechecked the archived MB08 integrity evidence, including24 reused CTC checkpoint records across seeds101/202. Neural historical checkpoints were not retained; this is qualified archived score/source provenance, not invented checkpoint replay.

The exact MB03 producer is present: `experiments/rq1_method_bench_03/agent_c/train_speaker_adversary.py`. It contains feature construction, PLAIN Detector (with dormant speaker branch), fit-local scaler/weights, optimizer, initialization/shuffle and epoch loop. Initial agents' source-absence claims were incorrect; supervisor recovered the file and C explicitly corrected the assertion. Historical unkeyed arrays are mapped by exact producer manifest traversal within folds, not just matching counts.

Independent archived MB03 reproduction: AP-S .4232088151, AP-O .2767063392, AP-I .1621248970, Macro3 Error AP .2873466837, insertion P@560 .2428571429. Historical selected MB08 AP-S/O/I .402741820/.330783489/.162124897 and Macro3 AP .298550068 reproduce. All136 archived MB08 artifact hashes and283 phase0 source/config/report hashes are unchanged.

## 4. Runner defect, fixes and restart evidence

C correctly found initial `gap_input` produced N rows and shifted left/right contexts. Original tests incorrectly expected N. That implementation is preserved under `agent_a/prior_version/`; accepted source now emits N+1 with exact zero edges and correct neighbors. Current amended BCF2 source/test/config freeze matches; seven A tests pass. Stale intermediate C two-failure conclusions are not final evidence.

Supervisor went beyond manual checkpoint fixtures: actual target/gap/CTC fit loops are interrupted mid-epoch and at an epoch boundary, then resumed through tiny synthetic two-epoch runs. Model, optimizer, RNG, sampler, order, cursor and epoch match uninterrupted runs **exactly**. All four real zero39 references and all official grouped plans are checked. Combined runner/policy/strict-inference suite:24 passed. Independent legacy EVAL/policy suite:41 passed.

A's real TRAIN smoke has24 utterances/six eligible fit speakers,738 targets/762 gaps, one epoch, three full-state checkpoints;61.955s wall/maxRSS865,252KiB. Supervisor independently rechecks all input/source/checkpoint hashes and replays target/gap scores from checkpoint/scalers in `runner_receipt_verification.json`. This is in-fit plumbing only, not a held-out model metric or full-budget feasibility result.

## 5. Independent policy and critical metric audit

B executed only necessary bounded gap work:12 six-speaker inner fits and4 nine-speaker outer fits, fixed5 epochs and phase01 seed schedule. Exact model/scaler/fit/held IDs,16 checkpoints and keyed score NPZs are retained. Combined phase elapsed wall33.57s (not CPU process seconds), two threads, no GPU. Inner thresholds froze before outer gap fits/application.

C independently recalculated96 cutoff occurrences across48 policy pairs and complete outer metrics; supervisor independently repeated cutoffs, validated16 checkpoint/scaler/ID records and invoked official EVAL-V3.1 for all12 method/budget pooled outputs. All class counts, wrong-type FP, Macro3 F1/AP, FWR and coverage match. Supervisor also reconstructed held edge/interior features and replayed all16 gap checkpoints (max probability error1.20e-7). Full-domain keyed policy exports remove target/gap compact-ID ambiguity.

Fixed50 primary B=.2: AP-S/O/I .402742/.330783/.174642, Macro3 AP .302723, Macro3 F1 .161780,364 false warnings including29 wrong types,920 total warnings, **jointFWR .202222**. This is valid bounded **archived-target/new-gap mixed evidence**, not formal3-seed training. FWR slightly exceeds nominal.2; per-speaker maximum.36. No exact matched-achieved-rate or safety claim is justified. Conditional speaker bootstrap and exploratory interpolation are separately labeled in the joint report.

Historical qualifications remain preserved: MB07 total-warning/FWR mislabeling; last42 prior slice includes three acoustic dimensions; ordinary learned meta crossfit does not establish entire-base exclusion. Primary fixed fusion avoids that prior/meta dependency. Historical MB07 missing GPU lease is not precedent; current preflight used CPU only.

## 6. Frozen design and remaining risks

Losses/scalers, official groups, seeds101/202/303, fixed terminal target12/gap5/CTC8, budgets, baselines and mandatory acoustic/reference controls remain unchanged from the pre-outcome MB08 specification. Shared-trunk/staged models remain NOT_TESTED. No evidence supports replacing the independent branches with a unified network.

Authoritative machine risk register: `supervisor/risk_register.csv`. The three former launch blockers are now **RESOLVED**:

1. `train.py` and its modules schedule the full official seed/fold/inner protocol, bind eligible streams, seal policies before outer refit, and export/evaluate fold/speaker/pooled outputs.
2. Each branch resets seed before construction, persists source/scaler/fit/held IDs and full optimizer/RNG/sampler state, archives every epoch with immutable hashes, and reloads only the fixed terminal checkpoint. Corrupt archives/decisions and incomplete historical indexes fail closed; only the just-completed archive commit can recover by exact state equality.
3. Persistent cumulative2700s/seed budget covers population initialization/loading through evaluation, stops safely on cap/signals, retains used time across resume and disallows silent extension. Budget control—not a full45-minute training run—was tested with generated data and a deterministic clock.

Independent fresh-context native `gpt-6-luna` review initially found budget initialization, two-file policy commit and archive-index recovery issues; each was fixed with a regression fixture and the final review reports no remaining blocker among them. One atomic policy document now contains payload and digest. The final suite **91 passed** (65 prior+26 new; one benign existing empty-N warning). Full generated-data CLI rehearsal completed4outer/12inner bundles,48 branch fits and96 epoch archives in10.236s process wall/703,240KiB child maxRSS. Tiny fixture models and generated events are not real-data performance evidence. New-backend E4 competitors/LLRs numerically match the verified MB07 producer at1e-12 on generated inputs.

Full formal-seed wall/RSS remains NOT_TESTED, not a fabricated feasibility measurement. Current resources support bounded execution and the hard-cap contract prevents unbounded work; future authorized runs must report partial completion if capped. Prompt-only lineage, missing historical neural checkpoints, repeated TRAIN exploration and speaker FWR dispersion remain explicit claim/generalization limits. No protected cohort or real model refit was used to close the engineering findings.

## 7. Reproducibility and stop

Code/commands/hashes/receipts: C's `independent_train_audit.py` and freeze/receipt, B's freezes/raw scores/checkpoints/cutoffs, supervisor `verify_{baseline,inner_targets,joint,gap_checkpoints,runner_receipts}.py`, strict inference and actual fit-loop tests. `supervisor/COMMANDS.md` lists verification commands; final evidence manifest indexes all outputs and six reports.

**MEASURED:** bounded fits, smoke, metrics, runtime/RSS/tests. **VERIFIED:** integrity/identity/exclusion/lineage within scope, model/scaler replays and exact actual-loop resume. **INCONCLUSIVE:** text-only derivation/external safety. **NOT_TESTED:** formal controlled run, full-budget orchestration/feasibility, fresh encoder extraction and architecture ablations.

No formal training, CUDA/lease, DEV_SEARCH tuning, protected/TEST_SEALED reads, destructive changes or remote push. Stop at the **GO infrastructure-preflight gate**. Formal execution still requires separate explicit operator authorization. Closure evidence/reviews/source freezes are under `preflight/orchestration_v2/`; the earlier C reconciliation remains historical, not a claim that later implementation was absent.
