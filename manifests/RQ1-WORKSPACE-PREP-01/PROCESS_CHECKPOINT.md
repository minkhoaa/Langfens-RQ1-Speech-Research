# RQ1 two-workspace preparation — shared-base checkpoint

## Preparation record

State: **COMMON_DOCUMENTATION_PREPARED; WORKTREE_CREATION_NEXT; experiments NOT_AUTHORIZED.** This is a documentary continuation of RQ-RESEARCH-01, not an experiment result. Subsequent verification is appended separately in `WORKSPACE_VERIFICATION.md` when available. No current model artifact created; training NOT_STARTED, 0 successful optimizer updates, 0 skipped updates/no attempts.

### Inputs and preservation

- Initial local branch `research-phase`, HEAD `b25a7595360294798d8d06b37592dd65dd4c9c33`; remote tracking happened to match, but local working bytes were newer.
- Pre-existing uncommitted files: global checkpoint index and parent RQ-RESEARCH-01 checkpoint modifications, entire closure/revalidation directories, dirty nested `repo` at `baa4989243d5ff46b8b3bf5a3c280cdf2857382b`. Index initially empty. No reset/clean/stash/rebase/force/push/merge or nested repair.
- Created dedicated branch `prep/rq1-shared-base` without discarding changes. Existing closure/revalidation documentary bytes will be committed explicitly with shared preparation, not silently omitted. Nested gitlink and its mode/symlink dirt excluded from staging and preserved. Other worktrees untouched, including stale `/tmp/langfens-e1-launch` registration (not pruned).
- Read AGENTS, registry, latest coordinated closure/checkpoint, source/data/thesis/pilot reports, original UIT proposal and current RQ1 source seams. Latest distinct omission proposal not found in scoped documentary search; conformance remains conditional, as recorded in the addendum.

### Outputs frozen before branching

- `PROPOSAL_CONFORMANCE_ADDENDUM.md`: scope reconciliation and explicit missing-source disposition.
- `EVALUATION_CONTRACT_V1.md`: new immutable contract, not a historical contract replacement.
- `CONTRACT_SHA256`: exact shared contract digest.
- `SHARED_PROVENANCE.json`: current source/input/archive fingerprints, checkpoint identities as archival metadata only, future artifact-verification gates.
- This checkpoint and append-only registry update. No arm-specific documents exist on the frozen base.

### Verification actually executed before common commit

- Initial branch/HEAD/status/branches/worktree list inspected; requested sibling paths and branch names not occupied. No remote commit assumed newer than local state.
- Closure checksum inventory: 17 entries verified; revalidation inventory: 10 verified. Original thesis digest `98fa922584b750510d4b0c183c59a84455da9e64ab231b59bdb58862c56133d3`; split digest `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b` verified.
- Frozen TRAIN/VAL lists and speaker-disjoint reduced-TRAIN FIT/SELECT/CAL selectors computed solely from split metadata, not labels/audio. TRAIN 2,399; VAL 300; FIT 1,799; SELECT 300; CAL 300. TEST metadata used only for exclusion.
- Initial tracked-file hashes and outer/nested status/diffs saved at `/tmp/langfens-two-workspaces-audit/` (ephemeral audit aids, not durable artifacts). 640 tracked regular files fingerprinted, no model/data loaders run. Source/evaluator/config/archive contents unchanged except permitted registry additions; hashes are rechecked before commit and in final verification.
- Git inventory contains inherited prediction audit tables, dictionaries and paper PDFs (some >1 MiB); these already exist in history. No new dataset/audio/TextGrid/model/secret/large binary is staged or copied into Git by this process. Worktrees will check out existing tracked source/audit assets normally, not clone ignored datasets/checkpoints/credentials or nested checkout dirt. No publication/push authorized; inherited privacy issues remain.
- No weight/durable-store readback: E0/E1 identities are recorded from archived metadata, not freshly certified. Untracked ignored local weights are not loaded, copied or edited; Git-only verification cannot certify remote artifact availability.
- Executed source/input/archive verification: 57 hashes match; 37 staged files pass the explicit documentary allowlist, regular-file/no-symlink and <100,000-byte checks. New shared Markdown contains no Markdown-link targets; registry relative links are checked in final verification. Scoped `git diff --cached --check -- manifests/RQ1-WORKSPACE-PREP-01 docs/process-checkpoints/INDEX.md` is clean.
- Whole staged `git diff --cached --check` reports 12 pre-existing Markdown hard-line-break trailing-space findings in three original closure reports (B/C/E). Their checksummed historical bytes are intentionally preserved, not repaired or called clean. This is a documented whitespace exception, not a source/test failure. No unexecuted unit/model tests are PASS.

### Exact safe continuation / stop

1. Review/stage only explicit research-closure/revalidation and shared documentary paths plus index/parent checkpoint. Verify historical bytes, nested diff and protected source hashes.
2. Commit shared preparation on `prep/rq1-shared-base`; resolve `git rev-parse HEAD` as the **frozen common base SHA** and contract SHA. Record that exact SHA in both arm manifests/checkpoints after creation; this avoids a self-referential commit hash. The later coordinator verification commit may advance this preparation branch, but MUST NOT change the frozen base used by the arms.
3. Verify branches/paths still absent, then `git worktree add -b prep/rq1-p-score ../langfens-rq1-p-score <frozen-base>` and equivalent `prep/rq1-p-sup`. No `--force`, branch reset, copying dirty nested checkout or initializing submodules.
4. Prepare arm-only docs/checkpoint/registry in each respective worktree; both use exact same contract bytes. Commit docs only. Verify merge-base, both contract hashes, disjoint arm documents and unchanged source/E0/E1 artifacts. Append cross-workspace verification to coordinator branch and registry with exact base/tips.
5. **STOP.** Setup authority does not authorize scorer/model/trainer implementation, tests that instantiate models, any fitting/inference/backward/training/GPU/remote staging or sealed TEST access. Safe next step is document review plus separate bounded implementation authorization, missing-proposal disposition, prospective effect margins and P-SUP objective-policy amendment.

Retained references: E0 `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`; E1 fixed `3b686730cc0a1101a1bbc96b59dc7bd38a9fc3cc7da7effe26b2d9498f079d24` (archives only). DIAG-02B PARTIAL and E1 rollout NO-GO unchanged.

## Append-only completion record

**SETUP_COMPLETE_WITH_ACCOUNTED_SOURCE_LIMITATIONS; EXPERIMENTS_NOT_AUTHORIZED.** Frozen common base is `0d268ff8c444e16fd76aca2f4ad6ece966908edc`; shared contract SHA-256 `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`. Both isolated arm documentation commits are direct children: P-SCORE `de1db5cc2f0b4ed8b750ccc6120a314679dc3b6f`, P-SUP `d6ef18443752b0add3f3dfcfff42f7fa7be967c1`. Exact paths, evidence, unresolved source/objective/implementation gates, one corrected audit-predicate failure, preserved inherited whitespace and safe commands are in [WORKSPACE_VERIFICATION.md](WORKSPACE_VERIFICATION.md) and [JSON receipt](WORKSPACE_VERIFICATION.json). Original nested dirt remains unchanged. No implementation/tests/inference/fitting/backward/training/GPU/deployment/TEST access. Current successful/skipped updates remain 0/0; no model artifact. STOP after documentation. This coordinator completion commit records receipts only and does not change either arm's frozen base or authorize execution.

## Scientific QA & Comparative Audit Record — Agent D

**PREPARATION_AUDITED_CRITERIA_FROZEN; CANDIDATE_METRICS_PENDING_EXECUTION; EXPERIMENTS_HOLD.** Completed formal scientific QA and comparative evaluation report: [SCIENTIFIC_QA_AND_COMPARISON_REPORT.md](SCIENTIFIC_QA_AND_COMPARISON_REPORT.md).

- **Evaluation Contract V1 Verified:** SHA-256 `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2` verified bit-identically across coordinator and both arm worktrees. Data sealing invariants confirmed (Fold 1 TEST partition ABA/ASI/EBVS/HJK/LXC/PNV 900 utterances strictly absent from staging package).
- **Historical E0/E1 Provenance Reconciled:** Baseline R `best.pt` (3,600 updates, SHA-256 `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`). Historical AP numbers reconciled: online GPU FP16 AMP `0.316062`, offline CPU FP32 replay `0.317021` (explained by FP16/FP32 precision boundaries), cross-fold pilot `0.317012`, Protocol Z Aligned-Only `0.313530` ($N=9,919$). E1 benchmark confirmed omission AP `0.032252` and complete collapse of skip mass on deletions (`0.000000`).
- **Strict Label & Denominator Accounting:** Fold 1 validation population confirmed at 300 utterances and 9,956 canonical phone tokens ($N_{\text{cor}}=8,799$, $N_{\text{sub}}=1,012$, $N_{\text{del}}=145$). Distinct evaluation cohorts certified: Binary AP ($N=9,956$), Omission AP ($N=8,944$, substitutions strictly excluded), Substitution AP ($N=9,811$, deletions strictly excluded), Realized PER reference ($N_{\text{realized}}=9,881$). Deletion-vs-all metrics mislabeled as omission AP rejected.
- **Prospective Acceptance Criteria Frozen:** Prespecified margins frozen prior to candidate metric inspection: P-SCORE primary contrast SF-SD vs SF-S ($\Delta \text{AP}_{\text{omit}} \ge +0.020$, $\Delta \text{Recall@FPR15} \ge +0.030$, noninferiority $\ge -0.005$); P-SUP primary contrast T-OP vs B-CONTROL ($\Delta \text{AP}_{\text{omit}} \ge +0.030$, $\Delta \text{Recall@FPR15} \ge +0.040$, noninferiority $\ge -0.005$, $\Delta \text{PER} \le +0.005$).
- **Controls Audited:** Within-speaker cyclic shift-1 mismatch control (`within_speaker_cyclic`, baseline acoustic gap $+0.022299$) and canonical prior baseline ($\text{AP}=0.303063$, neural margin $+0.013958$) certified as mandatory falsifiers.
- **Candidate Predictions & Execution State:** Neither candidate has executed real-data inference or training (updates=0, attempts=0, model artifacts=0). Candidate comparative rankings are deferred until verified real-data predictions are produced. Inferring statistical superiority from synthetic tests or small sample sizes ($G=2$) is strictly prohibited.
## One-Shot Parallel Campaign Integration Record — Coordinator

**P-SCORE_BENCHMARK_COMPLETE; P-SUP_PIPELINE_STAGED_AWAITING_KAGGLE_WEB_UI; INFRASTRUCTURE_VERIFIED; CRITERIA_FROZEN.** Four independent Gemini subagents executed concurrently under strict platform boundaries (Local CPU evaluation, Kaggle T4 staging, zero sealed TEST access).

1. **P-SCORE Experimental Execution (commit `a9f4947985651253cee980086a346056885b51c6` on `prep/rq1-p-score`):**
   - 269 synthetic unit/regression tests passed.
   - Closed graph, normalization and clean-room independent author-code licensing blockers (`ctc-union-language-v1` S/SD graphs, exact posterior occupancy, raw primary scoring).
   - Generated FP32 log-emissions cache on CPU using frozen E0 checkpoint (`best.pt`, SHA-256 `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`) for all 300 Fold-1 VAL utterances (NJS: 150, RRBI: 150). Cache SHA-256 `a699fd9378c86cd348a0628a144ecacfef491ea8f90e0edd23b4e46932e3e829`.
   - Executed `p_score_infer.py` with `--mode frozen-cache` across 11 CPU workers; emitted 39,824 tokens passing `strict_runtime_validator.py` with 0 errors.
   - Primary contrast (**GOP-CTC-SF-SD minus GOP-CTC-SF-S** on $N=9,919$ aligned):
     - **Delta Omission AP**: **+0.021615** (+2.16 AP points, 0.051026 vs 0.029411, relative +73.5%), meeting prospective acceptance margin ($\ge +0.020$).
     - **Delta Recall @ FPR15**: **+0.182482** (+18.25 percentage points, 43.07% vs 24.82%), meeting prospective margin ($\ge +0.030$).
     - **Delta Substitution AP**: **-0.024256** (0.222544 vs 0.246800), identifying omission-substitution tradeoff and failing strict noninferiority ($\ge -0.005$).
     - **Delta Binary AP**: **-0.009045** (0.250049 vs 0.259094).
   - Controls: Greedy CTC PER **0.236818** (2,340 edits / 9,881 references), exactly replicating historical baseline R. GOP-SA (Viterbi control): Omission AP 0.049846, Binary AP 0.254370. Greedy DP: Omission AP 0.019002 (42.38% coverage).
   - Word-final consonants (93/137 deletions): Omission AP delta **+0.049194** (0.181265 vs 0.132071).

2. **P-SUP Implementation & Staging Execution (commit `10c4157` on `prep/rq1-p-sup`):**
   - 97 unit and pipeline tests passed (`tests/test_p_sup_pipeline.py`).
   - Implemented end-to-end dataset allowlist, loader, trainer, and evaluator under Evaluation Contract V1 connecting Fold-1 TRAIN (2,399 utts) and VAL (300 utts) with strict sealed test speaker rejection.
   - Registered arm-scoped objective policy amendment in `manifests/RQ1-P-SUP/OBJECTIVE_POLICY_AMENDMENT.md` for Subtype Treatment `T-OP` (`operation_ce`) and Matched Binary Control `B-CONTROL` (`binary_collapsed`), exactly matching parameter budget (266,667 head parameters).
   - Audited preflight gate (`STAGING_VERIFIED_PASSED`) and staged Kaggle deployment package to private kernel `minkhoaa/langfens-rq1-p-sup-t4-training` using official CLI `--no-run`.
   - Documented exact operational status: Kaggle Web UI user secret `HF_TOKEN` attachment and manual "Save & Run All" is required by AGENTS.md policy before background remote T4 execution begins. No local training fallback was attempted.

3. **Infrastructure & Storage Verification:**
   - Cryptographically verified authorized Fold-1 dataset `authorized_RQ1-TRN-02A_fold1_train_val_only.zip` (SHA-256 `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`).
   - Verified strict speaker isolation (16 train, 2 val, 0 test; zero test leakage).
   - Verified Baseline R E0 checkpoint `best.pt` (SHA-256 `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`, 3,600 updates, AP 0.316062).
   - Verified Kaggle weekly GPU quota: 28.64h available out of 30.00h. Probed Hugging Face private repository checkpoint write-readback-delete durability.

4. **Scientific QA & Comparison Synthesis:**
   - Authored master scientific report: [SCIENTIFIC_QA_AND_COMPARISON_REPORT.md](SCIENTIFIC_QA_AND_COMPARISON_REPORT.md).
   - Reconciled historical AP baselines (.316062 online Colab FP16, .317021 offline CPU FP32, .317012 cross-fold pilot, .313530 Protocol Z aligned).
   - Certified strict denominators: Binary ($N=9,956$), Omission ($N=8,944$, substitutions strictly excluded), Substitution ($N=9,811$, deletions strictly excluded).
   - Confirmed prospective criteria frozen prior to candidate evaluation.

## Scientific Evaluation Audit & Metric Reconciliation Record — Agent D

**POPULATION_RECONCILED; HONEST_TRADEOFF_RECORDED; SPECIFICATIONS_VERIFIED.** Comprehensive scientific evaluation audit report authored: [P_SUP_AUDIT_AND_METRIC_RECONCILIATION.md](P_SUP_AUDIT_AND_METRIC_RECONCILIATION.md).

- **Population Reconciliation (9,956 Full vs 9,919 Aligned):**
  - Mapped all 37 unaligned tokens in Protocol Z Levenshtein dynamic programming alignment (`sub_first` tie-break): 25 correct, 4 substitutions, 8 deletions (total 37).
  - Categorized root causes: 9 wh-aspirate discrepancies (`/HH W/` in annotator TextGrid vs `/W/` in CMUdict G2P), 2 contraction expansions ("we'll" `/W IY W IH L/`), 14 weak vowel/syllabicity variants ("privileges" `/IH/`, "could", "talk"), 8 gold acoustic deletions dropped by G2P frontend phonemization, and 4 phonetic substitutions.
  - Certified metrics on both populations:
    - Aligned population ($N=9,919$, $N_{\text{cor}}=8,774$, $N_{\text{sub}}=1,008$, $N_{\text{del}}=137$): Baseline R Binary AP 0.313530, Omission AP 0.032564, Recall@FPR15 39.42%, Substitution AP 0.304811, PER 0.236818.
    - Full population ($N=9,956$, $N_{\text{cor}}=8,799$, $N_{\text{sub}}=1,012$, $N_{\text{del}}=145$): Baseline R Binary AP 0.317021, Omission AP 0.035020, Recall@FPR15 40.00%, Substitution AP 0.307366, PER 0.236818.
- **Cross-Model Comparative Evaluation on Exact Same Population ($N=9,919$):**
  - Direct side-by-side comparison establishes:
    - `GOP-CTC-SF-S`: Omission AP 0.029411, Recall@FPR15 24.82%, Substitution AP 0.246800, Binary AP 0.259094, PER 0.236818.
    - `GOP-CTC-SF-SD`: Omission AP 0.051026, Recall@FPR15 43.07%, Substitution AP 0.222544, Binary AP 0.250049, PER 0.236818.
    - `GOP-SA` (Viterbi Control): Omission AP 0.049846, Recall@FPR15 41.61%, Substitution AP 0.228104, Binary AP 0.254370, PER 0.236818.
    - `Greedy DP`: Omission AP 0.019002, Recall@FPR15 38.10%, Substitution AP 0.175058, Binary AP 0.186678, PER 0.236818 (coverage 42.46%).
- **Prospective Gate Compliance Audit & Omission-Substitution Tradeoff:**
  - Primary Omission AP: $\Delta = +0.021615$ vs target $\ge +0.020$ $\to$ **PASS** (+73.5% relative).
  - Operating Sensitivity: $\Delta \text{Recall@FPR15} = +0.182482$ (+18.25 pp) vs target $\ge +0.030$ $\to$ **PASS** (43.07% vs 24.82%).
  - Substitution AP Noninferiority: $\Delta = -0.024256$ vs target $\ge -0.005$ $\to$ **FAIL (Severe Noninferiority Breach)**.
  - Binary AP Noninferiority: $\Delta = -0.009045$ vs SF-S; $\Delta = -0.063481$ vs Baseline R $\to$ **FAIL**.
  - Omission-Substitution Tradeoff thoroughly documented: The $\epsilon$-deletion shortcut in the denominator lattice bypasses substituted acoustic segments, artificially deflating substitution posteriors and degrading substitution diagnosis by 9.83% relative.
- **Matched T-OP vs B-CONTROL Training & Evaluation Specifications:**
  - Verified identical random initialization (`seed: 1337`) via `matched_initial_pair()`.
  - Verified identical head architecture: `CompactOperationHead` comprises exactly 266,667 trainable parameters across both arms.
  - Verified identical frozen backbone: `facebook/wav2vec2-base-960h` (94.4M params, frozen in eval).
  - Verified identical optimization hyperparameters: AdamW (lr 0.001, wd 0.01), cosine schedule over 15,000 updates, 100 warmup updates, batch size 2x2 = 4, CUDA FP16 AMP.
  - Enforced mandatory dual checkpoint accounting on Fold-1 VAL: Evaluation must report fixed-budget 3,600 updates (`fixed_budget_3600.pt`) AND validation-selected `best.pt` separately to avoid confounding update exposure with algorithmic performance.



