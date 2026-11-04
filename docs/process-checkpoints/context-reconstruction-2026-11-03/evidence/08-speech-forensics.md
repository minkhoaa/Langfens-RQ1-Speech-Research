# SEAM 8 — Speech Git, implementation, and evidence forensics

## 1. Scope and observation boundary

**Discovery time:** 2026-11-03 23:42 +07 (Asia/Ho_Chi_Minh), as observed with `date -Is`. Git commit, author, and ref times below are reported separately from this discovery time. All findings are read-only observations of local files and Git metadata; no test, model, inference, training, service, or remote mutation was run.

| Repository | Observed HEAD and branch | Local refs / state |
|---|---|---|
| Speech: `/home/khoa/Projects/KLTN/Dataset` | `1fa4a5b8de206bbf0bd7cb7856f721d2b1d0adb6`, `prep/rq1-shared-base`; commit time `2026-11-03T22:31:54+07:00` | 40 local heads, 45 remote-tracking refs, 0 tags. `origin` is `https://github.com/minkhoaa/Langfens-RQ1-Speech-Research.git`. |
| Nested `Dataset/repo` | `baa4989243d5ff46b8b3bf5a3c280cdf2857382b`, detached | Nested repository with 42 refs and no tags; 16 pre-existing dirty paths, described below. |
| Recovery worktree | `/home/khoa/Projects/KLTN/diag02b-recovery-worktree`, `00c684408f377300b84aacbec1483b8a2e812967`, `review/rq1-diag-02b-safe-delivery-parity-recovery` | Clean status at observation; the branch is a delivered recovery/review commit, not proof that every parity failure was repaired. |
| Other registered speech worktrees | `/home/khoa/Projects/KLTN/langfens-rq1-p-score` at `a9f4947985651253cee980086a346056885b51c6`; `/home/khoa/Projects/KLTN/langfens-rq1-p-sup` at `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f`; `/tmp/langfens-e1-launch` at `baa4989243d5ff46b8b3bf5a3c280cdf2857382b` detached | The `/tmp` worktree is marked prunable because its gitdir points to a nonexistent location. No worktree was changed. |
| Reflog / reachability | Speech `HEAD` reflog was inspected; latest entries cover 2026-11-03 preparation and pinning work, and earlier entries reach 2026-10-23. | Speech has 85 total refs; oldest reachable commit observed with `git log --reverse` is `085bfcd4e935ed708722cd161dd1892ae16eb3d8` (`2026-10-23T15:01:33+07:00`). This is local history, not an exhaustive inventory of unreachable objects. |
| Remote GitHub API | Not queried. | No verified current GitHub visibility, PRs/issues/releases, default branch, or live remote tips. `origin` URL alone does not prove present remote visibility or that local remote-tracking refs are current. |

Read `/home/khoa/Projects/KLTN/Dataset/AGENTS.md` and `docs/process-checkpoints/INDEX.md` before investigation. The index is locally modified and contains append-only updates; therefore it is evidence of the checked-out worktree’s current documented state, not a clean committed source artifact. No remote API or network fetch was used.

### Observed pre-existing dirt — preserved

- Speech superproject: `docs/process-checkpoints/INDEX.md` modified; nested `repo` gitlink modified; untracked `docs/process-checkpoints/context-reconstruction-2026-11-03/` containing `PROCESS_CHECKPOINT.md`, `SOURCE_DISCOVERY_SNAPSHOT.json`, and `SOURCE_FILE_INVENTORY.tsv`.
- Nested `repo` has 14 modified `audit_scripts/*.py` files with **zero-byte diffs** and deletions of `references/papers/baevski2020_wav2vec2.pdf` and `references/papers/zhang2021_speechocean762.pdf`. Files were not opened or restored. Thus the papers are inaccessible in this worktree; this does not establish whether copies exist elsewhere.
- Recovery worktree: clean status, but its committed source state differs from the current coordinator worktree.
- No staging/index or working-tree mutation was performed.

## 2. Evidence inventory and confidence

Confidence labels: **VERIFIED** = directly observed in local primary source or Git; **CORROBORATED** = consistent across independent local records; **INFERRED** = reasoned from those records; **UNKNOWN** = not established in the read-only scope.

1. **VERIFIED:** `AGENTS.md` sets the research execution boundary: remote GPU only for training; no local fallback; per-fold test isolation; no gold TextGrid timestamps as inference input; checkpoint and artifact provenance requirements. It is policy, not proof that every historical execution complied.
2. **VERIFIED:** `docs/process-checkpoints/INDEX.md` summarizes prior RQ1 phases and links to reports/checkpoints; this audit read its latest research and DIAG sections but did not verify every indexed artifact or historical training claim.
3. **VERIFIED:** Primary diagnostic record `manifests/RQ1-DIAG-02B/SCIENTIFIC_REPORT.md`, process checkpoint, delivery record, `evaluation-final/REPORT.json`, and `reconciliation-final/` corrections/hashes exist locally.
4. **VERIFIED:** The DIAG-02B report states execution base `dd2dee9a2063284d2a0f9fbf1c18315f3a7cd774`, the DIAG-02 source commit, and distinguishes the 2026-11-02 report/checkpoint timestamp from older checkpoint-time claims.
5. **VERIFIED:** DIAG-02B final archive-parity result is **9/80** utterances outside unchanged `0.000200` tolerance; max discrepancy `0.001304954`. Full 1,199-utterance forward replay was not performed. Its status is `PARTIAL_ARCHIVE_PARITY_FAILED; NO_GO_TRAINING_CAUSAL_GATE`.
6. **VERIFIED:** The report identifies local, bounded CPU inference; no training, backward pass, optimizer update, or GPU execution for DIAG-02B. It documents 80 selected omission-containing utterances across eight speakers, 66 CTC matched pairs, and a 32-token waveform subset; these are not population-wide intervention samples.
7. **VERIFIED:** `reconciliation-final/CORRECTIONS.json` and its CSV reconcile historical numeric/hash claims; `evaluation-final/ARTIFACT_SHA256.json` is the canonical final-run hash ledger according to the report. This investigation did not recompute every listed digest.
8. **VERIFIED:** `git log --all -- manifests/RQ1-DIAG-02B` shows the evidence/recovery history including `9428d15` (report/results), `00c6844` (recovery audit), and `bf45207` (delivery tracking); current top-level branch contains the indexed report evidence.
9. **CORROBORATED:** Current index entries and DIAG-02B report/checkpoint both supersede DIAG-02 causal-training GO with a NO-GO on present evidence and preserve the broader alignment hypothesis as unresolved.
10. **UNKNOWN:** Live GitHub project metadata and current remotes; Kaggle/Hugging Face state; current credentials or visibility; any remote execution after recorded local commits. No credential content was inspected.

## 3. Event reconstruction (local documented and Git time)

| Date/time (+07) | Goal, decision, rationale | Action / reported result / limitation | Current validity |
|---|---|---|---|
| 2026-10-23, from `15:01` | Establish speech data/provenance pipeline. | Earliest observed local history begins at `085bfcd`; subsequent commits add parsing, audio loading, datasets, validation, then training contracts. These commits establish code history, not execution by themselves. | Historical implementation sequence; artifact/run claims require their own evidence. |
| 2026-10-23–29 | Establish Baseline R and remote execution process. | Index records Fold-1 training, checkpoint recovery/evaluation, private release, later acoustic-conditioning experiments and operational corrections. It reports training outcomes and remote persistence; this forensic pass did not independently query remote stores or re-run those jobs. | Retain as historical documented evidence, not a new live-state attestation. |
| 2026-11-01–02 | Cross-fold Baseline R, then omission diagnostics. | Index records folds 2–4 and validation synthesis; `RQ1-DIAG-01` records omission/FN analysis and `RQ1-DIAG-02` reports temporal-attention findings and a training GO. | Cross-fold/diagnostic claims are documented; the DIAG-02 causal inference and training GO are explicitly superseded. |
| 2026-11-02 | Test whether DIAG-02 causal interpretation and CTC evidence support alignment-aware training. | DIAG-02B report identifies the historical forced-attention intervention as invalid: it averaged heads, substituted uniform weights, and multiplied raw hidden states, bypassing learned attention projections. It also finds matched CTC blank AUC about `0.653`, not reliable omission detection. | **Current scientific precedence:** no causal training authorization or gold-window inference clipping from DIAG-02 evidence. |
| 2026-11-02 | Reconcile numerical, provenance, and parity drift. | DIAG-02B reports 9/80 archived score mismatches, max `0.001304954`; no full forward replay. It corrects, among other values, the omission masking macro to `0.001720` (not `0.001845`) and omission enrichment to `1.3966×` (not `0.00×`). | Corrections supersede stale prose; parity remains unresolved. |
| 2026-11-02 | Recover delivery/parity evidence without overwriting prior attempts. | Review branch `00c684408f377300b84aacbec1483b8a2e812967` exists; `evaluation/`, `evaluation-continuation/`, and `evaluation-final/` retain separate attempts. Final report labels continuation as not a PASS. | Recovery worktree is clean but the scientific report remains explicitly partial; no evidence that stale registry statements were rewritten in every historical file. |
| 2026-11-03, before 17:00 | Prepare alternative RQ1 methods (P-SCORE/P-SUP) under frozen evaluation contract and execution holds. | Current index and `P_SUP_AUDIT_AND_METRIC_RECONCILIATION.md` describe frozen criteria, prospective thresholds, and experimental holds. They also report earlier synthetic/runtime QA; not every claim was re-executed in this investigation. | Preparation criteria are not authorization to execute. |
| 2026-11-03, 17:51 | Stage parallel P-SUP remote T4 kernels. | Reflog commit `6ee11a20919a9d9270657d0d82edfdbddbffa8b1`; index says kernels staged via `--no-run`, pinned to `df55f125f5e0e26211986dae32a2dac78c364aaa` and remote execution enforced. This is a local/indexed historical claim, not live Kaggle state. | Later committed notebook content contradicts the stated pin. |
| 2026-11-03, 18:13 | Correct deployment source pin and enforce remote execution. | Commit `0e6e3807d3c31923c0d338d921de204a2cd4eb88` is titled “pin Kaggle T4 kernels to immutable P-SUP SHA df55f12.” | The present notebook source does **not** contain that SHA, so the title/index claim does not describe current notebook contents. |
| 2026-11-03, 20:23 | Fix Kaggle archive discovery/mounting. | Commit `e20f236a98da1bae91b88063ec58f8844bc87700` records a deployment correction. | Current branch includes this history; no Kaggle execution verified here. |
| 2026-11-03, 22:31 | Fix trainer imports and pin kernels to newer P-SUP arm commit. | Current HEAD `1fa4a5b8...` title says “pin Kaggle kernels to dd2dd89”; actual `deploy/kaggle_psup_control_train/kaggle_train.ipynb` and treatment notebook both declare `PINNED_COMMIT_SHA = "dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f"`. P-SUP worktree HEAD equals that SHA. | **VERIFIED local source:** current train notebooks target dd2dd89, not the earlier df55f12. Remote kernels’ current contents/version remain UNKNOWN. |

## 4. DIAG-02 → DIAG-02B: scientific contradiction and precedence

- **Historical DIAG-02 claim (index and `manifests/RQ1-DIAG-02/checkpoints/01-temporal-alignment-root-cause-validation.md`):** forced attention caused approximately `+0.26` score drift, indicating downstream MLP OOD breakdown; local acoustic masking barely changed scores; CTC blank evidence on omission frames supported alignment-aware joint-training GO.
- **DIAG-02B audit (`SCIENTIFIC_REPORT.md`, sections 1, 3–5):** the forced-attention implementation was not the original MHA under an attention mask. It bypassed learned value/output projections and head semantics; the replayed `+0.25515` is therefore not causal evidence of OOD breakdown.
- Corrected matched CTC evidence is modest: blank-posterior ROC-AUC `0.653` (reported exact estimate `0.652663`), with diagnostic TPR `0.242` at selected in-sample FPR `0.136`; not “>84% omission confidence.”
- Projection-correct local-vs-random interventions show sensitivity, but do not establish that naturally omitted phonemes cause attention routing failure, that alignment training will help, or that gold-window inference clipping is safe.
- Historical score parity fails in 9/80 cases at the unchanged tolerance; a fresh-process replay reproduces saved logits on the bounded sample, but does not resolve cross-backend archived-score drift.
- The report also corrects duration-bin, masking, enrichment, attention-offset, and hash-ledger discrepancies. Examples include raw vs rounded `<40 ms` counts `293` vs `284`, and `500/629` exact endpoint matches. These do not by themselves validate acoustic absence or causal interpretations.
- **Precedence by claim type:** DIAG-02B’s explicit correction ledger and final primary report control the *current causal interpretation*; raw final evidence files control the *specific measured quantities*; DIAG-02 remains historical provenance, not current authorization. The global index’s older DIAG-02 “GO” bullet is retained as history but immediately followed by a historical-status correction and DIAG-02B NO-GO. Do not treat the older wording or passed legacy tests as proof.
- **Current registry caution:** the current main index contains corrected status language, but historical reports/index bullets still expose superseded GO claims. Recovery branch delivery is not equivalent to global cleanup. Preserve the history; do not silently rewrite it.

## 5. P-SUP kernel pin contradiction

**Direct primary source observation:** grep of the two current train notebooks finds:

- `deploy/kaggle_psup_control_train/kaggle_train.ipynb:111`: `PINNED_COMMIT_SHA = "dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f"`.
- `deploy/kaggle_psup_treatment_train/kaggle_train.ipynb:111`: same SHA.
- Local P-SUP worktree `HEAD` is `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f`.
- Earlier infrastructure/index claim names `df55f125f5e0e26211986dae32a2dac78c364aaa`. Current notebooks therefore embody a later pin than that earlier claim. It may be an intentional update, but this read-only evidence cannot establish whether both kernel packages were redeployed after the change or whether any remote version executes this source.
- Current deployment notebook content is a higher-precedence source for what the local notebook *would pin* than an older index summary or commit title. It is **not** evidence of Kaggle’s deployed kernel version.

## 6. Architecture / intent evolution

1. Early speech pipeline commits build manifest-driven corpus parsing, alignment, audio loading, and two-head CTC plus binary MDD training.
2. Fold-1 Baseline R and protocol/evaluation work establish an RQ1 speech-diagnosis baseline; policy restricts test partitions and gold TextGrid timing to evaluation labels, not inference inputs.
3. Fold-1 acoustic-conditioning, cross-fold Baseline R, and omission diagnostics explore limitations of binary error ranking, especially deletions/omissions.
4. DIAG-02 proposes temporal attention as an explanation and recommends alignment regularization; DIAG-02B retracts causal certainty while keeping the research hypothesis open.
5. Later RQ1 prep explores segmentation-free GOP alternatives, P-SCORE, and supervised P-SUP objectives with prospective evaluation gates. Shared base and isolated arm worktrees are intended to prevent arm contamination; current commits and source pins still require provenance checks.
6. These are RQ1 speech threads. `AGENTS.md` demarcates speech/cross-corpus work as RQ1 and reserves RQ2 for multi-agent test generation; no evidence here changes that scope.

## 7. Latest actual state, commitments, and boundaries

- **VERIFIED local source state:** Speech HEAD is `1fa4a5b8...`; `prep/rq1-shared-base` has the listed dirt. Nested model/code repo is at detached `baa4989...` with 16 dirty paths.
- **VERIFIED documentation state:** Current index says DIAG-02 is historical and superseded; DIAG-02B is partial/no-go; RQ1 alternative experiments remain gated/held in the cited index entries.
- **VERIFIED P-SUP source state:** local notebook pins are dd2dd89; arm worktree HEAD is dd2dd89. This is distinct from the earlier df55f12 documentation.
- **NOT VERIFIED current remote state:** GitHub branch/default/visibility/PR/issue/release/tip metadata; Kaggle deployment version or run status; Hugging Face artifacts or current access; any post-commit execution.
- **Historical commitments, not independently re-attested:** keep test partitions sealed, no local training fallback, remote-GPU-only execution, per-epoch durable checkpointing, explicit operator authorization, preserve dirty worktrees and historical reports.
- Public-origin privacy: the configured origin URL is the public GitHub-style URL `https://github.com/minkhoaa/Langfens-RQ1-Speech-Research.git`, but visibility was not queried. The speech repo carries extensive inherited research manifests and links/identifiers for data, storage, and private artifact locations. Whether any inherited-data metadata is exposed on a publicly visible origin is therefore a **privacy risk requiring review**, not a finding that the repository is currently public or that restricted audio/labels/credentials were observed. No credentials or personal records were inspected or reproduced.

## 8. Claim/evidence registry

| Claim | Evidence | Confidence / limit |
|---|---|---|
| DIAG-02 causal “proof” is invalidated | DIAG-02B report’s intervention audit and correction ledger | **VERIFIED** as the report’s documented code/evidence audit; no independent rerun here. |
| DIAG-02B does not authorize alignment training | DIAG-02B report, process checkpoint, current index | **CORROBORATED**; scientific NO-GO, not a claim alignment can never work. |
| Archive parity is fully resolved | Final report says 9/80 fail unchanged tolerance; no full-corpus replay | **FALSE** for current recorded result. |
| Recovery worktree repaired parity | Clean worktree at delivery commit; report still says partial | **NOT ESTABLISHED**; cleanliness means no local uncommitted diff, not scientific repair. |
| Current local P-SUP train notebook pin is df55f12 | Current notebook lines 111 and P-SUP HEAD | **FALSE**; both observed notebooks pin dd2dd89. |
| Kaggle now runs dd2dd89 | No permitted live query performed | **UNKNOWN**. |
| GitHub origin is publicly visible | URL has GitHub form, API not queried | **UNKNOWN**. |
| Private datasets/checkpoints were leaked | No such exposure tested or observed | **UNKNOWN**; do not infer leak from metadata references. |
| All historical training/index metrics are independently verified now | Index contains reports but this audit did not revalidate remote artifacts or executions | **NOT ESTABLISHED**. |
| Dirty source files are safe to discard | Dirt includes zero-byte script diffs and deleted papers; untouched | **UNKNOWN**; preserve. |

## 9. Unresolved questions and exact safe restart

1. **Parity/provenance:** What source/runtime/backend difference explains the 9/80 DIAG-02B archive score mismatches? The report’s saved-logit replay is bounded and does not answer it. Safe continuation: read `manifests/RQ1-DIAG-02B/PROCESS_CHECKPOINT.md`, `SCIENTIFIC_REPORT.md`, and `reconciliation-final/CORRECTIONS.json`; verify all hashes and whitelist constraints first; use new output directories only. No TEST partition access.
2. **Registry reconciliation:** Which historical index/report locations retain obsolete DIAG-02 “GO” language? Corrected current index and final DIAG-02B report should remain precedence anchors; any future changes should append an erratum with exact path/commit, not rewrite history.
3. **P-SUP deployment provenance:** Were remote Kaggle packages actually restaged after notebooks changed from df55f12 to dd2dd89, and which immutable source SHA is deployed now? Do not infer from local metadata or commit subject. This audit did not use Kaggle API.
4. **Public-origin data/privacy:** What is the live GitHub visibility and what inherited manifests, remote IDs, or restricted-data metadata are public? Requires an approved read-only remote metadata review; none was done here. Do not expose credentials or attempt mutation.
5. **Reachability and worktree continuity:** Do the four expected staged files in the untracked `context-reconstruction-2026-11-03/` directory and modified index represent a separate in-progress task? They were not read because they may contain pre-existing evidence outside this seam. Keep intact.
6. **Safe restart point:** Start from observed speech HEAD `1fa4a5b8de206bbf0bd7cb7856f721d2b1d0adb6` and record fresh time/ref/status; re-read `AGENTS.md` and index; inspect all dirty paths before any action. For DIAG-02B use review commit `00c684408f377300b84aacbec1483b8a2e812967` and its own report/checkpoint; for P-SUP use current arm HEAD `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f` and verify deployment provenance separately. No edits, resets, checkouts, fetches, pushes, remote writes, test runs, inference, or training are authorized by this audit.

## 10. Manual notes and residual risks

- No files, refs, indexes, datasets, services, remote state, or credentials were modified. No tests or commands requiring model execution were run.
- Highest-impact unresolved risks: failed DIAG-02B archive parity; historical stale GO claims; local notebook pin versus unverified deployed Kaggle version; unknown live GitHub visibility with inherited research metadata; pre-existing nested repo dirt/deleted reference PDFs.
- Findings apply only to the observed local refs and files at discovery time; refs may move.