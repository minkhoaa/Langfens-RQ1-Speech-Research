# SEAM 5 — Kaggle, Hugging Face, datasets, and experiment provenance

## 1. Scope, source inventory, and confidence

**Discovery time:** 2026-11-03 23:42 +07:00, from the host clock.  
**Repository observed:** `/home/khoa/Projects/KLTN/Dataset`, HEAD `1fa4a5b8de206bbf0bd7cb7856f721d2b1d0adb6`, branch `prep/rq1-shared-base`. **It was dirty** at discovery (`M docs/process-checkpoints/INDEX.md`, modified nested `repo`, untracked `docs/process-checkpoints/context-reconstruction-2026-11-03/`). All inspected evidence below is from the read-only worktree state; no files or refs were changed.

**Required reading:** root `AGENTS.md` and `docs/process-checkpoints/INDEX.md`. The index’s latest RQ1 summary records P-SCORE metrics and P-SUP as staged/awaiting operator execution, but a current read-only Kaggle status query returned `COMPLETE` for the two P-SUP kernels. That inconsistency is material; see §6. The root index itself is dirty, so its latest claims are not assumed to be committed or independently verified by this inspection.

| Source | Inspected material | Evidence value / limitation |
|---|---|---|
| Root process registry | `docs/process-checkpoints/INDEX.md` | Broad phase history and current claims; secondary to underlying artifacts/logs. |
| Baseline R Fold 1 | `manifests/RQ1-TRN-02A/PROCESS_CHECKPOINT.md`, `TRAINING_HISTORY.json`, `EVALUATION_REPORT.json`; detailed results summarized in INDEX and later reports | Historical evidence; no checkpoint loading performed. |
| Baseline R Folds 2–4 | `manifests/RQ1-TRN-03A/checkpoints/12-official-fold4-baseline-r-training-closure.md`, `13-four-fold-validation-synthesis-and-closure.md`; local fold 2/3 Kaggle logs | Fold 3 log directly inspected; fold 2 log present but the targeted metric grep returned no matching lines. Fold 4 report is a contemporaneous recorded account, not a fresh artifact readback here. |
| Baseline R cross-fold evaluation | `manifests/RQ1-TRN-03A/evaluation/FOUR_FOLD_VALIDATION_REPORT.json`, `four_fold_metrics_summary.csv`, paired prediction archive (not opened) | Reported aggregate results; no sealed test materials accessed. |
| C0/C1 acoustic ablation | `manifests/RQ1-TRN-02B/PAIRED_SCIENTIFIC_EVALUATION.md`, `evaluation/PAIRED_EVALUATION_REPORT.json` | Detailed historical report; not fresh rerun or weight verification. |
| E1 | `manifests/RQ-MODEL-01/evaluation/E1_FOLD1_BENCHMARK_SUMMARY.json` and `manifests/RQ1-WORKSPACE-PREP-01/SCIENTIFIC_QA_AND_COMPARISON_REPORT.md` | Historical benchmark and synthesis; no inference performed. |
| P-SCORE / P-SUP | `manifests/RQ1-WORKSPACE-PREP-01/SCIENTIFIC_QA_AND_COMPARISON_REPORT.md`, `P_SUP_AUDIT_AND_METRIC_RECONCILIATION.md`, `INDEX.md` | P-SCORE outcomes reported; P-SUP execution status conflicts with live remote status. |
| Dataset and split provenance | `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`, Fold 1–3 authorized train/validation archive paths, `AGENTS.md` | Frozen split digest and stated sealing constraints. No archives extracted or audio/labels opened. |
| Remote state | Read-only `kaggle kernels status` for P-SUP treatment/control and Fold 2/3 Baseline R kernels; anonymous Hugging Face API GET | Kaggle returned `KernelWorkerStatus.COMPLETE` for all four. Anonymous HF GET returned no model metadata (`id/sha/private` absent); private repo inaccessible without authentication. |
| Other repositories | Requested backend and frontend starting HEAD/status were checked for context, not used as experiment evidence | Backend `2ba12e5488ec81dfbee97cf74fdf31dd9609eb48`, branch `generation-harness`, dirty; frontend `2d9fe94a54e65680eca56aafbe40e691018113da`, branch `autonomous-generation`, dirty. They do not establish training or storage provenance. |
| Inaccessible/not examined | Private HF file metadata and kernel outputs for P-SUP; Hugging Face private model history; Google Drive contents; Kaggle attached datasets/runtime files; fold 2 detailed local run log; all weight payloads | No authenticated HF/API metadata or outputs were available through this read-only session. No weights/datasets downloaded. |

**Confidence vocabulary:** VERIFIED = directly observed in this session or directly inspected primary local record; CORROBORATED = multiple consistent records; INFERRED = interpretation; UNKNOWN = unavailable or not tested. A document’s claim that an earlier run was “verified” is not equivalent to fresh verification in this session.

## 2. Dated experiment and decision timeline

Event times below are normalized to Asia/Ho_Chi_Minh where the source records a time. “Recorded” is the document’s record time, not necessarily the actual event time.

| Event time / record time | Goal; decision and rationale | Action, reported result, alternative or limitation | Current validity |
|---|---|---|---|
| 2026-10-29 03:00 +07:00, report timestamp | Test whether prompt embedding dropout improves acoustic reliance and MDD; compare matched C0/C1 retraining from Reference R. | Historical report says Kaggle T4×2 training, each 3,600 successful updates, 0 AMP skips; C1−C0 Binary AP −0.007918, acoustic-gap difference −0.006120; hypothesis “NOT_SUPPORTED.” It also reports Reference R CPU replay AP 0.317021 versus original online AP 0.316062. No fresh run in this task. | **CORROBORATED as documentary history**; no present authorization implied. |
| 2026-10-31 22:00 +07:00, Fold 4 closure record | Finish Baseline R independent Fold 4 training with fixed-budget and validation-selected endpoints. | Report states Kaggle dual T4 FP16 AMP; 4,800 total updates, zero skipped AMP updates; fixed 3,600-update and best-at-1,800-update checkpoints; early stop at epoch 8. Fold 4 fixed-budget checkpoint SHA-256 `c502271feba097781a6b0bdfe413085271d3a03647353f272dccd29bbb1499f6`; best SHA-256 `3daa8ca0db8a632050e3f965c705df2d271d07a450cb7110a9780537d623274d`. This report references Kaggle logs and private HF readback, not freshly queried here. | **CORROBORATED by the fold-closure record and cross-fold report**; fresh remote file presence/hash **UNKNOWN**. |
| 2026-10-31 22:30 +07:00, four-fold synthesis record | Establish the cross-fold validation baseline and report fixed exposure separately from validation-selected checkpoints. | Report gives fixed-budget macro AP 0.279207, micro AP 0.272580 over 39,576 tokens; best-checkpoint macro AP 0.289351, micro AP 0.287469. It reports macro CTC PER 23.90%; macro acoustic gap +0.013064, with 4-fold df=3 and wide uncertainty. It explicitly warns against interpreting eight-speaker/four-fold results as broad cross-corpus proof. | **CORROBORATED as recorded evaluation**, not an independent recomputation here. |
| 2026-11-01 22:30 +07:00, Gate C closure record | Screen loss-only RQ1-TRN-03B pilots before spending compute on cross-fold arms. | Fold 1 arm contrasts were negative; formal paired permutation tests reported p=0.4790 and p=0.6190 for two treatment arms. Decision: halt Stage 2 for compute efficiency; four-fold candidate criteria remain unmeasured. Record states no additional GPU training in that closure. | **CORROBORATED as documented decision**. It does not authorize later experiments. |
| 2026-11-03, current registry and report | Prepare P-SCORE and P-SUP under a frozen evaluation contract; historical baseline and candidate metrics must remain distinct. | `SCIENTIFIC_QA_AND_COMPARISON_REPORT.md` says P-SCORE completed real-data CPU inference and lists results; it says P-SUP remote training was not started, 0 updates/no artifact, while INDEX describes P-SUP as staged and awaiting Web UI launch. | P-SCORE reported results are supported by registry/report, but fresh artifacts not re-read in this pass. P-SUP operational status conflicts with live Kaggle status below. |
| 2026-11-03 23:42 +07:00, discovery | Determine current status without disrupting jobs. | Read-only Kaggle status returned `COMPLETE` for `minkhoaa/langfens-rq1-psup-t-op-t4x2`, `...-psup-b-control-t4x2`, `...-fold-2-baseline-r-t4x2`, and `...-fold-3-baseline-r-t4x2`. No kernel was stopped, triggered, or queried for output. | **VERIFIED current status label**; whether a complete status means successful contracted training and durable artifacts is **UNKNOWN** without output/report metadata. |

## 3. Experiment lineage and meaning of reported metrics

### 3.1 Data and split boundaries

- The frozen split manifest is `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`, digest recorded as `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`.
- Root `AGENTS.md` states Fold 1 has 2,399 train, 300 validation, and 900 sealed test utterances. It requires folds 2–4 to be independent Baseline R training from the original pretrained initialization—not warm-started from Fold 1—and prohibits staging active test data.
- Fold 1 authorized train/validation archive SHA-256 is repeatedly recorded as `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`.
- Fold 2 and fold 3 authorized archive files are present locally (`datasets/L2-ARCTIC/authorized_RQ1-TRN_fold2_train_val_only.zip` and `...fold3...`). Their archive contents were not inspected. No fold 4 archive was inventoried in this task.
- **VERIFIED boundary for this session:** no sealed test audio or labels were opened, no archives extracted, and no weights loaded. Historical assertions of zero leakage are documentary evidence, not a new end-to-end isolation audit.

### 3.2 R / E0 — reference Baseline R

- Architecture and contract: frozen `facebook/wav2vec2-base-960h`, revision `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`; two trained heads/objectives CTC + binary MDD. Root contract prohibits using gold TextGrid timestamps as model inference inputs.
- Fold 1 reported production `best.pt`: epoch 6, 3,600 updates, SHA-256 `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`. Training continued to 6,600 updates/epoch 11; that final `last.pt` is a distinct training endpoint.
- Fold 1 AP is evaluator/precision dependent, not one interchangeable number: online Colab FP16 AMP 0.316062; CPU FP32 Protocol O replay 0.317021; cross-fold evaluator replay 0.317012; Protocol Z aligned-only 0.313530. The report attributes differences to precision and evaluator details; these figures must not be averaged or substituted.
- **Protocol O** uses gold-aligned token correspondence for benchmark evaluation. **Protocol Z** feeds audio and written text converted through G2P to the model; gold alignment may be used offline to compare tokens, not as model input. Protocol Z coverage is reported as 9,919/9,956 (99.63%).
- Cross-fold synthesis reports 4 folds, 8 validation speakers, 39,576 tokens. Fixed-budget AP macro 0.279207 versus micro 0.272580. Best checkpoint AP macro 0.289351 versus micro 0.287469. Fold-level generalization is heterogeneous; fold 2 is notably lower. Results concern validation cohorts, not sealed test or cross-corpus generalization.
- Training contracts prescribe both exactly 3,600-update fixed-budget checkpoint and validation-selected `best.pt`, because best checkpoint exposure differs by fold. Fold 4 record reports fixed endpoint 3,600 but best at 1,800; Fold 1 best at 3,600; fold 3 best at 3,600 per cross-fold report.

### 3.3 C0/C1 — prompt dropout acoustic conditioning

- C0 is p=0 control; C1 is whole-phone prompt embedding dropout p=0.25. The paired report says both trained from the same Reference R initialization, 3,600 updates, seed 1337, zero AMP skips, and used identical evaluation.
- Fold 1 Protocol O matched Binary AP: R 0.317021, C0 0.310304, C1 0.302386. Primary within-retraining contrast C1−C0 = −0.007918. Reported speaker-stratified bootstrap interval [−0.018328, +0.000385].
- Acoustic reliance gap (matched minus within-speaker cyclic-shift mismatched AP): R +0.022299, C0 +0.017543, C1 +0.011423. C1−C0 gap difference −0.006120; interval [−0.019675, +0.005596].
- Protocol Z reports 9,919 aligned tokens: R 0.313530, C0 0.307582, C1 0.300992. The report concludes the p=.25 hypothesis is not supported.
- These are one validation fold / two speakers; they do not establish four-fold effects. Their reported use of CTC PER as a guardrail does not change the MDD AP conclusion.

### 3.4 E1 — alignment-aware skip model

- E1 is a prior alignment-aware skip/diagnosis mechanism, distinct from Baseline R, P-SCORE, and P-SUP. The workspace synthesis reports E1 omission AP 0.032252, binary AP 0.309293, greedy CTC PER 0.232365, and skip mass exactly 0 on omissions.
- Reported E1 versus E0 changes: Binary AP −0.007728; omission AP −0.002768 (−7.90%); recall at 15% FPR −0.034483; CTC PER improves by 0.004453. The workspace conclusion is E1 rollout NO-GO because the target skip mass collapsed.
- These are historical Fold 1 validation benchmark claims and do not support current deployment or training authorization.

### 3.5 P-SCORE — segmentation-free CTC scoring

- The registry reports P-SCORE was executed using frozen E0 emissions for Fold 1 VAL (300 utterances; speakers NJS/RRBI), generated on CPU, cached FP32 emissions, and emitted 39,824 tokens. It reports cache SHA-256 `a699fd9378c86cd348a0628a144ecacfef491ea8f90e0edd23b4e46932e3e829`, E0 checkpoint SHA-256 `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`, and zero errors under the strict runtime validator.
- Primary reported contrast is `GOP-CTC-SF-SD minus GOP-CTC-SF-S` on 9,919 aligned tokens: omission AP 0.051026 vs 0.029411 (delta +0.021615); recall@FPR15 delta +0.182482; substitution AP 0.222544 vs 0.246800 (delta −0.024256); binary AP delta −0.009045. Greedy CTC PER 0.236818 reportedly replicates E0.
- Full canonical population N=9,956 and aligned population N=9,919 differ by 37 unaligned tokens (reported as 25 correct, 4 substitutions, 8 deletions). Omission, substitution, and binary AP use different denominators; the report defines omission AP on 8,944 and substitution AP on 9,811 for legacy subtype scoring. These denominators must not be conflated.
- P-SCORE represents a scoring/inference study over frozen emissions, not a new trained model and not proof of causal acoustic diagnosis. Its omission gain coexists with lower substitution and binary AP; the acceptance report itself records failure of substitution noninferiority.

### 3.6 P-SUP — supervised operation objective

- P-SUP is a separate training comparison: operation-level treatment (`operation_ce`) versus matched binary control (`binary_collapsed`), not the P-SCORE frozen-emission experiment. The registry records 266,667 head parameters per arm and planned 3,600-update budgets, seed 1337, dual checkpoint reporting, and FIT/SELECT/CAL/VAL partitions.
- The INDEX records staging to version 2 of two private kernels, but the live Kaggle status today is `COMPLETE` for both. A kernel’s completion status alone does **not** establish successful updates, exact budget, skip count, selected/fixed checkpoint identity, authorization, or HF mirroring. No output was queried.
- Current metrics, checkpoint hashes, experiment IDs, successful/skipped updates, and artifact presence are **UNKNOWN** to this investigation. The registry’s prior “awaiting Web UI” / “NOT_STARTED” statements must be treated as stale or unresolved until remote logs and HF artifact metadata are reconciled.

## 4. Actual observed remote state and limits

| Kernel / resource | Read-only observation at 2026-11-03 23:42 +07:00 | What it establishes / does not establish |
|---|---|---|
| `minkhoaa/langfens-rq1-psup-t-op-t4x2` | `KernelWorkerStatus.COMPLETE` | Establishes current Kaggle status label. Not proof of successful P-SUP contract completion. |
| `minkhoaa/langfens-rq1-psup-b-control-t4x2` | `KernelWorkerStatus.COMPLETE` | Same limitation. |
| `minkhoaa/langfens-rq1-fold-2-baseline-r-t4x2` | `KernelWorkerStatus.COMPLETE` | Operationally complete; underlying run report not obtained in this session. |
| `minkhoaa/langfens-rq1-fold-3-baseline-r-t4x2` | `KernelWorkerStatus.COMPLETE` | Same; local log provides additional Fold 3 evidence below. |
| HF model repo `minkhoaa/langfens-rq1-cross-fold-checkpoints` | Anonymous `GET https://huggingface.co/api/models/minkhoaa/langfens-rq1-cross-fold-checkpoints` did not return model metadata; parsed `id`, `sha`, `private` were absent and sibling count zero | Private repository is not verifiable anonymously. No credentials were read or printed. Does not prove repository absence. |
| Local Fold 3 Kaggle log `langfens-rq1-fold-3-baseline-r-t4x2.log` | Directly inspected machine log records | See precise output below. |
| Local Fold 2 log `langfens-rq1-fold-2-baseline-r-t4x2.log` | File exists, but grep for ordinary metric/status strings returned no matches | Likely structured/encoded format or different field representation; exact run result remains unverified here. |

Fold 3 log directly records: 600 updates/epoch, 3,600 fixed-budget updates at epoch 6, 0 skipped AMP attempts; validation AP peaked at 0.307479 at epoch 6; later run continued to 6,000 updates and terminated with `COMPLETED_3600_UPDATES`. Endpoint report in log gives fixed-budget SHA-256 `bbee9dd392806fc81c88e41018edc7adb270d71b5047930f5ce78a69c538a441`, best SHA-256 `679f15c545384b90d003d9184d42ab945c7c7adcc67029dfdef7f79c93f25eea`, last SHA-256 `804cbfa0d03280fa8b988400b254dbfe79e08c7a6b23449379acdb63490b866e`, and history SHA-256 `33174b40c12720d5c48c5b4cce220372b48fadc833172d4901c4056fb3171fcc`. This log is a local copy, not a fresh HF readback.

## 5. Claim/evidence registry

| Claim | Evidence | Confidence / qualification |
|---|---|---|
| P-SUP kernels now report complete | Four current Kaggle CLI status responses, two matching P-SUP slugs | **VERIFIED** status label only. |
| P-SUP has completed prescribed training | No remote output/report retrieved; INDEX says staged/awaiting UI | **UNKNOWN**. Do not report training success or failure. |
| Fold 3 Baseline R reached 3,600 fixed-budget updates, 0 AMP skips | Local log contains epoch 6 record and endpoint classification | **VERIFIED** local log contents; authentic remote origin is **CORROBORATED** by recorded Kaggle kernel and checkpoint report; no fresh remote artifact hash readback. |
| Fold 2 Baseline R successful update count and metrics | Kernel status complete, local log exists but search found no expected strings | **UNKNOWN** in this inspection. |
| Fold 4 Baseline R fixed/best endpoints and hashes | Fold 4 closure record and cross-fold report | **CORROBORATED**, not freshly re-read from private HF. |
| Four-fold aggregate fixed-budget AP 0.279207 macro / 0.272580 micro | Cross-fold evaluation report/checkpoint | **CORROBORATED as recorded analysis**; not recomputed here. |
| E0 Fold 1 best.pt checksum and 3,600 update provenance | Multiple reports and registry; no weights opened | **CORROBORATED documentary evidence**. |
| P-SCORE omission gain versus SF-S | Registry and scientific QA report | **CORROBORATED documentary evidence**; not recomputed in this task. Substitution AP and Binary AP declined. |
| P-SUP outcome metrics, artifact SHA, run IDs, skipped updates | No accessible output | **UNKNOWN**. |
| No test data was accessed in historical experiments | Root contract, reports and staging attestations | **CORROBORATED by records**, not independently audited across remote jobs in this task. |
| All experiments are currently authorized | No current authorization observed; root AGENTS explicitly requires operator authorization for training | **Not established**. A remote `COMPLETE` status is not authorization evidence. |

## 6. Contradictions, drift, and risks

1. **P-SUP operational status drift — material:** INDEX and scientific QA say staged/awaiting launch or NOT_STARTED; live Kaggle status says both P-SUP kernels `COMPLETE`. Resolution requires read-only job logs, exact kernel version/source commit, execution timestamps, endpoint classification, updates/skips, and checkpoint metadata. Until reconciled, neither “not started” nor “training completed successfully” is defensible.
2. **Kaggle completion is not evidence of durable checkpoint correctness:** No P-SUP logs, remote HF revision, file list, file hashes, or run metadata were inspected. A completed kernel can exit after failure or incomplete work.
3. **Fold 2 audit gap:** Kaggle reports complete, but expected strings were not found in its local log. No update counts, AP, skipped updates, endpoint hashes, or fresh remote mirror verification can be attested from this task.
4. **Historical summary relies on claims:** Fold 4 and four-fold results are supported by detailed checkpoint reports but not freshly verified against remote HF files. Do not present historical “independently verified” labels as this session’s independent verification.
5. **Protocol/precision drift:** Fold 1 AP 0.316062 / 0.317021 / 0.317012 / 0.313530 arise from different precision/evaluator/protocol/populations. Mixing them would manufacture a false trend.
6. **Population and metric drift:** Full canonical tokens, aligned-only tokens, omission-only and substitution-only denominators are not interchangeable. P-SCORE omission improvement does not establish broad MDD improvement.
7. **Best versus fixed budget:** Validation-selected checkpoints have unequal update exposure. Fold 2 and Fold 4 peaks are reported before 3,600 updates; cross-fold reporting must preserve both endpoint types.
8. **Small generalization sample:** Four folds and eight validation speakers yield weak inference for population-wide claims. Fold 2 performance is substantially lower than other folds; macro/micro summaries conceal cohort variation.
9. **No test-set/cross-corpus claim:** Only validation evidence is described. No sealed test data was touched here. SpeechOcean762 and held-out test evaluation are not established by this seam.
10. **Repository dirt:** All three requested repositories were dirty at observation; this work made no changes. No claim about staged status or file-level authorship is made.

## 7. Architecture and intent evolution

- **Baseline R / E0:** Frozen pretrained acoustic backbone, two-head CTC + binary MDD baseline; reference model and original-initialization requirement for each fold.
- **C0/C1:** Controlled prompt-dropout attempt to alter acoustic reliance. Reported effect was negative on Fold 1, so hypothesis rejected; not a reason to warm-start later folds.
- **E1:** Learned skip/null mechanism for omission detection. Reported target skip mass collapsed to zero on deletions; rollout held.
- **RQ1-TRN-03A:** Independent cross-fold Baseline R validation, with fixed-budget and best checkpoint accounting; this established a heterogeneous baseline, not deployment readiness.
- **RQ1-TRN-03B:** Loss-only modifications screened on Fold 1, then Gate C halted cross-fold expansion due to no positive pilot signal. Four-fold treatment criteria remain unmeasured.
- **P-SCORE:** Shifted from training a new detector to scoring frozen E0 emission evidence using segmentation-free CTC graph methods. Reported omission gains came with substitution/binary losses; no universal winner.
- **P-SUP:** Separate supervised operation objective versus parameter-matched binary control. It is a training experiment, distinct from P-SCORE. Current remote operational state is complete, while scientific outcome/provenance is unresolved.
- **Current validity:** None of the historical designs grants present training authorization. Root `AGENTS.md` requires operator authorization, remote GPU-only training, private persistent checkpoint mirroring and test isolation.

## 8. Unresolved questions and safe restart

### Unresolved

- What exact Kaggle kernel versions and run timestamps produced the two P-SUP `COMPLETE` states? Did either run execute training, and what were successful/skipped updates?
- Did each P-SUP arm reach the exact fixed budget, produce both fixed-budget and `best.pt`, and persist verified hashes to the private HF repository?
- What are the actual P-SUP metrics on the prespecified populations and thresholds? Were these values inspected before or after the candidate criteria were frozen?
- Does Fold 2’s log encode output as JSON/escaped stream records, and what are its endpoint report/hashes and true successful update count?
- Can private HF API metadata be read under already-authenticated access without exposing credentials? This session did not have access.
- Do remote P-SUP source commits and notebook versions match pinned source commit `df55f125f5e0e26211986dae32a2dac78c364aaa` and the stated Version 2 staging?
- Which Fold 2–4 checkpoint artifacts have verified durable readback, and are both fixed-budget and validation-selected checkpoints present?

### Exact safe restart

1. Preserve all dirty working trees and refs; record timestamp and repository HEAD/status before any continuation. Do not fetch, checkout, or alter files.
2. Use **read-only** Kaggle metadata/status/log and HF API GET only. Do not start, stop, rerun, upload, download weights/datasets, or access credentials. If remote outputs require interactive/auth changes, stop and report blocked.
3. For each P-SUP slug, record kernel version/source commit, execution event timestamps, terminal classification, update counts, AMP skips, `best_val_ap`, fixed/best/last filenames, endpoint report and SHA-256 values. Compare with immutable HF revision/file metadata if accessible.
4. For Fold 2, inspect only the already-local log’s format and endpoint metadata; do not extract archives or access test partitions. Reconcile the log against Kaggle status and existing checkpoint docs.
5. Update conclusions only after source, run, metric population, and durable artifact identity agree. Keep “kernel complete,” “training complete,” “artifact persisted,” and “scientific result valid” as four separate claims.
6. Any new training/inference/model loading, remote launch, or storage mutation requires explicit authorization and the safeguards in `AGENTS.md`.