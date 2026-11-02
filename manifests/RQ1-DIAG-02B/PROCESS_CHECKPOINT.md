# RQ1-DIAG-02B — Process checkpoint

**State:** `PARTIAL_ARCHIVE_PARITY_FAILED; SCIENTIFIC_REPORT_DELIVERED; NO_GO_TRAINING_CAUSAL_GATE`.

**Authorization:** CPU-first bounded inference, read-only existing checkpoints, validation only; commit/push review branch authorized by operator. No training authorization. STOP after report/recommendation.

## Verified starting state and source

- Clean starting worktree on `feat/rq1-diag-02-temporal-alignment-validation`, HEAD `dd2dee9a2063284d2a0f9fbf1c18315f3a7cd774`.
- Created `feat/rq1-diag-02b-counterfactual-validity` from that exact HEAD.
- Read `AGENTS.md`, global index, DIAG-01/02 overview and detailed checkpoints, existing diagnostic/model/evaluator sources.
- Model implementation, original checkpoints, manifest, registered training protocols and original numerical report files are unchanged. Only appended historical Markdown supersession notices affect previous phases.
- Execution occurred with new uncommitted source on base HEAD; reports correctly record base HEAD plus **actual executed script SHA-256**, not a fictitious implementation commit. Final source digest: `7d43173ef4bbade93aec017f10d27615d33eadc5f2946332475940e6ea6e1424`.

## Actual inputs and outputs

- Frozen split manifest: `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`.
- Original prediction archive: `355b6e23bcb172d20b99ba3f9f4e07dc15453846ac312eca847471ad476e5892`.
- All four original 3,600-update Baseline R checkpoint digests and configurations verified, strict state loading, frozen eval backbone; pretrained revision preserved. All original `skipped_amp_attempts_total` counters are **0**. Full endpoint paths/hashes/configs are in `evaluation-final/REPORT.json`.
- All four authorized train/val package digests verified against registered digests (recorded in JSON). Only whitelisted validation audio/annotation members read; per-fold TEST metadata used only as an exclusion guard.
- Reproduced census/archived AP on 1,199 utterances/39,576 tokens. New inference restricted to 80 omission-enriched utterances/2,951 active tokens/eight speakers. Matched CTC: 66 pairs (41 sampled omissions unmatched); 10-ms caliper: 58 pairs. Waveform audit: 16 pairs/32 tokens, 64 masked encoder forwards.
- Main scientific report: [`SCIENTIFIC_REPORT.md`](SCIENTIFIC_REPORT.md).
- Canonical JSON: [`evaluation-final/REPORT.json`](evaluation-final/REPORT.json), SHA-256 `38aa543a46d5c2da856e0d73f1887a12285576a205c66fc41c0faa194f8647b6`.
- Raw compressed CSV records and aggregate/speaker CSV/JSON reports with per-file hash ledgers: `evaluation-final/`.
- Corrections, historical hash audit, independent endpoint check and fresh-process replay: `reconciliation-final/`.
- Earlier interrupted/preliminary artifacts are intentionally preserved in `evaluation/`, `evaluation-continuation/`, `reconciliation/`; use `*-final/` as canonical, not whichever directory was created first.

## Execution and verification evidence

1. Initial test collection failed on an unclosed dictionary; repaired before inference. The seven executable inference/matching regressions then passed. No backward/optimizer tests were run.
2. First inference interrupted on archived probability error **0.000313178 > 0.000200**; recorded in `FAILED_ATTEMPT.md`. The first `RUN_STATE.json` is a historical start marker, **not an active process/completion claim**.
3. Explicit `--continue-on-archive-drift` continuation, four CPU threads, completed in **102.2 s**. Kept as preliminary; maximum archive discrepancy **0.002100974**.
4. Canonical one-thread deterministic/MKL-compatible run completed in **489.6 s**. Repeated forwards, diagnostic head and all-open masks have **zero logit difference** vs `model.forward()`. Archived scores fail the unchanged tolerance on **9/80 utterances**, maximum **0.001304954**; hence no PASS.
5. Fresh independent process reproduces all 80 saved baseline logits with **maximum error 0**. Scalar/vector sigmoid probability differences ≤**5.96e−8** are reported separately. The cause of CUDA archive drift is not conclusively established.
6. Independent per-head Q/K/V/output-projection tests verify masked MHA semantics, padding/length handling and rejection of fully masked windows. Regression tests recompute matched CTC metrics and intervention AP from raw records and verify executed source/artifact SHA-256.
7. Executed `PYTHONPATH=.:src HF_HUB_OFFLINE=1 .venv/bin/pytest -q tests/test_rq1_diag_*.py`: **31 passed**. Full repository suite **not run** because unrelated tests can invoke prohibited backward/optimizer behavior.
8. Scientific corrections: 293 raw vs 284 rounded <40-ms omissions, nine IDs saved; 545 vs 542 <80 ms; historical speaker macro masking **0.001720** not .001845; omission enrichment **1.3966×** not 0.00×. Seven of eight original DIAG-02 artifact hash claims are stale; actual files are byte-identical to the base commit. No original scientific JSON/CSV is overwritten.

## Research state and model artifacts

- Diagnostic successful optimizer updates: **0**; backward passes: **0**; skipped diagnostic updates: **0 / not applicable (no updates attempted)**.
- Last valid model artifacts: unchanged four original 3,600-successful-update endpoints, all original skipped-update counters zero. No new model artifact/checkpoint exists or is needed.
- Numerical reports and process Markdown are not PyTorch training-state checkpoints.
- Evidence: modest CTC matched discrimination (blank AUC **0.652663**), not >84% omission confidence. Corrected attention interventions expose sensitivity, not natural-omission causation or a validated alignment-training benefit. Feature and waveform masking differ and are not interchangeable.
- **Scientific recommendation:** NO-GO for causal training authorization or deployment clipping on present evidence. Broader alignment hypothesis remains open. No new architecture, loss, backward pass, optimizer step or training launch implemented/executed.

## Safe restart / handoff

1. Verify branch/implementation commit with `git log -1` and `git status --short`; source execution base remains `dd2dee9...`. Inspect any delivery receipt for pushed commit identity.
2. Read this checkpoint, the scientific report and `reconciliation-final/CORRECTIONS.json`. Check all `ARTIFACT_SHA256.json` entries against their files and source SHA against the current diagnostic script. Preserve existing directories.
3. Confirm the same four checkpoint, manifest/prediction and authorized-package hashes; reapply validation whitelists before reading annotation/audio members. Do not download/extract sealed TEST partitions.
4. For a CPU replay, use the exact commands in the report with **new output paths**. Without the explicit drift-continuation flag the unchanged parity tolerance fails closed. Never reinterpret the flag as acceptance.
5. Resolve archived-score parity/provenance before reporting acceptance; an unchanged small aggregate AP does not waive pointwise failures. G=8 and case-control/selection limitations remain.
6. **Stop here.** Further training/proposal implementation requires separate explicit authorization and remote-GPU compliance; never fall back to local training.

## Delivery tracking

This checkpoint is committed with the source/tests/reports on the dedicated feature branch. Commit and push verification is recorded in `DELIVERY.md` when available; do not infer successful push from this sentence alone.
