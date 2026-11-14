# Agent B — Phase-01 joint-policy preflight handoff

## Outcome

**Partial implementation; no fitted/evaluated Phase-01 outcome.** Implemented a deterministic, leakage-guarded S/O/I threshold utility, synthetic regression tests, an independent read-only audit of keyed MB03 target/gap row ordering, and a source/config freeze receipt. No formal training, gap refit, policy threshold fit/application, or outer-label evaluation was launched. Consequently all joint-system P/R/F1, Macro-3 Error F1, AP/FWR and uncertainty values for this lane are **null / NOT_ESTIMABLE**, not zero.

## Implemented and checked

- `joint_policy.py`: frozen .8-target/.2-gap allocation; B is passed as .2/.5/1/2; target false warnings include any wrong predicted S/O type; gap FP means predicted insertion on gold-negative; largest eligible prefix uses descending score then descending first-eight-byte SHA-256 of `LANGFENS-MB07-warning-tie-v1:` + event ID; immutable cutoff applies without outer labels. It validates fit-vs-held speakers, inner prediction speaker provenance, unique aligned IDs, and score range.
- `test_joint_policy.py`: four CPU tests cover wrong-type FPs, deterministic tied-score behavior/cutoff, speaker leakage and invalid inner speaker rejection, and duplicate-ID rejection.
- `verify_keyed_order.py`: independently joined official TRAIN manifest order to the reused MB07 fixed50 target stream and archived MB03 keyed stream. **VERIFIED:** exact 59,562 target and 61,362 N+1 gap event order, metadata and score shapes; keyed MB03 source SHA-256 `8a5692e43f64b08d60252b9f6a8c63d8be7f8c5e6d6dcfa4d0e26ef7e65bb42b`.
- `freeze_receipt.json`: hashes the authoritative spec/config, this lane's source/tests/audit, and read-only candidate prediction artifacts before any new prediction outcome or policy application. Reused artifacts are not modified.

## Validation evidence

Commands, from lane directory, with `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`:

1. `python -m unittest -v test_joint_policy` — **passed**, 4 tests.
2. `python verify_keyed_order.py` — **passed**, status `VERIFIED_ORDER`, 1,800 utterances / 59,562 targets / 61,362 gaps.

The environment has no system `pytest` module; an initial `python -m pytest -q` attempt returned `No module named pytest`. The test suite uses stdlib `unittest` and passes as above. No CUDA was used.

## Exact blockers / unresolved

1. **MB03 gap fit cannot be faithfully replayed from preserved source.** The read-only MB03 tree contains score/evaluation code, not the direct plain gap-head fit producer, its checkpoint, or fitted normalizer. Specification documents historical architecture/feature/scaler/weight/epoch constraints (MB03 PLAIN exact-gap; 5 epochs), but do not give an executable fit provenance sufficient to establish that a newly written producer is the exact historical recipe. Inventing/reconstructing missing optimizer, sampler, preprocessing, or initialization semantics would breach the instruction to recover the exact MB03 procedure. Therefore zero gap fits were attempted; no partial fit receipts or raw new predictions exist.
2. **Eligible reused target inner provenance is incomplete for the required joint stream.** Verified available files include direct inner OOF fold scores and nested CTC/E4 payloads, and fixed50 outer scores for seed101; the exact fixed50/E4/direct streams with speaker and score provenance needed across all inner partitions are not co-located as a verified complete keyed prediction set. Seed202 does not have a standalone fixed50 outer NPZ (the broader fusion artifact exists, but its required inner provenance was not certified here). No guessed/reconstructed target scores were used for thresholds.
3. No outer gap refits were needed or executed; matched direct/E4-only/fixed comparison, frozen inner thresholds, outer policy applications, class metrics, joint FWR, fold/speaker uncertainty and full-system report remain **NOT_TESTED / NOT_ESTIMABLE**.

Historical source details already documented in frozen spec remain: independent gap head 1614→128→64→2; 1614-D context from adjacent H6 edge crops and intended-phone encodings; fit-only gap scaling `sd=max(std(ddof0),1e-4)`; fit-only square-root class weights normalized to mean one; AdamW lr `8e-4`, wd `1e-4`, batch 1024, 5 epochs; phase01 inner seeds `101+10*outer+inner`, outer `101+100+outer`. This is the frozen protocol, **not evidence the legacy producer was recovered or a fit was run**.

## Changed files

Only this lane was written: `joint_policy.py`, `test_joint_policy.py`, `verify_keyed_order.py`, `freeze_receipt.json`, and this `HANDOFF.md`. Repository reports and legacy experiment sources/artifacts were read-only. No staged files were created.

## Supervisor handoff

Retain the coded policy utility and verified keyed ordering. Keep all joint policy / complete-system outcomes null and **CONDITIONAL / INCONCLUSIVE**. Before any continuation, supervisor must identify a hash-verifiable exact MB03 producer/checkpoint+normalizer lineage and certify all required eligible inner target score streams; only then may the bounded 12 inner + up to 4 outer seed101 fits start under the frozen 20-minute CPU cap. Do not interpret the legacy keyed outer MB03 insertion scores as a new consistently seeded stream or as threshold-calibrated joint-policy evidence.
