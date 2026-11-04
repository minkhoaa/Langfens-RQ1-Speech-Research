# LANGFENS-RQ1-TRN-04R — Workstream C Artifact Checkpoint & Source Reconciliation

**Status:** `ARTIFACT_BYTES_AND_METADATA_RECONCILED; HISTORICAL_SOURCE_REVISION_UNRECOVERED; COORDINATOR_RUNTIME_PENDING`
**Workstream:** C — checkpoint integrity and source reconciliation
**Worktree base/head at start:** `d332997e732315292d673a3bd42a03698182f5f6` (`review/trn04r-c`)
**Scope:** read-only artifact metadata, hashes, provenance and frozen partition metadata only. No benchmark inference was run.

## Inputs reviewed

- `AGENTS.md`, global process-checkpoint index, `manifests/RQ1-TRN-04/PROCESS_CHECKPOINT.md`, `manifests/RQ1-P-SUP-BENCHMARK-V1/final-v2/PROCESS_CHECKPOINT.md`, and frozen `EVALUATION_CONTRACT_V1.md` (SHA-256 `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`).
- Only the allowlisted checkpoint files beneath `/home/khoa/Projects/KLTN/Dataset/checkpoints/RQ1-TRN-04/{c0,t1,t2,t3}`: `fixed_budget_3600.pt`, `best.pt`, `training_summary.json`. Did not recurse into datasets/archives.
- Each arm's tracked `CHECKPOINT_INTEGRITY.json`, `TRAINING_RUN_MANIFEST.json`; root `CHECKPOINT_RECEIPT.json` in the parent checkout; frozen contract partition metadata.

## Findings

### Byte integrity and remote readback

All eight local model checkpoint files have the same byte lengths and full SHA-256 values as their arm `CHECKPOINT_INTEGRITY.json` and the parent checkout's persisted receipt. All four local training summaries also match the parent persisted receipt. Then, after authenticated Hugging Face API metadata confirmed the repository is private, each of these 12 existing payloads was downloaded read-only at pinned repository commit `af6ac3620fb78873ff213c064dffe71fb8acadd4`; downloaded bytes/hash match the local bytes. No uploads, deletions, or privacy changes occurred. Current immutable revision and exact file identities are in `ARTIFACT_PROVENANCE_LEDGER.json`.

The RQ1-TRN-04 master checkpoint's displayed artifact table contains shortened strings presented as SHA-256 checksums; these are not complete 64-hex digests. Do not use those truncated values as identities. Its sizes and update/skip account broadly correspond to the verified receipts, but ledger values here are the complete receipts. Original arm integrity manifests themselves contain correct full hashes.

### CPU metadata/state inspection (not inference)

Trusted `torch.load(..., map_location="cpu", weights_only=False)` was used only to inspect checkpoint metadata and tensor state. All 8 files deserialize; each saved head has 17 tensors, 267,179 parameters, the same recorded architecture shapes, and finite values. Artifact metadata confirms objectives, feature setting, frozen contract hash, seed 1337, fixed/best endpoint flags, epoch and successful-update counters. No model forward call or mutation occurred.

| Arm | Objective | 4D features | Fixed checkpoint: epoch / successful / skipped | SELECT-best checkpoint: epoch / successful | Head params |
|---|---|---:|---:|---:|---:|
| C0 | `operation_ce` | no | 9 / 3,600 / 3 | 5 / 2,247 | 267,179 |
| T1 | `class_focal` | no | 9 / 3,600 / 3 | 2 / 897 | 267,179 |
| T2 | `operation_ce` | yes | 9 / 3,600 / 3 | 9 / 3,600 | 267,179 |
| T3 | `class_focal` | yes | 9 / 3,600 / 3 | 5 / 2,247 | 267,179 |

All four `training_summary.json` files report terminal 3,600 successful updates, three skipped AMP attempts (3,603 attempts), nine epochs. The fixed endpoint checkpoint metadata agrees. A best checkpoint is a distinct SELECT-selected snapshot, not automatically the terminal checkpoint; its `successful_updates` is the actual snapshot counter shown above. The skipped-attempt value attached to a best snapshot is cumulative run metadata, not skipped attempts since that endpoint.

### Split, CAL/VAL and sealing

The frozen contract records split manifest SHA-256 `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`; TRAIN 2,399 utterances and ordered-ID digest `2ebe54f2fc77e2a3ee9890ce3dc21f725897ebbe85f812d3c017c3b495f14b27`; VAL 300 and digest `6d34642b15bbf80cbd5bd102b5574ab36f5adac74be6d6d0ae7218b002eff065`. Partition metadata identifies FIT 1,799 (12 speakers), SELECT 300 (MBMPS/TLV), CAL 300 (NCC/THV), and VAL 300 (NJS/RRBI). TEST remains sealed: metadata only (900 utterances, speakers ABA/ASI/EBVS/HJK/LXC/PNV); no TEST IDs, payloads, or labels were accessed or exported. No CAL/VAL scoring, inference, calibration, or tuning was performed in this workstream.

### Source identity limitation

The arm training manifests identify objective, configuration, kernel slug, partitions, checkpoint hashes and terminal counters, but no immutable training-source commit. The serialized artifact metadata and summaries also omit a source commit. The reviewed `PROCESS_CHECKPOINT.md` has a “Base Pinned Commit” but that is not an artifact-level training source attestation. Current review checkout (`d332997...`) is not evidence of historical training source and is not substituted for it. Exact historical code equivalence/reproducibility therefore remains unverified; this is explicitly recorded as a provenance gap, not silently inferred from current source.

No final benchmark metrics conclusion is made by this workstream. Parent/coordinator owns the integrated revision and final benchmark execution; runtime is pending.

## Boundaries and restart

This process performed no training, backward pass, optimizer step, checkpoint/model mutation, inference, VAL tuning, TEST payload access, publication, upload, or privacy change. Remote access was authenticated read-only and performed only after confirming the HF repo's private status. No weights or data were added to Git.

Safe continuation: coordinator may consume `ARTIFACT_PROVENANCE_LEDGER.json` as a read-only artifact identity map, while preserving its exact pinned revision and two endpoint counters. Verify final integrated code/contract hash before authorized benchmark runtime. Do not treat missing historical source commit as resolved; report it as a provenance limitation. Benchmark runtime remains pending and is coordinator-owned.
