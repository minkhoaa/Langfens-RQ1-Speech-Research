# Executed V2 reproduction

The complete report, actual commands, environment/source checksums, private immutable artifact inventory, evidence classifications and restart instructions are preserved at [`../REPRODUCTION_REPORT.md`](../REPRODUCTION_REPORT.md).

This version's local input/output manifests are alongside this file; raw runtime/offline predictions are intentionally **not in Git**. `RAW_ARTIFACT_INDEX.json` points to their exact approved private revision, relative paths, sizes and SHA-256 hashes. `*/protocol_*/metrics.json` contains the lightweight full-precision metric results. The private bundle also retains all raw CAL/mismatch/recognition records and the executed inference source snapshot.

Status: all four checkpoints freshly evaluated on identical eligible Fold-1VAL cohorts under separate O/Z protocols; metrics recomputed; private hashes read back. Full-control completion withheld because frozen FIT-only canonical prior remains unavailable. No training/optimizer/backward/TEST access occurred.
