# RQ1-TRN-02A-EVAL-R1 — checkpoint recovery ledger

Status: artifacts recovered from the configured Google Drive remote; evaluation replay not yet completed.

## Source and retrieval evidence

- Repository source: `a6e8035644ba0adeab347a350658e27d86ef17c5`.
- Training execution SHA recorded in checkpoint metadata: `caa56783113d40eac438e5f458cdb313b8932218`.
- Read-only listing succeeded for `kltn_gdrive:KLTN-Dataset/manifests/RQ1-TRN-02A/checkpoints/`.
- Download destination: `checkpoints/RQ1-TRN-02A/recovered/` (ignored by Git; checkpoint files must not be committed).
- All five requested file names and expected byte sizes matched the remote listing and downloaded local files.
- `history.json`, `EVALUATION_REPORT.json`, and `BASELINE_COMPARISON.json` compare byte-for-byte equal to the Git-tracked copies (`TRAINING_HISTORY.json`, `EVALUATION_REPORT.json`, and `BASELINE_COMPARISON.json`, respectively).

## Local SHA-256

| File | Bytes | SHA-256 |
|---|---:|---|
| `best.pt` | 412366029 | `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b` |
| `last.pt` | 412367181 | `bf46fab366363d46082082cf7f088b218f6d8d12851f5a9de706276b789efcb6` |
| `history.json` | 8819 | `270dab59626bf665c0e52964b6f3c5a90e46fbaede856df987046c641dbceff5` |
| `EVALUATION_REPORT.json` | 5611 | `e4d093dd48bcfa9ecc0c7a2f452a5db9a86ff92b7e6702099cc504c85611778f` |
| `BASELINE_COMPARISON.json` | 1227 | `e08f4328115e968b8dd2f18979e660ef5a89d884b2e3e6449ef6b050597f11f9` |

## Handling and reproducibility

The two PyTorch checkpoint dictionaries were deserialized on CPU in the repository virtual environment with `weights_only=False`; this is a pickle-based format and should only be loaded from the trusted original Drive source. Neither checkpoint was modified. No upload or remote write was attempted. The checkpoint binary files are excluded from Git; this report records their hashes and sizes only.
