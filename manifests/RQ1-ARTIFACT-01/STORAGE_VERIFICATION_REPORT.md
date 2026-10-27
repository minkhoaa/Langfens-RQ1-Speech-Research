# STORAGE VERIFICATION REPORT — LANGFENS RQ1-ARTIFACT-01

**Date:** 2026-10-27  
**Artifact ID:** `RQ1-ARTIFACT-01`  
**Google Drive Destination:** `kltn_gdrive:KLTN-Dataset` (Folder ID: `1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK`)  
**Status:** PASS — ALL INTEGRITY CHECKS CONFIRMED

---

## 1. Scope & Objective

This report provides the formal storage verification audit for the Google Drive persistent archive of **LANGFENS RQ1 Fold 1 Baseline R** under experiment folder `RQ1-TRN-02A`.

The audit verifies:
1. Destination folder ancestry, ownership, and write permissions within authorized Google Drive folder `1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK`.
2. Cryptographic checksums and byte-level file sizes for all training checkpoints, evaluation histories, and authorized dataset archives.
3. Successful upload, permission persistence, and read-back validation of `BASELINE_R_ARTIFACT_MANIFEST.json`.
4. Absence of test split leakage or redundant file duplication.

---

## 2. Google Drive Hierarchy & Folder Ancestry

The directory hierarchy and verified folder IDs within the remote workspace are:

```
[ROOT] KLTN-Dataset (ID: 1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK)
  └── manifests (ID: 1AHTjosYOGEHl9lqKou0cF_K4wdwZD1JU)
        └── RQ1-TRN-02A (ID: 11g-5JI5u5JKHxV8u9kyCo4I-SF2JWJLx)
              ├── BASELINE_R_ARTIFACT_MANIFEST.json (ID: 1yrTbDR9ki0u-kv_TyaCfwqdJw20CbTKP)
              ├── checkpoints/ (ID: 1MFtZUBXixhZHqC7Au92cRIqC9CYx-S7a)
              │     ├── best.pt (ID: 10SBhwWd0eAv_ZByBmfJjJZwHywA-LU-I)
              │     ├── last.pt (ID: 1VdlhoQHfd9MqjzIkMUEqdqMxhLlJiO-s)
              │     ├── history.json (ID: 128mkF6EFTHl7ztiVVy56rUEwmoqlKy-M)
              │     ├── EVALUATION_REPORT.json (ID: 1vehhJp8Q23uIFY7hIEtjrIZ_czFki1wV)
              │     └── BASELINE_COMPARISON.json (ID: 1PR5HLpyCJMNYM-TblNqTOPnpg0EQaNmQ)
              ├── authorized_data/ (ID: 13J3gB7zM8vcVw9s1ci3q07AJYvC2Sunp)
              │     └── fold1_train_val_only.zip (ID: 1xdypDWopd585OjuQeHjxlWOJfU2oiJiY)
              └── feasibility/ (ID: 1JHK1tf2LPEPg4VIbK-ZQnCaJWEWscHrj)
                    └── rq1_fold1_colab_t4_micro.pt (ID: 1dcn4kXKw0YmuXjlVn78Fa6t0-89yOWWX)
```

### Ancestry Verification Audit

- **Root Ancestry:** All paths strictly originate from Google Drive folder ID `1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK` (`KLTN-Dataset`).
- **Parent Isolation:** Neither personal drives, temporary folders, nor unrelated storage buckets were utilized.
- **Write Permissions:** Verified by uploading `BASELINE_R_ARTIFACT_MANIFEST.json` directly into `RQ1-TRN-02A` and reading back its complete binary contents.

---

## 3. Cryptographic Checksum Audit Table

| File Name | Drive Path | Google Drive ID | File Size (Bytes) | MD5 Digest | Verified SHA-256 Digest | Status |
|---|---|---|---|---|---|---|
| `best.pt` | `manifests/RQ1-TRN-02A/checkpoints/` | `10SBhwWd0eAv_ZByBmfJjJZwHywA-LU-I` | 412,366,029 | `70c16eff09235e85b4cebc056113994a` | `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b` | **MATCH** |
| `last.pt` | `manifests/RQ1-TRN-02A/checkpoints/` | `1VdlhoQHfd9MqjzIkMUEqdqMxhLlJiO-s` | 412,367,181 | `fa52f658103942bb36bbb530f60b2ca2` | `bf46fab366363d46082082cf7f088b218f6d8d12851f5a9de706276b789efcb6` | **MATCH** |
| `history.json` | `manifests/RQ1-TRN-02A/checkpoints/` | `128mkF6EFTHl7ztiVVy56rUEwmoqlKy-M` | 8,819 | `6ef71d8b85a8137d971e0729c7e41ade` | `270dab59626bf665c0e52964b6f3c5a90e46fbaede856df987046c641dbceff5` | **MATCH** |
| `EVALUATION_REPORT.json` | `manifests/RQ1-TRN-02A/checkpoints/` | `1vehhJp8Q23uIFY7hIEtjrIZ_czFki1wV` | 5,611 | `2d87afc5ee889b0d8e5a0fb327ddedb3` | `e4d093dd48bcfa9ecc0c7a2f452a5db9a86ff92b7e6702099cc504c85611778f` | **MATCH** |
| `BASELINE_COMPARISON.json` | `manifests/RQ1-TRN-02A/checkpoints/` | `1PR5HLpyCJMNYM-TblNqTOPnpg0EQaNmQ` | 1,227 | `106547eac8779477220fae3c6d107ce1` | `e08f4328115e968b8dd2f18979e660ef5a89d884b2e3e6449ef6b050597f11f9` | **MATCH** |
| `fold1_train_val_only.zip` | `manifests/RQ1-TRN-02A/authorized_data/` | `1xdypDWopd585OjuQeHjxlWOJfU2oiJiY` | 761,922,918 | `fec058e897d0da739127ba79bfef521a` | `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9` | **MATCH** |
| `BASELINE_R_ARTIFACT_MANIFEST.json` | `manifests/RQ1-TRN-02A/` | `1yrTbDR9ki0u-kv_TyaCfwqdJw20CbTKP` | 7,486 | `b7be3b64c7fe0a4f5f5fe844ebf32b85` | `6222c92b63ea58f584267b39fb3ffc51394562086f90de21bbab8c8c1abe1e22` | **MATCH** |

---

## 4. Manifest Read-Back & Consistency Verification

The deliverable manifest `BASELINE_R_ARTIFACT_MANIFEST.json` was generated and copied to Google Drive via `rclone copyto`.

Integrity verification was completed as follows:
1. Queried Google Drive file metadata via `rclone lsjson`.
2. Verified remote file ID: `1yrTbDR9ki0u-kv_TyaCfwqdJw20CbTKP`.
3. Read the complete remote payload back using `rclone cat`.
4. Calculated remote SHA-256: `6222c92b63ea58f584267b39fb3ffc51394562086f90de21bbab8c8c1abe1e22`.
5. Confirmed exact match with local staged manifest:
   ```
   Local SHA-256:  6222c92b63ea58f584267b39fb3ffc51394562086f90de21bbab8c8c1abe1e22
   Remote SHA-256: 6222c92b63ea58f584267b39fb3ffc51394562086f90de21bbab8c8c1abe1e22
   Status: PASS
   ```
6. Validated all cross-references to Hugging Face (`minkhoaa/langfens-rq1-fold1-baseline-r`), GitHub (`8486cbb3596a0f5d2603dcdf46fd891b5c849840`), and training commit (`caa56783113d40eac438e5f458cdb313b8932218`).

---

## 5. Storage Compliance Conclusion

- **Parent Ancestry:** Verified under `1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK`.
- **Checkpoint Immutability:** Untouched; identical hashes confirmed.
- **Dataset Sealing:** Only Fold 1 Train & Val present in authorized archive.
- **Drive Manifest Verification:** **PASS**.
