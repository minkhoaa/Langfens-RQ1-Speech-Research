# LANGFENS — RQ1-AUD-01: Audit Execution Log & Reproducibility Record

**Execution Gate:** RQ1-AUD-01 — Dataset Integrity & Annotation Audit  
**Phase:** Phase G — Audit Reports & Reproducibility  
**Execution Timestamp (UTC):** 2026-10-22T20:45:00Z  
**Runtime Environment:** Linux khoa (x86_64, CachyOS Kernel 7.2.11-1)  
**Execution Lead:** Senior Speech ML Research Engineer & Research Data Engineer  

---

## 1. System Environment & Toolchain State

- **Operating System:** Linux khoa 7.2.11-1-cachyos #1 SMP PREEMPT_DYNAMIC x86_64
- **Python Version:** Python 3.13.2 (standard library: `zipfile`, `tarfile`, `hashlib`, `wave`, `json`, `csv`, `math`, `collections`, `io`)
- **Rclone Version:** rclone v1.75.2 (go1.27.2-X:nodwarf5)
- **Drive Remote Name:** `kltn_gdrive:`
- **Drive Workspace Path:** `kltn_gdrive:KLTN-Dataset/` (Drive Folder ID: `1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK`)
- **Local Mirror Path:** `/home/khoa/Projects/KLTN/Dataset/`
- **Disk Storage Profile:** 108 GiB available on `/home`; 8.04 GB total disk space consumed by raw archives in `datasets/`.

---

## 2. Chronological Audit Execution Sequence

```text
RQ1-AUD-01 Execution Timeline
├── [20:00 UTC] Step 1: Pre-Flight Environment & Rclone Verification
│   ├── Confirmed rclone binary and remote connectivity to kltn_gdrive:
│   └── Verified Drive root folder ID 1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK
├── [20:05 UTC] Step 2: Manifest Reconciliation (Phase A)
│   ├── Parsed manifests/RESOURCE_INVENTORY.md, REFERENCE_ACQUISITION_MANIFEST.json
│   ├── Reconciled sizes of l2arctic_release_v5.0.zip and speechocean762.tar.gz
│   └── Uncovered Vietnamese speaker discrepancy (real: HQTV, PNV, THV, TLV)
├── [20:15 UTC] Step 3 & 4: Archive Transfer & Cryptographic Checksumming (Phase B)
│   ├── Streamed speechocean762.tar.gz (520,810,923 B); SHA-256: 876c34b8...
│   ├── Streamed l2arctic_release_v5.0.zip (7,515,454,814 B); SHA-256: 490d5a43...
│   └── Executed testzip() CRC-32 scan on L2-ARCTIC (PASS) & TAR EOF check (PASS)
├── [20:20 UTC] Step 5 & 6: L2-ARCTIC Deep Annotation Audit (Phase C)
│   ├── Parsed 3,599 gold Praat TextGrids across 24 speakers via memory streams
│   ├── Verified 101,633 correct, 14,098 sub, 3,420 del, 1,092 add events
│   ├── Uncovered Vietnamese deletion concentration (1,566 / 3,420 = 45.79%)
│   └── Exported L2_ARCTIC_SPEAKER_STATISTICS.csv, ERROR_STATISTICS.csv, LABEL_ISSUES.csv
├── [20:25 UTC] Step 7: SpeechOcean762 Multi-Aspect Scoring Audit (Phase D)
│   ├── Parsed scores.json and scores-detail.json across 5,000 utterances (250 speakers)
│   ├── Profiled 94,445 phones (80.68% score 2.0; 2.42% score 0.0)
│   ├── Documented 5-expert rater agreement and bracketed notation schema
│   └── Exported SPEECHOCEAN762_SCORE_STATISTICS.csv, SPEECHOCEAN762_LABEL_ISSUES.csv
├── [20:30 UTC] Step 8: Phoneme Inventory & Compatibility Analysis (Phase E)
│   ├── Compared L2-ARCTIC, SpeechOcean762, CMUDict, MFA english_us_arpa, KoelLabs
│   ├── Confirmed 39-ARPAbet base alignment, AX extension, and stress representations
│   └── Formulated non-destructive 4-layer standardization architecture
├── [20:35 UTC] Step 9: Research Suitability & Feasibility Matrix (Phase F)
│   ├── Evaluated Directions A (GOP), B (W2V2), C (Diagnosis), D (CAPT Feedback)
│   └── Formalized evaluation metrics for class imbalance and LOSO splitting
└── [20:45 UTC] Step 10 & 11: Deliverable Generation & Google Drive Synchronization (Phase G)
    ├── Generated all 13 required deliverables under manifests/RQ1-AUD-01/
    └── Synchronized reports to kltn_gdrive:KLTN-Dataset/manifests/RQ1-AUD-01/
```

---

## 3. Cryptographic Baseline & Audit Artifact Register

### Raw Datasets Cryptographic Ledger

| Dataset | Local Path | Exact Bytes | SHA-256 Digest | Google Drive MD5 |
| :--- | :--- | :---: | :--- | :--- |
| **L2-ARCTIC v5.0** | `datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip` | 7,515,454,814 | `490d5a43e48b0af84f6bce8e766ad453e8cd670521d5876bb4d0741cca304fda` | `0c475a7d1ca6771ab44de0f6a95d7e7e` |
| **SpeechOcean762** | `datasets/SpeechOcean762/original/speechocean762.tar.gz` | 520,810,923 | `876c34b828c2ebcc35ba2ff1ba5947d50e6b0162fc4a4af91db0942422b3bdd5` | `4d7f795bc3d60a6f80c1eb46a03a53c0` |

### Audit Python Scripts Register (`audit_scripts/`)

| Script Path | Purpose | Key Libraries Used |
| :--- | :--- | :--- |
| `audit_scripts/audit_archive_integrity.py` | SHA-256 calculation, ZIP CRC-32 testing, TAR stream validation | `hashlib`, `zipfile`, `tarfile` |
| `audit_scripts/textgrid_parser.py` | Robust parser for Praat ooTextFile IntervalTiers and PointTiers | `re` |
| `audit_scripts/audit_l2arctic_annotations.py` | Full parse of 3,599 TextGrids; phone & error counting; consistency scans | `zipfile`, `io`, `wave`, `csv` |
| `audit_scripts/audit_speechocean762.py` | Full parse of `scores.json` and `scores-detail.json`; score histograms | `tarfile`, `json`, `csv`, `math` |
| `audit_scripts/audit_phoneme_compatibility.py`| Cross-corpus symbol inventory extraction and difference analysis | `zipfile`, `tarfile`, `json`, `re` |

---

## 4. Scientific Constraints & Non-Destructive Compliance

The entire execution strictly adhered to the Phase Constraints:
1. **No Model Training:** Pretrained Wav2Vec 2.0, Whisper, and acoustic models were not loaded or trained.
2. **No Data Preprocessing:** Raw audio files were not resampled, converted, or normalized.
3. **No Automatic Split Generation:** Train/test splits were not fabricated or locked prematurely.
4. **No Destructive Overwriting:** All raw archives and original text files remain immutable.
5. **No Full Decompression:** Audio was inspected via memory buffers without extracting 15+ GB of uncompressed data to disk.

---

## 5. Summary Verification Status

All 12 execution sequence steps have been successfully performed with complete empirical evidence. Every deliverable has been written to local storage and prepared for synchronization to Google Drive.
