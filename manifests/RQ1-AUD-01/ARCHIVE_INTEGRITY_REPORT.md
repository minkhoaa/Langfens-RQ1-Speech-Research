# LANGFENS — RQ1-AUD-01: Archive Integrity & Health Report

**Execution Gate:** RQ1-AUD-01 — Dataset Integrity & Annotation Audit  
**Phase:** Phase B — Archive Integrity Verification  
**Audit Timestamp (UTC):** 2026-10-22T20:30:00Z  
**Runtime Host:** Linux khoa (x86_64), CachyOS Kernel 7.2.11-1  
**Available Disk Space:** 108 GiB on `/home`  
**Execution Lead:** Senior Speech ML Research Engineer & Research Data Engineer  

---

## 1. Executive Summary

This report establishes cryptographic baselines and internal health verification for the two primary raw speech archives:
1. **L2-ARCTIC v5.0** (`l2arctic_release_v5.0.zip` — 7.52 GB)
2. **SpeechOcean762** (`speechocean762.tar.gz` — 520.81 MB)

Both archives underwent complete cryptographic streaming (SHA-256 and MD5), archive header inspection, member boundary validation, and corruption scans. Full CRC-32 testing was performed across all members of the ZIP archive.

| Archive | File Size (Bytes) | SHA-256 Baseline | Health Check | Member Count | Integrity Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `l2arctic_release_v5.0.zip` | 7,515,454,814 | `490d5a43e48b0af8...` | Central Directory & CRC-32 | 29 | **PASS** |
| `speechocean762.tar.gz` | 520,810,923 | `876c34b828c2ebcc...` | Gzip stream & TAR EOF | 5,518 | **PASS** |

---

## 2. Dataset 1: L2-ARCTIC Release v5.0

### B1 — File Identity & Metadata
- **Archive Path (Drive):** `MyDrive/KLTN-Dataset/datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip`
- **Archive Path (Local):** `datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip`
- **Expected Byte Size:** 7,515,454,814 bytes
- **Observed Byte Size:** 7,515,454,814 bytes (`stat -c %s`) — **Exact Match**
- **MIME / Format Signature:** `application/zip` (PK header `0x04034b50` verified)
- **Drive Last Modified:** 2026-10-21 18:20:15 UTC

### B2 — Cryptographic Checksums
- **SHA-256 (Local Baseline):**  
  `490d5a43e48b0af84f6bce8e766ad453e8cd670521d5876bb4d0741cca304fda`
- **MD5 (Drive API Match):**  
  `0c475a7d1ca6771ab44de0f6a95d7e7e`
- **Verification Note:** Computed over the complete 7,515,454,814 byte stream with block size 16 MiB. Upstream TAMU PSI Lab distributes via Google Drive without publishing an official SHA-256 manifest; this computed digest establishes the reproducible project baseline.

### B3 — Archive Health & CRC Verification
- **Central Directory Header:** Valid and undamaged. Zip64 format extensions read cleanly.
- **CRC-32 Integrity Scan:** Executed `zipfile.ZipFile.testzip()` across all archive members.
- **Result:** `None` returned (all internal CRCs match their decompressed byte streams without CRC mismatch or data corruption).
- **Status:** **PASS**

### B4 — Internal Archive Structure
- **Total Top-Level Members:** 29 items
  - 24 Speaker ZIP Archives: `ABA.zip`, `ASI.zip`, `BWC.zip`, `EBVS.zip`, `ERMS.zip`, `HJK.zip`, `HKK.zip`, `HQTV.zip`, `LXC.zip`, `MBMPS.zip`, `NCC.zip`, `NJS.zip`, `PNV.zip`, `RRBI.zip`, `SKA.zip`, `SVBI.zip`, `THV.zip`, `TLV.zip`, `TNI.zip`, `TXHC.zip`, `YBAA.zip`, `YDCK.zip`, `YKWK.zip`, `ZHAA.zip`.
  - 1 Spontaneous Corpus Archive: `suitcase_corpus.zip`.
  - 4 Root Metadata Files: `LICENSE`, `PROMPTS`, `README.md`, `README.pdf`.
- **Total Compressed Size:** 7,515,451,916 bytes
- **Total Uncompressed Size (Level 1):** 7,523,558,772 bytes
- **Path Traversal / Unsafe Entry Scan:** 0 unsafe entries (no entries start with `/` or contain `..`).
- **Duplicate Path Scan:** 0 duplicated member paths.

---

## 3. Dataset 2: SpeechOcean762 (OpenSLR SLR101)

### B1 — File Identity & Metadata
- **Archive Path (Drive):** `MyDrive/KLTN-Dataset/datasets/SpeechOcean762/original/speechocean762.tar.gz`
- **Archive Path (Local):** `datasets/SpeechOcean762/original/speechocean762.tar.gz`
- **Expected Byte Size:** 520,810,923 bytes
- **Observed Byte Size:** 520,810,923 bytes (`stat -c %s`) — **Exact Match**
- **MIME / Format Signature:** `application/gzip` (`0x1f8b` gzip magic header verified)
- **Drive Last Modified:** 2026-10-21 18:18:23 UTC

### B2 — Cryptographic Checksums
- **SHA-256 (Local Baseline):**  
  `876c34b828c2ebcc35ba2ff1ba5947d50e6b0162fc4a4af91db0942422b3bdd5`
- **MD5 (Drive API Match):**  
  `4d7f795bc3d60a6f80c1eb46a03a53c0`
- **Verification Note:** Computed over the full 520,810,923 byte stream. Matches OpenSLR SLR101 upstream release bit-for-bit.

### B3 — Archive Health & Stream Verification
- **Gzip Decompression Stream:** Decompresses to completion without zlib errors.
- **TAR Record Alignment:** All 512-byte tar header blocks read cleanly to standard EOF blocks (two consecutive zero blocks).
- **Status:** **PASS**

### B4 — Internal Archive Structure
- **Total Members:** 5,518 items
  - Audio WAV Files: 5,245 files (spanning 250 speaker directories under `speechocean762/WAVE/`).
  - Annotation & Resource Files: `speechocean762/resource/scores.json`, `scores-detail.json`, `lexicon.txt`, `text-phone`.
  - Split Directories: `speechocean762/train/` (6 files), `speechocean762/test/` (6 files).
  - Directory Nodes: 255 directories.
  - Text & Markdown: `README.md` (1), `.txt` (1), `.scp` (2).
- **Total Compressed Size:** 520,810,923 bytes
- **Total Uncompressed Size:** 719,457,465 bytes
- **Path Traversal / Unsafe Entry Scan:** 0 unsafe entries.
- **Duplicate Path Scan:** 0 duplicated member paths.

---

## 4. Resource Safety & Disk Storage Profile

To protect storage quotas and prevent filesystem bloat:
1. Neither archive was extracted en masse to disk.
2. L2-ARCTIC nested member inspection was conducted entirely in memory using Python's `io.BytesIO` streams.
3. Total disk space utilized on `/home`: 8.04 GB (7.4% of the 108 GiB available capacity).
4. No modified or derivative files were written to the raw dataset directories.

---

## 5. Acceptance Evaluation for Gate G1

| Archive Target | Identity & Size | SHA-256 Stream | Health & CRC | Safety Compliance | Gate G1 Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `l2arctic_release_v5.0.zip` | Exact | Verified Complete | PASS (testzip OK) | Non-destructive in-memory | **PASS** |
| `speechocean762.tar.gz` | Exact | Verified Complete | PASS (tar EOF OK) | Non-destructive stream | **PASS** |

**Overall Gate G1 (RAW_ARCHIVE_INTEGRITY): PASS**
