# LANGFENS — RQ1-PREP-01: Split Provenance Time Erratum

**Execution Gate:** RQ1-PREP-01 — Preflight & Provenance Investigation  
**Corpus / Targets:**
- `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`
- `manifests/RQ1-SPLIT-01/l2arctic_loso_vietnamese_splits.json`
- `manifests/RQ1-SPLIT-01/speechocean762_official_splits.json`  
**Deliverable Associated Path:** `manifests/RQ1-PREP-01/SPLIT_PROVENANCE_TIME_ERRATUM.md`  
**Synced Location:** `kltn_gdrive:KLTN-Dataset/manifests/RQ1-PREP-01/`  
**Investigation Status:** `NOT_VERIFIED` (Historical Header UTC Timestamp) / `VERIFIED_MATCH` (Data Payload SHA-256)

---

## 1. Executive Summary

An independent reproducibility audit of the RQ1-SPLIT-01 manifests identified a temporal discrepancy:
- The metadata header in the split manifests explicitly records `"generation_timestamp_utc": "2026-10-23T13:30:00Z"`.
- However, Google Drive metadata and `rclone lsd` listings show the files were synchronized at `2026-10-23 13:26:52 +07:00` (which corresponds to `2026-10-23T06:26:52Z` UTC).
- Under strict UTC interpretation, the manifest claims to have been generated ~7 hours *after* it was uploaded to Google Drive.

This erratum presents the physical, forensic, and cryptographic investigation into this discrepancy.

---

## 2. Forensic Investigation & Evidence

### 2.1 Host Environment & Timezone Analysis
- **Operating System:** Linux x86_64
- **Host Timezone:** Indochina Time (`ICT`, `UTC+07:00`)
- **System Time (at run):** Friday, October 23, 2026 (UTC+07)

### 2.2 Filesystem Modification Times (Local NVMe Storage)
Physical inspection via `stat` on the host filesystem reveals the exact file creation and modification timestamps:

| Manifest File | File Size (Bytes) | Local Modification Time (`+07:00`) | Exact UTC Modification Time (`Z`) |
| :--- | :---: | :---: | :---: |
| `l2arctic_4fold_splits.json` | 448,900 | `2026-10-23 13:23:01.955569212 +0700` | `2026-10-23T06:23:01.955569212Z` |
| `l2arctic_loso_vietnamese_splits.json` | 582,840 | `2026-10-23 13:23:01.964833312 +0700` | `2026-10-23T06:23:01.964833312Z` |
| `speechocean762_official_splits.json` | 126,598 | `2026-10-23 13:23:14.995856708 +0700` | `2026-10-23T06:23:14.995856708Z` |
| `SPLIT_VALIDATION_REPORT.md` | 16,857 | `2026-10-23 13:25:13.897439252 +0700` | `2026-10-23T06:25:13.897439252Z` |
| `SPLIT_PROVENANCE.json` | 3,700 | `2026-10-23 13:25:42.675713437 +0700` | `2026-10-23T06:25:42.675713437Z` |

### 2.3 Remote Synchronization Ledger
`rclone lsd kltn_gdrive:KLTN-Dataset/manifests/` records:
- `RQ1-SPLIT-01` directory timestamp: `2026-10-23 13:26:52` (local display, corresponding to `06:26:52Z` UTC).
This demonstrates that the local files were written at ~13:23 local time and synchronized to Google Drive at 13:26 local time.

### 2.4 Generator Script Inspection (`audit_scripts/generate_rq1_splits.py`)
Direct source code inspection of `audit_scripts/generate_rq1_splits.py` reveals lines 27–28:
```python
SPLIT_PROTOCOL_VERSION = "RQ1-SPLIT-01"
DEFAULT_TIMESTAMP_UTC = "2026-10-23T13:30:00Z"
```
And lines 598–600:
```python
# Allow override via environment variable if requested, else fixed reproducible timestamp
timestamp_utc = os.environ.get("LANGFENS_SPLIT_TIMESTAMP", DEFAULT_TIMESTAMP_UTC)
```

**Root Cause:**
The generator script author set a static fallback timestamp `"2026-10-23T13:30:00Z"` to ensure bitwise reproducible test runs. Because the author was operating in the UTC+07:00 timezone (where local time was ~13:23 at run time), the string was conflated: `13:30` was chosen as a rounded local planning time, but the literal character `Z` (indicating zero UTC offset) was appended instead of `+07:00` or converting `13:30 +07:00` to `06:30Z`.

---

## 3. Cryptographic Independence of `payload_sha256`

Crucially, the integrity of the speaker partitions and sample assignments is **completely immune** to this metadata timestamp error.

### 3.1 Canonical Serialization Specification
Project Langfens computes `payload_sha256` strictly on the payload dictionary, excluding all external metadata headers (`generation_timestamp_utc`, `master_seed`, `generation_algorithm`):
$$\text{payload\_bytes} = \text{json.dumps}(\text{payload\_dict}, \text{sort\_keys}=\text{True}).\text{encode}(\text{"utf-8"})$$
$$\text{payload\_sha256} = \text{SHA256}(\text{payload\_bytes})$$

### 3.2 Payload Targets per Manifest
1. `l2arctic_4fold_splits.json`:
   - Payload: `manifest["folds"]` (Folds 1 to 4 dictionary)
   - Recorded `payload_sha256`: `7d4fc2ef9aa0cf8b382023cbdf79d38c64265538e12d1bbfbf66cbf663e005be`
   - Re-computed independently: `7d4fc2ef9aa0cf8b382023cbdf79d38c64265538e12d1bbfbf66cbf663e005be` (**EXACT MATCH**)
2. `l2arctic_loso_vietnamese_splits.json`:
   - Payload: `{"regime_a": manifest["regime_a_cross_vietnamese_loso"]["rounds"], "regime_b": manifest["regime_b_zero_shot_vietnamese_transfer"]}`
   - Recorded `payload_sha256`: `1f9076628f80453f650577fb57eb7fc8d5f49ab6d093aa6d2bda061e88be6043`
   - Re-computed independently: `1f9076628f80453f650577fb57eb7fc8d5f49ab6d093aa6d2bda061e88be6043` (**EXACT MATCH**)
3. `speechocean762_official_splits.json`:
   - Payload: `manifest["partitions"]` (`train`, `val`, `test` mappings)
   - Recorded `payload_sha256`: `3d9c79e604f3d17e7fc88a100fb6a60cefa89b27a3c306d860e60882e7b1659a`
   - Re-computed independently: `3d9c79e604f3d17e7fc88a100fb6a60cefa89b27a3c306d860e60882e7b1659a` (**EXACT MATCH**)

---

## 4. Formal Erratum Policy & Adjudication

1. **Non-Destruction of Frozen Manifests:**
   The split manifests in `manifests/RQ1-SPLIT-01/` are immutable research artifacts with frozen SHA-256 sums:
   - `l2arctic_4fold_splits.json`: `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`
   - `l2arctic_loso_vietnamese_splits.json`: `7a59b63ad73c7616af2ef583760b50253c973090c4c7c76aadc46029cfe69118`
   - `speechocean762_official_splits.json`: `bc3f9ec97b5077402b3c67d89c857b9efe694649bef6e5d43776f07b528bad17`
   They must NOT be rewritten merely to correct header metadata, as doing so would invalidate downstream provenance tracking.
2. **Provenance Status Adjudication:**
   - Historical UTC Wall-Clock Timestamp: **`NOT_VERIFIED`** (proven to be a hardcoded local/UTC conflation).
   - Local Creation Wall-Clock Timestamp: **`2026-10-23 13:23:01 to 13:23:15 +07:00`** (forensically verified via inode metadata).
   - Data Partition Payloads: **`VERIFIED_MATCH`** (100% bitwise and cryptographic reproducibility).
