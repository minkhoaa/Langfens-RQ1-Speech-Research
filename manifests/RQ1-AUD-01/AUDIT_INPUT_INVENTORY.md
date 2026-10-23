# LANGFENS — RQ1-AUD-01: Audit Input Inventory & Manifest Reconciliation

**Execution Gate:** RQ1-AUD-01 — Dataset Integrity & Annotation Audit  
**Phase:** Phase A — Acquisition Manifest Reconciliation  
**Audit Timestamp (UTC):** 2026-10-22T20:25:00Z  
**Storage Path (Drive):** `MyDrive/KLTN-Dataset/` (Folder ID: `1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK`)  
**Storage Path (Local):** `/home/khoa/Projects/KLTN/Dataset/`  
**Execution Lead:** Senior Speech ML Research Engineer, Phonetics Researcher & Research Data Engineer  

---

## 1. Executive Summary & Objective

Phase A performs an independent reconciliation between the assets recorded in the acquisition phase (`RQ1-ACQ-01`) and the physical files present on Google Drive and the local audit environment.
In accordance with scientific audit standards, information reported in acquisition manifests was not taken as verified fact until physically checked and reconciled against raw corpus files, primary documentation, and cryptographic digests.

---

## 2. Manifest Verification & Readability Status

All five acquisition deliverable manifests generated during `RQ1-ACQ-01` were inspected and confirmed readable:

| Manifest Path | File Size (Bytes) | Format | Status | Verification Notes |
| :--- | :---: | :---: | :---: | :--- |
| `manifests/RESOURCE_INVENTORY.md` | 13,296 | Markdown | `VERIFIED` | Full 34-item ledger present; paths match directory hierarchy. |
| `manifests/REFERENCE_ACQUISITION_MANIFEST.json` | 18,941 | JSON | `VERIFIED` | Valid JSON syntax; 26 machine-readable resource objects. |
| `manifests/RESEARCH_LIBRARY.md` | 13,447 | Markdown | `VERIFIED` | 10 research papers taxonomy (P01–P10) mapped to 5 pillars. |
| `manifests/ACQUISITION_REPORT.md` | 12,028 | Markdown | `VERIFIED` | Records `REFERENCE_PACK_COMPLETE: PASS`, 0 failed downloads. |
| `manifests/BLOCKERS_AND_NEXT_ACTIONS.md` | 6,346 | Markdown | `VERIFIED` | Lists AUD-01 to AUD-05 next actions; flags pending checksums. |

---

## 3. Physical Archive Reconciliation

The two raw speech dataset archives recorded in `manifests/RESOURCE_INVENTORY.md` were inspected directly on Google Drive (via `rclone`) and in local storage:

| Resource ID | Canonical Path | Reported Size | Verified Local Size | Drive Metadata Size | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `CORPUS_L2ARCTIC` | `datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip` | 7,515,454,814 B | 7,515,454,814 B | 7,515,454,814 B | `EXACT_MATCH` |
| `CORPUS_SO762` | `datasets/SpeechOcean762/original/speechocean762.tar.gz` | 520,810,923 B | 520,810,923 B | 520,810,923 B | `EXACT_MATCH` |

### Checksum Reconciliation
- **L2-ARCTIC v5.0:**
  - Google Drive MD5: `0c475a7d1ca6771ab44de0f6a95d7e7e`
  - Local Computed SHA-256: `490d5a43e48b0af84f6bce8e766ad453e8cd670521d5876bb4d0741cca304fda`
- **SpeechOcean762 (SLR101):**
  - Google Drive MD5: `4d7f795bc3d60a6f80c1eb46a03a53c0`
  - Local Computed SHA-256: `876c34b828c2ebcc35ba2ff1ba5947d50e6b0162fc4a4af91db0942422b3bdd5`

---

## 4. Dataset Documentation & Linguistic Resource Ledger

All documentation and linguistic tools recorded in acquisition manifests were verified:

| Resource Category | Verified Path | Version / Upstream Hash | License / Status |
| :--- | :--- | :--- | :--- |
| **L2-ARCTIC Docs** | `references/dataset-documentation/L2-ARCTIC/` | TAMU PSI Lab Release v5.0 (March 2020) | CC BY-NC 4.0 |
| **SpeechOcean762 Docs** | `references/dataset-documentation/SpeechOcean762/` | OpenSLR SLR101 / GitHub `613968e` | CC BY 4.0 |
| **KoelLabs Dataset Card** | `references/dataset-documentation/KoelLabs/` | Hugging Face commit `e9f2aaca` | CC BY-NC 4.0 (Gated) |
| **CMUDict Dictionary** | `resources/CMUDict/cmudict.dict` | CMUSphinx master `74790861` | BSD-style |
| **CMUDict Symbols** | `resources/CMUDict/cmudict.symbols` | 84 stress-marked ARPAbet symbols | BSD-style |
| **MFA Acoustic Model** | `resources/MFA/acoustic-models/english_us_arpa.zip` | Montreal Corpus Tools `v3.0.0` | Open Source |
| **MFA Dictionary** | `resources/MFA/dictionaries/english_us_arpa.dict` | Montreal Corpus Tools `v3.0.0` | Open Source |

---

## 5. Critical Discrepancies & Nuances Discovered

During Phase A reconciliation, the audit cross-examined the claims made in acquisition notes against the actual metadata found in `README.md` inside `l2arctic_release_v5.0.zip`:

### Discrepancy 1: Vietnamese Speaker Identifiers (CRITICAL FINDING)
- **Acquisition Report Note:** Stated Vietnamese speakers as `TLV`, `TNI`, `TXHC`, `BWC`.
- **Physical Dataset Verification:**
  - Inspection of `README.md` and speaker directories inside `l2arctic_release_v5.0.zip` proves that:
    - `BWC`: **Chinese** L1 (Male) — Directory `BWC.zip`
    - `TXHC`: **Chinese** L1 (Male) — Directory `TXHC.zip`
    - `TNI`: **Hindi** L1 (Female) — Directory `TNI.zip`
    - The actual Vietnamese L1 speakers in L2-ARCTIC are:
      1. `HQTV` (Male, Vietnamese)
      2. `PNV` (Female, Vietnamese)
      3. `THV` (Female, Vietnamese)
      4. `TLV` (Male, Vietnamese)
  - **Corpus Naming Rule:** All L2-ARCTIC speaker IDs encode their native language as the final letter:
    - `A` = Arabic (`ABA`, `SKA`, `YBAA`, `ZHAA`)
    - `C` = Chinese (`BWC`, `LXC`, `NCC`, `TXHC`)
    - `I` = Hindi (`ASI`, `RRBI`, `SVBI`, `TNI`)
    - `K` = Korean (`HJK`, `HKK`, `YDCK`, `YKWK`)
    - `S` = Spanish (`EBVS`, `ERMS`, `MBMPS`, `NJS`)
    - `V` = Vietnamese (`HQTV`, `PNV`, `THV`, `TLV`)
- **Resolution:** Updated all subsequent audit pipelines to bind Vietnamese analysis strictly to `HQTV`, `PNV`, `THV`, and `TLV`.

### Discrepancy 2: Archive Internal Structure of L2-ARCTIC
- **Acquisition Assumption:** Suggested `l2arctic_release_v5.0.zip` was a flat hierarchy of directories.
- **Physical Verification:** `l2arctic_release_v5.0.zip` is a composite archive containing 24 individual speaker ZIP files (`ABA.zip` ... `ZHAA.zip`), one spontaneous narrative archive (`suitcase_corpus.zip`), and root metadata files (`PROMPTS`, `README.md`, `README.pdf`, `LICENSE`).
- **Resolution:** Implemented nested in-memory extraction via Python's `zipfile` and `io.BytesIO`, avoiding 15+ GB of uncompressed disk explosion.

### Discrepancy 3: SpeechOcean762 `mispronunciations` Field
- **Acquisition Assumption:** Presumed `scores.json` contained an explicit `mispronunciations` field based on the Hugging Face dataset card representation.
- **Physical Verification:** The raw OpenSLR SLR101 `scores.json` contains `phones-accuracy` scalar values (0.0 to 2.0) and does not have a `mispronunciations` list. Explicit rater markings exist only in `scores-detail.json` as bracket notations (`{PHONE}` for heavy accent, `(PHONE)` for incorrect/omitted, `[PHONE]` for insertions).
- **Resolution:** Evaluated in Phase D as a critical constraint for downstream MDD model architecture.

---

## 6. Audit Input Inventory Sign-Off

Phase A reconciliation confirms that all raw and auxiliary assets are physically present, accessible, and accurately cataloged. The discrepancies identified have been documented and resolved for all subsequent phases.
