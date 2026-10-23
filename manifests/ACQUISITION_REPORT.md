# LANGFENS — RQ1-ACQ-01: Research Resources Acquisition Report

**Execution Gate:** RQ1-ACQ-01  
**Project:** LANGFENS  
**Lead Roles:** Senior ML Research Engineer, Speech Research Assistant, Research Data Engineer  
**Date of Execution:** 2026-10-22  
**Target Storage:** Google Drive `MyDrive/KLTN-Dataset/` (Folder ID: `1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK`) & Local Workspace `/home/khoa/Projects/KLTN/Dataset/`  
**Gate Evaluation Status:** **REFERENCE_PACK_COMPLETE: PASS**  

---

## 1. Executive Summary

This report concludes the **RQ1-ACQ-01 Research Resources Acquisition** phase for Project Langfens.
In strict adherence to the **ACQUISITION ONLY** directives:
- No models were trained or fine-tuned.
- No audio files were resampled, converted, or modified.
- No raw datasets were extracted or manipulated.
- All downloads were conducted with temporary files, bounded retries, and strict integrity verification.
- Pre-existing files were preserved without destructive overwrites.

A total of **34 resource assets** were tracked across the research pipeline. All mandatory research literature (P0, P1, P2), dataset documentation (D01, D02, D03, D-Optional), and linguistic resources (CMUDict, MFA) were inventoried, acquired, and cryptographically verified.

---

## 2. Inventory & Acquisition Ledger

### A. Pre-Existing Resources Verified (10 items)
1. `datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip` (7,515,454,814 bytes) — Verified present on Google Drive with exact byte size match. SHA-256 calculation deferred to audit phase (`INTEGRITY_PENDING`).
2. `datasets/SpeechOcean762/original/speechocean762.tar.gz` (520,810,923 bytes) — Verified present on Google Drive with exact byte size match. SHA-256 calculation deferred (`INTEGRITY_PENDING`).
3. `datasets/L2Arctic-KoelLabs/original/` — Confirmed directory structure. Intentionally kept empty as KoelLabs is a processed derivative dataset.
4. `resources/CMUDict/cmudict.dict` (3,618,488 bytes) — Cryptographically verified (SHA-256: `81917843...`). Matches CMUSphinx master commit `74790861`.
5. `resources/CMUDict/cmudict.phones` (382 bytes) — Cryptographically verified (SHA-256: `ffb588a5...`).
6. `resources/CMUDict/cmudict.symbols` (281 bytes) — Cryptographically verified (SHA-256: `408ccaae...`).
7. `resources/CMUDict/LICENSE` (1,754 bytes) — Cryptographically verified (SHA-256: `bd4ce8e4...`).
8. `resources/MFA/acoustic-models/english_mfa.zip` (92,175,348 bytes) — Cryptographically verified (SHA-256: `a5c26c36...`). Preserved untouched.
9. `references/papers/wav2vec_2.0.pdf` (740,863 bytes, P04) — Cryptographically verified (SHA-256: `4a881708...`), 19 pages complete.
10. `references/papers/speechocean762.pdf` (512,281 bytes, P05) — Cryptographically verified (SHA-256: `5edd602a...`), 5 pages complete.

### B. Newly Acquired Resources (21 items)
1. **P01:** `references/papers/zhao2018_l2_arctic.pdf` (721,794 bytes) — Interspeech 2018 Open Access PDF from ISCA Archive. SHA-256: `03a4b836...`.
2. **P03:** `references/papers/peng2021_wav2vec2_mdd.pdf` (1,442,794 bytes) — Interspeech 2021 Open Access PDF from ISCA Archive. SHA-256: `8c56fc60...`.
3. **P06:** `references/papers/gong2022_gopt.pdf` (351,260 bytes) — ICASSP 2022 / arXiv:2205.03432 PDF. SHA-256: `44c5f27e...`.
4. **P07:** `references/papers/shahin2023_phonological_wav2vec2_mdd_preprint.pdf` (4,767,893 bytes) — arXiv:2311.07037v1 preprint PDF. SHA-256: `202d722d...`.
5. **P08:** `references/papers/tu2025_retrieval_mdd.pdf` (737,280 bytes) — arXiv:2511.20107v1 / ICASSP 2026 PDF. SHA-256: `2eb40576...`.
6. **P10:** `references/papers/radford2023_whisper.pdf` (823,459 bytes) — ICML 2023 Open Access PDF from PMLR. SHA-256: `84496ed0...`.
7. **Linguistic:** `resources/MFA/acoustic-models/english_us_arpa.zip` (91,928,208 bytes) — Official MFA release `v3.0.0` ARPAbet acoustic model. SHA-256: `d35ce271...`.
8. **Linguistic:** `resources/MFA/dictionaries/english_us_arpa.dict` (7,186,498 bytes) — Official MFA release `v3.0.0` ARPAbet dictionary. SHA-256: `e8c6c7b0...`.
9. **Linguistic:** `resources/CMUDict/README` (1,922 bytes) — Upstream CMUSphinx README. SHA-256: `00c34e75...`.
10. **Doc D01:** `references/dataset-documentation/L2-ARCTIC/L2_ARCTIC_OVERVIEW.md` (7,163 bytes).
11. **Doc D01:** `references/dataset-documentation/L2-ARCTIC/L2_ARCTIC_DOCUMENTATION.md` (9,566 bytes).
12. **Doc D01:** `references/dataset-documentation/L2-ARCTIC/l2-arctic-corpus.html` (72,391 bytes).
13. **Doc D01:** `references/dataset-documentation/L2-ARCTIC/l2-arctic-corpus-docs.html` (58,541 bytes).
14. **Doc D01:** `references/dataset-documentation/L2-ARCTIC/LICENSE.txt` (14,484 bytes, CC BY-NC 4.0).
15. **Doc D02:** `references/dataset-documentation/SpeechOcean762/README.md` (12,951 bytes).
16. **Doc D02:** `references/dataset-documentation/SpeechOcean762/openslr_101.html` (3,524 bytes).
17. **Doc D02:** `references/dataset-documentation/SpeechOcean762/LICENSE.txt` (19,301 bytes, CC BY 4.0).
18. **Doc D03:** `references/dataset-documentation/KoelLabs/KOELLABS_L2ARCTIC_CARD.md` (2,709 bytes).
19. **Doc D-Opt:** `references/dataset-documentation/MRPL2_METADATA.md` (1,950 bytes).
20. **License:** `references/licenses/CC-BY-NC-4.0.txt` (14,484 bytes).
21. **License:** `references/licenses/CC-BY-4.0.txt` (19,301 bytes).

### C. Restricted Resources Documented (3 items)
1. **P02:** `references/papers/P02_witt2000_gop_citation.md` — Witt & Young (2000), *Speech Communication* (Elsevier). Status: `ACCESS_RESTRICTED`. Full citation, DOI, and abstract preserved.
2. **P09:** `references/papers/P09_amrate2024_capt_citation.md` — Amrate & Tsai (2024), *ReCALL* (Cambridge University Press). Status: `ACCESS_RESTRICTED`. Full citation, DOI, and pedagogical review summary preserved.
3. **Corpus:** `MRPL2` — Zenodo Record `22690685`. Status: `ACCESS_RESTRICTED`. Full metadata documented; no gated downloads performed.

### D. Failed Downloads / Errors
- **None.** All attempted downloads from valid open sources succeeded with HTTP 200/302, valid byte counts, and proper PDF/text signatures.

### E. Already Present / Skipped Files
- `datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip` (Skipped re-download per idempotent rule).
- `datasets/SpeechOcean762/original/speechocean762.tar.gz` (Skipped re-download per idempotent rule).
- `references/papers/wav2vec_2.0.pdf` (Preserved intact without overwrite).
- `references/papers/speechocean762.pdf` (Preserved intact without overwrite).
- `resources/MFA/acoustic-models/english_mfa.zip` (Preserved intact without overwrite).

---

## 3. Provenance & Version Resolution

1. **CMUDict Provenance:**
   - Upstream repository: `https://github.com/cmusphinx/cmudict`
   - Master commit: `74790861f652b15e4ac49015a90074ad62a27690`
   - Verification: All dictionary files match upstream master bit-for-bit.
2. **MFA Provenance:**
   - Acoustic model: `acoustic-english_us_arpa-v3.0.0` from `MontrealCorpusTools/mfa-models`.
   - Dictionary: `dictionary-english_us_arpa-v3.0.0` from `MontrealCorpusTools/mfa-models`.
   - Phone set compatibility: Exactly matches standard ARPAbet phone set used in CMUDict and L2-ARCTIC.
3. **L2-ARCTIC Version:**
   - Verified as v5.0 (March 2020 release), containing 24 speakers across 6 L1 backgrounds (including 4 Vietnamese speakers added in v3.0, and suitcase spontaneous speech corpus added in v5.0).
4. **SpeechOcean762 Version:**
   - Verified as OpenSLR SLR101, matching GitHub repository commit `613968e3b0b789fc33936fb5eba1973176ba7d11`.

---

## 4. Readiness Assessment & Acceptance Gate

### Gate: `REFERENCE_PACK_COMPLETE`

| Criterion | Requirement | Actual Status | Evidence |
| :--- | :--- | :---: | :--- |
| **Physical Inventory** | Actual check of all Drive directories | **MET** | Complete `rclone lsjson` audit; byte-size verification. |
| **Dataset Documentation** | D01–D03 official documentation acquired | **MET** | D01 (L2-ARCTIC docs & overview), D02 (SpeechOcean762 README & OpenSLR), D03 (KoelLabs card). |
| **Mandatory Literature (P0)** | P01, P03, P04, P05 full-text; P02 restricted | **MET** | P01, P03, P04, P05 PDFs verified intact; P02 DOI and abstract documented. |
| **High-Priority Literature (P1)** | P06, P07, P08 full-text; P09 restricted | **MET** | P06, P07, P08 PDFs verified intact; P09 DOI and abstract documented. |
| **Optional Literature (P2)** | P10 full-text | **MET** | P10 (Whisper) PDF verified intact (27 pages). |
| **Linguistic Resources** | CMUDict with provenance; MFA ARPAbet pair | **MET** | CMUDict master commit verified + README; MFA `english_us_arpa` v3.0.0 model & dictionary acquired. |
| **Manifests & Reports** | All 5 required deliverable manifests generated | **MET** | All 5 files produced in `manifests/`. |
| **Non-Destructive Execution** | No overwrites of existing files; no training | **MET** | Existing PDFs and corpora preserved; ACQUISITION ONLY strictly followed. |

**Final Evaluation:** **PASS**

*(Note: `REFERENCE_PACK_COMPLETE` satisfies literature, documentation, and linguistic resource readiness. It does NOT assert `DATA_READY`, `ENV_READY`, or `READY_TO_FINE_TUNE`, which belong to subsequent audit and preparation phases.)*
