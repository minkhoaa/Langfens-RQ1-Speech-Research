# LANGFENS — RQ1-ACQ-01: Resource Inventory

**Scope:** RQ1-ACQ-01 Research Resources Acquisition  
**Date of Audit & Acquisition:** 2026-10-22  
**Root Path on Drive:** `MyDrive/KLTN-Dataset/` (Folder ID: `1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK`)  
**Local Mirror Path:** `/home/khoa/Projects/KLTN/Dataset/`  
**Execution Lead:** Senior ML Research Engineer / Research Data Engineer  

---

## 1. Summary of Inventory

| Category | Total Tracked | Pre-Existing | Newly Acquired | Restricted / Metadata Only | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Raw Datasets (Corpora)** | 3 | 2 | 0 | 1 (KoelLabs Gated) | 2 Active / 1 Gated |
| **Linguistic Resources** | 8 | 5 | 3 | 0 | VERIFIED |
| **Research Papers** | 10 | 2 | 6 | 2 (Paywalled P02, P09) | VERIFIED / DOCUMENTED |
| **Dataset Documentation & Licenses** | 13 | 1 (partial) | 12 | 0 | VERIFIED |
| **Total Inventory Items** | **34** | **10** | **21** | **3** | **REFERENCE_PACK_COMPLETE: PASS** |

---

## 2. Detailed Resource Breakdown

### A. Raw Corpora & Datasets (`datasets/`)

| Resource ID | Path | Size (Bytes) | MIME / Type | Status | Provenance & Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `CORPUS_L2ARCTIC` | `datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip` | 7,515,454,814 | `application/zip` | `INTEGRITY_PENDING` | Official release v5.0 (TAMU PSI Lab, March 2020). 24 non-native speakers, 27.1h speech, 3,599 gold verified TextGrids. Byte count matches Google Drive snapshot exactly. Full SHA-256 deferred to Audit phase per rule. |
| `CORPUS_SO762` | `datasets/SpeechOcean762/original/speechocean762.tar.gz` | 520,810,923 | `application/x-gzip` | `INTEGRITY_PENDING` | OpenSLR SLR101 release (Zhang et al., 2021). 5,000 utterances, multi-aspect scores, 5 raters. Exact size match with Drive metadata. Full SHA-256 deferred to Audit phase per rule. |
| `CORPUS_KOELLABS` | `datasets/L2Arctic-KoelLabs/original/` | 0 | Directory | `GATED_EMPTY` | KoelLabs processed derivative (Parquet/16kHz/IPA). Gated on Hugging Face (`KoelLabs/L2Arctic`, commit `e9f2aaca`). Directory intentionally empty for acquisition phase; documentation acquired in Phase B. |

---

### B. Linguistic Resources (`resources/`)

| Resource ID | Path | Size (Bytes) | SHA-256 Checksum | Status | Provenance & Upstream Version |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `LING_CMU_DICT` | `resources/CMUDict/cmudict.dict` | 3,618,488 | `81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22` | `VERIFIED` | CMUSphinx cmudict master (commit `74790861f652b15e4ac49015a90074ad62a27690`). Standard ARPAbet pronunciations. Pre-existing. |
| `LING_CMU_PHONES` | `resources/CMUDict/cmudict.phones` | 382 | `ffb588a5e55684723582c7256e1d2f9fadb130011392d9e59237c76e34c2cfd6` | `VERIFIED` | CMUSphinx cmudict phone definitions. Pre-existing. |
| `LING_CMU_SYMBOLS` | `resources/CMUDict/cmudict.symbols` | 281 | `408ccaae803641c6d7b626b6299949320c2dbca96b2220fd3fb17887b023b027` | `VERIFIED` | CMUSphinx cmudict symbol table. Pre-existing. |
| `LING_CMU_LICENSE` | `resources/CMUDict/LICENSE` | 1,754 | `bd4ce8e44170a5f9f481310ca85c51de3c4f851a65e679b40e603b143bd3542a` | `VERIFIED` | CMUDict BSD-style license. Pre-existing. |
| `LING_CMU_README` | `resources/CMUDict/README` | 1,922 | `00c34e7564f1f6a68de02e12c123d801471da92bc3091f7d89b605f238bf8554` | `VERIFIED` | Upstream official README from CMUSphinx repository. **Newly acquired**. |
| `LING_MFA_EN_LEGACY` | `resources/MFA/acoustic-models/english_mfa.zip` | 92,175,348 | `a5c26c360b741e9a8e692416d942dae1fb2a8957dccc0826c4d6a87c98945f73` | `VERIFIED` | Montreal Forced Aligner IPA acoustic model asset. Pre-existing on Drive, preserved untouched. |
| `LING_MFA_ARPA_AM` | `resources/MFA/acoustic-models/english_us_arpa.zip` | 91,928,208 | `d35ce271ded357d833d2f4b8d1041dc3748b9538567ba13f2c697f4e4126711b` | `VERIFIED` | MFA official release `acoustic-english_us_arpa-v3.0.0`. ARPAbet acoustic model for US English. **Newly acquired**. |
| `LING_MFA_ARPA_DICT` | `resources/MFA/dictionaries/english_us_arpa.dict` | 7,186,498 | `e8c6c7b036ae2b7c78d2768b8dc6b1f9359175b842956d00b48c53c9c332e6b0` | `VERIFIED` | MFA official release `dictionary-english_us_arpa-v3.0.0`. Pronunciation dictionary matching ARPAbet phone set. **Newly acquired**. |

---

### C. Research Literature (`references/papers/`)

| ID | Filename / Path | Size (Bytes) | SHA-256 Checksum | Status | Publication & Reference Info |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `P01` | `references/papers/zhao2018_l2_arctic.pdf` | 721,794 | `03a4b836bca32e6bdf3439bc363f8dbc1610d2186c577f492f4d6c243da5a022` | `VERIFIED` | Zhao et al. (2018), *L2-ARCTIC: A Non-native English Speech Corpus*, Interspeech 2018, pp. 2783-2787. DOI: 10.21437/Interspeech.2018-1110. **Newly acquired**. |
| `P02` | `references/papers/P02_witt2000_gop_citation.md` | 1,423 | `f42b322a...` | `ACCESS_RESTRICTED` | Witt & Young (2000), *Phone-level Pronunciation Scoring and Assessment for Interactive Language Learning*, Speech Communication. DOI: 10.1016/S0167-6393(99)00044-8. Paywalled. Full metadata & abstract documented. |
| `P03` | `references/papers/peng2021_wav2vec2_mdd.pdf` | 1,442,794 | `8c56fc608b97146b401aa86ddd5390b405e9809ed31ca3f4ac73d267f4aacce9` | `VERIFIED` | Peng et al. (2021), *A Study on Fine-Tuning wav2vec2.0 Model for the Task of Mispronunciation Detection and Diagnosis*, Interspeech 2021, pp. 4448-4452. DOI: 10.21437/Interspeech.2021-381. **Newly acquired**. |
| `P04` | `references/papers/wav2vec_2.0.pdf` *(alias: baevski2020_wav2vec2.pdf)* | 740,863 | `4a881708378143547a5c053f62090104afb49782a4c71d78441dde0a818758ff` | `VERIFIED` | Baevski et al. (2020), *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*, NeurIPS 2020 / arXiv:2006.11477. Pre-existing. |
| `P05` | `references/papers/speechocean762.pdf` *(alias: zhang2021_speechocean762.pdf)* | 512,281 | `5edd602a0ee383e3dc20dcc6a462541dd6daf0c2f22c2b3d5d877770391bd657` | `VERIFIED` | Zhang et al. (2021), *speechocean762: An Open-Source Non-native English Speech Corpus for Pronunciation Assessment*, Interspeech 2021 / arXiv:2104.01378. Pre-existing. |
| `P06` | `references/papers/gong2022_gopt.pdf` | 351,260 | `44c5f27ea0977b40396e6fc538a1636074b1484c6b574b968997789b0ceb851c` | `VERIFIED` | Gong et al. (2022), *Transformer-Based Multi-Aspect Multi-Granularity Non-Native English Speaker Pronunciation Assessment*, ICASSP 2022 / arXiv:2205.03432. **Newly acquired**. |
| `P07` | `references/papers/shahin2023_phonological_wav2vec2_mdd_preprint.pdf` | 4,767,893 | `202d722d4b0c15705c1bc0a5f11c924f13cec136705e4b05410ff32a79f4d456` | `VERIFIED` | Shahin et al., *Phonological Level wav2vec2-based Mispronunciation Detection and Diagnosis Method*, arXiv:2311.07037v1. Formal publication DOI: 10.1016/j.specom.2025.103249. **Newly acquired**. |
| `P08` | `references/papers/tu2025_retrieval_mdd.pdf` | 737,280 | `2eb40576a3c2f4bf59f1471c8bb3c240711d0f13d9af31c19e0454532fac65d1` | `VERIFIED` | Tu et al. (2025/2026), *Mispronunciation Detection and Diagnosis Without Model Training: A Retrieval-Based Approach*, arXiv:2511.20107 / IEEE ICASSP 2026 (DOI: 10.1109/ICASSP55912.2026.11462649). **Newly acquired**. |
| `P09` | `references/papers/P09_amrate2024_capt_citation.md` | 1,874 | `b18d89e1...` | `ACCESS_RESTRICTED` | Amrate & Tsai (2024), *Computer-assisted pronunciation training: A systematic review*, ReCALL, Cambridge University Press. DOI: 10.1017/S0958344024000181. Paywalled / Cloudflare protected. Full metadata & pedagogical framework documented. |
| `P10` | `references/papers/radford2023_whisper.pdf` | 823,459 | `84496ed07ac01bf3d00866a1934f81173ddbba99f7488de59bb8ae3b28f0f85e` | `VERIFIED` | Radford et al. (2023), *Robust Speech Recognition via Large-Scale Weak Supervision*, ICML 2023 (PMLR 202:28492-28518). **Newly acquired**. |

---

### D. Dataset Documentation, Licenses & Research Notes (`references/`)

| Path | Size (Bytes) | SHA-256 Checksum | Status | Description & Source |
| :--- | :---: | :---: | :---: | :--- |
| `references/dataset-documentation/L2-ARCTIC/L2_ARCTIC_OVERVIEW.md` | 7,163 | `6ae4f9...` | `VERIFIED` | Official corpus overview, speaker L1 table, version history v1.0–v5.0 from TAMU PSI Lab. |
| `references/dataset-documentation/L2-ARCTIC/L2_ARCTIC_DOCUMENTATION.md` | 9,566 | `6c10eb...` | `VERIFIED` | Comprehensive documentation: annotation schema, phone statistics, Praat TextGrid structure. |
| `references/dataset-documentation/L2-ARCTIC/l2-arctic-corpus.html` | 72,391 | `e437b0...` | `VERIFIED` | Raw HTML archive of `https://psi.engr.tamu.edu/l2-arctic-corpus/`. |
| `references/dataset-documentation/L2-ARCTIC/l2-arctic-corpus-docs.html` | 58,541 | `44ce1f...` | `VERIFIED` | Raw HTML archive of `https://psi.engr.tamu.edu/l2-arctic-corpus-docs/`. |
| `references/dataset-documentation/L2-ARCTIC/LICENSE.txt` | 14,484 | `12cf2c...` | `VERIFIED` | Legal text of Creative Commons CC BY-NC 4.0 International license. |
| `references/dataset-documentation/SpeechOcean762/README.md` | 12,951 | `a7d187...` | `VERIFIED` | Official repository README from `jimbozhang/speechocean762` (commit `613968e3`). |
| `references/dataset-documentation/SpeechOcean762/openslr_101.html` | 3,524 | `1ca1f8...` | `VERIFIED` | Raw HTML mirror of OpenSLR SLR101 entry page. |
| `references/dataset-documentation/SpeechOcean762/LICENSE.txt` | 19,301 | `7b3223...` | `VERIFIED` | Creative Commons CC BY 4.0 International legal license text. |
| `references/dataset-documentation/KoelLabs/KOELLABS_L2ARCTIC_CARD.md` | 2,709 | `05e94b...` | `VERIFIED` | Full metadata card, schema, audio normalization info, IPA conversion, split distribution. |
| `references/dataset-documentation/MRPL2_METADATA.md` | 1,950 | `705822...` | `VERIFIED` | Zenodo record metadata, MRP British English dialect info, evaluation potential. |
| `references/licenses/CC-BY-NC-4.0.txt` | 14,484 | `12cf2c...` | `VERIFIED` | Canonical copy of CC BY-NC 4.0 license for L2-ARCTIC and KoelLabs. |
| `references/licenses/CC-BY-4.0.txt` | 19,301 | `7b3223...` | `VERIFIED` | Canonical copy of CC BY 4.0 license for SpeechOcean762 and MRPL2. |
| `references/paywalled_citations.txt` | 809 | `da7e84...` | `VERIFIED` | Citations register for paywalled/restricted research items (P02, P09). |
