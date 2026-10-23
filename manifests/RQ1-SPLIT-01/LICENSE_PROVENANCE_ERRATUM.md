# LANGFENS — RQ1-SPLIT-01: License Provenance Erratum & Compliance Notice

**Document Type:** Formal License Erratum & Provenance Correction  
**Phase / Milestone:** RQ1-SPLIT-01 — Deterministic Speaker-Independent Split Manifest Generation  
**Engineering Leads:** Senior Speech ML Research Engineer & Research Data Engineer  
**Target Resource:** L2-ARCTIC Corpus Release v5.0 (Texas A&M University Phonetic Speech Information Lab)  
**Deliverable File Path:** `manifests/RQ1-SPLIT-01/LICENSE_PROVENANCE_ERRATUM.md`  
**Remote Sync Destination:** `kltn_gdrive:KLTN-Dataset/manifests/RQ1-SPLIT-01/LICENSE_PROVENANCE_ERRATUM.md`  
**Effective Date:** October 23, 2026 (UTC)  
**Status:** BINDING SCIENTIFIC & LEGAL ERRATUM  

---

## 1. Executive Summary

This formal erratum rectifies an erroneous license citation identified in the Project Langfens Phase `RQ1-STD-01` documentation suite. Specifically, in `manifests/RQ1-STD-01/reconciliation/SPECIFICATION_CONSISTENCY_REPORT.md` (Section 2, Line 59, Item 9), the license for the **L2-ARCTIC corpus** was erroneously recorded as **Creative Commons Attribution 4.0 International (`CC BY 4.0`)**.

The official, authoritative license designated by the dataset creators at Texas A&M University (TAMU) PSI Lab is **Creative Commons Attribution-NonCommercial 4.0 International (`CC BY-NC 4.0`)**.

This erratum establishes the definitive license provenance, details the non-commercial usage restrictions, provides the official required scientific attribution, analyzes legal implications for derived research assets, and provides an authoritative reference for all downstream project phases (`RQ1-SPLIT-01`, `RQ1-PREP-01`, `RQ1-TRN-01`, and `RQ1-EVAL-01`).

In accordance with strict reproducibility protocols, **no original archived audit reports, historical git commit logs, or raw dataset archives are altered or rewritten**. This document acts as a binding erratum that supersedes all earlier erroneous references.

---

## 2. Erratum Citation Ledger

| Metadata Field | Original Incorrect Citation | Authoritative Correct Specification | Source & Forensic Verification |
| :--- | :--- | :--- | :--- |
| **Dataset Title** | L2-ARCTIC: A Non-native English Speech Corpus | L2-ARCTIC: A Non-native English Speech Corpus | Official TAMU PSI Lab Release |
| **Applicable Version** | Release v5.0 (March 2020) | Release v5.0 (March 2020) | Archive SHA-256: `490d5a43...` |
| **License Name** | Creative Commons Attribution 4.0 International | **Creative Commons Attribution-NonCommercial 4.0 International** | Authoritative PSI Lab Repository |
| **SPDX Identifier** | `CC-BY-4.0` | **`CC-BY-NC-4.0`** | SPDX License List v3.22 |
| **Commercial Use** | Permitted (commercial rights granted) | **Prohibited (strictly NonCommercial)** | CC BY-NC 4.0 Section 2(a)(1) |
| **Primary Erroneous Source** | `RQ1-STD-01/reconciliation/SPECIFICATION_CONSISTENCY_REPORT.md` (L59) | Formally superseded by this erratum | Identified during pre-split audit |
| **Official Source URL** | https://psi.engr.tamu.edu/l2-arctic-corpus/ | https://psi.engr.tamu.edu/l2-arctic-corpus/ | Verified PSI Lab portal |
| **Repository Legal File** | `references/dataset-documentation/L2-ARCTIC/LICENSE.txt` | `references/licenses/CC-BY-NC-4.0.txt` | Both files contain CC BY-NC 4.0 |

---

## 3. Official Source & Licensing Text Analysis

### 3.1 Primary Official Source
- **Originating Institution:** Phonetic Speech Information (PSI) Lab, Department of Electrical & Computer Engineering, Texas A&M University, College Station, TX, USA.
- **Principal Investigators:** Prof. Ricardo Gutierrez-Osuna and Prof. John Levis.
- **Authoritative Distribution Portal:** https://psi.engr.tamu.edu/l2-arctic-corpus/
- **Corpus Version:** Release v5.0 (Released March 2020; packaged as `l2arctic_release_v5.0.zip`).
- **Archive Checksum (SHA-256):** `490d5a43e48b0af84f6bce8e766ad453e8cd670521d5876bb4d0741cca304fda`

### 3.2 Non-Commercial Restriction Analysis
Under the terms of the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** public license:
1. **NonCommercial Clause (Section 1(k)):**
   > *"NonCommercial means not primarily intended for or directed towards commercial advantage or monetary compensation."*
2. **Grant of Rights (Section 2(a)(1)):**
   Subject to the terms and conditions, the licensor grants a worldwide, royalty-free, non-sublicensable, non-exclusive license to reproduce and share the licensed material, in whole or in part, **for NonCommercial purposes only**, and to produce, reproduce, and share adapted material for NonCommercial purposes only.
3. **Prohibition of Commercial Exploitation:**
   - Any deployment of the raw corpus, phonetic annotations, trained model weights, acoustic alignment checkpoints, or derived pronunciation scoring APIs for commercial SaaS products, fee-charging tutoring software, or proprietary monetization is strictly prohibited without an explicit commercial license granted directly by Texas A&M University.

---

## 4. Mandatory Scholarly Attribution

All academic publications, technical reports, data sheets, and thesis defense deliverables originating from Project Langfens must cite the primary L2-ARCTIC publication:

```bibtex
@inproceedings{zhao2018l2arctic,
  author    = {Guanlong Zhao and Sinem Sonsaat and Alif Silpachai and Inas Lucic and Evgeny Chukharev-Hudilainen and John Levis and Ricardo Gutierrez-Osuna},
  title     = {{L2-ARCTIC: A Non-native English Speech Corpus}},
  year      = {2018},
  booktitle = {Proc. Interspeech 2018},
  pages     = {2783--2787},
  doi       = {10.21437/Interspeech.2018-1110},
  url       = {https://doi.org/10.21437/Interspeech.2018-1110}
}
```

In addition, derived assets must maintain the following notice:
> *"L2-ARCTIC Corpus v5.0 © 2018–2020 Texas A&M University PSI Lab, licensed under CC BY-NC 4.0. Used under academic research terms in Project Langfens."*

---

## 5. Comparative Licensing Analysis: L2-ARCTIC vs. SpeechOcean762

Project Langfens relies on two primary non-native speech corpora that carry distinct licensing terms:

| Dimension | L2-ARCTIC (v5.0) | SpeechOcean762 (SLR101) |
| :--- | :--- | :--- |
| **Licensor / Provenance** | Texas A&M University PSI Lab | Junbo Zhang et al. / OpenSLR SLR101 |
| **License Type** | **CC BY-NC 4.0** | **CC BY 4.0** |
| **Commercial Exploitation** | **STRICTLY PROHIBITED** | **PERMITTED** (with attribution) |
| **Redistribution Rights** | Permitted under non-commercial terms | Permitted (commercial or non-commercial) |
| **Modification / Adaptation** | Permitted with attribution & NC tag | Permitted with attribution |
| **Impact on Composite Models** | **Imposes Non-Commercial Restriction** | Fully open / permissive |

### The "Contagious Non-Commercial Constraint" on Composite Models
When a neural speech model (e.g. Wav2Vec 2.0 / HuBERT) is fine-tuned jointly or sequentially on both L2-ARCTIC and SpeechOcean762:
- The resulting model checkpoint incorporates learned parameters and phonetic representations derived directly from L2-ARCTIC training speech and annotations.
- Under copyright and database adaptation principles, the fine-tuned model checkpoint is considered an **Adapted Material** derived from L2-ARCTIC.
- Therefore, **the composite model checkpoint automatically inherits the CC BY-NC 4.0 Non-Commercial restriction**.
- While SpeechOcean762 itself allows commercialization, composite checkpoints containing L2-ARCTIC training signals **cannot be commercialized**.

---

## 6. Implications for Derived Research Assets in Project Langfens

1. **Academic Thesis Alignment:**
   Project Langfens is conducted exclusively as an undergraduate engineering capstone thesis (Khóa Luận Tốt Nghiệp — KLTN) at the university level. It is purely academic, educational, and open scientific research. Its operation is 100% compliant with CC BY-NC 4.0.
2. **Derived Split Manifests (`RQ1-SPLIT-01`):**
   The split manifests generated in this phase (`l2arctic_4fold_splits.json`, `l2arctic_loso_vietnamese_splits.json`) are metadata indices referencing the corpus. They are published under **CC BY-NC 4.0** and carry explicit attribution metadata.
3. **Acoustic Preprocessing & Alignment Tensors (`RQ1-PREP-01`):**
   Derived spectrograms, 16 kHz resampled WAVs, and CTC frame alignments remain governed by **CC BY-NC 4.0**.
4. **Trained Checkpoints (`RQ1-TRN-01`):**
   All weights, fine-tuned acoustic encoders, CTC heads, and MDD classification projection matrices trained on L2-ARCTIC folds must be released under **CC BY-NC 4.0** with strict NonCommercial clauses.
5. **Future Open-Source Release Safeguards:**
   Any public GitHub repository, Hugging Face Hub dataset card, or Model Hub checkpoint release must explicitly tag the repository as `license: cc-by-nc-4.0` and prominently display the non-commercial disclaimer.

---

## 7. Operational Sign-Off & Status

- **Erratum Status:** **RATIFIED & BINDING**.
- **Historical Report Modification Policy:** In strict adherence to scientific audit standards, original historical documents (`manifests/RQ1-STD-01/reconciliation/SPECIFICATION_CONSISTENCY_REPORT.md`, etc.) remain unaltered. This erratum provides the authoritative legal and scientific bridge.
- **Immediate Action:** All manifest JSON files produced in `RQ1-SPLIT-01` explicitly embed `"license": "CC BY-NC 4.0"` and `"license_source": "https://psi.engr.tamu.edu/l2-arctic-corpus/"`.
