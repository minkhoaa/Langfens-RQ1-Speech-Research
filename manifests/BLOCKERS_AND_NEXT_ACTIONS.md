# LANGFENS — RQ1-ACQ-01: Blockers, Nuances & Next Actions

**Scope:** Phase Transition & Integrity Boundary  
**Project:** LANGFENS  
**Gate:** RQ1-ACQ-01 (Research Resources Acquisition)  
**Date:** 2026-10-22  

---

## 1. Categorized Issues & Status

To maintain academic rigor and transparent engineering, all findings are categorized into strict ontological groups:

```text
Status Classification
├── True Blockers (Action required before proceeding to next gate)
├── Optional Resources & Deferred Assets (Non-blocking)
├── Access-Restricted Literature (Bibliographic metadata only)
├── Research Gaps & Linguistic Nuances (To address in model design)
└── Next Actions (Data Integrity & Annotation Audit phase)
```

---

### A. True Blockers (Blocking Issues)
- **None for Phase RQ1-ACQ-01.**
- All requirements for the `REFERENCE_PACK_COMPLETE` gate have been fulfilled. Google Drive synchronization is established and authenticated via `rclone`, and all required literature, linguistic resources, and documentation are stored in reproducible paths.

---

### B. Optional Resources & Deferred Assets (Non-Blocking)
1. **KoelLabs/L2Arctic Full Parquet Dataset:**
   - *Status:* Gated on Hugging Face (`KoelLabs/L2Arctic`, commit `e9f2aaca`).
   - *Analysis:* KoelLabs is a processed derivative (audio converted to float32 16kHz, ARPAbet converted to IPA, filtered annotations). It is **not** an independent primary speech corpus. The primary corpus is the official `l2arctic_release_v5.0.zip` (7.5 GB), which is already present on Google Drive. Therefore, omitting the KoelLabs full download does **not** block the project.
2. **MRPL2 Speech Corpus:**
   - *Status:* `ACCESS_RESTRICTED` on Zenodo (`recid: 22690685`, v1.1).
   - *Analysis:* MRPL2 contains 4,500 utterances of British English (Modern Received Pronunciation) non-native speech. It is an optional external cross-dialect evaluation candidate. It is not required for fine-tuning or primary evaluation on American English datasets (L2-ARCTIC and SpeechOcean762).
3. **Corpus Archive Full Checksums (`INTEGRITY_PENDING`):**
   - *Status:* `l2arctic_release_v5.0.zip` (7,515,454,814 bytes) and `speechocean762.tar.gz` (520,810,923 bytes) have exact byte-size matches on Google Drive metadata.
   - *Analysis:* Per Section 8 of the acquisition directives, computing full SHA-256 over 8 GB of archives is explicitly deferred to the next phase (`RQ1-AUD-01: Data Integrity & Annotation Audit`) to avoid redundant I/O and runtime overhead during acquisition.

---

### C. Access-Restricted Literature (Bibliographic Records Preserved)
1. **P02 — Witt & Young (2000), Phone-level Pronunciation Scoring and Assessment for Interactive Language Learning:**
   - *Journal:* *Speech Communication*, Elsevier.
   - *Status:* `ACCESS_RESTRICTED` (Paywalled).
   - *Handling:* Complied with Section 5 rules (no unauthorized paywall circumvention). Full DOI, bibliographic metadata, abstract, and formulation details are documented in `references/papers/P02_witt2000_gop_citation.md` and `references/paywalled_citations.txt`.
2. **P09 — Amrate & Tsai (2024), Computer-assisted pronunciation training: A systematic review:**
   - *Journal:* *ReCALL*, Cambridge University Press.
   - *Status:* `ACCESS_RESTRICTED` (Cambridge Core Subscription Gate & Cloudflare Bot Protection).
   - *Handling:* Full bibliographic metadata, abstract, and pedagogical review taxonomy are documented in `references/papers/P09_amrate2024_capt_citation.md` and `references/paywalled_citations.txt`.

---

### D. Research Gaps & Linguistic Nuances Identified

1. **ARPAbet vs. IPA Phoneme Alignment (Linguistic Nuance):**
   - *Finding:* L2-ARCTIC annotations, CMUDict, and the newly acquired MFA model (`english_us_arpa`) use standard 39-phone ARPAbet representations (with or without lexical stress numerals `0`, `1`, `2`). In contrast, SpeechOcean762 and general Wav2Vec2 phoneme recognition models frequently map to IPA or custom phone inventories.
   - *Impact on Future Phases:* A deterministic ARPAbet $\leftrightarrow$ IPA mapping table must be formally audited and locked down during the standardization phase before fine-tuning.
2. **Manual Annotations vs. Forced Alignment in L2-ARCTIC:**
   - *Finding:* In L2-ARCTIC v5.0, only 3,599 utterances across the 24 speakers were manually examined and annotated with phonetic errors (substitutions, deletions, additions). The remainder of the ~27 hours consists of automated forced alignments.
   - *Impact on Future Phases:* MDD evaluation metrics must be strictly computed on the gold manually annotated subset (or explicit canonical splits as utilized in Peng et al., 2021 and Tu et al., 2025).

---

## 2. Recommended Next Actions (Phase RQ1-AUD-01)

The following sequence of tasks should be executed in the upcoming **Data Integrity & Annotation Audit** phase:

1. **Action AUD-01 (Archive Integrity Verification):**
   - Stream or block-read `l2arctic_release_v5.0.zip` and `speechocean762.tar.gz` to compute their cryptographic SHA-256 hashes and verify ZIP/TAR central directory headers without full extraction.
2. **Action AUD-02 (L2-ARCTIC Annotation Audit):**
   - Inspect the structure of gold TextGrids for the 3,599 verified utterances across all 24 speakers (including the 4 Vietnamese speakers `TLV`, `TNI`, `TXHC`, `BWC`).
   - Audit error-tag distributions (`sub`, `del`, `add`, `err`).
3. **Action AUD-03 (SpeechOcean762 Scoring Distribution Audit):**
   - Parse `scores.json` and examine score histograms across accuracy, completeness, fluency, prosody, and total scores to detect potential rater bias or class imbalances.
4. **Action AUD-04 (MFA Verification Test):**
   - Verify compatibility of `english_us_arpa.zip` and `english_us_arpa.dict` with the MFA CLI environment (without executing alignment on the entire dataset).
5. **Action AUD-05 (Canonical Split Definition):**
   - Document and freeze speaker-independent train/dev/test splits aligned with standard literature baselines (e.g., Peng et al., 2021).
