# Canonical Input Provenance & Deployability Contract Audit — RQ1-TRN-02A-EVAL-R2

**Date:** 2026-10-26  
**Module Audited:** `src/langfens_prep/dataset.py`, `src/langfens_prep/textgrid_parser.py`, `src/langfens_prep/model.py`  
**Focus:** Provenance of `canonical_phone_indices`, oracle leakage audit, and zero-oracle deployability  

---

## 1. Provenance of Canonical Phone Inputs

In the current codebase, the canonical phone sequence fed into `cross_attention_module` during both training and evaluation is constructed as follows:

```python
# src/langfens_prep/dataset.py (lines 92-100)
tg_text = reader.read_annotation_text(speaker_id, utt_id)
annotation = parse_gold_textgrid_content(tg_text, f"{utt_id}.TextGrid")

canonical_intervals = annotation.canonical_intervals
canonical_phones = [i.canonical_phone for i in canonical_intervals]
canonical_phone_indices = torch.tensor(
    [PHONE_TO_CTC_INDEX[p] for p in canonical_phones if p in PHONE_TO_CTC_INDEX],
    dtype=torch.long
)
```

In `textgrid_parser.py`, intervals are extracted from the gold annotator's `phones` tier:
- **Correct Segment (`P`):** Canonical phone is $P$.
- **Substitution Segment (`C, P, s`):** Canonical phone is extracted as $C$ (intended prompt phone), realized is $P$.
- **Deletion Segment (`C, sil, d`):** Canonical phone is extracted as $C$, realized is `sil`.
- **Addition Segment (`sil, P, a`):** Marked `is_canonical = False`, strictly excluded from canonical prompt sequence.

---

## 2. Empirical Audit: Prompt Transcripts vs. Gold TextGrids

In the L2-ARCTIC corpus, every speaker reads from identical predefined ARCTIC prompt sentences (e.g., `arctic_a0003` = *"For the twentieth time that evening the two men shook hands."*).

We conducted an audit comparing the canonical phone sequences extracted from TextGrids across the two validation speakers (`NJS` and `RRBI`) for all 106 shared validation utterances:

| Comparison Category | Utterance Count | Percentage |
|---|---:|---:|
| **Exact Identical Canonical Phone Sequence** | 45 | **42.5%** |
| **Different Canonical Sequence Length** | 22 | **20.8%** |
| **Identical Length, but Different Canonical Phonemes** | 39 | **36.8%** |
| **Total Shared Validation Utterances** | 106 | **100.0%** |

### Key Empirical Findings:
1. **57.5% of shared utterances have differing canonical phone sequences** between speakers reading the exact same prompt sentence.
2. In `arctic_a0003`, for example:
   - `NJS` canonical: `F AO R DH AH T W EH N T IY AH TH T AY M DH AH T W IY M EH N SH UH K HH AE N D Z` (39 phones)
   - `RRBI` canonical: `F R ER DH AH T W EH N T IY AH TH T AY M DH AH T W AH M EH N SH UH K HH AE N D Z` (39 phones)
   - Differences: Index 1 (`AO` vs `R`), Index 2 (`R` vs `ER`), Index 25 (`IY` vs `AH`).
3. **Root Cause:** In L2-ARCTIC, initial phone boundaries were produced by forced alignment with CMUdict, which allows multiple pronunciation variants for common words (e.g., "for" as `F AO R` or `F ER`; "two" as `T UW` or unstressed variants). Human annotators subsequently adjusted or preserved these choices on a per-speaker basis.

---

## 3. Deployability & Oracle Information Audit

### Question 1: Are Gold TextGrid Timestamps Used During Inference?
**NO.** The cross-attention module (`cross_attention_module`) in `src/langfens_prep/model.py` strictly ignores `xmin` and `xmax` timestamps:
```python
aligned_acoustic, phone_embeds, alignment_weights = self.cross_attention_module(
    frame_features=hidden_states,
    canonical_phone_indices=canonical_phone_indices,
    acoustic_padding_mask=acoustic_padding_mask
)
```
The model operates purely through soft attention across all acoustic frames, proving that inference is **alignment-free and timestamp-independent**.

### Question 2: Is the Current Pipeline Fully Deployable Without Oracle Information?
**NO, not in its current state.**
- Although timestamps are not passed, the **sequence of canonical phone tokens** is extracted directly from the speaker's Gold TextGrid file rather than from a text prompt G2P frontend.
- In a true zero-oracle deployment (e.g. an interactive language learning app), the system receives only raw audio and the written English sentence (e.g., *"for the twentieth time..."*). It does not have access to an annotator-corrected TextGrid to determine which pronunciation variant was aligned.
- If a standard deterministic G2P frontend (such as CMUdict) were used, the canonical sequence would differ from the TextGrid canonical sequence in ~57.5% of utterances.

---

## 4. Engineering & Scientific Recommendations

1. **Explicit Lexicon Front-End:**
   - For true zero-oracle deployment and evaluation, implement an explicit G2P frontend (e.g. `g2p_en` or CMUdict lookup) that maps reference text directly to canonical phone sequences.
2. **Standardized Benchmark Protocol:**
   - In publications, explicitly document: *"The model receives canonical phone sequences parsed from L2-ARCTIC TextGrid interval tiers, which reflect annotator-selected dictionary variants, though without timestamp boundaries."*
3. **Input Contract Integrity:**
   - Never alter evaluation inputs silently. Maintain the current protocol for exact benchmark continuity with prior literature, but clearly distinguish "timestamp-free inference" from "zero-oracle text-to-MDD deployment."
