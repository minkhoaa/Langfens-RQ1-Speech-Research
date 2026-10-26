# Canonical Sequence Difference Audit: G2P Frontend vs. Gold TextGrids — RQ1-TRN-02A-EVAL-R2-CORR

**Author:** Senior Speech ML Evaluation & Reproducibility Auditor  
**Date:** 2026-10-26  
**Subject:** Empirical sequence-level audit comparing deterministic CMUdict G2P phonemization against human annotator Gold TextGrids  

---

## 1. Executive Summary

This audit establishes the exact numerical divergence between a deterministic, text-only G2P frontend (CMUdict v0.7b primary pronunciations) and the canonical phone sequences extracted from L2-ARCTIC Gold TextGrids across the 300 Fold 1 validation utterances (`NJS` and `RRBI`).

### Key Findings:
1. **72.67% of validation utterances** (218 / 300) have canonical phoneme sequences that differ between deterministic CMUdict G2P and the annotator's Gold TextGrid. Only **27.33%** (82 / 300) are exact bit-for-bit matches.
2. Across all 9,956 validation phone tokens, the phone-level Levenshtein edit rate is **4.22%** (420 total edits).
3. The vast majority of edits are **pronunciation variant substitutions (3.45%)**, such as reduction of vowels in function words (`the`, `for`, `was`, `that`). Length mismatches occur in **17.67%** of utterances, with insertions and deletions each accounting for $< 0.40\%$ of tokens.
4. This actual G2P-to-Gold divergence (**72.67%**) is distinct from and larger than the inter-speaker Gold TextGrid divergence (**57.55%**).

---

## 2. Quantitative Sequence Comparison Across Validation Split

| Dimension | Overall Validation (300 utts) | Speaker NJS (150 utts) | Speaker RRBI (150 utts) |
|---|---:|---:|---:|
| **Exact Sequence Matches** | **82 / 300 (27.33%)** | 34 / 150 (22.67%) | 48 / 150 (32.00%) |
| **Differing Sequences** | **218 / 300 (72.67%)** | 116 / 150 (77.33%) | 102 / 150 (68.00%) |
| **Sequence Length Discrepancies** | **53 / 300 (17.67%)** | 23 / 150 (15.33%) | 30 / 150 (20.00%) |
| **Identical Length but Divergent Phones** | **165 / 300 (55.00%)** | 93 / 150 (62.00%) | 72 / 150 (48.00%) |
| **Total Gold Phone Tokens** | 9,956 | 5,011 | 4,945 |
| **Total CMUdict G2P Phone Tokens** | 9,953 | 5,009 | 4,944 |
| **Token Ratio (G2P / Gold)** | 0.9997 | 0.9996 | 0.9998 |
| **Total Levenshtein Edits** | **420 (Edit Rate: 4.22%)** | **246 (Edit Rate: 4.91%)** | **174 (Edit Rate: 3.52%)** |
| **- Phoneme Substitutions** | 343 (3.45%) | 206 (4.11%) | 137 (2.77%) |
| **- Phone Deletions (Omitted by G2P)** | 40 (0.40%) | 21 (0.42%) | 19 (0.38%) |
| **- Phone Insertions (Extra in G2P)** | 37 (0.37%) | 19 (0.38%) | 18 (0.36%) |

---

## 3. Inter-Speaker Gold TextGrid Comparison (106 Shared Sentences)

To separate frontend dictionary variation from human annotator variability, we evaluated the 106 prompt sentences that appear in both the NJS and RRBI validation partitions:

| Comparison Category | Shared Utterances Count | Percentage of Shared Sentences |
|---|---:|---:|
| **Exact Identical Gold Sequence Across Both Speakers** | 45 | **42.45%** |
| **Different Sequence Length in Gold TextGrids** | 22 | **20.75%** |
| **Identical Length but Divergent Canonical Phonemes** | 39 | **36.79%** |
| **Total Inter-Speaker Divergent Sentences** | **61** | **57.55%** |

### Why G2P Divergence Differs from Inter-Speaker Divergence:
- **Inter-speaker divergence (57.55%):** Reflects annotator subjectivity and forced-alignment variability between two speakers pronouncing the same sentence.
- **G2P-to-Gold divergence (72.67%):** Reflects the distance between an objective, static lexicon (CMUdict v0.7b) and acoustic-contextual annotator decisions across both speakers.
- Inferring G2P frontend error from inter-speaker divergence is statistically invalid because an objective dictionary is fixed, whereas annotator decisions vary per speaker.

---

## 4. Frequent Phonological Discrepancy Patterns

Inspection of the 343 substitution edits reveals four systematic phonetic divergence categories:

1. **Vowel Reduction in Function Words:**
   - Word *"the"*: CMUdict primary is `DH AH`. Gold TextGrids frequently transcribe `DH IY` (especially when preceding vowel-initial words or in stressed positions).
   - Word *"that"*: CMUdict primary is `DH AE T`. Gold TextGrids frequently transcribe the reduced form `DH AH T`.
   - Word *"for"*: CMUdict primary is `F AO R`. Gold TextGrids frequently transcribe `F ER` or rhotic vowel `F R ER`.
2. **Unstressed Syllable Allophones:**
   - High front vowels in unstressed suffixes (e.g. *-ic*, *-ed*, *-es*): CMUdict alternates between `IH` and `AH`, whereas annotators aligned to acoustic formants.
3. **Compound & Contraction Handling:**
   - Words with apostrophes or compound nouns (e.g., *twentieth*, *don't*): CMUdict aligns as full standard forms, while annotators sometimes collapsed unstressed diphthongs.

---

## 5. Architectural Implications

- The small aggregate edit rate (**4.22%**) confirms that CMUdict v0.7b provides an extraordinarily accurate proxy for canonical English prompts.
- However, the high sentence-level divergence (**72.67%**) proves that benchmark evaluations that feed Gold TextGrid intervals to the model are benefiting from oracle dictionary selection.
- In all subsequent phases, Protocol O and Protocol Z must be evaluated in parallel.
