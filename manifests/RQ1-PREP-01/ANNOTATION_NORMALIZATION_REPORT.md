# LANGFENS — RQ1-PREP-01: Annotation Normalization & Phoneme Mapping Report

**Execution Gate:** RQ1-PREP-01 — Phase B Non-Destructive Normalization  
**Corpus Targets:** L2-ARCTIC Release v5.0 and SpeechOcean762  
**Deliverable Associated Path:** `manifests/RQ1-PREP-01/ANNOTATION_NORMALIZATION_REPORT.md`  
**Synced Location:** `kltn_gdrive:KLTN-Dataset/manifests/RQ1-PREP-01/`  
**Schema Compliance:** `manifests/RQ1-STD-01/PHONEME_MAPPING_SPEC.md`

---

## 1. Executive Summary

Phonetic annotations across L2-ARCTIC and SpeechOcean762 originate from different acoustic annotation frameworks: Praat TextGrids with non-native pronunciation tags vs. Kaldi-style phone alignments with scalar grading.

To enable robust acoustic model training and zero-shot transfer without data corruption, all phonetic labels pass through a **non-destructive, strictly reversible normalization layer**. Original labels and transformation metadata are preserved in each record.

---

## 2. Phoneme Inventory Standardization (39-Class ARPAbet)

### 2.1 The Standard 39-Phoneme Inventory
Project Langfens adopts the standard 39-class CMU ARPAbet inventory as the primary phonemic diagnosis space:

```text
Vowels (15):
  AA, AE, AH, AO, AW, AY, EH, ER, EY, IH, IY, OW, OY, UH, UW

Consonants (24):
  Stops (6):       B, D, G, P, T, K
  Fricatives (8):   DH, F, HH, S, SH, TH, V, Z, ZH
  Affricates (2):   CH, JH
  Nasals (3):       M, N, NG
  Liquids (2):      L, R
  Glides (2):       W, Y
```

### 2.2 Reversible Transformation Operations

| Raw Annotation Feature | Normalized Representation | Metadata Recorded | Reversibility Guarantee |
| :--- | :--- | :--- | :--- |
| **Stress Digits (`0`, `1`, `2`)** | Stripped from base phone (e.g., `AH0` $\to$ `AH`) | `stress: Optional[int]` | Exact (`reconstruct()` appends stress digit) |
| **Allophonic Asterisk (`*`)** | Stripped from base phone (e.g., `T*` $\to$ `T`) | `has_asterisk: bool` | Exact (`reconstruct()` appends `*`) |
| **Unstressed Mid-Central Schwa (`AX`)** | Folded to `AH` (per CMUDict standard) | `was_ax: bool` | Exact (`reconstruct()` restores `AX`) |
| **Distorted / Uncertain (`err`)** | Mapped to `<ERR>` sentinel; masked from diagnosis | `is_err: bool` | Exact (restores `err`) |
| **Keyboard Slip Typo (`V8`)** | Normalized to `V` (Speaker SVBI keyboard slip) | `is_typo_correction: bool` | Exact (restores `V8`) |
| **Acoustic Silence (`sil`, `sp`, `""`)** | Designated as `sil` (non-canonical pause) | `error_type: "silence"` | Separated from CTC `<BLANK>` |

---

## 3. Disambiguation: CTC `<BLANK>` vs. Acoustic Silence vs. Deletions

A critical failure mode in speech recognition MDD pipelines is confusing the CTC emission blank with acoustic silence or phone deletions. The Langfens data contract enforces strict mathematical disambiguation:

```text
Phonetic & Acoustic State Disambiguation
├── CTC <BLANK> (Index 0)
│   └── Reserved strictly for CTC emission blank. Never emitted for acoustic silence.
│
├── Acoustic Silence (sil / sp / spn)
│   └── Represents non-speech acoustic intervals (pauses, leading/trailing room tone).
│   └── Excluded from canonical prompt denominator (N_canonical).
│
├── Phonemic Deletion (CPL,sil,d)
│   └── Prompt phoneme that the speaker omitted.
│   └── Included in canonical prompt denominator (N_canonical).
│   └── Assigned binary label b_i = 1 (Error).
│   └── Has NO realized phone target in CTC alignment (length zero in realization).
│
└── Epenthetic Addition (sil,PPL,a)
    └── Unprompted acoustic phone inserted by learner.
    └── Excluded from canonical prompt denominator (N_canonical).
    └── Tracked as independent insertion error event.
```

---

## 4. International Phonetic Alphabet (IPA) Display Mappings

For user display and diagnosis visualization, normalized ARPAbet tokens map deterministically to Unicode IPA symbols:

| ARPAbet | IPA | ARPAbet | IPA | ARPAbet | IPA | ARPAbet | IPA |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `AA` | ɑ | `DH` | ð | `K` | k | `T` | t |
| `AE` | æ | `EH` | ɛ | `L` | l | `TH` | θ |
| `AH` | ʌ | `ER` | ɝ | `M` | m | `UH` | ʊ |
| `AO` | ɔ | `EY` | eɪ | `N` | n | `UW` | u |
| `AW` | aʊ | `F` | f | `NG` | ŋ | `V` | v |
| `AY` | aɪ | `G` | ɡ | `OW` | oʊ | `W` | w |
| `B` | b | `HH` | h | `OY` | ɔɪ | `Y` | j |
| `CH` | tʃ | `IH` | ɪ | `P` | p | `Z` | z |
| `D` | d | `IY` | i | `R` | ɹ | `ZH` | ʒ |
| `JH` | dʒ | `S` | s | `SH` | ʃ | — | — |

---

## 5. Lossy Mapping Audit

The only lossy transformation is when downstream neural acoustic representations discard lexical stress digits and asterisk flags during loss calculation. Because every `LangfensUtteranceRecord` stores both raw tokens (`raw_token`) and normalized representations alongside boolean flags (`has_asterisk`, `was_ax`, `stress`), zero linguistic information is permanently lost on disk.
