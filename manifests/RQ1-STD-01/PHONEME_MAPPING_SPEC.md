# LANGFENS — RQ1-STD-01: Cross-Resource Phoneme Standardization Specification

**Execution Gate:** RQ1-STD-01 — Data Standardization & Evaluation Protocol Specification  
**Focus Area:** Task B — Reversible Multi-Layer Phoneme Representations & CTC Mapping  
**Author / Engineering Lead:** Senior Speech ML Research Engineer & Phonetics Specialist  
**Evaluated Resources:**
1. **CMUDict v0.7b** (Master commit `74790861`, CMUSphinx)
2. **SpeechOcean762** (OpenSLR SLR101 / `scores.json` / `lexicon.txt`)
3. **L2-ARCTIC Release v5.0** (3,599 Gold Manual Praat TextGrids)
4. **MFA English US ARPA** (`english_us_arpa.dict` v3.0.0, Montreal Corpus Tools)
**Deliverable Associated Path:** `manifests/RQ1-STD-01/PHONEME_MAPPING_SPEC.md`  
**Synced Location:** `kltn_gdrive:KLTN-Dataset/manifests/RQ1-STD-01/`  
**Source Reports Referenced:**
- `manifests/RQ1-AUD-01/reconciliation/PHONEME_INVENTORY_RECONCILIATION.md`
- `manifests/RQ1-AUD-01/reconciliation/SCIENTIFIC_CONSISTENCY_REPORT.md`
- `manifests/RQ1-AUD-01/PHONEME_COMPATIBILITY_AUDIT.md`

---

## 1. Executive Summary & Harmonization Architecture

A critical vulnerability in multi-corpus speech ML research is phonetic inventory incompatibility: symbol naming collisions, mismatched stress numerals, silence token divergence, corrupted tokenizer vocabularies, and lossy transformations.
This specification formally designs a **lossless, reversible, multi-layer phoneme standardization architecture** for Project Langfens.

### Core Architectural Guarantees
1. **The Shared 39-Phoneme Core:** CMUDict v0.7b, SpeechOcean762, L2-ARCTIC, and Montreal Forced Aligner (MFA) strictly share the standard 39 American English phonemic categories (15 vowels, 24 consonants).
2. **MFA v3.0 Dictionary Disambiguation:** Upstream `english_us_arpa.dict` entries include four acoustic probability floating-point fields (`pron_prob`, `silence_after_prob`, `silence_before_prob`, `duration_scale`). Filtering numeric columns reveals **exactly 70 clean tokens (69 ARPAbet stress variants + `spn`)**. There are **zero** syllable boundary dots.
3. **SpeechOcean762 67 Phone Tokens:** Consists of 24 consonants, 41 stressed vowels, and 2 unstressed offglide vowels (`IH`, `UH` without digits) used for centring diphthongs (`HERE`, `YOUR`).
4. **Schwa (`AX`) vs. `AH0` Separation:** L2-ARCTIC's explicit `AX` phone is preserved losslessly in raw and diagnostic layers, with deterministic mapping to `AH0` for standard 39-phone acoustic models.
5. **CTC Target Independence:** CTC `<BLANK>` emissions are mathematically separated from acoustic silence (`<SIL>`) and canonical phone deletions (`CPL,sil,d`).

---

## 2. Multi-Layer Phoneme Representation Architecture

Project Langfens organizes all phonetic data into four strictly bounded layers:

```text
Multi-Layer Phone Architecture
├── Layer 0: Raw Immutable Storage (Preserved 100% untouched)
│   ├── L2-ARCTIC: Praat TextGrids (39 phones + AX + tags 's'/'d'/'a' + typos)
│   ├── SpeechOcean762: JSON (67 phones + expert bracket notes)
│   └── CMUDict / MFA: Raw lexicon files
│
├── Layer 1: Canonical Base Phoneme Space (39 Base Phonemes)
│   ├── Target for ASR pre-training, GOP scoring, and canonical sequence encoding
│   ├── Vowels stripped of stress numerals: {AA, AE, AH, ..., UW}
│   ├── Consonants: {B, CH, D, ..., ZH}
│   └── Normalization: AX -> AH; Typos normalized; Lowercase uppercased
│
├── Layer 2: Lexical Stress Space (69 Phone Classes)
│   ├── Target for Word Stress Assessment on SpeechOcean762
│   ├── 15 vowels × 3 stress levels {0, 1, 2} = 45 vowel classes
│   ├── 24 consonants (stress-free)
│   └── SpeechOcean unstressed offglides IH, UH -> mapped to IH0, UH0
│
└── Layer 3: Diagnostic CTC Space (43 Classes)
    ├── Target for Non-Native Realized Phone Sequence Prediction & MDD Diagnosis
    ├── Index 0:  <BLANK>      (CTC Alignment Blank)
    ├── Index 1:  <SIL>        (Acoustic Silence Interval)
    ├── Index 2:  <UNK_PHONE>  (Captures 269 'err' tags and unresolvable sounds)
    ├── Index 3:  AX           (Preserved diagnostic schwa)
    └── Index 4..42: The 39 Core Base Phonemes
```

---

## 3. The 39 Core ARPAbet Phonemes & Exhaustive IPA Mapping

The table below defines the complete 39-phone core inventory, their phonetic classification, standard Unicode IPA transcription, example English words, and presence across all four project resources:

| Phone | Phonetic Category | Manner / Articulation | IPA Unicode | Example Word | CMUDict | SpeechOcean762 | L2-ARCTIC | MFA ARPA |
| :---: | :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| `AA` | Low back unrounded vowel | Open back unrounded | ɑ | b**al**m, f**a**ther | Stressed | Stressed | Stressed | Stressed |
| `AE` | Low front unrounded vowel | Near-open front | æ | b**a**t, c**a**t | Stressed | Stressed | Stressed | Stressed |
| `AH` | Mid central unrounded vowel | Open-mid back / Central | ʌ / ə | b**u**tt, c**u**p | Stressed | Stressed | Stressed | Stressed |
| `AO` | Low-mid back rounded vowel | Open-mid back rounded | ɔ | c**augh**t, l**aw** | Stressed | Stressed | Stressed | Stressed |
| `AW` | Diphthong | Open front to near-close back | aʊ | b**ou**t, c**ow** | Stressed | Stressed | Stressed | Stressed |
| `AY` | Diphthong | Open front to near-close front | aɪ | b**i**te, m**y** | Stressed | Stressed | Stressed | Stressed |
| `B` | Voiced bilabial stop | Plosive | b | **b**et, ro**b** | Present | Present | Present | Present |
| `CH` | Voiceless postalveolar affricate | Affricate | tʃ | **ch**in, ma**tch** | Present | Present | Present | Present |
| `D` | Voiced alveolar stop | Plosive | d | **d**ebt, be**d** | Present | Present | Present | Present |
| `DH` | Voiced dental fricative | Fricative | ð | **th**is, ba**the** | Present | Present | Present | Present |
| `EH` | Mid front unrounded vowel | Open-mid front | ɛ | b**e**t, s**e**t | Stressed | Stressed | Stressed | Stressed |
| `ER` | Rhoticized central vowel | R-colored mid-central | ɝ / ɚ | b**ir**d, h**er**d | Stressed | Stressed | Stressed | Stressed |
| `EY` | Diphthong | Close-mid to near-close front | eɪ | b**ai**t, m**a**ke | Stressed | Stressed | Stressed | Stressed |
| `F` | Voiceless labiodental fricative | Fricative | f | **f**at, lau**gh** | Present | Present | Present | Present |
| `G` | Voiced velar stop | Plosive | ɡ | **g**o, ba**g** | Present | Present | Present | Present |
| `HH` | Voiceless glottal transition | Fricative | h | **h**at, a**h**ead | Present | Present | Present | Present |
| `IH` | High-mid front lax vowel | Near-close near-front | ɪ | b**i**t, s**i**t | Stressed | Stressed + Offglide | Stressed | Stressed |
| `IY` | High front tense vowel | Close front | i | b**ea**t, s**ee** | Stressed | Stressed | Stressed | Stressed |
| `JH` | Voiced postalveolar affricate | Affricate | dʒ | **j**am, e**dg**e | Present | Present | Present | Present |
| `K` | Voiceless velar stop | Plosive | k | **k**ick, **c**at | Present | Present | Present | Present |
| `L` | Alveolar lateral liquid | Lateral approximant | l / ɫ | **l**et, coo**l** | Present | Present | Present | Present |
| `M` | Bilabial nasal | Nasal | m | **m**at, ca**m** | Present | Present | Present | Present |
| `N` | Alveolar nasal | Nasal | n | **n**et, te**n** | Present | Present | Present | Present |
| `NG` | Velar nasal | Nasal | ŋ | si**ng**, ri**ng** | Present | Present | Present | Present |
| `OW` | Diphthong | Close-mid to near-close back | oʊ | b**oa**t, g**o** | Stressed | Stressed | Stressed | Stressed |
| `OY` | Diphthong | Open-mid back to near-close front | ɔɪ | b**oy** | Stressed | Stressed | Stressed | Stressed |
| `P` | Voiceless bilabial stop | Plosive | p | **p**et, to**p** | Present | Present | Present | Present |
| `R` | Alveolar / retroflex liquid | Approximant | ɹ | **r**ed, ca**r** | Present | Present | Present | Present |
| `S` | Voiceless alveolar fricative | Sibilant fricative | s | **s**it, mi**ss** | Present | Present | Present | Present |
| `SH` | Voiceless postalveolar fricative | Sibilant fricative | ʃ | **sh**ut, mi**ssi**on | Present | Present | Present | Present |
| `T` | Voiceless alveolar stop | Plosive | t | **t**op, ca**t** | Present | Present | Present | Present |
| `TH` | Voiceless dental fricative | Fricative | θ | **th**in, ba**th** | Present | Present | Present | Present |
| `UH` | High-mid back lax vowel | Near-close near-back | ʊ | b**oo**k, p**u**t | Stressed | Stressed + Offglide | Stressed | Stressed |
| `UW` | High back tense vowel | Close back | u | b**oo**t, t**wo** | Stressed | Stressed | Stressed | Stressed |
| `V` | Voiced labiodental fricative | Fricative | v | **v**et, ha**ve** | Present | Present | Present | Present |
| `W` | Voiced labial-velar glide | Approximant | w | **w**et, a**w**ay | Present | Present | Present | Present |
| `Y` | Voiced palatal glide | Approximant | j | **y**et, **y**ou | Present | Present | Present | Present |
| `Z` | Voiced alveolar fricative | Sibilant fricative | z | **z**ip, bu**zz** | Present | Present | Present | Present |
| `ZH` | Voiced postalveolar fricative | Sibilant fricative | ʒ | vi**si**on, a**z**ure | Present | Present | Present | Present |
| `AX` | Mid central reduced vowel (Schwa) | Reduced central | ə | comm**a**, **a**bout | Absent | Absent | **Present** (4,057 utts) | Absent |

---

## 4. Phonetic Discrepancy Reconciliation & Handling

### 1. The `AX` (Schwa) vs. `AH0` Harmonization
- **Corpus Occurrence:** In L2-ARCTIC, annotators utilized `AX` in 4,057 canonical intervals and 4,874 realized intervals. CMUDict, SpeechOcean762, and MFA lack `AX`, transcribing schwa as `AH0`.
- **Linguistic Reality:** While CMUDict collapses unstressed [ʌ] and [ə] into `AH0`, phoneticians distinguish them: /ʌ/ has a definite open-mid back quality, whereas /ə/ is centralized and durationally reduced. In non-native speech, learners often fail to reduce schwa, producing full vowels like [a] or [o].
- **Standardization Contract:**
  1. **Layer 0 & Layer 3:** Preserve `AX` losslessly as a distinct token.
  2. **Layer 1 (Canonical 39-Phone Space):** Apply deterministic reduction:
     $$\mathcal{M}_{39}(\text{AX}) = \text{AH}$$
  3. **Layer 2 (Stress Space):** Apply deterministic reduction:
     $$\mathcal{M}_{69}(\text{AX}) = \text{AH0}$$
  4. **Montreal Forced Aligner Alignment:** When generating forced alignments using `english_us_arpa.dict`, substitute `AX` $\rightarrow$ `AH0` in input pronunciation transcripts prior to passing to MFA.

### 2. SpeechOcean762 Unstressed Glide Vowels (`IH` & `UH`)
- **Corpus Occurrence:** `scores.json` and `lexicon.txt` contain `IH` (440 occurrences) and `UH` (98 occurrences) without lexical stress digits.
- **Phonological Reason:** Non-rhotic / British-influenced transcriptions of centring diphthongs where primary stress is assigned to the centering offglide (`AH`):
  - *here* = `HH IH AH0`
  - *your* = `Y UH AH0`
  - *idea* = `AY0 D IH AH1`
- **Standardization Contract:**
  - **Layer 1 (39-Phone):** Maps naturally to base `IH` and `UH`.
  - **Layer 2 (Stress Space):** Mapped deterministically to unstressed forms:
    $$\mathcal{M}_{69}(\text{IH}) = \text{IH0}, \quad \mathcal{M}_{69}(\text{UH}) = \text{UH0}$$

### 3. MFA `english_us_arpa.dict` (v3.0.0) Parsing Contract
- **The Issue:** Upstream MFA v3.0 dictionaries use Kaldi-style lines:
  ```text
  absolute   0.99   0.28   4.07   0.52   AE1 B S AH0 L UW2 T
  ```
- **Parsing Specification for Dataloaders:**
  ```python
  def parse_mfa_dictionary_line(line: str) -> tuple[str, list[str]]:
      tokens = line.strip().split()
      word = tokens[0]
      # Columns 1 to 4 are acoustic probability floats (pron_prob, sil_after, sil_before, dur_scale)
      # True phone sequence begins strictly at index 5
      phone_tokens = []
      for tok in tokens[1:]:
          try:
              float(tok)
              continue  # Skip numeric probability columns
          except ValueError:
              phone_tokens.append(tok)
      return word, phone_tokens
  ```
  This guarantees that probability floats are never ingested as phonetic symbols.

---

## 5. Diagnostic CTC Target Space Specification (43 Classes)

When fine-tuning acoustic encoders (e.g. Wav2Vec 2.0 / HuBERT) with Connectionist Temporal Classification (CTC) for phoneme sequence decoding, the output layer must support the **43-token Langfens Diagnostic Vocabulary**:

```text
CTC Token Vocabulary (Size = 43)
├── Index 0:  <BLANK>      (CTC Blank emission for dynamic programming alignment)
├── Index 1:  <SIL>        (Acoustic Silence Interval; maps to 'sil', 'sp', 'spn')
├── Index 2:  <UNK_PHONE>  (Unidentifiable sound; maps to 'err' and anomalous tokens)
├── Index 3:  AX           (Preserved Reduced Central Schwa)
└── Index 4..42: Standard 39 ARPAbet Base Phonemes (Alphabetical Order)
```

### Full CTC Vocabulary Lookup Table

| Index | Token String | Phonetic Category | Supervised Ground Truth in L2-ARCTIC | Supervised in SpeechOcean762 |
| :---: | :---: | :--- | :--- | :--- |
| **0** | `<BLANK>` | CTC Alignment Blank | CTC null emission | CTC null emission |
| **1** | `<SIL>` | Acoustic Silence | Interval `sil`, `sp`, `spn`, and Deletion `CPL,sil,d` | Inter-word pauses |
| **2** | `<UNK_PHONE>` | Uncertain / Distorted | Interval `CPL,err,s` (269) and `sil,err,a` (4) | Absent |
| **3** | `AX` | Schwa | Canonical/realized `AX` | Absent |
| **4** | `AA` | Low back vowel | Canonical/realized `AA0`, `AA1`, `AA2` | `AA0`, `AA1`, `AA2` |
| **5** | `AE` | Low front vowel | Canonical/realized `AE0`, `AE1`, `AE2` | `AE0`, `AE1`, `AE2` |
| **6** | `AH` | Mid central vowel | Canonical/realized `AH0`, `AH1`, `AH2` | `AH0`, `AH1`, `AH2` |
| **7** | `AO` | Low-mid back vowel | Canonical/realized `AO0`, `AO1`, `AO2` | `AO0`, `AO1`, `AO2` |
| **8** | `AW` | Diphthong | Canonical/realized `AW0`, `AW1`, `AW2` | `AW0`, `AW1`, `AW2` |
| **9** | `AY` | Diphthong | Canonical/realized `AY0`, `AY1`, `AY2` | `AY0`, `AY1`, `AY2` |
| **10** | `B` | Voiced bilabial stop | Canonical/realized `B` | `B` |
| **11** | `CH` | Voiceless affricate | Canonical/realized `CH` | `CH` |
| **12** | `D` | Voiced alveolar stop | Canonical/realized `D` | `D` |
| **13** | `DH` | Voiced dental fricative | Canonical/realized `DH` | `DH` |
| **14** | `EH` | Mid front vowel | Canonical/realized `EH0`, `EH1`, `EH2` | `EH0`, `EH1`, `EH2` |
| **15** | `ER` | Rhotic vowel | Canonical/realized `ER0`, `ER1`, `ER2` | `ER0`, `ER1`, `ER2` |
| **16** | `EY` | Diphthong | Canonical/realized `EY0`, `EY1`, `EY2` | `EY0`, `EY1`, `EY2` |
| **17** | `F` | Voiceless labiodental fricative | Canonical/realized `F` | `F` |
| **18** | `G` | Voiced velar stop | Canonical/realized `G` | `G` |
| **19** | `HH` | Voiceless glottal transition | Canonical/realized `HH` | `HH` |
| **20** | `IH` | High front lax vowel | Canonical/realized `IH0`, `IH1`, `IH2` | `IH`, `IH0`, `IH1`, `IH2` |
| **21** | `IY` | High front tense vowel | Canonical/realized `IY0`, `IY1`, `IY2` | `IY0`, `IY1` |
| **22** | `JH` | Voiced affricate | Canonical/realized `JH` | `JH` |
| **23** | `K` | Voiceless velar stop | Canonical/realized `K` | `K` |
| **24** | `L` | Alveolar liquid | Canonical/realized `L` | `L` |
| **25** | `M` | Bilabial nasal | Canonical/realized `M` | `M` |
| **26** | `N` | Alveolar nasal | Canonical/realized `N` | `N` |
| **27** | `NG` | Velar nasal | Canonical/realized `NG` | `NG` |
| **28** | `OW` | Diphthong | Canonical/realized `OW0`, `OW1`, `OW2` | `OW0`, `OW1` |
| **29** | `OY` | Diphthong | Canonical/realized `OY0`, `OY1`, `OY2` | `OY0`, `OY1` |
| **30** | `P` | Voiceless bilabial stop | Canonical/realized `P` | `P` |
| **31** | `R` | Alveolar/retroflex liquid | Canonical/realized `R` | `R` |
| **32** | `S` | Voiceless alveolar fricative | Canonical/realized `S` | `S` |
| **33** | `SH` | Voiceless postalveolar fricative | Canonical/realized `SH` | `SH` |
| **34** | `T` | Voiceless alveolar stop | Canonical/realized `T` | `T` |
| **35** | `TH` | Voiceless dental fricative | Canonical/realized `TH` | `TH` |
| **36** | `UH` | High back lax vowel | Canonical/realized `UH0`, `UH1`, `UH2` | `UH`, `UH0`, `UH1` |
| **37** | `UW` | High back tense vowel | Canonical/realized `UW0`, `UW1`, `UW2` | `UW0`, `UW1`, `UW2` |
| **38** | `V` | Voiced labiodental fricative | Canonical/realized `V` | `V` |
| **39** | `W` | Labial-velar glide | Canonical/realized `W` | `W` |
| **40** | `Y` | Palatal glide | Canonical/realized `Y` | `Y` |
| **41** | `Z` | Voiced alveolar fricative | Canonical/realized `Z` | `Z` |
| **42** | `ZH` | Voiced postalveolar fricative | Canonical/realized `ZH` | `ZH` |

---

## 6. Disambiguation: CTC Blank vs. Silence vs. Deletion

A critical conceptual error in speech recognition applied to MDD is conflating the CTC Blank token with acoustic silence or phone deletions. They are mathematically and physically distinct:

```text
Disambiguation Matrix
├── 1. CTC Blank Token (<BLANK>, Index 0)
│   ├── Nature: A mathematical artifact of CTC dynamic programming loss formulation
│   ├── Acoustic Meaning: NONE (Does NOT mean silence; means "no label emitted at this acoustic frame")
│   └── Lifetime: Stripped during CTC greedy/beam search collapse: B(a, <BLANK>, a) -> 'a a'
│
├── 2. Acoustic Silence (<SIL>, Index 1)
│   ├── Nature: Physical acoustic absence of speech energy / ambient background room noise
│   ├── Acoustic Meaning: Energy < threshold, flat spectral envelope, voiced/unvoiced absence
│   └── TextGrid Grounding: 'sil', 'sp', 'spn', and leading/trailing pause intervals
│
└── 3. Deleted Reference Phone (CPL,sil,d)
    ├── Nature: A linguistic error event in the prompt alignment
    ├── Acoustic Meaning: The speaker omitted phone CPL; acoustically, silence or adjacent phone transition occurs
    └── MDD Grounding: Ground-truth label for Task 3 (Deletion Detection)
```

**Implementation Rule:** Downstream CTC loss functions must never compute loss against `<BLANK>` as if it were a physical phone label. `<BLANK>` must remain reserved exclusively as the CTC null emission symbol.

---

## 7. Reversibility & Deterministic Testing Rules

All mappings between layers must satisfy the following formal properties:
1. **Determinism:** Given raw string $s \in \text{Layer 0}$, the mapped representation $\mathcal{M}(s)$ is uniquely determined by static lookup tables and pure regex functions.
2. **Provenance Preservation:** Every mapped token object maintains a `.raw_source` field containing the original, unmodified string from the gold file.
3. **Automated Unit Test Suite:** During `RQ1-PREP-01`, a test suite must verify:
   - All 3,599 TextGrids parse without crashing or throwing unrecognized token exceptions.
   - Every token in `scores.json` and `lexicon.txt` of SpeechOcean762 maps cleanly to Layer 1 and Layer 2.
   - Normalizing and re-exporting does not alter start or end millisecond timestamps.
