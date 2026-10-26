# G2P Frontend Specification & Lexicon Provenance — RQ1-TRN-02A-ZERO-ORACLE-VERIFY

**Author:** Senior Speech ML Evaluation & Reproducibility Auditor  
**Date:** 2026-10-26  
**Subject:** Pinned G2P frontend specification, tokenization rules, compound decomposition, and OOV handling for Protocol Z  

---

## 1. Objective and Architectural Boundary

Protocol Z requires that model inference operates strictly as a deployable text+audio pipeline:
$$\text{Audio Waveform } \mathbf{x} + \text{Written Prompt Text } \mathbf{w} \xrightarrow{\text{Model}} \hat{\mathbf{y}}$$
No human-annotated Praat TextGrids, interval boundaries, or annotator-selected pronunciation variants may enter the input generation or inference pipeline.

To achieve this, we specify a deterministic, offline, and cryptographically pinned G2P frontend based on CMUdict v0.7b.

---

## 2. Pinned Lexicon Provenance

The primary pronunciation dictionary is pinned to the pre-existing CMUSphinx master dictionary stored in the repository:

- **Resource File:** `resources/CMUDict/cmudict.dict`
- **File Size:** $3,618,488$ bytes
- **SHA-256 Checksum:** `81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22`
- **Upstream Source:** CMUSphinx CMUdict master commit `74790861f652b15e4ac49015a90074ad62a27690`
- **Total Primary Lexical Entries:** $126,052$ words (alternative pronunciation variants `(2)`, `(3)`, etc. are strictly excluded to enforce determinism).

### Supporting Phone Inventory:
- **Phone Symbol Table:** `resources/CMUDict/cmudict.symbols` (SHA-256: `408ccaae803641c6d7b626b6299949320c2dbca96b2220fd3fb17887b023b027`)
- **Normalized Inventory:** Mapped strictly to the Standard 39-ARPAbet phonemes via `normalize_phoneme` in `langfens_prep/phoneme_normalizer.py`.
- **Stress Handling:** All stress markers (`0`, `1`, `2`) are stripped. All unstressed schwa allophones (`AX`) are folded to `AH`.

---

## 3. Written Prompt Tokenization Contract

The tokenizer converts raw English sentences into an ordered sequence of uppercase lexical tokens:

1. **Dash and Hyphen Normalization:**
   - Double hyphens (`--`), em-dashes (`—`), and en-dashes (`–`) are replaced with spaces.
2. **Word Boundary Segmentation:**
   - Words are segmented using the regular expression `(?:'?[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?)`.
   - Punctuation (`.`, `,`, `;`, `:`, `!`, `?`, `"`, `'`, `(`, `)`) is stripped from token boundaries.
3. **Contractions:**
   - Common English contractions (`DON'T`, `I'LL`, `THAT'S`, `THERE'S`, `COULDN'T`, `DIDN'T`, `WHAT'S`, `WE'VE`, `IT'S`, `HE'S`) are matched directly in CMUdict.
   - Initial-apostrophe colloquial contractions (`'EM`, `'CAUSE`, `'ROUND`, `'BOUT`, `'TIL`) are preserved and mapped to their dictionary entries.

---

## 4. Phonological Inflection & Compound Decomposition Rules

When a word token is not found directly in CMUdict, the following deterministic rules are applied in strict sequence:

### Rule 1: Possessive Suffix (`'S`)
If a token ends in `'S`:
1. The base word $\mathbf{w}_{\text{base}}$ is looked up in CMUdict.
2. If found with phone sequence $\mathbf{p} = [p_1, \dots, p_k]$, standard English phonological voicing assimilation is applied to attach the suffix:
   - **Sibilant Coda:** If $p_k \in \{\text{S}, \text{Z}, \text{SH}, \text{ZH}, \text{CH}, \text{JH}\}$, append `[IH, Z]`.
   - **Voiceless Non-Sibilant Coda:** If $p_k \in \{\text{P}, \text{T}, \text{K}, \text{F}, \text{TH}\}$, append `[S]`.
   - **Voiced Non-Sibilant Coda:** If $p_k \in \{\text{Vowels}, \text{B}, \text{D}, \text{G}, \text{V}, \text{DH}, \text{M}, \text{N}, \text{NG}, \text{L}, \text{R}, \text{W}, \text{Y}\}$, append `[Z]`.
   *(Example: `PROMOTER'S` $\to$ `PROMOTER` (`P R AH M OW T ER`) $+$ `Z` $\to$ `P R AH M OW T ER Z`).*

### Rule 2: Hyphenated Compounds
Tokens containing internal hyphens (e.g. `WOLF-DOG`, `SMOOTH-SHAVEN`, `GRUB-THIEVES`, `JOY-THRILL`, `CROOKED-LEG`) are split on hyphens into subwords, each of which is independently resolved in CMUdict.

### Rule 3: Known Unhyphenated Compounds
A pre-declared lookup table handles the closed set of non-hyphenated compounds appearing across CMU ARCTIC prompt sentences:
- `NIGHTGLOW` $\to$ `NIGHT` (`N AY T`) $+$ `GLOW` (`G L OW`)
- `TOMFOOLERY` $\to$ `TOM` (`T AA M`) $+$ `FOOLERY` (`F UW L ER IY`)
- `ROADMATE` $\to$ `ROAD` (`R OW D`) $+$ `MATE` (`M EY T`)
- `SEAFARING` $\to$ `SEA` (`S IY`) $+$ `FARING` (`F EH R IH NG`)
- `UNQUENCHABLE` $\to$ `UN` (`AH N`) $+$ `QUENCH` (`K W EH N CH`) $+$ `ABLE` (`EY B AH L`)

---

## 5. Out-of-Vocabulary (OOV) Policy & Verification Census

### Policy Enforcement:
- **`policy="strict"` (Default for benchmarking):** Raises an explicit `KeyError` identifying the offending token, the full utterance ID, and prompt text. Evaluation halts immediately to prevent silent data corruption.
- **`policy="skip"`:** Logs a structured warning and omits the OOV token (allowed only in exploratory modes).

### Corpus-Wide Verification Census:
Across all **1,132 CMU ARCTIC prompt sentences** ($77,059$ characters):
- Exactly **31 unique word types** are not in raw CMUdict primary entries.
- 18 are regular possessives (`BRODIE'S`, `COMPANION'S`, `DAUGHTRY'S`, `DAYLIGHT'S`, `DENNIN'S`, `DOANE'S`, `EILEEN'S`, `FACTOR'S`, `HANRAHAN'S`, `JEANNE'S`, `KERFOOT'S`, `MCFEE'S`, `NAKATA'S`, `PASCAL'S`, `PEARCE'S`, `PROMOTER'S`, `SELDEN'S`, `STEWARD'S`, `THORPE'S`), which resolve via Rule 1.
- 5 are compound words (`NIGHTGLOW`, `ROADMATE`, `SEAFARING`, `TOMFOOLERY`, `UNQUENCHABLE`), which resolve via Rule 3.
- Across the **300 Fold 1 validation utterances**, exactly **0 OOV words** remain unresolved under the strict contract ($100.0\%$ prompt coverage).

---

## 6. Execution Command

The G2P frontend is instantiated directly from `langfens_prep.g2p`:
```python
from langfens_prep.g2p import CMUDictG2P

g2p = CMUDictG2P(dict_path="resources/CMUDict/cmudict.dict", verify_checksum=True, oov_policy="strict")
phones = g2p.phonemize("For the twentieth time that evening the two men shook hands.")
# Output: ['F', 'AO', 'R', 'DH', 'AH', 'T', 'W', 'EH', 'N', 'T', 'IY', 'AH', 'TH', ...]
indices = g2p.encode_indices("For the twentieth time that evening the two men shook hands.")
# Output: [14, 4, 28, 10, 3, 31, 36, 11, ...]
```
