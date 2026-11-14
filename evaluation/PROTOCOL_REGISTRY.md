# Protocol Registry: English L2 Phoneme-Level Pronunciation Assessment

This registry records candidate evaluation protocols designed from first principles, their scientific properties, operational constraints, and formal selection rationale.

---

## 1. Candidate Protocol Alpha: Deployable Target-Prompt Monotonic Phonetic Alignment

### 1.1 Inputs at Deployment
- Learner audio waveform: raw mono audio resampled to 16 kHz.
- Prompt text: canonical written prompt sentence / words provided to the learner.
- External reference lexicon: standard CMUDict G2P mapping words to canonical phonemes.
- **Strictly Forbidden at Inference:** No human realization TextGrids, no ground-truth phone boundaries, no manual error flags, no oracle transcriptions.

### 1.2 Prediction Unit & Object
- **Expected Target Phone Tokens ($N$ tokens):** Each target canonical phoneme $p_i \in \{1, \dots, N\}$ in the prompt is assigned a discrete prediction $\hat{y}_i \in \{\text{CORRECT}, \text{OMISSION}, \text{SUBSTITUTION}\}$ along with posterior probabilities $P(\text{CORRECT}), P(\text{OMISSION}), P(\text{SUBSTITUTION})$ and predicted replacement symbol $\hat{s}_i$ if substituted.
- **Inter-Phoneme Gaps ($N+1$ gaps):** Each gap $g_j \in \{0, \dots, N\}$ (including utterance prefix and suffix) is evaluated for an $\text{INSERTION}$ event with score $P(\text{INSERTION})$ and predicted inserted phone symbol $\hat{q}_j$.

### 1.3 Error Matching & Scoring Logic
- Monotonic acoustic-phonetic alignment maps acoustic evidence intervals to target phone positions.
- If acoustic evidence indicates no significant realization of the target phoneme, $p_i$ is flagged as $\text{OMISSION}$.
- If acoustic evidence shows a clear phoneme realization that differs significantly from canonical $p_i$, it is flagged as $\text{SUBSTITUTION}$.
- If non-canonical acoustic phoneme activity occurs in gap $g_j$, an $\text{INSERTION}$ event is registered.
- Ground truth labels are derived independently from original corpus annotations (TextGrids in L2-ARCTIC; scores/phones in SpeechOcean762) after time-boundary stripping.

---

## 2. Candidate Protocol Beta: Free Phonetic Recognition + Weighted Levenshtein Alignment

### 2.1 Inputs at Deployment
- Learner audio waveform: raw mono audio at 16 kHz.
- Prompt text: canonical written prompt.

### 2.2 Prediction Unit & Object
- A free acoustic phoneme recognizer outputs a sequence of realized phonemes $\hat{R} = (\hat{r}_1, \dots, \hat{r}_M)$ with frame timestamps and posteriors, unconstrained by prompt text.
- A phonological-distance-weighted Levenshtein alignment aligns canonical target sequence $T = (t_1, \dots, t_N)$ with $\hat{R}$.
- Diagnosis:
  - Exact match: $\text{CORRECT}$.
  - Target phone aligned to empty token $\epsilon$: $\text{OMISSION}$.
  - Target phone aligned to different realized phone: $\text{SUBSTITUTION}$.
  - Realized phone aligned to empty target token $\epsilon$: $\text{INSERTION}$.

### 2.3 Trade-offs
- Protocol Beta provides complete freedom from prompt anchoring, but errors in ASR phoneme recognition directly cascade into alignment errors, especially for heavily accented L2 speech where acoustic-phonetic models frequently produce deletions or substitutions due to acoustic confusion.
- Protocol Alpha conditions directly on the expected prompt (which is authentic to CAPT / computer-assisted pronunciation training applications), yielding much higher construct validity and localized diagnostic precision.

---

## 3. Protocol Selection & Lock Decision

**Selected Primary Protocol:** **Protocol Alpha (Deployable Target-Prompt Monotonic Phonetic Alignment)**
- **Construct validity:** Matches authentic deployment where the prompt is known.
- **Coverage:** Guarantees 100% target phone indexing for all expected prompt phonemes.
- **Gap anchoring:** Separates target-phone evaluation (CORRECT, OMISSION, SUBSTITUTION) from gap-anchored evaluation (INSERTION) with explicit, mathematically rigorous denominators.
- **Protocol Beta:** Maintained as a secondary diagnostic / unconstrained baseline protocol.
