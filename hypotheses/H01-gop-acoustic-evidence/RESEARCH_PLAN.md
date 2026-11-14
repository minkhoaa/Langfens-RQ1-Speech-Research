# Research Plan: Hypothesis H01

1. **Step 1: Implement Baselines and Negative Controls**
   - Run `EXP-01-PRIOR`: Predict prior probabilities based on training class frequencies.
   - Run `EXP-02-MISMATCH`: Mismatched audio control to test leakage.
2. **Step 2: Implement Pretrained Acoustic Feature / Phoneme Alignment Pipeline**
   - Use `facebook/wav2vec2-base-960h` / CTC character-phone or phoneme-level acoustic model.
   - Compute frame-level phone log-posteriors and perform alignment to prompt canonical phones.
   - Compute GOP per target phone: $\text{GOP}(p_i) = \frac{1}{|T_i|} \sum_{t \in T_i} \log P(p_i \mid \mathbf{x}_t)$.
   - Diagnose: if $\text{GOP}(p_i) < \tau_{\text{err}}$, check if frame energy/silence probability exceeds threshold $\implies \text{OMISSION}$; otherwise $\implies \text{SUBSTITUTION}$ with $\arg\max_{q \neq p_i} \sum_{t \in T_i} \log P(q \mid \mathbf{x}_t)$.
   - For gap $g_j$ between $T_j$ and $T_{j+1}$, inspect maximum non-blank frame probability $\implies \text{INSERTION}$ if $> \tau_{\text{ins}}$.
3. **Step 3: Evaluate on `DEV_SEARCH`**
   - Full denominator accounting: 19,755 target phones and 20,354 gaps across 599 utterances.
   - Compute AP, Precision, Recall, F1 for CORRECT, OMISSION, SUBSTITUTION, INSERTION.
   - Compute ECE, False Warning Rate, Weighted Utility.
4. **Step 4: Save Evidence, Review Failures, and Enqueue Next Run**
