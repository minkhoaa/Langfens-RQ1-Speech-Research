# Hypothesis H33: Hierarchical Broad-Category Cascaded Classification (HTC-Net)

## 1. Context and Problem Statement
Phonetic science establishes a fundamental dichotomy between vowels and consonants:
1. **Vowels** are high-energy periodic resonant events produced with an unconstricted vocal tract. Their pronunciation errors primarily manifest as spectral formant centroid shifts (SUBSTITUTIONS like /IH/ vs /IY/, /AE/ vs /EH/), while complete deletions (OMISSIONS) are relatively infrequent.
2. **Consonants** are low-energy, transient, or turbulent events involving vocal tract constriction or occlusion. Their errors are heavily dominated by coda cluster deletions (OMISSIONS), devoicing, and plosive unreleased bursts.

In all previous single-model classifiers (EXP-04 through EXP-34), a single target diagnostic head was trained across all phonemes simultaneously. Consequently, the optimal decision boundary for detecting vowel substitutions is compromised by the conflicting acoustic properties of consonant deletions.

## 2. Formal Hypothesis
Decoupling the diagnostic decision layer into a **Hierarchical Two-Stage Cascaded Network (HTC-Net)**:
- Canonical routing: If target $p_t \in \text{Vowels}$, route to a specialized **Vocalic Diagnostic Head** $\mathcal{H}_{\text{vowel}}$.
- If target $p_t \in \text{Consonants}$, route to a specialized **Consonantal Diagnostic Head** $\mathcal{H}_{\text{cons}}$.
Both branches receive the 63-dimensional phonotactic and prosodic representations, but optimize category-specific loss functions and thresholds on `DEV_CAL`.

This cascaded formulation will:
1. Reduce inter-category acoustic interference, pushing standalone Substitution AP on `DEV_SEARCH` beyond $0.3800$ and Omission AP beyond $0.2000$.
2. Elevate standalone model Utility to $\ge 0.3850$ on `DEV_SEARCH`.
3. Provide specialized vocalic and consonantal diagnostic confidence estimates.

## 3. Predicted Evidence & Significance
If validated, HTC-Net demonstrates that phonological broad-category routing resolves the intrinsic acoustic tension between vowel spectral shifts and consonant energy drops.
