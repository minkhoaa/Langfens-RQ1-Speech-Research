# Hypothesis H64: Interlanguage Phonotactic Error Prior Mapping (IPTM-Net)

## 1. Context & Motivation
Across our 67 completed experiments, the network has learned to detect mispronunciations from acoustic representations (5ms dynamics, prosody, MFCC residuals) conditioned on canonical prompt text and native English CMUDict n-grams.

However, second language acquisition (SLA) theory and contrastive linguistics demonstrate that pronunciation errors are governed by **Interlanguage Phonological Transfer**:
- English native bigram probabilities $P_{\text{native}}(p_k \mid p_{k-1})$ describe what native speakers say, but give no information about which phonemes are phonetically vulnerable in L2 interlanguage.
- Specific phonemes exhibit drastically skewed error priors in L2 speech:
  * Interdental fricatives (`/TH/`, `/DH/`) have substitution error rates exceeding $40\%$.
  * Word-final alveolar stops (`/T/`, `/D/`) have omission error rates exceeding $30\%$.
  * Tense/lax vowel pairs (`/IY/` vs `/IH/`, `/UW/` vs `/UH/`) have systematic vowel-shortening substitution priors.
  * In contrast, stable consonants (`/M/`, `/F/`, `/S/`) have correct production rates exceeding $90\%$.

When a model is blind to these phoneme-specific interlanguage error priors, it must infer them entirely from noisy acoustic frames, causing hesitation and coarticulation to be mistaken for errors in inherently stable sounds, while true subtle substitutions in vulnerable sounds are missed.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. Constructing an **Interlanguage Phonotactic Error Prior Matrix (IPTM)** computed strictly on the training partition (`train.jsonl`):
   $$\pi(p) = \left[ P(\text{CORRECT} \mid p), P(\text{SUBSTITUTION} \mid p), P(\text{OMISSION} \mid p) \right] \in \mathbb{R}^3$$
   for every canonical phone $p \in \{1, \dots, 39\}$, and concatenating this 3-dimensional interlanguage transfer prior directly into the canonical prompt stream ($44 \rightarrow 47$ dimensions):
   $$\text{PromptStream}(k) = \left[ \text{onehot}_{39}, M_k, Pl_k, V_k, \log P_{\text{fwd}}, \log P_{\text{bwd}}, \pi(p_k) \right]$$
   will provide the cross-attention mechanism with explicit prior knowledge of each target sound's phonological vulnerability.
2. IPTM-Net coupled with DSCA-Net (EXP-55) will achieve:
   - **Substitution AP** breaking $\ge 0.4500$ in a standalone model.
   - **Substitution Precision** rising to $\ge 52.00\%$.
   - **Omission AP** maintaining $\ge 0.4400$.
   - **Macro AP** surging to $\ge 0.6250$ (62.50%).
   - **Weighted Utility** breaking $\ge 0.4950$.
   - False Warning Rate maintaining $\le 1.65$ false alarms per utterance.

## 3. Predicted Experimental Outcomes
1. **DEV_REPLICATION Macro AP:** $\ge 0.6250$.
2. **DEV_REPLICATION Weighted Utility:** $\ge 0.4950$.
3. **DEV_REPLICATION Substitution AP:** $\ge 0.4500$.
4. **DEV_REPLICATION Substitution Precision:** $\ge 52.00\%$.
