# Hypothesis H29: Explicit Phonotactic N-Gram Prior Conditioning (PNC-Net)

## 1. Context and Problem Statement
In human second-language acquisition (SLA) and phonology, non-native pronunciation errors are heavily shaped by target language phonotactics and inter-language transfer.
For example:
- **Consonant Cluster Epenthesis:** Learners often insert vowels (/AH/, /IH/) between difficult consonant clusters (e.g., /S/ + /P/, /T/ + /R/) to satisfy CV syllable structures.
- **Coda Consonant Deletion:** Final stops and fricatives at syllable codas have a much higher base omission probability than onset consonants.
- **Phonotactically Illegal Transitions:** Certain substitution hypotheses are phonotactically absurd in English (e.g. producing an affricate after an unvoiced fricative in onset position).

While BiLSTMs learn contextual distributions implicitly, explicit lexical-phonotactic n-gram statistics derived from `cmudict.dict` and the training corpus provide an orthogonal inductive prior.

## 2. Formal Hypothesis
Constructing explicit phonotactic bigram and trigram transition matrices $P(p_t \mid p_{t-1})$, $P(p_t \mid p_{t-1}, p_{t-2})$, and cluster legality flags $\text{is\_valid\_cluster}(p_t, p_{t+1})$ and conditioning the sequence diagnostic model on these prior probabilities (**PNC-Net**) will:
1. Prune acoustic false alarms on phonotactically stable clusters, lowering False Warning Rate ($\text{FWR} \le 1.60$).
2. Elevate standalone Substitution Precision ($\ge 50.0\%$) and Insertion Precision ($\ge 35.0\%$) on `DEV_SEARCH`.
3. Improve overall Utility ($\ge 0.3800$).

## 3. Predicted Evidence & Significance
If validated, PNC-Net demonstrates that combining data-driven acoustic evidence with linguistic phonotactic constraints filters out spurious acoustic classifier hypotheses without manual feature engineering.
