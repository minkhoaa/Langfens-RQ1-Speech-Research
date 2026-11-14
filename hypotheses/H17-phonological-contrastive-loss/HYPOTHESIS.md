# Hypothesis H17: Phonological Supervised Contrastive Margin Regularization (PSC-Net)

## 1. Proposition
Cross-entropy and focal loss treat phoneme labels as discrete orthogonal categories, failing to enforce geometric angular separation in the acoustic latent space. Consequently, minor allophonic accent deviations that lie near the decision boundary frequently produce false substitution warnings.

By introducing a Supervised Phonological Contrastive Margin Loss (SupCon Margin) operating on $L_2$-normalized bottleneck embeddings:
$$\mathcal{L}_{\text{margin}} = \frac{1}{N} \sum_{i=1}^N \max(0, m - \cos(\mathbf{z}_i, \mathbf{c}_{y_i}) + \max_{k \neq y_i} \cos(\mathbf{z}_i, \mathbf{c}_k))$$
with margin $m = 0.35$, the network is regularized to:
1. Pull allophonic realizations of canonical targets into compact angular clusters around their target prototypes $\mathbf{c}_{c_i}$,
2. Enforce a minimum angular gap between legitimate targets and true substitute competitors,
3. Elevate `SUBSTITUTION` Precision beyond **52%** and F1 beyond **0.40** on held-out replication holdouts,
4. Push total Weighted Utility to a new campaign high ($> 0.430$).

## 2. Experimental Plan
- Method ID: `M22-contrastive-margin-model`
- Run ID: `EXP-22-PHONOLOGICAL-CONTRASTIVE-MARGIN`
- Train on `TRAIN` (1,800 utterances) with joint focal loss + articulatory loss + phonological margin loss.
- Calibrate on `DEV_CAL`.
- Evaluate on `DEV_SEARCH` and `DEV_REPLICATION` with primary locked evaluator.
