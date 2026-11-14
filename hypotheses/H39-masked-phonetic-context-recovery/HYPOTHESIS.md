# Hypothesis H39: Dynamic Masked Context Regularization & Acoustic Reconstruction (DMCT-Net)

## 1. Context and Problem Statement
In human L2 speech, pronunciation errors rarely occur in isolation. Rather, a mispronounced or omitted sound distorts the coarticulatory transitions of preceding and succeeding phonemes.
In previous models (EXP-08 through EXP-38), networks were trained with 100% visible, uncorrupted input sequence vectors. Consequently, when an adjacent phone is omitted or heavily accented at test time, the bidirectional LSTM and attention representations can suffer from context mismatch.

In self-supervised representation learning, dynamic sequence masking forces models to infer latent phonetic properties from non-local context.
Applying dynamic token masking ($p_{\text{mask}} = 0.15$) directly to the sequence feature vectors during training will prevent the network from relying excessively on fragile local adjacent frames.

## 2. Formal Hypothesis
Training the cascaded HTC-Net architecture under **Dynamic Masked Context Training (DMCT-Net)**:
- During each training step, randomly replace $15\%$ of phoneme sequence vectors with a learned mask token $\mathbf{e}_{\text{mask}} \in \mathbb{R}^{63}$.
- Add an auxiliary masked phoneme reconstruction loss:
  $$\mathcal{L}_{\text{recon}} = -\sum_{i \in \text{Masked}} \log P(p_i \mid \mathbf{h}_i)$$
  to force latent states to maintain phonetic identity even when acoustic features are obscured.

This will:
1. Elevate standalone Omission Recall on `DEV_SEARCH` ($\ge 46.0\%$) and Omission AP ($\ge 0.2500$).
2. Maintain high Substitution AP ($\ge 0.3800$) and high Macro AP ($\ge 0.5300$) on `DEV_SEARCH`.
3. Improve robustness against coarticulation degradation.

## 3. Predicted Evidence & Significance
If validated, DMCT-Net demonstrates that self-supervised masked context regularization directly fortifies phonetic diagnostic networks against real-world L2 coarticulatory corruption.
