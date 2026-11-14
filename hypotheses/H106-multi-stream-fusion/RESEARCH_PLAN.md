# Research Plan: Experiment EXP-111 (Multi-Stream Acoustic-Articulatory Fusion - MSAF-Net)

## 1. Network Architecture (MSAF-Net)
1. **Input Streams:**
   - Base Acoustic: 72-dimensional temporal pyramid features.
   - Low-Rank Adapter: LoRA bottleneck ($W_{\text{down}} \in \mathbb{R}^{8 \times 72}, W_{\text{up}} \in \mathbb{R}^{72 \times 8}, \text{scale}=2.0$).
     $$x_{\text{lora}} = x_{\text{base}} + 2.0 \cdot (W_{\text{up}} W_{\text{down}} x_{\text{base}})$$
   - Physical Area Stream: 8-dimensional vocal tract log-area function $A_{\text{tract}}$ derived from reflection coefficients.
   - Fused Input: $x_{\text{fused}} = [x_{\text{lora}}, A_{\text{tract}}] \in \mathbb{R}^{80}$.
2. **Context & Prompt Cross-Attention:**
   - BiLSTM (96 hidden) on $x_{\text{fused}} \to 128$ dimensions.
   - Bidirectional Multi-Head Cross-Attention (4 heads) against 47-dimensional canonical IPTM prompt stream.
   - Monotonic Local Gaussian Attention ($\sigma=3.0$).
3. **Disentangled Latent Space:**
   - 128-dimensional latent space partitioned into:
     $z_{\text{manner}} \in \mathbb{R}^{32}$, $z_{\text{place}} \in \mathbb{R}^{32}$, $z_{\text{voicing}} \in \mathbb{R}^{16}$, $z_{\text{timbre}} \in \mathbb{R}^{48}$.
   - Reconstruction Decoder: $128 \to 80$ dimensions.
4. **Joint Loss Optimization:**
   $$\mathcal{L} = \mathcal{L}_{\text{cls}} + 0.6 \mathcal{L}_{\text{gap}} + 0.25 \mathcal{L}_{\text{recon}} + 0.05 \mathcal{L}_{\text{ortho}} + 0.15 \mathcal{L}_{\text{lattice}}$$
   where $\mathcal{L}_{\text{gap}}$ uses BCE with `pos_weight=7.0` and triad category embeddings.

## 2. Experimental Execution
1. Implement `methods/M111-multi-stream-trainer/run.py`.
2. Train on GPU for 10 epochs using AdamW and cosine learning rate decay.
3. Calibrate thresholds on `dev_cal`.
4. Evaluate on `dev_search` and `dev_replication`.
5. Audit against H106 falsification criteria.
6. Save all artifacts in `experiments/EXP-111-MULTI-STREAM-ACOUSTIC-ARTICULATORY-FUSION/`.
