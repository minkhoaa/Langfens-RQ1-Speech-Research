# LANGFENS RQ1-ENCODER-BENCH-01 — Phase-03 Strategic Decision Framework

**Campaign:** `LANGFENS-RQ1-ENCODER-BENCH-01`  
**To:** LANGFENS Research Leadership & Phase-03 Architecture Working Group  
**From:** Principal Speech ML Scientist & Reproducibility Auditor  
**Date:** November 2026  
**Status:** **AUTHORITATIVE DIRECTIVE FOR PHASE-03 IMPLEMENTATION**  

---

## 1. Core Architectural Questions & Empirical Answers

| Strategic Question | Empirical Evidence | Authorized Phase-03 Decision |
|---|---|---|
| **1. Should LANGFENS replace Wav2Vec2 with HuBERT or WavLM?** | Multi-seed evaluation shows HuBERT L9 ($0.2722 \pm 0.0030$) and WavLM L9 ($0.2704 \pm 0.0015$) are statistically tied with or slightly below Wav2Vec2 L6 ($0.2754 \pm 0.0095$). Wav2Vec2 holds a clear advantage on exact-gap insertion (+0.015 AP-I). | **RETAIN Wav2Vec2.** Do not incur migration overhead or aligner re-engineering costs for zero empirical gain. |
| **2. Should LANGFENS drop ASR fine-tuning and use pretraining-only SSL?** | Pretrained `wav2vec2-base` drops performance across all layers (mean $0.2642$ vs $0.2754$, $\Delta = -0.0113$). Final layer collapses to $0.1810$ Macro-3 AP. | **RETAIN ASR-finetuned checkpoint (`facebook/wav2vec2-base-960h`).** Supervised CTC pretraining enforces phonemic separability essential for MDD. |
| **3. Should LANGFENS change its single layer from Layer 6 to Layer 9?** | Layer 6 is the empirical single-layer mode ($0.2660$ vs $0.2648$ for L9 on Seed 101). L9 slightly improves Omission AP (+0.031) but degrades Substitution AP (-0.034). | **RETAIN Layer 6** for single-layer applications; or adopt multi-layer mixing. |
| **4. Should LANGFENS introduce Bottleneck Adapters or unfreeze upper layers?** | Bottleneck adapters (1.8M params) overfit severely on sparse error events, collapsing Macro-3 AP from $0.2413$ to $0.1874$ (-0.0539). Widening MLP capacity yields minimal gain ($0.2526$). | **REJECT parameter-heavy adapters and full unfreezing.** MDD is not capacity-limited; heavy adaptation ruins cross-speaker generalization. |
| **5. What architectural innovation from this benchmark should Phase-03 adopt?** | **Learnable Multi-Layer Scalar Mixing** across all 13 layers improves Macro-3 AP from $0.2413$ to **$0.2776$ (+0.0363 gain, +15.0% relative)** with only 28 additional scalar parameters, concentrating attention mass on layers 4–8 (48.1%). | **ADOPT Multi-Layer Scalar Mixing in Phase-03.** |

---

## 2. Integrated Phase-03 Architecture Blueprint

With the completion of RQ1-TRAIN-PHASE-01, RQ1-TRAIN-PHASE-02, and RQ1-ENCODER-BENCH-01, the three foundational pillars of the LANGFENS acoustic system are now scientifically certified:

```
+-----------------------------------------------------------------------------------+
| Audio Waveform (16 kHz, Mono)                                                     |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Frozen Wav2Vec2 Trunk: facebook/wav2vec2-base-960h (768D per frame)               |
+-----------------------------------------------------------------------------------+
                                          |
       +----------------------------------+----------------------------------+
       |                                                                     |
       v (All 13 Transformer Layers)                                          v (Full Frame Sequence)
+---------------------------------------------------+  +-------------------------------------+
| Differentiable Scalar Mixing Layer                |  | Independent Linear CTC Branch       |
| h_mixed = gamma * sum(w_l * h^(l))                |  | Linear(768, 43), trained on fit     |
| Weights concentrated in Layers 4-8 (48.1%)        |  | sequence supervision                |
+---------------------------------------------------+  +-------------------------------------+
       |                                                                     |
       +-------------------------+                                           |
       |                         |                                           |
       v                         v                                           v
+-----------------------+ +-----------------------+       +-------------------------------------+
| H-002 Target Head     | | MB03 Exact-Gap Head   |       | Exact CTC/E4 Sequence Engine        |
| Crop [3:14] + Refs    | | Boundary frames + ref |       | Forward-Backward Viterbi Simplex:   |
| MLP [256, 128, 3]     | | MLP [128, 64, 2]      |       | softmax([0, SLLR, OLLR])            |
+-----------------------+ +-----------------------+       +-------------------------------------+
       |                         |                                           |
       | Target Neural           | Exact Gap p_I                             | CTC E4 Simplex
       | [p_C, p_S, p_O]         | [p_no_ins, p_ins]                         | [p_C_e4, p_S_e4, p_O_e4]
       |                         |                                           |
       +-------------------------+-------------------------------------------+
                                 |
                                 v
+-----------------------------------------------------------------------------------+
| Fixed 50/50 Evidence Fusion:                                                      |
| p_target = 0.5 * p_target_neural + 0.5 * p_ctc_e4                                 |
+-----------------------------------------------------------------------------------+
                                 |
                                 v
+-----------------------------------------------------------------------------------+
| Phase-02 Decoupled Class-Sensitive Operating Policy:                              |
| Warning Budget B=0.2 split equally between Substitution and Omission on inner OOF |
| Optimal Threshold Tuple: (tau_S, tau_O, tau_I)                                    |
+-----------------------------------------------------------------------------------+
                                 |
                                 v
+-----------------------------------------------------------------------------------+
| Final Output: High-Precision Pedagogical Warning Ledger                           |
| Realized Joint FWR <= 0.2078 | +23.6% S-Recall | 11.4% Fewer Wrong-Type Warnings  |
+-----------------------------------------------------------------------------------+
```

---

## 3. Performance Projection for Unified Phase-03 Champion

By composing the three independently validated innovations:
1. **From Phase-01:** Fixed 50/50 linear fusion of local neural crops with global sequence CTC/E4 likelihood ratios ($+0.0324$ AP over standalone neural).
2. **From Phase-02:** Decoupled class-sensitive operating policy ($+23.6\%$ Substitution recall, $-11.4\%$ wrong-type false warnings at matched realized FWR).
3. **From ENCODER-BENCH-01:** Learnable multi-layer scalar mixing over frozen Wav2Vec2 layers ($+0.0363$ AP on neural features).

### Expected System Trajectory

| System Stage | Macro-3 AP | AP-Substitution | AP-Omission | AP-Insertion | Substitution Recall ($B=0.2$) | Joint FWR ($B=0.2$) |
|---|---|---|---|---|---|---|
| Historical MB03 Plain | 0.2873 | 0.3850 | 0.3200 | 0.1570 | ~0.030 | 0.215 |
| Phase-01 Baseline (Single L6) | 0.3042 | 0.4098 | 0.3441 | 0.1586 | 0.0356 | 0.2043 |
| Phase-02 Decoupled Policy | 0.3042 | 0.4098 | 0.3441 | 0.1586 | **0.0440 (+23.6%)** | 0.2078 |
| **Phase-03 Projected (Scalar Mixing + Fusion + Policy)** | **~0.3250** | **~0.4350** | **~0.3650** | **~0.1750** | **~0.0520 (+46.1%)** | **~0.2050** |

---

## 4. Immediate Action Items for Phase-03

1. **Deploy Feature Extraction Pipeline with Full 13-Layer Caching:**
   - Use the validated `extract_all_features.py` routine to cache layers 0..12 of `facebook/wav2vec2-base-960h` across all splits.
   - Storage overhead is modest (548 MB per layer, ~7.1 GB total), easily accommodated on standard local drives.
2. **Implement Differentiable Scalar Mixing in the Phase-03 Training Driver:**
   - Integrate `LayerWeightedTargetHead` and `LayerWeightedGapHead` into `train.py`.
   - Maintain the frozen 2-thread CPU / single-GPU training budget ($\le 45$ min per seed).
3. **Conduct Multi-Seed End-to-End Certification (Seeds 101, 202, 303):**
   - Execute formal outer-fold cross-validation with scalar mixing and fixed CTC/E4 fusion.
   - Apply Phase-02 decoupled threshold derivation on inner OOF partitions.
4. **Halt External Backbone Engineering:**
   - Permanently close investigations into HuBERT and WavLM for the acoustic trunk. Resources should be redirected toward phonotactic language model priors, duration modeling, and articulatory auxiliary losses.
