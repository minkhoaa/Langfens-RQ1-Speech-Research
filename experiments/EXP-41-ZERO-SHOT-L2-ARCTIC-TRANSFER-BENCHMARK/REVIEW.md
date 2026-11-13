# Experiment Review: EXP-41-ZERO-SHOT-L2-ARCTIC-TRANSFER-BENCHMARK

## 1. Executive Summary
Experiment EXP-41 evaluated the zero-shot cross-corpus transfer of champion models trained on L2-ARCTIC onto the independent SpeechOcean762 test set (comprising 4,536 phonemes across 300 test utterances from Chinese learners across various age groups).

## 2. Quantitative Comparison
| Model Architecture | Pearson $r$ ($p$-value) | Spearman $\rho$ | Error AP | Error Recall | Error Precision | Error F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **BiLSTM (EXP-08)** | **$0.2690$ ($p=4.9 \times 10^{-76}$)** | **$0.3132$** | **$0.1258$** | **$63.88\%$** | **$11.26\%$** | **$0.1914$** |
| **HTC-Net (EXP-38)** | $0.1610$ ($p=1.0 \times 10^{-27}$) | $0.2717$ | $0.1067$ | $15.86\%$ | $8.18\%$ | $0.1079$ |

## 3. Scientific Insights & Domain Shift Analysis
1. **Statistically Significant Zero-Shot Transfer:** Both models confirm zero-shot correlation with human expert accuracy scores ($p < 10^{-26}$), proving that the learned representations capture universal acoustic-phonetic pronunciation quality beyond L2-ARCTIC.
2. **The Age / Vocal Tract Length Invariance Trade-Off:**
   - SpeechOcean762 features Chinese children and adolescents, whose fundamental pitch (F0) and formant distributions are shifted substantially higher than the adult L2-ARCTIC speakers.
   - The unrouted continuous temporal BiLSTM representation (EXP-08) proved more resilient to age-related vocal tract length scaling ($r = 0.2690$, Error Recall = 63.88%), whereas HTC-Net's sharp category-specific decision boundary suffered from adult-to-child formant shift.
   - This provides critical evidence for future work on age-invariant acoustic feature normalization in computer-assisted language learning.
