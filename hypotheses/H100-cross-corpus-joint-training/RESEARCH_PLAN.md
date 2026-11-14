# Research Plan: Experiment EXP-105 (Joint Cross-Corpus Multi-Accent Co-Training - JCCM-Net)

## 1. Multi-Corpus Ingestion & Representation
- **L2-ARCTIC Training Split:** 1,800 utterances (hard/soft targets).
- **SpeechOcean762 Auxiliary Training Split:**
  * Extract 300 diverse training utterances from `speechocean762.tar.gz` (`speechocean762/train/wav.scp`).
  * Compute 72-dimensional acoustic representations (Charsiu SSL + 5ms sub-frame dynamics).
  * Map continuous human accuracy scores $S \in [0, 10]$ to soft target distributions:
    - $S \ge 8.0$: $y_{\text{so}} = [0.85, 0.10, 0.05]$ (CORRECT)
    - $4.0 \le S < 8.0$: $y_{\text{so}} = [0.20, 0.70, 0.10]$ (SUBSTITUTION)
    - $S < 4.0$: $y_{\text{so}} = [0.10, 0.30, 0.60]$ (OMISSION)
- **Shared Architecture:**
  ADAE-Net with orthogonal latent spaces (manner: 32, place: 32, voicing: 16, timbre: 48) and 47-dim canonical IPTM prompt stream.
- **Evaluation:**
  1. Primary In-Domain: Held-out `DEV_REPLICATION` under locked `PROTOCOL_ALPHA_TARGET_MONOTONIC_V1`.
  2. Cross-Corpus Domain Generalization: Benchmark on SpeechOcean762-Test (300 utterances, 4,536 phonemes), evaluating Pearson $r$ and Spearman $\rho$.
- Audit against H100 falsification criteria.
- Save all artifacts in `experiments/EXP-105-CROSS-CORPUS-FULL-TRAINING-EXPANSION/`.
