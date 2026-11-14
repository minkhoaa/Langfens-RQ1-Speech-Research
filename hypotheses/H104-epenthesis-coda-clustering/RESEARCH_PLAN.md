# Research Plan: Experiment EXP-109 (Coda Release & Epenthesis Clustering Specializer - CREC-Net)

## 1. Feature Engineering (16-Dimensional Gap Stream)
- For each inter-phone gap $g \in \{0, \dots, N\}$ in 16kHz audio:
  * 13 base features: RMS energy, relative energy, log energy, ZCR, phonotactic cluster bigram log-probability, consonant-cluster flag, 5ms onset gradient, 5ms offset gradient, and context frames.
  * 3 specialized coda dynamics features:
    1. $r_{\text{post-coda}}$: Ratio of energy in the post-consonant interval ($[t_e, t_e + 35\text{ms}]$) to coda energy.
    2. $\Delta C_{\text{spec}}$: Drop in spectral centroid between consonant release and post-coda interval (measuring vocalic formant emergence).
    3. $\text{ZCR}_{\text{ratio}}$: Ratio of post-coda ZCR to consonant ZCR (identifying periodicity).
- **Network Architecture:**
  * Target stream: 72-dimensional temporal pyramid features + ADAE-Net orthogonal latent spaces (manner: 32, place: 32, voicing: 16, timbre: 48).
  * Prompt stream: 47-dimensional canonical IPTM prompt stream.
  * Gap stream: 16-dimensional input + 16-dim triad category embedding (pause, cluster, coda).
  * Gap loss: Focal loss ($\gamma=2.0, \alpha=0.75$) to focus on difficult coda releases.

## 2. Experimental Execution
1. Implement `methods/M109-epenthesis-specializer-trainer/run.py`.
2. Train on GPU for 10 epochs using AdamW and cosine scheduling.
3. Calibrate thresholds on `dev_cal`.
4. Evaluate on `dev_search` and `dev_replication`.
5. Audit against H104 falsification criteria.
6. Save all artifacts in `experiments/EXP-109-EPENTHESIS-CODA-CLUSTERING-SPECIALIZER/`.
