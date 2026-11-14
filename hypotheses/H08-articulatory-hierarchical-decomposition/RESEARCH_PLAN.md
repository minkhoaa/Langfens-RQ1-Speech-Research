# Research Plan: Hypothesis H08 (HM-ADN)

1. **Articulatory Mapping Formulation:**
   - Map all 39 ARPAbet phonemes to Place (8 classes), Manner (7 classes), and Voicing (2 classes).
   - Add Class 0 for `SILENCE` / `OMISSION` across all three articulatory dimensions.
2. **Model Architecture:**
   - Input Sequence: $\mathbf{X}_{\text{target}} \in \mathbb{R}^{B \times N \times 51}$
   - Shared Encoder: 2-layer BiLSTM (hidden=96, output=192) + Linear fusion (128)
   - Auxiliary Heads:
     - `Manner Head`: Linear(128, 8)
     - `Place Head`: Linear(128, 9)
     - `Voicing Head`: Linear(128, 3)
   - Primary Diagnostic Head:
     - Input: $\mathbf{h}_i \oplus \mathbf{p}_{\text{manner}} \oplus \mathbf{p}_{\text{place}} \oplus \mathbf{p}_{\text{voicing}}$ (148 dims)
     - Output: Linear(148, 64) -> GELU -> Linear(64, 3)
   - Gap Head: Linear(262, 64) -> GELU -> Linear(64, 1)
3. **Training & Calibration:**
   - Train for 30 epochs with AdamW (lr=1e-3) and multi-task loss.
   - Pareto calibration on `DEV_CAL`.
   - Primary locked evaluation on `DEV_SEARCH`.
