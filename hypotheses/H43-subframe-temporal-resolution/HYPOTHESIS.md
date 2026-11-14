# Hypothesis H43: High-Resolution Sub-Frame Temporal Dynamics for Micro-Acoustic Omission Diagnosis

## 1. Context & Motivation
Across our 47 research experiments, acoustic representations have relied primarily on 10ms frame intervals (100 Hz). While sufficient for vowels and steady-state fricatives, 10ms windowing introduces critical temporal smearing for transient phonetic events:
1. **Stop Consonant Bursts (`/p/`, `/t/`, `/k/`, `/b/`, `/d/`, `/g/`)**: The plosive burst release duration is often between 5ms and 15ms. In standard 10ms frames, a weak release can be averaged out with surrounding silence or aspiration.
2. **Affricate Transitions (`/ch/`, `/jh/`)**: The occlusion-to-friction transition occurs over an ultra-short window of 8-15ms.
3. **Flap & Tap Allophones**: Flapped `/t/` and `/d/` have closure durations as brief as 10-20ms.

Because the model lacks sub-frame temporal resolution, weak plosive bursts in L2 speech are disproportionately misdiagnosed as `OMISSION` (false omission alarms) or genuine omissions of unreleased stops are missed.

## 2. Formal Hypothesis Statement
We hypothesize that:
1. By extracting **5ms sub-frame acoustic micro-dynamics**—specifically:
   - $E_{\text{sub1}} / E_{\text{sub2}}$ (energy ratio between first and second 5ms half of the slot boundary)
   - Maximum instantaneous energy derivative: $\max |dE/dt|_{5\text{ms}}$
   - Micro-temporal crest factor: $C_{5\text{ms}} = \max |x[n]| / \text{RMS}_{5\text{ms}}$
   - High-frequency burst onset energy ratio (energy above 2 kHz in 5ms burst window relative to vowel baseline)
2. Integrating these 4 micro-temporal burst features ($63 \rightarrow 67$ dimensions) into the champion **DMCT-Net** will sharpen transient occlusion and release boundaries, yielding:
   - **DEV_SEARCH Omission AP** breaking the 30% barrier ($\text{Omission AP} \ge 0.3000$).
   - **DEV_REPLICATION Omission AP** rising to $\ge 0.4350$ and Omission Precision rising to $\ge 0.5000$ (50%).
   - Stop consonant omission false alarm reduction by $\ge 10\%$.

## 3. Predicted Experimental Outcomes
1. **DEV_SEARCH Omission AP:** $\ge 0.3000$ (breaking previous search record of 0.2965).
2. **DEV_REPLICATION Omission AP:** $\ge 0.4350$ with precision $\ge 0.5000$.
3. **Macro AP:** Maintaining $\ge 0.6050$ (certifying 60% milestone preservation).
