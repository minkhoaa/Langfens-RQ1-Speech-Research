# Research Plan: Experiment EXP-93 (Duodeviginti-Plus Extended Router)

## 1. Ensemble Composition (23 Models)
The Viginti-Plus-Pareto super-ensemble unites:
1. Base Undeviginti/Viginti components:
   - PSC-Net, SRAD-Net, NNA-Net, Conf-Net, JSAC-Net, APED-Net, PNC-Net, HTC-Net, VTLN-Net, DMCT-Net, HRTD-Net, CDMR-Net, DSCA-Net, CHTD-Net, UCR-Net, TDIS-Net, P-ALDS-Net, MRTP-Net, IPTM-Net, V-CASR-Net, DP-ALDS-Net, TPD-Net, IP-ALDS-Net, CCSP-Net, ADWA-Net.
2. New Candidate 1: **BAPM-Net (EXP-91)** (Acoustic Boundary-Gated Posterior Mapper, Subs AP = 0.4501).
3. New Candidate 2: **FRTE-Net (EXP-92)** (Formant Resonance Dynamics, Ins Recall = 42.55%, Ins F1 = 0.4061).

## 2. Experimental Execution
1. Load raw prediction posteriors from EXP-86 (Viginti-Pareto), EXP-91 (BAPM-Net), and EXP-92 (FRTE-Net) across `dev_cal`, `dev_search`, and `dev_replication`.
2. Optimize convex mixture weights on `dev_cal` via Dirichlet-constrained grid search:
   $$P_{\text{target}}(k) = (1 - w_1 - w_2) P_{\text{viginti}}(k) + w_1 P_{\text{bapm}}(k) + w_2 P_{\text{frte}}(k)$$
3. Calibrate class thresholds on `dev_cal`.
4. Evaluate on `dev_search` and `dev_replication`.
5. Audit against H88 falsification criteria.
6. Save all artifacts, predictions, metrics, and checksums under `experiments/EXP-93-DUODEVIGINTI-PLUS-EXTENDED-ROUTER/`.
