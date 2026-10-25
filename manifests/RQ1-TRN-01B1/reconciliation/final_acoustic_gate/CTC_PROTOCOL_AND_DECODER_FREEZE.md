# LANGFENS — RQ1-TRN-01B1-R2: CTC Protocol & Decoder Policy Freeze

**Gate Reference:** RQ1-TRN-01B1-R2 — Acoustic-Conditioned MDD & Final Source Provenance Gate  
**Deliverable Path:** `manifests/RQ1-TRN-01B1/reconciliation/final_acoustic_gate/CTC_PROTOCOL_AND_DECODER_FREEZE.md`  
**Experimental Metric References:**  
- `manifests/RQ1-TRN-01B1/reconciliation/TRAIN_ONLY_GENERALIZATION_METRICS.json`  
- `manifests/RQ1-TRN-01B1/reconciliation/CTC_LENGTH_RECOVERY_REPORT.md`  
**Lead Speech ML Systems Engineer & Scientific Reproducibility Auditor:** Senior Speech ML Research Engineer  
**Policy Status:** `CTC_PROTOCOL_FREEZE: APPROVED & FROZEN`  

---

## 1. Executive Summary & Policy Directives

In the RQ1-TRN-01B1 pilot, CTC training on real speech suffered catastrophic blank collapse:
- Decoded length ratio: $L_{\text{hyp}} / L_{\text{ref}} = 0.221$
- Deletions: $77.9\%$ of reference tokens missed
- PER: $> 0.81$

The reconciliation campaign successfully diagnosed and resolved this collapse by pinning a **frozen-backbone optimization protocol** and scaling training exposure to $256$ utterances ($9,308$ tokens), achieving:
- Greedy decoded length ratio: $\mathbf{0.8573}$ (Seed 1337) / $\mathbf{0.8582}$ (Seed 42)
- Greedy Phone Error Rate (PER): $\mathbf{0.4844}$ (Seed 1337) / $\mathbf{0.4863}$ (Seed 42)
- Blank fraction: $\mathbf{0.8133}$ (healthy speech framing, no collapse)

Task E mandates:
1. Retaining this verified frozen-backbone CTC recovery as the official experimental baseline.
2. Formalizing decoder penalty tuning ($\beta \in [0.0, 4.0]$) as **exploratory internal-probe optimization**.
3. Strictly prohibiting claims that $\beta = 1.0$ is an independently confirmed optimum.
4. Preserving a predeclared default decoder (Greedy $\beta = 0.0$) and deferring hyperparameter selection to an authorized tuning partition.
5. Conserving GPU budget by prohibiting repetitive re-execution of verified CTC baselines.

---

## 2. Pinned Frozen-Backbone CTC Baseline Specification

The following baseline protocol is frozen across all multi-task experiments:

| Protocol Parameter | Specification | Verification Standard |
| :--- | :--- | :--- |
| **Acoustic Backbone** | Hugging Face `facebook/wav2vec2-base-960h` | SHA `22aad52d435eb6dbaf354bdad9b0da84ce7d6156` |
| **Backbone Freezing** | Strictly frozen feature extractor and transformer encoder | `requires_grad = False` across all 12 transformer layers |
| **CTC Head Architecture** | Linear projection $\mathbb{R}^{768 \to 40}$ | $40$ classes: index $0 = \text{blank}$, indices $1..39 = \text{ARPAbet}$ |
| **Loss Formulation** | PyTorch `nn.CTCLoss(blank=0, zero_infinity=False)` | Enforced via strict runtime length contract assertions |
| **Predeclared Default Decoder**| **Standard Greedy CTC Collapse ($\beta = 0.0$)** | $\arg\max_c \to \text{collapse adjacent identical} \to \text{remove blank}$ |
| **Holdout Ref Tokens ($N=64$)**| $2,081$ phonemes across speakers `SVBI` & `ZHAA` | Exact protocol manifest verification |
| **Holdout Hyp Tokens (Greedy)**| $1,784$ phonemes | Decoded length ratio = $\mathbf{0.8573}$ |
| **Greedy Holdout PER** | $\mathbf{0.4844}$ (Seed 1337) / $\mathbf{0.4863}$ (Seed 42) | Cross-seed variance $\sigma_{\text{PER}} \le 0.0019$ |
| **Acoustic Dependence** | Strict | Mismatched Audio PER $= \mathbf{0.9471}$; Silence emits $0$ tokens |

---

## 3. Exploratory Internal-Probe Decoder Sweep ($\beta \in [0.0, 4.0]$)

In the reconciliation campaign, a blank penalty $\beta$ was applied to the CTC log-probabilities during frame decoding:
$$\tilde{z}_t[0] = z_t[0] - \beta, \quad \tilde{z}_t[c] = z_t[c] \quad (c \ge 1)$$

The resulting empirical sweep on `internal_holdout_probe_64` produced:

| Penalty Parameter $\beta$ | Hyp Tokens $L_{\text{hyp}}$ | Ref Tokens $L_{\text{ref}}$ | Length Ratio $L_{\text{hyp}}/L_{\text{ref}}$ | Total Edits | Substitutions | Deletions | Insertions | Phone Error Rate (PER) | Exploration Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$\beta = 0.0$ (Default)** | **$1,784$** | **$2,081$** | **$0.8573$** | **$1,008$** | **$575$** | **$365$** | **$68$** | **$0.4844$** | **Predeclared Official Default** |
| **$\beta = 1.0$** | $2,156$ | $2,081$ | $1.0360$ | $970$ | $663$ | $116$ | $191$ | $0.4661$ | Exploratory Probe Observation |
| **$\beta = 2.0$** | $2,370$ | $2,081$ | $1.1389$ | $1,047$ | $684$ | $37$ | $326$ | $0.5031$ | Exploratory Probe Observation |
| **$\beta = 3.0$** | $2,532$ | $2,081$ | $1.2167$ | $1,160$ | $673$ | $18$ | $469$ | $0.5574$ | Exploratory Probe Observation |
| **$\beta = 4.0$** | $2,678$ | $2,081$ | $1.2869$ | $1,287$ | $662$ | $14$ | $611$ | $0.6185$ | Exploratory Probe Observation |

### 3.1 Scientific Erratum & Over-Fitting Warning
- **The Finding:** While $\beta = 1.0$ produced a lower PER ($0.4661$ vs $0.4844$) and near-unity length ratio ($1.036$) on the 64-utterance holdout, **this result is strictly an exploratory observation on a two-speaker partition**.
- **The Risk:** Selecting $\beta = 1.0$ as an established optimum based on this internal probe constitutes hyperparameter tuning on the evaluation partition. Spanish (`SVBI`) and Arabic (`ZHAA`) speakers have idiosyncratic speaking rates and acoustic phoneme durations that may favor $\beta = 1.0$, while other accents (e.g. Mandarin, Korean, Hindi) may experience insertion over-generation.
- **Formal Erratum:** Any prior reference describing $\beta = 1.0$ as an "independently confirmed optimum" is formally corrected. $\beta = 1.0$ is designated as an **exploratory pilot observation**.

---

## 4. Frozen Decoder Selection Protocol for Future Phases

To maintain absolute scientific reproducibility, the following decoder policy is established:

1. **Predeclared Default Baseline:**  
   All future reporting of the frozen-backbone CTC baseline must cite **Standard Greedy Decoding ($\beta = 0.0$)**:
   $$\text{PER}_{\text{baseline}} = \mathbf{0.4844} \quad (L_{\text{hyp}}/L_{\text{ref}} = 0.8573)$$
2. **Authorized Tuning Partition Required:**  
   Hyperparameter optimization for:
   - Decoding blank penalty $\beta \in [0.0, 3.0]$
   - Prefix beam search beam width $W \in [10, 100]$
   - N-gram phoneme language model weight $\alpha_{\text{LM}} \in [0.0, 2.0]$
   must be conducted **exclusively on an authorized development tuning partition** (e.g. the official Fold 1 validation partition containing speakers `NJS` and `RRBI`) once formal Phase 2 calibration is authorized.
3. **No Retrospective Test Tuning:**  
   Hyperparameters tuned on the authorized development partition must be locked before running inference on the official held-out test split.
4. **Conservation of GPU Budget:**  
   No further GPU compute will be expended running redundant CTC training runs on the internal probe. The frozen CTC baseline is preserved bit-identically in checkpoint `rq1_trn_01b1_r_frozen_probe_seed1337_step200.pt`.
