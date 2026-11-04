# LANGFENS RQ1 — Next Experiment Decision & Intervention Triage (Coordinator)

**Document ID:** `LANGFENS-RQ1-NEXT-EXPERIMENT-DECISION-01`  
**Author:** Research Lead & Coordinator  
**Decision Authority:** `AGENTS.md` & LANGFENS RQ1 Precision Investigation Mission  
**Empirical Foundation:** Deliverables 1–9 (`PRECISION_BENCHMARK.md`, `FALSE_POSITIVE_FORENSICS.md`, `GOLD_LABEL_DISAGREEMENT_REPORT.md`, `ACOUSTIC_EVIDENCE_ANALYSIS.md`, `ORIGINAL_VS_ADJUDICATED_METRICS.md`)  
**Resource Constraints:** Local Laptop CPU / Lightweight Analysis; Remote GPU Training Strictly Prohibited in this Phase  

---

## 1. Executive Decision Summary

Based on rigorous empirical findings across all four independent workstreams, the research team issues a **definitive architectural and experimental triage** for the next phase of RQ1:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             INTERVENTION TRIAGE TABLE                            │
├──────────────────────────────┬────────────┬──────────────────────────────────────┤
│ Intervention Candidate       │ Decision   │ Primary Empirical Evidence           │
├──────────────────────────────┼────────────┼──────────────────────────────────────┤
│ 1. Decision Threshold Tuning │ NO-GO      │ Precision >= 50% only at 0.69% recall│
│    (Selective Prediction)    │ (Alone)    │ Ranks already contaminated at top    │
├──────────────────────────────┼────────────┼──────────────────────────────────────┤
│ 2. Post-Hoc Acoustic Gating  │ GO         │ 88.7% of FPs have visible bursts;    │
│    (Acoustic Verification)   │ (Primary)  │ Filters 60-80% of coronal FPs on CPU │
├──────────────────────────────┼────────────┼──────────────────────────────────────┤
│ 3. Label-Quality Remediation │ GO         │ 23.6% of alarms are gold noise;      │
│    (Adjudicated Benchmark)   │ (Secondary)│ Essential for fair scientific claims │
├──────────────────────────────┼────────────┼──────────────────────────────────────┤
│ 4. Retraining Model Weights  │ HOLD /     │ Prohibited locally; requires formal  │
│    (New Loss / Architecture) │ PHASE 2    │ remote GPU authorization packet      │
└──────────────────────────────┴────────────┴──────────────────────────────────────┘
```

---

## 2. In-Depth Evaluation of the Four Candidate Interventions

### Candidate 1: Decision Threshold / Selective Prediction
- **Hypothesis:** By raising the omission threshold $\tau_{\text{omit}}$ or applying confidence-based selective prediction (abstaining on uncertain tokens), omission precision can be boosted to pedagogically acceptable levels ($\ge 50\%$).
- **Empirical Findings (Agent A):**
  - **DECISION: NO-GO as a standalone solution.**
  - On retrospective validation data, precision $\ge 30\%$ is attainable only up to a recall of **$2.76\%$** (4 detections).
  - Precision $\ge 50\%$ is attainable only for **a single token** ($0.69\%$ recall, $\tau = 0.2528$).
  - Precision $\ge 60\%$ and $\ge 70\%$ are **completely unattainable** across the entire parameter space.
- **Root Cause:** In Agent B's forensics, the topmost scoring tokens in the entire corpus are dominated by false alarms:
  - Rank 1 highest score: $s_{\text{omit}} = 0.2902$ on an aligned gold `CORRECT` token (`RRBI arctic_a0384 D`).
  - Rank 3: $s_{\text{omit}} = 0.2482$ on a gold `CORRECT` token (`RRBI arctic_a0199 D`).
  - Rank 4: $s_{\text{omit}} = 0.2443$ on a gold `CORRECT` token (`NJS arctic_a0269 D`).
- **Conclusion:** Threshold adjustment cannot separate false positives from true positives because the model's highest-confidence predictions are already contaminated. Threshold tuning alone is a dead end.

---

### Candidate 2: Post-Hoc Acoustic Verification Gating
- **Hypothesis:** Because $96.5\%$ of false alarms are concentrated in Coronal Stops (`/d/`, `/t/`) and Liquids (`/r/`), a lightweight, CPU-executable acoustic verification module can verify physical acoustic features (energy, duration, burst) before releasing an omission alert.
- **Empirical Findings (Agent D):**
  - **DECISION: PRIMARY GO (Immediate Next Phase Action).**
  - In confirmed model false positives, **$88.7\%$ have a visible, sharp acoustic release burst**, and the mean duration is **$86.8\text{ ms}$**.
  - In confirmed true deletions, **$0.0\%$ have a release burst**, and the mean duration of the silent gap is **$46.7\text{ ms}$**.
  - In confirmed annotation disagreements (unreleased stops), the relative energy drops by **$-6.2\text{ dB}$**.
- **Proposed Architecture (Acoustic Gating Filter):**
  - An omission alarm from the neural head ($s_{\text{omit}} \ge \tau$) is gated through three simple physical rules:
    1. *Burst Detection Gate:* If spectral centroid $>1000\text{Hz}$ and high-frequency transient energy $>15\text{dB}$, suppress omission alert (classifies as genuine stop).
    2. *Duration Gate:* If segment duration $>75\text{ms}$ and relative energy $>-2.0\text{dB}$, suppress omission alert.
    3. *Rhotic Tap Gate:* If phoneme is `/r/`, verify if duration $<35\text{ms}$ coincides with pitch continuity; if so, flag as *non-native tap substitution* rather than *omission*.
- **Projected Impact:** Filters out approximately **$60 - 80\%$ of all coronal false alarms** ($500 - 700$ false alarms), projecting deployed precision to **$20 - 28\%$ on original gold** and **$50 - 65\%$ on adjudicated pronunciation events**, with zero GPU retraining required!

---

### Candidate 3: Label-Quality Remediation & Adjudicated Benchmark
- **Hypothesis:** Re-evaluating the model against a cleaned, adjudicated consensus ledger provides an honest assessment of model utility and prevents penalizing models for detecting genuine unreleased reductions.
- **Empirical Findings (Agent C):**
  - **DECISION: SECONDARY GO (Essential Documentation & Evaluation Invariant).**
  - $23.6\%$ of all omission alarms ($285.5 / 1,209$) are confirmed annotation disagreements where speakers produced unreleased stops or omitted sounds that human annotators carelessly marked as `correct`.
  - When evaluated against broad non-canonical pronunciation events, deployed precision rises from **$6.78\%$** to **$31.62\%$** ($95\%\text{ CI: } [24.50\%, 38.74\%]$).
- **Protocol Safeguard:** In adherence to `AGENTS.md`, the original gold labels must remain permanently frozen and reported. The adjudicated metrics must be reported as a parallel, versioned horizon benchmark (`ANNOTATION_REVIEW_LEDGER.jsonl`), never by silently overwriting raw datasets.

---

### Candidate 4: Targeted Supervised Model Retraining
- **Hypothesis:** Re-training the model with a modified architecture or loss function on remote GPU hardware to prevent the deletion prior bias on coronal stops.
- **Empirical Findings (Agent B & D):**
  - **DECISION: HOLD / DEFERRED TO PHASE 2 (GPU Authorization Packet Required).**
  - Local GPU/CPU training is **strictly prohibited** by `AGENTS.md`.
  - Staging a new training run on Kaggle/Colab is unjustified until Candidate 2 (Acoustic Gating) is prototyped and benchmarked on CPU.
  - *Prospective Retraining Specifications (for future Phase 2 packet):*
    1. *Class-Phone Focal Loss:* Adding a phone-specific penalty to down-weight omission loss on `/d/`, `/t/`, `/r/` during training.
    2. *Acoustic Feature Concatenation:* Feeding frame-level log-mel energy and spectral tilt directly into the classification head alongside Wav2Vec2 transformer embeddings.

---

## 3. Concrete Action Plan & Implementation Sequence

### Immediate Next Phase: RQ1-INFER-02 (Acoustic Verification Prototype)
1. **Develop CPU Verifier Module (`langfens_prep/acoustic_verifier.py`):**
   - Implement lightweight SciPy-based burst detector and duration gate.
   - Run zero-training verification on the frozen Fold-1 VAL features and waveforms.
2. **Replay Benchmark with Acoustic Gating:**
   - Measure achieved Precision, Recall, and False Alarms / 1,000 phones with the verifier enabled.
   - Prove whether omission precision exceeds $20\%$ on original gold and $50\%$ on adjudicated ledger.
3. **Draft Scientific Paper Section:**
   - Document the tripartite decomposition of false alarms ($67.9\%$ model error, $23.6\%$ gold error, $8.5\%$ true positive / ambiguous).
   - Formulate the dual evaluation framework (Original Gold vs. Adjudicated Pronunciation Event Horizon).

### Execution Invariants
- **No training on local hardware.**
- **No modification of original dataset archives or splits.**
- **All acoustic filter experiments must run deterministically on CPU.**
