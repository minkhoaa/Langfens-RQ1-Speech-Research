# LANGFENS — Frozen Model Integration Handoff
## Model Identifier: `LANGFENS-SM13-ASYMMETRIC-ROUTING-V1`
**Governing Protocol:** LANGFENS-RQ1-POSTTEST-CLOSEOUT-01  
**Architecture Role:** RQ1 Research-Selected Primary Production Model  
**Evaluator Standard:** EVAL-V3.1 (`UnifiedEvaluatorV3`, Exact $N$ Targets, $N+1$ Gaps)  
**Safety Certification:** Joint False Warning Rate $\le 0.2000$ (Realized $0.1633$ on held-out non-native speakers)  
**Document Purpose:** Complete, self-contained engineering specification and deployment contract for integrating the frozen SM13 model into downstream speech applications.

---

## 1. Executive Asset Ledger & Cryptographic Integrity

All checkpoints and policy parameters are permanently locked. Before initialization, consuming applications should cryptographically verify the assets against the SHA-256 ledger:

| Component Role | Canonical Repository Path | Verified SHA-256 Checksum | File Size (Bytes) |
|---|---|---|:---:|
| **Target Head Checkpoint** | `experiments/rq1_sm13_formal_train_01/checkpoints/target_sm13_seed101_fulltrain.pt` | `551c5f6079109e8decd49d6fe92ed16ec394a41177b1b87af9e64c95618f6ba6` | 2,829,321 |
| **Gap Head Checkpoint** | `experiments/rq1_sm13_formal_train_01/checkpoints/gap_sm13_seed101_fulltrain.pt` | `34de5bbfc2003176730bfdacbf23b1cfb54ffb861f42d8b0181295ed912b4f98` | 1,774,489 |
| **CTC Projection Checkpoint** | `experiments/rq1_sm13_formal_train_01/checkpoints/ctc_registered_seed101.pt` | `63d3d6147db237a38ec673b5142ba4f4f29ac43991b6124aead745b2a05b015a` | 134,817 |
| **Asymmetric Decision Policy** | `FROZEN_ASYMMETRIC_POLICY.json` | `e74ea31804f2d00d731aefffbfef2d613e6112d7965c4cbf60807b0bcee45023` | 8,258 |
| **Compiled C++ CTC Kernel** | `experiments/rq1_method_bench_04/agent_a/ctc_batch.so` | `e1d4d3b664724b07cfa20216b2cfd3bf9953d6a978f8fc6cbb451f28b438ea2f` | 31,856 |
| **Production Inference Service** | `experiments/rq1_sm13_posttest_closeout_01/sm13_inference_service.py` | Executable Service Script | — |

---

## 2. System Architecture & Feature Extraction Pipeline

The inference pipeline takes raw speech audio and a canonical CMU phoneme transcription, processes multi-layer representations through frozen Wav2Vec2 layers, and routes posterior scores through asymmetric thresholds.

```
Raw Audio (16 kHz) + Canonical Phones (N)
           │
           ▼
┌────────────────────────────────────────┐
│  Forced Alignment (CharSiu Wav2Vec2)  │ ──► Center timestamps c_i (i = 0..N-1)
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│   Frozen Wav2Vec2-960h (13 Layers)     │
└─────────┬────────────────────┬─────────┘
          │                    │
Target Windows (2304D)    Gap Context (1536D)
          │                    │
          ▼                    ▼
┌────────────────────┐ ┌────────────────────┐
│ SM13 Target Head   │ │ SM13 Gap Head      │
│ (13-layer scalar)  │ │ (13-layer scalar)  │
└─────────┬──────────┘ └─────────┬──────────┘
          │                      │
          │ p_neural_S           │ p_gap_I
          ▼                      ▼
┌────────────────────┐     s_I >= 0.886658? ──► INSERTION Alert
│ CTC L6 + E4 Simplex│                          (else NO_INSERTION)
│ 50/50 Fusion       │
└─────────┬──────────┘
          │
          │ p_fixed_O
          ▼
┌────────────────────────────────────────────────────────┐
│              Asymmetric Routing Engine                 │
│  s_S >= 0.949374?  s_O >= 0.857454?                    │
│  Conflict Arbitration: score_margin (s_S - tau_S vs.)  │
└──────────────────────────┬─────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
       SUBSTITUTION                   OMISSION
             (Neither triggered -> CORRECT)
```

### 2.1 Feature Representation Details
1. **Target Feature ($2,304\text{D} \times 13\text{ layers}$):**
   - For each phone center $c_i$, extract an acoustic window of $[-8, +8]$ frames ($17$ frames total at $20\text{ ms}$ step).
   - Take central frames $[3:14]$ ($11$ frames).
   - Feature vector: $\text{Concatenate}[\text{mean}(c), \text{std}(c), c[-1] - c[0]]$ ($768 \times 3 = 2,304\text{D}$).
   - Layer mixing: Weighted sum over $13$ layers using learned softmax layer logits $\mathbf{w}_T \in \mathbb{R}^{13}$ and learned scale $\alpha_T$.
   - Concatenated with 39D one-hot canonical phoneme vector $\mathbf{r}_i \in \mathbb{R}^{39}$.
   - Passed through MLP: $2343 \to 256 \to 128 \to 3$ logits (`CORRECT`, `SUBSTITUTION`, `OMISSION`).

2. **Gap Feature ($1,536\text{D} \times 13\text{ layers}$):**
   - Extracted across $N+1$ boundaries (including boundary before phone $0$ and boundary after phone $N-1$).
   - Left slice: frames $[11:16]$ of preceding phone ($768\text{D}$).
   - Right slice: frames $[1:6]$ of subsequent phone ($768\text{D}$).
   - Boundary feature: $\text{Concatenate}[\text{left}, \text{right}]$ ($1,536\text{D}$).
   - Layer mixing: Weighted sum over $13$ layers using learned softmax layer logits $\mathbf{w}_G \in \mathbb{R}^{13}$ and learned scale $\alpha_G$.
   - Concatenated with 78D one-hot boundary context vector (left 39D + right 39D).
   - Passed through MLP: $1614 \to 128 \to 64 \to 2$ logits (`NO_INSERTION`, `INSERTION`).

3. **E4 Sequence Likelihood Ratios (SLLR, OLLR):**
   - Layer 6 hidden states ($768\text{D}$) projected via frozen `LinearCTC` $\to 43$ classes (42 phones + blank).
   - Fast C++ forward algorithm (`score_ctc_batch`) evaluates $1 + 5N$ hypotheses:
     - $1$ canonical sequence baseline.
     - $4$ substitution competitor hypotheses per position $i$ $\to$ Average Likelihood Ratio $\text{SLLR}_i$.
     - $1$ canonical deletion hypothesis per position $i$ $\to$ Deletion Likelihood Ratio $\text{OLLR}_i$.
   - Form simplex: $p_{\text{E4}} = \text{Softmax}([0, \text{SLLR}_i, \text{OLLR}_i])$.
   - Fused Target Simplex: $p_{\text{fixed}} = 0.5 \times p_{\text{neural}} + 0.5 \times p_{\text{E4}}$.

---

## 3. Asymmetric Decision Rules & Frozen Thresholds

The system decodes errors using decoupled class thresholds rather than a joint simplex argmax:

| Target Class | Operating Score Source | Exact Frozen Threshold ($\tau$) | Tie Cutoff Hash ($2^{64}$ scale) |
|---|---|:---:|:---:|
| **Substitution** | $s_S = p_{\text{neural}}[i, 1]$ (`ScalarMixingTargetHead`) | **`0.9493744373321533`** | `12445346866078397902` |
| **Omission** | $s_O = p_{\text{fixed}}[i, 2]$ ($50/50$ Fused Simplex) | **`0.8574540615081787`** | `2357324174070838657` |
| **Insertion** | $s_I = p_{\text{gap}}[j, 1]$ (`ScalarMixingGapHead`) | **`0.8866584300994873`** | `9157070711166496197` |

### 3.1 Conflict Arbitration Rule (`score_margin`)
When both substitution and omission conditions are simultaneously met on target position $i$:
$$\Delta_S = s_S - \tau_S, \quad \Delta_O = s_O - \tau_O$$
- If $\Delta_S > \Delta_O \implies \mathbf{SUBSTITUTION}$
- If $\Delta_O > \Delta_S \implies \mathbf{OMISSION}$
- If $\Delta_S == \Delta_O \implies \text{tie\_hash}(event\_id) \pmod 2 == 1 \ ? \ \mathbf{SUBSTITUTION} : \mathbf{OMISSION}$
- If neither threshold is crossed $\implies \mathbf{CORRECT}$ (label `0`)

### 3.2 Deterministic Tie-Breaking Specification
When score equals threshold exactly ($s == \tau$):
```python
def tie_hash(event_id: str) -> int:
    prefix = "LANGFENS-MB07-warning-tie-v1:"
    h = hashlib.sha256((prefix + str(event_id)).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")

# Alert triggers if score > tau, OR (score == tau and tie_hash(event_id) >= cutoff)
```

---

## 4. Input & Output Contract Specification

### 4.1 Input Schema
The pipeline accepts:
1. `audio`: A 16,000 Hz, single-channel (mono), 32-bit floating point audio signal normalized in $[-1.0, 1.0]$.
2. `canonical_phones`: An ordered Python list of uppercase CMU phoneme strings of length $N$:
   `["F", "AO", "R", "DH", "AH", "T", "W", "EH", "N", "T", "IY"]`
3. `utterance_id` (optional): Unique string identifier for tracking.
4. `speaker_id` (optional): Identifier of speaker (for accent-aware logging).

### 4.2 Output Schema
The pipeline returns a structured JSON object satisfying the EVAL-V3.1 exact cardinality contract ($N$ target items, $N+1$ gap items):

```json
{
  "utterance_id": "arctic_a0003",
  "speaker_id": "ABA",
  "model_version": "LANGFENS-SM13-ASYMMETRIC-ROUTING-V1",
  "pedagogical_safety_guarantee": "Realized Joint FWR <= 0.2000 (0.1633 on held-out test cohort)",
  "summary": {
    "num_targets": 39,
    "num_gaps": 40,
    "total_warnings_emitted": 2,
    "target_warnings": 2,
    "gap_warnings": 0
  },
  "targets": [
    {
      "target_index": 37,
      "canonical_phone": "D",
      "event_id": "L2_ARCTIC|ABA|arctic_a0003|target|37",
      "prediction": "SUBSTITUTION",
      "is_warning": true,
      "score_substitution": 0.961205,
      "score_omission": 0.012450,
      "threshold_substitution": 0.9493744373321533,
      "threshold_omission": 0.8574540615081787,
      "margin": 0.011831,
      "e4_sllr": 3.8412,
      "e4_ollr": -4.1205
    }
  ],
  "gaps": [
    {
      "gap_index": 24,
      "left_phone": "NG",
      "right_phone": "DH",
      "event_id": "L2_ARCTIC|ABA|arctic_a0003|gap|24",
      "prediction": "INSERTION",
      "is_warning": true,
      "score_insertion": 0.914520,
      "threshold_insertion": 0.8866584300994873,
      "margin": 0.027862
    }
  ]
}
```

---

## 5. Warning Semantics & UI Guidance

### 5.1 What an Alert Means
- **High Pedagogical Confidence:** An alert indicates the model has detected severe phonetic divergence from the canonical target with high empirical precision ($75.6\%$ on substitutions, $68.8\%$ on omissions).
- **Actionable Corrective Target:** Applications should provide clear corrective instruction (e.g., highlighting articulatory place/manner errors for substitutions, or noting missing sounds for omissions).

### 5.2 What Silence (Absence of Alert) Means
- **Safety Ceiling Enforcement:** To guarantee that learners do not experience more than 1 false alarm per 5 utterances ($\text{FWR} \le 0.20$), decision thresholds are positioned in the extreme posterior tail ($\approx 0.95$ for substitution).
- **The Low-Recall Reality:**
  - Substitution recall is $4.34\%$ ($2,118$ out of $2,214$ errors not flagged).
  - Omission recall is $14.84\%$ ($568$ out of $667$ errors not flagged).
  - Insertion recall is $12.30\%$ ($107$ out of $122$ errors not flagged).
- **UI Warning:** **Never tell a learner their pronunciation was "100% native" or "perfect" simply because no alert fired.** The UI should state: *"No severe pronunciation errors detected."*

---

## 6. Minimal Python Integration Snippet

Downstream developers can instantiate and run inference using `SM13InferencePipeline`:

```python
from pathlib import Path
from experiments.rq1_sm13_posttest_closeout_01.sm13_inference_service import SM13InferencePipeline

# 1. Initialize Pipeline (validates SHA-256 checkpoints automatically)
pipeline = SM13InferencePipeline(device="cuda")

# 2. Define inputs
audio_path = "data/derived/l2arctic/wavs_16k/ABA_arctic_a0003.wav"
canonical_phones = [
    "F", "AO", "R", "DH", "AH", "T", "W", "EH", "N", "T", "IY", "IH", "TH",
    "T", "AY", "M", "DH", "AE", "T", "IY", "V", "N", "IH", "NG", "DH", "AH",
    "T", "UW", "M", "EH", "N", "SH", "UH", "K", "HH", "AE", "N", "D", "Z"
]

# 3. Execute Diagnostic Forward Pass
result = pipeline.diagnose_utterance(
    wav_path_or_array=audio_path,
    canonical_phones=canonical_phones,
    utterance_id="arctic_a0003",
    speaker_id="ABA"
)

# 4. Process Output
print(f"Total Warnings: {result['summary']['total_warnings_emitted']}")
for t in result["targets"]:
    if t["is_warning"]:
        print(f"  Alert at Phone {t['target_index']} ({t['canonical_phone']}): {t['prediction']} (Score={t['score_substitution']:.4f})")
for g in result["gaps"]:
    if g["is_warning"]:
        print(f"  Alert at Gap {g['gap_index']} ({g['left_phone']}-{g['right_phone']}): INSERTION (Score={g['score_insertion']:.4f})")
```

---

## 7. Reproducible CLI Verification Command

To verify the frozen model end-to-end on any local audio file:

```bash
# Execute standalone CLI inference
/home/khoa/Projects/New-Method-Research/.venv/bin/python \
  experiments/rq1_sm13_posttest_closeout_01/sm13_inference_service.py \
  --wav data/derived/l2arctic/wavs_16k/ABA_arctic_a0003.wav \
  --phones "F AO R DH AH T W EH N T IY IH TH T AY M DH AE T IY V N IH NG DH AH T UW M EH N SH UH K HH AE N D Z" \
  --device cuda \
  --output /tmp/sm13_handoff_verification.json
```

---

## 8. Handoff Certification

The frozen SM13 Asymmetric Routing model artifacts and inference contract are complete, validated, and ready for downstream integration.

**Certified by:** LANGFENS RQ1 Research Lead & Systems Auditor  
**Date:** November 11, 2026
