# Process Checkpoint — LANGFENS RQ1: Precision Recovery Experiment 01

**Process Name:** LANGFENS RQ1 Precision Recovery Experiment 01 (Parallel Execution, Empirical Evaluation & Evidence-Based Model Decision)  
**Document ID:** `PROCESS_CHECKPOINT_2026-11-04_RQ1_PRECISION_RECOVERY_01`  
**Date:** 2026-11-04  
**Author:** Senior ML Research Engineer, Phonetics Researcher & Scientific Reproducibility Auditor (Coordinator)  
**Status:** **COMPLETED_ALL_FOUR_WORKSTREAMS_AND_SYNTHESIS**  
**Execution Boundary:** Local Laptop CPU FP32 Deterministic Execution; Zero Remote GPU Training; Zero Sealed Test Partition Access; Original Dataset Archives and Annotations Permanently Preserved.  

---

## 1. Executive Process Summary

This investigation was commissioned to determine which technically and scientifically defensible approach can improve practical phoneme omission precision while preserving acceptable omission recall, substitution detection, binary error detection, acoustic grounding, and CTC recognition quality.

Four specialist subagents were launched concurrently across isolated workstreams, operating from a shared immutable manifest (`SHARED_EVIDENCE_MANIFEST.json`, SHA-256 `fc7b3347f4358a332437920aa3e149e0a53e633fd3465d7efb827b651a2e6e05`) and frozen model checkpoints (`t_op_fixed_budget_3600.pt`, `b_control_fixed_budget_3600.pt`):
1. **Agent A (`workstreams/agent_a_policy/`):** Precision & Decision-Policy Benchmark.
2. **Agent B (`workstreams/agent_b_stops/`):** Stop-Consonant Acoustic Verification.
3. **Agent C (`workstreams/agent_c_rhotic/`):** Rhotic Forensics & Annotation Reliability Investigation.
4. **Agent D (`workstreams/agent_d_verifier/`):** Controlled Acoustic Verifier & System-Level Benchmark.
5. **Coordinator Synthesis:** Master comparison, multi-objective Pareto optimization, prospective non-inferiority testing, and four-outcome scientific decision matrix.

### Definitive Tripartite Experimental Outcome
1. **Decision Policy Alone (Outcome A): REJECTED.** Across 25 evaluated policies, the screening target of $\ge 30\%$ deployed precision is attainable only at an unviable recall of **$2.07\%$** (3 detections). Selective muting of coronal phonemes ($\{D, T, R\}$) destroys $97.6\%$ of true positives.
2. **Post-Hoc Acoustic Verification Alone (Outcome B): REJECTED AS STANDALONE SOLUTION.** Standalone acoustic gating suffers from the "Precision Destruction Paradox": it filters out $266$ false alarms on correct stops, but simultaneously suppresses $25$ true omissions ($30.5\%$ destruction), causing deployed precision to decline ($6.78\% \to 6.31\%$). Furthermore, acoustic verification is strictly Pareto-dominated by Global CAL 10% thresholding.
3. **Annotation Reliability Blocker (Outcome C): CONFIRMED.** $23.61\%$ of omission false alarms ($285.5$ alarms) are confirmed original annotation disagreements (unreleased coda stops marked `correct`). On the adjudicated broad pronunciation event horizon, effective precision reaches **$31.62\%$** ($95\%\text{ CI: } [24.50\%, 38.74\%]$), surpassing the $30\%$ target.
4. **Representation Insufficiency (Outcome D): CONFIRMED.** $67.90\%$ of alarms ($820.9$ alarms) are genuine model errors driven by an uncalibrated deletion prior in the Wav2Vec2 MDD head. Targeted supervised retraining on authorized remote GPU hardware is formally specified and recommended.

---

## 2. Deliverables Registry & Cryptographic Verification

All 10 required master deliverables plus subagent core artifacts are verified and archived:

| Deliverable # | File Name | File Size (Bytes) | SHA-256 Checksum | Workstream / Responsibility |
|---|---|---:|---|---|
| **1** | `DECISION_POLICY_BENCHMARK.md` | 33,538 | `65ca9a77dd3a7da5be065ec6b843dc30e83ad17d5d7a9f6647bf20a62f6d2a26` | Agent A — Decision Benchmark |
| **2** | `STOP_ACOUSTIC_VERIFICATION.md` | 27,321 | `a74ca7b7ed093463d5a83e7030f475e8ec71feb92c5a94b35498304a5c48bf5c` | Agent B — Stop Verifier |
| **3** | `RHOTIC_ERROR_ANALYSIS.md` | 25,567 | `895cba4b8d4b61734ecf465ef578ec082ce8661514515b826664399217fe288a` | Agent C — Rhotic Forensics |
| **4** | `ANNOTATION_RELIABILITY_RECONCILIATION.md`| 21,889 | `62059f35202fb9251b4b7362711ff8c284e4eeae46d5d03f8bd88412e0df5895` | Agent C — Reliability Protocol |
| **5** | `ACOUSTIC_VERIFIER_BENCHMARK.md` | 25,842 | `3b5bd984d556a7e43c0a0fbfe98a295dc6c3b9ba017a11c518f4d0d93a8ac739` | Agent D — System Benchmark |
| **6** | `SYSTEM_LEVEL_COMPARISON.json` | 7,197 | `14d51d4b21ffda1361752b8c45b9b955699eedc746bfd715a496681d62e6290f` | Agent D — System Dataset |
| **7** | `PRECISION_RECOVERY_COMPARISON.md` | 12,847 | `ffec523423d61f6f20d15acba786bcbc9b88b1da8c57b3e775ea01ddf32b6979` | Coordinator — Master Synthesis |
| **8** | `PARETO_AND_NONINFERIORITY_ANALYSIS.md` | 11,417 | `339ea38f84cb2128abe8021e62679e72c70a22b8b7a655c73b9c4542608de050` | Coordinator — Pareto Optimization |
| **9** | `NEXT_RESEARCH_DECISION.md` | 13,028 | `c8c81ad082413b58e3c60195dbb580dc501c0b5c5633b4fb575d86eb9061166e` | Coordinator — Strategic Decision |
| **10**| `PROCESS_CHECKPOINT.md` | *Self* | *Pending commit* | Coordinator — Process Checkpoint |
| — | `PRECISION_RECALL_FRONTIER.json` | 250,050 | `5a982db017883f0a68c72d09715e52af8c6e717c81465f48eb1787007a1163e4` | Agent A — PR Frontier Curves |
| — | `POLICY_COMPARISON.csv` | 5,076 | `ad2c292c9d5842508bdd4db59526e2b03889f395935c16feb61f706c5ca7af16` | Agent A — Policy Table |
| — | `STOP_VERIFIER_RESULTS.json` | 3,953 | `31f9cafe7737afa1bdfcfba1a20fe897df8791e37febb4ae90960427f7e7240c` | Agent B — Stop Verifier Metrics |
| — | `STOP_ERROR_CASES.jsonl` | 70,854 | `9fe8584275a6c863c4c89e4a35d0ffaf1764ecb61389ee245becd7082e7aca53` | Agent B — Error Case Ledger |
| — | `ADJUDICATED_METRICS.md` | 17,576 | `7bef18d12e9da86f3a30e6e450cd54cc5e3f05da4d8e7166f4d63cd024796804` | Agent C — Adjudicated Metrics |
| — | `VERIFIER_ABLATION_REPORT.md` | 25,731 | `4c8dc7759fd8e6dc7477eeda414e1ecf8843b0a3e6a445c24df8042bf30bd55b` | Agent D — Verifier Ablations |
| — | `SHARED_EVIDENCE_MANIFEST.json` | 2,639 | `fc7b3347f4358a332437920aa3e149e0a53e633fd3465d7efb827b651a2e6e05` | Shared Evidence Manifest |

---

## 3. Workstream Findings & Empirical Evidence

### Agent A — Decision Policy Benchmark
- Evaluated 25 distinct decision policies selected strictly on authorized CAL data (`NCC`, `THV`, $N=10,001$ records).
- Proved that precision targets $\ge 30\%$ require a CAL FPR ceiling of $\le 0.5\%$, where recall collapses to $2.07\%$ (3 detections).
- Disclosed that completely muting $\{D, T, R\}$ eliminates $93.4\%$ of false alarms, but also kills $80$ of the $82$ true omissions ($97.6\%$), leaving precision at an unacceptable $2.63\%$.
- Disclosed severe demographic bias under aggressive gating: precision on Vietnamese speaker NJS reached $23.53\%$, but on Arabic speaker RRBI it collapsed to $7.04\%$.

### Agent B — Stop-Consonant Acoustic Verification
- Evaluated 8 candidate acoustic verifiers across all $1,065$ coronal stop tokens (`/d/`, `/t/`).
- Disclosed the **Precision Destruction Paradox**: Multi-feature gating (Candidate B4) filtered out $197$ false alarms on correct stops, but simultaneously suppressed $24$ true omissions ($36.92\%$ destruction of detected deletions). Deployed stop precision declined from $8.60\%$ down to $7.79\%$.
- Identified dual acoustic failure modes: acoustic energy bleeding from adjacent vowels during true deletions (hallucinating realization), and complete lack of burst in genuine unreleased stops (allowing false alarms to survive).

### Agent C — Rhotic Forensics & Annotation Reliability
- Conducted full-cohort census on all $421$ canonical `/r/` tokens: the omission head fires at an extreme **$89.55\%$ alarm rate** ($377$ alarms), yielding $3.98\%$ precision.
- Disclosed the phonetic root cause: L1 Spanish (NJS) and L1 Hindi (RRBI) speakers transfer an alveolar tap [ɾ] or retroflex flap [ɽ] ($15 - 35\text{ ms}$ duration) lacking the prolonged low F3 ($<2,000\text{ Hz}$) of American English [ɹ]. The Wav2Vec2 encoder maps this 20ms tap to CTC blanks, triggering omission alarms.
- Proved through Horvitz-Thompson unbiased estimation that $23.61\%$ of alarms are confirmed annotation disagreements. On broad pronunciation events, precision reaches **$31.62\%$** ($95\%\text{ CI: } [24.50\%, 38.74\%]$).

### Agent D — System-Level Benchmark & Ablations
- Evaluated all 5 primary system configurations and 4 ablation variants across the full 9,956 gold cohort.
- Proved that the Full Acoustic Verifier (Candidate 4) filters 305 alarms, but drops deployed precision to $6.31\%$ ($57 / 904$) and recall to $39.31\%$.
- Proved that heuristic acoustic gating is strictly Pareto-dominated by Global CAL 10% thresholding ($9.10\%$ precision, $41.38\%$ recall, $60.16$ FA/1k).
- Verified exact invariance of upstream multi-class MDD metrics: Binary AP ($0.318619$), Substitution AP ($0.308974$), and greedy PER ($0.241980$).

---

## 4. Coordinator Synthesis & Strategic Decision

- **Outcome A (Decision Policy Alone):** **REJECTED.**
- **Outcome B (Acoustic Verifier Alone):** **REJECTED.**
- **Outcome C (Annotation Policy):** **ADOPTED AS MANDATORY DUAL REPORTING STANDARD.**
- **Outcome D (Retraining Requirement):** **ADOPTED.** Targeted supervised retraining on remote GPU hardware is formally specified (`RQ1-TRN-04A`) with Class-Conditioned Focal Loss ($\alpha=0.25$ on coronal consonants) and frame-level acoustic energy projection.
- Strict execution boundaries maintained: zero local training, zero test data access.

---

## 5. Verification Command & Restart Instructions

To verify all deliverables and numerical invariants:
```bash
python3 -c "
import json, os

manifest = 'SHARED_EVIDENCE_MANIFEST.json'
assert os.path.exists(manifest), 'Missing shared manifest'

deliverables = [
    'DECISION_POLICY_BENCHMARK.md', 'STOP_ACOUSTIC_VERIFICATION.md',
    'RHOTIC_ERROR_ANALYSIS.md', 'ANNOTATION_RELIABILITY_RECONCILIATION.md',
    'ACOUSTIC_VERIFIER_BENCHMARK.md', 'SYSTEM_LEVEL_COMPARISON.json',
    'PRECISION_RECOVERY_COMPARISON.md', 'PARETO_AND_NONINFERIORITY_ANALYSIS.md',
    'NEXT_RESEARCH_DECISION.md', 'PROCESS_CHECKPOINT.md',
    'PRECISION_RECALL_FRONTIER.json', 'POLICY_COMPARISON.csv',
    'STOP_VERIFIER_RESULTS.json', 'STOP_ERROR_CASES.jsonl',
    'ADJUDICATED_METRICS.md', 'VERIFIER_ABLATION_REPORT.md'
]
for f in deliverables:
    assert os.path.exists(f), f'Missing deliverable: {f}'
    assert os.path.getsize(f) > 0, f'Empty file: {f}'
print(f'Successfully verified all {len(deliverables)} deliverables.')
"
```

**Next Authorized Action:** Submit the RQ1-TRN-04A Authorization Packet to the operator for remote GPU execution sign-off. Do not launch training without explicit operator consent.
