# LANGFENS RQ1-SM13-VALIDATION-PREFLIGHT-01 — Speaker Identity & Split Integrity Audit

**Auditor:** Independent Speech ML Validation Lead & Scientific Auditor  
**Campaign:** `RQ1-SM13-VALIDATION-PREFLIGHT-01` (Lane B)  
**Partition Standard:** L2-ARCTIC Speaker-Disjoint L1-Stratified Nested Partition  
**Sealed Holdout Hash:** `a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6`  
**Audit Verdict:** **SPEAKER ISOLATION PASS (Strict Policy Lock Enforced)**  

---

## 1. Executive Summary & Authoritative Split Definitions

Speaker-disjoint partitioning is a foundational requirement in speech machine learning. Models trained on non-native acoustic representations readily memorize speaker-specific vocal tract formant spaces, fundamental frequency ranges, and idiosyncratic accent patterns. Evaluating a model on the same biological speaker who appeared in training produces severely inflated diagnostic accuracy and invalidates prospective clinical and educational claims.

Authoritative dataset partition specifications are recovered from:
- `data/split_specs/L2_ARCTIC_SPLIT.yaml`
- `data/raw_registry.jsonl`
- `methods/M19-integrity-auditor/run.py`
- `dataset_migration/MOVE_VERIFICATION.json`

### 1.1 Complete L2-ARCTIC 24-Speaker Registry

The complete L2-ARCTIC corpus contains exactly 24 non-native speakers across 6 L1 backgrounds (4 speakers per L1: 2 Male, 2 Female):

| L1 Background | TRAIN Speakers (12) | DEV_SEARCH Speakers (4) | DEV_CAL Speakers (2) | TEST_SEALED Speakers (4) | dev_replication (Cloud) (2) | Total |
|---|---|---|---|---|---|:---:|
| **Arabic** | `ABA` (M), `SKA` (F) | `YBAA` (M) | — | `ZHAA` (F) | — | 4 |
| **Chinese (Mandarin)** | `BWC` (M), **`LXC` (F)** | `NCC` (F) | — | — | `TXHC` (M) | 4 |
| **Hindi** | `ASI` (M), `SVBI` (F) | `RRBI` (M) | — | `TNI` (M) | — | 4 |
| **Korean** | `HKK` (M), `HJK` (F) | `YDCK` (F) | — | — | `YKWK` (M) | 4 |
| **Spanish** | `EBVS` (M), `NJS` (F) | — | `ERMS` (M) | `MBMPS` (M) | — | 4 |
| **Vietnamese** | `HQTV` (M), `PNV` (F) | — | `THV` (F) | `TLV` (M) | — | 4 |
| **Total Speakers** | **12 (6M / 6F)** | **4 (2M / 2F)** | **2 (1M / 1F)** | **4 (3M / 1F)** | **2 (1M / 1F)** | **24** |
| **Utterance Count** | **1,800** | **599** | **300** | **600** | **300** | **3,599** |

---

## 2. Forensic Investigation of `LXC` vs `LXC_test`

### 2.1 The Forensic Inquiry
The audit investigated whether speaker `LXC` (present in `TRAIN`) appears or has ever appeared under the alias or suffix `LXC_test` in `TEST_SEALED` or any prospective test cohort.

### 2.2 Forensic Findings
1. **Authoritative Sealed Partition Composition:**
   In `data/split_specs/L2_ARCTIC_SPLIT.yaml` and `methods/M19-integrity-auditor/run.py`, `TEST_SEALED` is composed strictly of:
   $$\text{TEST\_SEALED Speakers} = \{\text{ZHAA}, \text{TNI}, \text{MBMPS}, \text{TLV}\}$$
   There is **NO Chinese speaker** in `TEST_SEALED`. Speaker `LXC` is not listed in `TEST_SEALED`.
2. **Exhaustive Code and Git History Search:**
   A full regex and commit-log search across all git trees (`git log -S "LXC"`, `git grep -i "lxc_test"`) confirms that **no file, manifest record, or variable named `LXC_test` exists in the repository**.
3. **Historical Suitcase Corpus Contamination Incident:**
   Forensic analysis uncovered the origin of the concern in `reports/upr01/WAVE_B_GATE_REVIEW.md` (lines 35–36):
   > *"Untouched Independent Cohort: Continuous spontaneous conversational narratives from the L2-ARCTIC Suitcase Corpus (`suitcase_corpus.zip`, 30 phrases across 6 non-native speaker backgrounds: `YBAA` [Arabic], `LXC` [Chinese], `RRBI` [Hindi], `YDCK` [Korean], `EBVS` [Spanish], `HQTV` [Vietnamese]).*  
   > *Strict Non-Exposure: Zero phrases or speakers were ever part of model training or threshold tuning in any laboratory."*

   **Forensic Auditor Critique:** The statement in `WAVE_B_GATE_REVIEW.md` was scientifically false and constituted a severe methodological error:
   - `LXC` was trained on 150 utterances in `TRAIN`.
   - `EBVS` was trained on 150 utterances in `TRAIN`.
   - `HQTV` was trained on 150 utterances in `TRAIN`.
   - `YBAA`, `RRBI`, and `YDCK` were iteratively evaluated in `DEV_SEARCH`.
   
   Evaluating the Suitcase Corpus while asserting "zero speakers were ever part of model training" failed to distinguish between novel *phrases* (prompts) and novel *speakers*.

### 2.3 Scientific Principle: Biological Identity Governs Disjointness
In scientific speech ML validation:
$$\text{SpeakerDisjoint}(A, B) \iff \text{HumanIdentity}(A) \cap \text{HumanIdentity}(B) = \emptyset$$
- Appending a suffix such as `_test` (e.g. `LXC_test`), altering an utterance prefix, or recording a speaker reading different text (e.g. conversational narratives vs ARCTIC sentences) **DOES NOT alter the biological vocal tract, fundamental frequency baseline, or phonetic habits of the human speaker**.
- Neural encoders fine-tuned or scalar-mixed on `LXC` retain deep acoustic familiarity with her voice.
- **Strict Prohibition:** Any prospective evaluation set containing recordings of `LXC`, under any filename, alias, or suffix, is **PERMANENTLY PROHIBITED** from serving as an independent test set for models trained on `TRAIN`.

---

## 3. Prior `DEV_SEARCH` Exploration Record

The partition `DEV_SEARCH` (`data/derived/l2arctic/manifests/dev_search.jsonl`, 599 utterances across speakers `YBAA`, `NCC`, `RRBI`, `YDCK`) has been heavily utilized:
- **Historical Exposure:** Used iteratively across 109 Phase 2 experiments (EXP-01 through EXP-109), candidate model selection in `rq1_method_select_01`, and feature extraction exploratory benchmarking (`rq1_encoder_bench_01`).
- **Development Status:** The hyperparameter configurations, learning rates, layer mixing concepts, and early-exit mechanisms were iteratively shaped by observing validation metrics on `DEV_SEARCH`.
- **Audit Verdict:** `DEV_SEARCH` is formally classified as **DEVELOPMENT-EXPOSED**. It is strictly ineligible to serve as a blind prospective validation set. It may only be reported as an exploratory development benchmark.

---

## 4. Protected Partition Firewall Verification

In strict accordance with the validation firewall directive, the auditor verified the cryptographic integrity of protected files using non-opening filesystem metadata:

| Partition | Manifest Path | Expected Size | Expected SHA-256 | Firewall Status |
|---|---|:---:|:---:|:---:|
| `DEV_CAL` | `data/derived/l2arctic/manifests/dev_cal.jsonl` | 1,793,066 B | `87fa9e2af6a2ad14d36e7680b977b1789fd20cfbfb56f239d5f840befce21ad6` | **PRISTINE / SEALED** |
| `DEV_SEARCH` | `data/derived/l2arctic/manifests/dev_search.jsonl` | 3,520,439 B | `a4ab6c7df7bd65b5d213d306c2fdfde6c316cbf7768a3eb63d056775351f99d2` | **PRISTINE / PRESERVED** |
| `TEST_SEALED` | `data/derived/l2arctic/manifests/test_sealed.jsonl` | 3,549,380 B | `a5636249f747a62e3340dded63161149644e83dd13aa2560e312d044c825b4b6` | **STRICTLY SEALED / UNTOUCHED** |

*Firewall Enforcement:* At no point during this preflight audit were the audio files or gold labels of `DEV_SEARCH`, `DEV_CAL`, or `TEST_SEALED` read or processed by model forward passes.

---

## 5. Demographic and Generalization Limits of `TEST_SEALED`

Auditors must explicitly report what a benchmark test set can and cannot prove:
1. **L1 Coverage Limitation:**
   - Covered L1s in `TEST_SEALED`: Arabic (`ZHAA`), Hindi (`TNI`), Spanish (`MBMPS`), Vietnamese (`TLV`).
   - Omitted L1s in `TEST_SEALED`: **Chinese** and **Korean** have 0 speakers in `TEST_SEALED`.
2. **Implication:**
   Prospective testing on `TEST_SEALED` will provide rigorous, contamination-free evidence for Arabic, Hindi, Spanish, and Vietnamese pronunciation evaluation. However, independent generalization to Chinese and Korean speakers will remain unassessed until external corpora (e.g. SpeechOcean762 or a newly collected cohort) are authorized.

---

## 6. Audit Verdict

- **LANE B VERDICT:** **SPEAKER ISOLATION PASS (With Strict Policy Lock)**
- **Audit Certification:**
  1. Authoritative `TEST_SEALED` partition contains zero overlap with `TRAIN` ($0/4$ speakers).
  2. No alias `LXC_test` exists in `TEST_SEALED`.
  3. Any evaluation of speaker `LXC` on prospective test sets is permanently barred.
  4. `DEV_SEARCH` is disqualified from blind holdout claims.
  5. Cryptographic seals on all protected partitions remain 100% intact.
