# LANGFENS — RQ1-TRN-03A / 01-Cross-Fold Research Plan & Test Isolation

**Status:** `COMPLETED`

**Recorded at:** 2026-10-29T14:35:00+07:00

**Evidence source:** Commits `7a3b9ecf5e9efedc480b477f531ebe9318ae3415`, `manifests/RQ1-TRN-03A/CROSS_FOLD_RESEARCH_PLAN.md`, `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`

**Evidence level:** `INDEPENDENTLY_VERIFIED`

**Branch:** `review/rq1-trn-03a-cross-fold-planning`

---

## 1. Objective
Formulate a mathematically rigorous, leak-free research plan for RQ1 4-fold cross-validation across Folds 2, 3, and 4. Audit speaker partitioning, resolve the perceived conflict between Fold 1 sealed test data and cross-fold staging, institute a three-tier test-isolation policy, plan independent Baseline R training without checkpoint reuse, and specify comprehensive cross-fold statistical aggregation.

---

## 2. Inputs & Dependencies
- Base commit: `7a3b9ecf5e9efedc480b477f531ebe9318ae3415` (Step 16 gate closure).
- Master split manifest: `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json` (SHA-256 `7d6bfdf6...`).
- Baseline R architecture: Two-head `LangfensWav2Vec2MDD` (frozen `facebook/wav2vec2-base-960h`, CTC head, cross-attention + binary head).
- Master policy: `AGENTS.md` (no local training, remote GPU boundaries, mandatory process checkpoints).

---

## 3. Work Performed
1. **Preserved Fold 1 R/C0/C1 Evidence:** Maintained historical findings; reaffirmed Reference R as authoritative baseline anchor; confirmed Treatment C1 whole-phone prompt dropout is `NOT_SUPPORTED` on Fold 1.
2. **Conducted Speaker Partition & Overlap Audit:**
   - Evaluated all 24 L2-ARCTIC speakers across 4 folds.
   - Identified the staging conflict where Fold 1 test speakers (`ABA`, `ASI`, `EBVS`, `HJK`, `LXC`, `PNV`) are required training speakers for Folds 2, 3, 4.
   - Clarified that test sealing applies per active fold ($S_{\text{test}}^{(k)}$), not globally across the entire corpus.
3. **Established Three-Tier Test-Isolation Policy:**
   - Tier 1: Cross-Fold Validation Phase. All 4 test sets ($S_{\text{test}}^{(1..4)}$) are 100% sealed and omitted from remote staging archives.
   - Tier 2: Model Improvement Phase. Evaluated strictly on pooled 4-fold validation sets (39,576 tokens).
   - Tier 3: Final Independent Testing. One-shot post-freeze evaluation on the 4 test sets using frozen thresholds $\tau_k^*$.
4. **Defined Independent Baseline R Training Protocol:**
   - Fresh initialization from pretrained `facebook/wav2vec2-base-960h` (rev `22aad52d...`).
   - Strict prohibition of warm-starting or fine-tuning from Fold 1 `best.pt`.
   - Budget: 6 epochs / 3,600 updates per fold (max 10 epochs early stopping).
5. **Specified Cross-Fold Evaluation Metrics & Aggregation:**
   - Per-fold: Protocol O AP, Mismatched AP, Acoustic Reliance Gap, Greedy CTC PER, Protocol Z AP, and per-speaker metrics.
   - Macro-averaging: Fold mean $\overline{\text{AP}}$, standard error, Student's $t$ 95% CI ($df=3$).
   - Micro-averaging: Pooled token-level PR curve across 39,576 tokens.
   - Cluster bootstrap: 1,000 speaker-stratified resamples for non-parametric CIs.
   - L1-accent disaggregation across 6 native language backgrounds.

---

## 4. Results & Verification
- Acceptance gate: `RQ1-TRN-03A-PLAN: PASS`.
- Split manifest verification: `python3 audit_scripts/validate_splits.py` passed with 0 errors.
- Unit and regression tests: 236 passed, 4 skipped with 0 failures.
- Zero test data staged or unsealed; physical fail-closed isolation confirmed.
- Documented in [CROSS_FOLD_RESEARCH_PLAN.md](../CROSS_FOLD_RESEARCH_PLAN.md) and [PROCESS_CHECKPOINT.md](../PROCESS_CHECKPOINT.md).

---

## 5. Output Artifacts
- Research monograph: `manifests/RQ1-TRN-03A/CROSS_FOLD_RESEARCH_PLAN.md`
- Phase overview: `manifests/RQ1-TRN-03A/PROCESS_CHECKPOINT.md`
- Step checkpoint: `manifests/RQ1-TRN-03A/checkpoints/01-cross-fold-research-plan-and-test-isolation.md`

---

## 6. Restart / Recovery Instructions
- Review `manifests/RQ1-TRN-03A/CROSS_FOLD_RESEARCH_PLAN.md`.
- Static planning milestone complete; no runtime state to recover.
- Training remains blocked pending explicit operator approval of Section 10 action packet.

---

## 7. Remaining Work & Safety Boundaries
- **Prohibited Actions:** No local training, no test set extraction, no GPU job provisioning without operator authorization.
- **Next Step:** Obtain operator approval on the 5-item action packet; implement `audit_scripts/build_fold_packages.py` and fold training configs.
