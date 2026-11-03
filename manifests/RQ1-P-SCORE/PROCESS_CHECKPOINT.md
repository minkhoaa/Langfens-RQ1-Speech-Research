# P-SCORE Execution Process Checkpoint

State: **EXECUTION_COMPLETED; BENCHMARK_REPORTED.**
Continuation of RQ-RESEARCH-01 and P-SCORE implementation readiness. All authorized tasks completed without local GPU or training.
Training NOT_STARTED; 0 successful optimizer updates, 0 skipped updates/no attempts; no new model artifact.

## 1. Identity and Provenance

- **Workspace**: `/home/khoa/Projects/KLTN/langfens-rq1-p-score`, branch `prep/rq1-p-score`.
- **Common Base**: `0d268ff8c444e16fd76aca2f4ad6ece966908edc`.
- **Binding Contract**: `manifests/RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md`, SHA-256 `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`.
- **Split Manifest**: `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`, SHA-256 `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`.
- **Frozen E0 Checkpoint**: `/home/khoa/Projects/KLTN/Dataset/checkpoints/RQ1-TRN-02A/recovered/best.pt`, SHA-256 `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`.
- **Authorized Dataset**: `/home/khoa/Projects/KLTN/Dataset/datasets/L2-ARCTIC/authorized_RQ1-TRN-02A_fold1_train_val_only.zip` (strictly containing train and validation speakers only).
- **Sealed Test Isolation**: Speakers `ABA`, `ASI`, `EBVS`, `HJK`, `LXC`, `PNV` were strictly sealed, never extracted, staged, or inspected.

## 2. Blockers Closed and Architecture Disposition

1. **Author Topology & License Disposition**: Clean-room independent implementation of Cao et al. (arXiv:2507.16838v3) Eq. 4/12 and S/SD output languages. Author code was not copied or borrowed, eliminating all third-party licensing encumbrances. The graph revision is formally registered as `ctc-union-language-v1` with all 39 non-blank classes as alternatives (no next-neighbor wildcard exclusion). Blank is strictly the CTC path mechanics symbol. Epsilon target-deletion in the SD graph reconstructs the standard CTC state sequence, preserving the requirement for an intervening blank between identical adjacent phones.
2. **Eq. 13 Occupancy Difference**: The published paper's Eq. 13 uses an approximate normalized-forward heuristic floored at 1.0. Our implementation computes exact conditional posterior target-state occupancy through a forward reward semiring (`log_occupancy_mass - log_mass`). Primary scoring is designated strictly as the raw error-oriented score `s_error = log P(relaxed) - log P(canonical)` (`raw_gop = -s_error`). Occupancy normalization is retained as an auxiliary analysis (`posterior_occupancy_normalized_error`), avoiding heuristic distortion of primary likelihood ratios.
3. **Proposal Disposition**: The missing UIT proposal attachment limitation is accepted and formally bounded.

## 3. Execution Summary

1. **Synthetic Verification**: 269 synthetic unit/regression tests passed (`pytest tests/test_p_score_ctc.py tests/test_p_score_runtime.py tests/test_p_score_shared_schema.py`).
2. **Emission Cache Generation**: Generated FP32 log-emissions on CPU using frozen E0 checkpoint (`facebook/wav2vec2-base-960h` backbone + `ctc_head`, `torch.no_grad`, `weights_only=False`) on the 300 Fold-1 VAL utterances (NJS: 150, RRBI: 150).
   - Cache file: `/home/khoa/Projects/KLTN/Dataset/cache/rq1_p_score_fold1_val_emissions.json`
   - Cache SHA-256: `a699fd9378c86cd348a0628a144ecacfef491ea8f90e0edd23b4e46932e3e829` (39,645,570 bytes)
   - Authorization file: `manifests/RQ1-P-SCORE/authorization/p_score_authorization_fold1_val.json`
   - Authorization SHA-256: `5432c000111a8244949c241e3298c422b15e5062d5a6835e2db64383f8758b15`
   - Verified with `validate_cache` and `authorize_cache` before execution.
3. **Inference Execution**: Executed `audit_scripts/p_score_infer.py` with `--mode frozen-cache` across 11 CPU worker processes:
   - Evaluated all 4 scorers: `sf_s`, `sf_sd`, `greedy_dp`, `gop_sa`.
   - Total runtime: 637.5 seconds (~10.6 minutes).
   - Total prompt tokens scored: 39,824 (9,956 positions × 4 scorers).
   - Strict runtime schema validation (`strict_runtime_validator.py`): 39,824 / 39,824 rows valid (0 schema errors).
   - Run output directory: `/home/khoa/Projects/KLTN/Dataset/p_score_runs/rq1_p_score_e0_fold1_val/`.
4. **Offline Join & Benchmark**: Aligned 9,956 G2P prompt tokens with 9,956 Gold TextGrid canonical tokens across 300 utterances via deterministic Levenshtein alignment (tie-break `sub_first`):
   - Aligned tokens: 9,919 (99.628% coverage).
   - Unaligned gold tokens: 37 (omitted from G2P sequence).
   - Unaligned G2P tokens: 37 (extra prompt tokens).

## 4. Benchmark Results

### Primary Comparison Across Scorers (Aligned Population N=9,919)

| Scorer | Valid / Total Records | Coverage | Omission AP | Substitution AP | Binary AP | Recall @ FPR15 | Precision @ FPR15 | Achieved FPR15 | Greedy CTC PER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **GOP-SF-S** (`sf_s`) | 9,919 / 9,919 | 100.0% | 0.029411 | **0.246800** | **0.259094** | 0.248175 | 0.025185 | 0.149989 | 0.236818 |
| **GOP-SF-SD** (`sf_sd`) | 9,919 / 9,919 | 100.0% | **0.051026** | 0.222544 | 0.250049 | **0.430657** | **0.042909** | 0.149989 | 0.236818 |
| **GOP-SA** (`gop_sa`) | 9,919 / 9,919 | 100.0% | 0.049846 | 0.228104 | 0.254370 | 0.416058 | 0.041515 | 0.149989 | 0.236818 |
| **Greedy DP** (`greedy_dp`) | 4,212 / 9,919 | 42.38% | 0.019002 | 0.175058 | 0.186678 | 0.380952 | 0.031873 | 0.129669 | 0.236818 |

### Primary Contrast: GOP-CTC-SF-SD minus GOP-CTC-SF-S

- **Delta Omission AP**: **+0.021615** (+2.16 AP points, relative gain +73.5%)
- **Delta Substitution AP**: **-0.024256** (-2.43 AP points, relative drop -9.8%)
- **Delta Binary AP**: **-0.009045** (-0.90 AP points, modest binary shift)
- **Delta Recall @ FPR15**: **+0.182482** (+18.25 percentage points, from 24.82% to 43.07%)

### Per-Speaker Breakdown (NJS vs RRBI)

| Scorer | Speaker | Tokens | Deletions | Substitutions | Correct | Omission AP | Substitution AP | Binary AP | Recall @ FPR15 | Achieved FPR15 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sf_s` | NJS | 4,998 | 78 | 560 | 4,360 | 0.043556 | 0.278724 | 0.294825 | 0.294872 | 0.150000 |
| `sf_sd` | NJS | 4,998 | 78 | 560 | 4,360 | **0.065212** | 0.251297 | 0.282350 | **0.474359** | 0.150000 |
| `sf_s` | RRBI | 4,921 | 59 | 448 | 4,414 | 0.020773 | 0.222045 | 0.229324 | 0.186441 | 0.149977 |
| `sf_sd` | RRBI | 4,921 | 59 | 448 | 4,414 | **0.041845** | 0.200570 | 0.223330 | **0.372881** | 0.149977 |

Both validation speakers demonstrate consistent Omission AP improvements (+0.0217 for NJS, +0.0211 for RRBI) and ~+18% Recall@FPR15 gains (+17.95 pp for NJS, +18.64 pp for RRBI).

### Breakdown by Word Position (Word-Final Consonants)

Deletions in non-native speech heavily concentrate in word-final consonants (93 of the 137 aligned deletions, or 67.9% of all omissions):

| Scorer | Stratum | Total Tokens | Deletions | Errors | Omission AP | Binary AP |
|---|---|---:|---:|---:|---:|---:|
| `sf_s` | Word-Final Consonants | 1,866 | 93 | 358 | 0.132071 | 0.415449 |
| `sf_sd` | Word-Final Consonants | 1,866 | 93 | 358 | **0.181265** | 0.393035 |
| `sf_s` | Other Positions | 8,053 | 44 | 787 | 0.013500 | 0.234791 |
| `sf_sd` | Other Positions | 8,053 | 44 | 787 | **0.029799** | 0.221892 |

In word-final consonants, admitting the target deletion alternative (`sf_sd`) increases Omission AP by **+0.049194** (+4.92 AP points, from 0.1321 to 0.1813).

### Breakdown by Confusable Phone Groups

| Confusable Group | Deletions | `sf_s` Omiss AP | `sf_sd` Omiss AP | Delta Omiss AP | `sf_s` Bin AP | `sf_sd` Bin AP |
|---|---:|---:|---:|---:|---:|---:|
| Dental Stops / Fricatives (`TH`, `DH`, `T`, `D`, `S`, `Z`) | 86 | 0.097995 | **0.130771** | **+0.032776** | 0.477450 | 0.438935 |
| Liquids / Glides (`L`, `R`, `W`, `Y`) | 25 | 0.072332 | **0.108112** | **+0.035780** | 0.132209 | 0.139764 |
| Labiodental / Bilabial (`B`, `V`, `P`, `F`) | 9 | 0.042483 | **0.056293** | **+0.013810** | 0.358589 | 0.350931 |
| Tense / Lax Vowels (`IY`, `IH`, `UW`, `UH`, `AE`, `EH`) | 4 | 0.026854 | 0.022146 | -0.004708 | 0.155216 | 0.144492 |
| Other Phones | 13 | 0.016140 | **0.064129** | **+0.047989** | 0.229958 | 0.219763 |

### Greedy CTC Phone Error Rate Fidelity Guardrail

Greedy CTC recognition was evaluated against 9,881 chronological realized reference phones:
- Total Reference Tokens: 9,881
- Substitutions: 1,707 (17.28%)
- Deletions: 431 (4.36%)
- Insertions: 202 (2.04%)
- Total Edits: 2,340
- **Greedy CTC PER**: **0.236818** (23.68%), exactly replicating the historical production Baseline R E0 benchmark.

## 5. Artifact Ledger

The following artifacts have been produced and verified in `manifests/RQ1-P-SCORE/evaluation/`:
- `BENCHMARK_REPORT.json`: Comprehensive report containing all metrics, operating points, and breakdowns.
- `metrics_summary.csv`: Summary of primary, secondary, and auxiliary metrics across all 4 scorers.
- `metrics_by_speaker.csv`: Detailed metric breakdown by speaker (NJS vs RRBI).
- `breakdown_word_final_consonants.csv`: Evaluation stratified by word-final consonants.
- `breakdown_confusable_phones.csv`: Evaluation stratified by confusable phonological classes.
- `offline_token_evaluation_sf_s.jsonl.gz`: Full joined offline token records for `sf_s`.
- `offline_token_evaluation_sf_sd.jsonl.gz`: Full joined offline token records for `sf_sd`.
- `offline_token_evaluation_greedy_dp.jsonl.gz`: Full joined offline token records for `greedy_dp`.
- `offline_token_evaluation_gop_sa.jsonl.gz`: Full joined offline token records for `gop_sa`.
- `run_manifest.json`: Complete execution run manifest with cryptographic source and config hashes.
- `SHA256SUMS`: Verification digests for all output records.
- External raw run bundle: `/home/khoa/Projects/KLTN/Dataset/p_score_runs/rq1_p_score_e0_fold1_val/`.
- External emission cache: `/home/khoa/Projects/KLTN/Dataset/cache/rq1_p_score_fold1_val_emissions.json` (SHA-256 `a699fd9378c86cd348a0628a144ecacfef491ea8f90e0edd23b4e46932e3e829`).
- Authorization record: `manifests/RQ1-P-SCORE/authorization/p_score_authorization_fold1_val.json` (SHA-256 `5432c000111a8244949c241e3298c422b15e5062d5a6835e2db64383f8758b15`).

## 6. Restart & Audit Commands

```bash
cd /home/khoa/Projects/KLTN/langfens-rq1-p-score
git status
git log -1 --format='%H %s'
PYTHONPATH=.:src /home/khoa/Projects/KLTN/Dataset/.venv/bin/python -m pytest -q tests/test_p_score_ctc.py tests/test_p_score_runtime.py tests/test_p_score_shared_schema.py
sha256sum manifests/RQ1-P-SCORE/authorization/p_score_authorization_fold1_val.json
sha256sum /home/khoa/Projects/KLTN/Dataset/cache/rq1_p_score_fold1_val_emissions.json
```
