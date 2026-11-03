# P-SUP recovered-checkpoint benchmark — protocol V1

## Scope and immutable inputs

Two completed experiment arms only: T-OP (`operation_ce`) and B-Control (`binary_collapsed`), each at `fixed_budget_3600.pt` and SELECT-selected `best.pt`. No retraining, architecture/weight changes, new optimization, literature review, or sealed TEST access. A selected checkpoint is not assumed to contain3600updates. Report its actual recovered update count separately from the run's final count.

Frozen evaluation contract: `manifests/RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md`, SHA-256 `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`. Split SHA-256 `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`. Authorized Fold-1 TRAIN/VAL archive SHA-256 `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`. VAL must be exactly300utterances, NJS/RRBI150each, ordered-ID SHA-256 `6d34642b15bbf80cbd5bd102b5574ab36f5adac74be6d6d0ae7218b002eff065`. Validate actual inputs rather than trusting these expected constants.

Backbone identity is `facebook/wav2vec2-base-960h@22aad52d435eb6dbaf354bdad9b0da84ce7d6156`, frozen eval. All four endpoints must use identical hardware/precision/preprocessing/batching within each comparison; record actual source hashes, dirty status and dependency identities. CPU FP32 inference, if used, is a distinct numerical stratum from historical CUDA FP16 training/validation. No local training fallback is allowed.

## Separate protocol namespaces

- **O-ARCHIVE / O oracle-canonical:** annotation-derived canonical phone queries are explicitly oracle information, even without gold times. Full gold population expected9956 =8799correct +1012substitution +145deletion.
- **Z-PROMPT-V1:** waveform and frozen written-prompt/CMUDict first pronunciation only, strict OOV policy and pinned normalizer. No TextGrid-derived canonical identities, labels, timing or word boundaries in runtime. Save/freeze runtime predictions and hashes before offline gold access/join. Use pinned `alignment.py` deterministic `sub_first` DP. Expected9919aligned positions (8774correct,1008substitution,137deletion) and37unmapped gold (25correct,4substitution,8deletion); verify rather than force these counts.

Original P-SUP runtime files falsely say Z although evaluator's dataset builds queries from TextGrid canonical labels. Preserve the original bytes and report actual protocol O separately. A schema validator accepting the mislabeled rows is not evidence of a gold-free runtime. The archival audit and Luna report document this correction; no silent historical overwrite.

Every allowed gold position must be accounted for offline. Unmapped/invalid scores abstain; no gold-prevalence imputation or fabricated omission probability. Aligned AP is conditional coverage. Include unanswered true errors in all-gold operating-point miss counts and unresolved-correct FPR bounds. Never rank9956-token O and9919-token Z AP directly.

## Score semantics and detection measurements

Higher scores mean more suspected error. Binary: `s_error` on substitution+deletion versus correct. Legacy omission: `s_error` on deletion versus correct only (exclude substitutions). Legacy substitution: `s_error` on substitution versus correct only (exclude deletions). Dedicated scores: `s_omit=p_deletion`, `s_sub=p_substitution`, only semantically supported for T-OP. B-Control's subtype logits are unidentified: dedicated AP/PR endpoints must be null with unsupported-semantics reason, never zero or a fabricated probability.

Full O denominators: binary9956, omission8944, substitution9811; Z aligned denominators: binary9919, omission8911, substitution9782. AP is noninterpolated recall-increment times precision with equal-score groups included together. Preserve full machine precision in JSON; round only display. Report label/prevalence counts, scores, endpoint, protocol, condition, cohort/hash and coverage for every metric.

For registered scores report precision, recall, F1, TP/FP/TN/FN, achieved correct-token FPR, score>=threshold tie policy and threshold origin at ceilings5%,10%,15%. Primary operating thresholds require allowed CAL correct scores (NCC/THV); SELECT is MBMPS/TLV, not VAL. VAL-derived ceilings are explicitly retrospective diagnostic curves, never held-out calibrated guarantees. Undefined metrics are null with reasons/counts. Omission subset precision is not deployed omission-feedback precision across all token types; distinguish them.

## Recognition, controls and strata

Greedy CTC collapse repeats then remove blank; score against chronological valid realized phones, including additions and excluding human deletions/silence/unknown realized phones. Save decoded utterance predictions, integer S/D/I/reference counts; expected historical realized reference9881must be checked. CTC recognition deletions are NOT human omission events.

Required waveform condition is matched versus within-speaker cyclic shift-1 audio in manifest order; save mapping/hash and prove no fixed points. Same query/prompt attached to different same-speaker waveform. No waveform masking, model updates or causal-diagnosis claim. Canonical-only prior must reuse a verified frozen permitted-training-only prior when available; do not fit a new model locally or construct it from VAL labels. If its artifact/procedure is inaccessible, report unavailable rather than silently substituting a new prior.

Per-speaker metrics include both NJS/RRBI. Word-final consonants are an explicitly offline gold descriptive stratum following the existing midpoint-to-word policy; no gold timings/word boundaries in runtime. Report sparse counts/coverage. With only two VAL speakers, do not claim population-level significance or diagnostic readiness.

## Evidence classification, storage and reproduction

Every metric is `reproduced`, `archival-only`, `reported-only`, or `unavailable`. Newly recalculating AP from historical fullprecision predictions reproduces arithmetic but leaves inference provenance **archival-only**. Historical token-only files do not independently reproduce PER without decoded utterance hypotheses.

Private durable store is the existing verified-private HF repository `minkhoaa/langfens-rq1-cross-fold-checkpoints`; use new `benchmarks/psup-v1/` prefixes, immutable revisions and per-file SHA-256 readback. No raw predictions, gold joins, audio, weights, credentials or prompt text newly enter Git. Git stores only lightweight reports/receipts/hashes and code. Original P-SUP and baseline raw mirrors already have receipts in this directory; fresh output durability requires its own receipt.

Reproduction commands and actual inference precision/source/run results belong in `REPRODUCTION_REPORT.md` after execution. Independent archival audit commands currently executed:

```sh
.venv/bin/python audit_scripts/verify_psup_archival_v1.py \
  --archive-root manifests/RQ1-P-SUP/evaluation \
  --output /tmp/psup-archival-recomputation-new.json
.venv/bin/python audit_scripts/verify_psup_baselines_v1.py \
  --dataset-root "$PWD" \
  --p-score-evaluation ../langfens-rq1-p-score/manifests/RQ1-P-SCORE/evaluation \
  --output /tmp/psup-baselines-recomputation-new.json
.venv/bin/python -m pytest -q \
  tests/test_verify_psup_archival_v1.py tests/test_persist_psup_benchmark_v1.py
```

Output paths must be new; helpers intentionally refuse overwriting old receipts. Remote GPU inference may be used only with verified allocation/privacy; Kaggle CLI may stage `--no-run`, but starting requires operator Web UI HF_TOKEN attachment and Save & Run All. Do not bypass this workflow or provision Colab with exhausted compute units.
