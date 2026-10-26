# RQ1-TRN-02B Baseline Freeze Ledger

**Gate:** design-only; no training or inference initiated. Source baseline: `fix/protocol-z-numerical-closure` at `c4a658ffbd5a2afd2ebdce172e47af43ed9bb63e`. Historical artifacts are references only and remain unmodified.

## Frozen Fold 1 contract

| Item | Frozen value / evidence |
|---|---|
| Split | `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`, SHA-256 `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b` |
| Train / validation / sealed test | 2,399 utterances / 300 (NJS, RRBI) / 900 sealed (ABA, ASI, EBVS, HJK, LXC, PNV); never stage/access test |
| Baseline config | `configs/rq1_trn_02a_fold1.yaml`; seed 1337; batch 2 × accumulation 2; max 25 epochs; patience 5; AdamW 1e-3, weight decay .01; cosine/100 warm-up; frozen Wav2Vec2 |
| Backbone | `facebook/wav2vec2-base-960h`, revision `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`; eval/frozen |
| Two-head objective | CTC + Binary BCE, weights 1.0 each; no diagnosis training |
| Production checkpoint | `checkpoints/RQ1-TRN-02A/recovered/best.pt`, SHA-256 `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`; epoch index 5 (display epoch 6), 3,600 successful updates, best validation AP .316062; original training SHA `caa56783113d40eac438e5f458cdb313b8932218` |
| Architecture source | `src/langfens_prep/model.py`; CTC linear head, canonical-query→audio cross-attention, binary MLP over concatenated aligned acoustic vector + phone embedding; two-head trainable parameters are CTC head, cross-attention module, binary head |

## Evaluator and controls inventory

| Protocol / control | Frozen evidence and result |
|---|---|
| Protocol O (oracle canonical alignment) | Existing historical evaluator: `audit_scripts/evaluate_fold1_ctc_binary.py`, SHA-256 `c29d45cf6667200ab7a24a266d9e28d4bd40d15fbd2392ecc3c847e877e32280`. Historical validation Binary AP .317021; CTC PER 23.64% (also reported as 23.68% in a separate replay—preserve provenance, do not silently reconcile). No gold timestamps as inference inputs outside explicitly labeled oracle Protocol O. |
| Protocol Z evaluator | `audit_scripts/evaluate_protocol_z.py`, SHA-256 `ab13414a852be4cd67c58721b6b1d0eaba10e0540de4cb4f85f1b066e99f76f5`; prediction record/report identify evaluator execution SHA `03854dca2dce81fbfcf1e2715eaf399af477dd62` and checkpoint SHA above. Uses CMUdict G2P input, strict validation-only. |
| Protocol Z numerical closure | `.../zero_oracle_verification/NUMERICAL_CLOSURE_REPORT.md`; aligned-only AP .313530, 9,919 matched phone positions, coverage 99.63%; threshold .20 confusion TP 605 / FP 1,510 / FN 540 / TN 7,264; threshold .50 TP 208 / FP 213 / FN 937 / TN 8,561. Protocol O historical artifacts unchanged. |
| Canonical-only controls | `manifests/RQ1-TRN-02A/ACOUSTIC_BASELINE_COMPARISON.md`; train-only phone prior AP .303063; validation prevalence .116211. Use the matched protocol/population for any inferential comparison, as there are discrepancies between reports/protocols. |
| Audio swaps | `.../ACOUSTIC_CONTRIBUTION_CONTROLS.md` and `.../zero_oracle_verification/ACOUSTIC_CONTROL_ERRATUM.md`; matched Protocol O AP .317021 vs within-speaker shift-1 mismatch AP .294722; canonical prior .303063. At .50 overall agreement 96.76% hides positive-class agreement 86.60%; matched audio produces 201 vs 172 TP and 208 vs 250 FP in corrected class-specific control. |

## Immutable baseline rule

Do not edit, replace, or re-label historical prediction records, checkpoint, split, evaluator outputs, or reports. Any new model/checkpoints/metrics belong under a new RQ1-TRN-02B output location, carry source/config/data/checkpoint hashes, and identify fresh evaluation executions separately. Fresh local SHA-256 verification during this design task matched both the production checkpoint (`a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`) and split manifest (`7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`). This verifies file identity only; no weights were loaded or altered.

## Additive clarification for DESIGN-R1

Under DESIGN-R1 (`docs/rq1-trn-02b-acoustic-conditioning-design`), the immutable production checkpoint above (`checkpoints/RQ1-TRN-02A/recovered/best.pt`, SHA-256 `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`) serves two distinct roles:
1. **Reference R**: The frozen historical research anchor (no optimizer updates, no model modifications, retaining historical Protocol O and Protocol Z evaluation results).
2. **Initial Model Weights**: The common weight initialization for both **Control C0** (`p=0.00` fine-tuning) and **Treatment C1** (`p=0.25` prompt dropout fine-tuning).

All frozen split definitions, checkpoint weights, SHA-256 hashes, evaluator contracts, and historical baseline metrics remain completely unmodified.
