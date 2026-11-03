# Shared V1 contract audit — implementation addendum

V1 remains byte-immutable: `../EVALUATION_CONTRACT_V1.md`, SHA256 `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`. This audit resolves transport/authorization interpretation, not historical evaluator semantics or scientific performance.

## Populations, labels and inference boundary

| Contract | Audit result / mandatory behavior |
|---|---|
| Fold 1 envelope | TRAIN 2,399; VAL 300 (NJS/RRBI); TEST 900 **SEALED**. Check manifest hash and ordered TRAIN/VAL IDs only; no TEST ID export or contents read. Speaker metadata may denylist active TEST. |
| Reduced development | FIT 1,799; SELECT 300 MBMPS/TLV; CAL 300 NCC/THV; VAL 300 NJS/RRBI. Same split/routing/budget/initial tensors and selection opportunities for future P-SUP treatment/control. E0 trained on all TRAIN, including SELECT/CAL: P-SCORE CAL is not independent of E0 acoustic training. |
| Strict error semantics | `s` remains substitution/error even starred same-base phones normalize equal; never KEEP-by-identity. Unknown `err` retains known operation/binary error but masks realization/diagnosis. Deletion has canonical slot/error, no realized phone, no fictitious silence CTC target. |
| CTC/diagnosis maps | Blank/pad 0, CTC phones 1..39; diagnosis phones 0..38, ignore -100. Realized chronology includes known additions/correct/substitutions; excludes deletion/silence/unknown realization. Blank is not learner deletion. |
| Insertions | Addition has no canonical slot. Separate gap/sequence outputs; no insertion class smuggled into canonical-token AP. Unknown/invalid canonical identity fails explicitly rather than silently shortening positional tensors. |
| Variants/repeats | Pinned primary dictionary pronunciation and OOV handling; alternative variants are named sensitivity policies. Prompt mismatch/DP deletion is not a learner omission. Repeated phones require positional keys, CTC blank separation and tie/ambiguity accounting. |
| O versus Z | O archive uses annotation-derived canonical identities (oracle) even without timestamps. New Z runtime: waveform + written prompt only, prompt-derived phones/word indices; **no** gold fields, perceived sequences, TextGrid identities/timing, operation labels or hidden annotation readers. |
| Offline join | Freeze runtime file/hash first; gold-canonical versus G2P `sub_first` DP occurs only in a separate offline evaluator, preserving every gold position/unmapped row. No gold corpus join is authorized now. |

Actual source reinspection: `phoneme_normalizer.py`, `textgrid_parser.py` §§substitution/deletion/addition, `dataset.py:97–153`, complete `collate.py`, `model.py:340–456`, `alignment.py:73–223`, `evaluate_e1_mdd.py:108–267` and archived E0/E1 JSON. No source imported/executed. Static inherited blinded-collator `padded_aux_targets` defect remains documented, **not repaired**. Historical schema examples claiming deletion SIL/blank/UNK conflict with executed filtering/masks; preserved as history, not reused as source truth.

## Metric contract (future offline evaluation only)

Archival counts are documentary, not fresh corpus access/reproduction:

| Endpoint | Positives / negatives | N | Score identity |
|---|---|---:|---|
| Binary AP | 1,012 substitutions +145 deletions /8,799 correct | 9,956 | `s_error` |
| Legacy omission AP | 145 deletion /8,799 correct; substitutions excluded | 8,944 | `s_error` |
| Legacy substitution AP | 1,012 substitution /8,799 correct; deletions excluded | 9,811 | `s_error` |
| Dedicated subtype AP | Same respective populations | as above | `s_omit`/`s_sub`, separately named |

AP ranks full-precision continuous scores, descending grouped ties; sum recall increments times precision at distinct scores, not interpolated AP or thresholded predictions. Constant-score reference equals the actual subset prevalence; random finite-sample expected AP need not exactly equal prevalence. Pooled AP differs from mean per-speaker AP. No candidate-specific favorable dropping or deletion-vs-all masquerading as legacy omission AP.

For each score/threshold: TP/FP/TN/FN, recall=TP/(TP+FN), precision=TP/(TP+FP), FPR=FP/(FP+TN), FNR and F1, denominator/tie policy and achieved FPR at ceilings .05/.10/.15. FPR is not false-discovery fraction. Undefined values null+reason/count. Omission-subset precision differs from complete deployed-alert precision, including substitution confusions. Select threshold on CAL correct scores, `score>=tau`, lowest available threshold meeting empirical ceiling including reject-all; retrospective VAL curves explicitly diagnostic, not deployment guarantees.

Greedy realized-phone PER: collapse consecutive repeats **then** remove blank; `(S+D+I)/N_realized`; preserve integer edits/reference counts. Historical N_realized=9,881, not canonical 9,956. Recognition deletions are not human omissions. Auxiliary alternate decoding does not replace this endpoint.

## Coverage, uncertainty and claims

Use the shared schema and full utterance/position failure accounting; finite JSON only. Unmapped/invalid score policy is **abstain**, not VAL prevalence imputation, arbitrary max/.90 fusion or renamed complete AP. Report aligned-only AP as conditional. No full-population AP superiority with incomplete scores. At operating points abstained true errors count undetected; disclose answered/all-gold denominators, false reassurance and best/worst unresolved-correct FPR bounds, risk versus coverage and per-speaker/label failures.

No learned fusion/calibration, validation-trained priors or local fitting is authorized. A TRAIN-derived fallback is a separately frozen extension, never retroactive V1 repair. Raw GOP/log-ratio is not probability. Three-way `s_error=1-p_correct`; hierarchical omission score requires joint `p_error*p_delete_given_error`, not conditional score alone. Binary-only capacity control cannot identify meaningful subtypes.

G=2 reused Fold-1 VAL speakers cannot establish population significance, Vietnamese-group claims or deployment safety. More tokens/seeds are not more speakers; any permitted speaker bootstrap is exploratory. Acoustic mismatch/prior controls, natural reductions/short realized phones/repeats/variants, feedback expert review and measured intended-workload resources remain unexecuted requirements. Never use projection-bypassing forced attention or gold intervals as runtime evidence.

## Immutable scientific limitations

DIAG-02B remains **PARTIAL**, 9/80 archive comparisons exceed unchanged .000200 tolerance; no causal training GO. E1 rollout remains **NO-GO**. No source-only or synthetic JSON check repairs drift, authenticates actual weights or reconstructs missing full-precision E1 records. Future experimental interpretation must resolve historical parity or prospectively name a new evaluator stratum with its own contemporaneous control; no tolerance relaxation/baseline substitution/history patch.
