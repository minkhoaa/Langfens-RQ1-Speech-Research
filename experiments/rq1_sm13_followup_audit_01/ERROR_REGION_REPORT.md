# Where the frozen SM13 score misses vowel and stop substitutions

**Population:** 12-speaker TRAIN OOF, seed 101.  
**Operating points:** For each manner separately, the retrospective threshold with maximum true positives subject to joint FWR ≤0.20, with all other frozen alert streams unchanged. These are diagnostic upper bounds on the current score, not deployable thresholds. Vowel and stop rows cannot simultaneously use the same warning allowance.

| Target | Gold S | TP | FP | Missed S | Largest missed normalized pair |
|---|---:|---:|---:|---:|---|
| Vowel /IH/ | 467 | 8 | 29 | 459 | IH→IY: 367 missed |
| Vowel /AH/ | 463 | 18 | 19 | 445 | AH→AO: 148 missed |
| Vowel /OW/ | 263 | 26 | 44 | 237 | OW→AO: 210 missed |
| Vowel /AE/ | 223 | 9 | 11 | 214 | AE→AA: 122 missed |
| Stop /D/ | 263 | 9 | 24 | 254 | D→T: 104 missed |
| Stop /T/ | 269 | 18 | 40 | 251 | T→T: 136 missed; raw labels often T* |
| Stop /P/ | 177 | 44 | 49 | 133 | P→B: 95 missed |

`IH`, `AH`, and `OW` account for **1,193/2,280 (52.3%)** vowel substitutions. `D` and `T` account for **532/862 (61.7%)** stop substitutions. A useful next model should state beforehand whether it targets these high-support contrasts and should report their TP/FP separately. The `T→T` normalized pair often retains the `*` deviation marker in its raw TextGrid label, so it should not be treated as an annotation error without listening.

Exact phone and speaker rows are in `ERROR_REGION_BY_PHONE.csv` and `ERROR_REGION_BY_SPEAKER.csv`. Reproduce with:

```bash
.venv/bin/python experiments/rq1_sm13_followup_audit_01/analyze_error_regions.py
```
