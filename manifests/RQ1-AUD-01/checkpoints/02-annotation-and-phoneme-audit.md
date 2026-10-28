# LANGFENS — RQ1-AUD-01 / 02-Annotation and Phoneme Audit

**Status:** COMPLETED

**Recorded at:** 2026-10-22T21:15:00+07:00

**Evidence source:** Commit `5ccb4dc2cc986036175161c660b73a206de188d2`, `manifests/RQ1-AUD-01/L2_ARCTIC_ANNOTATION_AUDIT.md`

**Evidence level:** INDEPENDENTLY_VERIFIED

## 1. Objective
Parse all 3,599 Praat TextGrids in L2-ARCTIC and 5,000 scored utterances in SpeechOcean762, extract error distributions across speaker cohorts, and evaluate phonetic inventory compatibility across target corpora.

## 2. Inputs & Dependencies
- Verified raw archives: `datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip`, `datasets/SpeechOcean762/original/speechocean762.tar.gz`.
- Linguistic resources: CMUDict v0.7b, MFA ARPA dictionary, KoelLabs phone inventory.
- Parsing script: `audit_scripts/textgrid_parser.py`, `audit_scripts/audit_l2arctic_annotations.py`.

## 3. Work Performed
1. Implemented in-memory streaming parser for Praat ooTextFile IntervalTiers and PointTiers.
2. Parsed 3,599 gold TextGrids across 24 speakers, profiling phone boundaries and alignment tiers.
3. Audited 5,000 SpeechOcean762 JSON annotations, analyzing 5-expert score distributions across 94,445 phones.
4. Compared phoneme sets against 39-ARPAbet standard, documenting stress markings, AX schwa, and typo tags (`err`, `*`).
5. Exported structured error inventories and speaker profiles to CSV.

## 4. Results & Verification
- Acceptance gate: `ANNOTATION_AUDIT: PASS`.
- Profiled 101,633 correct phone tokens and 17,518 mispronunciations in L2-ARCTIC.
- Isolated high deletion rates in Vietnamese speakers (HQTV, PNV, THV, TLV: 1,566 deletions).
- SpeechOcean762: 80.68% score 2.0 (correct), 2.42% score 0.0 (severe mispronunciation).
- Documented in [L2_ARCTIC_ANNOTATION_AUDIT.md](../L2_ARCTIC_ANNOTATION_AUDIT.md) and [PHONEME_COMPATIBILITY_AUDIT.md](../PHONEME_COMPATIBILITY_AUDIT.md).

## 5. Output Artifacts
- Statistics CSVs: `manifests/RQ1-AUD-01/L2_ARCTIC_SPEAKER_STATISTICS.csv`, `L2_ARCTIC_ERROR_STATISTICS.csv`, `L2_ARCTIC_LABEL_ISSUES.csv`.
- SpeechOcean762 CSVs: `manifests/RQ1-AUD-01/SPEECHOCEAN762_SCORE_STATISTICS.csv`, `SPEECHOCEAN762_LABEL_ISSUES.csv`.
- Summary JSON: `audit_scripts/l2arctic_audit_summary.json`, `audit_scripts/speechocean_audit_summary.json`.

## 6. Restart / Recovery Instructions
- Environment: Python standard library (`zipfile`, `tarfile`, `json`, `csv`, `re`).
- Audit execution command:
  ```bash
  python audit_scripts/audit_l2arctic_annotations.py
  ```
- Safe read-only execution; no authorization required.

## 7. Remaining Work & Safety Boundaries
- Unresolved issues: Initial denominator ambiguity between reference phones and realized phones identified.
- Prohibited actions: Do not alter raw TextGrid files or apply ad-hoc phone mappings.
- Next step: [03-scientific-consistency-reconciliation.md](03-scientific-consistency-reconciliation.md).
