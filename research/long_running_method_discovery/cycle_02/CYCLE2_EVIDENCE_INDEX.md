# Cycle 2 evidence index

| Topic | Existing evidence reviewed | Cycle 2 interpretation |
|---|---|---|
| EVAL-V3 | `reports/EVAL_V3_CLOSURE_REPORT.md`, `reports/EVAL_V3_SCIENTIFIC_PROTOCOL.md`, `experiments/evaluation_v3/closure/release_final/` | Authoritative metric implementation; preserve full target and N+1 gap events. DEV_SEARCH is exposed. |
| TMCI attribution | `reports/TMCI_FINAL_SCIENTIFIC_REPORT.md`, component attribution and reproduction reports | EXP-24/34/28 motivate hypotheses. Historical SEARCH values do not establish independent validation or causal attribution. |
| Phase 3 Gold/KD | `reports/PHASE3_FINAL_SCIENTIFIC_REPORT.md`, `reports/PHASE3_GOLD_VS_KD_PAIRED.md`, `reports/PHASE3_TEACHER_OOF_PROVENANCE.md` | Gold-Only/KD-All broadly comparable across three paired seeds; no KD superiority established. |
| S/O failure | `reports/PHASE3_SO_FAILURE_DIAGNOSIS.md` | Low S/O warning recall and shared policy constraints remain; do not retrofit SEARCH thresholds. |
| Phase 4 | `reports/PHASE4_FINAL_SCIENTIFIC_REPORT.md`, `reports/PHASE4_NEXT_HANDOFF.md`, agents A-D reports | The detailed handoff says no Phase 4 neural training; existing released scores are development evidence. Preserve prior implementation. |
| Cycle 1 | `reports/long_running_method_discovery/NEXT_CYCLE.md`, `RESEARCH_STATE.md`, hypothesis/proposal/literature reports | H-001 had priority; a later EVAL-V3 waveform replay was found with frame outputs, enabling H-001 A/B scoring in this cycle. |
| Runtime model | Initial Pi/Codex task API rejected full alias `openai-codex/gpt-6-luna`; native collaboration subagent mechanism accepted `gpt-6-luna` after operator steered to a small model | All six workstreams dispatched with the explicitly requested small Luna model; no substitute. |
| CUDA | `nvidia-smi` failure; `coordination/GPU_RESOURCE_LEDGER.jsonl` latest entries are historical | No present CUDA capacity/lease. No neural Cycle 2 job executed. |
| Frame-score assets | H-003 inspected the full waveform replay, which contains frame outputs for all 2,699 authorized utterances; receipt reports parity for TRAIN/CAL/SEARCH | Cache field name `ctc_logits` is misleading: H-001 traced the model to CharSiU Wav2Vec2 frame classification trained on forced-aligned labels. It supports matched frozen-emission path scoring, but not a claimed CTC sequence likelihood without further model provenance. |

New metrics exist for H-001 and H-003; their event/AP provenance was independently checked and is listed in the experiment registry. Both SEARCH findings remain exploratory, not independent confirmation. H-005 numbers are from pre-existing EVAL-V3 closure artifacts, not a new calibration fit. No neural Cycle 2 experiment is verified.
