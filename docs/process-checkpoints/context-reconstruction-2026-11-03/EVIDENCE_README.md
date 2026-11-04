# Evidence bundle — read this before raw agent reports

## Interpretation order

1. [PROJECT_CONTEXT_RECONSTRUCTION.md](../../../PROJECT_CONTEXT_RECONSTRUCTION.md) is the coordinator's adjudicated synthesis at the repository root.
2. `COORDINATOR_VERIFICATION_NOTES.md` preserves direct checks and explicit corrections, including corrections to reviewer overclaims.
3. `SOURCE_REGISTRY.md` accounts for examined, inventoried-only, restricted and unknown sources.
4. Direct JSON/TSV receipts preserve observation timestamps, exact refs, metadata and limited inspection scope.
5. `evidence/*.md` preserves the ten returned reports unchanged. These contain known errors/overclaims; successful agent completion is not scientific acceptance. Do not follow their restart/test commands as present authorization.
6. `PROCESS_CHECKPOINT.md`, `WORKFLOW_RECEIPT.json`, `DOCUMENTATION_VERIFICATION.json`, `PRESERVATION_CHECK.json` and `SHA256SUMS` record handoff and verification.

The requested investigation was read-only except these research outputs and the global index entry. No experiment, benchmark, model inference, training, service/database mutation or deployment was performed. Research-agent model calls were explicitly authorized. No credentials or raw/private dataset payloads were copied into this bundle. Weight identifiers are metadata, not weight files.

## Native orchestration receipt

Workflow: `ea49dbe6-c72f-494d-b433-77701f7d9e45`; mission `465ab78f-1a75-4650-a7fb-efb005b9b2f9`.

| Lane | Model | Run ID | Preserved report |
|---|---|---|---|
| Sessions | openai-codex/gpt-6-luna | 8d082acb-b530-446a-89ae-a0bfda72768f | evidence/01-sessions.md |
| Backend Git | openai-codex/gpt-6-luna | 90e84a15-a1b3-486c-80ce-968117a6f8ed | evidence/02-backend-git.md |
| Backend goals | antigravity/gemini-3.8-flash | d346c962-4638-4c8a-b521-494d8d7fa099 | evidence/03-backend-goals.md |
| Drive/Colab | antigravity/gemini-3.8-flash | ab1fc8ad-6c19-40a2-ba35-e57dfced9b39 | evidence/04-drive-colab.md |
| Experiments/storage | openai-codex/gpt-6-luna | 790ace38-1628-47c9-9fce-df671d0a9f97 | evidence/05-experiments.md |
| Methodology/literature | antigravity/gemini-3.8-flash | aa219c66-b900-4370-8420-4750e8cf6354 | evidence/06-methodology.md |
| Integration/frontend | openai-codex/gpt-6-luna | 83df997b-a41b-4fa6-8e13-7084cc7865a8 | evidence/07-integration.md |
| Speech forensics | openai-codex/gpt-6-luna | 616bad65-ad7a-4d34-98cf-25a430367b53 | evidence/08-speech-forensics.md |
| Platform cross-check | openai-codex/gpt-6-luna | ccbe5581-da83-4c22-8f82-24bb47b9ac0e | evidence/09-verify-platform.md |
| Speech cross-check | antigravity/gemini-3.8-flash | c4924484-4d76-483b-992a-10b86e3f79aa | evidence/10-verify-speech.md |

Eight initial lanes ran concurrently; the two reviewers depended on their returned reports. Coordinator retained acceptance/publication authority, independently checked primary records and rejected unsupported interpretations. No lane needed a failed-run fallback or alternate CLI execution path. Watchdog inactivity notices were inspected; children continued and all ten returned.

## Important correction map

- Child01/07: reject software-test interpretation of RQ2; fix session date labels from literal timestamps.
- Child03: reject publication-grade/definitive benchmark and live LoRA deployment assertions; its Qwen chronology is corrected by direct Git history to May31,2026 inclusion.
- Child04/06: distinguish archive/name listing from remote digest verification, document date from approval, counts from data sufficiency, ProtocolZ feasibility from readiness, observational diagnostics from causality.
- Child05: anonymous HF access failure did not mean authenticated access was unavailable; parent recovered P-SUP summaries and actual visibility.
- Child08: parent filled live GitHub/Kaggle/HF gaps; current notebook pin verified dd2dd89.
- Reviewer10: realized CTC excluding deleted canonical phones is not no deletion supervision; binary already labels errors. Displayed E1 mean .000000 is not proof every token is exactly zero. No verified final VAL result is not proof none exists anywhere.

## Integrity and limitations

`SHA256SUMS` covers the final root report and evidence/checkpoint files except itself. It verifies file identity only, not truth of historical claims. `DOCUMENTATION_VERIFICATION.json` records local Markdown/JSON/inventory checks and limited credential-pattern screening; no functional tests were run. `PRESERVATION_CHECK.json` compares original inventoried size/mtime and Git metadata, not ignored runtime trees or cryptographic source contents.
