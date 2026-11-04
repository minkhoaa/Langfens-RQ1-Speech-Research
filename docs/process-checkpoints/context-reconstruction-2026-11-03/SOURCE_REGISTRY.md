# Source registry and coverage accounting

This is an inventory, not a claim that every file/transcript was read. Final conclusions belong in `PROJECT_CONTEXT_RECONSTRUCTION.md`; coordinator corrections control over unreviewed child prose.

## Durable local metadata observations

| Source | Receipt | Coverage and limits |
|---|---|---|
| Speech/backend/frontend Git | `SOURCE_DISCOVERY_SNAPSHOT.json` | Exact roots, HEADs, branches, all local refs, tags, reachable commit lists, reflogs, worktrees and dirty status; no fetch; not unreachable-object recovery. |
| Source file paths | `SOURCE_FILE_INVENTORY.tsv` | 5,196 tracked/nonignored-untracked paths across three primary checkouts; metadata only; ignored data, cache, models and hidden external stores not exhaustively inventoried. |
| Pi histories | `PI_SESSION_INVENTORY.tsv` | 260 project-related top-level JSONL session headers; current mission included as inventory only, not independent evidence; no claim all bodies read. |
| Selected historical decisions | `evidence/01-sessions.md` | User-message indexing of 73 speech and 183 backend sessions; selected body review; zero frontend files in named Pi cwd root. Some raw date labels are wrong: see coordinator C-05. |
| Other agent stores | `evidence/01-sessions.md` | OMP, Antigravity/CLI, OpenCode and Orca presence/count/name-search inventory only; not exhaustive content search, not proof of no relevant records. Parent also discovered `~/.claude/session-data` (not read) and no standard session subtree at bounded depth in `~/.codex`; no credential contents opened. |
| Original thesis intent | `COORDINATOR_VERIFICATION_NOTES.md` C-01 | Original proposal line14 directly read; proposal date line16 directly read. RQ1 speech and feedback; RQ2 educational content, not software-test generation. No independent approval/signature authenticity verification. |
| Backend original models | Coordinator notes C-08 | Qwen trainer state/card and pronunciation CSV/source inspected, no weight loading. Full dataset/training-platform/authorization provenance unresolved. |
| Generation historical results | Coordinator notes C-09 | Actual completion manifest, benchmark report and two computed metric headers directly read; exact r4/projection text-document hashes independently recomputed. No run/test replay. |
| Speech diagnostic correction | `manifests/RQ1-DIAG-02B/SCIENTIFIC_REPORT.md` | Parent and lane8 inspected current report/final artifact references; no new inference or parity replay. |

## Direct remote observations (read-only)

| Ecosystem | Receipt | Established / not established |
|---|---|---|
| Three main GitHub repos | `GITHUB_METADATA_OBSERVATION.json` | Public visibility, default branch, branch tips, all returned PR/issue/release metadata; one page100 per endpoint, returned counts<100. CI history/status, PR comments/checks and deployed runtime not comprehensively reviewed. |
| Related fourth GitHub repo | `RELATED_SOURCE_DISCOVERY.json` | Public `minkhoaa/langfens-ielts-dataset`, main fcc18c97eee4b93a25ff464fe912399ce1c98e99; root metadata inventoried, no data payloads opened. No README content returned. |
| HF model repositories | `HF_METADATA_OBSERVATION.json` | Existing authenticated metadata access to baseline R (public), C0/C1 (private), cross-fold/P-SUP (private); file sizes/LFS hashes and immutable revisions, not payload readback. |
| HF author and commit inventory | `HF_SOURCE_REGISTRY.json` | Three project model repos returned; no matching dataset repos returned; latest22 cross-fold commits sampled, not full upload history or proof of no other assets. |
| P-SUP historical training records | `P_SUP_REMOTE_METADATA.json` | Authenticated small JSON GETs at immutable revision; each arm records3600 successful/0 skipped; selection AP only. No final VAL results verified, no independent training replay or checkpoint tensor validation. |
| Current P-SUP Kaggle notebooks | `KAGGLE_METADATA_OBSERVATION.json` | Status COMPLETE, current source pins dd2dd89, remote-execution flag, T4 metadata; server reports notebook is_private=false. Does not identify all historical version/run linkage or prove public data exposure. |
| Kaggle author inventory | `RELATED_SOURCE_DISCOVERY.json` |31 kernels returned,29 related retained; four related staging datasets returned. Platform naive timestamps retained raw. No submissions/competitions discovered as relevant; not a full account archaeology. |
| Kaggle staging datasets | `KAGGLE_DATASET_METADATA.json` | All four private, versions3/2/1/1, metadata CC BY-NC4.0; no raw content or sealed audio/labels read. |
| Colab live state | `DRIVE_COLAB_OBSERVATION.json` |0.00 compute units,0 assignments, no active sessions; no new runtime/auth/mount/stop calls. |
| Drive paths/notebooks | `evidence/04-drive-colab.md` | Existing authenticated rclone listings, root folders/notebook names; original archives and checkpoint IDs/hashes mostly historical references, not freshly verified payload digests. Native Google Docs/Sheets search and notebook bodies/execution histories not exhaustively inspected. |
| Archived Drive projects | `DRIVE_COLAB_OBSERVATION.json`, `DRIVE_BACKUP_INVENTORY.json` | `langfens-projects-2026-08-20` has backend and two frontend-named directory roots; backend docs/.claude paths inventoried. Backup name is not proof of snapshot time or archived Git ref. No secret files, .git objects or model/data payloads read. |

## Concurrent-document observation

`CONCURRENT_DOCUMENT_OBSERVATION.json` inventories an additional untracked backend-root `PROJECT_CONTEXT_RECONSTRUCTION.md` that appeared at final validation. It was not read or used as independent historical evidence; author UNKNOWN. B/F original inventoried source metadata and HEADs remained stable, but strict identical backend status is false. Initial validation/preservation receipts are preserved with `_INITIAL` suffixes; no reset/delete was attempted.

## Explicitly limited / deliberately excluded

- Original and staged audio/annotation/weight payloads: not newly downloaded, loaded or extracted; sealed TEST restrictions preserved.
- Credentials, secret stores and unrelated personal conversations/files: excluded.
- Paywalled literature: citation/primary access limits retained; no claim of full-text reading where blocked.
- Old proposal variants, missing distinct omission proposal, deleted/unreachable sessions/commits, archived source bodies and other accounts: not exhaustively recovered.
- No new project tests/builds, model inference, training, generation benchmarks, service startup, migrations, deployment, database access or remote mutation. The ten research-agent model calls/workflows were explicitly requested; this exclusion does not claim zero LLM usage for the investigation.
- This registry supports a reconstruction with accounted gaps, not a declaration that all requested universal completeness criteria are satisfied.
