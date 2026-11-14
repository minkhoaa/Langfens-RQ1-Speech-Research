# Bounded corrective continuation (no formal training)

After initial three parallel lanes and final review, supervisor inspected returned source directly. Initial child claims of unavailable data/MB03 producer/inner target provenance were incorrect and are not accepted as established blockers.

## Verified recovery sources

- `experiments/rq1_method_bench_03/agent_c/train_speaker_adversary.py`: exact `get_arrays`, `fit_predict`, `Detector`, including PLAIN dormant speaker head and RNG consumption. Complete optimizer/sampler/scaler/class-weight recipe exists. Legacy checkpoints remain absent; this is source recovery, not checkpoint replay.
- `experiments/rq1_method_bench_07/agent_a/nested_direct_seed101_inner_oof.npz`: inner target scores plus exact global indices.
- `experiments/rq1_method_bench_07/agent_a/ctc_nested_e4_seed101_inner.npz`: per outer/inner UID S/O LLR payload. Its receipt and saved CTC checkpoint metadata identify fit/held groups and hashes.
- `experiments/rq1_method_bench_07/agent_a/evaluate_nested_e4_fusion.py`: producer's exact keyed inner join and fixed .5 fusion reconstruction. Do not run its module body; it writes old outputs.
- `experiments/rq1_method_bench_08/supervisor/audit_inputs.py`: prior read-only integrity/exclusion logic. New supervisor verification confirms all archived evidence unchanged.
- H002 TRAIN crops/references/event IDs exist under `research/long_running_method_discovery/cycle_02/h002_ssl/normalized_v1/`; staging to lane is unnecessary for read-only access.

## Actual initial defect

Agent A gap_input allocates N rows, not N+1, and shifts adjacent features. Its passing synthetic test incorrectly expects N. This is a genuine blocker; preserve initial evidence and repair exact leading/interior/trailing values plus zero-length topology before using helper. Initial tests alone do not validate the contract.

## Unknown semantics clarification

The 39D neural/gap reference unknowns remain four zero vectors with slots intact. The separately frozen CTC42 inventory explicitly contains SPN and documents aliases ER)→ER, D_→D, Z_→Z (also Y_→Y, AH)→AH). These explicit legacy normalization rules are not an AA/phone0 fallback and need not erase the original raw token or zero39 reference. Any unregistered canonical token must fail closed. Prompt-only provenance remains unverified and is not claimed.

## Orchestration / authority

Only unfinished bounded requirements were resumed in retained native `openai-codex/gpt-6-luna` lanes: A `bcf2c149-1ef5-4ffd-8df1-3401f172e5c2`; B `26d5c734-d03d-49f6-b7b2-552e0bd50cb5`; C `0eb6c855-90a4-4ccb-89e2-9a58c67782bc`. Same sole-writer boundaries, CPU-only rules and unchanged scientific acceptance apply. There was no tool/launch/runtime failure or alternate CLI fallback. Initial incomplete work is not portrayed as a successful preflight. New code/config freezes must precede further tests/outcomes. B retains the original at-most12 inner+4 outer gap refits/20-minute CPU cap; no completed experiments or target/CTC fits are restarted. Supervisor owns final acceptance.
