# Cycle 2 failures and blockers

## Initial Pi/Codex task API attempt

- Requested model: `openai-codex/gpt-6-luna`.
- Mechanism checked: native Pi/Codex `create_thread`.
- Validation attempt: R1/H-001 startup, thread `01a1b5e4-866c-7b13-901d-5e01fa8c47d4`.
- Exact startup error: `The 'openai-codex/gpt-6-luna' model is not supported when using Codex with a ChatGPT account.`
- Outcome: this API route failed for the requested full alias. After the operator explicitly requested a small model in-session, native collaboration accepted `gpt-6-luna`; all six workstreams were dispatched with that model. R1 was restarted through native collaboration. No model substitution was made.
- This initial route failure does not block the six active collaboration agents.

## CUDA gate

- Requested device: NVIDIA GTX 1650 4GB.
- Probe: `nvidia-smi`.
- Exact result: `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.`
- Outcome: no live VRAM measurement, no CUDA lease, no model inference/training. Historical Phase 3 lockfiles and ledger entries do not establish current availability. CPU-side source, literature and statistical work remains permitted and is underway.
- Needed change: restore device driver/runtime and verify a live lease/capacity under the project scheduler before any neural job.

## Scientific constraints retained

No TEST_SEALED/protected data accessed. No historical checkpoint/output modified. No CPU neural fallback. H-001 and H-003 executed conventional/frozen-score comparisons with independent AP checks, but neither is independent confirmation; no neural experiment is verified. Existing SEARCH metrics remain development-exposed.

## CUDA screening and bounded failures (2026-11-07)

- `.venv-cuda` is available at `/home/khoa/Projects/KLTN/Dataset/.venv-cuda`; PyTorch 2.14.1+cu130 sees GTX 1650. Initial absent `/dev/nvidia*` visibility was resolved after the operator installed drivers. Smoke and Gold-Only baseline succeeded. Full receipts are in `baseline_control/`.
- H-004 raw OOF KD had two pre-training wrapper attempts that exited 1 because `.venv-cuda` lacked sklearn and the copied script's import path was incomplete. Both leases released; no outputs were used. The fixed run `...-R4` succeeded. Receipt clarifies raw, uncalibrated OOF TRAIN teacher targets.
- H-006 first wrapper attempt exited 1 before training because importing the source teacher module pulled missing sklearn. The run was corrected to use the verified source phone order literal, then treatment and capacity-matched zero-auxiliary control succeeded under `.venv-cuda`. No neural updates came from the failed attempt.
- H-004 CPU teacher-embedding preflight cache was generated to verify event joins and shapes, then explicitly disqualified after supervisor direction that new neural teacher forward passes use `.venv-cuda` and the GPU wrapper. It is retained but not used for training. The CUDA extractor avoids the sklearn imports and writes distinct `*_target_cuda_oof_embedding.npz`; it must wait for the active H-002 lease.

- H-004 CUDA teacher-feature extraction attempt `RC2-H004-OOF-TEACHER-FEATURE-EXTRACT-CUDA130` exited 1 before teacher forwards because the inline O architecture had extra BatchNorm/GELU layers; saved checkpoint shape inspection identified exact expected layer indices. Corrected R2 run succeeded strictly under `.venv-cuda` lease; no output from R1 was used.
- H-004 feature-KD training attempt `RC2-H004-FEATURE-KD-CUDA130-S101` was interrupted by the supervisor before epoch 1 after repeated compressed-NPZ indexing caused high RSS (~21 GiB) while GPU remained idle. The wrapper process ended by SIGINT; supervisor verified child/lock release and appended a reconciled `LEASE_END` return code 130. Joined arrays were pre-cached outside the lease, and the V2 treatment plus capacity-matched zero-loss control completed successfully under new job IDs.

## H-002 input-contract and retry trail (2026-11-07)

- `RC2-H002-ASR-FINETUNED-EXTRACT-S101-R2` failed early because `.venv-cuda` torchaudio required missing TorchCodec. The R3 attempt encountered non-normalized annotation spellings (`ER)`, `D_`, `Z_`, `Y_`) and was corrected to match the existing pipeline's OOV/SIL behavior.
- A subsequent raw-wave extraction/head job ran CUDA, but the pinned Wav2Vec2 `preprocessor_config.json` states `do_normalize=true`. The hidden states therefore violated the checkpoint input contract. Its cache and checkpoint are preserved under `h002_ssl/failures/attempt4_raw_input_contract/`; the head also exited before prediction publication on a metadata dtype mismatch. No scores from this arm were evaluated or used.
- Correct run `RC2-H002-ASR-FINETUNED-NORM-S101` used the local `AutoFeatureExtractor` with configured normalization, a separate `normalized_v1/` cache, exact event checks, CUDA extraction and head training; inference perturbation controls followed. It completed EVAL-V3.1. This is explicitly ASR-finetuned sensitivity evidence, not pure pretraining-only SSL.

## Unrelated GPU process

After all RC2 GPU leases were released, `nvidia-smi` showed an external `Pronunciation-Scientific-Recovery/.venv/bin/python` process using about 134 MiB. It was not stopped or altered. No RC2 CUDA child was active at the last check.
