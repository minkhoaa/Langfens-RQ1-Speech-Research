# LANGFENS Phase 4 — Agent D Unified Student

## Scope and status (2026-11-07)

This is an initial D0/D1/D2 audit and input-contract scaffold, not a training/evaluation result. Work is restricted to this report and `experiments/phase4/unified_student/`. No SEARCH-based model selection, sealed data access, or neural training was performed. D3+ remain blocked on provenance gates.

## D0 — Input and data contract

- The validated primary interface is genuine local WAV audio plus an intended ARPAbet phone sequence. `data_contract.py` checks RIFF/WAVE `fmt` and `data` chunks, supported PCM/IEEE-float encoding, non-empty frame count, finite IEEE-float samples, 16 kHz default sample rate, and supported phone symbols. It does not accept gold labels or realized-phone timings as model inputs.
- Exercised the loader against local TRAIN `ABA_arctic_a0003.wav`: 16,000 Hz, 78,720 frames, caller-supplied intended phones `DH AH` accepted. This was file/input validation only; no neural inference/training was run.
- The contract permits an empty reference, matching the already documented empty-reference inference case. For N reference phones, gold target cardinality is N and gap cardinality is N+1, with gap slots including both utterance boundaries. Padding masks expose exactly those valid slots, including one valid gap when N=0.
- This data interface validates raw input only; it does not claim an alignment, phone inventory lineage, or intended-reference independence. UPED handoff documents CMUdict/G2P as future components and reports 6.5% OOV on its cited population. A no-rejection/OOV-zero gate needs an approved G2P and variant-resolution contract before it can be claimed.

## D1 — Gold-Only GRU audit

The corrected Phase 3 receipts document `STU-01-GOLD-ONLY` as a 78,661-parameter, bidirectional GRU student, trained for 12 epochs / 684 updates on CUDA. Seed 101 receipt reports 49.985 seconds and 44.094 MiB peak allocated VRAM, with required CUDA packed-sequence lengths, N/N+1 masks, TRAIN-derived class weights, and unweighted gold DEV_CAL target+gap loss for checkpoint selection. Seeds 101/202/303 are represented in receipts; their gold-only headline Macro-4 AP values are 0.236917/0.234517/0.231667 in the EVAL-V3 leaderboard.

These are historical Phase 3 evidence, not Phase 4 runs or evidence that inputs are independently generated from WAV+orthographic text. The audio inference receipt establishes one sampled CUDA WAV+intended-phones run, not a full-population gate. Phase 4 will freeze seed list, data hashes, initialization, optimizer/update budget, selection rule, and runtime receipt schema before any matched run. No new training has been started.

## D2 — EXP24 architecture/training contract audit

The UPED handoff calls EXP24 a standalone `JointSoftAttentionCTCNetwork` / M24-ctc-attention-joint-decoder, reports 533,272 parameters, 55-dimensional target inputs and 6-dimensional gap inputs, with reported Macro-3 Error AP 0.276819, AP omission 0.225092, and exact-gap AP insertion 0.247628 in the EVAL-V3 closure report. Those are published comparison values, not a reproduced Phase 4 baseline.

The retrieved handoff and closure summary do not provide enough primary-source detail here to reconstruct EXP24's full preprocessing, acoustic checkpoint identity, feature normalization, initialization, loss, optimizer, epochs/update budget, stopping/selection, or split provenance as a matched training contract. Therefore do not claim architecture parity or train a purported matched EXP24 replication until those items are traced to primary artifacts and independently frozen. The proposed UPED design in the handoff (Wav2Vec2, Viterbi alignment, BiLSTM, monotonic local attention and multitask heads) is a proposal, not verified EXP24 implementation detail.

## Resource and selection policy

A point-in-time `nvidia-smi` check showed GTX 1650 4 GiB, 2 MiB used, 0% utilization, and no listed process. This is capacity evidence only, not an acquired lease. Before any neural forward/backward/training, require the project CUDA lease and recheck capacity; abort rather than CPU fallback or oversubscription. Neural work remains unrun. No SEARCH tuning or candidate selection was conducted.

## Files and validation

- `experiments/phase4/unified_student/data_contract.py`: pure validation and mask/label interface.
- `experiments/phase4/unified_student/test_data_contract.py`: tests for reference validation, mask cardinality, empty-reference edge gap, and label IDs.
- Tests: `python -m unittest discover -s experiments/phase4/unified_student -p 'test_*.py'` (four tests).

## Blockers / gates before D3+

1. Freeze authoritative EXP24 source/checkpoint and complete preprocessing/training contract; resolve apparent distinctions between the Phase 3 GRU baseline and proposed UPED architecture.
2. Establish intended-reference provenance and approved text-to-phone/OOV/variant resolver; verify genuine audio and reference lineage on authorized data without gold/realized-phone conditioning.
3. Freeze paired seeds and budgets, train/validation boundaries, target/gap objective and checkpoint-selection rule before execution.
4. Obtain a valid exclusive CUDA lease and record device, software, timing, memory, source/data/checkpoint hashes. Never fall back to CPU.
5. Do not access DEV_SEARCH for fitting or selection; any later use must follow its authorized final evaluation role.
