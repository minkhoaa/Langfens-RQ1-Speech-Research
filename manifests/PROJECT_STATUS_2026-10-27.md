# LANGFENS / KLTN — State Snapshot (2026-10-27, Asia/Ho_Chi_Minh)

**Purpose:** Durable handoff / current-state checkpoint, not an authorization to execute training, merge, publish, or upload artifacts.

## 1. Overall project stage

- Research pipeline: L2-ARCTIC phoneme-level mispronunciation detection (Binary MDD); a diagnosis head exists but is not trained in the current two-head experiment.
- **RQ1-TRN-02A baseline:** completed and scientifically audited on Fold 1.
- **RQ1-TRN-02B DESIGN-R1:** accepted, controlled no-dropout C0 vs Binary-only phone-embedding dropout C1.
- **RQ1-TRN-02B implementation preflight through R2:** audited; safety patches in source. No training for C0/C1 has occurred.
- **Current stage:** `RQ1-TRN-02B-COLAB-AUTH-R1` operational-packet corrections. **Operational authorization BLOCKED** as of this snapshot. This is a resource/staging/launch-command/provenance gate, not a request to redesign the scientific experiment.
- No new Colab GPU runtime, backward pass, optimizer update, official six-speaker TEST evaluation, HF model publication, or merge authorized.

## 2. GitHub provenance (verified by connected GitHub read)

Repository: `minkhoaa/Langfens-RQ1-Speech-Research`

- Historical RQ1-TRN-02A scientific baseline: `c4a658ffbd5a2afd2ebdce172e47af43ed9bb63e`.
- DESIGN-R1: `428061de9d0b18ac292e0c31008047e757de6ef0`.
- Implementation preflight: `2f6345e426279dd541cc830d4e31dd2d7823a7e7`.
- Preflight R1: `9fe993b6af0346a0d8170e328b5416cab30bdbdd`.
- Preflight R2: `6b4cf47fba8cb75ff338540f1dd07f7c8d267d14`.
- Additional output conflict protection: `23977681408cf56b8246b4a99c28c7b85636ed6a`.
- **Latest reviewed execution-packet branch HEAD:** `8486cbb3596a0f5d2603dcdf46fd891b5c849840` on `fix/rq1-trn-02b-preflight-r2`.
- Packet file: `manifests/RQ1-TRN-02B/COLAB_EXECUTION_AUTHORIZATION_PACKET.md`. The packet's embedded SHA `96729fe...` does not correspond to the verified head and must not be treated as immutable provenance.
- This status snapshot is on a separate documentation branch; it does **not** modify the execution branch.

## 3. Frozen Fold 1 research artifacts and contract

- Frozen split file: `manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json`, SHA-256 `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`.
- TRAIN: 2,399 utterances / 16 speakers: `BWC, ERMS, HKK, HQTV, MBMPS, NCC, SKA, SVBI, THV, TLV, TNI, TXHC, YBAA, YDCK, YKWK, ZHAA`.
- VAL: 300 utterances / `NJS, RRBI`.
- SEALED TEST: 900 utterances / `ABA, ASI, EBVS, HJK, LXC, PNV`. Do not stage or access it.
- Baseline production `best.pt`: SHA-256 `a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b`. Epoch 6, 3,600 successful updates; different historical AP evaluations include 0.316062 (selection) and 0.317021 (matched acoustic control).
- Historical Protocol Z aligned-only AP 0.313530, 9,919/9,956 gold-position alignment coverage (~99.63%); phone-only train-prior AP 0.303063; within-speaker mismatched-audio AP 0.294722 vs matched 0.317021. Preserve different evaluation provenances.
- Frozen Wav2Vec2 backbone: `facebook/wav2vec2-base-960h` revision `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`.
- Authorized train/val-only ZIP expected SHA-256: `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`.
- All hashes above come from the project ledger; Google Drive objects located in this snapshot were **not re-downloaded and independently rehashed** during this inspection.

## 4. Controlled RQ1-TRN-02B experiment (NOT YET RUN)

- Reference R: frozen historical production `best.pt`, no further training.
- Control C0: from same checkpoint, fine-tune with Binary prompt dropout `p=0`.
- Treatment C1: from same checkpoint, fine-tune with Binary-only whole-phone-embedding masking `p=0.25`, post-cross-attention, not changing canonical queries or CTC path at the point of masking.
- Identical non-intervention configs (seed 1337; AdamW lr 0.001; batch 2; gradient accumulation 2; CTC + Binary BCE; frozen acoustic backbone; diagnosis inactive).
- Maximum **6 epochs AND 3,600 successful optimizer updates per arm**; combined ceiling 7,200 additional updates. AMP skips/guardrails may make a fixed-budget endpoint incomplete; do not relabel it as completed.
- Primary controlled contrast: C1 − C0 at a matched validated endpoint, measured on Protocol O matched/mismatched audio. Protocol Z is separately reported and never used to select checkpoints.
- Agent reported 157 PASS and 4 SKIP regression tests after packet preparation; not independently re-executed in this session.
- The C0 and C1 trained checkpoints, reports, and scientific conclusions **do not yet exist**.

## 5. Google Drive inventory (independently confirmed by connected Drive metadata/listing on 2026-10-27)

The project **does have prior-phase data and weights on Google Drive**, even though RQ1-TRN-02B does not have a discovered output folder/checkpoint.

- Root folder `KLTN-Dataset`: https://drive.google.com/drive/folders/1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK
- `KLTN-Dataset/manifests/RQ1-TRN-02A/checkpoints/best.pt` (Drive lists ~412 MB): https://drive.google.com/file/d/10SBhwWd0eAv_ZByBmfJjJZwHywA-LU-I/view
- `last.pt` (Drive lists ~412 MB): https://drive.google.com/file/d/1VdlhoQHfd9MqjzIkMUEqdqMxhLlJiO-s/view
- Fold 1 authorized `train_val_only.zip` (Drive lists ~762 MB) under `RQ1-TRN-02A/authorized_data`: https://drive.google.com/file/d/1xdypDWopd585OjuQeHjxlWOJfU2oiJiY/view
- Other RQ1-TRN-02A evidence: `history.json`, `EVALUATION_REPORT.json`, `BASELINE_COMPARISON.json` in Drive checkpoint folder.
- Drive search for `RQ1-TRN-02B`: no results at snapshot time. Do not infer Drive is entirely empty.
- Access is confirmed through the connected account's Drive listing, **not** proof that the planned new Colab VM already has an authenticated Drive mount, or that packet paths are correct.

## 6. Hugging Face inventory

- No LANGFENS-specific publicly indexed model/dataset repository was located by limited web search, and `huggingface.co/minkhoaa` could not be opened through the available lookup.
- **Publication is NOT verified**. This does **not** rule out private repositories, different account names, or resources not indexed.
- The project uses a *third-party pretrained backbone* from Hugging Face; that is distinct from publishing a trained LANGFENS checkpoint.
- No HF upload should happen without explicit operator approval and review of reproducibility, licensing, sealed-test leakage and model card.

## 7. Outstanding operational blockers (not new model research)

1. **Correct the authorization packet's false speaker inventory:** it includes sealed TEST speakers in its claimed TRAIN list; frozen split itself is correct.
2. **Correct Colab CLI launch commands:** documented `colab exec ... -- "<shell strings>"` must be tested against the actual CLI; do not assume shell strings are supported.
3. **Stage the non-Git checkpoint:** Git repo intentionally excludes `best.pt`; the new VM must receive it from approved storage and verify SHA-256.
4. **Stage authorized ZIP** from a verified Drive location and verify SHA-256 on VM-local disk.
5. **Drive persistence/access:** do not assume `/content/drive/MyDrive` is available on a fresh CLI-provisioned runtime; confirm an authorized mount or non-interactive mechanism, with write/read/hash checks.
6. **Fix execution SHA:** use immutable checked-out commit, not a floating branch or the packet's inconsistent embedded SHA.
7. **Colab quota:** agent reported 0.00 available compute units; current actual balance was not independently read during this snapshot. Do not start training with insufficient quota.

## 8. Next steps and stop policy

- Next authorized work: `RQ1-TRN-02B-COLAB-AUTH-R1` documentation/operational corrections, no GPU.
- After operator verifies quota and staging access, revisit explicit authorization for C0 then C1 (estimated ~7–9 T4 GPU-hours for both; estimate only).
- Once **both** valid primary endpoint checkpoints exist, run paired Protocol O/Z and report AP, acoustic sensitivity, CTC PER, uncertainty and limitations.
- Further folds/seeds, diagnosis training, official TEST release/evaluation and public HF publication are **separate future decisions**, not approved by this snapshot.
- Do not silently start GPU allocation, local fallback training, publish model, merge code or access sealed TEST.

**State of record:** GitHub provenance and preflight code are present; Google Drive has baseline data/checkpoints, not discovered RQ1-TRN-02B outputs; Hugging Face release not verified; Colab execution gate remains BLOCKED pending operational work.
