# LANGFENS — SESSION HANDOFF (2026-10-28)

**Status:** HANDOFF_FROZEN. **Last verified operational HEAD:** ba93aaea759710ac796ed75b182f495d3cc17e67 on fix/rq1-trn-02b-colab-auth-r3. **Frozen R3 source:** ba7be0dc407313a4f5e31577051e6746278adeed.

## Read first
- [AGENTS.md](https://github.com/minkhoaa/Langfens-RQ1-Speech-Research/blob/ba93aaea759710ac796ed75b182f495d3cc17e67/AGENTS.md)
- [Checkpoint INDEX](https://github.com/minkhoaa/Langfens-RQ1-Speech-Research/blob/ba93aaea759710ac796ed75b182f495d3cc17e67/docs/process-checkpoints/INDEX.md)
- [RQ1-TRN-02B overview](https://github.com/minkhoaa/Langfens-RQ1-Speech-Research/blob/ba93aaea759710ac796ed75b182f495d3cc17e67/manifests/RQ1-TRN-02B/PROCESS_CHECKPOINT.md)
- [Checkpoint 08 — live T4 probe](https://github.com/minkhoaa/Langfens-RQ1-Speech-Research/blob/ba93aaea759710ac796ed75b182f495d3cc17e67/manifests/RQ1-TRN-02B/checkpoints/08-live-gpu-runtime-and-staging-verification.md)

## Current state
C0 p=0.00 and C1 p=0.25: **NOT_STARTED**, zero official optimizer updates, no trained artifacts. Baseline R: completed; historical matched AP 0.317021, mismatched AP 0.294722, canonical prior AP 0.303063, CTC PER 0.2364, Protocol Z AP 0.313530. Never conflate metric protocols. Agent reports R3 software 190 passed / 4 skipped; full suite was not rerun by the assistant. No official TEST audio/labels were accessed for this step.

**Step 08 agent report:** paid Colab compute-unit balance 0.00, free-tier tokens reportedly allowed one Tesla T4 test session (15 GiB, CUDA available). Drive was unmounted; interactive OAuth required /dev/tty and blocked automated staging. Test GPU was stopped. This is an agent-reported runtime probe preserved in GitHub, not independently remeasured by ChatGPT.

**Latest user CPU question:** Colab UI said \"At your current usage level, this runtime may last up to 36 hours 20 minutes.\" This is not an uptime guarantee nor CPU-training ETA. Current AGENTS.md and trainer require Colab GPU CUDA FP16. Proposed Colab CPU FP32 alternative is **NOT AUTHORIZED/IMPLEMENTED/BENCHMARKED**; it needs explicit protocol change on a review branch, then a disposable 50 successful optimizer-update CPU benchmark (not counted toward official C0) to measure seconds/update, RAM and forecast 3600 updates. If feasible and approved, official C0 then independent C1 under matched CPU FP32 conditions. Do not silently mix GPU FP16 and CPU FP32 or reduce the frozen update budget.

## Scientific invariants
Fold 1: 2399 TRAIN/16 speakers, 300 VAL (NJS, RRBI), 900 SEALED TEST/6 speakers. Frozen manifest SHA256 7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b. Baseline best.pt SHA256 a2445c8d17910bf3c03101f50fc419675e8d1bc012af43af10267a28dbba948b. Authorized TRAIN/VAL ZIP SHA256 b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9. Both arms seed 1337, batch 2, gradient accumulation 2, frozen Wav2Vec2, 6 epochs / 3600 successful optimizer updates max per arm, independently cold-start from the same Baseline R checkpoint. Do not access official TEST or train diagnosis head.

## Artifact sources
Drive root https://drive.google.com/drive/folders/1Zms4bwv3QYZcmDQAeYolbq-ovrZVRlnK ; baseline folder https://drive.google.com/drive/folders/11g-5JI5u5JKHxV8u9kyCo4I-SF2JWJLx ; Baseline best.pt file ID 10SBhwWd0eAv_ZByBmfJjJZwHywA-LU-I ; authorized ZIP ID 1xdypDWopd585OjuQeHjxlWOJfU2oiJiY. Private HF model https://huggingface.co/minkhoaa/langfens-rq1-fold1-baseline-r (agent-reported release, private readback not independently authenticated). Private Hugging Face metadata-only Dataset publication remains **UNVERIFIED**, despite preparation of a release ZIP.

## History
Historical phase registry backfill: 11 phase checkpoints and initial 26 step checkpoints, branch docs/process-checkpoints-backfill, commit aa6415a8d3a073a3550f209a53e1baa44461ce97; later step checkpoints expanded through step 08. RQ1-TRN-02B correction commits in order: 8486cbb3596a0f5d2603dcdf46fd891b5c849840 (preflight), a497ec3f0b6dd4c760c4e40296481f61b6426d8d (R1), ea8762079735cfb5520de49eaca4528a458a1f38 (R2), 4b5baccbb40d7e00bb710b42979196eaef4ce4d3 (R2 hotfix), ba7be0dc407313a4f5e31577051e6746278adeed (R3), ba93aaea759710ac796ed75b182f495d3cc17e67 (step-08 live probe documentation).

## Restart protocol
Verify latest GitHub HEAD, read checkpoint 08, check actual current Colab state and confirm CPU FP32 protocol authorization vs original GPU FP16 route. Do not start more broad preflight-only rounds. If CPU is selected, execute an actual (authorized) 50-update Colab CPU benchmark with real timings before assessing feasibility; do not promise 36-hour completion. If GPU is selected, use verified Drive OAuth/authentic storage staging, suitable T4 availability and operator authorization. Do not fabricate mount attestation, results or checkpoint metadata. Any completed, interrupted or blocked process MUST create a new numbered Markdown checkpoint and update INDEX + phase overview. No training was performed in this handoff.

## New-session message
\"Tiếp tục LANGFENS từ HANDOFF checkpoint 2026-10-28 trên branch docs/session-handoff-2026-10-28-cpu-decision. Đọc AGENTS.md, INDEX, checkpoint 08. C0/C1 và CPU benchmark chưa chạy. Tôi muốn ưu tiên Colab CPU feasibility nhưng phải benchmark thực tế và xử lý thay đổi GPU-only policy minh bạch. Không được suy luận 36h20 là training ETA; không đụng sealed TEST. Hãy triển khai bước thực tế tiếp theo khi có quyền, tạo Markdown checkpoint.\"
