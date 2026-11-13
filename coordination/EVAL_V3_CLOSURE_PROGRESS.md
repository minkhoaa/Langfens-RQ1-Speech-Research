# EVAL-V3 closure — ĐANG THỰC THI, không kết thúc ở cc4e331

## Mốc nguồn

HEAD `cc4e33117a73974859c0dc2cdea66a823be40751`; initial1499source freeze giữ nguyên. Closure baseline85file V3: `experiments/evaluation_v3/closure/CLOSURE_BASELINE_SHA256.json`. Parent duy nhất writer vào repo; child writers chỉ trong `/tmp/langfens-evalv3-closure-cc4e331/*`. Tất cả child dùng `openai-codex/gpt-6-luna`; operator cũng cho phép Gemini nhưng chưa cần dùng.

## Đã đóng thêm bằng thực nghiệm

1. `research/evaluation_v3/waveform_replay.py`: full2699authorized WAVs→frozen CTC→reference-derived features; features **delta0 tất cả2699**, old14student checkpoints × CAL/SEARCH **28/28score parity delta0**. TRAIN1800/59562/61362, CAL300/10043/10343, SEARCH599/19755/20354. Một acoustic forward/utterance,9pilot cache được reuse. Output `experiments/evaluation_v3/closure/waveform_full/`, receipt+trace đầy đủ; 2660orthographic-mode PASS,39OOV abstain. Orthographic events không remap sang canonicalgold; canonical reference vẫn annotation-derived.5traces malformed unknownphone dùng exact historical SIL/zero-onehot fallback, khai báo rõ.
2. `checkpoint_recovery.py`: actual CUDA inference full CAL/SEARCH cho EXP04 và EXP06 từ saved frozen MLP checkpoints, không retrain; source rootchỉlưu500predictions trước đây. Outputs: `closure/checkpoint_replays/<EXPID>/*_scores.npz`, receipts actual CHECKPOINT_INFERENCE_REPLAY.
3. `teacher_checkpoint_replay.py`: cả9teacher frozen checkpoints đã chạy CAL/SEARCH bằng native FP16 autocast. Source-ledger protocol dùng AMP; lần thử FP32 đầu có delta0,000698 trên S_EXP01, được giữ diagnostic trong `closure/teacher_replays_fp32_diagnostic/`, không sửa ledger. Native AMP replay đã chạy thành công ở `closure/teacher_replays/`.
4. Calibration hardening: mỗi fit event bắt buộc speaker/event ID, null/duplicate từ chối, receipt thêm hashes binding scores/labels/IDs. Test Numerical contract cập nhật caller đúng per-event IDs. `closure/confidence_hardened/` đã tái xuất metrics/maps bằng code hardening.
5. Full2699reference-lineage audit:830strictmatches theo legacy CMUlookup,467lengthmismatches,51legacyOOVutterances/8wordforms; malformed reference inventory D_, ER), SPN, Y_, Z_. Existing lexicon drops apostrophes; new independent waveform lookup preserves them, remaining39OOV. Report margin fix: chỉ competitor margin dim46 khác -10→legacy-15; silence margin47 đã-15. Không sửa oldinfer_student.py hoặc canonical gold.

## Phát hiện validity quan trọng đang audit độc lập

Một số native historical models (ví dụM48/M69/M94) truyền `target_m` từ gold realized/error-conditioned auxiliary labels vào **inference vowel/consonant gate**. Không được coi như supervised-training-only. Native source có `target_m=None` averaging fallback; replay mới phải gắn variant `NONORACLE_NATIVE_HEAD_AVERAGE`, không giả historicalparity. M31forward không nhận target_m và vẫn clean. Independent Luna C đang audit toàn bộ .pt model families để xác nhận ảnh hưởng và sourcecalls.

## Các lane còn chạy / artifact handoff

- EARLY: retained run `3cdef46b-9ec9-42c9-853c-8d85c65a4323` đã xong module2models EXP08/10 trong `EARLY/checkpoint_families_early.py`, strictCPUload/tests; parent chưa tích hợp/chạyCUDA.7families13/17/21/22/24/26/28 còn pending, không phải missing.
- MID: retained run `8a4a77a0-a328-4878-91f5-73e12783e5dc` đã xong EXP31/34/38 adapter và8tests/feature-native-vs-scrubbed parity. `MID/mid_adapters.py`, tests. Parent chưa tích hợp/CUDA. Cần review unknown `split` chỉ ở error-messagepaths. M38 usesnonoracle fallback.
- LATE: retained run `328dba99-160e-4e6a-aebc-538b72e72e33` đang implement actualM94/104 inputs, không chấp nhận 'native inputs absent' khi có thể derive từbaseX/WAV/CMU/TRAINstats. SourceM94 importsM48build_hrtd +M68pyramid +M69priors/prompt +M63gapcategories.
- Independent sourcegate C: `26c2e407-5b34-43e1-a6a3-36bc1638a1c9`, source validity audit of gold-conditioned inference vs training-only auxiliarylabels.
- B recovery metadata đã xong (`e6b407e2-70e0-4722-a70d-e4faeb4eac96`); adapter2models integrated. Broader sourceckptaudit57files:4checkpointfiles supported,48ADAPTER_PENDING,5serialization/runtimeconditions. Không gọi adapterpending là unavailable.

## Những phần phải tiếp tục trước khi chốt

- Thực chạy các adapter08/10/31/34/38 đã review, mở rộng float families còn reconstructible; quantized native CPU cases cần xem contract GPU-first, không silentlydequantize/fallback.
- Sửa recovery_audit.py administrative classification: EXP24 là neural MONOTONIC_LOCAL_ATTENTION_NETWORK, không phải NOT_A_MODEL dù tên chứaROADMAP. Numeric-only aliasesA102→EXP102 đã loại trong latestBversion. Candidate không đồng nghĩa recovery verified.
- Tạo unified closure evaluator cho new checkpoint scores, CAL-selected policies/calibration/reliability/topK/robustness cho mọi model có compatible CAL scores; complete requirement matrix mỗi gap trueblock hoặc completed evidence.
- Cập nhật inventory/status, reports/leaderboard/hashes/tests versioned; giữ cc4sourceoutputs có snapshot trongGit/baseline, không ghi đè initialfreeze. Scoped commit bổ sung khi toàn bộ đường khả thi đã kiểm chứng hoặc có blocker cụ thể.

Không neural training/backward, không restartPhase3B, không network dataset/model download, không đọc TEST_SEALED/GPG/replication hoặc external corpus. Không dừng chỉ vì operator hỏi estimate.
