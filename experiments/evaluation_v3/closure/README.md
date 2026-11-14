# EVAL-V3 — bằng chứng closure bổ sung

**CLOSED — bản bàn giao chính thức:** `release_final/` (1340 results/840 new operating rows/291 experiment-seed inventory rows). Báo cáo: `reports/EVAL_V3_CLOSURE_REPORT.md`; acceptance: `reports/EVAL_V3_REQUIREMENTS_ACCEPTANCE.md`. Calibration, all92 robustness, classification selective, canonical Top-K và341supported-head reliability panels đã hoàn tất;159tests passed. Blocker thật/reference independence chưa xác minh giữ riêng, không tuyên bố blind/zero-leakage/deployment certification.

Đây là evidence bổ sung sau baseline `cc4e331`, không ghi đè sourcefreeze1499files hoặc930legacy metric outputs. Các progress/snapshot cũ phía dưới phản ánh thời điểm trước publication cuối. Bản chính thức dùng `inventory_final/`, `classification_selective_v2/`, `stratified_full/`, `reliability_figures_final/`, `independent_numerical_final/`.

## Bằng chứng đã materialize

- `checkpoint_replays/`: **50** experiment frozenCUDA, full DEV_CAL và DEV_SEARCH; mỗi thư mục có rawlogits, nativeprobabilities, exacteventmetadata, TRAINnormalizers và `CHECKPOINT_INFERENCE_RECEIPT.json`.
- `teacher_replays/`: **9** frozenbinary specialists, nativeFP16autocast; SEARCH9/9scoreparitydelta0. Các headS/O/I độc lập, không phải3classsimplex.
- `waveform_full/`: full2699utterance WAVreplay; features2699/2699delta0 và14students×CAL/SEARCH28/28scoreparitydelta0. `WAVEFORM_TRACE.jsonl` và aggregatefeatures/scores đượcversioned.5398per-utterancecachefiles giữlocal, hashindexversioned, khôngcommitcachetrùnglặp~190MB.
- `reference_lineage/`: toàn2699authorizedutterances; canonicalreferenceannotation-derived, orthographicmodekhôngforcejoinvàabstainOOV.
- `source_validity/` và`checkpoint_source_chain/`: **38** nativeoldinferencegoldmanner-gatedcases; **2** cleanboundedcallpaths,1candidate,67unverified; kết luậnsourceboundSHA256. Fallback`target_m=None` là inference mode mới, không giảhistoricalparity.
- `QUANTIZED_RUNTIME_BLOCKERS.json`:5native dynamicINT8checkpointfiles, dynamiclinearCPUdispatchavailable/CUDAdispatchabsent; khôngCPUforward/dequantize.
- `adapter_sources/<sha256>/`: exactexecutioncodebytes khớp mỗi adapterSHA củareceipt, kể cảadapterđãmởrộngsau đó.
- `CLOSURE_RECEIPT_AUDIT.json`: all100NN+18teacher scoreoutputs/hash/bounds/exactIDs **PASS**, 1490originalsourcehashesverified,9protected/replicationpathsintentionallynotopened.
- `confidence_hardened/`: calibrationprovenancehardening củastudentconfidenceanalyses; khôngsửainitialoutput.
- `diagnostics/`, `teacher_replays_fp32_diagnostic/`: lỗiintermediatevàpartialtimeoutđượcgiữ, khôngđượcđếmbenchmark hoặcdiscovery nhưmôhìnhđãhoànthành.

## Phân biệt phạm vi khoa học

`CHECKPOINT_INFERENCE_REPLAY` không đồng nghĩa model đã có independent intendedreference hoặc blindtest. DEV_SEARCH development-exposed. EXP74VTLN dùng unsupervisedaudioadaptation TRAIN/CAL/SEARCH; M68normalizeddurationwindowquirk giữnativevàkhai báo; EXP105khôngđọc corpusngoài đểreplay. Xem`coordination/EVAL_V3_EXECUTION_INCIDENTS.md` vềsyntheticCPUtestvi phạmdochild (đãloạikhỏiacceptance), khôngtuyênbốtoànexecutionchưa từngcóCPUlearnedforward.

## Tái kiểm tra không inference

```bash
PYTHONPATH=. .venv/bin/python -m research.evaluation_v3.closure_receipt_audit \
  --root "$PWD" --output /tmp/eval-v3-closure-receipt-audit.json
PYTHONPATH=. .venv/bin/python -m pytest -q research/evaluation_v3 tools/test_eval_v3_safety.py
```

Không gọi mọiadapter`--experiments all` vàooutputđãcó: fresh-outputguard cấmoverwrite. Đểreproducefrozeninference, dùngfreshoutputvàexactID/config củareceipt; runners tựgiữcooperativeGPUlease, CUDA-only, khôngtraining/network. Điểmdừngchophầnkhôngthểphụchồi phải làartifact/backendblockercụthể, khôngphải 'adapterđa dạng'.
