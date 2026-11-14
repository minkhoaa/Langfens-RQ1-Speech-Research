# Closure EVAL-V3 — 50 checkpoint experiments đã thực chạy, chưa chốt toàn bộ benchmark

## Evidence mới đã kiểm tra

- **50** experiment legacy float, mỗi experiment có frozen CUDA inference trên đủ DEV_CAL 300 utterances /10043 targets/10343 gaps và DEV_SEARCH599 /19755/20354. Hai head checkpoints của EXP04/06 được tính là một experiment, không nhầm số weight files với số mô hình.
- **9** teacher checkpoints đã chạy CAL/SEARCH bằng exact nativeCUDAFP16autocast; SEARCH **9/9 native score delta0** so với frozen source ledgers. Không trộn các teacher binary probabilities thành simplex.
- Tất cả100NNsplit outputs và18teacher outputs có outputSHA256 đúng receipt, eventID unique/full-population đúng manifest, finite probability bounds; NNtargetsimplex đúng. Command:
  `PYTHONPATH=. .venv/bin/python -m research.evaluation_v3.closure_receipt_audit --root "$PWD" --output experiments/evaluation_v3/closure/CLOSURE_RECEIPT_AUDIT.json`
- Receipt audit **PASS**:50NN/9teachers; **1490** sourcefreeze files hash khớp;9path có dấu protected/replication bị chủ động bỏ qua không mở. Không giả1499file đều đã đọc lại.
- Source binding của adapter chạy thực được snapshot ở `closure/adapter_sources/<sha256>/<module>.py`. Sửa adapter sau một increment không làm mất sourcebytes thực dùng cho receipt cũ.
- Test suite hiện tại: **119 passed /6 warnings**, log `/tmp/eval-v3-closure-tests.log`. Warnings từ native feature helper mean-of-empty-slice; full-cohort arrays được fail-closed finite check, không thay native feature behavior để che lỗi.

## Phạm vi inference mới

EXP04/06/08/10/13/17/21/22/24/26/28/31/34/38/42/44/48/49/53/54/55/56/59/61/62/63/65/68/69/71/74/75/77/81/82/83/85/87/91/92/94/95/97/101/102/103/104/105/108/109.

- EXP105 không cần truy cập corpus ngoài để inference: nativecode fit normalization từL2TRAIN trước external training ingestion. Chỉ load saved jccmweights +M94architecture +authorizedL2features; không gọi ingest/training/network.
- EXP75 constructor prompt63 vì dialect embedding16 nội bộ; actual intended prompt47, dialect index từspeakerID +sourcefixedmapping. Không dùng gold phonetic label.
- Diffusion59/65/75/77/82 dùng exact native `forward_eval_ddim(ddim_steps=4)` frozenweights. Không gọi forward_train/backward.
- Native target_m-dependent heads chạy fallback `target_m=None` (**NONORACLE_NATIVE_HEAD_AVERAGE**), không gọi đó là historical score parity. Sourceaudit38nativegold-conditioned cases được giữ cùng sourcehash evidence, không suy sạch từbaseX.
- M68pyramid sử dụng normalizeddim48 làm duration theo source; giữ đúngquirk và khai báo. Không thay bằng gold timing. M74 unsupervised VTLNspeakerstats dùng authorized TRAIN/CAL/SEARCHaudio, ghi transductiveaudio adaptation riêng; không gọi làTRAIN-onlysignal.

## Blocker có chứng cứ

**5 dynamically-quantized checkpoint files**: EXP33/47/66(two)/96. Source và installedPyTorch dispatch metadata chứng minh `quantized::linear_dynamic` /`linear_dynamic_fp16` cóCPU nhưng khôngCUDA. Không được chạyCPU neural, không dequantize thànhmodelkhác. `closure/QUANTIZED_RUNTIME_BLOCKERS.json` ghi actualsafecheckpointmetadata/sourcehash, không forward.

OriginalF4 LR coefficients thiếu vẫn là blocker historical-replay; không fit mới để giả hồi phục lịch sử. Các old cohort/event mismatches không ép remap.

## Công việc vẫn đang làm

- Luna-only lane A: thống nhất consumer EVAL-V3 cho recovered scores, CAL-only calibrators/operating policies/reliability/curves/robustness, giữrawnative score AP và canonical eventIDs. Parentreview đã yêu cầu sửa draftlogit-vsprobabilityranking, canonicaldatasetID, calibrationprojection vàsecondscorer trongteacherpolicyrows; chưa tích hợp draft chưa đạt.
- Luna-only lane B: closure mapping toàn bộ inventory290variants, status thực qua50receipts, administrative/sourceblockers exact không dùngnameheuristic. EXP24 đãfixkhôngcònNOT_A_MODEL.
- Luna-only lane C: independentreview adapters/helper/sourcecontracts/selectedarithmetic. Không thêmNNforward.
- Bổ sung kiểm thử vàreports/leaderboard/versionedoutputs/acceptance, scopedcommit khi đủ bằng chứng. HEAD vẫncc4e331, không push, Phase3B vẫnstopped.

## Incidents công khai

Xem `coordination/EVAL_V3_EXECUTION_INCIDENTS.md`: childEARLYsyntheticCPUforward vi phạm đã loại khỏiacceptance vàthaymetadata/preprocessingtests cóguard. ParentCUDAreplays vẫn tách biệt. UPPER65 timeout đầu đểlạipartialCAL giữdiagnostic; parentkhôngrerunwholecompletedfamilies. Một lầnCAL65 cần chạy lại để córeceipt hoàn chỉnh; subsequent65/68completed.69preprocessingtimeout do parseCMU mỗiutterance, sửa cacheimmutablelexicon oncepersplit; không đổi featuremath hayweights.
