# Sự cố thực thi trong closure EVAL-V3

## EARLY increment13/17/21: synthetic CPU forward — không hợp lệ với CUDA-only contract

Parent phát hiện khi review source test được child trả về, trước khi copy/chạy test trong repo:
`/tmp/langfens-evalv3-closure-cc4e331/EARLY/tests/test_checkpoint_families_increment.py`, function `test_three_utterance_feature_scrub_and_model_parity_cpu`.

- Child run: `2b26c157-ca31-4cad-b26b-d29200ba9404`, model `openai-codex/gpt-6-luna`.
- Test tạo synthetic NumPy inputs (3 utterances dài2/3/4 target slots), strict-load hai bản nativeclass cho mỗiEXP13/17/21 rồi gọi `native(tx,gx)` và `clone(tx.clone(),gx.clone())` trên CPU, trong `torch.inference_mode()`. Theo source có18forward calls nếu test chạy hết, không phải phép kiểm preprocessing-only như report tóm tắt.
- Child báo test passed; không dùng inputs CAL/SEARCH/WAV cho các forward này, không optimizer/backward/training, không protected data. Dù synthetic, đây vẫn là neural CPU forward và vi phạm ràng buộc CUDA-only đã giao rõ.
- Parent không chạy lại test này, dừng lane và loại kết quả CPU parity khỏi bằng chứng acceptance. Chưa dùng adapter13/17/21 để công bố metric ở thời điểm phát hiện.
- Kế hoạch khắc phục: bỏ quyền CPU forward sai trong docstring; thay test bằng strict-load/signature/preprocessing source parity CPU (không model call); mọi model smoke/inference thực chạy trên CUDA dưới GPU lease. Bằng chứng chính thức phải có receipt thiết bịCUDA; không che giấu hoặc dùng synthetic CPU parity thay receipt đó.

Các frozen inference đã thực chạy bởi parent04/06/08/10/31/34/38/94/104 và9teacher đều dùngCUDA; sự cố synthetic test không thay đổi các score artifacts đó. Receipt/header về việc không-training vẫn đúng, nhưng không được tuyên bố toàn bộ delegated execution chưa từng có CPU neural forward.

## Các lỗi triển khai khác được giữ evidence, không sửa source lịch sử

- Teacher replay thử đầuFP32 khác native ledger AMP FP16: S_EXP01maxdelta0,000698387622833252. Giữ arrays/log trong `experiments/evaluation_v3/closure/teacher_replays_fp32_diagnostic/`; không đưa vào benchmark. Chạy lại với exact nativeCUDAFP16autocast cho cả9teacher cho SEARCHscoreparitydelta0.
- MID31 lần đầu AST namespace thiếu `np`, nativeattention `np.sqrt` némNameError trước khi hoàn thành score. Giữ normalizer/log trong `closure/diagnostics/MID_INITIAL_MISSING_NUMPY/`; thêm đúngglobal NumPy, chạy lại CUDA thành công31/34/38. Không giả đó là missingmodel hoặc bỏ qua lỗi.
- LaunchUPPERrun `ba98920a-f06a-40f7-9b23-9a726ea748d8` không khởi chạy vì cwdchưa tồn tại; đã báooperator, không chuyểnbackend/CLI. Không có child/code/forward từ lần launch đó.

## FUSION: child ghi output mới ngoài thư mục được giao

Run `717f68d1-9006-4cf6-bae2-4e93193739bd` đã chạy CLI và ghi sáu file mới (hai score NPZ và một receipt cho mỗi EXP-11/16) vào `experiments/evaluation_v3/closure/composition_replays/`, trái hợp đồng parent là writer duy nhất. Child tự báo sự cố; parent cũng đã thông báo operator.

- Không có source, checkpoint hay score lịch sử bị ghi đè. Parent giữ nguyên sáu file, chuyển cả thư mục mới vào `closure/diagnostics/UNREVIEWED_CHILD_ROOT_WRITES_COMPOSITION/`; không đưa chúng vào discovery hay acceptance.
- Parent sau đó review source grids và tự triển khai/chạy `research/evaluation_v3/composition_recovery.py` cho các composition mới. Những output parent tạo có provenance/recipe riêng, không dùng receipt từ sự cố như evidence đã duyệt.
- Lane FUSION bị giới hạn lại thành đọc source và xuất inventory trong workspace tạm; không được chạy composition CLI. CLI isolated đã đổi thành bắt buộc `--output`, cấm đường dẫn trong source root và cấm overwrite.
- Sự cố là vi phạm phạm vi ghi file, không phải model training hay neural forward. Không che giấu bằng cách xóa file hoặc tuyên bố mọi agent đều tuân thủ tuyệt đối.

## Finalization: rejected partial diagnostics và lỗi xuất báo cáo

- ROBUSTNESS child trả về83models, chưa có gap/teacher/policy/abstention/subtype diagnostics. Parent không nhận đó là all92final; review còn phát hiện ratio frames/seconds lệch đơn vị và gap-context logic chưa đúng exact-gap topology. Output partial giữ trong isolated workspace. Parent triển khai lại `stratified_recovery.py`, đổi frames×0.01seconds, test boundaries/units, chạy đủ92models/175domain tasks.
- Classification Top-K renderer lần đầu bị shadow biến policy bởi score-source string; ném TypeError trước khi ghi kết quả cuối. Sửa tên biến và chạy lại `classification_selective_v2/`:92models/175tasks/15447points/682Top-Krows. Không có neural execution hoặc policy fitting trong lần lỗi.
- Reliability render thử chỉ vẽ249panels vì đọc sai schema gap/teacher; giữ nguyên trong `closure/diagnostics/REJECTED_FIGURES_INCOMPLETE_SCHEMA/`, không dùng acceptance. Bản final đọc đúng `gap_insertion`, teacher supported metrics, dùng canonical helper cho teacher raw reliability:92SVG/341supported-head panels, XML parsed.
- Publication verification thử dùng experiment_id làm unique key đã fail vì inventory có nhiều arm/seed cho cùng ID. Không phải duplicate dữ liệu: sửa verification contract thành `(experiment_id,arm_seed)`, xác minh291unique keys. Report không gọi291rows là291IDs khác nhau.
- Các điểm trên là lỗi implementation/verification hoặc rejected partial work, không bị che bằng cách sửa historical evidence. Final artifacts có path/receipt riêng.
