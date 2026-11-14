# Kiểm toán số học độc lập — kết quả được chấp nhận

Reviewer Luna kiểm tra trực tiếp NPZ có receipt/hash và exact event IDs; không import evaluator/policies chính, không neural forward hoặc fit lại CAL. Đây là verification, không phải scorer production thứ hai.

Population599SEARCH utterances; C17778/S1768/O209/I125, noI20229.

| Model | AP S | AP O | AP I | Macro3 AP | F1 tại CAL max-error-F1 |
|---|---:|---:|---:|---:|---:|
| EXP13 | .350209391 | .200963280 | .244673979 | .265282217 | .316576798 |
| EXP24 | .357736712 | .225091580 | .247627966 | .276818753 | .305855362 |
| EXP38 | .200322657 | .090618853 | .238641975 | .176527828 | .235468903 |
| EXP11 | .361735082 | .156441428 | .233677305 | .250617938 | .306407862 |
| EXP86 | .251552566 | .151768130 | .304869674 | .236063456 | .297606324 |

Mỗi model: sáu selected policy rows khớp AP/F1≤1e-10 và TP/FP/FN/TN tuyệt đối. Chín complete raw-AP policy rows/model giữ AP bất biến. TeacherS_EXP01: AP S .388469522,1768positive/17987nonS; class-specific và risk-budget.2 khớp counts/F1/AP; tám complete S-head rows giữ AP bất biến. Không suy ra O/I scores.

EXP11/86 direct saved arrays có hash đúng receipt; independent weighted reconstruction target/gap đều maxdelta0, metadata order đúng. Streaming PR119696rows EXP13/24S/O/I: counts,precision,recall đều khớp; gzip SHA25639a40858cb6a999e8b834ff622199068cd4aa76c8354c40d90060f2a0a646492.

Machine evidence: `independent_numerical_closure.json`, `selected_metrics.csv`. Source audit module: `research/evaluation_v3/independent_numerical_closure.py`, SHA256d21a4d7172cdba9bdf4d53b82656a4b86f07bf7f1da8e252e4692da3a010c2df. Root unit tests chỉ đổi import path khi tích hợp package; computation code giữ byte-identical với reviewer. Không gọi phạm vi này là numerical audit đủ92models. Reference lineage/development exposure giữ caveat như báo cáo chính.
