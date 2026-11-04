# P-SUP recovery and executed benchmark — reproduction report

## Actual state

**Four configurations evaluated, O and genuine Z reproduced; full requested control set remains PARTIAL because the frozen FIT-only canonical prior artifact is unavailable.** No checkpoint/dataset access blocker prevented inference. No training, architecture modification, backward pass, optimizer update, new experiment, literature review or sealed TEST access occurred.

Authoritative final result stratum: [`final-v2/BENCHMARK_COMPARISON.md`](final-v2/BENCHMARK_COMPARISON.md). Earlier initial/V2 agent attempts are historical, not interchangeable with this stratum. `recovered-final-v1-20261104` terminated during CAL feature extraction and contains only preliminary mapping; it is not a completed inference result. Its exact termination cause was not established. The new V2 invocation below completed with exit 0.

## Identities and environment

- Coordinator execution HEAD: `ea16b83ca906e6bbd867296ed702bdf57e677feb`, `review/psup-recovery-benchmark-v1`, dirty/untracked final runner explicitly recorded. The subsequent reviewable commit packages these exact source bytes; it is not falsely presented as the earlier execution HEAD.
- Exact inference file SHA-256: `f19d2e1f2d62b1646979a06c9bfe7b0ddb47c689ca01981827d2ba541ef35d40`. Source text is privately preserved in `inference_source.json`; imported inference source hashes are in `final-v2/RUN_MANIFEST.json`.
- Original training SHA: `dd2dd8953fc5d946c0b88d72e915be53e7cd0d9f`. The head, original frozen encoder/trainer, runtime serializer and label implementation were byte-compared against this commit and are unchanged.
- Backbone: `facebook/wav2vec2-base-960h@22aad52d435eb6dbaf354bdad9b0da84ce7d6156`; safetensors SHA-256 `8aa76ab2243c81747a1f832954586bc566090c83a0ac167df6f31f0fa917d74a` (377,607,901 bytes).
- Dataset: authorized train/val-only ZIP SHA-256 `b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9`; split SHA-256 `7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b`.
- Contract SHA-256 `4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2`. Prompt and CMUDict hashes verified before inference; strict OOV and first dictionary pronunciation; original deterministic `sub_first` offline mapping.
- CPU inference on 12th Gen Intel Core i5-12450H (12 logical CPUs; `lscpu` verified), FP32, batch 1, four threads across every arm/endpoint/protocol/control; original encoder features reused exactly. Torch/NumPy seed 1337. `.venv-cuda` is only the interpreter directory name; this invocation did not move models/audio to CUDA.
- Python 3.11.17; PyTorch 2.14.1 (+cu130 build), Transformers 5.20.2, NumPy 2.4.6, SciPy 1.17.1, SoundFile 0.14.0, huggingface-hub 2.3.0. Platform: Linux 7.3.1-1-cachyos x86_64, glibc2.44. Versioned run manifest records package versions; changing kernels/versions can change full-precision AP ranks slightly.
- Elapsed inference plus offline gold joins: 162.979087148 seconds; excludes subsequent recomputation, validation and upload. Do not interpret this as a per-utterance service latency or total research-process time.

The pretrained loader logs unexpected `lm_head` and missing `masked_spec_embed`: the original encoder loads a bare Wav2Vec2Model from a CTC checkpoint. The removed upstream LM head is not the trained compact head. The mask embedding is inactive in frozen eval inference. Loader warnings are retained; no training is requested or performed. Parameter fingerprints before/after are equal.

## Durable private artifact inventory

Approved private HF repository: `minkhoaa/langfens-rq1-cross-fold-checkpoints`.

- Recovered weights/summaries/histories: revision `4aa9f2a68f0709889eacacf170b6446d3c45c73d`, prefixes `psup/t_op/` and `psup/b_control/`; exact local/remote paths, sizes and hashes in `final-v2/RECOVERY_MANIFEST.json`. All six checkpoint payloads reverified in `final-v2/CHECKPOINT_VERIFICATION.json`.
- **Authoritative final raw benchmark:** revision `7a9eae3d7e7ae4709f03e3766c2310991e1f314a`, prefix `benchmarks/psup-v1/recovered-final-v2-20261104`; **120 files / 308,086,566 bytes**, all remotely read back and SHA-256 verified. See `FINAL_PRIVATE_STORAGE_RECEIPT.json`. No visibility setting changed.
- Local authoritative raw directory: `/home/khoa/.cache/langfens/psup-benchmark-v1/recovered-final-v2-20261104`.
- Per configuration, each `protocol_o/` and `protocol_z/` has matched `runtime_predictions.jsonl`, `offline_token_evaluation.jsonl`, utterance runtime and decoded/reference records. `calibration/` contains actual CAL predictions/joins, and `mismatch/` contains independent mismatched-audio predictions/joins. Generated full-precision metrics are in private `recomputed/<configuration>/<protocol>/metrics.json`; permitted lightweight copies are in Git `final-v2/<configuration>/<protocol>/metrics.json`.
- `RUN_MANIFEST.json` hashes inference-stage files; `FINAL_PRIVATE_STORAGE_RECEIPT.json` additionally binds the later recomputed metrics and `EXECUTION.log`. The manifest intentionally does not self-hash or claim to hash files generated after it.
- Original P-SUP, E0/P-SCORE, initial fresh inference and recovered logs remain at their prior immutable revisions in the corresponding storage receipts. Original raw bytes were not overwritten.

## Executed commands

Run from the repository root. Use new output filenames/directories for EVERY invocation; tools refuse overwrites. Existing private HF authentication and the unchanged authorized dataset ZIP are prerequisites. Do not print credentials. Do not run training commands found in historical logs.

```sh
export PYTHONPATH=.:src
PY=.venv-cuda/bin/python
B=manifests/RQ1-P-SUP-BENCHMARK-V1
RAW=/home/khoa/.cache/langfens/psup-benchmark-v1/recovered-final-v2-20261104

# Safe selected tests only: no model fitting/backward/optimizer calls.
$PY -m pytest -q tests/test_run_psup_recovered_benchmark_v1.py \
  tests/test_recompute_psup_benchmark_v1.py tests/test_verify_psup_archival_v1.py \
  tests/test_persist_psup_benchmark_v1.py
# Executed: 25 passed. Four safe T-OP recovery tests are separately recorded.

$PY audit_scripts/verify_recovered_psup_checkpoints_v1.py \
  --benchmark-dir "$B" --output "$B/final-v2/CHECKPOINT_VERIFICATION.json" \
  --recovery-manifest "$B/final-v2/RECOVERY_MANIFEST.json"
# Executed: all six payloads, optimizer counters, hashes, objective/data/source identities verified.

# Actual successful inference command (do not rerun with the same existing RAW path):
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $PY -u \
  audit_scripts/run_psup_recovered_benchmark_v1.py infer \
  --output-root "$RAW" --threads 4

# Independent token auditor; no model/audio loading:
$PY audit_scripts/recompute_psup_benchmark_v1.py --raw-dir "$RAW" \
  --split-manifest manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json \
  --output "$B/FINAL_RUN_INDEPENDENT_RECOMPUTATION.json"
# Executed: all eight cells, zero validation issues; exact runtime/offline score joins.

# Second complete recomputation including CAL, PER, controls, speakers and WFC:
$PY audit_scripts/run_psup_recovered_benchmark_v1.py recompute \
  --input-root "$RAW" \
  --output-root /home/khoa/.cache/langfens/psup-benchmark-v1/recovered-final-v2-independent-recompute-20261104
# Executed: SUMMARY objects exactly equal; all AP/cohort hashes agree with independent auditor.

# Executed private persistence; remote prefix now exists, so use a new version for future uploads:
$PY audit_scripts/persist_psup_benchmark_v1.py --source-dir "$RAW" \
  --remote-prefix benchmarks/psup-v1/recovered-final-v2-20261104 \
  --receipt "$B/FINAL_PRIVATE_STORAGE_RECEIPT.json"
```

The successful execution also used `set -o pipefail` and `tee` to the sibling `.execution.log`; that log was copied into `RAW/EXECUTION.log` before persistence. Bare repository-wide pytest was intentionally not executed: inherited tests include local backward/optimizer operations prohibited by the project. The selected green suite is not represented as a full-suite PASS.

## Reproduce metrics without model/data access

Authenticate to the existing private repository, download only the final benchmark prefix at the immutable revision and verify all receipt hashes before recomputation:

```python
import hashlib, json
from pathlib import Path
from huggingface_hub import hf_hub_download
b = Path('manifests/RQ1-P-SUP-BENCHMARK-V1')
r = json.loads((b/'FINAL_PRIVATE_STORAGE_RECEIPT.json').read_text())
dest = Path.home()/'.cache/langfens/psup-benchmark-v1/readback-reproduction-new'
dest.mkdir(parents=True, exist_ok=False)
for item in r['files']:
    cached = Path(hf_hub_download(r['repository'], filename=r['prefix']+'/'+item['path'], revision=r['revision']))
    content = cached.read_bytes()
    assert hashlib.sha256(content).hexdigest() == item['sha256']
    target = dest/item['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as f:
        f.write(content)
```

Then run the two model-free commands above with that input and new output paths. Runtime raw scores retain original FP32 values serialized without decimal rounding. Every gold key, unmatched row and extra prompt row is retained. No checkpoint is necessary for model-free metric reproduction.

For full inference reproduction, recover the ten required artifacts using the pinned `remote_repository`, `remote_revision` and `remote_path` fields in the recovery manifest and verify SHA-256 before loading. `local_path` fields are machine-specific: in an isolated reproduction checkout, update only these local path locators to the verified recovered files (never the expected hashes). Supply the unchanged archive via `--zip-path`; stage the pinned backbone in the ordinary HF cache. Keep the same inference-source checksum and imported-source hashes. No dataset re-upload, sealed TEST download or remote GPU provisioning is needed for this CPU FP32 stratum.

## Acceptance and remaining restart point

1. Required recovered checkpoints: verified; actual updates fixed/last 3600 each, best 2700/2250, zero historical skips. Fixed embedded history3150 snapshot qualification retained.
2. Four configurations and both protocols: executed on exactly the same 300 NJS/RRBI VAL IDs. O and Z eligible populations separately constant across all four cells; Z eligible-key digest `172a4eaed698056e6929b104c2f336fa39a5943c5c632fa3501b1b38331ba783` also equals E0/SF-S/SF-SD/GOP-SA archived cohorts.
3. Saved full-precision prediction recomputation: executed; independent auditor zero issues, second complete metric objects equal, CTC decodings/references retained.
4. Valid comparisons/reconciliation: explicit in final comparison. No B-Control subtype invention or mixed-population AP ranking. Historical E0 CAL exposure limitation applies to E0, not to P-SUP initialization: P-SUP uses the pinned original pretrained backbone and fresh compact heads, not E0 warm-start.
5. Raw artifact durability: executed with private SHA-256 readback, immutable revision above.
6. **Remaining unavailable measurement:** frozen FIT1799 canonical-only prior lookup/control. Do not fit a replacement. Safe continuation is to recover an already frozen, authorized prior artifact with provenance, or ask the operator to waive/version this control requirement. No new training or TEST authorization is implied. Stop here rather than falsely declare the entire requested control matrix complete.

Parallel Gemini recovery rechecks and benchmark review are retained under `audit/GEMINI_*`. Their reports are evidence, not authority; coordinator corrections are in `audit/FINAL_REVIEW_RECONCILIATION.md` (including an incorrect Agent A environment version and review numeric/storage/initialization statements). No push, merge or publication is performed.
