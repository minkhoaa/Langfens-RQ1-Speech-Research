

## Additive correction — exact historical digest claims and reproducible readback

This correction supplements the earlier prose in this report without changing the historical `manifests/RQ1-TRN-04/PROCESS_CHECKPOINT.md`. Its eight endpoint checksum strings are each exactly 61 hexadecimal characters. Six are malformed/conflicting with the actual full SHA-256 before the displayed string ends; only T2's two strings are incomplete prefixes of the actual digests. The exact original strings, classifications and resolved digests are recorded in `ARTIFACT_PROVENANCE_LEDGER.json` under `historical_checkpoint_claims` and on each artifact. Do not treat the six mismatches as harmless truncations.

| Arm / endpoint | Exact historical PROCESS_CHECKPOINT.md string (61 chars) | Verified actual SHA-256 | Relation |
|---|---|---|---|
| C0 fixed | `ca4cff18ff38729ae553483017cf70678d975db353bb895e6f6a7d5fa21c4` | `ca4cff18ff38729a221fbeaba459f85afad23b595be41d7100b513b7917a2338` | conflicting/malformed |
| C0 best | `0b7092d44bae013fa89369e96f1fa2c263795ba429b6f849ff0dc4d830b56` | `0b7092d44bae013fb971ce64130c173f14acb45bc3f46f99c429880213a4571c` | conflicting/malformed |
| T1 fixed | `0d3cdad4a0ab786b51eb5825a071d227dddb5eb8d197607d727b1ee6c7647` | `0d3cdad4a0ab786bcfc26e4d05c2353ddc0d59e117577574ff543bfdfe663589` | conflicting/malformed |
| T1 best | `88a39b3617219db3f28cf67484df463aa1f40212f453c072c4161bb740ca3` | `88a39b3617219db313340fba9db0de082f5286908234db43a9818afff5cdf3ee` | conflicting/malformed |
| T2 fixed | `e8b00002aef9ee1587990a69162d5a5438b092e02c83d5654ae5aea0e1ec9` | `e8b00002aef9ee1587990a69162d5a5438b092e02c83d5654ae5aea0e1ec9c13` | matching but incomplete prefix |
| T2 best | `98d3fc1ecf6a5539d211028e32c9cbd2d5668b3942ad4d8fcae8226a99927` | `98d3fc1ecf6a5539d211028e32c9cbd2d5668b3942ad4d8fcae8226a99927fb1` | matching but incomplete prefix |
| T3 fixed | `ca32da07eac2094676451eebc95350c33748be7c4c1a719c8366ce2b451ec` | `ca32da07eac20946af125539ca1ef7ca3835a35ce05e1ec4897d3a790e90eeb9` | conflicting/malformed |
| T3 best | `1feb311ec9f88569c735d487cb1f912e75e1b6f00db1bbad478bf11979ad7` | `1feb311ec9f88569915fa43aefaa923e24c55f7fadb77a821a2b2c2c8dcf6107` | conflicting/malformed |

### Remote readback count and identities

Correction to the earlier ledger sentence: exactly **12 payloads total** were read back at pinned revision `af6ac3620fb78873ff213c064dffe71fb8acadd4`: 8 model endpoint files (fixed and best for four arms) plus 4 `training_summary.json` files. Each of their remote-readback SHA-256 values and byte counts is stored with the individual artifact entry in the ledger. Readback was authenticated, private-repository-checked first, revision-pinned, and read-only.

The command below is the reproducible verification procedure used. Run from the repository root with the parent checkout `.venv` available and authenticated Hugging Face cache configured. It prints all 12 remote paths, byte counts and digests, asserts each against both local bytes and ledger receipts, asserts the repository is private and at the recorded revision, and performs no writes to the remote repository or checkpoints.

```bash
cd /home/khoa/Projects/KLTN/trn04r-c
HF_HUB_DISABLE_TELEMETRY=1 /home/khoa/Projects/KLTN/Dataset/.venv/bin/python - <<'PY'
import hashlib, json, os
from huggingface_hub import HfApi, hf_hub_download
repo = "minkhoaa/langfens-rq1-cross-fold-checkpoints"
revision = "af6ac3620fb78873ff213c064dffe71fb8acadd4"
info = HfApi().repo_info(repo, repo_type="model")
assert info.private is True and info.sha == revision
ledger = json.load(open("manifests/RQ1-TRN-04R/ARTIFACT_PROVENANCE_LEDGER.json"))
count = 0
for arm, entry in ledger["arms"].items():
    for name, item in entry["artifacts"].items():
        remote = hf_hub_download(repo_id=repo, filename=item["remote_path"],
                                revision=revision, repo_type="model")
        remote_sha = hashlib.sha256(open(remote, "rb").read()).hexdigest()
        remote_bytes = os.path.getsize(remote)
        local_sha = hashlib.sha256(open(item["path"], "rb").read()).hexdigest()
        assert remote_sha == local_sha == item["sha256"]
        assert remote_bytes == item["bytes"]
        print(f"{arm}/{name}\t{remote_bytes}\t{remote_sha}")
        count += 1
assert count == 12
print(f"PASS private={info.private} revision={revision} readback_files={count}")
PY
```

### Explicit speaker and constants provenance correction

Frozen partition/speaker documentation identifies Fold-1 VAL speaker **NJS as Spanish L1, female** and **RRBI as Hindi L1, male** (`manifests/RQ1-STD-01/SPEAKER_SPLIT_PROTOCOL.md`; `manifests/RQ1-SPLIT-01/SPLIT_VALIDATION_REPORT.md`, sections 3 and 4). This is metadata-only; no audio or labels were inspected.

`src/langfens_prep/trn04/features.py` defines frozen original constants `FIT_LOG_ENERGY_MEAN = -6.047091` and `FIT_LOG_ENERGY_STD = 2.544367`; code comments and `FACTORIAL_DESIGN_AND_CONTROLS.md` state these were derived from 1,799 FIT utterances. The reviewed receipts do not include a reproducible calculation artifact or independent empirical provenance for these values. Therefore describe them as frozen original constants whose FIT empirical derivation is asserted by source documentation but not independently verified here. This does not alter constants or evaluation code.

### Source revision and execution state

The reviewed existing checkpoint/summary/run-manifest and persisted-receipt evidence does not establish an immutable historical training source commit. This provenance gap remains unresolved; current review HEAD is not a substitute. No inference or training was performed by Workstream C. Corrected inference/runtime remains pending coordinator execution.
