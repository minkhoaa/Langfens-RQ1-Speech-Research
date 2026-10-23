import hashlib
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict

FROZEN_SPLIT_HASHES: Dict[str, str] = {
    "l2arctic_4fold_splits.json": "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b",
    "l2arctic_loso_vietnamese_splits.json": "7a59b63ad73c7616af2ef583760b50253c973090c4c7c76aadc46029cfe69118",
    "speechocean762_official_splits.json": "bc3f9ec97b5077402b3c67d89c857b9efe694649bef6e5d43776f07b528bad17"
}

RAW_ARCHIVE_HASHES: Dict[str, str] = {
    "l2arctic_release_v5.0.zip": "490d5a43e48b0af84f6bce8e766ad453e8cd670521d5876bb4d0741cca304fda",
    "speechocean762.tar.gz": "876c34b828c2ebcc35ba2ff1ba5947d50e6b0162fc4a4af91db0942422b3bdd5"
}

def compute_file_sha256(filepath: str, chunk_size: int = 1048576) -> str:
    """Computes SHA-256 digest of a local file in streaming chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()

def compute_canonical_payload_sha256(payload: Any) -> str:
    """Computes SHA-256 of payload serialized canonically with sorted keys in UTF-8."""
    canonical_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()

def verify_split_manifest_hashes(manifest_dir: str = "manifests/RQ1-SPLIT-01") -> Dict[str, bool]:
    """Verifies that all frozen split manifests exist and match their exact immutable SHA-256 hashes."""
    results = {}
    for filename, expected_hash in FROZEN_SPLIT_HASHES.items():
        filepath = os.path.join(manifest_dir, filename)
        if not os.path.exists(filepath):
            results[filename] = False
            continue
        actual_hash = compute_file_sha256(filepath)
        results[filename] = (actual_hash == expected_hash)
    return results

def investigate_provenance_timestamp() -> Dict[str, Any]:
    """Investigates discrepancy between manifest metadata timestamp and Google Drive upload times."""
    # Read host local timezone
    local_tz = time.tzname
    is_dst = time.daylight and time.localtime().tm_isdst > 0
    tz_offset_hours = -(time.altzone if is_dst else time.timezone) / 3600
    
    # Check filesystem mtimes of split manifests
    split_dir = "manifests/RQ1-SPLIT-01"
    files_info = {}
    for filename in FROZEN_SPLIT_HASHES.keys():
        filepath = os.path.join(split_dir, filename)
        if os.path.exists(filepath):
            st = os.stat(filepath)
            mtime_utc = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
            files_info[filename] = {
                "mtime_local": datetime.fromtimestamp(st.st_mtime).isoformat(),
                "mtime_utc": mtime_utc,
                "size_bytes": st.st_size
            }
            
    return {
        "status": "NOT_VERIFIED",
        "recorded_timestamp_utc": "2026-10-23T13:30:00Z",
        "host_timezone": f"UTC{tz_offset_hours:+03.0f}:00",
        "timezone_names": local_tz,
        "files_mtime_analysis": files_info,
        "root_cause": (
            "DEFAULT_TIMESTAMP_UTC was statically hardcoded as '2026-10-23T13:30:00Z' "
            "in audit_scripts/generate_rq1_splits.py:28 without applying host local timezone (+07:00) "
            "offset conversion. Local execution occurred at ~13:23 local time (+07:00), which corresponds "
            "to ~06:23 UTC. rclone sync to Google Drive recorded creation at ~06:26 UTC, creating a ~7-hour "
            "discrepancy against the literal UTC string in manifest metadata."
        ),
        "recommendation": (
            "Preserve frozen manifests unmodified (DO NOT REWRITE) to maintain payload digest reproducibility. "
            "Designate the historical UTC wall-clock timestamp as NOT_VERIFIED in SPLIT_PROVENANCE_TIME_ERRATUM.md."
        )
    }
