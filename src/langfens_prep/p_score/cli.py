"""Inference-only cache CLI. Default is synthetic; no audio/weight loader exists."""

import argparse
import concurrent.futures
from datetime import datetime, timezone
import platform
from pathlib import Path
import re
import subprocess
import sys

from .cache import (CONFIG, CONTRACT_SHA256, PHONE_MAP, authorize_cache, decoded_record,
                    json_bytes, parse_json, sha256, validate_cache)
from .ctc import GRAPH_REVISION
from .runtime import PROTOCOL, SCHEMA, runtime_key, score_record, validate_bundle

BASE = "0d268ff8c444e16fd76aca2f4ad6ece966908edc"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _score_worker(args: tuple[dict, str, str]) -> tuple[list[dict], dict]:
    encoded, scorer_run, scorer = args
    record = decoded_record(encoded)
    return score_record(record, scorer_run, scorer)


def run(cache_path: Path, output: Path, config_path: Path, run_id: str,
        *, mode: str = "synthetic", authorization_path: Path | None = None,
        authorization_sha256: str | None = None, repo: Path | None = None,
        workers: int = 1) -> dict:
    repo = repo or Path(__file__).resolve().parents[3]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id must be a safe nonempty identifier")
    if output.exists():
        raise ValueError("output exists; new run/output required (no overwrites)")
    contract = repo / "manifests/RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md"
    if sha256(contract.read_bytes()) != CONTRACT_SHA256:
        raise ValueError("frozen V1 contract hash mismatch")
    config_data = config_path.read_bytes()
    if parse_json(config_data) != CONFIG:
        raise ValueError("unregistered config/settings; only raw primary implemented")
    # Authenticate registration bytes before opening the external emission cache.
    authorization = None
    if mode == "frozen-cache":
        if authorization_path is None or authorization_sha256 is None:
            raise ValueError("frozen-cache requires separately approved SHA-pinned authorization")
        data = authorization_path.read_bytes()
        if sha256(data) != authorization_sha256:
            raise ValueError("authorization hash mismatch")
        authorization = parse_json(data)
    elif mode != "synthetic" or authorization_path is not None or authorization_sha256 is not None:
        raise ValueError("invalid mode/authorization combination")
    cache_data = cache_path.read_bytes()
    cache_hash = sha256(cache_data)
    cache = validate_cache(parse_json(cache_data), mode)
    if authorization is not None:
        authorize_cache(cache, cache_hash, authorization,
                        repo / "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json")
    started = datetime.now(timezone.utc).isoformat()
    bundles = {}
    for scorer in CONFIG["scorers"]:
        print(f"[SCORER] Starting {scorer} scoring ({len(cache['records'])} records, workers={workers})...", flush=True)
        scorer_run = run_id + "_" + scorer
        rows, utterances, expected = [], [], {}
        if workers > 1 and len(cache["records"]) > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                items = [(rec, scorer_run, scorer) for rec in cache["records"]]
                results = list(executor.map(_score_worker, items))
        else:
            results = [_score_worker((rec, scorer_run, scorer)) for rec in cache["records"]]
        for (scored, utterance), encoded in zip(results, cache["records"]):
            record = decoded_record(encoded)
            utterance.update(emission_cache_sha256=cache_hash,
                             emission_shape=[len(record["log_emissions"]), 40],
                             source_speaker_id=record["source_speaker_id"],
                             source_utterance_id=record["source_utterance_id"])
            rows.extend(scored)
            utterances.append(utterance)
            for i, phone in enumerate(record["prompt_phone_ids"]):
                key = (scorer_run, PROTOCOL, record["condition"], record["speaker_id"], record["utterance_id"], i)
                expected[key] = (phone, record["word_indices"][i])
        counts = validate_bundle(rows, expected)
        if len(utterances) != len(cache["records"]):
            raise ValueError("incomplete utterance/failure accounting")
        bundles[scorer] = (rows, utterances, counts)
        print(f"[SCORER] Completed {scorer} scoring (statuses: {counts}).", flush=True)
    source_hashes = {str(p.relative_to(repo)): sha256(p.read_bytes())
                     for p in sorted((repo / "src/langfens_prep/p_score").glob("*.py"))}
    source_hashes["audit_scripts/p_score_infer.py"] = sha256((repo / "audit_scripts/p_score_infer.py").read_bytes())
    allowed_ids = list(dict.fromkeys(r["speaker_id"] + "/" + r["utterance_id"] for r in cache["records"]))
    prompt_records = [{k: r[k] for k in ("speaker_id", "utterance_id", "condition", "prompt_phone_ids", "word_indices")}
                      for r in cache["records"]]
    status = _git(repo, "status", "--porcelain")
    manifest = {
        "schema_version": "p-score-run-manifest-v1", "runtime_schema": SCHEMA,
        "contract_version": "EVALUATION_CONTRACT_V1", "contract_sha256": CONTRACT_SHA256,
        "arm": "P-SCORE", "run_id": run_id, "protocol": PROTOCOL, "mode": mode,
        "graph_revision": GRAPH_REVISION, "scorers": CONFIG["scorers"],
        "normalization": "raw", "occupancy_normalization": "auxiliary_exact_posterior_floor_1_not_paper_Eq13",
        "score_orientation": "higher_is_error", "probabilities_supported": False,
        "score_semantics": "raw uncalibrated compatibility; no subtype confidence or scientific performance claim",
        "git_base": BASE, "git_tip": _git(repo, "rev-parse", "HEAD"),
        "git_branch": _git(repo, "branch", "--show-current"), "git_dirty": bool(status),
        "git_status": status, "source_sha256": source_hashes, "config_sha256": sha256(config_data),
        "package_sha256": None, "package_status": "no package loaded; external cache only",
        "label_map_sha256": sha256(json_bytes(PHONE_MAP)),
        "prompt_sha256": sha256(json_bytes(prompt_records)),
        "g2p_sha256": cache["provenance"]["prompt_frontend_sha256"],
        "weight_sha256": cache["provenance"]["model_sha256"],
        "weights_loaded": False, "cache_sha256": cache_hash, "cache_provenance": cache["provenance"],
        "allowed_id_sha256": sha256(("\n".join(allowed_ids) + "\n").encode()),
        "fit_speaker_ids": [] if authorization is None else authorization["fit_speakers"],
        "selection_speaker_ids": [] if authorization is None else authorization["selection_speakers"],
        "calibration_speaker_ids": [] if authorization is None else authorization["calibration_speakers"],
        "evaluation_speaker_ids": sorted({r["speaker_id"] for r in cache["records"]}),
        "calibration_identity": None, "calibration_limitation": "E0 acoustic TRAIN overlaps historical CAL/SELECT; no fitting here",
        "seed": 1337, "rng": "not used; deterministic exhaustive sums and fixed ties",
        "hardware": {"platform": platform.platform(), "processor": platform.processor(), "device": "CPU"},
        "precision": "Python float64 log-space arithmetic; FP32 cache declaration; no acoustic execution",
        "determinism": "stable record order, sorted alternative strings; deterministic tie rules",
        "dependencies": {"python": platform.python_version(), "implementation": platform.python_implementation()},
        "endpoint": "frozen emission scoring only", "training": "NOT_STARTED",
        "successful_optimizer_updates": 0, "skipped_updates": 0, "optimizer_attempts": 0,
        "new_model_artifact": None,
        "authorization_id": "operator_synthetic_implementation_only" if authorization is None else authorization["authorization_id"],
        "authorization_sha256": authorization_sha256,
        "attempted_settings": [CONFIG], "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(), "output_sha256": {},
        "coverage": {s: {"utterances": len(u), "prompt_positions": len(r), "statuses": c,
                          "utterance_statuses": {k: sum(x["status"] == k for x in u) for k in ("ok", "abstain", "failure")}}
                     for s, (r, u, c) in bundles.items()},
    }
    # Build/validate all bytes before exclusive output creation. No partial scoring
    # silently dropped; hard I/O interruption leaves an incomplete, non-certified run.
    files = {}
    for scorer, (rows, utterances, _) in bundles.items():
        failures = [{"runtime_key": list(runtime_key(r)), "status": r["status"],
                     "reason": r["abstention_reason"]} for r in rows if r["status"] != "ok"]
        files[scorer + "/runtime_predictions.jsonl"] = b"".join(json_bytes(r) for r in rows)
        files[scorer + "/utterance_predictions.jsonl"] = b"".join(json_bytes(u) for u in utterances)
        files[scorer + "/failures.jsonl"] = b"".join(json_bytes(f) for f in failures)
    manifest["output_sha256"] = {name: sha256(data) for name, data in files.items()}
    files["run_manifest.json"] = json_bytes(manifest)
    files["SHA256SUMS"] = "".join(f"{sha256(data)}  {name}\n" for name, data in sorted(files.items())).encode()
    output.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        path = output / name
        path.parent.mkdir(exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P-SCORE forward-only synthetic/frozen-E0 emission-cache scorer. No audio or weight loader; real-cache use requires separate authorization.")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new private output directory; never overwrite")
    parser.add_argument("--config", type=Path, default=Path("configs/rq1_p_score_raw.json"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("synthetic", "frozen-cache"), default="synthetic")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--workers", type=int, default=1, help="worker processes for record scoring")
    args = parser.parse_args(argv)
    try:
        manifest = run(args.cache, args.output, args.config, args.run_id, mode=args.mode,
                       authorization_path=args.authorization, authorization_sha256=args.authorization_sha256,
                       workers=args.workers)
    except (ValueError, TypeError, KeyError, OSError, StopIteration) as exc:
        print(json_bytes({"status": "failure", "reason": str(exc), "output_certified": False}).decode().strip(), file=sys.stderr)
        return 2
    print(json_bytes({"status": "completed", "scope": "implementation_only_not_scientific_PASS", "coverage": manifest["coverage"]}).decode().strip())
    return 0
