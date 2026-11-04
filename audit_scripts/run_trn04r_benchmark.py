#!/usr/bin/env python3
"""Frozen Protocol Z inference for TRN-04 checkpoints; never trains or tunes.

This command is coordinator-run only after the integrated revision is frozen.
It writes raw/private records outside the Git checkout. It does not read gold
annotations until every CAL and VAL runtime prediction has been frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langfens_prep.trn04.model import CompactFactorialOperationHead, FactorialHeadConfig
from langfens_prep.trn04.runtime import (
    ARMS, ENDPOINTS, FEATURES_ENABLED, OBJECTIVES, SEALED_SPEAKERS,
    join_gold_rows, make_head_forward, offline_recognition_row, runtime_rows, sha256_file, write_jsonl,
)

SPLIT_SHA = "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b"
ZIP_SHA = "b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9"
CONTRACT_SHA = "4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2"
BACKBONE = "facebook/wav2vec2-base-960h"
BACKBONE_REVISION = "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
ARMS_DIR = {"C0": "c0", "T1": "t1", "T2": "t2", "T3": "t3"}
BEST_UPDATES = {"C0": 2247, "T1": 897, "T2": 3600, "T3": 2247}
EXPECTED_SKIPPED_ATTEMPTS = 3
PROMPT_SHA = "87843d363bc7e2bbe849842efe16974cb6634dfe0539efb42d666aed7e17f11e"
DICTIONARY_SHA = "81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22"
SOURCE_FILES = [
    "audit_scripts/run_trn04r_benchmark.py",
    "src/langfens_prep/trn04/__init__.py", "src/langfens_prep/trn04/runtime.py",
    "src/langfens_prep/trn04/model.py", "src/langfens_prep/trn04/features.py",
    "src/langfens_prep/trn04/objectives.py", "src/langfens_prep/trn04/evaluator.py",
    "src/langfens_prep/dataset.py", "src/langfens_prep/alignment.py",
    "src/langfens_prep/g2p.py", "src/langfens_prep/phoneme_normalizer.py",
    "src/langfens_prep/textgrid_parser.py", "src/langfens_prep/audio_loader.py",
    "src/langfens_prep/p_sup/labels.py", "src/langfens_prep/p_sup/trainer.py",
]


def file_digest(path: Path) -> str:
    return sha256_file(path)


def checkpoint_digest(arm: str, endpoint: str) -> str:
    integrity = ROOT / "manifests" / "RQ1-TRN-04" / ARMS_DIR[arm] / "CHECKPOINT_INTEGRITY.json"
    record = json.loads(integrity.read_text())["fixed_budget_3600.pt" if endpoint == "fixed_budget_3600" else "best.pt"]
    return record["sha256"]


def load_head(path: Path, arm: str, endpoint: str):
    import torch
    expected = checkpoint_digest(arm, endpoint)
    actual = file_digest(path)
    if actual != expected:
        raise ValueError(f"checkpoint hash not allowlisted for {arm}/{endpoint}: {actual} != {expected}")
    # Checkpoint file hash is verified before any deserialization.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    expected_updates = 3600 if endpoint == "fixed_budget_3600" else BEST_UPDATES[arm]
    endpoint_flag = "is_fixed_budget_endpoint" if endpoint == "fixed_budget_3600" else "is_best_endpoint"
    if (metadata.get("arm") != arm or metadata.get("objective") != OBJECTIVES[arm]
            or metadata.get("contract_sha256") != CONTRACT_SHA
            or metadata.get("features_enabled") is not FEATURES_ENABLED[arm]
            or metadata.get(endpoint_flag) is not True
            or payload.get("successful_updates") != expected_updates
            or payload.get("skipped_amp_attempts_total") != EXPECTED_SKIPPED_ATTEMPTS):
        raise ValueError(f"checkpoint metadata/counter mismatch: {arm}/{endpoint}")
    head = CompactFactorialOperationHead(FactorialHeadConfig()).float().cpu().eval().requires_grad_(False)
    head.load_state_dict(payload["head"], strict=True)
    for param in head.parameters():
        if param.requires_grad:
            raise AssertionError("loaded model must be frozen")
    return head, {"path": str(path), "sha256": actual, "metadata": metadata,
                  "successful_updates": expected_updates,
                  "skipped_updates": EXPECTED_SKIPPED_ATTEMPTS}


def fingerprint(module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=ROOT / "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json")
    parser.add_argument("--zip", dest="archive", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(ARMS), action="append")
    parser.add_argument("--endpoint", choices=sorted(ENDPOINTS), action="append")
    parser.add_argument("--backbone-cache", type=Path, help="Local Hugging Face cache root; downloads are disabled")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")

    import numpy as np
    import torch
    from transformers import Wav2Vec2Model
    from langfens_prep.dataset import ZeroOracleL2ArcticDataset
    from langfens_prep.g2p import CMUDictG2P
    from langfens_prep.phoneme_normalizer import CTC_INDEX_TO_PHONE, STANDARD_39_ARPABET
    from langfens_prep.audio_loader import get_l2arctic_reader
    from langfens_prep.textgrid_parser import parse_gold_textgrid_content
    from langfens_prep.p_sup.labels import build_operation_labels
    from langfens_prep.alignment import align_canonical_sequences
    from langfens_prep.trn04.features import extract_frame_acoustic_features

    start = time.time()
    torch.set_num_threads(args.threads)
    torch.manual_seed(1337)
    np.random.seed(1337)
    split_path = args.split.resolve()
    archive = args.archive.resolve()
    contract = ROOT / "manifests/RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md"
    prompts_file = ROOT / "resources/L2-ARCTIC/prompts.json"
    dictionary_file = ROOT / "resources/CMUDict/cmudict.dict"
    source_hashes_before = {name: file_digest(ROOT / name) for name in SOURCE_FILES}
    if file_digest(prompts_file) != PROMPT_SHA or file_digest(dictionary_file) != DICTIONARY_SHA:
        raise ValueError("pinned prompt table or CMUDict checksum mismatch")
    if file_digest(split_path) != SPLIT_SHA or file_digest(contract) != CONTRACT_SHA:
        raise ValueError("frozen split or evaluation contract hash mismatch")
    if file_digest(archive) != ZIP_SHA:
        raise ValueError("authorized TRAIN/VAL archive hash mismatch")
    output_root = args.private_output.expanduser().resolve()
    if output_root.is_relative_to(ROOT.resolve()) or output_root.exists():
        raise ValueError("private output must be a new path outside the repository")
    output_root.mkdir(parents=True, mode=0o700)
    os.chmod(output_root, 0o700)
    os.umask(0o077)

    manifest_data = json.loads(split_path.read_text())
    fold = next(row for row in manifest_data["folds"] if row["fold_index"] == 1)
    ids = {"cal": [x for x in fold["train"]["utterances"] if x.split("/")[0] in {"NCC", "THV"}],
           "val": list(fold["val"]["utterances"])}
    if len(ids["cal"]) != 300 or len(ids["val"]) != 300 or any(x.split("/")[0] in SEALED_SPEAKERS for values in ids.values() for x in values):
        raise ValueError("CAL/VAL allowlist failed")
    selected_arms = args.arm or sorted(ARMS)
    selected_endpoints = args.endpoint or ["fixed_budget_3600", "best"]

    # All model files are checksum-verified and loaded only after exact byte allowlisting.
    heads = {}
    checkpoint_info = {}
    for arm in selected_arms:
        for endpoint in selected_endpoints:
            path = args.checkpoint_root / ARMS_DIR[arm] / ("fixed_budget_3600.pt" if endpoint == "fixed_budget_3600" else "best.pt")
            heads[arm, endpoint], checkpoint_info[arm, endpoint] = load_head(path, arm, endpoint)

    backbone_cache = args.backbone_cache.resolve() if args.backbone_cache else None
    backbone_kwargs = {"revision": BACKBONE_REVISION, "local_files_only": True, "cache_dir": backbone_cache}
    from transformers.utils.hub import cached_file
    backbone_weights = Path(cached_file(BACKBONE, "model.safetensors", revision=BACKBONE_REVISION,
                                        local_files_only=True, cache_dir=backbone_cache))
    backbone_sha256 = file_digest(backbone_weights)
    if backbone_sha256 != "8aa76ab2243c81747a1f832954586bc566090c83a0ac167df6f31f0fa917d74a":
        raise ValueError(f"cached backbone hash mismatch: {backbone_sha256}")
    encoder = Wav2Vec2Model.from_pretrained(BACKBONE, **backbone_kwargs).float().cpu().eval().requires_grad_(False)
    from langfens_prep.p_sup.trainer import FrozenWav2Vec2AudioEncoder
    frozen_encoder = FrozenWav2Vec2AudioEncoder(encoder).float().cpu().eval().requires_grad_(False)
    model_before = {f"{a}:{e}": fingerprint(h) for (a, e), h in heads.items()}
    backbone_before = fingerprint(frozen_encoder)

    g2p = CMUDictG2P(dict_path=dictionary_file, verify_checksum=True, oov_policy="strict")
    cache = {}
    feature_cache_receipt = {}
    for partition, split_name in (("cal", "train"), ("val", "val")):
        dataset = ZeroOracleL2ArcticDataset(str(split_path), 1, split_name, prompts_source=str(prompts_file),
                                             g2p_frontend=g2p, zip_path=str(archive))
        allow = set(ids[partition])
        dataset.utterance_entries = [entry for entry in dataset.utterance_entries if "/".join(entry) in allow]
        if ["/".join(entry) for entry in dataset.utterance_entries] != ids[partition]:
            raise ValueError(f"ordered {partition} allowlist mismatch")
        for item in dataset:
            uid = item["speaker_id"] + "/" + item["utterance_id"]
            waveform = item["audio_waveform_16k"].detach().cpu().float().contiguous()
            prompt_ids = item["canonical_phone_indices"].detach().cpu().long().contiguous()
            if not waveform.numel() or not prompt_ids.numel():
                raise ValueError(f"empty audio/prompt: {uid}")
            with torch.inference_mode():
                features, lengths = frozen_encoder(waveform.unsqueeze(0), torch.tensor([waveform.numel()], dtype=torch.long))
                acoustics = extract_frame_acoustic_features(waveform.unsqueeze(0), lengths).float().cpu()
            feature_bytes = features.detach().cpu().numpy().tobytes()
            acoustic_bytes = acoustics.detach().cpu().numpy().tobytes()
            feature_cache_receipt[uid] = {
                "feature_shape": list(features.shape), "frame_count": int(lengths[0]),
                "fp32_feature_sha256": hashlib.sha256(feature_bytes).hexdigest(),
                "fp32_acoustic_feature_sha256": hashlib.sha256(acoustic_bytes).hexdigest(),
                "fp32_waveform_sha256": hashlib.sha256(waveform.numpy().tobytes()).hexdigest(),
            }
            cache[uid] = {"waveform": waveform, "prompt_ids": prompt_ids,
                          "prompt_phones": item["canonical_phones"], "prompt_text": item["prompt_text"],
                          "features": features.float().cpu(), "lengths": lengths.cpu(), "acoustics": acoustics}

    # Freeze all matched, CAL, VAL and mismatch raw Z outputs before any annotation read.
    raw_by_config = {}
    freeze_hashes = {}
    val_map = {}
    by_speaker = {}
    for uid in ids["val"]:
        by_speaker.setdefault(uid.split("/")[0], []).append(uid)
    for speaker_ids in by_speaker.values():
        if len(speaker_ids) < 2:
            raise ValueError("within-speaker cyclic mismatch requires at least two utterances per speaker")
        val_map.update({uid: speaker_ids[(i + 1) % len(speaker_ids)] for i, uid in enumerate(speaker_ids)})
    if (set(val_map) != set(ids["val"]) or set(val_map.values()) != set(ids["val"])
            or any(target == source or target.split("/")[0] != source.split("/")[0]
                   for target, source in val_map.items())):
        raise ValueError("mismatch permutation must cover VAL exactly with same-speaker nonfixed points")
    mapping_rows = [{"target_id": uid, "audio_source_id": val_map[uid]} for uid in ids["val"]]
    mapping_digest = hashlib.sha256(json.dumps(mapping_rows, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    mismatch_mapping_path = output_root / "AUDIO_MISMATCH_MAPPING.json"
    mismatch_mapping_path.write_text(json.dumps({"policy": "within_speaker_cyclic_shift_1", "ordered_mapping": mapping_rows,
                                                "mapping_sha256": mapping_digest}, indent=2, sort_keys=True) + "\n")
    for (arm, endpoint), head in heads.items():
        key = f"{arm}_{endpoint}"
        cfg_dir = output_root / key
        cfg_dir.mkdir()
        groups = {"runtime_Z.jsonl": [], "cal_runtime_Z.jsonl": [], "mismatched_runtime_Z.jsonl": [], "recognition_runtime_Z.jsonl": []}
        prompts_by_uid = {}
        for partition in ("cal", "val"):
            for uid in ids[partition]:
                data = cache[uid]
                output = head(features=data["features"], acoustic_features=data["acoustics"] if FEATURES_ENABLED[arm] else torch.zeros_like(data["acoustics"]),
                              frame_lengths=data["lengths"], prompt_phone_ids=data["prompt_ids"].unsqueeze(0),
                              prompt_lengths=torch.tensor([data["prompt_ids"].numel()], dtype=torch.long))
                rows, recognition = runtime_rows(logits=output["operation_logits"][0, :data["prompt_ids"].numel()],
                    ctc_logits=output["ctc_logits"][0, :int(data["lengths"][0])],
                    prompt_phone_ids=data["prompt_ids"].tolist(), canonical_phones=data["prompt_phones"],
                    speaker_id=uid.split("/")[0], utterance_id=uid.split("/")[1],
                    arm=arm, endpoint=endpoint, condition="matched", audio_source_id=uid)
                prompts_by_uid[uid] = rows
                groups["cal_runtime_Z.jsonl" if partition == "cal" else "runtime_Z.jsonl"].extend(rows)
                if partition == "val":
                    groups["recognition_runtime_Z.jsonl"].append(recognition)
        mismatch_rows = []
        for uid in ids["val"]:
            source_uid = val_map[uid]
            source = cache[source_uid]
            target = cache[uid]
            # Reuse the immutable batch-one cache for this source waveform; only target prompt varies.
            with torch.inference_mode():
                output = head(features=source["features"],
                              acoustic_features=source["acoustics"] if FEATURES_ENABLED[arm] else torch.zeros_like(source["acoustics"]),
                              frame_lengths=source["lengths"], prompt_phone_ids=target["prompt_ids"].unsqueeze(0),
                              prompt_lengths=torch.tensor([target["prompt_ids"].numel()], dtype=torch.long))
            rows, _ = runtime_rows(logits=output["operation_logits"][0, :target["prompt_ids"].numel()],
                ctc_logits=output["ctc_logits"][0, :int(source["lengths"][0])], prompt_phone_ids=target["prompt_ids"].tolist(),
                canonical_phones=target["prompt_phones"],
                speaker_id=uid.split("/")[0], utterance_id=uid.split("/")[1], arm=arm, endpoint=endpoint,
                condition="within_speaker_cyclic", audio_source_id=source_uid)
            mismatch_rows.extend(rows)
        groups["mismatched_runtime_Z.jsonl"] = mismatch_rows
        for filename, rows in groups.items():
            freeze_hashes[f"{key}/{filename}"] = write_jsonl(cfg_dir / filename, rows)
        raw_by_config[key] = {"matched": prompts_by_uid, "mismatch": mismatch_rows,
                              "recognition_runtime": groups["recognition_runtime_Z.jsonl"]}
        print(f"Frozen Z runtime: {key}", flush=True)
    freeze_receipt = {"contract_sha256": CONTRACT_SHA, "gold_reads_before_freeze": 0,
                      "runtime_hashes": freeze_hashes, "audio_mismatch_mapping_sha256": mapping_digest,
                      "allowed_id_hashes": {
                          part: hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest() for part, values in ids.items()}}
    freeze_path = output_root / "Z_RUNTIME_FREEZE.json"
    freeze_path.write_text(json.dumps(freeze_receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")

    # First TextGrid access; only offline metadata and join rows below this point.
    reader = get_l2arctic_reader(str(archive))
    gold = {}
    vowels = {"AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"}
    consonants = set(STANDARD_39_ARPABET) - vowels
    for partition in ("cal", "val"):
        for uid in ids[partition]:
            speaker, utterance = uid.split("/")
            annotation = parse_gold_textgrid_content(reader.read_annotation_text(speaker, utterance), utterance + ".TextGrid")
            labels = build_operation_labels(annotation)
            positions = []
            for i, interval in enumerate(annotation.canonical_intervals):
                midpoint = (interval.xmin + interval.xmax) / 2
                word = next((w for w in annotation.words_intervals if w["xmin"] <= midpoint <= w["xmax"]), None)
                word_positions = [j for j, candidate in enumerate(annotation.canonical_intervals)
                                  if word and word["xmin"] <= (candidate.xmin + candidate.xmax) / 2 <= word["xmax"]]
                operation = int(labels.operation[i])
                positions.append({"canonical_phone": interval.canonical_phone,
                                  "operation_target": operation, "binary_target": int(operation != 0),
                                  "word_final_consonant": len(word_positions) > 1 and i == word_positions[-1] and interval.canonical_phone in consonants})
            reference_ids = [int(phone_id) for phone_id in labels.ctc.tolist()]
            if any(phone_id not in CTC_INDEX_TO_PHONE or phone_id == 0 for phone_id in reference_ids):
                raise ValueError(f"invalid realized CTC reference for {uid}")
            gold[uid] = {"positions": positions, "reference_phone_ids": reference_ids}

    for (arm, endpoint), _head in heads.items():
        key = f"{arm}_{endpoint}"
        cfg_dir = output_root / key
        matched = raw_by_config[key]["matched"]
        for partition, output_name in (("cal", "cal_tokens_Z.jsonl"), ("val", "tokens_Z.jsonl")):
            joined = []
            for uid in ids[partition]:
                joined.extend(join_gold_rows(gold_positions=gold[uid]["positions"], prompt_phones=cache[uid]["prompt_phones"],
                                             runtime=matched[uid], align=align_canonical_sequences))
            write_jsonl(cfg_dir / output_name, joined)
        mismatch_join = []
        mismatch_index = {}
        for row in raw_by_config[key]["mismatch"]:
            row_uid = f"{row['speaker_id']}/{row['utterance_id']}"
            mismatch_index.setdefault(row_uid, []).append(row)
        for uid in ids["val"]:
            mismatch_join.extend(join_gold_rows(gold_positions=gold[uid]["positions"], prompt_phones=cache[uid]["prompt_phones"],
                                                runtime=mismatch_index[uid], align=align_canonical_sequences))
        write_jsonl(cfg_dir / "mismatched_tokens_Z.jsonl", mismatch_join)

        runtime_recognition = {(row["speaker_id"], row["utterance_id"]): row
                               for row in raw_by_config[key]["recognition_runtime"]}
        offline_recognition = [offline_recognition_row(runtime_recognition[tuple(uid.split("/"))],
                                                       gold[uid]["reference_phone_ids"])
                               for uid in ids["val"]]
        write_jsonl(cfg_dir / "recognition_Z.jsonl", offline_recognition)

    for relative, expected_hash in freeze_hashes.items():
        if file_digest(output_root / relative) != expected_hash:
            raise ValueError(f"gold-free runtime file changed after freeze: {relative}")
    if {f"{arm}:{endpoint}": fingerprint(head) for (arm, endpoint), head in heads.items()} != model_before:
        raise ValueError("checkpoint model state changed during inference")
    if fingerprint(frozen_encoder) != backbone_before:
        raise ValueError("frozen backbone changed during inference")

    cache_receipt_path = output_root / "FEATURE_CACHE_IDENTITIES.json"
    cache_receipt_path.write_text(json.dumps({"version": "trn04r-fp32-batch1-v1", "utterances": feature_cache_receipt},
                                             indent=2, sort_keys=True, allow_nan=False) + "\n")
    source_hashes_after = {name: file_digest(ROOT / name) for name in SOURCE_FILES}
    if source_hashes_after != source_hashes_before:
        raise ValueError("runtime source files changed during inference")
    if (file_digest(prompts_file) != PROMPT_SHA or file_digest(dictionary_file) != DICTIONARY_SHA
            or file_digest(split_path) != SPLIT_SHA or file_digest(contract) != CONTRACT_SHA):
        raise ValueError("frozen prompt/G2P/split/contract input changed during inference")
    manifest = {"schema_version": "trn04r-v1", "status": "INFERENCE_RECORDS_FROZEN",
        "contract_sha256": CONTRACT_SHA, "protocol": "Z-PROMPT-V1", "git_tip": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        "source_hashes": source_hashes_before,
        "source_hashes_verified_unchanged": source_hashes_after == source_hashes_before,
        "split_sha256": file_digest(split_path), "authorized_archive_sha256": file_digest(archive),
        "prompt_table_sha256": file_digest(prompts_file), "g2p_dictionary_sha256": file_digest(dictionary_file),
        "arms_endpoints": {f"{arm}:{endpoint}": checkpoint_info[arm, endpoint] for arm, endpoint in heads},
        "backbone": {"repo": BACKBONE, "revision": BACKBONE_REVISION, "model_safetensors_sha256": backbone_sha256},
        "id_hashes": freeze_receipt["allowed_id_hashes"], "feature_cache_sha256": file_digest(cache_receipt_path),
        "audio_mismatch_mapping_sha256": mapping_digest,
        "precision": "CPU FP32", "device": "cpu", "batch_size": 1,
        "threads": torch.get_num_threads(), "elapsed_seconds": time.time() - start,
        "command": sys.argv, "interpreter": sys.executable, "python_version": platform.python_version(),
        "hardware": {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count()},
        "dependencies": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "soundfile", "scipy", "huggingface-hub")},
        "backward_passes": 0, "optimizer_updates": 0, "gold_free_z_frozen_before_annotation": True,
        "outputs": {str(path.relative_to(output_root)): file_digest(path) for path in sorted(output_root.rglob("*")) if path.is_file()}}
    (output_root / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"Complete; private outputs: {output_root}", flush=True)


if __name__ == "__main__":
    main()
