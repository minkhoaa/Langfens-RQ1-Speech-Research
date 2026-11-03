#!/usr/bin/env python3
"""Offline package audit only: no corpus/model load, CUDA probe, service or training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from langfens_prep.p_sup.runtime import BASE_SHA, CONTRACT_SHA256


def audit_training_staging_package(root: Path) -> dict:
    staging_dir = root / "deploy/kaggle_p_sup_train"
    if not staging_dir.is_dir():
        return {"status": "NOT_PRESENT"}

    metadata_path = staging_dir / "kernel-metadata.json"
    if not metadata_path.is_file():
        raise ValueError("Missing kernel-metadata.json in training staging package")

    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    if meta.get("machine_shape") != "NvidiaTeslaT4":
        raise ValueError(f"Expected machine_shape NvidiaTeslaT4, got {meta.get('machine_shape')}")
    if meta.get("enable_gpu") is not True:
        raise ValueError("enable_gpu must be true")
    if meta.get("is_private") is not True:
        raise ValueError("is_private must be true")
    if meta.get("dataset_sources") != ["minkhoaa/langfens-rq1-fold1-staging"]:
        raise ValueError(f"dataset_sources mismatch: {meta.get('dataset_sources')}")

    notebook_path = staging_dir / "kaggle_train.ipynb"
    if not notebook_path.is_file():
        raise ValueError("Missing kaggle_train.ipynb in training staging package")
    json.loads(notebook_path.read_text(encoding="utf-8"))

    # Verify trainer and dataset modules are in place
    trainer_script = root / "audit_scripts/train_p_sup.py"
    if not trainer_script.is_file():
        raise ValueError("Missing audit_scripts/train_p_sup.py")

    dataset_module = root / "src/langfens_prep/p_sup/dataset.py"
    if not dataset_module.is_file():
        raise ValueError("Missing src/langfens_prep/p_sup/dataset.py")

    policy_doc = root / "manifests/RQ1-P-SUP/OBJECTIVE_POLICY_AMENDMENT.md"
    if not policy_doc.is_file():
        raise ValueError("Missing manifests/RQ1-P-SUP/OBJECTIVE_POLICY_AMENDMENT.md")

    return {
        "status": "STAGING_VERIFIED_PASSED",
        "package_dir": str(staging_dir.relative_to(root)),
        "kernel_id": meta.get("id"),
        "machine_shape": meta.get("machine_shape"),
        "enable_gpu": meta.get("enable_gpu"),
        "is_private": meta.get("is_private"),
        "dataset_sources": meta.get("dataset_sources")
    }


def preflight(root: Path):
    contract = root / "manifests/RQ1-WORKSPACE-PREP-01/EVALUATION_CONTRACT_V1.md"
    if hashlib.sha256(contract.read_bytes()).hexdigest() != CONTRACT_SHA256:
        raise ValueError("frozen V1 hash mismatch")
    config_path = root / "configs/rq1_p_sup_v1.json"
    config = json.loads(config_path.read_text())
    expected = {"seed": 1337, "max_successful_updates": 3600, "warmup_updates": 100,
                "cosine_horizon_updates": 15000, "batch_size": 2, "gradient_accumulation_steps": 2,
                "backbone_revision": "22aad52d435eb6dbaf354bdad9b0da84ce7d6156",
                "backbone": "facebook/wav2vec2-base-960h", "backbone_frozen_eval": True,
                "execute_training": False, "contract_sha256": CONTRACT_SHA256,
                "objectives": ["operation_ce", "binary_collapsed"], "fit_utterances": 1799,
                "selection_utterances": 300, "calibration_utterances": 300, "evaluation_utterances": 300,
                "head": {"feature_dim": 768, "width": 128, "heads": 4, "max_prompt_length": 256, "dropout": 0.1},
                "ctc_weight": 1.0, "task_weight": 1.0, "warmstart": None,
                "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.01,
                "gradient_clip": 1.0, "grad_scaler_initial_scale": 4096,
                "flush_tail_divisor": "actual_group_size", "workers": 0, "length_bucketing": False}
    for key, value in expected.items():
        if config.get(key) != value or type(config[key]) is not type(value):
            raise ValueError(f"registration mismatch: {key}")
    # Train/val ID metadata only; no corpus/label readers and no TEST ID access/export.
    split_path = root / "manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json"
    if hashlib.sha256(split_path.read_bytes()).hexdigest() != "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b":
        raise ValueError("split digest mismatch")
    splits = json.loads(split_path.read_text())
    fold = next(f for f in splits["folds"] if f["fold_index"] == 1)
    provenance = json.loads((root / "manifests/RQ1-WORKSPACE-PREP-01/SHARED_PROVENANCE.json").read_text())
    partitions = provenance["fold1_partitions"]
    for config_name, source_name in (("fit", "fit"), ("selection", "selection"),
                                     ("calibration", "calibration"), ("evaluation", "val")):
        registered = partitions[source_name]
        if config[config_name + "_speakers"] != registered["speakers"]:
            raise ValueError("speaker registration mismatch")
        envelope = fold["val" if source_name == "val" else "train"]["utterances"]
        ids = [u for u in envelope if u.split("/")[0] in registered["speakers"]]
        digest = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
        if len(ids) != registered["utterances"] or digest != registered["ordered_id_sha256"]:
            raise ValueError("ordered partition ID mismatch")
    package = root / "deploy/kaggle_p_sup_preflight"
    metadata_path = package / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if (metadata.get("machine_shape") != "NvidiaTeslaT4" or metadata.get("is_private") is not True
            or metadata.get("enable_gpu") is not True or metadata.get("enable_internet") is not False
            or metadata.get("kernel_type") != "script" or metadata.get("code_file") != "preflight.py"):
        raise ValueError("private offline T4 preflight metadata required")
    if metadata.get("dataset_sources") or metadata.get("kernel_sources") or metadata.get("model_sources"):
        raise ValueError("preparation package must not attach data/models")
    package_files = {p.name for p in package.iterdir() if p.is_file()}
    if package_files != {"kernel-metadata.json", "preflight.py"}:
        raise ValueError("preflight package must contain metadata and standalone gate only")
    package_code = (package / "preflight.py").read_text()
    if any(marker in package_code for marker in ("torch", "transformers", "huggingface", "kaggle", "boto3")):
        raise ValueError("standalone preflight must not load models or contact remote services")
    if config.get("precision") != "cuda_fp16_amp" or config.get("license") != "CC BY-NC 4.0":
        raise ValueError("precision/license mismatch")
    base = subprocess.check_output(["git", "merge-base", "HEAD", BASE_SHA], cwd=root, text=True).strip()
    if base != BASE_SHA:
        raise ValueError("wrong ancestry")

    staging_audit = audit_training_staging_package(root)

    return {"status": "PREPARATION_AUDITED_EXPERIMENT_HOLD", "contract_sha256": CONTRACT_SHA256,
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "git_tip": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "training": "NOT_STARTED", "successful_updates": 0, "skipped_updates": 0, "attempts": 0,
            "new_model_artifact": None, "training_entrypoint_available": False,
            "staging_package_audit": staging_audit,
            "blockers": ["remote data adapter and allowlist/hash verification not implemented",
                         "per-epoch atomic private durability and strict resume integration not implemented",
                         "end-to-end authorized T4 FP16 trainer not implemented",
                         "future distinct-objective policy amendment and execution authorization required",
                         "missing-proposal disposition, margins, licenses and subtype support unverified"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(preflight(args.root), indent=2, allow_nan=False))
