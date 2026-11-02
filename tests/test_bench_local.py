"""Source-named, inference-only tests for the single benchmark runtime."""
import gzip
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

from langfens_bench import runtime as bench

import numpy as np
import pytest
import torch


def frozen_contract():
    return json.loads((bench.REPOSITORY / "manifests/RQ1-BENCH-01/contract.json").read_text())


def frozen_split():
    return json.loads((bench.REPOSITORY / frozen_contract()["split"]["path"]).read_text())


def test_validate_ids_exact_validation_subset_and_test_exclusion():
    contract, split = frozen_contract(), frozen_split()
    assert len(bench.validate_ids(contract, split)) == 4
    contract["utterances"][0]["speaker"] = split["folds"][0]["test"]["speakers"][0]
    u = contract["utterances"][0]
    u["prediction_id"] = f"1/{u['speaker']}/{u['utterance_id']}"
    with pytest.raises(bench.BenchmarkBlocked, match="Non-validation/active TEST"):
        bench.validate_ids(contract, split)


def test_validate_ids_duplicate_refused():
    contract = frozen_contract()
    contract["utterances"][1] = contract["utterances"][0]
    with pytest.raises(bench.BenchmarkBlocked, match="Duplicate/invalid"):
        bench.validate_ids(contract, frozen_split())


def package(tmp_path, extra=None):
    fold = {"fold_index": 1,
            "train": {"speakers": ["A"], "utterances": ["A/u1"]},
            "val": {"speakers": ["B"], "utterances": ["B/u2"]},
            "test": {"speakers": ["SEALED"]}}
    path = tmp_path / "package.zip"
    with zipfile.ZipFile(path, "w") as master:
        for speaker, uid in (("A", "u1"), ("B", "u2")):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as inner:
                inner.writestr(f"{speaker}/wav/{uid}.wav", b"not a real wav")
                inner.writestr(f"{speaker}/annotation/{uid}.TextGrid", "not a real TextGrid")
                if extra and speaker == "A":
                    inner.writestr(extra, b"sealed")
            master.writestr(f"{speaker}.zip", buf.getvalue())
    return path, fold


def test_audit_package_never_reads_payload_members(tmp_path, monkeypatch):
    path, fold = package(tmp_path)
    original_read = zipfile.ZipFile.read
    def guarded_read(self, name, *args, **kwargs):
        assert str(name).endswith(".zip"), "WAV/TextGrid payload was read"
        return original_read(self, name, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, "read", guarded_read)
    result = bench.audit_package(path, fold)
    assert result["payload_members_read"] == 0
    assert result["members"] == 4


@pytest.mark.parametrize("extra", ["SEALED/wav/u.wav", "A/wav/unlisted.wav", "../escape.wav"])
def test_audit_package_rejects_test_unlisted_and_traversal(tmp_path, extra):
    path, fold = package(tmp_path, extra)
    with pytest.raises(bench.BenchmarkBlocked, match="Unauthorized"):
        bench.audit_package(path, fold)


def test_hash_and_root_escape_fail_closed(tmp_path):
    p = tmp_path / "artifact"
    p.write_bytes(b"data")
    with pytest.raises(bench.BenchmarkBlocked, match="SHA-256"):
        bench.verify_hash(tmp_path, {"path": "artifact", "sha256": "0" * 64})
    with pytest.raises(bench.BenchmarkBlocked, match="escapes root"):
        bench.repository_path("../sealed", tmp_path)


def test_preflight_hash_failure_precedes_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "audit_package", lambda *a: pytest.fail("inventory before all hashes"))
    monkeypatch.setattr(bench, "configure_torch", lambda *a: (_ for _ in ()).throw(bench.BenchmarkBlocked("test hardware blocker")))
    result, _ = bench.preflight(frozen_contract(), tmp_path, "LOCAL_FP32")
    assert result["status"] == "BLOCKED"
    assert "archive_inventories" not in result["checks"]
    assert result["blockers"]


def test_attention_metrics_active_lengths_uniform_and_peak():
    # Padded row/column must never contaminate the diagnostic.
    a = np.array([[.25, .25, .25, .25, 999], [999, 999, 999, 999, 999]], dtype=np.float16)
    result = bench.attention_metrics(a, [SimpleNamespace(xmin=.02, xmax=.04)], 4)
    assert result["in_window_mass"] == .25
    assert result["uniform_in_window_mass"] == .25
    assert result["normalized_entropy"] == 1.0
    assert result["absolute_peak_offset_seconds"] == .03
    with pytest.raises(bench.BenchmarkBlocked, match="Invalid attention"):
        bench.attention_metrics(np.array([[np.nan]]), [SimpleNamespace(xmin=0, xmax=.02)], 1)


def example_record():
    return {"prediction_id": "1/A/u", "fold": 1, "labels": [0, 1, 1],
            "error_types": ["correct", "substitution", "deletion"], "binary_scores": [.1, .8, .3],
            "ctc_ids": [1, 2], "ctc_reference_ids": [1, 2, 3],
            "attention_metrics": {"tokens": [{"in_window_mass": .3, "uniform_in_window_mass": .2,
                                                "normalized_entropy": .9, "absolute_peak_offset_seconds": .4}]},
            "protocol_z": {"aligned_pairs": [[0, 0, 0], [2, 1, 0]], "binary_scores": [.1, .3],
                           "unaligned_gold_count": 1, "unaligned_g2p_count": 0}, "latency_seconds": .5}


def test_aggregate_metrics_ap_auc_per_omissions_alignment_and_float64():
    result = bench.aggregate_metrics([example_record()])
    o, z = result["quality"], result["quality"]["protocol_z"]
    assert o["ap"] == o["roc_auc"] == 1
    assert o["per"] == pytest.approx(1 / 3)
    assert o["omission_recall_0.2"] == 1
    assert o["omission_fnr_0.5"] == 1
    assert z["ap"] == z["roc_auc"] == 1
    assert z["coverage"] == pytest.approx(2 / 3)
    assert result["performance"]["throughput_utterances_per_second"] == 2
    assert bench.binary_metrics([0, 0], [.5, .5])["ap"] is None
    assert bench.binary_metrics([1, 1], [.5, .5])["roc_auc"] is None
    assert bench.binary_metrics([1, 0], [.5, .5])["ap"] == .5
    assert bench.binary_metrics([1, 0], [.5, .5])["roc_auc"] == .5
    json.dumps(result, allow_nan=False)


def test_protocol_z_sub_first_is_diagonal_deletion_insertion():
    from langfens_prep.alignment import align_canonical_sequences
    result = align_canonical_sequences(["A", "B"], ["B", "A"], tie_break="sub_first")
    assert result["aligned_pairs"] == [(0, 0, 1), (1, 1, 1)]
    assert bench.SEMANTICS["protocol_z"].endswith("diag_del_ins (= sub_first)")


@pytest.mark.parametrize("profile", bench.PROFILES)
def test_forward_pair_inference_only_no_gold_times_and_fp16_autocast_only(profile, monkeypatch):
    contexts = []
    class Context:
        def __enter__(self):
            contexts.append("enter")
        def __exit__(self, *a):
            contexts.append("exit")
    def autocast(**kwargs):
        assert kwargs == {"device_type": "cuda", "dtype": torch.float16}
        return Context()
    monkeypatch.setattr(torch, "autocast", autocast)
    calls = []
    class Model:
        def __call__(self, **kwargs):
            assert torch.is_inference_mode_enabled()
            assert set(kwargs) == {"waveforms", "audio_lengths", "canonical_phone_indices", "canonical_phone_lengths"}
            assert kwargs["waveforms"].dtype == torch.float32
            calls.append(kwargs)
            return "output"
    outputs = bench.forward_pair(Model(), torch.ones(100), ["AA"], ["B", "AA"], profile, torch.device("cpu"))
    assert outputs[:2] == ("output", "output")
    assert len(calls) == 2
    assert contexts == (["enter", "exit"] if profile == "KAGGLE_FP16" else [])


def test_configure_torch_local_determinism():
    device, hardware = bench.configure_torch("LOCAL_FP32")
    assert device.type == hardware["device"] == "cpu"
    assert torch.get_num_threads() == 1
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32


def test_remote_evaluation_authorization_checked_before_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "KAGGLE_WORKING", tmp_path)
    monkeypatch.setattr(bench, "preflight", lambda *a: pytest.fail("preflight must not run without authorization"))
    out = tmp_path / "remote"
    result = bench.main(["evaluate", "--profile", "KAGGLE_FP16", "--contract", "manifests/RQ1-BENCH-01/contract.json",
                         "--artifact-root", str(tmp_path), "--outdir", str(out)])
    assert result == 2
    environment = json.loads((out / "environment.json").read_text())
    assert environment["metadata"]["status"] == "BLOCKED"
    assert "authorize-remote-inference" in environment["error"]["message"]
    assert not (out / "predictions.json.gz").exists()


def test_offline_cache_missing_refused(monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **kw: None)
    with pytest.raises(bench.BenchmarkBlocked, match="Missing offline pretrained cache"):
        bench.check_offline_cache()


def test_main_outputs_provenance_artifact_hashes_and_no_overwrite(tmp_path, monkeypatch):
    contract = frozen_contract()
    records = [{**example_record(), **u} for u in contract["utterances"]]
    monkeypatch.setattr(bench, "preflight", lambda *a: ({"status": "READY", "checks": {}, "blockers": []}, {}))
    monkeypatch.setattr(bench, "evaluate", lambda *a: (records, {
        "peak_ram_bytes": 1024, "peak_vram_allocated_bytes": None, "peak_vram_reserved_bytes": None}))
    out = tmp_path / "evaluation"
    args = ["evaluate", "--profile", "LOCAL_FP32", "--contract", "manifests/RQ1-BENCH-01/contract.json",
            "--artifact-root", str(tmp_path), "--outdir", str(out)]
    assert bench.main(args) == 0
    environment = json.loads((out / "environment.json").read_text())
    metrics = json.loads((out / "metrics.json").read_text())
    with gzip.open(out / "predictions.json.gz", "rt") as stream:
        predictions = json.load(stream)
    assert predictions["metadata"] == metrics["metadata"] == environment["metadata"]
    assert predictions["metadata"]["status"] == "COMPLETED"
    assert predictions["metadata"]["scope_fold"] is None
    assert len(predictions["metadata"]["evaluator_semantics_sha256"]) == 64
    assert "completion_status" not in predictions["metadata"]
    assert "evaluator_semantic_sha256" not in predictions["metadata"]
    assert set(metrics["quality"]) == {
        "ap", "roc_auc", "per", "omission_recall_0.2", "omission_fnr_0.2",
        "omission_recall_0.5", "omission_fnr_0.5", "attention_mass", "attention_entropy",
        "attention_peak_offset_seconds", "protocol_z"}
    assert set(metrics["quality"]["protocol_z"]) == {"ap", "roc_auc", "coverage"}
    assert set(metrics["performance"]) == {
        "latency_p50_seconds", "latency_p95_seconds", "throughput_utterances_per_second",
        "peak_ram_bytes", "peak_vram_allocated_bytes", "peak_vram_reserved_bytes"}
    assert set(environment["artifacts"]) == {"metrics.json", "predictions.json.gz"}
    assert len(predictions["utterances"]) == 80
    for name, digest in environment["artifacts"].items():
        assert digest == bench.sha256(out / name)
    with pytest.raises(SystemExit):
        bench.main(args)


def test_evaluation_error_records_failure_without_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "preflight", lambda *a: ({"status": "READY", "checks": {}, "blockers": []}, {}))
    def fail(*a):
        raise RuntimeError("injected forward failure")
    monkeypatch.setattr(bench, "evaluate", fail)
    out = tmp_path / "failed"
    assert bench.main(["evaluate", "--profile", "LOCAL_FP32", "--contract", "manifests/RQ1-BENCH-01/contract.json",
                       "--artifact-root", str(tmp_path), "--outdir", str(out)]) == 2
    assert json.loads((out / "environment.json").read_text())["metadata"]["status"] == "FAILED"
    assert not (out / "predictions.json.gz").exists()


def test_deterministic_gzip_and_nonfinite_json_refused(tmp_path):
    first, second = tmp_path / "a.gz", tmp_path / "b.gz"
    bench.write_predictions(first, {"utterances": []})
    bench.write_predictions(second, {"utterances": []})
    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(ValueError):
        bench.json_bytes({"score": float("nan")})


def test_evaluate_fp16_postprocessing_casts_before_sigmoid_and_warms_once(monkeypatch):
    from contextlib import nullcontext
    import audit_scripts.evaluate_fold1_ctc_binary as existing_eval
    import langfens_prep.audio_loader as audio
    import langfens_prep.textgrid_parser as textgrid
    calls = []
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            self.eval()
        def forward(self, **kwargs):
            assert torch.is_inference_mode_enabled()
            assert "phone_intervals" not in kwargs and "target_binary" not in kwargs
            calls.append(kwargs)
            return SimpleNamespace(
                binary_logits=torch.tensor([[8., -8.]], dtype=torch.float16),
                ctc_logits=torch.tensor([[[0., 1., 0.], [0., 0., 1.]]], dtype=torch.float16),
                alignment_weights=torch.tensor([[[.5, .5], [.5, .5]]], dtype=torch.float16),
                input_lengths=torch.tensor([2]))
    cfg = {"pretrained_model": bench.PRETRAINED_MODEL, "pretrained_revision": bench.PRETRAINED_REVISION,
           "loss_weights": {"ctc": 1., "binary": 1.}}
    monkeypatch.setattr(torch, "load", lambda *a, **kw: {"model": {}})
    monkeypatch.setattr(torch, "autocast", lambda **kw: nullcontext())
    monkeypatch.setattr(bench, "checkpoint_config", lambda *a: cfg)
    monkeypatch.setattr(existing_eval, "build_eval_model", lambda *a, **kw: Model())
    closed = []
    reader = SimpleNamespace(read_annotation_text=lambda *a: "gold scoring only", close=lambda: closed.append(True))
    monkeypatch.setattr(audio, "get_l2arctic_reader", lambda *a: reader)
    monkeypatch.setattr(audio, "load_l2arctic_audio_from_zip", lambda *a: (torch.ones(1600), 16000, None))
    intervals = [SimpleNamespace(canonical_phone=p, binary_label_strict=y, error_type=e,
                                 realized_phone=p, has_err_tag=False, xmin=0., xmax=.02)
                 for p, y, e in (("AA", 0, "correct"), ("B", 1, "deletion"))]
    annotation = SimpleNamespace(canonical_intervals=intervals, intervals=intervals)
    monkeypatch.setattr(textgrid, "parse_gold_textgrid_content", lambda *a: annotation)
    contract = {"folds": [{"fold": 1, "checkpoint": {"sha256": "test-only"}},
                          {"fold": 2, "checkpoint": {"sha256": "must-not-load"}}],
                "utterances": [{"fold": 1, "speaker": "A", "utterance_id": "u", "prediction_id": "1/A/u"},
                               {"fold": 2, "speaker": "SEALED_FOR_FOLD_1", "utterance_id": "blocked",
                                "prediction_id": "2/SEALED_FOR_FOLD_1/blocked"}]}
    paths = {"device": torch.device("cpu"), "offline_model": "offline-snapshot", "z_phones": {"1/A/u": ["AA", "B"]},
             1: {"checkpoint": Path("synthetic.pt"), "package": Path("synthetic.zip")}}
    monkeypatch.setattr(bench, "KAGGLE_INPUT", Path.cwd())  # synthetic CPU-only precision fixture
    records, memory = bench.evaluate(contract, paths, "KAGGLE_FP16", selected_fold=1)
    assert len(calls) == 4  # warmup O/Z, then timed O/Z
    assert closed == [True]
    expected = torch.tensor([8., -8.], dtype=torch.float32).sigmoid().tolist()
    assert records[0]["binary_scores"] == records[0]["protocol_z"]["binary_scores"] == expected
    assert expected != torch.tensor([8., -8.], dtype=torch.float16).sigmoid().tolist()
    assert records[0]["ctc_reference_ids"] == [1]  # deleted B excluded from CTC target
    assert records[0]["ctc_ids"] == [1, 2]
    assert records[0]["attention_metrics"]["normalized_entropy"] == 1
    assert memory["peak_vram_allocated_bytes"] is None


def test_checkpoint_config_rejects_wrong_fold_budget_and_half_weights():
    cfg = {"fold": 1, "pretrained_model": bench.PRETRAINED_MODEL, "pretrained_revision": bench.PRETRAINED_REVISION,
           "backbone_frozen": True, "loss_weights": {"ctc": 1., "binary": 1.}}
    checkpoint = {"metadata": {"config": cfg}, "successful_updates": 3600,
                  "model": {"weight": torch.ones(1)}}
    assert bench.checkpoint_config(checkpoint, 1) == cfg
    with pytest.raises(bench.BenchmarkBlocked, match="contract violation"):
        bench.checkpoint_config(checkpoint, 2)
    checkpoint["successful_updates"] = 3598
    with pytest.raises(bench.BenchmarkBlocked, match="contract violation"):
        bench.checkpoint_config(checkpoint, 1)
    checkpoint["successful_updates"] = 3600
    checkpoint["model"]["weight"] = torch.ones(1, dtype=torch.float16)
    with pytest.raises(bench.BenchmarkBlocked, match="not FP32"):
        bench.checkpoint_config(checkpoint, 1)


def test_remote_hardware_requires_kaggle_identity_before_cuda(monkeypatch):
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("must refuse identity before CUDA"))
    with pytest.raises(bench.BenchmarkBlocked, match="Kaggle runtime identity"):
        bench.configure_torch("KAGGLE_FP32")


@pytest.mark.parametrize("selected_fold", [1, 2, 3, 4])
def test_fold_preflight_never_resolves_hashes_opens_or_inventories_other_fold_artifacts(tmp_path, monkeypatch, selected_fold):
    contract = frozen_contract()
    selected = next(f for f in contract["folds"] if f["fold"] == selected_fold)
    accessed, inventoried, loaded = [], [], []
    def verify(root, entry):
        if root == bench.REPOSITORY:
            return bench.REPOSITORY / entry["path"]
        assert entry in [selected["checkpoint"], selected["package"]], "unselected artifact accessed"
        accessed.append(entry["path"])
        return tmp_path / entry["path"]
    def inventory(path, fold):
        assert fold["fold_index"] == selected_fold
        inventoried.append(fold["fold_index"])
        return {"active_test_excluded": True}
    cfg = {"fold": selected_fold, "pretrained_model": bench.PRETRAINED_MODEL,
           "pretrained_revision": bench.PRETRAINED_REVISION, "backbone_frozen": True,
           "loss_weights": {"ctc": 1., "binary": 1.}}
    def load(path, **kwargs):
        assert path == tmp_path / selected["checkpoint"]["path"]
        loaded.append(path)
        return {"metadata": {"config": cfg}, "successful_updates": 3600, "epoch": 5, "model": {"weight": torch.ones(1)}}
    monkeypatch.setattr(bench, "verify_hash", verify)
    monkeypatch.setattr(bench, "audit_package", inventory)
    monkeypatch.setattr(bench, "check_offline_cache", lambda: {"snapshot_path": "offline-snapshot"})
    monkeypatch.setattr(torch, "load", load)
    result, paths = bench.preflight(contract, tmp_path, "LOCAL_FP32", selected_fold)
    assert result["status"] == "READY", result["blockers"]
    assert accessed == [selected["checkpoint"]["path"], selected["package"]["path"]]
    assert inventoried == [selected_fold]
    assert len(loaded) == 1
    assert {k for k in paths if isinstance(k, int)} == {selected_fold}
    assert len(paths["z_phones"]) == 20
    assert all(key.startswith(f"{selected_fold}/") for key in paths["z_phones"])
    assert result["checks"]["selected_folds"] == [selected_fold]
    assert set(result["checks"]["verified_artifacts"]["checkpoint_sha256"]) == {str(selected_fold)}
    metadata = bench.provenance(bench.REPOSITORY / "manifests/RQ1-BENCH-01/contract.json", contract, "LOCAL_FP32", selected_fold)
    assert metadata["scope"] == "fold" and metadata["fold"] == metadata["scope_fold"] == selected_fold
    assert len(metadata["ordered_prediction_ids"]) == 20
    assert set(metadata["checkpoint_sha256"]) == set(metadata["package_sha256"]) == {"1", "2", "3", "4"}


@pytest.mark.parametrize("profile", ["KAGGLE_FP32", "KAGGLE_FP16"])
def test_remote_preflight_and_evaluate_refuse_unscoped_inputs_without_reads(profile, tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "verify_hash", lambda *a: pytest.fail("must not access any artifact"))
    monkeypatch.setattr(bench, "configure_torch", lambda *a: pytest.fail("must not probe GPU before fold scope"))
    result, paths = bench.preflight(frozen_contract(), tmp_path, profile)
    assert result["status"] == "BLOCKED" and not paths
    assert "require --fold" in result["blockers"][0]
    with pytest.raises(bench.BenchmarkBlocked, match="require --fold"):
        bench.evaluate(frozen_contract(), {}, profile)


@pytest.mark.parametrize("command", ["preflight", "evaluate"])
def test_remote_cli_requires_fold_before_preflight_even_when_authorized(tmp_path, monkeypatch, command):
    monkeypatch.setattr(bench, "KAGGLE_WORKING", tmp_path)
    monkeypatch.setattr(bench, "preflight", lambda *a: pytest.fail("unscoped remote execution must stop"))
    out = tmp_path / command
    assert bench.main([command, "--profile", "KAGGLE_FP32", "--authorize-remote-inference",
                       "--contract", "manifests/RQ1-BENCH-01/contract.json",
                       "--artifact-root", str(tmp_path), "--outdir", str(out)]) == 2
    assert "require --fold" in json.loads((out / "environment.json").read_text())["error"]["message"]


def test_fold_cli_outputs_selected_order_scope_and_counts_with_all_contract_pins(tmp_path, monkeypatch):
    contract = frozen_contract()
    selected_fold = 3
    records = [{**example_record(), **u} for u in contract["utterances"] if u["fold"] == selected_fold]
    def preflight(c, root, profile, fold):
        assert c == contract and fold == selected_fold
        return {"status": "READY", "checks": {}, "blockers": []}, {}
    def evaluate(c, paths, profile, fold):
        assert c == contract and fold == selected_fold
        return records, {"peak_ram_bytes": 1024, "peak_vram_allocated_bytes": None, "peak_vram_reserved_bytes": None}
    monkeypatch.setattr(bench, "preflight", preflight)
    monkeypatch.setattr(bench, "evaluate", evaluate)
    out = tmp_path / "fold3"
    assert bench.main(["evaluate", "--profile", "LOCAL_FP32", "--fold", "3",
                       "--contract", "manifests/RQ1-BENCH-01/contract.json",
                       "--artifact-root", str(tmp_path), "--outdir", str(out)]) == 0
    metrics = json.loads((out / "metrics.json").read_text())
    metadata = metrics["metadata"]
    assert metadata["status"] == "COMPLETED" and metadata["scope"] == "fold" and metadata["fold"] == 3
    assert metadata["ordered_prediction_ids"] == [r["prediction_id"] for r in records]
    assert set(metadata["checkpoint_sha256"]) == {"1", "2", "3", "4"}
    assert metrics["counts"]["utterances"] == 20
    assert set(metrics["by_fold"]) == {"3"}
    assert bench.scoped_contract(contract)["utterances"] == contract["utterances"]
    with pytest.raises(bench.BenchmarkBlocked, match="Fold scope"):
        bench.scoped_contract(contract, 5)


def test_compute_quality_shared_merge_entrypoint():
    assert bench.compute_quality([example_record()]) == bench.aggregate_metrics([example_record()])["quality"]
    assert "omission_recall_0.3" in bench.compute_quality([example_record()], thresholds=(.3,))


def test_remote_mount_inventory_isolation_and_known_other_fold_rejection(tmp_path, monkeypatch):
    contract = frozen_contract()
    mount = tmp_path / "input"
    root = mount / "private-fold"
    monkeypatch.setattr(bench, "KAGGLE_INPUT", mount)
    selected = contract["folds"][0]
    for kind in ("checkpoint", "package"):
        path = root / selected[kind]["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"inventory only; hashes checked later")
    assert bench.audit_remote_inputs(contract, root, 1)["payload_reads"] == 0
    unexpected = mount / "other-mounted-data.bin"
    unexpected.write_bytes(b"unknown")
    with pytest.raises(bench.BenchmarkBlocked, match="isolation unverified"):
        bench.audit_remote_inputs(contract, root, 1)
    unexpected.unlink()
    other = root / contract["folds"][1]["package"]["path"]
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_bytes(b"must not read")
    with pytest.raises(bench.BenchmarkBlocked, match="Other fold artifact"):
        bench.audit_remote_inputs(contract, root, 1)
    other.unlink()
    (mount / "alias").symlink_to(root, target_is_directory=True)
    with pytest.raises(bench.BenchmarkBlocked, match="symlink"):
        bench.audit_remote_inputs(contract, root, 1)
    with pytest.raises(bench.BenchmarkBlocked, match="under /kaggle/input"):
        bench.audit_remote_inputs(contract, tmp_path / "outside", 1)


def test_remote_output_outside_working_refused_before_directory_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "KAGGLE_WORKING", tmp_path / "working")
    out = tmp_path / "outside"
    with pytest.raises(SystemExit):
        bench.main(["preflight", "--profile", "KAGGLE_FP32", "--fold", "1",
                    "--contract", "manifests/RQ1-BENCH-01/contract.json",
                    "--artifact-root", str(tmp_path), "--outdir", str(out)])
    assert not out.exists()
