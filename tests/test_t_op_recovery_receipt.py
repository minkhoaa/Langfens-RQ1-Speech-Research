"""Unit and regression tests for Treatment T-OP independent recovery and verification receipt."""

import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
RECEIPT_PATH = REPO_ROOT / "manifests/RQ1-P-SUP-BENCHMARK-V1/recovery/t_op/RECOVERY_RECEIPT.json"
CHECKPOINT_PATH = REPO_ROOT / "manifests/RQ1-P-SUP-BENCHMARK-V1/recovery/t_op/PROCESS_CHECKPOINT.md"


class TestTreatmentTOpRecovery(unittest.TestCase):
    def setUp(self):
        self.assertTrue(RECEIPT_PATH.is_file(), f"Receipt missing: {RECEIPT_PATH}")
        self.assertTrue(CHECKPOINT_PATH.is_file(), f"Process checkpoint missing: {CHECKPOINT_PATH}")
        with open(RECEIPT_PATH, "r", encoding="utf-8") as f:
            self.receipt = json.load(f)

    def test_receipt_structure_and_status(self):
        self.assertEqual(self.receipt.get("arm"), "treatment")
        self.assertEqual(self.receipt.get("arm_identifier"), "T-OP")
        self.assertEqual(self.receipt.get("objective"), "operation_ce")
        self.assertEqual(self.receipt.get("recovery_status"), "PASSED")
        self.assertEqual(self.receipt["process_update_accounting"]["this_process_successful_updates"], 0)
        self.assertEqual(self.receipt["process_update_accounting"]["this_process_skipped_updates"], 0)
        self.assertEqual(self.receipt["process_update_accounting"]["historical_training_successful_updates"], 3600)
        self.assertEqual(self.receipt["process_update_accounting"]["historical_training_skipped_updates"], 0)

    def test_canonical_artifact_hashes_and_files(self):
        artifacts = self.receipt["recovered_artifacts"]
        for key in ["fixed_budget_3600.pt", "best.pt", "last.pt", "training_summary.json", "history.json"]:
            self.assertIn(key, artifacts)
            item = artifacts[key]
            scratch_path = Path(item["scratch_path"])
            self.assertTrue(scratch_path.is_file(), f"Scratch file missing: {scratch_path}")
            self.assertEqual(scratch_path.stat().st_size, item["size_bytes"])
            computed_sha = hashlib.sha256(scratch_path.read_bytes()).hexdigest()
            self.assertEqual(computed_sha, item["sha256"], f"SHA mismatch for {key}")

    def test_checkpoint_payload_integrity(self):
        safe_globals = [np._core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType]
        artifacts = self.receipt["recovered_artifacts"]

        # 1. fixed_budget_3600.pt
        fixed_path = artifacts["fixed_budget_3600.pt"]["scratch_path"]
        with torch.serialization.safe_globals(safe_globals):
            fixed_data = torch.load(fixed_path, map_location="cpu", weights_only=True)
        self.assertEqual(fixed_data["successful_updates"], 3600)
        self.assertEqual(fixed_data["skipped_amp_attempts_total"], 0)
        self.assertEqual(fixed_data["epoch"], 7)
        self.assertTrue(fixed_data["is_fixed_budget_endpoint"])
        self.assertEqual(fixed_data["metadata"]["objective"], "operation_ce")
        self.assertEqual(fixed_data["metadata"]["contract_sha256"], "4d99723e4b3f62f506165ced089eca82d4417279c76973b35470bd337cedfab2")
        self.assertEqual(fixed_data["metadata"]["split_manifest_sha256"], "7d6bfdf62bc6702b449048968f20e8c78df5b174be341b8d081667982980804b")

        # 2. best.pt
        best_path = artifacts["best.pt"]["scratch_path"]
        with torch.serialization.safe_globals(safe_globals):
            best_data = torch.load(best_path, map_location="cpu", weights_only=True)
        self.assertEqual(best_data["successful_updates"], 2700)
        self.assertEqual(best_data["skipped_amp_attempts_total"], 0)
        self.assertEqual(best_data["epoch"], 5)
        self.assertFalse(best_data["is_fixed_budget_endpoint"])
        self.assertAlmostEqual(best_data["best_ap"], 0.46497542138006975, places=6)

        # 3. last.pt
        last_path = artifacts["last.pt"]["scratch_path"]
        with torch.serialization.safe_globals(safe_globals):
            last_data = torch.load(last_path, map_location="cpu", weights_only=True)
        self.assertEqual(last_data["successful_updates"], 3600)
        self.assertEqual(last_data["skipped_amp_attempts_total"], 0)
        self.assertEqual(last_data["epoch"], 7)
        self.assertFalse(last_data["is_fixed_budget_endpoint"])

        # Parameter counts and finiteness
        head = fixed_data["head"]
        total_params = sum(p.numel() for p in head.values() if torch.is_tensor(p))
        self.assertEqual(total_params, 266667)
        for name, param in head.items():
            if torch.is_tensor(param):
                self.assertTrue(torch.isfinite(param).all().item(), f"Non-finite tensor: {name}")

    def test_speaker_partitions_and_zero_test_leakage(self):
        parts = self.receipt["payload_verification"]["data_partitions"]
        fit_spk = set(parts["fit_speakers"])
        sel_spk = set(parts["selection_speakers"])
        cal_spk = set(parts["calibration_speakers"])
        eval_spk = set(parts["evaluation_speakers"])
        test_spk = set(parts["sealed_test_speakers"])

        # Pairwise disjoint
        self.assertTrue(fit_spk.isdisjoint(sel_spk))
        self.assertTrue(fit_spk.isdisjoint(cal_spk))
        self.assertTrue(fit_spk.isdisjoint(eval_spk))
        self.assertTrue(fit_spk.isdisjoint(test_spk))
        self.assertTrue(eval_spk.isdisjoint(test_spk))
        self.assertEqual(len(fit_spk) + len(sel_spk) + len(cal_spk), 16)
        self.assertEqual(len(eval_spk), 2)
        self.assertEqual(len(test_spk), 6)


if __name__ == "__main__":
    unittest.main()
