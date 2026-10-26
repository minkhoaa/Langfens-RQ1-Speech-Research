#!/usr/bin/env python3
"""Stage C & D Verification Gate on Colab VM."""
import hashlib, os, shutil, sys
from pathlib import Path

EXPECTED_SHA256 = "b750eb3bf3971afccd9ab9058a64fa6b3f5781399b9ef0f1294d29bf580784a9"

def verify_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("[1/5] Checking Google Drive mount...")
    drive_mount = Path("/content/drive")
    if not drive_mount.exists():
        print("ERROR: /content/drive does not exist.")
        sys.exit(1)

    # Search for fold1_train_val_only.zip in Drive
    possible_paths = [
        Path("/content/drive/MyDrive/KLTN-Dataset/manifests/RQ1-TRN-02A/authorized_data/fold1_train_val_only.zip"),
        Path("/content/drive/My Drive/KLTN-Dataset/manifests/RQ1-TRN-02A/authorized_data/fold1_train_val_only.zip"),
    ]
    remote_zip = None
    for p in possible_paths:
        if p.exists():
            remote_zip = p
            break

    if remote_zip is None:
        # Search anywhere under /content/drive
        print("Searching for fold1_train_val_only.zip under /content/drive...")
        for root, dirs, files in os.walk("/content/drive"):
            if "fold1_train_val_only.zip" in files:
                remote_zip = Path(root) / "fold1_train_val_only.zip"
                break

    if remote_zip is None:
        print("ERROR: fold1_train_val_only.zip not found in Google Drive!")
        sys.exit(1)

    print(f"Found archive on Drive: {remote_zip} ({remote_zip.stat().st_size} bytes)")

    # Copy to ephemeral local VM disk
    local_dir = Path("/content/datasets/L2-ARCTIC")
    local_dir.mkdir(parents=True, exist_ok=True)
    local_zip = local_dir / "authorized_RQ1-TRN-02A_fold1_train_val_only.zip"

    print(f"[2/5] Copying archive from Drive to local VM disk: {local_zip}...")
    shutil.copyfile(str(remote_zip), str(local_zip))
    print(f"Copied {local_zip.stat().st_size} bytes to local VM disk.")

    print("[3/5] Verifying SHA-256 of transferred archive...")
    local_sha = verify_file_sha256(local_zip)
    print(f"Computed SHA-256: {local_sha}")
    if local_sha != EXPECTED_SHA256:
        print(f"ERROR: Checksum mismatch! Expected {EXPECTED_SHA256}, got {local_sha}")
        sys.exit(1)
    print("SHA-256 integrity verified!")

    print("[4/5] Testing Dataset Loader compatibility on Colab...")
    sys.path.insert(0, "/content/src")
    from langfens_prep.dataset import L2ArcticMDDDataset
    manifest = "/content/manifests/RQ1-SPLIT-01/l2arctic_4fold_splits.json"
    train_ds = L2ArcticMDDDataset(manifest, fold=1, split="train", policy="strict", zip_path=str(local_zip))
    val_ds = L2ArcticMDDDataset(manifest, fold=1, split="val", policy="strict", zip_path=str(local_zip))
    print(f"Loaded train_ds={len(train_ds)}, val_ds={len(val_ds)}")
    assert len(train_ds) == 2399 and len(val_ds) == 300
    _ = train_ds[0]
    _ = val_ds[0]
    print("Dataset loader: 100% PASS!")

    print("[5/5] Testing Drive checkpoint persistence...")
    drive_ckpt_dir = remote_zip.parent.parent / "checkpoints"
    drive_ckpt_dir.mkdir(parents=True, exist_ok=True)
    test_probe = drive_ckpt_dir / ".write_probe.tmp"
    test_probe.write_text("LANGFENS DRIVE PERSISTENCE TEST")
    if test_probe.exists() and test_probe.read_text() == "LANGFENS DRIVE PERSISTENCE TEST":
        test_probe.unlink()
        print(f"Drive checkpoint write access verified at: {drive_ckpt_dir}")
    else:
        print("ERROR: Drive write test failed!")
        sys.exit(1)

    print("\nALL STAGE C GATES PASSED! Ready for Stage D / E.")

if __name__ == "__main__":
    main()
