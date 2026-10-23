import os
import sys
import zipfile
import tarfile
import hashlib
import json
from collections import Counter

def audit_zip(zip_path):
    print(f"=== Auditing ZIP: {zip_path} ===")
    stat = os.stat(zip_path)
    file_size = stat.st_size
    print(f"File size: {file_size} bytes")

    # Compute SHA-256 and MD5
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    with open(zip_path, "rb") as f:
        while chunk := f.read(1024 * 1024 * 16):
            sha256.update(chunk)
            md5.update(chunk)
    sha256_hex = sha256.hexdigest()
    md5_hex = md5.hexdigest()
    print(f"SHA-256: {sha256_hex}")
    print(f"MD5: {md5_hex}")

    zf = zipfile.ZipFile(zip_path, 'r')
    infolist = zf.infolist()
    total_members = len(infolist)
    print(f"Total members: {total_members}")

    extensions = Counter()
    total_uncompressed = 0
    total_compressed = 0
    unsafe_entries = []
    seen_names = set()
    duplicates = []
    top_level_dirs = set()

    for info in infolist:
        name = info.filename
        if name in seen_names:
            duplicates.append(name)
        seen_names.add(name)

        if name.startswith("/") or ".." in name:
            unsafe_entries.append(name)

        parts = name.strip("/").split("/")
        if parts:
            top_level_dirs.add(parts[0])

        total_uncompressed += info.file_size
        total_compressed += info.compress_size

        if not info.is_dir():
            _, ext = os.path.splitext(name)
            extensions[ext.lower()] += 1
        else:
            extensions["<DIR>"] += 1

    print(f"Top level entries: {top_level_dirs}")
    print(f"Extensions breakdown: {dict(extensions)}")
    print(f"Total uncompressed size: {total_uncompressed} bytes")
    print(f"Total compressed size: {total_compressed} bytes")
    print(f"Duplicates: {len(duplicates)}")
    print(f"Unsafe entries: {len(unsafe_entries)}")

    print("Running testzip() for CRC validation across all members...")
    first_bad = zf.testzip()
    if first_bad is None:
        print("CRC verification: ALL MEMBERS PASS (testzip() returned None)")
        crc_status = "PASS"
    else:
        print(f"CRC verification: FAILED on member {first_bad}")
        crc_status = f"FAIL (member: {first_bad})"

    zf.close()
    return {
        "file_size": file_size,
        "sha256": sha256_hex,
        "md5": md5_hex,
        "total_members": total_members,
        "extensions": dict(extensions),
        "total_uncompressed_bytes": total_uncompressed,
        "total_compressed_bytes": total_compressed,
        "duplicates_count": len(duplicates),
        "unsafe_count": len(unsafe_entries),
        "top_level": list(top_level_dirs),
        "crc_status": crc_status
    }

def audit_targz(tar_path):
    print(f"\n=== Auditing TAR.GZ: {tar_path} ===")
    stat = os.stat(tar_path)
    file_size = stat.st_size
    print(f"File size: {file_size} bytes")

    # Compute SHA-256 and MD5
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    with open(tar_path, "rb") as f:
        while chunk := f.read(1024 * 1024 * 16):
            sha256.update(chunk)
            md5.update(chunk)
    sha256_hex = sha256.hexdigest()
    md5_hex = md5.hexdigest()
    print(f"SHA-256: {sha256_hex}")
    print(f"MD5: {md5_hex}")

    tf = tarfile.open(tar_path, "r:gz")
    total_members = 0
    total_uncompressed = 0
    extensions = Counter()
    unsafe_entries = []
    seen_names = set()
    duplicates = []
    top_level_dirs = set()

    for member in tf:
        total_members += 1
        name = member.name
        if name in seen_names:
            duplicates.append(name)
        seen_names.add(name)

        if name.startswith("/") or ".." in name:
            unsafe_entries.append(name)

        parts = name.strip("/").split("/")
        if parts:
            top_level_dirs.add(parts[0])

        total_uncompressed += member.size

        if member.isfile():
            _, ext = os.path.splitext(name)
            extensions[ext.lower()] += 1
        elif member.isdir():
            extensions["<DIR>"] += 1
        else:
            extensions["<OTHER>"] += 1

    tf.close()
    print(f"Total members: {total_members}")
    print(f"Top level entries: {top_level_dirs}")
    print(f"Extensions breakdown: {dict(extensions)}")
    print(f"Total uncompressed size: {total_uncompressed} bytes")
    print(f"Duplicates: {len(duplicates)}")
    print(f"Unsafe entries: {len(unsafe_entries)}")

    return {
        "file_size": file_size,
        "sha256": sha256_hex,
        "md5": md5_hex,
        "total_members": total_members,
        "extensions": dict(extensions),
        "total_uncompressed_bytes": total_uncompressed,
        "duplicates_count": len(duplicates),
        "unsafe_count": len(unsafe_entries),
        "top_level": list(top_level_dirs),
        "tar_stream_status": "PASS"
    }

if __name__ == "__main__":
    res_l2 = audit_zip("datasets/L2-ARCTIC/original/l2arctic_release_v5.0.zip")
    res_so = audit_targz("datasets/SpeechOcean762/original/speechocean762.tar.gz")
    with open("audit_scripts/archive_integrity_summary.json", "w") as f:
        json.dump({"l2arctic": res_l2, "speechocean762": res_so}, f, indent=2)
    print("\nSummary saved to audit_scripts/archive_integrity_summary.json")
