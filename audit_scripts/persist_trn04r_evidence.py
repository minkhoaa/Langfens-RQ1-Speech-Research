"""Persist a new TRN-04R evidence package atomically to approved private HF.

No weights/audio, overwrites, visibility changes or publication are permitted.
Every uploaded byte is downloaded at the immutable commit and SHA-256 verified.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath

from persist_psup_benchmark_v1 import APPROVED_REPO, inventory, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--remote-prefix', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    source = args.source_dir.resolve(strict=True)
    prefix = args.remote_prefix.rstrip('/')
    if not prefix.startswith('benchmarks/trn04r/') or '..' in PurePosixPath(prefix).parts or prefix.startswith('/'):
        raise ValueError('Only a new versioned benchmarks/trn04r/ prefix is allowed')
    if args.receipt.exists() or args.receipt.resolve().is_relative_to(source):
        raise ValueError('Receipt must be new and outside the upload source')
    files = inventory(source)
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    api = HfApi()
    info = api.model_info(APPROVED_REPO)
    if info.private is not True:
        raise RuntimeError('Approved repository is not verified private; no upload')
    existing = api.list_repo_files(APPROVED_REPO, revision=info.sha)
    if any(p == prefix or p.startswith(prefix + '/') for p in existing):
        raise FileExistsError('Prefix already exists; original evidence must be preserved')
    commit = api.create_commit(repo_id=APPROVED_REPO, parent_commit=info.sha,
        commit_message=f'Preserve private TRN-04R corrective evaluation: {prefix}',
        operations=[CommitOperationAdd(path_in_repo=f'{prefix}/{f["path"]}',
                                      path_or_fileobj=str(source / f['path'])) for f in files])
    if api.model_info(APPROVED_REPO, revision=commit.oid).private is not True:
        raise RuntimeError('Private readback not verified')
    if inventory(source) != files:
        raise RuntimeError('Local evidence changed during persistence')
    for f in files:
        downloaded = hf_hub_download(APPROVED_REPO, filename=f'{prefix}/{f["path"]}',
                                     revision=commit.oid, force_download=True)
        if sha256(downloaded) != f['sha256']:
            raise RuntimeError(f'SHA-256 readback mismatch: {f["path"]}')
    receipt = dict(status='VERIFIED_PRIVATE_SHA256_READBACK', repository=APPROVED_REPO,
        private=True, revision_before=info.sha, revision=commit.oid, prefix=prefix,
        source_directory=str(source), files=files, source_script_sha256=sha256(__file__),
        license='CC-BY-NC-4.0', no_weights_or_audio=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open('x') as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({k: receipt[k] for k in ('status', 'revision', 'prefix')}))


if __name__ == '__main__':
    main()
