"""Atomically persist benchmark-only files to the approved PRIVATE HF repository.

Never creates a repo, changes visibility, replaces an existing prefix, or uploads
weights/audio. Checks remote privacy before commit and cryptographic readback of
every uploaded file at the returned immutable revision. Uses existing HF auth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

APPROVED_REPO = "minkhoaa/langfens-rq1-cross-fold-checkpoints"
ALLOWED_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".md", ".txt", ".log"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inventory(source):
    files = []
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            raise ValueError(f"Symlink refused: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        permitted_type = (path.suffix.lower() in ALLOWED_SUFFIXES
                          or path.name.endswith('.jsonl.gz') or path.name == 'SHA256SUMS')
        if not permitted_type or any(
            word in relative.lower() for word in ('credential', 'secret', '.env', 'token.txt', 'token.json', 'hf_token')):
            raise ValueError(f"Non-benchmark artifact refused: {relative}")
        files.append(dict(path=relative, sha256=sha256(path), bytes=path.stat().st_size))
    if not files:
        raise ValueError('Empty source')
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--remote-prefix', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    source = args.source_dir.resolve(strict=True)
    prefix = args.remote_prefix.rstrip('/')
    if not prefix.startswith('benchmarks/psup-v1/') or '..' in PurePosixPath(prefix).parts or prefix.startswith('/'):
        raise ValueError('Only a new versioned benchmarks/psup-v1/ prefix is allowed')
    if args.receipt.exists() or args.receipt.resolve().is_relative_to(source):
        raise ValueError('Receipt must be new and outside the upload source')
    files = inventory(source)
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    api = HfApi()
    info = api.model_info(APPROVED_REPO)
    if info.private is not True:
        raise RuntimeError('Approved repository is not verified private; no upload')
    remote_files = api.list_repo_files(APPROVED_REPO, revision=info.sha)
    if any(path == prefix or path.startswith(prefix+'/') for path in remote_files):
        raise FileExistsError('Remote prefix already exists; preserve it and use a new version')
    # Parent revision avoids concurrent lost updates; conflict safely requires a rerun.
    commit = api.create_commit(repo_id=APPROVED_REPO,
        parent_commit=info.sha, commit_message=f'Persist private benchmark evidence: {prefix}',
        operations=[CommitOperationAdd(path_in_repo=f'{prefix}/{item["path"]}',
            path_or_fileobj=str(source/item['path'])) for item in files])
    if api.model_info(APPROVED_REPO, revision=commit.oid).private is not True:
        raise RuntimeError('Privacy could not be confirmed after commit; no durability PASS')
    if inventory(source) != files:
        raise RuntimeError('Source changed during upload; no durability PASS')
    for item in files:
        remote = hf_hub_download(APPROVED_REPO, filename=f'{prefix}/{item["path"]}',
            revision=commit.oid, force_download=True)
        if sha256(remote) != item['sha256']:
            raise RuntimeError(f'Readback hash mismatch: {item["path"]}')
    receipt = dict(status='VERIFIED_PRIVATE_SHA256_READBACK', repository=APPROVED_REPO,
        private=True, revision_before=info.sha, revision=commit.oid,
        prefix=prefix, files=files, source_directory=str(source),
        source_script_sha256=sha256(__file__))
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: receipt[key] for key in ('status','repository','revision','prefix')}))


if __name__ == '__main__':
    main()
