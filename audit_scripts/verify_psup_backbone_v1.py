"""Verify the cached pinned public backbone against immutable HF LFS metadata.

Metadata read only; no dependency installation, model download or model execution.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from huggingface_hub import HfApi
    from transformers.utils.hub import cached_file
    repo='facebook/wav2vec2-base-960h'
    revision='22aad52d435eb6dbaf354bdad9b0da84ce7d6156'
    info=HfApi().model_info(repo,revision=revision,files_metadata=True)
    assert info.sha==revision
    siblings={s.rfilename:s for s in info.siblings}
    artifacts={}
    for name in ['config.json','model.safetensors']:
        path=Path(cached_file(repo,name,revision=revision,local_files_only=True))
        actual=sha(path)
        expected=siblings[name].lfs.sha256 if siblings[name].lfs else None
        if expected is not None:
            assert actual==expected,(name,actual,expected)
        artifacts[name]=dict(local_path=str(path),resolved_path=str(path.resolve()),
            bytes=path.stat().st_size,sha256=actual,lfs_sha256=expected,
            verification='LFS_SHA256_MATCH' if expected else 'PINNED_CONFIG_LOCAL_HASH_ONLY')
    result=dict(status='PINNED_BACKBONE_WEIGHT_LFS_VERIFIED',repository=repo,revision=revision,
        artifacts=artifacts,verification_script_sha256=sha(__file__),
        interpreter=sys.executable,python=platform.python_version(),platform=platform.platform(),
        dependencies={p:importlib.metadata.version(p) for p in ['torch','transformers','numpy','soundfile','scipy','huggingface-hub']},
        model_executed=False,downloaded_model=False,optimizer_steps=0)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
