"""Losslessly package new private evaluation evidence without changing originals."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    source=args.source.resolve(strict=True)
    output=args.output.resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError('Use a new independent output directory')
    output.mkdir(parents=True,mode=0o700)
    records=[]
    for original in sorted(source.rglob('*')):
        if original.is_symlink():
            raise ValueError('Symlinks forbidden')
        if not original.is_file():continue
        relative=original.relative_to(source)
        destination=output/relative
        if original.suffix=='.jsonl':destination=destination.with_suffix('.jsonl.gz')
        destination.parent.mkdir(parents=True,exist_ok=True)
        if original.suffix=='.jsonl':
            with original.open('rb') as src,destination.open('xb') as dst:
                with gzip.GzipFile(filename='',mode='wb',fileobj=dst,mtime=0,compresslevel=6) as zipped:
                    shutil.copyfileobj(src,zipped)
            with gzip.open(destination,'rb') as unzipped:
                assert hashlib.sha256(unzipped.read()).hexdigest()==sha(original)
        else:shutil.copyfile(original,destination)
        records.append(dict(original_path=relative.as_posix(),original_sha256=sha(original),
            original_bytes=original.stat().st_size,packaged_path=destination.relative_to(output).as_posix(),
            packaged_sha256=sha(destination),packaged_bytes=destination.stat().st_size))
    with (output/'LOSSLESS_PACKAGE_INDEX.json').open('x') as f:
        json.dump(dict(status='LOSSLESS_READBACK_VERIFIED',source=str(source),files=records,
            restoration='Decompress *.jsonl.gz with gzip to original_path; verify original_sha256 before reproduction.'),f,indent=2)
        f.write('\n')
    print(json.dumps(dict(files=len(records),original_bytes=sum(r['original_bytes'] for r in records),packaged_bytes=sum(r['packaged_bytes'] for r in records))))


if __name__=='__main__':main()
