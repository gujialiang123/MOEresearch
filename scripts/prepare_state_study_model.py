#!/usr/bin/env python3
"""Resolve and download the original BF16 checkpoint at an immutable revision."""
import argparse
import datetime
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-id', default='Qwen/Qwen3-30B-A3B-Instruct-2507')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--out', required=True, help='Explicit model directory on a data volume, outside Home')
    args = parser.parse_args()
    target = Path(args.out).expanduser().resolve()
    if target.is_relative_to(Path.home().resolve()):
        raise RuntimeError('Model weights must be stored outside Home; choose a data volume such as /scratch/jgu7/models')
    os.environ.setdefault('HF_HOME', '/scratch/jgu7/cache/huggingface')
    from huggingface_hub import HfApi, snapshot_download
    target.mkdir(parents=True, exist_ok=True)
    provenance = target / 'study_download_provenance.json'
    if provenance.exists():
        previous = json.loads(provenance.read_text())
        if previous['model_id'] != args.model_id:
            raise RuntimeError('Existing download belongs to a different model')
        revision = previous['revision']
        if args.revision != 'main' and args.revision != revision:
            raise RuntimeError('Refusing to mix checkpoint revisions in one directory')
    else:
        revision = HfApi().model_info(args.model_id, revision=args.revision).sha
        provenance.write_text(json.dumps({
            'model_id': args.model_id, 'revision': revision,
            'resolved_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'source': 'official Hugging Face model repository',
            'historical_checkpoint_identity': 'Old local weight hashes unavailable; same model ID, current pinned official revision',
            'status': 'downloading',
        }, indent=2) + '\n')
    print(f'Downloading {args.model_id}@{revision} to {target}', flush=True)
    snapshot_download(repo_id=args.model_id, revision=revision, local_dir=str(target),
                      allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', '*.model', 'README.md'],
                      max_workers=8)
    record = json.loads(provenance.read_text())
    record['status'] = 'complete'
    record['files'] = {str(p.relative_to(target)): p.stat().st_size
                       for p in target.glob('*') if p.is_file() and p != provenance}
    provenance.write_text(json.dumps(record, indent=2) + '\n')
    print('Checkpoint download complete', flush=True)


if __name__ == '__main__':
    main()
