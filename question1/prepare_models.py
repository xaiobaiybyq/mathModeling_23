"""Download only public pretrained weights; never upload the competition data."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import os
import urllib.request
import time

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'models'

def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=90) as response, path.with_suffix(path.suffix+'.part').open('wb') as out:
                while chunk := response.read(1024*1024):
                    out.write(chunk)
            path.with_suffix(path.suffix+'.part').replace(path)
            print('Downloaded', path.name, path.stat().st_size, flush=True)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)

def main():
    MODELS.mkdir(exist_ok=True)
    lock_path = MODELS / 'model_manifest.json'
    if lock_path.exists():
        manifest = json.loads(lock_path.read_text('utf-8'))
    else:
        manifest = {'models': {}}
    repositories = {
        'text': ('distilbert/distilbert-base-uncased', ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt']),
        'alignment': ('facebook/wav2vec2-base-960h', ['config.json', 'model.safetensors', 'preprocessor_config.json', 'tokenizer_config.json', 'vocab.json', 'special_tokens_map.json'])
    }
    jobs = []
    for name, (repo, files) in repositories.items():
        if name not in manifest['models']:
            with urllib.request.urlopen('https://huggingface.co/api/models/'+repo, timeout=60) as r:
                info = json.load(r)
            revision = info['sha']
            existing = {x['rfilename'] for x in info['siblings']}
            # Some repositories have only PyTorch weights.
            if 'model.safetensors' not in existing:
                files = ['pytorch_model.bin' if x == 'model.safetensors' else x for x in files]
            files = [x for x in files if x in existing]
            manifest['models'][name] = {'repository': repo, 'revision': revision, 'files': files}
        spec = manifest['models'][name]
        for filename in spec['files']:
            jobs.append(('https://huggingface.co/'+repo+'/resolve/'+spec['revision']+'/'+filename, MODELS/name/filename))
    face_url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
    manifest['face'] = {'url': face_url, 'version': 1}
    lock_path.write_text(json.dumps(manifest, indent=2), 'utf-8')
    jobs.append((face_url, MODELS/'face_landmarker.task'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda p: fetch(*p), jobs))
    previous_hashes = manifest.get('sha256', {})
    manifest['sha256'] = {}
    for _, path in jobs:
        digest = hashlib.sha256()
        with path.open('rb') as f:
            while block := f.read(1024*1024):
                digest.update(block)
        relative = path.relative_to(MODELS).as_posix()
        if relative in previous_hashes and previous_hashes[relative] != digest.hexdigest():
            raise ValueError('Model hash mismatch: '+relative+'; replace the corrupted file before proceeding')
        manifest['sha256'][relative] = digest.hexdigest()
    lock_path.write_text(json.dumps(manifest, indent=2), 'utf-8')
    print('All pretrained models downloaded and hashed.', flush=True)

if __name__ == '__main__':
    main()
