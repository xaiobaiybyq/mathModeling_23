"""Recompute word pooling from saved raw features, without rerunning models.

This preserves extraction provenance and records the changed aggregation code.
Useful when changing only numerical precision of pool_intervals.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from alignment import pool_intervals

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    cfg=json.loads((ROOT/'config.json').read_text('utf-8'))
    fingerprint=hashlib.sha256((json.dumps(cfg,sort_keys=True)+digest(ROOT/'pipeline.py')+digest(ROOT/'alignment.py')+digest(ROOT/'models/model_manifest.json')).encode()).hexdigest()
    environment=json.loads((OUT/'run_environment.json').read_text('utf-8'))
    previous_code=environment['code_sha256'].copy()
    manifest=json.loads((OUT/'manifest.json').read_text('utf-8'))
    with (OUT/'reaggregation_log.jsonl').open('a',encoding='utf-8') as log:
        for row in manifest:
            p=OUT/'metadata'/(row['id']+'.json')
            meta=json.loads(p.read_text('utf-8'))
            path=OUT/meta['feature_file']
            assert digest(path)==meta['feature_sha256']
            with np.load(path,allow_pickle=False) as data:
                x={k:data[k] for k in data.files}
            prior_hash=meta['feature_sha256']
            for modal in ['audio','vision']:
                feature,mask,mapping=pool_intervals(x['word_intervals'],x[modal+'_intervals'],x[modal+'_raw'],x[modal+'_raw_mask'])
                assert mapping==meta[modal+'_source_indices_per_word']
                assert np.array_equal(mask,x[modal+'_mask'])
                x[modal]=feature
            np.savez_compressed(path,**x)
            meta['feature_sha256']=digest(path)
            meta['processing_fingerprint']=fingerprint
            meta['postprocessing']={'operation':'float64 accumulation for word pooling; stored as float32',
                                    'previous_feature_sha256':prior_hash,
                                    'aggregation_code_sha256':digest(ROOT/'alignment.py')}
            p.write_text(json.dumps(meta,ensure_ascii=False,indent=2),'utf-8')
            log.write(json.dumps({'id':row['id'],**meta['postprocessing'],'feature_sha256':meta['feature_sha256']})+'\n')
    environment['code_sha256']['alignment.py']=digest(ROOT/'alignment.py')
    environment['postprocessing']={'script':'reaggregate.py','script_sha256':digest(Path(__file__)),
                                  'original_extraction_code_sha256':previous_code,
                                  'operation':'recomputed only word-level pooling from unchanged raw arrays, using float64 accumulators'}
    (OUT/'run_environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2),'utf-8')
    print('Reaggregated all 100 samples; raw features, times and masks unchanged.')

if __name__=='__main__':
    main()
