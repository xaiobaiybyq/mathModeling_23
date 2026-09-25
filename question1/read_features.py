"""Read NPZ without pickle and collate full variable-length sequences."""
from pathlib import Path
import json
import numpy as np

def read_sample(results, sample_id):
    results = Path(results)
    metadata = json.loads((results/'metadata'/(sample_id+'.json')).read_text('utf-8'))
    with np.load(results/metadata['feature_file'],allow_pickle=False) as f:
        arrays = {k:f[k] for k in f.files}
    return arrays, metadata

def collate(samples):
    """samples: list of dictionaries returned as read_sample(...)[0].

    padding_mask=True means padding; modal masks=True means observed/valid.
    alignment_mask must additionally be checked when using time alignment.
    """
    if not samples:
        raise ValueError('Empty batch')
    lengths = np.array([int(s['valid_length']) for s in samples],np.int32)
    maximum = int(lengths.max())
    batch = {'lengths':lengths,'padding_mask':np.ones((len(samples),maximum),bool)}
    for key in ['text','audio','vision']:
        batch[key] = np.zeros((len(samples),maximum,samples[0][key].shape[1]),np.float32)
    for key in ['text_mask','audio_mask','vision_mask','alignment_mask','alignment_review_mask']:
        batch[key] = np.zeros((len(samples),maximum),bool)
    batch['word_intervals'] = np.full((len(samples),maximum,2),-1,np.float32)
    for i,(sample,n) in enumerate(zip(samples,lengths)):
        batch['padding_mask'][i,:n] = False
        for key in ['text','audio','vision','text_mask','audio_mask','vision_mask','alignment_mask','alignment_review_mask','word_intervals']:
            batch[key][i,:n] = sample[key]
    return batch

if __name__ == '__main__':
    root=Path(__file__).resolve().parent/'results'
    manifest=json.loads((root/'manifest.json').read_text('utf-8'))
    samples=[read_sample(root,x['id'])[0] for x in manifest[:2]]
    for k,v in collate(samples).items():
        print(k,v.shape,v.dtype)
