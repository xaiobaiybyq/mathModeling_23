"""Question 1: extract all 100 supplied videos without label-based fitting.

All times are seconds relative to the source container start. NPZ uses no pickle.
Run prepare_models.py before this script. Inference itself is entirely offline.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import sysconfig
import tempfile
import time
import traceback

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Some managed Windows shells omit this standard architecture variable.
# Derive it from Python's wheel platform, never from a guessed machine type.
if os.name == 'nt' and sysconfig.get_platform() == 'win-amd64':
    os.environ.setdefault('PROCESSOR_ARCHITECTURE', 'AMD64')
import av
import numpy as np
import opensmile
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from transformers import AutoModel, AutoTokenizer, AutoModelForCTC, AutoProcessor
import mediapipe as mp
from alignment import tokenize_words, ctc_viterbi, pool_intervals, complement_intervals

ROOT = Path(__file__).resolve().parent
NUMBER_NORMALIZATION = {
    '2008': 'TWO THOUSAND EIGHT', '1989': 'NINETEEN EIGHTY NINE',
    '10th': 'TENTH', '1,500': 'FIFTEEN HUNDRED',
    '100,000': 'ONE HUNDRED THOUSAND', '20,000': 'TWENTY THOUSAND', '50': 'FIFTY'
}

def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), 'utf-8')

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(1024*1024):
            h.update(block)
    return h.hexdigest()

def read_manifest(data_root):
    import openpyxl
    sheets = list(data_root.rglob('label-100.xlsx'))
    if len(sheets) != 1:
        raise ValueError(f'Expected exactly one label-100.xlsx, found {len(sheets)}')
    source = sheets[0]
    wb = openpyxl.load_workbook(source, read_only=True, data_only=True)
    values = iter(wb.active.values)
    columns = next(values)
    rows = []
    for values_row in values:
        if all(v is None for v in values_row):
            continue
        r = dict(zip(columns, values_row))
        video_id = str(r['video_id'])
        clip = str(r['clip_id'])
        if clip.endswith('.0'):
            clip = clip[:-2]
        filename = source.parent / video_id / (clip+'.mp4')
        if not filename.is_file():
            raise FileNotFoundError(f'Missing source video: {video_id}/{clip}.mp4')
        label = float(r['label'])
        annotation = 'Positive' if label > 0 else 'Negative' if label < 0 else 'Neutral'
        if annotation != r['annotation'] or not -3 <= label <= 3:
            raise ValueError('Invalid source label: '+video_id+'/'+clip)
        rows.append({'id': video_id+'$_$'+clip, 'video_id': video_id, 'clip_id': clip,
                     'source_video': video_id+'/'+clip+'.mp4', 'path': filename,
                     'raw_text': str(r['text']), 'label': label, 'annotation': annotation})
    wb.close()
    if len(rows) != 100 or len({r['id'] for r in rows}) != 100:
        raise ValueError('Attachment 1 must have exactly 100 unique samples')
    actual = {p.resolve() for p in source.parent.rglob('*.mp4')}
    if actual != {r['path'].resolve() for r in rows}:
        raise ValueError('Video set and spreadsheet rows differ')
    return rows, source

def media_audio(path, sample_rate):
    """Decode PTS-aware audio; preserve video/audio offsets and missing intervals."""
    with av.open(str(path)) as c:
        origin = (c.start_time or 0)/av.time_base
        duration = c.duration/av.time_base if c.duration else 0.0
        video = c.streams.video[0]
        info = {'duration_seconds': duration, 'container_origin_seconds': origin,
                'video_fps': float(video.average_rate or 0),
                'video_width': video.width, 'video_height': video.height,
                'audio_source_rate': c.streams.audio[0].rate if c.streams.audio else None}
        pieces = []
        if c.streams.audio:
            resampler = av.AudioResampler(format='fltp', layout='mono', rate=sample_rate)
            cursor = 0.0
            for frame in c.decode(audio=0):
                for converted in resampler.resample(frame):
                    start = float(converted.pts * converted.time_base)-origin if converted.pts is not None else cursor
                    data = converted.to_ndarray().reshape(-1).astype(np.float32)
                    pieces.append((start, data))
                    cursor = start+len(data)/sample_rate
            for converted in resampler.resample(None):
                start = float(converted.pts * converted.time_base)-origin if converted.pts is not None else cursor
                data = converted.to_ndarray().reshape(-1).astype(np.float32)
                pieces.append((start, data))
                cursor = start+len(data)/sample_rate
        duration = max(duration, max((s+len(x)/sample_rate for s,x in pieces), default=0))
    if duration <= 0:
        raise ValueError('No positive media duration')
    wave = np.zeros(int(np.ceil(duration*sample_rate)), np.float32)
    observed = np.zeros(len(wave), bool)
    for start, data in pieces:
        a = round(start*sample_rate)
        left, right = max(a, 0), min(a+len(data), len(wave))
        if right > left:
            wave[left:right] = data[left-a:right-a]
            observed[left:right] = True
    info['duration_seconds'] = len(wave)/sample_rate
    info['audio_observed_seconds'] = float(observed.sum()/sample_rate)
    return wave, observed, info

class Extractor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fingerprint = hashlib.sha256((json.dumps(cfg, sort_keys=True)+sha256(ROOT/'pipeline.py')+sha256(ROOT/'alignment.py')+sha256(ROOT/'models/model_manifest.json')).encode()).hexdigest()
        torch.set_num_threads(cfg['torch_threads'])
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        self.text_tokenizer = AutoTokenizer.from_pretrained(ROOT/'models/text', local_files_only=True)
        self.text_model = AutoModel.from_pretrained(ROOT/'models/text', local_files_only=True).eval()
        self.processor = AutoProcessor.from_pretrained(ROOT/'models/alignment', local_files_only=True)
        self.ctc = AutoModelForCTC.from_pretrained(ROOT/'models/alignment', local_files_only=True).eval()
        self.vocab = self.processor.tokenizer.get_vocab()
        # openSMILE's C wrapper accepts ASCII config paths only on Windows.
        # Copy its unchanged config tree to a temporary ASCII path; raw videos
        # stay in place and are passed as in-memory audio arrays.
        self.smile_temp = tempfile.TemporaryDirectory(prefix='q1_opensmile_')
        smile_root = Path(self.smile_temp.name)/'config'
        shutil.copytree(Path(opensmile.__file__).parent/'core/config', smile_root)
        class PortableSmile(opensmile.Smile):
            @property
            def default_config_root(self):
                return str(smile_root)
        self.smile = PortableSmile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                   feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=(ROOT/'models/face_landmarker.task').read_bytes()),
            running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=3,
            min_face_detection_confidence=cfg['face_detection_threshold'],
            min_face_presence_confidence=cfg['face_presence_threshold'],
            output_face_blendshapes=True, output_facial_transformation_matrixes=True)
        self.face = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self.vision_names = None

    def text_features(self, words):
        output = np.zeros((len(words), self.text_model.config.dim), np.float32)
        mapping = []
        size = self.cfg['text_chunk_words']
        for first in range(0, len(words), size):
            chunk = words[first:first+size]
            encoded = self.text_tokenizer(chunk, is_split_into_words=True, return_tensors='pt', truncation=False)
            if encoded.input_ids.shape[1] > self.text_model.config.max_position_embeddings:
                raise ValueError('Text chunk exceeds model length; reduce text_chunk_words')
            word_ids = encoded.word_ids()
            with torch.inference_mode():
                h = self.text_model(**encoded).last_hidden_state[0].numpy()
            for j in range(len(chunk)):
                idx = [k for k,w in enumerate(word_ids) if w == j]
                if not idx:
                    raise ValueError('Word has no text subtoken')
                output[first+j] = h[idx].mean(0)
                mapping.append({'word_index': first+j, 'chunk_first_word': first,
                                'subtoken_positions': idx,
                                'subtoken_ids': encoded.input_ids[0, idx].tolist()})
        return output, mapping

    def align(self, wave, words, sample_rate):
        # The original text is never overwritten. CTC expects uppercase English.
        normalized, owners, target = [], [], []
        for i, word in enumerate(words):
            norm = NUMBER_NORMALIZATION.get(word, word.upper().replace('’', "'")).replace(' ', '|')
            # Digits need explicit spoken-form normalization, never silent deletion.
            if any(c not in self.vocab for c in norm):
                raise ValueError('Unsupported CTC word; provide explicit normalization: '+word)
            if i:
                target.append(self.vocab['|'])
                owners.append(-1)
            normalized.append(norm)
            target.extend(self.vocab[c] for c in norm)
            owners.extend([i]*len(norm))
        encoded = self.processor(wave, sampling_rate=sample_rate, return_tensors='pt')
        with torch.inference_mode():
            logits = self.ctc(**encoded).logits[0]
            log_probs = logits.log_softmax(-1).numpy()
        path, spans, confidence = ctc_viterbi(log_probs, target, self.ctc.config.pad_token_id)
        # Conv kernels define the receptive-field center; CTC boundaries are
        # midpoint cell boundaries around those centers (about 20 ms resolution).
        stride, receptive = 1, 1
        for kernel, step in zip(self.ctc.config.conv_kernel, self.ctc.config.conv_stride):
            receptive += (kernel-1)*stride
            stride *= step
        center0 = (receptive-1)/2/sample_rate
        step_s = stride/sample_rate
        boundaries = np.clip(center0+(np.arange(len(log_probs)+1)-.5)*step_s, 0, len(wave)/sample_rate)
        times, scores = [], []
        owners = np.asarray(owners)
        for i in range(len(words)):
            chars = np.flatnonzero(owners == i)
            start, end = spans[chars[0],0], spans[chars[-1],1]
            times.append((float(boundaries[start]), float(boundaries[end])))
            scores.append(float(np.exp(np.log(np.maximum(confidence[chars], 1e-30)).mean())))
        greedy = self.processor.batch_decode(logits.argmax(-1).unsqueeze(0))[0]
        return np.asarray(times, np.float32), np.asarray(scores, np.float32), {
            'normalized_words': normalized, 'greedy_transcript_for_qc_only': greedy,
            'ctc_step_seconds': step_s, 'ctc_receptive_field_seconds': receptive/sample_rate,
            'confidence_definition': 'geometric mean of character emission probabilities along forced CTC path; not calibrated boundary accuracy'}

    def audio_features(self, wave, observed, rate):
        frame = self.smile.process_signal(wave, rate)
        starts = frame.index.get_level_values('start').total_seconds().to_numpy()
        ends = frame.index.get_level_values('end').total_seconds().to_numpy()
        times = np.column_stack((starts, ends)).astype(np.float32)
        data = frame.to_numpy(dtype=np.float32)
        cumulative = np.r_[0, np.cumsum(observed, dtype=np.int64)]
        a = np.clip(np.rint(starts*rate).astype(int), 0, len(observed))
        b = np.clip(np.rint(ends*rate).astype(int), 0, len(observed))
        valid = np.isfinite(data).all(1) & ((cumulative[b]-cumulative[a]) >= .95*np.maximum(b-a,1))
        data[~valid] = 0
        return data, times, valid, list(frame.columns)

    def visual_features(self, path, origin, duration):
        fps = self.cfg['video_sample_hz']
        frames, timestamps, ids, valid, counts, boxes = [], [], [], [], [], []
        next_time = 0.0
        with av.open(str(path)) as container:
            for frame_index, frame in enumerate(container.decode(video=0)):
                if frame.pts is None:
                    raise ValueError('Video frame without PTS')
                stamp = float(frame.pts*frame.time_base)-origin
                if stamp < next_time-1e-6:
                    continue
                next_time = (np.floor(stamp*fps)+1)/fps
                rgb = frame.to_ndarray(format='rgb24')
                if rgb.shape[1] > self.cfg['video_max_width']:
                    width = self.cfg['video_max_width']
                    rgb = np.asarray(Image.fromarray(rgb).resize((width, round(rgb.shape[0]*width/rgb.shape[1]))))
                result = self.face.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
                count = len(result.face_landmarks)
                feature = np.zeros(55, np.float32)
                box = [0.,0.,0.,0.]
                ok = False
                if count:
                    all_boxes = []
                    for landmarks in result.face_landmarks:
                        xy = np.array([(p.x, p.y) for p in landmarks])
                        all_boxes.append([*xy.min(0), *xy.max(0)])
                    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in all_boxes]
                    selected = int(np.argmax(areas))
                    box = all_boxes[selected]
                    blend = result.face_blendshapes[selected]
                    names = [v.category_name for v in blend]
                    if self.vision_names is None:
                        self.vision_names = names+['head_euler_x_rad','head_euler_y_rad','head_euler_z_rad']
                    if names != self.vision_names[:52]:
                        raise ValueError('Unexpected face blendshape ordering')
                    matrix = np.asarray(result.facial_transformation_matrixes[selected])[:3,:3]
                    u, _, vh = np.linalg.svd(matrix)
                    rot = u@vh
                    if np.linalg.det(rot) < 0:
                        u[:,-1] *= -1
                        rot = u@vh
                    angles = Rotation.from_matrix(rot).as_euler('xyz')
                    feature = np.r_[[v.score for v in blend], angles].astype(np.float32)
                    ok = bool(np.isfinite(feature).all())
                frames.append(feature if ok else np.zeros(55, np.float32))
                timestamps.append(max(stamp, 0))
                ids.append(frame_index)
                valid.append(ok)
                counts.append(count)
                boxes.append(box)
        t = np.asarray(timestamps, np.float32)
        if not len(t):
            raise ValueError('No decoded video frames')
        # Sample-and-hold cells bounded by adjacent sampled PTS midpoints. Cap
        # support at 0.075 s either side so large decoding gaps remain missing.
        edges = np.r_[max(0, t[0]-.5/fps), (t[:-1]+t[1:])/2, min(duration, t[-1]+.5/fps)]
        intervals = np.column_stack((np.maximum(edges[:-1], t-.75/fps), np.minimum(edges[1:],t+.75/fps)))
        return np.asarray(frames), intervals.astype(np.float32), np.asarray(valid), t, np.asarray(ids), np.asarray(counts), np.asarray(boxes,np.float32)

def process_one(extractor, row, out):
    begin = time.perf_counter()
    cfg = extractor.cfg
    sample_id = row['id']
    wave, observed, info = media_audio(row['path'], cfg['sample_rate'])
    duration = info['duration_seconds']
    words, offsets = tokenize_words(row['raw_text'])
    text, subwords = extractor.text_features(words)
    issues = []
    if any(any(c.isdigit() for c in word) for word in words):
        issues.append('numeric_pronunciation_needs_review')
    try:
        intervals, scores, alignment_info = extractor.align(wave, words, cfg['sample_rate'])
        alignment_mask = np.ones(len(words), bool)
    except ValueError as error:
        # Retain all text positions and raw modal features, explicitly invalid.
        intervals = np.zeros((len(words),2), np.float32)
        scores = np.zeros(len(words), np.float32)
        alignment_mask = np.zeros(len(words), bool)
        alignment_info = {'error': str(error)}
        issues.append('alignment_failed')
    audio_raw, audio_times, audio_valid, audio_names = extractor.audio_features(wave, observed, cfg['sample_rate'])
    vision_raw, vision_times, vision_valid, vision_pts, frame_ids, face_counts, face_boxes = extractor.visual_features(row['path'], info['container_origin_seconds'], duration)
    audio, audio_mask, audio_map = pool_intervals(intervals, audio_times, audio_raw, audio_valid)
    vision, vision_mask, vision_map = pool_intervals(intervals, vision_times, vision_raw, vision_valid)
    low_conf = (scores < cfg['alignment_review_threshold']) | ~alignment_mask
    if low_conf.any():
        issues.append('low_alignment_confidence')
    if not vision_mask.all():
        issues.append('some_words_without_valid_face')
    if (face_counts > 1).any():
        issues.append('multiple_faces_largest_face_selected')
    if not audio_mask.all():
        issues.append('some_words_without_valid_audio')
    gaps = complement_intervals(intervals[alignment_mask], duration)
    arrays = {
        'text': text, 'audio': audio, 'vision': vision,
        'word_intervals': intervals, 'word_char_offsets': offsets,
        'alignment_confidence': scores, 'alignment_mask': alignment_mask,
        'alignment_review_mask': low_conf, 'text_mask': np.ones(len(words),bool),
        'audio_mask': audio_mask, 'vision_mask': vision_mask,
        'padding_mask': np.zeros(len(words),bool), 'valid_length': np.array(len(words),np.int32),
        'audio_raw': audio_raw, 'audio_intervals': audio_times, 'audio_raw_mask': audio_valid,
        'vision_raw': vision_raw, 'vision_intervals': vision_times, 'vision_raw_mask': vision_valid,
        'vision_pts': vision_pts, 'vision_frame_ids': frame_ids.astype(np.int32),
        'face_counts': face_counts.astype(np.int16), 'face_boxes': face_boxes,
        'unassigned_intervals': gaps
    }
    for name, x in arrays.items():
        if np.issubdtype(x.dtype,np.floating) and not np.isfinite(x).all():
            raise ValueError('Non-finite output: '+name)
    feature_path = out/'features'/(sample_id+'.npz')
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(feature_path, **arrays)
    metadata = {k:v for k,v in row.items() if k != 'path'}
    metadata.update(info)
    metadata.update({'source_sha256': sha256(row['path']), 'words': words,
                     'processing_fingerprint': extractor.fingerprint,
                     'subword_mapping': subwords, 'alignment': alignment_info,
                     'numeric_normalization': NUMBER_NORMALIZATION,
                     'audio_source_indices_per_word': audio_map, 'vision_source_indices_per_word': vision_map,
                     'feature_file': 'features/'+feature_path.name,
                     'feature_sha256': sha256(feature_path),
                     'warnings': issues, 'padding_rule': 'ragged full sequence; no truncation or padding on disk',
                     'face_selection': 'largest detected face independently per sampled frame; not verified active speaker',
                     'status': 'review_required' if issues else 'ok',
                     'processing_seconds': round(time.perf_counter()-begin,3)})
    dump(out/'metadata'/(sample_id+'.json'), metadata)
    names = {'text': [f'distilbert_{i:03d}' for i in range(text.shape[1])],
             'audio_raw': audio_names, 'vision_raw': extractor.vision_names,
             'audio': ['mean_'+x for x in audio_names]+['std_'+x for x in audio_names],
             'vision': None if extractor.vision_names is None else ['mean_'+x for x in extractor.vision_names]+['std_'+x for x in extractor.vision_names]}
    dump(out/'feature_names.json', names)
    return metadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=ROOT.parent/'E题')
    parser.add_argument('--output', type=Path, default=ROOT/'results')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = json.loads((ROOT/'config.json').read_text('utf-8'))
    rows, source = read_manifest(args.data_root)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    dump(out/'manifest.json', [{k:v for k,v in r.items() if k!='path'} for r in rows])
    versions = {}
    for name in ['numpy','scipy','av','opensmile','mediapipe','transformers','torch','pandas','openpyxl','pillow']:
        versions[name] = importlib.metadata.version(name)
    dump(out/'run_environment.json', {'python':platform.python_version(), 'platform':platform.system(),
         'packages':versions, 'config':cfg, 'source_table_sha256':sha256(source),
         'code_sha256':{p.name:sha256(p) for p in [ROOT/'pipeline.py',ROOT/'alignment.py']},
         'models':json.loads((ROOT/'models/model_manifest.json').read_text('utf-8'))})
    extractor = Extractor(cfg)
    selected = rows[:args.limit] if args.limit else rows
    failures = 0
    with (out/'processing_log.jsonl').open('a', encoding='utf-8') as log:
        for index, row in enumerate(selected):
            meta_path = out/'metadata'/(row['id']+'.json')
            if args.resume and meta_path.exists():
                previous = json.loads(meta_path.read_text('utf-8'))
                feature = out/previous['feature_file']
                if feature.exists() and previous.get('processing_fingerprint') == extractor.fingerprint and previous['source_sha256'] == sha256(row['path']) and previous['feature_sha256'] == sha256(feature):
                    print(f'[{index+1}/{len(selected)}] cached {row["id"]}',flush=True)
                    continue
            print(f'[{index+1}/{len(selected)}] extracting {row["id"]}',flush=True)
            try:
                result = process_one(extractor,row,out)
                record = {'id':row['id'],'status':result['status'],'seconds':result['processing_seconds'],'warnings':result['warnings']}
            except Exception as e:
                failures += 1
                record = {'id':row['id'],'status':'failed','error_type':type(e).__name__,'error':str(e)}
                print(traceback.format_exc(),flush=True)
            log.write(json.dumps(record,ensure_ascii=False)+'\n')
            log.flush()
            print(json.dumps(record,ensure_ascii=False),flush=True)
    extractor.face.close()
    extractor.smile_temp.cleanup()
    if failures:
        raise SystemExit(f'{failures} samples failed; see processing_log.jsonl')

if __name__ == '__main__':
    main()
