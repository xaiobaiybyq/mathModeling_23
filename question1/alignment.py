"""CTC Viterbi forced alignment and time-overlap pooling, independent of models."""
import re
import unicodedata
import numpy as np

def tokenize_words(text):
    # Preserve the source string separately; offsets refer to that source string.
    matches = list(re.finditer(r"\d+(?:,\d{3})*(?:st|nd|rd|th)?|[A-Za-z]+(?:['’][A-Za-z]+)*", text))
    words = [m.group() for m in matches]
    return words, np.asarray([(m.start(), m.end()) for m in matches], dtype=np.int32).reshape(-1, 2)

def ctc_viterbi(log_probs, target, blank=0):
    """Full CTC topology: repeated characters require a blank between them.

    Returns target-index path (-1 for blank) and per-target frame spans.
    Leading/trailing blank states allow untranscribed silence, not extra speech.
    """
    target = np.asarray(target, dtype=np.int64)
    if not len(target):
        raise ValueError('Empty CTC target')
    T, _ = log_probs.shape
    labels = np.full(2*len(target)+1, blank, dtype=np.int64)
    labels[1::2] = target
    S = len(labels)
    prev = np.full(S, -np.inf)
    prev[0] = log_probs[0, blank]
    prev[1] = log_probs[0, target[0]]
    back = np.zeros((T, S), dtype=np.uint8)
    allow_skip = np.zeros(S, dtype=bool)
    allow_skip[2:] = (labels[2:] != blank) & (labels[2:] != labels[:-2])
    for t in range(1, T):
        stay = prev
        step = np.r_[-np.inf, prev[:-1]]
        skip = np.r_[-np.inf, -np.inf, prev[:-2]]
        skip[~allow_skip] = -np.inf
        candidates = np.stack((stay, step, skip))
        choice = candidates.argmax(0)
        prev = candidates[choice, np.arange(S)] + log_probs[t, labels]
        back[t] = choice
    state = S-1 if prev[-1] >= prev[-2] else S-2
    if not np.isfinite(prev[state]):
        raise ValueError('No valid CTC path: audio shorter than target')
    path = np.empty(T, np.int32)
    for t in range(T-1, -1, -1):
        path[t] = state//2 if state % 2 else -1
        if t:
            state -= int(back[t, state])
    spans, scores = [], []
    for j, token in enumerate(target):
        frames = np.flatnonzero(path == j)
        if not len(frames):
            raise ValueError('CTC target was skipped')
        spans.append((int(frames[0]), int(frames[-1]+1)))
        scores.append(float(np.exp(log_probs[frames, token].mean())))
    return path, np.asarray(spans, np.int32), np.asarray(scores, np.float32)

def pool_intervals(intervals, frame_intervals, features, valid):
    """Overlap-weighted mean and std; invalid source rows never contribute."""
    # Accumulate in float64: constant large-valued descriptors should have
    # zero standard deviation, not float32 rounding residuals.
    intervals = np.asarray(intervals, dtype=np.float64)
    frame_intervals = np.asarray(frame_intervals, dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    n, d = len(intervals), features.shape[1]
    output = np.zeros((n, d*2), np.float32)
    mask = np.zeros(n, bool)
    mapping = []
    for i, (start, end) in enumerate(intervals):
        weights = np.maximum(0, np.minimum(end, frame_intervals[:, 1])-np.maximum(start, frame_intervals[:, 0]))
        weights *= valid
        idx = np.flatnonzero(weights > 0)
        mapping.append(idx.tolist())
        if len(idx):
            w = weights[idx] / weights[idx].sum()
            mean = (w[:, None]*features[idx]).sum(0)
            var = (w[:, None]*(features[idx]-mean)**2).sum(0)
            output[i] = np.r_[mean, np.sqrt(np.maximum(var, 0))]
            mask[i] = True
    return output, mask, mapping

def complement_intervals(intervals, duration):
    gaps, cursor = [], 0.0
    for start, end in intervals:
        if start > cursor + 1e-7:
            gaps.append((cursor, float(start)))
        cursor = max(cursor, float(end))
    if cursor < duration:
        gaps.append((cursor, duration))
    return np.asarray(gaps, dtype=np.float32).reshape(-1, 2)
