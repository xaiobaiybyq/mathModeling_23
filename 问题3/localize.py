#!/usr/bin/env python3
"""将附件4的词元位置关联到原文、自动CTC时间和实际解码视频帧。"""
from __future__ import annotations

import csv
import difflib
import json
import os
import struct
import sys
import unicodedata
import zlib
from pathlib import Path

import numpy as np
import torch

import problem3 as q3

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = str(q3.HERE / ".hf_cache")

import av  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoConfig, AutoModelForCTC, AutoProcessor, AutoTokenizer  # noqa: E402

sys.path.insert(0, str(q3.Q1))
from alignment import ctc_viterbi, tokenize_words  # noqa: E402

MODEL_DIR = q3.Q1 / "models" / "alignment"
TOKENIZER_DIR = q3.Q1 / "models" / "text"
NUMBER_NORMALIZATION = {"28": "TWENTY EIGHT", "2008": "TWO THOUSAND EIGHT",
                        "1989": "NINETEEN EIGHTY NINE", "10th": "TENTH",
                        "1,500": "FIFTEEN HUNDRED", "100,000": "ONE HUNDRED THOUSAND",
                        "20,000": "TWENTY THOUSAND", "50": "FIFTY"}


def load_ctc():
    processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCTC.from_config(config)
    state = load_file(str(MODEL_DIR / "model.safetensors"))
    prefix = "wav2vec2.encoder.pos_conv_embed.conv."
    # 问题一的transformers 4.57保存weight_g/v；当前CPU环境4.40以parametrizations键读取。
    if prefix + "weight_g" in state:
        state[prefix + "parametrizations.weight.original0"] = state.pop(prefix + "weight_g")
        state[prefix + "parametrizations.weight.original1"] = state.pop(prefix + "weight_v")
    # 训练时遮挡向量在纯推理中不使用，置零以实现严格的状态检查。
    state["wav2vec2.masked_spec_embed"] = torch.zeros_like(model.state_dict()["wav2vec2.masked_spec_embed"])
    model.load_state_dict(state, strict=True)
    model.eval()
    return processor, model


def decode_audio(path, rate):
    with av.open(str(path)) as container:
        origin = (container.start_time or 0) / av.time_base
        duration = container.duration / av.time_base if container.duration else 0.0
        if not container.streams.audio:
            raise ValueError("video has no audio stream")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=rate)
        pieces = []
        cursor = 0.0
        for frame in container.decode(audio=0):
            for converted in resampler.resample(frame):
                start = float(converted.pts * converted.time_base) - origin if converted.pts is not None else cursor
                data = converted.to_ndarray().reshape(-1).astype(np.float32)
                pieces.append((start, data))
                cursor = start + len(data) / rate
        for converted in resampler.resample(None):
            start = float(converted.pts * converted.time_base) - origin if converted.pts is not None else cursor
            data = converted.to_ndarray().reshape(-1).astype(np.float32)
            pieces.append((start, data))
            cursor = start + len(data) / rate
        duration = max(duration, max((s + len(data) / rate for s, data in pieces), default=0))
    if duration <= 0:
        raise ValueError("nonpositive video duration")
    wave = np.zeros(int(np.ceil(duration * rate)), dtype=np.float32)
    observed = np.zeros(len(wave), dtype=bool)
    for start, data in pieces:
        left = round(start * rate)
        a, b = max(left, 0), min(left + len(data), len(wave))
        if b > a:
            wave[a:b] = data[a - left:b - left]
            observed[a:b] = True
    if not observed.any():
        raise ValueError("no observed audio samples")
    return wave, float(len(wave) / rate), float(observed.mean())


def normalized_word(word, vocab):
    fixed = NUMBER_NORMALIZATION.get(word, word.upper().replace("’", "'"))
    fixed = "".join(c for c in unicodedata.normalize("NFD", fixed) if unicodedata.category(c) != "Mn")
    fixed = fixed.replace(" ", "|")
    if any(c not in vocab for c in fixed):
        raise ValueError("unsupported characters in transcript")
    return fixed


def align_words(processor, model, wave, words, rate):
    vocab = processor.tokenizer.get_vocab()
    target, owners = [], []
    cleaned = []
    for i, word in enumerate(words):
        norm = normalized_word(word, vocab)
        if i:
            target.append(vocab["|"]); owners.append(-1)
        target.extend(vocab[c] for c in norm)
        owners.extend([i] * len(norm))
        cleaned.append(norm.replace("|", " "))
    batch = processor(wave, sampling_rate=rate, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**batch).logits[0]
        log_probs = logits.log_softmax(-1).cpu().numpy()
    _, spans, confidence = ctc_viterbi(log_probs, target, model.config.pad_token_id)
    stride, receptive = 1, 1
    for kernel, step in zip(model.config.conv_kernel, model.config.conv_stride):
        receptive += (kernel - 1) * stride
        stride *= step
    boundaries = np.clip((receptive - 1) / (2 * rate) +
                         (np.arange(len(log_probs) + 1) - 0.5) * stride / rate,
                         0, len(wave) / rate)
    owners = np.asarray(owners)
    times, scores = [], []
    for i in range(len(words)):
        chars = np.flatnonzero(owners == i)
        start, end = spans[chars[0], 0], spans[chars[-1], 1]
        times.append((float(boundaries[start]), float(boundaries[end])))
        scores.append(float(np.exp(np.log(np.maximum(confidence[chars], 1e-30)).mean())))
    greedy = processor.batch_decode(logits.argmax(-1).unsqueeze(0))[0]
    transcript = " ".join(cleaned).upper()
    similarity = difflib.SequenceMatcher(None,
        "".join(c for c in transcript if c.isalnum()),
        "".join(c for c in greedy.upper() if c.isalnum())).ratio()
    return times, scores, float(similarity)


def match_word(token_start, token_end, word_offsets):
    if token_end <= token_start:
        return None
    overlaps = np.maximum(0, np.minimum(token_end, word_offsets[:, 1]) - np.maximum(token_start, word_offsets[:, 0]))
    if overlaps.max() > 0:
        return int(overlaps.argmax())
    # 标点没有落入词区间时，关联字符距离最近的词并标记近邻匹配。
    distances = np.maximum(0, np.maximum(word_offsets[:, 0] - token_end, token_start - word_offsets[:, 1]))
    return int(distances.argmin())


def locate_window(raw_text, offsets, word_offsets, word_times, start, end):
    useful = [(int(a), int(b)) for a, b in offsets[start:end] if b > a]
    if not useful:
        return {"text_excerpt": "", "char_start": "", "char_end_exclusive": "",
                "time_start_seconds": "", "time_end_seconds": ""}
    char_start = min(a for a, _ in useful)
    char_end = max(b for _, b in useful)
    matched = [match_word(a, b, word_offsets) for a, b in useful]
    matched = [m for m in matched if m is not None]
    return {"text_excerpt": raw_text[char_start:char_end], "char_start": char_start,
            "char_end_exclusive": char_end,
            "time_start_seconds": min(word_times[m][0] for m in matched) if matched and word_times else "",
            "time_end_seconds": max(word_times[m][1] for m in matched) if matched and word_times else ""}


def png_chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)


def write_png(path, rgb):
    height, width, channels = rgb.shape
    assert channels == 3
    rows = b"".join(b"\0" + rgb[y].tobytes() for y in range(height))
    payload = (b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
               + png_chunk(b"IDAT", zlib.compress(rows, 6)) + png_chunk(b"IEND", b""))
    path.write_bytes(payload)


def extract_nearest_frame(video_path, target_time, output_path):
    with av.open(str(video_path)) as container:
        origin = (container.start_time or 0) / av.time_base
        best_frame, best_time, best_index, best_gap = None, None, None, float("inf")
        for index, frame in enumerate(container.decode(video=0)):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * frame.time_base) - origin
            gap = abs(timestamp - target_time)
            if gap < best_gap:
                best_frame, best_time, best_index, best_gap = frame, timestamp, index, gap
            if timestamp > target_time and gap > best_gap:
                break
        if best_frame is None:
            raise ValueError("no timestamped video frame")
        width = min(480, best_frame.width)
        height = max(1, int(round(best_frame.height * width / best_frame.width)))
        rgb = best_frame.reformat(width=width, height=height, format="rgb24").to_ndarray()
        write_png(output_path, rgb)
        return best_time, best_index, best_gap


def localize(target, raw_texts, predictions, evidence):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True, use_fast=True)
    processor, model = load_ctc()
    evidence_by_id = {}
    for row in evidence:
        evidence_by_id.setdefault(row["sample_id"], []).append(row)
    frame_dir = q3.OUT / "frames"
    frame_dir.mkdir(exist_ok=True)
    diagnostics = []
    for i, (prediction, raw_text) in enumerate(zip(predictions, raw_texts)):
        sample_id = prediction["sample_id"]
        coded = tokenizer(raw_text, truncation=True, max_length=50, return_offsets_mapping=True)
        expected = target["tokens"][i, target["effective"][i, 0]].tolist()
        if coded["input_ids"] != expected:
            raise ValueError(f"text token mapping differs for {sample_id}")
        offsets = coded["offset_mapping"]
        words, word_offsets = tokenize_words(raw_text)
        video = q3.TARGET / "videos" / f"{sample_id}.mp4"
        assert video.is_file() and len(words)
        status = "自动CTC定位_需人工回听"
        fail_reason = ""
        mean_confidence, similarity = None, None
        times = []
        warning_reasons = []
        try:
            wave, duration, audio_coverage = decode_audio(video, q3.CONFIG["alignment_sample_rate"])
            times, scores, similarity = align_words(processor, model, wave, words, q3.CONFIG["alignment_sample_rate"])
            mean_confidence = float(np.mean(scores))
            if mean_confidence < q3.CONFIG["low_alignment_confidence"]:
                warning_reasons.append("low_ctc_confidence")
            if similarity < q3.CONFIG["low_transcript_similarity"]:
                warning_reasons.append("low_transcript_similarity")
            if warning_reasons:
                status = "自动定位质量警示_需人工回听"
        except (ValueError, RuntimeError, av.AVError) as exc:
            duration, audio_coverage = None, None
            status = "未能自动定位"
            fail_reason = f"{type(exc).__name__}: {str(exc)[:90]}"
        for row in evidence_by_id[sample_id]:
            if row["start_position"] == "":
                location = {"text_excerpt": "", "char_start": "", "char_end_exclusive": "",
                            "time_start_seconds": "", "time_end_seconds": ""}
            else:
                location = locate_window(raw_text, offsets, word_offsets, times,
                                         row["start_position"], row["end_position_exclusive"])
            row.update(location)
            row["location_status"] = status if row["start_position"] != "" else "原始非零窗口不存在"
            row["source_video_file"] = sample_id + ".mp4"
            row["video_frame_path"] = ""
            row["video_frame_timestamp_seconds"] = ""
            row["video_frame_index_zero_based"] = ""
            if row["modality"] == "vision" and row["time_start_seconds"] != "":
                midpoint = (row["time_start_seconds"] + row["time_end_seconds"]) / 2
                visual_path = frame_dir / f"{sample_id}_vision.png"
                try:
                    timestamp, index, _ = extract_nearest_frame(video, midpoint, visual_path)
                    row["video_frame_path"] = str(visual_path.relative_to(q3.HERE))
                    row["video_frame_timestamp_seconds"] = timestamp
                    row["video_frame_index_zero_based"] = index
                except (ValueError, RuntimeError, av.AVError) as exc:
                    fail_reason = (fail_reason + "; " if fail_reason else "") + f"vision frame: {type(exc).__name__}"
        key = next(r for r in evidence_by_id[sample_id] if r["modality"] == prediction["primary_modality"])
        prediction.update({"key_text_excerpt": key["text_excerpt"],
                           "key_char_start": key["char_start"], "key_char_end_exclusive": key["char_end_exclusive"],
                           "key_time_start_seconds": key["time_start_seconds"],
                           "key_time_end_seconds": key["time_end_seconds"],
                           "localization_status": status,
                           "ctc_mean_confidence": mean_confidence if mean_confidence is not None else "",
                           "transcript_similarity": similarity if similarity is not None else "",
                           "video_duration_seconds": duration if duration is not None else "",
                           "video_frame_path": "", "video_frame_timestamp_seconds": "",
                           "video_frame_index_zero_based": "", "frame_time_gap_seconds": ""})
        if times and key["time_start_seconds"] != "":
            midpoint = (key["time_start_seconds"] + key["time_end_seconds"]) / 2
            path = frame_dir / f"{sample_id}.png"
            try:
                timestamp, index, gap = extract_nearest_frame(video, midpoint, path)
                prediction["video_frame_path"] = str(path.relative_to(q3.HERE))
                prediction["video_frame_timestamp_seconds"] = timestamp
                prediction["video_frame_index_zero_based"] = index
                prediction["frame_time_gap_seconds"] = gap
            except (ValueError, RuntimeError, av.AVError) as exc:
                fail_reason = (fail_reason + "; " if fail_reason else "") + f"frame: {type(exc).__name__}"
        diagnostics.append({"sample_id": sample_id, "tokenizer_exact": True,
                            "active_tokens": len(expected), "raw_word_count": len(words),
                            "text_truncated_to_50": len(tokenizer(raw_text, add_special_tokens=True)["input_ids"]) > 50,
                            "video_duration_seconds": duration, "audio_coverage_fraction": audio_coverage,
                            "ctc_word_count": len(times), "ctc_mean_confidence": mean_confidence,
                            "transcript_similarity": similarity, "status": status,
                            "warning_reasons": "+".join(warning_reasons),
                            "frame_extracted": bool(prediction["video_frame_path"]), "failure": fail_reason})
        print("attachment4 localized", i + 1, "/20", status, flush=True)
    q3.write_csv(q3.OUT / "localization_diagnostics.csv", diagnostics)
    review = [{"sample_id": row["sample_id"], "video_file": row["video_file"],
               "key_text_excerpt": row["key_text_excerpt"],
               "key_time_start_seconds": row["key_time_start_seconds"],
               "key_time_end_seconds": row["key_time_end_seconds"],
               "localization_status": row["localization_status"],
               "manual_audio_match": "", "manual_video_match": "", "review_notes": ""}
              for row in predictions]
    q3.write_csv(q3.OUT / "manual_review_template.csv", review)
    audit = {"samples": len(predictions), "tokenizer_exact_count": sum(r["tokenizer_exact"] for r in diagnostics),
             "ctc_success_count": sum(r["ctc_word_count"] > 0 for r in diagnostics),
             "quality_warning_count": sum(bool(r["warning_reasons"]) for r in diagnostics),
             "low_ctc_confidence_count": sum("low_ctc_confidence" in r["warning_reasons"] for r in diagnostics),
             "low_transcript_similarity_count": sum("low_transcript_similarity" in r["warning_reasons"] for r in diagnostics),
             "frame_extracted_count": sum(r["frame_extracted"] for r in diagnostics),
             "visual_evidence_frame_count": sum(bool(r["video_frame_path"]) for r in evidence if r["modality"] == "vision"),
             "visual_evidence_unavailable_count": sum(r["direction"] == "no_nonzero_feature_window" for r in evidence if r["modality"] == "vision"),
             "truncated_text_count": sum(r["text_truncated_to_50"] for r in diagnostics),
             "manual_time_verification_completed": False,
             "speech_model_source": "facebook/wav2vec2-base-960h",
             "speech_model_revision": json.loads((q3.Q1 / "models/model_manifest.json").read_text(encoding="utf-8"))["models"]["alignment"]["revision"],
             "tokenizer_source": "distilbert/distilbert-base-uncased",
             "time_mapping_rule": "exact tokenizer offsets to raw_text; independent CTC forced alignment of transcript to video audio; feature-position to word-time correspondence is inferred from aligned_50, not a supplied timestamp"}
    return predictions, evidence, audit


def verify(target, raw_texts, probs, strengths):
    path = q3.OUT / "attachment4_predictions_explanations.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(raw_texts) == 20
    assert len({r["sample_id"] for r in rows}) == 20
    assert {r["sample_id"] for r in rows} == {p.stem for p in q3.TARGET.glob("*.pkl")}
    for i, row in enumerate(rows):
        assert row["sample_id"] == target["id"][i]
        assert row["source_file"] == row["sample_id"] + ".pkl"
        assert row["video_file"] == row["sample_id"] + ".mp4"
        assert int(row["polarity_class"]) == int(probs[i].argmax())
        assert row["polarity_name"] == q3.NAMES[int(probs[i].argmax())]
        assert -3 <= float(row["predicted_strength"]) <= 3
        assert abs(float(row["predicted_strength"]) - float(strengths[i])) < 1e-5
        assert abs(sum(float(row["prob_" + name]) for name in ("negative", "neutral", "positive")) - 1) < 1e-5
        for name, value in zip(("negative", "neutral", "positive"), probs[i]):
            assert abs(float(row["prob_" + name]) - float(value)) < 1e-5
        assert row["primary_modality"] in q3.MODS
        assert abs(sum(float(row["share_" + name]) for name in q3.MODS) - 1) < 1e-6
        assert 1 <= int(row["key_start_position"]) < int(row["key_end_position_exclusive"]) <= int(target["lengths"][i]) - 1
        if row["key_char_start"]:
            assert row["key_text_excerpt"] == raw_texts[i][int(row["key_char_start"]):int(row["key_char_end_exclusive"])]
        if row["key_time_start_seconds"]:
            assert 0 <= float(row["key_time_start_seconds"]) <= float(row["key_time_end_seconds"]) <= float(row["video_duration_seconds"]) + 0.1
        if row["video_frame_path"]:
            assert (q3.HERE / row["video_frame_path"]).is_file()
    return True
