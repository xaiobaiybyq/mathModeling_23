#!/usr/bin/env python3
"""附件4自动复核：自由解码语音与证据帧逐字节核查。

这是 AI 辅助审计，不填写人工回听模板，也不重新选择预测模型或解释规则。
运行：从 E 目录执行 .problem2-venv/bin/python 问题3/ai_review.py
"""
from __future__ import annotations

import csv
import difflib
import os
import struct
import zlib
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import av
import numpy as np
import torch

import localize
import problem3 as q3

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RATE = q3.CONFIG["alignment_sample_rate"]


def clean(text):
    return "".join(c for c in text.upper() if c.isalnum())


def greedy_asr(processor, model, wave):
    """不提供参考转写，对音频直接做CTC自由解码并保留字符时间。"""
    with torch.inference_mode():
        batch = processor(wave, sampling_rate=RATE, return_tensors="pt")
        ids = model(**batch).logits[0].argmax(-1).cpu().numpy()
    stride, receptive = 1, 1
    for kernel, step in zip(model.config.conv_kernel, model.config.conv_stride):
        receptive += (kernel - 1) * stride
        stride *= step
    vocab = processor.tokenizer.convert_ids_to_tokens
    chars, times = [], []
    previous = None
    for index, token_id in enumerate(ids):
        token_id = int(token_id)
        if token_id != model.config.pad_token_id and token_id != previous:
            symbol = vocab(token_id)
            moment = ((receptive - 1) / 2 + (index + 0.5) * stride) / RATE
            for character in symbol:
                if character.isalnum():
                    chars.append(character.upper())
                    times.append(float(moment))
        previous = token_id
    return "".join(chars), times


def best_phrase_match(phrase, asr, times):
    expected = clean(phrase)
    if len(expected) < 4 or not asr:
        return 0.0, None, None
    width_min = max(3, len(expected) // 2)
    width_max = min(len(asr), int(len(expected) * 1.6) + 2)
    best = (0.0, None, None)
    for left in range(len(asr)):
        for width in range(width_min, width_max + 1):
            right = left + width
            if right > len(asr):
                break
            score = difflib.SequenceMatcher(None, expected, asr[left:right], autojunk=False).ratio()
            if score > best[0]:
                best = (float(score), times[left], times[right - 1])
    return best


def decode_png(path):
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("not PNG")
    width = height = None
    chunks = []
    offset = 8
    while offset < len(data):
        length = struct.unpack_from(">I", data, offset)[0]
        tag = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        if tag == b"IHDR":
            width, height, depth, colour, *_ = struct.unpack(">IIBBBBB", payload)
            if depth != 8 or colour != 2:
                raise ValueError("unexpected PNG format")
        elif tag == b"IDAT":
            chunks.append(payload)
        elif tag == b"IEND":
            break
        offset += length + 12
    raw = zlib.decompress(b"".join(chunks))
    row_bytes = width * 3
    rows = []
    for y in range(height):
        start = y * (row_bytes + 1)
        if raw[start] != 0:
            raise ValueError("unexpected PNG row filter")
        rows.append(raw[start + 1:start + 1 + row_bytes])
    return np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(height, width, 3)


def check_frame(video_path, frame_path, expected_index, expected_time):
    saved = decode_png(frame_path)
    with av.open(str(video_path)) as container:
        origin = (container.start_time or 0) / av.time_base
        for index, frame in enumerate(container.decode(video=0)):
            if index != expected_index:
                continue
            timestamp = float(frame.pts * frame.time_base) - origin
            width = min(480, frame.width)
            height = max(1, int(round(frame.height * width / frame.width)))
            actual = frame.reformat(width=width, height=height, format="rgb24").to_ndarray()
            return bool(np.array_equal(saved, actual)), abs(timestamp - expected_time)
    return False, float("inf")


def audit():
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    with (RESULTS / "attachment4_predictions_explanations.csv").open(encoding="utf-8-sig", newline="") as file:
        predictions = list(csv.DictReader(file))
    assert len(predictions) == 20
    processor, model = localize.load_ctc()
    rows = []
    for prediction in predictions:
        sample_id = prediction["sample_id"]
        video = q3.TARGET / "videos" / prediction["video_file"]
        wave, duration, coverage = localize.decode_audio(video, RATE)
        asr, times = greedy_asr(processor, model, wave)
        score, matched_start, matched_end = best_phrase_match(prediction["key_text_excerpt"], asr, times)
        expected_start = float(prediction["key_time_start_seconds"])
        expected_end = float(prediction["key_time_end_seconds"])
        if matched_start is None:
            offset = None
            audio_status = "uncertain"
        else:
            offset = (matched_start + matched_end - expected_start - expected_end) / 2
            if score >= 0.8 and abs(offset) <= 0.75:
                audio_status = "supports_time"
            elif score >= 0.8 and abs(offset) > 1.5:
                audio_status = "possible_time_conflict"
            else:
                audio_status = "uncertain"
        frame_path = HERE / prediction["video_frame_path"]
        frame_exact, frame_time_error = check_frame(
            video, frame_path, int(prediction["video_frame_index_zero_based"]),
            float(prediction["video_frame_timestamp_seconds"]))
        rows.append({
            "sample_id": sample_id,
            "localization_status": prediction["localization_status"],
            "asr_phrase_similarity": round(score, 4),
            "asr_match_midpoint_offset_seconds": "" if offset is None else round(offset, 3),
            "audio_check": audio_status,
            "frame_pixel_exact": frame_exact,
            "frame_timestamp_error_seconds": round(frame_time_error, 6),
            "audio_coverage_fraction": round(coverage, 6),
            "review_type": "AI_assisted_not_human",
        })
        print("AI audit", len(rows), "/20", sample_id, audio_status, flush=True)
    output = RESULTS / "ai_review_results.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["audio_check"] for row in rows)
    flagged = [row["sample_id"] for row in rows if row["audio_check"] != "supports_time"]
    frame_ok = sum(row["frame_pixel_exact"] and row["frame_timestamp_error_seconds"] < 1e-5 for row in rows)
    report = f"""# 第三问 AI 辅助核对报告

审计范围：附件4对齐版全部 {len(rows)} 条视频与问题三主证据。结果明细见 [ai_review_results.csv](results/ai_review_results.csv)。

## 方法

1. 对每条原视频音频使用本地 `facebook/wav2vec2-base-960h` 做不输入参考文本的 CTC 自由解码，搜索主证据短语的最佳字符匹配，并对照已有自动时间区间。
2. 重新从源视频指定帧序号解码，逐像素对照导出的证据 PNG，核查帧时间戳。
3. 语音短语相似度不低于 0.8 且中点偏移不超过 0.75 秒，记为 `supports_time`；相似度不低于 0.8 但偏移超过 1.5 秒，记为 `possible_time_conflict`；其余记为 `uncertain`。这只是审计规则，不是模型选择或真实标签。

## 结果

- 语音时间得到自由解码支持：**{counts['supports_time']}/{len(rows)}**。
- 可能存在时间冲突：**{counts['possible_time_conflict']}/{len(rows)}**。
- 自由解码不能确定：**{counts['uncertain']}/{len(rows)}**。
- 导出帧与源视频帧逐像素一致且时间戳一致：**{frame_ok}/{len(rows)}**。
- 建议人工优先回听的编号：**{', '.join(flagged) if flagged else '无'}**；原有定位质量警示编号 **07、15、18** 仍应优先检查。

## 解释边界

自由解码和先前的强制对齐使用同一个声学检查点，因而不能视为完全独立的语音真值。帧像素一致只证明图片来自指定视频时间，不能证明画面在情感意义上支持预测。附件4没有真实情感标签。本审计不填写 `manual_review_template.csv` 的人工栏，也不将 `manual_time_verification_completed` 改为 true；论文应继续使用“自动估计时间”，直至有人逐条回听、回看。
"""
    (HERE / "AI辅助核对报告.md").write_text(report, encoding="utf-8")
    print("SUMMARY", dict(counts), "frame_exact", frame_ok, "flagged", len(flagged), flush=True)


if __name__ == "__main__":
    audit()
