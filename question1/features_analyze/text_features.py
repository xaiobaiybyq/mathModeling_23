from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import re

# ========== 离线环境配置 ==========
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

ROOT = Path(__file__).resolve().parent
MODEL_TEXT_PATH = "../models/text"
cfg = {
    "text_chunk_words": 32
}

# 加载本地DistilBERT
text_tokenizer = AutoTokenizer.from_pretrained(MODEL_TEXT_PATH, local_files_only=True)
text_model = AutoModel.from_pretrained(MODEL_TEXT_PATH, local_files_only=True).eval()

# tokenize_words：原始句子 → 单词列表 + 字符偏移
def tokenize_words(raw_text: str):
    pattern = re.compile(r"[A-Za-z0-9']+")
    words = []
    offsets = []
    for match in pattern.finditer(raw_text):
        w = match.group(0).strip(".,!?;:\"()[]")
        if not w:
            continue
        words.append(w)
        offsets.append((match.start(), match.end()))
    return words, offsets

# 独立版 text_features
def text_features(words, text_tokenizer, text_model, text_chunk_words):
    output = np.zeros((len(words), text_model.config.dim), np.float32)
    mapping = []
    size = text_chunk_words
    for first in range(0, len(words), size):
        chunk = words[first:first+size]
        print(first)
        encoded = text_tokenizer(chunk, is_split_into_words=True, return_tensors='pt', truncation=False)
        print(encoded.keys())
        if encoded.input_ids.shape[1] > text_model.config.max_position_embeddings:
            raise ValueError('Text chunk exceeds model length; reduce text_chunk_words')
        word_ids = encoded.word_ids()
        with torch.inference_mode():
            h = text_model(**encoded).last_hidden_state[0].numpy()
        for j in range(len(chunk)):
            idx = [k for k,w in enumerate(word_ids) if w == j]
            if not idx:
                raise ValueError('Word has no text subtoken')
            output[first+j] = h[idx].mean(0)
            mapping.append({
                'word_index': first+j,
                'chunk_first_word': first,
                'subtoken_positions': idx,
                'subtoken_ids': encoded.input_ids[0, idx].tolist()
            })
    return output, mapping


if __name__ == "__main__":
    raw_sentence = "Power is similar to strength"
    words, offsets = tokenize_words(raw_sentence)
    print("原始句子：", raw_sentence)
    print("分词得到单词列表：")
    print(words)
    print("单词数量：", len(words))

    feat, map_info = text_features(words, text_tokenizer, text_model, cfg["text_chunk_words"])
    print("\n特征矩阵shape =", feat.shape)
    print("每个单词对应768维向量")

    print("\n==== 子词映射样例 ====")
    for m in map_info:
        print(f"单词索引{m['word_index']}, 单词：{words[m['word_index']]}, 子词下标:{m['subtoken_positions']}")
