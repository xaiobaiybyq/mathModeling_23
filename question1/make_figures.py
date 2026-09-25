# -*- coding: utf-8 -*-
"""生成第一问论文中文配图。"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)

# 中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

C_MAIN = '#276875'   # 深青
C_ACC  = '#5b83b4'   # 蓝
C_ORNG = '#ad6b32'   # 橙
C_GRID = '#dce3e8'

def load(name):
    return json.loads((RES / name).read_text('utf-8'))

manifest = load('manifest.json')
names = load('feature_names.json')
stats = load('validation.json')

# 读取 sample_summary.csv 数值列
import csv
def read_csv(name):
    with (RES / name).open('r', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

rows = read_csv('sample_summary.csv')
for r in rows:
    for k in ['duration_s','word_count','aligned_words','audio_valid_words','vision_valid_words',
              'alignment_confidence_mean','low_confidence_words','face_detection_rate','multiface_frames',
              'audio_frames','video_sampled_frames']:
        r[k] = float(r[k]) if r[k] != '' else 0.0

dur = [r['duration_s'] for r in rows]
wc  = [int(r['word_count']) for r in rows]
face_rate = [r['face_detection_rate'] for r in rows]
conf = [r['alignment_confidence_mean'] for r in rows]
lowc = [int(r['low_confidence_words']) for r in rows]

# ---------- 图1 整体流程示意 ----------
def fig_flow():
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.axis('off')
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    def box(x, y, w, h, text, fc, tc='white', fs=10):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec='none', zorder=2, alpha=0.95))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', color=tc, fontsize=fs, zorder=3)
    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#5a6b75', lw=1.4))
    # 源视频
    box(0.2, 3.6, 1.8, 1.4, '原始视频\n(100条 MP4)\n+ 转写文本', '#34495e')
    # 三条模态分支
    box(2.9, 4.9, 2.6, 1.0, '文本：分词 + DistilBERT\n词级 768 维', C_MAIN)
    box(2.9, 3.2, 2.6, 1.0, '语音：16kHz + openSMILE\n帧级 25 维', C_ACC)
    box(2.9, 1.5, 2.6, 1.0, '视觉：10Hz + MediaPipe\n帧级 55 维', C_ORNG)
    # 对齐
    box(6.2, 3.2, 2.6, 1.8, 'Wav2Vec2 CTC 强制对齐\nViterbi 求词级时间区间', '#8e44ad')
    # 输出
    box(6.4, 0.4, 2.2, 1.0, '逐词聚合\nmean/std', '#c0392b')
    box(8.9, 0.2, 1.0, 1.6, 'NPZ\n特征', '#2c3e50', fs=9)
    arrow(2.0, 4.3, 2.9, 5.2)
    arrow(2.0, 4.3, 2.9, 3.7)
    arrow(2.0, 4.3, 2.9, 2.0)
    arrow(5.5, 5.4, 6.2, 4.4)
    arrow(5.5, 3.7, 6.2, 4.0)
    arrow(5.5, 2.0, 6.2, 3.6)
    arrow(7.5, 3.2, 7.5, 1.4)
    arrow(8.6, 0.9, 8.9, 0.9)
    fig.tight_layout()
    fig.savefig(FIG/'fig1_flow.png', dpi=180, facecolor='white')
    plt.close(fig)

# ---------- 图2 时长/词数分布 ----------
def fig_dist():
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    ax[0].hist(dur, bins=16, color=C_MAIN, edgecolor='white')
    ax[0].set_xlabel('视频时长 / 秒'); ax[0].set_ylabel('样本数')
    ax[0].set_title('(a) 100 条样本时长分布')
    ax[1].hist(wc, bins=16, color=C_ACC, edgecolor='white')
    ax[1].set_xlabel('词数'); ax[1].set_ylabel('样本数')
    ax[1].set_title('(b) 样本词数分布')
    for a in ax:
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG/'fig2_dist.png', dpi=180, facecolor='white', bbox_inches='tight')
    plt.close(fig)

# ---------- 图3 三模态覆盖率 ----------
def fig_coverage():
    n = stats['words']
    aligned = stats['aligned_words']
    audio = stats['audio_words']
    vision = stats['vision_words']
    cats = ['词级时间区间\n(CTC 对齐)', '有效语音特征', '有效视觉特征']
    vals = [aligned/n*100, audio/n*100, vision/n*100]
    colors = ['#8e44ad', C_ACC, C_ORNG]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    bars = ax.bar(cats, vals, color=colors, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+1.5, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('覆盖率 / %'); ax.set_ylim(0, 110)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG/'fig3_coverage.png', dpi=180, facecolor='white')
    plt.close(fig)

# ---------- 图4 典型样本词级对齐图 ----------
def fig_alignment():
    sid = 'example'  # 用 example.json 里选出的样本
    ex = load('example.json')
    sid = ex['id']
    meta = load('metadata/' + sid + '.json')
    arr = np.load(RES / ('features/' + sid + '.npz'), allow_pickle=False)
    x = {k: arr[k] for k in arr.files}
    words = meta['words']
    times = x['word_intervals']
    audio_intervals = x['audio_intervals']
    audio_raw = x['audio_raw']
    vision_intervals = x['vision_intervals']
    vision_raw = x['vision_raw']
    vision_pts = x['vision_pts']
    vnames = names['vision_raw']
    anames = names['audio_raw']

    fig, ax = plt.subplots(4, 1, figsize=(10, 6.2), sharex=True,
                           gridspec_kw={'height_ratios':[1.1,1,1,1]})
    # 词条
    for i, ((a,b), w) in enumerate(zip(times, words)):
        c = '#d8edf0' if i % 2 == 0 else '#eef4f5'
        ax[0].axvspan(a, b, color=c, zorder=1)
        ax[0].text((a+b)/2, 0.55+0.3*(i%2), w, ha='center', va='center', fontsize=7.5, rotation=40, zorder=2)
    ax[0].set_ylim(0, 1); ax[0].set_yticks([])
    ax[0].set_ylabel('词级对齐', fontsize=9)
    # 语音帧
    am = audio_intervals.mean(1)
    pitch_idx = next((i for i, nm in enumerate(anames) if 'F0' in nm), 0)
    ax[1].plot(am, audio_raw[:, pitch_idx], color=C_ORNG, lw=0.8)
    ax[1].set_ylabel('基频描述符', fontsize=9)
    # 视觉帧（微笑 + 张嘴）
    for nm in ['mouthSmileLeft', 'mouthSmileRight', 'jawOpen']:
        if nm in vnames:
            j = vnames.index(nm)
            y = vision_raw[:, j].copy()
            y[~x['vision_raw_mask']] = np.nan
            ax[2].plot(vision_pts, y, lw=1, label=nm)
    ax[2].legend(loc='upper right', ncol=3, fontsize=7)
    ax[2].set_ylabel('面部系数', fontsize=9)
    # 采样帧
    ax[3].vlines(vision_pts, 0, 1, color=C_MAIN, lw=1, alpha=0.7)
    ax[3].set_ylim(0, 1); ax[3].set_yticks([])
    ax[3].set_ylabel('采样帧', fontsize=9)
    ax[3].set_xlabel('时间 / 秒（相对片段起点）', fontsize=9)
    for a in ax:
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
        for (s, e) in times:
            a.axvline(s, color='#b0b0b0', lw=0.3, alpha=0.5)
    sid_disp = sid.replace('$', '_')
    fig.tight_layout(rect=[0,0,1,1])
    fig.savefig(FIG/'fig4_alignment.png', dpi=180, facecolor='white')
    plt.close(fig)

# ---------- 图5 人脸检测率 + 对齐置信度 ----------
def fig_quality():
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    ax[0].bar(np.arange(len(face_rate))+1, face_rate, color=C_ACC, width=1)
    ax[0].set_xlabel('样本序号（按源表顺序）'); ax[0].set_ylabel('人脸检出率')
    ax[0].set_ylim(0, 1.05); ax[0].set_title('(a) 各样本视觉帧人脸检出率')
    sc = ax[1].scatter(dur, conf, c=lowc, cmap='YlOrRd', s=28, edgecolor='#666', linewidth=0.3)
    ax[1].set_xlabel('视频时长 / 秒'); ax[1].set_ylabel('平均对齐分数')
    ax[1].set_ylim(0, 1.05); ax[1].set_title('(b) 对齐分数 vs 时长（色=低分词数）')
    fig.colorbar(sc, ax=ax[1], label='低置信度词数')
    for a in ax:
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG/'fig5_quality.png', dpi=180, facecolor='white', bbox_inches='tight')
    plt.close(fig)

# ---------- 图6 情感标签分布 ----------
def fig_label():
    labels = [r['label'] for r in manifest]
    ann = [r['annotation'] for r in manifest]
    from collections import Counter
    c = Counter(ann)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    order = ['Negative', 'Neutral', 'Positive']
    vals = [c.get(o, 0) for o in order]
    colors = ['#c0392b', '#95a5a6', '#27ae60']
    ax.bar(['负向', '中性', '正向'], vals, color=colors, width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v+0.5, str(v), ha='center', fontweight='bold')
    ax.set_ylabel('样本数'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG/'fig6_label.png', dpi=180, facecolor='white')
    plt.close(fig)

fig_flow()
fig_dist()
fig_coverage()
fig_alignment()
fig_quality()
fig_label()
print('figures generated:', sorted(p.name for p in FIG.glob('*.png')))
