"""Validate numerical artifacts and generate a readable report and CSV tables."""
from pathlib import Path
import argparse
import base64
import csv
import hashlib
import html
import json
import os
import sys
import sysconfig
import textwrap
import zipfile

if os.name == 'nt' and sysconfig.get_platform() == 'win-amd64':
    os.environ.setdefault('PROCESSOR_ARCHITECTURE', 'AMD64')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import av
from PIL import Image

ROOT = Path(__file__).resolve().parent
WARNING_LABELS = {
    'low_alignment_confidence':'部分词对齐分数低',
    'some_words_without_valid_face':'部分词区间无人脸特征',
    'multiple_faces_largest_face_selected':'多人脸画面需确认说话人',
    'some_words_without_valid_audio':'部分词区间无有效语音特征',
    'numeric_pronunciation_needs_review':'数字读法需回听核对',
    'alignment_failed':'强制对齐未成功'
}
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','text.parse_math':False})

def load(path):
    return json.loads(path.read_text('utf-8'))

def write_csv(path, rows, columns=None):
    if columns is None:
        columns = list(rows[0]) if rows else []
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w = csv.DictWriter(f,fieldnames=columns)
        w.writeheader()
        w.writerows(rows)

def reference_pool(intervals, frame_intervals, features, valid):
    """Separate scalar interval implementation for checking the saved output."""
    output=np.zeros((len(intervals),2*features.shape[1]),np.float32)
    masks=np.zeros(len(intervals),bool)
    mappings=[]
    for i,(start,end) in enumerate(intervals):
        selected=[]
        weights=[]
        for j,(left,right) in enumerate(frame_intervals):
            weight=max(0.,min(float(end),float(right))-max(float(start),float(left)))
            if valid[j] and weight>0:
                selected.append(j)
                weights.append(weight)
        mappings.append(selected)
        if selected:
            mean=np.average(features[selected].astype(np.float64),axis=0,weights=weights)
            var=np.average((features[selected]-mean)**2,axis=0,weights=weights)
            output[i,:features.shape[1]]=mean
            output[i,features.shape[1]:]=np.sqrt(var)
            masks[i]=True
    return output,masks,mappings

def validate(out):
    manifest = load(out/'manifest.json')
    errors, rows, detail, word_rows, diagnostics = [], [], [], [], []
    names = load(out/'feature_names.json')
    totals = dict(words=0,aligned_words=0,audio_words=0,vision_words=0,low_confidence_words=0,video_frames=0,valid_face_frames=0,multiface_frames=0)
    for item in manifest:
        sid = item['id']
        meta_file = out/'metadata'/(sid+'.json')
        feature_file = out/'features'/(sid+'.npz')
        if not meta_file.exists() or not feature_file.exists():
            errors.append(sid+': missing result')
            rows.append({'id':sid,'status':'failed'})
            continue
        meta = load(meta_file)
        with np.load(feature_file,allow_pickle=False) as data:
            x = {k:data[k] for k in data.files}
        n = len(meta['words'])
        try:
            assert meta['id']==sid and meta['raw_text']==item['raw_text'], 'source identity/text'
            assert meta['label']==item['label'] and meta['annotation']==item['annotation'], 'source labels'
            assert hashlib.sha256(feature_file.read_bytes()).hexdigest() == meta['feature_sha256'], 'feature hash'
            assert int(x['valid_length']) == n > 0, 'valid length'
            assert not x['padding_mask'].any(), 'ragged sequence padding'
            for field in ['text','audio','vision']:
                assert x[field].shape == (n,len(names[field])), field+' shape'
                assert np.isfinite(x[field]).all(), field+' finiteness'
            for word,(a,b) in zip(meta['words'],x['word_char_offsets']):
                assert meta['raw_text'][a:b] == word, 'word source offset'
            times = x['word_intervals'][x['alignment_mask']]
            assert (times[:,1]>times[:,0]).all(), 'positive word duration'
            assert (times>=0).all() and (times<=meta['duration_seconds']+1e-5).all(), 'time range'
            assert (times[1:,0]>=times[:-1,1]-1e-5).all(), 'word monotonicity'
            assert (np.diff(x['vision_pts'])>0).all(), 'video PTS ordering'
            for field in ['audio','vision']:
                recomputed, mask, mapping = reference_pool(x['word_intervals'],x[field+'_intervals'],x[field+'_raw'],x[field+'_raw_mask'])
                np.testing.assert_allclose(recomputed,x[field],rtol=2e-5,atol=2e-5)
                assert np.array_equal(mask,x[field+'_mask']),field+' mask'
                assert mapping == meta[field+'_source_indices_per_word'],field+' source mapping'
                assert (x[field][~mask]==0).all(),field+' missing fill'
        except Exception as e:
            errors.append(sid+': '+str(e))
        low = int(x['alignment_review_mask'].sum())
        row = {'id':sid,'duration_s':round(meta['duration_seconds'],4),'word_count':n,
               'text_dim':x['text'].shape[1],'audio_dim':x['audio'].shape[1],'vision_dim':x['vision'].shape[1],
               'alignment_granularity':'word','aligned_words':int(x['alignment_mask'].sum()),
               'audio_valid_words':int(x['audio_mask'].sum()),'vision_valid_words':int(x['vision_mask'].sum()),
               'alignment_confidence_mean':round(float(x['alignment_confidence'].mean()),4),
               'low_confidence_words':low,'face_detection_rate':round(float(x['vision_raw_mask'].mean()),4),
               'multiface_frames':int((x['face_counts']>1).sum()),
               'audio_frames':len(x['audio_raw']),'video_sampled_frames':len(x['vision_raw']),
               'status':meta['status'],'warnings':';'.join(meta['warnings']),
               'warning_explanation':'；'.join(WARNING_LABELS.get(w,w) for w in meta['warnings']),
               'label':meta['label'],'annotation':meta['annotation'],
               'feature_file':meta['feature_file'],'feature_bytes':feature_file.stat().st_size}
        rows.append(row)
        diagnostics.append({'id':sid,'source_transcript':meta['raw_text'],
                            'greedy_asr_for_diagnosis':meta['alignment'].get('greedy_transcript_for_qc_only',''),
                            'mean_forced_path_score':row['alignment_confidence_mean'],
                            'low_score_words':low,
                            'diagnosis':'疑似转写与音频内容不匹配，或语音识别严重失败；需逐条回听确认' if row['alignment_confidence_mean']<.1 else '按低分词及模态掩码复核',
                            'source_transcript_modified':False})
        for modal in ['text','audio','vision']:
            detail.append({'id':sid,'modality':modal,'original_duration_s':row['duration_s'],
                           'sequence_positions':n,'feature_dimension':x[modal].shape[1],
                           'valid_positions':int(x[modal+'_mask'].sum()),
                           'granularity':'word','padding':'none; full ragged sequence',
                           'source_video':meta['source_video']})
        for i,word in enumerate(meta['words']):
            word_rows.append({'id':sid,'word_index':i,'word':word,'start_s':float(x['word_intervals'][i,0]),
                              'end_s':float(x['word_intervals'][i,1]),'alignment_valid':bool(x['alignment_mask'][i]),
                              'alignment_score':float(x['alignment_confidence'][i]),'review_required':bool(x['alignment_review_mask'][i]),
                              'audio_valid':bool(x['audio_mask'][i]),'vision_valid':bool(x['vision_mask'][i])})
        for key, value in {'words':n,'aligned_words':row['aligned_words'],'audio_words':row['audio_valid_words'],
                           'vision_words':row['vision_valid_words'],'low_confidence_words':low,
                           'video_frames':row['video_sampled_frames'],'valid_face_frames':int(x['vision_raw_mask'].sum()),
                           'multiface_frames':row['multiface_frames']}.items():
            totals[key] += value
    fields = list(next(r for r in rows if 'duration_s' in r))
    write_csv(out/'sample_summary.csv',rows,fields)
    write_csv(out/'modality_summary.csv',detail)
    write_csv(out/'word_alignment.csv',word_rows)
    write_csv(out/'alignment_diagnostics.csv',diagnostics)
    stats = {'expected_samples':len(manifest),'produced_samples':len(detail)//3,
             'duration_sum_s':sum(r.get('duration_s',0) for r in rows),**totals,
             'feature_bytes':sum(r.get('feature_bytes',0) for r in rows),
             'samples_needing_review':sum(r.get('status')=='review_required' for r in rows),
             'samples_mean_alignment_score_below_0_1':sum(r.get('alignment_confidence_mean',1)<.1 for r in rows),
             'structural_validation_errors':errors,
             'manual_boundary_accuracy':'not measured; no human word-boundary reference supplied'}
    (out/'validation.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False),'utf-8')
    return rows,stats,word_rows

def plot_quality(out, rows):
    good = [r for r in rows if 'duration_s' in r]
    fig,ax = plt.subplots(2,2,figsize=(12,7.5),layout='constrained')
    ax[0,0].hist([r['duration_s'] for r in good],bins=15,color='#276875',edgecolor='white')
    ax[0,0].set(xlabel='Video duration (s)',ylabel='Samples',title='Source duration distribution')
    ax[0,1].hist([r['word_count'] for r in good],bins=15,color='#276875',edgecolor='white')
    ax[0,1].set(xlabel='Words per sample',ylabel='Samples',title='Full sequences retained')
    rates = [r['face_detection_rate'] for r in good]
    ax[1,0].bar(np.arange(len(rates))+1,rates,color='#5b83b4',width=1)
    ax[1,0].set(xlabel='Sample order in source table',ylabel='Valid face / sampled frames',ylim=(0,1.05),title='Visual extraction coverage')
    ax[1,1].scatter([r['duration_s'] for r in good],[r['alignment_confidence_mean'] for r in good],c=[r['low_confidence_words'] for r in good],cmap='viridis',s=30)
    ax[1,1].set(xlabel='Video duration (s)',ylabel='Mean forced-path score',ylim=(0,1.05),title='Alignment diagnostic (not accuracy)')
    fig.savefig(out/'quality_overview.png',dpi=160)
    plt.close(fig)

def make_example(out, rows, data_root):
    available = [r for r in rows if r.get('aligned_words',0)==r.get('word_count',-1) and r.get('face_detection_rate',0)>.8 and r.get('multiface_frames',1)==0 and r.get('low_confidence_words',0)==0]
    if not available:
        available = [r for r in rows if 'duration_s' in r]
    # Prefer the first source-table sample satisfying the declared quality rule.
    row = available[0]
    sid = row['id']
    meta = load(out/'metadata'/(sid+'.json'))
    with np.load(out/row['feature_file']) as f:
        x = {k:f[k] for k in f.files}
    names = load(out/'feature_names.json')
    source = next(data_root.rglob('label-100.xlsx')).parent/meta['source_video']
    # Display the same PTS-aware audio timeline used by the extractor.
    from pipeline import media_audio
    signal,_,_=media_audio(source,16000)
    sr=16000
    fig,axes=plt.subplots(4,1,figsize=(14,8.5),sharex=True,gridspec_kw={'height_ratios':[1.5,1,1,1]},layout='constrained')
    times=x['word_intervals']
    for i,((a,b),word) in enumerate(zip(times,meta['words'])):
        color='#d8edf0' if i%2==0 else '#eef4f5'
        axes[0].axvspan(a,b,color=color)
        axes[0].text((a+b)/2,.25+.42*(i%2),word,ha='center',va='center',fontsize=8,rotation=35)
    axes[0].set(ylim=(0,1),yticks=[],title='Word-level alignment | '+sid)
    stride=max(1,len(signal)//12000)
    axes[1].plot(np.arange(0,len(signal),stride)/sr,signal[::stride],lw=.5,color='#276875')
    axes[1].set_ylabel('Waveform')
    audio_mid=x['audio_intervals'].mean(1)
    pitch_idx=next((i for i,n in enumerate(names['audio_raw']) if 'F0' in n),0)
    axes[2].plot(audio_mid,x['audio_raw'][:,pitch_idx],lw=.8,color='#ad6b32')
    axes[2].set_ylabel('Pitch descriptor')
    for name in ['mouthSmileLeft','mouthSmileRight','jawOpen']:
        if name in names['vision_raw']:
            col=names['vision_raw'].index(name)
            y=x['vision_raw'][:,col].copy()
            y[~x['vision_raw_mask']]=np.nan
            axes[3].plot(x['vision_pts'],y,label=name,lw=1)
    axes[3].legend(loc='upper right',ncol=3,fontsize=8)
    axes[3].set(xlabel='Seconds from clip start',ylabel='Face coefficients',xlim=(0,meta['duration_seconds']))
    for ax in axes[1:]:
        for a,b in times:
            ax.axvline(a,color='#a0a0a0',lw=.3,alpha=.4)
    fig.savefig(out/'example_alignment.png',dpi=170)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,4.2),layout='constrained')
    scale=float(np.quantile(abs(x['text']),.99))
    heat=ax.imshow(x['text'].T,aspect='auto',cmap='RdBu_r',vmin=-scale,vmax=scale,interpolation='nearest')
    ax.set_xticks(np.arange(len(meta['words'])),meta['words'],rotation=40,ha='right',fontsize=8)
    ax.set(xlabel='Word position (same order as alignment table)',ylabel='Text embedding coordinate',title='All 768 contextual text features per word')
    fig.colorbar(heat,ax=ax,label='Feature value (display clipped at 99th percentile)')
    fig.savefig(out/'example_text_features.png',dpi=150)
    plt.close(fig)
    positions=np.linspace(0,len(times)-1,min(6,len(times)),dtype=int)
    wanted={int(x['vision_frame_ids'][np.argmin(abs(x['vision_pts']-times[i].mean()))]):i for i in positions}
    fig, axes=plt.subplots(1,len(wanted),figsize=(15,3.1),layout='constrained',squeeze=False)
    at=0
    with av.open(str(source)) as c:
        for frame_id,frame in enumerate(c.decode(video=0)):
            if frame_id in wanted:
                i=wanted[frame_id]
                stamp=float(frame.pts*frame.time_base)-meta['container_origin_seconds']
                axes[0,at].imshow(frame.to_ndarray(format='rgb24'))
                axes[0,at].set_title(f'{meta["words"][i]}\n{stamp:.2f}s / frame {frame_id}',fontsize=10)
                axes[0,at].axis('off')
                at+=1
    fig.savefig(out/'example_frames.jpg',dpi=150)
    plt.close(fig)
    example={'id':sid,'selection_rule':'first source-table sample with all words aligned, zero low-confidence words, face rate > 0.8 and no multiple-face frames',
             'raw_text':meta['raw_text'],'source_video':meta['source_video'],
             'word_rows':[{'index':i,'word':w,'start_s':float(times[i,0]),'end_s':float(times[i,1]),
                           'score':float(x['alignment_confidence'][i]),'audio_frames':len(meta['audio_source_indices_per_word'][i]),
                           'visual_frames':len(meta['vision_source_indices_per_word'][i])} for i,w in enumerate(meta['words'])]}
    (out/'example.json').write_text(json.dumps(example,ensure_ascii=False,indent=2),'utf-8')
    write_csv(out/'example_word_mapping.csv',example['word_rows'])
    return example

def build_report(out, rows, stats, example):
    n=stats['words']
    ratio=lambda a,b: f'{a/b:.2%}' if b else 'n.a.'
    text=f'''# 第一问结果报告

## 实际完成情况

本次从附件1原始视频自主提取特征，未使用附件2特征代替，未训练或微调情感模型。共生成 {stats['produced_samples']}/{stats['expected_samples']} 条样本的特征文件，总时长 {stats['duration_sum_s']:.2f} 秒，保留 {n} 个完整词级位置。结构与数值复算检查发现 {len(stats['structural_validation_errors'])} 项错误。

| 指标 | 实测结果 |
|---|---:|
| 词级时间区间生成率 | {ratio(stats['aligned_words'],n)} |
| 有效语音特征覆盖词比例 | {ratio(stats['audio_words'],n)} |
| 有效视觉特征覆盖词比例 | {ratio(stats['vision_words'],n)} |
| 采样视频帧人脸检测率 | {ratio(stats['valid_face_frames'],stats['video_frames'])} |
| 对齐低置信度词数 | {stats['low_confidence_words']} |
| 多人脸采样帧数 | {stats['multiface_frames']} |
| 含质量复核提示的样本数 | {stats['samples_needing_review']} |
| 压缩特征文件总大小 | {stats['feature_bytes']/1024**2:.2f} MiB |

以上覆盖率不等同于对齐准确率。原数据没有人工逐词边界，本次未声称人工时间误差、识别准确率或情感预测成绩。低置信度位置保留在数据内并单独标记，建议按 manual_review_template.csv 进行人工回看。

本次有 {stats['samples_mean_alignment_score_below_0_1']} 条样本的平均对齐分数低于0.1。这些样本的赛题文本与模型直接识别出的语音文本往往差异较大，可能是转写与音频不匹配，也可能是识别失败。两种文本并列保存在 alignment_diagnostics.csv，供回听判断；未自行改写赛题文本或替换视频。合法CTC路径仅说明算法找到了受约束路径，不保证该样本具有可靠的语义时间对应。

## 方法与特征

1. 文本：冻结 DistilBERT 英文预训练编码器。按原文词语分组，对子词最后层向量求均值，得到每词768维上下文特征。原始转写全文与字符偏移保留。
2. 语音：音频按真实时间戳放入16kHz单声道时间轴。openSMILE eGeMAPSv02 提取25维低层描述符；按词区间的时间重叠长度加权计算均值、标准差，得到每词50维特征。这里不是整句88维统计特征。
3. 视觉：按10Hz目标频率选取具有真实PTS的视频帧；MediaPipe输出52个面部blendshape系数及3个头部欧拉角。按采样帧支持区间与词区间的重叠长度计算均值、标准差，得到每词110维特征。blendshape不是FACS动作单元，不应在论文中称为OpenFace AU。多人脸时选最大人脸并记录复核提示，不宣称已经验证其为说话人。
4. 对齐：冻结Wav2Vec2英文CTC声学模型，以赛题转写作为约束，用完整CTC状态转移和Viterbi算法寻找最大概率路径。重复字符之间必须经过blank；保留字符到原词的映射。时间分辨率约20ms，不代表实际边界误差小于20ms。
5. 数字读法：少数数字和序数词使用明确登记的英文展开；原文不改动。具体映射在metadata中。数字读法仍应通过视频回听核验。

若第i个词区间为I_i，原特征区间为J_j，则w_ij=|I_i∩J_j|m_j，μ_i=Σw_ij x_j/Σw_ij，σ_i=√[Σw_ij(x_j−μ_i)²/Σw_ij]。无有效贡献时特征置0，模态mask置False。检测失败与补齐位置分别记录。

存储采用完整变长序列，无50位置截断，不在磁盘上补齐。批量输入时按batch最大长度补0，padding_mask=True表示补齐；modality mask=True表示该模态具有有效观测。原始声学帧、视觉帧、支持区间和词间未分配区间均保留。未分配区间不自动等同于静音。

## 典型样本

样本：`{example['id']}`。选取规则为按原表顺序选择首条全部词可对齐、无低分词、人脸有效率超过80%且无多人脸采样帧的样本。这是展示用案例，不用于替代全量质量统计。

原文：{example['raw_text']}

![时间对齐](example_alignment.png)

![原始视频帧](example_frames.jpg)

![词级文本特征](example_text_features.png)

逐词时间、聚合帧数见 example_word_mapping.csv；每个词对应的源帧索引见该样本metadata JSON。图示用于可视核验，不构成人工边界精度测量。

## 质量分布

![质量概览](quality_overview.png)

## 全量结果与提交

- sample_summary.csv：100条样本汇总，包括时长、维度、有效词数、提示信息。
- modality_summary.csv：每样本三行，100条样本共300行，可整理为论文全量结果表。
- word_alignment.csv：全部词的位置、时间区间、置信度与模态有效性。
- alignment_diagnostics.csv：赛题转写与声学模型直接识别文本的逐样本对照，用于排查异常，不用于替换原文。
- features/*.npz：三模态对齐特征及原始帧级数值特征。
- metadata/*.json：原始文本、标签、源视频相对路径、字符映射、聚合源帧索引及哈希。
- validation.json：自动结构校验与独立聚合复算的结果。
- run_environment.json：依赖版本、模型修订号、配置、模型及源文件哈希。

局限：语音转写与声学模型不匹配时可能强制对齐到错误位置；低置信度分数只是复核线索。人脸遮挡、画外音和多人脸会影响视觉可用性；最大人脸不一定是说话人。视觉10Hz抽样会遗漏更短暂的变化。本次尚未进行人工逐词精度标注或情感预测评价。

## 官方工具资料

- DistilBERT：https://huggingface.co/distilbert/distilbert-base-uncased
- Wav2Vec2：https://huggingface.co/facebook/wav2vec2-base-960h
- openSMILE：https://audeering.github.io/opensmile-python/usage.html
- MediaPipe：https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python
'''
    (out/'第一问结果报告.md').write_text(text,'utf-8')
    def picture(filename):
        mime='image/jpeg' if filename.endswith('.jpg') else 'image/png'
        return f'<img src="data:{mime};base64,{base64.b64encode((out/filename).read_bytes()).decode()}" alt="{filename}">'
    columns=[('id','样本编号'),('duration_s','时长/秒'),('word_count','词数'),('audio_valid_words','语音有效词'),('vision_valid_words','视觉有效词'),('low_confidence_words','低分词'),('face_detection_rate','人脸检出率'),('warning_explanation','复核提示')]
    table='<table><thead><tr>'+''.join('<th>'+v+'</th>' for _,v in columns)+'</tr></thead><tbody>'
    for r in rows:
        table+='<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k,_ in columns)+'</tr>'
    table+='</tbody></table>'
    paragraphs=''
    for paragraph in text.split('\n\n'):
        if paragraph.startswith(('![','|','# ')):
            continue
        if paragraph.startswith('## '):
            paragraphs+='<h3>'+html.escape(paragraph[3:])+'</h3>'
        elif paragraph.startswith('- '):
            paragraphs+='<ul>'+''.join('<li>'+html.escape(line[2:])+'</li>' for line in paragraph.splitlines())+'</ul>'
        else:
            paragraphs+='<p>'+html.escape(paragraph)+'</p>'
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第一问结果报告</title><style>body{font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif;color:#203344;margin:40px auto;max-width:1200px;padding:0 24px;background:#f8fafb}h1{font-size:32px}p{white-space:pre-wrap}img{width:100%;background:white;border-radius:8px;margin:12px 0}table{border-collapse:collapse;width:100%;font-size:13px;background:white}th,td{border-bottom:1px solid #dce3e8;padding:9px;text-align:left}th{background:#e1eef1;position:sticky;top:0}td:last-child{font-size:11px}.metrics{display:flex;gap:16px;flex-wrap:wrap}.metric{background:white;padding:18px 28px;border-left:4px solid #276875}.metric b{font-size:28px;display:block}.scroll{overflow:auto}input{padding:10px;width:350px;max-width:90%;margin:14px 0}</style><h1>第一问特征提取与时序对齐</h1>'''
    document+=f'<div class="metrics"><div class="metric">样本已生成<b>{stats["produced_samples"]}/100</b></div><div class="metric">词级位置<b>{n}</b></div><div class="metric">结构检查错误<b>{len(stats["structural_validation_errors"])}</b></div><div class="metric">低置信度待复核词<b>{stats["low_confidence_words"]}</b></div></div>'
    document+='<p>自动生成结果。覆盖率不代表对齐准确率；尚未进行人工边界误差测量。</p><h2>典型样本的时间对应</h2><p>'+html.escape(example['id']+'\n'+example['raw_text'])+'</p>'
    document+=picture('example_alignment.png')+picture('example_frames.jpg')+picture('example_text_features.png')+'<h2>全量质量分布</h2>'+picture('quality_overview.png')
    document+='<h2>100条样本汇总</h2><input id="filter" placeholder="筛选样本编号或复核提示"><div class="scroll">'+table+'</div><h2>方法和结果说明</h2>'+paragraphs
    document+='''<script>document.getElementById('filter').addEventListener('input',e=>{for(const r of document.querySelectorAll('tbody tr'))r.hidden=!r.textContent.toLowerCase().includes(e.target.value.toLowerCase())});</script></html>'''
    (out/'第一问结果报告.html').write_text(document,'utf-8')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'results')
    p.add_argument('--data-root',type=Path,default=ROOT.parent/'E题')
    args=p.parse_args()
    rows,stats,words=validate(args.output)
    plot_quality(args.output,rows)
    example=make_example(args.output,rows,args.data_root)
    review=[{**w,'manual_start_s':'','manual_end_s':'','review_note':''} for w in words if w['review_required'] or w['id']==example['id'] or any(c.isdigit() for c in w['word'])]
    write_csv(args.output/'manual_review_template.csv',review)
    build_report(args.output,rows,stats,example)
    print(json.dumps(stats,ensure_ascii=False,indent=2))
    if stats['structural_validation_errors']:
        raise SystemExit('Structural validation failed; do not submit before fixing')

if __name__=='__main__':
    main()
