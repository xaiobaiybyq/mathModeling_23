# 第一问多模态特征提取与时序对齐

本项目从附件1的100条原始视频生成三模态时序特征。保留原始样本、转写、标签和完整序列；不使用附件2的现成特征替代提取，不使用标签训练特征提取器。实际结果及已知局限见 `results/第一问结果报告.html` 和同名 Markdown 文件。

## 开始查看

1. 打开 `results/第一问结果报告.html`：查看质量统计、典型样本时间轴、视频帧和可筛选的100条汇总。
2. 打开 `results/sample_summary.csv`：每样本一行。`modality_summary.csv` 每样本每模态一行，共300行。
3. `results/features` 保存100个 NPZ，`results/metadata` 保存对应 JSON。
4. `results/manual_review_template.csv` 是人工回看记录模板，空白人工边界表示尚未核验，不表示误差为0。
5. `results/alignment_diagnostics.csv` 并列保存赛题转写和声学模型直接识别文本，优先回看平均对齐分数低于0.1的18条样本。它们可能存在文本与音频不匹配或识别失败，不应因为生成了合法CTC路径就宣称已可靠对齐。

## 环境与运行

测试环境为 Windows 和 Python 3.12；实际依赖版本见 `requirements-lock.txt`，模型修订号与 SHA256 见 `models/model_manifest.json`。CPU可运行，不要求GPU。本项目模型缓存约620MiB，虚拟环境另占空间；两者不打入竞赛提交包。

以下命令从包含 `question1` 的项目根目录执行。先用本机 Python 创建虚拟环境，然后安装依赖：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r question1/requirements-lock.txt --index-url https://pypi.org/simple
.venv/Scripts/python.exe question1/prepare_models.py
.venv/Scripts/python.exe question1/test_alignment.py
.venv/Scripts/python.exe question1/pipeline.py --data-root E题 --resume
.venv/Scripts/python.exe question1/summarize.py --data-root E题
.venv/Scripts/python.exe question1/read_features.py
.venv/Scripts/python.exe question1/package_results.py
```

`--data-root` 指向能够递归找到唯一 `label-100.xlsx` 的目录，该表旁应是37个视频子目录。提供的 `run.ps1` 支持 `-DataRoot` 参数。已有环境可以直接运行提取和汇总，无需再次安装。

预训练模型下载只访问公开模型，不上传数据。模型下载完成后，提取程序强制离线。准备脚本使用保存的模型修订号；如果完全删除 `models/model_manifest.json`，重新准备时会解析最新修订号，不能再称为完全相同的复现实验。

`--limit 2` 可先验证两条；随后去掉 limit 并使用 `--resume` 跑全量。resume核对处理代码、配置、模型清单、源视频和结果哈希，匹配才跳过；配置或代码变化需要重新计算。

Windows中文项目路径已作处理：openSMILE原配置原样复制到系统临时目录，仅通过内存传入音频，结束后清理该临时副本。若系统临时目录本身也含非ASCII字符，需要将TEMP设置到可写的英文路径。预训练Wav2Vec2可能提示`masked_spec_embed`未从检查点加载；该参数用于训练时频谱遮挡，本程序始终eval模式且不传遮挡索引，推理不使用它。不要因此对附件1重新训练模型。

## 方法与数学定义

### 1 原始资料登记

唯一编号为 `video_id$_$clip_id`。程序核对标签表100行与100条视频集合完全对应，不删除异常样本。原标签0只属于Neutral；标签仅进入metadata和汇总，不输入特征模型。

### 2 文本与强制对齐

按正则提取词，保留原文字符偏移、原始全文和子词映射。连字符两侧视为两个词；撇号保留。DistilBERT冻结参数，每100词为一块编码，不截断；超过模型子词长度时明确报错。当前数据不需要长文本分块。词向量取所属子词最后隐层向量的平均，维度768。

Wav2Vec2输出CTC字符概率。将原始转写按明确规则大写化，并展开少数数字读法。Viterbi动态规划在blank和字符交替的状态序列上寻找最大对数概率路径：每步可原地、前进1个状态，或在字符不同且非blank的条件下前进2个状态。由字符的首末发射帧得到词区间。卷积感受野中心及步长转换为秒，不把全部单词均匀摊在整段时长上。

数字规则只做声学匹配所需的表示变换：2008→TWO THOUSAND EIGHT；1989→NINETEEN EIGHTY NINE；10th→TENTH；1,500→FIFTEEN HUNDRED；100,000→ONE HUNDRED THOUSAND；20,000→TWENTY THOUSAND；50→FIFTY。原表文本不改写，具体读法应回听核对。

每词score为CTC路径上各字符发射概率的几何平均；阈值0.1仅用作质量复核提示，未经边界真值校准，不作为精度指标。不确定位置保留。路径无解则明确alignment_mask=False，同时保存其原始模态序列，不能伪造时间戳。

### 3 音频

PyAV解码并重采样为16kHz单声道，按照真实PTS放在容器起始时间为0的公共轴上。记录实际观测范围，时间间隙不视作真实静音。openSMILE eGeMAPSv02低层描述符25维，使用工具返回的帧起止时间。具体名称见feature_names.json。

对词区间 I_i 与帧区间 J_j，定义 w_ij=max(0,min(e_i,e_j)-max(s_i,s_j))*m_j。计算加权均值μ_i与加权标准差σ_i，按[μ_i,σ_i]拼接为50维。语音有效mask只表明存在合格源特征，不代表该词的对齐已人工确认。

### 4 视频

按真实帧PTS进行10Hz目标抽样，图像最大宽度640，保留零起始的原视频帧序号。MediaPipe Face Landmarker输出52个面部blendshape系数；对面部变换矩阵旋转部分做SVD正交化，求xyz欧拉角3维。共55维原始视觉特征。它们不是情绪类别，也不是OpenFace/FACS动作单元。

每帧最多检测3张脸，选面积最大者，检测阈值0.5。检测失败时原始视觉mask=False；多张脸时保存数量及所选框，提醒检查，未做说话人身份确认。

每个采样帧的时间支持区间由相邻采样时间中点界定，并限制不超过左右各0.075秒，避免跨越大时间间隙。按词区间重叠加权聚合均值、标准差，得到110维视觉特征。这是一种明确的采样保持近似，不是每个词都重新截视频做人脸识别。

### 5 时间与填充规则

词、声学帧和视频帧时间均为相对片段时间（秒）。词区间可以包含词内部CTC blank；词之间未分配的区间单独保存，不能全部解释为静音。原始语音、视觉时序均保留，因此词外信息没有从结果中删除。

磁盘保存变长完整序列，不截成50词，不做padding。`valid_length`为实际词数，不等于所有模态都有效的词数。`padding_mask`在原文件中全False；`read_features.collate`批量补0，补齐位置padding_mask=True。`audio_mask/vision_mask=True`表示该位置有有效观测，`alignment_mask=True`表示存在合法强制对齐路径，`alignment_review_mask=True`表示需复核。

## 结果字段

| NPZ字段 | 形状/含义 |
|---|---|
| text | L×768，词级上下文向量 |
| audio | L×50，25维低层声学描述符的均值和标准差 |
| vision | L×110，55维面部特征的均值和标准差 |
| word_intervals | L×2，开始/结束秒数 |
| word_char_offsets | L×2，原始文本左闭右开字符偏移 |
| alignment_confidence | L，未校准的CTC路径分数 |
| *_mask | 有效性、对齐复核或填充标记，详见上文 |
| audio_raw / audio_intervals | A×25 / A×2，原始声学特征和帧时间 |
| vision_raw / vision_intervals | V×55 / V×2，原始视觉特征和采样支持区间 |
| vision_pts / vision_frame_ids | V，真实时间和零起始原视频帧序号 |
| face_counts / face_boxes | 检出脸数量及所选脸归一化框xmin,ymin,xmax,ymax |
| unassigned_intervals | 不属于已对齐词区间的时间段，保留原始模态数据 |

metadata中的`audio_source_indices_per_word`与`vision_source_indices_per_word`索引对应NPZ原始帧级数组。视觉索引还可通过`vision_frame_ids`还原到原视频帧。没有有效贡献时，对应列表为空。

## 质量校验与论文使用

汇总程序检查100条覆盖、原文字符映射、时间单调性、边界范围、维度、有限值、mask语义及文件哈希，并从原始帧重算全部词级声学与视觉聚合，核对结果和源索引。`test_alignment.py`另外测试重复字符、不可能路径、重叠加权和缺失区间。

聚合计算使用float64累加，最终以float32存储，避免常值大数特征出现浮点舍入造成的伪标准差。本次通过`reaggregate.py`从保留的原始帧特征重新计算了聚合数值，原特征、时间和掩码不变；变更前后哈希和原提取代码版本已登记。首次从头运行pipeline会直接使用修正后的累加方式，无需额外运行该脚本。

这些自动检查验证程序与数据的一致性，不提供人工对齐准确率。报告中给出实际覆盖率、低置信度数量、检测率、典型图及缺陷。若要报告边界误差，应先对抽样片段人工听看标注，填入manual_review_template.csv，再计算误差；不要把20ms模型步长当成20ms边界误差。

论文应呈现方法公式、关键参数、实际结果和100条全量汇总，并展示至少一个词—语音—视频帧对应案例。第一问无需情感预测分数；与第二、三问的768/74/35维现成接口不相同，不能直接把本结果送入按附件2训练的模型。

## 提交大小

`package_results.py`只打包代码、配置、锁定版本、模型清单及结果，不包含虚拟环境、原视频和可重新下载的预训练权重。输出question1_submission.zip，并打印实际大小。整个竞赛附件50MB限额还要留给第二、三问，第一问压缩包不能单独视为整题提交完成。

## 官方来源

- https://huggingface.co/distilbert/distilbert-base-uncased
- https://huggingface.co/facebook/wav2vec2-base-960h
- https://audeering.github.io/opensmile-python/usage.html
- https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python
