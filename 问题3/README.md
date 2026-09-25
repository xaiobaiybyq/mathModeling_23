# E题问题三：可解释性情感预测

本目录实现并保存了第三问的真实实验。成果总览见[第三问成果交接](第三问成果交接.md)，论文内容见[论文问题三正文](论文问题三正文.md)；完整20条输出见[附件4预测与解释CSV](results/attachment4_predictions_explanations.csv)。程序沿用[问题二锁定的完整集成模型](../问题2/README.md)，不使用附件4调模型或解释窗口参数，不读取附件2 test 标签。问题一生成的100条特征不直接输入问题二模型；问题一保存的英文词表和声学对齐模型只用于把附件4证据关联回原素材。

## 数据和方法

- 模型输入：附件2 `aligned_50.pkl` 的 train/valid 与附件4对齐版20条样本，统一使用 `text_bert`、`audio`、`vision`。附件4的 `text` 字段未作为额外模态。
- 标准化：直接使用问题二仅在 train 有效位置计算的 `results/normalization.json`。填充、原始零位、解释时制造的人工遮挡分别记录。
- 模型：问题二两个 BERT Tiny 动态融合分支的概率等权平均，再以15%权重融合普通动态融合模型；强度是两个 BERT Tiny 分支的均值。检查点已保存在问题二目录，本目录不复制。
- 解释：对完整集成输出的原预测类别概率逐窗遮挡，记录正负方向及强度变化。模态作用是该模态所有有效窗口上的平均绝对概率变化，三个模态归一化；原始全零窗口仍纳入此统计，但不作为可展示的关键内容证据。此数值同时反映特征移除与掩码覆盖率变化，是对遮挡的敏感度，不是因果贡献。
- 规则选择：valid 固定种子分层抽取90条，比较3、5、7位置窗口；取主模态关键窗口扩展2位置后的概率变化，与同模态同长度随机窗口比较，以配对差值除以扩展宽度平方根选择窗口。选择结果见 `results/selection.json`。
- 定位：用问题一的 DistilBERT uncased 词表精确核对附件4词元与 `raw_text`；用问题一保存的 `facebook/wav2vec2-base-960h` 在原视频语音上独立CTC强制对齐；按实际视频帧时间戳提取主证据及视觉证据帧。词元字符匹配已核验，但“附件4数值特征位置→原视频时间”的关系仍是近似推断，全部自动时间都标为需人工回听。

## 输出文件

|文件|内容|
|---|---|
|`第三问成果交接.md`|三问合稿所需的成果总览、数据边界和待核验事项|
|`results/validation_metrics.json`|完整valid的Accuracy、宏F1、MAE、Pearson|
|`results/validation_window_candidates.csv`|三种候选窗口的解释验证指标|
|`results/validation_explanation_cases.csv`|选定宽度的90条valid配对核查|
|`results/attachment4_predictions_explanations.csv`|附件4全量20条预测、三模态作用、关键位置、时间与帧|
|`results/evidence_windows.csv`|每条样本三个模态的关键证据或空证据说明|
|`results/target_windows.csv`|附件4所有连续窗口的扰动响应|
|`results/localization_diagnostics.csv`|逐样本词元匹配、CTC质量与视频帧核查|
|`results/manual_review_template.csv`|20条原视频的人工回听/回看记录模板，核验栏保持空白|
|`results/frames/*.png`|从原视频解码的证据帧|
|`results/modality_share.svg`、`results/local_importance_curve.svg`|论文图|

## 运行

本机使用 Python 3.8、NumPy 1.24.4、PyTorch 2.3.1+cpu、Transformers 4.40.2、PyAV 12.3.0、safetensors 0.5.3。依赖以问题二的隔离环境为基础，仅额外安装 `requirements.txt` 中的视频和声学推理库。原始题目附件需放在相邻 `../E题数据`；问题一声学权重和词表在相邻 `../问题1/models` 或 `../question1/models`；问题二锁定检查点在相邻 `../问题2/results`。Git 仓库只提供问题一模型清单，不包含大型预训练权重；重新做原视频自动定位前须按问题一 README 准备权重。无需下载新情感数据。

从 `E` 目录运行：

```bash
.problem2-venv/bin/python -m pip install -r 问题3/requirements.txt
.problem2-venv/bin/python 问题3/problem3.py all
.problem2-venv/bin/python 问题3/problem3.py verify
.problem2-venv/bin/python 问题3/report.py
.problem2-venv/bin/python 问题3/package_check.py
```

`all` 会重算valid完整指标、解释规则选择、附件4推理和全部原素材定位；`verify` 重新推理附件4并检查20行CSV的编号、类别、概率、强度、作用占比及证据字段，不重新运行CTC；`report.py` 只用已保存的真实指标、CSV重建论文图文。若只需排版论文，可直接使用已保存的报告、CSV、SVG和PNG。第三问不包含新的可训练参数文件；原始视频和问题一大型预训练权重不复制进本目录。附件4没有真实标签，不能报告其分类或回归准确度。

`package_check.py` 将问题一的100条特征及代码、问题二选定模型与核心结果、问题三全量CSV和论文使用的2张示例帧整合进 `技术附件候选.zip`。压缩包内问题一目录名为 `question1`，本程序可自动识别本机的 `问题1` 或打包后的 `question1`。候选包约47.6 MB，50 MB上限的剩余字节数见 `results/package_audit.json`。候选包未包含其余37张本地证据帧、原视频和大型预训练源权重；视觉证据的原视频编号、时间与帧序号仍完整保存在CSV，图片可运行 `all` 重建。候选包**尚未包含最终三问整合论文**，最终提交前须将论文加入并重新核验总大小。
