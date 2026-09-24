# E题第二问复现实验

## 当前推荐结果

当前推荐使用 `results/modeling_experiment/attachment3_predictions_two_seed.csv`：两个独立训练的 BERT Tiny 动态融合模型等权集成，再将分类概率与普通动态融合模型按 85%/15% 融合；强度为两个 BERT Tiny 回归输出的均值。完整 valid Accuracy 0.6044、宏 F1 0.5763、MAE 0.6511、Pearson 0.5461；完整50个缺失条件和退化场景见 `建模优化报告.md`。三个模型参数分别为 `results/optimization/tiny_weighted.pt`、`results/modeling_experiment/tiny_weighted.pt`、`results/checkpoint_full.pt`。先前推荐的单种子结果保留在 `results/calibration/attachment3_predictions_calibrated.csv`，比较见 `进一步优化报告.md`。数据边界审计见 `数据边界审计报告.md`；原优化模型结果见 `优化报告.md`。`results/checkpoint_fixed.pt` 保留供固定权重消融对照。修正前的小型指标文件已归档于 `results/pre_boundary_audit`，旧指标不能作为修正后模型的结果。整个 `问题2` 目录低于50 MB限制。

环境：Python 3.8、NumPy 1.24.4、PyTorch 2.3.1+cpu。使用附件2的 `aligned_50.pkl` 和附件3的对齐版30个文件；均按相对路径从 `../E题数据` 读取。本机已安装的隔离环境位于 `../.problem2-venv`，无需 sudo，直接运行：

```bash
../.problem2-venv/bin/python optimize.py verify
../.problem2-venv/bin/python optimize.py predict
../.problem2-venv/bin/python calibrate.py verify
../.problem2-venv/bin/python modeling_experiment.py verify
```

在其他有完整 `venv`/`pip` 的 Python 3.8 环境中可重新安装依赖：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

运行：

`problem2.py all` 从头训练五个模型，在验证集的固定50种条件下评价，选择模型、预测附件3并生成报告。`optimize.py all` 从头训练三种预训练方案并生成优化报告。两个脚本的 `predict` 用保存参数重新生成对应预测 CSV；`verify` 核验CSV并重算预测。`boundary_checks.py` 检查有效位、原始零值、人工遮挡的边界，汇总修正前后指标并生成审计报告。随机种子、五档缺失比例及参数分别见 `config.json` 与 `optimize_config.json`。附件2的 `test` 不参与选择或评估。

`calibrate.py run` 在固定 valid 面板上评估15个预设的分类融合/中性偏置候选，保存50场景指标和最终预测；`calibrate.py verify` 从上述两个检查点与 `results/calibration/selection.json` 重算30条预测。附件3仅在候选选定后读取。

`modeling_experiment.py run --pretrained /path/to/model.safetensors` 用固定的新种子再训练一次同结构模型，在预设 valid 五场景规则下评估等权双种子集成，并保存50场景指标和推荐预测。`modeling_experiment.py refresh` 从保存的参数重新生成50场景表、错误审阅并核验预测；`verify` 只核验30条预测，均无需预训练源文件。新模型权重以 float16 保存，加载到 float32 网络后推理。附件2 test 未读取；附件3只在模型采用后进行最终推理。

`problem2.py all` 可从头复现普通五模型与消融；优化版复用相同的有效位定义、训练集标准化统计和固定验证方案。优化版使用 [Google BERT Tiny](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2) 通用语言预训练权重，来源、架构和 SHA-256 均记录在 `优化报告.md`。要从头训练优化版，先从该模型页下载 `model.safetensors`，然后运行：

```bash
../.problem2-venv/bin/python optimize.py all --pretrained /path/to/model.safetensors
```

下载的预训练源文件只在初始化时使用，优化版保存参数已包含它的权重，`predict` 与 `verify` 无需再次下载。所有包只安装在独立虚拟环境中；虚拟环境、原始数据和下载文件不纳入竞赛提交附件。最终提交请按竞赛统一规范打包。
