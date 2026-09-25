#!/usr/bin/env python3
"""从已经保存的 valid/test 汇总结果生成第二问最终中文报告，不重读 test 原始数据。"""
import csv
import json
from pathlib import Path

import numpy as np

import problem2 as p2


HERE = Path(__file__).resolve().parent
OUT = p2.OUT / "final_summary"


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    numeric = {"accuracy", "macro_f1", "mae", "pearson", "nominal_rate",
               "accuracy_relative_drop", "macro_f1_relative_drop", "pearson_relative_drop",
               "mae_relative_increase"}
    for row in rows:
        for key in numeric & row.keys():
            row[key] = float(row[key])
    return rows


def fmt(row):
    return f"{row['accuracy']:.4f} / {row['macro_f1']:.4f} / {row['mae']:.4f} / {row['pearson']:.4f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    valid = [row for row in read_rows(p2.OUT / "modeling_experiment" / "validation_scenarios.csv")
             if row["model"] == "two_seed"]
    test = read_rows(p2.OUT / "final_test" / "test_metrics.csv")
    assert len(valid) == 50 and len(test) == 5
    valid_by = {row["scenario"]: row for row in valid}
    test_by = {row["scenario"]: row for row in test}
    assert set(test_by) == set(json.loads((HERE / "final_test_plan.json").read_text())["scenario_names"])
    audit = json.loads((p2.OUT / "data_audit.json").read_text())
    errors = json.loads((p2.OUT / "modeling_experiment" / "error_summary.json").read_text())
    test_summary = json.loads((p2.OUT / "final_test" / "summary.json").read_text())
    selection = json.loads((p2.OUT / "modeling_experiment" / "selection.json").read_text())
    assert selection["accepted"] and not test_summary["test_used_for_selection"]
    singles = [r for r in valid if r["missing_modality"] in p2.MODS]
    factors = []
    for factor, keys in (("modality", p2.MODS), ("position", ("front", "middle", "back")),
                         ("rate", tuple(p2.CONFIG["rates"]))):
        for key in keys:
            field = {"modality": "missing_modality", "position": "position", "rate": "nominal_rate"}[factor]
            matches = [r for r in singles if r[field] == key]
            factors.append({"factor": factor, "level": key, "conditions": len(matches),
                            "mean_accuracy": float(np.mean([r["accuracy"] for r in matches])),
                            "mean_macro_f1": float(np.mean([r["macro_f1"] for r in matches])),
                            "mean_mae": float(np.mean([r["mae"] for r in matches]))})
    p2.write_csv(OUT / "validation_factor_summary.csv", factors)
    p2.svg_plot([{"model": "final", **r} for r in valid], "final", OUT / "validation_missing_factors.svg")
    def factor_value(factor, key):
        return next(r["mean_macro_f1"] for r in factors if r["factor"] == factor and r["level"] == key)
    baseline = read_rows(p2.OUT / "validation_scenarios.csv")
    abl = [{"model": name, **next(r for r in baseline if r["model"] == name and r["scenario"] == "complete")}
           for name in ("baseline", "no_augment", "no_mask", "fixed", "full")]
    optimized = next(r for r in read_rows(p2.OUT / "optimization" / "validation_scenarios.csv") if r["scenario"] == "complete")
    labels = ("负向", "中性", "正向")
    cm = test_summary["confusion_matrix_complete"]
    f1 = [2 * cm[i][i] / max(1, sum(cm[i]) + sum(row[i] for row in cm)) for i in range(3)]
    lines = ["# 第二十三届中国研究生数学建模竞赛 E 题：第二问最终实验报告", "",
             "## 1. 任务与数据边界", "",
             "以附件2 `aligned_50.pkl` 的 `train` 训练，`valid` 选择模型及阈值；最终模型锁定后，对 `test` 做一次独立评估。附件3对齐版30条无标签样本只用于接口核查和最终预测，不参与训练、遮挡概率设定或调参。未使用其他情感数据集。", "",
             f"train {audit['split_counts']['train']['samples']} 条，valid {audit['split_counts']['valid']['samples']} 条，独立 test {test_summary['sample_count']} 条。核验标签关系：负向强度小于0，中性强度恰好等于0，正向强度大于0；连续强度范围为[-3,3]。统一输入为 `text_bert`（3×50）、`audio`（50×74）、`vision`（50×35），不使用仅附件2存在的768维 `text` 字段。附件3浮点存储的词元编号经整数性校验后转为 int64；源文件无 `id`，预测编号采用文件名。", "",
             "尾部填充由共同有效长度界定；文本有效位依据注意力掩码，音视频两端结构性零位排除。内部原始整行零值仍保留为有效位置，不推定每个零值都是人为缺失。定义有效掩码 $e_{mt}$、原始零值记录 $z_{mt}$ 与人工连续遮挡掩码 $s_{mt}$；模型观测掩码为 $o_{mt}=e_{mt}(1-s_{mt})$。", "",
             f"train 的音频第0位3395/3395条为零；有效内部原始零行涉及音频{audit['split_counts']['train']['samples_with_raw_zero_within_effective']['audio']}条、视觉{audit['split_counts']['train']['samples_with_raw_zero_within_effective']['vision']}条。附件3中29/30条的音视频原始零位模式完全相同，27/30条在文本有效时段内部存在零值；这些是数据观察，不是缺失成因标签。音视频标准化仅使用 train 各{audit['split_counts']['train']['effective_positions']['audio']}个有效位置计算均值和标准差，并保留掩码。", "",
             "## 2. 模型与训练", "",
             "普通基线对三模态做轻量时序编码与掩码均值/最大池化；完整鲁棒模型以覆盖率 $c_m=\\sum_t o_{mt}/\\sum_t e_{mt}$ 和模态表示 $r_m$ 计算样本级权重 $w_m=\\operatorname{softmax}_m(g_m([r_m,c_m]))$，再融合 $r=\\sum_m w_mr_m$。分类头输出三类概率，强度头输出 $3\\tanh(\\cdot)$。门控权重仅是模型内部计算量，不能直接解释为模态的因果作用。", "",
             "最终两个强化分支使用 [Google BERT Tiny](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2) 通用语言预训练编码器（2层、隐藏维128、2个注意力头）；预训练权重 SHA-256 为 `7fb69ad9f6866d8983183c930e33828f326470bf6ad8bbb2ad4ed957a92e9414`，训练时载入39个匹配张量。文本采用池化向量和有效词元均值，音视频采用线性层、一维卷积及掩码池化；融合层同时接收三模态表示和覆盖率。每个强化分支约449万参数，适合本机 CPU。", "",
             "优化模型在 train 上使用按类别频数平方根倒数加权的交叉熵与 $0.5\\times\\mathrm{Huber}_{\\delta=1}$；AdamW，批量64，最多12轮，耐心值3，BERT学习率3e-5，预测头1e-3，种子20260923与20260924。每轮保留约25%完整样本；其余按固定概率选择一或二个模态，在内部有效时段随机起点、按10/25/40/55/70%长度制造连续遮挡。没有随机散点置零。Pearson和完整/缺失一致性损失未加入最终模型：后者在 train 内检查折未达到采用门槛。", "",
             "最终类别概率为 $0.85(p_{B1}+p_{B2})/2+0.15p_{普通完整模型}$，强度为 $(u_{B1}+u_{B2})/2$。0.15分类融合权重由 valid 的有限候选网格选择，两个强化模型等权是预先固定的。所有分支使用同一输入与标准化接口。", "",
             "## 3. valid 模型比较与消融", "",
             "以下顺序均为 Accuracy / 宏 F1 / MAE / Pearson。普通模型消融使用同一固定验证遮挡方案；`no_mask` 仍保留有效位，但不显式屏蔽人工遮挡。", "",
             "|模型|完整输入 valid 四指标|", "|---|---|"]
    for row in abl:
        lines.append(f"|{row['model']}|{fmt(row)}|")
    lines += [f"|BERT Tiny 单分支|{fmt(optimized)}|", f"|最终双种子融合|{fmt(valid_by['complete'])}|", "",
              "最终融合相对单 BERT Tiny 的提升发生在用于选择的 valid 上，可能有选择偏倚。train 内另试完整/缺失一致性、文本缺失增强与分类专用组合：前两项未达预设 train 内联合门槛，分类专用组合在 valid 上虽有局部压力场景收益，但完整宏 F1 从0.5763降至0.5708，因此未替换最终模型。", "",
              "## 4. 固定的50组 valid 缺失实验", "",
              "含完整输入、3模态×3位置×5比例的45组单模态连续缺失、3组双模态中段40%缺失，以及仅文本可用。每组四指标与相对下降见[完整机器表](results/modeling_experiment/validation_scenarios.csv)，三因素曲线见[图](results/final_summary/validation_missing_factors.svg)。", "",
              "|场景|Accuracy|宏 F1|MAE|Pearson|宏 F1 相对完整下降|", "|---|---:|---:|---:|---:|---:|"]
    for name in ("complete", "text_middle_40", "text_middle_70", "audio_middle_70",
                 "vision_middle_70", "audio_vision_middle_40", "text_only"):
        r = valid_by[name]
        loss = (valid_by["complete"]["macro_f1"] - r["macro_f1"]) / valid_by["complete"]["macro_f1"]
        lines.append(f"|{name}|{r['accuracy']:.4f}|{r['macro_f1']:.4f}|{r['mae']:.4f}|{r['pearson']:.4f}|{loss:+.1%}|")
    lines += ["", "单模态45组的平均宏 F1：" + "、".join(f"{name} {factor_value('modality', name):.4f}" for name in p2.MODS) + "。文本缺失损害最大。",
              "位置平均宏 F1：" + "、".join(f"{name} {factor_value('position', name):.4f}" for name in ("front", "middle", "back")) + "；在当前样本和遮挡设置下，中段略低，不作因果解释。",
              "比例平均宏 F1：" + "、".join(f"{int(rate*100)}% {factor_value('rate', rate):.4f}" for rate in p2.CONFIG["rates"]) + "；总体随区间长度增加而下降。", "",
              "## 5. 锁定模型后的一次独立 test", "",
              f"测试方案及模型文件校验值先固化在[测试计划](final_test_plan.json)，锁定提交 `{test_summary['locked_commit'][:7]}`。test 共{test_summary['sample_count']}条；仅评价计划中的五个场景，不选择结构、权重或阈值，也不改动附件3预测。", "",
              "|test 场景|Accuracy|宏 F1|MAE|Pearson|Accuracy 相对完整下降|", "|---|---:|---:|---:|---:|---:|"]
    for row in test:
        lines.append(f"|{row['scenario']}|{row['accuracy']:.4f}|{row['macro_f1']:.4f}|{row['mae']:.4f}|{row['pearson']:.4f}|{row['accuracy_relative_drop']:+.1%}|")
    lines += ["", "独立 test 完整输入 Accuracy 0.6039，与 valid 的0.6044接近；宏 F1 由0.5763降至0.5511，MAE由0.6511升至0.7295。test 的文本中段70%缺失使 Accuracy 相对完整下降9.1%，说明文本信息受损仍是主要薄弱场景。五场景只是一组固定压力测试，不能代表所有可能的缺失机制。", "",
              "## 6. 错误分析与局限", "",
              f"最终模型在 valid 完整输入误分类{errors['misclassified']}/{errors['sample_count']}，中性类 F1 为{errors['class_f1'][1]:.4f}。test 完整输入的混淆矩阵如下（行真值、列预测；负/中/正）：", "",
              "|真值\\预测|负|中|正|", "|---|---:|---:|---:|"]
    for label, row in zip(labels, cm):
        lines.append(f"|{label}|{row[0]}|{row[1]}|{row[2]}|")
    lines += ["", f"test 各类 F1：负向{f1[0]:.4f}、中性{f1[1]:.4f}、正向{f1[2]:.4f}。中性158条中仅52条判对，72条判为正向，是宏 F1 的突出限制。valid 的12条高误差记录见[错误案例表](results/modeling_experiment/error_cases.csv)；其强度残差、有效长度和原始零值计数可辅助人工复核，但不能仅凭这些数字断定错误由哪一模态造成。", "",
              "原始整行零值的成因无法从特征单独判定；固定验证场景也不能覆盖所有真实缺失。valid 既用于轮次与方案选择又用于指标展示，存在选择偏倚。test 仅评估锁定模型一次；附件3没有真实标签，因此只交付预测，不能报告其 Accuracy 或 F1。", "",
              "## 7. 附件3结果和复现", "",
              "最终[预测 CSV](results/modeling_experiment/attachment3_predictions_two_seed.csv)有30条，样本编号与30个源文件名逐一对应，类别取0/1/2，强度限制在[-3,3]，并含三类概率。`python modeling_experiment.py verify` 从保存的模型重算并核验行数、编号唯一性、类别、概率、强度及数值一致性。", "",
              "`final_test.py` 使用锁定模型完成一次独立评价，防止在已生成结果时重复运行；`final_report.py` 仅读保存的汇总指标，可重复生成本报告与图表，不再次读取 test 原始数据。依赖、训练命令和权重来源见[README](README.md)。竞赛包不包含原始数据、虚拟环境和大型预训练源文件，当前目录低于50 MB。", ""]
    (HERE / "第二问最终报告.md").write_text("\n".join(lines), encoding="utf-8")
    print("generated final report, validation factor summary and plot")


if __name__ == "__main__":
    main()
