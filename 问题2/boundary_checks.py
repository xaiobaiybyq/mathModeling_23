#!/usr/bin/env python3
"""核验第二问数据边界，并汇总修正前后的真实验证指标。"""
import csv
import json
from pathlib import Path

import numpy as np

import problem2 as p2


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "results" / "pre_boundary_audit"


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def mask_checks():
    data = p2.load_data()  # 仅读取 train、valid
    stats = p2.fit_normalization(data["train"])
    summary = {}
    for name, x in data.items():
        assert not x["synthetic_missing"].any()
        assert np.array_equal(x["obs"], x["effective"])
        internal = x["raw_zero"] & x["effective"]
        summary[name] = {
            "samples": len(x["id"]),
            "raw_zero_inside_effective_positions": {p2.MODS[j]: int(internal[:, j].sum()) for j in range(3)},
            "samples_with_raw_zero_inside_effective": {p2.MODS[j]: int(internal[:, j].any(1).sum()) for j in range(3)},
            "audio_first_position_zero_samples": int(x["raw_zero"][:, 1, 0].sum()),
            "vision_first_position_zero_samples": int(x["raw_zero"][:, 2, 0].sum()),
            "audio_last_position_zero_samples": int(x["raw_zero"][np.arange(len(x["id"])), 1, x["lengths"] - 1].sum()),
            "vision_last_position_zero_samples": int(x["raw_zero"][np.arange(len(x["id"])), 2, x["lengths"] - 1].sum()),
        }
        p2.transform(x, stats)
        for j, key in enumerate(("audio", "vision"), 1):
            assert np.all(x[key][~x["effective"][:, j]] == 0)
            zeros = internal[:, j]
            expected = np.clip(-np.asarray(stats[key]["mean"]) / np.asarray(stats[key]["std"]), -8, 8)
            if zeros.any():
                assert np.allclose(x[key][zeros], expected.astype(np.float32), atol=2e-5)
    sample = p2.subset(data["valid"], slice(0, 32))
    p2.apply_missing(sample, ("audio", "vision"), 0.4, "middle")
    assert np.array_equal(sample["obs"], sample["effective"] & ~sample["synthetic_missing"])
    assert not sample["synthetic_missing"][:, 0].any()
    assert sample["synthetic_missing"][:, 1:].any()
    only_text = p2.subset(data["valid"], slice(0, 32))
    p2.apply_missing(only_text, ("audio", "vision"), 1.0, "all")
    assert np.array_equal(only_text["obs"][:, 0], only_text["effective"][:, 0])
    assert not only_text["obs"][:, 1:].any()
    assert np.array_equal(only_text["synthetic_missing"][:, 1:], only_text["effective"][:, 1:])
    x = p2.load_targets(stats)
    target = p2.write_target_boundary_audit(x)
    target["audio_first_position_zero_samples"] = int(x["raw_zero"][:, 1, 0].sum())
    target["vision_first_position_zero_samples"] = int(x["raw_zero"][:, 2, 0].sum())
    target["audio_last_position_zero_samples"] = int(x["raw_zero"][np.arange(len(x["id"])), 1, x["lengths"] - 1].sum())
    target["vision_last_position_zero_samples"] = int(x["raw_zero"][np.arange(len(x["id"])), 2, x["lengths"] - 1].sum())
    assert target["samples"] == 30 and target["text_attention_continuous_samples"] == 30
    assert all(summary[name][k] == summary[name]["samples"] for name in ("train", "valid") for k in ("audio_first_position_zero_samples", "vision_first_position_zero_samples", "audio_last_position_zero_samples", "vision_last_position_zero_samples"))
    assert all(target[k] == 30 for k in ("audio_first_position_zero_samples", "vision_first_position_zero_samples", "audio_last_position_zero_samples", "vision_last_position_zero_samples"))
    assert target["synthetic_missing_positions"] == 0
    assert len(p2.scenario_list()) == 50
    return {"train_valid": summary, "target": target,
            "checks": ["train-only normalization includes internal raw zeros", "padding excluded",
                       "original zero and synthetic missing masks separate", "AV local and text-only masks valid",
                       "fixed 50-scenario panel", "attachment3 unified interface"]}


def comparisons():
    files = [
        ("ordinary", ARCHIVE / "results__validation_scenarios.csv", HERE / "results" / "validation_scenarios.csv",
         ARCHIVE / "results__selection.json", HERE / "results" / "selection.json"),
        ("optimized", ARCHIVE / "results__optimization__validation_scenarios.csv",
         HERE / "results" / "optimization" / "validation_scenarios.csv",
         ARCHIVE / "results__optimization__selection.json", HERE / "results" / "optimization" / "selection.json"),
    ]
    out = []
    for family, old_path, new_path, old_sel_path, new_sel_path in files:
        if not all(p.exists() for p in (old_path, new_path, old_sel_path, new_sel_path)):
            continue
        old_sel = json.loads(old_sel_path.read_text())
        new_sel = json.loads(new_sel_path.read_text())
        old_name = old_sel["selected"] if family == "ordinary" else old_sel["selected"]["variant"]
        new_name = new_sel["selected"] if family == "ordinary" else new_sel["selected"]["variant"]
        old_rows = read_rows(old_path)
        new_rows = read_rows(new_path)
        pairs = [("ordinary_selected", old_name, new_name), ("ordinary_fixed", "fixed", "fixed")] if family == "ordinary" else [("optimized_selected", old_name, new_name)]
        for label, old_model, new_model in pairs:
            prior = [r for r in old_rows if family != "ordinary" or r["model"] == old_model]
            current = [r for r in new_rows if family != "ordinary" or r["model"] == new_model]
            old = {r["scenario"]: r for r in prior}
            new = {r["scenario"]: r for r in current}
            for scenario in ("complete", "text_middle_40", "audio_middle_40", "vision_middle_40", "audio_vision_middle_40", "text_only"):
                if scenario not in new:
                    continue
                row = {"family": label, "scenario": scenario, "old_model": old_model, "corrected_model": new_model,
                       "old_available": scenario in old}
                for metric in ("accuracy", "macro_f1", "mae", "pearson"):
                    row["old_" + metric] = old[scenario][metric] if scenario in old else ""
                    row["corrected_" + metric] = new[scenario][metric]
                    row["corrected_minus_old_" + metric] = (float(new[scenario][metric]) - float(old[scenario][metric])) if scenario in old else ""
                out.append(row)
    if out:
        p2.write_csv(HERE / "results" / "boundary_before_after.csv", out)
    return len(out)


def write_report(summary):
    root = HERE / "results"
    before = read_rows(root / "boundary_before_after.csv")
    ordinary = read_rows(root / "validation_scenarios.csv")
    optimized = read_rows(root / "optimization" / "validation_scenarios.csv")
    old_selected = json.loads((ARCHIVE / "results__selection.json").read_text())["selected"]
    new_selected = json.loads((root / "selection.json").read_text())["selected"]
    old_opt = json.loads((ARCHIVE / "results__optimization__selection.json").read_text())["selected"]["variant"]
    new_opt = json.loads((root / "optimization" / "selection.json").read_text())["selected"]["variant"]
    def fmt(r, prefix=""):
        return " / ".join(f"{float(r[prefix + m]):.4f}" for m in ("accuracy", "macro_f1", "mae", "pearson"))
    lines = ["# 第二问数据边界情况审计", "",
             "## 范围与判定", "",
             "本次只读附件2 `aligned_50.pkl` 的 train、valid；附件2 test 未读取。附件3的30条对齐样本仅用于接口核查及最终推理，其零值比例没有进入训练采样概率、模型选择或阈值。原始整行零值是观察结果，不能标注为人为缺失。", "",
             "共同时间轴在有效末端后为填充。文本有效位置由 `text_bert` 注意力位确定；音频和视觉两端特殊位置排除，内部位置即使原始整行零值也保留。人工遮挡由连续区间生成过程单独记录，模型观测掩码等于有效位置减去人工遮挡；原始零值掩码仅用于审计。", "",
             "|数据|条数|语音第0位零值条数|视觉第0位零值条数|内部语音零值条数/位置|内部视觉零值条数/位置|", "|---|---:|---:|---:|---:|---:|"]
    for name in ("train", "valid"):
        s = summary["train_valid"][name]
        lines.append(f"|{name}|{s['samples']}|{s['audio_first_position_zero_samples']}|{s['vision_first_position_zero_samples']}|{s['samples_with_raw_zero_inside_effective']['audio']} / {s['raw_zero_inside_effective_positions']['audio']}|{s['samples_with_raw_zero_inside_effective']['vision']} / {s['raw_zero_inside_effective_positions']['vision']}|")
    t = summary["target"]
    lines += ["", f"两端排除规则依据附件2 train/valid 的全零结构制定；附件3仅作接口核验，其音视频两端也全部为零。附件3：文本注意力连续 {t['text_attention_continuous_samples']}/30；音视频原始零位模式完全相同 {t['audio_vision_identical_raw_zero_pattern_samples']}/30；音视频有效内部至少一处原始零值 {t['samples_with_audio_or_vision_raw_zero_inside_effective']}/30。附件3文件没有 `id` 字段，预测编号取源文件名。", "",
              "## 代码与实验修正", "",
              "旧版把音视频整行零值直接从观测掩码、标准化和池化中排除，并用共同长度作为覆盖率分母。这会影响参数与指标。修正后，标准化仅由 train 的有效内部位置计算，包含内部原始零值；模型池化用观测掩码，覆盖率以各模态有效位置数为分母。连续遮挡和原始零值分离。普通模型五组及 BERT Tiny 三组均重新训练、选轮次、验证与预测。", "",
              "固定验证集共50场景：完整输入、45组单模态缺失、3组双模态中段40%局部缺失，以及音视频内部全遮挡的“仅文本可用”。语音与视觉同时局部缺失单列为 `audio_vision_middle_40`。每组四指标与相对下降见两个 `validation_scenarios.csv`。", "",
              "## 修正前后对照", "",
              "以下指标均来自真实保存的验证结果，顺序为 Accuracy / 宏 F1 / MAE / Pearson。旧结果仅作审计对照，不能当作修正后模型指标。模型选择可能变化，因此“所选模型”行不能单独用于归因代码改动。", "",
              f"普通所选模型：旧 {old_selected}，新 {new_selected}；优化所选模型：旧 {old_opt}，新 {new_opt}。", "",
              "|模型族|场景|修正前四指标|修正后四指标|", "|---|---|---|---|"]
    for family in ("ordinary_selected", "ordinary_fixed", "optimized_selected"):
        for scenario in ("complete", "audio_vision_middle_40", "text_only"):
            r = next(x for x in before if x["family"] == family and x["scenario"] == scenario)
            old = fmt(r, "old_") if r["old_available"] == "True" else "旧实验未设置"
            lines.append(f"|{family}|{scenario}|{old}|{fmt(r, 'corrected_')}|")
    lines += ["", "## 修正后消融与压力场景", "",
              "下表为修正后普通模型的真实验证指标，顺序同上。固定权重融合、无人工遮挡显式掩码、无连续缺失增强与普通基线均单列；`no_mask` 仍保留有效位以排除填充。优化版也按同一遮挡方案评价。", "",
              "|模型|完整输入|音视频同时中段40%缺失|仅文本可用|", "|---|---|---|---|"]
    for model in ("baseline", "no_augment", "no_mask", "fixed", "full"):
        def ordinary_row(s): return next(r for r in ordinary if r["model"] == model and r["scenario"] == s)
        lines.append(f"|{model}|{fmt(ordinary_row('complete'))}|{fmt(ordinary_row('audio_vision_middle_40'))}|{fmt(ordinary_row('text_only'))}|")
    def opt_row(s): return next(r for r in optimized if r["scenario"] == s)
    lines.append(f"|{new_opt}|{fmt(opt_row('complete'))}|{fmt(opt_row('audio_vision_middle_40'))}|{fmt(opt_row('text_only'))}|")
    lines += ["", "优化模型在“音视频同时中段40%缺失”和“仅文本可用”场景的宏 F1 略高于完整输入。这是此固定验证集上的非单调现象；不能据此断定音视频普遍有害或原始零值的成因。", "",
              "## 复核与限制", "",
              "`boundary_checks.json` 记录掩码及标准化断言；`boundary_before_after.csv` 给出可机读的前后对照。两个预测 CSV 均可由保存参数重新生成，并逐项核验30行、源文件编号唯一、类别、概率和强度范围。附件3无标签，不能计算其准确率。验证集既用于选模型又用于报告结果，存在选择偏倚；零值的真实形成原因不能从这些特征单独确定。", ""]
    (HERE / "数据边界审计报告.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    summary = mask_checks()
    summary["before_after_rows"] = comparisons()
    (HERE / "results" / "boundary_checks.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if summary["before_after_rows"] and (HERE / "results" / "optimization" / "validation_scenarios.csv").exists():
        write_report(summary)
    print("boundary checks passed; comparison rows:", summary["before_after_rows"])


if __name__ == "__main__":
    main()
