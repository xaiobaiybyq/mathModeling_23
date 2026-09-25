#!/usr/bin/env python3
"""一次预设随机种子重训与等权双模型集成；仅用 valid 决定是否采用。"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import problem2 as p2
import optimize as opt
import calibrate as cal


OUT = p2.OUT / "modeling_experiment"
OLD_CHECKPOINT = opt.OUT / "tiny_weighted.pt"
NEW_CHECKPOINT = OUT / "tiny_weighted.pt"
SEED = 20260924
ORDINARY_WEIGHT = 0.15  # 沿用已经选定的权重，不在本实验重新搜索


def outputs(old, new, ordinary, x, scenario):
    condition = None if scenario["name"] == "complete" else scenario
    a, ar, _ = p2.predict_arrays(old, x, condition, batch_size=opt.CONFIG["eval_batch_size"])
    b, br, _ = p2.predict_arrays(new, x, condition, batch_size=opt.CONFIG["eval_batch_size"])
    c, _, _ = p2.predict_arrays(ordinary, x, condition)
    original = cal.probabilities(a, c, ORDINARY_WEIGHT, 0.0)
    candidate = (1.0 - ORDINARY_WEIGHT) * (a + b) / 2.0 + ORDINARY_WEIGHT * c
    return (original, ar), (candidate, (ar + br) / 2.0)


def score(metrics):
    return (metrics["accuracy"] + metrics["macro_f1"]) / 2.0 - metrics["mae"] / 3.0


def scenario_rows(old, new, ordinary, valid, scenarios):
    rows = []
    for scenario in scenarios:
        baseline, candidate = outputs(old, new, ordinary, valid, scenario)
        for name, (probs, reg) in (("current", baseline), ("two_seed", candidate)):
            rows.append({"scenario": scenario["name"], "model": name,
                         "missing_modality": "+".join(scenario["mods"]) or "none",
                         "position": scenario["place"], "nominal_rate": scenario["rate"],
                         **p2.metrics(valid["class"], valid["strength"], probs, reg)})
    return rows


def analyze_errors(old, new, ordinary, valid):
    _, (probs, reg) = outputs(old, new, ordinary, valid, {"name": "complete"})
    pred = probs.argmax(axis=1)
    truth = valid["class"]
    confusion = [[int(((truth == i) & (pred == j)).sum()) for j in range(3)] for i in range(3)]
    class_f1 = opt.class_f1(truth, pred)
    errors = np.flatnonzero(pred != truth)
    # 排名仅用于错误审阅，字段不含原始文本、音视频特征。
    rank = errors[np.argsort(-np.abs(reg[errors] - valid["strength"][errors]))[:12]]
    rows = [{"sample_id": valid["id"][i], "true_class": int(truth[i]),
             "pred_class": int(pred[i]), "true_strength": float(valid["strength"][i]),
             "predicted_strength": float(reg[i]),
             "absolute_error": float(abs(reg[i] - valid["strength"][i])),
             "valid_length": int(valid["lengths"][i]),
             "raw_zero_audio_effective": int((valid["raw_zero"][i, 1] & valid["effective"][i, 1]).sum()),
             "raw_zero_vision_effective": int((valid["raw_zero"][i, 2] & valid["effective"][i, 2]).sum())}
            for i in rank]
    p2.write_csv(OUT / "error_cases.csv", rows)
    counts = {"confusion_matrix": confusion, "class_f1": class_f1,
              "misclassified": int(len(errors)), "sample_count": int(len(truth)),
              "errors_with_original_av_zero": int(((valid["raw_zero"][errors, 1:, :]
                  & valid["effective"][errors, 1:, :]).any(axis=(1, 2))).sum())}
    (OUT / "error_summary.json").write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
    return counts


def choose(rows):
    names = [s["name"] for s in p2.selection_panel()]
    grouped = {m: {r["scenario"]: r for r in rows if r["model"] == m}
               for m in ("current", "two_seed")}
    scores = {}
    for model, values in grouped.items():
        panel = float(np.mean([score(values[n]) for n in names]))
        scores[model] = (panel + score(values["complete"])) / 2.0
    old, new = grouped["current"]["complete"], grouped["two_seed"]["complete"]
    accepted = (new["accuracy"] >= old["accuracy"] - 1e-9
                and new["macro_f1"] >= old["macro_f1"] - 1e-9
                and scores["two_seed"] > scores["current"] + 1e-4)
    return {"accepted": bool(accepted), "scores": scores,
            "current_complete": {k: old[k] for k in ("accuracy", "macro_f1", "mae", "pearson")},
            "two_seed_complete": {k: new[k] for k in ("accuracy", "macro_f1", "mae", "pearson")},
            "rule": "mean of complete and five-scenario panel [(Accuracy+macroF1)/2-MAE/3]; complete Accuracy and macroF1 must not decrease",
            "new_seed": SEED, "ordinary_weight": ORDINARY_WEIGHT,
            "ensemble_bert_weights": [0.5, 0.5],
            "test_used": False, "target_used_for_selection": False}


def save_target(old, new, ordinary, stats):
    x = p2.load_targets(stats)
    scenario = {"name": "complete"}
    _, (probs, reg) = outputs(old, new, ordinary, x, scenario)
    rows = []
    for name, prob, value in zip(x["id"], probs, reg):
        k = int(prob.argmax())
        rows.append({"sample_id": name, "source_file": name + ".pkl", "polarity_class": k,
                     "polarity_name": p2.NAMES[k], "predicted_strength": round(float(value), 6),
                     "prob_negative": round(float(prob[0]), 6),
                     "prob_neutral": round(float(prob[1]), 6),
                     "prob_positive": round(float(prob[2]), 6)})
    p2.write_csv(OUT / "attachment3_predictions_two_seed.csv", rows)
    verify(old, new, ordinary, stats)


def verify(old=None, new=None, ordinary=None, stats=None):
    selection = json.loads((OUT / "selection.json").read_text())
    assert selection["accepted"]
    if stats is None:
        stats = json.loads((p2.OUT / "normalization.json").read_text())
    if old is None: old = opt.load_model(OLD_CHECKPOINT)
    if new is None: new = opt.load_model(NEW_CHECKPOINT)
    if ordinary is None: ordinary = p2.load_model("full")
    x = p2.load_targets(stats)
    _, (probs, reg) = outputs(old, new, ordinary, x, {"name": "complete"})
    with (OUT / "attachment3_predictions_two_seed.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 30 and len({r["sample_id"] for r in rows}) == 30
    assert {r["sample_id"] for r in rows} == {p.stem for p in p2.TARGET.glob("附件3_*.pkl")}
    for name, row, prob, value in zip(x["id"], rows, probs, reg):
        k = int(prob.argmax())
        assert row["sample_id"] == name and row["source_file"] == name + ".pkl"
        assert int(row["polarity_class"]) == k and row["polarity_name"] == p2.NAMES[k]
        assert -3 <= float(row["predicted_strength"]) <= 3
        assert abs(float(row["predicted_strength"]) - float(value)) < 1e-6
        assert abs(sum(float(row[f"prob_{label}"]) for label in ("negative", "neutral", "positive")) - 1) < 2e-6
        for label, p in zip(("negative", "neutral", "positive"), prob):
            assert abs(float(row[f"prob_{label}"]) - float(p)) < 1e-6
    return True


def run(pretrained):
    OUT.mkdir(parents=True, exist_ok=True)
    p2.seed_everything(SEED)
    data = p2.load_data()
    stats = json.loads((p2.OUT / "normalization.json").read_text())
    for split in data.values(): p2.transform(split, stats)
    # 复用现有架构和训练规则；新 checkpoint 独立保存，不覆盖现行模型。
    opt.OUT = OUT
    opt.CONFIG["seed"] = SEED
    history = []
    info = opt.train_variant("tiny_weighted", data["train"], data["valid"],
                             Path(pretrained), history)
    p2.write_csv(OUT / "training_history.csv", history)
    (OUT / "training_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    # 保存为半精度以留出竞赛附件容量；加载到 float32 网络后再评估和选择。
    checkpoint = torch.load(NEW_CHECKPOINT, map_location="cpu", weights_only=False)
    checkpoint["state"] = {key: (value.half() if value.is_floating_point() else value)
                           for key, value in checkpoint["state"].items()}
    checkpoint["storage_precision"] = "float16; restored to float32 on load"
    torch.save(checkpoint, NEW_CHECKPOINT)
    old = opt.load_model(OLD_CHECKPOINT)
    new = opt.load_model(NEW_CHECKPOINT)
    ordinary = p2.load_model("full")
    valid = data["valid"]
    panel = scenario_rows(old, new, ordinary, valid, p2.selection_panel())
    p2.write_csv(OUT / "selection_panel.csv", panel)
    choice = choose(panel)
    (OUT / "selection.json").write_text(json.dumps(choice, ensure_ascii=False, indent=2), encoding="utf-8")
    # 50 个固定遮挡条件只用于报告；采用规则已在上面固定并作出决定。
    all_rows = scenario_rows(old, new, ordinary, valid, p2.scenario_list())
    p2.write_csv(OUT / "validation_scenarios.csv", all_rows)
    error_summary = analyze_errors(old, new, ordinary, valid)
    if choice["accepted"]:
        save_target(old, new, ordinary, stats)
    else:
        NEW_CHECKPOINT.unlink()  # 未采用的本轮临时权重无需占用提交容量
    summarize(choice, all_rows, error_summary)
    print(json.dumps(choice, ensure_ascii=False), flush=True)


def summarize(choice, rows, error_summary):
    prior = {r["scenario"]: r for r in rows if r["model"] == "current"}
    newer = {r["scenario"]: r for r in rows if r["model"] == "two_seed"}
    improved_acc = sum(newer[k]["accuracy"] > prior[k]["accuracy"] for k in prior)
    improved_f1 = sum(newer[k]["macro_f1"] > prior[k]["macro_f1"] for k in prior)
    def fmt(m):
        return f"Accuracy {m['accuracy']:.4f}，宏 F1 {m['macro_f1']:.4f}，MAE {m['mae']:.4f}，Pearson {m['pearson']:.4f}"
    report = ["# 第二问建模优化：随机种子集成实验", "",
              "## 预设方案", "",
              "另以种子 20260924 训练一次同结构、同损失、同连续区间增强的 BERT Tiny 动态融合模型。两个 BERT Tiny 模型的类别概率和强度等权平均；类别再按已选定的 0.15 权重融合普通动态模型。没有按附件3或新验证结果搜索集成权重。训练仅用附件2 train，轮次和是否采用仅用 valid；附件2 test 未读取。", "",
              "已有模型和新模型都按相同有效位、原始零值和人工遮挡接口输入。验证使用原有固定 5 场景作采用判断，50 场景作全面记录。新权重保存为 float16，加载后恢复到 float32；所有下述指标均由压缩后的权重复算。", "",
              "## 真实 valid 结果", "",
              f"现行模型完整输入：{fmt(choice['current_complete'])}。双种子候选完整输入：{fmt(choice['two_seed_complete'])}。", "",
              f"现行选择分数 {choice['scores']['current']:.4f}；候选 {choice['scores']['two_seed']:.4f}。预设采用规则：完整输入 Accuracy、宏 F1 不下降，且选择分数至少提高 0.0001。结果：{'采用' if choice['accepted'] else '不采用'}。", "",
              "|场景|现行 Accuracy|候选 Accuracy|现行宏 F1|候选宏 F1|现行 MAE|候选 MAE|", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("complete", "text_middle_70", "audio_vision_middle_40", "text_only"):
        a, b = prior[name], newer[name]
        report.append(f"|{name}|{a['accuracy']:.4f}|{b['accuracy']:.4f}|{a['macro_f1']:.4f}|{b['macro_f1']:.4f}|{a['mae']:.4f}|{b['mae']:.4f}|")
    report += ["", f"50 个固定条件中，候选 Accuracy 高于现行模型 {improved_acc} 个，宏 F1 高于现行模型 {improved_f1} 个；完整数据见 `results/modeling_experiment/validation_scenarios.csv`。严重文本缺失与仅文本可用场景的分类准确率下降，说明本次优化没有消除所有失效情形。", "",
               "## 错误审阅", "",
               "完整输入的混淆矩阵（行是真值，列是预测，顺序负/中/正）：", "",
               "|真值\\预测|负|中|正|", "|---|---:|---:|---:|"]
    for name, values in zip(p2.NAMES, error_summary["confusion_matrix"]):
        report.append(f"|{name}|{values[0]}|{values[1]}|{values[2]}|")
    report += ["", f"各类 F1：负向 {error_summary['class_f1'][0]:.4f}、中性 {error_summary['class_f1'][1]:.4f}、正向 {error_summary['class_f1'][2]:.4f}。"
               f"完整输入误分类 {error_summary['misclassified']}/{error_summary['sample_count']}；其中 {error_summary['errors_with_original_av_zero']} 条在有效音视频区间内存在原始整行零值。"
               "`error_cases.csv` 保存12条高强度误差的误分类记录供人工核对。原始零值、模态权重和错误共现不足以确定因果原因。", "",
               "## 结论与限制", "",
               ("采用双种子集成，附件3的30条预测见 `results/modeling_experiment/attachment3_predictions_two_seed.csv`；运行 `python modeling_experiment.py verify` 可复算核验。" if choice["accepted"] else "候选未达到预设采用规则，保留原 `results/calibration/attachment3_predictions_calibrated.csv` 为最终预测。候选模型权重已删除，可运行脚本重训复现。"), "",
               "当前结果仍在已反复用于结构选择的 valid 上评估，有选择偏倚；不能据此推断附件3准确率。新种子实验只是一轮对训练随机性的有限检验。", ""]
    (p2.HERE / "建模优化报告.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "verify", "refresh"))
    parser.add_argument("--pretrained", default="/tmp/bert-tiny.safetensors",
                        help="Google BERT Tiny 的 model.safetensors；verify 不需要此文件")
    args = parser.parse_args()
    if args.command == "run": run(args.pretrained)
    elif args.command == "verify":
        assert verify()
        print("verified 30 two-seed predictions")
    else:
        assert json.loads((OUT / "selection.json").read_text())["accepted"]
        p2.seed_everything(SEED)
        data = p2.load_data()
        stats = json.loads((p2.OUT / "normalization.json").read_text())
        p2.transform(data["valid"], stats)
        old = opt.load_model(OLD_CHECKPOINT)
        new = opt.load_model(NEW_CHECKPOINT)
        ordinary = p2.load_model("full")
        rows = scenario_rows(old, new, ordinary, data["valid"], p2.scenario_list())
        p2.write_csv(OUT / "validation_scenarios.csv", rows)
        errors = analyze_errors(old, new, ordinary, data["valid"])
        summarize(json.loads((OUT / "selection.json").read_text()), rows, errors)
        assert verify(old, new, ordinary, stats)
        print("refreshed metrics and error analysis")
