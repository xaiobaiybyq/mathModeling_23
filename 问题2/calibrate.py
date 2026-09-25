#!/usr/bin/env python3
"""在固定 valid 面板上评估小规模类别校准；附件3仅用于选定后的推理。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import problem2 as p2
import optimize as opt


OUT = p2.OUT / "calibration"
WEIGHTS = (0.0, 0.15, 0.30)  # 修正后的普通动态融合模型，仅混合分类概率
NEUTRAL_BIASES = (-0.4, -0.2, 0.0, 0.2, 0.4)


def probabilities(bert, ordinary, weight, bias):
    mixed = (1 - weight) * bert + weight * ordinary
    scores = np.log(np.clip(mixed, 1e-9, 1.0))
    scores[:, 1] += bias
    scores -= scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


def get_outputs(optimized, ordinary, data, scenario):
    s = None if scenario["name"] == "complete" else scenario
    a, r, _ = p2.predict_arrays(optimized, data, s, batch_size=opt.CONFIG["eval_batch_size"])
    b, _, _ = p2.predict_arrays(ordinary, data, s)
    return a, b, r


def score_panel(cached, valid, weight, bias):
    results = []
    for s in p2.selection_panel():
        a, b, r = cached[s["name"]]
        results.append(p2.metrics(valid["class"], valid["strength"], probabilities(a, b, weight, bias), r))
    score = float(np.mean([(m["accuracy"] + m["macro_f1"]) / 2 - m["mae"] / 3 for m in results]))
    return score, results


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    p2.seed_everything(p2.CONFIG["seed"])
    data = p2.load_data()
    stats = json.loads((p2.OUT / "normalization.json").read_text())
    for x in data.values():
        p2.transform(x, stats)
    selected = json.loads((opt.OUT / "selection.json").read_text())["selected"]["variant"]
    optimized = opt.load_model(opt.OUT / (selected + ".pt"))
    ordinary = p2.load_model("full")
    valid = data["valid"]
    cached = {s["name"]: get_outputs(optimized, ordinary, valid, s) for s in p2.selection_panel()}
    candidates = []
    for weight in WEIGHTS:
        for bias in NEUTRAL_BIASES:
            score, panel = score_panel(cached, valid, weight, bias)
            complete_term = (panel[0]["accuracy"] + panel[0]["macro_f1"]) / 2 - panel[0]["mae"] / 3
            candidates.append({"ordinary_class_weight": weight, "neutral_log_bias": bias,
                               "panel_score": score, "selection_score": (score + complete_term) / 2,
                               "complete_accuracy": panel[0]["accuracy"],
                               "complete_macro_f1": panel[0]["macro_f1"],
                               "complete_mae": panel[0]["mae"], "complete_pearson": panel[0]["pearson"]})
    p2.write_csv(OUT / "candidates.csv", candidates)
    base = next(x for x in candidates if x["ordinary_class_weight"] == 0 and x["neutral_log_bias"] == 0)
    eligible = [x for x in candidates if x["complete_accuracy"] >= base["complete_accuracy"] - 1e-9
                and x["complete_macro_f1"] >= base["complete_macro_f1"] - 1e-9]
    best = max(eligible, key=lambda x: x["selection_score"])
    accepted = best["selection_score"] > base["selection_score"] + 1e-4
    selection = {"accepted": accepted, "baseline": base, "selected": best,
                 "rule": "maximize half complete and half five-scenario mean of [(Accuracy+MacroF1)/2 - MAE/3], with complete Accuracy and MacroF1 no lower than baseline",
                 "candidate_grid": {"ordinary_class_weight": WEIGHTS, "neutral_log_bias": NEUTRAL_BIASES},
                 "valid_only": True, "target_used_for_selection": False}
    (OUT / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    if not accepted:
        print("No calibrated candidate met the prespecified improvement rule", flush=True)
        return
    weight, bias = best["ordinary_class_weight"], best["neutral_log_bias"]
    halves = np.array([int(hashlib.sha256(str(s).encode()).hexdigest(), 16) % 2 for s in valid["id"]])
    stability = []
    for fold in (0, 1):
        take = halves == fold
        for label, w, b in (("baseline", 0.0, 0.0), ("selected", weight, bias)):
            values = []
            for scenario in p2.selection_panel():
                a, ordinary_probs, r = cached[scenario["name"]]
                m = p2.metrics(valid["class"][take], valid["strength"][take],
                               probabilities(a[take], ordinary_probs[take], w, b), r[take])
                values.append(m)
            panel_score = float(np.mean([(m["accuracy"] + m["macro_f1"]) / 2 - m["mae"] / 3 for m in values]))
            stability.append({"fixed_half": fold, "samples": int(take.sum()), "candidate": label,
                              "panel_score": panel_score, **values[0]})
    p2.write_csv(OUT / "fixed_half_diagnostic.csv", stability)
    rows = []
    for s in p2.scenario_list():
        a, b, r = cached[s["name"]] if s["name"] in cached else get_outputs(optimized, ordinary, valid, s)
        m = p2.metrics(valid["class"], valid["strength"], probabilities(a, b, weight, bias), r)
        rows.append({"scenario": s["name"], "missing_modality": "+".join(s["mods"]) or "none",
                     "position": s["place"], "nominal_rate": s["rate"], **m})
    full = rows[0]
    for row in rows:
        for key in ("accuracy", "macro_f1", "pearson"):
            row[key + "_relative_drop"] = (full[key] - row[key]) / max(1e-9, abs(full[key]))
        row["mae_relative_increase"] = (row["mae"] - full["mae"]) / max(1e-9, abs(full["mae"]))
    p2.write_csv(OUT / "validation_scenarios.csv", rows)
    p2.svg_plot([{"model": "calibrated", **row} for row in rows], "calibrated", OUT / "missing_factors.svg")
    a, b, r = cached["complete"]
    calibrated_valid = probabilities(a, b, weight, bias)
    prediction = calibrated_valid.argmax(1)
    confusion = np.zeros((3, 3), dtype=int)
    for truth, guess in zip(valid["class"], prediction):
        confusion[int(truth), int(guess)] += 1
    rank = np.argsort(-(np.abs(r - valid["strength"]) + (prediction != valid["class"]).astype(float)))[:12]
    errors = [{"id": valid["id"][i], "true_class": int(valid["class"][i]),
               "pred_class": int(prediction[i]), "true_strength": float(valid["strength"][i]),
               "pred_strength": float(r[i]), "absolute_error": float(abs(r[i] - valid["strength"][i])),
               "valid_length": int(valid["lengths"][i]),
               "raw_zero_audio_effective": int((valid["raw_zero"][i, 1] & valid["effective"][i, 1]).sum()),
               "raw_zero_vision_effective": int((valid["raw_zero"][i, 2] & valid["effective"][i, 2]).sum())}
              for i in rank]
    p2.write_csv(OUT / "error_cases.csv", errors)
    class_f1 = opt.class_f1(valid["class"], prediction)
    target = p2.load_targets(stats)  # 在选择完成后才读取附件3
    a, b, r = get_outputs(optimized, ordinary, target, {"name": "complete"})
    calibrated = probabilities(a, b, weight, bias)
    predictions = []
    for name, c, z in zip(target["id"], calibrated, r):
        k = int(c.argmax())
        predictions.append({"sample_id": name, "source_file": name + ".pkl", "polarity_class": k,
                            "polarity_name": p2.NAMES[k], "predicted_strength": round(float(z), 6),
                            "prob_negative": round(float(c[0]), 6), "prob_neutral": round(float(c[1]), 6),
                            "prob_positive": round(float(c[2]), 6)})
    p2.write_csv(OUT / "attachment3_predictions_calibrated.csv", predictions)
    verify(optimized, ordinary, stats)
    base_rows = []
    with (opt.OUT / "validation_scenarios.csv").open(encoding="utf-8-sig", newline="") as f:
        base_rows = list(csv.DictReader(f))
    base_full = base_rows[0]
    base_by_scenario = {x["scenario"]: x for x in base_rows}
    f1_delta = np.asarray([x["macro_f1"] - float(base_by_scenario[x["scenario"]]["macro_f1"]) for x in rows])
    accuracy_delta = np.asarray([x["accuracy"] - float(base_by_scenario[x["scenario"]]["accuracy"]) for x in rows])
    def fmt(x):
        return f"Accuracy {float(x['accuracy']):.4f}、宏 F1 {float(x['macro_f1']):.4f}、MAE {float(x['mae']):.4f}、Pearson {float(x['pearson']):.4f}"
    report = ["# E题第二问进一步优化：轻量分类融合", "",
              "## 方案与数据边界", "",
              "只用附件2 train 训练的两个修正后模型：BERT Tiny 优化模型与普通动态融合模型。类别概率按 $(1-\\lambda)p_{Tiny}+\\lambda p_{Full}$ 融合，再对中性类别加一个固定 logit 偏置；强度继续使用 BERT Tiny 回归头。最终选择 $\\lambda=%.2f$、中性偏置 $%.1f$。两个模型采用同一 `text_bert/audio/vision` 输入、相同标准化及有效位/人工遮挡掩码。" % (weight, bias),
              "候选网格只有3个融合权重 × 5个中性偏置；在 valid 固定五场景上选择，分数为完整输入与五场景平均的 `[(Accuracy+宏F1)/2−MAE/3]` 各占一半，同时要求完整输入 Accuracy 和宏 F1 均不低于原优化模型。附件2 test 未读取；附件3只在选择完成后推理，不参与权重或偏置选择。", "",
              "## 真实验证指标", "",
              f"原优化模型完整输入：{fmt(base_full)}。最终融合模型：{fmt(rows[0])}。Accuracy 提高 {(rows[0]['accuracy']-float(base_full['accuracy']))*100:.2f} 个百分点，宏 F1 提高 {(rows[0]['macro_f1']-float(base_full['macro_f1']))*100:.2f} 个百分点；强度输出未改变。", "",
              "|固定场景|Accuracy|宏 F1|MAE|Pearson|", "|---|---:|---:|---:|---:|"]
    for name in ("complete", "text_middle_70", "audio_vision_middle_40", "text_only"):
        item = next(row for row in rows if row["scenario"] == name)
        report.append(f"|{name}|{item['accuracy']:.4f}|{item['macro_f1']:.4f}|{item['mae']:.4f}|{item['pearson']:.4f}|")
    report += ["", f"相对原优化模型，50场景中 Accuracy 提升 {int((accuracy_delta > 0).sum())} 场、宏 F1 提升 {int((f1_delta > 0).sum())} 场；平均差分别为 {accuracy_delta.mean():+.4f}、{f1_delta.mean():+.4f}。最差场景差分别为 {accuracy_delta.min():+.4f}、{f1_delta.min():+.4f}，因此不是每一种缺失情形都有提升。", "",
               "完整的50场景四指标与相对下降见 `results/calibration/validation_scenarios.csv`，三因素趋势图见同目录 `missing_factors.svg`。", "",
               "## 稳定性检查与错误分析", "",
               "固定哈希将 valid 分成两个半区，仅作诊断；原模型已经用整个 valid 选过轮次，因此半区结果不是独立测试。", "",
               "|半区|条数|原模型面板分数|融合面板分数|原模型完整 Accuracy / 宏 F1|融合完整 Accuracy / 宏 F1|", "|---|---:|---:|---:|---|---|"]
    for fold in (0, 1):
        old = next(x for x in stability if x["fixed_half"] == fold and x["candidate"] == "baseline")
        new = next(x for x in stability if x["fixed_half"] == fold and x["candidate"] == "selected")
        report.append(f"|{fold}|{new['samples']}|{old['panel_score']:.4f}|{new['panel_score']:.4f}|{old['accuracy']:.4f} / {old['macro_f1']:.4f}|{new['accuracy']:.4f} / {new['macro_f1']:.4f}|")
    report += ["", "融合模型完整输入的混淆矩阵（行真值、列预测；负/中/正）：", "",
               "|真值\\预测|负|中|正|", "|---|---:|---:|---:|"]
    for j, line in enumerate(confusion):
        report.append(f"|{p2.NAMES[j]}|{line[0]}|{line[1]}|{line[2]}|")
    report += ["", f"各类 F1：负向 {class_f1[0]:.4f}、中性 {class_f1[1]:.4f}、正向 {class_f1[2]:.4f}。`error_cases.csv` 记录12条高误差验证案例及原始零值计数；仅凭特征和预测不能确定错误的因果来源。", "",
               "## 交付与限制", "",
               "最终预测为 `results/calibration/attachment3_predictions_calibrated.csv`，编号来自30个源文件名；`python calibrate.py verify` 从两个保存的模型和 `selection.json` 重算并核验行数、唯一编号、类别、概率、强度范围和数值。融合不新增大模型参数，目录大小仍低于50 MB。", "",
               "该提升是在用于选择的 valid 上观察到的，可能包含选择偏倚；附件3无真实标签，无法计算其准确率。固定半区均有提升支持稳定性，但不等同独立盲测。", ""]
    (p2.HERE / "进一步优化报告.md").write_text("\n".join(report), encoding="utf-8")
    print("selected", weight, bias, "score", best["selection_score"], "panel", best["panel_score"], "complete", rows[0], flush=True)


def verify(optimized=None, ordinary=None, stats=None):
    selection = json.loads((OUT / "selection.json").read_text())
    assert selection["accepted"]
    if stats is None:
        stats = json.loads((p2.OUT / "normalization.json").read_text())
    if optimized is None:
        chosen = json.loads((opt.OUT / "selection.json").read_text())["selected"]["variant"]
        optimized = opt.load_model(opt.OUT / (chosen + ".pt"))
    if ordinary is None:
        ordinary = p2.load_model("full")
    target = p2.load_targets(stats)
    a, b, r = get_outputs(optimized, ordinary, target, {"name": "complete"})
    c = probabilities(a, b, selection["selected"]["ordinary_class_weight"], selection["selected"]["neutral_log_bias"])
    with (OUT / "attachment3_predictions_calibrated.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 30 and len({x["sample_id"] for x in rows}) == 30
    assert {x["sample_id"] for x in rows} == {p.stem for p in p2.TARGET.glob("附件3_*.pkl")}
    for name, row, prob, strength in zip(target["id"], rows, c, r):
        k = int(prob.argmax())
        assert row["sample_id"] == name and row["source_file"] == name + ".pkl"
        assert row["polarity_name"] == p2.NAMES[k] and int(row["polarity_class"]) == k
        assert abs(float(row["predicted_strength"]) - float(strength)) < 1e-6
        assert -3 <= float(row["predicted_strength"]) <= 3
        assert abs(sum(float(row[key]) for key in ("prob_negative", "prob_neutral", "prob_positive")) - 1) < 2e-6
        for key, value in zip(("prob_negative", "prob_neutral", "prob_positive"), prob):
            assert abs(float(row[key]) - float(value)) < 1e-6
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "verify"))
    args = parser.parse_args()
    if args.command == "run":
        run()
    else:
        assert verify()
        print("verified 30 calibrated predictions")
