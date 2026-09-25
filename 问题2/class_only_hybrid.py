#!/usr/bin/env python3
"""文本缺失增强仅用于分类；回归保持现行双种子输出。"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import problem2 as p2
import optimize as opt
import modeling_experiment as prior
import consistency_experiment as common
import focused_mask_experiment as focus


OUT = p2.OUT / "class_only_hybrid"
CHECKPOINT = OUT / "text_focus_tiny.pt"
EVAL_NAMES = ("complete", "text_middle_40", "audio_middle_40", "vision_middle_40",
              "audio_vision_middle_40", "text_middle_70", "text_only")


def cv_gate():
    with (focus.OUT / "cv_panel.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    a = {r["scenario"]: r for r in rows if r["model"] == "baseline"}
    b = {r["scenario"]: r for r in rows if r["model"] == "text_focus"}
    def gain(name, key): return float(b[name][key]) - float(a[name][key])
    gains = {f"{name}_{key}": gain(name, key)
             for name in ("complete", "text_middle_40", "audio_vision_middle_40")
             for key in ("accuracy", "macro_f1")}
    accepted = (gain("complete", "accuracy") >= 0.01
                and gain("complete", "macro_f1") >= -0.005
                and gain("text_middle_40", "accuracy") >= 0.01
                and gain("text_middle_40", "macro_f1") >= -0.005
                and gain("audio_vision_middle_40", "macro_f1") >= 0)
    return {"accepted": bool(accepted), "gains": gains,
            "rule": "complete and text-middle-40 Accuracy +0.01, their macroF1 loss <=0.005, AV-middle-40 macroF1 nondecrease"}


def outputs(old_a, old_b, focus_model, ordinary, x, scenario):
    before, three = common.final_outputs(old_a, old_b, focus_model, ordinary, x, scenario)
    return before, (three[0], before[1])


def compare(old_a, old_b, focus_model, ordinary, valid, scenarios):
    rows = []
    for scenario in scenarios:
        before, after = outputs(old_a, old_b, focus_model, ordinary, valid, scenario)
        for name, (probs, reg) in (("current", before), ("class_only_hybrid", after)):
            rows.append({"scenario": scenario["name"], "model": name,
                         "missing_modality": "+".join(scenario["mods"]) or "none",
                         "position": scenario["place"], "nominal_rate": scenario["rate"],
                         **p2.metrics(valid["class"], valid["strength"], probs, reg)})
    return rows


def score(m):
    return (m["accuracy"] + m["macro_f1"]) / 2


def valid_gate(rows):
    a = {r["scenario"]: r for r in rows if r["model"] == "current"}
    b = {r["scenario"]: r for r in rows if r["model"] == "class_only_hybrid"}
    old_score = (score(a["complete"]) + np.mean([score(a[n]) for n in EVAL_NAMES])) / 2
    new_score = (score(b["complete"]) + np.mean([score(b[n]) for n in EVAL_NAMES])) / 2
    accepted = (new_score >= old_score + 0.003
                and b["complete"]["accuracy"] >= a["complete"]["accuracy"] + 0.005
                and b["complete"]["macro_f1"] >= a["complete"]["macro_f1"] - 1e-9
                and b["text_middle_40"]["macro_f1"] >= a["text_middle_40"]["macro_f1"] - 0.005
                and b["audio_vision_middle_40"]["macro_f1"] >= a["audio_vision_middle_40"]["macro_f1"] - 0.005
                and b["text_middle_70"]["macro_f1"] >= a["text_middle_70"]["macro_f1"] - 0.01
                and b["text_only"]["macro_f1"] >= a["text_only"]["macro_f1"] - 0.01)
    return {"accepted": bool(accepted), "old_score": float(old_score),
            "candidate_score": float(new_score), "complete_current": {k: a["complete"][k] for k in ("accuracy", "macro_f1", "mae", "pearson")},
            "complete_candidate": {k: b["complete"][k] for k in ("accuracy", "macro_f1", "mae", "pearson")},
            "rule": "7 fixed conditions: score gain>=0.003, complete Accuracy gain>=0.005, complete F1 no drop, moderate missing F1 drop<=0.005, severe missing F1 drop<=0.01",
            "test_used": False, "target_used_for_selection": False}


def save_target(old_a, old_b, focus_model, ordinary, stats):
    target = p2.load_targets(stats)
    _, (probs, strength) = outputs(old_a, old_b, focus_model, ordinary, target, {"name": "complete"})
    rows = []
    for name, prob, value in zip(target["id"], probs, strength):
        label = int(prob.argmax())
        rows.append({"sample_id": name, "source_file": name + ".pkl", "polarity_class": label,
                     "polarity_name": p2.NAMES[label], "predicted_strength": round(float(value), 6),
                     "prob_negative": round(float(prob[0]), 6), "prob_neutral": round(float(prob[1]), 6),
                     "prob_positive": round(float(prob[2]), 6)})
    p2.write_csv(OUT / "attachment3_predictions_class_only.csv", rows)
    verify(old_a, old_b, focus_model, ordinary, stats)


def verify(old_a=None, old_b=None, focus_model=None, ordinary=None, stats=None):
    assert json.loads((OUT / "valid_selection.json").read_text())["accepted"]
    if stats is None: stats = json.loads((p2.OUT / "normalization.json").read_text())
    if old_a is None: old_a = opt.load_model(prior.OLD_CHECKPOINT)
    if old_b is None: old_b = opt.load_model(prior.NEW_CHECKPOINT)
    if focus_model is None: focus_model = opt.load_model(CHECKPOINT)
    if ordinary is None: ordinary = p2.load_model("full")
    target = p2.load_targets(stats)
    _, (probs, strength) = outputs(old_a, old_b, focus_model, ordinary, target, {"name": "complete"})
    with (OUT / "attachment3_predictions_class_only.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 30 and len({r["sample_id"] for r in rows}) == 30
    assert {r["sample_id"] for r in rows} == {p.stem for p in p2.TARGET.glob("附件3_*.pkl")}
    for name, row, prob, value in zip(target["id"], rows, probs, strength):
        label = int(prob.argmax())
        assert row["sample_id"] == name and row["source_file"] == name + ".pkl"
        assert int(row["polarity_class"]) == label and row["polarity_name"] == p2.NAMES[label]
        assert -3 <= float(row["predicted_strength"]) <= 3
        assert abs(float(row["predicted_strength"]) - float(value)) < 1e-6
        assert abs(sum(float(row[f"prob_{key}"]) for key in ("negative", "neutral", "positive")) - 1) < 2e-6
        for key, p in zip(("negative", "neutral", "positive"), prob):
            assert abs(float(row[f"prob_{key}"]) - float(p)) < 1e-6
    return True


def report(cv, final=None, all_rows=None):
    lines = ["# 第二问继续优化：分类专用文本缺失增强模型", "",
             "## 决策过程", "",
             "先尝试完整/缺失一致性训练及文本缺失增强的联合任务模型，均未通过 train 内联合指标门槛。检查折显示文本增强使分类 Accuracy 上升但 MAE 变差，因此进一步评估一个分类专用组合：三个 BERT Tiny 分类概率等权平均，再按原已定的15%权重混入普通动态模型；回归保持现行双种子输出。这一方案及门槛在访问 valid 评价该组合前固定。", "",
             "train 内分类门槛：完整与文本中段40%缺失的 Accuracy 均提高至少0.01、宏 F1 降幅不超过0.005，音视频同时中段40%缺失的宏 F1 不下降。结果：" + ("通过" if cv["accepted"] else "未通过") + "。详细结果见 `results/focused_mask_experiment/cv_panel.csv`。", ""]
    if final is None:
        lines += ["未进行全量 train 重训或 valid 选择；现行预测不变。", ""]
    else:
        def fmt(m):
            return f"Accuracy {m['accuracy']:.4f}、宏 F1 {m['macro_f1']:.4f}、MAE {m['mae']:.4f}、Pearson {m['pearson']:.4f}"
        lines += ["## 全量 train 后的 valid 检查", "",
                  f"现行模型：{fmt(final['complete_current'])}。分类专用候选：{fmt(final['complete_candidate'])}。",
                  f"固定七场景分类分数 {final['old_score']:.4f} 与 {final['candidate_score']:.4f}。按预设完整和缺失指标门槛：{'采用' if final['accepted'] else '不采用'}。", ""]
        a = {r["scenario"]: r for r in all_rows if r["model"] == "current"}
        b = {r["scenario"]: r for r in all_rows if r["model"] == "class_only_hybrid"}
        lines += ["|固定场景|现行 Accuracy / 宏 F1|候选 Accuracy / 宏 F1|", "|---|---:|---:|"]
        for name in ("complete", "text_middle_70", "audio_vision_middle_40", "text_only"):
            x, y = a[name], b[name]
            lines.append(f"|{name}|{float(x['accuracy']):.4f} / {float(x['macro_f1']):.4f}|{float(y['accuracy']):.4f} / {float(y['macro_f1']):.4f}|")
        if final["accepted"]:
            lines += ["", "50个固定场景的四指标见 `results/class_only_hybrid/valid_scenarios.csv`；30条附件3预测见 `attachment3_predictions_class_only.csv`。", ""]
        else:
            lines += ["", "七场景四指标见 `results/class_only_hybrid/valid_selection_panel.csv`。未达到预设门槛，候选权重已删除，现行预测保持不变。", ""]
        lines += ["附件2 test 未参与训练、选择或评价；附件3只在模型采用后推理。valid 已参与此前多轮选择，当前验证提升可能有选择偏倚。", ""]
    (p2.HERE / "分类专用优化实验报告.md").write_text("\n".join(lines), encoding="utf-8")


def run(pretrained):
    OUT.mkdir(parents=True, exist_ok=True)
    cv = cv_gate()
    (OUT / "cv_gate.json").write_text(json.dumps(cv, ensure_ascii=False, indent=2), encoding="utf-8")
    if not cv["accepted"]:
        report(cv)
        print(json.dumps({"cv": cv, "final": "not trained"}, ensure_ascii=False), flush=True)
        return
    p2.seed_everything(common.SEED)
    data = p2.load_data()
    stats = json.loads((p2.OUT / "normalization.json").read_text())
    for split in data.values(): p2.transform(split, stats)
    history = []
    info = common.train_one("text_focus", data["train"], data["valid"], pretrained, CHECKPOINT, history)
    p2.write_csv(OUT / "training_history.csv", history)
    (OUT / "training_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    checkpoint["state"] = {k: (v.half() if v.is_floating_point() else v) for k, v in checkpoint["state"].items()}
    checkpoint["storage_precision"] = "float16; restored to float32 on load"
    torch.save(checkpoint, CHECKPOINT)
    old_a = opt.load_model(prior.OLD_CHECKPOINT)
    old_b = opt.load_model(prior.NEW_CHECKPOINT)
    new = opt.load_model(CHECKPOINT)
    ordinary = p2.load_model("full")
    scenarios = [s for s in p2.scenario_list() if s["name"] in EVAL_NAMES]
    seven = compare(old_a, old_b, new, ordinary, data["valid"], scenarios)
    p2.write_csv(OUT / "valid_selection_panel.csv", seven)
    final = valid_gate(seven)
    (OUT / "valid_selection.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    if final["accepted"]:
        all_rows = compare(old_a, old_b, new, ordinary, data["valid"], p2.scenario_list())
        p2.write_csv(OUT / "valid_scenarios.csv", all_rows)
        save_target(old_a, old_b, new, ordinary, stats)
    else:
        all_rows = None
        CHECKPOINT.unlink()
    report(cv, final, all_rows if all_rows is not None else seven)
    print(json.dumps({"cv": cv, "final": final}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--pretrained", default="/tmp/bert-tiny.safetensors")
    args = parser.parse_args()
    if args.command == "run": run(args.pretrained)
    else:
        assert verify()
        print("verified 30 classification-only predictions")
