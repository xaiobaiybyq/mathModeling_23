#!/usr/bin/env python3
"""用 train 内检查折验证文本连续缺失增强，符合门槛时才全量训练。"""
import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

import problem2 as p2
import optimize as opt
import modeling_experiment as prior
import consistency_experiment as common


OUT = p2.OUT / "focused_mask_experiment"
CHECKPOINT = OUT / "text_focus_tiny.pt"


def compare_all(old_a, old_b, candidate, ordinary, valid):
    rows = []
    for scenario in p2.scenario_list():
        before, after = common.final_outputs(old_a, old_b, candidate, ordinary, valid, scenario)
        for name, (probs, strength) in (("current", before), ("three_seed_text_focus", after)):
            rows.append({"scenario": scenario["name"], "model": name,
                         "missing_modality": "+".join(scenario["mods"]) or "none",
                         "position": scenario["place"], "nominal_rate": scenario["rate"],
                         **p2.metrics(valid["class"], valid["strength"], probs, strength)})
    p2.write_csv(OUT / "valid_scenarios.csv", rows)
    return rows


def save_target(old_a, old_b, candidate, ordinary, stats):
    target = p2.load_targets(stats)
    _, (probs, strength) = common.final_outputs(old_a, old_b, candidate, ordinary,
                                                 target, {"name": "complete"})
    rows = []
    for name, prob, value in zip(target["id"], probs, strength):
        label = int(prob.argmax())
        rows.append({"sample_id": name, "source_file": name + ".pkl",
                     "polarity_class": label, "polarity_name": p2.NAMES[label],
                     "predicted_strength": round(float(value), 6),
                     "prob_negative": round(float(prob[0]), 6),
                     "prob_neutral": round(float(prob[1]), 6),
                     "prob_positive": round(float(prob[2]), 6)})
    p2.write_csv(OUT / "attachment3_predictions_text_focus.csv", rows)
    verify(old_a, old_b, candidate, ordinary, stats)


def verify(old_a=None, old_b=None, candidate=None, ordinary=None, stats=None):
    final = json.loads((OUT / "valid_selection.json").read_text())
    assert final["accepted"]
    if stats is None: stats = json.loads((p2.OUT / "normalization.json").read_text())
    if old_a is None: old_a = opt.load_model(prior.OLD_CHECKPOINT)
    if old_b is None: old_b = opt.load_model(prior.NEW_CHECKPOINT)
    if candidate is None: candidate = opt.load_model(CHECKPOINT)
    if ordinary is None: ordinary = p2.load_model("full")
    target = p2.load_targets(stats)
    _, (probs, strength) = common.final_outputs(old_a, old_b, candidate, ordinary,
                                                 target, {"name": "complete"})
    with (OUT / "attachment3_predictions_text_focus.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 30 and len({row["sample_id"] for row in rows}) == 30
    assert {row["sample_id"] for row in rows} == {p.stem for p in p2.TARGET.glob("附件3_*.pkl")}
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


def write_report(cv, final=None, rows=None):
    report = ["# 第二问继续优化：文本连续缺失增强实验", "",
              "在附件2 train 的固定20%检查折上比较新候选与原训练方式。候选保留25%完整样本；其余样本以50%概率优先遮挡文本连续区间，以50%概率使用原本的一/二模态随机遮挡。缺失长度仍从10%、25%、40%、55%、70%均匀抽取，随机起点；概率依据建模假设预设，未参考附件3的零值比例。特征接口、标准化及原始零值处理保持一致。", "",
              f"train 内原训练分数 {cv['baseline_score']:.4f}，文本增强候选 {cv['candidate_score']:.4f}；按预设规则：{'通过' if cv['accepted'] else '未通过'}。五场景指标见 `results/focused_mask_experiment/cv_panel.csv`。", ""]
    if final is None:
        report += ["未通过 train 内检查，故没有全量训练，也没有用 valid、test 或附件3评价或选择本候选。现行模型保持不变。", ""]
    else:
        def fmt(m):
            return f"Accuracy {m['accuracy']:.4f}、宏 F1 {m['macro_f1']:.4f}、MAE {m['mae']:.4f}、Pearson {m['pearson']:.4f}"
        report += ["全量 train 重训后，现行模型完整 valid 为 " + fmt(final["complete_current"]) + "；三模型候选为 " + fmt(final["complete_candidate"]) + "。",
                   f"固定五场景的选择分数为 {final['old_score']:.4f} 与 {final['candidate_score']:.4f}；{'采用' if final['accepted'] else '不采用'}。", ""]
        if final["accepted"]:
            before = {r["scenario"]: r for r in rows if r["model"] == "current"}
            after = {r["scenario"]: r for r in rows if r["model"] == "three_seed_text_focus"}
            report += ["|场景|现行 Accuracy / 宏 F1|候选 Accuracy / 宏 F1|", "|---|---:|---:|"]
            for name in ("complete", "text_middle_70", "audio_vision_middle_40", "text_only"):
                a, b = before[name], after[name]
                report.append(f"|{name}|{a['accuracy']:.4f} / {a['macro_f1']:.4f}|{b['accuracy']:.4f} / {b['macro_f1']:.4f}|")
            report += ["", "50场景指标见 `results/focused_mask_experiment/valid_scenarios.csv`；附件3的30条预测见 `attachment3_predictions_text_focus.csv`。", ""]
        else:
            report += ["候选未达到 valid 上预设的完整输入和缺失场景门槛，现行预测不变。", ""]
        report += ["附件2 test 未读取。valid 已多次用于已有模型选择，因此即使本次提升也可能有选择偏倚；附件3无真实标签。", ""]
    (p2.HERE / "文本缺失增强实验报告.md").write_text("\n".join(report), encoding="utf-8")


def run(pretrained):
    OUT.mkdir(parents=True, exist_ok=True)
    p2.seed_everything(common.SEED)
    raw = p2.load_data()
    train_indices, check_indices = common.split_train(raw["train"])
    inner_train = common.take(raw["train"], train_indices)
    check = common.take(raw["train"], check_indices)
    stats = p2.fit_normalization(inner_train)
    p2.transform(inner_train, stats)
    p2.transform(check, stats)
    with (common.OUT / "cv_panel.csv").open(encoding="utf-8-sig", newline="") as f:
        base = [{k: (float(v) if k not in ("model", "scenario") else v) for k, v in row.items()}
                for row in csv.DictReader(f) if row["model"] == "baseline"]
    assert len(base) == len(p2.selection_panel())
    history = []
    with tempfile.TemporaryDirectory(prefix="problem2_text_focus_") as temp:
        temp_checkpoint = Path(temp) / "text_focus.pt"
        info = common.train_one("text_focus", inner_train, check, pretrained, temp_checkpoint, history)
        candidate = opt.load_model(temp_checkpoint)
        _, values = common.panel(candidate, check)
    p2.write_csv(OUT / "cv_training_history.csv", history)
    (OUT / "cv_training_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    p2.write_csv(OUT / "cv_panel.csv", [{"model": name, **row} for name, panel in
                  (("baseline", base), ("text_focus", values)) for row in panel])
    cv = common.cv_accept(base, values)
    (OUT / "cv_selection.json").write_text(json.dumps(cv, ensure_ascii=False, indent=2), encoding="utf-8")
    if not cv["accepted"]:
        write_report(cv)
        print(json.dumps({"cv": cv, "final": "not trained"}, ensure_ascii=False), flush=True)
        return
    full_stats = json.loads((p2.OUT / "normalization.json").read_text())
    for split in raw.values(): p2.transform(split, full_stats)
    full_history = []
    full_info = common.train_one("text_focus", raw["train"], raw["valid"], pretrained, CHECKPOINT, full_history)
    p2.write_csv(OUT / "full_training_history.csv", full_history)
    (OUT / "full_training_info.json").write_text(json.dumps(full_info, indent=2), encoding="utf-8")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    checkpoint["state"] = {k: (v.half() if v.is_floating_point() else v) for k, v in checkpoint["state"].items()}
    checkpoint["storage_precision"] = "float16; restored to float32 on load"
    torch.save(checkpoint, CHECKPOINT)
    old_a = opt.load_model(prior.OLD_CHECKPOINT)
    old_b = opt.load_model(prior.NEW_CHECKPOINT)
    candidate = opt.load_model(CHECKPOINT)
    ordinary = p2.load_model("full")
    common.OUT = OUT
    final = common.final_select(old_a, old_b, candidate, ordinary, raw["valid"], "three_seed_text_focus")
    (OUT / "valid_selection.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    if final["accepted"]:
        rows = compare_all(old_a, old_b, candidate, ordinary, raw["valid"])
        save_target(old_a, old_b, candidate, ordinary, full_stats)
    else:
        rows = None
        CHECKPOINT.unlink()
    write_report(cv, final, rows)
    print(json.dumps({"cv": cv, "final": final}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--pretrained", default="/tmp/bert-tiny.safetensors")
    args = parser.parse_args()
    if args.command == "run": run(args.pretrained)
    else:
        assert verify()
        print("verified 30 focused-mask predictions")
