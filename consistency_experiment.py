#!/usr/bin/env python3
"""train 内留出折检验完整/缺失一致性训练，达标后才进行全量训练和 valid 检查。"""
import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import problem2 as p2
import optimize as opt
import modeling_experiment as prior


OUT = p2.OUT / "consistency_experiment"
SEED = 20260925
TEMPERATURE = 2.0
KL_WEIGHT = 0.1
REG_CONSISTENCY_WEIGHT = 0.1
MASKED_TASK_WEIGHT = 0.75
ORDINARY_WEIGHT = 0.15
NEW_CHECKPOINT = OUT / "consistency_tiny.pt"
CV_MIN_GAIN = 0.005
FINAL_MIN_GAIN = 0.003


def take(x, indices):
    return {key: ([value[i] for i in indices] if key == "id" else value[indices].copy())
            for key, value in x.items()}


def split_train(train):
    selected = np.zeros(len(train["id"]), dtype=bool)
    for label in range(3):
        indices = np.flatnonzero(train["class"] == label)
        keyed = sorted(indices, key=lambda i: hashlib.sha256((str(train["id"][i]) + "|cv20260925").encode()).hexdigest())
        selected[keyed[:round(len(indices) * 0.2)]] = True
    return np.flatnonzero(~selected), np.flatnonzero(selected)


def task_loss(logits, reg, labels, strength, class_weight):
    return F.cross_entropy(logits, labels, weight=class_weight) + opt.CONFIG["huber_weight"] * F.huber_loss(reg, strength, delta=1.0)


def text_focus_mask(batch, rng):
    """保留25%完整输入；增强样本的一半优先遮挡文本，其余沿原规则抽取。"""
    for i, length in enumerate(batch["lengths"]):
        if rng.random() >= p2.CONFIG["augment_probability"]:
            continue
        if rng.random() < 0.5:
            mods = [0]
        else:
            count = 2 if rng.random() < p2.CONFIG["two_modality_probability"] else 1
            mods = rng.choice(3, size=count, replace=False)
        rate = float(rng.choice(p2.CONFIG["rates"]))
        start, end = p2.interval(length, rate, "random", rng)
        for j in mods:
            if j == 0: batch["tokens"][i, start:end] = 0
            elif j == 1: batch["audio"][i, start:end] = 0
            else: batch["vision"][i, start:end] = 0
            batch["synthetic_missing"][i, j, start:end] |= batch["effective"][i, j, start:end]
            batch["obs"][i, j, start:end] = batch["effective"][i, j, start:end] & ~batch["synthetic_missing"][i, j, start:end]


def panel(model, x):
    rows = []
    for scenario in p2.selection_panel():
        s = None if scenario["name"] == "complete" else scenario
        probs, strength, _ = p2.predict_arrays(model, x, s, batch_size=opt.CONFIG["eval_batch_size"])
        rows.append({"scenario": scenario["name"], **p2.metrics(x["class"], x["strength"], probs, strength)})
    score = (prior.score(rows[0]) + np.mean([prior.score(r) for r in rows])) / 2
    return float(score), rows


def train_one(mode, train, select, pretrained, checkpoint, history):
    p2.seed_everything(SEED)
    rng = np.random.default_rng(SEED + 101)
    model = opt.OptimizedNet()
    model.initialize_from_pretrained(pretrained)
    groups = [{"params": [p for name, p in model.named_parameters() if not name.startswith("bert.")],
               "lr": opt.CONFIG["head_lr"]},
              {"params": model.bert.parameters(), "lr": opt.CONFIG["bert_lr"]}]
    optimizer = torch.optim.AdamW(groups, weight_decay=opt.CONFIG["weight_decay"])
    counts = np.bincount(train["class"], minlength=3).astype(float)
    weights = 1 / np.sqrt(counts)
    class_weight = torch.tensor(weights / weights.mean(), dtype=torch.float32)
    best, epoch_best, stale = -1e9, 0, 0
    n = len(train["id"])
    for epoch in range(1, opt.CONFIG["epochs"] + 1):
        model.train()
        order = rng.permutation(n)
        losses = []
        for lo in range(0, n, opt.CONFIG["batch_size"]):
            indices = order[lo:lo + opt.CONFIG["batch_size"]]
            full = p2.subset(train, indices)
            masked = p2.subset(train, indices)
            if mode == "text_focus": text_focus_mask(masked, rng)
            else: p2.train_mask(masked, rng)
            labels = torch.from_numpy(train["class"][indices])
            strength = torch.from_numpy(train["strength"][indices])
            logits_m, reg_m, _ = model(p2.to_torch(masked))
            loss_m = task_loss(logits_m, reg_m, labels, strength, class_weight)
            if mode == "consistency":
                logits_f, reg_f, _ = model(p2.to_torch(full))
                loss_f = task_loss(logits_f, reg_f, labels, strength, class_weight)
                kl = F.kl_div(F.log_softmax(logits_m / TEMPERATURE, dim=1),
                              F.softmax(logits_f.detach() / TEMPERATURE, dim=1),
                              reduction="batchmean") * TEMPERATURE ** 2
                reg_match = F.huber_loss(reg_m / 3, reg_f.detach() / 3, delta=1.0)
                loss = (MASKED_TASK_WEIGHT * loss_m + (1 - MASKED_TASK_WEIGHT) * loss_f
                        + KL_WEIGHT * kl + REG_CONSISTENCY_WEIGHT * reg_match)
            else:
                loss = loss_m
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.item()))
        selection_score, values = panel(model, select)
        history.append({"mode": mode, "epoch": epoch, "train_loss": float(np.mean(losses)),
                        "selection_score": selection_score, "complete_accuracy": values[0]["accuracy"],
                        "complete_macro_f1": values[0]["macro_f1"], "complete_mae": values[0]["mae"]})
        print(mode, "epoch", epoch, "loss", round(float(np.mean(losses)), 4),
              "score", round(selection_score, 4), "f1", round(values[0]["macro_f1"], 4), flush=True)
        if selection_score > best + 1e-5:
            best, epoch_best, stale = selection_score, epoch, 0
            torch.save({"state": model.state_dict(), "mode": mode, "seed": SEED,
                        "epoch": epoch, "score": best}, checkpoint)
        else:
            stale += 1
            if stale >= opt.CONFIG["patience"]:
                break
    return {"mode": mode, "best_epoch": epoch_best, "selection_score": best}


def cv_accept(base, candidate):
    a = {r["scenario"]: r for r in base}
    b = {r["scenario"]: r for r in candidate}
    base_score = (prior.score(a["complete"]) + np.mean([prior.score(x) for x in base])) / 2
    new_score = (prior.score(b["complete"]) + np.mean([prior.score(x) for x in candidate])) / 2
    accepted = (new_score >= base_score + CV_MIN_GAIN
                and b["complete"]["accuracy"] >= a["complete"]["accuracy"] - 0.01
                and b["complete"]["macro_f1"] >= a["complete"]["macro_f1"] - 0.01
                and b["text_middle_40"]["macro_f1"] >= a["text_middle_40"]["macro_f1"] - 0.01)
    return {"accepted": bool(accepted), "baseline_score": float(base_score),
            "candidate_score": float(new_score), "min_gain": CV_MIN_GAIN,
            "complete_accuracy_tolerance": 0.01, "complete_macro_f1_tolerance": 0.01,
            "text_middle_40_macro_f1_tolerance": 0.01}


def final_outputs(old_a, old_b, candidate, ordinary, x, scenario):
    s = None if scenario["name"] == "complete" else scenario
    probs, regs = [], []
    for model in (old_a, old_b, candidate):
        p, r, _ = p2.predict_arrays(model, x, s, batch_size=opt.CONFIG["eval_batch_size"])
        probs.append(p)
        regs.append(r)
    regular, _, _ = p2.predict_arrays(ordinary, x, s)
    old = ((1 - ORDINARY_WEIGHT) * (probs[0] + probs[1]) / 2 + ORDINARY_WEIGHT * regular,
           (regs[0] + regs[1]) / 2)
    new = ((1 - ORDINARY_WEIGHT) * (probs[0] + probs[1] + probs[2]) / 3 + ORDINARY_WEIGHT * regular,
           (regs[0] + regs[1] + regs[2]) / 3)
    return old, new


def final_select(old_a, old_b, candidate, ordinary, valid, candidate_name="three_seed_consistency"):
    rows = []
    for scenario in p2.selection_panel():
        before, after = final_outputs(old_a, old_b, candidate, ordinary, valid, scenario)
        for name, output in (("current", before), (candidate_name, after)):
            rows.append({"scenario": scenario["name"], "model": name,
                         **p2.metrics(valid["class"], valid["strength"], *output)})
    p2.write_csv(OUT / "valid_selection_panel.csv", rows)
    grouped = {name: [r for r in rows if r["model"] == name]
               for name in ("current", candidate_name)}
    a, b = grouped["current"], grouped[candidate_name]
    old_score = (prior.score(a[0]) + np.mean([prior.score(x) for x in a])) / 2
    new_score = (prior.score(b[0]) + np.mean([prior.score(x) for x in b])) / 2
    by_a = {r["scenario"]: r for r in a}
    by_b = {r["scenario"]: r for r in b}
    accepted = (new_score >= old_score + FINAL_MIN_GAIN
                and by_b["complete"]["accuracy"] >= by_a["complete"]["accuracy"] - 1e-9
                and by_b["complete"]["macro_f1"] >= by_a["complete"]["macro_f1"] - 1e-9
                and by_b["text_middle_40"]["macro_f1"] >= by_a["text_middle_40"]["macro_f1"] - 1e-9
                and by_b["audio_vision_middle_40"]["macro_f1"] >= by_a["audio_vision_middle_40"]["macro_f1"] - 1e-9)
    return {"accepted": bool(accepted), "old_score": float(old_score), "candidate_score": float(new_score),
            "min_gain": FINAL_MIN_GAIN, "complete_current": by_a["complete"],
            "complete_candidate": by_b["complete"],
            "rule": "score gain >=0.003, no drop in complete Accuracy/F1 or text-middle-40 and AV-middle-40 F1",
            "test_used": False, "target_used_for_selection": False}


def summarize(cv, final=None):
    lines = ["# 第二问继续优化：完整/缺失一致性实验", "",
             "## 方案和数据边界", "",
             "在附件2 train 内按类别及样本编号的固定哈希划出约20%检查折，剩余样本分别训练原损失基线和一致性候选。候选以完整输入和连续局部缺失输入共享网络：遮挡任务损失权重0.75、完整任务损失0.25，另以系数0.1加入分类 KL 蒸馏及归一化强度 Huber 一致性。温度2，种子20260925；模型结构、BERT Tiny 预训练来源、原始零值和显式掩码规则与现有模型一致。附件3未用于训练、采样设置或模型选择。", "",
             "## train 内检查折", "",
             f"原损失分数 {cv['baseline_score']:.4f}；一致性候选 {cv['candidate_score']:.4f}。预先规定候选至少提高 {CV_MIN_GAIN:.3f}，完整 Accuracy/F1 及文本中段40%缺失 F1 的容差各为0.01。结果：{'通过' if cv['accepted'] else '未通过'}。详细五场景指标见 `results/consistency_experiment/cv_panel.csv`。", ""]
    if final is None:
        lines += ["一致性候选未通过 train 内检查，因此未用附件2 valid 评价或选择本候选；test 与附件3也未参与本实验。当前推荐模型及预测保持不变。", ""]
    else:
        def fmt(m):
            return f"Accuracy {m['accuracy']:.4f}、宏 F1 {m['macro_f1']:.4f}、MAE {m['mae']:.4f}、Pearson {m['pearson']:.4f}"
        lines += ["## 全量 train 重训后的 valid 检查", "",
                  f"现行模型：{fmt(final['complete_current'])}。三模型候选：{fmt(final['complete_candidate'])}。现行分数 {final['old_score']:.4f}，候选 {final['candidate_score']:.4f}。按预设规则：{'采用' if final['accepted'] else '不采用'}。五场景指标见 `results/consistency_experiment/valid_selection_panel.csv`。", "",
                  "附件2 test 未读取。valid 已用于先前多轮选择，因此此处的提升可能包含选择偏倚；附件3无真实标签，不能计算其准确率。", ""]
    (p2.HERE / "一致性训练实验报告.md").write_text("\n".join(lines), encoding="utf-8")


def run(pretrained):
    OUT.mkdir(parents=True, exist_ok=True)
    p2.seed_everything(SEED)
    raw = p2.load_data()
    train_indices, check_indices = split_train(raw["train"])
    inner_train = take(raw["train"], train_indices)
    check = take(raw["train"], check_indices)
    stats = p2.fit_normalization(inner_train)
    p2.transform(inner_train, stats)
    p2.transform(check, stats)
    (OUT / "cv_split.json").write_text(json.dumps({"seed": SEED, "stratified_by": "class",
        "split_rule": "within each class sha256(id+'|cv20260925'), first round(20%) held out",
        "train_count": len(train_indices), "check_count": len(check_indices),
        "train_class_counts": np.bincount(inner_train["class"], minlength=3).tolist(),
        "check_class_counts": np.bincount(check["class"], minlength=3).tolist()}, indent=2), encoding="utf-8")
    history = []
    with tempfile.TemporaryDirectory(prefix="problem2_consistency_") as temp:
        base_path, candidate_path = Path(temp) / "base.pt", Path(temp) / "candidate.pt"
        base_info = train_one("baseline", inner_train, check, pretrained, base_path, history)
        candidate_info = train_one("consistency", inner_train, check, pretrained, candidate_path, history)
        p2.write_csv(OUT / "cv_training_history.csv", history)
        (OUT / "cv_training_info.json").write_text(json.dumps([base_info, candidate_info], indent=2), encoding="utf-8")
        base_model = opt.load_model(base_path)
        candidate_model = opt.load_model(candidate_path)
        _, base_panel = panel(base_model, check)
        _, candidate_panel = panel(candidate_model, check)
    p2.write_csv(OUT / "cv_panel.csv", [{"model": name, **row} for name, values in
                  (("baseline", base_panel), ("consistency", candidate_panel)) for row in values])
    cv = cv_accept(base_panel, candidate_panel)
    (OUT / "cv_selection.json").write_text(json.dumps(cv, ensure_ascii=False, indent=2), encoding="utf-8")
    if not cv["accepted"]:
        summarize(cv)
        print(json.dumps({"cv": cv, "final": "not trained"}, ensure_ascii=False), flush=True)
        return
    full_stats = json.loads((p2.OUT / "normalization.json").read_text())
    for split in raw.values(): p2.transform(split, full_stats)
    full_history = []
    full_info = train_one("consistency", raw["train"], raw["valid"], pretrained, NEW_CHECKPOINT, full_history)
    p2.write_csv(OUT / "full_training_history.csv", full_history)
    (OUT / "full_training_info.json").write_text(json.dumps(full_info, indent=2), encoding="utf-8")
    checkpoint = torch.load(NEW_CHECKPOINT, map_location="cpu", weights_only=False)
    checkpoint["state"] = {k: (v.half() if v.is_floating_point() else v) for k, v in checkpoint["state"].items()}
    checkpoint["storage_precision"] = "float16; restored to float32 on load"
    torch.save(checkpoint, NEW_CHECKPOINT)
    old_a = opt.load_model(prior.OLD_CHECKPOINT)
    old_b = opt.load_model(prior.NEW_CHECKPOINT)
    candidate = opt.load_model(NEW_CHECKPOINT)
    ordinary = p2.load_model("full")
    final = final_select(old_a, old_b, candidate, ordinary, raw["valid"])
    (OUT / "valid_selection.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    if not final["accepted"]:
        NEW_CHECKPOINT.unlink()
    summarize(cv, final)
    print(json.dumps({"cv": cv, "final": final}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--pretrained", default="/tmp/bert-tiny.safetensors")
    args = parser.parse_args()
    run(args.pretrained)
