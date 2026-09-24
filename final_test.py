#!/usr/bin/env python3
"""已锁定模型的一次独立 test 评价；输出汇总指标，不进行模型选择。"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import problem2 as p2
import optimize as opt
import modeling_experiment as final_model


HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / "final_test_plan.json").read_text(encoding="utf-8"))
OUT = p2.OUT / "final_test"


def verify_lock():
    assert PLAN["selection_on_test"] is False
    assert json.loads((p2.OUT / "modeling_experiment" / "selection.json").read_text())["accepted"]
    for relative, expected in PLAN["checkpoint_sha256"].items():
        actual = hashlib.sha256((HERE / relative).read_bytes()).hexdigest()
        assert actual == expected, f"锁定文件变更：{relative}"
    available = {scenario["name"]: scenario for scenario in p2.scenario_list()}
    assert len(set(PLAN["scenario_names"])) == len(PLAN["scenario_names"])
    assert set(PLAN["scenario_names"]) <= set(available)
    return available


def load_test(stats):
    source = p2.read_pickle(p2.SOURCE)
    split = source["test"]
    ids = split["id"]
    x = p2.raw_inputs(split, ids)
    classes = np.asarray(split["classification_labels"]).astype(np.int64).reshape(-1)
    strength = np.asarray(split["regression_labels"]).astype(np.float32).reshape(-1)
    assert len(x["id"]) == len(classes) == len(strength)
    assert np.isin(classes, [0, 1, 2]).all() and np.isfinite(strength).all()
    assert strength.min() >= -3 and strength.max() <= 3
    assert np.array_equal(classes, np.where(strength < 0, 0, np.where(strength > 0, 2, 1)))
    x["class"] = classes
    x["strength"] = strength
    p2.transform(x, stats)
    return x


def run():
    available = verify_lock()
    OUT.mkdir(parents=True, exist_ok=True)
    metrics_path = OUT / "test_metrics.csv"
    assert not metrics_path.exists(), "独立 test 已完成；禁止在本实验内重复评价"
    p2.seed_everything(p2.CONFIG["seed"])
    stats = json.loads((p2.OUT / "normalization.json").read_text())
    old = opt.load_model(final_model.OLD_CHECKPOINT)
    new = opt.load_model(final_model.NEW_CHECKPOINT)
    ordinary = p2.load_model("full")
    test = load_test(stats)
    rows = []
    confusion = None
    for name in PLAN["scenario_names"]:
        scenario = available[name]
        _, (probs, reg) = final_model.outputs(old, new, ordinary, test, scenario)
        metrics = p2.metrics(test["class"], test["strength"], probs, reg)
        rows.append({"scenario": name, "sample_count": len(test["id"]),
                     "missing_modality": "+".join(scenario["mods"]) or "none",
                     "position": scenario["place"], "nominal_rate": scenario["rate"], **metrics})
        if name == "complete":
            pred = probs.argmax(axis=1)
            confusion = [[int(((test["class"] == i) & (pred == j)).sum()) for j in range(3)]
                         for i in range(3)]
    full = rows[0]
    for row in rows:
        for key in ("accuracy", "macro_f1", "pearson"):
            row[key + "_relative_drop"] = (full[key] - row[key]) / max(1e-9, abs(full[key]))
        row["mae_relative_increase"] = (row["mae"] - full["mae"]) / max(1e-9, full["mae"])
    p2.write_csv(metrics_path, rows)
    summary = {"locked_commit": PLAN["locked_commit"], "plan": "final_test_plan.json",
               "sample_count": len(test["id"]), "label_counts": np.bincount(test["class"], minlength=3).tolist(),
               "confusion_matrix_complete": confusion, "scenario_count": len(rows),
               "test_used_for_selection": False, "attachment3_recomputed": False}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "metrics": rows}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run()
