#!/usr/bin/env python3
"""由锁定的附件3预测 CSV 更新论文问题二正文中的全量预测表。"""
import csv
import hashlib
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "results" / "modeling_experiment" / "attachment3_predictions_two_seed.csv"
PAPER = HERE / "论文问题二正文.md"
START = "<!-- ATTACHMENT3_PREDICTIONS_START -->"
END = "<!-- ATTACHMENT3_PREDICTIONS_END -->"
LABELS = {0: "负向", 1: "中性", 2: "正向"}


def main():
    plan = json.loads((HERE / "final_test_plan.json").read_text(encoding="utf-8"))
    expected = plan["checkpoint_sha256"]["results/modeling_experiment/attachment3_predictions_two_seed.csv"]
    assert hashlib.sha256(CSV_PATH.read_bytes()).hexdigest() == expected, "预测 CSV 与锁定版本不一致"
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 30 and len({row["sample_id"] for row in rows}) == 30
    table = ["|样本编号（源文件名，不含扩展名）|极性类别|预测情感强度|",
             "|---|---|---:|"]
    for row in rows:
        name = row["sample_id"]
        label = int(row["polarity_class"])
        value = float(row["predicted_strength"])
        assert row["source_file"] == name + ".pkl"
        assert label in LABELS and -3 <= value <= 3
        assert row["polarity_name"] == ("Negative", "Neutral", "Positive")[label]
        table.append(f"|{name.replace('|', '&#124;')}|{label}（{LABELS[label]}）|{value:.4f}|")
    source = PAPER.read_text(encoding="utf-8")
    pattern = re.escape(START) + r".*?" + re.escape(END)
    replacement = START + "\n\n" + "\n".join(table) + "\n\n" + END
    updated, count = re.subn(pattern, lambda _: replacement, source, count=1, flags=re.DOTALL)
    assert count == 1
    PAPER.write_text(updated, encoding="utf-8")
    print("updated 30 prediction rows in paper section")


if __name__ == "__main__":
    main()
