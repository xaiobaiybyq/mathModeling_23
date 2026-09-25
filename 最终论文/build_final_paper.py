#!/usr/bin/env python3
"""由三问已运行报告、全量CSV和网络核实文献生成整合论文Markdown。"""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
E = HERE.parent
Q1 = E / "问题1" if (E / "问题1").is_dir() else E / "question1"

Q2_CITATIONS = {1: 1, 2: 9, 3: 2, 4: 10, 5: 11, 6: 12, 7: 13, 8: 14}
Q3_CITATIONS = {9: 15, 10: 4, 11: 3}


def references_to_global(body: str, mapping: dict[int, int]) -> str:
    def replace(match: re.Match[str]) -> str:
        old = int(match.group(1))
        return f"[{mapping[old]}]" if old in mapping else match.group(0)

    return re.sub(r"\[(\d+)\]", replace, body)


def relocate_links(body: str, source_dir: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        address = match.group(1)
        if address.startswith(("http://", "https://", "#")):
            return match.group(0)
        target = (source_dir / address).resolve()
        return "](" + os.path.relpath(target, HERE).replace(os.sep, "/") + ")"

    return re.sub(r"\]\(([^)]+)\)", replace, body)


def load_chapter(path: Path, mapping: dict[int, int]) -> str:
    body = path.read_text(encoding="utf-8").split("## 参考文献", 1)[0].strip()
    body = body.replace("<!-- ATTACHMENT3_PREDICTIONS_START -->", "")
    body = body.replace("<!-- ATTACHMENT3_PREDICTIONS_END -->", "")
    body = references_to_global(body, mapping)
    body = relocate_links(body, path.parent)
    return body


def table_100() -> str:
    with (Q1 / "results/sample_summary.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len({row["id"] for row in rows}) == 100
    assert sum(int(row["word_count"]) for row in rows) == 1931
    assert sum(int(row["audio_valid_words"]) for row in rows) == 1931
    assert sum(int(row["vision_valid_words"]) for row in rows) == 1601
    assert all(
        (row["text_dim"], row["audio_dim"], row["vision_dim"]) == ("768", "50", "110")
        for row in rows
    )
    lines = [
        "# 附录A 附件1全部100条样本的特征结果",
        "",
        "附表A-1保留源样本编号。每条均含文本T、语音A、视觉V，词级对齐粒度；维度分别为768/50/110。时长按原视频解码时间轴，A/V为有效贡献词数。完整逐词时间、源帧索引和特征矩阵在问题一结果目录及技术附件中。",
        "",
        "|编号|解码时长(s)|词数|模态及维度|对齐粒度|A有效词|V有效词|",
        "|---|---:|---:|---|---|---:|---:|",
    ]
    for row in rows:
        sample_id = row["id"].replace("|", r"\|")
        lines.append(
            f"|{sample_id}|{float(row['duration_s']):.4f}|{row['word_count']}"
            f"|T768/A50/V110|词级|{row['audio_valid_words']}|{row['vision_valid_words']}|"
        )
    lines.extend(
        [
            "",
            "合计100条、1931个词位置；A有效1931词、V有效1601词。文件逐条对应情况见[问题一汇总CSV](../question1/results/sample_summary.csv)。",
        ]
    )
    return "\n".join(lines)


CONCLUSION = r"""# 6 综合评价与结论

## 6.1 三问之间的关系

问题一解决从原视频到可追溯词级表示的生成问题，输出完整变长序列、显式质量掩码和源帧对应；问题二以附件2已对齐特征为独立输入接口，解决连续局部缺失时的分类与回归预测；问题三冻结问题二模型，通过连续遮挡响应定位影响预测的模态和片段，再用原视频生成自动时间与关键帧。三问方法可衔接，但问题一的自提取特征与附件2的数值维度不同，不能把两套特征视为同一模型输入。

## 6.2 结果与适用范围

问题一完成100/100条视频的特征生成与结构检查，视觉有效词占82.91%；问题二最终模型在728条valid完整输入上达到Accuracy 0.6044、宏F1 0.5763、MAE 0.6511、Pearson 0.5461，在锁定后一次独立test完整输入上分别为0.6039、0.5511、0.7295、0.5495。连续缺失验证显示文本缺失，尤其较长的中段缺失，平均影响更大，但这是指定遮挡规则和该模型下的统计表现。问题三对附件4的20条无标签样本完成预测与解释，主要参考模态均为文本；此结论是模型敏感度，不是人类情感判断的因果规律。

## 6.3 局限

问题一缺少人工逐词时间真值，原始转写与音频不一致会使CTC强制对齐错误；问题三的原视频时间关联也因此需要人工回听，07、15、18号质量警示样本应优先复核。问题二的valid经历多轮模型选择，最终结果存在选择偏倚；中性类仍易被判为正向。分类与强度两头独立，专项预测中存在符号不一致输出，本文如实保留。附件3和附件4均无真实情感标签，不能对它们报告预测准确率，也不能用其零值比例反推缺失成因或调整模型。

## 6.4 可复现材料

训练代码、固定配置、所选检查点、100条自主特征、30条附件3预测和20条附件4预测与解释均由各问目录提供。三问技术附件候选包大小约47.6 MB，尚未纳入最终整合论文；正式提交时须重新核对论文与附件合计体积及身份信息。本文研究引用的外部论文和工具来源均在下列参考文献及[网络核验记录](参考文献网络核验.md)中列明，外部文献不作为本地实验成绩的证据。"""


def main() -> None:
    intro = (HERE / "前言与问题一.md").read_text(encoding="utf-8").strip()
    q2 = load_chapter(E / "问题2/论文问题二正文.md", Q2_CITATIONS)
    q2 = q2.replace("竞赛正文应将本节与问题一、问题三章节统一编号和排版；", "")
    q3 = load_chapter(E / "问题3/论文问题三正文.md", Q3_CITATIONS)
    audit = (HERE / "参考文献网络核验.md").read_text(encoding="utf-8")
    bibliography = audit.split("## 核实的参考文献条目", 1)[1].strip()
    paper = "\n\n".join(
        [
            intro,
            q2,
            q3,
            CONCLUSION,
            "# 参考文献\n\n" + bibliography,
            table_100(),
        ]
    ) + "\n"
    output = HERE / "最终论文_三问整合稿.md"
    output.write_text(paper, encoding="utf-8")

    relative_links = re.findall(r"\]\(([^)]+)\)", paper)
    missing = [
        link
        for link in relative_links
        if not link.startswith(("https://", "http://", "#"))
        and not (HERE / link).exists()
    ]
    assert not missing, missing
    assert paper.count("|T768/A50/V110|词级|") == 100
    with (E / "问题2/results/modeling_experiment/attachment3_predictions_two_seed.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        attachment3_count = sum(1 for _ in csv.DictReader(stream))
    with (E / "问题3/results/attachment4_predictions_explanations.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        attachment4_count = sum(1 for _ in csv.DictReader(stream))
    assert attachment3_count == 30 and attachment4_count == 20
    assert all(f"[{n}]" in paper for n in range(1, 16))
    summary = {
        "paper": output.name,
        "bytes": output.stat().st_size,
        "bibliography_entries": 15,
        "question1_table_rows": 100,
        "attachment3_csv_rows": attachment3_count,
        "attachment4_csv_rows": attachment4_count,
        "relative_links_checked": len(relative_links),
        "missing_relative_links": missing,
    }
    (HERE / "整合核验.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
