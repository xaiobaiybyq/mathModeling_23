#!/usr/bin/env python3
"""制作三问技术附件候选包并核查50MB体积；最终论文另行合并。"""
from __future__ import annotations

import json
import re
import zipfile
import csv
from pathlib import Path

E_ROOT = Path(__file__).resolve().parent.parent
Q1 = E_ROOT / "问题1" if (E_ROOT / "问题1").is_dir() else E_ROOT / "question1"
Q2 = E_ROOT / "问题2"
Q3 = E_ROOT / "问题3"
LIMIT = 50_000_000

Q2_FILES = [
    "README.md", "requirements.txt", "config.json", "optimize_config.json",
    "problem2.py", "optimize.py", "modeling_experiment.py", "calibrate.py",
    "final_test.py", "final_test_plan.json", "paper_table.py", "final_report.py",
    "论文问题二正文.md", "第二问最终报告.md", "工作交付说明.md",
    "results/normalization.json", "results/checkpoint_full.pt",
    "results/optimization/tiny_weighted.pt", "results/modeling_experiment/tiny_weighted.pt",
    "results/ablation_summary.csv", "results/validation_scenarios.csv",
    "results/modeling_experiment/validation_scenarios.csv",
    "results/modeling_experiment/selection.json",
    "results/modeling_experiment/attachment3_predictions_two_seed.csv",
    "results/modeling_experiment/error_cases.csv",
    "results/final_test/test_metrics.csv", "results/final_test/summary.json",
    "results/final_summary/validation_factor_summary.csv",
    "results/final_summary/validation_missing_factors.svg",
]
Q3_FILES = [
    "question3.txt", "README.md", "requirements.txt", "config.json", "problem3.py",
    "localize.py", "report.py", "package_check.py", "论文问题三正文.md", "问题三实验报告.md",
    "第三问成果交接.md",
]


def make_package():
    with zipfile.ZipFile(Q1 / "question1_submission.zip") as first:
        assert sum(name.startswith("question1/results/features/") and name.endswith(".npz")
                   for name in first.namelist()) == 100
    with (Q3 / "results/attachment4_predictions_explanations.csv").open(encoding="utf-8-sig", newline="") as f:
        predictions = list(csv.DictReader(f))
    assert len(predictions) == len({row["sample_id"] for row in predictions}) == 20
    for file in Q2_FILES:
        assert (Q2 / file).is_file(), file
    q3_files = [Q3 / file for file in Q3_FILES]
    q3_files += [p for p in (Q3 / "results").iterdir()
                 if p.is_file() and p.name != "package_audit.json" and p.suffix in (".csv", ".json", ".svg")]
    paper = (Q3 / "论文问题三正文.md").read_text(encoding="utf-8")
    frame_paths = sorted(set(re.findall(r"results/frames/[^)\s]+\.png", paper)))
    for relative in frame_paths:
        q3_files.append(Q3 / relative)
    for file in q3_files:
        assert file.is_file(), file
    output = Q3 / "技术附件候选.zip"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        with zipfile.ZipFile(Q1 / "question1_submission.zip") as first:
            for item in first.infolist():
                assert item.filename.startswith("question1/") and not item.is_dir()
                archive.writestr(item.filename, first.read(item), compress_type=zipfile.ZIP_DEFLATED,
                                 compresslevel=6)
        for relative in Q2_FILES:
            archive.write(Q2 / relative, "问题2/" + relative)
        for file in q3_files:
            archive.write(file, "问题3/" + str(file.relative_to(Q3)))
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(set(archive.namelist()))
        assert "问题3/results/attachment4_predictions_explanations.csv" in archive.namelist()
        assert sum(name.startswith("question1/results/features/") and name.endswith(".npz")
                   for name in archive.namelist()) == 100
        archive_count = len(archive.namelist())
    size = output.stat().st_size
    summary = {"candidate_package": output.name, "candidate_size_bytes": size,
               "limit_bytes": LIMIT, "reserve_bytes_for_final_paper": LIMIT - size,
               "fits_without_final_paper": size <= LIMIT, "files": archive_count,
               "contains_question1_100_features": True, "contains_question2_selected_checkpoints": True,
               "contains_question3_all_20_csv": True,
               "contains_question3_paper_card_frames": len(frame_paths),
               "contains_original_video_or_large_source_models": False,
               "final_competition_paper_included": False}
    (Q3 / "results" / "package_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    make_package()
