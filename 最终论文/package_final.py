#!/usr/bin/env python3
"""把三问技术附件与论文Markdown组合成50 MB以内的候选包。"""
from __future__ import annotations

import json
import posixpath
import re
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
E = HERE.parent
TECHNICAL = E / "问题3/技术附件候选.zip"
OUTPUT = HERE / "三问论文与技术附件_候选.zip"
LIMIT = 50_000_000
EXTRA = [
    "README.md",
    "前言与问题一.md",
    "最终论文_三问整合稿.md",
    "参考文献网络核验.md",
    "整合核验.json",
    "build_final_paper.py",
    "package_final.py",
]


def main() -> None:
    with zipfile.ZipFile(TECHNICAL) as source, zipfile.ZipFile(
        OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as target:
        assert source.testzip() is None
        for item in source.infolist():
            target.writestr(
                item.filename,
                source.read(item),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            )
        for relative in EXTRA:
            target.write(HERE / relative, "最终论文/" + relative)

    with zipfile.ZipFile(OUTPUT) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
        assert len(names) == len(archive.namelist())
        assert sum(
            x.startswith("question1/results/features/") and x.endswith(".npz")
            for x in names
        ) == 100
        assert "问题2/results/checkpoint_full.pt" in names
        assert "问题3/results/attachment4_predictions_explanations.csv" in names
        assert "最终论文/参考文献网络核验.md" in names
        text = archive.read("最终论文/最终论文_三问整合稿.md").decode("utf-8")
        links = re.findall(r"\]\(([^)]+)\)", text)
        unresolved = [
            posixpath.normpath(posixpath.join("最终论文", url))
            for url in links
            if not url.startswith(("http://", "https://", "#"))
            and posixpath.normpath(posixpath.join("最终论文", url)) not in names
        ]
        assert not unresolved, unresolved
        count = len(names)

    size = OUTPUT.stat().st_size
    assert size <= LIMIT, (size, LIMIT)
    audit = {
        "candidate_package": OUTPUT.name,
        "size_bytes": size,
        "limit_bytes": LIMIT,
        "reserve_bytes": LIMIT - size,
        "files": count,
        "paper_and_reference_audit_included": True,
        "paper_format": "Markdown source; competition layout still required",
        "unresolved_paper_links": [],
    }
    (HERE / "打包核验.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
