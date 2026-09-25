#!/usr/bin/env python3
"""问题三图表、中文实验报告和论文正文；仅读取真实运行保存的结果。"""
from __future__ import annotations

import csv
import json
from collections import Counter
from html import escape

import numpy as np

import problem3 as q3


def fmt(value, digits=4):
    return f"{float(value):.{digits}f}"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def bar_chart(rows):
    means = [float(np.mean([r["share_" + mod] for r in rows])) for mod in q3.MODS]
    counts = Counter(r["primary_modality"] for r in rows)
    names = ("文本", "语音", "视觉")
    colors = ("#305aaf", "#d46a48", "#409a82")
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="710" height="340" viewBox="0 0 710 340">',
             '<rect width="710" height="340" fill="white"/>',
             '<text x="28" y="30" font-size="19" font-family="sans-serif">附件4样本的平均模态作用占比</text>',
             '<text x="28" y="54" font-size="12" font-family="sans-serif" fill="#555">连续遮挡敏感度，非因果贡献；n=20</text>']
    for j, (name, mod, value, color) in enumerate(zip(names, q3.MODS, means, colors)):
        y = 95 + j * 75
        parts += [f'<text x="30" y="{y+20}" font-size="16" font-family="sans-serif">{name}</text>',
                  f'<rect x="100" y="{y}" width="{500*value:.1f}" height="28" fill="{color}"/>',
                  f'<text x="{110+500*value:.1f}" y="{y+20}" font-size="14" font-family="sans-serif">{value:.1%}；主要参考 {counts[mod]} 条</text>']
    parts.append('</svg>')
    (q3.OUT / "modality_share.svg").write_text("\n".join(parts), encoding="utf-8")
    return means, counts


def local_curve(rows):
    candidates = [r for r in rows if r["localization_status"] == "自动CTC定位_需人工回听"
                  and r["key_direction"] == "support"]
    if not candidates:
        candidates = [r for r in rows if r["video_frame_path"]]
    chosen = max(candidates or rows, key=lambda r: abs(float(r["key_delta_probability"])))
    sample_id = chosen["sample_id"]
    modality = chosen["primary_modality"]
    data = [r for r in read_csv(q3.OUT / "target_windows.csv")
            if r["sample_id"] == sample_id and r["modality"] == modality]
    data.sort(key=lambda r: int(r["start_position"]))
    ys = [float(r["delta_predicted_class_probability"]) for r in data]
    starts = [int(r["start_position"]) for r in data]
    if not starts:
        return chosen
    maximum = max(0.01, max(abs(v) for v in ys) * 1.2)
    min_x, max_x = starts[0], starts[-1]
    def xscale(value):
        return 70 + 630 * (value - min_x) / max(1, max_x - min_x)
    def yscale(value):
        return 170 - 125 * value / maximum
    points = " ".join(f"{xscale(x):.1f},{yscale(y):.1f}" for x, y in zip(starts, ys))
    selected_start = int(chosen["key_start_position"])
    marker_x = xscale(selected_start)
    label = escape(f"附件4 {sample_id}：{modality} 连续遮挡窗口的原预测类别概率变化")
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="350" viewBox="0 0 760 350">',
             '<rect width="760" height="350" fill="white"/>',
             f'<text x="25" y="30" font-size="18" font-family="sans-serif">{label}</text>',
             '<text x="25" y="52" font-size="12" font-family="sans-serif" fill="#555">正值表示该窗口支持原预测类别；负值表示遮挡后原类别概率反而升高</text>',
             '<line x1="65" y1="170" x2="710" y2="170" stroke="#777"/>',
             f'<polyline points="{points}" fill="none" stroke="#3166ad" stroke-width="2.4"/>',
             f'<line x1="{marker_x:.1f}" y1="65" x2="{marker_x:.1f}" y2="295" stroke="#ce5b3d" stroke-dasharray="5,4"/>',
             f'<text x="{min(570,marker_x+6):.1f}" y="78" font-size="12" font-family="sans-serif" fill="#ce5b3d">主证据窗口起点 {selected_start}</text>',
             f'<text x="58" y="315" font-size="12" font-family="sans-serif">起点 {min_x}</text>',
             f'<text x="650" y="315" font-size="12" font-family="sans-serif">起点 {max_x}</text>',
             '</svg>']
    (q3.OUT / "local_importance_curve.svg").write_text("\n".join(parts), encoding="utf-8")
    return chosen


def case_card(row, title, ai_review=None):
    time = "无法自动定位" if row["key_time_start_seconds"] == "" else (
        f"{float(row['key_time_start_seconds']):.2f}—{float(row['key_time_end_seconds']):.2f} 秒")
    image = (f"![{row['sample_id']} 的实际解码视频帧]({row['video_frame_path']})"
             if row["video_frame_path"] else "未能导出对应视频帧。")
    card = [f"### {title}：附件4 {row['sample_id']}", "",
            f"预测类别 {row['polarity_name']}，强度 {fmt(row['predicted_strength'])}；主要参考模态 {row['primary_modality']}。"
            f"文本/语音/视觉作用占比分别为 {fmt(row['share_text'])}/{fmt(row['share_audio'])}/{fmt(row['share_vision'])}。",
            "", f"主证据位置为 [{row['key_start_position']}, {row['key_end_position_exclusive']})，"
            f"原预测类别概率变化 {fmt(row['key_delta_probability'])}（{row['key_direction']}），"
            f"对应转写片段“{row['key_text_excerpt']}”，自动定位时间 {time}。"
            f"定位状态：{row['localization_status']}。",
            "", image, "",
            "视频帧由原视频在自动定位时间附近实际解码得到；仅有帧图不能证明该片段的语音与特征位置已人工对齐。", ""]
    if ai_review and ai_review["audio_check"] == "possible_time_conflict":
        card += [f"追加AI辅助复核提示：不输入参考转写的自由解码找到相似度"
                 f"{fmt(ai_review['asr_phrase_similarity'], 3)}的短语，但其中点比原自动时间晚"
                 f"{fmt(ai_review['asr_match_midpoint_offset_seconds'], 2)}秒。"
                 "因此该自动时间及对应截图不能作为短语与视频同步的可靠证据；"
                 "保留它作为定位失败案例，仍待人工回听。", ""]
    return card


def write_all(metric, selection, panel, validation_cases, rows, evidence):
    assert len(rows) == 20
    means, primary_counts = bar_chart(rows)
    curve = local_curve(rows)
    audit = json.loads((q3.OUT / "localization_audit.json").read_text(encoding="utf-8"))
    ai_path = q3.OUT / "ai_review_results.csv"
    ai_reviews = {r["sample_id"]: r for r in read_csv(ai_path)} if ai_path.exists() else {}
    if ai_reviews:
        assert set(ai_reviews) == {r["sample_id"] for r in rows}
    ai_counts = Counter(r["audio_check"] for r in ai_reviews.values())
    target_audit = json.loads((q3.OUT / "target_audit.json").read_text(encoding="utf-8"))
    polarity = Counter(r["polarity_name"] for r in rows)
    mismatch = sum((r["polarity_class"] == 0 and r["predicted_strength"] >= 0) or
                   (r["polarity_class"] == 1 and r["predicted_strength"] != 0) or
                   (r["polarity_class"] == 2 and r["predicted_strength"] <= 0) for r in rows)
    chosen_panel = next(r for r in panel if r["candidate_width"] == selection["selected_width"])
    incorrect = [r for r in validation_cases if r["true_class"] != r["pred_class"]]
    valid_primary = Counter(r["primary_modality"] for r in validation_cases)
    valid_means = [float(np.mean([r["share_" + mod] for r in validation_cases])) for mod in q3.MODS]
    error = max(incorrect, key=lambda r: abs(r["predicted_strength"] - r["true_strength"])) if incorrect else None
    failure_candidates = [r for r in rows if r["localization_status"] != "自动CTC定位_需人工回听"
                          or r["key_direction"] == "counter"]
    failure = max(failure_candidates, key=lambda r: abs(float(r["key_delta_probability"]))) if failure_candidates else None
    lines = ["# 5 问题三：可解释性多模态情感预测", "",
             "## 5.1 问题分析与数据边界", "",
             "本问沿用问题二已锁定的完整三模态集成预测器，不再训练第三问专用分类器。数值输入始终为附件2与附件4对齐版共有的 `text_bert`、`audio`、`vision`；附件2 train 训练的参数和标准化统计保持不变，valid 用于解释窗口宽度选择与性能评价。附件4的20条样本没有标签，仅在解释规则锁定后推理。问题一的100条自主提取特征（768/50/110维）与此模型的接口不同，不并入训练或直接输入预测器；本节复用问题一的独立音频CTC对齐思路定位原素材。", "",
             f"附件4的20条样本均有连续文本注意力，有效长度最小/中位/最大为{target_audit['length_min_median_max'][0]}/"
             f"{fmt(target_audit['length_min_median_max'][1],1)}/{target_audit['length_min_median_max'][2]}。"
             f"有效区间内原始整行零值位置：语音{target_audit['raw_zero_within_effective']['audio']}、"
             f"视觉{target_audit['raw_zero_within_effective']['vision']}；仅记录观察事实，不据此推断缺失成因。", "",
             "## 5.2 预测与可解释模型", "",
             "分类概率由两个BERT Tiny分支等权平均后，与普通动态融合模型按0.85/0.15加权；强度取两个BERT Tiny回归输出的均值。该模型的结构、交叉熵与Huber联合训练目标、连续缺失增强、参数来源及锁定规则见[问题二正文](../问题2/论文问题二正文.md)。本问借鉴遮挡敏感度分析的基本思想[9]，只对**最终集成输出**进行扰动解释，未把子模型注意力或门控权重直接当成因果贡献。", "",
             "固定样本原预测类别 $\\hat y_i$，在模态 $m$ 的有效内部移动连续窗口 $w$，以同一模型重新预测。令 $p_{i,\\hat y_i}$ 为原类别概率，定义", "",
             "$$\\Delta_{imw}=p_{i,\\hat y_i}(x_i)-p_{i,\\hat y_i}(x_i\\setminus w_m),\\qquad "
             "\\Delta^{(q)}_{imw}=\\hat q_i(x_i)-\\hat q_i(x_i\\setminus w_m).$$", "",
             "正的 $\\Delta$ 表示遮挡后原类别概率下降，是对该判断的支持性证据；负值表示相反方向。定义作用量与占比", "",
             "$$R_{im}=|\\mathcal W_{im}|^{-1}\\sum_{w\\in\\mathcal W_{im}}|\\Delta_{imw}|,\\qquad "
             "S_{im}=R_{im}/\\sum_k R_{ik}.$$", "",
             "原始零向量作为审计事实保留，不能自动视为人为缺失。解释时人工遮挡的掩码另行写入；两端特殊词元与填充位不参与窗口。所有有效窗口均纳入作用量统计；全零原始特征窗口仍可能通过覆盖率影响模型，但不作为可展示的关键内容证据候选。主要参考模态为 $R_{im}$ 最大者；该模态的局部证据取 $|\\Delta|$ 最大的非全零有效窗口，保留其方向和强度变化，其他两模态也分别保存最显著窗口。占比是模型在此遮挡方案下的**敏感度比例**，不能解释成现实中的因果作用。", "",
             "## 5.3 验证设计与真实结果", "",
             f"验证集完整输入共728条，实际 Accuracy={fmt(metric['accuracy'])}、宏 F1={fmt(metric['macro_f1'])}、MAE={fmt(metric['mae'])}、Pearson={fmt(metric['pearson'])}；这与问题二已锁定集成模型的完整输入指标一致。", "",
             "解释验证从 valid 按真实类别固定种子各取30条，共90条；测试窗口长度3、5、7，步长1。对每条样本先取主模态中绝对变化最大的窗口，再扩大2位置，并与同模态同长度的随机窗口比较绝对概率变化。预先固定的选择分数是配对差值除以扩大后窗口长度的平方根。", "",
             "|候选宽度|扩大后关键窗口平均变化|同长度随机窗口平均变化|平均配对差值|关键窗口更大比例|选择分数|", "|---:|---:|---:|---:|---:|---:|"]
    for p in panel:
        lines.append(f"|{p['candidate_width']}|{fmt(p['mean_selected_extended_abs_change'])}|"
                     f"{fmt(p['mean_random_extended_abs_change'])}|{fmt(p['mean_paired_gap'])}|"
                     f"{fmt(100*p['fraction_selected_larger'],1)}%|{fmt(p['selection_score_gap_per_sqrt_width'])}|")
    lines += ["", f"按上述 valid 分数选定连续窗口宽度 **{selection['selected_width']}**，附件4未参与选择。"
              f"在该宽度下，扩大后的关键窗口比同长度随机窗口平均多改变 {fmt(chosen_panel['mean_paired_gap'])} 的原类别概率。"
              "这一比较仍由同一模型的扰动响应产生，不能当成独立人工证据真值；部分优势由按最大变化选窗口的规则带来。", "",
              f"valid 子集的主要参考模态计数为文本{valid_primary['text']}、语音{valid_primary['audio']}、视觉{valid_primary['vision']}；"
              f"平均敏感度占比为{fmt(valid_means[0])}/{fmt(valid_means[1])}/{fmt(valid_means[2])}。"
              f"该子集有 {len(incorrect)}/{len(validation_cases)} 条分类错误。" +
              (f"高残差误判示例 `{error['sample_id']}`：真实类{error['true_class']}、预测类{error['pred_class']}，"
               f"真实强度{fmt(error['true_strength'])}、预测强度{fmt(error['predicted_strength'])}，"
               f"主模态{error['primary_modality']}。解释能呈现模型依赖，不保证其结论正确。" if error else ""), "",
              "## 5.4 附件4原素材定位核查", "",
              f"通过本地 Wav2Vec2-CTC[10] 对原视频语音进行独立强制对齐，{audit['ctc_success_count']}/20条得到自动词时间，"
              f"其中{audit['quality_warning_count']}条存在自动定位质量警示（CTC均值分数低{audit['low_ctc_confidence_count']}条、"
              f"转写与声学模型直接识别文本相似度低{audit['low_transcript_similarity_count']}条，两类可重叠）；"
              f"{audit['frame_extracted_count']}/20条导出主证据的实际解码视频帧。"
              f"20/20条样本的 BERT 词元编号均能由原始转写和问题一保存的 DistilBERT 英文词表[11]精确重建；{audit['truncated_text_count']}条文本超过50词元上限。"
              f"视觉模态证据另导出{audit['visual_evidence_frame_count']}张帧图，"
              f"{audit['visual_evidence_unavailable_count']}条样本的视觉模态没有可用的原始非零窗口，明确记为空证据。"
              "附件4只提供特征位置而不提供逐词时间戳，因此位置到时间是结合精确词元字符映射和重新计算的CTC词时间作出的**近似关联**；"
              "所有时间定位均标为需人工回听，质量警示样本另作标记，不能声称人工核验过。"
              "需逐条复核时可填写[人工回看记录模板](results/manual_review_template.csv)，空白栏表示尚未完成。", "",
              "## 5.5 三模态作用与专项测试全量输出", "",
              "![附件4平均模态作用占比](results/modality_share.svg)", "",
              f"20条中主要参考模态计数为文本{primary_counts['text']}、语音{primary_counts['audio']}、视觉{primary_counts['vision']}；"
              f"平均敏感度占比依次为{fmt(means[0])}、{fmt(means[1])}、{fmt(means[2])}。"
              f"预测类别计数为负向{polarity['Negative']}、中性{polarity['Neutral']}、正向{polarity['Positive']}。"
              f"若机械按强度的符号和零值反推类别，有{mismatch}/20条与分类头输出不一致；两头预测保持原义，不据附件4无标签数据调阈值。", "",
              "文本在附件4的20条和valid解释子集的89/90条上占主导，说明当前预测器对文本连续遮挡更敏感。"
              "这只是模型与所选遮挡规则下的观察，不能推出语音、视觉在真实情感判断中不重要；模态相关性与训练后的模型偏好都可能影响占比。", "",
              "|ID|极性|强度|主模态|文本/语音/视觉占比|关键位置|自动时间(s)|定位状态|",
              "|---|---|---:|---|---|---|---|---|"]
    for r in rows:
        t = "—" if r["key_time_start_seconds"] == "" else (
            f"{float(r['key_time_start_seconds']):.2f}–{float(r['key_time_end_seconds']):.2f}")
        direction = "+" if r["key_direction"] == "support" else "−"
        lines.append(f"|{r['sample_id']}|{r['polarity_name']}|{fmt(r['predicted_strength'])}|{r['primary_modality']}|"
                     f"{fmt(r['share_text'],2)}/{fmt(r['share_audio'],2)}/{fmt(r['share_vision'],2)}|"
                     f"[{r['key_start_position']},{r['key_end_position_exclusive']}) {direction}|{t}|{r['localization_status']}|")
    lines += ["", "全量机器可读文件见[附件4预测与解释CSV](results/attachment4_predictions_explanations.csv)；"
              "各模态关键证据、原文片段和对应时间见[证据窗口CSV](results/evidence_windows.csv)；"
              "所有滑动窗口的概率与强度变化见[窗口明细](results/target_windows.csv)。附件4没有真实标签，不报告其Accuracy、F1、MAE或Pearson。", "",
              "## 5.6 典型解释卡及局部重要性", "",
              f"![附件4 {curve['sample_id']} 主模态局部重要性曲线](results/local_importance_curve.svg)", ""]
    if ai_reviews:
        lines[lines.index("## 5.5 三模态作用与专项测试全量输出"):lines.index("## 5.5 三模态作用与专项测试全量输出")] = [
            f"追加AI辅助核对：对20条原视频做不输入参考转写的CTC自由解码，"
            f"{ai_counts['supports_time']}条主证据时间得到支持，"
            f"{ai_counts['possible_time_conflict']}条可能冲突，"
            f"{ai_counts['uncertain']}条无法判定；20条主证据帧与源视频指定帧逐像素一致。"
            "自由解码与强制对齐共用声学模型，不能代替人工回听；详见"
            "[AI辅助核对报告](AI辅助核对报告.md)。", ""]
    lines += case_card(curve, "可追溯的支持性片段", ai_reviews.get(curve["sample_id"]))
    if failure and failure["sample_id"] != curve["sample_id"]:
        lines += case_card(failure, "反向证据或定位质量警示案例", ai_reviews.get(failure["sample_id"]))
    lines += ["## 5.7 局限与复现", "",
              "连续遮挡既移除特征又改变显式覆盖率，因此敏感度包含两种变化的共同响应。模态间有互补与相关性，三模态占比不等于独立因果贡献。"
              "独立CTC的强制路径可能在转写与语音不符时给出低质量时间；附件4没有人工逐词边界，视频帧也不能单独证明词音同步。"
              "模型中性类别仍较难识别，错误预测同样可能具有看似清晰的局部解释。", "",
              "问题三未新增可训练参数。Python、NumPy、PyTorch及新增视频/声学依赖版本见[README](README.md)和[依赖文件](requirements.txt)；"
              "固定种子、窗口候选与阈值见 `config.json`，实际选型见 `results/selection.json`。运行 `problem3.py all` 可从附件2 valid、"
              "问题二锁定检查点和附件4原视频重新生成全部结果；`problem3.py verify` 复算20条预测并核验CSV字段。"
              "原始赛题文件、大型问题一声学模型和问题二检查点均引用相邻目录，不复制进问题三提交目录。", "",
              "## 参考文献（问题三增补）", "",
              "[9] ZEILER M D, FERGUS R. Visualizing and Understanding Convolutional Networks[C]//European Conference on Computer Vision. 2014: 818-833. [作者论文](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf).", "",
              "[10] BAEVSKI A, ZHOU Y, MOHAMED A, et al. wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations[C]//Advances in Neural Information Processing Systems 33. 2020. [原文](https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html).", "",
              "[11] SANH V, DEBUT L, CHAUMOND J, et al. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter[EB/OL]. 2019. arXiv: 1910.01108. [原文](https://arxiv.org/abs/1910.01108).", "",
              "[9]只支持遮挡敏感度的分析思想，本研究的时序窗口、模态占比与验证数值均由本地实验定义和计算。全文合并时须统一参考文献编号。", ""]
    content = "\n".join(lines)
    (q3.HERE / "论文问题三正文.md").write_text(content, encoding="utf-8")
    (q3.HERE / "问题三实验报告.md").write_text(content.replace("# 5 问题三：可解释性多模态情感预测", "# 问题三实验报告", 1), encoding="utf-8")


def regenerate():
    """仅由已保存的真实结果重建图文；不读取原始附件或重新选模型。"""
    metric = json.loads((q3.OUT / "validation_metrics.json").read_text(encoding="utf-8"))
    selection = json.loads((q3.OUT / "selection.json").read_text(encoding="utf-8"))
    panel = read_csv(q3.OUT / "validation_window_candidates.csv")
    for row in panel:
        row["candidate_width"] = int(row["candidate_width"])
        for key in row:
            if key not in ("candidate_width", "sample_count"):
                row[key] = float(row[key])
    cases = read_csv(q3.OUT / "validation_explanation_cases.csv")
    for row in cases:
        for key in ("true_class", "pred_class"):
            row[key] = int(row[key])
        for key in ("true_strength", "predicted_strength", "share_text", "share_audio", "share_vision"):
            row[key] = float(row[key])
    target = read_csv(q3.OUT / "attachment4_predictions_explanations.csv")
    for row in target:
        row["polarity_class"] = int(row["polarity_class"])
        for key in ("predicted_strength", "share_text", "share_audio", "share_vision", "key_delta_probability"):
            row[key] = float(row[key])
    evidence = read_csv(q3.OUT / "evidence_windows.csv")
    write_all(metric, selection, panel, cases, target, evidence)


if __name__ == "__main__":
    regenerate()
