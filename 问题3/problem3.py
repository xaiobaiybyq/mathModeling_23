#!/usr/bin/env python3
"""E题问题三：锁定问题二集成模型，按连续窗口扰动解释附件4。"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
E_ROOT = HERE.parent
Q2 = E_ROOT / "问题2"
Q1 = E_ROOT / "问题1" if (E_ROOT / "问题1").is_dir() else E_ROOT / "question1"
sys.path.insert(0, str(Q2))

import problem2 as p2  # noqa: E402
import optimize as opt  # noqa: E402
import modeling_experiment as ensemble  # noqa: E402
import final_test as final_lock  # noqa: E402

CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
OUT = HERE / "results"
TARGET = E_ROOT / "E题数据" / "附件4-可解释专项视频样本与特征文件" / "附件4-可解释专项视频样本与特征文件" / "对齐版本"
MODS = ("text", "audio", "vision")
NAMES = ("Negative", "Neutral", "Positive")


def seed_all():
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.set_num_threads(CONFIG["threads"])


def numpy_pickle_compat():
    """附件4由NumPy 2写入；兼容本实验NumPy 1.24的pickle模块路径。"""
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def load_valid(stats):
    source = p2.read_pickle(p2.SOURCE)
    split = source["valid"]
    x = p2.raw_inputs(split, split["id"])
    x["class"] = np.asarray(split["classification_labels"], dtype=np.int64).reshape(-1)
    x["strength"] = np.asarray(split["regression_labels"], dtype=np.float32).reshape(-1)
    assert len(x["id"]) == 728
    assert np.array_equal(x["class"], np.where(x["strength"] < 0, 0, np.where(x["strength"] > 0, 2, 1)))
    del source
    return p2.transform(x, stats)


def load_target(stats):
    numpy_pickle_compat()
    files = sorted(TARGET.glob("*.pkl"))
    videos = sorted((TARGET / "videos").glob("*.mp4"))
    assert len(files) == len(videos) == 20
    assert {f.stem for f in files} == {v.stem for v in videos}
    records = []
    for path in files:
        with path.open("rb") as f:
            record = pickle.load(f)
        assert {"text_bert", "audio", "vision", "raw_text"} <= set(record)
        assert str(record.get("id", path.stem)) == path.stem
        records.append(record)
    ids = [path.stem for path in files]
    stacked = {k: np.stack([np.asarray(rec[k]) for rec in records]) for k in ("text_bert", "audio", "vision")}
    x = p2.raw_inputs(stacked, ids)
    raw_text = [str(rec["raw_text"]) for rec in records]
    audit = {
        "samples": len(ids), "video_pairs": len(videos),
        "input_fields": ["text_bert", "audio", "vision"],
        "length_min_median_max": [int(x["lengths"].min()), float(np.median(x["lengths"])), int(x["lengths"].max())],
        "continuous_text_attention": int(sum(np.array_equal(m, np.arange(50) < m.sum()) for m in x["effective"][:, 0])),
        "raw_zero_within_effective": {m: int((x["raw_zero"][:, j] & x["effective"][:, j]).sum()) for j, m in enumerate(MODS)},
        "artificial_missing_positions_before_explanation": int(x["synthetic_missing"].sum()),
        "labels_present": False,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "target_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return p2.transform(x, stats), raw_text


def load_models():
    final_lock.verify_lock()  # 只校验锁定检查点与计划，不读取test标签。
    assert json.loads((Q2 / "results/modeling_experiment/selection.json").read_text())["accepted"]
    return (opt.load_model(ensemble.OLD_CHECKPOINT),
            opt.load_model(ensemble.NEW_CHECKPOINT),
            p2.load_model("full"))


@torch.inference_mode()
def predict_tensor(models, batch):
    first, second, ordinary = models
    a, ar, _ = first(batch)
    b, br, _ = second(batch)
    c, _, _ = ordinary(batch)
    probs = 0.85 * (torch.softmax(a, 1) + torch.softmax(b, 1)) / 2 + 0.15 * torch.softmax(c, 1)
    strength = (ar + br) / 2
    return probs.cpu().numpy(), strength.cpu().numpy()


def predict_data(models, x):
    ps, rs = [], []
    for lo in range(0, len(x["id"]), CONFIG["batch_size"]):
        hi = min(len(x["id"]), lo + CONFIG["batch_size"])
        b = p2.subset(x, slice(lo, hi))
        p, r = predict_tensor(models, p2.to_torch(b))
        ps.append(p); rs.append(r)
    return np.concatenate(ps), np.concatenate(rs)


def windows(x, index, width, modality=None):
    """仅在有效内部形成连续窗口；两端特殊词元、填充均不遮挡。"""
    length = int(x["lengths"][index])
    if length <= 2:
        return []
    span = min(width, length - 2)
    mods = range(3) if modality is None else [modality]
    result = []
    for m in mods:
        effective = x["effective"][index, m]
        for start in range(1, length - span):
            end = start + span
            if np.all(effective[start:end]):
                result.append((m, start, end))
    return result


def masked_predict(models, x, index, requests):
    """requests为(modality,start,end)列表；零值与人工遮挡掩码分开记录。"""
    all_p, all_r = [], []
    for lo in range(0, len(requests), CONFIG["batch_size"]):
        chunk = requests[lo:lo + CONFIG["batch_size"]]
        ix = np.full(len(chunk), index, dtype=np.int64)
        batch = p2.subset(x, ix)
        for row, (mod, start, end) in enumerate(chunk):
            assert np.all(batch["effective"][row, mod, start:end])
            batch["synthetic_missing"][row, mod, start:end] = True
            batch["obs"][row, mod, start:end] = False
            if mod == 0:
                batch["tokens"][row, start:end] = 0
            elif mod == 1:
                batch["audio"][row, start:end] = 0
            else:
                batch["vision"][row, start:end] = 0
        p, r = predict_tensor(models, p2.to_torch(batch))
        all_p.append(p); all_r.append(r)
    return np.concatenate(all_p), np.concatenate(all_r)


def explain_item(models, x, index, width, base_probs, base_strength):
    requests = windows(x, index, width)
    if not requests:
        raise ValueError(f"no valid occlusion windows: {x['id'][index]}")
    occluded_p, occluded_r = masked_predict(models, x, index, requests)
    category = int(np.argmax(base_probs))
    rows = []
    for (mod, start, end), p, strength in zip(requests, occluded_p, occluded_r):
        zero_fraction = float(np.mean(x["raw_zero"][index, mod, start:end])) if mod else 0.0
        rows.append({"sample_id": x["id"][index], "modality": MODS[mod], "modality_index": mod,
                     "start_position": start, "end_position_exclusive": end,
                     "masked_effective_positions": end - start,
                     "raw_zero_fraction": zero_fraction,
                     "delta_predicted_class_probability": float(base_probs[category] - p[category]),
                     "delta_strength": float(base_strength - strength)})
    by_mod = []
    top = []
    for mod in range(3):
        all_mod = [r for r in rows if r["modality_index"] == mod]
        eligible = [r for r in rows if r["modality_index"] == mod and r["raw_zero_fraction"] < 1]
        by_mod.append(float(np.mean([abs(r["delta_predicted_class_probability"]) for r in all_mod])) if all_mod else 0.0)
        top.append(max(eligible, key=lambda r: abs(r["delta_predicted_class_probability"])) if eligible else None)
    total = sum(by_mod)
    shares = [v / total for v in by_mod] if total > 0 else [1 / 3] * 3
    primary = int(np.argmax(by_mod))
    if top[primary] is None:
        raise ValueError(f"no informative evidence window: {x['id'][index]}")
    return {"rows": rows, "effect": by_mod, "shares": shares, "primary": primary, "top": top}


def balanced_valid_indices(valid):
    rng = np.random.default_rng(CONFIG["seed"])
    take = CONFIG["validation_per_class"]
    ix = np.concatenate([rng.choice(np.flatnonzero(valid["class"] == c), size=take, replace=False)
                         for c in range(3)])
    return np.sort(ix)


def expanded_window(x, index, request, extension):
    mod, start, end = request
    length = int(x["lengths"][index])
    width = min(end - start + extension, length - 2)
    center = (start + end - 1) / 2
    new_start = max(1, min(length - 1 - width, int(round(center - (width - 1) / 2))))
    return mod, new_start, new_start + width


def validate_explanation(models, valid, probs, strengths):
    chosen = balanced_valid_indices(valid)
    rng = np.random.default_rng(CONFIG["seed"] + 11)
    panel = []
    records_by_width = {}
    for width in CONFIG["candidate_window_widths"]:
        records = []
        for index in chosen:
            item = explain_item(models, valid, int(index), width, probs[index], strengths[index])
            mod = item["primary"]
            selected = item["top"][mod]
            request = (mod, selected["start_position"], selected["end_position_exclusive"])
            expanded = expanded_window(valid, int(index), request, CONFIG["selection_extension"])
            candidates = windows(valid, int(index), expanded[2] - expanded[1], mod)
            candidates = [r for r in candidates if float(np.mean(valid["raw_zero"][index, mod, r[1]:r[2]])) < 1]
            random_request = candidates[int(rng.integers(len(candidates)))]
            altered_p, _ = masked_predict(models, valid, int(index), [expanded, random_request])
            category = int(probs[index].argmax())
            selected_change = float(abs(probs[index, category] - altered_p[0, category]))
            random_change = float(abs(probs[index, category] - altered_p[1, category]))
            records.append({"sample_id": valid["id"][index], "true_class": int(valid["class"][index]),
                            "pred_class": category, "predicted_strength": float(strengths[index]),
                            "true_strength": float(valid["strength"][index]),
                            "candidate_width": width, "primary_modality": MODS[mod],
                            "selected_start": selected["start_position"],
                            "selected_end_exclusive": selected["end_position_exclusive"],
                            "selected_direction": "support" if selected["delta_predicted_class_probability"] > 0 else "counter",
                            "selected_delta_probability": selected["delta_predicted_class_probability"],
                            "extended_selected_abs_change": selected_change,
                            "extended_random_abs_change": random_change,
                            "extended_width": expanded[2] - expanded[1],
                            "share_text": item["shares"][0], "share_audio": item["shares"][1],
                            "share_vision": item["shares"][2]})
        records_by_width[width] = records
        gaps = np.array([r["extended_selected_abs_change"] - r["extended_random_abs_change"] for r in records])
        adjusted = np.array([v / np.sqrt(r["extended_width"]) for v, r in zip(gaps, records)])
        panel.append({"candidate_width": width, "sample_count": len(records),
                      "mean_selected_extended_abs_change": float(np.mean([r["extended_selected_abs_change"] for r in records])),
                      "mean_random_extended_abs_change": float(np.mean([r["extended_random_abs_change"] for r in records])),
                      "mean_paired_gap": float(gaps.mean()),
                      "fraction_selected_larger": float((gaps > 0).mean()),
                      "selection_score_gap_per_sqrt_width": float(adjusted.mean()),
                      "supporting_top_fraction": float(np.mean([r["selected_direction"] == "support" for r in records]))})
        print("validation explanation width", width, "gap", round(float(gaps.mean()), 5), flush=True)
    selected_width = max(panel, key=lambda r: (r["selection_score_gap_per_sqrt_width"], -r["candidate_width"]))["candidate_width"]
    write_csv(OUT / "validation_window_candidates.csv", panel)
    write_csv(OUT / "validation_explanation_cases.csv", records_by_width[selected_width])
    selection = {"selected_width": selected_width, "stride": CONFIG["window_stride"],
                 "rule": "mean paired absolute class-probability change of selected versus random expanded window, divided by sqrt(expanded width); candidates 3/5/7, fixed stratified valid subset",
                 "validation_subset_size": len(chosen), "validation_indices_seed": CONFIG["seed"],
                 "attachment4_used_for_selection": False, "test_used_for_selection": False,
                 "top_evidence_rule": "maximum absolute class-probability change among windows with at least one nonzero original feature; retain direction; no merging"}
    (OUT / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    return selection, panel, records_by_width[selected_width]


def write_csv(path, rows):
    assert rows
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def target_explanations(models, target, probs, strengths, width):
    rows, evidence, windows_all = [], [], []
    for i, sample_id in enumerate(target["id"]):
        result = explain_item(models, target, i, width, probs[i], strengths[i])
        category = int(probs[i].argmax())
        key = result["top"][result["primary"]]
        rows.append({"sample_id": sample_id, "source_file": sample_id + ".pkl",
                     "video_file": sample_id + ".mp4", "polarity_class": category,
                     "polarity_name": NAMES[category], "predicted_strength": float(strengths[i]),
                     "prob_negative": float(probs[i, 0]), "prob_neutral": float(probs[i, 1]),
                     "prob_positive": float(probs[i, 2]), "primary_modality": MODS[result["primary"]],
                     "effect_text": result["effect"][0], "effect_audio": result["effect"][1],
                     "effect_vision": result["effect"][2],
                     "share_text": result["shares"][0], "share_audio": result["shares"][1],
                     "share_vision": result["shares"][2],
                     "key_start_position": key["start_position"],
                     "key_end_position_exclusive": key["end_position_exclusive"],
                     "key_direction": "support" if key["delta_predicted_class_probability"] > 0 else "counter",
                     "key_delta_probability": key["delta_predicted_class_probability"],
                     "key_delta_strength": key["delta_strength"],
                     "key_raw_zero_fraction": key["raw_zero_fraction"]})
        windows_all.extend(result["rows"])
        for mod, top in enumerate(result["top"]):
            if top is None:
                evidence.append({"sample_id": sample_id, "modality": MODS[mod],
                                 "start_position": "", "end_position_exclusive": "",
                                 "direction": "no_nonzero_feature_window",
                                 "delta_predicted_class_probability": "", "delta_strength": "",
                                 "raw_zero_fraction": 1.0, "mean_absolute_effect": 0.0,
                                 "normalized_share": result["shares"][mod]})
                continue
            evidence.append({"sample_id": sample_id, "modality": MODS[mod],
                             "start_position": top["start_position"],
                             "end_position_exclusive": top["end_position_exclusive"],
                             "direction": "support" if top["delta_predicted_class_probability"] > 0 else "counter",
                             "delta_predicted_class_probability": top["delta_predicted_class_probability"],
                             "delta_strength": top["delta_strength"],
                             "raw_zero_fraction": top["raw_zero_fraction"],
                             "mean_absolute_effect": result["effect"][mod],
                             "normalized_share": result["shares"][mod]})
        print("attachment4 explained", i + 1, "/20", flush=True)
    write_csv(OUT / "target_windows.csv", windows_all)
    return rows, evidence


def verify_explanations(models, target, probs, strengths):
    width = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))["selected_width"]
    with (OUT / "attachment4_predictions_explanations.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(target["id"]) == 20
    for i, saved in enumerate(rows):
        recomputed = explain_item(models, target, i, width, probs[i], strengths[i])
        primary = recomputed["primary"]
        key = recomputed["top"][primary]
        assert saved["primary_modality"] == MODS[primary]
        for j, mod in enumerate(MODS):
            assert abs(float(saved["effect_" + mod]) - recomputed["effect"][j]) < 1e-6
            assert abs(float(saved["share_" + mod]) - recomputed["shares"][j]) < 1e-6
        assert int(saved["key_start_position"]) == key["start_position"]
        assert int(saved["key_end_position_exclusive"]) == key["end_position_exclusive"]
        assert abs(float(saved["key_delta_probability"]) - key["delta_predicted_class_probability"]) < 1e-6
        assert abs(float(saved["key_delta_strength"]) - key["delta_strength"]) < 1e-6
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("all", "verify"))
    args = parser.parse_args()
    seed_all()
    stats = json.loads((Q2 / "results/normalization.json").read_text(encoding="utf-8"))
    models = load_models()
    if args.command == "verify":
        import localize
        target, raw_text = load_target(stats)
        probs, strengths = predict_data(models, target)
        localize.verify(target, raw_text, probs, strengths)
        verify_explanations(models, target, probs, strengths)
        print("verified all 20 attachment4 outputs", flush=True)
        return
    valid = load_valid(stats)
    valid_probs, valid_strengths = predict_data(models, valid)
    metric = p2.metrics(valid["class"], valid["strength"], valid_probs, valid_strengths)
    (OUT / "validation_metrics.json").write_text(json.dumps({"sample_count": len(valid["id"]),
        "model": "locked two-BERT plus ordinary 0.15 ensemble, complete input", **metric},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("valid complete metrics", json.dumps(metric), flush=True)
    selection, panel, validation_cases = validate_explanation(models, valid, valid_probs, valid_strengths)
    target, raw_text = load_target(stats)
    target_probs, target_strengths = predict_data(models, target)
    target_rows, evidence = target_explanations(models, target, target_probs, target_strengths, selection["selected_width"])
    import localize
    target_rows, evidence, location_audit = localize.localize(target, raw_text, target_rows, evidence)
    write_csv(OUT / "attachment4_predictions_explanations.csv", target_rows)
    write_csv(OUT / "evidence_windows.csv", evidence)
    (OUT / "localization_audit.json").write_text(json.dumps(location_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    import report
    report.write_all(metric, selection, panel, validation_cases, target_rows, evidence)
    localize.verify(target, raw_text, target_probs, target_strengths)
    verify_explanations(models, target, target_probs, target_strengths)
    print("completed attachment4", len(target_rows), "width", selection["selected_width"], flush=True)


if __name__ == "__main__":
    main()
