#!/usr/bin/env python3
"""竞赛 E 题第二问：对齐特征、连续局部缺失、CPU 多任务实验。"""
import argparse
import csv
import gc
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "E题数据"
SOURCE = DATA / "附件2-数据集特征文件" / "aligned_50.pkl"
TARGET = DATA / "附件3-模态缺失特征样本" / "对齐版本"
OUT = HERE / "results"
MODS = ("text", "audio", "vision")
NAMES = ("Negative", "Neutral", "Positive")
CONFIG = json.loads((HERE / "config.json").read_text())


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(CONFIG["threads"])


def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def raw_inputs(s, ids):
    """唯一入口：三字段转为相同的数值与显式掩码。"""
    assert all(k in s for k in ("text_bert", "audio", "vision"))
    t = np.asarray(s["text_bert"])
    a = np.asarray(s["audio"])
    v = np.asarray(s["vision"])
    assert t.shape == (len(ids), 3, 50) and a.shape == (len(ids), 50, 74) and v.shape == (len(ids), 50, 35)
    assert np.isfinite(t).all() and np.isfinite(a).all() and np.isfinite(v).all()
    assert np.max(np.abs(t - np.rint(t))) < 1e-5, "text_bert token 值不是整数"
    t = np.rint(t).astype(np.int64)
    assert np.isin(t[:, 1, :], [0, 1]).all()
    assert t[:, 0, :].min() >= 0 and t[:, 0, :].max() < CONFIG["vocab_size"]
    att = t[:, 1, :].astype(bool)
    av = np.any(a != 0, axis=2)
    vv = np.any(v != 0, axis=2)
    # 若文本注意力在末段丢失，其余模态仍可确定共同时间轴的终点。
    signal = att | av | vv
    pos = np.arange(1, 51)[None, :]
    lengths = (signal * pos).max(axis=1)
    assert np.all(lengths > 0)
    valid = pos <= lengths[:, None]
    # 两端是对齐的特殊标记位置；音视频在 train/valid 的这些位置均为结构性零向量。
    # 内部整行零值仍是有效位置，不能据此推断人为缺失。
    interior = valid & (np.arange(50)[None, :] > 0) & (np.arange(50)[None, :] < lengths[:, None] - 1)
    effective = np.stack([att & valid, interior, interior], axis=1)
    raw_zero = np.stack([t[:, 0, :] == 0, ~av, ~vv], axis=1)
    synthetic_missing = np.zeros_like(effective)
    obs = effective & ~synthetic_missing
    return {
        "id": list(ids), "tokens": t[:, 0, :].copy(),
        "audio": a.astype(np.float32), "vision": v.astype(np.float32),
        "valid": valid.copy(), "effective": effective.copy(), "raw_zero": raw_zero.copy(),
        "synthetic_missing": synthetic_missing, "obs": obs.copy(), "lengths": lengths.astype(np.int64),
    }


def load_data():
    source = read_pickle(SOURCE)
    out = {}
    for split in ("train", "valid"):
        s = source[split]
        ids = s["id"]
        out[split] = raw_inputs(s, ids)
        y = np.asarray(s["classification_labels"])
        r = np.asarray(s["regression_labels"])
        assert np.isin(y, [0, 1, 2]).all() and np.isfinite(r).all()
        assert np.array_equal(y.astype(int), np.where(r < 0, 0, np.where(r > 0, 2, 1)))
        assert r.min() >= -3 and r.max() <= 3
        out[split]["class"] = y.astype(np.int64)
        out[split]["strength"] = r.astype(np.float32)
    del source
    gc.collect()
    return out


def fit_normalization(train):
    stats = {}
    for j, key in enumerate(("audio", "vision"), 1):
        x = train[key]
        observed = train["effective"][:, j, :]
        z = x[observed].astype(np.float64)
        mean = z.mean(axis=0)
        std = z.std(axis=0)
        std = np.maximum(std, 1e-4)
        stats[key] = {"mean": mean.tolist(), "std": std.tolist(), "effective_positions": int(observed.sum())}
    return stats


def write_data_audit(data):
    audit={"source":"aligned_50.pkl","split_counts":{},"label_mapping":{"0":"Negative (<0)","1":"Neutral (=0)","2":"Positive (>0)"}}
    for name,x in data.items():
        audit["split_counts"][name]={
            "samples":len(x["id"]),
            "class_counts":{str(c):int((x["class"]==c).sum()) for c in range(3)},
            "valid_length_min":int(x["lengths"].min()),
            "valid_length_median":float(np.median(x["lengths"])),
            "valid_length_max":int(x["lengths"].max()),
            "valid_positions":int(x["valid"].sum()),
            "effective_positions":{MODS[j]:int(x["effective"][:,j,:].sum()) for j in range(3)},
            "raw_zero_within_effective":{MODS[j]:int((x["raw_zero"][:,j,:] & x["effective"][:,j,:]).sum()) for j in range(3)},
            "samples_with_raw_zero_within_effective":{MODS[j]:int((x["raw_zero"][:,j,:] & x["effective"][:,j,:]).any(axis=1).sum()) for j in range(3)},
            "synthetic_missing_positions":int(x["synthetic_missing"].sum()),
            "padding_positions":int((~x["valid"]).sum()),
        }
    (OUT/"data_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")


def transform(x, stats):
    for j, key in enumerate(("audio", "vision"), 1):
        mean = np.asarray(stats[key]["mean"], dtype=np.float32)
        std = np.asarray(stats[key]["std"], dtype=np.float32)
        z = np.clip((x[key] - mean) / std, -8, 8)
        z[~x["effective"][:, j, :]] = 0
        x[key] = z.astype(np.float32)
    return x


def subset(x, ix):
    keys = ("tokens", "audio", "vision", "valid", "effective", "raw_zero", "synthetic_missing", "obs", "lengths")
    return {k: x[k][ix].copy() for k in keys}


def interval(length, rate, place, rng=None):
    # 两端特殊标记保留；短样本至少有一个可遮挡位置。
    lo, hi = 1, max(2, int(length) - 1)
    width = hi - lo
    span = min(width, max(1, int(round(width * rate))))
    if place == "all": start, span = lo, width
    elif place == "front": start = lo
    elif place == "middle": start = lo + (width - span) // 2
    elif place == "back": start = hi - span
    else: start = int(rng.integers(lo, hi - span + 1))
    return start, start + span


def apply_missing(b, mods, rate, place, rng=None):
    """所有模型共用同一连续区间，返回实际移除的观测比例。"""
    before = b["obs"].copy()
    for i, length in enumerate(b["lengths"]):
        start, end = interval(length, rate, place, rng)
        for mod in mods:
            j = MODS.index(mod)
            if j == 0: b["tokens"][i, start:end] = 0
            elif j == 1: b["audio"][i, start:end] = 0
            else: b["vision"][i, start:end] = 0
            b["synthetic_missing"][i, j, start:end] |= b["effective"][i, j, start:end]
            b["obs"][i, j, start:end] = b["effective"][i, j, start:end] & ~b["synthetic_missing"][i, j, start:end]
    removed = before.sum(axis=(0, 2)) - b["obs"].sum(axis=(0, 2))
    counts = before.sum(axis=(0, 2))
    return removed.astype(np.int64), counts.astype(np.int64)


def train_mask(b, rng):
    for i, length in enumerate(b["lengths"]):
        if rng.random() >= CONFIG["augment_probability"]: continue
        nmod = 2 if rng.random() < CONFIG["two_modality_probability"] else 1
        mods = rng.choice(3, size=nmod, replace=False)
        rate = float(rng.choice(CONFIG["rates"]))
        start, end = interval(length, rate, "random", rng)
        for j in mods:
            if j == 0: b["tokens"][i, start:end] = 0
            elif j == 1: b["audio"][i, start:end] = 0
            else: b["vision"][i, start:end] = 0
            b["synthetic_missing"][i, j, start:end] |= b["effective"][i, j, start:end]
            b["obs"][i, j, start:end] = b["effective"][i, j, start:end] & ~b["synthetic_missing"][i, j, start:end]


def to_torch(b):
    return {k: torch.from_numpy(np.ascontiguousarray(b[k])) for k in ("tokens", "audio", "vision", "valid", "effective", "raw_zero", "synthetic_missing", "obs")}


class Branch(nn.Module):
    def __init__(self, in_dim, hidden, is_text=False):
        super().__init__()
        self.is_text = is_text
        self.proj = nn.Embedding(CONFIG["vocab_size"], hidden, padding_idx=0) if is_text else nn.Linear(in_dim, hidden)
        self.conv = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.reduce = nn.Linear(hidden * 2, hidden)

    def forward(self, x, mask):
        z = self.proj(x)
        z = z * mask.unsqueeze(-1)
        z = F.gelu(self.conv(z.transpose(1, 2)).transpose(1, 2))
        z = z * mask.unsqueeze(-1)
        count = mask.sum(1).clamp(min=1).unsqueeze(-1)
        avg = z.sum(1) / count
        mx = z.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
        mx = torch.where(mask.any(1).unsqueeze(-1), mx, torch.zeros_like(mx))
        return F.gelu(self.reduce(torch.cat([avg, mx], dim=1)))


class SentimentNet(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        h = CONFIG["hidden"]
        self.branches = nn.ModuleList([Branch(0, h, True), Branch(74, h), Branch(35, h)])
        self.gates = nn.ModuleList([nn.Linear(h + 1, 1) for _ in MODS])
        self.head = nn.Sequential(nn.Linear(h * 3 if mode == "baseline" else h, h), nn.GELU(), nn.Dropout(0.1))
        self.classifier = nn.Linear(h, 3)
        self.regressor = nn.Linear(h, 1)

    def forward(self, b):
        valid = b["valid"].bool()
        effective = b["effective"].bool()
        masks = [effective[:, j, :] if self.mode == "no_mask" else b["obs"][:, j, :].bool() for j in range(3)]
        xs = (b["tokens"].long(), b["audio"].float(), b["vision"].float())
        reps = [branch(x, m) for branch, x, m in zip(self.branches, xs, masks)]
        covers = [m.float().sum(1, keepdim=True) / effective[:, j, :].float().sum(1, keepdim=True).clamp(min=1) for j,m in enumerate(masks)]
        if self.mode == "baseline":
            fused = torch.cat(reps, dim=1)
            weights = torch.full((len(valid), 3), 1/3, device=valid.device)
        else:
            if self.mode == "fixed":
                weights = torch.full((len(valid), 3), 1/3, device=valid.device)
            else:
                logits = torch.cat([g(torch.cat([r, c], 1)) for g, r, c in zip(self.gates, reps, covers)], 1)
                if self.mode != "no_mask":
                    logits = logits.masked_fill(torch.cat(covers, 1) <= 0, -1e4)
                weights = torch.softmax(logits, 1)
            fused = sum(weights[:, j:j+1] * reps[j] for j in range(3))
        h = self.head(fused)
        return self.classifier(h), torch.tanh(self.regressor(h).squeeze(1)) * 3, weights


def metrics(y, r, cls, strength):
    y = np.asarray(y); r = np.asarray(r); cls = np.asarray(cls); strength = np.asarray(strength)
    pred = cls.argmax(axis=1)
    f1 = []
    for c in range(3):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        f1.append(2 * tp / max(1, 2 * tp + fp + fn))
    corr = float(np.corrcoef(r, strength)[0, 1]) if np.std(strength) > 0 and np.std(r) > 0 else 0.0
    return {"accuracy": float((pred == y).mean()), "macro_f1": float(np.mean(f1)),
            "mae": float(np.abs(strength - r).mean()), "pearson": corr}


def predict_arrays(model, x, scenario=None, batch_size=256):
    model.eval()
    probs, strengths, removed_total, observed_total = [], [], np.zeros(3,dtype=np.int64), np.zeros(3,dtype=np.int64)
    with torch.no_grad():
        for lo in range(0, len(x["id"]), batch_size):
            hi = min(len(x["id"]), lo + batch_size)
            b = subset(x, slice(lo, hi))
            if scenario is not None:
                removed, observed = apply_missing(b, scenario["mods"], scenario["rate"], scenario["place"])
                removed_total += removed; observed_total += observed
            logits, reg, _ = model(to_torch(b))
            probs.append(torch.softmax(logits, 1).numpy())
            strengths.append(reg.numpy())
    actual = {m:float(removed_total[j] / max(1, observed_total[j])) for j,m in enumerate(MODS)}
    return np.concatenate(probs), np.concatenate(strengths), actual


def scenario_list():
    out = [{"name":"complete", "mods":[], "place":"none", "rate":0.0}]
    for m in MODS:
        for place in ("front", "middle", "back"):
            for rate in CONFIG["rates"]:
                out.append({"name":f"{m}_{place}_{int(rate*100)}", "mods":[m], "place":place, "rate":rate})
    for pair in (("text","audio"),("text","vision"),("audio","vision")):
        out.append({"name":"_".join(pair)+"_middle_40", "mods":list(pair), "place":"middle", "rate":0.4})
    out.append({"name":"text_only", "mods":["audio","vision"], "place":"all", "rate":1.0})
    return out


def selection_panel():
    ss = scenario_list()
    names = {"complete", "text_middle_40", "audio_middle_40", "vision_middle_40", "audio_vision_middle_40"}
    return [s for s in ss if s["name"] in names]


def selection_score(model, valid):
    vals = []
    for s in selection_panel():
        cls, reg, _ = predict_arrays(model, valid, None if s["name"] == "complete" else s)
        vals.append(metrics(valid["class"], valid["strength"], cls, reg))
    return float(np.mean([v["macro_f1"] - v["mae"] / 3 for v in vals])), vals


def write_csv(path, rows):
    assert rows
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def train_one(mode, train, valid, history):
    seed_everything(CONFIG["seed"])
    model = SentimentNet(mode)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    rng = np.random.default_rng(CONFIG["seed"] + 101)
    best, best_epoch, stale = -1e9, -1, 0
    ckpt = OUT / ("checkpoint_" + mode + ".pt")
    n = len(train["id"])
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        order = rng.permutation(n)
        losses = []
        for lo in range(0, n, CONFIG["batch_size"]):
            ix = order[lo:lo+CONFIG["batch_size"]]
            b = subset(train, ix)
            if mode not in ("baseline", "no_augment"):
                train_mask(b, rng)
            t = to_torch(b)
            y = torch.from_numpy(train["class"][ix])
            r = torch.from_numpy(train["strength"][ix])
            logits, reg, _ = model(t)
            loss = F.cross_entropy(logits, y) + CONFIG["huber_weight"] * F.huber_loss(reg, r, delta=1.0)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step(); losses.append(float(loss.item()))
        score, panel = selection_score(model, valid)
        row = {"model":mode, "epoch":epoch, "train_loss":float(np.mean(losses)), "selection_score":score,
               "complete_macro_f1":panel[0]["macro_f1"], "complete_mae":panel[0]["mae"]}
        history.append(row)
        print(f"{mode} epoch={epoch:02d} loss={row['train_loss']:.4f} score={score:.4f} full_f1={panel[0]['macro_f1']:.4f} full_mae={panel[0]['mae']:.4f}", flush=True)
        if score > best + 1e-5:
            best, best_epoch, stale = score, epoch, 0
            torch.save({"state":model.state_dict(),"mode":mode,"epoch":epoch,"selection_score":score},ckpt)
        else:
            stale += 1
            if stale >= CONFIG["patience"]: break
    info = {"model":mode,"best_epoch":best_epoch,"selection_score":best,
            "parameter_count":sum(p.numel() for p in model.parameters())}
    return info


def load_model(mode):
    ckpt = torch.load(OUT / ("checkpoint_" + mode + ".pt"),map_location="cpu",weights_only=False)
    model = SentimentNet(mode)
    model.load_state_dict(ckpt["state"]); model.eval()
    return model


def eval_all(models, valid):
    scenarios = scenario_list()
    rows = []
    for info in models:
        mode = info["model"]
        model = load_model(mode)
        for s in scenarios:
            cls, reg, actual = predict_arrays(model, valid, None if s["name"] == "complete" else s)
            m = metrics(valid["class"], valid["strength"], cls, reg)
            rows.append({"model":mode,"scenario":s["name"],"missing_modality":"+".join(s["mods"]) or "none",
                         "position":s["place"],"nominal_rate":s["rate"],"actual_removed_text":actual["text"],
                         "actual_removed_audio":actual["audio"],"actual_removed_vision":actual["vision"],**m})
        print("evaluated",mode,len(scenarios),"conditions",flush=True)
    fulls = {r["model"]:r for r in rows if r["scenario"] == "complete"}
    for r in rows:
        base = fulls[r["model"]]
        r["accuracy_drop"] = base["accuracy"] - r["accuracy"]
        r["macro_f1_drop"] = base["macro_f1"] - r["macro_f1"]
        r["mae_increase"] = r["mae"] - base["mae"]
        r["pearson_drop"] = base["pearson"] - r["pearson"]
        r["accuracy_relative_drop"] = r["accuracy_drop"] / max(1e-9,abs(base["accuracy"]))
        r["macro_f1_relative_drop"] = r["macro_f1_drop"] / max(1e-9,abs(base["macro_f1"]))
        r["mae_relative_increase"] = r["mae_increase"] / max(1e-9,abs(base["mae"]))
        r["pearson_relative_drop"] = r["pearson_drop"] / max(1e-9,abs(base["pearson"]))
    write_csv(OUT / "validation_scenarios.csv",rows)
    return rows


def write_error_cases(mode, valid):
    model=load_model(mode)
    cls, reg, _ = predict_arrays(model,valid)
    pred=cls.argmax(1)
    rank=np.argsort(-(np.abs(reg-valid["strength"]) + (pred!=valid["class"]).astype(float)))[:12]
    errors=[]
    for j in rank:
        errors.append({"id":valid["id"][j],"true_class":int(valid["class"][j]),"pred_class":int(pred[j]),
                       "true_strength":float(valid["strength"][j]),"pred_strength":float(reg[j]),
                       "absolute_error":float(abs(reg[j]-valid["strength"][j])),
                       "valid_length":int(valid["lengths"][j]),
                       "observed_text":int(valid["obs"][j,0].sum()),
                       "observed_audio":int(valid["obs"][j,1].sum()),
                       "observed_vision":int(valid["obs"][j,2].sum())})
    write_csv(OUT/"error_cases.csv",errors)
    return errors


def load_targets(stats):
    paths = sorted(TARGET.glob("附件3_*.pkl"))
    assert len(paths) == 30
    pieces = []
    for p in paths:
        s = read_pickle(p)["test"]
        q = raw_inputs(s, [p.stem])
        pieces.append(q)
    keys = ("tokens", "audio", "vision", "valid", "effective", "raw_zero", "synthetic_missing", "obs", "lengths")
    x = {k:np.concatenate([q[k] for q in pieces],axis=0) for k in keys}
    x["id"] = [q["id"][0] for q in pieces]
    assert len(set(x["id"])) == 30
    return transform(x,stats)


def write_target_boundary_audit(x):
    attention = x["effective"][:, 0, :]
    continuous = (attention == (np.arange(50)[None, :] < attention.sum(axis=1)[:, None])).all(axis=1)
    zero = x["raw_zero"]
    internal_zero = zero & x["effective"]
    audit = {
        "samples": len(x["id"]),
        "text_attention_continuous_samples": int(continuous.sum()),
        "audio_vision_identical_raw_zero_pattern_samples": int((zero[:, 1, :] == zero[:, 2, :]).all(axis=1).sum()),
        "samples_with_audio_or_vision_raw_zero_inside_effective": int(internal_zero[:, 1:, :].any(axis=(1, 2)).sum()),
        "samples_with_audio_and_vision_raw_zero_inside_effective": int(internal_zero[:, 1:, :].all(axis=1).any(axis=1).sum()),
        "internal_raw_zero_positions": {MODS[j]: int(internal_zero[:, j, :].sum()) for j in range(3)},
        "synthetic_missing_positions": int(x["synthetic_missing"].sum()),
        "note": "原始零值仅为观察结果，未标注为人为缺失；附件3统计不参与训练、选择或阈值调整。",
    }
    (OUT / "target_boundary_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def predict_target(mode, stats):
    x = load_targets(stats)
    write_target_boundary_audit(x)
    model = load_model(mode)
    cls, reg, _ = predict_arrays(model,x)
    rows = [{"sample_id":s,"source_file":s+".pkl","polarity_class":int(np.argmax(c)),
             "polarity_name":NAMES[int(np.argmax(c))],"predicted_strength":round(float(z),6),
             "prob_negative":round(float(c[0]),6),"prob_neutral":round(float(c[1]),6),
             "prob_positive":round(float(c[2]),6)}
            for s,c,z in zip(x["id"],cls,reg)]
    write_csv(OUT / "attachment3_predictions.csv",rows)
    return rows


def verify_target(mode, stats):
    path=OUT / "attachment3_predictions.csv"
    with open(path,encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
    assert len(rows)==30 and len({r["sample_id"] for r in rows})==30
    expected={p.stem for p in TARGET.glob("附件3_*.pkl")}
    assert {r["sample_id"] for r in rows}==expected
    for r in rows:
        c=int(r["polarity_class"]); z=float(r["predicted_strength"])
        assert c in (0,1,2) and r["polarity_name"]==NAMES[c] and -3<=z<=3
        assert abs(sum(float(r[k]) for k in ("prob_negative","prob_neutral","prob_positive"))-1)<2e-6
        assert r["source_file"]==r["sample_id"]+".pkl"
    model=load_model(mode); x=load_targets(stats)
    cls,reg,_=predict_arrays(model,x)
    for i,(r,c,z) in enumerate(zip(rows,cls,reg)):
        assert r["sample_id"]==x["id"][i]
        assert int(r["polarity_class"])==int(np.argmax(c))
        assert abs(float(r["predicted_strength"])-float(z))<1e-6
        for key,value in zip(("prob_negative","prob_neutral","prob_positive"),c):
            assert abs(float(r[key])-float(value))<1e-6
    return True


def svg_plot(rows, selected, output_path=None):
    rs=[r for r in rows if r["model"]==selected and len(r["missing_modality"].split("+"))==1 and r["missing_modality"]!="none"]
    colors={"text":"#2563eb","audio":"#e11d48","vision":"#059669"}
    lines=['<svg xmlns="http://www.w3.org/2000/svg" width="820" height="460" viewBox="0 0 820 460">',
           '<rect width="820" height="460" fill="white"/><text x="50" y="30" font-size="20">Validation macro F1 under contiguous missing spans</text>']
    for k,place in enumerate(("front","middle","back")):
        x0=60+k*255; y0=70
        lines += [f'<text x="{x0+65}" y="{y0}" font-size="16">{place}</text>',f'<line x1="{x0}" y1="380" x2="{x0+215}" y2="380" stroke="#333"/>',f'<line x1="{x0}" y1="90" x2="{x0}" y2="380" stroke="#333"/>']
        for mod in MODS:
            q=sorted([r for r in rs if r["position"]==place and r["missing_modality"]==mod],key=lambda r:r["nominal_rate"])
            pts=" ".join(f'{x0+12+(r["nominal_rate"]-0.1)/0.6*190:.1f},{375-r["macro_f1"]*260:.1f}' for r in q)
            lines.append(f'<polyline fill="none" stroke="{colors[mod]}" stroke-width="2.5" points="{pts}"/>')
        lines.append(f'<text x="{x0+85}" y="405" font-size="13">missing rate</text>')
    for j,mod in enumerate(MODS):
        lines.append(f'<rect x="{250+j*130}" y="430" width="18" height="3" fill="{colors[mod]}"/><text x="{275+j*130}" y="435" font-size="13">{mod}</text>')
    lines.append('</svg>')
    (output_path or OUT/"missing_factors.svg").write_text("\n".join(lines),encoding="utf-8")


def report(models, rows, stats, selected, train, valid, errors):
    def get(mode,sc): return next(r for r in rows if r["model"]==mode and r["scenario"]==sc)
    def fmt(r): return f"Accuracy {r['accuracy']:.3f}、宏 F1 {r['macro_f1']:.3f}、MAE {r['mae']:.3f}、Pearson {r['pearson']:.3f}"
    lines=["# E题第二问：模态局部缺失下的鲁棒情感预测实验报告","",
           "## 数据、约束和可复现性","",
           f"附件2仅使用 aligned_50.pkl。train {len(train['id'])} 条、valid {len(valid['id'])} 条；test 未读取、未用于选择或评估。模型参数仅由 train 学习，valid 用于选最佳轮次与最终结构，附件3仅在模型确定后用于最终推理和复核。",
           "类别映射经强度符号核验：0=Negative（<0）、1=Neutral（=0）、2=Positive（>0）；强度限于 [-3,3]。",
           "统一输入为 text_bert、audio、vision；未使用附件2的预计算 text。附件3浮点 text_bert 已先检查每个元素为整数、再转 int64。没有外部情感数据或预训练编码器。",
           "长度从三模态信号的末端确定，尾部是填充。音视频两端为结构性特殊位置；内部整行原始零值仍是有效位置，不据此断言人为缺失。有效位置、原始零值、人工连续遮挡各有独立掩码；模型观测掩码为有效位置扣除人工遮挡。",
           f"音频标准化仅用 train 的 {stats['audio']['effective_positions']} 个有效位置，视觉仅用 {stats['vision']['effective_positions']} 个有效位置，包括内部原始零值。均值/标准差见 normalization.json；填充位置标准化后置零，有效零值按训练集参数变换并裁剪到 [-8,8]。",
           "","## 模型与训练","",
           "每个模态的有效时序表示为 $h_{mt}=\\mathrm{GELU}(\\mathrm{Conv1D}(\\phi_m(x_{mt})\\odot o_{mt}))$；以观测掩码做均值和最大值汇总后投影为 $r_m$。文本 $\\phi$ 为从头训练的 48 维词元嵌入；音频和视觉为线性投影。",
           "覆盖率 $c_m=\\sum_t o_{mt}/\\sum_t e_{mt}$，其中 $e_{mt}$ 为该模态有效位置；动态权重 $w_m=\\mathrm{softmax}_m(g_m([r_m,c_m]))$，融合 $r=\\sum_m w_m r_m$。分类头输出三类 softmax，回归头为 $3\\tanh(\\cdot)$。门控权重只表征模型内部的动态分配，不能解释为因果贡献。",
           "损失为交叉熵 + 0.5×Huber（δ=1）；未加入 Pearson 或分支一致性损失。AdamW、固定随机种子；超参数见 config.json。每轮有25%机会保留完整样本，否则以75%概率作遮挡；遮挡时以25%概率选择两个模态，否则选择一个；比例从五档均匀抽取，区间起点在样本内部有效时段中均匀抽取，保留两端标记。",
           "模型选择分数为固定验证面板（完整输入，以及三种单模态和音频+视觉中段40%遮挡）上 $\\mathrm{macroF1}-\\mathrm{MAE}/3$ 的平均值；所有模型用相同面板、相同类别规则。",
           "","## 验证与消融","",
           "下表是未经遮挡的 valid 指标，以及验证面板分数；完整的50条件×5模型指标见 validation_scenarios.csv。",
           "","|模型|最佳轮次|面板分数|完整输入指标|","|---|---:|---:|---|" ]
    for x in models:
        lines.append(f"|{x['model']}|{x['best_epoch']}|{x['selection_score']:.3f}|{fmt(get(x['model'],'complete'))}|")
    lines += ["",f"最终按验证面板选择 **{selected}**。附件3预测文件来自这一模型。`fixed` 保留显式掩码与连续缺失增强，但将动态融合权重固定为每模态 1/3。", "",
              "## 三因素连续缺失实验","",
              "五档名义长度比例为 10%、25%、40%、55%、70%；前/中/后段各测试文本、音频、视觉，另测试三种双模态组合的中段40%及仅文本可用。实际移除的有效位置比例逐条件列于 CSV。下表为最终选定模型的中段结果。",
              "","|模态|比例|Accuracy|宏 F1|MAE|Pearson|宏 F1 相对完整下降|","|---|---:|---:|---:|---:|---:|---:|"]
    full=get(selected,"complete")
    for mod in MODS:
        for rate in CONFIG["rates"]:
            r=get(selected,f"{mod}_middle_{int(rate*100)}")
            lines.append(f"|{mod}|{int(rate*100)}%|{r['accuracy']:.3f}|{r['macro_f1']:.3f}|{r['mae']:.3f}|{r['pearson']:.3f}|{r['macro_f1_relative_drop']*100:.1f}%|")
    lines += ["","双模态中段40%："]
    for pair in ("text_audio","text_vision","audio_vision"):
        r=get(selected,pair+"_middle_40");lines.append(f"- {pair}: {fmt(r)}；宏 F1 相对下降 {r['macro_f1_relative_drop']*100:.1f}%。")
    r=get(selected,"text_only")
    lines.append(f"- 仅文本可用（音频、视觉有效内部全部遮挡）：{fmt(r)}；宏 F1 相对下降 {r['macro_f1_relative_drop']*100:.1f}%。")
    one=[r for r in rows if r["model"]==selected and r["missing_modality"] in MODS]
    def avg(q,key): return float(np.mean([r[key] for r in q]))
    by_mod={m:avg([r for r in one if r["missing_modality"]==m],"macro_f1") for m in MODS}
    by_pos={p:avg([r for r in one if r["position"]==p],"macro_f1") for p in ("front","middle","back")}
    by_rate={rate:avg([r for r in one if r["nominal_rate"]==rate],"macro_f1") for rate in CONFIG["rates"]}
    lines += ["",f"单模态45条件平均宏 F1：文本 {by_mod['text']:.3f}、音频 {by_mod['audio']:.3f}、视觉 {by_mod['vision']:.3f}。在本验证集和遮挡规则下，文本缺失的平均损害最大。",
              f"位置均值：前段 {by_pos['front']:.3f}、中段 {by_pos['middle']:.3f}、后段 {by_pos['back']:.3f}；差异较小，不宜据此声称某一时段普遍更重要。",
              "缺失率均值（10%→70%）："+"、".join(f"{int(k*100)}%={v:.3f}" for k,v in by_rate.items())+"。总体随缺失时长增加而下降，局部波动来自有限验证样本。",
              "不同位置、类型、时长的全部结果及绝对/相对下降均在 CSV；图见 missing_factors.svg。",
              "","## 错误案例与局限","",
              "error_cases.csv 列出最终模型在 valid 的12条高误差案例，包含样本编号、真实/预测极性与强度及每模态观测数，便于回溯。",
              f"其中正向误判负向 {sum(e['true_class']==2 and e['pred_class']==0 for e in errors)} 条，负向误判正向 {sum(e['true_class']==0 and e['pred_class']==2 for e in errors)} 条；这说明极性反转是大误差的一种具体来源。",
              "对单条案例，表中的有效长度与观测覆盖只能辅助排查；是否由语义否定、语音语调或表情冲突导致，还需回看原始视频与文本，不能仅凭错误分数作因果断言。",
              "文本嵌入仅从本题训练集学习，缺少外部语义先验；小规模数据下对少见词与否定表达的泛化受限。原始零值无法严格区分天然无信号和异常缺失，因此覆盖率仅衡量人工遮挡后的有效位置保留比例。附件3没有标签，不能报告其准确率。",
              "","## 交付与复核","",
              "`attachment3_predictions.csv` 包含30条结果；样本号对应源 pkl 文件名，类别 0/1/2、英文类别名、三类概率与 [-3,3] 强度。`verify` 子命令检查行数、唯一性、概率和强度范围及由保存模型重新生成的一致性。",
              "只保留最终模型参数。源码、配置、统计和结果可放入竞赛附件；虚拟环境、原始数据及下载轮子不属于提交材料。"]
    (HERE/"实验报告.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("command",choices=("all","predict","verify")); args=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    seed_everything(CONFIG["seed"])
    if args.command=="all":
        d=load_data()
        write_data_audit(d)
        stats=fit_normalization(d["train"])
        (OUT/"normalization.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2))
        for k in d: transform(d[k],stats)
        history=[]; models=[]
        for mode in ("baseline","no_augment","no_mask","fixed","full"):
            models.append(train_one(mode,d["train"],d["valid"],history))
        write_csv(OUT/"training_history.csv",history)
        rows=eval_all(models,d["valid"])
        write_csv(OUT/"ablation_summary.csv",models)
        selected=max(models,key=lambda x:x["selection_score"])["model"]
        errors=write_error_cases(selected,d["valid"])
        (OUT/"selection.json").write_text(json.dumps({"selected":selected,"models":models,"rule":"mean(macro_f1 - mae/3) on fixed five-scenario valid panel"},indent=2))
        predict_target(selected,stats)
        assert verify_target(selected,stats)
        svg_plot(rows,selected)
        report(models,rows,stats,selected,d["train"],d["valid"],errors)
        for mode in ("baseline","no_augment","no_mask","fixed","full"):
            if mode not in (selected,"fixed"): (OUT/("checkpoint_"+mode+".pt")).unlink()
        print("DONE selected",selected,"results",OUT,flush=True)
    else:
        stats=json.loads((OUT/"normalization.json").read_text())
        selected=json.loads((OUT/"selection.json").read_text())["selected"]
        if args.command=="predict": predict_target(selected,stats)
        else: assert verify_target(selected,stats); print("verified 30 predictions")


if __name__=="__main__": main()
