#!/usr/bin/env python3
"""用 Google BERT-Tiny 增强统一三字段输入，完整重训并在固定缺失面板选择。"""
import argparse
import csv
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import problem2 as p2

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "optimization"
CONFIG = json.loads((HERE / "optimize_config.json").read_text())
SHA256 = "7fb69ad9f6866d8983183c930e33828f326470bf6ad8bbb2ad4ed957a92e9414"
SOURCE_URL = "https://huggingface.co/google/bert_uncased_L-2_H-128_A-2"


def load_safetensors(path):
    raw = Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA256, "预训练权重校验失败"
    length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8+length])
    base = 8 + length
    tensors = {}
    for key, item in header.items():
        if key == "__metadata__": continue
        assert item["dtype"] == "F32"
        start, end = item["data_offsets"]
        array = np.frombuffer(raw[base+start:base+end], dtype="<f4").reshape(item["shape"]).copy()
        tensors[key] = torch.from_numpy(array)
    return tensors


class BertEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(30522,128,padding_idx=0)
        self.position_embeddings = nn.Embedding(512,128)
        self.token_type_embeddings = nn.Embedding(2,128)
        self.LayerNorm = nn.LayerNorm(128,eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, ids):
        b,t=ids.shape
        pos=torch.arange(t,device=ids.device).unsqueeze(0).expand(b,t)
        types=torch.zeros_like(ids)
        x=self.word_embeddings(ids)+self.position_embeddings(pos)+self.token_type_embeddings(types)
        return self.dropout(self.LayerNorm(x))


class BertSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query=nn.Linear(128,128)
        self.key=nn.Linear(128,128)
        self.value=nn.Linear(128,128)
        self.dropout=nn.Dropout(0.1)

    def forward(self,x,mask):
        b,t,h=x.shape
        def heads(z): return z.reshape(b,t,2,64).transpose(1,2)
        q,k,v=heads(self.query(x)),heads(self.key(x)),heads(self.value(x))
        attention=(q @ k.transpose(-2,-1))/8.0
        attention=attention.masked_fill(~mask[:,None,None,:],-1e4)
        attention=self.dropout(torch.softmax(attention,dim=-1))
        return (attention @ v).transpose(1,2).reshape(b,t,h)


class BertAttentionOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense=nn.Linear(128,128)
        self.LayerNorm=nn.LayerNorm(128,eps=1e-12)
        self.dropout=nn.Dropout(0.1)

    def forward(self,x,residual):
        return self.LayerNorm(self.dropout(self.dense(x))+residual)


class BertAttention(nn.Module):
    def __init__(self):
        super().__init__()
        setattr(self,"self",BertSelfAttention())
        self.output=BertAttentionOutput()

    def forward(self,x,mask):
        return self.output(getattr(self,"self")(x,mask),x)


class BertIntermediate(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense=nn.Linear(128,512)

    def forward(self,x): return F.gelu(self.dense(x))


class BertOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense=nn.Linear(512,128)
        self.LayerNorm=nn.LayerNorm(128,eps=1e-12)
        self.dropout=nn.Dropout(0.1)

    def forward(self,x,residual):
        return self.LayerNorm(self.dropout(self.dense(x))+residual)


class BertLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention=BertAttention()
        self.intermediate=BertIntermediate()
        self.output=BertOutput()

    def forward(self,x,mask):
        x=self.attention(x,mask)
        return self.output(self.intermediate(x),x)


class BertEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=nn.ModuleList([BertLayer(),BertLayer()])

    def forward(self,x,mask):
        for layer in self.layer: x=layer(x,mask)
        return x


class BertPooler(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense=nn.Linear(128,128)

    def forward(self,x): return torch.tanh(self.dense(x[:,0]))


class TinyBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings=BertEmbeddings()
        self.encoder=BertEncoder()
        self.pooler=BertPooler()

    def forward(self,ids,mask):
        x=self.encoder(self.embeddings(ids),mask)
        mean=(x*mask.unsqueeze(-1)).sum(1)/mask.sum(1).clamp(min=1).unsqueeze(-1)
        return torch.cat([self.pooler(x),mean],dim=1)


class OptimizedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert=TinyBert()
        self.text_reduce=nn.Sequential(nn.Linear(256,64),nn.GELU())
        self.audio=p2.Branch(74,64)
        self.vision=p2.Branch(35,64)
        self.gates=nn.ModuleList([nn.Linear(65,1) for _ in range(3)])
        self.head=nn.Sequential(nn.Linear(259,128),nn.GELU(),nn.Dropout(0.2),nn.Linear(128,64),nn.GELU())
        self.classifier=nn.Linear(64,3)
        self.regressor=nn.Linear(64,1)

    def forward(self,b):
        valid=b["valid"].bool()
        effective=b["effective"].bool()
        masks=[b["obs"][:,j,:].bool() for j in range(3)]
        text=self.text_reduce(self.bert(b["tokens"].long(),masks[0]))
        audio=self.audio(b["audio"].float(),masks[1])
        vision=self.vision(b["vision"].float(),masks[2])
        reps=[text,audio,vision]
        coverage=torch.stack([m.float().sum(1)/effective[:,j,:].float().sum(1).clamp(min=1) for j,m in enumerate(masks)],dim=1)
        logits=torch.cat([g(torch.cat([rep,coverage[:,j:j+1]],dim=1)) for j,(g,rep) in enumerate(zip(self.gates,reps))],dim=1)
        logits=logits.masked_fill(coverage<=0,-1e4)
        weights=torch.softmax(logits,dim=1)
        fused=sum(weights[:,j:j+1]*reps[j] for j in range(3))
        h=self.head(torch.cat([fused,*reps,coverage],dim=1))
        return self.classifier(h),3*torch.tanh(self.regressor(h).squeeze(1)),weights

    def initialize_from_pretrained(self,path):
        pretrained=load_safetensors(path)
        own=self.state_dict()
        matched={k:v for k,v in pretrained.items() if k in own and tuple(v.shape)==tuple(own[k].shape)}
        assert len(matched)==39,(len(matched),len(pretrained))
        own.update(matched)
        self.load_state_dict(own)
        return len(matched)


def model_metrics(model,x,scenario=None):
    cls,reg,_=p2.predict_arrays(model,x,scenario,batch_size=CONFIG["eval_batch_size"])
    return p2.metrics(x["class"],x["strength"],cls,reg)


def score_panel(model,valid):
    values=[]
    for s in p2.selection_panel():
        values.append(model_metrics(model,valid,None if s["name"]=="complete" else s))
    score=float(np.mean([x["macro_f1"]-x["mae"]/3 for x in values]))
    return score,values


def train_variant(name,train,valid,weights_path,history):
    p2.seed_everything(CONFIG["seed"])
    rng=np.random.default_rng(CONFIG["seed"]+101)
    model=OptimizedNet()
    count=model.initialize_from_pretrained(weights_path)
    frozen=name=="tiny_frozen"
    weighted=name=="tiny_weighted"
    if frozen:
        for p in model.bert.parameters(): p.requires_grad=False
    groups=[{"params":[p for n,p in model.named_parameters() if not n.startswith("bert.")],"lr":CONFIG["head_lr"]}]
    if not frozen:
        groups.append({"params":model.bert.parameters(),"lr":CONFIG["bert_lr"]})
    opt=torch.optim.AdamW(groups,weight_decay=CONFIG["weight_decay"])
    class_weight=None
    if weighted:
        counts=np.bincount(train["class"],minlength=3).astype(float)
        w=1/np.sqrt(counts);w=w/w.mean()
        class_weight=torch.tensor(w,dtype=torch.float32)
    best=-1e9;best_epoch=0;stale=0
    ckpt=OUT/(name+".pt")
    n=len(train["id"])
    for epoch in range(1,CONFIG["epochs"]+1):
        model.train();order=rng.permutation(n);losses=[]
        for lo in range(0,n,CONFIG["batch_size"]):
            ix=order[lo:lo+CONFIG["batch_size"]]
            b=p2.subset(train,ix)
            p2.train_mask(b,rng)
            t=p2.to_torch(b)
            y=torch.from_numpy(train["class"][ix]);r=torch.from_numpy(train["strength"][ix])
            logits,reg,_=model(t)
            loss=F.cross_entropy(logits,y,weight=class_weight)+CONFIG["huber_weight"]*F.huber_loss(reg,r,delta=1.0)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.0);opt.step()
            losses.append(float(loss.item()))
        score,panel=score_panel(model,valid)
        row={"variant":name,"epoch":epoch,"train_loss":float(np.mean(losses)),"selection_score":score,
             "complete_accuracy":panel[0]["accuracy"],"complete_macro_f1":panel[0]["macro_f1"],"complete_mae":panel[0]["mae"]}
        history.append(row)
        print(name,"epoch",epoch,"loss",round(row["train_loss"],4),"score",round(score,4),"accuracy",round(panel[0]["accuracy"],4),"f1",round(panel[0]["macro_f1"],4),flush=True)
        if score>best+1e-5:
            best=score;best_epoch=epoch;stale=0
            torch.save({"state":model.state_dict(),"variant":name,"epoch":epoch,"score":score},ckpt)
        else:
            stale+=1
            if stale>=CONFIG["patience"]:break
    return {"variant":name,"best_epoch":best_epoch,"selection_score":best,"loaded_pretrained_tensors":count,
            "parameters":sum(p.numel() for p in model.parameters())}


def load_model(path):
    ckpt=torch.load(path,map_location="cpu",weights_only=False)
    model=OptimizedNet();model.load_state_dict(ckpt["state"]);model.eval()
    return model


def evaluate(model,valid):
    rows=[]
    for s in p2.scenario_list():
        cls,reg,actual=p2.predict_arrays(model,valid,None if s["name"]=="complete" else s,batch_size=CONFIG["eval_batch_size"])
        m=p2.metrics(valid["class"],valid["strength"],cls,reg)
        rows.append({"scenario":s["name"],"missing_modality":"+".join(s["mods"]) or "none","position":s["place"],
                     "nominal_rate":s["rate"],"actual_removed_text":actual["text"],"actual_removed_audio":actual["audio"],
                     "actual_removed_vision":actual["vision"],**m})
    base=rows[0]
    for r in rows:
        for key in ("accuracy","macro_f1","pearson"):
            r[key+"_relative_drop"]=(base[key]-r[key])/max(1e-9,abs(base[key]))
        r["mae_relative_increase"]=(r["mae"]-base["mae"])/max(1e-9,base["mae"])
    return rows


def save_predictions(model,stats):
    x=p2.load_targets(stats)
    p2.write_target_boundary_audit(x)
    cls,reg,_=p2.predict_arrays(model,x,batch_size=CONFIG["eval_batch_size"])
    rows=[]
    for name,c,z in zip(x["id"],cls,reg):
        k=int(c.argmax())
        rows.append({"sample_id":name,"source_file":name+".pkl","polarity_class":k,"polarity_name":p2.NAMES[k],
                     "predicted_strength":round(float(z),6),"prob_negative":round(float(c[0]),6),
                     "prob_neutral":round(float(c[1]),6),"prob_positive":round(float(c[2]),6)})
    p2.write_csv(OUT/"attachment3_predictions_optimized.csv",rows)
    return rows


def verify(model,stats):
    with (OUT/"attachment3_predictions_optimized.csv").open(encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
    assert len(rows)==30 and len({r["sample_id"] for r in rows})==30
    assert {r["sample_id"] for r in rows}=={p.stem for p in p2.TARGET.glob("附件3_*.pkl")}
    x=p2.load_targets(stats)
    cls,reg,_=p2.predict_arrays(model,x,batch_size=CONFIG["eval_batch_size"])
    for i,(row,c,z) in enumerate(zip(rows,cls,reg)):
        assert row["sample_id"]==x["id"][i] and row["source_file"]==row["sample_id"]+".pkl"
        assert int(row["polarity_class"])==int(c.argmax()) and row["polarity_name"]==p2.NAMES[int(c.argmax())]
        assert -3<=float(row["predicted_strength"])<=3 and abs(float(row["predicted_strength"])-float(z))<1e-6
        assert abs(sum(float(row[k]) for k in ("prob_negative","prob_neutral","prob_positive"))-1)<2e-6
        for key,value in zip(("prob_negative","prob_neutral","prob_positive"),c):
            assert abs(float(row[key])-float(value))<1e-6
    return True


def class_f1(y,pred):
    out=[]
    for c in range(3):
        tp=int(((y==c)&(pred==c)).sum())
        fp=int(((y!=c)&(pred==c)).sum())
        fn=int(((y==c)&(pred!=c)).sum())
        out.append(2*tp/max(1,2*tp+fp+fn))
    return out


def finalize(model,data,stats):
    valid=data["valid"]
    rows=evaluate(model,valid)
    p2.write_csv(OUT/"validation_scenarios.csv",rows)
    p2.svg_plot([{"model":"optimized",**r} for r in rows],"optimized",OUT/"missing_factors.svg")
    old=p2.load_model("fixed")
    prior=[]
    for s in p2.scenario_list():
        c,r,_=p2.predict_arrays(old,valid,None if s["name"]=="complete" else s)
        prior.append({"scenario":s["name"],**p2.metrics(valid["class"],valid["strength"],c,r)})
    comparison=[]
    for a,b in zip(prior,rows):
        comparison.append({"scenario":a["scenario"],
                           **{"old_"+k:a[k] for k in ("accuracy","macro_f1","mae","pearson")},
                           **{"optimized_"+k:b[k] for k in ("accuracy","macro_f1","mae","pearson")},
                           "accuracy_gain":b["accuracy"]-a["accuracy"],
                           "macro_f1_gain":b["macro_f1"]-a["macro_f1"],
                           "mae_reduction":a["mae"]-b["mae"]})
    p2.write_csv(OUT/"comparison_to_previous.csv",comparison)
    cls,reg,_=p2.predict_arrays(model,valid)
    pred=cls.argmax(1)
    old_cls,old_reg,_=p2.predict_arrays(old,valid)
    conf=np.zeros((3,3),dtype=int)
    for y,z in zip(valid["class"],pred):conf[int(y),int(z)]+=1
    rank=np.argsort(-(np.abs(reg-valid["strength"])+(pred!=valid["class"]).astype(float)))[:12]
    errors=[]
    for i in rank:
        errors.append({"id":valid["id"][i],"true_class":int(valid["class"][i]),"pred_class":int(pred[i]),
                       "true_strength":float(valid["strength"][i]),"pred_strength":float(reg[i]),
                       "absolute_error":float(abs(reg[i]-valid["strength"][i])),
                       "valid_length":int(valid["lengths"][i]),
                       "observed_text":int(valid["obs"][i,0].sum()),
                       "observed_audio":int(valid["obs"][i,1].sum()),
                       "observed_vision":int(valid["obs"][i,2].sum())})
    p2.write_csv(OUT/"error_cases.csv",errors)
    save_predictions(model,stats)
    assert verify(model,stats)
    sel=json.loads((OUT/"selection.json").read_text())
    chosen=sel["selected"]
    def fmt(x):return f"Accuracy {x['accuracy']:.4f}，宏 F1 {x['macro_f1']:.4f}，MAE {x['mae']:.4f}，Pearson {x['pearson']:.4f}"
    report=["# E题第二问优化实验报告","",
            "## 结论与验证边界","",
            f"优化模型为 {chosen['variant']}（最佳轮次 {chosen['best_epoch']}）。在 train 上训练，在 valid 上选择轮次、类别权重方案及最终模型；附件2 test 未读取，附件3只在选择结束后预测，无真实标签可用于评价。",
            f"完整输入的修正后普通固定权重模型：{fmt(prior[0])}。修正后优化模型：{fmt(rows[0])}。两者差值：Accuracy {comparison[0]['accuracy_gain']*100:.2f} 个百分点、宏 F1 {comparison[0]['macro_f1_gain']*100:.2f} 个百分点、MAE 降低 {comparison[0]['mae_reduction']:.4f}。修正前后指标另见《数据边界审计报告》。",
            "这些是同一个验证集上的选择后结果；未使用独立测试集确认泛化提升。", "",
            "## 输入、预训练来源和模型","",
            "仍只使用 aligned_50.pkl 的 text_bert、audio、vision 三字段，附件3对齐版同接口。文本特殊词元 [CLS]=101、[SEP]=102 与所选编码器词表兼容。附件3浮点词元经过整数性校验再转 int64。",
            f"文本编码器为 [Google BERT Tiny]({SOURCE_URL})：2层、隐藏维128、2个注意力头；通用语言预训练权重 SHA-256 `{SHA256}`，Apache-2.0，下载后校验并载入39个张量。未引入任何其他情感数据集。",
            "文本表示由 [CLS] 池化与有效词元均值构成；音频和视觉各用线性层及一维卷积编码，并按观测掩码池化。模型依据三模态表示和有效覆盖率产生样本级融合权重，同时将融合表示与各模态表示输入双任务预测头。门控权重不可视作因果解释。",
            "音视频标准化参数只取 train 的有效内部位置，包含原始零向量。有效位置、原始零值、人工连续遮挡分别记录；模型观测掩码是有效位置扣除人工遮挡。原始零值不能直接判定为人为缺失。覆盖率以各模态有效位置数为分母，未人工遮挡的原始零值保留在池化中。", "",
            "## 训练与选择","",
            "比较了冻结预训练编码器、全量微调、类别加权微调三方案。三方案在相同随机种子及连续区间增强下训练；损失为三分类交叉熵 + 0.5×Huber，类别加权版的交叉熵按 train 的类别频数平方根倒数加权。微调编码器学习率 3e-5，预测头学习率 1e-3，AdamW；其余参数见 optimize_config.json。",
            "每轮用同一固定验证面板（完整输入、三种单模态中段40%及音频+视觉中段40%）计算平均 `宏F1−MAE/3` 选择最佳轮次。三方案结果：", "",
            "|方案|最佳轮次|验证面板分数|","|---|---:|---:|"]
    for variant in sel["variants"]:
        report.append(f"|{variant['variant']}|{variant['best_epoch']}|{variant['selection_score']:.4f}|")
    report += ["","## 三因素缺失与消融对比","",
               "完整及45组单模态（3模态×3位置×5比例）、3组双模态局部缺失及仅文本可用，共50条件；优化模型和普通模型使用完全相同的验证遮挡。逐条件四指标、相对下降和与普通模型的差值见 CSV。", "",
               "|条件|Accuracy|宏 F1|MAE|Pearson|宏 F1 相对完整下降|","|---|---:|---:|---:|---:|---:|"]
    wanted=["complete","text_middle_70","audio_middle_70","vision_middle_70","text_audio_middle_40","text_vision_middle_40","audio_vision_middle_40","text_only"]
    for name in wanted:
        x=next(r for r in rows if r["scenario"]==name)
        report.append(f"|{name}|{x['accuracy']:.4f}|{x['macro_f1']:.4f}|{x['mae']:.4f}|{x['pearson']:.4f}|{x['macro_f1_relative_drop']*100:.1f}%|")
    single=[r for r in rows if r["missing_modality"] in p2.MODS]
    by_mod={m:float(np.mean([r["macro_f1"] for r in single if r["missing_modality"]==m])) for m in p2.MODS}
    by_pos={loc:float(np.mean([r["macro_f1"] for r in single if r["position"]==loc])) for loc in ("front","middle","back")}
    by_rate={rate:float(np.mean([r["macro_f1"] for r in single if r["nominal_rate"]==rate])) for rate in p2.CONFIG["rates"]}
    report += ["",f"单模态45条件的宏 F1 均值：文本 {by_mod['text']:.4f}、音频 {by_mod['audio']:.4f}、视觉 {by_mod['vision']:.4f}；文本缺失影响最大。",
               f"位置均值：前段 {by_pos['front']:.4f}、中段 {by_pos['middle']:.4f}、后段 {by_pos['back']:.4f}。",
               "缺失比例均值："+"、".join(f"{int(k*100)}%={v:.4f}" for k,v in by_rate.items())+"。",
               "具体条件存在小幅非单调变化，不能把少量验证样本上的波动解释成稳定的时间因果规律。", "",
               "## 错误分析与限制","",
               "优化模型的验证集混淆矩阵（行是真值，列是预测；顺序负/中/正）：", "",
               "|真值\\预测|负|中|正|","|---|---:|---:|---:|"]
    for i,line in enumerate(conf):report.append(f"|{p2.NAMES[i]}|{line[0]}|{line[1]}|{line[2]}|")
    f1=class_f1(valid["class"],pred)
    report += ["",f"各类 F1：负向 {f1[0]:.4f}、中性 {f1[1]:.4f}、正向 {f1[2]:.4f}。",
               "error_cases.csv 给出12条高误差案例的编号、真值/预测及观测覆盖，供原始素材回看。误差案例中的极性反转与强度偏差可核验，但仅凭预测数值无法确定语义、语音或视觉的因果来源。",
               "局限：验证集同时用于轮次和方案选择，指标有选择偏倚；缺少独立盲测标签；BERT Tiny 只提供通用语言预训练，不保证领域匹配。文本被大比例遮挡时仍明显退化。",
               "","## 可复现文件","",
               "`optimize.py`、`optimize_config.json`、保存的唯一优化模型参数、全部验证指标、比较表、图及30条预测 CSV 已保留。原始数据、预训练源文件与虚拟环境不在提交目录。执行 `python optimize.py verify` 可从模型重新计算并核验30条预测。"]
    (HERE/"优化报告.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    return rows[0]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("command",choices=["all","finalize","predict","verify"])
    ap.add_argument("--pretrained",default="/tmp/bert-tiny.safetensors")
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    p2.seed_everything(CONFIG["seed"])
    if args.command=="all":
        data=p2.load_data()
        stats=p2.fit_normalization(data["train"])
        for x in data.values():p2.transform(x,stats)
        history=[]; variants=[]
        for name in ("tiny_frozen","tiny_finetune","tiny_weighted"):
            variants.append(train_variant(name,data["train"],data["valid"],args.pretrained,history))
        p2.write_csv(OUT/"training_history.csv",history)
        p2.write_csv(OUT/"model_selection.csv",variants)
        selected=max(variants,key=lambda x:x["selection_score"])
        model=load_model(OUT/(selected["variant"]+".pt"))
        (OUT/"selection.json").write_text(json.dumps({"selected":selected,"variants":variants,"source":SOURCE_URL,"sha256":SHA256},indent=2))
        complete=finalize(model,data,stats)
        for v in variants:
            if v["variant"]!=selected["variant"]:(OUT/(v["variant"]+".pt")).unlink()
        print("DONE",selected,"complete",complete,flush=True)
    else:
        selected=json.loads((OUT/"selection.json").read_text())["selected"]["variant"]
        model=load_model(OUT/(selected+".pt"))
        stats=json.loads((p2.OUT/"normalization.json").read_text())
        if args.command=="finalize":
            data=p2.load_data()
            for x in data.values():p2.transform(x,stats)
            complete=finalize(model,data,stats)
            print("finalized",complete,flush=True)
        elif args.command=="predict":save_predictions(model,stats)
        else: assert verify(model,stats);print("verified 30 optimized predictions")


if __name__=="__main__": main()
