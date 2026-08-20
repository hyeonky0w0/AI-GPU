"""Residual MLP/DCN의 temporal outer 평가와 gated final 학습."""
from __future__ import annotations
import argparse, gc, hashlib, json, os, platform, time
from pathlib import Path
import joblib, numpy as np, pandas as pd, psutil, torch
from torch.utils.data import DataLoader,TensorDataset
from build_dataset import build_frame,build_test_frame
from contract import *
from evaluate import bootstrap_error_delta,metrics,residual_stats
from inspect_assets import inspect
from models import ResidualDCN,ResidualMLP,final_probability,make_preprocessor,numpy_probability

def write_json(path,value): path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8"); os.replace(tmp,path)
def seed_all(seed): np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
def row_digest(frame):
    h=hashlib.sha256()
    for value in frame[ID].astype(str): h.update(value.encode()); h.update(b"\0")
    return h.hexdigest()
def split_inner(frame,year):
    if year==2024: return frame[frame.fold.eq(2022)],frame[frame.fold.eq(2023)]
    source=frame[frame.fold.eq(2022)]; cut=int(len(source)*.8); return source.iloc[:cut],source.iloc[cut:]
def predict(model,x,device,batch):
    model.eval(); result=[]
    with torch.no_grad():
        for i in range(0,len(x),batch): result.append(model(torch.as_tensor(x[i:i+batch],dtype=torch.float32,device=device)).cpu().numpy())
    return np.concatenate(result)

def fit(train,pred,kind,seed,epochs,config,device,stage,ckpt,early=True):
    target=ckpt/stage/kind/f"seed_{seed}"; marker=target/"complete.json"; raw_path=target/"raw.npy"
    signature={"train_rows":len(train),"predict_rows":len(pred),"train_row_sha256":row_digest(train),"predict_row_sha256":row_digest(pred),"kind":kind,"seed":seed,"epochs":epochs,"learning_rate":config["learning_rate"],"weight_decay":config["weight_decay"],"train_lambda":config["train_lambda"]}
    if marker.is_file() and raw_path.is_file():
        m=json.loads(marker.read_text(encoding="utf-8")); raw=np.load(raw_path)
        if m.get("signature")==signature and raw.shape==(len(pred),) and np.isfinite(raw).all(): return raw,m["best_epoch"]
    target.mkdir(parents=True,exist_ok=True); pre=make_preprocessor(); x=pre.fit_transform(train[MODEL_FEATURES]).astype("float32"); xp=pre.transform(pred[MODEL_FEATURES]).astype("float32"); joblib.dump(pre,target/"preprocessor.joblib")
    seed_all(seed); cls=ResidualMLP if kind=="mlp" else ResidualDCN; model=cls(x.shape[1]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
    ds=TensorDataset(torch.from_numpy(x),torch.from_numpy(train.p_915.to_numpy("float32")),torch.from_numpy(train.target.to_numpy("float32")))
    loader=DataLoader(ds,batch_size=config["batch_size"],shuffle=True,generator=torch.Generator().manual_seed(seed)); best=np.inf; wait=0; best_epoch=0
    for epoch in range(1,epochs+1):
        model.train(); total=0.0
        for xb,bb,yb in loader:
            xb,bb,yb=xb.to(device),bb.to(device),yb.to(device); opt.zero_grad(set_to_none=True); p=final_probability(bb,model(xb),config["train_lambda"]); loss=((p-yb)**2).mean()
            if not torch.isfinite(loss): raise FloatingPointError("비유한 loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),config["gradient_clip"]); opt.step(); total+=float(loss.detach())*len(xb)
        torch.save({"model":model.state_dict(),"input_dim":x.shape[1],"epoch":epoch,"kind":kind},target/f"epoch_{epoch}.pt")
        if early:
            validation_raw=predict(model,xp,device,config["predict_batch_size"])
            validation_probability=numpy_probability(pred.p_915.to_numpy(),validation_raw,config["train_lambda"])[0]
            score=float(np.mean((validation_probability-pred.target.to_numpy())**2))
        else:
            score=total/len(ds)
        if score<best-1e-8: best=score; best_epoch=epoch; wait=0
        else: wait+=1
        if not early: best_epoch=epoch
        if early and wait>=config["early_stopping_patience"]: break
    saved=torch.load(target/f"epoch_{best_epoch}.pt",map_location=device,weights_only=True); model.load_state_dict(saved["model"]); raw=predict(model,xp,device,config["predict_batch_size"]); np.save(raw_path,raw); write_json(marker,{"signature":signature,"best_epoch":best_epoch}); del x,xp,model; gc.collect(); return raw,best_epoch

def choose_lambda(inner,raw):
    values={str(l):float(np.mean((numpy_probability(inner.p_915,raw,l)[0]-inner.target)**2)) for l in LAMBDAS}; return float(min(values,key=values.get)),values

def evaluate_candidate(frame,kind,seeds,config,device,ckpt):
    outputs=[]; selections={}; selection_history={}; best_epochs={}
    for year,train_years in OUTER.items():
        outer_train=frame[frame.fold.isin(train_years)]; outer_eval=frame[frame.fold.eq(year)]
        if set(outer_train[ID])&set(outer_eval[ID]): raise ValueError("temporal train/eval overlap")
        inner_train,inner_val=split_inner(frame,year); inner_raw=[]
        for seed in seeds:
            r,e=fit(inner_train,inner_val,kind,seed,config["epochs"],config,device,f"outer_{year}_inner",ckpt); inner_raw.append(r); best_epochs[f"{year}_inner_{seed}"]=e
        lam,history=choose_lambda(inner_val,np.mean(inner_raw,axis=0)); selections[str(year)]=lam; selection_history[str(year)]=history
        outer_raw=[]
        for seed in seeds:
            r,e=fit(outer_train,outer_eval,kind,seed,max(1,round(np.mean([best_epochs[f'{year}_inner_{s}'] for s in seeds]))),config,device,f"outer_{year}_refit",ckpt,False); outer_raw.append(r); best_epochs[f"{year}_refit_{seed}"]=e
        raw=np.mean(outer_raw,axis=0); out=outer_eval[[ID,FOLD,"target","p_915"]].copy(); out["raw_residual"]=raw; out["bounded_residual"]=np.tanh(raw)
        for l in LAMBDAS: out[f"p_lambda_{l:.2f}"]=numpy_probability(out.p_915,raw,l)[0]
        out["selected_lambda"]=lam; out["p_final"]=numpy_probability(out.p_915,raw,lam)[0]; outputs.append(out)
    return pd.concat(outputs,ignore_index=True),selections,selection_history,best_epochs

def summarize(pred,kind,selections,history,seeds,config):
    rows=[]
    for scope,g in [(str(y),pred[pred.fold.eq(y)]) for y in [2023,2024]]+[("overall",pred)]:
        base=metrics(g.target,g.p_915)
        for l in LAMBDAS:
            p=g[f"p_lambda_{l:.2f}"]; m=metrics(g.target,p); deployable=bool(scope!="overall" and l==selections[scope]); rows.append({"model":kind,"scope":scope,"lambda":l,"deployable":deployable,**m,"brier_delta":m["brier"]-base["brier"],"correlation_with_915":float(np.corrcoef(g.p_915,p)[0,1]),"error_delta_bootstrap_ci":bootstrap_error_delta(g.target.to_numpy(),g.p_915.to_numpy(),p.to_numpy(),repeats=config["bootstrap_repeats"]) if deployable else None,**residual_stats(g.raw_residual)})
    base=metrics(pred.target,pred.p_915); m=metrics(pred.target,pred.p_final)
    rows.append({"model":kind,"scope":"overall_deployable","lambda":"temporal_selected","deployable":True,**m,"brier_delta":m["brier"]-base["brier"],"correlation_with_915":float(np.corrcoef(pred.p_915,pred.p_final)[0,1]),"error_delta_bootstrap_ci":bootstrap_error_delta(pred.target.to_numpy(),pred.p_915.to_numpy(),pred.p_final.to_numpy(),repeats=config["bootstrap_repeats"]),**residual_stats(pred.raw_residual)})
    table=pd.DataFrame(rows); deploy=[]
    for y in [2023,2024]: deploy.append(table[(table.scope.eq(str(y)))&(table["lambda"].eq(selections[str(y)]))].iloc[0])
    overall_delta=float(np.average([x.brier_delta for x in deploy],weights=[len(pred[pred.fold.eq(y)]) for x,y in zip(deploy,[2023,2024])]))
    reject=any(x.brier_delta>=.0001 for x in deploy) or any(x.residual_saturation_fraction>config["acceptance"]["max_saturation_fraction"] for x in deploy)
    passed=(not reject and all(x.brier_delta<=0 for x in deploy) and overall_delta<=-config["acceptance"]["min_overall_improvement"] and any(float(x["lambda"])>0 for x in deploy))
    return table,{"model":kind,"selected_lambdas":selections,"lambda_selection_history":history,"deployable_overall_delta":overall_delta,"passed":bool(passed),"rejected":bool(reject),"seeds":seeds}

def run_final(frame,test,prior,cfg,device,ckpt,out):
    dcn=next(x for x in prior["models"] if x["model"]=="dcn")
    if not dcn.get("passed") or prior.get("decision") not in {"CONDITIONAL","ACCEPT"}: raise PermissionError("검증 통과 DCN report가 아님")
    lam=float(dcn["selected_lambdas"]["2024"]); seeds=dcn["seeds"]; raws=[]
    for seed in seeds:
        raw,_=fit(frame,test,"dcn",int(seed),cfg["epochs"],cfg,device,"final_train",ckpt,False); raws.append(raw)
    raw=np.mean(raws,axis=0); probability=numpy_probability(test.p_915,np.mean(raws,axis=0),lam)[0]
    if not np.isfinite(probability).all() or not np.all((0<=probability)&(probability<=1)): raise ValueError("final 확률 계약 실패")
    submission=pd.DataFrame({ID:test[ID].to_numpy(),TARGET:probability})
    if not np.array_equal(submission[ID].astype(str),test[ID].astype(str)) or submission[ID].duplicated().any(): raise ValueError("final row_id 계약 실패")
    submission.to_csv(out/"submission.csv",index=False)
    return {"status":"completed","phase":"final_train","model":"dcn","lambda":lam,"seeds":seeds,"rows":len(submission),**residual_stats(raw)}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--phase",choices=["seed42","three_seed","final_train"],default="seed42"); p.add_argument("--data-root",required=True); p.add_argument("--asset-root",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--checkpoint-dir",required=True); p.add_argument("--mlp-oof-path"); p.add_argument("--smoke",action="store_true"); p.add_argument("--cpu-smoke",action="store_true"); a=p.parse_args()
    cfg=load_json(root()/"configs/experiment.json"); device=torch.device("cpu" if a.cpu_smoke else "cuda")
    if not a.cpu_smoke and not torch.cuda.is_available(): raise RuntimeError("full mode는 CUDA 필수")
    assets=inspect(Path(a.asset_root),Path(a.data_root),Path(a.mlp_oof_path) if a.mlp_oof_path else None,a.phase=="final_train")
    frame=build_frame(Path(a.data_root),Path(a.asset_root),Path(a.mlp_oof_path) if a.mlp_oof_path else None); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); ckpt=Path(a.checkpoint_dir)/a.phase
    if a.smoke: frame=pd.concat([frame[frame.fold.eq(y)].head(cfg["smoke_rows_per_fold"]) for y in YEARS],ignore_index=True); cfg["epochs"]=2; cfg["bootstrap_repeats"]=20
    if a.phase=="final_train":
        prior_path=Path(a.checkpoint_dir)/"three_seed"/"report.json"
        if not prior_path.is_file(): prior_path=Path(a.checkpoint_dir)/"seed42"/"report.json"
        if not prior_path.is_file(): raise PermissionError("final_train 검증 report 없음")
        test=build_test_frame(Path(a.data_root),Path(a.asset_root)); report=run_final(frame,test,json.loads(prior_path.read_text(encoding="utf-8")),cfg,device,ckpt,out); report["asset_audit"]=assets; write_json(out/"report.json",report); write_json(ckpt/"report.json",report); return
    if a.phase=="three_seed":
        prior=Path(a.checkpoint_dir)/"seed42"/"report.json"
        if not prior.is_file() or not json.loads(prior.read_text(encoding="utf-8")).get("seed42_extend"): raise PermissionError("seed42 통과 report 없음")
    seeds=cfg["seeds"][a.phase]; reports=[]; tables=[]; predictions=[]; started=time.time()
    for kind in ["mlp","dcn"]:
        pred,sel,hist,epochs=evaluate_candidate(frame,kind,seeds,cfg,device,ckpt); table,report=summarize(pred,kind,sel,hist,seeds,cfg); report["best_epochs"]=epochs; reports.append(report); tables.append(table); predictions.append(pred.assign(model=kind))
    dcn=next(x for x in reports if x["model"]=="dcn"); mlp=next(x for x in reports if x["model"]=="mlp"); dcn["better_or_more_stable_than_mlp"]=dcn["deployable_overall_delta"]<=mlp["deployable_overall_delta"]; extend=bool(dcn["passed"] and dcn["better_or_more_stable_than_mlp"])
    report={"status":"completed","decision":"CONDITIONAL" if extend else "REJECT","phase":a.phase,"seed42_extend":extend,"models":reports,"asset_audit":assets,"lambda_zero_max_error":float(max(np.max(np.abs(x.p_915-x["p_lambda_0.00"])) for x in predictions)),"runtime_seconds":time.time()-started,"peak_rss_mib":psutil.Process().memory_info().rss/2**20,"environment":{"python":platform.python_version(),"torch":torch.__version__,"device":str(device)}}
    if report["lambda_zero_max_error"]>2e-15: raise AssertionError("lambda=0 baseline 불일치")
    pd.concat(tables).to_csv(out/"metrics.csv",index=False); pd.concat(predictions).to_csv(out/"residual_oof_predictions.csv.gz",index=False,compression="gzip"); write_json(out/"report.json",report); write_json(ckpt/"report.json",report)
if __name__=="__main__": main()
