"""038 Router 구조를 고정하고 lightweight MLP와 실제 890 MLP만 교체 비교한다."""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from contract import FEATURES as RAW_FEATURES, FOLDS, ID, TARGET, sha256

PRED=["p_mlp","p_lgb","p_cat","p_mean","p_std","p_range","p_mlp_minus_lgb","p_mlp_minus_cat","p_lgb_minus_cat"]
CAT=["game_type","base_state","top_bottom","pitcher_hand","batter_hand"]
CTX=["season","inning","balls_before","strikes_before","outs_before","score_diff_pitcher_team","asof_pitcher_n","asof_pitcher_success_rate","asof_pitcher_prev1_game_success_rate","asof_pitcher_prev3_game_success_rate","asof_pitcher_prev5_game_success_rate","asof_batter_n","asof_batter_success_rate"]
ROUTER_FEATURES=PRED+CAT+CTX; OUTER={2023:[2022],2024:[2022,2023]}; L2=1e-3; SHIFT=-0.03946
SOURCE_HASHES={
    "router_sources/catboost/fold_2022.npz":"0b412e1b738f7af6a34f2d57af16eb89754cacb644c57edfe0af8cf0ab4aa2bc",
    "router_sources/catboost/fold_2023.npz":"5bed5b35ec5dbd5fba871b90657f601331ab51017ece9637182945ea3b16c9e9",
    "router_sources/catboost/fold_2024.npz":"cdefdbfb4a78870f71725b1e1b9fa2f95610ccdb2d8d731e52e01bddceb437f0",
    "router_sources/lightgbm/fold_2022.npz":"5697cb8750529c5be1273c2a3e52e857183610e20b66de43362a0279bb457035",
    "router_sources/lightgbm/fold_2023.npz":"561f41a3dc13c8281fc8b89679ecf550d8276e725d3dd88a977593171742f0797",
    "router_sources/lightgbm/fold_2024.npz":"13df97e6e8d3481f9338a393b1fe6fcc683d945ad4f64bfc6e58840693bab7e6",
}

def softmax3(z2):
    z=np.column_stack([z2,np.zeros(len(z2))]); z-=z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)

def fit_gate(x,base,y):
    d=np.column_stack([np.ones(len(x)),x]); k=d.shape[1]
    def obj(flat):
        t=flat.reshape(k,2); w=softmax3(d@t); p=(w*base).sum(1); err=p-y; g=np.empty_like(t)
        for j in range(2): g[:,j]=d.T@(2*err*w[:,j]*(base[:,j]-p))/len(y)
        penalty=np.vstack([np.zeros((1,2)),t[1:]])
        return float(np.mean(err**2)+L2*np.sum(t[1:]**2)),(g+2*L2*penalty).ravel()
    r=minimize(obj,np.zeros(k*2),jac=True,method="L-BFGS-B",options={"maxiter":100,"ftol":1e-12,"gtol":1e-8})
    if not r.success: raise RuntimeError(f"Router 최적화 실패: {r.message}")
    return r.x.reshape(k,2)

def prep_frame(frame,p_mlp):
    out=frame.copy(); out["p_mlp"]=p_mlp; b=out[["p_mlp","p_lgb","p_cat"]].to_numpy()
    out["p_mean"]=b.mean(1); out["p_std"]=b.std(1); out["p_range"]=b.max(1)-b.min(1)
    out["p_mlp_minus_lgb"]=out.p_mlp-out.p_lgb; out["p_mlp_minus_cat"]=out.p_mlp-out.p_cat; out["p_lgb_minus_cat"]=out.p_lgb-out.p_cat
    return out

def metric(y,p):
    return {"brier":float(np.mean((p-y)**2)),"mean_prediction":float(p.mean()),"actual_positive_rate":float(y.mean()),"calibration_bias":float(p.mean()-y.mean()),"auc":float(roc_auc_score(y,p)),"logloss":float(log_loss(y,np.clip(p,1e-7,1-1e-7)))}

def load_base(data,asset_root):
    rows=[]
    for year in FOLDS:
        val=data[data.season.eq(year)].copy(); y=val[TARGET].to_numpy(); ids=val[ID].astype(str).to_numpy()
        cz=np.load(asset_root/f"router_sources/catboost/fold_{year}.npz",allow_pickle=True); lz=np.load(asset_root/f"router_sources/lightgbm/fold_{year}.npz",allow_pickle=True)
        if (not np.array_equal(cz["y_true"],y) or not np.array_equal(lz["y_true"],y)
                or not np.array_equal(cz["row_id"].astype(str),ids)
                or not np.array_equal(lz["row_id"].astype(str),ids)):
            raise ValueError(f"base OOF row/target 불일치: {year}")
        f=val[[ID,TARGET,*CAT,*CTX]].copy().rename(columns={TARGET:"target"}); f["fold"]=year; f["p_lgb"]=lz["lightgbm_k10"]; f["p_cat"]=cz["seed_777"]; rows.append(f)
    return pd.concat(rows,ignore_index=True)

def run_router(frame,label):
    outputs=[]
    for year,years in OUTER.items():
        tr=frame[frame.fold.isin(years)]; va=frame[frame.fold.eq(year)]
        if set(tr[ID])&set(va[ID]): raise ValueError("Router train/eval row overlap")
        pre=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),PRED+CTX),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),CAT)])
        xt=pre.fit_transform(tr[ROUTER_FEATURES]); xv=pre.transform(va[ROUTER_FEATURES]); theta=fit_gate(xt,tr[["p_mlp","p_lgb","p_cat"]].to_numpy(),tr.target.to_numpy()); w=softmax3(np.column_stack([np.ones(len(xv)),xv])@theta); p=(w*va[["p_mlp","p_lgb","p_cat"]].to_numpy()).sum(1)
        o=va[[ID,"fold","target","p_range"]].copy(); o[[f"w_{x}" for x in ["mlp","lgb","cat"]]]=w; o[f"p_router_{label}"]=p; outputs.append(o)
    return pd.concat(outputs,ignore_index=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-root",required=True); ap.add_argument("--asset-root",required=True); ap.add_argument("--mlp-oof",required=True); ap.add_argument("--output-dir",required=True); args=ap.parse_args(); started=time.perf_counter()
    root=Path(args.data_root); assets=Path(args.asset_root); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    for relative,expected in SOURCE_HASHES.items():
        path=assets/relative
        if sha256(path)!=expected: raise ValueError(f"Router source SHA-256 불일치: {relative}")
    data=pd.read_csv(root/"train.csv",usecols=[ID,TARGET,*RAW_FEATURES],encoding="utf-8-sig",low_memory=False); base=load_base(data,assets)
    real=pd.read_csv(args.mlp_oof); light=pd.read_csv(assets/"router_sources/lightweight_oof_predictions.csv.gz")
    expected=base.set_index(ID)[["fold","target"]].sort_index()
    for name,x,col in [("real",real,"p_mlp_real_890"),("light",light,"p_mlp")]:
        if x[ID].duplicated().any() or set(x[ID].astype(str))!=set(base[ID].astype(str)): raise ValueError(f"{name} OOF row_id 계약 실패")
        aligned=x.assign(**{ID:x[ID].astype(str)}).set_index(ID).loc[expected.index]
        if not np.array_equal(aligned["fold"].to_numpy(),expected["fold"].to_numpy()) or not np.array_equal(aligned["target"].to_numpy(),expected["target"].to_numpy()): raise ValueError(f"{name} OOF fold/target 계약 실패")
        probability=aligned[col].to_numpy()
        if not np.isfinite(probability).all() or not ((probability>=0)&(probability<=1)).all(): raise ValueError(f"{name} OOF 확률 계약 실패")
    indexed=base.set_index(ID); rp=real.set_index(ID).loc[indexed.index,"p_mlp_real_890"].to_numpy(); lp=light.set_index(ID).loc[indexed.index,"p_mlp"].to_numpy()
    real_frame=prep_frame(base,rp); light_frame=prep_frame(base,lp); rr=run_router(real_frame,"real_890"); lr=run_router(light_frame,"lightweight")
    joined=rr.merge(lr[[ID,"p_router_lightweight"]],on=ID,validate="one_to_one"); eval_real=real_frame.set_index(ID).loc[joined[ID]]; y=joined.target.to_numpy()
    pre=.75*eval_real.p_lgb.to_numpy()+.25*eval_real.p_mlp.to_numpy(); shifted=1/(1+np.exp(-(np.log(np.clip(pre,1e-7,1-1e-7)/(1-np.clip(pre,1e-7,1-1e-7)))+SHIFT)))
    candidates={"A_baseline_915":shifted,"B_router_lightweight":joined.p_router_lightweight.to_numpy(),"C_router_real_890":joined.p_router_real_890.to_numpy(),"D_fixed_blend_real_890":pre}
    rows=[]
    for year in OUTER:
        m=joined.fold.eq(year).to_numpy()
        for name,p in candidates.items(): rows.append({"scope":str(year),"candidate":name,"rows":int(m.sum()),**metric(y[m],p[m])})
    for name,p in candidates.items(): rows.append({"scope":"overall","candidate":name,"rows":len(y),**metric(y,p)})
    pd.DataFrame(rows).to_csv(out/"router_comparison.csv",index=False)
    weights=[]
    for year,g in joined.groupby("fold"):
        for scope,mask in [("bottom_50",g.p_range<=g.p_range.quantile(.5)),("50_90",(g.p_range>g.p_range.quantile(.5))&(g.p_range<=g.p_range.quantile(.9))),("top_10",g.p_range>g.p_range.quantile(.9))]:
            x=g.loc[mask]; weights.append({"season":year,"disagreement":scope,"rows":len(x),"mean_mlp_weight":x.w_mlp.mean(),"median_mlp_weight":x.w_mlp.median(),"std_mlp_weight":x.w_mlp.std(),"min_mlp_weight":x.w_mlp.min(),"max_mlp_weight":x.w_mlp.max(),"mean_lgb_weight":x.w_lgb.mean(),"mean_cat_weight":x.w_cat.mean()})
    pd.DataFrame(weights).to_csv(out/"router_weight_analysis.csv",index=False); joined.to_csv(out/"router_oof_predictions.csv.gz",index=False,compression="gzip")
    report={"status":"completed_no_selection_decision","outer_folds":OUTER,"router_architecture_changed":False,"router_training_rows_scored":False,"row_id_unique":bool(joined[ID].is_unique),"weight_sum_max_error":float(np.abs(joined[["w_mlp","w_lgb","w_cat"]].sum(1)-1).max()),"runtime_seconds":time.perf_counter()-started}
    (out/"router_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")

if __name__=="__main__": main()
