"""실제 890 MLP 계약의 temporal rolling OOF 생성기."""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler

from contract import (CHECKPOINT_SHA256, FEATURES, FOLDS, ID, NUMERIC, ONEHOT, SEEDS,
                      SNAPSHOTS, TARGET, USED, sha256)

EXP=Path(__file__).resolve().parents[1]

def make_pre() -> ColumnTransformer:
    return ColumnTransformer([
        ("qt",Pipeline([("imp",SimpleImputer(strategy="median")),("qt",QuantileTransformer(output_distribution="normal",n_quantiles=1000,subsample=200_000,random_state=0))]),NUMERIC),
        ("na",MissingIndicator(features="missing-only"),NUMERIC),
        ("season",StandardScaler(),["season"]),
        ("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),ONEHOT),
    ],remainder="drop",verbose_feature_names_out=False)

def make_model(seed:int) -> MLPClassifier:
    return MLPClassifier(hidden_layer_sizes=(256,128),activation="relu",solver="adam",alpha=1e-4,batch_size=512,learning_rate_init=1e-3,max_iter=1,shuffle=True,random_state=seed,verbose=False)

def validate_reference(path:Path, sample:pd.DataFrame) -> dict:
    if sha256(path)!=CHECKPOINT_SHA256: raise ValueError("890 checkpoint SHA-256 불일치")
    model=joblib.load(path)
    if list(model.feature_names_in_)!=FEATURES or model.n_features_in_!=47: raise ValueError("890 checkpoint 입력 계약 불일치")
    vote=model.named_steps["clf"]
    if len(vote.estimators_)!=30: raise ValueError("890 checkpoint member 수 불일치")
    members=[member.estimator for member in vote.estimators_]
    first=members[0]
    expected={"hidden_layer_sizes":(256,128),"activation":"relu","solver":"adam","alpha":1e-4,"batch_size":512,"learning_rate_init":1e-3,"random_state":42}
    if any(first.get_params()[k]!=v for k,v in expected.items()): raise ValueError("890 checkpoint 모델 파라미터 불일치")
    if [member.random_state for member in members] != [seed for seed in SEEDS for _ in SNAPSHOTS]: raise ValueError("890 checkpoint seed/member 순서 불일치")
    common={k:v for k,v in expected.items() if k!="random_state"}
    if any(any(member.get_params()[k]!=v for k,v in common.items()) for member in members): raise ValueError("890 checkpoint member 파라미터 불일치")
    x=sample[FEATURES]; got=model.predict_proba(x)[:,1]; xt=model.named_steps["pre"].transform(x)
    manual=np.mean([member.predict_proba(xt)[:,1] for member in members],axis=0)
    error=float(np.max(np.abs(got-manual)))
    if error>1e-12 or not np.isfinite(got).all() or not ((got>=0)&(got<=1)).all(): raise ValueError("890 checkpoint inference 재현 실패")
    return {"rows":len(sample),"max_absolute_difference":error,"mean_absolute_difference":float(np.mean(np.abs(got-manual))),"correlation":float(np.corrcoef(got,manual)[0,1]) if len(got)>1 else 1.0}

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--data-root",required=True); ap.add_argument("--asset-root",required=True); ap.add_argument("--output-dir",required=True); ap.add_argument("--smoke",action="store_true"); args=ap.parse_args()
    root=Path(args.data_root); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(root/"train.csv",usecols=[ID,*FEATURES,TARGET],encoding="utf-8-sig",low_memory=False)
    if len(FEATURES)!=47 or len(USED)!=37 or len(SEEDS)*len(SNAPSHOTS)!=30: raise ValueError("실제 890 상수 계약 위반")
    if data[ID].isna().any() or data[ID].duplicated().any(): raise ValueError("원본 row_id 결측/중복")
    if TARGET in FEATURES or ID in FEATURES or list(data[FEATURES].columns)!=FEATURES: raise ValueError("feature/target 계약 위반")
    reference=validate_reference(Path(args.asset_root)/"mlp_snap345.pkl",data.head(32))
    seeds=[42] if args.smoke else SEEDS; snapshots=[1] if args.smoke else SNAPSHOTS
    parts=[]; fold_metrics=[]; started=time.perf_counter()
    for year,train_years in FOLDS.items():
        train=data[data.season.isin(train_years)]; val=data[data.season.eq(year)]
        if args.smoke:
            train=train.groupby("season",group_keys=False).head(128); val=val.head(256)
        if set(train[ID])&set(val[ID]) or not (train.season<year).all() or not (val.season==year).all(): raise ValueError(f"fold {year} temporal leakage")
        pre=make_pre(); xtr=pre.fit_transform(train[FEATURES]); xval=pre.transform(val[FEATURES]); y=train[TARGET].to_numpy(); preds=[]
        if getattr(pre,"n_features_in_",0)!=47: raise ValueError("preprocessor feature count 불일치")
        for seed in seeds:
            model=make_model(seed)
            for epoch in range(1,max(snapshots)+1):
                model.partial_fit(xtr,y,classes=np.array([0,1]))
                if epoch in snapshots: preds.append(model.predict_proba(xval)[:,1].copy())
        p=np.mean(np.vstack(preds),axis=0)
        if p.shape!=(len(val),) or not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all(): raise ValueError("OOF prediction 계약 위반")
        frame=pd.DataFrame({ID:val[ID].astype(str),"fold":year,"target":val[TARGET].astype(np.int8),"p_mlp_real_890":p})
        parts.append(frame); fold_metrics.append({"fold":year,"train_seasons":train_years,"train_rows":len(train),"validation_rows":len(val),"target_rate":float(val[TARGET].mean()),"prediction_mean":float(p.mean()),"calibration_bias":float(p.mean()-val[TARGET].mean()),"brier":float(np.mean((p-val[TARGET].to_numpy())**2)),"seeds":seeds,"snapshot_epochs":snapshots})
        np.savez_compressed(out/f"mlp_fold_{year}.npz",row_id=frame[ID].to_numpy(),y_true=frame.target.to_numpy(),p_mlp_real_890=p,train_seasons=np.array(train_years))
    oof=pd.concat(parts,ignore_index=True)
    if oof[ID].duplicated().any(): raise ValueError("fold 간 row_id 중복")
    oof.to_csv(out/("mlp_oof_smoke.csv.gz" if args.smoke else "mlp_oof_predictions.csv.gz"),index=False,compression="gzip")
    report={"mode":"smoke" if args.smoke else "full","contract":"real_890_snap3_5","reference_inference":reference,"fold_metrics":fold_metrics,"leakage":{"target_in_features":False,"validation_excluded_from_training":True,"future_season_excluded":True,"preprocessor_fit_on_training_only":True,"row_id_unique":True},"runtime_seconds":time.perf_counter()-started}
    (out/("smoke_report.json" if args.smoke else "mlp_oof_report.json")).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")

if __name__=="__main__": main()
