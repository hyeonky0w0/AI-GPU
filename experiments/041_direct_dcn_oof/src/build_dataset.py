"""원본 train과 검증된 기존 OOF 자산을 row_id로 엄격 정렬한다."""
from pathlib import Path
import numpy as np, pandas as pd
from contract import *

def load_train(data_root):
 df=pd.read_csv(Path(data_root)/"train.csv",usecols=[ID,TARGET,*RAW_FEATURES],low_memory=False)
 if df[ID].isna().any() or df[ID].duplicated().any() or len(RAW_FEATURES)!=37: raise ValueError("train/37-feature 계약 실패")
 return df

def build_reference_frame(data_root,asset_root,mlp_path=None):
 cfg=load_json(root()/"configs/assets.json"); train=load_train(data_root); mlp=pd.read_csv(mlp_path or Path(asset_root)/cfg["sources"]["real_890_mlp_oof"]); parts=[]
 for year in YEARS:
  val=train[train.season.eq(year)].reset_index(drop=True); mm=mlp[mlp.fold.eq(year)].reset_index(drop=True)
  with np.load(Path(asset_root)/cfg["sources"]["lightgbm"].format(year=year),allow_pickle=True) as lz, np.load(Path(asset_root)/cfg["sources"]["catboost"].format(year=year)) as cz:
   ids=val[ID].astype(str).to_numpy(); y=val[TARGET].to_numpy()
   if not (np.array_equal(ids,lz["row_id"].astype(str)) and np.array_equal(ids,mm[ID].astype(str)) and np.array_equal(y,lz["y_true"]) and np.array_equal(y,cz["y_true"]) and np.array_equal(y,mm.target)): raise ValueError(f"OOF 정렬 계약 실패: {year}")
   out=val[[ID,"season","game_type","asof_pitcher_n"]].copy(); out[ID]=out[ID].astype(str); out[FOLD]=year; out["target"]=y.astype("int8"); out["p_cat"]=cz["seed_777"].astype(float); out["p_lgb"]=lz["lightgbm_k10"].astype(float); out["p_mlp"]=mm["p_mlp_real_890"].to_numpy(float)
   pre=.75*out.p_lgb.to_numpy()+.25*out.p_mlp.to_numpy(); out["p_915"]=1/(1+np.exp(-(np.log(np.clip(pre,1e-6,1-1e-6)/(1-np.clip(pre,1e-6,1-1e-6)))-.03946))); parts.append(out)
 result=pd.concat(parts,ignore_index=True)
 if result[ID].duplicated().any(): raise ValueError("OOF row_id 중복")
 return result
