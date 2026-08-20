"""검증된 rolling OOF만 row_id 순서로 결합한다."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from contract import *

def _prediction_columns(out: pd.DataFrame) -> pd.DataFrame:
    base = out[["p_915", "p_lgb", "p_cat"]].to_numpy(float)
    out["p_lgb_minus_cat"] = out.p_lgb-out.p_cat
    out["p_915_minus_lgb"] = out.p_915-out.p_lgb
    out["p_915_minus_cat"] = out.p_915-out.p_cat
    out["prediction_mean"] = base.mean(1); out["prediction_std"] = base.std(1)
    out["prediction_range"] = base.max(1)-base.min(1)
    return out

def build_frame(data_root: Path, asset_root: Path, mlp_path: Path | None=None) -> pd.DataFrame:
    cfg=load_json(root()/"configs/assets.json")
    train=pd.read_csv(data_root/"train.csv",usecols=list(dict.fromkeys([ID,TARGET,*RAW_FEATURES])),low_memory=False)
    mlp=pd.read_csv(mlp_path or asset_root/cfg["sources"]["real_890_mlp_oof"])
    parts=[]
    for year in YEARS:
        val=train.loc[train.season.eq(year)].reset_index(drop=True); mm=mlp.loc[mlp.fold.eq(year)].reset_index(drop=True)
        with np.load(asset_root/cfg["sources"]["lightgbm"].format(year=year),allow_pickle=True) as lz, np.load(asset_root/cfg["sources"]["catboost"].format(year=year)) as cz:
            ids=val[ID].astype(str).to_numpy(); y=val[TARGET].to_numpy()
            ok=(np.array_equal(lz["row_id"].astype(str),ids) and np.array_equal(mm[ID].astype(str),ids)
                and np.array_equal(lz["y_true"],y) and np.array_equal(cz["y_true"],y)
                and np.array_equal(mm["target"],y) and np.array_equal(cz["seed_777"],lz["catboost_seed777"]))
            if not ok: raise ValueError(f"row_id/fold/target OOF 정렬 계약 실패: {year}")
            out=val.copy(); out[FOLD]=year; out["target"]=y.astype(np.int8)
            out["p_lgb"]=lz["lightgbm_k10"].astype(float); out["p_cat"]=cz["seed_777"].astype(float)
            out["p_915"]=shifted(.75*out.p_lgb.to_numpy()+.25*mm.p_mlp_real_890.to_numpy(float))
            parts.append(_prediction_columns(out))
    frame=pd.concat(parts,ignore_index=True)
    if len(frame)!=746_504 or frame[ID].duplicated().any(): raise ValueError("최종 OOF 행 계약 실패")
    return frame

def build_test_frame(data_root: Path, asset_root: Path) -> pd.DataFrame:
    cfg=load_json(root()/"configs/assets.json"); test=pd.read_csv(data_root/"test.csv",usecols=[ID,*RAW_FEATURES])
    pred=pd.read_csv(asset_root/cfg["sources"]["final_test_predictions"])
    if not np.array_equal(test[ID].astype(str),pred[ID].astype(str)): raise ValueError("test row_id 순서 불일치")
    out=test.copy()
    for c in ["p_915","p_lgb","p_cat"]: out[c]=pred[c].to_numpy(float)
    return _prediction_columns(out)
