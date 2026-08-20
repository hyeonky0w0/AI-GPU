"""검증된 OOF 원본을 row_id 순서로 결합해 residual 학습 frame을 만든다."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from contract import BASELINE_SHIFT, ID, MODEL_FEATURES, RAW_FEATURES, TARGET, YEARS, experiment_root, load_json, sigmoid_logit_shift


def build_frame(data_root: Path, asset_root: Path, mlp_oof_path: Path | None = None) -> pd.DataFrame:
    assets = load_json(experiment_root() / "configs/assets.json")
    usecols = [ID, TARGET, "pitcher_id", *RAW_FEATURES]
    usecols = list(dict.fromkeys(usecols))
    train = pd.read_csv(data_root / "train.csv", usecols=usecols, low_memory=False)
    mlp = pd.read_csv(mlp_oof_path or asset_root / assets["sources"]["real_890_mlp_oof"])
    parts = []
    for year in YEARS:
        val = train.loc[train["season"].eq(year)].reset_index(drop=True)
        mm = mlp.loc[mlp["fold"].eq(year)].reset_index(drop=True)
        with np.load(asset_root / assets["sources"]["lightgbm"].format(year=year), allow_pickle=True) as lz, \
             np.load(asset_root / assets["sources"]["catboost"].format(year=year), allow_pickle=False) as cz:
            ids, y = val[ID].astype(str).to_numpy(), val[TARGET].to_numpy()
            if not (np.array_equal(lz["row_id"].astype(str), ids)
                    and np.array_equal(mm[ID].astype(str).to_numpy(), ids)
                    and np.array_equal(lz["y_true"], y) and np.array_equal(cz["y_true"], y)
                    and np.array_equal(mm["target"].to_numpy(), y)
                    and np.array_equal(cz["seed_777"], lz["catboost_seed777"])):
                raise ValueError(f"row_id/target/base prediction 정렬 계약 실패: {year}")
            out = val.copy()
            out["fold"] = year
            out["target"] = y.astype(np.int8)
            out["p_lgb"] = lz["lightgbm_k10"].astype(np.float64)
            out["p_cat"] = cz["seed_777"].astype(np.float64)
            pre = 0.75 * out["p_lgb"].to_numpy() + 0.25 * mm["p_mlp_real_890"].to_numpy(float)
            out["p_915"] = sigmoid_logit_shift(pre, BASELINE_SHIFT)
            base = out[["p_915", "p_lgb", "p_cat"]].to_numpy()
            out["p_lgb_minus_cat"] = out["p_lgb"] - out["p_cat"]
            out["p_915_minus_lgb"] = out["p_915"] - out["p_lgb"]
            out["p_915_minus_cat"] = out["p_915"] - out["p_cat"]
            out["prediction_mean"] = base.mean(axis=1)
            out["prediction_std"] = base.std(axis=1)
            out["prediction_range"] = base.max(axis=1) - base.min(axis=1)
            parts.append(out)
    frame = pd.concat(parts, ignore_index=True)
    if len(frame) != 746_504 or frame[ID].duplicated().any() or frame[[*MODEL_FEATURES, "target"]].isna().all(axis=1).any():
        raise ValueError("residual dataset 최종 계약 실패")
    return frame


def build_test_frame(data_root: Path, asset_root: Path) -> pd.DataFrame:
    assets=load_json(experiment_root()/"configs/assets.json")
    usecols=list(dict.fromkeys([ID,"pitcher_id",*RAW_FEATURES]))
    test=pd.read_csv(data_root/"test.csv",usecols=usecols,low_memory=False)
    predictions=pd.read_csv(asset_root/assets["sources"]["final_test_predictions"])
    if not np.array_equal(test[ID].astype(str).to_numpy(),predictions[ID].astype(str).to_numpy()):
        raise ValueError("915 test prediction row_id 순서가 test.csv와 다릅니다")
    out=test.copy()
    for column in ["p_915","p_lgb","p_cat"]: out[column]=predictions[column].to_numpy(np.float64)
    base=out[["p_915","p_lgb","p_cat"]].to_numpy(); out["p_lgb_minus_cat"]=out.p_lgb-out.p_cat; out["p_915_minus_lgb"]=out.p_915-out.p_lgb; out["p_915_minus_cat"]=out.p_915-out.p_cat; out["prediction_mean"]=base.mean(1); out["prediction_std"]=base.std(1); out["prediction_range"]=base.max(1)-base.min(1)
    if out[ID].duplicated().any() or not np.isfinite(out[["p_915","p_lgb","p_cat"]]).all().all(): raise ValueError("test prediction 계약 실패")
    return out
