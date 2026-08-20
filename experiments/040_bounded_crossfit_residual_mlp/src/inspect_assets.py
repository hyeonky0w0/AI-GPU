"""040 source 자산을 읽기 전용으로 감사하고 915 OOF 계약을 재현한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from contract import BASELINE_SHIFT, ID, TARGET, YEARS, experiment_root, load_json, sha256, sigmoid_logit_shift


def inspect(asset_root: Path, data_root: Path, require_final: bool = False) -> dict:
    contract = load_json(experiment_root() / "configs/assets.json")
    required = list(contract["sha256"])
    if require_final:
        final = contract["final_test_contract"]
        if not final.get("sha256"):
            raise ValueError("final_train 차단: 정확한 915 test prediction SHA256이 assets.json에 등록되지 않았습니다")
        required.append(contract["sources"]["final_test_predictions"])
        if not (data_root / "test.csv").is_file():
            raise FileNotFoundError(f"final_train test.csv 누락: {data_root / 'test.csv'}")
    missing = [str(asset_root / rel) for rel in required if not (asset_root / rel).is_file()]
    if not (data_root / "train.csv").is_file():
        missing.append(str(data_root / "train.csv"))
    if missing:
        raise FileNotFoundError("040 필수 자산 누락 (검사 root: " + str(asset_root) + "):\n  - " + "\n  - ".join(missing))
    checked = {}
    for rel in required:
        expected = contract["sha256"].get(rel) or contract["final_test_contract"].get("sha256")
        actual = sha256(asset_root / rel)
        if actual != expected:
            raise ValueError(f"SHA256 불일치: {rel}\nexpected={expected}\nactual={actual}")
        checked[rel] = {"sha256": actual, "bytes": (asset_root / rel).stat().st_size}
    if require_final:
        final_predictions = pd.read_csv(asset_root / contract["sources"]["final_test_predictions"])
        expected = contract["final_test_contract"]["required_columns"]
        if list(final_predictions.columns) != expected or final_predictions[ID].isna().any() or final_predictions[ID].duplicated().any():
            raise ValueError(f"915 test prediction schema/row_id 계약 불일치: {list(final_predictions.columns)}")

    train = pd.read_csv(data_root / "train.csv", usecols=[ID, "season", TARGET], low_memory=False)
    mlp_path = asset_root / contract["sources"]["real_890_mlp_oof"]
    mlp = pd.read_csv(mlp_path)
    expected_columns = [ID, "fold", "target", "p_mlp_real_890"]
    if list(mlp.columns) != expected_columns or len(mlp) != 746_504:
        raise ValueError(f"실제 890 MLP OOF schema/행 수 불일치: {mlp.shape}, {list(mlp.columns)}")
    if mlp[ID].isna().any() or mlp[ID].duplicated().any() or mlp.isna().any().any():
        raise ValueError("실제 890 MLP OOF row_id 중복 또는 결측")

    folds = []
    all_y, all_p = [], []
    for year in YEARS:
        val = train.loc[train["season"].eq(year)].reset_index(drop=True)
        mm = mlp.loc[mlp["fold"].eq(year)].reset_index(drop=True)
        lpath = asset_root / contract["sources"]["lightgbm"].format(year=year)
        cpath = asset_root / contract["sources"]["catboost"].format(year=year)
        with np.load(lpath, allow_pickle=True) as lz, np.load(cpath, allow_pickle=False) as cz:
            lkeys, ckeys = sorted(lz.files), sorted(cz.files)
            needed_l = {"row_id", "y_true", "lightgbm_k10", "catboost_seed777"}
            needed_c = {"y_true", "seed_777"}
            if not needed_l.issubset(lz.files) or not needed_c.issubset(cz.files):
                raise KeyError(f"NPZ schema 불일치 {year}: lgb={lkeys}, cat={ckeys}")
            ids = val[ID].astype(str).to_numpy()
            y = val[TARGET].to_numpy()
            if not np.array_equal(lz["row_id"].astype(str), ids):
                raise ValueError(f"LightGBM row_id/원본 순서 불일치: {year}")
            if not np.array_equal(mm[ID].astype(str).to_numpy(), ids):
                raise ValueError(f"MLP row_id/원본 순서 불일치: {year}")
            if not np.array_equal(lz["y_true"], y) or not np.array_equal(cz["y_true"], y) or not np.array_equal(mm["target"], y):
                raise ValueError(f"target 정렬 불일치: {year}")
            if not np.array_equal(cz["seed_777"], lz["catboost_seed777"]):
                raise ValueError(f"CatBoost/LightGBM embedded 예측 순서 불일치: {year}")
            pre = 0.75 * lz["lightgbm_k10"].astype(float) + 0.25 * mm["p_mlp_real_890"].to_numpy(float)
            p915 = sigmoid_logit_shift(pre, BASELINE_SHIFT)
            folds.append({"year": year, "rows": len(y), "brier_915": float(np.mean((p915-y)**2)),
                          "prediction_mean": float(p915.mean()), "target_rate": float(y.mean()),
                          "lightgbm_keys": lkeys, "catboost_keys": ckeys})
            all_y.append(y); all_p.append(p915)
    combined_y, combined_p = np.concatenate(all_y), np.concatenate(all_p)
    report = {"status": "PASS", "asset_root": str(asset_root.resolve()), "data_root": str(data_root.resolve()),
              "checked_assets": checked, "mlp_oof_rows": len(mlp), "folds": folds,
              "brier_915_all_three_folds": float(np.mean((combined_p-combined_y)**2)),
              "brier_915_outer_2023_2024": float(np.mean((np.concatenate(all_p[1:])-np.concatenate(all_y[1:]))**2))}
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output")
    parser.add_argument("--require-final", action="store_true")
    args = parser.parse_args()
    report = inspect(Path(args.asset_root), Path(args.data_root), args.require_final)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
