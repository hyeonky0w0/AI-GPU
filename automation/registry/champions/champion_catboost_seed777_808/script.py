from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


def main() -> None:
    root = Path(__file__).resolve().parent
    config = json.loads((root / "model" / "config.json").read_text(encoding="utf-8"))
    test = pd.read_csv(root / "data" / "test.csv", low_memory=False, encoding="utf-8-sig")
    features = config["features"]
    missing = [column for column in features if column not in test]
    if missing:
        raise ValueError(f"test 피처 누락: {missing}")
    X = test.loc[:, features].copy()
    if X.columns.tolist() != features:
        raise ValueError("추론 피처 순서 불일치")
    for column in config["categorical_columns"]:
        X[column] = X[column].fillna("__MISSING__").astype(str)
    model = CatBoostClassifier()
    model.load_model(root / "model" / config["model_file"])
    prediction = model.predict_proba(X)[:, 1]
    if len(prediction) != len(test) or not np.isfinite(prediction).all() or ((prediction < 0) | (prediction > 1)).any():
        raise ValueError("예측 검증 실패")
    output = pd.DataFrame({"row_id": test["row_id"], "control_success": prediction})
    (root / "output").mkdir(exist_ok=True)
    output.to_csv(root / "output" / "submission.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
