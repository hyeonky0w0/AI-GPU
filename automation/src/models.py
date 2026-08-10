from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import psutil
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_logistic(X: pd.DataFrame, seed: int, max_iter: int) -> Pipeline:
    categorical = [column for column in X if pd.api.types.is_string_dtype(X[column].dtype)]
    numeric = [column for column in X if column not in categorical]
    preprocessor = ColumnTransformer(
        [
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
            ]), categorical),
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("pre", preprocessor),
        ("model", LogisticRegression(max_iter=max_iter, random_state=seed, solver="lbfgs")),
    ])


def prepare_lightgbm_frames(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, int]]:
    """학습 기준 범주 사전을 고정하고 StringDtype/미지 범주를 안전하게 처리한다."""
    train = X_train.copy()
    validation = X_val.copy()
    if train.columns.tolist() != validation.columns.tolist():
        raise ValueError("LightGBM train/validation 피처 순서 불일치")
    categorical = [column for column in categorical_columns if column in train]
    conversion_failures: dict[str, int] = {}
    for column in categorical:
        train_values = train[column].fillna("__MISSING__").astype(str)
        known = sorted(train_values.unique().tolist())
        categories = known + (["__UNSEEN__"] if "__UNSEEN__" not in known else [])
        validation_values = validation[column].fillna("__MISSING__").astype(str)
        validation_values = validation_values.where(validation_values.isin(known), "__UNSEEN__")
        train[column] = pd.Categorical(train_values, categories=categories)
        validation[column] = pd.Categorical(validation_values, categories=categories)
        if train[column].isna().any() or validation[column].isna().any():
            raise ValueError(f"LightGBM 범주형 null 처리 실패: {column}")
    for column in train.columns:
        if column in categorical:
            continue
        if pd.api.types.is_string_dtype(train[column].dtype) or pd.api.types.is_object_dtype(train[column].dtype):
            converted_train = pd.to_numeric(train[column], errors="coerce")
            converted_val = pd.to_numeric(validation[column], errors="coerce")
            failures = int((train[column].notna() & converted_train.isna()).sum()) + int(
                (validation[column].notna() & converted_val.isna()).sum()
            )
            conversion_failures[column] = failures
            if failures:
                raise ValueError(f"수치형 대상 문자열 변환 실패: {column}={failures}")
            train[column] = converted_train
            validation[column] = converted_val
    return train, validation, categorical, conversion_failures


def _resource_callback(started: float, max_seconds: float, min_available_bytes: int):
    def callback(_env: Any) -> None:
        if time.monotonic() - started > max_seconds:
            raise TimeoutError(f"LightGBM 모델 시간 예산 초과: {max_seconds:.0f}초")
        if psutil.virtual_memory().available < min_available_bytes:
            raise MemoryError("LightGBM 학습 중 가용 메모리 하한 미달")
    callback.order = 0
    callback.before_iteration = False
    return callback


def select_nonnegative_oof_weight(
    y_true: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    grid_size: int = 101,
) -> tuple[float, float]:
    """과거 OOF에서 p = w*a + (1-w)*b의 Brier 최소 가중치를 고른다."""
    if not (len(y_true) == len(prediction_a) == len(prediction_b)) or len(y_true) == 0:
        raise ValueError("OOF 가중치 계산 입력 길이가 잘못되었습니다.")
    weights = np.linspace(0.0, 1.0, grid_size)
    briers = [float(np.mean((weight * prediction_a + (1 - weight) * prediction_b - y_true) ** 2)) for weight in weights]
    index = int(np.argmin(briers))
    return float(weights[index]), briers[index]


def fit_predict(model_type: str, X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, config: dict[str, Any]) -> tuple[np.ndarray, Any]:
    if model_type == "logistic":
        model = build_logistic(X_train, int(config["seed"]), int(config.get("max_iter", 500)))
        model.fit(X_train, y_train)
        return model.predict_proba(X_val)[:, 1], model
    if model_type == "constant":
        probability = float(y_train.mean())
        return np.full(len(X_val), probability), {"probability": probability}
    if model_type == "previous_season_constant":
        if "season" not in X_train:
            raise ValueError("직전 시즌 평균 기준선에 season 컬럼이 필요합니다.")
        latest_season = X_train["season"].max()
        probability = float(y_train[X_train["season"].to_numpy() == latest_season].mean())
        return np.full(len(X_val), probability), {"probability": probability, "source_season": int(latest_season)}
    if model_type == "catboost":
        from catboost import CatBoostClassifier
        train = X_train.copy()
        validation = X_val.copy()
        categorical = [column for column in config.get("categorical", []) if column in train]
        for column in categorical:
            train[column] = train[column].fillna("__MISSING__").astype(str)
            validation[column] = validation[column].fillna("__MISSING__").astype(str)
        seeds = [int(seed) for seed in config.get("seeds", [config.get("seed", 42)])]
        seed_predictions: list[np.ndarray] = []
        seed_results: list[dict[str, Any]] = []
        models: list[Any] = []
        for seed in seeds:
            started = time.monotonic()
            if psutil.virtual_memory().available < int(config.get("min_available_memory_bytes", 0)):
                raise MemoryError("CatBoost 학습 전 가용 메모리 하한 미달")
            model = CatBoostClassifier(
                iterations=int(config.get("iterations", 2000)), depth=7, learning_rate=0.05,
                loss_function="Logloss", eval_metric="Logloss",
                early_stopping_rounds=int(config.get("early_stopping_rounds", 100)),
                allow_writing_files=False, thread_count=8, random_seed=seed, verbose=False,
            )
            model.fit(train, y_train, cat_features=categorical, eval_set=(validation, config["y_val"]), verbose=False)
            elapsed = time.monotonic() - started
            if elapsed > float(config.get("max_model_seconds", 3600)):
                raise TimeoutError("CatBoost 모델 시간 예산 초과")
            prediction = model.predict_proba(validation)[:, 1]
            if not np.isfinite(prediction).all() or ((prediction < 0) | (prediction > 1)).any():
                raise ValueError(f"CatBoost seed={seed} 예측 유한성/범위 검사 실패")
            seed_predictions.append(prediction)
            seed_results.append({
                "seed": seed,
                "brier": float(np.mean((prediction - np.asarray(config["y_val"])) ** 2)),
                "best_iteration": int(model.get_best_iteration() + 1),
                "runtime_seconds": elapsed,
            })
            models.append(model)
        matrix = np.column_stack(seed_predictions)
        ensemble_prediction = matrix.mean(axis=1)
        return ensemble_prediction, {
            "models": models,
            "seed_predictions": matrix,
            "seed_results": seed_results,
            "ensemble_brier": float(np.mean((ensemble_prediction - np.asarray(config["y_val"])) ** 2)),
            "categorical_columns": categorical,
            "numeric_conversion_failures": {},
        }
    if model_type == "lightgbm":
        import lightgbm as lgb

        train, validation, categorical, conversion_failures = prepare_lightgbm_frames(
            X_train, X_val, list(config.get("categorical", []))
        )
        seeds = [int(seed) for seed in config.get("seeds", [config.get("seed", 42)])]
        seed_predictions: list[np.ndarray] = []
        seed_results: list[dict[str, Any]] = []
        models: list[Any] = []
        for seed in seeds:
            model_started = time.monotonic()
            model = lgb.LGBMClassifier(
                objective="binary",
                n_estimators=int(config.get("n_estimators", 3000)),
                learning_rate=float(config.get("learning_rate", 0.05)),
                num_leaves=int(config.get("num_leaves", 63)),
                min_child_samples=int(config.get("min_child_samples", 200)),
                colsample_bytree=float(config.get("colsample_bytree", 0.8)),
                subsample=float(config.get("subsample", 0.8)),
                subsample_freq=int(config.get("subsample_freq", 1)),
                random_state=seed,
                n_jobs=int(config.get("n_jobs", 8)),
                verbose=-1,
            )
            callbacks = [
                lgb.early_stopping(int(config.get("early_stopping_rounds", 100)), verbose=False),
                _resource_callback(
                    model_started,
                    float(config.get("max_model_seconds", 3600)),
                    int(config.get("min_available_memory_bytes", 0)),
                ),
            ]
            model.fit(
                train,
                y_train,
                eval_X=validation,
                eval_y=config["y_val"],
                eval_metric="binary_logloss",
                categorical_feature=categorical,
                callbacks=callbacks,
            )
            prediction = model.predict_proba(validation)[:, 1]
            if not np.isfinite(prediction).all() or ((prediction < 0) | (prediction > 1)).any():
                raise ValueError(f"LightGBM seed={seed} 예측 유한성/범위 검사 실패")
            seed_predictions.append(prediction)
            seed_results.append({
                "seed": seed,
                "brier": float(np.mean((prediction - np.asarray(config["y_val"])) ** 2)),
                "best_iteration": int(model.best_iteration_ or config.get("n_estimators", 0)),
                "runtime_seconds": time.monotonic() - model_started,
            })
            models.append(model)
        matrix = np.column_stack(seed_predictions)
        ensemble_prediction = matrix.mean(axis=1)
        return ensemble_prediction, {
            "models": models,
            "seed_predictions": matrix,
            "seed_results": seed_results,
            "ensemble_brier": float(np.mean((ensemble_prediction - np.asarray(config["y_val"])) ** 2)),
            "categorical_columns": categorical,
            "numeric_conversion_failures": conversion_failures,
        }
    if model_type == "oof_blend":
        lightgbm_prediction, lightgbm_details = fit_predict(
            "lightgbm", X_train, y_train, X_val, {**config["lightgbm"], "y_val": config["y_val"]}
        )
        catboost_prediction, catboost_details = fit_predict(
            "catboost", X_train, y_train, X_val, {**config["catboost"], "y_val": config["y_val"]}
        )
        return (lightgbm_prediction + catboost_prediction) / 2, {
            "component_predictions": {
                "lightgbm": lightgbm_prediction,
                "catboost": catboost_prediction,
            },
            "component_seed_results": {
                "lightgbm": lightgbm_details["seed_results"],
                "catboost": catboost_details["seed_results"],
            },
        }
    raise ValueError(f"지원하지 않는 model_type: {model_type}")
