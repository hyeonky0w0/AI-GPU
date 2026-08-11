from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def training_sample_weights(X_train: pd.DataFrame, policy: str) -> pd.Series | None:
    """검증 정보 없이 학습 시즌만으로 사전 고정 sample weight를 만든다."""
    if policy == "none":
        return None
    if policy != "latest_season_double":
        raise ValueError(f"지원하지 않는 sample weight 정책: {policy}")
    if "season" not in X_train or X_train["season"].isna().any():
        raise ValueError("최근 시즌 가중치에는 결측 없는 season 컬럼이 필요합니다.")
    latest_season = X_train["season"].max()
    weights = pd.Series(1.0, index=X_train.index, dtype=float)
    weights.loc[X_train["season"] == latest_season] = 2.0
    return weights


def apply_training_window_policy(split: dict[str, Any], policy: str) -> dict[str, Any]:
    """검증 시즌을 건드리지 않고 fold의 학습 시즌 범위만 제한한다."""
    prepared = {**split, "train_seasons": list(split["train_seasons"])}
    if policy == "recent_two_seasons":
        past_seasons = sorted(
            season for season in prepared["train_seasons"]
            if season < prepared["validation_season"]
        )
        if len(past_seasons) < 2:
            raise ValueError("최근 2개 시즌 정책에 과거 학습 시즌이 두 개 이상 필요합니다.")
        prepared["train_seasons"] = past_seasons[-2:]
    elif policy != "all":
        raise ValueError(f"지원하지 않는 training window policy: {policy}")
    return prepared


def feature_columns(train_path: Path, test_path: Path, target: str, identifier: str) -> list[str]:
    train_columns = pd.read_csv(train_path, nrows=0, encoding="utf-8-sig").columns.tolist()
    test_columns = pd.read_csv(test_path, nrows=0, encoding="utf-8-sig").columns.tolist()
    if target not in train_columns or target in test_columns or identifier not in test_columns:
        raise ValueError("train/test 필수 컬럼 계약이 맞지 않습니다.")
    missing = [column for column in test_columns if column not in train_columns]
    if missing:
        raise ValueError(f"test 피처가 train에 없습니다: {missing}")
    return [column for column in test_columns if column != identifier]


def load_smoke_rows(
    train_path: Path,
    usecols: list[str],
    splits: list[dict[str, Any]],
    train_limit: int,
    validation_limit: int,
) -> pd.DataFrame:
    needed = sorted({season for split in splits for season in split["train_seasons"]} | {s["validation_season"] for s in splits})
    per_season_limit = max(train_limit, validation_limit)
    parts: list[pd.DataFrame] = []
    counts = {season: 0 for season in needed}
    for chunk in pd.read_csv(train_path, usecols=usecols, chunksize=50_000, low_memory=False, encoding="utf-8-sig"):
        for season in needed:
            remaining = per_season_limit - counts[season]
            if remaining > 0:
                part = chunk.loc[chunk["season"].eq(season)].head(remaining)
                if not part.empty:
                    parts.append(part)
                    counts[season] += len(part)
        if all(value >= per_season_limit for value in counts.values()):
            break
    missing = [season for season, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"smoke에 필요한 시즌 데이터가 없습니다: {missing}")
    return pd.concat(parts, ignore_index=True)


def select_fold(
    data: pd.DataFrame,
    split: dict[str, Any],
    features: list[str],
    target: str,
    smoke: bool,
    train_limit: int,
    validation_limit: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    train = data.loc[data["season"].isin(split["train_seasons"])]
    validation = data.loc[data["season"].eq(split["validation_season"])]
    if smoke:
        per_season = max(1, train_limit // len(split["train_seasons"]))
        train = pd.concat(
            [train.loc[train["season"].eq(season)].head(per_season) for season in split["train_seasons"]],
            ignore_index=True,
        ).head(train_limit)
        validation = validation.head(validation_limit)
    if train.empty or validation.empty:
        raise ValueError(f"빈 rolling fold: {split['name']}")
    return (
        train.loc[:, features].copy(), train[target].astype(int).to_numpy(),
        validation.loc[:, features].copy(), validation[target].astype(int).to_numpy(),
    )


def group_metrics_frame(validation: pd.DataFrame, y: np.ndarray, prediction: np.ndarray) -> pd.DataFrame:
    frame = validation.copy()
    frame["_target"] = y
    frame["_prediction"] = prediction
    frame["_squared_error"] = (prediction - y) ** 2
    frame["prediction_bin"] = pd.cut(prediction, bins=np.linspace(0, 1, 11), include_lowest=True).astype(str)
    groups = ["season", "game_month", "inning", "base_state", "balls_before", "strikes_before", "pitcher_id", "batter_id", "prediction_bin"]
    records: list[dict[str, Any]] = []
    for column in groups:
        if column not in frame:
            continue
        summary = frame.groupby(column, dropna=False).agg(
            rows=("_target", "size"), brier=("_squared_error", "mean"),
            target_rate=("_target", "mean"), prediction_mean=("_prediction", "mean"),
        ).reset_index()
        for row in summary.itertuples(index=False):
            records.append({
                "group_type": column, "group_value": str(row[0]), "rows": int(row.rows),
                "brier": float(row.brier), "target_rate": float(row.target_rate),
                "prediction_mean": float(row.prediction_mean), "unstable_small_group": int(row.rows) < 100,
            })
    return pd.DataFrame(records)


def population_stability_index(reference: pd.Series, comparison: pd.Series, bins: int = 10) -> float:
    """reference에서 bin을 만들고 두 분포의 PSI를 계산한다."""
    reference = pd.to_numeric(reference, errors="coerce")
    comparison = pd.to_numeric(comparison, errors="coerce")
    valid = reference.dropna()
    if valid.nunique() < 2:
        return 0.0
    edges = np.unique(valid.quantile(np.linspace(0, 1, bins + 1)).to_numpy())
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_bins = pd.cut(reference, bins=edges, include_lowest=True)
    cmp_bins = pd.cut(comparison, bins=edges, include_lowest=True)
    categories = ref_bins.cat.categories
    ref_dist = ref_bins.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    cmp_dist = cmp_bins.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    epsilon = 1e-6
    ref_dist = np.clip(ref_dist, epsilon, None)
    cmp_dist = np.clip(cmp_dist, epsilon, None)
    return float(np.sum((cmp_dist - ref_dist) * np.log(cmp_dist / ref_dist)))


def training_only_psi_drop_columns(X_train: pd.DataFrame, top_n: int = 5) -> tuple[list[str], dict[str, float]]:
    """validation을 보지 않고 학습기간의 과거 시즌과 최신 시즌만 비교한다."""
    if "season" not in X_train:
        raise ValueError("PSI 계산에 season 컬럼이 필요합니다.")
    seasons = sorted(X_train["season"].dropna().unique().tolist())
    if len(seasons) < 2:
        raise ValueError("PSI 계산에 서로 다른 학습 시즌이 두 개 이상 필요합니다.")
    latest = seasons[-1]
    reference = X_train.loc[X_train["season"].isin(seasons[:-1])]
    comparison = X_train.loc[X_train["season"].eq(latest)]
    scores: dict[str, float] = {}
    for column in X_train.columns:
        if column == "season" or not pd.api.types.is_numeric_dtype(X_train[column].dtype):
            continue
        scores[column] = population_stability_index(reference[column], comparison[column])
    dropped = [column for column, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_n]]
    return dropped, scores


def apply_feature_policy(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, float]]:
    if X_train.columns.tolist() != X_val.columns.tolist():
        raise ValueError("피처 정책 적용 전 순서 불일치")
    dropped: list[str] = []
    psi_scores: dict[str, float] = {}
    if policy == "drop_season":
        dropped = ["season"] if "season" in X_train else []
    elif policy == "drop_player_ids":
        dropped = [column for column in ("pitcher_id", "batter_id") if column in X_train]
    elif policy == "training_only_psi_top":
        dropped, psi_scores = training_only_psi_drop_columns(X_train, top_n=5)
    elif policy != "all":
        raise ValueError(f"지원하지 않는 feature policy: {policy}")
    features = [column for column in X_train.columns if column not in dropped]
    return X_train.loc[:, features].copy(), X_val.loc[:, features].copy(), dropped, psi_scores
