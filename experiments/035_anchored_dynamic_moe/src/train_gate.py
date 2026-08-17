"""035 ADMoE의 OOF gate 학습기.

실행 전제: 모든 p_*는 해당 fold의 라벨을 보지 않고 생성된 base-model OOF여야 한다.
이 스크립트는 base model을 학습하지 않으며, 첫 검증 fold에는 과거 OOF가 없으므로
anchor를 그대로 반환한다. 이는 stacking 단계의 시간 누수를 막기 위한 의도된 제약이다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
CFG = json.loads((EXP / "config.json").read_text(encoding="utf-8"))
OOF_DIR = EXP / "outputs" / "oof"


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y - p) ** 2))


def make_gate():
    """실제 OOF gate 경로에서만 pandas/scikit-learn을 요구한다."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    cats = CFG["categorical_gate_features"]
    nums = [x for x in CFG["gate_features"] if x not in cats]
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), nums),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cats),
    ], verbose_feature_names_out=False)
    return Pipeline([("pre", pre), ("reg", HistGradientBoostingRegressor(
        max_depth=2, learning_rate=0.05, max_iter=80, min_samples_leaf=500,
        l2_regularization=5.0, random_state=42,
    ))])


def gated_prediction(anchor: np.ndarray, experts: dict[str, np.ndarray],
                     gains: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """양의 예상 개선만 비례 배분해 anchor에서 최대 30%만 이동한다."""
    positive = {name: np.maximum(value, 0.0) for name, value in gains.items()}
    total = np.sum(list(positive.values()), axis=0)
    scale = float(CFG["gain_scale"])
    strength = float(CFG["max_total_intervention"]) * total / (total + scale)
    weights = {name: np.divide(strength * value, total, out=np.zeros_like(total), where=total > 0)
               for name, value in positive.items()}
    pred = anchor.copy()
    for name, weight in weights.items():
        pred += weight * (experts[name] - anchor)
    return np.clip(pred, 0.0, 1.0), np.sum(list(weights.values()), axis=0)


def load_fold(year: int):
    import pandas as pd

    packed = np.load(OOF_DIR / f"fold_{year}.npz", allow_pickle=False)
    needed = {"row_id", "y_true", "p_anchor"}
    if not needed.issubset(packed.files):
        raise ValueError(f"{year}: OOF 필수 키 누락: {needed - set(packed.files)}")
    experts = {k.removeprefix("p_"): packed[k] for k in packed.files
               if k.startswith("p_") and k != "p_anchor"}
    if not experts:
        raise ValueError(f"{year}: p_<expert> 후보가 없음")
    feature_path = OOF_DIR / f"features_{year}.parquet"
    frame = pd.read_parquet(feature_path, columns=["row_id", *CFG["gate_features"]])
    if not np.array_equal(frame.row_id.astype(str).to_numpy(), packed["row_id"].astype(str)):
        raise ValueError(f"{year}: OOF와 feature의 row_id 순서 불일치")
    n = len(frame)
    arrays = [packed["y_true"], packed["p_anchor"], *experts.values()]
    if any(len(a) != n or not np.isfinite(a).all() for a in arrays):
        raise ValueError(f"{year}: 예측 길이 또는 유한성 오류")
    return frame, packed["y_true"].astype(float), packed["p_anchor"].astype(float), experts


def fit_segment_bias(pred: np.ndarray, anchor: np.ndarray, y: np.ndarray,
                     segment) -> dict[str, float]:
    """교정용 OOF에서만 anchor 대비 잔차 이동을 추정한다."""
    corrections: dict[str, float] = {}
    cap = float(CFG["max_segment_bias_correction"])
    values = segment.astype(str)
    for value in values.unique():
        mask = values.to_numpy() == value
        correction = np.mean(y[mask] - pred[mask]) - np.mean(y[mask] - anchor[mask])
        corrections[value] = float(np.clip(correction, -cap, cap))
    return corrections


def apply_segment_bias(pred: np.ndarray, segment,
                       corrections: dict[str, float]) -> np.ndarray:
    """미래 행에는 이미 과거 OOF로 고정한 구간별 보정만 적용한다."""
    out = pred.copy()
    for value in segment.astype(str).unique():
        mask = segment.astype(str).to_numpy() == value
        out[mask] += corrections.get(value, 0.0)
    return np.clip(out, 0.0, 1.0)


def predict_from_prior(prior_data: list[tuple],
                       target_frame, target_anchor: np.ndarray,
                       target_experts: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """지정 target보다 과거인 OOF만 써서 raw dynamic prediction을 만든다."""
    import pandas as pd

    if not prior_data:
        return target_anchor.copy(), np.zeros_like(target_anchor)
    train_x = pd.concat([v[0] for v in prior_data], ignore_index=True)
    gains: dict[str, np.ndarray] = {}
    for name in target_experts:
        if any(name not in v[3] for v in prior_data):
            raise ValueError(f"전문가 {name}가 모든 과거 OOF에 없음")
        yy = np.concatenate([v[1] for v in prior_data])
        aa = np.concatenate([v[2] for v in prior_data])
        ee = np.concatenate([v[3][name] for v in prior_data])
        model = make_gate().fit(train_x, (yy - aa) ** 2 - (yy - ee) ** 2)
        gains[name] = model.predict(target_frame)
    return gated_prediction(target_anchor, target_experts, gains)


def smoke() -> None:
    rng = np.random.default_rng(42)
    anchor = rng.uniform(.3, .7, 1000)
    experts = {"cat": np.clip(anchor + rng.normal(0, .08, 1000), 0, 1)}
    pred, total = gated_prediction(anchor, experts, {"cat": np.full(1000, .1)})
    assert np.all(total <= CFG["max_total_intervention"] + 1e-12)
    assert np.isfinite(pred).all() and np.all((0 <= pred) & (pred <= 1))
    zero, zero_total = gated_prediction(anchor, experts, {"cat": np.full(1000, -.1)})
    assert np.array_equal(zero, anchor) and np.array_equal(zero_total, np.zeros(1000))
    print("SMOKE OK: 개입량 상한, 음의 예상개선 차단, 확률 범위를 확인했습니다.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--years", default="2022,2023,2024")
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    import pandas as pd

    years = [int(x) for x in args.years.split(",")]
    loaded = {year: load_fold(year) for year in years}
    rows = []
    for i, year in enumerate(years):
        frame, y, anchor, experts = loaded[year]
        prior = years[:i]
        if not prior:
            pred, intervention = anchor.copy(), np.zeros_like(anchor)
            status = "cold_start_anchor_only"
        else:
            train = [loaded[p] for p in prior]
            pred, intervention = predict_from_prior(train, frame, anchor, experts)
            # 마지막 과거 fold는 더 이른 OOF로 만든 gate 예측만 보정 추정에 쓴다.
            if len(train) >= 2:
                cal_frame, cal_y, cal_anchor, cal_experts = train[-1]
                cal_pred, _ = predict_from_prior(train[:-1], cal_frame, cal_anchor, cal_experts)
                corrections = fit_segment_bias(cal_pred, cal_anchor, cal_y, cal_frame["game_type"])
                pred = apply_segment_bias(pred, frame["game_type"], corrections)
                status = f"gate_trained_on_{','.join(map(str, prior))}; bias_from_{prior[-1]}"
            else:
                status = f"gate_trained_on_{','.join(map(str, prior))}; bias_cold_start"
        rows.append({"year": year, "status": status, "brier_anchor": brier(y, anchor),
                     "brier_gate": brier(y, pred), "delta_brier": brier(y, anchor) - brier(y, pred),
                     "mean_intervention": float(intervention.mean()), "max_intervention": float(intervention.max())})
    out = EXP / "outputs"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "rolling_gate_metrics.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
