from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "048_count_state_hand_matchup"
EXP_046 = ROOT / "experiments" / "046_count_state_cross"
EXP_047 = ROOT / "experiments" / "047_hand_matchup_cross"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"모듈 로드 실패: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE_046 = load_module("experiment_046_for_048", EXP_046 / "run_experiment.py")
BASE_047 = load_module("experiment_047_for_048", EXP_047 / "run_experiment.py")
YEARS = (2022, 2023, 2024)
EXPECTED_915 = {2022: 0.24340969885134944, 2023: 0.25086441827051154, 2024: 0.24806077772944432}
EXPECTED_046 = {2022: 0.24339423113866543, 2023: 0.2508085713597035, 2024: 0.24804501355858163}


def brier(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((prediction - y) ** 2))


def validate_prediction(prediction: np.ndarray, rows: int, label: str) -> None:
    if len(prediction) != rows:
        raise ValueError(f"{label} 행 수 불일치")
    if not np.isfinite(prediction).all():
        raise ValueError(f"{label} NaN/inf")
    if not ((prediction >= 0) & (prediction <= 1)).all():
        raise ValueError(f"{label} 확률 범위 위반")


def validate_counts(data: pd.DataFrame) -> None:
    balls = pd.to_numeric(data["balls_before"], errors="coerce")
    strikes = pd.to_numeric(data["strikes_before"], errors="coerce")
    valid = balls.notna() & strikes.notna() & balls.between(0, 3) & strikes.between(0, 2)
    valid &= balls.eq(np.floor(balls)) & strikes.eq(np.floor(strikes))
    if not bool(valid.all()):
        bad = data.loc[~valid, ["row_id", "season", "balls_before", "strikes_before"]].head(10)
        raise ValueError(f"비정상 count 발견:\n{bad.to_string(index=False)}")
    state = (3 * balls.astype(np.int8) + strikes.astype(np.int8)).to_numpy()
    if not np.array_equal(np.unique(state), np.arange(12)):
        raise ValueError(f"count_state 고유성 계약 실패: {np.unique(state).tolist()}")


def metric_record(year: int, candidate: str, y: np.ndarray, prediction: np.ndarray, p_915: np.ndarray, p_046: np.ndarray) -> dict:
    validate_prediction(prediction, len(y), candidate)
    value = brier(y, prediction)
    return {
        "validation_year": year,
        "candidate": candidate,
        "rows": len(y),
        "brier": value,
        "delta_vs_915": value - brier(y, p_915),
        "delta_vs_046": value - brier(y, p_046),
        "prediction_mean": float(prediction.mean()),
        "target_rate": float(y.mean()),
        "prediction_bias": float(prediction.mean() - y.mean()),
        "prediction_std": float(prediction.std()),
        "auc": float(roc_auc_score(y, prediction)),
        "logloss": float(log_loss(y, prediction)),
        "probability_min": float(prediction.min()),
        "probability_max": float(prediction.max()),
        "nan_count": int(np.isnan(prediction).sum()),
        "inf_count": int(np.isinf(prediction).sum()),
        "row_id_duplicates": 0,
    }


def load_full_assets(data: pd.DataFrame, mlp: pd.DataFrame, year: int) -> dict[str, np.ndarray]:
    validation = data[data.season.eq(year)].reset_index(drop=True)
    row_id = validation.row_id.astype(str).to_numpy()
    y = validation[BASE_046.TARGET].astype(np.int8).to_numpy()
    z46 = np.load(EXP_046 / "outputs" / f"fold_{year}.npz", allow_pickle=True)
    z47 = np.load(EXP_047 / "outputs" / f"fold_{year}.npz", allow_pickle=True)
    mlp_fold = mlp[mlp.fold.eq(year)].reset_index(drop=True)
    for label, saved in (("046", z46), ("047", z47)):
        if not np.array_equal(saved["row_id"].astype(str), row_id) or not np.array_equal(saved["y_true"], y):
            raise ValueError(f"{label} row_id/fold/target 불일치: {year}")
    if not np.array_equal(z46["row_id"].astype(str), z47["row_id"].astype(str)) or not np.array_equal(z46["y_true"], z47["y_true"]):
        raise ValueError(f"046·047 상호 정렬 실패: {year}")
    if not np.array_equal(mlp_fold.row_id.astype(str).to_numpy(), row_id) or not np.array_equal(mlp_fold.target.to_numpy(), y):
        raise ValueError(f"MLP 정렬 실패: {year}")
    if len(np.unique(row_id)) != len(row_id):
        raise ValueError(f"row_id 중복: {year}")
    result = {
        "row_id": row_id,
        "y": y,
        "p_915": z46["p_915_existing"],
        "p_046": z46["p_915_count_state"],
        "p_047": z47["p_915_count_hand"],
        "p_mlp": mlp_fold.p_mlp_real_890.to_numpy(),
    }
    for key in ("p_915", "p_046", "p_047", "p_mlp"):
        validate_prediction(result[key], len(y), f"{key}_{year}")
    if abs(brier(y, result["p_915"]) - EXPECTED_915[year]) > 1e-12:
        raise ValueError(f"915 Brier 재현 실패: {year}")
    if abs(brier(y, result["p_046"]) - EXPECTED_046[year]) > 1e-12:
        raise ValueError(f"046 Brier 재현 실패: {year}")
    return result


def aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, group in metrics.groupby("candidate", sort=False):
        weights = group.rows.to_numpy()
        rows.append({
            "candidate": candidate,
            "rows": int(group.rows.sum()),
            "brier": float(np.average(group.brier, weights=weights)),
            "delta_vs_915": float(np.average(group.delta_vs_915, weights=weights)),
            "delta_vs_046": float(np.average(group.delta_vs_046, weights=weights)),
            "prediction_mean": float(np.average(group.prediction_mean, weights=weights)),
            "target_rate": float(np.average(group.target_rate, weights=weights)),
            "prediction_bias": float(np.average(group.prediction_bias, weights=weights)),
            "auc_macro": float(group.auc.mean()),
            "logloss": float(np.average(group.logloss, weights=weights)),
            "probability_min": float(group.probability_min.min()),
            "probability_max": float(group.probability_max.max()),
            "nan_count": int(group.nan_count.sum()),
            "inf_count": int(group.inf_count.sum()),
            "row_id_duplicates": int(group.row_id_duplicates.sum()),
            "worst_fold_brier": float(group.brier.max()),
        })
    return pd.DataFrame(rows)


def acceptance(candidate: str, metrics: pd.DataFrame, overall: pd.DataFrame, config: dict) -> dict:
    current = metrics[metrics.candidate.eq(candidate)].set_index("validation_year")
    overall_row = overall[overall.candidate.eq(candidate)].iloc[0]
    overall_046 = float(overall[overall.candidate.eq("count_state_046")].iloc[0].brier)
    overall_915 = float(overall[overall.candidate.eq("baseline_915")].iloc[0].brier)
    improvement_vs_915 = overall_915 - float(overall_row.brier)
    checks = {
        "2022_non_worse_vs_915": float(current.loc[2022, "delta_vs_915"]) <= config["acceptance"]["max_2022_delta_vs_915"],
        "2023_delta_vs_915_at_most_0_00001": float(current.loc[2023, "delta_vs_915"]) <= config["acceptance"]["max_2023_delta_vs_915"],
        "2024_non_worse_vs_915": float(current.loc[2024, "delta_vs_915"]) <= config["acceptance"]["max_2024_delta_vs_915"],
        "overall_below_046": float(overall_row.brier) < overall_046,
        "overall_improvement_exceeds_046": improvement_vs_915 > config["acceptance"]["must_exceed_046_improvement"],
        "integrity_checks": int(overall_row.nan_count) == 0 and int(overall_row.inf_count) == 0 and int(overall_row.row_id_duplicates) == 0,
    }
    return {
        "candidate": candidate,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "overall_brier": float(overall_row.brier),
        "improvement_vs_915": improvement_vs_915,
        "worst_fold_brier": float(overall_row.worst_fold_brier),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads((EXP / "config.json").read_text(encoding="utf-8"))
    if config["candidate_b"] != {"name": "fixed_046_047_blend", "weight_046": 0.5, "weight_047": 0.5}:
        raise ValueError("후보 B 고정 50:50 계약 위반")
    output = EXP / ("smoke" if args.smoke else "outputs")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"기존 결과 보호: {output}")
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(BASE_046.TRAIN_PATH, usecols=["row_id", *BASE_046.FEATURES, BASE_046.TARGET], low_memory=False)
    validate_counts(data)
    mlp = pd.read_csv(BASE_046.MLP_PATH)
    seeds = [42] if args.smoke else config["seeds"]
    if args.smoke:
        data = pd.concat([part.head(3000) for _, part in data.groupby("season", sort=True)], ignore_index=True)
    started_all = time.perf_counter()
    metric_rows = []
    fold_audit = {}
    for year in YEARS:
        if time.perf_counter() - started_all > config["max_runtime_seconds"]:
            (output / "checkpoint_timeout.json").write_text(json.dumps({"completed_years": list(fold_audit), "runtime_seconds": time.perf_counter()-started_all}, indent=2), encoding="utf-8")
            raise TimeoutError("1시간 제한 초과")
        train = data[data.season.isin(BASE_046.OUTER_TRAIN[year])]
        validation = data[data.season.eq(year)].reset_index(drop=True)
        y = validation[BASE_046.TARGET].astype(np.int8).to_numpy()
        if args.smoke:
            mean = float(train[BASE_046.TARGET].mean())
            assets = {"row_id": validation.row_id.astype(str).to_numpy(), "y": y, "p_915": np.full(len(y), mean), "p_046": np.full(len(y), mean), "p_047": np.full(len(y), mean), "p_mlp": np.full(len(y), mean)}
        else:
            assets = load_full_assets(data, mlp, year)
        p_lgb_a, audit = BASE_047.train_candidate(train, validation, seeds)
        p_pre_a = config["candidate_a"]["lgb_weight"] * p_lgb_a + config["candidate_a"]["mlp_weight"] * assets["p_mlp"]
        p_a = BASE_046.sigmoid(BASE_046.logit(p_pre_a) + config["candidate_a"]["logit_shift"])
        p_b = config["candidate_b"]["weight_046"] * assets["p_046"] + config["candidate_b"]["weight_047"] * assets["p_047"]
        for label, prediction in (("baseline_915", assets["p_915"]), ("count_state_046", assets["p_046"]), ("hand_matchup_047", assets["p_047"]), ("candidate_a_joint", p_a), ("candidate_b_fixed_blend", p_b)):
            metric_rows.append(metric_record(year, label, y, prediction, assets["p_915"], assets["p_046"]))
        reproduction_error = float(np.max(np.abs(p_a - assets["p_047"]))) if not args.smoke else None
        fold_audit[str(year)] = {"candidate_a_vs_047_max_abs": reproduction_error, "training": audit}
        np.savez_compressed(output / f"fold_{year}.npz", row_id=assets["row_id"], y_true=y, p_915=assets["p_915"], p_046=assets["p_046"], p_047=assets["p_047"], p_candidate_a=p_a, p_candidate_b=p_b)
        print(f"year={year} A={brier(y,p_a):.12f} B={brier(y,p_b):.12f} A_vs_047={reproduction_error}", flush=True)
    metrics = pd.DataFrame(metric_rows)
    overall = aggregate(metrics)
    metrics.to_csv(output / "metrics_by_fold.csv", index=False)
    overall.to_csv(output / "metrics_overall.csv", index=False)
    a_result = acceptance("candidate_a_joint", metrics, overall, config)
    b_result = acceptance("candidate_b_fixed_blend", metrics, overall, config)
    if a_result["passed"] and b_result["passed"]:
        selected = min((a_result, b_result), key=lambda x: (x["overall_brier"], x["worst_fold_brier"]))
        decision = "ACCEPT_JOINT" if selected["candidate"] == "candidate_a_joint" else "ACCEPT_FIXED_BLEND"
    elif a_result["passed"]:
        decision = "ACCEPT_JOINT"
    elif b_result["passed"]:
        decision = "ACCEPT_FIXED_BLEND"
    else:
        decision = "KEEP_046"
    runtime = time.perf_counter() - started_all
    if runtime > config["max_runtime_seconds"]:
        decision = "BLOCKED"
    report = {"decision": decision, "candidate_a": a_result, "candidate_b": b_result, "fold_audit": fold_audit, "runtime_seconds": runtime, "smoke": args.smoke, "environment": {"python": platform.python_version(), "lightgbm": lgb.__version__, "numpy": np.__version__, "pandas": pd.__version__}}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.smoke:
        by = metrics.set_index(["candidate", "validation_year"])
        lines = ["# 048 Count-state + hand-matchup 결합 결과", "", f"## 최종 판정: **{decision}**", "", "## 연도별 Brier", "", "| 후보 | 2022 | 2023 | 2024 | 전체 |", "| --- | ---: | ---: | ---: | ---: |"]
        for candidate in ("baseline_915", "count_state_046", "hand_matchup_047", "candidate_a_joint", "candidate_b_fixed_blend"):
            total = float(overall[overall.candidate.eq(candidate)].iloc[0].brier)
            lines.append(f"| {candidate} | {by.loc[(candidate,2022),'brier']:.12f} | {by.loc[(candidate,2023),'brier']:.12f} | {by.loc[(candidate,2024),'brier']:.12f} | {total:.12f} |")
        lines += ["", "## 판정", "", f"- 후보 A: `{'PASS' if a_result['passed'] else 'FAIL'}`", f"- 후보 B: `{'PASS' if b_result['passed'] else 'FAIL'}`", f"- 실행시간: `{runtime:.3f}초`", "", "## 해석/가설", "", "- 후보 A는 047 정의를 변경하지 않고 full K10으로 다시 학습했다.", "- 후보 B는 동일 최종 확률 단계의 046·047 OOF를 정확히 50:50 결합했다.", "", "## 다음 행동", "", "- 판정에 따라 선택 후보를 challenger로 보존하며 추가 weight·feature·parameter 탐색은 하지 않는다.", ""]
        (EXP / "RESULT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
