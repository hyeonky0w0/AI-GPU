from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from core import (
    REGISTRY_FIELDS, ResearchBudget, aggregate_fold_metrics, append_registry_atomic,
    atomic_write_json, atomic_write_text, calculate_metrics, champion_decision, config_hash, now_iso,
    derive_stage_state, migrate_registry_schema, read_registry, sha256_file, static_leakage_check, successful_hashes,
    is_resume_candidate, write_failure_artifacts,
)
from data_pipeline import (
    apply_feature_policy, apply_training_window_policy, feature_columns, group_metrics_frame, load_smoke_rows, select_fold,
)
from models import fit_predict, select_nonnegative_oof_weight


ROOT = Path(__file__).resolve().parents[2]
AUTOMATION = ROOT / "automation"
CONFIG_PATH = AUTOMATION / "config.json"
REGISTRY_DIR = AUTOMATION / "registry"
RUNS_DIR = AUTOMATION / "runs"
REPORTS_DIR = AUTOMATION / "reports"
LOGS_DIR = AUTOMATION / "logs"
CATALOG_PATH = AUTOMATION / "catalog" / "hypotheses.json"
REGISTRY_CSV = REGISTRY_DIR / "EXPERIMENT_REGISTRY.csv"
REGISTRY_MD = REGISTRY_DIR / "EXPERIMENT_REGISTRY.md"
CHAMPION_PATH = REGISTRY_DIR / "CHAMPION.json"
QUEUE_PATH = REGISTRY_DIR / "HYPOTHESIS_QUEUE.json"
LEADERBOARD_PATH = REGISTRY_DIR / "LEADERBOARD_SCORES.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["check", "smoke", "research", "resume", "report", "record-score"])
    parser.add_argument("--max-experiments", type=int)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--experiment-id")
    parser.add_argument("--leaderboard-score", type=float)
    parser.add_argument("--score-type", choices=["public", "private", "unknown"], default="unknown")
    parser.add_argument("--notes", default="")
    parser.add_argument("--candidate-id")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
        return completed.stdout.strip()
    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD") or None, "branch": run("branch", "--show-current") or None, "dirty": bool(status)}


def new_run_id(mode: str) -> str:
    base = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{mode}"
    candidate = base
    index = 1
    while (RUNS_DIR / candidate).exists():
        index += 1
        candidate = f"{base}_{index}"
    return candidate


def setup_logger(run_id: str, run_dir: Path) -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(run_id)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for path in [run_dir / "run.log", LOGS_DIR / f"{run_id}.log"]:
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    return logger


def write_run_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    """호환 이름 두 개를 항상 원자적으로 저장한다."""
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    atomic_write_json(run_dir / "manifest.json", manifest)


def paths_from_config(config: dict[str, Any]) -> tuple[Path, Path]:
    return ROOT / config["data"]["train_path"], ROOT / config["data"]["test_path"]


def environment_snapshot(config: dict[str, Any], include_checksum: bool = True) -> dict[str, Any]:
    train_path, test_path = paths_from_config(config)
    disk = psutil.disk_usage(str(ROOT))
    memory = psutil.virtual_memory()
    libraries = {name: package_version(name) for name in ["numpy", "pandas", "scikit-learn", "catboost", "lightgbm", "xgboost", "psutil"]}
    result = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "libraries": libraries,
        "cpu_logical": psutil.cpu_count(logical=True),
        "cpu_physical": psutil.cpu_count(logical=False),
        "memory_total_bytes": memory.total,
        "memory_available_bytes": memory.available,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "gpu": "nvidia-smi unavailable; GPU not confirmed",
        "git": git_info(),
        "data": {
            "train_path": str(train_path), "test_path": str(test_path),
            "train_size": train_path.stat().st_size, "test_size": test_path.stat().st_size,
        },
    }
    if include_checksum:
        result["data"]["train_sha256"] = sha256_file(train_path)
        result["data"]["test_sha256"] = sha256_file(test_path)
    return result


def data_summary(config: dict[str, Any]) -> dict[str, Any]:
    train_path, test_path = paths_from_config(config)
    target, season = config["target"], config["season_column"]
    columns = feature_columns(train_path, test_path, target, config["id_column"])
    counts: dict[str, dict[str, float]] = {}
    total_rows = 0
    for chunk in pd.read_csv(train_path, usecols=[season, target], chunksize=100_000, encoding="utf-8-sig"):
        total_rows += len(chunk)
        grouped = chunk.groupby(season)[target].agg(["size", "sum"])
        for year, row in grouped.iterrows():
            item = counts.setdefault(str(int(year)), {"rows": 0, "positives": 0})
            item["rows"] += int(row["size"])
            item["positives"] += int(row["sum"])
    for item in counts.values():
        item["target_rate"] = item["positives"] / item["rows"]
    header = pd.read_csv(train_path, nrows=100, encoding="utf-8-sig")
    duplicate_sample = int(header[config["id_column"]].duplicated().sum())
    return {
        "train_rows": total_rows, "test_rows_sample": sum(1 for _ in test_path.open("r", encoding="utf-8-sig")) - 1,
        "feature_count": len(columns), "features": columns, "season_summary": counts,
        "sample_duplicate_row_ids": duplicate_sample,
    }


def check_system(config: dict[str, Any], logger: logging.Logger) -> dict[str, Any]:
    train_path, test_path = paths_from_config(config)
    errors: list[str] = []
    warnings: list[str] = []
    if not train_path.exists() or not test_path.exists():
        errors.append("train.csv 또는 test.csv 누락")
    if Path(sys.executable).resolve() != (ROOT / ".venv" / "Scripts" / "python.exe").resolve():
        warnings.append("프로젝트 .venv가 아닌 Python 사용")
    environment = environment_snapshot(config)
    summary = data_summary(config)
    leakage = static_leakage_check(summary["features"], config["target"], config["id_column"], config["rolling_splits"])
    errors.extend(leakage["violations"])
    warnings.extend(leakage["warnings"])
    if environment["libraries"]["lightgbm"] is None:
        warnings.append("LightGBM 미설치: 기존 001 모델/코드는 현재 환경에서 재현 불가")
    if environment["libraries"]["xgboost"] is None:
        warnings.append("XGBoost 미설치: 후보 자동 생략")
    free_gb = environment["disk_free_bytes"] / 1024**3
    available_gb = environment["memory_available_bytes"] / 1024**3
    if free_gb < config["limits"]["min_disk_free_gb"]:
        errors.append(f"디스크 여유 부족: {free_gb:.1f} GB")
    if available_gb < config["limits"]["memory_warning_available_gb"]:
        warnings.append(f"가용 메모리 경고: {available_gb:.1f} GB")
    logger.info("check: rows=%s features=%s errors=%s warnings=%s", summary["train_rows"], summary["feature_count"], len(errors), len(warnings))
    return {"passed": not errors, "errors": errors, "warnings": warnings, "environment": environment, "data_summary": summary, "leakage": leakage}


def candidate_config(candidate: dict[str, Any], config: dict[str, Any], data_checksum: str, stage: str) -> dict[str, Any]:
    model_type = candidate.get("model_type", "unsupported")
    model_parameters: dict[str, Any] = {"seed": config["smoke"]["seed"]}
    if model_type == "logistic":
        model_parameters["max_iter"] = config["smoke"]["max_iter"] if stage == "smoke" else 500
    if model_type == "catboost":
        model_parameters.update({
            "iterations": 20 if stage == "smoke" else 2000,
            "depth": 7, "learning_rate": 0.05, "early_stopping_rounds": 100,
            "categorical": ["top_bottom", "game_type", "base_state"],
            "seeds": config["limits"]["seeds"] if candidate["id"] == "catboost_seed_ensemble" else [config["smoke"]["seed"]],
            "max_model_seconds": config["limits"]["max_model_minutes"] * 60,
            "min_available_memory_bytes": int(config["limits"]["memory_stop_available_gb"] * 1024**3),
        })
    if model_type == "lightgbm":
        model_parameters.update({
            "seeds": config["limits"]["seeds"],
            "n_estimators": 30 if stage == "smoke" else 3000,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 20 if stage == "smoke" else 200,
            "colsample_bytree": 0.8,
            "subsample": 0.8,
            "subsample_freq": 1,
            "early_stopping_rounds": 10 if stage == "smoke" else 100,
            "categorical": ["top_bottom", "game_type", "base_state"],
            "n_jobs": 8,
            "max_model_seconds": config["limits"]["max_model_minutes"] * 60,
            "min_available_memory_bytes": int(config["limits"]["memory_stop_available_gb"] * 1024**3),
        })
    if model_type == "oof_blend":
        common_resource = {
            "max_model_seconds": config["limits"]["max_model_minutes"] * 60,
            "min_available_memory_bytes": int(config["limits"]["memory_stop_available_gb"] * 1024**3),
        }
        model_parameters.update({
            "lightgbm": {
                "seed": config["smoke"]["seed"], "seeds": [config["smoke"]["seed"]],
                "n_estimators": 30 if stage == "smoke" else 3000, "learning_rate": 0.05,
                "num_leaves": 63, "min_child_samples": 20 if stage == "smoke" else 200,
                "colsample_bytree": 0.8, "subsample": 0.8, "subsample_freq": 1,
                "early_stopping_rounds": 10 if stage == "smoke" else 100,
                "categorical": ["top_bottom", "game_type", "base_state"], "n_jobs": 8,
                **common_resource,
            },
            "catboost": {
                "seed": config["smoke"]["seed"], "seeds": [config["smoke"]["seed"]],
                "iterations": 20 if stage == "smoke" else 2000,
                "early_stopping_rounds": 100,
                "categorical": ["top_bottom", "game_type", "base_state"],
                **common_resource,
            },
        })
    return {
        "data_checksum": data_checksum,
        "split": config["rolling_splits"],
        "feature_definition": {
            "source": "test.csv", "exclude": [config["id_column"]],
            "candidate": candidate["id"], "policy": candidate.get("feature_policy", "all"),
        },
        "training_window_policy": candidate.get("training_window_policy", "all"),
        "model": model_type,
        "model_parameters": model_parameters,
        "seed": model_parameters["seed"],
        "sample_weight": None,
        "calibration": None,
        "ensemble": None,
        "stage": stage,
    }


def run_experiment(candidate: dict[str, Any], config: dict[str, Any], run_id: str, run_dir: Path, stage: str, logger: logging.Logger) -> dict[str, Any]:
    train_path, test_path = paths_from_config(config)
    base_features = feature_columns(train_path, test_path, config["target"], config["id_column"])
    leakage = static_leakage_check(base_features, config["target"], config["id_column"], config["rolling_splits"])
    if not leakage["passed"]:
        raise ValueError(f"정적 누출 검사 실패: {leakage['violations']}")
    checksum = sha256_file(train_path)
    experiment_config = candidate_config(candidate, config, checksum, stage)
    digest = config_hash(experiment_config)
    if digest in successful_hashes(read_registry(REGISTRY_CSV)):
        return {"status": "skipped", "reason": "duplicate_config_hash", "config_hash": digest}
    experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{candidate['id']}_{stage}"
    experiment_dir = run_dir / "experiments" / experiment_id
    prediction_dir = experiment_dir / "predictions"
    model_dir = experiment_dir / "models"
    prediction_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    atomic_write_json(experiment_dir / "config.json", experiment_config)
    atomic_write_text(experiment_dir / "hypothesis.md", f"# {candidate['title']}\n\n- 가설: {candidate['reason']}\n- 누출 위험: {candidate['leakage_risk']}\n")
    started = time.monotonic()
    try:
        if stage == "smoke":
            data = load_smoke_rows(
                train_path, base_features + [config["target"]], config["rolling_splits"],
                config["limits"]["smoke_train_rows_per_fold"], config["limits"]["smoke_validation_rows_per_fold"],
            )
        else:
            data = pd.read_csv(train_path, usecols=base_features + [config["target"]], low_memory=False, encoding="utf-8-sig")
        active_splits = config["rolling_splits"] if stage != "quick" else [config["rolling_splits"][-1]]
        fold_results: list[dict[str, Any]] = []
        group_frames: list[pd.DataFrame] = []
        past_oof_y: list[np.ndarray] = []
        past_oof_lightgbm: list[np.ndarray] = []
        past_oof_catboost: list[np.ndarray] = []
        for split in active_splits:
            effective_split = apply_training_window_policy(
                split, candidate.get("training_window_policy", "all")
            )
            X_train, y_train, X_val, y_val = select_fold(
                data, effective_split, base_features, config["target"], stage == "smoke",
                config["limits"]["smoke_train_rows_per_fold"], config["limits"]["smoke_validation_rows_per_fold"],
            )
            feature_policy = candidate.get("feature_policy", "all")
            X_train, X_val, dropped_features, psi_scores = apply_feature_policy(X_train, X_val, feature_policy)
            model_features = X_train.columns.tolist()
            if X_train.columns.tolist() != X_val.columns.tolist():
                raise ValueError("후보 전처리 후 train/validation 피처 순서 불일치")
            model_config = dict(experiment_config["model_parameters"])
            model_config["y_val"] = y_val
            fold_started = time.monotonic()
            prediction, model_details = fit_predict(candidate["model_type"], X_train, y_train, X_val, model_config)
            blend_details: dict[str, Any] | None = None
            if candidate["model_type"] == "oof_blend":
                components = model_details["component_predictions"]
                if past_oof_y:
                    weight, weight_fit_brier = select_nonnegative_oof_weight(
                        np.concatenate(past_oof_y),
                        np.concatenate(past_oof_lightgbm),
                        np.concatenate(past_oof_catboost),
                    )
                    weight_source = "previous_folds_oof"
                else:
                    weight, weight_fit_brier, weight_source = 0.5, None, "no_previous_oof_equal_weight"
                prediction = weight * components["lightgbm"] + (1 - weight) * components["catboost"]
                blend_details = {
                    "lightgbm_weight": weight,
                    "catboost_weight": 1 - weight,
                    "weight_source": weight_source,
                    "weight_fit_brier": weight_fit_brier,
                    "prediction_correlation": float(np.corrcoef(components["lightgbm"], components["catboost"])[0, 1]),
                    "lightgbm_brier": float(np.mean((components["lightgbm"] - y_val) ** 2)),
                    "catboost_brier": float(np.mean((components["catboost"] - y_val) ** 2)),
                }
                past_oof_y.append(y_val.copy())
                past_oof_lightgbm.append(components["lightgbm"].copy())
                past_oof_catboost.append(components["catboost"].copy())
            metrics = calculate_metrics(y_val, prediction, float(y_train.mean()))
            fold_result = {
                "fold": effective_split["name"], "train_seasons": effective_split["train_seasons"],
                "validation_season": effective_split["validation_season"],
                "train_rows": len(y_train), "validation_rows": len(y_val), "train_target_rate": float(y_train.mean()),
                "runtime_seconds": time.monotonic() - fold_started, "metrics": metrics,
                "feature_count": len(model_features), "features": model_features,
                "dropped_features": dropped_features, "psi_scores": psi_scores,
            }
            if isinstance(model_details, dict) and model_details.get("seed_results"):
                fold_result["seed_results"] = model_details["seed_results"]
                fold_result["seed_brier_mean"] = float(np.mean([item["brier"] for item in model_details["seed_results"]]))
                fold_result["ensemble_brier"] = model_details["ensemble_brier"]
                fold_result["categorical_columns"] = model_details["categorical_columns"]
                fold_result["numeric_conversion_failures"] = model_details["numeric_conversion_failures"]
            if blend_details is not None:
                fold_result["blend"] = blend_details
            fold_results.append(fold_result)
            predictions = pd.DataFrame({"validation_season": split["validation_season"], "y_true": y_val, "prediction": prediction})
            if isinstance(model_details, dict) and model_details.get("seed_results"):
                for index, item in enumerate(model_details["seed_results"]):
                    seed = item["seed"]
                    predictions[f"prediction_seed_{seed}"] = model_details["seed_predictions"][:, index]
            if candidate["model_type"] == "oof_blend":
                predictions["prediction_lightgbm"] = model_details["component_predictions"]["lightgbm"]
                predictions["prediction_catboost"] = model_details["component_predictions"]["catboost"]
            predictions.to_csv(prediction_dir / f"{split['name']}.csv", index=False)
            group_frame = group_metrics_frame(X_val, y_val, prediction)
            group_frame.insert(0, "validation_season", split["validation_season"])
            group_frames.append(group_frame)
            logger.info("experiment=%s fold=%s brier=%.8f baseline=%.8f", experiment_id, split["name"], metrics["brier"], metrics["baseline_brier"])
        aggregate = aggregate_fold_metrics(fold_results, config["recent_season_weights"])
        all_seed_briers = [item["brier"] for fold in fold_results for item in fold.get("seed_results", [])]
        result = {
            "experiment_id": experiment_id, "run_id": run_id, "status": "completed", "stage": stage,
            "candidate": candidate, "config_hash": digest, "leakage": leakage, "folds": fold_results,
            "aggregate": aggregate, "runtime_seconds": time.monotonic() - started,
            "seed_brier_mean": float(np.mean(all_seed_briers)) if all_seed_briers else aggregate["mean_brier"],
            "seed_prediction_ensemble_brier": aggregate["mean_brier"] if all_seed_briers else None,
            "note": "seed별 Brier 평균과 seed 예측 평균 앙상블 Brier를 별도 기록",
        }
        pd.DataFrame([{**{"fold": f["fold"], "validation_season": f["validation_season"], "train_rows": f["train_rows"], "validation_rows": f["validation_rows"], "seed_brier_mean": f.get("seed_brier_mean"), "ensemble_brier": f.get("ensemble_brier"), "seed_results": json.dumps(f.get("seed_results", []), ensure_ascii=False)}, **f["metrics"]} for f in fold_results]).to_csv(experiment_dir / "results.csv", index=False, encoding="utf-8-sig")
        pd.concat(group_frames, ignore_index=True).to_csv(experiment_dir / "group_metrics.csv", index=False, encoding="utf-8-sig")
        atomic_write_json(experiment_dir / "result.json", result)
        atomic_write_text(experiment_dir / "RESULT.md", render_experiment_result(result))
        append_registry_atomic(REGISTRY_CSV, registry_row(result, experiment_dir))
        return result
    except BaseException as error:
        error_value = {"experiment_id": experiment_id, "run_id": run_id, "status": "failed", "stage": stage, "exception_type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(), "occurred_at": now_iso()}
        atomic_write_json(experiment_dir / "error.json", error_value)
        append_registry_atomic(REGISTRY_CSV, {"experiment_id": experiment_id, "config_hash": digest, "date": now_iso(), "status": "failed", "stage": stage, "hypothesis_id": candidate["id"], "hypothesis": candidate["title"], "model": candidate.get("model_type"), "model_type": candidate.get("model_type"), "leakage_risk": candidate["leakage_risk"], "rejection_reason": str(error), "result_file": str(experiment_dir / "error.json")})
        raise


def registry_row(result: dict[str, Any], experiment_dir: Path) -> dict[str, Any]:
    folds = result["folds"]
    mean_auc = np.mean([f["metrics"]["auc"] for f in folds if f["metrics"]["auc"] is not None])
    mean_logloss = np.mean([f["metrics"]["logloss"] for f in folds])
    return {
        "experiment_id": result["experiment_id"], "config_hash": result["config_hash"], "date": now_iso(), "status": "completed",
        "stage": result["stage"], "hypothesis_id": result["candidate"]["id"], "hypothesis": result["candidate"]["title"], "model": result["candidate"].get("model_type"), "expected_reason": result["candidate"]["reason"],
        "baseline_model": "deployable_train_mean", "changed_elements": result["candidate"]["id"], "fixed_elements": "rolling splits, test feature order",
        "model_type": result["candidate"].get("model_type"), "feature_changes": result["candidate"]["id"], "parameter_changes": "catalog config",
        "data_split": json.dumps([f["fold"] for f in folds]),
        "seed": json.dumps([item["seed"] for item in folds[0].get("seed_results", [])] or [42]),
        "leakage_risk": result["candidate"]["leakage_risk"],
        "expected_cost": result["candidate"]["cost"], "runtime_seconds": result["runtime_seconds"],
        "season_briers": json.dumps({str(f["validation_season"]): f["metrics"]["brier"] for f in folds}),
        "mean_brier": result["aggregate"]["mean_brier"], "weighted_mean_brier": result["aggregate"]["recent_weighted_brier"],
        "bss": result["aggregate"]["recent_weighted_bss"], "auc": mean_auc, "logloss": mean_logloss,
        "calibration": json.dumps({str(f["validation_season"]): f["metrics"]["calibration_error_ece10"] for f in folds}),
        "result_file": str(experiment_dir / "result.json"), "conclusion": "smoke pipeline passed" if result["stage"] == "smoke" else "evaluation completed",
        "next_recommendation": "quick는 baseline 대비 임계 개선 확인 후 rolling 승격",
    }


def render_experiment_result(result: dict[str, Any]) -> str:
    lines = [f"# {result['experiment_id']}", "", f"- 상태: {result['status']}", f"- 단계: {result['stage']}", f"- 가설: {result['candidate']['title']}", "", "| 시즌 | Brier | 기준 Brier | BSS | AUC | LogLoss | ECE |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for fold in result["folds"]:
        m = fold["metrics"]
        lines.append(f"| {fold['validation_season']} | {m['brier']:.8f} | {m['baseline_brier']:.8f} | {m['bss']:.6f} | {m['auc']:.6f} | {m['logloss']:.6f} | {m['calibration_error_ece10']:.6f} |")
        if fold.get("seed_results"):
            lines.append("")
            lines.append(f"- {fold['validation_season']} seed별 Brier: " + ", ".join(f"{item['seed']}={item['brier']:.8f}" for item in fold["seed_results"]))
            lines.append(f"- seed Brier 평균: {fold['seed_brier_mean']:.8f}")
            lines.append(f"- seed 예측 평균 앙상블 Brier: {fold['ensemble_brier']:.8f}")
    lines += ["", "Smoke 결과는 성능 우열 근거가 아니며 champion 승격 대상이 아니다.", ""]
    return "\n".join(lines)


def refresh_reports() -> dict[str, Any]:
    migrate_registry_schema(REGISTRY_CSV)
    registry = read_registry(REGISTRY_CSV)
    lines = ["# Experiment Registry", "", "| Experiment | Hypothesis ID | Model | 상태 | 단계 | 가설 | Weighted Brier | BSS |", "|---|---|---|---|---|---|---|---:|---:|"]
    for row in registry:
        lines.append(f"| {row['experiment_id']} | {row['hypothesis_id']} | {row['model']} | {row['status']} | {row['stage']} | {row['hypothesis']} | {row['weighted_mean_brier']} | {row['bss']} |")
    atomic_write_text(REGISTRY_MD, "\n".join(lines) + "\n")
    catalog = load_json(CATALOG_PATH)
    queue = []
    for candidate in catalog:
        eligibility_reason = None
        missing_dependencies = [name for name in candidate.get("requires_all", []) if package_version(name) is None]
        if missing_dependencies:
            eligibility_reason = f"필수 패키지 미설치: {missing_dependencies}"
        elif candidate.get("requires") and package_version(candidate["requires"]) is None:
            eligibility_reason = f"필수 패키지 {candidate['requires']} 미설치"
        elif not candidate.get("implemented"):
            eligibility_reason = "안전 실행 템플릿 미구현"
        eligible = eligibility_reason is None
        stage_results = candidate_stage_results(candidate["id"], registry)
        stage_state = candidate_stage_state(
            candidate, eligible, eligibility_reason, stage_results,
            load_json(CONFIG_PATH)["promotion"]["min_quick_brier_improvement"],
        )
        queue.append({
            **candidate,
            "eligible": eligible,
            "eligibility_reason": eligibility_reason,
            "skip_reason": eligibility_reason,
            **stage_state,
        })
    queue.sort(key=lambda item: item["priority"], reverse=True)
    atomic_write_json(QUEUE_PATH, queue)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text(REPORTS_DIR / "RESEARCH_PLAN.md", render_plan(queue))
    return {
        "registry_rows": len(registry),
        "eligible_candidates": sum(item["eligible"] for item in queue),
        "actionable_candidates": actionable_candidate_count(queue),
    }


def candidate_stage_state(
    candidate: dict[str, Any],
    eligible: bool,
    eligibility_reason: str | None,
    stage_results: dict[str, dict[str, Any]],
    min_quick_brier_improvement: float,
) -> dict[str, Any]:
    if candidate.get("stage_policy") == "benchmark_only":
        if not eligible:
            return {"next_stage": None, "stage_gate_status": "blocked", "stage_gate_reason": eligibility_reason}
        if stage_results.get("benchmark", {}).get("status") == "completed":
            return {"next_stage": None, "stage_gate_status": "completed", "stage_gate_reason": "benchmark_already_completed"}
        return {"next_stage": "benchmark", "stage_gate_status": "pending", "stage_gate_reason": "benchmark_not_started"}
    return derive_stage_state(
        eligible, eligibility_reason, stage_results, min_quick_brier_improvement
    )


def actionable_candidate_count(queue: list[dict[str, Any]]) -> int:
    return sum(
        1
        for item in queue
        if item.get("eligible") is True and item.get("next_stage") is not None
    )


def candidate_stage_results(candidate_id: str, registry: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for row in registry:
        row_candidate = row.get("hypothesis_id") or row.get("changed_elements")
        if row_candidate != candidate_id or row.get("stage") not in {"benchmark", "smoke", "quick", "rolling"}:
            continue
        value: dict[str, Any] = {"status": row.get("status"), "experiment_id": row.get("experiment_id")}
        result_path = Path(row.get("result_file", "")) if row.get("result_file") else None
        if row.get("status") == "completed" and result_path and result_path.exists():
            saved = load_json(result_path)
            if row.get("stage") == "quick" and saved.get("folds"):
                value["brier_improvement"] = saved["folds"][0]["metrics"]["brier_improvement"]
        results[row["stage"]] = value
    return results


def render_plan(queue: list[dict[str, Any]]) -> str:
    lines = [
        "# 향후 연구 계획", "",
        "후보는 구현 eligibility와 stage 진행 상태를 분리한다.",
        "`MaxHours`는 실행 상한이며 최소 실행 시간이 아니다. actionable candidate가 없으면 즉시 정상 종료한다.",
        "가짜 대기나 동일 실험 반복으로 시간을 채우지 않는다.", "",
        "| 우선순위 | 후보 | Eligible | Next stage | Gate | 사유 |", "|---:|---|---|---|---|---|",
    ]
    for item in queue:
        lines.append(f"| {item['priority']} | {item['title']} | {'예' if item['eligible'] else '아니오'} | {item.get('next_stage') or '-'} | {item.get('stage_gate_status')} | {item.get('stage_gate_reason')} |")
    return "\n".join(lines) + "\n"


def record_score(args: argparse.Namespace) -> None:
    if not args.experiment_id or args.leaderboard_score is None:
        raise ValueError("record-score에는 ExperimentId와 LeaderboardScore가 필요합니다.")
    registry = read_registry(REGISTRY_CSV)
    matches = [row for row in registry if row["experiment_id"] == args.experiment_id]
    if not matches:
        raise ValueError(f"레지스트리에 없는 experiment_id: {args.experiment_id}")
    row = matches[-1]
    exists = LEADERBOARD_PATH.exists()
    LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEADERBOARD_PATH.open("a", encoding="utf-8-sig", newline="") as handle:
        fields = ["experiment_id", "submitted_at", "score_type", "leaderboard_score", "local_rolling_brier", "local_bss", "local_rank", "notes", "submission_number"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        previous = 0
        if exists:
            with LEADERBOARD_PATH.open("r", encoding="utf-8-sig") as read_handle:
                previous = sum(1 for value in csv.DictReader(read_handle) if value["experiment_id"] == args.experiment_id)
        writer.writerow({"experiment_id": args.experiment_id, "submitted_at": now_iso(), "score_type": args.score_type, "leaderboard_score": args.leaderboard_score, "local_rolling_brier": row["weighted_mean_brier"], "local_bss": row["bss"], "local_rank": "", "notes": args.notes, "submission_number": previous + 1})


def choose_candidate(queue: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [
        candidate for candidate in queue
        if candidate.get("eligible")
        and candidate.get("next_stage")
        and candidate.get("stage_gate_status") in {"pending", "passed"}
    ]
    return max(eligible, key=lambda item: item["priority"], default=None)


def assess_and_store_champion(result: dict[str, Any], config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    current = load_json(CHAMPION_PATH)
    fold_briers = {str(fold["validation_season"]): fold["metrics"]["brier"] for fold in result["folds"]}
    if current.get("status") != "verified" or not current.get("fold_briers"):
        decision = {"decision": "needs_confirmation", "reason": "검증된 기존 champion 또는 다중 seed 확인 결과가 없음"}
    elif len(config["limits"]["seeds"]) > 1 and result.get("seed_prediction_ensemble_brier") is None:
        decision = {"decision": "needs_confirmation", "reason": "단일 seed 결과이므로 다중 seed 방향성 미확인"}
    else:
        status, reason = champion_decision(
            {
                "stage": result["stage"], "leakage_passed": result["leakage"]["passed"],
                "recent_weighted_brier": result["aggregate"]["recent_weighted_brier"],
                "fold_briers": fold_briers,
            },
            current,
            config["promotion"],
        )
        decision = {"decision": status, "reason": reason}
        if status == "promote":
            updated = {
                "status": "verified", "experiment_id": result["experiment_id"],
                "model": result["candidate"].get("model_type"),
                "recent_weighted_brier": result["aggregate"]["recent_weighted_brier"],
                "fold_briers": fold_briers, "updated_at": now_iso(),
            }
            atomic_write_json(CHAMPION_PATH, updated)
    atomic_write_json(run_dir / "experiments" / result["experiment_id"] / "promotion_decision.json", decision)
    return decision


def run_research_loop(config: dict[str, Any], run_id: str, run_dir: Path, logger: logging.Logger, max_experiments: int, max_hours: float) -> dict[str, Any]:
    budget = ResearchBudget(max_experiments, max_hours, time.monotonic())
    completed = failed = skipped = 0
    selected_experiments = 0
    exclusions: list[dict[str, Any]] = []
    runtime_blocked: set[str] = set()
    stop_reason = None
    while True:
        exhausted, stop_reason = budget.exhausted()
        if exhausted:
            break
        refresh_reports()
        queue = load_json(QUEUE_PATH)
        selectable = [item for item in queue if item["id"] not in runtime_blocked]
        candidate = choose_candidate(selectable)
        if candidate is None:
            stop_reason = "no_actionable_candidates"
            exclusions.extend([
                {
                    "hypothesis_id": item["id"],
                    "eligible": item.get("eligible"),
                    "next_stage": item.get("next_stage"),
                    "stage_gate_status": item.get("stage_gate_status"),
                    "reason": item.get("stage_gate_reason") or item.get("eligibility_reason") or "not_selected",
                }
                for item in queue
                if item["id"] not in {value["hypothesis_id"] for value in exclusions}
            ])
            break
        stage = candidate["next_stage"]
        selected_experiments += 1
        logger.info("candidate selected: id=%s stage=%s", candidate["id"], stage)
        try:
            result = run_experiment(candidate, config, run_id, run_dir, stage, logger)
            if result.get("status") == "skipped":
                skipped += 1
                reason = result.get("reason", "duplicate_config_hash")
                exclusions.append({
                    "hypothesis_id": candidate["id"], "eligible": True,
                    "next_stage": stage, "stage_gate_status": "skipped", "reason": reason,
                })
                runtime_blocked.add(candidate["id"])
                budget.completed += 1
                continue
            completed += int(result.get("status") == "completed")
            if stage == "rolling" and result.get("status") == "completed":
                promotion = assess_and_store_champion(result, config, run_dir)
                logger.info("champion 판정: %s (%s)", promotion["decision"], promotion["reason"])
        except Exception as error:
            failed += 1
            logger.exception("연구 실험 실패")
            exclusions.append({
                "hypothesis_id": candidate["id"], "eligible": True,
                "next_stage": stage, "stage_gate_status": "failed",
                "reason": f"{stage}_execution_failed:{type(error).__name__}",
            })
            runtime_blocked.add(candidate["id"])
            if failed >= config["limits"]["max_failures"]:
                stop_reason = "max_failures"
                break
        budget.completed += 1
    return {
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "selected_experiments": selected_experiments,
        "candidate_exclusions": exclusions,
        "stop_reason": stop_reason,
    }


def main() -> int:
    args = parse_args()
    config = load_json(CONFIG_PATH)
    if args.mode == "record-score":
        record_score(args)
        refresh_reports()
        return 0
    run_id = new_run_id(args.mode)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True)
    logger = setup_logger(run_id, run_dir)
    snapshot = environment_snapshot(config)
    manifest = {
        "research_run_id": run_id, "mode": args.mode, "status": "running",
        "started_at": now_iso(), "finished_at": None,
        "python_path": sys.executable, "python_version": sys.version,
        "library_versions": snapshot["libraries"], "git": snapshot["git"],
        "data": snapshot["data"], "split": config["rolling_splits"],
        "max_experiments": args.max_experiments or config["limits"]["max_experiments"],
        "max_hours": args.max_hours or config["limits"]["max_hours"],
        "completed_experiments": 0, "failed_experiments": 0, "skipped_experiments": 0,
        "selected_experiments": 0, "candidate_exclusions": [],
        "previous_champion": load_json(CHAMPION_PATH) if CHAMPION_PATH.exists() else None,
        "current_champion": None, "stop_reason": None, "result_files": [], "error_summary": None,
    }
    write_run_manifest(run_dir, manifest)
    try:
        if args.mode == "check":
            result = check_system(config, logger)
            atomic_write_json(run_dir / "check_result.json", result)
            manifest["result_files"].append(str(run_dir / "check_result.json"))
            if not result["passed"]:
                raise RuntimeError(f"check 실패: {result['errors']}")
        elif args.mode == "smoke":
            check = check_system(config, logger)
            atomic_write_json(run_dir / "check_result.json", check)
            if not check["passed"]:
                raise RuntimeError(f"smoke 사전 check 실패: {check['errors']}")
            catalog = load_json(CATALOG_PATH)
            candidate_id = args.candidate_id or config["smoke"]["candidate_id"]
            candidate = next((item for item in catalog if item["id"] == candidate_id), None)
            if candidate is None:
                raise ValueError(f"카탈로그에 없는 candidate_id: {candidate_id}")
            if not candidate.get("implemented"):
                raise ValueError(f"안전 실행 템플릿이 없는 candidate_id: {candidate_id}")
            missing_dependencies = [name for name in candidate.get("requires_all", []) if package_version(name) is None]
            if missing_dependencies:
                raise RuntimeError(f"필수 패키지 미설치: {missing_dependencies}")
            if candidate.get("requires") and package_version(candidate["requires"]) is None:
                raise RuntimeError(f"필수 패키지 미설치: {candidate['requires']}")
            result = run_experiment(candidate, config, run_id, run_dir, "smoke", logger)
            manifest["completed_experiments"] = int(result.get("status") == "completed")
            manifest["skipped_experiments"] = int(result.get("status") == "skipped")
            manifest["result_files"].append(str(run_dir / "experiments"))
        elif args.mode == "research":
            outcome = run_research_loop(config, run_id, run_dir, logger, manifest["max_experiments"], manifest["max_hours"])
            manifest["completed_experiments"] = outcome["completed"]
            manifest["failed_experiments"] = outcome["failed"]
            manifest["skipped_experiments"] = outcome["skipped"]
            manifest["selected_experiments"] = outcome["selected_experiments"]
            manifest["candidate_exclusions"] = outcome["candidate_exclusions"]
            manifest["stop_reason"] = outcome["stop_reason"]
            print(f"selected_experiments={outcome['selected_experiments']}")
            print(f"stop_reason={outcome['stop_reason']}")
        elif args.mode == "resume":
            interrupted = sorted([p for p in RUNS_DIR.glob("*_research/run_manifest.json") if is_resume_candidate(load_json(p))], key=lambda p: p.stat().st_mtime, reverse=True)
            atomic_write_json(run_dir / "resume_candidates.json", [str(path) for path in interrupted])
            outcome = run_research_loop(config, run_id, run_dir, logger, manifest["max_experiments"], manifest["max_hours"])
            manifest["completed_experiments"] = outcome["completed"]
            manifest["failed_experiments"] = outcome["failed"]
            manifest["skipped_experiments"] = outcome["skipped"]
            manifest["selected_experiments"] = outcome["selected_experiments"]
            manifest["candidate_exclusions"] = outcome["candidate_exclusions"]
            manifest["stop_reason"] = "resumed_safely_in_new_run: " + str(outcome["stop_reason"])
            print(f"selected_experiments={outcome['selected_experiments']}")
            print(f"stop_reason={manifest['stop_reason']}")
        elif args.mode == "report":
            report = refresh_reports()
            atomic_write_json(run_dir / "report_result.json", report)
            manifest["result_files"].append(str(REPORTS_DIR / "RESEARCH_PLAN.md"))
        refresh_reports()
        manifest["status"] = "completed"
        manifest["finished_at"] = now_iso()
        manifest["current_champion"] = load_json(CHAMPION_PATH) if CHAMPION_PATH.exists() else None
        write_run_manifest(run_dir, manifest)
        atomic_write_text(run_dir / "research_summary.md", f"# {run_id}\n\n- 상태: completed\n- 모드: {args.mode}\n- 완료 실험: {manifest['completed_experiments']}\n- 실패 실험: {manifest['failed_experiments']}\n")
        return 0
    except KeyboardInterrupt:
        manifest.update({"status": "interrupted", "finished_at": now_iso(), "stop_reason": "user_interrupt"})
        write_run_manifest(run_dir, manifest)
        return 130
    except BaseException as error:
        manifest["failed_experiments"] += 1
        write_failure_artifacts(run_dir, manifest, error)
        logger.exception("run 실패")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
