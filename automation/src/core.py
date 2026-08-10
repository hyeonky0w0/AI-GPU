from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REGISTRY_FIELDS = [
    "experiment_id", "parent_experiment_id", "config_hash", "date", "status", "stage",
    "hypothesis_id", "hypothesis", "model", "expected_reason", "baseline_model", "changed_elements", "fixed_elements",
    "model_type", "feature_changes", "parameter_changes", "data_split", "seed", "leakage_risk",
    "expected_cost", "runtime_seconds", "season_briers", "mean_brier", "weighted_mean_brier",
    "bss", "auc", "logloss", "calibration", "champion_improvement", "result_file", "conclusion",
    "rejection_reason", "next_recommendation",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_predictions(y: np.ndarray, prediction: np.ndarray) -> None:
    if len(y) != len(prediction):
        raise ValueError("예측 길이가 타깃 길이와 다릅니다.")
    if not np.isfinite(prediction).all():
        raise ValueError("예측에 NaN 또는 무한대가 있습니다.")
    if ((prediction < 0) | (prediction > 1)).any():
        raise ValueError("예측 확률이 [0, 1] 범위를 벗어났습니다.")


def expected_calibration_error(y: np.ndarray, prediction: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.minimum(np.digitize(prediction, edges[1:-1]), bins - 1)
    total = len(y)
    error = 0.0
    for index in range(bins):
        mask = indices == index
        if mask.any():
            error += float(mask.mean()) * abs(float(y[mask].mean()) - float(prediction[mask].mean()))
    return error


def calibration_slope_intercept(y: np.ndarray, prediction: np.ndarray) -> tuple[float | None, float | None]:
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(y)) < 2:
        return None, None
    clipped = np.clip(prediction, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
    model.fit(logit, y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def calculate_metrics(y: np.ndarray, prediction: np.ndarray, baseline_probability: float) -> dict[str, Any]:
    from sklearn.metrics import log_loss, roc_auc_score

    y = np.asarray(y, dtype=int)
    prediction = np.asarray(prediction, dtype=float)
    validate_predictions(y, prediction)
    baseline = np.full(len(y), float(baseline_probability))
    brier = float(np.mean((prediction - y) ** 2))
    baseline_brier = float(np.mean((baseline - y) ** 2))
    bss = float(1.0 - brier / baseline_brier) if baseline_brier > 0 else float("nan")
    slope, intercept = calibration_slope_intercept(y, prediction)
    return {
        "brier": brier,
        "baseline_brier": baseline_brier,
        "brier_improvement": baseline_brier - brier,
        "bss": bss,
        "competition_score_approx": max(0.0, 100000.0 * bss) if math.isfinite(bss) else None,
        "auc": float(roc_auc_score(y, prediction)) if len(np.unique(y)) == 2 else None,
        "logloss": float(log_loss(y, np.clip(prediction, 1e-6, 1 - 1e-6))),
        "target_rate": float(y.mean()),
        "prediction_mean": float(prediction.mean()),
        "mean_bias": float(prediction.mean() - y.mean()),
        "prediction_std": float(prediction.std()),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
        "calibration_error_ece10": expected_calibration_error(y, prediction),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def aggregate_fold_metrics(folds: list[dict[str, Any]], season_weights: dict[str, float]) -> dict[str, Any]:
    briers = np.array([fold["metrics"]["brier"] for fold in folds], dtype=float)
    rows = np.array([fold["validation_rows"] for fold in folds], dtype=float)
    weights = np.array([season_weights[str(fold["validation_season"])] for fold in folds], dtype=float)
    baseline_briers = np.array([fold["metrics"]["baseline_brier"] for fold in folds], dtype=float)
    weighted_brier = float(np.average(briers, weights=weights))
    weighted_baseline = float(np.average(baseline_briers, weights=weights))
    return {
        "mean_brier": float(briers.mean()),
        "row_weighted_brier": float(np.average(briers, weights=rows)),
        "recent_weighted_brier": weighted_brier,
        "worst_season_brier": float(briers.max()),
        "recent_weighted_baseline_brier": weighted_baseline,
        "recent_weighted_bss": float(1 - weighted_brier / weighted_baseline),
        "improved_seasons_vs_baseline": int(sum(f["metrics"]["brier"] < f["metrics"]["baseline_brier"] for f in folds)),
    }


FORBIDDEN_FEATURE_NAMES = {
    "control_success", "target", "label", "pitch_result", "actual_pitch_type",
    "plate_x", "plate_z", "zone_result",
}


def static_leakage_check(features: list[str], target: str, identifier: str, splits: list[dict[str, Any]]) -> dict[str, Any]:
    violations: list[str] = []
    lowered = {feature.lower() for feature in features}
    if target in features:
        violations.append("타깃이 피처에 포함됨")
    if identifier in features:
        violations.append("row_id가 피처에 포함됨")
    suspicious = sorted(lowered & FORBIDDEN_FEATURE_NAMES)
    if suspicious:
        violations.append(f"금지 또는 의심 피처: {suspicious}")
    for split in splits:
        if not split["train_seasons"] or max(split["train_seasons"]) >= split["validation_season"]:
            violations.append(f"시간 역전 split: {split['name']}")
    return {
        "passed": not violations,
        "violations": violations,
        "warnings": ["asof_*는 제공 문서상 허용되지만 원시 생성 코드는 저장소에 없어 독립 재감사는 미완료"],
    }


def past_expanding_rate(target: np.ndarray, groups: np.ndarray, prior: float = 0.5, strength: float = 10.0) -> np.ndarray:
    """현재 행을 제외한 과거 행만으로 smoothed expanding rate를 만든다."""
    sums: dict[Any, float] = {}
    counts: dict[Any, int] = {}
    output = np.empty(len(target), dtype=float)
    for index, (value, group) in enumerate(zip(target, groups)):
        total = sums.get(group, 0.0)
        count = counts.get(group, 0)
        output[index] = (total + prior * strength) / (count + strength)
        sums[group] = total + float(value)
        counts[group] = count + 1
    return output


def champion_decision(candidate: dict[str, Any], champion: dict[str, Any], thresholds: dict[str, Any]) -> tuple[str, str]:
    if candidate.get("stage") != "rolling" or not candidate.get("leakage_passed"):
        return "reject", "rolling 완료 또는 누출 통과 조건 미충족"
    candidate_folds = candidate["fold_briers"]
    champion_folds = champion.get("fold_briers", {})
    if set(candidate_folds) != set(champion_folds):
        return "reject", "필수 시즌 비교 결과 불완전"
    improvement = champion["recent_weighted_brier"] - candidate["recent_weighted_brier"]
    improved = sum(candidate_folds[s] < champion_folds[s] for s in candidate_folds)
    regression = max(candidate_folds[s] - champion_folds[s] for s in candidate_folds)
    if improved < thresholds["required_improved_seasons"]:
        return "reject", "개선 시즌 수 부족"
    if regression > thresholds["max_single_season_regression"]:
        return "reject", "단일 시즌 허용 악화 초과"
    if improvement < thresholds["min_champion_brier_improvement"]:
        return "needs_confirmation", "개선폭이 즉시 승격 임계값 미만"
    return "promote", "모든 champion 승격 조건 통과"


@dataclass
class ResearchBudget:
    max_experiments: int
    max_hours: float
    started_monotonic: float
    completed: int = 0

    def exhausted(self) -> tuple[bool, str | None]:
        if self.completed >= self.max_experiments:
            return True, "max_experiments"
        if time.monotonic() - self.started_monotonic >= self.max_hours * 3600:
            return True, "max_hours"
        return False, None


def read_registry(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def append_registry_atomic(path: Path, row: dict[str, Any]) -> None:
    rows = read_registry(path)
    normalized = {field: row.get(field, "") for field in REGISTRY_FIELDS}
    rows.append({key: str(value) if value is not None else "" for key, value in normalized.items()})
    lines: list[str] = []
    from io import StringIO
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=REGISTRY_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def migrate_registry_schema(path: Path) -> None:
    rows = read_registry(path)
    from io import StringIO
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=REGISTRY_FIELDS, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        normalized = {field: row.get(field, "") for field in REGISTRY_FIELDS}
        normalized["hypothesis_id"] = normalized["hypothesis_id"] or row.get("changed_elements", "")
        normalized["model"] = normalized["model"] or row.get("model_type", "")
        writer.writerow(normalized)
    atomic_write_text(path, buffer.getvalue())


def successful_hashes(registry: list[dict[str, str]]) -> set[str]:
    return {row["config_hash"] for row in registry if row.get("status") == "completed"}


def write_failure_artifacts(run_dir: Path, manifest: dict[str, Any], error: BaseException) -> None:
    manifest = dict(manifest)
    manifest.update({
        "status": "failed", "finished_at": now_iso(),
        "error_summary": f"{type(error).__name__}: {error}",
    })
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    atomic_write_json(run_dir / "manifest.json", manifest)
    atomic_write_json(run_dir / "error.json", {
        "type": type(error).__name__, "message": str(error), "occurred_at": now_iso(),
    })


def is_resume_candidate(manifest: dict[str, Any]) -> bool:
    return manifest.get("mode") == "research" and manifest.get("status") in {"running", "interrupted", "failed"}


def derive_stage_state(
    eligible: bool,
    eligibility_reason: str | None,
    stage_results: dict[str, dict[str, Any]],
    min_quick_brier_improvement: float,
) -> dict[str, Any]:
    """구현 eligibility와 실험 stage 진행 상태를 독립적으로 판정한다."""
    if not eligible:
        return {
            "next_stage": None,
            "stage_gate_status": "blocked",
            "stage_gate_reason": eligibility_reason or "ineligible",
        }
    rolling = stage_results.get("rolling")
    if rolling and rolling.get("status") == "completed":
        return {
            "next_stage": None,
            "stage_gate_status": "completed",
            "stage_gate_reason": "rolling_already_completed",
        }
    quick = stage_results.get("quick")
    if quick and quick.get("status") == "completed":
        improvement = quick.get("brier_improvement")
        if improvement is None or improvement < min_quick_brier_improvement:
            return {
                "next_stage": None,
                "stage_gate_status": "failed",
                "stage_gate_reason": "quick_metric_gate_failed",
            }
        return {
            "next_stage": "rolling",
            "stage_gate_status": "passed",
            "stage_gate_reason": "quick_metric_gate_passed",
        }
    smoke = stage_results.get("smoke")
    if smoke and smoke.get("status") == "completed":
        # smoke 지표는 축소 표본 파이프라인 검사값이며 성능 gate로 사용하지 않는다.
        return {
            "next_stage": "quick",
            "stage_gate_status": "passed",
            "stage_gate_reason": "smoke_already_completed",
        }
    if smoke and smoke.get("status") == "failed":
        return {
            "next_stage": None,
            "stage_gate_status": "failed",
            "stage_gate_reason": "smoke_pipeline_failed",
        }
    return {
        "next_stage": "smoke",
        "stage_gate_status": "pending",
        "stage_gate_reason": "smoke_not_started",
    }
