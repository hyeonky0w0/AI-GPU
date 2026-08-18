"""036 GPU Tabular MLP smoke 학습기."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


FEATURES = [
    "season", "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before", "run_top_before", "run_bot_before",
    "run_total_before", "score_diff_home", "score_diff_pitcher_team", "runner_on_1b",
    "runner_on_2b", "runner_on_3b", "num_runners_on", "base_state", "home_win_expectancy",
    "away_win_expectancy", "li", "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id", "asof_pitcher_n", "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate", "asof_batter_n", "asof_batter_success_rate",
    "asof_batter_middle_rate", "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]
CATEGORICAL = ["top_bottom", "game_type", "base_state"]
TARGET = "control_success"


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode(frame: pd.DataFrame, train_frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, object]]:
    result = np.empty((len(frame), len(FEATURES)), dtype=np.float32)
    medians: dict[str, float] = {}
    categories: dict[str, dict[str, int]] = {}
    for index, column in enumerate(FEATURES):
        if column in CATEGORICAL:
            train_values = train_frame[column].fillna("__MISSING__").astype(str)
            mapping = {value: position for position, value in enumerate(sorted(train_values.unique()))}
            values = frame[column].fillna("__MISSING__").astype(str).map(mapping).fillna(-1)
            result[:, index] = values.to_numpy(dtype=np.float32)
            categories[column] = mapping
        else:
            train_values = pd.to_numeric(train_frame[column], errors="coerce")
            median = float(train_values.median()) if train_values.notna().any() else 0.0
            values = pd.to_numeric(frame[column], errors="coerce").fillna(median).to_numpy(dtype=np.float32)
            result[:, index] = values
            medians[column] = median
    numeric_indices = [i for i, column in enumerate(FEATURES) if column not in CATEGORICAL]
    for index in numeric_indices:
        column = FEATURES[index]
        train_values = pd.to_numeric(train_frame[column], errors="coerce").fillna(medians[column]).to_numpy(dtype=np.float32)
        mean = float(train_values.mean())
        std = float(train_values.std())
        result[:, index] = (result[:, index] - mean) / (std if std > 1e-8 else 1.0)
    return result, {"medians": medians, "categories": categories}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_config = config["model"]
    seed = int(model_config["seed"])
    seed_everything(seed)
    output = Path(__file__).resolve().parents[1] / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(message: str) -> None:
        print(message, flush=True)
        log_lines.append(message)

    data_path = Path(args.data_root) / "train.csv"
    if not data_path.is_file():
        raise FileNotFoundError(f"Network Volume에 train.csv가 없습니다: {data_path}")
    usecols = FEATURES + [TARGET]
    frame = pd.read_csv(data_path, usecols=usecols, low_memory=False)
    train_frame = frame.loc[frame["season"] < 2024].copy()
    validation = frame.loc[frame["season"] == 2024].copy()
    max_rows = model_config.get("max_train_rows")
    if max_rows is not None and len(train_frame) > int(max_rows):
        train_frame = train_frame.head(int(max_rows)).copy()
    if train_frame.empty or validation.empty:
        raise ValueError("시간순 train/validation 행이 비어 있습니다")
    train_x, _ = encode(train_frame, train_frame)
    validation_x, _ = encode(validation, train_frame)
    y_train = train_frame[TARGET].astype(np.float32).to_numpy()
    y_validation = validation[TARGET].astype(np.float32).to_numpy()

    cuda_available = bool(torch.cuda.is_available())
    device = torch.device("cuda" if cuda_available else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    cuda_version = torch.version.cuda
    log(f"CUDA available: {cuda_available}")
    log(f"Device: {device}")
    log(f"GPU: {gpu_name}")
    log(f"CUDA version: {cuda_version}")
    log(f"PyTorch version: {torch.__version__}")
    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=int(model_config["batch_size"]), shuffle=True,
                        num_workers=0, pin_memory=cuda_available)
    model = TabularMLP(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(model_config["learning_rate"]),
                                  weight_decay=float(model_config["weight_decay"]))
    criterion = nn.BCEWithLogitsLoss()
    validation_tensor = torch.from_numpy(validation_x).to(device)
    started = time.perf_counter()
    best_brier = math.inf
    best_logloss = math.inf
    for epoch in range(1, int(model_config["max_epochs"]) + 1):
        model.train()
        losses = []
        for batch_index, (features, target) in enumerate(loader):
            features, target = features.to(device, non_blocking=True), target.to(device, non_blocking=True)
            if batch_index == 0:
                log(f"First batch device: features={features.device}, target={target.device}")
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(model(validation_tensor)).detach().cpu().numpy()
        probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
        brier = float(np.mean((probabilities - y_validation) ** 2))
        logloss = float(-np.mean(y_validation * np.log(probabilities) + (1 - y_validation) * np.log(1 - probabilities)))
        best_brier = min(best_brier, brier)
        best_logloss = min(best_logloss, logloss)
        log(f"epoch={epoch} train_loss={np.mean(losses):.6f} validation_brier={brier:.6f} validation_logloss={logloss:.6f}")
    train_seconds = time.perf_counter() - started
    metrics = {
        "experiment_id": config["experiment_id"], "status": "success", "gpu_name": gpu_name,
        "cuda_available": cuda_available, "cuda_version": cuda_version,
        "torch_version": torch.__version__, "device": str(device),
        "train_rows": len(train_frame), "validation_rows": len(validation),
        "epochs": int(model_config["max_epochs"]), "train_seconds": train_seconds,
        "best_brier": best_brier, "best_logloss": best_logloss,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "train.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        output = Path(__file__).resolve().parents[1] / "outputs"
        output.mkdir(parents=True, exist_ok=True)
        (output / "error.log").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise
