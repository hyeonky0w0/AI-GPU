import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from torch import nn


def load_config(path: Path) -> dict:
    # experiment.yaml is intentionally JSON-compatible YAML.
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== GPU EXPERIMENT ===", flush=True)
    print(f"device={device}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"cuda={torch.version.cuda}", flush=True)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    if gpu_name:
        print(f"gpu={gpu_name}", flush=True)

    # Deterministic synthetic regression data for pipeline verification.
    n = int(config.get("data", {}).get("sample_rows", 2000))
    train_ratio = float(config.get("data", {}).get("train_ratio", 0.8))

    x = torch.rand(n, 4, device=device)
    noise = 0.03 * torch.randn(n, device=device)
    y = (
        2.0 * x[:, 0]
        - 1.3 * x[:, 1]
        + 0.7 * x[:, 2]
        + 0.4 * x[:, 3]
        + noise
    ).unsqueeze(1)

    split = max(1, int(n * train_ratio))
    train_x, valid_x = x[:split], x[split:]
    train_y, valid_y = y[:split], y[split:]

    model = nn.Sequential(
        nn.Linear(4, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    ).to(device)

    learning_rate = float(config.get("model", {}).get("learning_rate", 0.01))
    epochs = int(config.get("model", {}).get("epochs", 100))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(train_x)
        loss = loss_fn(prediction, train_y)
        loss.backward()
        optimizer.step()

        if epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"epoch={epoch + 1}/{epochs} train_mse={loss.item():.6f}", flush=True)

    with torch.no_grad():
        valid_pred = model(valid_x)
        diff = valid_pred - valid_y
        rmse = math.sqrt(torch.mean(diff.square()).item())
        mae = torch.mean(diff.abs()).item()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    metrics = {
        "status": "success",
        "model": "MLP_gpu_smoke_test",
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "val_rmse": round(rmse, 6),
        "val_mae": round(mae, 6),
        "epochs": epochs,
        "sample_rows": n,
        "train_seconds": round(time.time() - started, 3),
        "experiment_id": os.getenv("EXPERIMENT_ID", "local"),
        "branch": os.getenv("GIT_BRANCH", "local"),
        "commit_sha": os.getenv("GIT_SHA", "local"),
    }

    (results_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
