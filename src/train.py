import json
import os
import subprocess
import time
from pathlib import Path

from evaluate import evaluate
from features import build_dataset
from model import fit_and_predict


def _gpu_name():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().splitlines()[0] or None
    except (FileNotFoundError, subprocess.SubprocessError, IndexError):
        return None


def main():
    started = time.perf_counter()
    config = json.loads(Path("config/experiment.yaml").read_text(encoding="utf-8"))
    train_x, train_y, valid_x, valid_y = build_dataset(config)
    predictions = fit_and_predict(train_x, train_y, valid_x, config)
    scores = evaluate(valid_y, predictions)

    metrics = {
        "experiment_id": os.getenv("EXPERIMENT_ID", "local"),
        "branch": os.getenv("GIT_BRANCH", "local"),
        "sha": os.getenv("GIT_SHA", "local"),
        "model": config["model"]["type"],
        **scores,
        "train_seconds": round(time.perf_counter() - started, 4),
        "gpu": _gpu_name(),
    }
    results_dir = Path(os.getenv("RESULTS_DIR", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (results_dir / "predictions.csv").write_text(
        "actual,prediction\n" + "".join(
            f"{actual},{prediction}\n" for actual, prediction in zip(valid_y, predictions)
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
