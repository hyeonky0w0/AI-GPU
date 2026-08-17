import json
import subprocess
from pathlib import Path

import runpod


def handler(event):
    print("===== EXPERIMENT START =====")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    result = subprocess.run(
        ["python", "src/train.py"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    with (results_dir / "train.log").open("w", encoding="utf-8") as file:
        file.write(result.stdout)
        file.write("\n")
        file.write(result.stderr)

    if result.returncode != 0:
        return {
            "status": "failed",
            "error": result.stderr,
        }

    with (results_dir / "metrics.json").open(encoding="utf-8") as file:
        metrics = json.load(file)

    return {
        "status": "success",
        "metrics": metrics,
    }


runpod.serverless.start({"handler": handler})
