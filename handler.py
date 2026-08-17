import json
import subprocess
import sys
from pathlib import Path

import runpod


APP_DIR = Path("/app")
RESULTS_DIR = APP_DIR / "results"
TRAIN_SCRIPT = APP_DIR / "src" / "train.py"


def handler(job):
    job_input = job.get("input", {})

    experiment_id = job_input.get("experiment_id", "local-test")
    branch = job_input.get("branch", "unknown")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("GPU EXPERIMENT START", flush=True)
    print(f"experiment_id={experiment_id}", flush=True)
    print(f"branch={branch}", flush=True)
    print("=" * 60, flush=True)

    try:
        process = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT)],
            cwd=str(APP_DIR),
            capture_output=True,
            text=True,
        )

        train_log = process.stdout or ""
        error_log = process.stderr or ""

        # 항상 로그 저장
        (RESULTS_DIR / "train.log").write_text(
            train_log,
            encoding="utf-8",
        )

        (RESULTS_DIR / "error.log").write_text(
            error_log,
            encoding="utf-8",
        )

        metrics_path = RESULTS_DIR / "metrics.json"

        # train.py가 metrics.json을 만들었는지 확인
        if metrics_path.exists():
            metrics = json.loads(
                metrics_path.read_text(encoding="utf-8")
            )
        else:
            metrics = {}

        metrics["experiment_id"] = experiment_id
        metrics["branch"] = branch
        metrics["exit_code"] = process.returncode

        metrics_path.write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )

        if process.returncode != 0:
            return {
                "status": "failed",
                "experiment_id": experiment_id,
                "branch": branch,
                "exit_code": process.returncode,
                "metrics": metrics,
                "train_log": train_log,
                "error_log": error_log,
            }

        return {
            "status": "success",
            "experiment_id": experiment_id,
            "branch": branch,
            "metrics": metrics,
            "train_log": train_log,
            "error_log": error_log,
        }

    except Exception as exc:
        error_message = str(exc)

        (RESULTS_DIR / "error.log").write_text(
            error_message,
            encoding="utf-8",
        )

        return {
            "status": "failed",
            "experiment_id": experiment_id,
            "branch": branch,
            "error": error_message,
        }


runpod.serverless.start({"handler": handler})