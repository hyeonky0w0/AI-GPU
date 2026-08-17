import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import runpod


APP_DIR = Path("/app")
BUNDLED_SRC = APP_DIR / "src"
BUNDLED_CONFIG = APP_DIR / "config"
MAX_RESULT_ZIP_BYTES = 6 * 1024 * 1024  # leaves headroom under RunPod's 10 MB /run payload


def safe_extract(zip_bytes: bytes, destination: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for member in archive.infolist():
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe bundle path: {member.filename}")
        archive.extractall(destination)


def zip_results(results_dir: Path) -> str:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(results_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(results_dir).as_posix())

    raw = buffer.getvalue()
    if len(raw) > MAX_RESULT_ZIP_BYTES:
        raise ValueError(
            f"results.zip is {len(raw) / 1024 / 1024:.2f} MiB; "
            f"limit for inline return is {MAX_RESULT_ZIP_BYTES / 1024 / 1024:.0f} MiB. "
            "Move large predictions/models to object storage or a RunPod Network Volume."
        )
    return base64.b64encode(raw).decode("ascii")


def handler(job):
    job_input = job.get("input", {}) or {}
    metadata = job_input.get("metadata", {}) or {}

    experiment_id = str(metadata.get("experiment_id") or job_input.get("experiment_id") or "local-test")
    branch = str(metadata.get("branch") or job_input.get("branch") or "unknown")
    commit_sha = str(metadata.get("sha") or job_input.get("sha") or "unknown")

    work_root = Path(tempfile.mkdtemp(prefix=f"experiment-{experiment_id}-"))
    results_dir = work_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        src_dir = BUNDLED_SRC
        config_dir = BUNDLED_CONFIG

        bundle_b64 = job_input.get("bundle_b64")
        if bundle_b64:
            bundle_dir = work_root / "bundle"
            bundle_dir.mkdir()
            safe_extract(base64.b64decode(bundle_b64), bundle_dir)
            src_dir = bundle_dir / "src"
            config_dir = bundle_dir / "config"

        train_script = src_dir / "train.py"
        config_file = config_dir / "experiment.yaml"

        if not train_script.exists():
            raise FileNotFoundError(f"train.py not found: {train_script}")
        if not config_file.exists():
            raise FileNotFoundError(f"experiment.yaml not found: {config_file}")

        env = os.environ.copy()
        env.update(
            {
                "EXPERIMENT_ID": experiment_id,
                "GIT_BRANCH": branch,
                "GIT_SHA": commit_sha,
            }
        )

        command = [
            sys.executable,
            str(train_script),
            "--config",
            str(config_file),
            "--results-dir",
            str(results_dir),
        ]

        process = subprocess.run(
            command,
            cwd=str(work_root),
            capture_output=True,
            text=True,
            env=env,
        )

        train_log = process.stdout or ""
        error_log = process.stderr or ""

        (results_dir / "train.log").write_text(train_log, encoding="utf-8")
        (results_dir / "error.log").write_text(error_log, encoding="utf-8")

        metrics_path = results_dir / "metrics.json"
        metrics = {}
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                error_log += f"\nmetrics.json is invalid JSON: {exc}\n"
                (results_dir / "error.log").write_text(error_log, encoding="utf-8")

        metrics.update(
            {
                "experiment_id": experiment_id,
                "branch": branch,
                "commit_sha": commit_sha,
                "exit_code": process.returncode,
            }
        )
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        result_zip_b64 = zip_results(results_dir)

        status = "success" if process.returncode == 0 else "failed"
        return {
            "status": status,
            "experiment_id": experiment_id,
            "branch": branch,
            "commit_sha": commit_sha,
            "exit_code": process.returncode,
            "metrics": metrics,
            "train_log_tail": train_log[-12000:],
            "error_log_tail": error_log[-12000:],
            "results_zip_b64": result_zip_b64,
        }

    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        (results_dir / "error.log").write_text(error_message + "\n", encoding="utf-8")
        return {
            "status": "failed",
            "experiment_id": experiment_id,
            "branch": branch,
            "commit_sha": commit_sha,
            "error": error_message,
        }
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


runpod.serverless.start({"handler": handler})
