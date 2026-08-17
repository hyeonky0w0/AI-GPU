import base64
import io
import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import runpod


MAX_BUNDLE_BYTES = 8 * 1024 * 1024
MAX_RESULT_BYTES = 8 * 1024 * 1024


def _decode_bundle(encoded_bundle, destination):
    try:
        payload = base64.b64decode(encoded_bundle, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("input.bundle_b64 is not valid base64") from exc
    if len(payload) > MAX_BUNDLE_BYTES:
        raise ValueError("decoded code bundle exceeds 8 MiB")

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe bundle path: {member.filename}")
        archive.extractall(destination)


def _encode_results(results_dir):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(results_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(results_dir))
    payload = buffer.getvalue()
    if len(payload) > MAX_RESULT_BYTES:
        raise ValueError("result archive exceeds 8 MiB; use external object storage")
    return base64.b64encode(payload).decode("ascii")


def handler(job):
    job_input = job.get("input") or {}
    bundle_b64 = job_input.get("bundle_b64")
    if not isinstance(bundle_b64, str) or not bundle_b64:
        return {"status": "failed", "error": "input.bundle_b64 is required"}

    metadata = job_input.get("metadata") or {}
    print(f"===== EXPERIMENT {metadata.get('experiment_id', job.get('id'))} =====", flush=True)

    with tempfile.TemporaryDirectory(prefix="experiment-") as temp_dir:
        workspace = Path(temp_dir)
        results_dir = workspace / "results"
        results_dir.mkdir()
        try:
            _decode_bundle(bundle_b64, workspace)
            train_file = workspace / "src" / "train.py"
            if not train_file.is_file():
                raise ValueError("bundle does not contain src/train.py")

            environment = os.environ.copy()
            environment.update({
                "EXPERIMENT_ID": str(metadata.get("experiment_id", job.get("id", "unknown"))),
                "GIT_BRANCH": str(metadata.get("branch", "unknown")),
                "GIT_SHA": str(metadata.get("sha", "unknown")),
                "RESULTS_DIR": str(results_dir),
            })
            result = subprocess.run(
                ["python", "-u", str(train_file)],
                cwd=workspace,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            (results_dir / "train.log").write_text(result.stdout, encoding="utf-8")
            (results_dir / "error.log").write_text(result.stderr, encoding="utf-8")

            metrics_file = results_dir / "metrics.json"
            metrics = None
            if metrics_file.is_file():
                metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            return {
                "status": "success" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "metrics": metrics,
                "results_zip_b64": _encode_results(results_dir),
            }
        except Exception as exc:
            (results_dir / "error.log").write_text(
                f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
            )
            return {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "results_zip_b64": _encode_results(results_dir),
            }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
