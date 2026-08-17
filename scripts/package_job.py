import base64
import io
import json
import os
import zipfile
from pathlib import Path


buffer = io.BytesIO()
with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
    for root in (Path("src"), Path("config")):
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, path.as_posix())

payload = {
    "input": {
        "bundle_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "metadata": {
            "experiment_id": os.environ["EXPERIMENT_ID"],
            "branch": os.environ["GIT_BRANCH"],
            "sha": os.environ["GIT_SHA"],
        },
    },
    "policy": {
        "executionTimeout": int(os.getenv("EXECUTION_TIMEOUT_MS", "3600000")),
        "ttl": int(os.getenv("JOB_TTL_MS", "7200000")),
    },
}
Path("job-request.json").write_text(json.dumps(payload), encoding="utf-8")
