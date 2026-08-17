import base64
import io
import json
import os
import zipfile
from pathlib import Path


buffer = io.BytesIO()

with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
    for root in (Path("src"), Path("config")):
        if not root.exists():
            raise FileNotFoundError(root)

        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, path.as_posix())

raw = buffer.getvalue()

# /run has a 10 MB payload limit. Keep substantial headroom for JSON/base64 overhead.
if len(raw) > 6 * 1024 * 1024:
    raise RuntimeError(
        f"Experiment bundle is {len(raw) / 1024 / 1024:.2f} MiB. "
        "Keep code/config below 6 MiB or move large assets to shared storage."
    )

payload = {
    "input": {
        "bundle_b64": base64.b64encode(raw).decode("ascii"),
        "metadata": {
            "experiment_id": os.environ["EXPERIMENT_ID"],
            "branch": os.environ["GIT_BRANCH"],
            "sha": os.environ["GIT_SHA"],
        },
    }
}

Path("job-request.json").write_text(
    json.dumps(payload),
    encoding="utf-8",
)
