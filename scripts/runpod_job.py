import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path, PurePosixPath


API_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


def request(path, method="GET", body=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(BASE_URL + path, data=data, headers=HEADERS, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"RunPod HTTP {exc.code}: {detail}") from exc


submission = request("/run", "POST", json.loads(Path("job-request.json").read_text()))
job_id = submission["id"]
print(f"Submitted RunPod job: {job_id}", flush=True)

deadline = time.monotonic() + int(os.getenv("POLL_TIMEOUT_SECONDS", "7200"))
delay = 5
while True:
    status = request(f"/status/{job_id}")
    state = status.get("status")
    print(f"RunPod job {job_id}: {state}", flush=True)
    if state in TERMINAL:
        break
    if time.monotonic() >= deadline:
        raise TimeoutError(f"polling timed out for job {job_id}")
    time.sleep(delay)
    delay = min(delay + 5, 30)

Path("runpod-response.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
output = status.get("output") or {}
encoded = output.get("results_zip_b64")
if encoded:
    results = Path("results")
    results.mkdir(exist_ok=True)
    with zipfile.ZipFile(BytesIO(base64.b64decode(encoded))) as archive:
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe result path: {member.filename}")
        archive.extractall(results)

if state != "COMPLETED" or output.get("status") != "success":
    print(json.dumps(status, indent=2), file=sys.stderr)
    sys.exit(1)
