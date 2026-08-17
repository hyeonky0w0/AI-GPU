"""RunPod 비동기 job 제출·상태 대기·작은 결과만 안전하게 회수한다."""
from __future__ import annotations

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


ROOT = Path(__file__).resolve().parents[3]
API_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


def request(path: str, method: str = "GET", body: object | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request_obj = urllib.request.Request(BASE_URL + path, data=data, headers={
        "Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}, method=method)
    try:
        with urllib.request.urlopen(request_obj, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"RunPod HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')}") from exc


def extract_results(encoded: str) -> None:
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    with zipfile.ZipFile(BytesIO(base64.b64decode(encoded))) as archive:
        for item in archive.infolist():
            path = PurePosixPath(item.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"안전하지 않은 결과 경로: {item.filename}")
        archive.extractall(results)


def main() -> None:
    submission = request("/run", "POST", json.loads((ROOT / "job-request.json").read_text(encoding="utf-8")))
    job_id = submission["id"]
    print(f"Submitted RunPod job: {job_id}", flush=True)
    deadline = time.monotonic() + int(os.getenv("POLL_TIMEOUT_SECONDS", "7200"))
    delay = 5
    while True:
        response = request(f"/status/{job_id}")
        state = response.get("status")
        print(f"RunPod job {job_id}: {state}", flush=True)
        if state in TERMINAL:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"RunPod polling 시간 초과: {job_id}")
        time.sleep(delay)
        delay = min(delay + 5, 30)
    (ROOT / "runpod-response.json").write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
    output = response.get("output") or {}
    if output.get("results_zip_b64"):
        extract_results(output["results_zip_b64"])
    if state != "COMPLETED" or output.get("status") != "success":
        print(json.dumps(response, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
