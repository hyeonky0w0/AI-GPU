"""LG Aimers의 필요한 코드만 기존 RunPod worker 형식으로 압축한다."""
from __future__ import annotations

import base64
import io
import json
import os
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "runpod" / "lg_aimers"
EXPERIMENTS = {
    "gate_smoke": ROOT / "experiments" / "035_anchored_dynamic_moe",
    "verify_volume": ROOT / "experiments" / "035_anchored_dynamic_moe",
    "gate_rolling": ROOT / "experiments" / "035_anchored_dynamic_moe",
    "gpu_mlp_smoke": ROOT / "experiments" / "036_gpu_tabular_mlp_smoke",
}
MAX_BUNDLE_BYTES = 6 * 1024 * 1024


def add_file(archive: zipfile.ZipFile, source: Path, target: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    archive.write(source, target)


def main() -> None:
    mode = os.getenv("LG_AIMERS_EXECUTION_MODE", "gate_smoke")
    if mode not in EXPERIMENTS:
        raise ValueError(f"지원하지 않는 LG_AIMERS_EXECUTION_MODE: {mode}")
    experiment = EXPERIMENTS[mode]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        add_file(archive, RUNTIME / "src" / "train.py", "src/train.py")
        config_name = "experiment.json" if mode != "gpu_mlp_smoke" else "experiment_036.json"
        config = json.loads((RUNTIME / "config" / config_name).read_text(encoding="utf-8"))
        config["execution_mode"] = mode
        archive.writestr("config/experiment.yaml", json.dumps(config, ensure_ascii=False, indent=2))
        for path in sorted(experiment.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and "outputs" not in path.parts:
                archive.write(path, (Path("src") / "experiments" / experiment.name / path.relative_to(experiment)).as_posix())

    raw = buffer.getvalue()
    if len(raw) > MAX_BUNDLE_BYTES:
        raise RuntimeError(f"실험 bundle이 {len(raw) / 1024 / 1024:.2f} MiB입니다. 제한은 6 MiB입니다.")
    payload = {
        "input": {
            "bundle_b64": base64.b64encode(raw).decode("ascii"),
            "metadata": {
                "experiment_id": config["experiment_id"],
                "branch": os.environ["GIT_BRANCH"],
                "sha": os.environ["GIT_SHA"],
            },
        }
    }
    (ROOT / "job-request.json").write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
