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
    "real_mlp_oof": ROOT / "experiments" / "039_real_mlp_oof",
    "bounded_crossfit_residual_mlp": ROOT / "experiments" / "040_bounded_crossfit_residual_mlp",
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
        worker = "train_040.py" if mode == "bounded_crossfit_residual_mlp" else "train.py"
        add_file(archive, RUNTIME / "src" / worker, "src/train.py")
        config_name = {"gpu_mlp_smoke":"experiment_036.json", "real_mlp_oof":"experiment_039.json", "bounded_crossfit_residual_mlp":"experiment_040.json"}.get(mode,"experiment.json")
        config = json.loads((RUNTIME / "config" / config_name).read_text(encoding="utf-8"))
        config["execution_mode"] = mode
        if mode == "bounded_crossfit_residual_mlp":
            phase = os.getenv("LG_AIMERS_EXPERIMENT_PHASE", "seed42")
            if phase not in {"seed42", "three_seed", "final_train"}:
                raise ValueError(f"지원하지 않는 040 phase: {phase}")
            config["phase"] = phase
        archive.writestr("config/experiment.yaml", json.dumps(config, ensure_ascii=False, indent=2))
        if mode in {"gpu_mlp_smoke", "real_mlp_oof", "bounded_crossfit_residual_mlp"}:
            add_file(archive, experiment / "requirements.txt", "requirements.txt")
        for path in sorted(experiment.rglob("*")):
            if (path.is_file() and "__pycache__" not in path.parts and "outputs" not in path.parts
                    and not (mode in {"gpu_mlp_smoke", "real_mlp_oof", "bounded_crossfit_residual_mlp"} and path == experiment / "requirements.txt")):
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
