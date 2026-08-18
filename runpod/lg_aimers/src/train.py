"""기존 RunPod worker가 실행하는 LG Aimers 실험 어댑터.

Docker 이미지의 handler.py 계약(src/train.py, config/experiment.yaml)을 그대로 사용한다.
실험 코드는 GitHub Actions가 job bundle로 전달하며, 원본 데이터와 대형 OOF는
Network Volume에만 둔다.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def gpu_metadata() -> dict[str, object]:
    try:
        import torch

        return {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_version": torch.__version__,
        }
    except Exception as exc:  # GPU 정보 수집 실패가 실험 자체를 가리면 안 된다.
        return {"cuda_available": False, "gpu_metadata_error": f"{type(exc).__name__}: {exc}"}


def require_volume(config: dict[str, object]) -> Path:
    volume = config["network_volume"]
    root = Path(volume["root"])
    missing = [str(root / name) for name in volume["required_data_files"] if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "RunPod Network Volume에 필수 LG Aimers 데이터가 없습니다: " + ", ".join(missing)
        )
    return root


def copy_small_outputs(experiment_dir: Path, results_dir: Path) -> list[str]:
    source = experiment_dir / "outputs"
    if not source.exists():
        return []
    destination = results_dir / "experiment_outputs"
    copied: list[str] = []
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".gz", ".npz", ".json", ".md", ".txt", ".log"}:
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(str(relative))
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    started = time.monotonic()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    mode = config["execution_mode"]
    experiment_dir = Path(__file__).resolve().parent / config["entrypoint"]
    experiment_root = experiment_dir.parents[1]
    metadata: dict[str, object] = {
        "experiment_id": config["experiment_id"],
        "execution_mode": mode,
        "branch": os.getenv("GIT_BRANCH", "unknown"),
        "commit_sha": os.getenv("GIT_SHA", "unknown"),
        **gpu_metadata(),
    }

    if mode == "verify_volume":
        root = require_volume(config)
        metadata["network_volume"] = str(root)
        metadata["status"] = "success"
    elif mode == "gate_smoke":
        process = subprocess.run([sys.executable, str(experiment_dir), "--smoke"], text=True)
        if process.returncode:
            raise RuntimeError(f"gate smoke 실패 (exit_code={process.returncode})")
        metadata["status"] = "success"
    elif mode == "gate_rolling":
        if config["safety"]["allow_full_training"] is not True:
            raise PermissionError("gate_rolling은 설정에서 명시적으로 승인되어야 합니다.")
        root = require_volume(config)
        oof_source = root / config["network_volume"]["oof_directory"]
        if not oof_source.is_dir():
            raise FileNotFoundError(f"OOF 디렉터리가 없습니다: {oof_source}")
        output_dir = experiment_root / "outputs"
        output_dir.mkdir(exist_ok=True)
        oof_link = output_dir / "oof"
        if oof_link.exists() or oof_link.is_symlink():
            raise FileExistsError(f"bundle 안의 OOF 경로가 비어 있지 않습니다: {oof_link}")
        oof_link.symlink_to(oof_source, target_is_directory=True)
        process = subprocess.run([sys.executable, str(experiment_dir)], text=True)
        if process.returncode:
            raise RuntimeError(f"rolling gate 실패 (exit_code={process.returncode})")
        metadata["network_volume"] = str(root)
        metadata["copied_outputs"] = copy_small_outputs(experiment_root, results_dir)
        metadata["status"] = "success"
    elif mode == "gpu_mlp_smoke":
        root = require_volume(config)
        bundle_root = Path(args.config).resolve().parent.parent
        requirements_path = bundle_root / "requirements.txt"
        if not requirements_path.is_file():
            raise FileNotFoundError(f"GPU MLP bundle requirements.txt가 없습니다: {requirements_path}")
        install = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_path),
            "--disable-pip-version-check",
        ], text=True, capture_output=True)
        pip_output = (install.stdout or "") + (install.stderr or "")
        (results_dir / "pip_install.log").write_text(pip_output, encoding="utf-8")
        if install.returncode:
            raise RuntimeError(
                "GPU MLP dependency 설치 실패 (pip install -r requirements.txt):\n"
                + pip_output[-12000:]
            )
        process = subprocess.run([
            sys.executable, str(experiment_dir), "--data-root", str(root),
            "--config", str(args.config),
        ], text=True)
        if process.returncode:
            raise RuntimeError(f"GPU MLP smoke 실패 (exit_code={process.returncode})")
        metrics_path = experiment_root / "outputs" / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"GPU MLP metrics가 없습니다: {metrics_path}")
        metadata.update(json.loads(metrics_path.read_text(encoding="utf-8")))
        metadata["network_volume"] = str(root)
        metadata["copied_outputs"] = copy_small_outputs(experiment_root, results_dir)
        for filename in ("train.log", "error.log"):
            source = experiment_root / "outputs" / filename
            target = results_dir / filename
            if source.is_file():
                shutil.copy2(source, target)
            elif filename == "error.log":
                target.write_text("", encoding="utf-8")
        metadata["status"] = "success"
    elif mode == "real_mlp_oof":
        if config["safety"]["allow_full_training"] is not True:
            raise PermissionError("real_mlp_oof full training이 config에서 승인되지 않았습니다.")
        root = require_volume(config)
        gpu = gpu_metadata()
        if not gpu.get("cuda_available"):
            raise RuntimeError("GPU runner 계약 위반: CUDA device를 찾지 못했습니다.")
        bundle_root = Path(args.config).resolve().parent.parent
        requirements_path = bundle_root / "requirements.txt"
        install = subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_path), "--disable-pip-version-check"], text=True, capture_output=True)
        (results_dir / "pip_install.log").write_text((install.stdout or "")+(install.stderr or ""),encoding="utf-8")
        if install.returncode: raise RuntimeError("039 dependency 설치 실패")
        output_dir = experiment_root / "outputs"; output_dir.mkdir(exist_ok=True)
        build = subprocess.run([sys.executable,str(experiment_dir),"--data-root",str(root),"--asset-root",str(root/"assets"),"--output-dir",str(output_dir)],text=True)
        if build.returncode: raise RuntimeError(f"real 890 MLP OOF 실패 (exit_code={build.returncode})")
        router = Path(__file__).resolve().parent / config["router_entrypoint"]
        evaluate = subprocess.run([sys.executable,str(router),"--data-root",str(root),"--asset-root",str(root),"--mlp-oof",str(output_dir/"mlp_oof_predictions.csv.gz"),"--output-dir",str(output_dir)],text=True)
        if evaluate.returncode: raise RuntimeError(f"Router 평가 실패 (exit_code={evaluate.returncode})")
        metadata.update(gpu); metadata["network_volume"]=str(root); metadata["copied_outputs"]=copy_small_outputs(experiment_root,results_dir); metadata["status"]="success"
    else:
        raise ValueError(f"지원하지 않는 execution_mode: {mode}")

    worker_seconds = round(time.monotonic() - started, 6)
    metadata["worker_seconds"] = worker_seconds
    metadata.setdefault("train_seconds", worker_seconds)
    (results_dir / "metrics.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # 기존 handler는 subprocess 실패 시에도 결과 폴더를 ZIP으로 회수한다.
        # 여기서 먼저 실패 metric을 남겨야 CI artifact만으로 실패 원인을 판별할 수 있다.
        if "--results-dir" in sys.argv:
            failure_dir = Path(sys.argv[sys.argv.index("--results-dir") + 1])
            failure_dir.mkdir(parents=True, exist_ok=True)
            (failure_dir / "metrics.json").write_text(json.dumps({
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "branch": os.getenv("GIT_BRANCH", "unknown"),
                "commit_sha": os.getenv("GIT_SHA", "unknown"),
                **gpu_metadata(),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            (failure_dir / "error.log").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise
