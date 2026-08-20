"""RunPod worker 전용 040 어댑터. 기존 실행 mode와 분리한다."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--results-dir",required=True); args=parser.parse_args()
    config=json.loads(Path(args.config).read_text(encoding="utf-8")); results=Path(args.results_dir); results.mkdir(parents=True,exist_ok=True)
    if config.get("execution_mode")!="bounded_crossfit_residual_mlp": raise ValueError("train_040은 040 mode만 실행합니다")
    env_name=config.get("network_volume_root_env","LG_AIMERS_NETWORK_VOLUME_ROOT"); root=Path(os.getenv(env_name,config["network_volume"]["root"]))
    diagnostics={"selected_root":str(root),"source":f"environment:{env_name}" if os.getenv(env_name) else "config:network_volume.root","exists":root.exists(),"realpath":str(root.resolve()),"entries":sorted(p.name for p in root.iterdir())[:100] if root.is_dir() else []}
    (results/"volume_diagnostics.json").write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2),encoding="utf-8")
    missing=[str(root/p) for p in config["network_volume"]["required_data_files"] if not (root/p).is_file()]
    if missing: raise FileNotFoundError(f"040 필수 Network Volume 자산 누락 (root={root}):\n"+"\n".join(f"  - {x}" for x in missing))
    bundle=Path(args.config).resolve().parent.parent; requirements=bundle/"requirements.txt"
    install=subprocess.run([sys.executable,"-m","pip","install","-r",str(requirements),"--disable-pip-version-check"],text=True,capture_output=True)
    (results/"pip_install.log").write_text((install.stdout or "")+(install.stderr or ""),encoding="utf-8")
    if install.returncode: raise RuntimeError("040 dependency 설치 실패")
    import torch
    gpu={"cuda_available":torch.cuda.is_available(),"torch":torch.__version__,"cuda":torch.version.cuda,"gpu_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    (results/"gpu_metadata.json").write_text(json.dumps(gpu,ensure_ascii=False,indent=2),encoding="utf-8")
    if not torch.cuda.is_available(): raise RuntimeError("040 full mode는 CUDA GPU가 필수입니다")
    utilization=subprocess.run(["nvidia-smi","--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu","--format=csv,noheader,nounits"],text=True,capture_output=True)
    (results/"gpu_utilization_before_train.log").write_text((utilization.stdout or "")+(utilization.stderr or ""),encoding="utf-8")
    if utilization.returncode: raise RuntimeError("nvidia-smi GPU utilization 확인 실패")
    experiment=bundle/"src"/"experiments"/"040_bounded_crossfit_residual_mlp"; output=experiment/"outputs"; output.mkdir(parents=True,exist_ok=True)
    checkpoint=root/config["network_volume"]["checkpoint_directory"]
    command=[sys.executable,str(bundle/"src"/config["entrypoint"]),"--phase",config["phase"],"--data-root",str(root),"--asset-root",str(root),"--output-dir",str(output),"--checkpoint-dir",str(checkpoint)]
    log=results/"train.log"
    with log.open("w",encoding="utf-8") as stream:
        process=subprocess.run(command,text=True,stdout=stream,stderr=subprocess.STDOUT)
    if process.returncode: raise RuntimeError(f"040 {config['phase']} 실패 (exit_code={process.returncode}); train.log 확인")
    target=results/"experiment_outputs"; shutil.copytree(output,target,dirs_exist_ok=True)
    for name in ["README.md","RESULT.md"]: shutil.copy2(experiment/name,results/name)
    shutil.copy2(Path(args.config),results/"runpod_experiment_config.json")
    shutil.copytree(experiment/"configs",results/"experiment_configs",dirs_exist_ok=True)
    phase_checkpoints=checkpoint/config["phase"]
    if phase_checkpoints.is_dir(): shutil.copytree(phase_checkpoints,results/"checkpoints"/config["phase"],dirs_exist_ok=True)
    metadata={"status":"success","experiment_id":config["experiment_id"],"execution_mode":config["execution_mode"],"phase":config["phase"],"network_volume":str(root),**gpu}
    (results/"metrics.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    try: main()
    except Exception as exc:
        if "--results-dir" in sys.argv:
            out=Path(sys.argv[sys.argv.index("--results-dir")+1]); out.mkdir(parents=True,exist_ok=True); (out/"error.log").write_text(f"{type(exc).__name__}: {exc}\n",encoding="utf-8"); (out/"metrics.json").write_text(json.dumps({"status":"failed","error_type":type(exc).__name__,"error_message":str(exc)},ensure_ascii=False,indent=2),encoding="utf-8")
        raise
