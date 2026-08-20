"""RunPod의 bounded_residual_dcn 전용 어댑터."""
from __future__ import annotations
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--results-dir",required=True); a=p.parse_args(); cfg=json.loads(Path(a.config).read_text(encoding="utf-8")); results=Path(a.results_dir); results.mkdir(parents=True,exist_ok=True)
    if cfg.get("execution_mode")!="bounded_residual_dcn": raise ValueError("전용 adapter mode 불일치")
    env=cfg.get("network_volume_root_env","LG_AIMERS_NETWORK_VOLUME_ROOT"); root=Path(os.getenv(env,cfg["network_volume"]["root"])); (results/"volume_diagnostics.json").write_text(json.dumps({"root":str(root),"exists":root.exists(),"entries":sorted(x.name for x in root.iterdir())[:100] if root.is_dir() else []},ensure_ascii=False,indent=2),encoding="utf-8")
    missing=[str(root/x) for x in cfg["network_volume"]["required_data_files"] if not (root/x).is_file()]
    if missing: raise FileNotFoundError("Network Volume 자산 누락:\n"+"\n".join(missing))
    bundle=Path(a.config).resolve().parent.parent; req=bundle/"requirements.txt"; install=subprocess.run([sys.executable,"-m","pip","install","-r",str(req),"--disable-pip-version-check"],capture_output=True,text=True); (results/"pip_install.log").write_text((install.stdout or "")+(install.stderr or ""),encoding="utf-8")
    if install.returncode: raise RuntimeError("dependency 설치 실패")
    import torch
    gpu={"cuda_available":torch.cuda.is_available(),"torch":torch.__version__,"cuda":torch.version.cuda,"gpu_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}; (results/"gpu_metadata.json").write_text(json.dumps(gpu,ensure_ascii=False,indent=2),encoding="utf-8")
    if not torch.cuda.is_available(): raise RuntimeError("full mode CUDA 필수")
    experiment=bundle/"src"/"experiments"/"040_bounded_residual_dcn"; output=experiment/"outputs"; output.mkdir(exist_ok=True); checkpoint=root/cfg["network_volume"]["checkpoint_directory"]
    command=[sys.executable,str(bundle/"src"/cfg["entrypoint"]),"--phase",cfg["phase"],"--data-root",str(root),"--asset-root",str(root),"--output-dir",str(output),"--checkpoint-dir",str(checkpoint)]
    with (results/"train.log").open("w",encoding="utf-8") as stream: process=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,text=True)
    if process.returncode: raise RuntimeError(f"bounded residual DCN 실패: {process.returncode}")
    shutil.copytree(output,results/"experiment_outputs",dirs_exist_ok=True); shutil.copytree(experiment/"configs",results/"experiment_configs",dirs_exist_ok=True)
    phase=checkpoint/cfg["phase"]
    if phase.is_dir(): shutil.copytree(phase,results/"checkpoints"/cfg["phase"],dirs_exist_ok=True)
    (results/"metrics.json").write_text(json.dumps({"status":"success","execution_mode":cfg["execution_mode"],"phase":cfg["phase"],**gpu},ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__":
    try: main()
    except Exception as exc:
        if "--results-dir" in sys.argv:
            out=Path(sys.argv[sys.argv.index("--results-dir")+1]); out.mkdir(parents=True,exist_ok=True); (out/"error.log").write_text(f"{type(exc).__name__}: {exc}\n",encoding="utf-8"); (out/"metrics.json").write_text(json.dumps({"status":"failed","error_type":type(exc).__name__,"error":str(exc)},ensure_ascii=False),encoding="utf-8")
        raise
