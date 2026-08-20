"""Windows 로컬 CPU의 검증/smoke/seed42/evaluate 전용 진입점."""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/"src"
def existing(value,label,directory=False):
    if not value: raise ValueError(f"{label} 인자가 필요합니다")
    path=Path(value).resolve()
    if not (path.is_dir() if directory else path.is_file()): raise FileNotFoundError(f"{label} 경로가 없습니다: {path}")
    return path
def local_path(value,label):
    if not value: raise ValueError(f"{label} 인자가 필요합니다")
    path=Path(value).resolve()
    if "040_bounded_crossfit_residual_mlp" in str(path): raise ValueError(f"기존 Residual MLP 경로 사용 금지: {path}")
    return path

def main():
    p=argparse.ArgumentParser(description="040 bounded residual DCN Windows CPU runner")
    p.add_argument("stage",choices=["verify-assets","smoke","seed42","evaluate"]); p.add_argument("--data-root"); p.add_argument("--asset-root"); p.add_argument("--mlp-oof-path"); p.add_argument("--output-dir"); p.add_argument("--checkpoint-dir"); p.add_argument("--cpu-workers",type=int,default=1); p.add_argument("--dataloader-workers",type=int,default=0); p.add_argument("--max-hours",type=float)
    a=p.parse_args()
    if a.cpu_workers<1 or a.dataloader_workers<0: raise ValueError("worker 수는 cpu>=1, dataloader>=0 이어야 합니다")
    if a.stage=="smoke": command=[sys.executable,str(SRC/"smoke_test.py")]
    else:
        data=existing(a.data_root,"--data-root",True); assets=existing(a.asset_root,"--asset-root",True); mlp=existing(a.mlp_oof_path,"--mlp-oof-path")
        if a.stage=="verify-assets":
            command=[sys.executable,str(SRC/"inspect_assets.py"),"--data-root",str(data),"--asset-root",str(assets),"--mlp-oof-path",str(mlp)]
            if a.output_dir:
                output=Path(a.output_dir).resolve(); output.mkdir(parents=True,exist_ok=True); command += ["--output",str(output/"asset_verification.json")]
        else:
            output=local_path(a.output_dir,"--output-dir"); checkpoint=local_path(a.checkpoint_dir,"--checkpoint-dir")
            command=[sys.executable,str(SRC/"train_residual.py"),"--phase","seed42","--local-cpu","--data-root",str(data),"--asset-root",str(assets),"--mlp-oof-path",str(mlp),"--output-dir",str(output),"--checkpoint-dir",str(checkpoint),"--cpu-workers",str(a.cpu_workers),"--dataloader-workers",str(a.dataloader_workers)]
            if a.stage=="evaluate": command.append("--evaluate-only")
            if a.max_hours is not None: command += ["--max-hours",str(a.max_hours)]
    env=os.environ.copy(); env["OMP_NUM_THREADS"]=str(a.cpu_workers); env["MKL_NUM_THREADS"]=str(a.cpu_workers); env["OPENBLAS_NUM_THREADS"]=str(a.cpu_workers)
    print("실행 명령:",subprocess.list2cmdline(command),flush=True)
    try: completed=subprocess.run(command,env=env)
    except KeyboardInterrupt: print("Ctrl+C를 전달했습니다. 마지막 완료 epoch에서 같은 명령으로 재개하십시오.",flush=True); raise SystemExit(130)
    raise SystemExit(completed.returncode)
if __name__=="__main__": main()
