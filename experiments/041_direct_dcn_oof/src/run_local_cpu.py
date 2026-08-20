"""Windows 로컬 CPU 통합 runner."""
from __future__ import annotations
import argparse,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/"src"
def existing(v,label,file=False):
 if not v: raise ValueError(f"{label} 필요")
 p=Path(v).resolve()
 if not (p.is_file() if file else p.is_dir()): raise FileNotFoundError(p)
 return p
def safe_output(v,label):
 if not v: raise ValueError(f"{label} 필요")
 p=Path(v).resolve()
 if "041_direct_dcn_oof" not in str(p): raise ValueError("041 전용 경로만 허용")
 return p
def main():
 p=argparse.ArgumentParser(); p.add_argument("stage",choices=["verify-assets","smoke","seed42","evaluate","three-seed","final-train"]); p.add_argument("--data-root"); p.add_argument("--asset-root"); p.add_argument("--mlp-oof-path"); p.add_argument("--output-dir"); p.add_argument("--checkpoint-dir"); p.add_argument("--cpu-workers",type=int,default=1); p.add_argument("--dataloader-workers",type=int,default=0); p.add_argument("--max-hours",type=float); a=p.parse_args()
 if a.cpu_workers<1 or a.dataloader_workers<0: raise ValueError("worker 계약 실패")
 if a.stage=="smoke":
  command=[sys.executable,str(SRC/"smoke_test.py")]
  if a.output_dir: command += ["--output-dir",str(safe_output(a.output_dir,"--output-dir"))]
 else:
  data=existing(a.data_root,"--data-root"); assets=existing(a.asset_root,"--asset-root"); mlp=existing(a.mlp_oof_path,"--mlp-oof-path",True)
  if a.stage=="verify-assets":
   command=[sys.executable,str(SRC/"inspect_assets.py"),"--data-root",str(data),"--asset-root",str(assets),"--mlp-oof-path",str(mlp)]
   if a.output_dir: command += ["--output",str(safe_output(a.output_dir,"--output-dir")/"asset_verification.json")]
  else:
   out=safe_output(a.output_dir,"--output-dir"); ckpt=safe_output(a.checkpoint_dir,"--checkpoint-dir"); phase={"seed42":"seed42","evaluate":"seed42","three-seed":"three-seed","final-train":"final-train"}[a.stage]; command=[sys.executable,str(SRC/"train_direct.py"),"--phase",phase,"--data-root",str(data),"--asset-root",str(assets),"--mlp-oof-path",str(mlp),"--output-dir",str(out),"--checkpoint-dir",str(ckpt),"--cpu-workers",str(a.cpu_workers),"--dataloader-workers",str(a.dataloader_workers)]
   if a.stage=="evaluate": command.append("--evaluate-only")
   if a.max_hours is not None: command += ["--max-hours",str(a.max_hours)]
 env=os.environ.copy(); env.update({"OMP_NUM_THREADS":str(a.cpu_workers),"MKL_NUM_THREADS":str(a.cpu_workers),"OPENBLAS_NUM_THREADS":str(a.cpu_workers)}); print("실행 명령:",subprocess.list2cmdline(command),flush=True)
 try: result=subprocess.run(command,env=env)
 except KeyboardInterrupt: print("Ctrl+C 전달: 마지막 epoch checkpoint부터 동일 명령으로 재개",flush=True); raise SystemExit(130)
 raise SystemExit(result.returncode)
if __name__=="__main__": main()
