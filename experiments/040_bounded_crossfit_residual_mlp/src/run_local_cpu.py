"""Windows 로컬 CPU에서 040 단계를 안전하게 실행하는 전용 진입점."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"src"


def require_path(value: str | None, label: str, directory: bool = False) -> Path:
    if not value: raise ValueError(f"{label} 인자가 필요합니다")
    path=Path(value).resolve()
    valid=path.is_dir() if directory else path.is_file()
    if not valid: raise FileNotFoundError(f"{label} 경로가 없습니다: {path}")
    return path


def main() -> None:
    parser=argparse.ArgumentParser(description="040 bounded residual MLP 로컬 CPU runner")
    parser.add_argument("stage",choices=["verify-assets","smoke","seed42","three-seed","evaluate","final-train"])
    parser.add_argument("--data-root"); parser.add_argument("--asset-root"); parser.add_argument("--mlp-oof-path")
    parser.add_argument("--output-dir"); parser.add_argument("--checkpoint-dir")
    parser.add_argument("--cpu-workers",type=int,default=1); parser.add_argument("--max-hours",type=float)
    parser.add_argument("--memory-reserve-gb",type=float,default=4.0)
    parser.add_argument("--evaluation-phase",choices=["seed42","three_seed"],default="seed42")
    args=parser.parse_args()
    if args.cpu_workers<1: raise ValueError("--cpu-workers는 1 이상이어야 합니다")
    if args.stage=="smoke":
        command=[sys.executable,str(SRC/"smoke_test.py")]
    else:
        data=require_path(args.data_root,"--data-root",directory=True); assets=require_path(args.asset_root,"--asset-root",directory=True); mlp=require_path(args.mlp_oof_path,"--mlp-oof-path")
        if args.stage=="verify-assets":
            command=[sys.executable,str(SRC/"inspect_assets.py"),"--data-root",str(data),"--asset-root",str(assets),"--mlp-oof-path",str(mlp)]
            if args.output_dir:
                output=Path(args.output_dir).resolve(); output.mkdir(parents=True,exist_ok=True); command += ["--output",str(output/"asset_verification.json")]
        else:
            if not args.output_dir or not args.checkpoint_dir: raise ValueError("학습/evaluate 단계에는 --output-dir과 --checkpoint-dir이 필요합니다")
            phase={"seed42":"seed42","three-seed":"three_seed","evaluate":args.evaluation_phase,"final-train":"final_train"}[args.stage]
            command=[sys.executable,str(SRC/"train_residual.py"),"--phase",phase,"--local-cpu","--data-root",str(data),"--asset-root",str(assets),"--mlp-oof-path",str(mlp),"--output-dir",str(Path(args.output_dir).resolve()),"--checkpoint-dir",str(Path(args.checkpoint_dir).resolve()),"--cpu-workers",str(args.cpu_workers),"--memory-reserve-gb",str(args.memory_reserve_gb)]
            if args.max_hours is not None: command += ["--max-hours",str(args.max_hours)]
            if args.stage=="evaluate": command.append("--evaluate-only")
    environment=os.environ.copy(); environment["OMP_NUM_THREADS"]=str(args.cpu_workers); environment["MKL_NUM_THREADS"]=str(args.cpu_workers); environment["OPENBLAS_NUM_THREADS"]=str(args.cpu_workers)
    print("실행 명령:",subprocess.list2cmdline(command),flush=True)
    try:
        completed=subprocess.run(command,env=environment)
    except KeyboardInterrupt:
        print("Ctrl+C 요청을 전달했습니다. 마지막 완료 epoch checkpoint에서 재개할 수 있습니다.",flush=True); raise SystemExit(130)
    raise SystemExit(completed.returncode)


if __name__=="__main__": main()
