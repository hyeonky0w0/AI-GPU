"""로컬 자산을 RunPod Network Volume 업로드용 디렉터리로 준비한다. 자동 실행하지 않는다."""
from __future__ import annotations
import argparse, shutil, zipfile
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--destination",required=True); args=ap.parse_args(); dest=Path(args.destination)
    (dest/"assets").mkdir(parents=True,exist_ok=False); (dest/"router_sources/catboost").mkdir(parents=True); (dest/"router_sources/lightgbm").mkdir(parents=True)
    with zipfile.ZipFile(ROOT/"experiments/oofrog_009_mlp_blend.zip") as z:
        (dest/"assets/mlp_snap345.pkl").write_bytes(z.read("011_mlp_blend/model/mlp_snap345.pkl"))
    for year in [2022,2023,2024]:
        shutil.copy2(ROOT/f"experiments/007_catboost_seed_ensemble/predictions/fold_{year}.npz",dest/f"router_sources/catboost/fold_{year}.npz")
        shutil.copy2(ROOT/f"experiments/020_catboost_lgbm_k10_blend/predictions/fold_{year}.npz",dest/f"router_sources/lightgbm/fold_{year}.npz")
    source=ROOT/"experiments/037_adaptive_router_oof/outputs/oof_predictions.csv"
    cols=["row_id","fold","target","p_mlp"]
    pd.read_csv(source,usecols=cols).to_csv(dest/"router_sources/lightweight_oof_predictions.csv.gz",index=False,compression="gzip")
    print(dest.resolve())

if __name__=="__main__": main()
