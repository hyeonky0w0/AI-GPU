"""로컬 원본을 RunPod Network Volume 업로드용 디렉터리로 준비하고 검증한다."""
from __future__ import annotations
import argparse, gzip, json, shutil, zipfile
from pathlib import Path
import pandas as pd
from verify_assets import verify

ROOT=Path(__file__).resolve().parents[3]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--destination",required=True); args=ap.parse_args(); dest=Path(args.destination)
    (dest/"assets").mkdir(parents=True,exist_ok=False); (dest/"router_sources/catboost").mkdir(parents=True); (dest/"router_sources/lightgbm").mkdir(parents=True)
    reference=ROOT/"experiments/oofrog_009_mlp_blend.zip"
    if not reference.is_file(): raise FileNotFoundError(f"실제 890 원본 ZIP이 없습니다: {reference}")
    with zipfile.ZipFile(reference) as z:
        member="011_mlp_blend/model/mlp_snap345.pkl"
        if member not in z.namelist(): raise FileNotFoundError(f"실제 890 checkpoint가 ZIP에 없습니다: {member}")
        (dest/"assets/mlp_snap345.pkl").write_bytes(z.read(member))
    for year in [2022,2023,2024]:
        shutil.copy2(ROOT/f"experiments/007_catboost_seed_ensemble/predictions/fold_{year}.npz",dest/f"router_sources/catboost/fold_{year}.npz")
        shutil.copy2(ROOT/f"experiments/020_catboost_lgbm_k10_blend/predictions/fold_{year}.npz",dest/f"router_sources/lightgbm/fold_{year}.npz")
    source=ROOT/"experiments/037_adaptive_router_oof/outputs/oof_predictions.csv"
    cols=["row_id","fold","target","p_mlp"]
    if not source.is_file(): raise FileNotFoundError(f"lightweight OOF 원본이 없습니다: {source}")
    frame=pd.read_csv(source,usecols=cols)
    lightweight=dest/"router_sources/lightweight_oof_predictions.csv.gz"
    with lightweight.open("wb") as raw:
        with gzip.GzipFile(filename="",mode="wb",fileobj=raw,mtime=0) as stream:
            frame.to_csv(stream,index=False,lineterminator="\n")
    print(json.dumps(verify(dest),ensure_ascii=False,indent=2))

if __name__=="__main__": main()
