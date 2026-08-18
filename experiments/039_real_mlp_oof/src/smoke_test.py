"""N<=1000 로컬 smoke: checkpoint 계약, 모델 구성, OOF schema와 leakage만 검사한다."""
from __future__ import annotations

import argparse, json, subprocess, sys, tempfile, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from evaluate_router import fit_gate, softmax3

EXP=Path(__file__).resolve().parents[1]; ROOT=EXP.parents[1]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--reference-zip",default=str(ROOT/"experiments/oofrog_009_mlp_blend.zip")); args=ap.parse_args()
    with tempfile.TemporaryDirectory() as td:
        tmp=Path(td); assets=tmp/"assets"; out=tmp/"outputs"; assets.mkdir()
        with zipfile.ZipFile(args.reference_zip) as z:
            (assets/"mlp_snap345.pkl").write_bytes(z.read("011_mlp_blend/model/mlp_snap345.pkl"))
        cmd=[sys.executable,str(EXP/"src/build_mlp_oof.py"),"--data-root",str(ROOT),"--asset-root",str(assets),"--output-dir",str(out),"--smoke"]
        subprocess.run(cmd,check=True)
        oof=pd.read_csv(out/"mlp_oof_smoke.csv.gz"); report=json.loads((out/"smoke_report.json").read_text(encoding="utf-8"))
        expected=["row_id","fold","target","p_mlp_real_890"]
        if oof.columns.tolist()!=expected or len(oof)>1000 or oof.row_id.duplicated().any(): raise AssertionError("smoke OOF schema/row 계약 실패")
        if not np.isfinite(oof.p_mlp_real_890).all() or not oof.p_mlp_real_890.between(0,1).all(): raise AssertionError("smoke 확률 계약 실패")
        leakage=report["leakage"]
        if leakage["target_in_features"] is not False or not all(leakage[k] for k in ["validation_excluded_from_training","future_season_excluded","preprocessor_fit_on_training_only","row_id_unique"]) or report["reference_inference"]["max_absolute_difference"]>1e-12: raise AssertionError("smoke leakage/reference 계약 실패")
        rng=np.random.default_rng(42); x=rng.normal(size=(300,6)); base=np.clip(rng.normal(.5,.05,size=(300,3)),0,1); y=rng.integers(0,2,size=300)
        theta=fit_gate(x,base,y); weights=softmax3(np.column_stack([np.ones(len(x)),x])@theta)
        if weights.shape!=(300,3) or np.max(np.abs(weights.sum(1)-1))>1e-12: raise AssertionError("Router softmax smoke 실패")
        print(json.dumps({"status":"PASS","rows":len(oof),"columns":expected,"folds":oof.fold.value_counts().sort_index().to_dict(),"reference":report["reference_inference"],"leakage":report["leakage"],"router":{"weight_shape":list(weights.shape),"weight_sum_max_error":float(np.max(np.abs(weights.sum(1)-1))) }},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
