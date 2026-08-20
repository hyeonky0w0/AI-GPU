"""915 rolling OOF 원본 schema/SHA/정렬/Brier 감사."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from build_dataset import build_frame
from contract import *

def inspect(asset_root:Path,data_root:Path,mlp_path:Path|None=None,require_final=False):
    cfg=load_json(root()/"configs/assets.json"); checked={}
    for rel,expected in cfg["sha256"].items():
        path=mlp_path if rel==cfg["sources"]["real_890_mlp_oof"] and mlp_path else asset_root/rel
        if not path or not path.is_file(): raise FileNotFoundError(path)
        actual=sha256(path)
        if actual!=expected: raise ValueError(f"SHA256 불일치: {rel}: {actual}")
        checked[rel]={"sha256":actual,"bytes":path.stat().st_size}
    if require_final:
        final=cfg["final_test_contract"]
        if not final.get("sha256"): raise PermissionError("final_train 차단: exact 915 test prediction SHA256 미등록")
        p=asset_root/cfg["sources"]["final_test_predictions"]
        if sha256(p)!=final["sha256"]: raise ValueError("915 test prediction SHA256 불일치")
    frame=build_frame(data_root,asset_root,mlp_path); folds=[]
    for year,g in frame.groupby(FOLD): folds.append({"year":int(year),"rows":len(g),"brier_915":float(np.mean((g.p_915-g.target)**2)),"prediction_mean":float(g.p_915.mean()),"target_rate":float(g.target.mean())})
    outer=frame[frame.fold.isin([2023,2024])]
    return {"status":"PASS","checked_assets":checked,"rows":len(frame),"folds":folds,
            "brier_915_all":float(np.mean((frame.p_915-frame.target)**2)),"brier_915_outer":float(np.mean((outer.p_915-outer.target)**2))}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--asset-root",required=True); p.add_argument("--data-root",required=True); p.add_argument("--mlp-oof-path"); p.add_argument("--output"); p.add_argument("--require-final",action="store_true"); a=p.parse_args()
    r=inspect(Path(a.asset_root),Path(a.data_root),Path(a.mlp_oof_path) if a.mlp_oof_path else None,a.require_final); s=json.dumps(r,ensure_ascii=False,indent=2)
    if a.output: Path(a.output).write_text(s,encoding="utf-8")
    print(s)
if __name__=="__main__": main()
