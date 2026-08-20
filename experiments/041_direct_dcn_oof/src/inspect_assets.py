"""입력 자산 SHA/schema/기준 Brier 검사."""
import argparse
from pathlib import Path
import numpy as np
from contract import *
from build_dataset import build_reference_frame
def inspect(data_root,asset_root,mlp_path=None,require_final=False):
 cfg=load_json(root()/"configs/assets.json"); checked={}
 for rel,expected in cfg["sha256"].items():
  p=Path(mlp_path) if mlp_path and rel==cfg["sources"]["real_890_mlp_oof"] else Path(asset_root)/rel
  if not p.is_file(): raise FileNotFoundError(p)
  actual=sha256(p)
  if actual!=expected: raise ValueError(f"SHA256 불일치: {rel}")
  checked[rel]={"sha256":actual,"bytes":p.stat().st_size}
 if require_final:
  final=cfg["final_test_contract"]
  if not final.get("sha256"): raise PermissionError("exact 915 test prediction SHA 미등록: final-train fail-closed")
  final_path=Path(asset_root)/cfg["sources"]["final_test_predictions"]
  if not final_path.is_file() or sha256(final_path)!=final["sha256"]: raise ValueError("exact 915 test prediction SHA 불일치")
 ref=build_reference_frame(data_root,asset_root,mlp_path)
 folds=[{"fold":int(y),"rows":len(g),"brier_915":float(np.mean((g.p_915-g.target)**2))} for y,g in ref.groupby(FOLD)]
 return {"status":"PASS","checked_assets":checked,"rows":len(ref),"folds":folds}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--data-root",required=True); p.add_argument("--asset-root",required=True); p.add_argument("--mlp-oof-path"); p.add_argument("--output"); p.add_argument("--require-final",action="store_true"); a=p.parse_args(); r=inspect(a.data_root,a.asset_root,a.mlp_oof_path,a.require_final)
 if a.output: atomic_json(a.output,r)
 print(r)
if __name__=="__main__": main()
