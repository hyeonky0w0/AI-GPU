"""외부 데이터 없는 축소 CPU smoke: 전처리·DCN·checkpoint·평가 산출물 검사."""
from __future__ import annotations
import argparse, tempfile
from pathlib import Path
import numpy as np,pandas as pd,torch
from contract import RAW_FEATURES,ONEHOT,ID,FOLD
from model import CrossLayer,DirectDCN,make_preprocessor
from evaluate import evaluate

def synthetic(n=288):
 rng=np.random.default_rng(42); d={c:rng.normal(size=n) for c in RAW_FEATURES}; d["season"]=np.repeat([2019,2020,2021,2022,2023,2024],n//6)
 for c in ONEHOT: d[c]=rng.choice(["A","B",None],n)
 d["asof_pitcher_n"]=rng.integers(0,250,n); d["game_type"]=rng.choice(["regular","post"],n); x=pd.DataFrame(d); x.loc[::11,"li"]=np.nan; signal=np.nan_to_num(x["inning"].to_numpy())*.15+np.nan_to_num(x["balls_before"].to_numpy())*np.nan_to_num(x["strikes_before"].to_numpy())*.3; x["control_success"]=(signal+rng.normal(size=n)>0).astype("int8"); x[ID]=[f"smoke_{i}" for i in range(n)]; return x
def main():
 p=argparse.ArgumentParser(); p.add_argument("--output-dir"); a=p.parse_args(); torch.manual_seed(42); frame=synthetic(); train=frame[frame.season.lt(2022)]; val=frame[frame.season.eq(2022)]; pre=make_preprocessor(); x=pre.fit_transform(train[RAW_FEATURES]).astype("float32"); xv=pre.transform(val[RAW_FEATURES]).astype("float32"); assert pre.n_features_in_==37
 cross=CrossLayer(x.shape[1]); assert cross(torch.from_numpy(x[:8]),torch.from_numpy(x[:8])).shape==(8,x.shape[1]); rate=train.control_success.mean(); model=DirectDCN(x.shape[1],rate); expected=float(np.log(rate/(1-rate))); assert abs(float(model.output.bias.detach())-expected)<1e-6
 opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.01); xt=torch.from_numpy(x); yt=torch.from_numpy(train.control_success.to_numpy("float32")); losses=[]
 for _ in range(20): opt.zero_grad(); probability=model.probability(xt); loss=((probability-yt)**2).mean(); losses.append(float(loss.detach())); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
 assert losses[-1]<losses[0] and all(torch.isfinite(v).all() for v in model.parameters()); pred=model.probability(torch.from_numpy(xv)).detach().numpy(); assert np.isfinite(pred).all() and np.all((0<=pred)&(pred<=1))
 with tempfile.TemporaryDirectory() as td:
  path=Path(td)/"epoch_1.pt"; torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"epoch":1,"rng":torch.get_rng_state()},path); clone=DirectDCN(x.shape[1],rate); saved=torch.load(path,weights_only=True); clone.load_state_dict(saved["model"]); assert torch.equal(model(torch.from_numpy(xv)),clone(torch.from_numpy(xv)))
 out=Path(a.output_dir) if a.output_dir else Path(tempfile.mkdtemp(prefix="041_smoke_")); refs=[]; oofs=[]
 for year in [2022,2023,2024]:
  g=frame[frame.season.eq(year)].copy(); y=g.control_success.to_numpy(); base=np.clip(.5+.03*np.sin(np.arange(len(g))),.01,.99); pdcn=np.clip(base+.02*np.cos(np.arange(len(g))),.01,.99); refs.append(pd.DataFrame({ID:g[ID].astype(str),FOLD:year,"target":y,"season":year,"game_type":g.game_type,"asof_pitcher_n":g.asof_pitcher_n,"p_cat":base,"p_lgb":base,"p_mlp":base,"p_915":base})); oofs.append(pd.DataFrame({ID:g[ID].astype(str),FOLD:year,"target":y,"p_dcn":pdcn}))
 report=evaluate(pd.concat(oofs),pd.concat(refs),out,{"smoke":True,"checkpoint_resume":True,"evaluate_only_regeneration":True}); required=["direct_dcn_oof_predictions.csv.gz","metrics_by_fold.csv","metrics_overall.csv","blend_analysis.csv","error_diversity.csv","calibration_by_decile.csv","report.json","RESULT.md"]
 assert all((out/f).is_file() for f in required); regenerated=out/"regenerated"; evaluate(pd.concat(oofs),pd.concat(refs),regenerated,{"evaluate_only":True}); assert all((regenerated/f).is_file() for f in required)
 print({"status":"PASS","initial_brier":losses[0],"final_brier":losses[-1],"output_dir":str(out),"required_files":required,"decision_is_diagnostic":report["decision"]})
if __name__=="__main__": main()
