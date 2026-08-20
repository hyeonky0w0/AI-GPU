"""Direct DCN temporal rolling OOF 학습기. CPU checkpoint/resume를 지원한다."""
from __future__ import annotations
import argparse, gc, os, platform, random, time
from pathlib import Path
import joblib, numpy as np, pandas as pd, psutil, torch
from torch.utils.data import DataLoader,TensorDataset
from contract import *
from build_dataset import load_train,build_reference_frame
from inspect_assets import inspect
from model import DirectDCN,make_preprocessor
from evaluate import evaluate

def seed_all(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
def write_torch(path,value): path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); torch.save(value,tmp); os.replace(tmp,path)
def predict(model,x,batch,device):
 model.eval(); out=[]
 with torch.no_grad():
  for i in range(0,len(x),batch): out.append(model.probability(torch.from_numpy(x[i:i+batch]).to(device)).cpu().numpy())
 return np.concatenate(out)
def signature(config,train,pred,seed,stage,input_sha): return object_sha({"config":config,"train_ids":object_sha(train[ID].astype(str).tolist()),"pred_ids":object_sha(pred[ID].astype(str).tolist()),"seed":seed,"stage":stage,"input_sha":input_sha})
def fit(train,pred,seed,epochs,cfg,device,stage,ckpt,early=True):
 target=Path(ckpt)/stage/f"seed_{seed}"; target.mkdir(parents=True,exist_ok=True); sig=signature({k:v for k,v in cfg.items() if not k.startswith("_")},train,pred,seed,stage,cfg["_input_sha"]); marker=target/"complete.json"; raw_path=target/"predictions.npy"; pre_path=target/"preprocessor.joblib"
 if marker.is_file() and raw_path.is_file():
  m=load_json(marker)
  if m.get("signature")==sig: return np.load(raw_path),int(m["best_epoch"]),True
 if pre_path.is_file() and (target/"preprocessor.json").is_file() and load_json(target/"preprocessor.json").get("signature")==sig: pre=joblib.load(pre_path)
 else:
  pre=make_preprocessor(); pre.fit(train[RAW_FEATURES]); tmp=pre_path.with_suffix(".joblib.tmp"); joblib.dump(pre,tmp); os.replace(tmp,pre_path); atomic_json(target/"preprocessor.json",{"signature":sig,"fit_seasons":sorted(map(int,train.season.unique()))})
 x=pre.transform(train[RAW_FEATURES]).astype("float32"); xp=pre.transform(pred[RAW_FEATURES]).astype("float32"); seed_all(seed); model=DirectDCN(x.shape[1],train[TARGET].mean()).to(device); opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"]); gen=torch.Generator().manual_seed(seed); loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(train[TARGET].to_numpy("float32"))),batch_size=cfg["batch_size"],shuffle=True,generator=gen,num_workers=cfg["_dataloader_workers"])
 best=float("inf"); best_epoch=0; wait=0; start=1; files=sorted(target.glob("epoch_*.pt"),key=lambda p:int(p.stem.split("_")[-1]))
 if files:
  saved=torch.load(files[-1],map_location=device,weights_only=False)
  if saved.get("signature")==sig:
   model.load_state_dict(saved["model"]); opt.load_state_dict(saved["optimizer"]); gen.set_state(saved["generator_state"]); random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"]); torch.set_rng_state(saved["torch_rng"]); best=saved["best"]; best_epoch=saved["best_epoch"]; wait=saved["wait"]; start=saved["epoch"]+1; print(f"[resume] {stage} seed={seed} epoch={start-1}",flush=True)
 for epoch in range(start,epochs+1):
  began=time.perf_counter(); model.train(); total=0.
  for xb,yb in loader:
   xb,yb=xb.to(device),yb.to(device); opt.zero_grad(set_to_none=True); p=model.probability(xb); loss=((p-yb)**2).mean()
   if not torch.isfinite(loss): raise FloatingPointError("비유한 Brier loss")
   loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["gradient_clip"]); opt.step(); total+=float(loss.detach())*len(xb)
  score=float(np.mean((predict(model,xp,cfg["predict_batch_size"],device)-pred[TARGET].to_numpy())**2)) if early else total/len(train)
  if score<best-1e-8: best=score; best_epoch=epoch; wait=0
  else: wait+=1
  if not early: best_epoch=epoch
  write_torch(target/f"epoch_{epoch}.pt",{"signature":sig,"epoch":epoch,"model":model.state_dict(),"optimizer":opt.state_dict(),"generator_state":gen.get_state(),"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"best":best,"best_epoch":best_epoch,"wait":wait,"input_dim":x.shape[1]})
  elapsed=time.perf_counter()-began; eta=elapsed*(epochs-epoch); rss=psutil.Process().memory_info().rss/2**20; print(f"[epoch] stage={stage} seed={seed} {epoch}/{epochs} inner_brier={score:.9f} seconds={elapsed:.1f} ETA={eta/3600:.2f}h RSS={rss:.1f}MiB",flush=True)
  if cfg.get("_deadline") and time.monotonic()>=cfg["_deadline"]: raise TimeoutError("--max-hours 도달; checkpoint 저장 완료")
  if early and wait>=cfg["early_stopping_patience"]: break
 saved=torch.load(target/f"epoch_{best_epoch}.pt",map_location=device,weights_only=False); model.load_state_dict(saved["model"]); probability=predict(model,xp,cfg["predict_batch_size"],device); tmp=raw_path.with_suffix(".npy.tmp");
 with tmp.open("wb") as f: np.save(f,probability)
 os.replace(tmp,raw_path); atomic_json(marker,{"signature":sig,"best_epoch":best_epoch,"completed":True}); del x,xp,model; gc.collect(); return probability,best_epoch,False
def train_oof(data,seeds,cfg,device,ckpt,smoke=False):
 outputs=[]; epochs={}; resume={}
 for year,years in OUTER.items():
  outer_train=data[data.season.isin(years)].copy(); outer_eval=data[data.season.eq(year)].copy(); inner_year=INNER_VALIDATION[year]; inner_train=outer_train[outer_train.season.lt(inner_year)].copy(); inner_val=outer_train[outer_train.season.eq(inner_year)].copy()
  if smoke:
   n=cfg["smoke_rows_per_season"]; outer_train=outer_train.groupby("season",group_keys=False).head(n); outer_eval=outer_eval.head(n); inner_train=inner_train.groupby("season",group_keys=False).head(n); inner_val=inner_val.head(n)
  if set(outer_train[ID])&set(outer_eval[ID]) or not (outer_train.season<year).all() or not (inner_train.season<inner_year).all(): raise ValueError("temporal leakage")
  inner_epochs=[]
  for seed in seeds:
   _,e,r=fit(inner_train,inner_val,seed,cfg["epochs"],cfg,device,f"outer_{year}_inner",ckpt,True); inner_epochs.append(e); resume[f"{year}_inner_{seed}"]=r
  selected=max(1,round(np.mean(inner_epochs))); preds=[]
  for seed in seeds:
   p,e,r=fit(outer_train,outer_eval,seed,selected,cfg,device,f"outer_{year}_refit",ckpt,False); preds.append(p); epochs[f"{year}_{seed}"]={"inner_best":inner_epochs[seeds.index(seed)],"refit":e}; resume[f"{year}_refit_{seed}"]=r
  p=np.mean(preds,axis=0)
  if not np.isfinite(p).all() or not np.all((p>=0)&(p<=1)): raise ValueError("probability 계약 실패")
  outputs.append(pd.DataFrame({ID:outer_eval[ID].astype(str),FOLD:year,"target":outer_eval[TARGET].astype("int8"),"p_dcn":p}))
 return pd.concat(outputs,ignore_index=True),epochs,resume
def final_train(data,data_root,asset_root,seeds,cfg,device,ckpt,out,prior):
 test=pd.read_csv(Path(data_root)/"test.csv",usecols=[ID,*RAW_FEATURES],low_memory=False); assets=load_json(root()/"configs/assets.json"); base=pd.read_csv(Path(asset_root)/assets["sources"]["final_test_predictions"])
 if not np.array_equal(test[ID].astype(str),base[ID].astype(str)) or base[ID].duplicated().any() or not np.isfinite(base.p_915).all(): raise ValueError("final 915/test row_id 계약 실패")
 historical=[v["inner_best"] for v in prior.get("metadata",{}).get("best_epochs",{}).values() if isinstance(v,dict) and "inner_best" in v]
 if not historical: raise PermissionError("사전 선택 epoch 기록 없음")
 selected=max(1,round(float(np.mean(historical)))); probabilities=[]
 for seed in seeds:
  p,_,_=fit(data,test,seed,selected,cfg,device,"final_train",ckpt,False); probabilities.append(p)
 p_dcn=np.mean(probabilities,axis=0); weight=float(prior["deployable_weights"].get("2024",prior["deployable_weights"].get(2024,0))); probability=(1-weight)*base.p_915.to_numpy(float)+weight*p_dcn
 submission=pd.DataFrame({ID:test[ID].to_numpy(),TARGET:probability})
 if not np.array_equal(submission[ID].astype(str),test[ID].astype(str)) or submission[ID].duplicated().any() or not np.isfinite(probability).all() or not np.all((0<=probability)&(probability<=1)): raise ValueError("submission 계약 실패")
 out=Path(out); out.mkdir(parents=True,exist_ok=True); tmp=out/"submission.csv.tmp"; submission.to_csv(tmp,index=False); os.replace(tmp,out/"submission.csv"); atomic_json(out/"report.json",{"status":"completed","phase":"final-train","seeds":seeds,"selected_epoch":selected,"fixed_blend_weight":weight,"rows":len(test)})
def main():
 p=argparse.ArgumentParser(); p.add_argument("--phase",choices=["seed42","three-seed","final-train"],default="seed42"); p.add_argument("--data-root",required=True); p.add_argument("--asset-root",required=True); p.add_argument("--mlp-oof-path"); p.add_argument("--output-dir",required=True); p.add_argument("--checkpoint-dir",required=True); p.add_argument("--cpu-workers",type=int,default=1); p.add_argument("--dataloader-workers",type=int,default=0); p.add_argument("--max-hours",type=float); p.add_argument("--smoke",action="store_true"); p.add_argument("--evaluate-only",action="store_true"); a=p.parse_args()
 if a.cpu_workers<1 or a.dataloader_workers<0: raise ValueError("worker 계약 실패")
 cfg=load_json(root()/"configs/experiment.json"); cfg["_dataloader_workers"]=a.dataloader_workers; cfg["_deadline"]=time.monotonic()+a.max_hours*3600 if a.max_hours else None; audit=inspect(a.data_root,a.asset_root,a.mlp_oof_path,a.phase=="final-train")
 if a.phase=="final-train":
  prior_path=Path(a.checkpoint_dir)/"three-seed"/"report.json"
  if not prior_path.is_file(): prior_path=Path(a.checkpoint_dir)/"seed42"/"report.json"
  if not prior_path.is_file(): raise PermissionError("검증 통과 report 없음")
  prior=load_json(prior_path)
  if not prior.get("seed42_extend"): raise PermissionError("검증 통과 조건 미충족")
  torch.set_num_threads(a.cpu_workers); torch.set_num_interop_threads(1); cfg["_input_sha"]=sha256(Path(a.data_root)/"train.csv"); cfg["_dataloader_workers"]=a.dataloader_workers; cfg["_deadline"]=time.monotonic()+a.max_hours*3600 if a.max_hours else None; data=load_train(a.data_root); final_train(data,a.data_root,a.asset_root,prior.get("metadata",{}).get("seeds",[42]),cfg,torch.device("cpu"),Path(a.checkpoint_dir)/"final-train",a.output_dir,prior); return
 if a.phase=="three-seed":
  prior=Path(a.checkpoint_dir)/"seed42"/"report.json"
  if not prior.is_file() or not load_json(prior).get("seed42_extend"): raise PermissionError("seed42 통과 marker 없음")
 torch.set_num_threads(a.cpu_workers); torch.set_num_interop_threads(1); device=torch.device("cpu"); data=load_train(a.data_root); cfg["_input_sha"]=sha256(Path(a.data_root)/"train.csv"); seeds=cfg["seeds"][a.phase]; ckpt=Path(a.checkpoint_dir)/a.phase
 if a.smoke: cfg["epochs"]=2
 if a.evaluate_only:
  parts=[]
  for year in YEARS:
   ids=data[data.season.eq(year)][ID].astype(str).to_numpy(); target=data[data.season.eq(year)][TARGET].to_numpy(); probabilities=[]
   for seed in seeds:
    path=ckpt/f"outer_{year}_refit"/f"seed_{seed}"/"predictions.npy"
    if not path.is_file(): raise FileNotFoundError(path)
    probabilities.append(np.load(path))
   parts.append(pd.DataFrame({ID:ids,FOLD:year,"target":target,"p_dcn":np.mean(probabilities,axis=0)}))
  oof=pd.concat(parts,ignore_index=True); epoch_info={}; resumed={"evaluate_only":True}
 else: oof,epoch_info,resumed=train_oof(data,seeds,cfg,device,ckpt,a.smoke)
 reference=build_reference_frame(a.data_root,a.asset_root,a.mlp_oof_path)
 if a.smoke: reference=pd.concat([reference[reference.fold.eq(y)].head(cfg["smoke_rows_per_season"]) for y in YEARS],ignore_index=True)
 meta={"phase":a.phase,"seeds":seeds,"best_epochs":epoch_info,"resumed":resumed,"asset_audit":audit,"config_sha":sha256(root()/"configs/experiment.json"),"input_sha":cfg["_input_sha"],"row_id_sha":object_sha(oof[ID].astype(str).tolist()),"device":str(device),"python":platform.python_version(),"torch":torch.__version__}
 report=evaluate(oof,reference,a.output_dir,meta); atomic_json(ckpt/"report.json",report); atomic_json(ckpt/"complete.json",{"completed":True,"config_sha":meta["config_sha"],"input_sha":meta["input_sha"],"row_id_sha":meta["row_id_sha"]})
if __name__=="__main__":
 try: main()
 except (KeyboardInterrupt,TimeoutError) as e: print(f"안전 중단: {e}. 동일 명령으로 재개하십시오.",flush=True); raise SystemExit(130)
