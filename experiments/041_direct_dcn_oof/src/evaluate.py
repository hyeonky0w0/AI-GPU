"""Direct DCN 단독·다양성·고정 blend 평가와 필수 산출물 생성."""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from contract import ID,FOLD,WEIGHTS,atomic_json

def metric(y,p):
 y=np.asarray(y); p=np.asarray(p,float)
 return {"brier":float(np.mean((p-y)**2)),"auc":float(roc_auc_score(y,p)) if np.unique(y).size>1 else float("nan"),"log_loss":float(log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1])),"mean_prediction":float(p.mean()),"target_rate":float(y.mean()),"prediction_bias":float(p.mean()-y.mean()),"prediction_std":float(p.std()),"probability_min":float(p.min()),"probability_max":float(p.max())}
def atomic_csv(frame,path,gzip=False):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); frame.to_csv(tmp,index=False,compression="gzip" if gzip else None); os.replace(tmp,path)
def _best_weight(g): return min(WEIGHTS,key=lambda w:np.mean((((1-w)*g.p_915+w*g.p_dcn)-g.target)**2))
def evaluate(oof,reference,out,meta):
 out=Path(out); out.mkdir(parents=True,exist_ok=True); frame=reference.merge(oof,on=[ID,FOLD,"target"],how="inner",validate="one_to_one")
 if len(frame)!=len(reference) or not np.isfinite(frame.p_dcn).all() or not frame.p_dcn.between(0,1).all(): raise ValueError("평가 OOF 계약 실패")
 models={"catboost_seed777":"p_cat","lightgbm_k10":"p_lgb","real_mlp":"p_mlp","champion_915":"p_915","direct_dcn":"p_dcn"}; rows=[]
 for scope,g in [(str(y),frame[frame.fold.eq(y)]) for y in sorted(frame.fold.unique())]+[("overall",frame)]:
  for name,col in models.items(): rows.append({"scope":scope,"model":name,**metric(g.target,g[col])})
 metrics=pd.DataFrame(rows); atomic_csv(metrics[metrics.scope.ne("overall")],out/"metrics_by_fold.csv"); atomic_csv(metrics[metrics.scope.eq("overall")],out/"metrics_overall.csv")
 cal=[]
 for name,col in models.items():
  for y,g0 in frame.groupby(FOLD):
   g=g0.copy(); g["decile"]=pd.qcut(g[col].rank(method="first"),10,labels=False,duplicates="drop")
   for d,x in g.groupby("decile"): cal.append({"fold":y,"model":name,"decile":int(d),"rows":len(x),"mean_prediction":x[col].mean(),"target_rate":x.target.mean(),"brier":np.mean((x[col]-x.target)**2)})
 atomic_csv(pd.DataFrame(cal),out/"calibration_by_decile.csv")
 blend=[]; deploy={2022:0.0}
 for y in sorted(frame.fold.unique()):
  history=frame[frame.fold.lt(y)]
  if y==2023: history=frame[frame.fold.eq(2022)]
  if y==2024: history=frame[frame.fold.isin([2022,2023])]
  deploy[int(y)]=float(_best_weight(history)) if len(history) else 0.0
  g=frame[frame.fold.eq(y)]; oracle=float(_best_weight(g))
  for w in WEIGHTS:
   p=(1-w)*g.p_915+w*g.p_dcn; blend.append({"fold":int(y),"weight":w,"brier":np.mean((p-g.target)**2),"delta_vs_915":np.mean((p-g.target)**2)-np.mean((g.p_915-g.target)**2),"deployable_selected":w==deploy[int(y)],"diagnostic_oracle":w==oracle})
 atomic_csv(pd.DataFrame(blend),out/"blend_analysis.csv")
 diversity=[]
 for scope,g in [(str(y),frame[frame.fold.eq(y)]) for y in sorted(frame.fold.unique())]+[("overall",frame)]:
  e1=g.p_915-g.target; e2=g.p_dcn-g.target; se1=e1**2; se2=e2**2
  diversity.append({"scope":scope,"segment":"all","value":"all","rows":len(g),"prediction_correlation":g.p_915.corr(g.p_dcn),"residual_correlation":e1.corr(e2),"squared_error_correlation":se1.corr(se2),"dcn_right_915_wrong_fraction":float((se2<se1).mean()),"915_right_dcn_wrong_fraction":float((se1<se2).mean()),"dcn_brier":se2.mean(),"brier_915":se1.mean()})
 for segment,series in [("game_type",frame.game_type.astype(str)),("pitcher_n_bucket",pd.cut(frame.asof_pitcher_n,[-np.inf,0,10,50,200,np.inf]).astype(str)),("pitcher_status",np.where(frame.asof_pitcher_n.fillna(0)<=0,"new","existing"))]:
  for value,g in frame.assign(_segment=series).groupby("_segment"):
   e1=g.p_915-g.target; e2=g.p_dcn-g.target; diversity.append({"scope":"overall","segment":segment,"value":value,"rows":len(g),"prediction_correlation":g.p_915.corr(g.p_dcn),"residual_correlation":e1.corr(e2),"squared_error_correlation":(e1**2).corr(e2**2),"dcn_right_915_wrong_fraction":float(((e2**2)<(e1**2)).mean()),"915_right_dcn_wrong_fraction":float(((e1**2)<(e2**2)).mean()),"dcn_brier":np.mean(e2**2),"brier_915":np.mean(e1**2)})
 disagreement=pd.qcut((frame.p_dcn-frame.p_915).abs().rank(method="first"),10,labels=False)
 for d,g in frame.assign(_segment=disagreement).groupby("_segment"): diversity.append({"scope":"overall","segment":"disagreement_decile","value":int(d),"rows":len(g),"dcn_brier":np.mean((g.p_dcn-g.target)**2),"brier_915":np.mean((g.p_915-g.target)**2)})
 atomic_csv(pd.DataFrame(diversity),out/"error_diversity.csv")
 deployed=frame.copy(); deployed["weight"]=deployed.fold.map(deploy); deployed["p_blend"]=(1-deployed.weight)*deployed.p_915+deployed.weight*deployed.p_dcn; outer=deployed[deployed.fold.isin([2023,2024])]
 deltas={str(y):float(np.mean((g.p_blend-g.target)**2)-np.mean((g.p_915-g.target)**2)) for y,g in deployed.groupby(FOLD)}; overall_delta=float(np.mean((outer.p_blend-outer.target)**2)-np.mean((outer.p_915-outer.target)**2))
 extend=bool(deltas.get("2023",1)<=0 and deltas.get("2024",1)<=0 and overall_delta<=-0.0002 and any(deploy[y]>0 for y in [2023,2024]) and abs(frame.p_dcn.corr(frame.p_915))<.999999)
 report={"status":"completed","decision":"CONDITIONAL" if extend else "REJECT","seed42_extend":extend,"deployable_weights":deploy,"deployable_fold_deltas":deltas,"deployable_outer_delta":overall_delta,"diagnostic_only_oracle_weights":{int(y):float(_best_weight(g)) for y,g in frame.groupby(FOLD)},"metadata":meta}
 atomic_csv(oof[[ID,FOLD,"target","p_dcn"]],out/"direct_dcn_oof_predictions.csv.gz",True); atomic_json(out/"report.json",report)
 result=("# RESULT\n\n## 관찰 사실\n\n- 판정: **%s**\n- deployable outer Brier delta: `%.9f`\n- deployable weight: `%s`\n\n## 해석/가설\n\n- 같은 연도 target으로 고른 oracle weight는 진단 전용이며 판정에 사용하지 않았다.\n\n## 다음 판별 실험\n\n- seed42 확장 조건: `%s`\n"%(report["decision"],overall_delta,deploy,extend)); (out/"RESULT.md").write_text(result,encoding="utf-8")
 return report
