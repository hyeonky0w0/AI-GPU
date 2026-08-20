"""cap별/그룹별 residual OOF 지표와 채택 판정을 생성한다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from contract import CAPS, ID
from model import numpy_probability


def metrics(y: np.ndarray, p: np.ndarray, base: np.ndarray, correction: np.ndarray, raw: np.ndarray) -> dict:
    improved = (p-y)**2 < (base-y)**2
    return {"rows": len(y), "brier": float(np.mean((p-y)**2)),
            "delta_vs_915": float(np.mean((p-y)**2)-np.mean((base-y)**2)),
            "mean_prediction": float(p.mean()), "actual_target_rate": float(y.mean()),
            "prediction_bias": float(p.mean()-y.mean()), "auc": float(roc_auc_score(y,p)),
            "log_loss": float(log_loss(y,np.clip(p,1e-7,1-1e-7))),
            "correction_mean": float(correction.mean()), "correction_std": float(correction.std()),
            "correction_min": float(correction.min()), "correction_max": float(correction.max()),
            "saturation_fraction": float(np.mean(np.abs(np.tanh(raw)) > .99)),
            "probability_min": float(p.min()), "probability_max": float(p.max()),
            "correlation_with_915": float(np.corrcoef(p,base)[0,1]),
            "improved_row_fraction": float(improved.mean()), "worsened_row_fraction": float((~improved & (p != base)).mean())}


def evaluate(frame: pd.DataFrame, selected_caps: dict[int, float], output_dir: Path) -> dict:
    records, predictions = [], []
    for year in [2023, 2024]:
        part = frame.loc[frame.fold.eq(year)].copy()
        y, base, raw = part.target.to_numpy(), part.p_915.to_numpy(), part.raw_correction.to_numpy()
        for cap in CAPS:
            p, correction = numpy_probability(base, raw, cap)
            rec = {"fold": year, "cap": cap, "selection": "diagnostic", **metrics(y,p,base,correction,raw)}
            records.append(rec)
        cap = float(selected_caps[year]); p, correction = numpy_probability(base, raw, cap)
        records.append({"fold": year, "cap": cap, "selection": "deployable", **metrics(y,p,base,correction,raw)})
        part["selected_cap"] = cap; part["correction"] = correction; part["p_final"] = p
        predictions.append(part)
    output = pd.concat(predictions, ignore_index=True)
    cap_metrics = pd.DataFrame(records)
    overall = []
    for label, cap in [("baseline_915",0.0),("deployable",None), *[(f"cap_{c:.2f}",c) for c in CAPS]]:
        values=[]
        for year in [2023,2024]:
            part=frame.loc[frame.fold.eq(year)]; chosen=selected_caps[year] if cap is None else cap
            p,corr=numpy_probability(part.p_915.to_numpy(),part.raw_correction.to_numpy(),chosen); values.append((part.target.to_numpy(),p,part.p_915.to_numpy(),corr))
        y=np.concatenate([x[0] for x in values]); p=np.concatenate([x[1] for x in values]); base=np.concatenate([x[2] for x in values]); corr=np.concatenate([x[3] for x in values])
        raw=np.concatenate([frame.loc[frame.fold.eq(year),"raw_correction"].to_numpy() for year in [2023,2024]])
        overall.append({"candidate":label,**metrics(y,p,base,corr,raw)})
    overall_df=pd.DataFrame(overall)
    group_rows=[]
    for year, part in output.groupby("fold"):
        history=pd.cut(part.asof_pitcher_n,[-np.inf,0,50,200,np.inf],labels=["new","1-50","51-200","200+"])
        for dimension, values in [("pitcher_status",np.where(part.asof_pitcher_n.eq(0),"new","existing")),("pitcher_history",history.astype(str)),("game_type",part.game_type.astype(str))]:
            for value, idx in pd.Series(values,index=part.index).groupby(values).groups.items():
                x=part.loc[idx]; group_rows.append({"fold":year,"dimension":dimension,"value":value,"rows":len(x),"brier_915":float(np.mean((x.p_915-x.target)**2)),"brier_final":float(np.mean((x.p_final-x.target)**2))})
    output_dir.mkdir(parents=True,exist_ok=True)
    cap_metrics.to_csv(output_dir/"metrics_by_fold.csv",index=False); overall_df.to_csv(output_dir/"metrics_overall.csv",index=False)
    cap_metrics.to_csv(output_dir/"cap_analysis.csv",index=False); pd.DataFrame(group_rows).to_csv(output_dir/"correction_analysis.csv",index=False)
    output[[ID,"fold","target","p_915","raw_correction","selected_cap","correction","p_final"]].to_csv(output_dir/"residual_oof_predictions.csv.gz",index=False,compression="gzip")
    base_zero=np.max(np.abs(output.p_915.to_numpy()-numpy_probability(output.p_915.to_numpy(),output.raw_correction.to_numpy(),0)[0]))
    dep=overall_df.loc[overall_df.candidate.eq("deployable")].iloc[0]
    fold_dep=cap_metrics.loc[cap_metrics.selection.eq("deployable")]
    extend=bool(base_zero<=1e-12 and (fold_dep.delta_vs_915<=0).all() and dep.delta_vs_915<=-0.0002 and all(c>0 for c in selected_caps.values()) and (fold_dep.saturation_fraction<.05).all())
    report={"status":"completed","baseline_cap0_max_error":float(base_zero),"selected_caps":{str(k):v for k,v in selected_caps.items()},"seed42_extend_three_seed":extend,"decision":"CONDITIONAL" if extend else "REJECT"}
    (output_dir/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report
