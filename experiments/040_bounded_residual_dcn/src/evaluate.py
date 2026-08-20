"""Brier 중심 지표와 paired bootstrap 진단."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

def metrics(y,p):
    y=np.asarray(y); p=np.asarray(p)
    return {"brier":float(np.mean((p-y)**2)),"mean_prediction":float(p.mean()),"target_rate":float(y.mean()),
            "prediction_bias":float(p.mean()-y.mean()),"auc":float(roc_auc_score(y,p)),
            "log_loss":float(log_loss(y,np.clip(p,1e-7,1-1e-7))),"p_min":float(p.min()),"p_max":float(p.max())}

def bootstrap_error_delta(y,base,p,seed=42,repeats=1000):
    delta=(p-y)**2-(base-y)**2; rng=np.random.default_rng(seed); n=len(y); means=[]
    # 메모리를 제한하기 위해 각 반복에서 인덱스 한 벡터만 만든다.
    for _ in range(repeats): means.append(float(delta[rng.integers(0,n,n)].mean()))
    return [float(x) for x in np.quantile(means,[.025,.975])]

def residual_stats(raw):
    r=np.tanh(np.asarray(raw,float))
    return {"residual_mean":float(r.mean()),"residual_std":float(r.std()),"residual_min":float(r.min()),
            "residual_max":float(r.max()),"residual_saturation_fraction":float((np.abs(r)>=.99).mean())}
