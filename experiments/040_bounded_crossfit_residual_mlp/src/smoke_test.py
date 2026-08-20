"""데이터 자산이나 GPU 없이 040 핵심 불변식을 검증한다."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from contract import ID, MODEL_FEATURES, NUMERIC, ONEHOT, PREDICTION_FEATURES, RAW_FEATURES
from model import ResidualMLP, final_probability, make_preprocessor, numpy_probability


def synthetic(rows: int = 96) -> pd.DataFrame:
    rng=np.random.default_rng(42); data={ID:[f"SMOKE_{i:04d}" for i in range(rows)],"target":rng.integers(0,2,rows),"fold":np.repeat([2022,2023,2024],rows//3)}
    for c in RAW_FEATURES:
        if c=="season": data[c]=data["fold"]
        elif c in ONEHOT: data[c]=rng.choice(["A","B","C"],rows)
        else: data[c]=rng.normal(size=rows)
    p=np.clip(rng.normal(.51,.03,rows),.05,.95); data["p_915"]=p; data["p_lgb"]=np.clip(p+rng.normal(0,.01,rows),.01,.99); data["p_cat"]=np.clip(p+rng.normal(0,.01,rows),.01,.99)
    base=np.column_stack([data["p_915"],data["p_lgb"],data["p_cat"]]); data["p_lgb_minus_cat"]=base[:,1]-base[:,2]; data["p_915_minus_lgb"]=base[:,0]-base[:,1]; data["p_915_minus_cat"]=base[:,0]-base[:,2]; data["prediction_mean"]=base.mean(1); data["prediction_std"]=base.std(1); data["prediction_range"]=base.max(1)-base.min(1)
    return pd.DataFrame(data)


def main() -> None:
    frame=synthetic(); train=frame.loc[frame.fold.eq(2022)]; evaluation=frame.loc[frame.fold.eq(2023)]
    assert not (set(train[ID])&set(evaluation[ID])) and frame[ID].is_unique
    pre=make_preprocessor(); x=np.asarray(pre.fit_transform(train[MODEL_FEATURES]),dtype=np.float32); xv=np.asarray(pre.transform(evaluation[MODEL_FEATURES]),dtype=np.float32)
    model=ResidualMLP(x.shape[1]); before=model(torch.from_numpy(xv)); base=torch.from_numpy(evaluation.p_915.to_numpy(np.float32)); p0=final_probability(base,before,.15)
    assert torch.equal(before,torch.zeros_like(before)); assert float(torch.max(torch.abs(p0-base)).detach())<1e-7
    exact,_=numpy_probability(evaluation.p_915.to_numpy(),np.ones(len(evaluation)),0); assert float(np.max(np.abs(exact-evaluation.p_915.to_numpy())))<1e-15
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4); target=torch.from_numpy(train.target.to_numpy(np.float32)); raw=model(torch.from_numpy(x)); loss=torch.mean((final_probability(torch.from_numpy(train.p_915.to_numpy(np.float32)),raw,.15)-target)**2); loss.backward(); optimizer.step()
    after=model(torch.from_numpy(xv)); assert not torch.equal(after,before); correction=.15*np.tanh(after.detach().numpy()); assert np.isfinite(correction).all() and np.max(np.abs(correction))<=.15
    probability,_=numpy_probability(evaluation.p_915.to_numpy(),after.detach().numpy(),.15); assert np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all()
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/"checkpoint.pt"; temporary=path.with_suffix(".pt.tmp"); torch.save(model.state_dict(),temporary); temporary.replace(path)
        resumed=ResidualMLP(x.shape[1]); resumed.load_state_dict(torch.load(path,weights_only=True)); assert torch.equal(resumed(torch.from_numpy(xv)),after)
    print("PASS: preprocessing, forward/backward, zero-init anchor, cap bound, checkpoint/resume, temporal/row/probability contracts")


if __name__=="__main__": main()
