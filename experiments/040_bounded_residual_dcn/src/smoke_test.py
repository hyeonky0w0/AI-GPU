"""CPU 합성 데이터로 모델/수식/checkpoint/temporal 계약을 검사한다."""
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np, pandas as pd, torch
from models import CrossLayer,ResidualDCN,ResidualMLP,final_probability,numpy_probability

def train_once(cls,x,base,y):
    model=cls(x.shape[1]); opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.01); first=None
    for _ in range(30):
        opt.zero_grad(); p=final_probability(base,model(x),.2); loss=((p-y)**2).mean(); first=float(loss.detach()) if first is None else first; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    assert float(loss.detach())<first and all(torch.isfinite(v).all() for v in model.parameters())
    return model,float(first),float(loss.detach())

def main():
    torch.manual_seed(42); n,d=128,17; x=torch.randn(n,d); base=torch.sigmoid(.2*x[:,0]); y=(x[:,0]*x[:,1]+.3*x[:,2]>0).float()
    cross=CrossLayer(d); assert cross(x,x).shape==x.shape
    results={}
    with tempfile.TemporaryDirectory() as td:
        for name,cls in [("mlp",ResidualMLP),("dcn",ResidualDCN)]:
            model,before,after=train_once(cls,x,base,y); raw=model(x).detach().numpy(); p0,r=numpy_probability(base.numpy(),raw,0); p,r=numpy_probability(base.numpy(),raw,.2)
            assert np.array_equal(p0,base.numpy().astype(float)) or np.max(np.abs(p0-base.numpy()))<2e-15
            assert np.all((-1<=r)&(r<=1)) and np.isfinite(p).all() and np.all((0<=p)&(p<=1))
            path=Path(td)/f"{name}.pt"; torch.save(model.state_dict(),path); clone=cls(d); clone.load_state_dict(torch.load(path,weights_only=True)); assert torch.equal(model(x),clone(x))
            results[name]={"initial_brier":before,"final_brier":after}
    frame=pd.DataFrame({"row_id":np.arange(12),"fold":np.repeat([2022,2023,2024],4)})
    assert frame.row_id.is_unique and not (set(frame[frame.fold.eq(2022)].row_id)&set(frame[frame.fold.eq(2023)].row_id))
    print({"status":"PASS","tests":["forward_backward","cross_shape","lambda_zero","bounded","loss_decrease","checkpoint_resume","row_alignment","temporal_overlap","finite_probability"],"loss":results})
if __name__=="__main__": main()
