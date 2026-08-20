"""동일 representation을 쓰는 Residual MLP와 DCN."""
from __future__ import annotations
import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler
from torch import nn
from contract import EPS, NUMERIC, ONEHOT, PREDICTION_FEATURES

def make_preprocessor():
    return ColumnTransformer([
        ("qt",Pipeline([("imp",SimpleImputer(strategy="median")),("qt",QuantileTransformer(output_distribution="normal",n_quantiles=1000,subsample=200_000,random_state=0))]),NUMERIC),
        ("na",MissingIndicator(features="missing-only"),NUMERIC), ("season",StandardScaler(),["season"]),
        ("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),ONEHOT),
        ("prediction",StandardScaler(),PREDICTION_FEATURES)],verbose_feature_names_out=False)

class ResidualMLP(nn.Module):
    def __init__(self,d:int):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,1)); nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,x): return self.net(x).squeeze(-1)

class CrossLayer(nn.Module):
    def __init__(self,d:int): super().__init__(); self.weight=nn.Parameter(torch.empty(d)); self.bias=nn.Parameter(torch.zeros(d)); nn.init.normal_(self.weight,std=0.01)
    def forward(self,x0,x): return x0*(x@self.weight).unsqueeze(1)+self.bias+x

class ResidualDCN(nn.Module):
    def __init__(self,d:int):
        super().__init__(); self.cross1=CrossLayer(d); self.cross2=CrossLayer(d); self.deep=nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU()); self.output=nn.Linear(d+128,1); nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
    def forward(self,x):
        cross=self.cross2(x,self.cross1(x,x)); return self.output(torch.cat([cross,self.deep(x)],1)).squeeze(-1)

def final_probability(base,raw,lam): return torch.sigmoid(torch.logit(base.detach().clamp(EPS,1-EPS))+float(lam)*torch.tanh(raw))
def numpy_probability(base,raw,lam):
    b=np.clip(np.asarray(base,float),EPS,1-EPS); residual=np.tanh(np.asarray(raw,float)); p=1/(1+np.exp(-(np.log(b/(1-b))+float(lam)*residual))); return p,residual
