"""37개 raw feature representation과 Direct DCN."""
from __future__ import annotations
import numpy as np, torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler
from torch import nn
from contract import NUMERIC, ONEHOT

def make_preprocessor():
 return ColumnTransformer([
  ("qt",Pipeline([("imp",SimpleImputer(strategy="median")),("qt",QuantileTransformer(output_distribution="normal",n_quantiles=1000,subsample=200_000,random_state=0))]),NUMERIC),
  ("na",MissingIndicator(features="missing-only"),NUMERIC),
  ("season",StandardScaler(),["season"]),
  ("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),ONEHOT)],remainder="drop",verbose_feature_names_out=False)

class CrossLayer(nn.Module):
 def __init__(self,d): super().__init__(); self.weight=nn.Parameter(torch.empty(d)); self.bias=nn.Parameter(torch.zeros(d)); nn.init.normal_(self.weight,std=.01)
 def forward(self,x0,x): return x0*(x@self.weight).unsqueeze(1)+self.bias+x

class DirectDCN(nn.Module):
 def __init__(self,d,target_rate):
  super().__init__(); self.cross1=CrossLayer(d); self.cross2=CrossLayer(d); self.deep=nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU()); self.output=nn.Linear(d+128,1)
  rate=float(np.clip(target_rate,1e-6,1-1e-6)); nn.init.zeros_(self.output.weight); nn.init.constant_(self.output.bias,float(np.log(rate/(1-rate))))
 def forward(self,x):
  cross=self.cross2(x,self.cross1(x,x)); return self.output(torch.cat([cross,self.deep(x)],1)).squeeze(-1)
 def probability(self,x): return torch.sigmoid(self(x))
