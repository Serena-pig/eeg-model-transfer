"""Multiplex signed-correlation graph + band-power learner, source-only.
No anatomy assumptions: fixed channel ordering, learned index embeddings.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import copy,random,time,json
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
import driver
from select_ensemble import ASSIGNMENTS,postprocess
from leaderboard_dualbranch_20261028 import groups
OUT=driver.JOB/'connectivity_graph_20261101';OUT.mkdir(exist_ok=True)
SEED=110131;EPOCHS=20

class Graph(nn.Module):
 def __init__(self):
  super().__init__();self.node=nn.Linear(8,24);self.position=nn.Parameter(torch.zeros(30,24));nn.init.normal_(self.position,std=.02);self.band=nn.Parameter(torch.zeros(7));self.layers=nn.ModuleList([nn.Linear(72,24),nn.Linear(72,24)]);self.norms=nn.ModuleList([nn.LayerNorm(24),nn.LayerNorm(24)]);self.attention=nn.Linear(24,1);self.head=nn.Sequential(nn.Linear(72,24),nn.GELU(),nn.Dropout(.4),nn.Linear(24,1))
 def forward(self,power,corr):
  b,k,_,_=power.shape;p=power-power.mean(1,keepdim=True);p=(p/p.std(1,unbiased=False,keepdim=True).clamp_min(.2)).clamp(-8,8);h=F.gelu(self.node(p.transpose(-1,-2))+self.position);centered=corr-corr.mean(1,keepdim=True);a=corr/corr.abs().sum(-1,keepdim=True).clamp_min(1.);ac=centered/centered.abs().sum(-1,keepdim=True).clamp_min(1.);weight=F.softmax(self.band,0)
  for layer,norm in zip(self.layers,self.norms):
   agg=torch.einsum('bkfij,bkjd,f->bkid',a,h,weight);diff=torch.einsum('bkfij,bkjd,f->bkid',ac,h,weight);h=norm(h+F.dropout(F.gelu(layer(torch.cat([h,agg,diff],-1))),.2,self.training))
  att=F.softmax(self.attention(h),-2);pooled=torch.cat([h.mean(-2),h.std(-2,unbiased=False),(att*h).sum(-2)],-1);z=self.head(pooled).squeeze(-1);return z-z.mean(1,keepdim=True)

class Data:
 def __init__(self):
  with np.load(driver.CONTROL/'full_evaluation_rows.npz') as a:self.d={k:a[k] for k in a.files}
  with np.load(driver.ART/'features.npz') as a:power=a['powers'];cov=a['covs']
  self.mapping=self.d['model_row_to_full_row'];self.eligible=np.zeros(len(self.d['label']),dtype=bool);self.eligible[self.mapping]=True;self.power=np.zeros((len(self.eligible),8,30),dtype=np.float32);self.power[self.mapping]=power;scale=np.sqrt(np.maximum(np.diagonal(cov,axis1=-2,axis2=-1),1e-12));c=cov/scale[:,:,:,None]/scale[:,:,None,:];c=np.clip(c,-1,1);self.corr=np.zeros((len(self.eligible),7,30,30),dtype=np.float32);self.corr[self.mapping]=c;assert np.isfinite(self.corr).all()
 def get(self,rr):return torch.tensor(self.power[rr]),torch.tensor(self.corr[rr])

def infer(model,data,rr):
 model.eval();out=[]
 with torch.no_grad():
  for start in range(0,len(rr),4):out.append(model(*data.get(rr[start:start+4])).numpy())
 return np.concatenate(out)

def fit(data,tr,va,path):
 torch.set_num_threads(1);torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED);rng=np.random.default_rng(SEED);d=data.d;model=Graph();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.05);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=.00005);people=np.unique(d['subject'][tr]);rows=[];diag=[]
 for s in people:
  ix=tr[d['subject'][tr]==s];rows.append(np.stack([ix[d['trial'][ix]==t][np.argsort(d['crop'][ix[d['trial'][ix]==t]])] for t in range(1,9)]));diag.append(d['diagnosis'][ix[0]])
 rows=np.array(rows);diag=np.array(diag);pools=[np.flatnonzero(diag==v) for v in [0,1]];steps=int(np.ceil(len(people)*5/4));assign=torch.tensor(ASSIGNMENTS.T,dtype=torch.float32);states=[]
 for epoch in range(1,EPOCHS+1):
  model.train()
  for step in range(steps):
   who=np.array([rng.choice(pools[j%2]) for j in range(4)]);crop=rng.integers(0,5,(4,8));rr=rows[who[:,None],np.arange(8)[None],crop];power,corr=data.get(rr);power=power+torch.randn_like(power)*.02;z=model(power,corr);y=torch.tensor(d['label'][rr],dtype=torch.float32);flat=z.flatten();yy=y.flatten();pair=F.softplus(flat[yy<.5][None]-flat[yy>.5][:,None]).mean();conditional=(torch.logsumexp(z@assign,1)-(z*y).sum(1)).mean()/4;loss=.5*F.binary_cross_entropy_with_logits(z,y*.98+.01)+.25*pair+.25*conditional+.002*z.square().mean();opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
  schedule.step()
  if epoch in [16,18,20]:states.append(copy.deepcopy(model.state_dict()))
 u,rr=groups(d,va);zz=[]
 for state in states:model.load_state_dict(state);zz.append(infer(model,data,rr))
 z=np.mean(zz,0);q=postprocess(z.ravel(),np.repeat(u,8),'conditional',1.);out=np.zeros(len(va));lookup={v:j for j,v in enumerate(va)};out[[lookup[v] for v in rr.ravel()]]=q;torch.save(dict(states=states,source_subjects=people.tolist(),epochs=20,seed=SEED),path);saved=torch.load(path,map_location='cpu',weights_only=True);replay=[]
 for state in saved['states']:model.load_state_dict(state);replay.append(infer(model,data,rr[:,::-1]))
 error=float(abs(np.mean(replay,0)[:,::-1]-z).max());assert error<1e-5;return out,error

def main():
 torch.set_num_threads(1);spec=dict(architecture='Two-layer7band signed correlation message passing +8band DE nodes + learned channel embeddings',seed=SEED,epochs=20,train_sampling='balanced source diagnosis, fixed4patient batch',primary_metric='unweighted pooled AUC',all5outer3inner=True,no_hyperparameter_selection_on_outer=True,no_public_or_holdout_used=True,script_sha256=driver.sha(__file__));p=OUT/'protocol.json'
 if p.exists():assert json.loads(p.read_text())==spec
 else:driver.save(p,spec)
 data=Data();d=data.d;count=0
 for f in driver.protocol()['folds']:
  for part in ['inner0','inner1','inner2','outer']:
   split=f if part=='outer' else f['inner'][int(part[-1])];tr=np.flatnonzero(np.isin(d['subject'],split['training_subjects'])&data.eligible);va=np.flatnonzero(np.isin(d['subject'],split['validation_subjects']));assert not set(d['subject'][tr])&set(d['subject'][va]);path=OUT/f'selected_fold{f["fold"]}_{part}.pt';cache=path.with_suffix('.npz')
   if cache.exists():
    with np.load(cache) as a:assert np.array_equal(a['validation'],va);assert str(a['model_sha256'])==driver.sha(path)
   else:
    started=time.time();driver.save(OUT/'progress.json',dict(stage='training',completed=count,total=20,fold=f['fold'],part=part));q,error=fit(data,tr,va,path);np.savez_compressed(cache,validation=va,probability=q,reload_error=error,model_sha256=np.array(driver.sha(path)));print('CONNECTIVITY GRAPH',f['fold'],part,'seconds',round(time.time()-started,1),flush=True)
   count+=1
  driver.save(OUT/f'fold{f["fold"]}_selection.json',dict(chosen='fixed_graph',no_outer_selection=True))
 driver.save(OUT/'completion.json',dict(all20fits_complete=True,public_data_used=False,holdout_used=False,official_submissions=0));print('GRAPH COMPLETE',flush=True)
if __name__=='__main__':main()
