"""Source-local permutation-equivariant context over eight trials.
Frozen matched-source graph encoder; mean context vs set self-attention.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import copy,json,time,random,shutil
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import driver
from connectivity_graph_20261101 import Data as GraphData,Graph
from leaderboard_dualbranch_20261028 import groups
from select_ensemble import ASSIGNMENTS,postprocess
OUT=driver.JOB/'trial_set_context_20261101';OUT.mkdir(exist_ok=True)
MODES=['mean','attention'];SEED=110171;EPOCHS=20

def encode(model,power,corr):
 p=power-power.mean(1,keepdim=True);p=(p/p.std(1,unbiased=False,keepdim=True).clamp_min(.2)).clamp(-8,8);h=F.gelu(model.node(p.transpose(-1,-2))+model.position);difference=corr-corr.mean(1,keepdim=True);a=corr/corr.abs().sum(-1,keepdim=True).clamp_min(1.);ac=difference/difference.abs().sum(-1,keepdim=True).clamp_min(1.);weights=F.softmax(model.band,0)
 for layer,norm in zip(model.layers,model.norms):h=norm(h+F.gelu(layer(torch.cat([h,torch.einsum('bkfij,bkjd,f->bkid',a,h,weights),torch.einsum('bkfij,bkjd,f->bkid',ac,h,weights)],-1))))
 att=F.softmax(model.attention(h),-2);pooled=torch.cat([h.mean(-2),h.std(-2,unbiased=False),(att*h).sum(-2)],-1);z=model.head(pooled).squeeze(-1);return pooled,z-z.mean(1,keepdim=True)

class Task:
 def __init__(self,data,tr,va,parent):
  self.d=data.d;checkpoint=torch.load(parent,map_location='cpu',weights_only=True);assert set(checkpoint['source_subjects'])==set(np.unique(self.d['subject'][tr]).tolist());_,rr=groups(self.d,np.r_[tr,va]);features=[];scores=[];model=Graph();model.eval()
  with torch.no_grad():
   for state in checkpoint['states']:
    model.load_state_dict(state);ff=[];zz=[]
    for start in range(0,len(rr),4):
     a,b=data.get(rr[start:start+4]);f,z=encode(model,a,b);reference=model(a,b);assert float(abs(reference-z).max())<1e-6;ff.append(f.numpy());zz.append(z.numpy())
    features.append(np.concatenate(ff));scores.append(np.concatenate(zz))
  self.x=np.full((len(self.d['label']),72),np.nan,dtype=np.float32);self.teacher=np.full(len(self.d['label']),np.nan,dtype=np.float32);self.x[rr.ravel()]=np.mean(features,0).reshape(-1,72);self.teacher[rr.ravel()]=np.mean(scores,0).ravel();self.mean=self.x[tr].mean(0);self.scale=np.maximum(self.x[tr].std(0),.1);self.x=(self.x-self.mean)/self.scale
 def get(self,rr):
  assert np.isfinite(self.x[rr]).all();assert np.isfinite(self.teacher[rr]).all()
  return torch.tensor(self.x[rr]),torch.tensor(self.teacher[rr])

class SetModel(nn.Module):
 def __init__(self,mode):
  super().__init__();self.mode=mode;self.project=nn.Sequential(nn.Linear(72,24),nn.LayerNorm(24),nn.GELU())
  if mode=='attention':self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(24,3,48,.2,activation='gelu',batch_first=True,norm_first=True),1,enable_nested_tensor=False)
  self.head=nn.Sequential(nn.Linear(72 if mode=='mean' else 96,32),nn.GELU(),nn.Dropout(.4),nn.Linear(32,1))
 def forward(self,x,teacher):
  h=self.project(x);context=torch.cat([h.mean(1),h.std(1,unbiased=False)],-1)[:,None].expand(-1,h.shape[1],-1);parts=[h,context]
  if self.mode=='attention':parts.append(self.encoder(h))
  residual=self.head(torch.cat(parts,-1)).squeeze(-1);z=.5*teacher.clamp(-4,4)+.5*residual;return z-z.mean(1,keepdim=True)

def infer(model,task,rr):
 model.eval();out=[]
 with torch.no_grad():
  for start in range(0,len(rr),4):out.append(model(*task.get(rr[start:start+4])).numpy())
 return np.concatenate(out)

def fit(task,tr,va,mode,parent,path):
 torch.set_num_threads(1);torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED);rng=np.random.default_rng(SEED);d=task.d;model=SetModel(mode);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.05);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=.00005);people=np.unique(d['subject'][tr]);rows=[];diag=[]
 for s in people:
  ix=tr[d['subject'][tr]==s];rows.append(np.stack([ix[d['trial'][ix]==t][np.argsort(d['crop'][ix[d['trial'][ix]==t]])] for t in range(1,9)]));diag.append(d['diagnosis'][ix[0]])
 rows=np.array(rows);diag=np.array(diag);pools=[np.flatnonzero(diag==v) for v in [0,1]];steps=int(np.ceil(len(people)*5/4));assign=torch.tensor(ASSIGNMENTS.T,dtype=torch.float32);states=[]
 for epoch in range(1,EPOCHS+1):
  model.train()
  for step in range(steps):
   who=np.array([rng.choice(pools[j%2]) for j in range(4)]);crop=rng.integers(0,5,4);rr=rows[who[:,None],np.arange(8)[None],crop[:,None]];permutations=np.stack([rng.permutation(8) for _ in range(4)]);rr=np.take_along_axis(rr,permutations,axis=1);z=model(*task.get(rr));y=torch.tensor(d['label'][rr],dtype=torch.float32);flat=z.flatten();yy=y.flatten();pair=F.softplus(flat[yy<.5][None]-flat[yy>.5][:,None]).mean();cond=(torch.logsumexp(z@assign,1)-(z*y).sum(1)).mean()/4;loss=.5*F.binary_cross_entropy_with_logits(z,y*.98+.01)+.25*pair+.25*cond+.002*z.square().mean();opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
  schedule.step()
  if epoch in [16,18,20]:states.append(copy.deepcopy(model.state_dict()))
 u,rr=groups(d,va);zz=[]
 for state in states:model.load_state_dict(state);zz.append(infer(model,task,rr))
 z=np.mean(zz,0);q=postprocess(z.ravel(),np.repeat(u,8),'conditional',1.);out=np.zeros(len(va));lookup={v:j for j,v in enumerate(va)};out[[lookup[v] for v in rr.ravel()]]=q;torch.save(dict(states=states,mode=mode,epochs=20,seed=SEED,feature_mean=torch.tensor(task.mean),feature_scale=torch.tensor(task.scale),source_subjects=people.tolist(),parent_sha256=driver.sha(parent)),path);saved=torch.load(path,map_location='cpu',weights_only=True);replay=[]
 for state in saved['states']:model.load_state_dict(state);replay.append(infer(model,task,rr[:,::-1]))
 error=float(abs(np.mean(replay,0)[:,::-1]-z).max());assert error<1e-5;return out,error

def main():
 torch.set_num_threads(1);spec=dict(modes=['old_static']+MODES,seed=SEED,epochs=20,parent='matched-source frozen static graph3checkpoints',feature_scaler_source_only=True,no_trial_position_encoding=True,source_trial_order_shuffled=True,inner_choice='max unweighted pooled3inner AUC; require.001gain otherwise old_static',primary_metric='unweighted_pooled_auc',public_data_used=False,holdout_used=False,script_sha256=driver.sha(__file__))
 if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==spec
 else:driver.save(OUT/'protocol.json',spec)
 data=GraphData();d=data.d;selections=[];completed=0
 for f in driver.protocol()['folds']:
  k=f['fold'];predictions={m:[] for m in ['old_static']+MODES};labels=[]
  for part in range(3):
   split=f['inner'][part];tr=np.flatnonzero(np.isin(d['subject'],split['training_subjects'])&data.eligible);va=np.flatnonzero(np.isin(d['subject'],split['validation_subjects']));assert not set(d['subject'][tr])&set(d['subject'][va]);parent=driver.JOB/f'connectivity_graph_20261101/selected_fold{k}_inner{part}.pt';task=Task(data,tr,va,parent);labels.append(d['label'][va])
   with np.load(parent.with_suffix('.npz')) as a:assert np.array_equal(a['validation'],va);predictions['old_static'].append(a['probability'])
   for mode in MODES:
    path=OUT/f'{mode}_fold{k}_inner{part}.pt';cache=path.with_suffix('.npz')
    if cache.exists():
     with np.load(cache) as a:assert str(a['model_sha256'])==driver.sha(path);q=a['probability']
    else:
     started=time.time();driver.save(OUT/'progress.json',dict(stage='training',completed_inner_fits=completed,total_inner_fits=30,fold=k,part=part,mode=mode));q,error=fit(task,tr,va,mode,parent,path);np.savez_compressed(cache,validation=va,probability=q,reload_error=error,model_sha256=np.array(driver.sha(path)));print('SET CONTEXT',k,part,mode,'seconds',round(time.time()-started,1),flush=True)
    predictions[mode].append(q);completed+=1
  yy=np.concatenate(labels);scores={m:float(roc_auc_score(yy,np.concatenate(v))) for m,v in predictions.items()};chosen=max(scores,key=scores.get)
  if scores[chosen]<scores['old_static']+.001:chosen='old_static'
  record=dict(fold=k,chosen=chosen,source_pooled_scores=scores);selections.append(record);driver.save(OUT/f'fold{k}_selection.json',record)
  for part in range(3):
   src=driver.JOB/f'connectivity_graph_20261101/selected_fold{k}_inner{part}.npz' if chosen=='old_static' else OUT/f'{chosen}_fold{k}_inner{part}.npz';shutil.copy2(src,OUT/f'selected_fold{k}_inner{part}.npz')
  cache=OUT/f'selected_fold{k}_outer.npz'
  if chosen=='old_static':shutil.copy2(driver.JOB/f'connectivity_graph_20261101/selected_fold{k}_outer.npz',cache)
  elif not cache.exists():
   tr=np.flatnonzero(np.isin(d['subject'],f['training_subjects'])&data.eligible);va=np.flatnonzero(np.isin(d['subject'],f['validation_subjects']));parent=driver.JOB/f'connectivity_graph_20261101/selected_fold{k}_outer.pt';task=Task(data,tr,va,parent);path=OUT/f'selected_fold{k}_outer.pt';q,error=fit(task,tr,va,chosen,parent,path);np.savez_compressed(cache,validation=va,probability=q,reload_error=error,model_sha256=np.array(driver.sha(path)))
  print('SET CONTEXT OUTER',k,'/5 COMPLETE',chosen,flush=True)
 driver.save(OUT/'completion.json',dict(selections=selections,all5outer3inner_complete=True,public_data_used=False,holdout_used=False,official_submissions=0));print('SET CONTEXT COMPLETE',flush=True)
if __name__=='__main__':main()
