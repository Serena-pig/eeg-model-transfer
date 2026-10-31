"""Matched whole10s/static vs five2s/dynamic graph edges with temporal nodes.
Only official source-development data; 3inner selection before outer fitting.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import copy,json,random,time,shutil
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import driver
from leaderboard_dualbranch_20261028 import groups
from select_ensemble import ASSIGNMENTS,postprocess
OUT=driver.JOB/'dynamic_connectivity_20261101';OUT.mkdir(exist_ok=True)
CACHE=driver.JOB/'local_connectivity_cache_20261101';MODES=['matched_static','dynamic'];SEED=110161;EPOCHS=20

class DynamicGraph(nn.Module):
 def __init__(self):
  super().__init__();self.node=nn.Linear(6,12);self.position=nn.Parameter(torch.empty(30,12));nn.init.normal_(self.position,std=.02);self.band=nn.Parameter(torch.zeros(7));self.message=nn.Linear(36,12);self.norm=nn.LayerNorm(12);self.node_attention=nn.Linear(12,1);self.gru=nn.GRU(36,12,batch_first=True,bidirectional=True);self.time_attention=nn.Linear(24,1);self.head=nn.Sequential(nn.Linear(72,32),nn.GELU(),nn.Dropout(.4),nn.Linear(32,1))
 def forward(self,local,corr):
  b,k,t,_,_=local.shape;x=local-local.mean((1,2),keepdim=True);sd=local.mean(2).std(1,unbiased=False).clamp_min(.2);x=(x/sd[:,None,None]).clamp(-8,8);h=F.gelu(self.node(x.transpose(-1,-2))+self.position);difference=corr-corr.mean(1,keepdim=True);weights=F.softmax(self.band,0);a=torch.einsum('bktfij,f->bktij',corr/corr.abs().sum(-1,keepdim=True).clamp_min(1.),weights);ac=torch.einsum('bktfij,f->bktij',difference/difference.abs().sum(-1,keepdim=True).clamp_min(1.),weights);h=self.norm(h+F.dropout(F.gelu(self.message(torch.cat([h,a@h,ac@h],-1))),.2,self.training));attention=F.softmax(self.node_attention(h),-2);pooled=torch.cat([h.mean(-2),h.std(-2,unbiased=False),(attention*h).sum(-2)],-1).reshape(b*k,t,36);z,_=self.gru(pooled);time_weight=F.softmax(self.time_attention(z),1);feature=torch.cat([z.mean(1),z.std(1,unbiased=False),(time_weight*z).sum(1)],-1);z=self.head(feature).reshape(b,k);return z-z.mean(1,keepdim=True)

class Data:
 def __init__(self):
  assert (CACHE/'completion.json').exists()
  with np.load(driver.CONTROL/'full_evaluation_rows.npz') as a:self.d={k:a[k] for k in a.files}
  with np.load(driver.ART/'features.npz') as a:local=a['local']
  self.mapping=self.d['model_row_to_full_row'];self.reverse=np.full(len(self.d['label']),-1,dtype=int);self.reverse[self.mapping]=np.arange(len(self.mapping));self.eligible=self.reverse>=0;self.local=np.zeros((len(self.eligible),5,6,30),dtype=np.float32);self.local[self.mapping]=local;self.static=np.load(CACHE/'static_corr.npy',mmap_mode='r');self.dynamic=np.load(CACHE/'local_corr.npy',mmap_mode='r')
 def get(self,rr,mode):
  indices=self.reverse[rr];valid=indices>=0;corr=np.zeros(rr.shape+(5,7,30,30),dtype=np.float32)
  if mode=='dynamic':corr[valid]=self.dynamic[indices[valid]]
  else:corr[valid]=np.asarray(self.static[indices[valid]],dtype=np.float32)[:,None]
  return torch.tensor(self.local[rr]),torch.tensor(corr)

def infer(model,data,rr,mode):
 model.eval();out=[]
 with torch.no_grad():
  for j in range(0,len(rr),4):out.append(model(*data.get(rr[j:j+4],mode)).numpy())
 return np.concatenate(out)

def fit(data,tr,va,mode,path):
 torch.set_num_threads(1);torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED);rng=np.random.default_rng(SEED);d=data.d;model=DynamicGraph();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.05);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=.00005);people=np.unique(d['subject'][tr]);rows=[];diag=[]
 for s in people:
  ix=tr[d['subject'][tr]==s];rows.append(np.stack([ix[d['trial'][ix]==t][np.argsort(d['crop'][ix[d['trial'][ix]==t]])] for t in range(1,9)]));diag.append(d['diagnosis'][ix[0]])
 rows=np.array(rows);diag=np.array(diag);pools=[np.flatnonzero(diag==v) for v in [0,1]];steps=int(np.ceil(len(people)*5/4));assign=torch.tensor(ASSIGNMENTS.T,dtype=torch.float32);states=[]
 for epoch in range(1,EPOCHS+1):
  model.train()
  for step in range(steps):
   who=np.array([rng.choice(pools[j%2]) for j in range(4)]);crop=rng.integers(0,5,4);rr=rows[who[:,None],np.arange(8)[None],crop[:,None]];local,corr=data.get(rr,mode);z=model(local+torch.randn_like(local)*.02,corr);y=torch.tensor(d['label'][rr],dtype=torch.float32);flat=z.flatten();yy=y.flatten();pair=F.softplus(flat[yy<.5][None]-flat[yy>.5][:,None]).mean();conditional=(torch.logsumexp(z@assign,1)-(z*y).sum(1)).mean()/4;loss=.5*F.binary_cross_entropy_with_logits(z,y*.98+.01)+.25*pair+.25*conditional+.002*z.square().mean();opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
  schedule.step()
  if epoch in [16,18,20]:states.append(copy.deepcopy(model.state_dict()))
 u,rr=groups(d,va);zz=[]
 for state in states:model.load_state_dict(state);zz.append(infer(model,data,rr,mode))
 z=np.mean(zz,0);q=postprocess(z.ravel(),np.repeat(u,8),'conditional',1.);result=np.zeros(len(va));lookup={v:j for j,v in enumerate(va)};result[[lookup[v] for v in rr.ravel()]]=q;torch.save(dict(states=states,mode=mode,source_subjects=people.tolist(),seed=SEED,epochs=20),path);saved=torch.load(path,map_location='cpu',weights_only=True);replay=[]
 for state in saved['states']:model.load_state_dict(state);replay.append(infer(model,data,rr[:,::-1],mode))
 error=float(abs(np.mean(replay,0)[:,::-1]-z).max());assert error<1e-5;return result,error

def main():
 torch.set_num_threads(1)
 while not (CACHE/'completion.json').exists():time.sleep(3)
 spec=dict(modes=['old_static']+MODES,epochs=20,seed=SEED,matched_architecture=True,matched_static_dynamic_difference='Static10s vs local2s correlation edges; same local power nodes and same classifier',source_inner_selection='max unweighted pooled AUC of source3inner; require .001 gain over old_static else fallback',primary_metric='unweighted_pooled_auc',feature_cache_completion_sha256=driver.sha(CACHE/'completion.json'),public_data_used=False,holdout_used=False,script_sha256=driver.sha(__file__))
 if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==spec
 else:driver.save(OUT/'protocol.json',spec)
 data=Data();d=data.d;completed=0;selections=[]
 for f in driver.protocol()['folds']:
  k=f['fold'];predictions={m:[] for m in ['old_static']+MODES};labels=[]
  for part in range(3):
   split=f['inner'][part];tr=np.flatnonzero(np.isin(d['subject'],split['training_subjects'])&data.eligible);va=np.flatnonzero(np.isin(d['subject'],split['validation_subjects']));assert not set(d['subject'][tr])&set(d['subject'][va]);labels.append(d['label'][va])
   with np.load(driver.JOB/f'connectivity_graph_20261101/selected_fold{k}_inner{part}.npz') as a:assert np.array_equal(a['validation'],va);predictions['old_static'].append(a['probability'])
   for mode in MODES:
    path=OUT/f'{mode}_fold{k}_inner{part}.pt';cache=path.with_suffix('.npz')
    if cache.exists():
     with np.load(cache) as a:assert str(a['model_sha256'])==driver.sha(path);q=a['probability']
    else:
     start=time.time();driver.save(OUT/'progress.json',dict(stage='training',completed_inner_fits=completed,total_inner_fits=30,fold=k,part=part,mode=mode));q,error=fit(data,tr,va,mode,path);np.savez_compressed(cache,validation=va,probability=q,reload_error=error,model_sha256=np.array(driver.sha(path)));print('LOCAL GRAPH',k,part,mode,'seconds',round(time.time()-start,1),flush=True)
    predictions[mode].append(q);completed+=1
  yy=np.concatenate(labels);scores={m:float(roc_auc_score(yy,np.concatenate(v))) for m,v in predictions.items()};chosen=max(scores,key=scores.get)
  if scores[chosen]<scores['old_static']+.001:chosen='old_static'
  selection=dict(fold=k,chosen=chosen,source_pooled_scores=scores);selections.append(selection);driver.save(OUT/f'fold{k}_selection.json',selection)
  for part in range(3):
   src=driver.JOB/f'connectivity_graph_20261101/selected_fold{k}_inner{part}.npz' if chosen=='old_static' else OUT/f'{chosen}_fold{k}_inner{part}.npz';shutil.copy2(src,OUT/f'selected_fold{k}_inner{part}.npz')
  cache=OUT/f'selected_fold{k}_outer.npz'
  if chosen=='old_static':shutil.copy2(driver.JOB/f'connectivity_graph_20261101/selected_fold{k}_outer.npz',cache)
  elif not cache.exists():
   tr=np.flatnonzero(np.isin(d['subject'],f['training_subjects'])&data.eligible);va=np.flatnonzero(np.isin(d['subject'],f['validation_subjects']));path=OUT/f'selected_fold{k}_outer.pt';q,error=fit(data,tr,va,chosen,path);np.savez_compressed(cache,validation=va,probability=q,reload_error=error,model_sha256=np.array(driver.sha(path)))
  print('LOCAL GRAPH OUTER',k,'/5 COMPLETE',chosen,flush=True)
 driver.save(OUT/'completion.json',dict(selections=selections,all5outer3inner_complete=True,public_data_used=False,holdout_used=False,official_submissions=0));print('LOCAL GRAPH COMPLETE',flush=True)
if __name__=='__main__':main()
