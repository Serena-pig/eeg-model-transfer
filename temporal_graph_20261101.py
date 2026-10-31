"""Source-local temporal extension of the frozen static correlation graph.
Five2s power views are inside each10s input; no target50s context is used.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import copy,json,random,time,shutil
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import driver
from connectivity_graph_20261101 import Data as StaticData,Graph,infer as static_infer
from leaderboard_dualbranch_20261028 import groups
from select_ensemble import ASSIGNMENTS,postprocess
OUT=driver.JOB/'temporal_graph_20261101';OUT.mkdir(exist_ok=True)
MODES=['summary','bigru'];SEED=110141;EPOCHS=20

class TemporalGraph(nn.Module):
 def __init__(self,mode):
  super().__init__();self.mode=mode;self.node=nn.Linear(6,12);self.position=nn.Parameter(torch.empty(30,12));nn.init.normal_(self.position,std=.02);self.message=nn.Linear(36,12);self.norm=nn.LayerNorm(12);self.node_attention=nn.Linear(12,1)
  if mode=='bigru':self.gru=nn.GRU(36,12,batch_first=True,bidirectional=True);self.time_attention=nn.Linear(24,1)
  self.head=nn.Sequential(nn.Linear(108 if mode=='summary' else 72,32),nn.GELU(),nn.Dropout(.4),nn.Linear(32,1))
 def forward(self,local,a,ac,teacher):
  b,k,t,band,ch=local.shape;x=local-local.mean((1,2),keepdim=True);scale=local.mean(2).std(1,unbiased=False).clamp_min(.2);x=(x/scale[:,None,None]).clamp(-8,8);h=F.gelu(self.node(x.transpose(-1,-2))+self.position);agg=a[:,:,None]@h;diff=ac[:,:,None]@h;h=self.norm(h+F.dropout(F.gelu(self.message(torch.cat([h,agg,diff],-1))),.2,self.training));attention=F.softmax(self.node_attention(h),-2);pooled=torch.cat([h.mean(-2),h.std(-2,unbiased=False),(attention*h).sum(-2)],-1).reshape(b*k,t,36)
  if self.mode=='summary':features=torch.cat([pooled.mean(1),pooled.std(1,unbiased=False),pooled[:,-1]-pooled[:,0]],-1)
  else:
   z,_=self.gru(pooled);weights=F.softmax(self.time_attention(z),1);features=torch.cat([z.mean(1),z.std(1,unbiased=False),(weights*z).sum(1)],-1)
  residual=self.head(features).reshape(b,k);z=.5*teacher.clamp(-4,4)+.5*residual;return z-z.mean(1,keepdim=True)

class TaskData:
 def __init__(self,data,tr,va,parent):
  self.d=data.d
  with np.load(driver.ART/'features.npz') as source:local=source['local']
  self.local=np.zeros((len(self.d['label']),5,6,30),dtype=np.float32);self.local[data.mapping]=local
  saved=torch.load(parent,map_location='cpu',weights_only=True);assert set(saved['source_subjects'])==set(np.unique(self.d['subject'][tr]).tolist());model=Graph();_,rr=groups(self.d,np.r_[tr,va]);logits=[]
  for state in saved['states']:model.load_state_dict(state);logits.append(static_infer(model,data,rr))
  self.teacher=np.full(len(self.d['label']),np.nan,dtype=np.float32);self.teacher[rr.ravel()]=np.mean(logits,0).ravel();weights=F.softmax(torch.stack([state['band'] for state in saved['states']]).mean(0),0).numpy();self.a=np.zeros((len(self.d['label']),30,30),dtype=np.float32);self.ac=np.zeros_like(self.a)
  for rows in rr:
   c=data.corr[rows];delta=c-c.mean(0,keepdims=True);normal=c/np.maximum(abs(c).sum(-1,keepdims=True),1.);centered=delta/np.maximum(abs(delta).sum(-1,keepdims=True),1.);self.a[rows]=np.einsum('gfij,f->gij',normal,weights);self.ac[rows]=np.einsum('gfij,f->gij',centered,weights)
 def get(self,rr):
  assert np.isfinite(self.teacher[rr]).all()
  return tuple(torch.tensor(x[rr]) for x in [self.local,self.a,self.ac,self.teacher])

def infer(model,data,rr):
 model.eval();out=[]
 with torch.no_grad():
  for j in range(0,len(rr),4):out.append(model(*data.get(rr[j:j+4])).numpy())
 return np.concatenate(out)

def fit(data,tr,va,mode,path,parent):
 torch.set_num_threads(1);torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED);rng=np.random.default_rng(SEED);d=data.d;model=TemporalGraph(mode);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.05);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=.00005);people=np.unique(d['subject'][tr]);rows=[];diag=[]
 for s in people:
  ix=tr[d['subject'][tr]==s];rows.append(np.stack([ix[d['trial'][ix]==t][np.argsort(d['crop'][ix[d['trial'][ix]==t]])] for t in range(1,9)]));diag.append(d['diagnosis'][ix[0]])
 rows=np.array(rows);diag=np.array(diag);pools=[np.flatnonzero(diag==v) for v in [0,1]];steps=int(np.ceil(len(people)*5/4));assign=torch.tensor(ASSIGNMENTS.T,dtype=torch.float32);states=[]
 for epoch in range(1,EPOCHS+1):
  model.train()
  for step in range(steps):
   who=np.array([rng.choice(pools[j%2]) for j in range(4)]);crop=rng.integers(0,5,4);rr=rows[who[:,None],np.arange(8)[None],crop[:,None]];local,a,ac,teacher=data.get(rr);local=local+torch.randn_like(local)*.02;z=model(local,a,ac,teacher);y=torch.tensor(d['label'][rr],dtype=torch.float32);flat=z.flatten();yy=y.flatten();pair=F.softplus(flat[yy<.5][None]-flat[yy>.5][:,None]).mean();cond=(torch.logsumexp(z@assign,1)-(z*y).sum(1)).mean()/4;loss=.5*F.binary_cross_entropy_with_logits(z,y*.98+.01)+.25*pair+.25*cond+.002*z.square().mean();opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
  sched.step()
  if epoch in [16,18,20]:states.append(copy.deepcopy(model.state_dict()))
 u,rr=groups(d,va);zz=[]
 for state in states:model.load_state_dict(state);zz.append(infer(model,data,rr))
 z=np.mean(zz,0);q=postprocess(z.ravel(),np.repeat(u,8),'conditional',1.);out=np.zeros(len(va));lookup={v:j for j,v in enumerate(va)};out[[lookup[v] for v in rr.ravel()]]=q;torch.save(dict(states=states,mode=mode,epochs=20,seed=SEED,source_subjects=people.tolist(),parent_graph_sha256=driver.sha(parent)),path);saved=torch.load(path,map_location='cpu',weights_only=True);replay=[]
 for state in saved['states']:model.load_state_dict(state);replay.append(infer(model,data,rr[:,::-1]))
 error=float(abs(np.mean(replay,0)[:,::-1]-z).max());assert error<1e-5;return out,error

def main():
 torch.set_num_threads(1);spec=dict(modes=['static']+MODES,epochs=20,seed=SEED,teacher='matched-source frozen static correlation graph',new_information='5two-second local power views inside each10second record',teacher_logit_factor=.5,temporal_factor=.5,selection='max pooled source3inner AUC; temporal requires .001 absolute gain, else static',primary_metric='unweighted_pooled_auc',source_sampling='2HC+2DEP, same crop for all8trials of a patient',target_input_seconds=10,no_public_or_holdout_used=True,script_sha256=driver.sha(__file__))
 if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==spec
 else:driver.save(OUT/'protocol.json',spec)
 source=StaticData();d=source.d;completed=0;selections=[]
 for f in driver.protocol()['folds']:
  k=f['fold'];predictions={m:[] for m in ['static']+MODES};labels=[]
  for part in range(3):
   split=f['inner'][part];tr=np.flatnonzero(np.isin(d['subject'],split['training_subjects'])&source.eligible);va=np.flatnonzero(np.isin(d['subject'],split['validation_subjects']));assert not set(d['subject'][tr])&set(d['subject'][va]);parent=driver.JOB/f'connectivity_graph_20261101/selected_fold{k}_inner{part}.pt';task=TaskData(source,tr,va,parent);labels.append(d['label'][va])
   with np.load(parent.with_suffix('.npz')) as a:assert np.array_equal(a['validation'],va);predictions['static'].append(a['probability'])
   for mode in MODES:
    path=OUT/f'{mode}_fold{k}_inner{part}.pt';cache=path.with_suffix('.npz')
    if cache.exists():
     with np.load(cache) as a:assert str(a['model_sha256'])==driver.sha(path);q=a['probability']
    else:
     started=time.time();driver.save(OUT/'progress.json',dict(stage='training',completed_inner_fits=completed,total_inner_fits=30,fold=k,part=part,mode=mode));q,error=fit(task,tr,va,mode,path,parent);np.savez_compressed(cache,validation=va,probability=q,reload_error=error,model_sha256=np.array(driver.sha(path)));print('TEMPORAL GRAPH',k,part,mode,'seconds',round(time.time()-started,1),flush=True)
    predictions[mode].append(q);completed+=1
  yy=np.concatenate(labels);scores={m:float(roc_auc_score(yy,np.concatenate(v))) for m,v in predictions.items()};chosen=max(scores,key=scores.get)
  if scores[chosen]<scores['static']+.001:chosen='static'
  selection=dict(fold=k,chosen=chosen,source_pooled_scores=scores);selections.append(selection);driver.save(OUT/f'fold{k}_selection.json',selection)
  for part in range(3):
   src=driver.JOB/f'connectivity_graph_20261101/selected_fold{k}_inner{part}.npz' if chosen=='static' else OUT/f'{chosen}_fold{k}_inner{part}.npz';shutil.copy2(src,OUT/f'selected_fold{k}_inner{part}.npz')
  cache=OUT/f'selected_fold{k}_outer.npz'
  if chosen=='static':shutil.copy2(driver.JOB/f'connectivity_graph_20261101/selected_fold{k}_outer.npz',cache)
  elif not cache.exists():
   tr=np.flatnonzero(np.isin(d['subject'],f['training_subjects'])&source.eligible);va=np.flatnonzero(np.isin(d['subject'],f['validation_subjects']));parent=driver.JOB/f'connectivity_graph_20261101/selected_fold{k}_outer.pt';task=TaskData(source,tr,va,parent);path=OUT/f'selected_fold{k}_outer.pt';q,error=fit(task,tr,va,chosen,path,parent);np.savez_compressed(cache,validation=va,probability=q,reload_error=error,model_sha256=np.array(driver.sha(path)))
  print('TEMPORAL GRAPH OUTER',k,'/5 COMPLETE',chosen,flush=True)
 driver.save(OUT/'completion.json',dict(selections=selections,all5outer3inner_complete=True,public_data_used=False,holdout_used=False,official_submissions=0));print('TEMPORAL GRAPH COMPLETE',flush=True)
if __name__=='__main__':main()
