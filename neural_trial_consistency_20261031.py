"""Matched20epoch DE-CNN consistency training; source-inner strength choice.
Existing gamma0 checkpoints reused. RNG of main supervised stream preserved.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,random,copy,time
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import driver
from leaderboard_dualbranch_20261028 import Net,Inputs,groups,infer,SEED,EPOCHS
from select_ensemble import ASSIGNMENTS,postprocess
OUT=driver.JOB/'neural_trial_consistency_20261031';OUT.mkdir(exist_ok=True)
GAMMAS=[.1,1.]

def fit(inputs,tr,va,gamma,path):
 d=inputs.d;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);rng=np.random.default_rng(SEED);pairrng=np.random.default_rng(SEED+77)
 model=Net('de_cnn');opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.03);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS,eta_min=.00005);people=np.unique(d['subject'][tr]);rows=[];diag=[]
 for s in people:
  ix=tr[d['subject'][tr]==s];rows.append(np.stack([ix[d['trial'][ix]==t][np.argsort(d['crop'][ix[d['trial'][ix]==t]])] for t in range(1,9)]));diag.append(d['diagnosis'][ix[0]])
 rows=np.array(rows);diag=np.array(diag);pools=[np.flatnonzero(diag==v) for v in [0,1]];assign=torch.tensor(ASSIGNMENTS.T,dtype=torch.float32);states=[];steps=int(np.ceil(len(people)*5/4))
 for epoch in range(1,EPOCHS+1):
  model.train()
  for step in range(steps):
   who=np.array([rng.choice(pools[j%2]) for j in range(4)]);crops=rng.integers(0,5,(4,8));rr=rows[who[:,None],np.arange(8)[None],crops];de,raw=inputs.get(rr,'de_cnn');de=de+torch.randn_like(de)*.02;y=torch.tensor(d['label'][rr],dtype=torch.float32);z=model(de,raw);flat=z.flatten();yy=y.flatten();pair=F.softplus(flat[yy<.5][None]-flat[yy>.5][:,None]).mean();cond=(torch.logsumexp(z@assign,1)-(z*y).sum(1)).mean()/4
   # A different crop of each SAME source trial; never another person's trial.
   crop2=(crops+pairrng.integers(1,5,(4,8)))%5;rr2=rows[who[:,None],np.arange(8)[None],crop2];assert np.array_equal(d['trial'][rr2],d['trial'][rr]);assert np.array_equal(d['subject'][rr2],d['subject'][rr])
   with torch.random.fork_rng(devices=[]):
    torch.manual_seed(SEED+epoch*1000+step);de2,raw2=inputs.get(rr2,'de_cnn');z2=model(de2,raw2)
   consistency=F.mse_loss(z,z2);loss=.5*F.binary_cross_entropy_with_logits(z,y*.98+.01)+.25*pair+.25*cond+.002*z.square().mean()+gamma*consistency
   opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
  sched.step()
  if epoch in [16,18,20]:states.append(copy.deepcopy(model.state_dict()))
 u,rr=groups(d,va);pred=[]
 for state in states:model.load_state_dict(state);pred.append(infer(model,inputs,rr))
 z=np.mean(pred,axis=0);q=postprocess(z.ravel(),np.repeat(u,8),'conditional',1.);out=np.zeros(len(va));lookup={v:j for j,v in enumerate(va)};out[[lookup[v] for v in rr.ravel()]]=q
 torch.save(dict(states=states,mode='de_cnn',epochs=20,gamma=gamma,source_subjects=people.tolist()),path)
 saved=torch.load(path,map_location='cpu',weights_only=True);replay=[]
 for state in saved['states']:model.load_state_dict(state);replay.append(infer(model,inputs,rr[:,::-1]))
 error=float(abs(np.mean(replay,axis=0)[:,::-1]-z).max());assert error<1e-5
 return out,error

def main():
 torch.set_num_threads(1);p=driver.protocol();spec=dict(gammas=[0.]+GAMMAS,epochs=20,source_inner_metric='mean equal-clinical-mass pooled AUC',same_source_trial_different_crops=True,source_diagnosis_balanced_sampling=True,baseline_checkpoints_reused=True,primary_supervised_RNG_preserved=True,outer_not_for_selection=True,public_data_used=False,holdout_used=False,script_sha256=driver.sha(__file__))
 old=OUT/'protocol.json'
 if old.exists():assert json.loads(old.read_text())==spec
 else:driver.save(old,spec)
 inputs=Inputs();d=inputs.d;selections=[];count=0
 for f in p['folds']:
  scores={gamma:[] for gamma in [0.]+GAMMAS}
  for part in range(3):
   sp=f['inner'][part];tr=np.flatnonzero(np.isin(d['subject'],sp['training_subjects'])&inputs.eligible);va=np.flatnonzero(np.isin(d['subject'],sp['validation_subjects']));assert not set(d['subject'][tr])&set(d['subject'][va]);dep=d['diagnosis'][va]==1
   with np.load(driver.JOB/f'leaderboard_dualbranch_20261028/de_cnn_fold{f["fold"]}_inner{part}.npz') as a:assert np.array_equal(a['validation'],va);scores[0.].append(float(roc_auc_score(d['label'][va],a['probability'],sample_weight=np.where(dep,2.,1.))))
   for gamma in GAMMAS:
    base=OUT/f'gamma{gamma}_fold{f["fold"]}_inner{part}';cache=base.with_suffix('.npz') if False else OUT/(base.name+'.npz');path=OUT/(base.name+'.pt')
    if cache.exists():
     with np.load(cache) as a:assert np.array_equal(a['validation'],va);assert str(a['model_sha256'])==driver.sha(path);q=a['probability']
    else:
     started=time.time();driver.save(OUT/'progress.json',dict(stage='training',completed_new_fits=count,total_inner_fits=30,fold=f['fold'],part=part,gamma=gamma));q,error=fit(inputs,tr,va,gamma,path);np.savez_compressed(cache,validation=va,probability=q,reload_error=error,model_sha256=np.array(driver.sha(path)));print('NEURAL CONSISTENCY inner',f['fold'],part,gamma,'seconds',round(time.time()-started,1),flush=True)
    scores[gamma].append(float(roc_auc_score(d['label'][va],q,sample_weight=np.where(dep,2.,1.))));count+=1
  chosen=max(scores,key=lambda v:np.mean(scores[v]));selection=dict(fold=f['fold'],chosen_gamma=chosen,inner_scores={str(k):v for k,v in scores.items()});selections.append(selection);driver.save(OUT/f'fold{f["fold"]}_selection.json',selection)
  # Selected gamma's3inner forecasts copied for source-only fusion fitting.
  import shutil
  for part in range(3):
   src=driver.JOB/f'leaderboard_dualbranch_20261028/de_cnn_fold{f["fold"]}_inner{part}.npz' if chosen==0 else OUT/f'gamma{chosen}_fold{f["fold"]}_inner{part}.npz';shutil.copy2(src,OUT/f'selected_fold{f["fold"]}_inner{part}.npz')
  va=np.flatnonzero(np.isin(d['subject'],f['validation_subjects']));path=OUT/f'selected_fold{f["fold"]}_outer.pt';cache=OUT/f'selected_fold{f["fold"]}_outer.npz'
  if chosen==0:
   shutil.copy2(driver.JOB/f'leaderboard_dualbranch_20261028/de_cnn_fold{f["fold"]}_outer.npz',cache)
  elif not cache.exists():
   tr=np.flatnonzero(np.isin(d['subject'],f['training_subjects'])&inputs.eligible);q,error=fit(inputs,tr,va,chosen,path);np.savez_compressed(cache,validation=va,probability=q,reload_error=error,model_sha256=np.array(driver.sha(path)))
  print('NEURAL CONSISTENCY OUTER',f['fold'],'/5 COMPLETE gamma',chosen,flush=True)
 driver.save(OUT/'selection_summary.json',dict(selections=selections,all5outer3inner_complete=True,outer_scores_not_used=True,official_submissions=0));print('NEURAL SELECTION COMPLETE',flush=True)
if __name__=='__main__':main()
